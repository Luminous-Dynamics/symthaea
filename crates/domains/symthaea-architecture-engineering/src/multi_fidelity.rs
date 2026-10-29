//! Cost-accounted multi-fidelity promotion decisions for architecture search.
//!
//! This module schedules which candidates advance from one pre-declared fidelity to the
//! next. It does not train models, predict full-fidelity scores, or establish that any
//! promotion policy is superior. Fidelity scores are only compared within the same stage.
//! Costs are cumulative units, so the next-stage incremental cost is charged as the
//! difference between adjacent stages.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// A rung in an evaluation ladder.
///
/// `cumulative_cost_units` is the total planned cost to evaluate a candidate to this
/// fidelity from a clean start. Adjacent differences are used for promotion accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FidelityLevel {
    pub id: String,
    pub cumulative_cost_units: u64,
}

/// An observation obtained at exactly one declared fidelity.
///
/// `mean_score` uses a maximize-is-better convention. `standard_error` must be a
/// finite, non-negative uncertainty estimate in the same score units.
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityObservation {
    pub candidate_id: String,
    pub fidelity_index: usize,
    pub mean_score: f64,
    pub standard_error: f64,
}

/// Pre-registered policy controls for one promotion step.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionPolicy {
    /// Fraction of observed candidates that should advance, rounded up.
    pub promotion_fraction: f64,
    /// Fraction of promotion slots reserved for seeded exploration among candidates
    /// outside the exploitation-ranked prefix. A positive share reserves at least one
    /// slot when pruning is possible.
    pub exploration_share: f64,
    /// Conservative ranking uses mean_score - confidence_multiplier * standard_error.
    pub confidence_multiplier: f64,
    /// Seed used only to make exploration sampling reproducible.
    pub seed: u64,
}

/// Auditable result of a single promotion decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionDecision {
    pub current_fidelity_index: usize,
    pub next_fidelity_index: usize,
    pub promoted_candidate_ids: Vec<String>,
    pub exploitation_count: usize,
    pub exploration_count: usize,
    pub observed_candidate_count: usize,
    pub estimated_incremental_cost_units: u64,
}

/// Validation or accounting failure. Invalid decisions never return partial promotions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FidelityError {
    pub field: String,
    pub message: String,
}

impl FidelityError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for FidelityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for FidelityError {}

/// Validate unique names and strictly increasing, positive cumulative cost units.
pub fn validate_fidelity_ladder(levels: &[FidelityLevel]) -> Result<(), FidelityError> {
    if levels.len() < 2 {
        return Err(FidelityError::new(
            "levels",
            "at least two fidelity levels are required for promotion",
        ));
    }

    let mut ids = HashSet::with_capacity(levels.len());
    let mut previous_cost = 0u64;
    for (index, level) in levels.iter().enumerate() {
        if level.id.trim().is_empty() {
            return Err(FidelityError::new(
                format!("levels[{index}].id"),
                "must be a non-empty string",
            ));
        }
        if !ids.insert(level.id.as_str()) {
            return Err(FidelityError::new(
                format!("levels[{index}].id"),
                format!("duplicate fidelity ID {:?}", level.id),
            ));
        }
        if level.cumulative_cost_units == 0 {
            return Err(FidelityError::new(
                format!("levels[{index}].cumulative_cost_units"),
                "must be positive",
            ));
        }
        if index > 0 && level.cumulative_cost_units <= previous_cost {
            return Err(FidelityError::new(
                format!("levels[{index}].cumulative_cost_units"),
                "must strictly increase because costs are cumulative",
            ));
        }
        previous_cost = level.cumulative_cost_units;
    }
    Ok(())
}

/// Select candidates for the next fidelity with explicit exploration and cost accounting.
///
/// Candidates are ranked only against observations from `current_fidelity_index`. The
/// primary exploitation score is a conservative lower confidence bound, not the raw mean.
/// A deterministic, seeded sample from the remaining candidates provides exploration when
/// the policy reserves a non-zero share. This is a scheduler primitive, not a complete
/// Hyperband/TPE/Bayesian optimizer; comparisons against those policies belong in a separate
/// benchmark with equal total cost budgets.
pub fn plan_promotions(
    levels: &[FidelityLevel],
    observations: &[FidelityObservation],
    current_fidelity_index: usize,
    policy: &PromotionPolicy,
) -> Result<PromotionDecision, FidelityError> {
    validate_fidelity_ladder(levels)?;

    if current_fidelity_index >= levels.len() - 1 {
        return Err(FidelityError::new(
            "current_fidelity_index",
            "must identify a level with a subsequent fidelity",
        ));
    }
    if !policy.promotion_fraction.is_finite()
        || policy.promotion_fraction <= 0.0
        || policy.promotion_fraction > 1.0
    {
        return Err(FidelityError::new(
            "promotion_fraction",
            "must be finite and in (0, 1]",
        ));
    }
    if !policy.exploration_share.is_finite()
        || policy.exploration_share < 0.0
        || policy.exploration_share > 1.0
    {
        return Err(FidelityError::new(
            "exploration_share",
            "must be finite and in [0, 1]",
        ));
    }
    if !policy.confidence_multiplier.is_finite() || policy.confidence_multiplier < 0.0 {
        return Err(FidelityError::new(
            "confidence_multiplier",
            "must be finite and non-negative",
        ));
    }
    if observations.is_empty() {
        return Err(FidelityError::new(
            "observations",
            "at least one observation is required",
        ));
    }

    let mut observed_ids = HashSet::with_capacity(observations.len());
    let mut ranked: Vec<(&FidelityObservation, f64)> = Vec::with_capacity(observations.len());
    for (index, observation) in observations.iter().enumerate() {
        let field = format!("observations[{index}]");
        if observation.candidate_id.trim().is_empty() {
            return Err(FidelityError::new(
                format!("{field}.candidate_id"),
                "must be a non-empty string",
            ));
        }
        if observation.fidelity_index != current_fidelity_index {
            return Err(FidelityError::new(
                format!("{field}.fidelity_index"),
                "all observations in one promotion decision must use the current fidelity",
            ));
        }
        if !observed_ids.insert(observation.candidate_id.as_str()) {
            return Err(FidelityError::new(
                format!("{field}.candidate_id"),
                format!("duplicate candidate ID {:?}", observation.candidate_id),
            ));
        }
        if !observation.mean_score.is_finite() {
            return Err(FidelityError::new(
                format!("{field}.mean_score"),
                "must be finite",
            ));
        }
        if !observation.standard_error.is_finite() || observation.standard_error < 0.0 {
            return Err(FidelityError::new(
                format!("{field}.standard_error"),
                "must be finite and non-negative",
            ));
        }
        let lower_confidence_bound =
            observation.mean_score - policy.confidence_multiplier * observation.standard_error;
        if !lower_confidence_bound.is_finite() {
            return Err(FidelityError::new(
                format!("{field}.lower_confidence_bound"),
                "overflow or non-finite arithmetic while applying uncertainty penalty",
            ));
        }
        ranked.push((observation, lower_confidence_bound));
    }

    // Deterministic ties make a fixed input independent of hash-map iteration order.
    ranked.sort_by(|(left, left_bound), (right, right_bound)| {
        right_bound
            .total_cmp(left_bound)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let candidate_count = ranked.len();
    let promotion_count = ((candidate_count as f64) * policy.promotion_fraction)
        .ceil()
        .max(1.0)
        .min(candidate_count as f64) as usize;

    // If everyone advances, an exploration reservation has no pruning decision to improve.
    // Otherwise, any positive exploration share reserves at least one promotion slot.
    let desired_exploration = if promotion_count == candidate_count
        || policy.exploration_share == 0.0
    {
        0
    } else {
        ((promotion_count as f64) * policy.exploration_share)
            .ceil()
            .max(1.0)
            .min(promotion_count as f64) as usize
    };
    let exploitation_count = promotion_count - desired_exploration;

    let mut promoted: Vec<String> = ranked
        .iter()
        .take(exploitation_count)
        .map(|(observation, _)| observation.candidate_id.clone())
        .collect();

    let mut exploration_pool: Vec<&FidelityObservation> = ranked
        .iter()
        .skip(exploitation_count)
        .map(|(observation, _)| *observation)
        .collect();
    shuffle_with_seed(&mut exploration_pool, policy.seed);
    promoted.extend(
        exploration_pool
            .iter()
            .take(desired_exploration)
            .map(|observation| observation.candidate_id.clone()),
    );

    let current_cost = levels[current_fidelity_index].cumulative_cost_units;
    let next_cost = levels[current_fidelity_index + 1].cumulative_cost_units;
    let incremental_cost_per_candidate = next_cost - current_cost;
    let estimated_incremental_cost_units = incremental_cost_per_candidate
        .checked_mul(promotion_count as u64)
        .ok_or_else(|| {
            FidelityError::new(
                "estimated_incremental_cost_units",
                "u64 overflow while accounting for promoted candidates",
            )
        })?;

    Ok(PromotionDecision {
        current_fidelity_index,
        next_fidelity_index: current_fidelity_index + 1,
        promoted_candidate_ids: promoted,
        exploitation_count,
        exploration_count: desired_exploration,
        observed_candidate_count: candidate_count,
        estimated_incremental_cost_units,
    })
}

fn shuffle_with_seed<T>(values: &mut [T], mut state: u64) {
    for index in (1..values.len()).rev() {
        state = splitmix64(state);
        let selected = (state as usize) % (index + 1);
        values.swap(index, selected);
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels() -> Vec<FidelityLevel> {
        vec![
            FidelityLevel { id: "proxy".into(), cumulative_cost_units: 10 },
            FidelityLevel { id: "short_train".into(), cumulative_cost_units: 40 },
            FidelityLevel { id: "full_train".into(), cumulative_cost_units: 100 },
        ]
    }

    fn observations() -> Vec<FidelityObservation> {
        (0..8)
            .map(|index| FidelityObservation {
                candidate_id: format!("candidate-{index}"),
                fidelity_index: 0,
                mean_score: index as f64,
                standard_error: 0.0,
            })
            .collect()
    }

    fn policy() -> PromotionPolicy {
        PromotionPolicy {
            promotion_fraction: 0.5,
            exploration_share: 0.0,
            confidence_multiplier: 1.0,
            seed: 17,
        }
    }

    #[test]
    fn validates_a_strictly_increasing_cost_ladder() {
        assert!(validate_fidelity_ladder(&levels()).is_ok());
        assert!(validate_fidelity_ladder(&[]).is_err());
        assert!(validate_fidelity_ladder(&[
            FidelityLevel { id: "a".into(), cumulative_cost_units: 10 },
            FidelityLevel { id: "b".into(), cumulative_cost_units: 10 },
        ]).is_err());
    }

    #[test]
    fn promotes_highest_conservative_scores_with_stable_ties() {
        let result = plan_promotions(&levels(), &observations(), 0, &policy()).unwrap();
        assert_eq!(result.promoted_candidate_ids, vec!["candidate-7", "candidate-6", "candidate-5", "candidate-4"]);
        assert_eq!(result.exploitation_count, 4);
        assert_eq!(result.exploration_count, 0);
    }

    #[test]
    fn uncertainty_can_change_the_ranking() {
        let inputs = vec![
            FidelityObservation { candidate_id: "high_mean_high_uncertainty".into(), fidelity_index: 0, mean_score: 0.9, standard_error: 0.3 },
            FidelityObservation { candidate_id: "lower_mean_certain".into(), fidelity_index: 0, mean_score: 0.7, standard_error: 0.01 },
        ];
        let result = plan_promotions(
            &levels(),
            &inputs,
            0,
            &PromotionPolicy { promotion_fraction: 0.5, exploration_share: 0.0, confidence_multiplier: 2.0, seed: 1 },
        ).unwrap();
        assert_eq!(result.promoted_candidate_ids, vec!["lower_mean_certain"]);
    }

    #[test]
    fn exploration_promotions_are_deterministic_and_come_from_outside_exploitation_prefix() {
        let inputs = observations();
        let configured = PromotionPolicy { promotion_fraction: 0.5, exploration_share: 0.25, confidence_multiplier: 0.0, seed: 99 };
        let first = plan_promotions(&levels(), &inputs, 0, &configured).unwrap();
        let second = plan_promotions(&levels(), &inputs, 0, &configured).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.promoted_candidate_ids.len(), 4);
        assert_eq!(first.exploitation_count, 3);
        assert_eq!(first.exploration_count, 1);
        let exploitation_ids = ["candidate-7", "candidate-6", "candidate-5"];
        assert_eq!(
            &first.promoted_candidate_ids[..3],
            &exploitation_ids.map(str::to_owned),
        );
        assert!(!exploitation_ids.contains(&first.promoted_candidate_ids[3].as_str()));
    }

    #[test]
    fn incremental_cost_is_charged_for_each_promotion() {
        let result = plan_promotions(&levels(), &observations(), 1, &policy()).unwrap();
        assert_eq!(result.promoted_candidate_ids.len(), 4);
        assert_eq!(result.estimated_incremental_cost_units, 240);
    }

    #[test]
    fn full_promotion_disables_moot_exploration_reservation() {
        let result = plan_promotions(
            &levels(),
            &observations(),
            0,
            &PromotionPolicy { promotion_fraction: 1.0, exploration_share: 0.8, confidence_multiplier: 0.0, seed: 2 },
        ).unwrap();
        assert_eq!(result.promoted_candidate_ids.len(), 8);
        assert_eq!(result.exploration_count, 0);
        assert_eq!(result.estimated_incremental_cost_units, 240);
    }

    #[test]
    fn rejects_mixed_fidelities_duplicate_candidates_and_non_finite_values() {
        let mut mixed = observations();
        mixed[0].fidelity_index = 1;
        assert!(plan_promotions(&levels(), &mixed, 0, &policy()).is_err());

        let mut duplicate = observations();
        duplicate[1].candidate_id = duplicate[0].candidate_id.clone();
        assert!(plan_promotions(&levels(), &duplicate, 0, &policy()).is_err());

        let mut invalid = observations();
        invalid[0].mean_score = f64::NAN;
        assert!(plan_promotions(&levels(), &invalid, 0, &policy()).is_err());

        let mut invalid_error = observations();
        invalid_error[0].standard_error = -0.1;
        assert!(plan_promotions(&levels(), &invalid_error, 0, &policy()).is_err());
    }

    #[test]
    fn rejects_invalid_policy_and_last_fidelity_promotion() {
        let mut invalid = policy();
        invalid.promotion_fraction = 0.0;
        assert!(plan_promotions(&levels(), &observations(), 0, &invalid).is_err());
        invalid.promotion_fraction = 0.5;
        invalid.exploration_share = 1.1;
        assert!(plan_promotions(&levels(), &observations(), 0, &invalid).is_err());
        assert!(plan_promotions(&levels(), &observations(), 2, &policy()).is_err());
    }

    #[test]
    fn no_observations_and_duplicate_level_names_fail_closed() {
        assert!(plan_promotions(&levels(), &[], 0, &policy()).is_err());
        let bad_ladder = vec![
            FidelityLevel { id: "same".into(), cumulative_cost_units: 10 },
            FidelityLevel { id: "same".into(), cumulative_cost_units: 20 },
        ];
        assert!(validate_fidelity_ladder(&bad_ladder).is_err());
    }

    #[test]
    fn output_is_invariant_to_input_order() {
        let first = plan_promotions(&levels(), &observations(), 0, &policy()).unwrap();
        let mut reversed = observations();
        reversed.reverse();
        let second = plan_promotions(&levels(), &reversed, 0, &policy()).unwrap();
        assert_eq!(first, second);
    }
}
