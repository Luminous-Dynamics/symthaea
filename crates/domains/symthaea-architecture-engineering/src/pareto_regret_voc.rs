//! Exact finite-posterior, two-objective Pareto-regret oracle for NAS.
//!
//! This module is a small exact reference kernel for multi-objective VoC research. It works
//! with normalized maximize-is-better utilities and a frozen hypervolume reference point.
//! A recommendation is a fixed set of at most K architectures. Its value is the posterior
//! expected 2-D hypervolume of that set. The Bayes action is the set maximizing expected
//! hypervolume, not the hypervolume of the posterior-mean objective vector.
//!
//! The oracle computes:
//!
//!   R_HV,K(p) = E_p[HV(all feasible candidates)] -
//!               max_{S: |S| <= K} E_p[HV(S)]
//!
//! For an observation action a, expected value of information is:
//!
//!   VOI_HV,K(a) = R_HV,K(p) - E_{o|a}[R_HV,K(p | o)]
//!
//! The joint particles preserve accuracy/latency (or other objective) correlations. Hard
//! minimum-utility constraints are feasibility filters, not scalar penalty weights.
//!
//! Exact recommendation-set enumeration is exponential. v0.1 deliberately caps the candidate
//! count at 16 and is an oracle for tests/small cohorts; it is not a scalable NAS acquisition
//! function. Higher-dimensional approximate hypervolume and large-space candidate generation
//! must be validated against this reference before deployment.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const MAX_EXACT_CANDIDATES: usize = 16;
const PROBABILITY_TOLERANCE: f64 = 1e-9;
const VALUE_TOLERANCE: f64 = 1e-10;

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoPosteriorParticle {
    pub particle_id: String,
    /// Prior probability mass. All particle weights must sum to one.
    pub probability: f64,
    /// Joint objective vector for each candidate, in candidate_ids order.
    /// Each objective is normalized into [0,1] and maximize-is-better.
    pub objective_values: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoPosterior {
    /// Strictly sorted IDs bind objective-vector columns and deterministic tie-breaking.
    pub candidate_ids: Vec<String>,
    pub particles: Vec<ParetoPosteriorParticle>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoObservationOutcome {
    pub outcome_id: String,
    /// P(outcome | posterior particle j), one value per particle.
    pub likelihood_by_particle: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoMeasurementAction {
    pub action_id: String,
    /// Cost uses the one resource unit frozen by the calling experiment manifest.
    pub incremental_cost_units: u64,
    pub outcomes: Vec<ParetoObservationOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HypervolumeReference {
    /// Lower-left reference point in normalized maximize-is-better utility space.
    pub point: [f64; 2],
    /// Hard lower bounds on both normalized objectives. Values below either floor are
    /// infeasible and contribute no Pareto hypervolume.
    pub minimum_feasible_utility: [f64; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoDecisionValue {
    /// Expected oracle hypervolume when every candidate can be recommended after observing
    /// the particle's true objective values.
    pub expected_oracle_hypervolume: f64,
    /// Maximum posterior expected hypervolume over all fixed candidate sets of size <= K.
    pub best_expected_recommendation_hypervolume: f64,
    /// Difference between the two terms above.
    pub expected_hypervolume_regret: f64,
    pub target_k: usize,
    pub recommended_candidate_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoValueOfInformation {
    pub current: ParetoDecisionValue,
    pub expected_posterior_regret: f64,
    pub expected_hypervolume_regret_reduction: f64,
    pub incremental_cost_units: u64,
    pub expected_hypervolume_regret_reduction_per_cost_unit: f64,
    pub expected_observation_probabilities: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoVoCError {
    pub field: String,
    pub message: String,
}

impl ParetoVoCError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ParetoVoCError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for ParetoVoCError {}

/// Exact dominated hypervolume area in two normalized maximize-is-better objectives.
///
/// The reference is the lower-left corner. Points dominated by the reference (or equal to
/// either reference coordinate) contribute zero. Duplicate and dominated points are harmless.
/// This routine is exact up to floating-point arithmetic.
pub fn hypervolume_2d(
    points: &[[f64; 2]],
    reference: [f64; 2],
) -> Result<f64, ParetoVoCError> {
    validate_unit_point("reference", reference)?;
    let mut eligible = Vec::with_capacity(points.len());
    for (index, point) in points.iter().copied().enumerate() {
        validate_unit_point(&format!("points[{index}]"), point)?;
        if point[0] > reference[0] && point[1] > reference[1] {
            eligible.push(point);
        }
    }
    if eligible.is_empty() {
        return Ok(0.0);
    }

    // Sweep from the largest x coordinate towards the reference. On each vertical strip,
    // the active rectangles are exactly those whose x coordinate reaches that strip; their
    // union height is the maximum y coordinate seen so far.
    eligible.sort_by(|left, right| {
        right[0]
            .total_cmp(&left[0])
            .then_with(|| right[1].total_cmp(&left[1]))
    });

    let mut area = 0.0;
    let mut active_max_y = reference[1];
    let mut index = 0;
    while index < eligible.len() {
        let current_x = eligible[index][0];
        while index < eligible.len() && eligible[index][0] == current_x {
            active_max_y = active_max_y.max(eligible[index][1]);
            index += 1;
        }
        let next_x = if index < eligible.len() {
            eligible[index][0].max(reference[0])
        } else {
            reference[0]
        };
        let strip_width = (current_x - next_x).max(0.0);
        let strip_height = (active_max_y - reference[1]).max(0.0);
        area += strip_width * strip_height;
    }
    Ok(area.max(0.0))
}

/// Return non-dominated point indices in deterministic input order.
///
/// Point i is dominated if another point is at least as good in both objectives and strictly
/// better in at least one. Equal points are represented by the first occurrence only.
pub fn nondominated_indices_2d(points: &[[f64; 2]]) -> Result<Vec<usize>, ParetoVoCError> {
    for (index, point) in points.iter().copied().enumerate() {
        validate_unit_point(&format!("points[{index}]"), point)?;
    }
    let mut output = Vec::new();
    for (index, point) in points.iter().copied().enumerate() {
        let dominated_or_duplicate = points.iter().copied().enumerate().any(|(other_index, other)| {
            if other_index == index {
                return false;
            }
            let no_worse = other[0] >= point[0] && other[1] >= point[1];
            let strictly_better = other[0] > point[0] || other[1] > point[1];
            let duplicate_with_earlier_index = other == point && other_index < index;
            (no_worse && strictly_better) || duplicate_with_earlier_index
        });
        if !dominated_or_duplicate {
            output.push(index);
        }
    }
    Ok(output)
}

/// Compute the exact Bayes recommendation set for a finite joint posterior.
///
/// The oracle enumerates every non-empty fixed candidate set of at most target_k candidates.
/// Each set's objective is its posterior expected hypervolume. The returned regret compares
/// that best posterior decision with the expected per-particle oracle frontier.
pub fn evaluate_pareto_decision(
    posterior: &ParetoPosterior,
    reference: HypervolumeReference,
    target_k: usize,
) -> Result<ParetoDecisionValue, ParetoVoCError> {
    validate_reference(reference)?;
    validate_posterior(posterior)?;

    let candidate_count = posterior.candidate_ids.len();
    if target_k == 0 || target_k >= candidate_count {
        return Err(ParetoVoCError::new(
            "target_k",
            "must be positive and strictly less than the exact candidate count",
        ));
    }
    if candidate_count > MAX_EXACT_CANDIDATES {
        return Err(ParetoVoCError::new(
            "candidate_ids",
            format!(
                "exact v0.1 oracle is limited to {MAX_EXACT_CANDIDATES} candidates; use it as a small-fixture reference"
            ),
        ));
    }

    let mut expected_oracle_hypervolume = 0.0;
    for particle in &posterior.particles {
        let feasible_points = feasible_objectives(&particle.objective_values, reference);
        let particle_oracle_hv = hypervolume_2d(&feasible_points, reference.point)?;
        expected_oracle_hypervolume += particle.probability * particle_oracle_hv;
    }

    let mut best_expected_hv = f64::NEG_INFINITY;
    let mut best_indices = Vec::new();
    let subset_limit = 1u64 << candidate_count;
    for mask in 1u64..subset_limit {
        let subset_size = mask.count_ones() as usize;
        if subset_size > target_k {
            continue;
        }
        let mut expected_hv = 0.0;
        for particle in &posterior.particles {
            let mut selected_points = Vec::with_capacity(subset_size);
            for candidate_index in 0..candidate_count {
                if (mask & (1u64 << candidate_index)) != 0 {
                    let point = particle.objective_values[candidate_index];
                    if is_feasible(point, reference.minimum_feasible_utility) {
                        selected_points.push(point);
                    }
                }
            }
            expected_hv += particle.probability
                * hypervolume_2d(&selected_points, reference.point)?;
        }

        let indices: Vec<usize> = (0..candidate_count)
            .filter(|index| (mask & (1u64 << *index)) != 0)
            .collect();
        let better = expected_hv > best_expected_hv + VALUE_TOLERANCE;
        let tied = (expected_hv - best_expected_hv).abs() <= VALUE_TOLERANCE;
        if better || (tied && lexicographically_preferred(&indices, &best_indices, &posterior.candidate_ids)) {
            best_expected_hv = expected_hv;
            best_indices = indices;
        }
    }

    let expected_hypervolume_regret =
        (expected_oracle_hypervolume - best_expected_hv).max(0.0);
    let recommended_candidate_ids = best_indices
        .iter()
        .map(|index| posterior.candidate_ids[*index].clone())
        .collect();

    Ok(ParetoDecisionValue {
        expected_oracle_hypervolume,
        best_expected_recommendation_hypervolume: best_expected_hv.max(0.0),
        expected_hypervolume_regret,
        target_k,
        recommended_candidate_ids,
    })
}

/// Evaluate the expected reduction in Bayes Pareto hypervolume regret from one discrete action.
///
/// Posterior outcome probabilities are integrated exactly under the finite particle model.
/// Outcomes with zero predictive probability are retained with probability zero and do not
/// require a posterior update.
pub fn expected_pareto_value_of_information(
    posterior: &ParetoPosterior,
    reference: HypervolumeReference,
    target_k: usize,
    action: &ParetoMeasurementAction,
) -> Result<ParetoValueOfInformation, ParetoVoCError> {
    if action.action_id.trim().is_empty() {
        return Err(ParetoVoCError::new("action_id", "must be non-empty"));
    }
    if action.incremental_cost_units == 0 {
        return Err(ParetoVoCError::new(
            "incremental_cost_units",
            "must be positive and use the manifest's declared resource unit",
        ));
    }
    validate_action(action, posterior)?;

    let current = evaluate_pareto_decision(posterior, reference, target_k)?;
    let mut expected_posterior_regret = 0.0;
    let mut expected_observation_probabilities = Vec::with_capacity(action.outcomes.len());

    for outcome in &action.outcomes {
        let outcome_probability: f64 = posterior
            .particles
            .iter()
            .zip(&outcome.likelihood_by_particle)
            .map(|(particle, likelihood)| particle.probability * likelihood)
            .sum();
        expected_observation_probabilities
            .push((outcome.outcome_id.clone(), outcome_probability));

        if outcome_probability <= 0.0 {
            continue;
        }

        let mut updated = posterior.clone();
        for (particle, likelihood) in updated
            .particles
            .iter_mut()
            .zip(&outcome.likelihood_by_particle)
        {
            particle.probability =
                particle.probability * likelihood / outcome_probability;
        }
        updated
            .particles
            .retain(|particle| particle.probability > 0.0);
        let posterior_decision =
            evaluate_pareto_decision(&updated, reference, target_k)?;
        expected_posterior_regret +=
            outcome_probability * posterior_decision.expected_hypervolume_regret;
    }

    let raw_reduction = current.expected_hypervolume_regret - expected_posterior_regret;
    if raw_reduction < -1e-8 {
        return Err(ParetoVoCError::new(
            "expected_pareto_value_of_information",
            "posterior expected Bayes regret increased beyond numerical tolerance",
        ));
    }

    let expected_hypervolume_regret_reduction = raw_reduction.max(0.0);
    Ok(ParetoValueOfInformation {
        current,
        expected_posterior_regret,
        expected_hypervolume_regret_reduction,
        incremental_cost_units: action.incremental_cost_units,
        expected_hypervolume_regret_reduction_per_cost_unit:
            expected_hypervolume_regret_reduction / action.incremental_cost_units as f64,
        expected_observation_probabilities,
    })
}

fn validate_action(
    action: &ParetoMeasurementAction,
    posterior: &ParetoPosterior,
) -> Result<(), ParetoVoCError> {
    if action.outcomes.is_empty() {
        return Err(ParetoVoCError::new(
            "outcomes",
            "at least one observation outcome is required",
        ));
    }
    let mut outcome_ids = HashSet::with_capacity(action.outcomes.len());
    let mut particle_sums = vec![0.0; posterior.particles.len()];
    for (outcome_index, outcome) in action.outcomes.iter().enumerate() {
        if outcome.outcome_id.trim().is_empty()
            || !outcome_ids.insert(outcome.outcome_id.as_str())
        {
            return Err(ParetoVoCError::new(
                format!("outcomes[{outcome_index}].outcome_id"),
                "outcome IDs must be non-empty and unique",
            ));
        }
        if outcome.likelihood_by_particle.len() != posterior.particles.len() {
            return Err(ParetoVoCError::new(
                format!("outcomes[{outcome_index}].likelihood_by_particle"),
                "requires exactly one likelihood per posterior particle",
            ));
        }
        for (particle_index, likelihood) in
            outcome.likelihood_by_particle.iter().copied().enumerate()
        {
            if !likelihood.is_finite() || !(0.0..=1.0).contains(&likelihood) {
                return Err(ParetoVoCError::new(
                    format!("outcomes[{outcome_index}].likelihood_by_particle[{particle_index}]"),
                    "likelihoods must be finite and in [0,1]",
                ));
            }
            particle_sums[particle_index] += likelihood;
        }
    }
    for (particle_index, sum) in particle_sums.into_iter().enumerate() {
        if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(ParetoVoCError::new(
                format!("outcomes[*].likelihood_by_particle[{particle_index}]"),
                "mutually exclusive action outcomes must sum to one for every particle",
            ));
        }
    }
    Ok(())
}

fn validate_posterior(posterior: &ParetoPosterior) -> Result<(), ParetoVoCError> {
    if posterior.candidate_ids.len() < 2 || posterior.candidate_ids.len() > MAX_EXACT_CANDIDATES {
        return Err(ParetoVoCError::new(
            "candidate_ids",
            format!("requires 2..={MAX_EXACT_CANDIDATES} candidates for the exact oracle"),
        ));
    }
    if posterior.candidate_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ParetoVoCError::new(
            "candidate_ids",
            "candidate IDs must be strictly sorted to bind objective columns and tie-breaking",
        ));
    }
    if posterior.particles.is_empty() {
        return Err(ParetoVoCError::new(
            "particles",
            "at least one posterior particle is required",
        ));
    }
    let mut particle_ids = HashSet::with_capacity(posterior.particles.len());
    let mut probability_sum = 0.0;
    for (particle_index, particle) in posterior.particles.iter().enumerate() {
        if particle.particle_id.trim().is_empty()
            || !particle_ids.insert(particle.particle_id.as_str())
        {
            return Err(ParetoVoCError::new(
                format!("particles[{particle_index}].particle_id"),
                "particle IDs must be non-empty and unique",
            ));
        }
        if !particle.probability.is_finite() || particle.probability < 0.0 {
            return Err(ParetoVoCError::new(
                format!("particles[{particle_index}].probability"),
                "posterior probabilities must be finite and non-negative",
            ));
        }
        if particle.objective_values.len() != posterior.candidate_ids.len() {
            return Err(ParetoVoCError::new(
                format!("particles[{particle_index}].objective_values"),
                "objective-vector count must match candidate count",
            ));
        }
        for (candidate_index, point) in particle.objective_values.iter().copied().enumerate() {
            validate_unit_point(
                &format!("particles[{particle_index}].objective_values[{candidate_index}]"),
                point,
            )?;
        }
        probability_sum += particle.probability;
    }
    if !probability_sum.is_finite()
        || (probability_sum - 1.0).abs() > PROBABILITY_TOLERANCE
    {
        return Err(ParetoVoCError::new(
            "particles",
            "posterior probabilities must sum to one",
        ));
    }
    Ok(())
}

fn validate_reference(reference: HypervolumeReference) -> Result<(), ParetoVoCError> {
    validate_unit_point("reference.point", reference.point)?;
    validate_unit_point(
        "reference.minimum_feasible_utility",
        reference.minimum_feasible_utility,
    )?;
    if reference.point[0] >= 1.0
        || reference.point[1] >= 1.0
        || reference.minimum_feasible_utility[0] < reference.point[0]
        || reference.minimum_feasible_utility[1] < reference.point[1]
    {
        return Err(ParetoVoCError::new(
            "reference",
            "reference point must be below one and no greater than either hard feasibility floor",
        ));
    }
    Ok(())
}

fn validate_unit_point(field: &str, point: [f64; 2]) -> Result<(), ParetoVoCError> {
    for (dimension, value) in point.into_iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ParetoVoCError::new(
                format!("{field}[{dimension}]"),
                "normalized objective values must be finite and in [0,1]",
            ));
        }
    }
    Ok(())
}

fn feasible_objectives(
    objectives: &[[f64; 2]],
    reference: HypervolumeReference,
) -> Vec<[f64; 2]> {
    objectives
        .iter()
        .copied()
        .filter(|point| is_feasible(*point, reference.minimum_feasible_utility))
        .collect()
}

fn is_feasible(point: [f64; 2], minimum: [f64; 2]) -> bool {
    point[0] >= minimum[0] && point[1] >= minimum[1]
}

fn lexicographically_preferred(
    candidate: &[usize],
    current: &[usize],
    candidate_ids: &[String],
) -> bool {
    let candidate_names: Vec<&str> = candidate
        .iter()
        .map(|index| candidate_ids[*index].as_str())
        .collect();
    let current_names: Vec<&str> = current
        .iter()
        .map(|index| candidate_ids[*index].as_str())
        .collect();
    candidate_names < current_names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> HypervolumeReference {
        HypervolumeReference {
            point: [0.0, 0.0],
            minimum_feasible_utility: [0.0, 0.0],
        }
    }

    fn posterior(
        particles: Vec<ParetoPosteriorParticle>,
    ) -> ParetoPosterior {
        ParetoPosterior {
            candidate_ids: vec!["A-latency".into(), "B-accuracy".into(), "C-balanced".into()],
            particles,
        }
    }

    #[test]
    fn exact_two_dimensional_hypervolume_for_one_point() {
        let hv = hypervolume_2d(&[[0.5, 0.4]], [0.0, 0.0]).unwrap();
        assert!((hv - 0.20).abs() < 1e-12);
    }

    #[test]
    fn hypervolume_union_does_not_double_count_overlap() {
        let hv = hypervolume_2d(&[[0.8, 0.3], [0.4, 0.7]], [0.0, 0.0]).unwrap();
        // 0.4*0.7 + (0.8-0.4)*0.3 = 0.40.
        assert!((hv - 0.40).abs() < 1e-12);
    }

    #[test]
    fn dominated_and_duplicate_points_do_not_change_hypervolume() {
        let baseline = hypervolume_2d(&[[0.8, 0.7], [0.6, 0.5]], [0.0, 0.0]).unwrap();
        let extended = hypervolume_2d(
            &[[0.8, 0.7], [0.6, 0.5], [0.3, 0.3], [0.8, 0.7]],
            [0.0, 0.0],
        )
        .unwrap();
        assert!((baseline - extended).abs() < 1e-12);
    }

    #[test]
    fn non_dominated_indices_keep_only_the_frontier() {
        let points = [[0.8, 0.7], [0.6, 0.5], [0.4, 0.9], [0.8, 0.7]];
        assert_eq!(nondominated_indices_2d(&points).unwrap(), vec![0, 2]);
    }

    #[test]
    fn bayes_recommendation_uses_expected_hypervolume_not_hypervolume_of_mean_vectors() {
        let posterior = posterior(vec![
            ParetoPosteriorParticle {
                particle_id: "p1".into(),
                probability: 0.5,
                objective_values: vec![[1.0, 0.2], [0.5, 0.5], [0.1, 0.1]],
            },
            ParetoPosteriorParticle {
                particle_id: "p2".into(),
                probability: 0.5,
                objective_values: vec![[0.2, 1.0], [0.5, 0.5], [0.1, 0.1]],
            },
        ]);
        // A's posterior mean is [0.6, 0.6], so HV(E[A])=0.36 > HV(E[B])=0.25.
        // But E[HV(A)] = 0.20 < E[HV(B)] = 0.25: B is the correct Bayes action.
        let result = evaluate_pareto_decision(&posterior, reference(), 1).unwrap();
        assert_eq!(result.recommended_candidate_ids, vec!["B-accuracy"]);
        assert!((result.expected_hypervolume_regret - 0.10).abs() < 1e-12);
    }

    #[test]
    fn hard_feasibility_threshold_excludes_an_infeasible_pareto_point() {
        let posterior = ParetoPosterior {
            candidate_ids: vec!["A".into(), "B".into(), "C".into()],
            particles: vec![ParetoPosteriorParticle {
                particle_id: "p1".into(),
                probability: 1.0,
                objective_values: vec![[0.99, 0.20], [0.80, 0.80], [0.10, 0.95]],
            }],
        };
        let constrained = HypervolumeReference {
            point: [0.0, 0.0],
            minimum_feasible_utility: [0.5, 0.5],
        };
        let result = evaluate_pareto_decision(&posterior, constrained, 1).unwrap();
        assert_eq!(result.recommended_candidate_ids, vec!["B"]);
    }

    #[test]
    fn uninformative_action_has_zero_pareto_value_of_information() {
        let posterior = posterior(vec![
            ParetoPosteriorParticle {
                particle_id: "p1".into(),
                probability: 0.5,
                objective_values: vec![[0.9, 0.2], [0.2, 0.9], [0.45, 0.45]],
            },
            ParetoPosteriorParticle {
                particle_id: "p2".into(),
                probability: 0.5,
                objective_values: vec![[0.8, 0.3], [0.3, 0.8], [0.45, 0.45]],
            },
        ]);
        let action = ParetoMeasurementAction {
            action_id: "no-information".into(),
            incremental_cost_units: 1,
            outcomes: vec![ParetoObservationOutcome {
                outcome_id: "same".into(),
                likelihood_by_particle: vec![1.0, 1.0],
            }],
        };
        let result =
            expected_pareto_value_of_information(&posterior, reference(), 2, &action).unwrap();
        assert!(result.expected_hypervolume_regret_reduction.abs() < 1e-10);
    }

    #[test]
    fn informative_action_reduces_pareto_decision_regret() {
        let posterior = posterior(vec![
            ParetoPosteriorParticle {
                particle_id: "p1".into(),
                probability: 0.5,
                objective_values: vec![[0.95, 0.5], [0.3, 0.4], [0.5, 0.4]],
            },
            ParetoPosteriorParticle {
                particle_id: "p2".into(),
                probability: 0.5,
                objective_values: vec![[0.3, 0.4], [0.95, 0.5], [0.5, 0.4]],
            },
        ]);
        let action = ParetoMeasurementAction {
            action_id: "reveals-which-objective-wins".into(),
            incremental_cost_units: 2,
            outcomes: vec![
                ParetoObservationOutcome {
                    outcome_id: "p1".into(),
                    likelihood_by_particle: vec![1.0, 0.0],
                },
                ParetoObservationOutcome {
                    outcome_id: "p2".into(),
                    likelihood_by_particle: vec![0.0, 1.0],
                },
            ],
        };
        let result =
            expected_pareto_value_of_information(&posterior, reference(), 1, &action).unwrap();
        assert!(result.current.expected_hypervolume_regret > 0.0);
        assert!(result.expected_hypervolume_regret_reduction > 0.0);
        assert!(result.expected_posterior_regret.abs() < 1e-10);
        assert_eq!(result.incremental_cost_units, 2);
        assert!((result.expected_hypervolume_regret_reduction_per_cost_unit
            - result.expected_hypervolume_regret_reduction / 2.0).abs() < 1e-12);
        assert_eq!(
            result
                .expected_observation_probabilities
                .iter()
                .map(|(_, probability)| *probability)
                .collect::<Vec<_>>(),
            vec![0.5, 0.5]
        );
    }

    #[test]
    fn invalid_action_likelihood_mass_fails_closed() {
        let posterior = posterior(vec![ParetoPosteriorParticle {
            particle_id: "p1".into(),
            probability: 1.0,
            objective_values: vec![[0.9, 0.2], [0.2, 0.9], [0.45, 0.45]],
        }]);
        let action = ParetoMeasurementAction {
            action_id: "bad".into(),
            incremental_cost_units: 1,
            outcomes: vec![
                ParetoObservationOutcome {
                    outcome_id: "one".into(),
                    likelihood_by_particle: vec![0.8],
                },
                ParetoObservationOutcome {
                    outcome_id: "two".into(),
                    likelihood_by_particle: vec![0.8],
                },
            ],
        };
        assert!(expected_pareto_value_of_information(&posterior, reference(), 1, &action).is_err());
    }

    #[test]
    fn exact_oracle_refuses_large_cohorts_instead_of_silently_approximating() {
        let candidate_ids = (0..17).map(|index| format!("C-{index:02}")).collect();
        let objectives = (0..17)
            .map(|index| [index as f64 / 20.0, (17 - index) as f64 / 20.0])
            .collect();
        let posterior = ParetoPosterior {
            candidate_ids,
            particles: vec![ParetoPosteriorParticle {
                particle_id: "p".into(),
                probability: 1.0,
                objective_values: objectives,
            }],
        };
        let error = evaluate_pareto_decision(&posterior, reference(), 2)
            .expect_err("large cohort requires an explicit approximate solver");
        assert!(error.to_string().contains("requires 2..=16 candidates"));
    }
}
