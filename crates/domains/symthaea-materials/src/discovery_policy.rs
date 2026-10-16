// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, auditable candidate-selection policies for materials discovery.
//!
//! This module ranks *requests for evidence*. It is not an atomistic, optical,
//! biological, or manufacturing-property predictor. Candidate feature vectors
//! must contain only fields permitted before the next oracle query. In
//! particular, hidden target labels and unrevealed oracle outputs must never be
//! encoded into `features`, `predicted_utility`, or `uncertainty`.
//!
//! `predicted_utility` is supplied by a caller-side policy/model and must be
//! normalized to [0, 1] for this interface. It is not a measured outcome.
//! `uncertainty` is a non-negative exploration signal, not a calibrated
//! probability unless the caller separately validates calibration.
//!
//! The selector intentionally does not own or invoke an oracle. A benchmark
//! loop should call this module first, record the decision receipt, then query
//! the oracle and append the resulting observation to the history. That
//! separation helps make hidden-label leakage auditable; it cannot prove that
//! upstream features or estimates were constructed without leakage.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// Candidate information available before a new expensive evaluation.
///
/// `features` may contain declared descriptors or a fixed representation
/// (including an HDC vector mapped to finite scalar values). Every candidate
/// in one selection call must use the same non-empty feature dimension.
/// Features must not include hidden outcome labels or post-query data.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveryCandidate {
    /// Stable, unique candidate identifier.
    pub id: String,
    /// Optional stable scientific-family identifier (parent structure, batch,
    /// geometry family, sequence family, or another domain-specific grouping).
    pub family_id: Option<String>,
    /// Pre-query features used for descriptor-distance novelty.
    pub features: Vec<f64>,
    /// Caller-provided predicted utility, normalized to [0, 1].
    pub predicted_utility: Option<f64>,
    /// Caller-provided uncertainty signal (non-negative; calibration is external).
    pub uncertainty: Option<f64>,
    /// Estimated positive query cost in caller-defined, consistent units.
    pub estimated_cost: Option<f64>,
}

/// Selection policies intended to be compared under equal query budgets.
///
/// These are transparent baselines, not claims of optimality. Use the
/// `SeededRandom` policy across multiple preregistered seeds for a random
/// acquisition baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoverySelectionPolicy {
    /// Stable pseudo-random candidate ordering for a fixed seed and candidate ID.
    SeededRandom { seed: u64 },
    /// Maximize predicted utility; lexicographic candidate ID breaks ties.
    GreedyUtility,
    /// Maximize predicted utility divided by estimated query cost.
    UtilityPerCost,
    /// Maximize uncertainty divided by estimated query cost.
    UncertaintyPerCost,
    /// Maximize descriptor novelty, preferring an unseen known family on ties.
    DescriptorDiversity,
    /// Choose from the Pareto frontier over utility, uncertainty, novelty and cost.
    ///
    /// Utility, uncertainty, novelty are maximized and cost is minimized.
    /// The final choice among frontier points is deterministic and novelty-first.
    ParetoDiverse,
}

impl DiscoverySelectionPolicy {
    /// Stable identifier suitable for a benchmark receipt.
    pub fn id(self) -> &'static str {
        match self {
            Self::SeededRandom { .. } => "seeded_random_v1",
            Self::GreedyUtility => "greedy_utility_v1",
            Self::UtilityPerCost => "utility_per_cost_v1",
            Self::UncertaintyPerCost => "uncertainty_per_cost_v1",
            Self::DescriptorDiversity => "descriptor_diversity_v1",
            Self::ParetoDiverse => "pareto_diverse_v1",
        }
    }
}

/// Auditable summary of one candidate-selection decision.
///
/// This receipt records estimates that were visible to the selector; it is
/// not an oracle result and should be stored before the chosen candidate is
/// evaluated. An outer benchmark ledger must also record the subsequent oracle
/// outcome, cost actually incurred, failures and total query-budget accounting.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoverySelectionDecision {
    /// Selected candidate identifier.
    pub candidate_id: String,
    /// Stable selection policy identifier.
    pub policy_id: &'static str,
    /// Predicted utility visible before selection, if supplied.
    pub predicted_utility: Option<f64>,
    /// Uncertainty signal visible before selection, if supplied.
    pub uncertainty: Option<f64>,
    /// Estimated cost visible before selection, if supplied.
    pub estimated_cost: Option<f64>,
    /// Minimum normalized descriptor distance to any previously selected candidate.
    ///
    /// Equals 1.0 when there is no prior selected candidate; this is a
    /// convention meaning "no prior point to compare", not a physical score.
    pub descriptor_novelty: f64,
    /// Number of candidates remaining after selected IDs are excluded.
    pub eligible_candidate_count: usize,
    /// Number of Pareto-frontier candidates, only populated for ParetoDiverse.
    pub pareto_frontier_size: Option<usize>,
    /// Number of prior selected IDs considered.
    pub history_count: usize,
}

/// Input-validation or selection failure. No error silently imputes missing
/// utility, uncertainty, cost, descriptor data, or history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoverySelectionError {
    /// The candidate pool is empty.
    EmptyPool,
    /// A candidate ID is empty or whitespace-only.
    EmptyCandidateId { index: usize },
    /// A candidate ID occurs more than once.
    DuplicateCandidateId(String),
    /// A supplied family ID is empty or whitespace-only.
    EmptyFamilyId { candidate_id: String },
    /// A candidate has no descriptor/feature values.
    EmptyFeatureVector { candidate_id: String },
    /// Candidate feature dimensions do not match.
    InconsistentFeatureDimensions {
        /// Candidate whose feature dimension differs.
        candidate_id: String,
        /// Expected feature dimension.
        expected: usize,
        /// Actual feature dimension.
        actual: usize,
    },
    /// A descriptor value is NaN or infinite.
    NonFiniteFeature {
        /// Candidate with the invalid feature.
        candidate_id: String,
        /// Zero-based feature index.
        dimension: usize,
    },
    /// Predicted utility must be finite and in [0, 1].
    InvalidPredictedUtility { candidate_id: String },
    /// Uncertainty must be finite and non-negative.
    InvalidUncertainty { candidate_id: String },
    /// Estimated cost must be finite and strictly positive.
    InvalidEstimatedCost { candidate_id: String },
    /// A selected-history ID appears more than once.
    DuplicateHistoryId(String),
    /// A selected-history ID does not exist in the supplied pool.
    HistoryCandidateMissing(String),
    /// Every candidate in the pool has already been selected.
    NoCandidatesRemaining,
    /// A policy needs a signal that was not supplied by the caller.
    MissingSignal {
        /// Candidate missing the required signal.
        candidate_id: String,
        /// Stable signal name, for example `predicted_utility`.
        signal: &'static str,
    },
    /// A derived score overflowed even though its inputs were finite.
    NonFiniteDerivedScore {
        /// Candidate whose derived score is not finite.
        candidate_id: String,
        /// Stable policy identifier.
        policy_id: &'static str,
    },
}

impl fmt::Display for DiscoverySelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPool => write!(f, "candidate pool is empty"),
            Self::EmptyCandidateId { index } => {
                write!(f, "candidate at index {index} has an empty ID")
            }
            Self::DuplicateCandidateId(id) => write!(f, "duplicate candidate ID: {id}"),
            Self::EmptyFamilyId { candidate_id } => {
                write!(f, "candidate {candidate_id} has an empty family ID")
            }
            Self::EmptyFeatureVector { candidate_id } => {
                write!(f, "candidate {candidate_id} has an empty feature vector")
            }
            Self::InconsistentFeatureDimensions {
                candidate_id,
                expected,
                actual,
            } => write!(
                f,
                "candidate {candidate_id} has {actual} features; expected {expected}"
            ),
            Self::NonFiniteFeature {
                candidate_id,
                dimension,
            } => write!(
                f,
                "candidate {candidate_id} has a non-finite feature at index {dimension}"
            ),
            Self::InvalidPredictedUtility { candidate_id } => write!(
                f,
                "candidate {candidate_id} has an invalid predicted utility (expected finite [0, 1])"
            ),
            Self::InvalidUncertainty { candidate_id } => write!(
                f,
                "candidate {candidate_id} has invalid uncertainty (expected finite >= 0)"
            ),
            Self::InvalidEstimatedCost { candidate_id } => write!(
                f,
                "candidate {candidate_id} has invalid estimated cost (expected finite > 0)"
            ),
            Self::DuplicateHistoryId(id) => write!(f, "duplicate selected-history ID: {id}"),
            Self::HistoryCandidateMissing(id) => {
                write!(f, "selected-history candidate is missing from the pool: {id}")
            }
            Self::NoCandidatesRemaining => write!(f, "no unselected candidates remain"),
            Self::MissingSignal {
                candidate_id,
                signal,
            } => write!(
                f,
                "candidate {candidate_id} is missing required selection signal {signal}"
            ),
            Self::NonFiniteDerivedScore {
                candidate_id,
                policy_id,
            } => write!(
                f,
                "candidate {candidate_id} produced a non-finite score under {policy_id}"
            ),
        }
    }
}

impl Error for DiscoverySelectionError {}

#[derive(Debug, Clone, Copy)]
struct RankedCandidate {
    index: usize,
    novelty: f64,
    new_family: bool,
    random_key: u64,
    utility: Option<f64>,
    uncertainty: Option<f64>,
    cost: Option<f64>,
    utility_per_cost: Option<f64>,
    uncertainty_per_cost: Option<f64>,
}

/// Select one candidate using only the supplied pre-query pool and prior
/// selected-ID history.
///
/// The function does not call an oracle or expose unobserved outcomes. The
/// caller owns query-budget enforcement and should persist the returned
/// decision before evaluating the selected candidate.
pub fn select_next_candidate(
    candidates: &[DiscoveryCandidate],
    already_selected_ids: &[String],
    policy: DiscoverySelectionPolicy,
) -> Result<DiscoverySelectionDecision, DiscoverySelectionError> {
    validate_pool(candidates)?;
    let pool_ids: HashSet<&str> = candidates.iter().map(|candidate| candidate.id.as_str()).collect();
    let mut selected_set: HashSet<String> = HashSet::with_capacity(already_selected_ids.len());

    for id in already_selected_ids {
        if !selected_set.insert(id.clone()) {
            return Err(DiscoverySelectionError::DuplicateHistoryId(id.clone()));
        }
        if !pool_ids.contains(id.as_str()) {
            return Err(DiscoverySelectionError::HistoryCandidateMissing(id.clone()));
        }
    }

    let eligible_indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            (!selected_set.contains(&candidate.id)).then_some(index)
        })
        .collect();

    if eligible_indices.is_empty() {
        return Err(DiscoverySelectionError::NoCandidatesRemaining);
    }

    for &index in &eligible_indices {
        require_policy_signals(&candidates[index], policy)?;
    }

    let normalized = normalize_features(candidates);
    let selected_indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            selected_set.contains(&candidate.id).then_some(index)
        })
        .collect();

    let mut ranked = Vec::with_capacity(eligible_indices.len());
    for index in eligible_indices.iter().copied() {
        let candidate = &candidates[index];
        let novelty = descriptor_novelty(index, &selected_indices, &normalized);
        let new_family = candidate.family_id.as_ref().is_some_and(|family| {
            selected_indices.iter().all(|&selected_index| {
                candidates[selected_index].family_id.as_ref() != Some(family)
            })
        });
        let utility_per_cost = match (candidate.predicted_utility, candidate.estimated_cost) {
            (Some(utility), Some(cost)) => Some(utility / cost),
            _ => None,
        };
        let uncertainty_per_cost = match (candidate.uncertainty, candidate.estimated_cost) {
            (Some(uncertainty), Some(cost)) => Some(uncertainty / cost),
            _ => None,
        };

        if matches!(policy, DiscoverySelectionPolicy::UtilityPerCost)
            && !utility_per_cost.is_some_and(f64::is_finite)
        {
            return Err(DiscoverySelectionError::NonFiniteDerivedScore {
                candidate_id: candidate.id.clone(),
                policy_id: policy.id(),
            });
        }
        if matches!(policy, DiscoverySelectionPolicy::UncertaintyPerCost)
            && !uncertainty_per_cost.is_some_and(f64::is_finite)
        {
            return Err(DiscoverySelectionError::NonFiniteDerivedScore {
                candidate_id: candidate.id.clone(),
                policy_id: policy.id(),
            });
        }

        ranked.push(RankedCandidate {
            index,
            novelty,
            new_family,
            random_key: seeded_candidate_key(
                match policy {
                    DiscoverySelectionPolicy::SeededRandom { seed } => seed,
                    _ => 0,
                },
                &candidate.id,
            ),
            utility: candidate.predicted_utility,
            uncertainty: candidate.uncertainty,
            cost: candidate.estimated_cost,
            utility_per_cost,
            uncertainty_per_cost,
        });
    }

    let pareto_frontier_size = if policy == DiscoverySelectionPolicy::ParetoDiverse {
        let frontier: Vec<RankedCandidate> = ranked
            .iter()
            .copied()
            .filter(|candidate| {
                !ranked.iter().any(|other| {
                    other.index != candidate.index
                        && dominates(other, candidate)
                })
            })
            .collect();
        let frontier_size = frontier.len();
        let selected = select_pareto_representative(frontier, candidates);
        return Ok(make_decision(
            selected,
            candidates,
            policy,
            eligible_indices.len(),
            Some(frontier_size),
            already_selected_ids.len(),
        ));
    } else {
        None
    };

    ranked.sort_by(|left, right| compare_for_policy(*left, *right, policy, candidates));
    let selected = ranked[0];
    Ok(make_decision(
        selected,
        candidates,
        policy,
        eligible_indices.len(),
        pareto_frontier_size,
        already_selected_ids.len(),
    ))
}

fn validate_pool(
    candidates: &[DiscoveryCandidate],
) -> Result<(), DiscoverySelectionError> {
    if candidates.is_empty() {
        return Err(DiscoverySelectionError::EmptyPool);
    }

    let expected_dimensions = candidates[0].features.len();
    if expected_dimensions == 0 {
        return Err(DiscoverySelectionError::EmptyFeatureVector {
            candidate_id: candidates[0].id.clone(),
        });
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.id.trim().is_empty() {
            return Err(DiscoverySelectionError::EmptyCandidateId { index });
        }
        if !ids.insert(candidate.id.as_str()) {
            return Err(DiscoverySelectionError::DuplicateCandidateId(
                candidate.id.clone(),
            ));
        }
        if let Some(family) = candidate.family_id.as_ref() {
            if family.trim().is_empty() {
                return Err(DiscoverySelectionError::EmptyFamilyId {
                    candidate_id: candidate.id.clone(),
                });
            }
        }
        if candidate.features.is_empty() {
            return Err(DiscoverySelectionError::EmptyFeatureVector {
                candidate_id: candidate.id.clone(),
            });
        }
        if candidate.features.len() != expected_dimensions {
            return Err(DiscoverySelectionError::InconsistentFeatureDimensions {
                candidate_id: candidate.id.clone(),
                expected: expected_dimensions,
                actual: candidate.features.len(),
            });
        }
        for (dimension, value) in candidate.features.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(DiscoverySelectionError::NonFiniteFeature {
                    candidate_id: candidate.id.clone(),
                    dimension,
                });
            }
        }
        if candidate
            .predicted_utility
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(DiscoverySelectionError::InvalidPredictedUtility {
                candidate_id: candidate.id.clone(),
            });
        }
        if candidate
            .uncertainty
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(DiscoverySelectionError::InvalidUncertainty {
                candidate_id: candidate.id.clone(),
            });
        }
        if candidate.estimated_cost.is_some_and(|cost| {
            !cost.is_finite() || cost < f64::MIN_POSITIVE
        }) {
            return Err(DiscoverySelectionError::InvalidEstimatedCost {
                candidate_id: candidate.id.clone(),
            });
        }
    }

    Ok(())
}

fn require_policy_signals(
    candidate: &DiscoveryCandidate,
    policy: DiscoverySelectionPolicy,
) -> Result<(), DiscoverySelectionError> {
    let requires_utility = matches!(
        policy,
        DiscoverySelectionPolicy::GreedyUtility
            | DiscoverySelectionPolicy::UtilityPerCost
            | DiscoverySelectionPolicy::ParetoDiverse
    );
    let requires_uncertainty = matches!(
        policy,
        DiscoverySelectionPolicy::UncertaintyPerCost
            | DiscoverySelectionPolicy::ParetoDiverse
    );
    let requires_cost = matches!(
        policy,
        DiscoverySelectionPolicy::UtilityPerCost
            | DiscoverySelectionPolicy::UncertaintyPerCost
            | DiscoverySelectionPolicy::ParetoDiverse
    );

    if requires_utility && candidate.predicted_utility.is_none() {
        return Err(DiscoverySelectionError::MissingSignal {
            candidate_id: candidate.id.clone(),
            signal: "predicted_utility",
        });
    }
    if requires_uncertainty && candidate.uncertainty.is_none() {
        return Err(DiscoverySelectionError::MissingSignal {
            candidate_id: candidate.id.clone(),
            signal: "uncertainty",
        });
    }
    if requires_cost && candidate.estimated_cost.is_none() {
        return Err(DiscoverySelectionError::MissingSignal {
            candidate_id: candidate.id.clone(),
            signal: "estimated_cost",
        });
    }
    Ok(())
}

fn normalize_features(candidates: &[DiscoveryCandidate]) -> Vec<Vec<f64>> {
    let dimensions = candidates[0].features.len();
    let mut minima = vec![f64::INFINITY; dimensions];
    let mut maxima = vec![f64::NEG_INFINITY; dimensions];

    for candidate in candidates {
        for dimension in 0..dimensions {
            minima[dimension] = minima[dimension].min(candidate.features[dimension]);
            maxima[dimension] = maxima[dimension].max(candidate.features[dimension]);
        }
    }

    candidates
        .iter()
        .map(|candidate| {
            (0..dimensions)
                .map(|dimension| {
                    let low = minima[dimension];
                    let high = maxima[dimension];
                    let scale = 1.0_f64.max(low.abs()).max(high.abs());
                    let scaled_low = low / scale;
                    let scaled_high = high / scale;
                    let range = scaled_high - scaled_low;
                    if range <= f64::EPSILON {
                        0.0
                    } else {
                        ((candidate.features[dimension] / scale - scaled_low) / range)
                            .clamp(0.0, 1.0)
                    }
                })
                .collect()
        })
        .collect()
}

fn descriptor_novelty(
    candidate_index: usize,
    selected_indices: &[usize],
    normalized_features: &[Vec<f64>],
) -> f64 {
    if selected_indices.is_empty() {
        return 1.0;
    }

    let candidate = &normalized_features[candidate_index];
    selected_indices
        .iter()
        .map(|&selected_index| {
            let selected = &normalized_features[selected_index];
            let mean_squared_distance = candidate
                .iter()
                .zip(selected.iter())
                .map(|(left, right)| {
                    let difference = left - right;
                    difference * difference
                })
                .sum::<f64>()
                / candidate.len() as f64;
            mean_squared_distance.sqrt().clamp(0.0, 1.0)
        })
        .fold(f64::INFINITY, f64::min)
}

fn seeded_candidate_key(seed: u64, id: &str) -> u64 {
    // FNV-1a followed by a SplitMix64 finalizer. This is a stable seeded
    // ordering for a baseline, not a cryptographic hash or RNG.
    let mut hash = 0xcbf29ce484222325_u64 ^ seed;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d049bb133111eb);
    hash ^ (hash >> 31)
}

fn compare_for_policy(
    left: RankedCandidate,
    right: RankedCandidate,
    policy: DiscoverySelectionPolicy,
    candidates: &[DiscoveryCandidate],
) -> Ordering {
    let left_id = &candidates[left.index].id;
    let right_id = &candidates[right.index].id;
    match policy {
        DiscoverySelectionPolicy::SeededRandom { .. } => left
            .random_key
            .cmp(&right.random_key)
            .then_with(|| left_id.cmp(right_id)),
        DiscoverySelectionPolicy::GreedyUtility => right
            .utility
            .expect("required by signal validation")
            .total_cmp(&left.utility.expect("required by signal validation"))
            .then_with(|| left_id.cmp(right_id)),
        DiscoverySelectionPolicy::UtilityPerCost => right
            .utility_per_cost
            .expect("required and finite after signal validation")
            .total_cmp(&left.utility_per_cost.expect("required and finite after signal validation"))
            .then_with(|| {
                right
                    .novelty
                    .total_cmp(&left.novelty)
            })
            .then_with(|| left_id.cmp(right_id)),
        DiscoverySelectionPolicy::UncertaintyPerCost => right
            .uncertainty_per_cost
            .expect("required and finite after signal validation")
            .total_cmp(&left.uncertainty_per_cost.expect("required and finite after signal validation"))
            .then_with(|| right.novelty.total_cmp(&left.novelty))
            .then_with(|| left_id.cmp(right_id)),
        DiscoverySelectionPolicy::DescriptorDiversity => right
            .novelty
            .total_cmp(&left.novelty)
            .then_with(|| right.new_family.cmp(&left.new_family))
            .then_with(|| left_id.cmp(right_id)),
        DiscoverySelectionPolicy::ParetoDiverse => compare_pareto_representative(left, right, candidates),
    }
}

fn dominates(left: RankedCandidate, right: RankedCandidate) -> bool {
    let left_utility = left.utility.expect("required by signal validation");
    let right_utility = right.utility.expect("required by signal validation");
    let left_uncertainty = left.uncertainty.expect("required by signal validation");
    let right_uncertainty = right.uncertainty.expect("required by signal validation");
    let left_cost = left.cost.expect("required by signal validation");
    let right_cost = right.cost.expect("required by signal validation");

    let no_worse = left_utility >= right_utility
        && left_uncertainty >= right_uncertainty
        && left.novelty >= right.novelty
        && left_cost <= right_cost;
    let strictly_better = left_utility > right_utility
        || left_uncertainty > right_uncertainty
        || left.novelty > right.novelty
        || left_cost < right_cost;
    no_worse && strictly_better
}

fn select_pareto_representative(
    mut frontier: Vec<RankedCandidate>,
    candidates: &[DiscoveryCandidate],
) -> RankedCandidate {
    frontier.sort_by(|left, right| compare_pareto_representative(*left, *right, candidates));
    frontier[0]
}

fn compare_pareto_representative(
    left: RankedCandidate,
    right: RankedCandidate,
    candidates: &[DiscoveryCandidate],
) -> Ordering {
    let left_id = &candidates[left.index].id;
    let right_id = &candidates[right.index].id;
    right
        .novelty
        .total_cmp(&left.novelty)
        .then_with(|| {
            right
                .utility
                .expect("required by signal validation")
                .total_cmp(&left.utility.expect("required by signal validation"))
        })
        .then_with(|| {
            right
                .uncertainty
                .expect("required by signal validation")
                .total_cmp(&left.uncertainty.expect("required by signal validation"))
        })
        .then_with(|| {
            left.cost
                .expect("required by signal validation")
                .total_cmp(&right.cost.expect("required by signal validation"))
        })
        .then_with(|| left_id.cmp(right_id))
}

fn make_decision(
    selected: RankedCandidate,
    candidates: &[DiscoveryCandidate],
    policy: DiscoverySelectionPolicy,
    eligible_candidate_count: usize,
    pareto_frontier_size: Option<usize>,
    history_count: usize,
) -> DiscoverySelectionDecision {
    let candidate = &candidates[selected.index];
    DiscoverySelectionDecision {
        candidate_id: candidate.id.clone(),
        policy_id: policy.id(),
        predicted_utility: candidate.predicted_utility,
        uncertainty: candidate.uncertainty,
        estimated_cost: candidate.estimated_cost,
        descriptor_novelty: selected.novelty,
        eligible_candidate_count,
        pareto_frontier_size,
        history_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        id: &str,
        family_id: &str,
        features: Vec<f64>,
        utility: Option<f64>,
        uncertainty: Option<f64>,
        cost: Option<f64>,
    ) -> DiscoveryCandidate {
        DiscoveryCandidate {
            id: id.to_string(),
            family_id: Some(family_id.to_string()),
            features,
            predicted_utility: utility,
            uncertainty,
            estimated_cost: cost,
        }
    }

    fn fixture() -> Vec<DiscoveryCandidate> {
        vec![
            candidate("a", "family-a", vec![0.0, 0.0], Some(0.70), Some(0.20), Some(3.0)),
            candidate("b", "family-b", vec![0.1, 0.0], Some(0.90), Some(0.10), Some(1.0)),
            candidate("c", "family-c", vec![9.0, 9.0], Some(0.60), Some(0.80), Some(4.0)),
        ]
    }

    #[test]
    fn seeded_random_is_stable_for_same_seed_and_pool() {
        let pool = fixture();
        let first = select_next_candidate(&pool, &[], DiscoverySelectionPolicy::SeededRandom { seed: 42 }).unwrap();
        let second = select_next_candidate(&pool, &[], DiscoverySelectionPolicy::SeededRandom { seed: 42 }).unwrap();
        assert_eq!(first.candidate_id, second.candidate_id);
        assert_eq!(first.policy_id, "seeded_random_v1");
    }

    #[test]
    fn greedy_utility_selects_highest_predicted_utility() {
        let decision = select_next_candidate(&fixture(), &[], DiscoverySelectionPolicy::GreedyUtility).unwrap();
        assert_eq!(decision.candidate_id, "b");
        assert_eq!(decision.predicted_utility, Some(0.90));
    }

    #[test]
    fn utility_per_cost_is_an_explicit_baseline() {
        let decision = select_next_candidate(&fixture(), &[], DiscoverySelectionPolicy::UtilityPerCost).unwrap();
        assert_eq!(decision.candidate_id, "b");
    }

    #[test]
    fn uncertainty_per_cost_selects_largest_uncertainty_per_cost() {
        let decision = select_next_candidate(&fixture(), &[], DiscoverySelectionPolicy::UncertaintyPerCost).unwrap();
        assert_eq!(decision.candidate_id, "c");
    }

    #[test]
    fn descriptor_diversity_prefers_far_candidate_after_selection() {
        let pool = fixture();
        let decision = select_next_candidate(&pool, &["a".to_string()], DiscoverySelectionPolicy::DescriptorDiversity).unwrap();
        assert_eq!(decision.candidate_id, "c");
        assert!(decision.descriptor_novelty > 0.9);
        assert_eq!(decision.history_count, 1);
    }

    #[test]
    fn pareto_policy_preserves_utility_uncertainty_novelty_cost_tradeoffs() {
        let pool = fixture();
        let decision = select_next_candidate(&pool, &["b".to_string()], DiscoverySelectionPolicy::ParetoDiverse).unwrap();
        assert_eq!(decision.candidate_id, "c");
        assert!(decision.pareto_frontier_size.unwrap() >= 1);
    }

    #[test]
    fn previously_selected_candidates_are_never_selected_again() {
        let pool = fixture();
        let decision = select_next_candidate(&pool, &["b".to_string()], DiscoverySelectionPolicy::GreedyUtility).unwrap();
        assert_ne!(decision.candidate_id, "b");
        assert_eq!(decision.eligible_candidate_count, 2);
    }

    #[test]
    fn missing_required_signal_fails_closed_instead_of_imputing_zero() {
        let mut pool = fixture();
        pool[1].estimated_cost = None;
        let error = select_next_candidate(&pool, &[], DiscoverySelectionPolicy::UtilityPerCost).unwrap_err();
        assert_eq!(
            error,
            DiscoverySelectionError::MissingSignal {
                candidate_id: "b".to_string(),
                signal: "estimated_cost",
            }
        );
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let mut pool = fixture();
        pool[1].id = "a".to_string();
        assert!(matches!(
            select_next_candidate(&pool, &[], DiscoverySelectionPolicy::GreedyUtility),
            Err(DiscoverySelectionError::DuplicateCandidateId(_))
        ));
    }

    #[test]
    fn non_finite_features_are_rejected() {
        let mut pool = fixture();
        pool[0].features[0] = f64::NAN;
        assert!(matches!(
            select_next_candidate(&pool, &[], DiscoverySelectionPolicy::DescriptorDiversity),
            Err(DiscoverySelectionError::NonFiniteFeature { .. })
        ));
    }

    #[test]
    fn non_finite_or_out_of_range_utility_is_rejected() {
        let mut pool = fixture();
        pool[0].predicted_utility = Some(1.5);
        assert!(matches!(
            select_next_candidate(&pool, &[], DiscoverySelectionPolicy::GreedyUtility),
            Err(DiscoverySelectionError::InvalidPredictedUtility { .. })
        ));
    }

    #[test]
    fn repeated_or_unknown_history_fails_closed() {
        let pool = fixture();
        assert!(matches!(
            select_next_candidate(&pool, &["a".to_string(), "a".to_string()], DiscoverySelectionPolicy::GreedyUtility),
            Err(DiscoverySelectionError::DuplicateHistoryId(_))
        ));
        assert!(matches!(
            select_next_candidate(&pool, &["unknown".to_string()], DiscoverySelectionPolicy::GreedyUtility),
            Err(DiscoverySelectionError::HistoryCandidateMissing(_))
        ));
    }

    #[test]
    fn fully_exhausted_pool_is_reported() {
        let pool = fixture();
        let error = select_next_candidate(
            &pool,
            &["a".to_string(), "b".to_string(), "c".to_string()],
            DiscoverySelectionPolicy::GreedyUtility,
        )
        .unwrap_err();
        assert_eq!(error, DiscoverySelectionError::NoCandidatesRemaining);
    }

    #[test]
    fn pareto_policy_can_fail_closed_for_unknown_uncertainty() {
        let mut pool = fixture();
        pool[2].uncertainty = None;
        let error = select_next_candidate(&pool, &[], DiscoverySelectionPolicy::ParetoDiverse).unwrap_err();
        assert_eq!(
            error,
            DiscoverySelectionError::MissingSignal {
                candidate_id: "c".to_string(),
                signal: "uncertainty",
            }
        );
    }
}
