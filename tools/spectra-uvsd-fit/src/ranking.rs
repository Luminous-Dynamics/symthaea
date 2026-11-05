// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Indifference-zone selection certificates for robust candidate experiments.
//!
//! This module consumes simultaneous per-action × scenario target-utility
//! intervals from score_bounds. It aggregates those intervals under the
//! frozen scenario objective, selects the action with the largest robust
//! lower bound, and certifies epsilon-optimality only when the largest
//! possible utility of any action is no more than epsilon above the selected
//! action's robust lower bound.
//!
//! The certificate is conditional on the finite scenario set, score ranges,
//! bounded-IID replication contract and simultaneous confidence bounds in
//! ScoreBoundsReceipt. It is not a claim that the scenario bank covers reality.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::scenario::{RobustUtility, ScenarioDefinition};
use super::score_bounds::{ScoreBoundsReceipt, SCORE_BOUND_METHOD_ID};

/// Robust-utility interval for one operational action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionRobustUtilityInterval {
    pub action_id: u32,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Whether the selected action has a simultaneous epsilon-optimality certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustSelectionStatus {
    CertifiedWithinIndifferenceZone,
    NoSelectionCertificate,
}

/// Receipt for robust action ranking under the frozen scenario set.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustSelectionCertificate {
    pub method_id: &'static str,
    pub score_bound_method_id: &'static str,
    pub action_set_digest: String,
    pub scenario_set_digest: String,
    pub score_contract_digest: String,
    pub source_digest: String,
    pub robust_utility: RobustUtility,
    pub epsilon: f64,
    pub familywise_confidence: f64,
    /// Highest robust lower bound; ties resolve to the lowest action ID.
    pub selected_action_id: u32,
    pub selected_utility_lower_bound: f64,
    /// Maximum robust upper bound across every frozen candidate action.
    pub best_possible_utility_upper_bound: f64,
    /// Simultaneous upper bound on regret of selected action.
    pub regret_upper_bound: f64,
    pub status: RobustSelectionStatus,
    pub action_intervals: Vec<ActionRobustUtilityInterval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustSelectionError {
    MissingActionOrScenarioSnapshot,
    InvalidFamilywiseConfidence,
    InvalidScenarioWeight,
    ScenarioWeightsNotNormalized,
    InvalidIndifferenceZone,
    InvalidTailMass,
    MetricCountMismatch,
    DuplicateMetricRecord,
    UnknownMetricAction,
    UnknownMetricScenario,
    MissingMetricRecord,
    InvalidUtilityInterval,
    InvalidScoreReceiptProvenance,
    NumericFailure,
}

impl fmt::Display for RobustSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingActionOrScenarioSnapshot => "score receipt must preserve nonempty action and scenario snapshots",
            Self::InvalidFamilywiseConfidence => "familywise confidence must be finite and in (0, 1]",
            Self::InvalidScenarioWeight => "scenario weights must be finite and positive",
            Self::ScenarioWeightsNotNormalized => "scenario weights must sum to one within the declared numeric tolerance",
            Self::InvalidIndifferenceZone => "indifference-zone epsilon must be finite and nonnegative",
            Self::InvalidTailMass => "lower-tail CVaR mass must be finite and in (0, 1]",
            Self::MetricCountMismatch => "score receipt must contain exactly one utility interval for every action × scenario pair",
            Self::DuplicateMetricRecord => "duplicate utility interval for an action × scenario pair",
            Self::UnknownMetricAction => "utility interval references an action outside the frozen action set",
            Self::UnknownMetricScenario => "utility interval references a scenario outside the frozen scenario set",
            Self::MissingMetricRecord => "utility interval is missing for a frozen action × scenario pair",
            Self::InvalidUtilityInterval => "target-utility interval must be finite, ordered and inside its declared range",
            Self::InvalidScoreReceiptProvenance => "score receipt method, digests, multiplicity or familywise confidence are inconsistent",
            Self::NumericFailure => "robust utility aggregation or selection certificate was not finite",
        };
        f.write_str(message)
    }
}

impl std::error::Error for RobustSelectionError {}

/// Certify an action as epsilon-optimal over the frozen robust scenario objective.
///
/// Let L(a) and U(a) be the robust lower and upper utility bounds for action a.
/// The selected action maximizes L(a). The certificate is issued only when
///
///   max_a U(a) - L(selected) <= epsilon.
///
/// Because the underlying utility intervals are simultaneous, this inequality
/// provides a (1 - familywise_alpha) confidence statement that the selected
/// action is within epsilon of the best action over the declared scenario set.
/// A failed certificate still returns the pessimistically best action, but
/// does not claim it is epsilon-optimal.
pub fn certify_epsilon_optimal_action(
    receipt: &ScoreBoundsReceipt,
    robust_utility: RobustUtility,
    epsilon: f64,
) -> Result<RobustSelectionCertificate, RobustSelectionError> {
    if receipt.action_snapshot.is_empty() || receipt.scenarios.is_empty() {
        return Err(RobustSelectionError::MissingActionOrScenarioSnapshot);
    }
    if !receipt.familywise_confidence.is_finite()
        || receipt.familywise_confidence <= 0.0
        || receipt.familywise_confidence > 1.0
    {
        return Err(RobustSelectionError::InvalidFamilywiseConfidence);
    }
    if receipt.method_id != SCORE_BOUND_METHOD_ID
        || receipt.action_set_digest.trim().is_empty()
        || receipt.scenario_set_digest.trim().is_empty()
        || receipt.score_contract_digest.trim().is_empty()
        || receipt.source_digest.trim().is_empty()
        || !receipt.familywise_alpha.is_finite()
        || receipt.familywise_alpha <= 0.0
        || receipt.familywise_alpha >= 1.0
        || (receipt.familywise_confidence - (1.0 - receipt.familywise_alpha)).abs() > 1.0e-12
    {
        return Err(RobustSelectionError::InvalidScoreReceiptProvenance);
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(RobustSelectionError::InvalidIndifferenceZone);
    }
    if let RobustUtility::LowerTailCvar { tail_mass } = robust_utility {
        if !tail_mass.is_finite() || tail_mass <= 0.0 || tail_mass > 1.0 {
            return Err(RobustSelectionError::InvalidTailMass);
        }
    }

    validate_scenario_weights(&receipt.scenarios)?;

    let action_ids: BTreeSet<u32> = receipt
        .action_snapshot
        .iter()
        .map(|action| action.action_id)
        .collect();
    let scenario_ids: BTreeSet<u16> = receipt
        .scenarios
        .iter()
        .map(|scenario| scenario.scenario_id)
        .collect();
    if action_ids.len() != receipt.action_snapshot.len()
        || scenario_ids.len() != receipt.scenarios.len()
    {
        return Err(RobustSelectionError::MissingActionOrScenarioSnapshot);
    }

    let expected_metrics = action_ids
        .len()
        .checked_mul(scenario_ids.len())
        .ok_or(RobustSelectionError::MetricCountMismatch)?;
    let expected_claim_count = expected_metrics
        .checked_mul(6)
        .ok_or(RobustSelectionError::MetricCountMismatch)?;
    let expected_per_claim_alpha =
        receipt.familywise_alpha / expected_claim_count as f64;
    if receipt.metrics.len() != expected_metrics {
        return Err(RobustSelectionError::MetricCountMismatch);
    }
    if receipt.claim_count != expected_claim_count
        || !receipt.per_claim_alpha.is_finite()
        || receipt.per_claim_alpha <= 0.0
        || (receipt.per_claim_alpha - expected_per_claim_alpha).abs() > 1.0e-12
    {
        return Err(RobustSelectionError::InvalidScoreReceiptProvenance);
    }

    let mut by_pair = BTreeMap::new();
    for metric in &receipt.metrics {
        if !action_ids.contains(&metric.action_id) {
            return Err(RobustSelectionError::UnknownMetricAction);
        }
        if !scenario_ids.contains(&metric.scenario_id) {
            return Err(RobustSelectionError::UnknownMetricScenario);
        }
        let utility = metric.target_utility;
        if !utility.point_estimate.is_finite()
            || !utility.lower_bound.is_finite()
            || !utility.upper_bound.is_finite()
            || utility.lower_bound > utility.point_estimate
            || utility.point_estimate > utility.upper_bound
            || utility.lower_bound < utility.declared_range.lower
            || utility.upper_bound > utility.declared_range.upper
            || !utility.declared_range.lower.is_finite()
            || !utility.declared_range.upper.is_finite()
            || utility.declared_range.lower > utility.declared_range.upper
            || !(utility.declared_range.upper - utility.declared_range.lower).is_finite()
        {
            return Err(RobustSelectionError::InvalidUtilityInterval);
        }
        if by_pair
            .insert((metric.action_id, metric.scenario_id), utility)
            .is_some()
        {
            return Err(RobustSelectionError::DuplicateMetricRecord);
        }
    }
    if by_pair.len() != expected_metrics {
        return Err(RobustSelectionError::MissingMetricRecord);
    }

    let mut action_intervals = Vec::with_capacity(action_ids.len());
    for action_id in action_ids.iter().copied() {
        let mut lower_scenario_values = Vec::with_capacity(receipt.scenarios.len());
        let mut upper_scenario_values = Vec::with_capacity(receipt.scenarios.len());
        for scenario in &receipt.scenarios {
            let utility = by_pair
                .get(&(action_id, scenario.scenario_id))
                .copied()
                .ok_or(RobustSelectionError::MissingMetricRecord)?;
            lower_scenario_values.push((
                scenario.scenario_id,
                scenario.weight,
                utility.lower_bound,
            ));
            upper_scenario_values.push((
                scenario.scenario_id,
                scenario.weight,
                utility.upper_bound,
            ));
        }
        let lower_bound = aggregate_robust_utility(&lower_scenario_values, robust_utility);
        let upper_bound = aggregate_robust_utility(&upper_scenario_values, robust_utility);
        if !lower_bound.is_finite() || !upper_bound.is_finite() || lower_bound > upper_bound {
            return Err(RobustSelectionError::NumericFailure);
        }
        action_intervals.push(ActionRobustUtilityInterval {
            action_id,
            lower_bound,
            upper_bound,
        });
    }

    let selected = action_intervals
        .iter()
        .copied()
        .reduce(|best, candidate| {
            if candidate.lower_bound > best.lower_bound
                || (candidate.lower_bound == best.lower_bound
                    && candidate.action_id < best.action_id)
            {
                candidate
            } else {
                best
            }
        })
        .ok_or(RobustSelectionError::MissingActionOrScenarioSnapshot)?;

    let best_possible_utility_upper_bound = action_intervals
        .iter()
        .map(|interval| interval.upper_bound)
        .fold(f64::NEG_INFINITY, f64::max);
    let regret_upper_bound =
        (best_possible_utility_upper_bound - selected.lower_bound).max(0.0);
    if !best_possible_utility_upper_bound.is_finite() || !regret_upper_bound.is_finite() {
        return Err(RobustSelectionError::NumericFailure);
    }

    let status = if regret_upper_bound <= epsilon {
        RobustSelectionStatus::CertifiedWithinIndifferenceZone
    } else {
        RobustSelectionStatus::NoSelectionCertificate
    };

    Ok(RobustSelectionCertificate {
        method_id: "simultaneous-robust-indifference-zone-v1",
        score_bound_method_id: receipt.method_id,
        action_set_digest: receipt.action_set_digest.clone(),
        scenario_set_digest: receipt.scenario_set_digest.clone(),
        score_contract_digest: receipt.score_contract_digest.clone(),
        source_digest: receipt.source_digest.clone(),
        robust_utility,
        epsilon,
        familywise_confidence: receipt.familywise_confidence,
        selected_action_id: selected.action_id,
        selected_utility_lower_bound: selected.lower_bound,
        best_possible_utility_upper_bound,
        regret_upper_bound,
        status,
        action_intervals,
    })
}

fn validate_scenario_weights(
    scenarios: &[ScenarioDefinition],
) -> Result<(), RobustSelectionError> {
    let mut total = 0.0;
    for scenario in scenarios {
        if !scenario.weight.is_finite() || scenario.weight <= 0.0 {
            return Err(RobustSelectionError::InvalidScenarioWeight);
        }
        total += scenario.weight;
    }
    if !total.is_finite() || (total - 1.0).abs() > 1.0e-9 {
        return Err(RobustSelectionError::ScenarioWeightsNotNormalized);
    }
    Ok(())
}

fn aggregate_robust_utility(
    values: &[(u16, f64, f64)],
    robust_utility: RobustUtility,
) -> f64 {
    match robust_utility {
        RobustUtility::Maximin => values
            .iter()
            .map(|(_, _, value)| *value)
            .fold(f64::INFINITY, f64::min),
        RobustUtility::LowerTailCvar { tail_mass } => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|left, right| {
                left.2
                    .total_cmp(&right.2)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let mut remaining = tail_mass;
            let mut weighted_value = 0.0;
            for (_, weight, value) in sorted {
                if remaining <= 0.0 {
                    break;
                }
                let included_mass = weight.min(remaining);
                weighted_value += included_mass * value;
                remaining -= included_mass;
            }
            weighted_value / tail_mass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::score_bounds::{
        bound_candidate_score_samples, BoundedScoreSamples, ScoreBoundsConfig, ScoreRange,
        ScoreRanges, ScoreBoundsReceipt,
    };
    use super::super::scenario::ScenarioSet;
    use super::super::sequential::{
        OperationalAction, OperationalActionSet, StimulusClass,
    };

    fn action_set() -> OperationalActionSet {
        OperationalActionSet::new(
            vec![
                OperationalAction {
                    action_id: 1,
                    audit_stratum: 1,
                    condition_id: 0,
                    stimulus_class: StimulusClass::Noise,
                    stimulus_level: 0.0,
                    response_scale_id: 1,
                    rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 2,
                    audit_stratum: 1,
                    condition_id: 0,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 1.0,
                    response_scale_id: 1,
                    rt_deadline_ms: None,
                },
            ],
            "sha256:actions",
        )
        .unwrap()
    }

    fn scenario_set() -> ScenarioSet {
        ScenarioSet::new(
            vec![
                super::super::scenario::ScenarioDefinition {
                    scenario_id: 10,
                    weight: 0.6,
                },
                super::super::scenario::ScenarioDefinition {
                    scenario_id: 20,
                    weight: 0.4,
                },
            ],
            "sha256:scenarios",
        )
        .unwrap()
    }

    fn bounded_receipt(
        utility_by_action_scenario: &[(u32, u16, f64)],
        n: usize,
    ) -> (ScoreBoundsReceipt, ScenarioSet) {
        let actions = action_set();
        let scenarios = scenario_set();
        let mut records = Vec::new();
        for action in actions.actions() {
            for scenario in scenarios.scenarios() {
                let utility = utility_by_action_scenario
                    .iter()
                    .find(|(action_id, scenario_id, _)| {
                        *action_id == action.action_id && *scenario_id == scenario.scenario_id
                    })
                    .map(|(_, _, value)| *value)
                    .unwrap();
                records.push(BoundedScoreSamples {
                    action_id: action.action_id,
                    scenario_id: scenario.scenario_id,
                    operational_mapping_qualified: true,
                    model_adequacy_qualified: true,
                    target_information: vec![0.5; n],
                    model_discrimination_information: vec![0.5; n],
                    representativeness: vec![0.9; n],
                    target_utility: vec![utility; n],
                    participant_burden_seconds: vec![1.0; n],
                });
            }
        }
        let config = ScoreBoundsConfig {
            familywise_alpha: 0.05,
            minimum_replicates: 2,
            ranges: ScoreRanges {
                target_information: ScoreRange { lower: 0.0, upper: 1.0 },
                model_discrimination_information: ScoreRange { lower: 0.0, upper: 1.0 },
                representativeness: ScoreRange { lower: 0.0, upper: 1.0 },
                target_utility: ScoreRange { lower: 0.0, upper: 1.0 },
                participant_burden_seconds: ScoreRange { lower: 0.0, upper: 2.0 },
            },
            score_contract_digest: "sha256:score".to_owned(),
            source_digest: "sha256:generator".to_owned(),
            root_seed: 42,
        };
        (
            bound_candidate_score_samples(&actions, &scenarios, &records, &config).unwrap(),
            scenarios,
        )
    }

    #[test]
    fn certifies_epsilon_optimal_action_when_robust_intervals_are_separated_enough() {
        let (receipt, _) = bounded_receipt(
            &[
                (1, 10, 0.9),
                (1, 20, 0.9),
                (2, 10, 0.1),
                (2, 20, 0.1),
            ],
            500,
        );
        let certificate = certify_epsilon_optimal_action(
            &receipt,
            RobustUtility::Maximin,
            0.25,
        )
        .unwrap();
        assert_eq!(certificate.selected_action_id, 1);
        assert_eq!(
            certificate.status,
            RobustSelectionStatus::CertifiedWithinIndifferenceZone
        );
        assert!(certificate.regret_upper_bound <= 0.25);
        assert_eq!(certificate.familywise_confidence, 0.95);
    }

    #[test]
    fn refuses_pcs_certificate_when_the_indifference_zone_is_too_small() {
        let (receipt, _) = bounded_receipt(
            &[
                (1, 10, 0.9),
                (1, 20, 0.9),
                (2, 10, 0.1),
                (2, 20, 0.1),
            ],
            500,
        );
        let certificate = certify_epsilon_optimal_action(
            &receipt,
            RobustUtility::Maximin,
            0.05,
        )
        .unwrap();
        assert_eq!(certificate.selected_action_id, 1);
        assert_eq!(
            certificate.status,
            RobustSelectionStatus::NoSelectionCertificate
        );
        assert!(certificate.regret_upper_bound > 0.05);
    }

    #[test]
    fn selects_by_robust_lower_bound_not_best_case_scenario() {
        let (receipt, _) = bounded_receipt(
            &[
                (1, 10, 0.95),
                (1, 20, 0.2),
                (2, 10, 0.6),
                (2, 20, 0.6),
            ],
            2_000,
        );
        let certificate = certify_epsilon_optimal_action(
            &receipt,
            RobustUtility::Maximin,
            0.5,
        )
        .unwrap();
        assert_eq!(certificate.selected_action_id, 2);
        assert_eq!(certificate.status, RobustSelectionStatus::CertifiedWithinIndifferenceZone);
    }

    #[test]
    fn supports_fractional_weighted_lower_tail_cvar_selection() {
        let (receipt, _) = bounded_receipt(
            &[
                (1, 10, 0.9),
                (1, 20, 0.2),
                (2, 10, 0.6),
                (2, 20, 0.6),
            ],
            2_000,
        );
        let certificate = certify_epsilon_optimal_action(
            &receipt,
            RobustUtility::LowerTailCvar { tail_mass: 0.5 },
            0.5,
        )
        .unwrap();
        assert_eq!(certificate.selected_action_id, 2);
    }

    #[test]
    fn invalid_inputs_and_corrupt_metric_grids_fail_closed() {
        let (mut receipt, _) = bounded_receipt(
            &[
                (1, 10, 0.9),
                (1, 20, 0.9),
                (2, 10, 0.1),
                (2, 20, 0.1),
            ],
            10,
        );
        assert_eq!(
            certify_epsilon_optimal_action(&receipt, RobustUtility::Maximin, f64::NAN),
            Err(RobustSelectionError::InvalidIndifferenceZone)
        );
        assert_eq!(
            certify_epsilon_optimal_action(
                &receipt,
                RobustUtility::LowerTailCvar { tail_mass: 0.0 },
                0.2
            ),
            Err(RobustSelectionError::InvalidTailMass)
        );
        receipt.metrics.pop();
        assert_eq!(
            certify_epsilon_optimal_action(&receipt, RobustUtility::Maximin, 0.2),
            Err(RobustSelectionError::MetricCountMismatch)
        );
    }
}
