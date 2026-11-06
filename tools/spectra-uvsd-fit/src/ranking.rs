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
use super::score_bounds::{
    verify_score_bounds_receipt, ScoreBoundsReceipt, ANYTIME_ALPHA_SPENDING_RULE_ID,
    ANYTIME_SCORE_BOUND_METHOD_ID, SCORE_BOUND_METHOD_ID,
};

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
    ScoreReceiptRecomputationFailed,
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
            Self::ScoreReceiptRecomputationFailed => "score receipt bounds or outputs do not reproduce from its raw Monte Carlo samples",
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
    let score_method_is_valid = match receipt.method_id {
        SCORE_BOUND_METHOD_ID => !receipt.time_uniform && receipt.alpha_spending_rule == "none",
        ANYTIME_SCORE_BOUND_METHOD_ID => {
            receipt.time_uniform
                && receipt.alpha_spending_rule == ANYTIME_ALPHA_SPENDING_RULE_ID
        }
        _ => false,
    };
    if !score_method_is_valid
        || receipt.action_set_digest.trim().is_empty()
        || receipt.scenario_set_digest.trim().is_empty()
        || receipt.score_contract_digest.trim().is_empty()
        || receipt.source_digest.trim().is_empty()
        || !receipt.familywise_alpha.is_finite()
        || receipt.familywise_alpha <= 0.0
        || receipt.familywise_alpha >= 1.0
        || receipt.minimum_replicates < 2
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
            || utility.declared_range != receipt.ranges.target_utility
            || utility.sample_count < receipt.minimum_replicates
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
    verify_score_bounds_receipt(receipt)
        .map_err(|_| RobustSelectionError::ScoreReceiptRecomputationFailed)?;

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


/// Recompute the robust regret gap after radius-only utility-bound overrides.
///
/// This is crate-private because it is a projection primitive, not a certificate:
/// callers must first verify the original score receipt and must never expose the
/// returned value as a statistical guarantee. Point estimates and all untouched
/// cells remain fixed; only the supplied utility endpoints are replaced.

/// Reusable exact projection state for independent action × scenario updates.
///
/// The index preserves the production Maximin/CVaR aggregation and caches the
/// best and runner-up action endpoints. A one-cell counterfactual therefore
/// recomputes only the affected action's scenario aggregate; all other action
/// intervals are reused. The full counterfactual function below remains the
/// reference implementation and parity oracle.
pub(crate) struct IndependentCounterfactualProjectionIndex {
    robust_utility: RobustUtility,
    scenario_values: BTreeMap<u32, Vec<(u16, f64, f64, f64, f64, f64)>>,
    lower_order: Vec<(f64, u32)>,
    upper_order: Vec<(f64, u32)>,
}

impl IndependentCounterfactualProjectionIndex {
    pub(crate) fn new(
        receipt: &ScoreBoundsReceipt,
        robust_utility: RobustUtility,
    ) -> Result<Self, RobustSelectionError> {
        validate_scenario_weights(&receipt.scenarios)?;

        let action_ids = receipt
            .action_snapshot
            .iter()
            .map(|action| action.action_id)
            .collect::<BTreeSet<_>>();
        let scenario_ids = receipt
            .scenarios
            .iter()
            .map(|scenario| scenario.scenario_id)
            .collect::<BTreeSet<_>>();
        if action_ids.is_empty()
            || scenario_ids.is_empty()
            || action_ids.len() != receipt.action_snapshot.len()
            || scenario_ids.len() != receipt.scenarios.len()
        {
            return Err(RobustSelectionError::MissingActionOrScenarioSnapshot);
        }

        let mut utility_by_pair = BTreeMap::new();
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
                || !utility.declared_range.lower.is_finite()
                || !utility.declared_range.upper.is_finite()
                || utility.declared_range.lower > utility.declared_range.upper
                || utility.lower_bound < utility.declared_range.lower
                || utility.upper_bound > utility.declared_range.upper
                || utility.declared_range != receipt.ranges.target_utility
            {
                return Err(RobustSelectionError::InvalidUtilityInterval);
            }
            if utility_by_pair
                .insert((metric.action_id, metric.scenario_id), utility)
                .is_some()
            {
                return Err(RobustSelectionError::DuplicateMetricRecord);
            }
        }

        let expected_metrics = action_ids
            .len()
            .checked_mul(scenario_ids.len())
            .ok_or(RobustSelectionError::MetricCountMismatch)?;
        if utility_by_pair.len() != expected_metrics || receipt.metrics.len() != expected_metrics {
            return Err(RobustSelectionError::MetricCountMismatch);
        }

        let mut scenario_values = BTreeMap::new();
        let mut lower_order = Vec::with_capacity(action_ids.len());
        let mut upper_order = Vec::with_capacity(action_ids.len());

        for action_id in action_ids {
            let mut values = Vec::with_capacity(receipt.scenarios.len());
            let mut lower_values = Vec::with_capacity(receipt.scenarios.len());
            let mut upper_values = Vec::with_capacity(receipt.scenarios.len());
            for scenario in &receipt.scenarios {
                let utility = utility_by_pair
                    .get(&(action_id, scenario.scenario_id))
                    .copied()
                    .ok_or(RobustSelectionError::MissingMetricRecord)?;
                values.push((
                    scenario.scenario_id,
                    scenario.weight,
                    utility.lower_bound,
                    utility.upper_bound,
                    utility.declared_range.lower,
                    utility.declared_range.upper,
                ));
                lower_values.push((
                    scenario.scenario_id,
                    scenario.weight,
                    utility.lower_bound,
                ));
                upper_values.push((
                    scenario.scenario_id,
                    scenario.weight,
                    utility.upper_bound,
                ));
            }
            let lower_bound = aggregate_robust_utility(&lower_values, robust_utility);
            let upper_bound = aggregate_robust_utility(&upper_values, robust_utility);
            if !lower_bound.is_finite() || !upper_bound.is_finite() || lower_bound > upper_bound {
                return Err(RobustSelectionError::NumericFailure);
            }
            lower_order.push((lower_bound, action_id));
            upper_order.push((upper_bound, action_id));
            scenario_values.insert(action_id, values);
        }

        lower_order.sort_by(|left, right| {
            right.0.total_cmp(&left.0).then_with(|| left.1.cmp(&right.1))
        });
        upper_order.sort_by(|left, right| {
            right.0.total_cmp(&left.0).then_with(|| left.1.cmp(&right.1))
        });

        Ok(Self {
            robust_utility,
            scenario_values,
            lower_order,
            upper_order,
        })
    }

    /// Recompute one action interval, then update the global regret gap using
    /// the cached best/runner-up endpoints from every untouched action.
    pub(crate) fn regret_upper_bound_with_override(
        &self,
        action_id: u32,
        scenario_id: u16,
        lower: f64,
        upper: f64,
    ) -> Result<f64, RobustSelectionError> {
        let values = self
            .scenario_values
            .get(&action_id)
            .ok_or(RobustSelectionError::UnknownMetricAction)?;
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(RobustSelectionError::InvalidUtilityInterval);
        }

        let mut scenario_found = false;
        let mut lower_values = Vec::with_capacity(values.len());
        let mut upper_values = Vec::with_capacity(values.len());
        for (id, weight, base_lower, base_upper, range_lower, range_upper) in values {
            let (cell_lower, cell_upper) = if *id == scenario_id {
                scenario_found = true;
                if lower < *range_lower || upper > *range_upper {
                    return Err(RobustSelectionError::InvalidUtilityInterval);
                }
                (lower, upper)
            } else {
                (*base_lower, *base_upper)
            };
            lower_values.push((*id, *weight, cell_lower));
            upper_values.push((*id, *weight, cell_upper));
        }
        if !scenario_found {
            return Err(RobustSelectionError::MissingMetricRecord);
        }

        let updated_lower = aggregate_robust_utility(&lower_values, self.robust_utility);
        let updated_upper = aggregate_robust_utility(&upper_values, self.robust_utility);
        if !updated_lower.is_finite()
            || !updated_upper.is_finite()
            || updated_lower > updated_upper
        {
            return Err(RobustSelectionError::NumericFailure);
        }

        let other_best_lower = maximum_excluding_action(&self.lower_order, action_id);
        let selected_lower = other_best_lower
            .map(|value| value.max(updated_lower))
            .unwrap_or(updated_lower);
        let other_best_upper = maximum_excluding_action(&self.upper_order, action_id);
        let best_possible_upper = other_best_upper
            .map(|value| value.max(updated_upper))
            .unwrap_or(updated_upper);
        let regret_upper_bound = (best_possible_upper - selected_lower).max(0.0);
        if !regret_upper_bound.is_finite() {
            return Err(RobustSelectionError::NumericFailure);
        }
        Ok(regret_upper_bound)
    }
}

fn maximum_excluding_action(sorted: &[(f64, u32)], excluded_action_id: u32) -> Option<f64> {
    match sorted.first() {
        Some((value, action_id)) if *action_id != excluded_action_id => Some(*value),
        Some(_) => sorted.get(1).map(|(value, _)| *value),
        None => None,
    }
}

pub(crate) fn counterfactual_regret_upper_bound(
    receipt: &ScoreBoundsReceipt,
    robust_utility: RobustUtility,
    utility_overrides: &[(u32, u16, f64, f64)],
) -> Result<f64, RobustSelectionError> {
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
    let mut by_pair = BTreeMap::new();
    for metric in &receipt.metrics {
        if !action_ids.contains(&metric.action_id) {
            return Err(RobustSelectionError::UnknownMetricAction);
        }
        if !scenario_ids.contains(&metric.scenario_id) {
            return Err(RobustSelectionError::UnknownMetricScenario);
        }
        if by_pair
            .insert((metric.action_id, metric.scenario_id), metric.target_utility)
            .is_some()
        {
            return Err(RobustSelectionError::DuplicateMetricRecord);
        }
    }
    let expected_metrics = action_ids
        .len()
        .checked_mul(scenario_ids.len())
        .ok_or(RobustSelectionError::MetricCountMismatch)?;
    if action_ids.is_empty() || scenario_ids.is_empty() || by_pair.len() != expected_metrics {
        return Err(RobustSelectionError::MetricCountMismatch);
    }

    let mut overrides = BTreeMap::new();
    for (action_id, scenario_id, lower, upper) in utility_overrides {
        if !action_ids.contains(action_id) {
            return Err(RobustSelectionError::UnknownMetricAction);
        }
        if !scenario_ids.contains(scenario_id) {
            return Err(RobustSelectionError::UnknownMetricScenario);
        }
        let metric = by_pair
            .get(&(*action_id, *scenario_id))
            .ok_or(RobustSelectionError::MissingMetricRecord)?;
        if !lower.is_finite()
            || !upper.is_finite()
            || lower > upper
            || *lower < metric.declared_range.lower
            || *upper > metric.declared_range.upper
        {
            return Err(RobustSelectionError::InvalidUtilityInterval);
        }
        if overrides
            .insert((*action_id, *scenario_id), (*lower, *upper))
            .is_some()
        {
            return Err(RobustSelectionError::DuplicateMetricRecord);
        }
    }

    let mut action_intervals = Vec::with_capacity(action_ids.len());
    for action_id in action_ids.iter().copied() {
        let mut lower_scenario_values = Vec::with_capacity(receipt.scenarios.len());
        let mut upper_scenario_values = Vec::with_capacity(receipt.scenarios.len());
        for scenario in &receipt.scenarios {
            let key = (action_id, scenario.scenario_id);
            let utility = by_pair
                .get(&key)
                .copied()
                .ok_or(RobustSelectionError::MissingMetricRecord)?;
            let (lower, upper) = overrides
                .get(&key)
                .copied()
                .unwrap_or((utility.lower_bound, utility.upper_bound));
            lower_scenario_values.push((scenario.scenario_id, scenario.weight, lower));
            upper_scenario_values.push((scenario.scenario_id, scenario.weight, upper));
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
    Ok(regret_upper_bound)
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
        bound_candidate_score_samples, bound_candidate_score_samples_anytime,
        BoundedScoreSamples, ScoreBoundsConfig, ScoreRange, ScoreRanges, ScoreBoundsReceipt,
        ANYTIME_SCORE_BOUND_METHOD_ID,
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
    fn accepts_time_uniform_receipt_for_data_dependent_robust_stopping() {
        let (fixed_receipt, scenarios) = bounded_receipt(
            &[
                (1, 10, 0.9),
                (1, 20, 0.9),
                (2, 10, 0.1),
                (2, 20, 0.1),
            ],
            500,
        );
        let actions = action_set();
        let config = ScoreBoundsConfig {
            familywise_alpha: fixed_receipt.familywise_alpha,
            minimum_replicates: fixed_receipt.minimum_replicates,
            ranges: fixed_receipt.ranges,
            score_contract_digest: fixed_receipt.score_contract_digest.clone(),
            source_digest: fixed_receipt.source_digest.clone(),
            root_seed: fixed_receipt.root_seed,
        };
        let anytime = bound_candidate_score_samples_anytime(
            &actions,
            &scenarios,
            &fixed_receipt.samples,
            &config,
        )
        .unwrap();
        assert_eq!(anytime.method_id, ANYTIME_SCORE_BOUND_METHOD_ID);
        assert!(anytime.time_uniform);
        let certificate = certify_epsilon_optimal_action(
            &anytime,
            RobustUtility::Maximin,
            0.25,
        )
        .unwrap();
        assert_eq!(certificate.selected_action_id, 1);
        assert_eq!(
            certificate.status,
            RobustSelectionStatus::CertifiedWithinIndifferenceZone
        );
    }

    #[test]
    fn selection_refuses_tampered_utility_interval_even_when_ordered() {
        let (mut receipt, _) = bounded_receipt(
            &[
                (1, 10, 0.9),
                (1, 20, 0.9),
                (2, 10, 0.1),
                (2, 20, 0.1),
            ],
            500,
        );
        receipt.metrics[0].target_utility.lower_bound += 0.01;
        assert_eq!(
            certify_epsilon_optimal_action(&receipt, RobustUtility::Maximin, 0.2),
            Err(RobustSelectionError::ScoreReceiptRecomputationFailed)
        );
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
        let mut corrupted_provenance = receipt.clone();
        corrupted_provenance.per_claim_alpha *= 2.0;
        assert_eq!(
            certify_epsilon_optimal_action(
                &corrupted_provenance,
                RobustUtility::Maximin,
                0.2
            ),
            Err(RobustSelectionError::InvalidScoreReceiptProvenance)
        );

        receipt.metrics.pop();
        assert_eq!(
            certify_epsilon_optimal_action(&receipt, RobustUtility::Maximin, 0.2),
            Err(RobustSelectionError::MetricCountMismatch)
        );
    }
}
