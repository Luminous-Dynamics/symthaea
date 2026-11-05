// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic adaptive Monte Carlo allocation for robust design selection.
//!
//! This first allocator is a robust interval-width heuristic, not OCBA or
//! Track-and-Stop. It requires time-uniform score confidence sequences, only
//! allocates to actions that could still satisfy the declared indifference
//! zone, and stops only when the robust selection certificate succeeds or
//! the total simulation budget is exhausted.

use std::collections::BTreeMap;
use std::fmt;

use super::ranking::{
    certify_epsilon_optimal_action, ActionRobustUtilityInterval, RobustSelectionCertificate,
    RobustSelectionStatus,
};
use super::scenario::{RobustUtility, ScenarioDefinition};
use super::score_bounds::{
    ScoreBoundsReceipt, ANYTIME_ALPHA_SPENDING_RULE_ID, ANYTIME_SCORE_BOUND_METHOD_ID,
};
use super::sequential::OperationalAction;

pub const ALLOCATION_SEED_METHOD_ID: &str = "spectra-splitmix64-seed-v1";

/// How reproducible simulation seeds are assigned to requested score replications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationSeedMode {
    /// Each action × scenario × replication index gets a distinct deterministic seed.
    IndependentPerCell,
    /// Actions within the same scenario and replication index share a seed
    /// to enable common-random-number coupling.
    CommonRandomNumbersWithinScenario,
}

/// Configuration for the next score-simulation allocation decision.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveAllocationConfig {
    pub policy_version: String,
    /// Caller-supplied digest of the complete canonical current score receipt.
    pub score_receipt_digest: String,
    pub root_seed: u64,
    pub seed_mode: ReplicationSeedMode,
    pub robust_utility: RobustUtility,
    /// Preregistered utility tolerance used by the robust selection certificate.
    pub epsilon: f64,
    /// Maximum total score replications across all action × scenario cells.
    pub max_total_replications: usize,
    /// Number of new score replications requested for the selected cell.
    pub next_batch_size: usize,
}

/// One action's part of a deterministic simulation batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationCellRequest {
    pub action_id: u32,
    /// First zero-based per-cell replication index to generate.
    pub first_replication_index: u32,
    pub replication_count: u32,
    /// One deterministic seed per requested replication, in index order.
    pub seeds: Vec<u64>,
}

/// A deterministic request for one or more action × scenario cells.
///
/// Common-random-number mode requests every action within one scenario using
/// the same replication indices and shared seed per index. Independent mode
/// requests only the priority action cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationBatchRequest {
    pub scenario_id: u16,
    pub cells: Vec<SimulationCellRequest>,
    pub seed_mode: ReplicationSeedMode,
}

/// Why the allocation procedure stopped or continues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationDisposition {
    CertifiedStop,
    BudgetExhaustedWithoutCertificate,
    BudgetInsufficientForAtomicBatch,
    AllocateMore,
}

/// Replay receipt for one adaptive score-allocation decision.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveAllocationReceipt {
    pub method_id: &'static str,
    pub policy_version: String,
    pub score_bound_method_id: &'static str,
    pub score_contract_digest: String,
    pub source_digest: String,
    pub action_set_digest: String,
    pub scenario_set_digest: String,
    /// Digest of the exact score snapshot that produced this allocation.
    pub score_receipt_digest: String,
    pub current_cell_sample_counts: Vec<CellSampleCount>,
    pub seed_method_id: &'static str,
    pub root_seed: u64,
    pub seed_mode: ReplicationSeedMode,
    pub robust_utility: RobustUtility,
    pub epsilon: f64,
    pub max_total_replications: usize,
    pub current_total_replications: usize,
    pub certificate: RobustSelectionCertificate,
    pub disposition: AllocationDisposition,
    pub selected_cell_priority: Option<f64>,
    pub competitive_action_ids: Vec<u32>,
    pub cell_priorities: Vec<CellPriority>,
    pub next_batch: Option<SimulationBatchRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellSampleCount {
    pub action_id: u32,
    pub scenario_id: u16,
    pub sample_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellPriority {
    pub action_id: u32,
    pub scenario_id: u16,
    pub sample_count: usize,
    /// Contribution to unresolved robust-utility interval width.
    pub priority: f64,
    pub competitive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveAllocationError {
    MissingPolicyVersion,
    MissingScoreReceiptDigest,
    InvalidEpsilon,
    InvalidBudget,
    FixedLookBoundsNotAllowed,
    InvalidTimeUniformSchedule,
    BudgetAlreadyExceeded,
    ReplicationCountOverflow,
    MissingScoreSampleCell,
    SampleMetricCountMismatch,
    NoActionableCell,
    InvalidScenarioWeights,
    InvalidCvarTailMass,
    CommonRandomNumberCountMismatch,
    NonFinitePriority,
    SeedIndexOverflow,
}

impl fmt::Display for AdaptiveAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingPolicyVersion => "adaptive allocation policy version must not be empty",
            Self::MissingScoreReceiptDigest => "score-receipt digest must not be empty",
            Self::InvalidEpsilon => "indifference-zone epsilon must be finite and nonnegative",
            Self::InvalidBudget => "maximum total and next batch budgets must be positive",
            Self::FixedLookBoundsNotAllowed => "adaptive allocation requires time-uniform score bounds",
            Self::InvalidTimeUniformSchedule => "score receipt does not contain the registered anytime alpha-spending schedule",
            Self::BudgetAlreadyExceeded => "current score replication total exceeds the configured maximum",
            Self::ReplicationCountOverflow => "total or per-cell replication count overflowed",
            Self::MissingScoreSampleCell => "score receipt is missing a raw sample cell",
            Self::SampleMetricCountMismatch => "metric score vectors do not have a common per-cell sample count",
            Self::NoActionableCell => "no competitive action/scenario cell can receive another replication",
            Self::InvalidScenarioWeights => "scenario weights must be finite, positive and normalized",
            Self::InvalidCvarTailMass => "CVaR tail mass must be finite and in (0, 1]",
            Self::CommonRandomNumberCountMismatch => "common-random-number mode requires equal replication counts across actions within the selected scenario",
            Self::NonFinitePriority => "adaptive allocation priority is not finite",
            Self::SeedIndexOverflow => "requested replication index exceeds u32 capacity",
        };
        f.write_str(message)
    }
}

impl std::error::Error for AdaptiveAllocationError {}

/// Decide whether to stop or request the next batch of Monte Carlo score simulations.
///
/// A valid time-uniform score receipt is required because cell sample counts are
/// selected adaptively. The returned request includes deterministic seeds and
/// a stable per-cell replication index. The caller must append the complete
/// five-metric result for each requested index and recompute an anytime receipt.
pub fn allocate_next_score_batch(
    receipt: &ScoreBoundsReceipt,
    config: &AdaptiveAllocationConfig,
) -> Result<AdaptiveAllocationReceipt, AdaptiveAllocationError> {
    if config.policy_version.trim().is_empty() {
        return Err(AdaptiveAllocationError::MissingPolicyVersion);
    }
    if config.score_receipt_digest.trim().is_empty() {
        return Err(AdaptiveAllocationError::MissingScoreReceiptDigest);
    }
    if !config.epsilon.is_finite() || config.epsilon < 0.0 {
        return Err(AdaptiveAllocationError::InvalidEpsilon);
    }
    if config.max_total_replications == 0 || config.next_batch_size == 0 {
        return Err(AdaptiveAllocationError::InvalidBudget);
    }
    if receipt.method_id != ANYTIME_SCORE_BOUND_METHOD_ID || !receipt.time_uniform {
        return Err(AdaptiveAllocationError::FixedLookBoundsNotAllowed);
    }
    if receipt.alpha_spending_rule != ANYTIME_ALPHA_SPENDING_RULE_ID {
        return Err(AdaptiveAllocationError::InvalidTimeUniformSchedule);
    }

    validate_receipt_sample_grid(receipt)?;
    let current_total_replications = receipt.samples.iter().try_fold(0usize, |total, cell| {
        let count = common_cell_sample_count(cell)?;
        total
            .checked_add(count)
            .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)
    })?;
    if current_total_replications > config.max_total_replications {
        return Err(AdaptiveAllocationError::BudgetAlreadyExceeded);
    }

    let certificate = certify_epsilon_optimal_action(
        receipt,
        config.robust_utility,
        config.epsilon,
    )
    .map_err(|_| AdaptiveAllocationError::InvalidTimeUniformSchedule)?;

    let competitive_action_ids = receipt
        .action_snapshot
        .iter()
        .filter_map(|action| {
            certificate
                .action_intervals
                .iter()
                .find(|interval| interval.action_id == action.action_id)
                .filter(|interval| {
                    interval.upper_bound
                        >= certificate.selected_utility_lower_bound - config.epsilon
                })
                .map(|interval| interval.action_id)
        })
        .collect::<Vec<_>>();

    if certificate.status == RobustSelectionStatus::CertifiedWithinIndifferenceZone {
        return Ok(build_stop_receipt(
            receipt,
            config,
            current_total_replications,
            certificate,
            competitive_action_ids,
            AllocationDisposition::CertifiedStop,
        )?);
    }

    if current_total_replications >= config.max_total_replications {
        return Ok(build_stop_receipt(
            receipt,
            config,
            current_total_replications,
            certificate,
            competitive_action_ids,
            AllocationDisposition::BudgetExhaustedWithoutCertificate,
        )?);
    }

    let remaining_budget = config.max_total_replications - current_total_replications;
    let priorities = compute_cell_priorities(receipt, &certificate, config.robust_utility)?;
    let mut actionable = priorities
        .iter()
        .filter(|cell| cell.competitive && cell.priority > 0.0)
        .cloned()
        .collect::<Vec<_>>();
    actionable.sort_by(|left, right| {
        right
            .priority
            .total_cmp(&left.priority)
            .then_with(|| left.sample_count.cmp(&right.sample_count))
            .then_with(|| left.action_id.cmp(&right.action_id))
            .then_with(|| left.scenario_id.cmp(&right.scenario_id))
    });
    let selected_cell = actionable
        .first()
        .ok_or(AdaptiveAllocationError::NoActionableCell)?;

    let target_action_ids = match config.seed_mode {
        ReplicationSeedMode::IndependentPerCell => vec![selected_cell.action_id],
        ReplicationSeedMode::CommonRandomNumbersWithinScenario => {
            let counts = receipt
                .samples
                .iter()
                .filter(|cell| cell.scenario_id == selected_cell.scenario_id)
                .map(|cell| common_cell_sample_count(cell).map(|count| (cell.action_id, count)))
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            if counts.len() != receipt.action_snapshot.len()
                || counts.values().any(|count| *count != selected_cell.sample_count)
            {
                return Err(AdaptiveAllocationError::CommonRandomNumberCountMismatch);
            }
            receipt
                .action_snapshot
                .iter()
                .map(|action| action.action_id)
                .collect()
        }
    };

    let cost_per_replication = target_action_ids.len();
    let affordable_count = remaining_budget / cost_per_replication;
    if affordable_count == 0 {
        return Ok(build_stop_receipt(
            receipt,
            config,
            current_total_replications,
            certificate,
            competitive_action_ids,
            AllocationDisposition::BudgetInsufficientForAtomicBatch,
        )?);
    }
    let requested_count = affordable_count.min(config.next_batch_size);
    let first_index = u32::try_from(selected_cell.sample_count)
        .map_err(|_| AdaptiveAllocationError::SeedIndexOverflow)?;
    let last_index_exclusive = selected_cell
        .sample_count
        .checked_add(requested_count)
        .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)?;
    if last_index_exclusive > (u32::MAX as usize) + 1 {
        return Err(AdaptiveAllocationError::SeedIndexOverflow);
    }

    let mut cell_requests = Vec::with_capacity(target_action_ids.len());
    for action_id in target_action_ids {
        let mut seeds = Vec::with_capacity(requested_count);
        for offset in 0..requested_count {
            let replication_index = selected_cell
                .sample_count
                .checked_add(offset)
                .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)?;
            let replication_index = u32::try_from(replication_index)
                .map_err(|_| AdaptiveAllocationError::SeedIndexOverflow)?;
            seeds.push(derive_replication_seed(
                config.root_seed,
                action_id,
                selected_cell.scenario_id,
                replication_index,
                config.seed_mode,
            ));
        }
        cell_requests.push(SimulationCellRequest {
            action_id,
            first_replication_index: first_index,
            replication_count: u32::try_from(requested_count)
                .map_err(|_| AdaptiveAllocationError::SeedIndexOverflow)?,
            seeds,
        });
    }
    let next_batch = SimulationBatchRequest {
        scenario_id: selected_cell.scenario_id,
        cells: cell_requests,
        seed_mode: config.seed_mode,
    };
    let selected_cell_priority = Some(selected_cell.priority);

    Ok(AdaptiveAllocationReceipt {
        method_id: "robust-interval-width-allocation-v1",
        policy_version: config.policy_version.clone(),
        score_bound_method_id: receipt.method_id,
        score_contract_digest: receipt.score_contract_digest.clone(),
        source_digest: receipt.source_digest.clone(),
        action_set_digest: receipt.action_set_digest.clone(),
        scenario_set_digest: receipt.scenario_set_digest.clone(),
        score_receipt_digest: config.score_receipt_digest.clone(),
        current_cell_sample_counts: current_cell_sample_counts(receipt)?,
        seed_method_id: ALLOCATION_SEED_METHOD_ID,
        root_seed: config.root_seed,
        seed_mode: config.seed_mode,
        robust_utility: config.robust_utility,
        epsilon: config.epsilon,
        max_total_replications: config.max_total_replications,
        current_total_replications,
        certificate,
        disposition: AllocationDisposition::AllocateMore,
        selected_cell_priority,
        competitive_action_ids,
        cell_priorities: priorities,
        next_batch: Some(next_batch),
    })
}

fn build_stop_receipt(
    receipt: &ScoreBoundsReceipt,
    config: &AdaptiveAllocationConfig,
    current_total_replications: usize,
    certificate: RobustSelectionCertificate,
    competitive_action_ids: Vec<u32>,
    disposition: AllocationDisposition,
) -> Result<AdaptiveAllocationReceipt, AdaptiveAllocationError> {
    Ok(AdaptiveAllocationReceipt {
        method_id: "robust-interval-width-allocation-v1",
        policy_version: config.policy_version.clone(),
        score_bound_method_id: receipt.method_id,
        score_contract_digest: receipt.score_contract_digest.clone(),
        source_digest: receipt.source_digest.clone(),
        action_set_digest: receipt.action_set_digest.clone(),
        scenario_set_digest: receipt.scenario_set_digest.clone(),
        score_receipt_digest: config.score_receipt_digest.clone(),
        current_cell_sample_counts: current_cell_sample_counts(receipt)?,
        seed_method_id: ALLOCATION_SEED_METHOD_ID,
        root_seed: config.root_seed,
        seed_mode: config.seed_mode,
        robust_utility: config.robust_utility,
        epsilon: config.epsilon,
        max_total_replications: config.max_total_replications,
        current_total_replications,
        certificate,
        disposition,
        selected_cell_priority: None,
        competitive_action_ids,
        cell_priorities: Vec::new(),
        next_batch: None,
    })
}

fn common_cell_sample_count(
    cell: &super::score_bounds::BoundedScoreSamples,
) -> Result<usize, AdaptiveAllocationError> {
    let count = cell.target_information.len();
    if cell.model_discrimination_information.len() != count
        || cell.representativeness.len() != count
        || cell.target_utility.len() != count
        || cell.participant_burden_seconds.len() != count
    {
        return Err(AdaptiveAllocationError::SampleMetricCountMismatch);
    }
    Ok(count)
}

fn compute_cell_priorities(
    receipt: &ScoreBoundsReceipt,
    certificate: &RobustSelectionCertificate,
    robust_utility: RobustUtility,
) -> Result<Vec<CellPriority>, AdaptiveAllocationError> {
    let competitive_ids: BTreeMap<u32, bool> = certificate
        .action_intervals
        .iter()
        .map(|interval| {
            (
                interval.action_id,
                interval.upper_bound
                    >= certificate.selected_utility_lower_bound - certificate.epsilon,
            )
        })
        .collect();
    let mut priorities = Vec::with_capacity(receipt.metrics.len());

    for action_interval in &certificate.action_intervals {
        let competitive = competitive_ids
            .get(&action_interval.action_id)
            .copied()
            .unwrap_or(false);
        let mut action_metrics = receipt
            .metrics
            .iter()
            .filter(|metric| metric.action_id == action_interval.action_id)
            .collect::<Vec<_>>();
        action_metrics.sort_by_key(|metric| metric.scenario_id);

        let tail_contributions = scenario_tail_contributions(
            &action_metrics,
            &receipt.scenarios,
            robust_utility,
        )?;
        for metric in action_metrics {
            let sample_count = metric.target_utility.sample_count;
            let width = metric.target_utility.upper_bound
                - metric.target_utility.lower_bound;
            let tail_mass = tail_contributions
                .get(&metric.scenario_id)
                .copied()
                .unwrap_or(0.0);
            let priority = width * tail_mass;
            if !priority.is_finite() || priority < 0.0 {
                return Err(AdaptiveAllocationError::NonFinitePriority);
            }
            priorities.push(CellPriority {
                action_id: metric.action_id,
                scenario_id: metric.scenario_id,
                sample_count,
                priority,
                competitive,
            });
        }
    }
    Ok(priorities)
}

fn current_cell_sample_counts(
    receipt: &ScoreBoundsReceipt,
) -> Result<Vec<CellSampleCount>, AdaptiveAllocationError> {
    let mut counts = receipt
        .samples
        .iter()
        .map(|cell| {
            Ok(CellSampleCount {
                action_id: cell.action_id,
                scenario_id: cell.scenario_id,
                sample_count: common_cell_sample_count(cell)?,
            })
        })
        .collect::<Result<Vec<_>, AdaptiveAllocationError>>()?;
    counts.sort_by_key(|cell| (cell.action_id, cell.scenario_id));
    Ok(counts)
}

fn validate_receipt_sample_grid(
    receipt: &ScoreBoundsReceipt,
) -> Result<(), AdaptiveAllocationError> {
    let expected = receipt
        .action_snapshot
        .len()
        .checked_mul(receipt.scenarios.len())
        .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)?;
    if receipt.samples.len() != expected {
        return Err(AdaptiveAllocationError::MissingScoreSampleCell);
    }
    let metric_by_pair: BTreeMap<_, _> = receipt
        .metrics
        .iter()
        .map(|metric| ((metric.action_id, metric.scenario_id), metric))
        .collect();
    if metric_by_pair.len() != expected {
        return Err(AdaptiveAllocationError::MissingScoreSampleCell);
    }
    if receipt.minimum_replicates < 2 {
        return Err(AdaptiveAllocationError::SampleMetricCountMismatch);
    }
    let mut seen = BTreeMap::new();
    for sample in &receipt.samples {
        let count = common_cell_sample_count(sample)?;
        if count < receipt.minimum_replicates {
            return Err(AdaptiveAllocationError::SampleMetricCountMismatch);
        }
        let key = (sample.action_id, sample.scenario_id);
        if seen.insert(key, count).is_some() {
            return Err(AdaptiveAllocationError::MissingScoreSampleCell);
        }
        let metric = metric_by_pair
            .get(&key)
            .copied()
            .ok_or(AdaptiveAllocationError::MissingScoreSampleCell)?;
        if metric.target_utility.sample_count != count
            || metric.target_information.sample_count != count
            || metric.model_discrimination_information.sample_count != count
            || metric.representativeness.sample_count != count
            || metric.participant_burden_seconds.sample_count != count
        {
            return Err(AdaptiveAllocationError::SampleMetricCountMismatch);
        }
        validate_score_vector(&sample.target_information, receipt.ranges.target_information)?;
        validate_score_vector(
            &sample.model_discrimination_information,
            receipt.ranges.model_discrimination_information,
        )?;
        validate_score_vector(&sample.representativeness, receipt.ranges.representativeness)?;
        validate_score_vector(&sample.target_utility, receipt.ranges.target_utility)?;
        validate_score_vector(
            &sample.participant_burden_seconds,
            receipt.ranges.participant_burden_seconds,
        )?;
    }
    if seen.len() != expected {
        return Err(AdaptiveAllocationError::MissingScoreSampleCell);
    }
    Ok(())
}

fn validate_score_vector(
    values: &[f64],
    range: super::score_bounds::ScoreRange,
) -> Result<(), AdaptiveAllocationError> {
    if !range.lower.is_finite()
        || !range.upper.is_finite()
        || range.lower > range.upper
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < range.lower || *value > range.upper)
    {
        return Err(AdaptiveAllocationError::SampleMetricCountMismatch);
    }
    Ok(())
}

fn scenario_tail_contributions(
    metrics: &[&super::score_bounds::ActionScenarioMetricBounds],
    scenarios: &[ScenarioDefinition],
    robust_utility: RobustUtility,
) -> Result<BTreeMap<u16, f64>, AdaptiveAllocationError> {
    if scenarios.is_empty() {
        return Err(AdaptiveAllocationError::InvalidScenarioWeights);
    }
    let mut total_weight = 0.0;
    let mut weights = BTreeMap::new();
    for scenario in scenarios {
        if !scenario.weight.is_finite() || scenario.weight <= 0.0 {
            return Err(AdaptiveAllocationError::InvalidScenarioWeights);
        }
        total_weight += scenario.weight;
        weights.insert(scenario.scenario_id, scenario.weight);
    }
    if !total_weight.is_finite() || (total_weight - 1.0).abs() > 1.0e-9 {
        return Err(AdaptiveAllocationError::InvalidScenarioWeights);
    }

    let mut lower_values = Vec::with_capacity(metrics.len());
    let mut upper_values = Vec::with_capacity(metrics.len());
    for metric in metrics {
        let weight = weights
            .get(&metric.scenario_id)
            .copied()
            .ok_or(AdaptiveAllocationError::InvalidScenarioWeights)?;
        lower_values.push((metric.scenario_id, weight, metric.target_utility.lower_bound));
        upper_values.push((metric.scenario_id, weight, metric.target_utility.upper_bound));
    }

    let mut contributions = BTreeMap::<u16, f64>::new();
    match robust_utility {
        RobustUtility::Maximin => {
            let min_lower = lower_values
                .iter()
                .map(|(_, _, value)| *value)
                .fold(f64::INFINITY, f64::min);
            let min_upper = upper_values
                .iter()
                .map(|(_, _, value)| *value)
                .fold(f64::INFINITY, f64::min);
            let lower_ties = lower_values
                .iter()
                .filter(|(_, _, value)| (*value - min_lower).abs() <= 1.0e-12)
                .count()
                .max(1);
            let upper_ties = upper_values
                .iter()
                .filter(|(_, _, value)| (*value - min_upper).abs() <= 1.0e-12)
                .count()
                .max(1);
            for (scenario_id, _, value) in &lower_values {
                if (*value - min_lower).abs() <= 1.0e-12 {
                    *contributions.entry(*scenario_id).or_insert(0.0) +=
                        1.0 / lower_ties as f64;
                }
            }
            for (scenario_id, _, value) in &upper_values {
                if (*value - min_upper).abs() <= 1.0e-12 {
                    *contributions.entry(*scenario_id).or_insert(0.0) +=
                        1.0 / upper_ties as f64;
                }
            }
        }
        RobustUtility::LowerTailCvar { tail_mass } => {
            if !tail_mass.is_finite() || tail_mass <= 0.0 || tail_mass > 1.0 {
                return Err(AdaptiveAllocationError::InvalidCvarTailMass);
            }
            let lower = lower_tail_mass(&lower_values, tail_mass);
            let upper = lower_tail_mass(&upper_values, tail_mass);
            for (scenario_id, mass) in lower {
                *contributions.entry(scenario_id).or_insert(0.0) += mass;
            }
            for (scenario_id, mass) in upper {
                *contributions.entry(scenario_id).or_insert(0.0) += mass;
            }
        }
    }
    Ok(contributions)
}

fn lower_tail_mass(
    values: &[(u16, f64, f64)],
    tail_mass: f64,
) -> BTreeMap<u16, f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| {
        left.2
            .total_cmp(&right.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut remaining = tail_mass;
    let mut included = BTreeMap::new();
    for (scenario_id, weight, _) in sorted {
        if remaining <= 0.0 {
            break;
        }
        let mass = weight.min(remaining);
        if mass > 0.0 {
            included.insert(scenario_id, mass);
            remaining -= mass;
        }
    }
    included
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::score_bounds::{
        bound_candidate_score_samples, bound_candidate_score_samples_anytime,
        BoundedScoreSamples, ScoreBoundsConfig, ScoreRange, ScoreRanges,
    };
    use super::super::scenario::ScenarioSet;
    use super::super::sequential::{OperationalActionSet, StimulusClass};

    fn action_set() -> OperationalActionSet {
        OperationalActionSet::new(
            vec![
                OperationalAction {
                    action_id: 1,
                    audit_stratum: 1,
                    condition_id: 1,
                    stimulus_class: StimulusClass::Noise,
                    stimulus_level: 0.0,
                    response_scale_id: 1,
                    rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 2,
                    audit_stratum: 1,
                    condition_id: 1,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 1.0,
                    response_scale_id: 1,
                    rt_deadline_ms: None,
                },
            ],
            "sha256:allocation-actions",
        )
        .unwrap()
    }

    fn scenario_set() -> ScenarioSet {
        ScenarioSet::new(
            vec![
                ScenarioDefinition { scenario_id: 10, weight: 0.6 },
                ScenarioDefinition { scenario_id: 20, weight: 0.4 },
            ],
            "sha256:allocation-scenarios",
        )
        .unwrap()
    }

    fn score_config() -> ScoreBoundsConfig {
        ScoreBoundsConfig {
            familywise_alpha: 0.05,
            minimum_replicates: 4,
            ranges: ScoreRanges {
                target_information: ScoreRange { lower: 0.0, upper: 1.0 },
                model_discrimination_information: ScoreRange { lower: 0.0, upper: 1.0 },
                representativeness: ScoreRange { lower: 0.0, upper: 1.0 },
                target_utility: ScoreRange { lower: 0.0, upper: 1.0 },
                participant_burden_seconds: ScoreRange { lower: 0.0, upper: 2.0 },
            },
            score_contract_digest: "sha256:allocation-score".to_owned(),
            source_digest: "sha256:allocation-generator".to_owned(),
            root_seed: 123,
        }
    }

    fn score_receipt(n: usize, action_utilities: [f64; 2], anytime: bool) -> ScoreBoundsReceipt {
        let actions = action_set();
        let scenarios = scenario_set();
        let mut samples = Vec::new();
        for action in actions.actions() {
            for scenario in scenarios.scenarios() {
                let utility = if action.action_id == 1 {
                    action_utilities[0]
                } else {
                    action_utilities[1]
                };
                samples.push(BoundedScoreSamples {
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
        if anytime {
            bound_candidate_score_samples_anytime(&actions, &scenarios, &samples, &score_config())
                .unwrap()
        } else {
            bound_candidate_score_samples(&actions, &scenarios, &samples, &score_config()).unwrap()
        }
    }

    fn config(epsilon: f64, max_total: usize, batch: usize) -> AdaptiveAllocationConfig {
        AdaptiveAllocationConfig {
            policy_version: "spectra-test-allocator-v1".to_owned(),
            score_receipt_digest: "sha256:test-score-receipt".to_owned(),
            root_seed: 77,
            seed_mode: ReplicationSeedMode::CommonRandomNumbersWithinScenario,
            robust_utility: RobustUtility::Maximin,
            epsilon,
            max_total_replications: max_total,
            next_batch_size: batch,
        }
    }

    #[test]
    fn rejects_fixed_look_bounds_for_adaptive_allocation() {
        let receipt = score_receipt(4, [0.8, 0.7], false);
        assert_eq!(
            allocate_next_score_batch(&receipt, &config(0.1, 100, 2)),
            Err(AdaptiveAllocationError::FixedLookBoundsNotAllowed)
        );
    }

    #[test]
    fn requests_more_samples_from_a_competitive_uncertain_cell() {
        let receipt = score_receipt(4, [0.8, 0.7], true);
        let decision = allocate_next_score_batch(&receipt, &config(0.01, 40, 3)).unwrap();
        assert_eq!(decision.disposition, AllocationDisposition::AllocateMore);
        assert_eq!(decision.current_total_replications, 16);
        assert_eq!(decision.competitive_action_ids, vec![1, 2]);
        let request = decision.next_batch.unwrap();
        assert!(decision
            .cell_priorities
            .iter()
            .any(|cell| cell.scenario_id == request.scenario_id
                && cell.priority == decision.selected_cell_priority.unwrap()));
        assert_eq!(decision.seed_mode, ReplicationSeedMode::CommonRandomNumbersWithinScenario);
        assert_eq!(decision.current_cell_sample_counts.len(), 4);
        assert_eq!(decision.seed_method_id, ALLOCATION_SEED_METHOD_ID);
        assert_eq!(request.cells.len(), 2);
        assert!(request.cells.iter().all(|cell| cell.first_replication_index == 4));
        assert!(request.cells.iter().all(|cell| cell.replication_count == 3));
        assert!(request.cells.iter().all(|cell| cell.seeds.len() == 3));
        assert_eq!(request.cells[0].seeds, request.cells[1].seeds);
        assert_eq!(
            request.cells[0].seeds[0],
            derive_replication_seed(
                77,
                request.cells[0].action_id,
                request.scenario_id,
                4,
                ReplicationSeedMode::CommonRandomNumbersWithinScenario,
            )
        );
    }

    #[test]
    fn stops_when_indifference_zone_certificate_is_reached() {
        let receipt = score_receipt(2_000, [0.9, 0.1], true);
        let decision = allocate_next_score_batch(&receipt, &config(0.3, 20_000, 10)).unwrap();
        assert_eq!(decision.disposition, AllocationDisposition::CertifiedStop);
        assert_eq!(
            decision.certificate.status,
            RobustSelectionStatus::CertifiedWithinIndifferenceZone
        );
        assert!(decision.next_batch.is_none());
    }

    #[test]
    fn exhausted_budget_returns_no_selection_certificate() {
        let receipt = score_receipt(4, [0.8, 0.7], true);
        let decision = allocate_next_score_batch(&receipt, &config(0.01, 16, 4)).unwrap();
        assert_eq!(
            decision.disposition,
            AllocationDisposition::BudgetExhaustedWithoutCertificate
        );
        assert_eq!(
            decision.certificate.status,
            RobustSelectionStatus::NoSelectionCertificate
        );
        assert!(decision.next_batch.is_none());
    }

    #[test]
    fn independent_mode_allocates_only_one_action_cell() {
        let receipt = score_receipt(4, [0.8, 0.7], true);
        let mut cfg = config(0.01, 40, 2);
        cfg.seed_mode = ReplicationSeedMode::IndependentPerCell;
        let decision = allocate_next_score_batch(&receipt, &cfg).unwrap();
        assert_eq!(decision.disposition, AllocationDisposition::AllocateMore);
        let request = decision.next_batch.unwrap();
        assert_eq!(request.cells.len(), 1);
        assert_eq!(request.cells[0].replication_count, 2);
        assert_ne!(
            derive_replication_seed(77, 1, request.scenario_id, 4, ReplicationSeedMode::IndependentPerCell),
            derive_replication_seed(77, 2, request.scenario_id, 4, ReplicationSeedMode::IndependentPerCell)
        );
    }

    #[test]
    fn common_random_number_mode_refuses_unequal_per_action_scenario_counts() {
        let fixed = score_receipt(4, [0.8, 0.7], true);
        let actions = action_set();
        let scenarios = scenario_set();
        let mut samples = fixed.samples.clone();
        samples[0].target_information.push(1.0);
        samples[0].model_discrimination_information.push(0.5);
        samples[0].representativeness.push(0.9);
        samples[0].target_utility.push(0.8);
        samples[0].participant_burden_seconds.push(1.0);
        let cfg = ScoreBoundsConfig {
            familywise_alpha: fixed.familywise_alpha,
            minimum_replicates: fixed.minimum_replicates,
            ranges: fixed.ranges,
            score_contract_digest: fixed.score_contract_digest.clone(),
            source_digest: fixed.source_digest.clone(),
            root_seed: fixed.root_seed,
        };
        let variable = bound_candidate_score_samples_anytime(
            &actions,
            &scenarios,
            &samples,
            &cfg,
        )
        .unwrap();
        assert_eq!(
            allocate_next_score_batch(&variable, &config(0.01, 100, 2)),
            Err(AdaptiveAllocationError::CommonRandomNumberCountMismatch)
        );
    }

    #[test]
    fn common_random_number_batch_is_atomic_under_compute_budget() {
        let receipt = score_receipt(4, [0.8, 0.7], true);
        let decision = allocate_next_score_batch(&receipt, &config(0.01, 17, 4)).unwrap();
        assert_eq!(
            decision.disposition,
            AllocationDisposition::BudgetInsufficientForAtomicBatch
        );
        assert!(decision.next_batch.is_none());
    }

    #[test]
    fn common_random_number_seeds_match_across_actions_but_independent_seeds_do_not() {
        let common_a = derive_replication_seed(
            7, 1, 20, 4, ReplicationSeedMode::CommonRandomNumbersWithinScenario,
        );
        let common_b = derive_replication_seed(
            7, 2, 20, 4, ReplicationSeedMode::CommonRandomNumbersWithinScenario,
        );
        assert_eq!(common_a, common_b);
        let independent_a =
            derive_replication_seed(7, 1, 20, 4, ReplicationSeedMode::IndependentPerCell);
        let independent_b =
            derive_replication_seed(7, 2, 20, 4, ReplicationSeedMode::IndependentPerCell);
        assert_ne!(independent_a, independent_b);
    }
}

/// Deterministic simulation seed derivation. The schedule must use the same
/// function and the caller's frozen root seed to replay score generation.
pub fn derive_replication_seed(
    root_seed: u64,
    action_id: u32,
    scenario_id: u16,
    replication_index: u32,
    mode: ReplicationSeedMode,
) -> u64 {
    let mut value = root_seed
        ^ (scenario_id as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ (replication_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if mode == ReplicationSeedMode::IndependentPerCell {
        value ^= (action_id as u64).wrapping_mul(0xA076_1D64_78BD_642F);
    }
    splitmix64_scramble(value)
}

fn splitmix64_scramble(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

