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

/// A deterministic request for additional simulations for one action × scenario cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationBatchRequest {
    pub action_id: u32,
    pub scenario_id: u16,
    /// First zero-based per-cell replication index to generate.
    pub first_replication_index: u32,
    pub replication_count: u32,
    /// One deterministic seed per requested replication, in index order.
    pub seeds: Vec<u64>,
    pub seed_mode: ReplicationSeedMode,
}

/// Why the allocation procedure stopped or continues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationDisposition {
    CertifiedStop,
    BudgetExhaustedWithoutCertificate,
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
    NonFinitePriority,
    SeedIndexOverflow,
}

impl fmt::Display for AdaptiveAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingPolicyVersion => "adaptive allocation policy version must not be empty",
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
        ));
    }

    if current_total_replications >= config.max_total_replications {
        return Ok(build_stop_receipt(
            receipt,
            config,
            current_total_replications,
            certificate,
            competitive_action_ids,
            AllocationDisposition::BudgetExhaustedWithoutCertificate,
        ));
    }

    let remaining_budget = config.max_total_replications - current_total_replications;
    let requested_count = remaining_budget.min(config.next_batch_size);
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

    let first_index = u32::try_from(selected_cell.sample_count)
        .map_err(|_| AdaptiveAllocationError::SeedIndexOverflow)?;
    let last_index_exclusive = selected_cell
        .sample_count
        .checked_add(requested_count)
        .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)?;
    if last_index_exclusive > (u32::MAX as usize) + 1 {
        return Err(AdaptiveAllocationError::SeedIndexOverflow);
    }
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
            selected_cell.action_id,
            selected_cell.scenario_id,
            replication_index,
            config.seed_mode,
        ));
    }

    let next_batch = SimulationBatchRequest {
        action_id: selected_cell.action_id,
        scenario_id: selected_cell.scenario_id,
        first_replication_index: first_index,
        replication_count: requested_count as u32,
        seeds,
        seed_mode: config.seed_mode,
    };
    let selected_cell_priority = Some(selected_cell.priority);

    Ok(AdaptiveAllocationReceipt {
        method_id: "robust-interval-width-allocation-v1",
        policy_version: config.policy_version.clone(),
        score_bound_method_id: receipt.method_id,
        score_contract_digest: receipt.score_contract_digest.clone(),
        source_digest: receipt.source_digest.clone(),
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
) -> AdaptiveAllocationReceipt {
    AdaptiveAllocationReceipt {
        method_id: "robust-interval-width-allocation-v1",
        policy_version: config.policy_version.clone(),
        score_bound_method_id: receipt.method_id,
        score_contract_digest: receipt.score_contract_digest.clone(),
        source_digest: receipt.source_digest.clone(),
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
    }
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
    let scenario_weights: BTreeMap<u16, f64> = receipt
        .scenarios
        .iter()
        .map(|scenario| (scenario.scenario_id, scenario.weight))
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
            let scenario_weight = scenario_weights
                .get(&metric.scenario_id)
                .copied()
                .ok_or(AdaptiveAllocationError::InvalidScenarioWeights)?;
                let priority = width * tail_mass;
            if !priority.is_finite() || priority < 0.0 {
                return Err(AdaptiveAllocationError::NonFinitePriority);
            }
            let _ = scenario_weight;
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
    let mut seen = BTreeMap::new();
    for sample in &receipt.samples {
        let count = common_cell_sample_count(sample)?;
        let key = (sample.action_id, sample.scenario_id);
        if seen.insert(key, count).is_some() {
            return Err(AdaptiveAllocationError::MissingScoreSampleCell);
        }
        let metric = metric_by_pair
            .get(&key)
            .copied()
            .ok_or(AdaptiveAllocationError::MissingScoreSampleCell)?;
        if metric.target_utility.sample_count != count {
            return Err(AdaptiveAllocationError::SampleMetricCountMismatch);
        }
    }
    if seen.len() != expected {
        return Err(AdaptiveAllocationError::MissingScoreSampleCell);
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

