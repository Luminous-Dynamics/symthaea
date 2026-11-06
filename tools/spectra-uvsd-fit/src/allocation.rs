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
    certify_epsilon_optimal_action, counterfactual_regret_upper_bound,
    ActionRobustUtilityInterval, IndependentCounterfactualProjectionIndex,
    RobustSelectionCertificate, RobustSelectionStatus,
};
use super::scenario::{RobustUtility, ScenarioDefinition};
use super::score_bounds::{
    verify_score_bounds_receipt, ScoreBoundsReceipt, ANYTIME_ALPHA_SPENDING_RULE_ID,
    ANYTIME_SCORE_BOUND_METHOD_ID,
};
use super::sequential::OperationalAction;

pub const ALLOCATION_SEED_METHOD_ID: &str = "spectra-splitmix64-seed-v1";
pub const INTERVAL_WIDTH_ALLOCATION_METHOD_ID: &str = "robust-interval-width-allocation-v1";
pub const DECISION_DIRECTED_ALLOCATION_METHOD_ID: &str =
    "robust-certificate-contraction-allocation-v1";

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
    NoCertificateContraction,
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
    /// Policy-specific allocation priority. For v1 this is interval width × tail mass;
    /// for certificate-contraction v1 this is projected regret-gap reduction / batch cost.
    pub priority: f64,
    /// Radius-only reduction in the global robust regret gap for the projected batch.
    pub certificate_gap_contraction: f64,
    /// Fully charged number of score replications in the projected batch.
    pub projected_batch_cost: usize,
    /// Counterfactual post-batch robust regret upper bound under frozen point estimates.
    pub projected_regret_upper_bound: f64,
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
    InvalidScoreReceipt,
    AllocationReceiptRecomputationMismatch,
    CounterfactualCertificateFailure,
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
            Self::InvalidScoreReceipt => "score receipt failed raw-sample recomputation before allocation replay",
            Self::AllocationReceiptRecomputationMismatch => "allocation receipt does not reproduce from the frozen score receipt and allocation configuration",
            Self::CounterfactualCertificateFailure => "counterfactual robust-certificate projection failed",
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
    allocate_next_score_batch_with_policy(receipt, config, false)
}

/// Allocate the next batch by projected contraction of the actual robust certificate gap.
///
/// This is a deterministic radius-only sensitivity heuristic: empirical point
/// estimates are frozen, hypothetical future radii are recomputed at the
/// projected sample count, and the production Maximin/CVaR aggregation is
/// applied to those counterfactual endpoints. It is not EVI or an optimality
/// theorem. The original interval-width allocator remains available as v1.
pub fn allocate_next_score_batch_decision_directed(
    receipt: &ScoreBoundsReceipt,
    config: &AdaptiveAllocationConfig,
) -> Result<AdaptiveAllocationReceipt, AdaptiveAllocationError> {
    allocate_next_score_batch_with_policy(receipt, config, true)
}

fn allocate_next_score_batch_with_policy(
    receipt: &ScoreBoundsReceipt,
    config: &AdaptiveAllocationConfig,
    decision_directed: bool,
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
    let allocation_method_id = if decision_directed {
        DECISION_DIRECTED_ALLOCATION_METHOD_ID
    } else {
        INTERVAL_WIDTH_ALLOCATION_METHOD_ID
    };

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
            allocation_method_id,
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
            allocation_method_id,
            current_total_replications,
            certificate,
            competitive_action_ids,
            AllocationDisposition::BudgetExhaustedWithoutCertificate,
        )?);
    }

    let remaining_budget = config.max_total_replications - current_total_replications;
    if decision_directed {
        let minimum_atomic_cost = match config.seed_mode {
            ReplicationSeedMode::IndependentPerCell => 1,
            ReplicationSeedMode::CommonRandomNumbersWithinScenario => {
                receipt.action_snapshot.len()
            }
        };
        if remaining_budget < minimum_atomic_cost {
            return Ok(build_stop_receipt(
                receipt,
                config,
                allocation_method_id,
                current_total_replications,
                certificate,
                competitive_action_ids,
                AllocationDisposition::BudgetInsufficientForAtomicBatch,
            )?);
        }
    }
    let priorities = if decision_directed {
        compute_decision_directed_cell_priorities(
            receipt,
            &certificate,
            config,
            remaining_budget,
        )?
    } else {
        compute_cell_priorities(receipt, &certificate, config.robust_utility)?
    };
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
    let selected_cell = match actionable.first() {
        Some(cell) => cell,
        None if decision_directed => {
            return Ok(build_stop_receipt(
                receipt,
                config,
                allocation_method_id,
                current_total_replications,
                certificate,
                competitive_action_ids,
                AllocationDisposition::NoCertificateContraction,
            )?);
        }
        None => return Err(AdaptiveAllocationError::NoActionableCell),
    };

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
            allocation_method_id,
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
        method_id: allocation_method_id,
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

/// Deterministically replay and verify an allocation receipt.
///
/// This verifier first reconstructs the score-bound receipt from its raw samples,
/// then reruns the registered allocator with the supplied frozen configuration
/// and compares every field of the resulting allocation receipt. It detects
/// modified priorities, certificates, dispositions, batch requests and seeds.
///
/// This is a replay verifier, not an independent second implementation of the
/// allocation algorithm. Caller-provided digest strings are compared as opaque
/// commitments; this function does not authenticate hashes or signatures.
pub fn verify_adaptive_allocation_receipt(
    score_receipt: &ScoreBoundsReceipt,
    config: &AdaptiveAllocationConfig,
    allocation_receipt: &AdaptiveAllocationReceipt,
) -> Result<(), AdaptiveAllocationError> {
    verify_score_bounds_receipt(score_receipt)
        .map_err(|_| AdaptiveAllocationError::InvalidScoreReceipt)?;

    let expected = match allocation_receipt.method_id {
        INTERVAL_WIDTH_ALLOCATION_METHOD_ID => allocate_next_score_batch(score_receipt, config)?,
        DECISION_DIRECTED_ALLOCATION_METHOD_ID => {
            allocate_next_score_batch_decision_directed(score_receipt, config)?
        }
        _ => return Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch),
    };
    if &expected != allocation_receipt {
        return Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch);
    }
    Ok(())
}

fn build_stop_receipt(
    receipt: &ScoreBoundsReceipt,
    config: &AdaptiveAllocationConfig,
    method_id: &'static str,
    current_total_replications: usize,
    certificate: RobustSelectionCertificate,
    competitive_action_ids: Vec<u32>,
    disposition: AllocationDisposition,
) -> Result<AdaptiveAllocationReceipt, AdaptiveAllocationError> {
    Ok(AdaptiveAllocationReceipt {
        method_id,
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
        let mut action_metrics = metrics_by_action
            .get(&action_interval.action_id)
            .cloned()
            .unwrap_or_default();
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
                certificate_gap_contraction: 0.0,
                projected_batch_cost: 0,
                projected_regret_upper_bound: certificate.regret_upper_bound,
                competitive,
            });
        }
    }
    Ok(priorities)
}


/// Compute radius-only certificate contraction for every competitive cell.
///
/// A CRN candidate projects one atomic batch over every action in the selected
/// scenario. Every action × replication is charged. A cell's score is zero when
/// its projected batch cannot change the global robust regret gap.
fn compute_decision_directed_cell_priorities(
    receipt: &ScoreBoundsReceipt,
    certificate: &RobustSelectionCertificate,
    config: &AdaptiveAllocationConfig,
    remaining_budget: usize,
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
    let mut metrics_by_action: BTreeMap<u32, Vec<&super::score_bounds::ActionScenarioMetricBounds>> =
        BTreeMap::new();
    for metric in &receipt.metrics {
        metrics_by_action
            .entry(metric.action_id)
            .or_default()
            .push(metric);
    }
    let mut priorities = Vec::with_capacity(receipt.metrics.len());
    // Independent mode repeatedly changes only one action × scenario cell.
    // Build one exact index and reuse all untouched action aggregates.
    let independent_projection_index = if config.seed_mode == ReplicationSeedMode::IndependentPerCell {
        Some(
            IndependentCounterfactualProjectionIndex::new(receipt, config.robust_utility)
                .map_err(|_| AdaptiveAllocationError::CounterfactualCertificateFailure)?,
        )
    } else {
        None
    };
    // For CRN, every candidate action in a scenario is the same atomic batch.
    // Cache its exact projection once per scenario instead of recomputing it A times.
    let mut crn_projection_cache =
        BTreeMap::<u16, (f64, f64, usize, f64)>::new();

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

        for metric in action_metrics {
            let sample_count = metric.target_utility.sample_count;
            let mut projection = (
                0.0,
                0.0,
                0,
                certificate.regret_upper_bound,
            );
            if competitive {
                if config.seed_mode == ReplicationSeedMode::CommonRandomNumbersWithinScenario {
                    if let Some(cached) =
                        crn_projection_cache.get(&metric.scenario_id).copied()
                    {
                        projection = cached;
                    } else {
                        projection = project_decision_directed_batch(
                            receipt,
                            certificate,
                            config,
                            metric.action_id,
                            metric.scenario_id,
                            sample_count,
                            remaining_budget,
                            None,
                            None,
                        )?;
                        crn_projection_cache.insert(metric.scenario_id, projection);
                    }
                } else {
                    projection = project_decision_directed_batch(
                        receipt,
                        certificate,
                        config,
                        metric.action_id,
                        metric.scenario_id,
                        sample_count,
                        remaining_budget,
                        independent_projection_index.as_ref(),
                        Some(metric),
                    )?;
                }
            }
            priorities.push(CellPriority {
                action_id: metric.action_id,
                scenario_id: metric.scenario_id,
                sample_count,
                priority: projection.0,
                certificate_gap_contraction: projection.1,
                projected_batch_cost: projection.2,
                projected_regret_upper_bound: projection.3,
                competitive,
            });
        }
    }
    Ok(priorities)
}

/// Returns (priority, certificate-gap contraction, fully charged cost, projected regret bound).
fn project_decision_directed_batch(
    receipt: &ScoreBoundsReceipt,
    certificate: &RobustSelectionCertificate,
    config: &AdaptiveAllocationConfig,
    candidate_action_id: u32,
    scenario_id: u16,
    sample_count: usize,
    remaining_budget: usize,
    independent_projection_index: Option<&IndependentCounterfactualProjectionIndex>,
    candidate_metric: Option<&super::score_bounds::ActionScenarioMetricBounds>,
) -> Result<(f64, f64, usize, f64), AdaptiveAllocationError> {
    let target_action_ids = match config.seed_mode {
        ReplicationSeedMode::IndependentPerCell => vec![candidate_action_id],
        ReplicationSeedMode::CommonRandomNumbersWithinScenario => {
            let mut counts = BTreeMap::new();
            for sample in receipt
                .samples
                .iter()
                .filter(|sample| sample.scenario_id == scenario_id)
            {
                counts.insert(sample.action_id, common_cell_sample_count(sample)?);
            }
            if counts.len() != receipt.action_snapshot.len()
                || counts.values().any(|count| *count != sample_count)
            {
                return Err(AdaptiveAllocationError::CommonRandomNumberCountMismatch);
            }
            receipt
                .action_snapshot
                .iter()
                .map(|action| action.action_id)
                .collect::<Vec<_>>()
        }
    };

    let cost_per_replication = target_action_ids.len();
    let affordable_count = remaining_budget / cost_per_replication;
    let batch_count = affordable_count.min(config.next_batch_size);
    if batch_count == 0 {
        return Ok((0.0, 0.0, 0, certificate.regret_upper_bound));
    }
    let projected_count = sample_count
        .checked_add(batch_count)
        .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)?;
    let mut overrides = Vec::with_capacity(target_action_ids.len());
    for target_action_id in &target_action_ids {
        let target_metric = match config.seed_mode {
            ReplicationSeedMode::IndependentPerCell => {
                let candidate = candidate_metric
                    .ok_or(AdaptiveAllocationError::MissingScoreSampleCell)?;
                if candidate.action_id != *target_action_id
                    || candidate.scenario_id != scenario_id
                {
                    return Err(AdaptiveAllocationError::MissingScoreSampleCell);
                }
                candidate
            }
            ReplicationSeedMode::CommonRandomNumbersWithinScenario => receipt
                .metrics
                .iter()
                .find(|candidate| {
                    candidate.action_id == *target_action_id
                        && candidate.scenario_id == scenario_id
                })
                .ok_or(AdaptiveAllocationError::MissingScoreSampleCell)?,
        };
        if target_metric.target_utility.sample_count != sample_count {
            return Err(AdaptiveAllocationError::CommonRandomNumberCountMismatch);
        }
        let radius = projected_anytime_hoeffding_radius(
            target_metric.target_utility.declared_range,
            projected_count,
            receipt.per_claim_alpha,
        )?;
        let point = target_metric.target_utility.point_estimate;
        let range = target_metric.target_utility.declared_range;
        let lower = (point - radius).max(range.lower);
        let upper = (point + radius).min(range.upper);
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(AdaptiveAllocationError::NonFinitePriority);
        }
        overrides.push((*target_action_id, scenario_id, lower, upper));
    }
    let projected_regret_upper_bound = match config.seed_mode {
        ReplicationSeedMode::IndependentPerCell => {
            if overrides.len() != 1 {
                return Err(AdaptiveAllocationError::CounterfactualCertificateFailure);
            }
            let (action_id, scenario_id, lower, upper) = overrides[0];
            independent_projection_index
                .ok_or(AdaptiveAllocationError::CounterfactualCertificateFailure)?
                .regret_upper_bound_with_override(action_id, scenario_id, lower, upper)
                .map_err(|_| AdaptiveAllocationError::CounterfactualCertificateFailure)?
        }
        ReplicationSeedMode::CommonRandomNumbersWithinScenario => {
            counterfactual_regret_upper_bound(receipt, config.robust_utility, &overrides)
                .map_err(|_| AdaptiveAllocationError::CounterfactualCertificateFailure)?
        }
    };
    let certificate_gap_contraction =
        (certificate.regret_upper_bound - projected_regret_upper_bound).max(0.0);
    let projected_batch_cost = batch_count
        .checked_mul(cost_per_replication)
        .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)?;
    let priority = certificate_gap_contraction / projected_batch_cost as f64;
    if !priority.is_finite() || priority < 0.0 {
        return Err(AdaptiveAllocationError::NonFinitePriority);
    }
    Ok((
        priority,
        certificate_gap_contraction,
        projected_batch_cost,
        projected_regret_upper_bound,
    ))
}

fn projected_anytime_hoeffding_radius(
    range: super::score_bounds::ScoreRange,
    sample_count: usize,
    delta: f64,
) -> Result<f64, AdaptiveAllocationError> {
    let width = range.upper - range.lower;
    if !width.is_finite() || width < 0.0 || sample_count == 0 || !delta.is_finite()
        || delta <= 0.0 || delta >= 1.0
    {
        return Err(AdaptiveAllocationError::NonFinitePriority);
    }
    if width == 0.0 {
        return Ok(0.0);
    }
    let next_count = sample_count
        .checked_add(1)
        .ok_or(AdaptiveAllocationError::ReplicationCountOverflow)?;
    let factor = ((-delta.ln() + (sample_count as f64).ln() + (next_count as f64).ln())
        / (2.0 * sample_count as f64))
        .sqrt();
    if !factor.is_finite() || factor < 0.0 {
        return Err(AdaptiveAllocationError::NonFinitePriority);
    }
    Ok(if factor >= 1.0 { width } else { width * factor })
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

 
    fn asymmetric_certificate_receipt() -> ScoreBoundsReceipt {
        let actions = action_set();
        let scenarios = scenario_set();
        let rows = [
            (1_u32, 10_u16, 20_851_usize, 0.825_f64),
            (1_u32, 20_u16, 4_609_usize, 0.870_f64),
            (2_u32, 10_u16, 58_usize, 0.550_f64),
            (2_u32, 20_u16, 133_usize, 0.595_f64),
        ];
        let samples = rows
            .iter()
            .map(|(action_id, scenario_id, n, utility)| BoundedScoreSamples {
                action_id: *action_id,
                scenario_id: *scenario_id,
                operational_mapping_qualified: true,
                model_adequacy_qualified: true,
                target_information: vec![0.5; *n],
                model_discrimination_information: vec![0.5; *n],
                representativeness: vec![0.9; *n],
                target_utility: vec![*utility; *n],
                participant_burden_seconds: vec![1.0; *n],
            })
            .collect::<Vec<_>>();
        bound_candidate_score_samples_anytime(
            &actions,
            &scenarios,
            &samples,
            &score_config(),
        )
        .unwrap()
    }

    #[test]
    fn decision_directed_policy_allocates_to_certificate_sensitive_endpoint() {
        let receipt = asymmetric_certificate_receipt();
        let mut cfg = config(0.01, 30_000, 1);
        cfg.seed_mode = ReplicationSeedMode::IndependentPerCell;

        let width = allocate_next_score_batch(&receipt, &cfg).unwrap();
        let width_batch = width.next_batch.unwrap();
        assert_eq!(width_batch.cells[0].action_id, 2);
        assert_eq!(width_batch.scenario_id, 10);

        let directed = allocate_next_score_batch_decision_directed(&receipt, &cfg).unwrap();
        assert_eq!(directed.method_id, DECISION_DIRECTED_ALLOCATION_METHOD_ID);
        assert_eq!(directed.disposition, AllocationDisposition::AllocateMore);
        let directed_batch = directed.next_batch.as_ref().unwrap();
        assert_eq!(directed_batch.cells[0].action_id, 1);
        assert_eq!(directed_batch.scenario_id, 10);
        let selected = directed
            .cell_priorities
            .iter()
            .find(|cell| cell.action_id == 1 && cell.scenario_id == 10)
            .unwrap();
        assert!(selected.certificate_gap_contraction > 0.0);
        assert_eq!(selected.projected_batch_cost, 1);
        assert!(selected.projected_regret_upper_bound < directed.certificate.regret_upper_bound);
        assert_eq!(
            verify_adaptive_allocation_receipt(&receipt, &cfg, &directed),
            Ok(())
        );
    }


    #[test]
    fn decision_directed_crn_cvar_charges_every_action_replication() {
        let receipt = score_receipt(1_000, [0.85, 0.70], true);
        let mut cfg = config(0.01, 5_000, 2);
        cfg.robust_utility = RobustUtility::LowerTailCvar { tail_mass: 0.5 };
        let allocation =
            allocate_next_score_batch_decision_directed(&receipt, &cfg).unwrap();
        assert_eq!(allocation.disposition, AllocationDisposition::AllocateMore);
        let batch = allocation.next_batch.as_ref().unwrap();
        assert_eq!(batch.seed_mode, ReplicationSeedMode::CommonRandomNumbersWithinScenario);
        assert_eq!(batch.cells.len(), 2);
        assert_eq!(batch.cells[0].replication_count, 2);
        assert_eq!(batch.cells[1].replication_count, 2);
        assert_eq!(batch.cells[0].seeds, batch.cells[1].seeds);

        let selected = allocation
            .cell_priorities
            .iter()
            .find(|cell| {
                cell.action_id == batch.cells[0].action_id
                    && cell.scenario_id == batch.scenario_id
            })
            .unwrap();
        assert_eq!(selected.projected_batch_cost, 4);
        assert!(selected.certificate_gap_contraction > 0.0);
        assert!(
            (selected.priority - selected.certificate_gap_contraction / 4.0).abs() < 1.0e-12
        );
        assert_eq!(
            verify_adaptive_allocation_receipt(&receipt, &cfg, &allocation),
            Ok(())
        );
    }



    #[test]
    fn crn_projection_cache_matches_direct_projection_for_every_action_candidate() {
        let receipt = score_receipt(1_000, [0.85, 0.70], true);
        let mut cfg = config(0.01, 5_000, 2);
        cfg.robust_utility = RobustUtility::LowerTailCvar { tail_mass: 0.5 };
        let allocation =
            allocate_next_score_batch_decision_directed(&receipt, &cfg).unwrap();
        let remaining_budget = cfg.max_total_replications - allocation.current_total_replications;

        for scenario_id in [10_u16, 20_u16] {
            let candidates = allocation
                .cell_priorities
                .iter()
                .filter(|cell| cell.scenario_id == scenario_id && cell.competitive)
                .collect::<Vec<_>>();
            assert!(!candidates.is_empty());
            let first = candidates[0];
            let cached_tuple = (
                first.priority,
                first.certificate_gap_contraction,
                first.projected_batch_cost,
                first.projected_regret_upper_bound,
            );
            for candidate in candidates {
                let direct = project_decision_directed_batch(
                    &receipt,
                    &allocation.certificate,
                    &cfg,
                    candidate.action_id,
                    scenario_id,
                    candidate.sample_count,
                    remaining_budget,
                    None,
                    None,
                )
                .unwrap();
                assert_eq!(direct, cached_tuple);
            }
        }
    }

    #[test]
    fn decision_directed_crn_refuses_partial_atomic_batch() {
        let receipt = score_receipt(1_000, [0.85, 0.70], true);
        let mut cfg = config(0.01, 4_001, 2);
        cfg.robust_utility = RobustUtility::LowerTailCvar { tail_mass: 0.5 };
        let allocation =
            allocate_next_score_batch_decision_directed(&receipt, &cfg).unwrap();
        assert_eq!(
            allocation.disposition,
            AllocationDisposition::BudgetInsufficientForAtomicBatch
        );
        assert!(allocation.next_batch.is_none());
        assert_eq!(
            verify_adaptive_allocation_receipt(&receipt, &cfg, &allocation),
            Ok(())
        );
    }

    #[test]
    fn decision_directed_policy_abstains_when_no_projected_batch_changes_the_gap() {
        // Both scenarios tie at the full-range Hoeffding cap. Contracting only
        // one scenario leaves the other tied endpoint controlling the Maximin
        // interval, so no candidate batch changes the global regret gap.
        let receipt = score_receipt(4, [0.8, 0.7], true);
        let cfg = config(0.01, 200, 1);
        let allocation =
            allocate_next_score_batch_decision_directed(&receipt, &cfg).unwrap();
        assert_eq!(
            allocation.disposition,
            AllocationDisposition::NoCertificateContraction
        );
        assert!(allocation.next_batch.is_none());
        assert!(allocation.cell_priorities.iter().all(|cell| cell.priority == 0.0));
        assert_eq!(
            verify_adaptive_allocation_receipt(&receipt, &cfg, &allocation),
            Ok(())
        );
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
    fn decision_directed_replay_rejects_tampered_projection_fields() {
        let receipt = asymmetric_certificate_receipt();
        let mut cfg = config(0.01, 30_000, 1);
        cfg.seed_mode = ReplicationSeedMode::IndependentPerCell;
        let mut allocation = allocate_next_score_batch_decision_directed(&receipt, &cfg).unwrap();
        allocation.cell_priorities[0].certificate_gap_contraction += 0.001;
        assert_eq!(
            verify_adaptive_allocation_receipt(&receipt, &cfg, &allocation),
            Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch)
        );
    }


    #[test]
    fn decision_directed_projection_rejects_overflow_and_non_finite_inputs() {
        let range = super::super::score_bounds::ScoreRange {
            lower: 0.0,
            upper: 1.0,
        };
        assert_eq!(
            projected_anytime_hoeffding_radius(range, usize::MAX, 0.01),
            Err(AdaptiveAllocationError::ReplicationCountOverflow)
        );
        assert_eq!(
            projected_anytime_hoeffding_radius(range, 10, f64::NAN),
            Err(AdaptiveAllocationError::NonFinitePriority)
        );
        assert_eq!(
            projected_anytime_hoeffding_radius(
                super::super::score_bounds::ScoreRange {
                    lower: 0.0,
                    upper: f64::INFINITY,
                },
                10,
                0.01,
            ),
            Err(AdaptiveAllocationError::NonFinitePriority)
        );
    }

    #[test]
    fn allocation_receipts_replay_for_batch_and_stop_dispositions() {
        let receipt = score_receipt(4, [0.8, 0.7], true);
        let cfg = config(0.01, 40, 3);
        let allocation = allocate_next_score_batch(&receipt, &cfg).unwrap();
        assert_eq!(allocation.disposition, AllocationDisposition::AllocateMore);
        assert_eq!(
            verify_adaptive_allocation_receipt(&receipt, &cfg, &allocation),
            Ok(())
        );

        let stop_score = score_receipt(2_000, [0.9, 0.1], true);
        let stop_cfg = config(0.3, 20_000, 10);
        let stop = allocate_next_score_batch(&stop_score, &stop_cfg).unwrap();
        assert_eq!(stop.disposition, AllocationDisposition::CertifiedStop);
        assert_eq!(
            verify_adaptive_allocation_receipt(&stop_score, &stop_cfg, &stop),
            Ok(())
        );
    }

    #[test]
    fn allocation_receipt_replay_rejects_mutated_priorities_seeds_and_disposition() {
        let score = score_receipt(4, [0.8, 0.7], true);
        let cfg = config(0.01, 40, 3);
        let valid = allocate_next_score_batch(&score, &cfg).unwrap();

        let mut bad_priority = valid.clone();
        bad_priority.cell_priorities[0].priority += 0.01;
        assert_eq!(
            verify_adaptive_allocation_receipt(&score, &cfg, &bad_priority),
            Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch)
        );

        let mut bad_seed = valid.clone();
        bad_seed.next_batch.as_mut().unwrap().cells[0].seeds[0] ^= 1;
        assert_eq!(
            verify_adaptive_allocation_receipt(&score, &cfg, &bad_seed),
            Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch)
        );

        let mut bad_disposition = valid.clone();
        bad_disposition.disposition = AllocationDisposition::BudgetExhaustedWithoutCertificate;
        assert_eq!(
            verify_adaptive_allocation_receipt(&score, &cfg, &bad_disposition),
            Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch)
        );

        let mut bad_competitors = valid.clone();
        bad_competitors.competitive_action_ids.clear();
        assert_eq!(
            verify_adaptive_allocation_receipt(&score, &cfg, &bad_competitors),
            Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch)
        );

        let mut bad_digest = valid.clone();
        bad_digest.score_receipt_digest.push_str("-tampered");
        assert_eq!(
            verify_adaptive_allocation_receipt(&score, &cfg, &bad_digest),
            Err(AdaptiveAllocationError::AllocationReceiptRecomputationMismatch)
        );

        let mut changed_config = cfg.clone();
        changed_config.root_seed ^= 1;
        assert!(verify_adaptive_allocation_receipt(&score, &changed_config, &valid).is_err());
    }

    #[test]
    fn allocation_receipt_replay_rejects_tampered_raw_score_samples() {
        let mut score = score_receipt(4, [0.8, 0.7], true);
        let cfg = config(0.01, 40, 3);
        let valid = allocate_next_score_batch(&score, &cfg).unwrap();

        score.samples[0].target_information[0] += 0.01;
        assert_eq!(
            verify_adaptive_allocation_receipt(&score, &cfg, &valid),
            Err(AdaptiveAllocationError::InvalidScoreReceipt)
        );
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

