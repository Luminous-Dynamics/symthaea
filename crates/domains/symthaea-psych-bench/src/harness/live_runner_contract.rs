// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compatibility adapter between the historical live runner and the typed
//! live-execution contract.
//!
//! The existing runner always executes one mutable episode and applies reward
//! after each scored trial. This module names that behavior explicitly and
//! refuses any stronger/different execution contract until the runtime can
//! actually enforce it.

use crate::harness::config::BenchmarkConfig;
use crate::harness::live_runner::LoopBenchmarkResult;
use crate::live_execution_contract::{
    LiveExecutionCapabilities, LiveExecutionContract, LiveExecutionContractError,
    LiveExecutionReceipt,
};
use serde::{Deserialize, Serialize};

/// Current hard-coded warmup used by `CognitiveLoopBenchmarkRunner`.
///
/// This remains a compatibility constant until the runner itself exposes its
/// warmup policy through a stable public contract.
pub const LEGACY_LIVE_RUNNER_WARMUP_CYCLES: usize = 100;

/// Raw legacy result paired with the execution receipt that constrains how the
/// result may be interpreted statistically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractedLoopBenchmarkResult {
    pub result: LoopBenchmarkResult,
    pub receipt: LiveExecutionReceipt,
}

/// Adapter failures are typed so callers cannot silently fall back to the
/// historical behavior after requesting a stronger experimental contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveRunnerContractError {
    /// The historical runner cannot enforce the requested semantics.
    UnsupportedByLegacyRunner,
    /// The receipt itself failed the fail-closed execution contract.
    Contract(LiveExecutionContractError),
}

impl From<LiveExecutionContractError> for LiveRunnerContractError {
    fn from(value: LiveExecutionContractError) -> Self {
        Self::Contract(value)
    }
}

/// Exact typed representation of the current historical `run_benchmark()` path.
pub const fn current_legacy_runner_contract() -> LiveExecutionContract {
    LiveExecutionContract::legacy_sequential_learning(LEGACY_LIVE_RUNNER_WARMUP_CYCLES)
}

/// Refuse any requested contract that the historical runner would violate.
///
/// In particular, the current runner cannot satisfy no-reward execution,
/// reset-per-trial identity, frozen learning, or a different warmup policy.
pub fn validate_legacy_runner_request(
    requested: &LiveExecutionContract,
) -> Result<(), LiveRunnerContractError> {
    if requested != &current_legacy_runner_contract() {
        return Err(LiveRunnerContractError::UnsupportedByLegacyRunner);
    }
    requested
        .validate_support(LiveExecutionCapabilities::legacy_current_runner())
        .map_err(LiveRunnerContractError::from)
}

/// Construct a receipt for an already executed historical live-runner result.
///
/// This does not reinterpret trial rows as independent replicates. The receipt
/// always declares `SeededEpisode` through the legacy sequential contract.
pub fn receipt_for_legacy_result(
    result: &LoopBenchmarkResult,
    config: &BenchmarkConfig,
) -> Result<LiveExecutionReceipt, LiveRunnerContractError> {
    let contract = current_legacy_runner_contract();
    validate_legacy_runner_request(&contract)?;
    let learning_observed = result.trials.iter().any(|trial| trial.learning_occurred);

    LiveExecutionReceipt::new(
        result.benchmark.clone(),
        config.label.clone(),
        config.seed,
        result.trials.len(),
        contract,
        LiveExecutionCapabilities::legacy_current_runner(),
        learning_observed,
    )
    .map_err(LiveRunnerContractError::from)
}

#[cfg(feature = "symthaea-backend")]
use crate::harness::live_runner::{CognitiveLoopBenchmarkRunner, LoopDrivable};

/// Explicit-contract execution for the current runner.
///
/// Only the exact historical semantics are accepted in this tranche. Stronger
/// modes return a typed refusal instead of running with weaker semantics.
#[cfg(feature = "symthaea-backend")]
pub trait CognitiveLoopBenchmarkRunnerContractExt {
    fn run_benchmark_with_contract(
        &mut self,
        bench: &dyn LoopDrivable,
        config: &BenchmarkConfig,
        contract: LiveExecutionContract,
    ) -> Result<ContractedLoopBenchmarkResult, LiveRunnerContractError>;
}

#[cfg(feature = "symthaea-backend")]
impl CognitiveLoopBenchmarkRunnerContractExt for CognitiveLoopBenchmarkRunner {
    fn run_benchmark_with_contract(
        &mut self,
        bench: &dyn LoopDrivable,
        config: &BenchmarkConfig,
        contract: LiveExecutionContract,
    ) -> Result<ContractedLoopBenchmarkResult, LiveRunnerContractError> {
        validate_legacy_runner_request(&contract)?;

        // Behavior-authoritative compatibility path: call the existing runner
        // unchanged, then attach a receipt describing what actually happened.
        let result = self.run_benchmark(bench, config);
        let receipt = receipt_for_legacy_result(&result, config)?;
        Ok(ContractedLoopBenchmarkResult { result, receipt })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_execution_contract::{
        LiveExecutionMode, OutcomeFeedbackPolicy, ReplicationUnit, WarmupPolicy,
    };

    #[test]
    fn current_contract_exactly_names_historical_runner_semantics() {
        let contract = current_legacy_runner_contract();
        assert_eq!(contract.mode, LiveExecutionMode::SequentialLearning);
        assert_eq!(contract.outcome_feedback, OutcomeFeedbackPolicy::ApplyReward);
        assert_eq!(
            contract.warmup,
            WarmupPolicy::FixedCycles {
                cycles: LEGACY_LIVE_RUNNER_WARMUP_CYCLES,
            }
        );
        assert!(contract.production_learning_permitted);
        assert_eq!(contract.replication_unit(), ReplicationUnit::SeededEpisode);
        assert!(validate_legacy_runner_request(&contract).is_ok());
    }

    #[test]
    fn no_feedback_request_is_refused_instead_of_silently_rewarding() {
        let requested = LiveExecutionContract {
            mode: LiveExecutionMode::SequentialLearning,
            outcome_feedback: OutcomeFeedbackPolicy::NoOutcomeFeedback,
            warmup: WarmupPolicy::FixedCycles {
                cycles: LEGACY_LIVE_RUNNER_WARMUP_CYCLES,
            },
            production_learning_permitted: true,
        };
        assert_eq!(
            validate_legacy_runner_request(&requested),
            Err(LiveRunnerContractError::UnsupportedByLegacyRunner)
        );
    }

    #[test]
    fn reset_per_trial_request_is_refused() {
        let requested = LiveExecutionContract {
            mode: LiveExecutionMode::ResetPerTrial,
            outcome_feedback: OutcomeFeedbackPolicy::NoOutcomeFeedback,
            warmup: WarmupPolicy::None,
            production_learning_permitted: false,
        };
        assert_eq!(
            validate_legacy_runner_request(&requested),
            Err(LiveRunnerContractError::UnsupportedByLegacyRunner)
        );
    }

    #[test]
    fn frozen_episode_request_is_refused() {
        let requested = LiveExecutionContract {
            mode: LiveExecutionMode::FrozenEpisode,
            outcome_feedback: OutcomeFeedbackPolicy::ObserveOutcomeOnly,
            warmup: WarmupPolicy::FixedCycles { cycles: 100 },
            production_learning_permitted: false,
        };
        assert_eq!(
            validate_legacy_runner_request(&requested),
            Err(LiveRunnerContractError::UnsupportedByLegacyRunner)
        );
    }

    #[test]
    fn changed_warmup_is_refused_because_it_is_not_current_behavior() {
        let requested = LiveExecutionContract::legacy_sequential_learning(20);
        assert_eq!(
            validate_legacy_runner_request(&requested),
            Err(LiveRunnerContractError::UnsupportedByLegacyRunner)
        );
    }

    #[test]
    fn forbidding_learning_is_refused_while_legacy_runner_may_learn() {
        let mut requested = current_legacy_runner_contract();
        requested.production_learning_permitted = false;
        assert_eq!(
            validate_legacy_runner_request(&requested),
            Err(LiveRunnerContractError::UnsupportedByLegacyRunner)
        );
    }
}
