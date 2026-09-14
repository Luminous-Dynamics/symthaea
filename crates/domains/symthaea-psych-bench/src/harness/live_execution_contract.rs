// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit experimental semantics for live cognitive-loop benchmark execution.
//!
//! This module does not change `live_runner::run_benchmark()` behavior. It
//! defines the authority-bearing contract that future live-loop execution paths
//! must satisfy before they may call themselves reset-per-trial or frozen.
//!
//! Core invariants:
//!
//! - static capability evaluation != sequential learning evaluation;
//! - trial rows inside one mutable episode are not independent replicates;
//! - outcome feedback and production learning are separate policy axes;
//! - unsupported reset/freeze semantics fail closed;
//! - receipts serialize the declared replication unit and observed learning.

use serde::{Deserialize, Serialize};

/// Stable schema identifier for serialized live-execution receipts.
pub const LIVE_EXECUTION_RECEIPT_SCHEMA_VERSION: &str = "psych-live-execution-v1";

/// State-persistence semantics for one scored benchmark execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveExecutionMode {
    /// Every scored trial begins from the same proven initial state.
    ResetPerTrial,
    /// One stateful episode persists, but production learning is frozen.
    FrozenEpisode,
    /// One mutable episode persists and production learning may occur.
    SequentialLearning,
}

/// How the benchmark outcome is exposed back to the cognitive system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeFeedbackPolicy {
    /// Correctness is scored externally and never fed back to the system.
    NoOutcomeFeedback,
    /// Outcome may be observed by the evaluation layer, but no reward is applied.
    ObserveOutcomeOnly,
    /// Outcome is converted to a reward signal and applied to the live system.
    ApplyReward,
}

/// Independent experimental unit represented by one result row/receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationUnit {
    Trial,
    SeededEpisode,
}

/// Warmup semantics are explicit because warmup mutates the live service state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WarmupPolicy {
    None,
    FixedCycles { cycles: usize },
}

/// Requested execution semantics for a live benchmark.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveExecutionContract {
    pub mode: LiveExecutionMode,
    pub outcome_feedback: OutcomeFeedbackPolicy,
    pub warmup: WarmupPolicy,
    /// Whether production learning/replay/Q-state mutation is permitted.
    pub production_learning_permitted: bool,
}

impl LiveExecutionContract {
    /// Current historical `run_benchmark()` semantics.
    ///
    /// This is intentionally explicit rather than inferred from implementation
    /// details: one service persists, outcomes become rewards, and learning may
    /// occur across the seeded episode.
    pub const fn legacy_sequential_learning(warmup_cycles: usize) -> Self {
        Self {
            mode: LiveExecutionMode::SequentialLearning,
            outcome_feedback: OutcomeFeedbackPolicy::ApplyReward,
            warmup: WarmupPolicy::FixedCycles {
                cycles: warmup_cycles,
            },
            production_learning_permitted: true,
        }
    }

    /// Statistical replication unit implied by the execution mode.
    pub const fn replication_unit(&self) -> ReplicationUnit {
        match self.mode {
            LiveExecutionMode::ResetPerTrial => ReplicationUnit::Trial,
            LiveExecutionMode::FrozenEpisode | LiveExecutionMode::SequentialLearning => {
                ReplicationUnit::SeededEpisode
            }
        }
    }

    /// Validate that the runtime has enough authority to advertise this mode.
    pub fn validate_support(
        &self,
        capabilities: LiveExecutionCapabilities,
    ) -> Result<(), LiveExecutionContractError> {
        match self.mode {
            LiveExecutionMode::ResetPerTrial if !capabilities.reset_identity_established => {
                Err(LiveExecutionContractError::ResetIdentityNotEstablished)
            }
            LiveExecutionMode::FrozenEpisode if !capabilities.learning_freeze_enforced => {
                Err(LiveExecutionContractError::LearningFreezeNotEnforced)
            }
            LiveExecutionMode::FrozenEpisode if self.production_learning_permitted => {
                Err(LiveExecutionContractError::FrozenEpisodePermitsLearning)
            }
            _ => Ok(()),
        }
    }
}

/// Runtime capabilities that must be established independently of the label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveExecutionCapabilities {
    /// The same declared initial state can be reconstructed/restored before
    /// each scored trial and that identity has been checked.
    pub reset_identity_established: bool,
    /// Production learner/replay/Q-state mutation is mechanically frozen.
    pub learning_freeze_enforced: bool,
}

impl LiveExecutionCapabilities {
    /// Capabilities established by the current legacy live runner.
    ///
    /// It supports sequential execution, but does not yet prove reset identity
    /// or a production-wide learning freeze.
    pub const fn legacy_current_runner() -> Self {
        Self {
            reset_identity_established: false,
            learning_freeze_enforced: false,
        }
    }
}

/// Fail-closed reasons a requested execution label cannot be established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveExecutionContractError {
    ResetIdentityNotEstablished,
    LearningFreezeNotEnforced,
    FrozenEpisodePermitsLearning,
    ObservedLearningWasForbidden,
    UnsupportedReceiptSchema,
    ReplicationUnitMismatch,
}

/// Serialized evidence receipt for one live benchmark execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveExecutionReceipt {
    pub schema_version: String,
    pub benchmark: String,
    pub config_label: Option<String>,
    pub benchmark_seed: u64,
    pub trial_count: usize,
    pub contract: LiveExecutionContract,
    pub replication_unit: ReplicationUnit,
    pub production_learning_observed: bool,
    pub reset_identity_established: bool,
    pub learning_freeze_enforced: bool,
}

impl LiveExecutionReceipt {
    /// Construct only from an execution contract that the supplied runtime
    /// capabilities can actually support.
    pub fn new(
        benchmark: impl Into<String>,
        config_label: Option<String>,
        benchmark_seed: u64,
        trial_count: usize,
        contract: LiveExecutionContract,
        capabilities: LiveExecutionCapabilities,
        production_learning_observed: bool,
    ) -> Result<Self, LiveExecutionContractError> {
        contract.validate_support(capabilities)?;
        if production_learning_observed && !contract.production_learning_permitted {
            return Err(LiveExecutionContractError::ObservedLearningWasForbidden);
        }

        Ok(Self {
            schema_version: LIVE_EXECUTION_RECEIPT_SCHEMA_VERSION.to_string(),
            benchmark: benchmark.into(),
            config_label,
            benchmark_seed,
            trial_count,
            replication_unit: contract.replication_unit(),
            contract,
            production_learning_observed,
            reset_identity_established: capabilities.reset_identity_established,
            learning_freeze_enforced: capabilities.learning_freeze_enforced,
        })
    }

    /// Revalidate a deserialized receipt before authority-bearing use.
    pub fn validate(&self) -> Result<(), LiveExecutionContractError> {
        if self.schema_version != LIVE_EXECUTION_RECEIPT_SCHEMA_VERSION {
            return Err(LiveExecutionContractError::UnsupportedReceiptSchema);
        }
        if self.replication_unit != self.contract.replication_unit() {
            return Err(LiveExecutionContractError::ReplicationUnitMismatch);
        }

        let capabilities = LiveExecutionCapabilities {
            reset_identity_established: self.reset_identity_established,
            learning_freeze_enforced: self.learning_freeze_enforced,
        };
        self.contract.validate_support(capabilities)?;

        if self.production_learning_observed && !self.contract.production_learning_permitted {
            return Err(LiveExecutionContractError::ObservedLearningWasForbidden);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_contract_names_current_mutable_episode_semantics() {
        let contract = LiveExecutionContract::legacy_sequential_learning(100);
        assert_eq!(contract.mode, LiveExecutionMode::SequentialLearning);
        assert_eq!(contract.outcome_feedback, OutcomeFeedbackPolicy::ApplyReward);
        assert!(contract.production_learning_permitted);
        assert_eq!(contract.replication_unit(), ReplicationUnit::SeededEpisode);
        assert_eq!(contract.warmup, WarmupPolicy::FixedCycles { cycles: 100 });
        assert!(
            contract
                .validate_support(LiveExecutionCapabilities::legacy_current_runner())
                .is_ok()
        );
    }

    #[test]
    fn reset_per_trial_fails_closed_without_reset_identity() {
        let contract = LiveExecutionContract {
            mode: LiveExecutionMode::ResetPerTrial,
            outcome_feedback: OutcomeFeedbackPolicy::NoOutcomeFeedback,
            warmup: WarmupPolicy::None,
            production_learning_permitted: false,
        };
        assert_eq!(
            contract.validate_support(LiveExecutionCapabilities::legacy_current_runner()),
            Err(LiveExecutionContractError::ResetIdentityNotEstablished)
        );
    }

    #[test]
    fn frozen_episode_fails_closed_without_learning_freeze() {
        let contract = LiveExecutionContract {
            mode: LiveExecutionMode::FrozenEpisode,
            outcome_feedback: OutcomeFeedbackPolicy::ObserveOutcomeOnly,
            warmup: WarmupPolicy::FixedCycles { cycles: 20 },
            production_learning_permitted: false,
        };
        assert_eq!(
            contract.validate_support(LiveExecutionCapabilities::legacy_current_runner()),
            Err(LiveExecutionContractError::LearningFreezeNotEnforced)
        );
    }

    #[test]
    fn frozen_episode_cannot_permit_production_learning() {
        let contract = LiveExecutionContract {
            mode: LiveExecutionMode::FrozenEpisode,
            outcome_feedback: OutcomeFeedbackPolicy::ApplyReward,
            warmup: WarmupPolicy::None,
            production_learning_permitted: true,
        };
        let capabilities = LiveExecutionCapabilities {
            reset_identity_established: false,
            learning_freeze_enforced: true,
        };
        assert_eq!(
            contract.validate_support(capabilities),
            Err(LiveExecutionContractError::FrozenEpisodePermitsLearning)
        );
    }

    #[test]
    fn reset_per_trial_receipt_declares_trial_replication() {
        let contract = LiveExecutionContract {
            mode: LiveExecutionMode::ResetPerTrial,
            outcome_feedback: OutcomeFeedbackPolicy::NoOutcomeFeedback,
            warmup: WarmupPolicy::None,
            production_learning_permitted: false,
        };
        let capabilities = LiveExecutionCapabilities {
            reset_identity_established: true,
            learning_freeze_enforced: false,
        };
        let receipt = LiveExecutionReceipt::new(
            "StaticCapability",
            Some("holdout".to_string()),
            42,
            64,
            contract,
            capabilities,
            false,
        )
        .unwrap();
        assert_eq!(receipt.replication_unit, ReplicationUnit::Trial);
        assert!(receipt.validate().is_ok());
    }

    #[test]
    fn sequential_receipt_declares_seeded_episode_replication() {
        let contract = LiveExecutionContract::legacy_sequential_learning(100);
        let receipt = LiveExecutionReceipt::new(
            "OnlineLearning",
            None,
            7,
            200,
            contract,
            LiveExecutionCapabilities::legacy_current_runner(),
            true,
        )
        .unwrap();
        assert_eq!(receipt.replication_unit, ReplicationUnit::SeededEpisode);
        assert!(receipt.validate().is_ok());
    }

    #[test]
    fn observed_learning_when_forbidden_is_rejected() {
        let contract = LiveExecutionContract {
            mode: LiveExecutionMode::ResetPerTrial,
            outcome_feedback: OutcomeFeedbackPolicy::NoOutcomeFeedback,
            warmup: WarmupPolicy::None,
            production_learning_permitted: false,
        };
        let capabilities = LiveExecutionCapabilities {
            reset_identity_established: true,
            learning_freeze_enforced: false,
        };
        assert!(matches!(
            LiveExecutionReceipt::new(
                "StaticCapability",
                None,
                11,
                10,
                contract,
                capabilities,
                true,
            ),
            Err(LiveExecutionContractError::ObservedLearningWasForbidden)
        ));
    }

    #[test]
    fn feedback_policy_does_not_silently_define_learning_permission() {
        let contract = LiveExecutionContract {
            mode: LiveExecutionMode::SequentialLearning,
            outcome_feedback: OutcomeFeedbackPolicy::ApplyReward,
            warmup: WarmupPolicy::None,
            production_learning_permitted: false,
        };
        assert_eq!(contract.outcome_feedback, OutcomeFeedbackPolicy::ApplyReward);
        assert!(!contract.production_learning_permitted);
        assert!(
            contract
                .validate_support(LiveExecutionCapabilities::legacy_current_runner())
                .is_ok()
        );
    }

    #[test]
    fn receipt_serialization_preserves_execution_semantics() {
        let receipt = LiveExecutionReceipt::new(
            "OnlineLearning",
            Some("seed-99".to_string()),
            99,
            20,
            LiveExecutionContract::legacy_sequential_learning(100),
            LiveExecutionCapabilities::legacy_current_runner(),
            true,
        )
        .unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: LiveExecutionReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, receipt);
        assert!(decoded.validate().is_ok());
    }

    #[test]
    fn missing_mode_fails_deserialization_instead_of_defaulting() {
        let json = r#"{
            "outcome_feedback":"apply_reward",
            "warmup":{"kind":"fixed_cycles","cycles":100},
            "production_learning_permitted":true
        }"#;
        let decoded = serde_json::from_str::<LiveExecutionContract>(json);
        assert!(decoded.is_err());
    }

    #[test]
    fn tampered_replication_unit_fails_validation() {
        let mut receipt = LiveExecutionReceipt::new(
            "OnlineLearning",
            None,
            5,
            8,
            LiveExecutionContract::legacy_sequential_learning(100),
            LiveExecutionCapabilities::legacy_current_runner(),
            false,
        )
        .unwrap();
        receipt.replication_unit = ReplicationUnit::Trial;
        assert_eq!(
            receipt.validate(),
            Err(LiveExecutionContractError::ReplicationUnitMismatch)
        );
    }
}
