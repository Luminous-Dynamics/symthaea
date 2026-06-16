// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Constraint Gate (MAGI Loop Step 3.5 - UPGRADE B)
//!
//! Safety gate that controls execution mode before autonomous action.
//!
//! ## The Problem
//!
//! An AGI system that learns quickly can also make unsafe decisions quickly.
//! The EFE (Expected Free Energy) loop can become capable before it's safe.
//!
//! ## The Solution
//!
//! The ConstraintGate ensures:
//! 1. **High-risk actions require supervision** - No destructive actions without human approval
//! 2. **Poor calibration forces exploration** - If the system doesn't know itself, dry-run first
//! 3. **Graduated autonomy** - Trust is earned through demonstrated calibration
//!
//! ## Execution Modes
//!
//! - **Autonomous**: Full execution without human intervention
//! - **DryRun**: Execute in simulation/preview mode first
//! - **Supervised**: Require human approval before execution
//!
//! ## Design Principle
//!
//! > "The first strong behavior must not be 'unsafe cleverness.'"

use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::calibration::BrierScoreTracker;
use super::world_prediction::{RiskTier, WorldActionContext};

// ═══════════════════════════════════════════════════════════════════════════════
// EXECUTION MODE
// ═══════════════════════════════════════════════════════════════════════════════

/// Mode of execution for an action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Full autonomous execution without human intervention
    Autonomous,

    /// Dry-run/simulation first, then optionally execute
    DryRun {
        /// Reason for dry-run
        reason: DryRunReason,
    },

    /// Require explicit human approval before execution
    Supervised {
        /// Reason supervision is required
        reason: SupervisionReason,
    },
}

impl ExecutionMode {
    /// Is this mode autonomous?
    pub fn is_autonomous(&self) -> bool {
        matches!(self, Self::Autonomous)
    }

    /// Is this mode requiring supervision?
    pub fn is_supervised(&self) -> bool {
        matches!(self, Self::Supervised { .. })
    }

    /// Is this a dry-run mode?
    pub fn is_dry_run(&self) -> bool {
        matches!(self, Self::DryRun { .. })
    }

    /// Get human-readable description
    pub fn description(&self) -> String {
        match self {
            Self::Autonomous => "Autonomous execution permitted".to_string(),
            Self::DryRun { reason } => format!("Dry-run required: {}", reason.description()),
            Self::Supervised { reason } => {
                format!("Supervision required: {}", reason.description())
            }
        }
    }
}

/// Reason for requiring dry-run mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DryRunReason {
    /// Calibration error too high
    PoorCalibration,
    /// Domain is new/unexplored
    UnexploredDomain,
    /// System requested exploration
    ExploratoryMode,
    /// Action involves state modification
    StateModifying,
}

impl DryRunReason {
    fn description(&self) -> &'static str {
        match self {
            Self::PoorCalibration => "System calibration is below threshold",
            Self::UnexploredDomain => "Domain has insufficient prediction history",
            Self::ExploratoryMode => "System is in exploration/learning mode",
            Self::StateModifying => "Action modifies state and requires preview",
        }
    }
}

/// Reason for requiring supervision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupervisionReason {
    /// Action has high risk tier
    HighRisk,
    /// Action is potentially destructive
    Destructive,
    /// Action is critical/irreversible
    Critical,
    /// System is in supervised-only mode
    ForcedSupervision,
    /// Domain has history of failures
    DomainUnsafe,
}

impl SupervisionReason {
    fn description(&self) -> &'static str {
        match self {
            Self::HighRisk => "Action risk tier exceeds autonomous threshold",
            Self::Destructive => "Action may have destructive consequences",
            Self::Critical => "Action has critical/irreversible effects",
            Self::ForcedSupervision => "System is in supervised-only mode",
            Self::DomainUnsafe => "Domain has history of unsafe failures",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT GATE CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the constraint gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintGateConfig {
    /// Risk tier threshold requiring supervision
    /// Actions at or above this tier require human approval
    pub supervision_threshold: RiskTier,

    /// ECE (Expected Calibration Error) threshold for dry-run
    /// If ECE is above this, force dry-run mode
    pub calibration_threshold: f64,

    /// Minimum predictions needed before autonomous execution allowed
    pub min_predictions_for_autonomy: usize,

    /// Whether to force dry-run for all state-modifying actions
    pub always_preview_state_changes: bool,

    /// Whether to force supervision for all destructive actions
    pub always_supervise_destructive: bool,

    /// Global override: force supervision for ALL actions
    pub force_supervised_mode: bool,

    /// Domains that should always require supervision
    pub always_supervised_domains: Vec<String>,

    /// Minimum accuracy for autonomous execution
    pub min_accuracy_for_autonomy: f64,
}

impl Default for ConstraintGateConfig {
    fn default() -> Self {
        Self {
            supervision_threshold: RiskTier::Destructive,
            calibration_threshold: 0.15, // ECE > 15% triggers dry-run
            min_predictions_for_autonomy: 50,
            always_preview_state_changes: true,
            always_supervise_destructive: true,
            force_supervised_mode: false,
            always_supervised_domains: Vec::new(),
            min_accuracy_for_autonomy: 0.7,
        }
    }
}

impl ConstraintGateConfig {
    /// Create a strict configuration (more supervision)
    pub fn strict() -> Self {
        Self {
            supervision_threshold: RiskTier::StateModifying,
            calibration_threshold: 0.10,
            min_predictions_for_autonomy: 100,
            always_preview_state_changes: true,
            always_supervise_destructive: true,
            force_supervised_mode: false,
            always_supervised_domains: Vec::new(),
            min_accuracy_for_autonomy: 0.8,
        }
    }

    /// Create a permissive configuration (more autonomy)
    pub fn permissive() -> Self {
        Self {
            supervision_threshold: RiskTier::Critical,
            calibration_threshold: 0.25,
            min_predictions_for_autonomy: 20,
            always_preview_state_changes: false,
            always_supervise_destructive: true,
            force_supervised_mode: false,
            always_supervised_domains: Vec::new(),
            min_accuracy_for_autonomy: 0.6,
        }
    }

    /// Create a fully supervised configuration (no autonomy)
    pub fn fully_supervised() -> Self {
        Self {
            force_supervised_mode: true,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT GATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Gate decision result
#[derive(Debug, Clone)]
pub struct GateDecision {
    /// The execution mode determined
    pub mode: ExecutionMode,

    /// Confidence in this decision
    pub confidence: f64,

    /// Factors that influenced the decision
    pub factors: Vec<GateFactor>,

    /// Suggested confidence adjustment for the action
    pub suggested_confidence: f64,
}

/// Factor that influenced a gate decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateFactor {
    /// Name of the factor
    pub name: String,
    /// Weight of this factor
    pub weight: f64,
    /// Value of this factor
    pub value: f64,
    /// Description
    pub description: String,
}

/// The constraint gate that controls execution mode
#[derive(Debug)]
pub struct ConstraintGate {
    /// Configuration
    config: ConstraintGateConfig,

    /// Total actions checked
    actions_checked: usize,

    /// Actions requiring supervision
    supervision_required: usize,

    /// Actions forced to dry-run
    dry_run_forced: usize,

    /// Actions allowed autonomous
    autonomous_allowed: usize,

    /// When this gate was created
    created_at: Instant,
}

impl ConstraintGate {
    /// Create a new constraint gate
    pub fn new(config: ConstraintGateConfig) -> Self {
        Self {
            config,
            actions_checked: 0,
            supervision_required: 0,
            dry_run_forced: 0,
            autonomous_allowed: 0,
            created_at: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ConstraintGateConfig::default())
    }

    /// Check an action and determine execution mode
    pub fn check(
        &mut self,
        action: &WorldActionContext,
        calibration: &BrierScoreTracker,
    ) -> GateDecision {
        self.actions_checked += 1;

        let mut factors = Vec::new();

        // Factor 1: Global forced supervision mode
        if self.config.force_supervised_mode {
            self.supervision_required += 1;
            return GateDecision {
                mode: ExecutionMode::Supervised {
                    reason: SupervisionReason::ForcedSupervision,
                },
                confidence: 1.0,
                factors: vec![GateFactor {
                    name: "force_supervised_mode".to_string(),
                    weight: 1.0,
                    value: 1.0,
                    description: "Global supervision mode is enabled".to_string(),
                }],
                suggested_confidence: action.urgency as f64 * 0.5,
            };
        }

        // Factor 2: Risk tier check
        let risk_factor = GateFactor {
            name: "risk_tier".to_string(),
            weight: 0.4,
            value: action.risk_tier.level(),
            description: format!("Action risk: {:?}", action.risk_tier),
        };
        factors.push(risk_factor);

        if action.risk_tier >= self.config.supervision_threshold {
            self.supervision_required += 1;
            let reason = match action.risk_tier {
                RiskTier::Critical => SupervisionReason::Critical,
                RiskTier::Destructive => SupervisionReason::Destructive,
                _ => SupervisionReason::HighRisk,
            };
            return GateDecision {
                mode: ExecutionMode::Supervised { reason },
                confidence: 1.0,
                factors,
                suggested_confidence: calibration
                    .adjust_confidence(super::world_prediction::PredictionDomain::ToolUse, 0.5),
            };
        }

        // Factor 3: Always supervise destructive (if enabled)
        if self.config.always_supervise_destructive && action.risk_tier >= RiskTier::Destructive {
            self.supervision_required += 1;
            return GateDecision {
                mode: ExecutionMode::Supervised {
                    reason: SupervisionReason::Destructive,
                },
                confidence: 1.0,
                factors,
                suggested_confidence: 0.5,
            };
        }

        // Factor 4: Calibration check
        let ece = calibration.expected_calibration_error();
        let calibration_factor = GateFactor {
            name: "calibration_error".to_string(),
            weight: 0.3,
            value: ece,
            description: format!("ECE: {:.3}", ece),
        };
        factors.push(calibration_factor);

        if ece > self.config.calibration_threshold {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::PoorCalibration,
                },
                confidence: 0.7,
                factors,
                suggested_confidence: calibration
                    .adjust_confidence(super::world_prediction::PredictionDomain::ToolUse, 0.6),
            };
        }

        // Factor 5: Minimum predictions check
        let total_predictions = calibration.total_predictions();
        let experience_factor = GateFactor {
            name: "prediction_experience".to_string(),
            weight: 0.2,
            value: (total_predictions as f64 / self.config.min_predictions_for_autonomy as f64)
                .min(1.0),
            description: format!(
                "Predictions: {}/{}",
                total_predictions, self.config.min_predictions_for_autonomy
            ),
        };
        factors.push(experience_factor);

        if total_predictions < self.config.min_predictions_for_autonomy {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::UnexploredDomain,
                },
                confidence: 0.6,
                factors,
                suggested_confidence: 0.5,
            };
        }

        // Factor 6: Accuracy check
        let accuracy = calibration.accuracy();
        let accuracy_factor = GateFactor {
            name: "accuracy".to_string(),
            weight: 0.25,
            value: accuracy,
            description: format!("Accuracy: {:.1}%", accuracy * 100.0),
        };
        factors.push(accuracy_factor);

        if accuracy < self.config.min_accuracy_for_autonomy {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::PoorCalibration,
                },
                confidence: 0.65,
                factors,
                suggested_confidence: calibration.adjust_confidence(
                    super::world_prediction::PredictionDomain::ToolUse,
                    accuracy,
                ),
            };
        }

        // Factor 7: State modification preview (if enabled)
        if self.config.always_preview_state_changes && action.risk_tier >= RiskTier::StateModifying
        {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::StateModifying,
                },
                confidence: 0.8,
                factors,
                suggested_confidence: calibration
                    .adjust_confidence(super::world_prediction::PredictionDomain::SystemState, 0.7),
            };
        }

        // All checks passed - allow autonomous execution
        self.autonomous_allowed += 1;

        // Calculate overall confidence based on factors
        let total_weight: f64 = factors.iter().map(|f| f.weight).sum();
        let weighted_confidence: f64 = factors
            .iter()
            .map(|f| f.weight * (1.0 - f.value.abs())) // Lower risk/error = higher confidence
            .sum::<f64>()
            / total_weight;

        GateDecision {
            mode: ExecutionMode::Autonomous,
            confidence: weighted_confidence.clamp(0.5, 0.95),
            factors,
            suggested_confidence: calibration
                .adjust_confidence(super::world_prediction::PredictionDomain::ToolUse, 0.8),
        }
    }

    /// Get statistics about gate decisions
    pub fn statistics(&self) -> GateStatistics {
        GateStatistics {
            total_checked: self.actions_checked,
            autonomous_allowed: self.autonomous_allowed,
            dry_run_forced: self.dry_run_forced,
            supervision_required: self.supervision_required,
            autonomy_rate: if self.actions_checked > 0 {
                self.autonomous_allowed as f64 / self.actions_checked as f64
            } else {
                0.0
            },
            uptime_seconds: self.created_at.elapsed().as_secs(),
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &ConstraintGateConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ConstraintGateConfig) {
        self.config = config;
    }
}

/// Statistics about gate decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateStatistics {
    pub total_checked: usize,
    pub autonomous_allowed: usize,
    pub dry_run_forced: usize,
    pub supervision_required: usize,
    pub autonomy_rate: f64,
    pub uptime_seconds: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::calibration::CalibrationConfig;
    use crate::consciousness::recursive_improvement::world_prediction::WorldActionContext;

    fn create_calibrated_tracker() -> BrierScoreTracker {
        let config = CalibrationConfig {
            min_predictions_for_ece: 5,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(config);

        // Add 60 well-calibrated predictions (above min threshold)
        for i in 0..60 {
            let mut pred = crate::consciousness::recursive_improvement::world_prediction::WorldPrediction::new(
                "test",
                crate::consciousness::recursive_improvement::world_prediction::OutcomeCategory::Success,
                0.8,
                WorldActionContext::new("test", "test"),
                crate::consciousness::recursive_improvement::world_prediction::ResolutionContract::shell_command(),
            );
            // 80% accuracy to match 80% confidence (well-calibrated)
            if i % 5 != 0 {
                pred.resolve_true(
                    crate::consciousness::recursive_improvement::world_prediction::OutcomeCategory::Success,
                    1.0,
                );
            } else {
                pred.resolve_false(
                    crate::consciousness::recursive_improvement::world_prediction::OutcomeCategory::SafeFailure,
                    1.0,
                );
            }
            tracker.record_prediction(&pred);
        }
        tracker
    }

    #[test]
    fn test_force_supervised_mode() {
        let config = ConstraintGateConfig::fully_supervised();
        let mut gate = ConstraintGate::new(config);
        let tracker = BrierScoreTracker::with_defaults();

        let action = WorldActionContext::new("test", "test").with_risk_tier(RiskTier::Observation);

        let decision = gate.check(&action, &tracker);
        assert!(decision.mode.is_supervised());
    }

    #[test]
    fn test_high_risk_requires_supervision() {
        let mut gate = ConstraintGate::with_defaults();
        let tracker = create_calibrated_tracker();

        let action = WorldActionContext::new("delete_files", "Delete files")
            .with_risk_tier(RiskTier::Destructive);

        let decision = gate.check(&action, &tracker);
        assert!(decision.mode.is_supervised());
    }

    #[test]
    fn test_low_risk_allows_autonomy() {
        let mut config = ConstraintGateConfig::default();
        config.always_preview_state_changes = false; // Disable for this test
        let mut gate = ConstraintGate::new(config);
        let tracker = create_calibrated_tracker();

        let action =
            WorldActionContext::new("read_file", "Read file").with_risk_tier(RiskTier::Observation);

        let decision = gate.check(&action, &tracker);
        assert!(decision.mode.is_autonomous());
    }

    #[test]
    fn test_poor_calibration_forces_dry_run() {
        let config = ConstraintGateConfig {
            calibration_threshold: 0.01, // Very strict
            min_predictions_for_autonomy: 5,
            always_preview_state_changes: false,
            ..Default::default()
        };
        let mut gate = ConstraintGate::new(config);

        // Create poorly calibrated tracker (overconfident)
        let cal_config = CalibrationConfig {
            min_predictions_for_ece: 5,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(cal_config);
        for i in 0..20 {
            let mut pred = crate::consciousness::recursive_improvement::world_prediction::WorldPrediction::new(
                "test",
                crate::consciousness::recursive_improvement::world_prediction::OutcomeCategory::Success,
                0.95, // High confidence
                WorldActionContext::new("test", "test"),
                crate::consciousness::recursive_improvement::world_prediction::ResolutionContract::shell_command(),
            );
            // But only 50% accuracy
            if i % 2 == 0 {
                pred.resolve_true(
                    crate::consciousness::recursive_improvement::world_prediction::OutcomeCategory::Success,
                    1.0,
                );
            } else {
                pred.resolve_false(
                    crate::consciousness::recursive_improvement::world_prediction::OutcomeCategory::SafeFailure,
                    1.0,
                );
            }
            tracker.record_prediction(&pred);
        }

        let action = WorldActionContext::new("test", "test").with_risk_tier(RiskTier::Reversible);

        let decision = gate.check(&action, &tracker);
        // Should be dry-run due to poor calibration or low accuracy
        assert!(!decision.mode.is_autonomous());
    }

    #[test]
    fn test_statistics_tracking() {
        let mut gate = ConstraintGate::with_defaults();
        let tracker = create_calibrated_tracker();

        // Check a few actions
        for risk in [
            RiskTier::Observation,
            RiskTier::Destructive,
            RiskTier::Critical,
        ] {
            let action = WorldActionContext::new("test", "test").with_risk_tier(risk);
            gate.check(&action, &tracker);
        }

        let stats = gate.statistics();
        assert_eq!(stats.total_checked, 3);
        assert!(stats.supervision_required >= 2); // Destructive and Critical
    }
}
