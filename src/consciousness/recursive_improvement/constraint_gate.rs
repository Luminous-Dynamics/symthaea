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
//! - **DryRun**: Execute in simulation/preview mode first, then optionally execute
//! - **Supervised**: Require human approval before execution
//!
//! ## Design Principle
//!
//! > "The first strong behavior must not be 'unsafe cleverness.'"

use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::calibration::BrierScoreTracker;
use super::world_prediction::{PredictionDomain, RiskTier, WorldActionContext};

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
    /// No explicit typed calibration domain was bound to the action.
    CalibrationDomainUnbound,
    /// Matching declared-domain evidence exists but has not reached the
    /// calibration measurement threshold.
    CalibrationUnmeasured,
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
            Self::CalibrationDomainUnbound => {
                "Action has no explicit typed calibration domain"
            }
            Self::CalibrationUnmeasured => {
                "Matching declared-domain calibration has not been measured"
            }
            Self::PoorCalibration => "Matching domain calibration is below threshold",
            Self::UnexploredDomain => "Domain has insufficient declared prediction history",
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
    /// If matching-domain ECE is above this, force dry-run mode
    pub calibration_threshold: f64,

    /// Minimum explicitly domain-bound predictions needed before autonomous execution allowed
    pub min_predictions_for_autonomy: usize,

    /// Whether to force dry-run for all state-modifying actions
    pub always_preview_state_changes: bool,

    /// Whether to force supervision for all destructive actions
    pub always_supervise_destructive: bool,

    /// Global override: force supervision for ALL actions
    pub force_supervised_mode: bool,

    /// Domains that should always require supervision
    pub always_supervised_domains: Vec<String>,

    /// Minimum matching-domain accuracy for autonomous execution
    pub min_accuracy_for_autonomy: f64,
}

impl Default for ConstraintGateConfig {
    fn default() -> Self {
        Self {
            supervision_threshold: RiskTier::Destructive,
            calibration_threshold: 0.15,
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

/// Gate decision result.
///
/// Deliberately contains no aggregate `confidence` or `suggested_confidence`
/// scalar. The gate decides a mode from typed rules and exposes the individual
/// diagnostic factors that caused that disposition. A heterogeneous mix of
/// risk, accuracy, experience and calibration error has no justified
/// probabilistic interpretation merely because each input is numeric.
#[derive(Debug, Clone)]
pub struct GateDecision {
    /// The execution mode determined
    pub mode: ExecutionMode,

    /// Factors that influenced the decision
    pub factors: Vec<GateFactor>,
}

/// One diagnostic factor that influenced a gate decision.
///
/// `value` retains the native numeric meaning documented by `name` and
/// `description`; values across different factors must not be averaged into a
/// probability or authority score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateFactor {
    /// Name of the factor
    pub name: String,
    /// Native value of this factor
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

    /// Check an action and determine execution mode.
    ///
    /// Risk supervision remains independent of calibration. If an action is
    /// otherwise eligible for autonomous execution, CAL-002A requires an
    /// explicit typed prediction domain plus sufficient *declared-domain*
    /// session evidence. CAL-003A preserves those exact decisions while
    /// removing the legacy pseudo-probability outputs from the gate result.
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
                factors: vec![GateFactor {
                    name: "force_supervised_mode".to_string(),
                    value: 1.0,
                    description: "Global supervision mode is enabled".to_string(),
                }],
            };
        }

        // Factor 2: Risk tier check. High risk never needs calibration evidence
        // merely to be denied autonomy; supervision wins first.
        factors.push(GateFactor {
            name: "risk_tier".to_string(),
            value: action.risk_tier.level(),
            description: format!("Action risk: {:?}", action.risk_tier),
        });

        if action.risk_tier >= self.config.supervision_threshold {
            self.supervision_required += 1;
            let reason = match action.risk_tier {
                RiskTier::Critical => SupervisionReason::Critical,
                RiskTier::Destructive => SupervisionReason::Destructive,
                _ => SupervisionReason::HighRisk,
            };
            return GateDecision {
                mode: ExecutionMode::Supervised { reason },
                factors,
            };
        }

        // Factor 3: Always supervise destructive (if enabled)
        if self.config.always_supervise_destructive && action.risk_tier >= RiskTier::Destructive {
            self.supervision_required += 1;
            return GateDecision {
                mode: ExecutionMode::Supervised {
                    reason: SupervisionReason::Destructive,
                },
                factors,
            };
        }

        // Factor 4: an authority-sensitive calibration cohort must be explicitly
        // typed. The legacy action-string heuristic remains usable for modeling
        // but cannot select the cohort that grants autonomy.
        let Some(domain) = action.declared_prediction_domain() else {
            self.dry_run_forced += 1;
            factors.push(GateFactor {
                name: "calibration_domain_bound".to_string(),
                value: 0.0,
                description: "No explicit typed prediction domain".to_string(),
            });
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::CalibrationDomainUnbound,
                },
                factors,
            };
        };

        let domain_calibration = calibration.declared_domain_calibration(domain);

        // Factor 5: Minimum matching declared-domain observations.
        factors.push(GateFactor {
            name: "declared_domain_prediction_experience".to_string(),
            value: domain_calibration.sample_count as f64,
            description: format!(
                "{:?} declared predictions: {}/{}",
                domain,
                domain_calibration.sample_count,
                self.config.min_predictions_for_autonomy
            ),
        });

        if domain_calibration.sample_count < self.config.min_predictions_for_autonomy {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::UnexploredDomain,
                },
                factors,
            };
        }

        // Factor 6: ECE must actually be measured for this declared domain.
        let Some(ece) = domain_calibration.ece else {
            self.dry_run_forced += 1;
            factors.push(GateFactor {
                name: "declared_domain_calibration_measured".to_string(),
                value: 0.0,
                description: format!("{:?} declared-domain ECE is unmeasured", domain),
            });
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::CalibrationUnmeasured,
                },
                factors,
            };
        };

        factors.push(GateFactor {
            name: "declared_domain_calibration_error".to_string(),
            value: ece,
            description: format!("{:?} declared-domain ECE: {:.3}", domain, ece),
        });

        if ece > self.config.calibration_threshold {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::PoorCalibration,
                },
                factors,
            };
        }

        // Factor 7: Matching-domain accuracy check.
        let accuracy = domain_calibration.accuracy;
        factors.push(GateFactor {
            name: "declared_domain_accuracy".to_string(),
            value: accuracy,
            description: format!("{:?} declared-domain accuracy: {:.1}%", domain, accuracy * 100.0),
        });

        if accuracy < self.config.min_accuracy_for_autonomy {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::PoorCalibration,
                },
                factors,
            };
        }

        // Factor 8: State modification preview (if enabled)
        if self.config.always_preview_state_changes && action.risk_tier >= RiskTier::StateModifying
        {
            self.dry_run_forced += 1;
            return GateDecision {
                mode: ExecutionMode::DryRun {
                    reason: DryRunReason::StateModifying,
                },
                factors,
            };
        }

        self.autonomous_allowed += 1;
        GateDecision {
            mode: ExecutionMode::Autonomous,
            factors,
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
    use crate::consciousness::recursive_improvement::world_prediction::{
        OutcomeCategory, ResolutionContract, WorldPrediction,
    };

    fn add_resolved_predictions(
        tracker: &mut BrierScoreTracker,
        domain: PredictionDomain,
        count: usize,
        confidence: f64,
        correct_numerator: usize,
        correct_denominator: usize,
        declared: bool,
    ) {
        for i in 0..count {
            let mut action = WorldActionContext::new("test", "test");
            if declared {
                action = action.with_prediction_domain(domain);
            }
            let mut pred = WorldPrediction::new(
                "test",
                OutcomeCategory::Success,
                confidence,
                action,
                ResolutionContract::shell_command(),
            );
            if i % correct_denominator < correct_numerator {
                pred.resolve_true(OutcomeCategory::Success, 1.0);
            } else {
                pred.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            }
            tracker.record_prediction(&pred);
        }
    }

    fn create_calibrated_tracker() -> BrierScoreTracker {
        let config = CalibrationConfig {
            min_predictions_for_ece: 5,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(config);
        add_resolved_predictions(
            &mut tracker,
            PredictionDomain::CodeExecution,
            60,
            0.8,
            4,
            5,
            true,
        );
        tracker
    }

    #[test]
    fn gate_decision_exposes_factors_not_aggregate_confidence() {
        let mut gate = ConstraintGate::with_defaults();
        let tracker = BrierScoreTracker::with_defaults();
        let action = WorldActionContext::new("test", "test").with_risk_tier(RiskTier::Observation);
        let decision = gate.check(&action, &tracker);
        assert!(!decision.factors.is_empty());
        assert!(decision.mode.is_dry_run());
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
    fn test_high_risk_requires_supervision_without_domain_binding() {
        let mut gate = ConstraintGate::with_defaults();
        let tracker = create_calibrated_tracker();

        let action = WorldActionContext::new("delete_files", "Delete files")
            .with_risk_tier(RiskTier::Destructive);

        let decision = gate.check(&action, &tracker);
        assert!(decision.mode.is_supervised());
    }

    #[test]
    fn unbound_low_risk_action_cannot_be_autonomous() {
        let mut config = ConstraintGateConfig::default();
        config.always_preview_state_changes = false;
        let mut gate = ConstraintGate::new(config);
        let tracker = create_calibrated_tracker();

        let action = WorldActionContext::new("test", "test").with_risk_tier(RiskTier::Observation);
        let decision = gate.check(&action, &tracker);
        assert!(matches!(
            decision.mode,
            ExecutionMode::DryRun {
                reason: DryRunReason::CalibrationDomainUnbound
            }
        ));
    }

    #[test]
    fn test_low_risk_allows_matching_domain_autonomy() {
        let mut config = ConstraintGateConfig::default();
        config.always_preview_state_changes = false;
        let mut gate = ConstraintGate::new(config);
        let tracker = create_calibrated_tracker();

        let action = WorldActionContext::new("read_file", "Read file")
            .with_risk_tier(RiskTier::Observation)
            .with_prediction_domain(PredictionDomain::CodeExecution);

        let decision = gate.check(&action, &tracker);
        assert!(decision.mode.is_autonomous());
    }

    #[test]
    fn calibrated_other_domain_cannot_subsidize_autonomy() {
        let mut config = ConstraintGateConfig::default();
        config.always_preview_state_changes = false;
        let mut gate = ConstraintGate::new(config);
        let tracker = create_calibrated_tracker();

        let action = WorldActionContext::new("tool", "Use tool")
            .with_risk_tier(RiskTier::Observation)
            .with_prediction_domain(PredictionDomain::ToolUse);

        let decision = gate.check(&action, &tracker);
        assert!(matches!(
            decision.mode,
            ExecutionMode::DryRun {
                reason: DryRunReason::UnexploredDomain
            }
        ));
    }

    #[test]
    fn inferred_history_cannot_subsidize_declared_domain_autonomy() {
        let cal_config = CalibrationConfig {
            min_predictions_for_ece: 5,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(cal_config);
        add_resolved_predictions(
            &mut tracker,
            PredictionDomain::CodeExecution,
            60,
            0.8,
            4,
            5,
            false,
        );

        let mut gate_config = ConstraintGateConfig::default();
        gate_config.always_preview_state_changes = false;
        let mut gate = ConstraintGate::new(gate_config);
        let action = WorldActionContext::new("compile", "Compile")
            .with_risk_tier(RiskTier::Observation)
            .with_prediction_domain(PredictionDomain::CodeExecution);

        let decision = gate.check(&action, &tracker);
        assert!(matches!(
            decision.mode,
            ExecutionMode::DryRun {
                reason: DryRunReason::UnexploredDomain
            }
        ));
    }

    #[test]
    fn sufficient_samples_without_measured_ece_still_fail_closed() {
        let cal_config = CalibrationConfig {
            min_predictions_for_ece: 10,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(cal_config);
        add_resolved_predictions(
            &mut tracker,
            PredictionDomain::CodeExecution,
            5,
            0.8,
            4,
            5,
            true,
        );

        let gate_config = ConstraintGateConfig {
            min_predictions_for_autonomy: 5,
            always_preview_state_changes: false,
            ..Default::default()
        };
        let mut gate = ConstraintGate::new(gate_config);
        let action = WorldActionContext::new("compile", "Compile")
            .with_risk_tier(RiskTier::Observation)
            .with_prediction_domain(PredictionDomain::CodeExecution);

        let decision = gate.check(&action, &tracker);
        assert!(matches!(
            decision.mode,
            ExecutionMode::DryRun {
                reason: DryRunReason::CalibrationUnmeasured
            }
        ));
    }

    #[test]
    fn test_poor_matching_domain_calibration_forces_dry_run() {
        let config = ConstraintGateConfig {
            calibration_threshold: 0.01,
            min_predictions_for_autonomy: 5,
            always_preview_state_changes: false,
            ..Default::default()
        };
        let mut gate = ConstraintGate::new(config);

        let cal_config = CalibrationConfig {
            min_predictions_for_ece: 5,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(cal_config);
        add_resolved_predictions(
            &mut tracker,
            PredictionDomain::CodeExecution,
            20,
            0.95,
            1,
            2,
            true,
        );

        let action = WorldActionContext::new("test", "test")
            .with_risk_tier(RiskTier::Reversible)
            .with_prediction_domain(PredictionDomain::CodeExecution);

        let decision = gate.check(&action, &tracker);
        assert!(!decision.mode.is_autonomous());
    }

    #[test]
    fn test_statistics_tracking() {
        let mut gate = ConstraintGate::with_defaults();
        let tracker = create_calibrated_tracker();

        for risk in [
            RiskTier::Observation,
            RiskTier::Destructive,
            RiskTier::Critical,
        ] {
            let action = WorldActionContext::new("test", "test")
                .with_risk_tier(risk)
                .with_prediction_domain(PredictionDomain::CodeExecution);
            gate.check(&action, &tracker);
        }

        let stats = gate.statistics();
        assert_eq!(stats.total_checked, 3);
        assert!(stats.supervision_required >= 2);
    }
}
