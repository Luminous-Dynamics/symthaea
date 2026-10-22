// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite-horizon sequential monitoring with a fixed calibration reference.
//!
//! This is a deliberately conservative baseline, not a conformal test martingale. Each audit
//! uses the same frozen score definition and calibration reference, and receives alpha/H where
//! H is the predeclared maximum number of audits. By the union bound, the probability of one or
//! more false alarms is at most the declared total alpha when each audit p-value is marginally
//! valid under the clean null. Dependence between alarms does not invalidate this union-bound
//! argument; marginal validity still requires the documented exchangeability/data protocol.
//!
//! The horizon is intentionally finite. The constructor rejects a calibration set whose minimum
//! attainable rank p-value is larger than alpha/H, because such a detector could never alarm.
//! This resolution check is necessary, not sufficient, for useful power. Do not refit the model,
//! tune the score after seeing audit data, or silently change the target distribution during an
//! epoch. Start a new epoch with fresh calibration when the model or scoring contract changes.
//!
//! This does not provide the tighter anytime-valid guarantees or adaptive betting strategies of
//! conditional conformal test martingales or predictive-rank martingales. Those are benchmark
//! targets for future work.

use crate::adequacy_calibration::{CalibrationError, ConformalAdequacyCalibrator};

/// Errors raised when configuring or advancing a finite-horizon monitor.
#[derive(Clone, Debug, PartialEq)]
pub enum SequentialMonitorError {
    /// Overall significance level must be finite and strictly between zero and one.
    InvalidOverallSignificanceLevel,
    /// The maximum number of audits must be positive.
    EmptyMonitoringHorizon,
    /// The calibration set cannot attain the per-audit rejection threshold.
    InsufficientCalibrationResolution {
        /// Number of clean calibration batches available.
        calibration_batches: usize,
        /// Smallest attainable conservative rank p-value.
        attainable_p_value_floor: f64,
        /// Bonferroni threshold used for each audit.
        per_audit_alpha: f64,
    },
    /// The audit p-value could not be computed under the calibrator contract.
    Calibration(CalibrationError),
    /// The monitor has already raised an alarm and requires a new epoch.
    AlreadyAlarmed,
    /// All predeclared audits have been consumed without an alarm.
    MonitoringHorizonExhausted,
}

/// Current state of a finite-horizon monitoring epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequentialMonitorState {
    /// The monitor may consume another independent audit batch.
    Monitoring,
    /// A predeclared audit crossed the spending threshold; stop and investigate.
    Alarmed,
    /// The declared horizon ended without an alarm; this is not proof of model adequacy.
    HorizonExhausted,
}

/// Outcome of one audit within the declared monitoring horizon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequentialAuditDecision {
    /// The audit p-value crossed the per-audit threshold.
    Alarm,
    /// No alarm on this audit; more predeclared audits remain.
    NoAlarmYet,
    /// No alarm on the last allowed audit; the result remains inconclusive.
    NoAlarmAtHorizon,
}

/// Auditable result for one sequential audit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequentialAuditResult {
    /// One-based audit index in the current frozen monitoring epoch.
    pub audit_index: usize,
    /// Conservative conformal rank p-value for this audit batch.
    pub p_value: f64,
    /// Per-audit Bonferroni threshold (overall alpha divided by the horizon).
    pub per_audit_alpha: f64,
    /// Union-bound false-alarm budget through this audit, capped at the declared total alpha.
    pub cumulative_familywise_alpha_bound: f64,
    /// Policy outcome; a non-alarm is never a proof that the model is correct.
    pub decision: SequentialAuditDecision,
}

/// Fixed-reference, finite-horizon monitor with explicit alpha spending.
///
/// The caller owns the scientific protocol: calibration and audit scores must use the same
/// predeclared statistic, and the calibration scores and each clean audit score must be
/// exchangeable under the null. Reusing the fixed reference does not make p-values independent;
/// the finite-horizon guarantee comes from Bonferroni's union bound, not a martingale argument.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialAdequacyMonitor {
    calibrator: ConformalAdequacyCalibrator,
    total_alpha: f64,
    per_audit_alpha: f64,
    max_audits: usize,
    audits_consumed: usize,
    state: SequentialMonitorState,
}

impl SequentialAdequacyMonitor {
    /// Create a monitor with a fixed maximum audit count and family-wise false-alarm budget.
    ///
    /// For example, alpha=0.05 over five audits requires at least 99 calibration batches so
    /// that the smallest attainable p-value, 1/(n+1), can reach the per-audit threshold 0.01.
    /// This says nothing about detection power at practical discrepancy magnitudes.
    pub fn new(
        calibrator: ConformalAdequacyCalibrator,
        total_alpha: f64,
        max_audits: usize,
    ) -> Result<Self, SequentialMonitorError> {
        if !total_alpha.is_finite() || total_alpha <= 0.0 || total_alpha >= 1.0 {
            return Err(SequentialMonitorError::InvalidOverallSignificanceLevel);
        }
        if max_audits == 0 {
            return Err(SequentialMonitorError::EmptyMonitoringHorizon);
        }

        let per_audit_alpha = total_alpha / max_audits as f64;
        if !per_audit_alpha.is_finite() || per_audit_alpha <= 0.0 {
            return Err(SequentialMonitorError::InvalidOverallSignificanceLevel);
        }

        let calibration_batches = calibrator.calibration_batch_count();
        let attainable_p_value_floor = 1.0 / (calibration_batches as f64 + 1.0);
        if attainable_p_value_floor > per_audit_alpha {
            return Err(SequentialMonitorError::InsufficientCalibrationResolution {
                calibration_batches,
                attainable_p_value_floor,
                per_audit_alpha,
            });
        }

        Ok(Self {
            calibrator,
            total_alpha,
            per_audit_alpha,
            max_audits,
            audits_consumed: 0,
            state: SequentialMonitorState::Monitoring,
        })
    }

    /// Score one fresh audit batch and spend one portion of the declared alpha budget.
    ///
    /// Invalid/non-independent audit inputs do not consume the horizon. After an alarm or
    /// after the last non-alarming audit, the epoch is terminal and cannot be reused.
    pub fn observe(
        &mut self,
        audit_score: f64,
        independent_audit: bool,
    ) -> Result<SequentialAuditResult, SequentialMonitorError> {
        match self.state {
            SequentialMonitorState::Alarmed => {
                return Err(SequentialMonitorError::AlreadyAlarmed);
            }
            SequentialMonitorState::HorizonExhausted => {
                return Err(SequentialMonitorError::MonitoringHorizonExhausted);
            }
            SequentialMonitorState::Monitoring => {}
        }

        let p_value = self
            .calibrator
            .audit_p_value(audit_score, independent_audit)
            .map_err(SequentialMonitorError::Calibration)?;
        let audit_index = self.audits_consumed + 1;
        let decision = if p_value <= self.per_audit_alpha {
            self.state = SequentialMonitorState::Alarmed;
            SequentialAuditDecision::Alarm
        } else if audit_index == self.max_audits {
            self.state = SequentialMonitorState::HorizonExhausted;
            SequentialAuditDecision::NoAlarmAtHorizon
        } else {
            SequentialAuditDecision::NoAlarmYet
        };
        self.audits_consumed = audit_index;

        Ok(SequentialAuditResult {
            audit_index,
            p_value,
            per_audit_alpha: self.per_audit_alpha,
            cumulative_familywise_alpha_bound: (self.per_audit_alpha * audit_index as f64)
                .min(self.total_alpha),
            decision,
        })
    }

    /// Current monitoring state.
    pub fn state(&self) -> SequentialMonitorState {
        self.state
    }

    /// Number of successfully scored audit batches consumed in this epoch.
    pub fn audits_consumed(&self) -> usize {
        self.audits_consumed
    }

    /// The predeclared total family-wise false-alarm level.
    pub fn total_alpha(&self) -> f64 {
        self.total_alpha
    }

    /// Per-audit Bonferroni rejection threshold.
    pub fn per_audit_alpha(&self) -> f64 {
        self.per_audit_alpha
    }

    /// Maximum number of audit batches in this frozen epoch.
    pub fn max_audits(&self) -> usize {
        self.max_audits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibrator(n: usize) -> ConformalAdequacyCalibrator {
        let scores = (0..n).map(|index| index as f64).collect::<Vec<_>>();
        ConformalAdequacyCalibrator::new(&scores, 0.05)
            .expect("fixture has enough calibration scores")
    }

    #[test]
    fn enforces_rank_resolution_for_declared_familywise_budget() {
        assert!(matches!(
            SequentialAdequacyMonitor::new(calibrator(80), 0.05, 5),
            Err(SequentialMonitorError::InsufficientCalibrationResolution { .. })
        ));

        let monitor = SequentialAdequacyMonitor::new(calibrator(99), 0.05, 5)
            .expect("99 calibration batches attain a 0.01 threshold");
        assert!((monitor.per_audit_alpha() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_budget_and_horizon() {
        assert!(matches!(
            SequentialAdequacyMonitor::new(calibrator(100), f64::NAN, 5),
            Err(SequentialMonitorError::InvalidOverallSignificanceLevel)
        ));
        assert!(matches!(
            SequentialAdequacyMonitor::new(calibrator(100), 0.05, 0),
            Err(SequentialMonitorError::EmptyMonitoringHorizon)
        ));
    }

    #[test]
    fn non_alarm_uses_spending_budget_and_ends_inconclusively() {
        let mut monitor =
            SequentialAdequacyMonitor::new(calibrator(100), 0.05, 5).unwrap();

        for index in 1..=5 {
            let result = monitor.observe(0.0, true).unwrap();
            assert_eq!(result.audit_index, index);
            assert_eq!(result.p_value, 1.0);
            assert_eq!(
                result.decision,
                if index == 5 {
                    SequentialAuditDecision::NoAlarmAtHorizon
                } else {
                    SequentialAuditDecision::NoAlarmYet
                }
            );
            assert!(result.cumulative_familywise_alpha_bound <= 0.05);
        }

        assert_eq!(monitor.state(), SequentialMonitorState::HorizonExhausted);
        assert!(matches!(
            monitor.observe(0.0, true),
            Err(SequentialMonitorError::MonitoringHorizonExhausted)
        ));
    }

    #[test]
    fn alarm_stops_epoch_and_does_not_exceed_declared_budget() {
        let mut monitor =
            SequentialAdequacyMonitor::new(calibrator(100), 0.05, 5).unwrap();
        let result = monitor.observe(100.0, true).unwrap();

        assert_eq!(result.p_value, 1.0 / 101.0);
        assert_eq!(result.decision, SequentialAuditDecision::Alarm);
        assert!(result.cumulative_familywise_alpha_bound <= 0.05);
        assert_eq!(monitor.state(), SequentialMonitorState::Alarmed);
        assert!(matches!(
            monitor.observe(100.0, true),
            Err(SequentialMonitorError::AlreadyAlarmed)
        ));
    }

    #[test]
    fn invalid_audit_does_not_consume_alpha_or_horizon() {
        let mut monitor =
            SequentialAdequacyMonitor::new(calibrator(100), 0.05, 5).unwrap();
        assert!(matches!(
            monitor.observe(0.0, false),
            Err(SequentialMonitorError::Calibration(
                CalibrationError::AuditNotIndependent
            ))
        ));
        assert!(matches!(
            monitor.observe(f64::INFINITY, true),
            Err(SequentialMonitorError::Calibration(
                CalibrationError::InvalidAuditScore
            ))
        ));
        assert_eq!(monitor.audits_consumed(), 0);
        assert_eq!(monitor.state(), SequentialMonitorState::Monitoring);
    }
}
