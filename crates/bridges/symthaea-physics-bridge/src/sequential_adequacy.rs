// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite-horizon sequential monitoring with a fixed calibration reference.
//!
//! This is a deliberately conservative baseline, not a conformal test martingale. Each audit
//! uses the same frozen score definition and calibration reference, and receives a predeclared
//! portion of the total alpha budget. By the union bound, the probability of one or more false
//! alarms is at most the declared total alpha when each audit p-value is marginally valid under
//! the clean null. The allocation may be equal or risk-weighted; it must be fixed before seeing
//! audit scores, and the allocated thresholds must sum to no more than the total alpha.
//!
//! Dependence between alarms does not invalidate this union-bound argument; marginal validity
//! still requires the documented exchangeability/data protocol. This is not anytime-valid and
//! does not provide the tighter guarantees of conditional conformal test martingales or
//! predictive-rank martingales. Those are benchmark targets for future work.
//!
//! The finite horizon is intentional. The constructor rejects a calibration set whose minimum
//! attainable rank p-value exceeds any allocated audit threshold, because that audit could never
//! alarm. This resolution check is necessary, not sufficient, for useful power. Do not refit the
//! model, tune the score after seeing audit data, or silently change the target distribution during
//! an epoch. Start a new epoch with fresh calibration when the model or scoring contract changes.

use crate::adequacy_calibration::{CalibrationError, ConformalAdequacyCalibrator};

/// Errors raised when configuring or advancing a finite-horizon monitor.
#[derive(Clone, Debug, PartialEq)]
pub enum SequentialMonitorError {
    /// Overall significance level must be finite and strictly between zero and one.
    InvalidOverallSignificanceLevel,
    /// The maximum number of audits must be positive.
    EmptyMonitoringHorizon,
    /// The alpha-spending weights are invalid or allocate more than the total budget.
    InvalidAlphaSpendingSchedule,
    /// The calibration set cannot attain an allocated rejection threshold.
    InsufficientCalibrationResolution {
        /// Number of clean calibration batches available.
        calibration_batches: usize,
        /// Smallest attainable conservative rank p-value.
        attainable_p_value_floor: f64,
        /// Allocated threshold for an audit in the schedule.
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
    /// Predeclared alpha threshold allocated to this audit.
    pub per_audit_alpha: f64,
    /// Union-bound false-alarm budget through this audit, capped at total alpha.
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
    alpha_spending: Vec<f64>,
    cumulative_alpha_bounds: Vec<f64>,
    audits_consumed: usize,
    state: SequentialMonitorState,
}

impl SequentialAdequacyMonitor {
    /// Create a monitor with equal alpha spending across a fixed number of audits.
    ///
    /// For example, alpha=0.05 over five audits uses threshold 0.01 at each audit and requires
    /// at least 99 calibration batches so that the smallest attainable p-value, 1/(n+1), can
    /// reach that threshold. This says nothing about detection power at practical discrepancies.
    pub fn new(
        calibrator: ConformalAdequacyCalibrator,
        total_alpha: f64,
        max_audits: usize,
    ) -> Result<Self, SequentialMonitorError> {
        if max_audits == 0 {
            return Err(SequentialMonitorError::EmptyMonitoringHorizon);
        }
        let equal_weight = 1.0 / max_audits as f64;
        Self::new_with_alpha_spending(calibrator, total_alpha, &vec![equal_weight; max_audits])
    }

    /// Create a monitor with a predeclared, risk-weighted allocation of the total alpha budget.
    ///
    /// `spending_weights` must be non-empty, finite, strictly positive, and sum to at most one.
    /// The threshold at audit i is `total_alpha * spending_weights[i]`; any unallocated fraction
    /// remains unused. Weights should be chosen before audit data are inspected, based on a
    /// domain-level reason for prioritizing particular parts of the horizon. The constructor
    /// rejects schedules whose thresholds are not attainable at the available calibration
    /// resolution. A schedule does not make this monitor anytime-valid.
    ///
    /// For example, `[0.5, 0.25, 0.15, 0.1]` allocates half the budget to the first audit and
    /// the remainder across later audits. This can improve early-shift sensitivity only if the
    /// larger first threshold is useful for the target discrepancy; it is not universally more
    /// powerful than equal spending.
    pub fn new_with_alpha_spending(
        calibrator: ConformalAdequacyCalibrator,
        total_alpha: f64,
        spending_weights: &[f64],
    ) -> Result<Self, SequentialMonitorError> {
        if !total_alpha.is_finite() || total_alpha <= 0.0 || total_alpha >= 1.0 {
            return Err(SequentialMonitorError::InvalidOverallSignificanceLevel);
        }
        if spending_weights.is_empty() {
            return Err(SequentialMonitorError::EmptyMonitoringHorizon);
        }
        if spending_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
        {
            return Err(SequentialMonitorError::InvalidAlphaSpendingSchedule);
        }

        let weight_sum = spending_weights.iter().sum::<f64>();
        if !weight_sum.is_finite() || weight_sum <= 0.0 || weight_sum > 1.0 + 32.0 * f64::EPSILON {
            return Err(SequentialMonitorError::InvalidAlphaSpendingSchedule);
        }
        // Correct only floating-point roundoff just above one; never silently shrink a materially
        // over-budget schedule into validity.
        let scale = if weight_sum > 1.0 {
            1.0 / weight_sum
        } else {
            1.0
        };
        let alpha_spending = spending_weights
            .iter()
            .map(|weight| total_alpha * (*weight * scale))
            .collect::<Vec<_>>();
        if alpha_spending
            .iter()
            .any(|threshold| !threshold.is_finite() || *threshold <= 0.0)
        {
            return Err(SequentialMonitorError::InvalidAlphaSpendingSchedule);
        }

        let calibration_batches = calibrator.calibration_batch_count();
        let attainable_p_value_floor = 1.0 / (calibration_batches as f64 + 1.0);
        if let Some(per_audit_alpha) = alpha_spending
            .iter()
            .copied()
            .find(|threshold| attainable_p_value_floor > *threshold)
        {
            return Err(SequentialMonitorError::InsufficientCalibrationResolution {
                calibration_batches,
                attainable_p_value_floor,
                per_audit_alpha,
            });
        }

        let mut cumulative = 0.0;
        let cumulative_alpha_bounds = alpha_spending
            .iter()
            .map(|threshold| {
                cumulative += *threshold;
                cumulative.min(total_alpha)
            })
            .collect::<Vec<_>>();

        Ok(Self {
            calibrator,
            total_alpha,
            alpha_spending,
            cumulative_alpha_bounds,
            audits_consumed: 0,
            state: SequentialMonitorState::Monitoring,
        })
    }

    /// Score one fresh audit batch and spend its predeclared portion of the alpha budget.
    ///
    /// Invalid/non-independent audit inputs do not consume the horizon. After an alarm or after
    /// the last non-alarming audit, the epoch is terminal and cannot be reused.
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
        let zero_based_index = self.audits_consumed;
        let audit_index = zero_based_index + 1;
        let per_audit_alpha = self.alpha_spending[zero_based_index];
        let decision = if p_value <= per_audit_alpha {
            self.state = SequentialMonitorState::Alarmed;
            SequentialAuditDecision::Alarm
        } else if audit_index == self.alpha_spending.len() {
            self.state = SequentialMonitorState::HorizonExhausted;
            SequentialAuditDecision::NoAlarmAtHorizon
        } else {
            SequentialAuditDecision::NoAlarmYet
        };
        self.audits_consumed = audit_index;

        Ok(SequentialAuditResult {
            audit_index,
            p_value,
            per_audit_alpha,
            cumulative_familywise_alpha_bound: self.cumulative_alpha_bounds[zero_based_index],
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

    /// Threshold for the first audit. Equal allocation makes this the threshold for every audit.
    ///
    /// For non-uniform schedules, use `alpha_for_audit` to inspect a particular threshold.
    pub fn per_audit_alpha(&self) -> f64 {
        self.alpha_spending[0]
    }

    /// Threshold for a one-based audit index, or `None` for an invalid index.
    pub fn alpha_for_audit(&self, audit_index: usize) -> Option<f64> {
        audit_index
            .checked_sub(1)
            .and_then(|index| self.alpha_spending.get(index).copied())
    }

    /// The full predeclared alpha thresholds, one per audit.
    pub fn alpha_spending_schedule(&self) -> &[f64] {
        &self.alpha_spending
    }

    /// Maximum number of audit batches in this frozen epoch.
    pub fn max_audits(&self) -> usize {
        self.alpha_spending.len()
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
        assert!(matches!(
            SequentialAdequacyMonitor::new_with_alpha_spending(calibrator(100), 0.05, &[]),
            Err(SequentialMonitorError::EmptyMonitoringHorizon)
        ));
        assert!(matches!(
            SequentialAdequacyMonitor::new_with_alpha_spending(
                calibrator(100),
                0.05,
                &[0.5, 0.5, 0.01]
            ),
            Err(SequentialMonitorError::InvalidAlphaSpendingSchedule)
        ));
        assert!(matches!(
            SequentialAdequacyMonitor::new_with_alpha_spending(
                calibrator(100),
                0.05,
                &[0.5, f64::NAN]
            ),
            Err(SequentialMonitorError::InvalidAlphaSpendingSchedule)
        ));
    }

    #[test]
    fn non_alarm_uses_spending_budget_and_ends_inconclusively() {
        let mut monitor = SequentialAdequacyMonitor::new(calibrator(100), 0.05, 5).unwrap();

        for index in 1..=5 {
            let result = monitor.observe(0.0, true).unwrap();
            assert_eq!(result.audit_index, index);
            assert_eq!(result.p_value, 1.0);
            assert_eq!(result.per_audit_alpha, 0.01);
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
        let mut monitor = SequentialAdequacyMonitor::new(calibrator(100), 0.05, 5).unwrap();
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
    fn risk_weighted_schedule_changes_detection_priority_without_overspending() {
        let weights = [0.5, 0.25, 0.15, 0.1];
        let mut early =
            SequentialAdequacyMonitor::new_with_alpha_spending(calibrator(999), 0.05, &weights)
                .unwrap();
        let mut late =
            SequentialAdequacyMonitor::new_with_alpha_spending(calibrator(999), 0.05, &weights)
                .unwrap();

        // With 1,000 calibration batches this score yields p=(1+10)/1000=0.011:
        // it crosses the early 0.025 threshold but not the final 0.005 threshold.
        let early_result = early.observe(989.0, true).unwrap();
        assert!((early_result.p_value - 0.011).abs() < 1e-12);
        assert_eq!(early_result.per_audit_alpha, 0.025);
        assert_eq!(early_result.decision, SequentialAuditDecision::Alarm);

        for _ in 0..3 {
            assert_eq!(
                late.observe(0.0, true).unwrap().decision,
                SequentialAuditDecision::NoAlarmYet
            );
        }
        let late_result = late.observe(989.0, true).unwrap();
        assert!((late_result.per_audit_alpha - 0.005).abs() < 1e-12);
        assert_eq!(
            late_result.decision,
            SequentialAuditDecision::NoAlarmAtHorizon
        );
        assert!(late_result.cumulative_familywise_alpha_bound <= 0.05);
        assert!((late_result.cumulative_familywise_alpha_bound - 0.05).abs() < 1e-12);
    }

    #[test]
    fn rejects_risk_weighted_schedule_when_any_threshold_is_unattainable() {
        assert!(matches!(
            SequentialAdequacyMonitor::new_with_alpha_spending(
                calibrator(100),
                0.05,
                &[0.5, 0.25, 0.15, 0.1]
            ),
            Err(SequentialMonitorError::InsufficientCalibrationResolution {
                per_audit_alpha,
                ..
            }) if (per_audit_alpha - 0.005).abs() < 1e-12
        ));
    }

    fn next_unit_interval(state: &mut u64) -> f64 {
        // Fixed-seed xorshift64* is sufficient for this deterministic smoke simulation; it is
        // not used for cryptographic or scientific production randomness.
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let value = (*state).wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    #[test]
    fn null_stream_smoke_simulation_exposes_repeated_test_inflation() {
        const STREAMS: usize = 20_000;
        const CALIBRATION_BATCHES: usize = 100;
        const HORIZON: usize = 5;

        let mut rng = 0xD1B5_4A32_D192_ED03_u64;
        let mut sequential_alarms = 0usize;
        let mut uncorrected_alarms = 0usize;

        for _ in 0..STREAMS {
            let calibration_scores = (0..CALIBRATION_BATCHES)
                .map(|_| next_unit_interval(&mut rng))
                .collect::<Vec<_>>();
            let calibrator = ConformalAdequacyCalibrator::new(&calibration_scores, 0.05).unwrap();
            let audit_scores = (0..HORIZON)
                .map(|_| next_unit_interval(&mut rng))
                .collect::<Vec<_>>();

            // Comparator only: repeatedly thresholding each marginal rank p-value at 0.05.
            if audit_scores
                .iter()
                .any(|score| calibrator.audit_p_value(*score, true).unwrap() <= 0.05)
            {
                uncorrected_alarms += 1;
            }

            let mut monitor = SequentialAdequacyMonitor::new(calibrator, 0.05, HORIZON).unwrap();
            for score in audit_scores {
                if monitor.observe(score, true).unwrap().decision == SequentialAuditDecision::Alarm
                {
                    sequential_alarms += 1;
                    break;
                }
            }
        }

        let sequential_rate = sequential_alarms as f64 / STREAMS as f64;
        let uncorrected_rate = uncorrected_alarms as f64 / STREAMS as f64;

        // These are a fixed-seed regression envelope, not a confidence interval or proof of
        // validity. The analytic finite-horizon claim is supplied by the union-bound argument
        // and its exchangeability assumptions, not by this Monte Carlo test.
        assert!(
            (0.04..=0.055).contains(&sequential_rate),
            "corrected null FWER smoke rate was {sequential_rate:.4}"
        );
        assert!(
            (0.16..=0.28).contains(&uncorrected_rate),
            "uncorrected repeated-test smoke rate was {uncorrected_rate:.4}"
        );
        assert!(
            uncorrected_rate > sequential_rate + 0.10,
            "correction did not produce the expected gap: corrected={sequential_rate:.4}, uncorrected={uncorrected_rate:.4}"
        );
    }

    #[test]
    fn invalid_audit_does_not_consume_alpha_or_horizon() {
        let mut monitor = SequentialAdequacyMonitor::new(calibrator(100), 0.05, 5).unwrap();
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