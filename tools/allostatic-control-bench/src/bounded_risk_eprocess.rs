//! Anytime-valid bounded-risk e-process for sequential runtime monitoring.
//!
//! Under the null hypothesis E[loss_t | history before t] <= risk_bound, the
//! factor 1 + bet * (loss_t - risk_bound) is a nonnegative supermartingale
//! factor when bet is predictable and 0 <= bet * risk_bound < 1. Ville's
//! inequality bounds the probability that the process ever crosses 1/delta.
//! For N parallel monitors, this type uses N/delta as each threshold so the
//! union bound allocates the total false-alarm budget across them.
//!
//! This is a statistical risk monitor, not an OOD detector or safety certificate.
//! Its validity depends on the conditional-mean null assumption; nominal
//! calibration alone does not establish that assumption under arbitrary drift.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedRiskEProcess {
    risk_bound: f64,
    bet: f64,
    total_delta: f64,
    simultaneous_monitors: usize,
    e_value: f64,
    alarmed: bool,
}

impl BoundedRiskEProcess {
    /// Construct a fixed-bet e-process.
    ///
    /// risk_bound is the tolerated upper bound on the conditional expected
    /// bounded loss under the null. bet must be predictable with respect to
    /// the current loss; this implementation uses a fixed bet. total_delta
    /// is the overall false-alarm budget allocated by a union bound across
    /// simultaneous_monitors parallel processes.
    pub fn new(
        risk_bound: f64,
        bet: f64,
        total_delta: f64,
        simultaneous_monitors: usize,
    ) -> Option<Self> {
        if !risk_bound.is_finite()
            || risk_bound <= 0.0
            || risk_bound >= 1.0
            || !bet.is_finite()
            || bet <= 0.0
            || bet * risk_bound >= 1.0
            || !total_delta.is_finite()
            || total_delta <= 0.0
            || total_delta >= 1.0
            || simultaneous_monitors == 0
        {
            return None;
        }

        Some(Self {
            risk_bound,
            bet,
            total_delta,
            simultaneous_monitors,
            e_value: 1.0,
            alarmed: false,
        })
    }

    pub const fn risk_bound(self) -> f64 {
        self.risk_bound
    }

    pub const fn bet(self) -> f64 {
        self.bet
    }

    pub const fn total_delta(self) -> f64 {
        self.total_delta
    }

    pub const fn simultaneous_monitors(self) -> usize {
        self.simultaneous_monitors
    }

    /// Per-monitor threshold after a Bonferroni allocation of the global delta.
    pub fn alarm_threshold(self) -> f64 {
        self.simultaneous_monitors as f64 / self.total_delta
    }

    pub const fn e_value(self) -> f64 {
        self.e_value
    }

    pub const fn alarmed(self) -> bool {
        self.alarmed
    }

    /// Observe a bounded loss in [0, 1]. Returns None on invalid input without
    /// mutating the process. Overflow is treated as decisive evidence and latches
    /// an alarm, rather than wrapping or returning a misleading finite value.
    pub fn observe(&mut self, loss: f64) -> Option<bool> {
        if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
            return None;
        }

        let factor = 1.0 + self.bet * (loss - self.risk_bound);
        if !factor.is_finite() || factor <= 0.0 {
            return None;
        }

        let next_value = self.e_value * factor;
        if !next_value.is_finite() {
            self.e_value = f64::MAX;
            self.alarmed = true;
            return Some(true);
        }

        self.e_value = next_value;
        self.alarmed |= self.e_value >= self.alarm_threshold();
        Some(self.alarmed)
    }

    /// Reset the evidence stream, for example when starting a distinct episode.
    pub fn reset(&mut self) {
        self.e_value = 1.0;
        self.alarmed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_parameters_and_losses() {
        assert!(BoundedRiskEProcess::new(0.0, 1.0, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(1.0, 0.5, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 5.0, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 1.0, 1.0, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 1.0, 0.05, 0).is_none());

        let mut process = BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1)
            .expect("valid process");
        let before = process.e_value();
        assert!(process.observe(f64::NAN).is_none());
        assert!(process.observe(-0.01).is_none());
        assert!(process.observe(1.01).is_none());
        assert_eq!(process.e_value(), before);
    }

    #[test]
    fn allocates_global_delta_across_parallel_monitors() {
        let process = BoundedRiskEProcess::new(0.2, 2.0, 0.05, 5).expect("valid process");
        assert!((process.alarm_threshold() - 100.0).abs() < f64::EPSILON);
        assert_eq!(process.simultaneous_monitors(), 5);
        assert_eq!(process.total_delta(), 0.05);
    }

    #[test]
    fn repeated_low_losses_reduce_evidence_without_alarms() {
        let mut process = BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1)
            .expect("valid process");
        for _ in 0..30 {
            assert_eq!(process.observe(0.05), Some(false));
        }
        assert!(process.e_value() < 1.0);
        assert!(!process.alarmed());
    }

    #[test]
    fn sustained_high_losses_cross_the_anytime_alarm_boundary() {
        let mut process = BoundedRiskEProcess::new(0.15, 0.5 / 0.15, 0.05, 5)
            .expect("valid process");
        let mut first_alarm = None;
        for step in 0..8 {
            if process.observe(0.65) == Some(true) {
                first_alarm.get_or_insert(step);
            }
        }
        assert!(first_alarm.is_some());
        assert!(process.e_value() >= process.alarm_threshold());
        assert!(process.alarmed());
    }

    #[test]
    fn valid_below_bound_losses_preserve_positive_evidence() {
        let mut process = BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1)
            .expect("valid process");
        for loss in [0.05, 0.10, 0.05, 0.10, 0.05, 0.10] {
            let _ = process.observe(loss);
        }
        assert!(process.e_value() > 0.0);
    }

    #[test]
    fn reset_starts_a_fresh_evidence_stream() {
        let mut process = BoundedRiskEProcess::new(0.15, 0.5 / 0.15, 0.05, 1)
            .expect("valid process");
        for _ in 0..8 {
            let _ = process.observe(0.65);
        }
        assert!(process.alarmed());
        process.reset();
        assert_eq!(process.e_value(), 1.0);
        assert!(!process.alarmed());
    }
}
