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
    log_e_value: f64,
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
            log_e_value: 0.0,
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

    /// E-value threshold after Bonferroni allocation of the global delta.
    pub fn e_value_alarm_threshold(self) -> f64 {
        self.simultaneous_monitors as f64 / self.total_delta
    }

    /// Log of the E-value threshold, used for stable comparisons in log space.
    pub fn log_alarm_threshold(self) -> f64 {
        self.e_value_alarm_threshold().ln()
    }

    pub const fn log_e_value(self) -> f64 {
        self.log_e_value
    }

    /// Exponentiated e-value for reporting. The monitor itself works in log
    /// space so very long streams do not silently underflow or overflow.
    pub fn e_value(self) -> f64 {
        self.log_e_value.exp()
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

        let next_log_value = self.log_e_value + factor.ln();
        if !next_log_value.is_finite() {
            self.log_e_value = f64::MAX;
            self.alarmed = true;
            return Some(true);
        }

        self.log_e_value = next_log_value;
        self.alarmed |= self.log_e_value >= self.log_alarm_threshold();
        Some(self.alarmed)
    }

    /// Reset the evidence stream, for example when starting a distinct episode.
    pub fn reset(&mut self) {
        self.log_e_value = 0.0;
        self.alarmed = false;
    }
}

/// Fixed-grid mixture of bounded-risk e-processes for unknown shift magnitude.
///
/// Each component uses a predeclared predictable bet fraction in (0, 1), with
/// bet = fraction / risk_bound. The arithmetic mixture of nonnegative
/// supermartingales is itself a nonnegative supermartingale under the same
/// conditional-risk null. This avoids selecting one bet using evaluation data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixtureBoundedRiskEProcess {
    risk_bound: f64,
    bet_fractions: [f64; 3],
    total_delta: f64,
    simultaneous_monitors: usize,
    component_log_e_values: [f64; 3],
    log_e_value: f64,
    alarmed: bool,
}

impl MixtureBoundedRiskEProcess {
    pub fn new(
        risk_bound: f64,
        bet_fractions: [f64; 3],
        total_delta: f64,
        simultaneous_monitors: usize,
    ) -> Option<Self> {
        if !risk_bound.is_finite()
            || risk_bound <= 0.0
            || risk_bound >= 1.0
            || bet_fractions
                .iter()
                .any(|fraction| !fraction.is_finite() || *fraction <= 0.0 || *fraction >= 1.0)
            || !total_delta.is_finite()
            || total_delta <= 0.0
            || total_delta >= 1.0
            || simultaneous_monitors == 0
        {
            return None;
        }

        Some(Self {
            risk_bound,
            bet_fractions,
            total_delta,
            simultaneous_monitors,
            component_log_e_values: [0.0; 3],
            log_e_value: 0.0,
            alarmed: false,
        })
    }

    pub const fn risk_bound(self) -> f64 {
        self.risk_bound
    }

    pub const fn bet_fractions(self) -> [f64; 3] {
        self.bet_fractions
    }

    pub fn bet_values(self) -> [f64; 3] {
        self.bet_fractions.map(|fraction| fraction / self.risk_bound)
    }

    pub const fn total_delta(self) -> f64 {
        self.total_delta
    }

    pub const fn simultaneous_monitors(self) -> usize {
        self.simultaneous_monitors
    }

    pub fn log_alarm_threshold(self) -> f64 {
        (self.simultaneous_monitors as f64 / self.total_delta).ln()
    }

    pub const fn component_log_e_values(self) -> [f64; 3] {
        self.component_log_e_values
    }

    pub const fn log_e_value(self) -> f64 {
        self.log_e_value
    }

    pub fn e_value(self) -> f64 {
        self.log_e_value.exp()
    }

    pub const fn alarmed(self) -> bool {
        self.alarmed
    }

    pub fn observe(&mut self, loss: f64) -> Option<bool> {
        if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
            return None;
        }

        for (index, fraction) in self.bet_fractions.iter().copied().enumerate() {
            let bet = fraction / self.risk_bound;
            let factor = 1.0 + bet * (loss - self.risk_bound);
            if !factor.is_finite() || factor <= 0.0 {
                return None;
            }
            self.component_log_e_values[index] += factor.ln();
        }

        let max_log_value = self
            .component_log_e_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_exp = self
            .component_log_e_values
            .iter()
            .map(|value| (*value - max_log_value).exp())
            .sum::<f64>()
            / self.component_log_e_values.len() as f64;
        let next_log_value = max_log_value + mean_exp.ln();

        if !next_log_value.is_finite() {
            self.log_e_value = f64::MAX;
            self.alarmed = true;
            return Some(true);
        }

        self.log_e_value = next_log_value;
        self.alarmed |= self.log_e_value >= self.log_alarm_threshold();
        Some(self.alarmed)
    }

    pub fn reset(&mut self) {
        self.component_log_e_values = [0.0; 3];
        self.log_e_value = 0.0;
        self.alarmed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixture_of_bets_allocates_delta_and_rejects_invalid_components() {
        assert!(MixtureBoundedRiskEProcess::new(0.15, [0.25, 0.5, 1.0], 0.05, 5).is_none());
        assert!(MixtureBoundedRiskEProcess::new(0.15, [0.25, 0.0, 0.75], 0.05, 5).is_none());
        let process = MixtureBoundedRiskEProcess::new(0.15, [0.25, 0.5, 0.75], 0.05, 5)
            .expect("valid bet mixture");
        assert!((process.log_alarm_threshold().exp() - 100.0).abs() < 1e-10);
        assert_eq!(process.bet_fractions(), [0.25, 0.5, 0.75]);
        assert_eq!(process.bet_values(), [
            0.25 / 0.15,
            0.5 / 0.15,
            0.75 / 0.15,
        ]);
    }

    #[test]
    fn mixture_preserves_nominal_evidence_and_detects_gradual_excess() {
        let mut nominal = MixtureBoundedRiskEProcess::new(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
        )
        .expect("valid mixture");
        for loss in [0.06, 0.08, 0.10, 0.12, 0.09, 0.13, 0.07, 0.10] {
            assert_eq!(nominal.observe(loss), Some(false));
        }
        assert!(nominal.log_e_value() < nominal.log_alarm_threshold());

        let mut gradual = MixtureBoundedRiskEProcess::new(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
        )
        .expect("valid mixture");
        let mut first_alarm = None;
        for (step, score) in [0.009, 0.012, 0.015, 0.018, 0.021, 0.025, 0.028, 0.031]
            .into_iter()
            .enumerate()
        {
            if gradual.observe((score / 0.05).min(1.0)) == Some(true) {
                first_alarm.get_or_insert(step);
            }
        }
        assert!(first_alarm.is_some());
        assert!(gradual.log_e_value() >= gradual.log_alarm_threshold());
    }

    #[test]
    fn mixture_logsumexp_matches_independent_component_recurrence() {
        let risk_bound = 0.15;
        let fractions = [0.25, 0.5, 0.75];
        let mut process = MixtureBoundedRiskEProcess::new(
            risk_bound,
            fractions,
            0.05,
            5,
        )
        .expect("valid mixture");

        let losses = [0.12, 0.20, 0.24, 0.08, 0.31];
        let mut expected_components = [0.0_f64; 3];
        for loss in losses {
            let _ = process.observe(loss);
            for (index, fraction) in fractions.iter().copied().enumerate() {
                let factor = 1.0 + (fraction / risk_bound) * (loss - risk_bound);
                expected_components[index] += factor.ln();
            }
            for (actual, expected) in process
                .component_log_e_values()
                .iter()
                .zip(expected_components.iter())
            {
                assert!((actual - expected).abs() < 1e-12);
            }
            let max_component = expected_components
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let expected_mix = max_component
                + ((expected_components[0] - max_component).exp()
                    + (expected_components[1] - max_component).exp()
                    + (expected_components[2] - max_component).exp())
                    .ln()
                - 3.0_f64.ln();
            assert!((process.log_e_value() - expected_mix).abs() < 1e-12);
        }
    }

    #[test]
    fn mixture_rejects_invalid_loss_without_mutating_state_and_can_reset() {
        let mut process = MixtureBoundedRiskEProcess::new(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
        )
        .expect("valid mixture");
        let initial = process.component_log_e_values();
        assert!(process.observe(f64::NAN).is_none());
        assert_eq!(process.component_log_e_values(), initial);

        for _ in 0..8 {
            let _ = process.observe(0.65);
        }
        assert!(process.alarmed());
        process.reset();
        assert_eq!(process.component_log_e_values(), [0.0; 3]);
        assert_eq!(process.log_e_value(), 0.0);
        assert!(!process.alarmed());
    }

    #[test]
    fn rejects_invalid_parameters_and_losses() {
        assert!(BoundedRiskEProcess::new(0.0, 1.0, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(1.0, 0.5, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 5.0, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 1.0, 1.0, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 1.0, 0.05, 0).is_none());

        let mut process =
            BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1).expect("valid process");
        let before = process.log_e_value();
        assert!(process.observe(f64::NAN).is_none());
        assert!(process.observe(-0.01).is_none());
        assert!(process.observe(1.01).is_none());
        assert_eq!(process.log_e_value(), before);
    }

    #[test]
    fn allocates_global_delta_across_parallel_monitors() {
        let process = BoundedRiskEProcess::new(0.2, 2.0, 0.05, 5).expect("valid process");
        assert!((process.e_value_alarm_threshold() - 100.0).abs() < 1e-10);
        assert_eq!(process.simultaneous_monitors(), 5);
        assert_eq!(process.total_delta(), 0.05);
    }

    #[test]
    fn repeated_low_losses_reduce_evidence_without_alarms() {
        let mut process =
            BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1).expect("valid process");
        for _ in 0..30 {
            assert_eq!(process.observe(0.05), Some(false));
        }
        assert!(process.log_e_value() < 0.0);
        assert!(!process.alarmed());
    }

    #[test]
    fn sustained_high_losses_cross_the_anytime_alarm_boundary() {
        let mut process =
            BoundedRiskEProcess::new(0.15, 0.5 / 0.15, 0.05, 5).expect("valid process");
        let mut first_alarm = None;
        for step in 0..8 {
            if process.observe(0.65) == Some(true) {
                first_alarm.get_or_insert(step);
            }
        }
        assert!(first_alarm.is_some());
        assert!(process.log_e_value() >= process.log_alarm_threshold());
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
        let mut process =
            BoundedRiskEProcess::new(0.15, 0.5 / 0.15, 0.05, 1).expect("valid process");
        for _ in 0..8 {
            let _ = process.observe(0.65);
        }
        assert!(process.alarmed());
        process.reset();
        assert_eq!(process.log_e_value(), 0.0);
        assert!(!process.alarmed());
    }
}
