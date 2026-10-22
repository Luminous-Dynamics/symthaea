//! Sequential prediction-residual monitor for model mismatch.
//!
//! The monitor consumes a non-negative, task-normalized innovation score per
//! control step. Its CUSUM statistic accumulates persistent excess over a
//! predeclared slack. Calibration uses the maximum statistic from each nominal
//! episode, so the threshold is an episode-level order statistic rather than a
//! per-step threshold. The false-alarm interpretation requires a frozen score,
//! disjoint exchangeable nominal calibration/evaluation episodes, and correctly
//! specified observation semantics. This detector is not a safety certificate:
//! a caller must route an alarm to an independently verified fallback.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualThreshold {
    statistic_threshold: f64,
    slack: f64,
    target_false_alarm_rate: f64,
    calibration_episodes: usize,
}

impl ResidualThreshold {
    pub const fn statistic_threshold(self) -> f64 {
        self.statistic_threshold
    }

    pub const fn slack(self) -> f64 {
        self.slack
    }

    pub const fn target_false_alarm_rate(self) -> f64 {
        self.target_false_alarm_rate
    }

    pub const fn calibration_episodes(self) -> usize {
        self.calibration_episodes
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualAssessment {
    step_index: usize,
    residual: f64,
    cumulative_statistic: f64,
    alarm_latched: bool,
    newly_alarmed: bool,
}

impl ResidualAssessment {
    pub const fn step_index(self) -> usize {
        self.step_index
    }

    pub const fn residual(self) -> f64 {
        self.residual
    }

    pub const fn cumulative_statistic(self) -> f64 {
        self.cumulative_statistic
    }

    pub const fn alarm_latched(self) -> bool {
        self.alarm_latched
    }

    pub const fn newly_alarmed(self) -> bool {
        self.newly_alarmed
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequentialResidualMonitor {
    threshold: ResidualThreshold,
    cumulative_statistic: f64,
    step_index: usize,
    alarm_latched: bool,
}

impl SequentialResidualMonitor {
    pub fn new(threshold: ResidualThreshold) -> Option<Self> {
        if !threshold.statistic_threshold.is_finite()
            || threshold.statistic_threshold < 0.0
            || !threshold.slack.is_finite()
            || threshold.slack < 0.0
            || !threshold.target_false_alarm_rate.is_finite()
            || !(0.0..1.0).contains(&threshold.target_false_alarm_rate)
            || threshold.calibration_episodes == 0
        {
            return None;
        }

        Some(Self {
            threshold,
            cumulative_statistic: 0.0,
            step_index: 0,
            alarm_latched: false,
        })
    }

    pub const fn threshold(self) -> ResidualThreshold {
        self.threshold
    }

    pub const fn step_index(self) -> usize {
        self.step_index
    }

    pub const fn alarm_latched(self) -> bool {
        self.alarm_latched
    }

    /// Update with a finite non-negative normalized residual. Invalid evidence
    /// returns None without modifying monitor state; callers should treat that
    /// result as an explicit loss of monitoring capability.
    pub fn observe(&mut self, residual: f64) -> Option<ResidualAssessment> {
        if !residual.is_finite() || residual < 0.0 {
            return None;
        }

        self.step_index = self.step_index.saturating_add(1);
        self.cumulative_statistic =
            (self.cumulative_statistic + residual - self.threshold.slack).max(0.0);
        let newly_alarmed = !self.alarm_latched
            && self.cumulative_statistic > self.threshold.statistic_threshold;
        self.alarm_latched |= newly_alarmed;

        Some(ResidualAssessment {
            step_index: self.step_index,
            residual,
            cumulative_statistic: self.cumulative_statistic,
            alarm_latched: self.alarm_latched,
            newly_alarmed,
        })
    }

    /// Reset the sequential state at an episode boundary while retaining the
    /// frozen, pre-calibrated threshold.
    pub fn reset(&mut self) {
        self.cumulative_statistic = 0.0;
        self.step_index = 0;
        self.alarm_latched = false;
    }
}

fn max_cusum(residuals: &[f64], slack: f64) -> Option<f64> {
    if residuals.is_empty()
        || !slack.is_finite()
        || slack < 0.0
        || residuals
            .iter()
            .any(|residual| !residual.is_finite() || *residual < 0.0)
    {
        return None;
    }

    let mut cumulative: f64 = 0.0;
    let mut maximum: f64 = 0.0;
    for residual in residuals {
        cumulative = (cumulative + residual - slack).max(0.0);
        maximum = maximum.max(cumulative);
    }
    Some(maximum)
}

/// Calibrate an episode-level CUSUM threshold from one residual sequence per
/// nominal calibration episode. The finite-sample rank is ceil((n+1)*(1-alpha)).
/// Invalid data, empty episodes, too few episodes for the requested rank, or an
/// invalid target returns None. Statistical interpretation requires exchangeable
/// independent calibration/evaluation episodes; it does not survive arbitrary
/// distribution shift.
pub fn calibrate_residual_threshold(
    nominal_episode_residuals: &[Vec<f64>],
    slack: f64,
    target_false_alarm_rate: f64,
) -> Option<ResidualThreshold> {
    let n = nominal_episode_residuals.len();
    if n == 0
        || !slack.is_finite()
        || slack < 0.0
        || !target_false_alarm_rate.is_finite()
        || !(0.0..1.0).contains(&target_false_alarm_rate)
    {
        return None;
    }

    let mut statistics = Vec::with_capacity(n);
    for residuals in nominal_episode_residuals {
        statistics.push(max_cusum(residuals, slack)?);
    }

    let rank = (((n + 1) as f64) * (1.0 - target_false_alarm_rate)).ceil() as usize;
    if rank == 0 || rank > n {
        return None;
    }
    statistics.sort_by(f64::total_cmp);

    Some(ResidualThreshold {
        statistic_threshold: *statistics.get(rank - 1)?,
        slack,
        target_false_alarm_rate,
        calibration_episodes: n,
    })
}

#[cfg(test)]
mod tests {
    use super::{calibrate_residual_threshold, SequentialResidualMonitor};

    fn nominal_sequences() -> Vec<Vec<f64>> {
        (0..159)
            .map(|episode| {
                (0..48)
                    .map(|step| {
                        let key = (episode * 37 + step * 19 + 11) % 41;
                        0.025 + key as f64 / 1_000.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn calibration_rejects_empty_invalid_and_undersized_inputs() {
        assert!(calibrate_residual_threshold(&[], 0.05, 0.05).is_none());
        assert!(calibrate_residual_threshold(&[vec![]], 0.05, 0.05).is_none());
        assert!(calibrate_residual_threshold(&[vec![f64::NAN]], 0.05, 0.05).is_none());
        assert!(calibrate_residual_threshold(&[vec![0.01]], 0.05, 0.0).is_none());
        assert!(calibrate_residual_threshold(&[vec![0.01]], 0.05, 0.01).is_none());
    }

    #[test]
    fn persistent_residual_shift_latches_alarm_within_sequence() {
        let threshold =
            calibrate_residual_threshold(&nominal_sequences(), 0.06, 0.05).expect("threshold");
        let mut monitor = SequentialResidualMonitor::new(threshold).expect("monitor");

        for _ in 0..12 {
            let assessment = monitor.observe(0.04).expect("valid nominal residual");
            assert!(!assessment.alarm_latched());
        }

        let shifted = monitor.observe(0.40).expect("valid shifted residual");
        assert!(shifted.newly_alarmed());
        assert!(shifted.alarm_latched());
        assert!(shifted.cumulative_statistic() > threshold.statistic_threshold());
    }

    #[test]
    fn an_alarm_is_latched_until_episode_reset() {
        let threshold =
            calibrate_residual_threshold(&nominal_sequences(), 0.06, 0.05).expect("threshold");
        let mut monitor = SequentialResidualMonitor::new(threshold).expect("monitor");
        monitor.observe(0.90).expect("shifted residual");
        assert!(monitor.alarm_latched());

        let later = monitor.observe(0.0).expect("finite residual");
        assert!(later.alarm_latched());
        assert!(!later.newly_alarmed());

        monitor.reset();
        assert_eq!(monitor.step_index(), 0);
        assert!(!monitor.alarm_latched());
    }

    #[test]
    fn invalid_observations_do_not_mutate_monitor_state() {
        let threshold =
            calibrate_residual_threshold(&nominal_sequences(), 0.06, 0.05).expect("threshold");
        let mut monitor = SequentialResidualMonitor::new(threshold).expect("monitor");
        assert!(monitor.observe(f64::INFINITY).is_none());
        assert!(monitor.observe(-0.1).is_none());
        assert_eq!(monitor.step_index(), 0);
        assert!(!monitor.alarm_latched());
    }

    #[test]
    fn identical_nominal_episode_sequences_produce_identical_thresholds() {
        let first = calibrate_residual_threshold(&nominal_sequences(), 0.06, 0.05)
            .expect("first threshold");
        let second = calibrate_residual_threshold(&nominal_sequences(), 0.06, 0.05)
            .expect("second threshold");
        assert_eq!(first, second);
    }
}
