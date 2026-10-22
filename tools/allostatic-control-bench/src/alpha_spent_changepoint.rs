//! Bounded-memory alpha-spent multi-start e-process for unknown change points.
//!
//! For each absolute stream index s, this monitor starts a new arithmetic
//! mixture of bounded-risk e-processes. Candidate s receives delta_s =
//! delta / K * 6 / (pi^2 * (s + 1)^2), whose total allocation over all s is
//! delta / K. A candidate alarms at log(1 / delta_s), which is compared with
//! the component-mixture log e-value. A union bound over start times and K
//! streams controls the familywise crossing probability under the declared
//! conditional-risk null.
//!
//! The fixed candidate window bounds memory and can only reduce detection power:
//! expired candidates are discarded, but their allocated alpha is never reused.
//! There is intentionally no reset API because resetting the stream index and
//! reusing early alpha allocations would invalidate the lifetime accounting.
//! As with every e-process, validity depends on E[L_t | F_(t-1)] <= risk_bound;
//! this does not follow from nominal calibration under arbitrary distribution
//! shift.

use std::f64::consts::PI;

const PI_SQUARED_OVER_SIX: f64 = PI * PI / 6.0;
const MAX_ACTIVE_STARTS_LIMIT: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChangePointAlarm {
    start_index: u64,
    detected_at: u64,
    log_e_value: f64,
    log_threshold: f64,
}

impl ChangePointAlarm {
    pub const fn start_index(self) -> u64 {
        self.start_index
    }

    pub const fn detected_at(self) -> u64 {
        self.detected_at
    }

    pub const fn log_e_value(self) -> f64 {
        self.log_e_value
    }

    pub const fn log_threshold(self) -> f64 {
        self.log_threshold
    }

    pub const fn evidence_excess(self) -> f64 {
        self.log_e_value - self.log_threshold
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChangePointUpdate {
    time_index: u64,
    active_candidates: usize,
    new_alarm: Option<ChangePointAlarm>,
    first_alarm: Option<ChangePointAlarm>,
}

impl ChangePointUpdate {
    pub const fn time_index(self) -> u64 {
        self.time_index
    }

    pub const fn active_candidates(self) -> usize {
        self.active_candidates
    }

    /// Present only on the step where the first alarm is latched.
    pub const fn new_alarm(self) -> Option<ChangePointAlarm> {
        self.new_alarm
    }

    /// The earliest time at which any candidate crossed its own allocated
    /// threshold; this remains latched for the life of the monitor.
    pub const fn first_alarm(self) -> Option<ChangePointAlarm> {
        self.first_alarm
    }

    pub const fn detected(self) -> bool {
        self.first_alarm.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StartCandidate {
    start_index: u64,
    component_log_e_values: [f64; 3],
    log_e_value: f64,
}

impl StartCandidate {
    const fn new(start_index: u64) -> Self {
        Self {
            start_index,
            component_log_e_values: [0.0; 3],
            log_e_value: 0.0,
        }
    }

    fn observe(&mut self, loss: f64, risk_bound: f64, bet_fractions: [f64; 3]) {
        for (index, fraction) in bet_fractions.into_iter().enumerate() {
            let bet = fraction / risk_bound;
            let factor = 1.0 + bet * (loss - risk_bound);
            self.component_log_e_values[index] += factor.ln();
        }

        let max_log = self
            .component_log_e_values
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
        let scaled_sum = self
            .component_log_e_values
            .into_iter()
            .map(|value| (value - max_log).exp())
            .sum::<f64>()
            / self.component_log_e_values.len() as f64;
        self.log_e_value = max_log + scaled_sum.ln();
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlphaSpentChangePointMonitor {
    risk_bound: f64,
    bet_fractions: [f64; 3],
    total_delta: f64,
    simultaneous_streams: usize,
    max_active_starts: usize,
    time_index: u64,
    candidates: Vec<StartCandidate>,
    first_alarm: Option<ChangePointAlarm>,
}

impl AlphaSpentChangePointMonitor {
    /// Construct a monitor with a global familywise delta allocated across K
    /// simultaneous streams and all possible absolute start indices.
    ///
    /// The candidate bank is bounded to max_active_starts recent start points.
    /// Its size must be 1..=4096. Older candidates are dropped without refunding
    /// their alpha allocation, preserving validity while possibly reducing power.
    pub fn new(
        risk_bound: f64,
        bet_fractions: [f64; 3],
        total_delta: f64,
        simultaneous_streams: usize,
        max_active_starts: usize,
    ) -> Option<Self> {
        if !risk_bound.is_finite()
            || risk_bound <= 0.0
            || risk_bound >= 1.0
            || bet_fractions
                .iter()
                .any(|fraction| !fraction.is_finite() || *fraction <= 0.0 || *fraction >= 1.0)
            || bet_fractions.iter().any(|fraction| {
                let bet = *fraction / risk_bound;
                !bet.is_finite() || bet * risk_bound >= 1.0
            })
            || !total_delta.is_finite()
            || total_delta <= 0.0
            || total_delta >= 1.0
            || simultaneous_streams == 0
            || max_active_starts == 0
            || max_active_starts > MAX_ACTIVE_STARTS_LIMIT
        {
            return None;
        }

        let base_log_threshold =
            (simultaneous_streams as f64).ln() - total_delta.ln() + PI_SQUARED_OVER_SIX.ln();
        if !base_log_threshold.is_finite() {
            return None;
        }

        Some(Self {
            risk_bound,
            bet_fractions,
            total_delta,
            simultaneous_streams,
            max_active_starts,
            time_index: 0,
            candidates: Vec::with_capacity(max_active_starts),
            first_alarm: None,
        })
    }

    pub const fn risk_bound(&self) -> f64 {
        self.risk_bound
    }

    pub const fn bet_fractions(&self) -> [f64; 3] {
        self.bet_fractions
    }

    pub const fn total_delta(&self) -> f64 {
        self.total_delta
    }

    pub const fn simultaneous_streams(&self) -> usize {
        self.simultaneous_streams
    }

    pub const fn max_active_starts(&self) -> usize {
        self.max_active_starts
    }

    pub const fn time_index(&self) -> u64 {
        self.time_index
    }

    pub fn active_candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub const fn first_alarm(&self) -> Option<ChangePointAlarm> {
        self.first_alarm
    }

    /// Return the absolute-start log threshold:
    /// log(K / delta) + log(pi^2 / 6) + 2 log(s + 1).
    ///
    /// Public for independent allocation-schedule checks in tests/tooling.
    pub fn log_threshold_for_start(
        start_index: u64,
        total_delta: f64,
        simultaneous_streams: usize,
    ) -> Option<f64> {
        if !total_delta.is_finite()
            || total_delta <= 0.0
            || total_delta >= 1.0
            || simultaneous_streams == 0
        {
            return None;
        }
        let log_threshold = (simultaneous_streams as f64).ln() - total_delta.ln()
            + PI_SQUARED_OVER_SIX.ln()
            + 2.0 * ((start_index as f64) + 1.0).ln();
        log_threshold.is_finite().then_some(log_threshold)
    }

    /// Observe one bounded loss. Invalid input or exhausted absolute index
    /// returns None without changing the stream, candidates or latched alarm.
    ///
    /// The monitor keeps running after its first alarm for research telemetry,
    /// but first_alarm remains fixed. Callers should use it for first-detection
    /// metrics. There is deliberately no reset method: start-index alpha is
    /// never recycled across a new episode on the same monitor.
    pub fn observe(&mut self, loss: f64) -> Option<ChangePointUpdate> {
        if !loss.is_finite()
            || !(0.0..=1.0).contains(&loss)
            || self.time_index.checked_add(1).is_none()
        {
            return None;
        }

        let time_index = self.time_index;
        let max_age = self.max_active_starts as u64;
        self.candidates
            .retain(|candidate| time_index.saturating_sub(candidate.start_index) < max_age);
        self.candidates.push(StartCandidate::new(time_index));

        let mut best_crossing: Option<ChangePointAlarm> = None;
        for candidate in &mut self.candidates {
            candidate.observe(loss, self.risk_bound, self.bet_fractions);
            let threshold = Self::log_threshold_for_start(
                candidate.start_index,
                self.total_delta,
                self.simultaneous_streams,
            )?;
            if candidate.log_e_value >= threshold {
                let crossing = ChangePointAlarm {
                    start_index: candidate.start_index,
                    detected_at: time_index,
                    log_e_value: candidate.log_e_value,
                    log_threshold: threshold,
                };
                let replace = best_crossing.is_none_or(|current| {
                    crossing.evidence_excess() > current.evidence_excess()
                        || (crossing.evidence_excess() == current.evidence_excess()
                            && crossing.start_index < current.start_index)
                });
                if replace {
                    best_crossing = Some(crossing);
                }
            }
        }

        let new_alarm = if self.first_alarm.is_none() {
            best_crossing
        } else {
            None
        };
        if self.first_alarm.is_none() {
            self.first_alarm = new_alarm;
        }
        self.time_index += 1;

        Some(ChangePointUpdate {
            time_index,
            active_candidates: self.candidates.len(),
            new_alarm,
            first_alarm: self.first_alarm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 0.15;
    const DELTA: f64 = 0.05;
    const STREAMS: usize = 5;
    const BETS: [f64; 3] = [0.02, 0.10, 0.25];

    fn monitor(window: usize) -> AlphaSpentChangePointMonitor {
        AlphaSpentChangePointMonitor::new(EPSILON, BETS, DELTA, STREAMS, window)
            .expect("valid change-point monitor")
    }

    #[test]
    fn rejects_invalid_configuration_and_observation_without_mutation() {
        assert!(AlphaSpentChangePointMonitor::new(0.0, BETS, DELTA, STREAMS, 64).is_none());
        assert!(AlphaSpentChangePointMonitor::new(EPSILON, [0.0, 0.1, 0.2], DELTA, STREAMS, 64).is_none());
        assert!(AlphaSpentChangePointMonitor::new(EPSILON, BETS, DELTA, 0, 64).is_none());
        assert!(AlphaSpentChangePointMonitor::new(EPSILON, BETS, DELTA, STREAMS, 0).is_none());
        assert!(AlphaSpentChangePointMonitor::new(
            EPSILON,
            BETS,
            DELTA,
            STREAMS,
            MAX_ACTIVE_STARTS_LIMIT + 1,
        )
        .is_none());

        let mut subject = monitor(64);
        let initial = subject.clone();
        assert!(subject.observe(f64::NAN).is_none());
        assert!(subject.observe(-0.01).is_none());
        assert!(subject.observe(1.01).is_none());
        assert_eq!(subject, initial);
    }

    #[test]
    fn alpha_spending_is_monotone_and_total_allocation_is_bounded() {
        let mut allocations = 0.0;
        let mut previous = 0.0;
        for start in 0..100_000_u64 {
            let threshold =
                AlphaSpentChangePointMonitor::log_threshold_for_start(start, DELTA, STREAMS)
                    .expect("finite allocation");
            assert!(threshold > previous);
            previous = threshold;
            allocations += (-threshold).exp() * STREAMS as f64;
        }
        assert!(allocations <= DELTA);
        let expected = (STREAMS as f64).ln()
            - DELTA.ln()
            + PI_SQUARED_OVER_SIX.ln()
            + 2.0 * 100_000.0_f64.ln();
        assert!((previous - expected).abs() < 1e-10);
    }

    #[test]
    fn candidate_bank_is_bounded_and_expired_alpha_is_not_recycled() {
        let mut subject = monitor(32);
        for _ in 0..1000 {
            let update = subject.observe(0.0).expect("valid loss");
            assert!(update.active_candidates() <= 32);
        }
        assert_eq!(subject.time_index(), 1000);
        assert_eq!(subject.active_candidate_count(), 32);
        assert!(subject.first_alarm().is_none());
        assert_eq!(
            AlphaSpentChangePointMonitor::log_threshold_for_start(999, DELTA, STREAMS),
            Some(
                (STREAMS as f64).ln() - DELTA.ln()
                    + PI_SQUARED_OVER_SIX.ln()
                    + 2.0 * 1000.0_f64.ln()
            )
        );
    }

    #[test]
    fn late_shift_alarms_from_recent_candidate_after_global_evidence_would_decay() {
        let mut subject = monitor(64);
        for _ in 0..192 {
            let update = subject.observe(0.0).expect("nominal loss");
            assert!(update.new_alarm().is_none());
        }
        let mut alarm = None;
        for _ in 0..64 {
            if let Some(update) = subject.observe(1.0) {
                if let Some(candidate_alarm) = update.new_alarm() {
                    alarm = Some(candidate_alarm);
                    break;
                }
            }
        }
        let alarm = alarm.expect("late high-risk sequence should cross a fresh-start boundary");
        assert!(alarm.start_index() >= 192);
        assert!(alarm.start_index() < alarm.detected_at());
        assert!(alarm.detected_at() - 192 < 64);
        assert!(alarm.log_e_value() >= alarm.log_threshold());
        assert!(subject.first_alarm().is_some());
    }

    #[test]
    fn identical_observations_produce_identical_change_point_evidence() {
        let mut left = monitor(64);
        let mut right = monitor(64);
        for step in 0..256 {
            let loss = if step < 208 { 0.0 } else { 1.0 };
            assert_eq!(left.observe(loss), right.observe(loss));
            assert_eq!(left.active_candidate_count(), right.active_candidate_count());
        }
        assert_eq!(left.first_alarm(), right.first_alarm());
    }
}
