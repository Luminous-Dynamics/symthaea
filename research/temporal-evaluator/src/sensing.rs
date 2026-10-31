//! Evaluator-side observability and aliasing controls for active temporal sensing.
//!
//! This module consumes hidden evaluator event schedules and candidate
//! observation/detection records. It must never be linked into or executed by
//! candidate code. It distinguishes directly observable transients, missed
//! observable events, unobserved events and candidate inferences without a
//! direct sample.
//!
//! The alias-pair control constructs two distinct piecewise-linear signals
//! that agree at all observed timestamps and diverge at evaluator-only probes.
//! This demonstrates an identifiability limit of the submitted schedule.

use std::collections::BTreeMap;
use std::fmt;

const PPM: u128 = 1_000_000;
const MAX_SIGNAL_TIME_US: u64 = 1_000_000_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingEpisode {
    pub episode_id: u64,
    pub duration_us: u64,
    pub observation_budget: u64,
    pub maximum_gap_us: u64,
    pub energy_budget_nj: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationSample {
    pub time_us: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HiddenTransient {
    pub event_id: u64,
    /// Event interval is half-open [start_us, end_us).
    pub start_us: u64,
    pub end_us: u64,
    /// Evaluator-owned amplitude in milli-units, retained for stratification.
    pub amplitude_milli: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateEventDecision {
    pub event_id: u64,
    pub decision_time_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingRun {
    pub episode: SamplingEpisode,
    pub observations: Vec<ObservationSample>,
    pub decisions: Vec<CandidateEventDecision>,
    /// Required when an energy budget is declared. Missing is not zero.
    pub measured_energy_nj: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectionLatency {
    pub event_id: u64,
    pub latency_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingObservabilityScore {
    pub episode_id: u64,
    pub observations_used: u64,
    pub observation_budget: u64,
    pub maximum_inter_observation_gap_us: u64,
    pub maximum_gap_deadline_missed: bool,
    pub observation_budget_exceeded: bool,
    pub energy_budget_exceeded: bool,
    pub measured_energy_nj: Option<u64>,
    pub energy_budget_nj: Option<u64>,
    pub total_hidden_events: usize,
    pub directly_observable_event_ids: Vec<u64>,
    pub detected_observable_event_ids: Vec<u64>,
    pub missed_observable_event_ids: Vec<u64>,
    pub unobserved_event_ids: Vec<u64>,
    pub inferred_without_direct_observation_ids: Vec<u64>,
    pub false_detection_ids: Vec<u64>,
    pub detection_latencies: Vec<DetectionLatency>,
    pub event_observability_ppm: u32,
    pub observable_event_recall_ppm: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplingEvaluationError {
    ZeroEpisodeDuration,
    ZeroMaximumGap,
    ObservationOutOfRange { index: usize },
    ObservationNotStrictlyIncreasing { index: usize },
    EventOutOfRange { event_id: u64 },
    DuplicateEventId { event_id: u64 },
    DuplicateDecision { event_id: u64 },
    DecisionOutOfRange { event_id: u64 },
    FutureEventLeak { event_id: u64, first_evidence_us: u64, decision_us: u64 },
    EnergyMeasurementMissing,
    ArithmeticOverflow,
}

impl fmt::Display for SamplingEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEpisodeDuration => write!(f, "sampling episode duration must be positive"),
            Self::ZeroMaximumGap => write!(f, "maximum observation gap must be positive"),
            Self::ObservationOutOfRange { index } => write!(f, "observation {index} is outside the episode"),
            Self::ObservationNotStrictlyIncreasing { index } => write!(f, "observation timestamps are not strictly increasing at index {index}"),
            Self::EventOutOfRange { event_id } => write!(f, "hidden event {event_id} has invalid episode bounds"),
            Self::DuplicateEventId { event_id } => write!(f, "duplicate hidden event ID {event_id}"),
            Self::DuplicateDecision { event_id } => write!(f, "duplicate candidate decision for event {event_id}"),
            Self::DecisionOutOfRange { event_id } => write!(f, "candidate decision for event {event_id} is outside the episode"),
            Self::FutureEventLeak { event_id, first_evidence_us, decision_us } => write!(f, "candidate decision for event {event_id} at {decision_us} predates first sampled evidence at {first_evidence_us}"),
            Self::EnergyMeasurementMissing => write!(f, "episode declares an energy budget but measured energy is missing"),
            Self::ArithmeticOverflow => write!(f, "sampling score arithmetic overflowed"),
        }
    }
}

impl std::error::Error for SamplingEvaluationError {}

/// Score observation coverage and transient decisions against evaluator truth.
///
/// An event is directly observable only if a sample timestamp lies in its
/// half-open event interval. A candidate decision for an unobserved event is
/// retained as an inference-without-direct-sample, not counted as an observed
/// hit. Unknown event IDs are false detections. Missing observable events
/// remain explicit misses.
pub fn score_sampling_observability(
    run: &SamplingRun,
    hidden_events: &[HiddenTransient],
) -> Result<SamplingObservabilityScore, SamplingEvaluationError> {
    let episode = &run.episode;
    if episode.duration_us == 0 {
        return Err(SamplingEvaluationError::ZeroEpisodeDuration);
    }
    if episode.maximum_gap_us == 0 {
        return Err(SamplingEvaluationError::ZeroMaximumGap);
    }

    let mut previous = None;
    for (index, sample) in run.observations.iter().enumerate() {
        if sample.time_us > episode.duration_us {
            return Err(SamplingEvaluationError::ObservationOutOfRange { index });
        }
        if previous.is_some_and(|prior| sample.time_us <= prior) {
            return Err(SamplingEvaluationError::ObservationNotStrictlyIncreasing { index });
        }
        previous = Some(sample.time_us);
    }

    let mut events_by_id = BTreeMap::new();
    for event in hidden_events {
        if event.start_us >= event.end_us || event.end_us > episode.duration_us {
            return Err(SamplingEvaluationError::EventOutOfRange { event_id: event.event_id });
        }
        if events_by_id.insert(event.event_id, *event).is_some() {
            return Err(SamplingEvaluationError::DuplicateEventId { event_id: event.event_id });
        }
    }

    let mut decisions_by_id = BTreeMap::new();
    for decision in &run.decisions {
        if decision.decision_time_us > episode.duration_us {
            return Err(SamplingEvaluationError::DecisionOutOfRange { event_id: decision.event_id });
        }
        if decisions_by_id.insert(decision.event_id, decision.decision_time_us).is_some() {
            return Err(SamplingEvaluationError::DuplicateDecision { event_id: decision.event_id });
        }
    }

    if episode.energy_budget_nj.is_some() && run.measured_energy_nj.is_none() {
        return Err(SamplingEvaluationError::EnergyMeasurementMissing);
    }

    let mut directly_observable = Vec::new();
    let mut detected_observable = Vec::new();
    let mut missed_observable = Vec::new();
    let mut unobserved = Vec::new();
    let mut inferred_without_direct_observation = Vec::new();
    let mut false_detections = Vec::new();
    let mut detection_latencies = Vec::new();

    for event in hidden_events {
        let first_evidence = run.observations.iter()
            .map(|sample| sample.time_us)
            .find(|time| *time >= event.start_us && *time < event.end_us);
        let decision_time = decisions_by_id.get(&event.event_id).copied();

        match (first_evidence, decision_time) {
            (Some(first), Some(decision)) => {
                if decision < first {
                    return Err(SamplingEvaluationError::FutureEventLeak {
                        event_id: event.event_id,
                        first_evidence_us: first,
                        decision_us: decision,
                    });
                }
                directly_observable.push(event.event_id);
                detected_observable.push(event.event_id);
                detection_latencies.push(DetectionLatency {
                    event_id: event.event_id,
                    latency_us: decision.saturating_sub(event.start_us),
                });
            }
            (Some(_), None) => {
                directly_observable.push(event.event_id);
                missed_observable.push(event.event_id);
            }
            (None, Some(_)) => {
                unobserved.push(event.event_id);
                inferred_without_direct_observation.push(event.event_id);
            }
            (None, None) => unobserved.push(event.event_id),
        }
    }

    for event_id in decisions_by_id.keys() {
        if !events_by_id.contains_key(event_id) {
            false_detections.push(*event_id);
        }
    }

    directly_observable.sort_unstable();
    detected_observable.sort_unstable();
    missed_observable.sort_unstable();
    unobserved.sort_unstable();
    inferred_without_direct_observation.sort_unstable();
    false_detections.sort_unstable();
    detection_latencies.sort_by_key(|entry| entry.event_id);

    let max_gap = maximum_inter_observation_gap(episode.duration_us, &run.observations);
    let total_events = hidden_events.len();
    let event_observability_ppm = ratio_ppm(directly_observable.len(), total_events)?;
    let observable_event_recall_ppm = if directly_observable.is_empty() {
        None
    } else {
        Some(ratio_ppm(detected_observable.len(), directly_observable.len())?)
    };

    Ok(SamplingObservabilityScore {
        episode_id: episode.episode_id,
        observations_used: u64::try_from(run.observations.len()).map_err(|_| SamplingEvaluationError::ArithmeticOverflow)?,
        observation_budget: episode.observation_budget,
        maximum_inter_observation_gap_us: max_gap,
        maximum_gap_deadline_missed: max_gap > episode.maximum_gap_us,
        observation_budget_exceeded: run.observations.len() as u128 > u128::from(episode.observation_budget),
        energy_budget_exceeded: matches!((episode.energy_budget_nj, run.measured_energy_nj), (Some(budget), Some(used)) if used > budget),
        measured_energy_nj: run.measured_energy_nj,
        energy_budget_nj: episode.energy_budget_nj,
        total_hidden_events: total_events,
        directly_observable_event_ids: directly_observable,
        detected_observable_event_ids: detected_observable,
        missed_observable_event_ids: missed_observable,
        unobserved_event_ids: unobserved,
        inferred_without_direct_observation_ids: inferred_without_direct_observation,
        false_detection_ids: false_detections,
        detection_latencies,
        event_observability_ppm,
        observable_event_recall_ppm,
    })
}

fn maximum_inter_observation_gap(duration_us: u64, observations: &[ObservationSample]) -> u64 {
    let mut maximum = 0_u64;
    let mut previous = 0_u64;
    for sample in observations {
        maximum = maximum.max(sample.time_us.saturating_sub(previous));
        previous = sample.time_us;
    }
    maximum.max(duration_us.saturating_sub(previous))
}

fn ratio_ppm(numerator: usize, denominator: usize) -> Result<u32, SamplingEvaluationError> {
    if denominator == 0 {
        return Ok(0);
    }
    let value = u128::try_from(numerator)
        .map_err(|_| SamplingEvaluationError::ArithmeticOverflow)?
        .checked_mul(PPM)
        .ok_or(SamplingEvaluationError::ArithmeticOverflow)?
        / u128::try_from(denominator)
            .map_err(|_| SamplingEvaluationError::ArithmeticOverflow)?;
    u32::try_from(value).map_err(|_| SamplingEvaluationError::ArithmeticOverflow)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignalKnot {
    pub time_us: u64,
    /// Signal value in milli-units; avoids floating-point equality in alias tests.
    pub value_milli: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PiecewiseLinearSignal {
    pub knots: Vec<SignalKnot>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AliasingQualification {
    pub observed_timestamp_count: usize,
    pub hidden_probe_count: usize,
    pub maximum_hidden_difference_milli: u64,
    pub divergence_probe_times_us: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AliasingError {
    InvalidSignal { index: usize },
    SignalTimeNotIncreasing { signal: usize, knot: usize },
    SignalTimeOutOfRange { signal: usize, knot: usize },
    ObservationTimeNotIncreasing { index: usize },
    HiddenProbeTimeNotIncreasing { index: usize },
    HiddenProbeMustBeBetweenObservations { time_us: u64 },
    EmptyObservationSchedule,
    EmptyHiddenProbeSchedule,
    ObservationValuesDiffer { time_us: u64 },
    NoHiddenDivergence,
    ArithmeticOverflow,
}

impl fmt::Display for AliasingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSignal { index } => write!(f, "signal {index} requires at least two knots"),
            Self::SignalTimeNotIncreasing { signal, knot } => write!(f, "signal {signal} knot time is not increasing at index {knot}"),
            Self::SignalTimeOutOfRange { signal, knot } => write!(f, "signal {signal} knot is outside the supported time range at index {knot}"),
            Self::ObservationTimeNotIncreasing { index } => write!(f, "observation timestamps are not strictly increasing at index {index}"),
            Self::HiddenProbeTimeNotIncreasing { index } => write!(f, "hidden probe timestamps are not strictly increasing at index {index}"),
            Self::HiddenProbeMustBeBetweenObservations { time_us } => write!(f, "hidden probe at {time_us} must lie strictly between observed timestamps"),
            Self::EmptyObservationSchedule => write!(f, "alias control requires at least one observation"),
            Self::EmptyHiddenProbeSchedule => write!(f, "alias control requires at least one hidden probe"),
            Self::ObservationValuesDiffer { time_us } => write!(f, "alias pair differs at observed timestamp {time_us}"),
            Self::NoHiddenDivergence => write!(f, "alias pair does not diverge at any hidden probe"),
            Self::ArithmeticOverflow => write!(f, "alias interpolation arithmetic overflowed"),
        }
    }
}

impl std::error::Error for AliasingError {}

/// Verify that two piecewise-linear latent signals agree at every observed
/// timestamp within the declared tolerance but diverge at evaluator-only probes.
pub fn verify_alias_pair(
    observations: &[ObservationSample],
    left: &PiecewiseLinearSignal,
    right: &PiecewiseLinearSignal,
    hidden_probe_times_us: &[u64],
    tolerance_milli: u32,
) -> Result<AliasingQualification, AliasingError> {
    validate_signal(left, 0)?;
    validate_signal(right, 1)?;
    if observations.is_empty() {
        return Err(AliasingError::EmptyObservationSchedule);
    }
    if hidden_probe_times_us.is_empty() {
        return Err(AliasingError::EmptyHiddenProbeSchedule);
    }
    for (index, pair) in observations.windows(2).enumerate() {
        if pair[1].time_us <= pair[0].time_us {
            return Err(AliasingError::ObservationTimeNotIncreasing { index: index + 1 });
        }
    }
    for (index, pair) in hidden_probe_times_us.windows(2).enumerate() {
        if pair[1] <= pair[0] {
            return Err(AliasingError::HiddenProbeTimeNotIncreasing { index: index + 1 });
        }
    }

    let first_observed = observations.first().expect("not empty").time_us;
    let last_observed = observations.last().expect("not empty").time_us;
    for sample in observations {
        let left_value = interpolate_fraction(left, sample.time_us, 0)?;
        let right_value = interpolate_fraction(right, sample.time_us, 1)?;
        if difference_exceeds(left_value, right_value, tolerance_milli)? {
            return Err(AliasingError::ObservationValuesDiffer { time_us: sample.time_us });
        }
    }

    let mut divergence_probe_times_us = Vec::new();
    let mut maximum_hidden_difference_milli = 0_u64;
    for time_us in hidden_probe_times_us {
        if *time_us <= first_observed || *time_us >= last_observed
            || observations.iter().any(|sample| sample.time_us == *time_us)
        {
            return Err(AliasingError::HiddenProbeMustBeBetweenObservations { time_us: *time_us });
        }
        let left_value = interpolate_fraction(left, *time_us, 0)?;
        let right_value = interpolate_fraction(right, *time_us, 1)?;
        let (exceeds, rounded_difference) = difference_and_round(left_value, right_value, tolerance_milli)?;
        maximum_hidden_difference_milli = maximum_hidden_difference_milli.max(rounded_difference);
        if exceeds {
            divergence_probe_times_us.push(*time_us);
        }
    }

    if divergence_probe_times_us.is_empty() {
        return Err(AliasingError::NoHiddenDivergence);
    }
    Ok(AliasingQualification {
        observed_timestamp_count: observations.len(),
        hidden_probe_count: hidden_probe_times_us.len(),
        maximum_hidden_difference_milli,
        divergence_probe_times_us,
    })
}

#[derive(Clone, Copy)]
struct Rational {
    numerator: i128,
    denominator: i128,
}

fn validate_signal(signal: &PiecewiseLinearSignal, signal_index: usize) -> Result<(), AliasingError> {
    if signal.knots.len() < 2 {
        return Err(AliasingError::InvalidSignal { index: signal_index });
    }
    for (index, knot) in signal.knots.iter().enumerate() {
        if knot.time_us > MAX_SIGNAL_TIME_US {
            return Err(AliasingError::SignalTimeOutOfRange { signal: signal_index, knot: index });
        }
        if index > 0 && knot.time_us <= signal.knots[index - 1].time_us {
            return Err(AliasingError::SignalTimeNotIncreasing { signal: signal_index, knot: index });
        }
    }
    Ok(())
}

fn interpolate_fraction(
    signal: &PiecewiseLinearSignal,
    time_us: u64,
    signal_index: usize,
) -> Result<Rational, AliasingError> {
    let first = signal.knots.first().ok_or(AliasingError::InvalidSignal { index: signal_index })?;
    let last = signal.knots.last().ok_or(AliasingError::InvalidSignal { index: signal_index })?;
    if time_us < first.time_us || time_us > last.time_us || time_us > MAX_SIGNAL_TIME_US {
        return Err(AliasingError::SignalTimeOutOfRange { signal: signal_index, knot: signal.knots.len() - 1 });
    }
    for knot in &signal.knots {
        if knot.time_us == time_us {
            return Ok(Rational { numerator: i128::from(knot.value_milli), denominator: 1 });
        }
    }
    let pair = signal.knots.windows(2)
        .find(|pair| pair[0].time_us < time_us && time_us < pair[1].time_us)
        .ok_or(AliasingError::SignalTimeOutOfRange { signal: signal_index, knot: signal.knots.len() - 1 })?;
    let dt = i128::from(pair[1].time_us - pair[0].time_us);
    let offset = i128::from(time_us - pair[0].time_us);
    let y0 = i128::from(pair[0].value_milli);
    let delta_y = i128::from(pair[1].value_milli)
        .checked_sub(y0).ok_or(AliasingError::ArithmeticOverflow)?;
    let numerator = y0.checked_mul(dt)
        .and_then(|value| delta_y.checked_mul(offset).and_then(|slope| value.checked_add(slope)))
        .ok_or(AliasingError::ArithmeticOverflow)?;
    Ok(Rational { numerator, denominator: dt })
}

fn difference_exceeds(left: Rational, right: Rational, tolerance_milli: u32) -> Result<bool, AliasingError> {
    Ok(difference_and_round(left, right, tolerance_milli)?.0)
}

fn difference_and_round(
    left: Rational,
    right: Rational,
    tolerance_milli: u32,
) -> Result<(bool, u64), AliasingError> {
    let left_scaled = left.numerator.checked_mul(right.denominator).ok_or(AliasingError::ArithmeticOverflow)?;
    let right_scaled = right.numerator.checked_mul(left.denominator).ok_or(AliasingError::ArithmeticOverflow)?;
    let difference = left_scaled.checked_sub(right_scaled)
        .ok_or(AliasingError::ArithmeticOverflow)?
        .checked_abs().ok_or(AliasingError::ArithmeticOverflow)?;
    let denominator = left.denominator.checked_mul(right.denominator).ok_or(AliasingError::ArithmeticOverflow)?;
    let tolerance = i128::from(tolerance_milli).checked_mul(denominator).ok_or(AliasingError::ArithmeticOverflow)?;
    let rounded = difference.checked_add(denominator - 1)
        .ok_or(AliasingError::ArithmeticOverflow)? / denominator;
    let rounded = u64::try_from(rounded).map_err(|_| AliasingError::ArithmeticOverflow)?;
    Ok((difference > tolerance, rounded))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn episode() -> SamplingEpisode {
        SamplingEpisode {
            episode_id: 91,
            duration_us: 100,
            observation_budget: 5,
            maximum_gap_us: 40,
            energy_budget_nj: Some(1_000),
        }
    }

    fn signal(knots: &[(u64, i32)]) -> PiecewiseLinearSignal {
        PiecewiseLinearSignal {
            knots: knots.iter().map(|(time_us, value_milli)| SignalKnot {
                time_us: *time_us,
                value_milli: *value_milli,
            }).collect(),
        }
    }

    #[test]
    fn observed_transient_is_detected_with_latency_and_budget_reported() {
        let run = SamplingRun {
            episode: episode(),
            observations: vec![ObservationSample { time_us: 0 }, ObservationSample { time_us: 20 }, ObservationSample { time_us: 40 }, ObservationSample { time_us: 80 }],
            decisions: vec![CandidateEventDecision { event_id: 7, decision_time_us: 25 }],
            measured_energy_nj: Some(900),
        };
        let events = [HiddenTransient { event_id: 7, start_us: 20, end_us: 30, amplitude_milli: 2_000 }];
        let score = score_sampling_observability(&run, &events).unwrap();
        assert_eq!(score.detected_observable_event_ids, vec![7]);
        assert!(score.missed_observable_event_ids.is_empty());
        assert_eq!(score.detection_latencies, vec![DetectionLatency { event_id: 7, latency_us: 5 }]);
        assert_eq!(score.event_observability_ppm, 1_000_000);
        assert_eq!(score.observable_event_recall_ppm, Some(1_000_000));
        assert!(!score.maximum_gap_deadline_missed);
    }

    #[test]
    fn observable_but_undetected_transient_is_a_miss() {
        let run = SamplingRun {
            episode: episode(),
            observations: vec![ObservationSample { time_us: 10 }, ObservationSample { time_us: 40 }],
            decisions: vec![],
            measured_energy_nj: Some(100),
        };
        let events = [HiddenTransient { event_id: 8, start_us: 20, end_us: 30, amplitude_milli: 900 }];
        let score = score_sampling_observability(&run, &events).unwrap();
        assert_eq!(score.missed_observable_event_ids, vec![8]);
        assert_eq!(score.observable_event_recall_ppm, Some(0));
    }

    #[test]
    fn transient_between_samples_is_unobserved_not_a_model_miss() {
        let run = SamplingRun {
            episode: episode(),
            observations: vec![ObservationSample { time_us: 10 }, ObservationSample { time_us: 40 }],
            decisions: vec![],
            measured_energy_nj: Some(100),
        };
        let events = [HiddenTransient { event_id: 9, start_us: 20, end_us: 30, amplitude_milli: 1_500 }];
        let score = score_sampling_observability(&run, &events).unwrap();
        assert_eq!(score.unobserved_event_ids, vec![9]);
        assert!(score.missed_observable_event_ids.is_empty());
        assert_eq!(score.event_observability_ppm, 0);
        assert_eq!(score.observable_event_recall_ppm, None);
    }

    #[test]
    fn inference_without_direct_sample_is_separate_from_observed_hit() {
        let run = SamplingRun {
            episode: episode(),
            observations: vec![ObservationSample { time_us: 10 }, ObservationSample { time_us: 40 }],
            decisions: vec![CandidateEventDecision { event_id: 9, decision_time_us: 40 }],
            measured_energy_nj: Some(100),
        };
        let events = [HiddenTransient { event_id: 9, start_us: 20, end_us: 30, amplitude_milli: 1_500 }];
        let score = score_sampling_observability(&run, &events).unwrap();
        assert_eq!(score.inferred_without_direct_observation_ids, vec![9]);
        assert!(score.detected_observable_event_ids.is_empty());
    }

    #[test]
    fn future_event_leak_is_rejected() {
        let run = SamplingRun {
            episode: episode(),
            observations: vec![ObservationSample { time_us: 20 }],
            decisions: vec![CandidateEventDecision { event_id: 7, decision_time_us: 19 }],
            measured_energy_nj: Some(100),
        };
        let events = [HiddenTransient { event_id: 7, start_us: 10, end_us: 30, amplitude_milli: 1_000 }];
        assert!(matches!(score_sampling_observability(&run, &events), Err(SamplingEvaluationError::FutureEventLeak { .. })));
    }

    #[test]
    fn invalid_schedule_events_and_missing_energy_fail_closed() {
        let mut run = SamplingRun {
            episode: episode(),
            observations: vec![ObservationSample { time_us: 10 }, ObservationSample { time_us: 10 }],
            decisions: vec![],
            measured_energy_nj: Some(100),
        };
        assert!(matches!(score_sampling_observability(&run, &[]), Err(SamplingEvaluationError::ObservationNotStrictlyIncreasing { .. })));
        run.observations = vec![ObservationSample { time_us: 101 }];
        assert!(matches!(score_sampling_observability(&run, &[]), Err(SamplingEvaluationError::ObservationOutOfRange { .. })));
        run.observations = vec![];
        run.measured_energy_nj = None;
        assert_eq!(score_sampling_observability(&run, &[]), Err(SamplingEvaluationError::EnergyMeasurementMissing));
    }

    #[test]
    fn budget_overruns_and_missed_maximum_gap_are_reported() {
        let mut run = SamplingRun {
            episode: episode(),
            observations: vec![ObservationSample { time_us: 10 }],
            decisions: vec![],
            measured_energy_nj: Some(1_001),
        };
        let score = score_sampling_observability(&run, &[]).unwrap();
        assert!(score.maximum_gap_deadline_missed);
        assert!(score.energy_budget_exceeded);
        run.episode.observation_budget = 0;
        let score = score_sampling_observability(&run, &[]).unwrap();
        assert!(score.observation_budget_exceeded);
    }

    #[test]
    fn alias_pair_equal_at_samples_but_different_between_samples_is_qualified() {
        let observations = [ObservationSample { time_us: 0 }, ObservationSample { time_us: 10 }];
        let left = signal(&[(0, 0), (10, 0)]);
        let right = signal(&[(0, 0), (5, 15_000), (10, 0)]);
        let result = verify_alias_pair(&observations, &left, &right, &[5], 0).unwrap();
        assert_eq!(result.observed_timestamp_count, 2);
        assert_eq!(result.divergence_probe_times_us, vec![5]);
        assert_eq!(result.maximum_hidden_difference_milli, 15_000);
    }

    #[test]
    fn alias_pair_that_differs_at_an_observation_is_rejected() {
        let observations = [ObservationSample { time_us: 0 }, ObservationSample { time_us: 5 }, ObservationSample { time_us: 10 }];
        let left = signal(&[(0, 0), (10, 0)]);
        let right = signal(&[(0, 0), (5, 15_000), (10, 0)]);
        assert_eq!(verify_alias_pair(&observations, &left, &right, &[7], 0), Err(AliasingError::ObservationValuesDiffer { time_us: 5 }));
    }

    #[test]
    fn alias_pair_requires_hidden_probe_between_observations() {
        let observations = [ObservationSample { time_us: 0 }, ObservationSample { time_us: 10 }];
        let left = signal(&[(0, 0), (10, 0)]);
        let right = signal(&[(0, 0), (5, 0), (10, 0)]);
        assert_eq!(verify_alias_pair(&observations, &left, &right, &[5], 0), Err(AliasingError::NoHiddenDivergence));
        assert!(matches!(verify_alias_pair(&observations, &left, &right, &[0], 0), Err(AliasingError::HiddenProbeMustBeBetweenObservations { time_us: 0 })));
    }
}
