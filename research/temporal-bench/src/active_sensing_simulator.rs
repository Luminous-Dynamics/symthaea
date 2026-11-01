//! Deterministic hidden-state environment for active-sensing policy comparisons.
//!
//! Candidate policies receive only PublicObservation and return PolicyAction.
//! Hidden event intervals, scenario seeds, and evaluator metrics are private to
//! this module. This is an API/data-flow boundary for research fixtures, not a
//! security sandbox: confirmatory runs still require a separately permissioned
//! candidate process and evaluator process.

use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;

const MAX_EPISODE_STEPS: u64 = 5_000_000;
const MAX_WAKE_INTERVAL_STEPS: u32 = 1_024;
const PROBABILITY_SCALE: u32 = 1_000_000;
const WAKE_ENERGY_NJ: u64 = 10;
const LOW_POWER_SAMPLE_ENERGY_NJ: u64 = 20;
const HIGH_FIDELITY_SAMPLE_ENERGY_NJ: u64 = 100;
const LOW_POWER_FALSE_POSITIVE_PPM: u32 = 80_000;
const LOW_POWER_FALSE_NEGATIVE_PPM: u32 = 140_000;
const HIGH_FIDELITY_FALSE_POSITIVE_PPM: u32 = 25_000;
const HIGH_FIDELITY_FALSE_NEGATIVE_PPM: u32 = 40_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSensingSimulationError {
    pub field: String,
    pub message: String,
}

impl ActiveSensingSimulationError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for ActiveSensingSimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for ActiveSensingSimulationError {}

/// Evaluator-side shift configuration. It is never passed to candidate code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActiveSensingShift {
    Stable,
    AbruptSensorShift {
        start_step: u64,
        extra_error_ppm: u32,
    },
    GradualSensorShift {
        start_step: u64,
        end_step: u64,
        max_extra_error_ppm: u32,
    },
    CompoundShift {
        start_step: u64,
        extra_sensor_error_ppm: u32,
        event_rate_multiplier_ppm: u32,
    },
}

/// Evaluator-side scenario definition. The seed and shift schedule are private
/// to the simulator and must not be forwarded to a candidate process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveSensingScenario {
    pub seed: u64,
    pub steps: u64,
    pub event_rate_ppm: u32,
    pub min_event_duration_steps: u32,
    pub max_event_duration_steps: u32,
    pub shift: ActiveSensingShift,
}

impl ActiveSensingScenario {
    pub fn validate(&self) -> Result<(), ActiveSensingSimulationError> {
        if self.steps == 0 || self.steps > MAX_EPISODE_STEPS {
            return Err(ActiveSensingSimulationError::new(
                "scenario.steps",
                format!("steps must be in 1..={MAX_EPISODE_STEPS}"),
            ));
        }
        if self.event_rate_ppm > PROBABILITY_SCALE {
            return Err(ActiveSensingSimulationError::new(
                "scenario.event_rate_ppm",
                "event rate must be in [0, 1,000,000] ppm",
            ));
        }
        if self.min_event_duration_steps == 0
            || self.min_event_duration_steps > self.max_event_duration_steps
        {
            return Err(ActiveSensingSimulationError::new(
                "scenario.event_duration",
                "event duration bounds must be positive and ordered",
            ));
        }
        match self.shift {
            ActiveSensingShift::Stable => {}
            ActiveSensingShift::AbruptSensorShift { start_step, extra_error_ppm } => {
                if start_step >= self.steps || extra_error_ppm > PROBABILITY_SCALE {
                    return Err(ActiveSensingSimulationError::new(
                        "scenario.shift",
                        "abrupt shift start must be within the episode and error <= 1,000,000 ppm",
                    ));
                }
            }
            ActiveSensingShift::GradualSensorShift {
                start_step,
                end_step,
                max_extra_error_ppm,
            } => {
                if start_step >= end_step
                    || end_step > self.steps
                    || max_extra_error_ppm > PROBABILITY_SCALE
                {
                    return Err(ActiveSensingSimulationError::new(
                        "scenario.shift",
                        "gradual shift requires 0 <= start < end <= steps and error <= 1,000,000 ppm",
                    ));
                }
            }
            ActiveSensingShift::CompoundShift {
                start_step,
                extra_sensor_error_ppm,
                event_rate_multiplier_ppm,
            } => {
                if start_step >= self.steps
                    || extra_sensor_error_ppm > PROBABILITY_SCALE
                    || event_rate_multiplier_ppm > 10 * PROBABILITY_SCALE
                {
                    return Err(ActiveSensingSimulationError::new(
                        "scenario.shift",
                        "compound shift bounds are invalid",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActiveSensingSensor {
    LowPower,
    HighFidelity,
}

/// The candidate sees a past measurement, never the hidden state at step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicSensorMeasurement {
    pub sample_step: u64,
    pub sensor: ActiveSensingSensor,
    pub positive: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicObservation {
    pub step: u64,
    pub last_measurement: Option<PublicSensorMeasurement>,
    pub last_measurement_age_steps: Option<u64>,
    pub samples_acquired: u64,
    pub low_power_samples: u64,
    pub high_fidelity_samples: u64,
}

/// One causal policy response. sensor=None means sleep without sampling.
/// Costs other than sensor/wake energy are candidate-declared counters, not
/// physical measurements; the system track must independently measure them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyAction {
    pub alarm: bool,
    pub sensor: Option<ActiveSensingSensor>,
    pub next_wake_after_steps: u32,
    pub compute_cycles: u64,
    pub transmit_bytes: u64,
    pub memory_write_bytes: u64,
}

impl PolicyAction {
    pub fn sleep(next_wake_after_steps: u32) -> Self {
        Self {
            alarm: false,
            sensor: None,
            next_wake_after_steps,
            compute_cycles: 0,
            transmit_bytes: 0,
            memory_write_bytes: 0,
        }
    }
}

/// Candidate policies are called only with the public observation contract.
/// A runner must instantiate a fresh policy per episode and pin its artifact.
pub trait ActiveSensingPolicy {
    fn decide(&mut self, observation: &PublicObservation) -> PolicyAction;
}

/// Fixed-rate comparator. It alarms on a positive latest measurement and
/// samples the selected sensor at the requested fixed interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedRateBaseline {
    pub sensor: ActiveSensingSensor,
    pub interval_steps: u32,
}

impl ActiveSensingPolicy for FixedRateBaseline {
    fn decide(&mut self, observation: &PublicObservation) -> PolicyAction {
        PolicyAction {
            alarm: observation.last_measurement.is_some_and(|m| m.positive),
            sensor: Some(self.sensor),
            next_wake_after_steps: self.interval_steps,
            compute_cycles: 1,
            transmit_bytes: 0,
            memory_write_bytes: 0,
        }
    }
}

/// Periodic confirmation comparator: cheap sensing while quiet, then a
/// high-fidelity observation immediately after a positive low-power reading.
/// It alarms only after a positive high-fidelity reading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HysteresisAdaptiveBaseline {
    pub quiet_interval_steps: u32,
    pub active_interval_steps: u32,
}

impl ActiveSensingPolicy for HysteresisAdaptiveBaseline {
    fn decide(&mut self, observation: &PublicObservation) -> PolicyAction {
        let latest = observation.last_measurement;
        let confirmed = latest.is_some_and(|m| {
            m.sensor == ActiveSensingSensor::HighFidelity && m.positive
        });
        let requires_confirmation = latest.is_some_and(|m| {
            m.sensor == ActiveSensingSensor::LowPower && m.positive
        });

        if confirmed {
            PolicyAction {
                alarm: true,
                sensor: Some(ActiveSensingSensor::HighFidelity),
                next_wake_after_steps: self.active_interval_steps,
                compute_cycles: 2,
                transmit_bytes: 0,
                memory_write_bytes: 8,
            }
        } else if requires_confirmation {
            PolicyAction {
                alarm: false,
                sensor: Some(ActiveSensingSensor::HighFidelity),
                next_wake_after_steps: self.active_interval_steps,
                compute_cycles: 2,
                transmit_bytes: 0,
                memory_write_bytes: 8,
            }
        } else {
            PolicyAction {
                alarm: false,
                sensor: Some(ActiveSensingSensor::LowPower),
                next_wake_after_steps: self.quiet_interval_steps,
                compute_cycles: 1,
                transmit_bytes: 0,
                memory_write_bytes: 0,
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HiddenEvent {
    start_step: u64,
    end_step_exclusive: u64,
}

#[derive(Clone, Debug)]
struct HiddenEpisode {
    events: Vec<HiddenEvent>,
    event_at_step: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper_exclusive: u64) -> u64 {
        if upper_exclusive == 0 { 0 } else { self.next_u64() % upper_exclusive }
    }
}

fn scenario_event_rate_ppm(scenario: &ActiveSensingScenario, step: u64) -> u32 {
    match scenario.shift {
        ActiveSensingShift::CompoundShift {
            start_step,
            event_rate_multiplier_ppm,
            ..
        } if step >= start_step => {
            let scaled = u64::from(scenario.event_rate_ppm)
                .saturating_mul(u64::from(event_rate_multiplier_ppm))
                / u64::from(PROBABILITY_SCALE);
            scaled.min(u64::from(PROBABILITY_SCALE)) as u32
        }
        _ => scenario.event_rate_ppm,
    }
}

fn extra_sensor_error_ppm(scenario: &ActiveSensingScenario, step: u64) -> u32 {
    match scenario.shift {
        ActiveSensingShift::AbruptSensorShift { start_step, extra_error_ppm }
            if step >= start_step => extra_error_ppm,
        ActiveSensingShift::GradualSensorShift {
            start_step,
            end_step,
            max_extra_error_ppm,
        } if step >= start_step => {
            let elapsed = step.saturating_sub(start_step).min(end_step - start_step);
            let span = end_step - start_step;
            (u64::from(max_extra_error_ppm) * elapsed / span) as u32
        }
        ActiveSensingShift::CompoundShift {
            start_step,
            extra_sensor_error_ppm,
            ..
        } if step >= start_step => extra_sensor_error_ppm,
        _ => 0,
    }
}

fn generate_hidden_episode(scenario: &ActiveSensingScenario) -> HiddenEpisode {
    let mut rng = SplitMix64::new(scenario.seed ^ 0x48494444454E5F31);
    let mut events = Vec::new();
    let mut remaining = 0_u32;
    let mut active_start = 0_u64;
    let duration_span = u64::from(
        scenario.max_event_duration_steps - scenario.min_event_duration_steps,
    ) + 1;

    for step in 0..scenario.steps {
        if remaining == 0
            && rng.below(u64::from(PROBABILITY_SCALE))
                < u64::from(scenario_event_rate_ppm(scenario, step))
        {
            let duration = scenario.min_event_duration_steps
                + rng.below(duration_span) as u32;
            remaining = duration;
            active_start = step;
        }
        if remaining > 0 {
            remaining -= 1;
            if remaining == 0 {
                events.push(HiddenEvent {
                    start_step: active_start,
                    end_step_exclusive: step + 1,
                });
            }
        }
    }
    if remaining > 0 {
        events.push(HiddenEvent {
            start_step: active_start,
            end_step_exclusive: scenario.steps,
        });
    }

    let mut event_at_step = vec![None; scenario.steps as usize];
    for (event_index, event) in events.iter().enumerate() {
        for step in event.start_step..event.end_step_exclusive {
            event_at_step[step as usize] = Some(event_index);
        }
    }
    HiddenEpisode { events, event_at_step }
}

fn sensor_energy_nj(sensor: ActiveSensingSensor) -> u64 {
    match sensor {
        ActiveSensingSensor::LowPower => LOW_POWER_SAMPLE_ENERGY_NJ,
        ActiveSensingSensor::HighFidelity => HIGH_FIDELITY_SAMPLE_ENERGY_NJ,
    }
}

fn acquire_measurement(
    rng: &mut SplitMix64,
    scenario: &ActiveSensingScenario,
    step: u64,
    hidden_positive: bool,
    sensor: ActiveSensingSensor,
) -> PublicSensorMeasurement {
    let (base_fp, base_fn) = match sensor {
        ActiveSensingSensor::LowPower => (
            LOW_POWER_FALSE_POSITIVE_PPM,
            LOW_POWER_FALSE_NEGATIVE_PPM,
        ),
        ActiveSensingSensor::HighFidelity => (
            HIGH_FIDELITY_FALSE_POSITIVE_PPM,
            HIGH_FIDELITY_FALSE_NEGATIVE_PPM,
        ),
    };
    let extra = extra_sensor_error_ppm(scenario, step);
    let false_positive = base_fp.saturating_add(extra).min(PROBABILITY_SCALE);
    let false_negative = base_fn.saturating_add(extra).min(PROBABILITY_SCALE);
    let draw = rng.below(u64::from(PROBABILITY_SCALE)) as u32;
    let positive = if hidden_positive {
        draw >= false_negative
    } else {
        draw < false_positive
    };
    PublicSensorMeasurement { sample_step: step, sensor, positive }
}

fn ppm(numerator: u64, denominator: u64) -> Option<u32> {
    if denominator == 0 {
        None
    } else {
        Some(((u128::from(numerator) * u128::from(PROBABILITY_SCALE)
            / u128::from(denominator))
            .min(u128::from(PROBABILITY_SCALE))) as u32)
    }
}

fn p95_nearest_rank(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (95 * sorted.len() + 99) / 100;
    Some(sorted[rank.saturating_sub(1)])
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSensingEpisodeResult {
    pub scenario_sha256: [u8; 32],
    pub policy_sha256: [u8; 32],
    pub trace_sha256: [u8; 32],
    pub steps: u64,
    pub event_count: u64,
    pub detected_events: u64,
    pub missed_events: u64,
    pub false_alarms: u64,
    pub duplicate_alarms: u64,
    pub event_recall_ppm: Option<u32>,
    /// Detections divided by detections + false alarms + duplicate alarms.
    pub alarm_precision_ppm: Option<u32>,
    /// Exposure denominator is the complete number of simulated steps.
    pub false_alarms_per_1_000_steps: u64,
    pub detection_delays_steps: Vec<u64>,
    pub detection_delay_p95_steps: Option<u64>,
    pub samples_acquired: u64,
    pub low_power_samples: u64,
    pub high_fidelity_samples: u64,
    pub wake_count: u64,
    pub sensor_energy_nj: u64,
    pub wake_energy_nj: u64,
    pub modeled_energy_nj: u64,
    pub declared_compute_cycles: u64,
    pub declared_transmit_bytes: u64,
    pub declared_memory_write_bytes: u64,
    pub modeled_energy_per_detected_event_nj: Option<u64>,
    pub result_sha256: [u8; 32],
}

fn scenario_digest(scenario: &ActiveSensingScenario) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.active-sensing-scenario.v1");
    hash.u64(scenario.seed);
    hash.u64(scenario.steps);
    hash.u32(scenario.event_rate_ppm);
    hash.u32(scenario.min_event_duration_steps);
    hash.u32(scenario.max_event_duration_steps);
    match scenario.shift {
        ActiveSensingShift::Stable => hash.u8(0),
        ActiveSensingShift::AbruptSensorShift { start_step, extra_error_ppm } => {
            hash.u8(1);
            hash.u64(start_step);
            hash.u32(extra_error_ppm);
        }
        ActiveSensingShift::GradualSensorShift {
            start_step,
            end_step,
            max_extra_error_ppm,
        } => {
            hash.u8(2);
            hash.u64(start_step);
            hash.u64(end_step);
            hash.u32(max_extra_error_ppm);
        }
        ActiveSensingShift::CompoundShift {
            start_step,
            extra_sensor_error_ppm,
            event_rate_multiplier_ppm,
        } => {
            hash.u8(3);
            hash.u64(start_step);
            hash.u32(extra_sensor_error_ppm);
            hash.u32(event_rate_multiplier_ppm);
        }
    }
    hash.finish()
}

fn result_digest(result: &ActiveSensingEpisodeResult) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.active-sensing-result.v1");
    hash.bytes(&result.scenario_sha256);
    hash.bytes(&result.policy_sha256);
    hash.bytes(&result.trace_sha256);
    for value in [
        result.steps,
        result.event_count,
        result.detected_events,
        result.missed_events,
        result.false_alarms,
        result.duplicate_alarms,
        result.false_alarms_per_1_000_steps,
        result.samples_acquired,
        result.low_power_samples,
        result.high_fidelity_samples,
        result.wake_count,
        result.sensor_energy_nj,
        result.wake_energy_nj,
        result.modeled_energy_nj,
        result.declared_compute_cycles,
        result.declared_transmit_bytes,
        result.declared_memory_write_bytes,
    ] {
        hash.u64(value);
    }
    hash.u32(result.event_recall_ppm.unwrap_or(u32::MAX));
    hash.u32(result.alarm_precision_ppm.unwrap_or(u32::MAX));
    hash.u64(result.detection_delay_p95_steps.unwrap_or(u64::MAX));
    hash.usize(result.detection_delays_steps.len());
    for delay in &result.detection_delays_steps {
        hash.u64(*delay);
    }
    hash.u64(result.modeled_energy_per_detected_event_nj.unwrap_or(u64::MAX));
    hash.finish()
}

/// Recompute summary invariants and the result digest. This validates internal
/// consistency; it does not authenticate the runner or power instrumentation.
pub fn validate_active_sensing_result(
    result: &ActiveSensingEpisodeResult,
) -> Result<(), ActiveSensingSimulationError> {
    if result.scenario_sha256 == [0; 32] || result.policy_sha256 == [0; 32] {
        return Err(ActiveSensingSimulationError::new(
            "result.identity",
            "scenario and policy digests must be nonzero",
        ));
    }
    if result.steps == 0
        || result.detected_events + result.missed_events != result.event_count
        || result.detection_delays_steps.len() as u64 != result.detected_events
        || result.low_power_samples + result.high_fidelity_samples != result.samples_acquired
        || result.wake_count == 0
        || result.wake_count > result.steps
    {
        return Err(ActiveSensingSimulationError::new(
            "result.denominators",
            "event, delay, sensor-sample, or wake denominators are inconsistent",
        ));
    }
    if result.modeled_energy_nj
        != result.sensor_energy_nj.saturating_add(result.wake_energy_nj)
    {
        return Err(ActiveSensingSimulationError::new(
            "result.resources",
            "modeled energy accounting is inconsistent",
        ));
    }
    let expected_recall = ppm(result.detected_events, result.event_count);
    let alarm_count = result.detected_events
        .saturating_add(result.false_alarms)
        .saturating_add(result.duplicate_alarms);
    let expected_precision = ppm(result.detected_events, alarm_count);
    let expected_false_alarm_rate = (u128::from(result.false_alarms) * 1_000
        / u128::from(result.steps)) as u64;
    if result.event_recall_ppm != expected_recall
        || result.alarm_precision_ppm != expected_precision
        || result.false_alarms_per_1_000_steps != expected_false_alarm_rate
        || result.detection_delay_p95_steps != p95_nearest_rank(&result.detection_delays_steps)
        || result.modeled_energy_per_detected_event_nj
            != if result.detected_events == 0 {
                None
            } else {
                Some(result.modeled_energy_nj / result.detected_events)
            }
    {
        return Err(ActiveSensingSimulationError::new(
            "result.metrics",
            "derived metric does not reproduce from the retained denominators",
        ));
    }
    if result.result_sha256 != result_digest(result) {
        return Err(ActiveSensingSimulationError::new(
            "result.result_sha256",
            "canonical result digest mismatch",
        ));
    }
    Ok(())
}

/// Run one deterministic episode. The policy never receives the scenario,
/// seed, event schedule, current hidden label, or evaluator result.
pub fn run_active_sensing_episode<P: ActiveSensingPolicy>(
    scenario: ActiveSensingScenario,
    policy_sha256: [u8; 32],
    policy: &mut P,
) -> Result<ActiveSensingEpisodeResult, ActiveSensingSimulationError> {
    scenario.validate()?;
    if policy_sha256 == [0; 32] {
        return Err(ActiveSensingSimulationError::new(
            "policy_sha256",
            "candidate artifact/configuration digest must be nonzero",
        ));
    }

    let hidden = generate_hidden_episode(&scenario);
    let mut sensor_rng = SplitMix64::new(scenario.seed ^ 0x53454E534F525F31);
    let mut last_measurement: Option<PublicSensorMeasurement> = None;
    let mut detected = vec![false; hidden.events.len()];
    let mut detection_delays_steps = Vec::new();
    let mut false_alarms = 0_u64;
    let mut duplicate_alarms = 0_u64;
    let mut low_power_samples = 0_u64;
    let mut high_fidelity_samples = 0_u64;
    let mut wake_count = 0_u64;
    let mut sensor_energy_nj = 0_u64;
    let mut wake_energy_nj = 0_u64;
    let mut declared_compute_cycles = 0_u64;
    let mut declared_transmit_bytes = 0_u64;
    let mut declared_memory_write_bytes = 0_u64;
    let mut trace = CanonicalHasher::new(b"symthaea.temporal.active-sensing-policy-trace.v1");

    let mut step = 0_u64;
    while step < scenario.steps {
        let observation = PublicObservation {
            step,
            last_measurement,
            last_measurement_age_steps: last_measurement
                .map(|measurement| step.saturating_sub(measurement.sample_step)),
            samples_acquired: low_power_samples + high_fidelity_samples,
            low_power_samples,
            high_fidelity_samples,
        };
        let action = policy.decide(&observation);
        if action.next_wake_after_steps == 0
            || action.next_wake_after_steps > MAX_WAKE_INTERVAL_STEPS
        {
            return Err(ActiveSensingSimulationError::new(
                "policy_action.next_wake_after_steps",
                format!("wake interval must be in 1..={MAX_WAKE_INTERVAL_STEPS}"),
            ));
        }

        wake_count = wake_count.checked_add(1).ok_or_else(|| {
            ActiveSensingSimulationError::new("result.wake_count", "counter overflow")
        })?;
        wake_energy_nj = wake_energy_nj.checked_add(WAKE_ENERGY_NJ).ok_or_else(|| {
            ActiveSensingSimulationError::new("result.wake_energy_nj", "counter overflow")
        })?;
        declared_compute_cycles = declared_compute_cycles
            .checked_add(action.compute_cycles)
            .ok_or_else(|| ActiveSensingSimulationError::new("result.compute_cycles", "overflow"))?;
        declared_transmit_bytes = declared_transmit_bytes
            .checked_add(action.transmit_bytes)
            .ok_or_else(|| ActiveSensingSimulationError::new("result.transmit_bytes", "overflow"))?;
        declared_memory_write_bytes = declared_memory_write_bytes
            .checked_add(action.memory_write_bytes)
            .ok_or_else(|| ActiveSensingSimulationError::new("result.memory_write_bytes", "overflow"))?;

        let event_index = hidden.event_at_step[step as usize];
        if action.alarm {
            match event_index {
                Some(index) if !detected[index] => {
                    detected[index] = true;
                    detection_delays_steps.push(step - hidden.events[index].start_step);
                }
                Some(_) => duplicate_alarms = duplicate_alarms.saturating_add(1),
                None => false_alarms = false_alarms.saturating_add(1),
            }
        }

        trace.u64(step);
        trace.u8(u8::from(action.alarm));
        trace.u32(action.next_wake_after_steps);
        trace.u64(action.compute_cycles);
        trace.u64(action.transmit_bytes);
        trace.u64(action.memory_write_bytes);
        match action.sensor {
            Some(ActiveSensingSensor::LowPower) => {
                low_power_samples = low_power_samples.checked_add(1).ok_or_else(|| {
                    ActiveSensingSimulationError::new("result.low_power_samples", "overflow")
                })?;
                sensor_energy_nj = sensor_energy_nj
                    .checked_add(LOW_POWER_SAMPLE_ENERGY_NJ)
                    .ok_or_else(|| ActiveSensingSimulationError::new("result.sensor_energy_nj", "overflow"))?;
                let measurement = acquire_measurement(
                    &mut sensor_rng,
                    &scenario,
                    step,
                    event_index.is_some(),
                    ActiveSensingSensor::LowPower,
                );
                trace.u8(0);
                trace.u8(u8::from(measurement.positive));
                last_measurement = Some(measurement);
            }
            Some(ActiveSensingSensor::HighFidelity) => {
                high_fidelity_samples = high_fidelity_samples.checked_add(1).ok_or_else(|| {
                    ActiveSensingSimulationError::new("result.high_fidelity_samples", "overflow")
                })?;
                sensor_energy_nj = sensor_energy_nj
                    .checked_add(HIGH_FIDELITY_SAMPLE_ENERGY_NJ)
                    .ok_or_else(|| ActiveSensingSimulationError::new("result.sensor_energy_nj", "overflow"))?;
                let measurement = acquire_measurement(
                    &mut sensor_rng,
                    &scenario,
                    step,
                    event_index.is_some(),
                    ActiveSensingSensor::HighFidelity,
                );
                trace.u8(1);
                trace.u8(u8::from(measurement.positive));
                last_measurement = Some(measurement);
            }
            None => trace.u8(2),
        }
        step = step.saturating_add(u64::from(action.next_wake_after_steps));
    }

    let event_count = hidden.events.len() as u64;
    let detected_events = detected.iter().filter(|value| **value).count() as u64;
    let missed_events = event_count - detected_events;
    let samples_acquired = low_power_samples + high_fidelity_samples;
    let modeled_energy_nj = sensor_energy_nj.checked_add(wake_energy_nj).ok_or_else(|| {
        ActiveSensingSimulationError::new("result.modeled_energy_nj", "counter overflow")
    })?;
    let alarm_count = detected_events
        .saturating_add(false_alarms)
        .saturating_add(duplicate_alarms);
    let mut result = ActiveSensingEpisodeResult {
        scenario_sha256: scenario_digest(&scenario),
        policy_sha256,
        trace_sha256: trace.finish(),
        steps: scenario.steps,
        event_count,
        detected_events,
        missed_events,
        false_alarms,
        duplicate_alarms,
        event_recall_ppm: ppm(detected_events, event_count),
        alarm_precision_ppm: ppm(detected_events, alarm_count),
        false_alarms_per_1_000_steps: (u128::from(false_alarms) * 1_000
            / u128::from(scenario.steps)) as u64,
        detection_delay_p95_steps: p95_nearest_rank(&detection_delays_steps),
        detection_delays_steps,
        samples_acquired,
        low_power_samples,
        high_fidelity_samples,
        wake_count,
        sensor_energy_nj,
        wake_energy_nj,
        modeled_energy_nj,
        declared_compute_cycles,
        declared_transmit_bytes,
        declared_memory_write_bytes,
        modeled_energy_per_detected_event_nj: if detected_events == 0 {
            None
        } else {
            Some(modeled_energy_nj / detected_events)
        },
        result_sha256: [0; 32],
    };
    result.result_sha256 = result_digest(&result);
    validate_active_sensing_result(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(shift: ActiveSensingShift) -> ActiveSensingScenario {
        ActiveSensingScenario {
            seed: 0xA11CE,
            steps: 2_000,
            event_rate_ppm: 35_000,
            min_event_duration_steps: 3,
            max_event_duration_steps: 8,
            shift,
        }
    }

    fn policy_digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn same_seed_policy_and_scenario_produce_identical_receipts() {
        let scenario = scenario(ActiveSensingShift::Stable);
        let mut first_policy = FixedRateBaseline {
            sensor: ActiveSensingSensor::LowPower,
            interval_steps: 1,
        };
        let mut second_policy = first_policy;
        let first = run_active_sensing_episode(scenario, policy_digest(1), &mut first_policy).unwrap();
        let second = run_active_sensing_episode(scenario, policy_digest(1), &mut second_policy).unwrap();
        assert_eq!(first, second);
        validate_active_sensing_result(&first).unwrap();
    }

    #[test]
    fn high_fidelity_sampling_has_higher_modeled_sensor_energy() {
        let scenario = scenario(ActiveSensingShift::Stable);
        let mut low = FixedRateBaseline {
            sensor: ActiveSensingSensor::LowPower,
            interval_steps: 1,
        };
        let mut high = FixedRateBaseline {
            sensor: ActiveSensingSensor::HighFidelity,
            interval_steps: 1,
        };
        let low_result = run_active_sensing_episode(scenario, policy_digest(2), &mut low).unwrap();
        let high_result = run_active_sensing_episode(scenario, policy_digest(3), &mut high).unwrap();
        assert_eq!(low_result.samples_acquired, high_result.samples_acquired);
        assert!(high_result.sensor_energy_nj > low_result.sensor_energy_nj);
        assert_eq!(low_result.wake_energy_nj, high_result.wake_energy_nj);
    }

    #[test]
    fn sleep_without_sampling_reduces_sample_energy_but_not_wake_cost() {
        struct SleepPolicy;
        impl ActiveSensingPolicy for SleepPolicy {
            fn decide(&mut self, _: &PublicObservation) -> PolicyAction {
                PolicyAction::sleep(8)
            }
        }
        let mut policy = SleepPolicy;
        let result = run_active_sensing_episode(
            scenario(ActiveSensingShift::Stable),
            policy_digest(4),
            &mut policy,
        )
        .unwrap();
        assert_eq!(result.samples_acquired, 0);
        assert_eq!(result.sensor_energy_nj, 0);
        assert!(result.wake_energy_nj > 0);
        assert_eq!(result.wake_count, result.wake_energy_nj / WAKE_ENERGY_NJ);
    }

    #[test]
    fn adaptive_confirmation_baseline_runs_with_valid_denominators() {
        let mut policy = HysteresisAdaptiveBaseline {
            quiet_interval_steps: 4,
            active_interval_steps: 1,
        };
        let result = run_active_sensing_episode(
            scenario(ActiveSensingShift::AbruptSensorShift {
                start_step: 1_000,
                extra_error_ppm: 120_000,
            }),
            policy_digest(5),
            &mut policy,
        )
        .unwrap();
        assert_eq!(result.event_count, result.detected_events + result.missed_events);
        assert_eq!(result.samples_acquired, result.low_power_samples + result.high_fidelity_samples);
        validate_active_sensing_result(&result).unwrap();
    }

    #[test]
    fn invalid_wake_interval_fails_closed() {
        struct InvalidPolicy;
        impl ActiveSensingPolicy for InvalidPolicy {
            fn decide(&mut self, _: &PublicObservation) -> PolicyAction {
                PolicyAction::sleep(0)
            }
        }
        let mut policy = InvalidPolicy;
        let error = run_active_sensing_episode(
            scenario(ActiveSensingShift::Stable),
            policy_digest(6),
            &mut policy,
        )
        .unwrap_err();
        assert_eq!(error.field, "policy_action.next_wake_after_steps");
    }

    #[test]
    fn result_validator_detects_metric_tampering() {
        let mut policy = FixedRateBaseline {
            sensor: ActiveSensingSensor::LowPower,
            interval_steps: 2,
        };
        let mut result = run_active_sensing_episode(
            scenario(ActiveSensingShift::Stable),
            policy_digest(7),
            &mut policy,
        )
        .unwrap();
        result.false_alarms = result.false_alarms.saturating_add(1);
        assert!(validate_active_sensing_result(&result).is_err());
    }

    #[test]
    fn compound_shift_changes_the_scenario_commitment() {
        let stable = scenario(ActiveSensingShift::Stable);
        let shifted = scenario(ActiveSensingShift::CompoundShift {
            start_step: 1_000,
            extra_sensor_error_ppm: 100_000,
            event_rate_multiplier_ppm: 2 * PROBABILITY_SCALE,
        });
        assert_ne!(scenario_digest(&stable), scenario_digest(&shifted));
    }

    #[test]
    fn scenario_rejects_invalid_shift_bounds() {
        let mut invalid = scenario(ActiveSensingShift::GradualSensorShift {
            start_step: 1_000,
            end_step: 1_000,
            max_extra_error_ppm: 10_000,
        });
        assert!(invalid.validate().is_err());
        invalid.shift = ActiveSensingShift::AbruptSensorShift {
            start_step: invalid.steps,
            extra_error_ppm: 1,
        };
        assert!(invalid.validate().is_err());
    }
}
