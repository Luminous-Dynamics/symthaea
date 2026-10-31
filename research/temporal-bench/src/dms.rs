//! Deterministic Rust delayed-match-to-sample (DMS) reference workload.
//!
//! This reproduces the task semantics of the pinned BrainTrace public DMS
//! script: eight motion directions, population tuning, sample/delay/test
//! periods, background firing, and Bernoulli spike conversion. It does not
//! reproduce the BrainTrace network/training code. The public task and oracle
//! are returned together for prototype testing; this is not process isolation.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::fmt;

const MAX_DMS_STEPS: usize = 500_000;
const MOTION_CHOICES: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DmsConfig {
    pub seed: u64,
    pub dt_ms: u32,
    pub fixation_ms: u32,
    pub sample_ms: u32,
    pub delay_ms: u32,
    pub test_ms: u32,
    pub input_channels: usize,
    pub firing_rate_hz: f64,
    pub background_rate_hz: f64,
    pub kappa: f64,
    /// Rotation in 45-degree direction bins. Zero means exact direction match.
    pub rotation_steps: u8,
}

impl DmsConfig {
    pub fn braintrace_reference(seed: u64) -> Self {
        Self {
            seed,
            dt_ms: 1,
            fixation_ms: 10,
            sample_ms: 500,
            delay_ms: 1_000,
            test_ms: 500,
            input_channels: 100,
            firing_rate_hz: 100.0,
            background_rate_hz: 1.0,
            kappa: 3.0,
            rotation_steps: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmsObservation {
    pub step: u32,
    /// Sparse indices for channels that emitted a spike at this time step.
    pub active_channels: Vec<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmsTask {
    pub stream_id: u64,
    pub dt_ms: u32,
    pub observations: Vec<DmsObservation>,
    /// Public sample presentation window; the sample direction remains hidden.
    pub sample_start_step: u32,
    pub sample_end_step: u32,
    /// Public delayed decision window; the test direction and class remain hidden.
    pub test_start_step: u32,
    pub test_end_step: u32,
    pub delay_steps: u32,
    pub input_channels: usize,
    pub rotation_steps: u8,
    pub firing_rate_hz_bits: u64,
    pub background_rate_hz_bits: u64,
    pub kappa_bits: u64,
    /// SHA-256 over the versioned canonical public DMS stream.
    pub content_sha256: [u8; 32],
}

/// Domain-separated canonical SHA-256 of the DMS public stream.
pub fn dms_content_digest(task: &DmsTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.dms.public.v1");
    hash.u64(task.stream_id);
    hash.u32(task.dt_ms);
    hash.u32(task.sample_start_step);
    hash.u32(task.sample_end_step);
    hash.u32(task.test_start_step);
    hash.u32(task.test_end_step);
    hash.u32(task.delay_steps);
    hash.usize(task.input_channels);
    hash.u8(task.rotation_steps);
    hash.u64(task.firing_rate_hz_bits);
    hash.u64(task.background_rate_hz_bits);
    hash.u64(task.kappa_bits);
    hash.usize(task.observations.len());
    for observation in &task.observations {
        hash.u32(observation.step);
        hash.usize(observation.active_channels.len());
        for channel in &observation.active_channels {
            hash.u16(*channel);
        }
    }
    hash.finish()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmsOracle {
    pub stream_id: u64,
    pub seed: u64,
    pub is_match: bool,
    pub sample_direction: u8,
    pub test_direction: u8,
    pub expected_label: u8,
    /// SHA-256 over the versioned canonical oracle record.
    pub content_sha256: [u8; 32],
}

/// Domain-separated canonical SHA-256 of the DMS evaluator oracle.
pub fn dms_oracle_digest(oracle: &DmsOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.dms.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u8(u8::from(oracle.is_match));
    hash.u8(oracle.sample_direction);
    hash.u8(oracle.test_direction);
    hash.u8(oracle.expected_label);
    hash.u64(oracle.seed);
    hash.finish()
}

#[derive(Clone, Debug, PartialEq)]
pub enum DmsGenerationError {
    ZeroTimeStep,
    ZeroSampleDuration,
    ZeroDelayDuration,
    ZeroTestDuration,
    DurationNotDivisibleByStep { duration_ms: u32, dt_ms: u32 },
    TooManySteps { requested: usize, maximum: usize },
    InvalidInputChannels { requested: usize, minimum: usize, maximum: usize },
    InvalidRotation { requested: u8, maximum: u8 },
    InvalidRate { name: &'static str },
    InvalidKappa,
    SpikeProbabilityOutOfRange { maximum_probability: f64 },
}

impl fmt::Display for DmsGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTimeStep => write!(f, "dt_ms must be greater than zero"),
            Self::ZeroSampleDuration => write!(f, "sample_ms must be greater than zero"),
            Self::ZeroDelayDuration => write!(f, "delay_ms must be greater than zero"),
            Self::ZeroTestDuration => write!(f, "test_ms must be greater than zero"),
            Self::DurationNotDivisibleByStep { duration_ms, dt_ms } => write!(
                f,
                "duration {duration_ms} ms is not divisible by dt {dt_ms} ms"
            ),
            Self::TooManySteps { requested, maximum } => {
                write!(f, "DMS stream requires {requested} steps; maximum is {maximum}")
            }
            Self::InvalidInputChannels { requested, minimum, maximum } => write!(
                f,
                "input_channels {requested} is outside the supported range {minimum}..={maximum}"
            ),
            Self::InvalidRotation { requested, maximum } => {
                write!(f, "rotation_steps {requested} exceeds {maximum}")
            }
            Self::InvalidRate { name } => write!(f, "{name} must be finite and non-negative"),
            Self::InvalidKappa => write!(f, "kappa must be finite and positive"),
            Self::SpikeProbabilityOutOfRange { maximum_probability } => write!(
                f,
                "maximum spike probability {maximum_probability} exceeds 1"
            ),
        }
    }
}

impl std::error::Error for DmsGenerationError {}

/// Generate a DMS spike stream from the reference task's population-tuning rule.
///
/// Spike observations are public; the match label and sample/test directions
/// are kept in the oracle value. A confirmatory harness must write these
/// artifacts to separately permissioned destinations and give the candidate
/// only the public stream. The reference tuning function is
/// exp(kappa * cos(delta)) / exp(kappa).
pub fn generate_dms_task(
    config: DmsConfig,
) -> Result<(DmsTask, DmsOracle), DmsGenerationError> {
    validate_config(config)?;

    let dt_ms = f64::from(config.dt_ms);
    let sample_probability = config.firing_rate_hz * dt_ms / 1_000.0;
    let background_probability = config.background_rate_hz * dt_ms / 1_000.0;
    let maximum_probability = sample_probability + background_probability;
    if maximum_probability > 1.0 {
        return Err(DmsGenerationError::SpikeProbabilityOutOfRange {
            maximum_probability,
        });
    }

    let stream_id = dms_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x444D_535F_5245_4631);
    let is_match = rng.next_u64() & 1 == 1;
    let sample_direction = (rng.next_u64() % u64::from(MOTION_CHOICES)) as u8;
    let rotated_sample = (sample_direction + config.rotation_steps) % MOTION_CHOICES;
    let test_direction = if is_match {
        rotated_sample
    } else {
        let non_match_choice = (rng.next_u64() % u64::from(MOTION_CHOICES - 1)) as u8;
        if non_match_choice >= rotated_sample {
            non_match_choice + 1
        } else {
            non_match_choice
        }
    };

    let fixation_steps = config.fixation_ms / config.dt_ms;
    let sample_steps = config.sample_ms / config.dt_ms;
    let delay_steps = config.delay_ms / config.dt_ms;
    let test_steps = config.test_ms / config.dt_ms;
    let sample_start = fixation_steps;
    let sample_end = sample_start.saturating_add(sample_steps);
    let test_start = sample_end.saturating_add(delay_steps);
    let test_end = test_start.saturating_add(test_steps);
    let total_steps = usize::try_from(test_end).unwrap_or(usize::MAX);

    let mut sample_tuning = Vec::with_capacity(config.input_channels);
    let mut test_tuning = Vec::with_capacity(config.input_channels);
    for channel in 0..config.input_channels {
        sample_tuning.push(motion_tuning(
            channel,
            config.input_channels,
            sample_direction,
            config.kappa,
        ));
        test_tuning.push(motion_tuning(
            channel,
            config.input_channels,
            test_direction,
            config.kappa,
        ));
    }

    let mut observations = Vec::with_capacity(total_steps);
    for step in 0..test_end {
        let mut active_channels = Vec::new();
        let in_sample = step >= sample_start && step < sample_end;
        let in_test = step >= test_start && step < test_end;
        for channel in 0..config.input_channels {
            let stimulus_probability = if in_sample {
                sample_probability * sample_tuning[channel]
            } else if in_test {
                sample_probability * test_tuning[channel]
            } else {
                0.0
            };
            let probability = (background_probability + stimulus_probability).min(1.0);
            let draw = rng.next_u64() >> 11;
            let threshold = (probability * 9_007_199_254_740_992.0) as u64;
            if draw < threshold {
                active_channels.push(u16::try_from(channel).unwrap_or(u16::MAX));
            }
        }
        observations.push(DmsObservation {
            step,
            active_channels,
        });
    }

    let mut public_task = DmsTask {
        stream_id,
        dt_ms: config.dt_ms,
        observations,
        sample_start_step: sample_start,
        sample_end_step: sample_end,
        test_start_step: test_start,
        test_end_step: test_end,
        delay_steps,
        input_channels: config.input_channels,
        rotation_steps: config.rotation_steps,
        content_sha256: [0; 32],
    };
    public_task.content_sha256 = dms_content_digest(&public_task);

    Ok((
        public_task,
        {
            let mut oracle = DmsOracle {
                stream_id,
                is_match,
                sample_direction,
                test_direction,
                expected_label: u8::from(is_match),
                content_sha256: [0; 32],
            };
            oracle.content_sha256 = dms_oracle_digest(&oracle);
            oracle
        },
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DmsVerificationError {
    StreamIdMismatch,
    ContentDigestMismatch,
    OracleDigestMismatch,
    ReproducibilityMismatch,
    InvalidDecisionWindow,
    InvalidOracle,
    InvalidObservation { expected_step: u32 },
    ChannelOutOfRange { step: u32, channel: u16 },
    ChannelsNotStrictlyOrdered { step: u32 },
}

impl fmt::Display for DmsVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "DMS public task and oracle stream IDs differ"),
            Self::ContentDigestMismatch => write!(f, "DMS public task SHA-256 digest does not match its contents"),
            Self::OracleDigestMismatch => write!(f, "DMS oracle SHA-256 digest does not match its contents"),
            Self::ReproducibilityMismatch => write!(f, "DMS public stream does not match the seeded generator"),
            Self::InvalidDecisionWindow => write!(f, "DMS decision window or observation length is invalid"),
            Self::InvalidOracle => write!(f, "DMS oracle direction or match relation is invalid"),
            Self::InvalidObservation { expected_step } => {
                write!(f, "DMS observation is missing or reordered at step {expected_step}")
            }
            Self::ChannelOutOfRange { step, channel } => {
                write!(f, "DMS observation at step {step} contains out-of-range channel {channel}")
            }
            Self::ChannelsNotStrictlyOrdered { step } => {
                write!(f, "DMS active channels are duplicated or unordered at step {step}")
            }
        }
    }
}

impl std::error::Error for DmsVerificationError {}

/// Evaluator-side structural and oracle check for a generated DMS stream.
pub fn verify_dms_trace(
    task: &DmsTask,
    oracle: &DmsOracle,
) -> Result<(), DmsVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(DmsVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != dms_content_digest(task) {
        return Err(DmsVerificationError::ContentDigestMismatch);
    }
    if oracle.content_sha256 != dms_oracle_digest(oracle) {
        return Err(DmsVerificationError::OracleDigestMismatch);
    }
    if task.sample_start_step >= task.sample_end_step
        || task.sample_end_step.saturating_add(task.delay_steps) != task.test_start_step
        || task.test_start_step >= task.test_end_step
        || usize::try_from(task.test_end_step).unwrap_or(usize::MAX) != task.observations.len()
        || task.input_channels == 0
        || task.input_channels > 4_096
        || task.dt_ms == 0
    {
        return Err(DmsVerificationError::InvalidDecisionWindow);
    }
    if oracle.sample_direction >= MOTION_CHOICES
        || oracle.test_direction >= MOTION_CHOICES
        || oracle.expected_label != u8::from(oracle.is_match)
        || task.rotation_steps >= MOTION_CHOICES
    {
        return Err(DmsVerificationError::InvalidOracle);
    }
    let rotated_sample = (oracle.sample_direction + task.rotation_steps) % MOTION_CHOICES;
    if (oracle.is_match && oracle.test_direction != rotated_sample)
        || (!oracle.is_match && oracle.test_direction == rotated_sample)
    {
        return Err(DmsVerificationError::InvalidOracle);
    }

    for (index, observation) in task.observations.iter().enumerate() {
        let expected_step = u32::try_from(index).unwrap_or(u32::MAX);
        if observation.step != expected_step {
            return Err(DmsVerificationError::InvalidObservation { expected_step });
        }
        let mut previous = None;
        for channel in &observation.active_channels {
            if usize::from(*channel) >= task.input_channels {
                return Err(DmsVerificationError::ChannelOutOfRange {
                    step: observation.step,
                    channel: *channel,
                });
            }
            if previous.is_some_and(|prior| prior >= *channel) {
                return Err(DmsVerificationError::ChannelsNotStrictlyOrdered {
                    step: observation.step,
                });
            }
            previous = Some(*channel);
        }
    }

    let config = DmsConfig {
        seed: oracle.seed,
        dt_ms: task.dt_ms,
        fixation_ms: task.sample_start_step.saturating_mul(task.dt_ms),
        sample_ms: task.sample_end_step.saturating_sub(task.sample_start_step)
            .saturating_mul(task.dt_ms),
        delay_ms: task.delay_steps.saturating_mul(task.dt_ms),
        test_ms: task.test_end_step.saturating_sub(task.test_start_step)
            .saturating_mul(task.dt_ms),
        input_channels: task.input_channels,
        firing_rate_hz: f64::from_bits(task.firing_rate_hz_bits),
        background_rate_hz: f64::from_bits(task.background_rate_hz_bits),
        kappa: f64::from_bits(task.kappa_bits),
        rotation_steps: task.rotation_steps,
    };
    let (expected_task, expected_oracle) =
        generate_dms_task(config).map_err(|_| DmsVerificationError::ReproducibilityMismatch)?;
    if expected_task != *task || expected_oracle != *oracle {
        return Err(DmsVerificationError::ReproducibilityMismatch);
    }
    Ok(())
}

fn validate_config(config: DmsConfig) -> Result<(), DmsGenerationError> {
    if config.dt_ms == 0 {
        return Err(DmsGenerationError::ZeroTimeStep);
    }
    if config.sample_ms == 0 {
        return Err(DmsGenerationError::ZeroSampleDuration);
    }
    if config.delay_ms == 0 {
        return Err(DmsGenerationError::ZeroDelayDuration);
    }
    if config.test_ms == 0 {
        return Err(DmsGenerationError::ZeroTestDuration);
    }
    for duration in [config.fixation_ms, config.sample_ms, config.delay_ms, config.test_ms] {
        if duration % config.dt_ms != 0 {
            return Err(DmsGenerationError::DurationNotDivisibleByStep {
                duration_ms: duration,
                dt_ms: config.dt_ms,
            });
        }
    }
    if !(8..=4_096).contains(&config.input_channels) {
        return Err(DmsGenerationError::InvalidInputChannels {
            requested: config.input_channels,
            minimum: 8,
            maximum: 4_096,
        });
    }
    if config.rotation_steps >= MOTION_CHOICES {
        return Err(DmsGenerationError::InvalidRotation {
            requested: config.rotation_steps,
            maximum: MOTION_CHOICES - 1,
        });
    }
    if !config.firing_rate_hz.is_finite() || config.firing_rate_hz < 0.0 {
        return Err(DmsGenerationError::InvalidRate {
            name: "firing_rate_hz",
        });
    }
    if !config.background_rate_hz.is_finite() || config.background_rate_hz < 0.0 {
        return Err(DmsGenerationError::InvalidRate {
            name: "background_rate_hz",
        });
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(DmsGenerationError::InvalidKappa);
    }
    let total_steps = u64::from(config.fixation_ms / config.dt_ms)
        .saturating_add(u64::from(config.sample_ms / config.dt_ms))
        .saturating_add(u64::from(config.delay_ms / config.dt_ms))
        .saturating_add(u64::from(config.test_ms / config.dt_ms));
    if total_steps > MAX_DMS_STEPS as u64 {
        return Err(DmsGenerationError::TooManySteps {
            requested: usize::try_from(total_steps).unwrap_or(usize::MAX),
            maximum: MAX_DMS_STEPS,
        });
    }
    Ok(())
}

fn dms_stream_id(config: DmsConfig) -> u64 {
    let mut value = config.seed ^ u64::from(config.dt_ms).rotate_left(5);
    value ^= u64::from(config.fixation_ms).rotate_left(11);
    value ^= u64::from(config.sample_ms).rotate_left(17);
    value ^= u64::from(config.delay_ms).rotate_left(23);
    value ^= u64::from(config.test_ms).rotate_left(29);
    value ^= (config.input_channels as u64).rotate_left(37);
    value ^= config.firing_rate_hz.to_bits().rotate_left(7);
    value ^= config.background_rate_hz.to_bits().rotate_left(19);
    value ^= config.kappa.to_bits().rotate_left(31);
    value ^= u64::from(config.rotation_steps).rotate_left(47);
    mix64(value)
}

fn motion_tuning(
    channel: usize,
    input_channels: usize,
    direction: u8,
    kappa: f64,
) -> f64 {
    let preferred_angle =
        std::f64::consts::TAU * (channel as f64 / input_channels as f64);
    let stimulus_angle =
        std::f64::consts::TAU * (f64::from(direction) / f64::from(MOTION_CHOICES));
    (kappa * ((stimulus_angle - preferred_angle).cos() - 1.0)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_config_matches_the_pinned_braintrace_dms_timing() {
        let config = DmsConfig::braintrace_reference(42);
        assert_eq!(config.dt_ms, 1);
        assert_eq!(config.fixation_ms, 10);
        assert_eq!(config.sample_ms, 500);
        assert_eq!(config.delay_ms, 1_000);
        assert_eq!(config.test_ms, 500);
        let (task, _) = generate_dms_task(config).unwrap();
        assert_eq!(task.observations.len(), 2_010);
        assert_eq!(task.sample_start_step, 10);
        assert_eq!(task.sample_end_step, 510);
        assert_eq!(task.delay_steps, 1_000);
        assert_eq!(task.test_start_step, 1_510);
        assert_eq!(task.test_end_step, 2_010);
    }

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let config = DmsConfig::braintrace_reference(7);
        assert_eq!(generate_dms_task(config), generate_dms_task(config));
    }

    #[test]
    fn match_trials_follow_the_declared_rotation() {
        for seed in 0..100 {
            let config = DmsConfig::braintrace_reference(seed);
            let (_, oracle) = generate_dms_task(config).unwrap();
            if oracle.is_match {
                assert_eq!(
                    oracle.test_direction,
                    (oracle.sample_direction + config.rotation_steps) % MOTION_CHOICES
                );
            } else {
                assert_ne!(
                    oracle.test_direction,
                    (oracle.sample_direction + config.rotation_steps) % MOTION_CHOICES
                );
            }
            assert_eq!(oracle.expected_label, u8::from(oracle.is_match));
        }
    }

    #[test]
    fn spike_channels_are_in_range_and_timesteps_are_ordered() {
        let config = DmsConfig::braintrace_reference(19);
        let (task, _) = generate_dms_task(config).unwrap();
        for (index, observation) in task.observations.iter().enumerate() {
            assert_eq!(observation.step as usize, index);
            assert!(observation
                .active_channels
                .iter()
                .all(|channel| usize::from(*channel) < config.input_channels));
        }
    }

    #[test]
    fn generated_dms_trace_passes_oracle_verifier() {
        let config = DmsConfig::braintrace_reference(93);
        let (task, oracle) = generate_dms_task(config).unwrap();
        assert_eq!(verify_dms_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn generated_oracle_has_valid_sha256() {
        let config = DmsConfig::braintrace_reference(93);
        let (_, oracle) = generate_dms_task(config).unwrap();
        assert_eq!(oracle.content_sha256, dms_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_oracle_mutation_after_generation() {
        let config = DmsConfig::braintrace_reference(93);
        let (task, mut oracle) = generate_dms_task(config).unwrap();
        oracle.expected_label ^= 1;
        assert_eq!(
            verify_dms_trace(&task, &oracle),
            Err(DmsVerificationError::OracleDigestMismatch)
        );
    }

    #[test]
    fn generated_public_stream_has_valid_sha256() {
        let config = DmsConfig::braintrace_reference(93);
        let (task, _) = generate_dms_task(config).unwrap();
        assert_eq!(task.content_sha256, dms_content_digest(&task));
    }

    #[test]
    fn verifier_rejects_recommitted_seed_inconsistent_sample() {
        let config = DmsConfig::braintrace_reference(93);
        let (mut task, oracle) = generate_dms_task(config).unwrap();
        let first = &mut task.observations[0].active_channels;
        if first.contains(&0) {
            first.retain(|channel| *channel != 0);
        } else {
            first.push(0);
            first.sort_unstable();
        }
        task.content_sha256 = dms_content_digest(&task);
        assert_eq!(
            verify_dms_trace(&task, &oracle),
            Err(DmsVerificationError::ReproducibilityMismatch)
        );
    }

    #[test]
    fn oracle_digest_binds_the_generator_seed() {
        let config = DmsConfig::braintrace_reference(94);
        let (_, mut oracle) = generate_dms_task(config).unwrap();
        oracle.seed ^= 1;
        assert_ne!(oracle.content_sha256, dms_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_post_generation_stream_mutation() {
        let config = DmsConfig::braintrace_reference(93);
        let (mut task, oracle) = generate_dms_task(config).unwrap();
        if let Some(observation) = task.observations.iter_mut().find(|row| !row.active_channels.is_empty()) {
            observation.active_channels.pop();
        }
        assert_eq!(
            verify_dms_trace(&task, &oracle),
            Err(DmsVerificationError::ContentDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_sample_delay_window_mismatch() {
        let config = DmsConfig::braintrace_reference(93);
        let (mut task, oracle) = generate_dms_task(config).unwrap();
        task.sample_end_step = task.sample_end_step.saturating_add(1);
        task.content_sha256 = dms_content_digest(&task);
        assert_eq!(
            verify_dms_trace(&task, &oracle),
            Err(DmsVerificationError::InvalidDecisionWindow)
        );
    }

    #[test]
    fn verifier_rejects_wrong_direction_relation() {
        let config = DmsConfig::braintrace_reference(93);
        let (task, mut oracle) = generate_dms_task(config).unwrap();
        let rotated_sample =
            (oracle.sample_direction + config.rotation_steps) % MOTION_CHOICES;
        oracle.test_direction = if oracle.is_match {
            (rotated_sample + 1) % MOTION_CHOICES
        } else {
            rotated_sample
        };
        oracle.content_sha256 = dms_oracle_digest(&oracle);
        assert_eq!(
            verify_dms_trace(&task, &oracle),
            Err(DmsVerificationError::InvalidOracle)
        );
    }

    #[test]
    fn verifier_rejects_duplicate_channel_indices() {
        let config = DmsConfig::braintrace_reference(93);
        let (mut task, oracle) = generate_dms_task(config).unwrap();
        let observation = task
            .observations
            .iter_mut()
            .find(|observation| !observation.active_channels.is_empty())
            .unwrap();
        let first_channel = observation.active_channels[0];
        observation.active_channels.push(first_channel);
        observation.active_channels.sort_unstable();
        task.content_sha256 = dms_content_digest(&task);
        assert!(matches!(
            verify_dms_trace(&task, &oracle),
            Err(DmsVerificationError::ChannelsNotStrictlyOrdered { .. })
        ));
    }

    #[test]
    fn verifier_rejects_stream_identity_mismatch() {
        let config = DmsConfig::braintrace_reference(93);
        let (task, mut oracle) = generate_dms_task(config).unwrap();
        oracle.stream_id ^= 1;
        assert_eq!(
            verify_dms_trace(&task, &oracle),
            Err(DmsVerificationError::StreamIdMismatch)
        );
    }

    #[test]
    fn invalid_time_step_is_rejected() {
        let mut config = DmsConfig::braintrace_reference(1);
        config.dt_ms = 0;
        assert!(matches!(
            generate_dms_task(config),
            Err(DmsGenerationError::ZeroTimeStep)
        ));
    }

    #[test]
    fn non_divisible_interval_is_rejected() {
        let mut config = DmsConfig::braintrace_reference(1);
        config.dt_ms = 7;
        assert!(matches!(
            generate_dms_task(config),
            Err(DmsGenerationError::DurationNotDivisibleByStep { .. })
        ));
    }

    #[test]
    fn excess_spike_probability_is_rejected() {
        let mut config = DmsConfig::braintrace_reference(1);
        config.dt_ms = 100;
        config.fixation_ms = 100;
        config.firing_rate_hz = 1_000.0;
        assert!(matches!(
            generate_dms_task(config),
            Err(DmsGenerationError::SpikeProbabilityOutOfRange { .. })
        ));
    }
}
