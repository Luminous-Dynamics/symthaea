//! Cross-cutting physical-time / observation-rate / internal-step sweep.
//!
//! One canonical continuous-time signal can be sampled at several observation
//! rates. For any one public observation stream, the same input bytes are reused
//! across internal-substep arms K={1,2,4,8,16}. K changes internal work only;
//! it must not add sensor observations or extend the physical horizon.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::fmt;

const MAX_SLOTS: usize = 100_000;
const MAX_DURATION_US: u64 = 86_400_000_000;
const MAX_NOISE_AMPLITUDE: u16 = 64;
const INTERNAL_SUBSTEP_ARMS: [u32; 5] = [1, 2, 4, 8, 16];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimeResolutionConfig {
    pub seed: u64,
    pub duration_us: u64,
    pub observation_interval_us: u64,
    pub period_before_us: u64,
    pub period_after_us: u64,
    pub transition_at_us: u64,
    pub phase_jump_permille: u16,
    pub noise_amplitude: u16,
}

impl TimeResolutionConfig {
    pub fn reference(seed: u64, observation_interval_us: u64) -> Self {
        Self {
            seed,
            duration_us: 4_000_000,
            observation_interval_us,
            period_before_us: 64_000,
            period_after_us: 96_000,
            transition_at_us: 2_000_000,
            phase_jump_permille: 125,
            noise_amplitude: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InternalStepArm {
    pub internal_substeps_per_observation: u32,
    pub internal_dt_us: u64,
    pub total_internal_updates: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeResolutionObservation {
    pub step: u32,
    pub time_us: u64,
    pub value: i16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeResolutionTask {
    pub stream_id: u64,
    pub duration_us: u64,
    pub observation_interval_us: u64,
    pub observations: Vec<TimeResolutionObservation>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeResolutionOraclePoint {
    pub step: u32,
    pub time_us: u64,
    pub phase_permille: u16,
    pub latent_value: i16,
    pub observed_value: i16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeResolutionOracle {
    pub stream_id: u64,
    pub initial_phase_permille: u16,
    pub period_before_us: u64,
    pub period_after_us: u64,
    pub transition_at_us: u64,
    pub phase_jump_permille: u16,
    pub points: Vec<TimeResolutionOraclePoint>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimeResolutionGenerationError {
    ZeroDuration,
    ZeroObservationInterval,
    DurationNotDivisibleByObservationInterval {
        duration_us: u64,
        observation_interval_us: u64,
    },
    TooFewObservations { requested: usize, minimum: usize },
    TooManyObservations { requested: usize, maximum: usize },
    DurationExceedsLimit,
    InvalidPeriod { name: &'static str, period_us: u64 },
    InvalidTransition { transition_at_us: u64, duration_us: u64 },
    InvalidPhaseJump { phase_jump_permille: u16, maximum: u16 },
    ExcessiveNoise { noise_amplitude: u16, maximum: u16 },
}

impl fmt::Display for TimeResolutionGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDuration => write!(f, "duration_us must be greater than zero"),
            Self::ZeroObservationInterval => {
                write!(f, "observation_interval_us must be greater than zero")
            }
            Self::DurationNotDivisibleByObservationInterval {
                duration_us,
                observation_interval_us,
            } => write!(
                f,
                "duration {duration_us} us is not divisible by observation interval {observation_interval_us} us"
            ),
            Self::TooFewObservations { requested, minimum } => {
                write!(f, "observation slots {requested} is below minimum {minimum}")
            }
            Self::TooManyObservations { requested, maximum } => {
                write!(f, "observation slots {requested} exceeds maximum {maximum}")
            }
            Self::DurationExceedsLimit => write!(f, "physical duration exceeds configured maximum"),
            Self::InvalidPeriod { name, period_us } => {
                write!(f, "{name} is invalid: {period_us} us")
            }
            Self::InvalidTransition { transition_at_us, duration_us } => write!(
                f,
                "transition time {transition_at_us} us must lie strictly inside duration {duration_us} us"
            ),
            Self::InvalidPhaseJump { phase_jump_permille, maximum } => write!(
                f,
                "phase jump {phase_jump_permille} permille exceeds maximum {maximum}"
            ),
            Self::ExcessiveNoise { noise_amplitude, maximum } => write!(
                f,
                "noise amplitude {noise_amplitude} exceeds maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for TimeResolutionGenerationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimeResolutionVerificationError {
    StreamIdMismatch,
    PublicDigestMismatch,
    OracleDigestMismatch,
    LengthMismatch,
    InvalidProfile { substeps: u32 },
    InvalidSchedule { step: usize },
    InvalidOracleParameters,
    OraclePhaseMismatch { step: usize },
    PublicOracleValueMismatch { step: usize },
}

impl fmt::Display for TimeResolutionVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "time-resolution task and oracle IDs differ"),
            Self::PublicDigestMismatch => write!(f, "time-resolution public stream SHA-256 mismatch"),
            Self::OracleDigestMismatch => write!(f, "time-resolution oracle SHA-256 mismatch"),
            Self::LengthMismatch => write!(f, "time-resolution public/oracle point counts differ"),
            Self::InvalidProfile { substeps } => {
                write!(f, "internal substep profile {substeps} is invalid for this observation interval")
            }
            Self::InvalidSchedule { step } => {
                write!(f, "time-resolution observation schedule is invalid at step {step}")
            }
            Self::InvalidOracleParameters => write!(f, "time-resolution oracle parameters are invalid"),
            Self::OraclePhaseMismatch { step } => {
                write!(f, "oracle phase at step {step} disagrees with its physical-time parameters")
            }
            Self::PublicOracleValueMismatch { step } => {
                write!(f, "public observation at step {step} disagrees with the oracle sample")
            }
        }
    }
}

impl std::error::Error for TimeResolutionVerificationError {}

/// Generate the public stream and its hidden continuous-time oracle.
///
/// The physical signal is defined over absolute microsecond time, so streams
/// sampled at different observation intervals share exactly matching values
/// at every timestamp common to both grids. Noise is also keyed by absolute
/// time, not by the sequential PRNG draw count, preserving cross-rate pairing.
pub fn generate_time_resolution_task(
    config: TimeResolutionConfig,
) -> Result<(TimeResolutionTask, TimeResolutionOracle), TimeResolutionGenerationError> {
    validate_config(config)?;
    let slots = usize::try_from(config.duration_us / config.observation_interval_us)
        .unwrap_or(usize::MAX);
    let stream_id = time_resolution_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x5453_505F_54494D45);
    let initial_phase_permille = u16::try_from(rng.next_u64() % 1_000).unwrap_or(0);

    let mut observations = Vec::with_capacity(slots);
    let mut points = Vec::with_capacity(slots);
    for step in 0..slots {
        let step_u64 = u64::try_from(step).unwrap_or(u64::MAX);
        let time_us = step_u64.saturating_mul(config.observation_interval_us);
        let phase = phase_at(
            time_us,
            initial_phase_permille,
            config.period_before_us,
            config.period_after_us,
            config.transition_at_us,
            config.phase_jump_permille,
        );
        let latent_value = triangle_value(phase);
        let noise = deterministic_noise(config.seed, time_us, config.noise_amplitude);
        let observed_value = latent_value.saturating_add(noise);
        let step_u32 = u32::try_from(step).unwrap_or(u32::MAX);

        observations.push(TimeResolutionObservation {
            step: step_u32,
            time_us,
            value: observed_value,
        });
        points.push(TimeResolutionOraclePoint {
            step: step_u32,
            time_us,
            phase_permille: phase,
            latent_value,
            observed_value,
        });
    }

    let mut task = TimeResolutionTask {
        stream_id,
        duration_us: config.duration_us,
        observation_interval_us: config.observation_interval_us,
        observations,
        content_sha256: [0; 32],
    };
    task.content_sha256 = time_resolution_task_digest(&task);

    let mut oracle = TimeResolutionOracle {
        stream_id,
        initial_phase_permille,
        period_before_us: config.period_before_us,
        period_after_us: config.period_after_us,
        transition_at_us: config.transition_at_us,
        phase_jump_permille: config.phase_jump_permille,
        points,
        content_sha256: [0; 32],
    };
    oracle.content_sha256 = time_resolution_oracle_digest(&oracle);

    Ok((task, oracle))
}

/// Construct internal-update arms that reuse the same public input stream.
///
/// This sweep changes only the number of internal updates per observation
/// interval; it does not change observation count, physical duration, or
/// public input bytes.
pub fn internal_step_arms(
    task: &TimeResolutionTask,
) -> Result<Vec<InternalStepArm>, TimeResolutionVerificationError> {
    let observation_interval_us = task.observation_interval_us;
    if observation_interval_us == 0 || task.observations.is_empty() {
        return Err(TimeResolutionVerificationError::InvalidProfile { substeps: 0 });
    }
    let observation_count = u64::try_from(task.observations.len()).unwrap_or(u64::MAX);
    let mut arms = Vec::new();
    for substeps in INTERNAL_SUBSTEP_ARMS {
        if observation_interval_us % u64::from(substeps) == 0 {
            arms.push(InternalStepArm {
                internal_substeps_per_observation: substeps,
                internal_dt_us: observation_interval_us / u64::from(substeps),
                total_internal_updates: observation_count.saturating_mul(u64::from(substeps)),
            });
        }
    }
    if arms.is_empty() {
        return Err(TimeResolutionVerificationError::InvalidProfile { substeps: 0 });
    }
    Ok(arms)
}

/// Bind an internal-update arm to a task and fill its exact update count.
pub fn bind_internal_step_arm(
    task: &TimeResolutionTask,
    substeps: u32,
) -> Result<InternalStepArm, TimeResolutionVerificationError> {
    if !INTERNAL_SUBSTEP_ARMS.contains(&substeps)
        || task.observation_interval_us % u64::from(substeps) != 0
    {
        return Err(TimeResolutionVerificationError::InvalidProfile { substeps });
    }
    let observation_count = u64::try_from(task.observations.len()).unwrap_or(u64::MAX);
    Ok(InternalStepArm {
        internal_substeps_per_observation: substeps,
        internal_dt_us: task.observation_interval_us / u64::from(substeps),
        total_internal_updates: observation_count.saturating_mul(u64::from(substeps)),
    })
}

pub fn time_resolution_task_digest(task: &TimeResolutionTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.time-resolution.public.v1");
    hash.u64(task.stream_id);
    hash.u64(task.duration_us);
    hash.u64(task.observation_interval_us);
    hash.usize(task.observations.len());
    for observation in &task.observations {
        hash.u32(observation.step);
        hash.u64(observation.time_us);
        hash.u16(observation.value as u16);
    }
    hash.finish()
}

pub fn time_resolution_oracle_digest(oracle: &TimeResolutionOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.time-resolution.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u16(oracle.initial_phase_permille);
    hash.u64(oracle.period_before_us);
    hash.u64(oracle.period_after_us);
    hash.u64(oracle.transition_at_us);
    hash.u16(oracle.phase_jump_permille);
    hash.usize(oracle.points.len());
    for point in &oracle.points {
        hash.u32(point.step);
        hash.u64(point.time_us);
        hash.u16(point.phase_permille);
        hash.u16(point.latent_value as u16);
        hash.u16(point.observed_value as u16);
    }
    hash.finish()
}

/// Verify public and oracle digests plus their exact physical-time relationship.
pub fn verify_time_resolution_trace(
    task: &TimeResolutionTask,
    oracle: &TimeResolutionOracle,
) -> Result<(), TimeResolutionVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(TimeResolutionVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != time_resolution_task_digest(task) {
        return Err(TimeResolutionVerificationError::PublicDigestMismatch);
    }
    if oracle.content_sha256 != time_resolution_oracle_digest(oracle) {
        return Err(TimeResolutionVerificationError::OracleDigestMismatch);
    }
    if task.observations.len() != oracle.points.len()
        || task.observations.len() > MAX_SLOTS
        || task.duration_us == 0
        || task.duration_us > MAX_DURATION_US
        || task.observation_interval_us == 0
        || task.duration_us % task.observation_interval_us != 0
        || usize::try_from(task.duration_us / task.observation_interval_us)
            .unwrap_or(usize::MAX)
            != task.observations.len()
    {
        return Err(TimeResolutionVerificationError::LengthMismatch);
    }
    if oracle.initial_phase_permille >= 1_000
        || oracle.period_before_us < 2
        || oracle.period_after_us < 2
        || oracle.transition_at_us == 0
        || oracle.transition_at_us >= task.duration_us
        || oracle.phase_jump_permille >= 1_000
    {
        return Err(TimeResolutionVerificationError::InvalidOracleParameters);
    }

    for (index, (observation, point)) in task
        .observations
        .iter()
        .zip(&oracle.points)
        .enumerate()
    {
        let expected_step = u32::try_from(index).unwrap_or(u32::MAX);
        let expected_time = u64::try_from(index)
            .unwrap_or(u64::MAX)
            .saturating_mul(task.observation_interval_us);
        if observation.step != expected_step
            || point.step != expected_step
            || observation.time_us != expected_time
            || point.time_us != expected_time
        {
            return Err(TimeResolutionVerificationError::InvalidSchedule { step: index });
        }
        let expected_phase = phase_at(
            point.time_us,
            oracle.initial_phase_permille,
            oracle.period_before_us,
            oracle.period_after_us,
            oracle.transition_at_us,
            oracle.phase_jump_permille,
        );
        if point.phase_permille != expected_phase
            || point.latent_value != triangle_value(expected_phase)
        {
            return Err(TimeResolutionVerificationError::OraclePhaseMismatch { step: index });
        }
        if observation.value != point.observed_value
            || (i32::from(point.observed_value) - i32::from(point.latent_value)).abs()
                > MAX_NOISE_AS_I32
        {
            return Err(TimeResolutionVerificationError::PublicOracleValueMismatch { step: index });
        }
    }
    Ok(())
}

const MAX_NOISE_AS_I32: i32 = 64;

pub fn verify_time_resolution_seeded(
    task: &TimeResolutionTask,
    oracle: &TimeResolutionOracle,
    config: TimeResolutionConfig,
) -> Result<(), TimeResolutionVerificationError> {
    verify_time_resolution_trace(task, oracle)?;
    let (expected_task, expected_oracle) =
        generate_time_resolution_task(config).map_err(|_| {
            TimeResolutionVerificationError::InvalidOracleParameters
        })?;
    if &expected_task != task || &expected_oracle != oracle {
        return Err(TimeResolutionVerificationError::PublicOracleValueMismatch { step: 0 });
    }
    Ok(())
}

fn validate_config(config: TimeResolutionConfig) -> Result<(), TimeResolutionGenerationError> {
    if config.duration_us == 0 {
        return Err(TimeResolutionGenerationError::ZeroDuration);
    }
    if config.observation_interval_us == 0 {
        return Err(TimeResolutionGenerationError::ZeroObservationInterval);
    }
    if config.duration_us % config.observation_interval_us != 0 {
        return Err(TimeResolutionGenerationError::DurationNotDivisibleByObservationInterval {
            duration_us: config.duration_us,
            observation_interval_us: config.observation_interval_us,
        });
    }
    let slots = usize::try_from(config.duration_us / config.observation_interval_us)
        .unwrap_or(usize::MAX);
    if slots < 2 {
        return Err(TimeResolutionGenerationError::TooFewObservations {
            requested: slots,
            minimum: 2,
        });
    }
    if slots > MAX_SLOTS {
        return Err(TimeResolutionGenerationError::TooManyObservations {
            requested: slots,
            maximum: MAX_SLOTS,
        });
    }
    if config.duration_us > MAX_DURATION_US {
        return Err(TimeResolutionGenerationError::DurationExceedsLimit);
    }
    if config.period_before_us < 2 || config.period_before_us > MAX_DURATION_US {
        return Err(TimeResolutionGenerationError::InvalidPeriod {
            name: "period_before_us",
            period_us: config.period_before_us,
        });
    }
    if config.period_after_us < 2 || config.period_after_us > MAX_DURATION_US {
        return Err(TimeResolutionGenerationError::InvalidPeriod {
            name: "period_after_us",
            period_us: config.period_after_us,
        });
    }
    if config.transition_at_us == 0 || config.transition_at_us >= config.duration_us {
        return Err(TimeResolutionGenerationError::InvalidTransition {
            transition_at_us: config.transition_at_us,
            duration_us: config.duration_us,
        });
    }
    if config.phase_jump_permille >= 1_000 {
        return Err(TimeResolutionGenerationError::InvalidPhaseJump {
            phase_jump_permille: config.phase_jump_permille,
            maximum: 999,
        });
    }
    if config.noise_amplitude > MAX_NOISE_AMPLITUDE {
        return Err(TimeResolutionGenerationError::ExcessiveNoise {
            noise_amplitude: config.noise_amplitude,
            maximum: MAX_NOISE_AMPLITUDE,
        });
    }
    Ok(())
}

fn time_resolution_stream_id(config: TimeResolutionConfig) -> u64 {
    let mut value = config.seed ^ config.duration_us.rotate_left(7);
    value ^= config.observation_interval_us.rotate_left(17);
    value ^= config.period_before_us.rotate_left(27);
    value ^= config.period_after_us.rotate_left(37);
    value ^= config.transition_at_us.rotate_left(47);
    value ^= u64::from(config.phase_jump_permille).rotate_left(11);
    value ^= u64::from(config.noise_amplitude).rotate_left(23);
    mix64(value)
}

fn phase_at(
    time_us: u64,
    initial_phase_permille: u16,
    period_before_us: u64,
    period_after_us: u64,
    transition_at_us: u64,
    phase_jump_permille: u16,
) -> u16 {
    let initial = u128::from(initial_phase_permille);
    let before = (u128::from(time_us.min(transition_at_us)) * 1_000)
        / u128::from(period_before_us);
    let phase = if time_us < transition_at_us {
        (initial + before) % 1_000
    } else {
        let at_transition =
            (initial + before + u128::from(phase_jump_permille)) % 1_000;
        let after = (u128::from(time_us - transition_at_us) * 1_000)
            / u128::from(period_after_us);
        (at_transition + after) % 1_000
    };
    u16::try_from(phase).unwrap_or(0)
}

fn triangle_value(phase_permille: u16) -> i16 {
    let phase = i32::from(phase_permille);
    let value = if phase < 500 {
        -1_000 + phase * 4
    } else {
        1_000 - ((phase - 500) * 4)
    };
    i16::try_from(value).unwrap_or(0)
}

fn deterministic_noise(seed: u64, time_us: u64, amplitude: u16) -> i16 {
    if amplitude == 0 {
        return 0;
    }
    let mixed = mix64(seed ^ time_us.rotate_left(19) ^ 0x4E4F_4953_455F_4B45);
    let width = u64::from(amplitude).saturating_mul(2).saturating_add(1);
    let signed = i64::try_from(mixed % width).unwrap_or(0) - i64::from(amplitude);
    i16::try_from(signed).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_stream_is_deterministic_and_seed_replayable() {
        let config = TimeResolutionConfig::reference(0x9496_2026, 10_000);
        let first = generate_time_resolution_task(config).unwrap();
        let second = generate_time_resolution_task(config).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            verify_time_resolution_seeded(&first.0, &first.1, config),
            Ok(())
        );
    }

    #[test]
    fn internal_substep_arms_change_compute_not_public_input() {
        let config = TimeResolutionConfig::reference(91, 10_000);
        let (task, oracle) = generate_time_resolution_task(config).unwrap();
        let arms = internal_step_arms(&task).unwrap();
        assert_eq!(
            arms.iter()
                .map(|arm| arm.internal_substeps_per_observation)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8, 16]
        );
        let baseline_bytes = time_resolution_task_digest(&task);
        for arm in arms {
            let bound = bind_internal_step_arm(&task, arm.internal_substeps_per_observation)
                .unwrap();
            assert_eq!(bound.total_internal_updates,
                u64::try_from(task.observations.len()).unwrap()
                    * u64::from(arm.internal_substeps_per_observation)
            );
            assert_eq!(time_resolution_task_digest(&task), baseline_bytes);
            assert_eq!(verify_time_resolution_trace(&task, &oracle), Ok(()));
        }
    }

    #[test]
    fn all_profiles_preserve_same_world_duration_and_observation_count() {
        let config = TimeResolutionConfig::reference(92, 10_000);
        let (task, _) = generate_time_resolution_task(config).unwrap();
        let arms = internal_step_arms(&task).unwrap();
        for arm in arms {
            assert_eq!(task.duration_us, 4_000_000);
            assert_eq!(task.observations.len(), 400);
            assert_eq!(
                arm.total_internal_updates,
                400 * u64::from(arm.internal_substeps_per_observation)
            );
        }
    }

    #[test]
    fn observation_rate_variants_match_at_common_physical_timestamps() {
        let fine = TimeResolutionConfig::reference(0xabc, 10_000);
        let coarse = TimeResolutionConfig::reference(0xabc, 20_000);
        let (fine_task, _) = generate_time_resolution_task(fine).unwrap();
        let (coarse_task, _) = generate_time_resolution_task(coarse).unwrap();
        for coarse_observation in &coarse_task.observations {
            let fine_observation = fine_task
                .observations
                .iter()
                .find(|row| row.time_us == coarse_observation.time_us)
                .unwrap();
            assert_eq!(fine_observation.value, coarse_observation.value);
        }
    }

    #[test]
    fn different_internal_substep_counts_do_not_change_input_stream_digest() {
        let config = TimeResolutionConfig::reference(93, 10_000);
        let (task, _) = generate_time_resolution_task(config).unwrap();
        let digest_before = time_resolution_task_digest(&task);
        let one = bind_internal_step_arm(&task, 1).unwrap();
        let sixteen = bind_internal_step_arm(&task, 16).unwrap();
        assert_eq!(one.internal_dt_us, 10_000);
        assert_eq!(sixteen.internal_dt_us, 625);
        assert_eq!(task.content_sha256, digest_before);
    }

    #[test]
    fn verifier_rejects_public_observation_mutation() {
        let config = TimeResolutionConfig::reference(94, 10_000);
        let (mut task, oracle) = generate_time_resolution_task(config).unwrap();
        task.observations[0].value = task.observations[0].value.saturating_add(1);
        assert_eq!(
            verify_time_resolution_trace(&task, &oracle),
            Err(TimeResolutionVerificationError::PublicDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_recommitted_but_wrong_oracle_phase() {
        let config = TimeResolutionConfig::reference(95, 10_000);
        let (task, mut oracle) = generate_time_resolution_task(config).unwrap();
        oracle.points[0].phase_permille = (oracle.points[0].phase_permille + 1) % 1_000;
        oracle.content_sha256 = time_resolution_oracle_digest(&oracle);
        assert_eq!(
            verify_time_resolution_trace(&task, &oracle),
            Err(TimeResolutionVerificationError::OraclePhaseMismatch { step: 0 })
        );
    }

    #[test]
    fn invalid_duration_grid_is_rejected() {
        let mut config = TimeResolutionConfig::reference(1, 10_000);
        config.duration_us = config.duration_us.saturating_sub(1);
        assert!(matches!(
            generate_time_resolution_task(config),
            Err(TimeResolutionGenerationError::DurationNotDivisibleByObservationInterval { .. })
        ));
    }

    #[test]
    fn invalid_transition_is_rejected() {
        let mut config = TimeResolutionConfig::reference(1, 10_000);
        config.transition_at_us = config.duration_us;
        assert!(matches!(
            generate_time_resolution_task(config),
            Err(TimeResolutionGenerationError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn invalid_internal_profile_is_rejected() {
        let config = TimeResolutionConfig::reference(1, 10_000);
        let (task, _) = generate_time_resolution_task(config).unwrap();
        assert!(matches!(
            bind_internal_step_arm(&task, 3),
            Err(TimeResolutionVerificationError::InvalidProfile { substeps: 3 })
        ));
    }
}
