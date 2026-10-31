//! Deterministic irregular-time and missing-observation workload.
//!
//! This task samples a known three-timescale latent process on an irregular
//! schedule, drops a controlled fraction of observations, and adds bounded
//! integer noise. Elapsed microseconds are explicit; a missing observation
//! creates a larger time gap rather than an invented zero-valued sample.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::fmt;

const MAX_SLOTS: usize = 100_000;
const MAX_DURATION_US: u64 = 86_400_000_000;
const MAX_NOISE: i16 = 15;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IrregularConfig {
    pub seed: u64,
    pub slots: usize,
    pub nominal_interval_us: u64,
    /// Maximum absolute interval jitter as a fraction of nominal interval, per 1,000.
    pub jitter_per_mille: u16,
    /// Independent observation-drop probability, per 1,000.
    pub missing_per_mille: u16,
    pub periods_us: [u64; 3],
}

impl IrregularConfig {
    pub fn reference(seed: u64) -> Self {
        Self {
            seed,
            slots: 512,
            nominal_interval_us: 10_000,
            jitter_per_mille: 250,
            missing_per_mille: 150,
            periods_us: [60_000, 250_000, 1_000_000],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IrregularObservation {
    pub time_us: u64,
    pub fast: i16,
    pub medium: i16,
    pub slow: i16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IrregularTask {
    pub stream_id: u64,
    pub nominal_interval_us: u64,
    pub jitter_per_mille: u16,
    pub missing_per_mille: u16,
    pub observations: Vec<IrregularObservation>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IrregularOraclePoint {
    pub time_us: u64,
    pub observed: [i16; 3],
    pub latent: [i16; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IrregularOracle {
    pub stream_id: u64,
    pub scheduled_slots: usize,
    pub dropped_slots: Vec<u32>,
    pub periods_us: [u64; 3],
    pub phases_us: [u64; 3],
    pub points: Vec<IrregularOraclePoint>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IrregularGenerationError {
    TooFewSlots { requested: usize, minimum: usize },
    TooManySlots { requested: usize, maximum: usize },
    ZeroNominalInterval,
    ExcessiveJitter { requested: u16, maximum: u16 },
    ExcessiveMissingness { requested: u16, maximum: u16 },
    InvalidPeriod { index: usize, period_us: u64 },
    DurationExceedsLimit,
    NoObservationsAfterMissingness,
}

impl fmt::Display for IrregularGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSlots { requested, minimum } => {
                write!(f, "slots {requested} is below minimum {minimum}")
            }
            Self::TooManySlots { requested, maximum } => {
                write!(f, "slots {requested} exceeds maximum {maximum}")
            }
            Self::ZeroNominalInterval => write!(f, "nominal_interval_us must be greater than zero"),
            Self::ExcessiveJitter { requested, maximum } => {
                write!(f, "jitter_per_mille {requested} exceeds maximum {maximum}")
            }
            Self::ExcessiveMissingness { requested, maximum } => {
                write!(f, "missing_per_mille {requested} exceeds maximum {maximum}")
            }
            Self::InvalidPeriod { index, period_us } => {
                write!(f, "period {index} is invalid: {period_us} microseconds")
            }
            Self::DurationExceedsLimit => {
                write!(f, "irregular stream duration exceeds the configured limit")
            }
            Self::NoObservationsAfterMissingness => {
                write!(f, "missingness removed every scheduled observation")
            }
        }
    }
}

impl std::error::Error for IrregularGenerationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IrregularVerificationError {
    StreamIdMismatch,
    PublicDigestMismatch,
    OracleDigestMismatch,
    SeedReplayMismatch,
    LengthMismatch,
    DroppedSlotManifestInvalid,
    InvalidTaskContract,
    InvalidOracleSchedule,
    InvalidLatentValue { index: usize },
    TimeNotStrictlyIncreasing { index: usize },
    OraclePointMismatch { index: usize },
    NoiseBoundExceeded { index: usize },
}

impl fmt::Display for IrregularVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "irregular task and oracle stream IDs differ"),
            Self::PublicDigestMismatch => {
                write!(f, "irregular public stream SHA-256 digest mismatch")
            }
            Self::OracleDigestMismatch => write!(f, "irregular oracle SHA-256 digest mismatch"),
            Self::SeedReplayMismatch => write!(
                f,
                "irregular task/oracle do not match the supplied seed and configuration"
            ),
            Self::LengthMismatch => {
                write!(f, "irregular public/oracle point count is inconsistent")
            }
            Self::DroppedSlotManifestInvalid => write!(f, "dropped-slot manifest is invalid"),
            Self::InvalidTaskContract => {
                write!(f, "irregular task timing or missingness contract is invalid")
            }
            Self::InvalidOracleSchedule => {
                write!(f, "irregular oracle schedule is outside the generator bounds")
            }
            Self::InvalidLatentValue { index } => {
                write!(
                    f,
                    "irregular latent value at point {index} disagrees with its period and phase"
                )
            }
            Self::TimeNotStrictlyIncreasing { index } => {
                write!(f, "observation timestamps are not strictly increasing at index {index}")
            }
            Self::OraclePointMismatch { index } => {
                write!(f, "observation at index {index} does not match the oracle point")
            }
            Self::NoiseBoundExceeded { index } => {
                write!(f, "observation noise at index {index} exceeds the declared bound")
            }
        }
    }
}

impl std::error::Error for IrregularVerificationError {}

/// Generate a multiscale latent stream on a jittered schedule with dropouts.
pub fn generate_irregular_task(
    config: IrregularConfig,
) -> Result<(IrregularTask, IrregularOracle), IrregularGenerationError> {
    validate_config(config)?;

    let stream_id = irregular_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x4952_5245_4755_4C41);
    let phases_us = [
        rng.next_u64() % config.periods_us[0],
        rng.next_u64() % config.periods_us[1],
        rng.next_u64() % config.periods_us[2],
    ];

    let mut observations = Vec::with_capacity(config.slots);
    let mut oracle_points = Vec::with_capacity(config.slots);
    let mut dropped_slots = Vec::new();
    let mut time_us = 0_u64;

    for slot in 0..config.slots {
        if slot > 0 {
            let maximum_jitter = u64::try_from(
                (u128::from(config.nominal_interval_us)
                    * u128::from(config.jitter_per_mille))
                    / 1_000,
            )
            .unwrap_or(u64::MAX);
            let width = maximum_jitter.saturating_mul(2).saturating_add(1);
            let offset = rng.next_u64() % width;
            let interval = i128::from(config.nominal_interval_us)
                + i128::from(offset)
                - i128::from(maximum_jitter);
            let interval = u64::try_from(interval).unwrap_or(1);
            time_us = time_us.saturating_add(interval);
        }

        let should_drop = rng.next_u64() % 1_000 < u64::from(config.missing_per_mille);
        if should_drop {
            dropped_slots.push(u32::try_from(slot).unwrap_or(u32::MAX));
            continue;
        }

        let latent = [
            triangle_wave(time_us, config.periods_us[0], phases_us[0]),
            triangle_wave(time_us, config.periods_us[1], phases_us[1]),
            triangle_wave(time_us, config.periods_us[2], phases_us[2]),
        ];
        let observed = [
            add_bounded_noise(latent[0], &mut rng),
            add_bounded_noise(latent[1], &mut rng),
            add_bounded_noise(latent[2], &mut rng),
        ];

        observations.push(IrregularObservation {
            time_us,
            fast: observed[0],
            medium: observed[1],
            slow: observed[2],
        });
        oracle_points.push(IrregularOraclePoint {
            time_us,
            observed,
            latent,
        });
    }

    if observations.is_empty() {
        return Err(IrregularGenerationError::NoObservationsAfterMissingness);
    }

    let mut public_task = IrregularTask {
        stream_id,
        nominal_interval_us: config.nominal_interval_us,
        jitter_per_mille: config.jitter_per_mille,
        missing_per_mille: config.missing_per_mille,
        observations,
        content_sha256: [0; 32],
    };
    public_task.content_sha256 = irregular_content_digest(&public_task);

    let mut oracle = IrregularOracle {
        stream_id,
        scheduled_slots: config.slots,
        dropped_slots,
        periods_us: config.periods_us,
        phases_us,
        points: oracle_points,
        content_sha256: [0; 32],
    };
    oracle.content_sha256 = irregular_oracle_digest(&oracle);

    Ok((public_task, oracle))
}

/// Canonical public-stream digest for an irregular task.
pub fn irregular_content_digest(task: &IrregularTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.irregular.public.v1");
    hash.u64(task.stream_id);
    hash.u64(task.nominal_interval_us);
    hash.u16(task.jitter_per_mille);
    hash.u16(task.missing_per_mille);
    hash.usize(task.observations.len());
    for observation in &task.observations {
        hash.u64(observation.time_us);
        hash.u16(observation.fast as u16);
        hash.u16(observation.medium as u16);
        hash.u16(observation.slow as u16);
    }
    hash.finish()
}

/// Canonical oracle digest for the hidden schedule and latent trajectories.
pub fn irregular_oracle_digest(oracle: &IrregularOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.irregular.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.usize(oracle.scheduled_slots);
    hash.usize(oracle.dropped_slots.len());
    for slot in &oracle.dropped_slots {
        hash.u32(*slot);
    }
    for period in oracle.periods_us {
        hash.u64(period);
    }
    for phase in oracle.phases_us {
        hash.u64(phase);
    }
    hash.usize(oracle.points.len());
    for point in &oracle.points {
        hash.u64(point.time_us);
        for value in point.observed {
            hash.u16(value as u16);
        }
        for value in point.latent {
            hash.u16(value as u16);
        }
    }
    hash.finish()
}

/// Evaluator-side structural, provenance-digest, and latent/noise consistency check.
pub fn verify_irregular_trace(
    task: &IrregularTask,
    oracle: &IrregularOracle,
) -> Result<(), IrregularVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(IrregularVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != irregular_content_digest(task) {
        return Err(IrregularVerificationError::PublicDigestMismatch);
    }
    if oracle.content_sha256 != irregular_oracle_digest(oracle) {
        return Err(IrregularVerificationError::OracleDigestMismatch);
    }
    if task.observations.len() != oracle.points.len()
        || task.observations.len().saturating_add(oracle.dropped_slots.len())
            != oracle.scheduled_slots
    {
        return Err(IrregularVerificationError::LengthMismatch);
    }
    if task.nominal_interval_us == 0
        || task.jitter_per_mille > 900
        || task.missing_per_mille >= 1_000
        || oracle.scheduled_slots < 2
        || oracle.scheduled_slots > MAX_SLOTS
    {
        return Err(IrregularVerificationError::InvalidTaskContract);
    }
    let mut previous_drop = None;
    for slot in &oracle.dropped_slots {
        if usize::try_from(*slot).unwrap_or(usize::MAX) >= oracle.scheduled_slots
            || previous_drop.is_some_and(|prior| prior >= *slot)
        {
            return Err(IrregularVerificationError::DroppedSlotManifestInvalid);
        }
        previous_drop = Some(*slot);
    }
    if oracle
        .periods_us
        .iter()
        .any(|period| *period < 2 || *period > MAX_DURATION_US)
        || oracle
            .phases_us
            .iter()
            .zip(oracle.periods_us)
            .any(|(phase, period)| *phase >= period)
    {
        return Err(IrregularVerificationError::InvalidOracleSchedule);
    }

    let mut observed_slots = Vec::with_capacity(task.observations.len());
    let mut next_drop = 0_usize;
    for slot in 0..oracle.scheduled_slots {
        if oracle.dropped_slots.get(next_drop).copied()
            == Some(u32::try_from(slot).unwrap_or(u32::MAX))
        {
            next_drop = next_drop.saturating_add(1);
        } else {
            observed_slots.push(u32::try_from(slot).unwrap_or(u32::MAX));
        }
    }
    if observed_slots.len() != task.observations.len() {
        return Err(IrregularVerificationError::DroppedSlotManifestInvalid);
    }

    let max_jitter = u64::try_from(
        (u128::from(task.nominal_interval_us) * u128::from(task.jitter_per_mille)) / 1_000,
    )
    .unwrap_or(u64::MAX);
    let minimum_interval = task.nominal_interval_us.saturating_sub(max_jitter);
    let maximum_interval = task.nominal_interval_us.saturating_add(max_jitter);

    for (index, ((observation, point), slot)) in task
        .observations
        .iter()
        .zip(&oracle.points)
        .zip(&observed_slots)
        .enumerate()
    {
        if index > 0
            && observation.time_us <= task.observations[index - 1].time_us
        {
            return Err(IrregularVerificationError::TimeNotStrictlyIncreasing { index });
        }
        let previous_slot = if index == 0 { 0 } else { observed_slots[index - 1] };
        let gap_slots = if index == 0 { *slot } else { slot.saturating_sub(previous_slot) };
        let delta_us = if index == 0 {
            observation.time_us
        } else {
            observation.time_us.saturating_sub(task.observations[index - 1].time_us)
        };
        let minimum_gap = u128::from(minimum_interval) * u128::from(gap_slots);
        let maximum_gap = u128::from(maximum_interval) * u128::from(gap_slots);
        if u128::from(delta_us) < minimum_gap || u128::from(delta_us) > maximum_gap {
            return Err(IrregularVerificationError::InvalidOracleSchedule);
        }

        let observed = [observation.fast, observation.medium, observation.slow];
        if observation.time_us != point.time_us || observed != point.observed {
            return Err(IrregularVerificationError::OraclePointMismatch { index });
        }
        let expected_latent = [
            triangle_wave(point.time_us, oracle.periods_us[0], oracle.phases_us[0]),
            triangle_wave(point.time_us, oracle.periods_us[1], oracle.phases_us[1]),
            triangle_wave(point.time_us, oracle.periods_us[2], oracle.phases_us[2]),
        ];
        if point.latent != expected_latent {
            return Err(IrregularVerificationError::InvalidLatentValue { index });
        }
        for (observed_value, latent_value) in observed.into_iter().zip(point.latent) {
            if (i32::from(observed_value) - i32::from(latent_value)).abs()
                > i32::from(MAX_NOISE)
            {
                return Err(IrregularVerificationError::NoiseBoundExceeded { index });
            }
        }
    }
    Ok(())
}

/// Verify a task against an evaluator-held seed and complete generator configuration.
///
/// This is stronger than digest checking: it rejects a semantically valid but
/// seed-inconsistent stream, even when an attacker recomputes both SHA-256
/// digests. The seed/configuration must itself be protected by the evaluator.
pub fn verify_irregular_trace_seeded(
    task: &IrregularTask,
    oracle: &IrregularOracle,
    config: IrregularConfig,
) -> Result<(), IrregularVerificationError> {
    verify_irregular_trace(task, oracle)?;
    let (expected_task, expected_oracle) = generate_irregular_task(config)
        .map_err(|_| IrregularVerificationError::SeedReplayMismatch)?;
    if task != &expected_task || oracle != &expected_oracle {
        return Err(IrregularVerificationError::SeedReplayMismatch);
    }
    Ok(())
}

fn validate_config(config: IrregularConfig) -> Result<(), IrregularGenerationError> {
    if config.slots < 2 {
        return Err(IrregularGenerationError::TooFewSlots {
            requested: config.slots,
            minimum: 2,
        });
    }
    if config.slots > MAX_SLOTS {
        return Err(IrregularGenerationError::TooManySlots {
            requested: config.slots,
            maximum: MAX_SLOTS,
        });
    }
    if config.nominal_interval_us == 0 {
        return Err(IrregularGenerationError::ZeroNominalInterval);
    }
    if config.jitter_per_mille > 900 {
        return Err(IrregularGenerationError::ExcessiveJitter {
            requested: config.jitter_per_mille,
            maximum: 900,
        });
    }
    if config.missing_per_mille >= 1_000 {
        return Err(IrregularGenerationError::ExcessiveMissingness {
            requested: config.missing_per_mille,
            maximum: 999,
        });
    }
    for (index, period) in config.periods_us.into_iter().enumerate() {
        if period < 2 || period > MAX_DURATION_US {
            return Err(IrregularGenerationError::InvalidPeriod {
                index,
                period_us: period,
            });
        }
    }
    let maximum_interval = (u128::from(config.nominal_interval_us)
        * (1_000 + u128::from(config.jitter_per_mille)))
        / 1_000;
    let total_duration = maximum_interval.saturating_mul(
        u128::try_from(config.slots.saturating_sub(1)).unwrap_or(u128::MAX),
    );
    if total_duration > u128::from(MAX_DURATION_US) {
        return Err(IrregularGenerationError::DurationExceedsLimit);
    }
    Ok(())
}

fn irregular_stream_id(config: IrregularConfig) -> u64 {
    let mut value = config.seed ^ (config.slots as u64).rotate_left(7);
    value ^= config.nominal_interval_us.rotate_left(17);
    value ^= u64::from(config.jitter_per_mille).rotate_left(29);
    value ^= u64::from(config.missing_per_mille).rotate_left(41);
    value ^= mix64(config.periods_us[0])
        ^ mix64(config.periods_us[1].rotate_left(13))
        ^ mix64(config.periods_us[2].rotate_left(31));
    mix64(value)
}

/// Symmetric, deterministic fixed-point triangle wave in [-1000, 1000].
fn triangle_wave(time_us: u64, period_us: u64, phase_us: u64) -> i16 {
    let position = ((time_us % period_us) + (phase_us % period_us)) % period_us;
    let first_half = period_us / 2;
    let value = if position < first_half {
        -1_000_i64
            + i64::try_from(
                (u128::from(position) * 2_000) / u128::from(first_half),
            )
            .unwrap_or(0)
    } else {
        1_000_i64
            - i64::try_from(
                (u128::from(position - first_half) * 2_000)
                    / u128::from(period_us - first_half),
            )
            .unwrap_or(0)
    };
    i16::try_from(value).unwrap_or(0)
}

fn add_bounded_noise(value: i16, rng: &mut SplitMix64) -> i16 {
    let noise = i16::try_from(rng.next_u64() % 31).unwrap_or(0) - MAX_NOISE;
    value.saturating_add(noise)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let config = IrregularConfig::reference(17);
        assert_eq!(
            generate_irregular_task(config),
            generate_irregular_task(config)
        );
    }

    #[test]
    fn generated_public_stream_passes_oracle_verifier() {
        let config = IrregularConfig::reference(21);
        let (task, oracle) = generate_irregular_task(config).unwrap();
        assert_eq!(verify_irregular_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn timestamps_are_strictly_monotonic() {
        let config = IrregularConfig::reference(22);
        let (task, _) = generate_irregular_task(config).unwrap();
        for pair in task.observations.windows(2) {
            assert!(pair[1].time_us > pair[0].time_us);
        }
    }

    #[test]
    fn missingness_is_explicit_in_oracle_manifest() {
        let config = IrregularConfig::reference(23);
        let (task, oracle) = generate_irregular_task(config).unwrap();
        assert_eq!(
            task.observations.len() + oracle.dropped_slots.len(),
            config.slots
        );
    }

    #[test]
    fn jitter_bounds_are_respected_without_dropouts() {
        let mut config = IrregularConfig::reference(24);
        config.missing_per_mille = 0;
        let (task, _) = generate_irregular_task(config).unwrap();
        let max_jitter = config.nominal_interval_us
            * u64::from(config.jitter_per_mille)
            / 1_000;
        for pair in task.observations.windows(2) {
            let delta = pair[1].time_us - pair[0].time_us;
            assert!(delta >= config.nominal_interval_us - max_jitter);
            assert!(delta <= config.nominal_interval_us + max_jitter);
        }
    }

    #[test]
    fn generated_public_stream_and_oracle_have_sha256_digests() {
        let config = IrregularConfig::reference(25);
        let (task, oracle) = generate_irregular_task(config).unwrap();
        assert_eq!(task.content_sha256, irregular_content_digest(&task));
        assert_eq!(oracle.content_sha256, irregular_oracle_digest(&oracle));
    }

    #[test]
    fn seeded_verifier_rejects_recommitted_but_seed_inconsistent_stream() {
        let config = IrregularConfig::reference(2_026);
        let (mut task, oracle) = generate_irregular_task(config).unwrap();
        task.observations[0].fast ^= 1;
        task.content_sha256 = irregular_content_digest(&task);
        assert_eq!(
            verify_irregular_trace_seeded(&task, &oracle, config),
            Err(IrregularVerificationError::SeedReplayMismatch)
        );
    }

    #[test]
    fn verifier_rejects_public_observation_mutation() {
        let config = IrregularConfig::reference(26);
        let (mut task, oracle) = generate_irregular_task(config).unwrap();
        task.observations[0].fast ^= 1;
        assert_eq!(
            verify_irregular_trace(&task, &oracle),
            Err(IrregularVerificationError::PublicDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_recommitted_oracle_with_wrong_latent_wave() {
        let config = IrregularConfig::reference(27);
        let (task, mut oracle) = generate_irregular_task(config).unwrap();
        oracle.points[0].latent[0] = oracle.points[0].latent[0].saturating_add(1);
        oracle.content_sha256 = irregular_oracle_digest(&oracle);
        assert_eq!(
            verify_irregular_trace(&task, &oracle),
            Err(IrregularVerificationError::InvalidLatentValue { index: 0 })
        );
    }

    #[test]
    fn verifier_rejects_oracle_mutation() {
        let config = IrregularConfig::reference(27);
        let (task, mut oracle) = generate_irregular_task(config).unwrap();
        oracle.periods_us[0] = oracle.periods_us[0].saturating_add(1);
        assert_eq!(
            verify_irregular_trace(&task, &oracle),
            Err(IrregularVerificationError::OracleDigestMismatch)
        );
    }

    #[test]
    fn invalid_jitter_is_rejected() {
        let mut config = IrregularConfig::reference(1);
        config.jitter_per_mille = 1_000;
        assert!(matches!(
            generate_irregular_task(config),
            Err(IrregularGenerationError::ExcessiveJitter { .. })
        ));
    }

    #[test]
    fn complete_missingness_is_rejected() {
        let mut config = IrregularConfig::reference(1);
        config.missing_per_mille = 1_000;
        assert!(matches!(
            generate_irregular_task(config),
            Err(IrregularGenerationError::ExcessiveMissingness { .. })
        ));
    }

    #[test]
    fn zero_period_is_rejected() {
        let mut config = IrregularConfig::reference(1);
        config.periods_us[0] = 0;
        assert!(matches!(
            generate_irregular_task(config),
            Err(IrregularGenerationError::InvalidPeriod { .. })
        ));
    }

    #[test]
    fn excessive_duration_is_rejected() {
        let mut config = IrregularConfig::reference(1);
        config.nominal_interval_us = MAX_DURATION_US;
        assert!(matches!(
            generate_irregular_task(config),
            Err(IrregularGenerationError::DurationExceedsLimit)
        ));
    }
}
