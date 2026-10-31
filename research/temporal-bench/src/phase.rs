//! Deterministic phase-sensitive prediction under hidden phase/frequency shifts.
//!
//! The candidate receives timestamped noisy samples from a periodic signal.
//! The oracle records the hidden phase trajectory and exact next rising
//! zero-crossing time. Frequency changes preserve fractional cycle phase;
//! phase shifts create a controlled discontinuity. All time and phase
//! arithmetic is integer/fixed-point.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::fmt;

const MAX_PHASE_SLOTS: usize = 100_000;
const MAX_PHASE_DURATION_US: u64 = 86_400_000_000;
const MIN_PERIOD_US: u64 = 400;
const MAX_NOISE: i16 = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseMode {
    Stable,
    PhaseShift,
    FrequencyShift,
    Mixed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseConfig {
    pub seed: u64,
    pub slots: usize,
    pub sample_interval_us: u64,
    pub base_period_us: u64,
    pub mode: PhaseMode,
}

impl PhaseConfig {
    pub fn reference(seed: u64, mode: PhaseMode) -> Self {
        Self {
            seed,
            slots: 4_096,
            sample_interval_us: 1_000,
            base_period_us: 64_000,
            mode,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseObservation {
    pub step: u32,
    pub time_us: u64,
    pub sample: i16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseTask {
    pub stream_id: u64,
    pub sample_interval_us: u64,
    pub slots: usize,
    pub observations: Vec<PhaseObservation>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseOraclePoint {
    pub step: u32,
    pub time_us: u64,
    pub phase_permille: u16,
    pub period_us: u64,
    pub latent_sample: i16,
    pub next_rising_crossing_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseOracle {
    pub stream_id: u64,
    pub mode: PhaseMode,
    pub initial_phase_us: u64,
    pub shift_step: Option<u32>,
    pub period_before_us: u64,
    pub period_after_us: u64,
    pub phase_jump_us: u64,
    pub points: Vec<PhaseOraclePoint>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhaseGenerationError {
    TooFewSlots { requested: usize, minimum: usize },
    TooManySlots { requested: usize, maximum: usize },
    ZeroSampleInterval,
    InvalidPeriod { period_us: u64 },
    DurationExceedsLimit,
}

impl fmt::Display for PhaseGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSlots { requested, minimum } => {
                write!(f, "phase stream slots {requested} is below minimum {minimum}")
            }
            Self::TooManySlots { requested, maximum } => {
                write!(f, "phase stream slots {requested} exceeds maximum {maximum}")
            }
            Self::ZeroSampleInterval => write!(f, "sample_interval_us must be greater than zero"),
            Self::InvalidPeriod { period_us } => write!(
                f,
                "period {period_us} must be at least {MIN_PERIOD_US} us and divisible by four"
            ),
            Self::DurationExceedsLimit => write!(f, "phase stream duration exceeds one day"),
        }
    }
}

impl std::error::Error for PhaseGenerationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhaseVerificationError {
    StreamIdMismatch,
    PublicDigestMismatch,
    OracleDigestMismatch,
    InvalidTaskContract,
    InvalidOracleMode,
    InvalidShiftSchedule,
    InvalidPhasePoint { index: usize },
    InvalidCrossingTime { index: usize },
    InvalidObservation { expected_step: u32 },
    NoiseBoundExceeded { step: u32 },
}

impl fmt::Display for PhaseVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "phase task and oracle stream IDs differ"),
            Self::PublicDigestMismatch => write!(f, "phase public stream SHA-256 digest mismatch"),
            Self::OracleDigestMismatch => write!(f, "phase oracle SHA-256 digest mismatch"),
            Self::InvalidTaskContract => write!(f, "phase task timing or stream length is invalid"),
            Self::InvalidOracleMode => write!(f, "phase oracle fields do not match the declared mode"),
            Self::InvalidShiftSchedule => write!(f, "phase/frequency shift time is outside the allowed range"),
            Self::InvalidPhasePoint { index } => {
                write!(f, "phase oracle point {index} disagrees with the fixed-point oscillator")
            }
            Self::InvalidCrossingTime { index } => {
                write!(f, "next rising-crossing oracle at point {index} is invalid")
            }
            Self::InvalidObservation { expected_step } => {
                write!(f, "phase observation is missing or reordered at step {expected_step}")
            }
            Self::NoiseBoundExceeded { step } => {
                write!(f, "phase observation noise exceeds the declared bound at step {step}")
            }
        }
    }
}

impl std::error::Error for PhaseVerificationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhasePrediction {
    pub step: u32,
    /// Predicted oscillator phase as a fraction of one cycle, in [0, 1000).
    pub phase_permille: u16,
    /// Absolute timestamp of the next rising zero crossing.
    pub next_rising_crossing_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseScore {
    pub evaluated_points: usize,
    pub circular_phase_error_permille: Vec<u16>,
    pub next_crossing_absolute_error_us: Vec<u64>,
}

impl PhaseScore {
    pub fn phase_error_percentile(&self, numerator: usize, denominator: usize) -> Option<u16> {
        nearest_rank(&self.circular_phase_error_permille, numerator, denominator)
    }

    pub fn crossing_error_percentile_us(
        &self,
        numerator: usize,
        denominator: usize,
    ) -> Option<u64> {
        nearest_rank(&self.next_crossing_absolute_error_us, numerator, denominator)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhaseScoringError {
    InvalidTrace(PhaseVerificationError),
    PredictionOutOfRange { step: u32 },
    InvalidPredictedPhase { step: u32, phase_permille: u16 },
    PredictionsNotStrictlyIncreasing { index: usize },
}

impl fmt::Display for PhaseScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTrace(error) => write!(f, "cannot score invalid phase task: {error}"),
            Self::PredictionOutOfRange { step } => {
                write!(f, "phase prediction at step {step} is outside the public stream")
            }
            Self::InvalidPredictedPhase { step, phase_permille } => {
                write!(f, "predicted phase {phase_permille} is invalid at step {step}")
            }
            Self::PredictionsNotStrictlyIncreasing { index } => {
                write!(f, "phase predictions are duplicated or reordered at index {index}")
            }
        }
    }
}

impl std::error::Error for PhaseScoringError {}

pub fn generate_phase_task(
    config: PhaseConfig,
) -> Result<(PhaseTask, PhaseOracle), PhaseGenerationError> {
    validate_config(config)?;

    let stream_id = phase_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x5048_4153_455F_3031);
    let initial_phase_us = rng.next_u64() % config.base_period_us;
    let shift_step = match config.mode {
        PhaseMode::Stable => None,
        PhaseMode::PhaseShift | PhaseMode::FrequencyShift | PhaseMode::Mixed => {
            let low = u64::try_from(config.slots / 3).unwrap_or(u64::MAX);
            let high = u64::try_from((config.slots * 2) / 3).unwrap_or(u64::MAX);
            let width = high.saturating_sub(low).saturating_add(1);
            Some(u32::try_from(low.saturating_add(rng.next_u64() % width)).unwrap_or(u32::MAX))
        }
    };

    let changes_frequency = matches!(config.mode, PhaseMode::FrequencyShift | PhaseMode::Mixed);
    let changes_phase = matches!(config.mode, PhaseMode::PhaseShift | PhaseMode::Mixed);
    let period_after_us = if changes_frequency {
        let scale_percent = 75 + (rng.next_u64() % 51);
        u64::try_from(
            (u128::from(config.base_period_us) * u128::from(scale_percent)) / 100,
        )
        .unwrap_or(config.base_period_us)
    } else {
        config.base_period_us
    };
    let phase_jump_us = if changes_phase {
        let jump_permille = 100 + (rng.next_u64() % 301);
        u64::try_from(
            (u128::from(config.base_period_us) * u128::from(jump_permille)) / 1_000,
        )
        .unwrap_or(0)
    } else {
        0
    };

    let mut oracle = PhaseOracle {
        stream_id,
        mode: config.mode,
        initial_phase_us,
        shift_step,
        period_before_us: config.base_period_us,
        period_after_us,
        phase_jump_us,
        points: Vec::with_capacity(config.slots),
        content_sha256: [0; 32],
    };
    let mut task = PhaseTask {
        stream_id,
        sample_interval_us: config.sample_interval_us,
        slots: config.slots,
        observations: Vec::with_capacity(config.slots),
        content_sha256: [0; 32],
    };

    for step in 0..config.slots {
        let step_u32 = u32::try_from(step).unwrap_or(u32::MAX);
        let time_us = u64::try_from(step)
            .unwrap_or(u64::MAX)
            .saturating_mul(config.sample_interval_us);
        let (phase_us, period_us) = phase_state_at(time_us, &oracle, config.sample_interval_us);
        let phase_permille = u16::try_from(
            (u128::from(phase_us) * 1_000) / u128::from(period_us),
        )
        .unwrap_or(0);
        let latent_sample = triangle_sample(phase_us, period_us);
        let next_rising_crossing_us =
            next_rising_crossing(time_us, &oracle, config.sample_interval_us);
        let noise = i16::try_from(rng.next_u64() % 41).unwrap_or(0) - MAX_NOISE;
        let sample = latent_sample.saturating_add(noise);

        task.observations.push(PhaseObservation {
            step: step_u32,
            time_us,
            sample,
        });
        oracle.points.push(PhaseOraclePoint {
            step: step_u32,
            time_us,
            phase_permille,
            period_us,
            latent_sample,
            next_rising_crossing_us,
        });
    }

    task.content_sha256 = phase_content_digest(&task);
    oracle.content_sha256 = phase_oracle_digest(&oracle);
    Ok((task, oracle))
}

pub fn phase_content_digest(task: &PhaseTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.phase.public.v1");
    hash.u64(task.stream_id);
    hash.u64(task.sample_interval_us);
    hash.usize(task.slots);
    hash.usize(task.observations.len());
    for observation in &task.observations {
        hash.u32(observation.step);
        hash.u64(observation.time_us);
        hash.u16(observation.sample as u16);
    }
    hash.finish()
}

pub fn phase_oracle_digest(oracle: &PhaseOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.phase.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u8(match oracle.mode {
        PhaseMode::Stable => 0,
        PhaseMode::PhaseShift => 1,
        PhaseMode::FrequencyShift => 2,
        PhaseMode::Mixed => 3,
    });
    hash.u64(oracle.initial_phase_us);
    match oracle.shift_step {
        Some(step) => {
            hash.u8(1);
            hash.u32(step);
        }
        None => hash.u8(0),
    }
    hash.u64(oracle.period_before_us);
    hash.u64(oracle.period_after_us);
    hash.u64(oracle.phase_jump_us);
    hash.usize(oracle.points.len());
    for point in &oracle.points {
        hash.u32(point.step);
        hash.u64(point.time_us);
        hash.u16(point.phase_permille);
        hash.u64(point.period_us);
        hash.u16(point.latent_sample as u16);
        hash.u64(point.next_rising_crossing_us);
    }
    hash.finish()
}

pub fn verify_phase_trace(
    task: &PhaseTask,
    oracle: &PhaseOracle,
) -> Result<(), PhaseVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(PhaseVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != phase_content_digest(task) {
        return Err(PhaseVerificationError::PublicDigestMismatch);
    }
    if oracle.content_sha256 != phase_oracle_digest(oracle) {
        return Err(PhaseVerificationError::OracleDigestMismatch);
    }
    if task.sample_interval_us == 0
        || task.slots < 3
        || task.slots > MAX_PHASE_SLOTS
        || task.observations.len() != task.slots
        || oracle.points.len() != task.slots
        || u128::from(task.sample_interval_us)
            .saturating_mul(u128::try_from(task.slots).unwrap_or(u128::MAX))
            > u128::from(MAX_PHASE_DURATION_US)
    {
        return Err(PhaseVerificationError::InvalidTaskContract);
    }

    if oracle.period_before_us < MIN_PERIOD_US
        || oracle.period_before_us % MIN_PERIOD_US != 0
        || oracle.period_after_us < MIN_PERIOD_US
        || oracle.period_after_us % (MIN_PERIOD_US / 4) != 0
        || oracle.initial_phase_us >= oracle.period_before_us
    {
        return Err(PhaseVerificationError::InvalidOracleMode);
    }

    let expected_mode_valid = match oracle.mode {
        PhaseMode::Stable => {
            oracle.shift_step.is_none()
                && oracle.period_after_us == oracle.period_before_us
                && oracle.phase_jump_us == 0
        }
        PhaseMode::PhaseShift => {
            oracle.shift_step.is_some()
                && oracle.period_after_us == oracle.period_before_us
                && oracle.phase_jump_us > 0
                && oracle.phase_jump_us < oracle.period_after_us
        }
        PhaseMode::FrequencyShift => {
            oracle.shift_step.is_some()
                && oracle.period_after_us != oracle.period_before_us
                && oracle.phase_jump_us == 0
        }
        PhaseMode::Mixed => {
            oracle.shift_step.is_some()
                && oracle.period_after_us != oracle.period_before_us
                && oracle.phase_jump_us > 0
                && oracle.phase_jump_us < oracle.period_after_us
        }
    };
    if !expected_mode_valid {
        return Err(PhaseVerificationError::InvalidOracleMode);
    }

    if let Some(shift_step) = oracle.shift_step {
        let low = u32::try_from(task.slots / 3).unwrap_or(u32::MAX);
        let high = u32::try_from((task.slots * 2) / 3).unwrap_or(u32::MAX);
        if shift_step < low || shift_step > high {
            return Err(PhaseVerificationError::InvalidShiftSchedule);
        }
    }

    for index in 0..task.slots {
        let expected_step = u32::try_from(index).unwrap_or(u32::MAX);
        let expected_time = u64::try_from(index)
            .unwrap_or(u64::MAX)
            .saturating_mul(task.sample_interval_us);
        let observation = &task.observations[index];
        let point = &oracle.points[index];
        if observation.step != expected_step
            || observation.time_us != expected_time
            || point.step != expected_step
            || point.time_us != expected_time
        {
            return Err(PhaseVerificationError::InvalidObservation { expected_step });
        }

        let (phase_us, period_us) =
            phase_state_at(expected_time, oracle, task.sample_interval_us);
        let phase_permille = u16::try_from(
            (u128::from(phase_us) * 1_000) / u128::from(period_us),
        )
        .unwrap_or(0);
        let latent_sample = triangle_sample(phase_us, period_us);
        if point.period_us != period_us
            || point.phase_permille != phase_permille
            || point.latent_sample != latent_sample
        {
            return Err(PhaseVerificationError::InvalidPhasePoint { index });
        }
        let expected_crossing =
            next_rising_crossing(expected_time, oracle, task.sample_interval_us);
        if point.next_rising_crossing_us != expected_crossing
            || expected_crossing <= expected_time
        {
            return Err(PhaseVerificationError::InvalidCrossingTime { index });
        }

        let noise = i32::from(observation.sample) - i32::from(latent_sample);
        if !(-i32::from(MAX_NOISE)..=i32::from(MAX_NOISE)).contains(&noise) {
            return Err(PhaseVerificationError::NoiseBoundExceeded { step: expected_step });
        }
    }
    Ok(())
}

pub fn score_phase_predictions(
    task: &PhaseTask,
    oracle: &PhaseOracle,
    predictions: &[PhasePrediction],
) -> Result<PhaseScore, PhaseScoringError> {
    verify_phase_trace(task, oracle).map_err(PhaseScoringError::InvalidTrace)?;
    let mut score = PhaseScore {
        evaluated_points: predictions.len(),
        circular_phase_error_permille: Vec::with_capacity(predictions.len()),
        next_crossing_absolute_error_us: Vec::with_capacity(predictions.len()),
    };

    for (index, prediction) in predictions.iter().enumerate() {
        let step_index = usize::try_from(prediction.step).unwrap_or(usize::MAX);
        let Some(point) = oracle.points.get(step_index) else {
            return Err(PhaseScoringError::PredictionOutOfRange {
                step: prediction.step,
            });
        };
        if prediction.phase_permille >= 1_000 {
            return Err(PhaseScoringError::InvalidPredictedPhase {
                step: prediction.step,
                phase_permille: prediction.phase_permille,
            });
        }
        if index > 0 && predictions[index - 1].step >= prediction.step {
            return Err(PhaseScoringError::PredictionsNotStrictlyIncreasing { index });
        }
        let phase_diff = u16::abs_diff(prediction.phase_permille, point.phase_permille);
        score
            .circular_phase_error_permille
            .push(phase_diff.min(1_000 - phase_diff));
        score.next_crossing_absolute_error_us.push(
            prediction
                .next_rising_crossing_us
                .abs_diff(point.next_rising_crossing_us),
        );
    }
    Ok(score)
}

fn validate_config(config: PhaseConfig) -> Result<(), PhaseGenerationError> {
    if config.slots < 3 {
        return Err(PhaseGenerationError::TooFewSlots {
            requested: config.slots,
            minimum: 3,
        });
    }
    if config.slots > MAX_PHASE_SLOTS {
        return Err(PhaseGenerationError::TooManySlots {
            requested: config.slots,
            maximum: MAX_PHASE_SLOTS,
        });
    }
    if config.sample_interval_us == 0 {
        return Err(PhaseGenerationError::ZeroSampleInterval);
    }
    if config.base_period_us < MIN_PERIOD_US
        || config.base_period_us % MIN_PERIOD_US != 0
    {
        return Err(PhaseGenerationError::InvalidPeriod {
            period_us: config.base_period_us,
        });
    }
    let duration = u128::from(config.sample_interval_us)
        .saturating_mul(u128::try_from(config.slots).unwrap_or(u128::MAX));
    if duration > u128::from(MAX_PHASE_DURATION_US) {
        return Err(PhaseGenerationError::DurationExceedsLimit);
    }
    Ok(())
}

fn phase_stream_id(config: PhaseConfig) -> u64 {
    let mut value = config.seed ^ (config.slots as u64).rotate_left(11);
    value ^= config.sample_interval_us.rotate_left(21);
    value ^= config.base_period_us.rotate_left(37);
    value ^= match config.mode {
        PhaseMode::Stable => 0x5354_4142_4C45_0001,
        PhaseMode::PhaseShift => 0x5048_4153_455F_0002,
        PhaseMode::FrequencyShift => 0x4652_4551_5F00_0003,
        PhaseMode::Mixed => 0x4D49_5845_445F_0004,
    };
    mix64(value)
}

fn phase_state_at(
    time_us: u64,
    oracle: &PhaseOracle,
    sample_interval_us: u64,
) -> (u64, u64) {
    let Some(shift_step) = oracle.shift_step else {
        return (
            oracle.initial_phase_us.saturating_add(time_us) % oracle.period_before_us,
            oracle.period_before_us,
        );
    };
    let shift_time_us = u64::from(shift_step).saturating_mul(sample_interval_us);
    if time_us < shift_time_us {
        return (
            oracle.initial_phase_us.saturating_add(time_us) % oracle.period_before_us,
            oracle.period_before_us,
        );
    }

    let phase_at_boundary =
        oracle.initial_phase_us.saturating_add(shift_time_us) % oracle.period_before_us;
    let mapped_phase = u64::try_from(
        (u128::from(phase_at_boundary) * u128::from(oracle.period_after_us))
            / u128::from(oracle.period_before_us),
    )
    .unwrap_or(0);
    let elapsed_after = time_us.saturating_sub(shift_time_us);
    (
        mapped_phase
            .saturating_add(oracle.phase_jump_us)
            .saturating_add(elapsed_after)
            % oracle.period_after_us,
        oracle.period_after_us,
    )
}

fn next_rising_crossing(
    time_us: u64,
    oracle: &PhaseOracle,
    sample_interval_us: u64,
) -> u64 {
    let (phase_us, period_us) = phase_state_at(time_us, oracle, sample_interval_us);
    let delay = until_next_rising_crossing(phase_us, period_us);
    if let Some(shift_step) = oracle.shift_step {
        let shift_time_us = u64::from(shift_step).saturating_mul(sample_interval_us);
        if time_us < shift_time_us && time_us.saturating_add(delay) >= shift_time_us {
            let (post_phase, post_period) =
                phase_state_at(shift_time_us, oracle, sample_interval_us);
            return shift_time_us
                .saturating_add(until_next_rising_crossing(post_phase, post_period));
        }
    }
    time_us.saturating_add(delay)
}

fn until_next_rising_crossing(phase_us: u64, period_us: u64) -> u64 {
    let crossing = period_us / 4;
    if phase_us < crossing {
        crossing - phase_us
    } else {
        period_us - phase_us + crossing
    }
}

fn triangle_sample(phase_us: u64, period_us: u64) -> i16 {
    let half = period_us / 2;
    let value = if phase_us < half {
        -1_000_i64
            + i64::try_from((u128::from(phase_us) * 2_000) / u128::from(half))
                .unwrap_or(0)
    } else {
        1_000_i64
            - i64::try_from(
                (u128::from(phase_us - half) * 2_000) / u128::from(period_us - half),
            )
            .unwrap_or(0)
    };
    i16::try_from(value).unwrap_or(0)
}

fn nearest_rank<T: Copy + Ord>(
    values: &[T],
    numerator: usize,
    denominator: usize,
) -> Option<T> {
    if values.is_empty()
        || denominator == 0
        || numerator == 0
        || numerator > denominator
    {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = sorted
        .len()
        .saturating_mul(numerator)
        .saturating_add(denominator - 1)
        / denominator;
    sorted.get(rank.saturating_sub(1)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_reference_has_no_hidden_shift() {
        let config = PhaseConfig::reference(11, PhaseMode::Stable);
        let (task, oracle) = generate_phase_task(config).unwrap();
        assert_eq!(oracle.shift_step, None);
        assert_eq!(oracle.period_before_us, oracle.period_after_us);
        assert_eq!(verify_phase_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn phase_shift_changes_phase_without_changing_period() {
        let config = PhaseConfig::reference(12, PhaseMode::PhaseShift);
        let (task, oracle) = generate_phase_task(config).unwrap();
        assert!(oracle.shift_step.is_some());
        assert_eq!(oracle.period_before_us, oracle.period_after_us);
        assert!(oracle.phase_jump_us > 0);
        assert_eq!(verify_phase_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn frequency_shift_preserves_fractional_cycle_phase_at_transition() {
        let config = PhaseConfig::reference(13, PhaseMode::FrequencyShift);
        let (_, oracle) = generate_phase_task(config).unwrap();
        let shift = oracle.shift_step.unwrap();
        let shift_time = u64::from(shift) * config.sample_interval_us;
        let phase_before = oracle.initial_phase_us + shift_time;
        let phase_before = phase_before % oracle.period_before_us;
        let fractional_before = (u128::from(phase_before) * 1_000_000)
            / u128::from(oracle.period_before_us);
        let (phase_after, period_after) =
            phase_state_at(shift_time, &oracle, config.sample_interval_us);
        let fractional_after = (u128::from(phase_after) * 1_000_000)
            / u128::from(period_after);
        assert!(fractional_before.abs_diff(fractional_after) <= 1);
    }

    #[test]
    fn mixed_mode_changes_both_phase_and_frequency() {
        let config = PhaseConfig::reference(14, PhaseMode::Mixed);
        let (task, oracle) = generate_phase_task(config).unwrap();
        assert_ne!(oracle.period_before_us, oracle.period_after_us);
        assert!(oracle.phase_jump_us > 0);
        assert_eq!(verify_phase_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let config = PhaseConfig::reference(15, PhaseMode::Mixed);
        assert_eq!(generate_phase_task(config), generate_phase_task(config));
    }

    #[test]
    fn public_observations_are_timestamped_and_monotonic() {
        let config = PhaseConfig::reference(16, PhaseMode::Stable);
        let (task, _) = generate_phase_task(config).unwrap();
        for (index, observation) in task.observations.iter().enumerate() {
            assert_eq!(observation.step as usize, index);
            assert_eq!(
                observation.time_us,
                index as u64 * config.sample_interval_us
            );
        }
    }

    #[test]
    fn phase_and_crossing_digests_are_bound_to_oracle_points() {
        let config = PhaseConfig::reference(17, PhaseMode::Mixed);
        let (_, oracle) = generate_phase_task(config).unwrap();
        assert_eq!(oracle.content_sha256, phase_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_public_sample_mutation() {
        let config = PhaseConfig::reference(18, PhaseMode::Stable);
        let (mut task, oracle) = generate_phase_task(config).unwrap();
        task.observations[0].sample ^= 1;
        assert_eq!(
            verify_phase_trace(&task, &oracle),
            Err(PhaseVerificationError::PublicDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_recommitted_wrong_latent_sample() {
        let config = PhaseConfig::reference(19, PhaseMode::Mixed);
        let (task, mut oracle) = generate_phase_task(config).unwrap();
        oracle.points[0].latent_sample ^= 1;
        oracle.content_sha256 = phase_oracle_digest(&oracle);
        assert_eq!(
            verify_phase_trace(&task, &oracle),
            Err(PhaseVerificationError::InvalidPhasePoint { index: 0 })
        );
    }

    #[test]
    fn verifier_rejects_recommitted_wrong_crossing_time() {
        let config = PhaseConfig::reference(20, PhaseMode::Mixed);
        let (task, mut oracle) = generate_phase_task(config).unwrap();
        oracle.points[0].next_rising_crossing_us =
            oracle.points[0].next_rising_crossing_us.saturating_add(1);
        oracle.content_sha256 = phase_oracle_digest(&oracle);
        assert_eq!(
            verify_phase_trace(&task, &oracle),
            Err(PhaseVerificationError::InvalidCrossingTime { index: 0 })
        );
    }

    #[test]
    fn phase_change_does_not_rewrite_pre_change_samples() {
        let config = PhaseConfig::reference(21, PhaseMode::Mixed);
        let (_, oracle) = generate_phase_task(config).unwrap();
        let shift = oracle.shift_step.unwrap() as usize;
        for point in oracle.points.iter().take(shift) {
            let (phase, period) = phase_state_at(
                point.time_us,
                &oracle,
                config.sample_interval_us,
            );
            assert_eq!(period, oracle.period_before_us);
            assert_eq!(point.latent_sample, triangle_sample(phase, period));
        }
    }

    #[test]
    fn scorer_gives_zero_error_for_oracle_predictions() {
        let config = PhaseConfig::reference(22, PhaseMode::Mixed);
        let (task, oracle) = generate_phase_task(config).unwrap();
        let predictions: Vec<_> = oracle.points.iter().step_by(17).map(|point| {
            PhasePrediction {
                step: point.step,
                phase_permille: point.phase_permille,
                next_rising_crossing_us: point.next_rising_crossing_us,
            }
        }).collect();
        let score = score_phase_predictions(&task, &oracle, &predictions).unwrap();
        assert!(score.circular_phase_error_permille.iter().all(|error| *error == 0));
        assert!(score.next_crossing_absolute_error_us.iter().all(|error| *error == 0));
    }

    #[test]
    fn scorer_rejects_duplicate_or_reordered_predictions() {
        let config = PhaseConfig::reference(23, PhaseMode::Stable);
        let (task, oracle) = generate_phase_task(config).unwrap();
        let predictions = [
            PhasePrediction { step: 10, phase_permille: 0, next_rising_crossing_us: 0 },
            PhasePrediction { step: 9, phase_permille: 0, next_rising_crossing_us: 0 },
        ];
        assert!(matches!(
            score_phase_predictions(&task, &oracle, &predictions),
            Err(PhaseScoringError::PredictionsNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn scorer_rejects_invalid_phase_fraction() {
        let config = PhaseConfig::reference(24, PhaseMode::Stable);
        let (task, oracle) = generate_phase_task(config).unwrap();
        let predictions = [
            PhasePrediction {
                step: 0,
                phase_permille: 1_000,
                next_rising_crossing_us: 0,
            },
        ];
        assert!(matches!(
            score_phase_predictions(&task, &oracle, &predictions),
            Err(PhaseScoringError::InvalidPredictedPhase { .. })
        ));
    }

    #[test]
    fn invalid_period_is_rejected() {
        let mut config = PhaseConfig::reference(1, PhaseMode::Stable);
        config.base_period_us = 500;
        assert!(matches!(
            generate_phase_task(config),
            Err(PhaseGenerationError::InvalidPeriod { .. })
        ));
    }

    #[test]
    fn excessive_duration_is_rejected() {
        let mut config = PhaseConfig::reference(1, PhaseMode::Stable);
        config.sample_interval_us = MAX_PHASE_DURATION_US;
        assert!(matches!(
            generate_phase_task(config),
            Err(PhaseGenerationError::DurationExceedsLimit)
        ));
    }

    #[test]
    fn shift_modes_reject_invalid_hidden_mode_combinations() {
        let config = PhaseConfig::reference(25, PhaseMode::PhaseShift);
        let (task, mut oracle) = generate_phase_task(config).unwrap();
        oracle.phase_jump_us = 0;
        oracle.content_sha256 = phase_oracle_digest(&oracle);
        assert_eq!(
            verify_phase_trace(&task, &oracle),
            Err(PhaseVerificationError::InvalidOracleMode)
        );
    }
}
