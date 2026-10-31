//! Deterministic always-on sparse-event detection workload.
//!
//! The candidate sees a continuous noisy scalar sensor stream with rare target
//! signatures and similar non-target transients. The oracle records event
//! locations, classes, codes, and amplitudes. This is a synthetic sensor/event
//! task, not a reproduction of MLPerf Tiny's audio wake-word workload.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::collections::BTreeSet;
use std::fmt;

const MAX_EVENT_SLOTS: usize = 1_000_000;
const BLOCK_SIZE: usize = 64;
const EVENT_WIDTH: usize = 8;
const MAX_NOISE: i16 = 40;
const MIN_AMPLITUDE: i16 = 300;
const MAX_AMPLITUDE: i16 = 600;
const TARGET_CODE: u8 = 0b1010_0110;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventDistractorSimilarity {
    /// Non-target motifs differ from the target code in 1-3 of 8 positions.
    High,
    /// Non-target motifs differ from the target code in 4-7 of 8 positions.
    Low,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FastEventConfig {
    pub seed: u64,
    pub slots: usize,
    pub sample_interval_us: u64,
    pub target_event_count: usize,
    pub distractor_event_count: usize,
    pub similarity: EventDistractorSimilarity,
}

impl FastEventConfig {
    /// Reference stream: about 16.4 seconds of always-on samples and rare events.
    pub fn reference(seed: u64) -> Self {
        Self {
            seed,
            slots: 16_384,
            sample_interval_us: 1_000,
            target_event_count: 8,
            distractor_event_count: 16,
            similarity: EventDistractorSimilarity::High,
        }
    }

    /// Negative control: continuous quiet-idle stream with no target/distractor events.
    pub fn quiet_idle(seed: u64) -> Self {
        Self {
            target_event_count: 0,
            distractor_event_count: 0,
            ..Self::reference(seed)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastEventObservation {
    pub step: u32,
    pub time_us: u64,
    pub sample: i16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastEventTask {
    pub stream_id: u64,
    pub sample_interval_us: u64,
    pub slots: usize,
    pub similarity: EventDistractorSimilarity,
    /// Maximum permitted decision lag after the final event sample.
    pub response_deadline_steps: u32,
    pub observations: Vec<FastEventObservation>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastEventTruth {
    pub start_step: u32,
    pub end_step: u32,
    pub is_target: bool,
    pub code: u8,
    pub amplitude: i16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastEventOracle {
    pub stream_id: u64,
    pub target_event_count: usize,
    pub distractor_event_count: usize,
    pub events: Vec<FastEventTruth>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FastEventGenerationError {
    TooFewSlots { requested: usize, minimum: usize },
    TooManySlots { requested: usize, maximum: usize },
    ZeroSampleInterval,
    DurationExceedsLimit,
    TooManyEvents { requested: usize, available_blocks: usize },
    TooManyDistinctDistractorCodes { requested: usize, maximum: usize },
}

impl fmt::Display for FastEventGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSlots { requested, minimum } => {
                write!(f, "slots {requested} is below minimum {minimum}")
            }
            Self::TooManySlots { requested, maximum } => {
                write!(f, "slots {requested} exceeds maximum {maximum}")
            }
            Self::ZeroSampleInterval => write!(f, "sample_interval_us must be greater than zero"),
            Self::DurationExceedsLimit => write!(f, "fast-event stream duration exceeds one day"),
            Self::TooManyEvents { requested, available_blocks } => write!(
                f,
                "requested {requested} events but only {available_blocks} non-overlapping event blocks exist"
            ),
            Self::TooManyDistinctDistractorCodes { requested, maximum } => write!(
                f,
                "requested {requested} distinct distractor codes but only {maximum} exist in the selected similarity band"
            ),
        }
    }
}

impl std::error::Error for FastEventGenerationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FastEventVerificationError {
    StreamIdMismatch,
    PublicDigestMismatch,
    OracleDigestMismatch,
    InvalidTaskContract,
    InvalidEventCount,
    InvalidEventSchedule { index: usize },
    InvalidEventCode { index: usize },
    InvalidObservation { expected_step: u32 },
    SignalMismatch { step: u32 },
    NoiseBoundExceeded { step: u32 },
}

impl fmt::Display for FastEventVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "fast-event task and oracle stream IDs differ"),
            Self::PublicDigestMismatch => write!(f, "fast-event public stream SHA-256 digest mismatch"),
            Self::OracleDigestMismatch => write!(f, "fast-event oracle SHA-256 digest mismatch"),
            Self::InvalidTaskContract => write!(f, "fast-event timing or task metadata is invalid"),
            Self::InvalidEventCount => write!(f, "fast-event oracle event counts are inconsistent"),
            Self::InvalidEventSchedule { index } => {
                write!(f, "fast event {index} violates the non-overlap or block schedule")
            }
            Self::InvalidEventCode { index } => {
                write!(f, "fast event {index} has a code outside the declared similarity band")
            }
            Self::InvalidObservation { expected_step } => {
                write!(f, "fast-event observation is missing or reordered at step {expected_step}")
            }
            Self::SignalMismatch { step } => {
                write!(f, "fast-event sample at step {step} is inconsistent with the oracle motif")
            }
            Self::NoiseBoundExceeded { step } => {
                write!(f, "fast-event noise at step {step} exceeds the declared bound")
            }
        }
    }
}

impl std::error::Error for FastEventVerificationError {}

/// Generate a continuous scalar stream with rare target and distractor motifs.
///
/// Event windows are placed in distinct 64-sample blocks. The candidate only
/// receives noisy samples and timestamps; event boundaries, target/distractor
/// labels, codes, and amplitudes remain in the oracle record.
pub fn generate_fast_event_task(
    config: FastEventConfig,
) -> Result<(FastEventTask, FastEventOracle), FastEventGenerationError> {
    validate_config(config)?;

    let stream_id = fast_event_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x4641_5354_4556_454E);
    let block_count = config.slots / BLOCK_SIZE;
    let event_count = config.target_event_count + config.distractor_event_count;

    let mut blocks: Vec<usize> = (0..block_count).collect();
    for index in (1..blocks.len()).rev() {
        let other = usize::try_from(rng.next_u64() % u64::try_from(index + 1).unwrap_or(u64::MAX))
            .unwrap_or(0);
        blocks.swap(index, other);
    }

    let mut events = Vec::with_capacity(event_count);
    let mut used_codes = BTreeSet::new();
    used_codes.insert(TARGET_CODE);

    for event_index in 0..event_count {
        let block = blocks[event_index];
        let offset = usize::try_from(rng.next_u64() % (BLOCK_SIZE - EVENT_WIDTH + 1) as u64)
            .unwrap_or(0);
        let start = block * BLOCK_SIZE + offset;
        let is_target = event_index < config.target_event_count;
        let code = if is_target {
            TARGET_CODE
        } else {
            unique_distractor_code(&mut rng, config.similarity, &mut used_codes)
        };
        let amplitude = MIN_AMPLITUDE
            + i16::try_from(rng.next_u64() % u64::try_from(MAX_AMPLITUDE - MIN_AMPLITUDE + 1).unwrap_or(1))
                .unwrap_or(0);
        events.push(FastEventTruth {
            start_step: u32::try_from(start).unwrap_or(u32::MAX),
            end_step: u32::try_from(start + EVENT_WIDTH).unwrap_or(u32::MAX),
            is_target,
            code,
            amplitude,
        });
    }
    events.sort_by_key(|event| event.start_step);

    let mut observations = Vec::with_capacity(config.slots);
    let mut event_index = 0_usize;
    for step in 0..config.slots {
        while events
            .get(event_index)
            .is_some_and(|event| u32::try_from(step).unwrap_or(u32::MAX) >= event.end_step)
        {
            event_index = event_index.saturating_add(1);
        }
        let noise = i16::try_from(rng.next_u64() % u64::try_from(MAX_NOISE * 2 + 1).unwrap_or(1))
            .unwrap_or(0)
            - MAX_NOISE;
        let base = events.get(event_index).and_then(|event| {
            let step_u32 = u32::try_from(step).unwrap_or(u32::MAX);
            if step_u32 >= event.start_step && step_u32 < event.end_step {
                let position = usize::try_from(step_u32 - event.start_step).unwrap_or(0);
                let bit = (event.code >> (EVENT_WIDTH - 1 - position)) & 1;
                Some(if bit == 1 { event.amplitude } else { -event.amplitude })
            } else {
                None
            }
        }).unwrap_or(0);
        let sample = base.saturating_add(noise);
        observations.push(FastEventObservation {
            step: u32::try_from(step).unwrap_or(u32::MAX),
            time_us: u64::try_from(step).unwrap_or(u64::MAX)
                .saturating_mul(config.sample_interval_us),
            sample,
        });
    }

    let mut task = FastEventTask {
        stream_id,
        sample_interval_us: config.sample_interval_us,
        slots: config.slots,
        similarity: config.similarity,
        response_deadline_steps: u32::try_from(EVENT_WIDTH).unwrap_or(u32::MAX),
        observations,
        content_sha256: [0; 32],
    };
    task.content_sha256 = fast_event_content_digest(&task);

    let mut oracle = FastEventOracle {
        stream_id,
        target_event_count: config.target_event_count,
        distractor_event_count: config.distractor_event_count,
        events,
        content_sha256: [0; 32],
    };
    oracle.content_sha256 = fast_event_oracle_digest(&oracle);

    Ok((task, oracle))
}

pub fn fast_event_content_digest(task: &FastEventTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.fast-event.public.v1");
    hash.u64(task.stream_id);
    hash.u64(task.sample_interval_us);
    hash.usize(task.slots);
    hash.u8(match task.similarity {
        EventDistractorSimilarity::High => 0,
        EventDistractorSimilarity::Low => 1,
    });
    hash.u32(task.response_deadline_steps);
    hash.usize(task.observations.len());
    for observation in &task.observations {
        hash.u32(observation.step);
        hash.u64(observation.time_us);
        hash.u16(observation.sample as u16);
    }
    hash.finish()
}

pub fn fast_event_oracle_digest(oracle: &FastEventOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.fast-event.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.usize(oracle.target_event_count);
    hash.usize(oracle.distractor_event_count);
    hash.usize(oracle.events.len());
    for event in &oracle.events {
        hash.u32(event.start_step);
        hash.u32(event.end_step);
        hash.u8(u8::from(event.is_target));
        hash.u8(event.code);
        hash.u16(event.amplitude as u16);
    }
    hash.finish()
}

/// Verify timing, event code similarity, sample-to-oracle consistency, and hashes.
pub fn verify_fast_event_trace(
    task: &FastEventTask,
    oracle: &FastEventOracle,
) -> Result<(), FastEventVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(FastEventVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != fast_event_content_digest(task) {
        return Err(FastEventVerificationError::PublicDigestMismatch);
    }
    if oracle.content_sha256 != fast_event_oracle_digest(oracle) {
        return Err(FastEventVerificationError::OracleDigestMismatch);
    }
    if task.sample_interval_us == 0
        || task.slots < BLOCK_SIZE
        || task.slots > MAX_EVENT_SLOTS
        || task.observations.len() != task.slots
        || task.response_deadline_steps != u32::try_from(EVENT_WIDTH).unwrap_or(u32::MAX)
        || task.sample_interval_us.saturating_mul(task.slots as u64) > 86_400_000_000
    {
        return Err(FastEventVerificationError::InvalidTaskContract);
    }
    let target_count = oracle.events.iter().filter(|event| event.is_target).count();
    let distractor_count = oracle.events.iter().filter(|event| !event.is_target).count();
    if target_count != oracle.target_event_count
        || distractor_count != oracle.distractor_event_count
        || oracle.events.len() != target_count.saturating_add(distractor_count)
        || oracle.events.len() > task.slots / BLOCK_SIZE
    {
        return Err(FastEventVerificationError::InvalidEventCount);
    }

    let mut previous_end = 0_u32;
    let mut used_blocks = BTreeSet::new();
    let mut used_distractor_codes = BTreeSet::new();
    for (index, event) in oracle.events.iter().enumerate() {
        let start = usize::try_from(event.start_step).unwrap_or(usize::MAX);
        let end = usize::try_from(event.end_step).unwrap_or(usize::MAX);
        if end != start.saturating_add(EVENT_WIDTH)
            || end > task.slots
            || (index > 0 && event.start_step < previous_end)
            || !used_blocks.insert(start / BLOCK_SIZE)
            || start % BLOCK_SIZE + EVENT_WIDTH > BLOCK_SIZE
            || event.amplitude < MIN_AMPLITUDE
            || event.amplitude > MAX_AMPLITUDE
        {
            return Err(FastEventVerificationError::InvalidEventSchedule { index });
        }
        let distance = (event.code ^ TARGET_CODE).count_ones();
        let code_valid = if event.is_target {
            event.code == TARGET_CODE
        } else {
            match task.similarity {
                EventDistractorSimilarity::High => (1..=3).contains(&distance),
                EventDistractorSimilarity::Low => (4..=7).contains(&distance),
            }
        };
        if !code_valid
            || (!event.is_target && !used_distractor_codes.insert(event.code))
        {
            return Err(FastEventVerificationError::InvalidEventCode { index });
        }
        previous_end = event.end_step;
    }

    let mut event_index = 0_usize;
    for (index, observation) in task.observations.iter().enumerate() {
        let expected_step = u32::try_from(index).unwrap_or(u32::MAX);
        if observation.step != expected_step
            || observation.time_us
                != u64::try_from(index).unwrap_or(u64::MAX)
                    .saturating_mul(task.sample_interval_us)
        {
            return Err(FastEventVerificationError::InvalidObservation { expected_step });
        }
        while oracle.events.get(event_index).is_some_and(|event| expected_step >= event.end_step) {
            event_index = event_index.saturating_add(1);
        }
        let base = oracle.events.get(event_index).and_then(|event| {
            if expected_step >= event.start_step && expected_step < event.end_step {
                let position = usize::try_from(expected_step - event.start_step).unwrap_or(0);
                let bit = (event.code >> (EVENT_WIDTH - 1 - position)) & 1;
                Some(if bit == 1 { event.amplitude } else { -event.amplitude })
            } else {
                None
            }
        }).unwrap_or(0);
        let noise = i32::from(observation.sample) - i32::from(base);
        if !(-i32::from(MAX_NOISE)..=i32::from(MAX_NOISE)).contains(&noise) {
            return Err(FastEventVerificationError::NoiseBoundExceeded { step: expected_step });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FastEventPrediction {
    /// Candidate decision time in sample steps. One target activation is one prediction.
    pub step: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastEventScore {
    pub true_positives: usize,
    pub false_negatives: usize,
    pub false_positive_idle: usize,
    pub false_positive_distractor: usize,
    pub duplicate_target_activations: usize,
    pub quiet_exposure_us: u64,
    /// One latency for each matched target event, in microseconds.
    pub detection_latency_us: Vec<u64>,
}

impl FastEventScore {
    pub fn false_positives(&self) -> usize {
        self.false_positive_idle
            .saturating_add(self.false_positive_distractor)
            .saturating_add(self.duplicate_target_activations)
    }

    pub fn precision(&self) -> Option<f64> {
        let predicted = self.true_positives.saturating_add(self.false_positives());
        (predicted > 0).then_some(self.true_positives as f64 / predicted as f64)
    }

    pub fn recall(&self) -> Option<f64> {
        let actual = self.true_positives.saturating_add(self.false_negatives);
        (actual > 0).then_some(self.true_positives as f64 / actual as f64)
    }

    /// Quiet-period false activations normalized to one monitored hour.
    pub fn idle_false_triggers_per_hour(&self) -> Option<f64> {
        (self.quiet_exposure_us > 0).then_some(
            self.false_positive_idle as f64 * 3_600_000_000.0
                / self.quiet_exposure_us as f64,
        )
    }

    /// Nearest-rank latency percentile, with numerator/denominator in [0, 1].
    pub fn latency_percentile_us(
        &self,
        numerator: usize,
        denominator: usize,
    ) -> Option<u64> {
        if self.detection_latency_us.is_empty()
            || denominator == 0
            || numerator == 0
            || numerator > denominator
        {
            return None;
        }
        let mut sorted = self.detection_latency_us.clone();
        sorted.sort_unstable();
        let rank = sorted
            .len()
            .saturating_mul(numerator)
            .saturating_add(denominator - 1)
            / denominator;
        sorted.get(rank.saturating_sub(1)).copied()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FastEventScoringError {
    InvalidTrace(FastEventVerificationError),
    PredictionOutOfRange { step: u32 },
    PredictionsNotStrictlyIncreasing { index: usize },
}

impl fmt::Display for FastEventScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTrace(error) => write!(f, "cannot score invalid task: {error}"),
            Self::PredictionOutOfRange { step } => {
                write!(f, "candidate prediction at step {step} is outside the stream")
            }
            Self::PredictionsNotStrictlyIncreasing { index } => {
                write!(f, "candidate predictions are duplicated or reordered at index {index}")
            }
        }
    }
}

impl std::error::Error for FastEventScoringError {}

/// Score event decisions with one-to-one target matching and explicit idle/distractor false alarms.
///
/// A target can be detected from its first sample through its exclusive end
/// boundary (the decision may occur immediately after consuming the final
/// motif sample). Extra activations for that same event are duplicates, not
/// additional true positives. Detections during distractors and quiet periods
/// are reported separately. Quiet exposure excludes every target and
/// distractor motif interval; no rate extrapolation is presented without its
/// exact monitored exposure.
pub fn score_fast_event_predictions(
    task: &FastEventTask,
    oracle: &FastEventOracle,
    predictions: &[FastEventPrediction],
) -> Result<FastEventScore, FastEventScoringError> {
    verify_fast_event_trace(task, oracle)
        .map_err(FastEventScoringError::InvalidTrace)?;

    let mut matched_targets = vec![false; oracle.events.len()];
    let mut score = FastEventScore {
        true_positives: 0,
        false_negatives: oracle.target_event_count,
        false_positive_idle: 0,
        false_positive_distractor: 0,
        duplicate_target_activations: 0,
        quiet_exposure_us: u64::try_from(task.slots)
            .unwrap_or(u64::MAX)
            .saturating_mul(task.sample_interval_us),
        detection_latency_us: Vec::new(),
    };

    for (index, prediction) in predictions.iter().enumerate() {
        if u64::from(prediction.step) > u64::try_from(task.slots).unwrap_or(u64::MAX) {
            return Err(FastEventScoringError::PredictionOutOfRange {
                step: prediction.step,
            });
        }
        if index > 0 && predictions[index - 1].step >= prediction.step {
            return Err(FastEventScoringError::PredictionsNotStrictlyIncreasing {
                index,
            });
        }

        let mut matched_target = false;
        let mut in_distractor = false;
        for (event_index, event) in oracle.events.iter().enumerate() {
            let within_target_window = event.is_target
                && prediction.step >= event.start_step
                && prediction.step
                    <= event.start_step.saturating_add(task.response_deadline_steps);
            if within_target_window {
                if matched_targets[event_index] {
                    score.duplicate_target_activations =
                        score.duplicate_target_activations.saturating_add(1);
                } else {
                    matched_targets[event_index] = true;
                    score.true_positives = score.true_positives.saturating_add(1);
                    score.false_negatives = score.false_negatives.saturating_sub(1);
                    let latency_steps = prediction.step.saturating_sub(event.start_step);
                    score.detection_latency_us.push(
                        u64::from(latency_steps).saturating_mul(task.sample_interval_us),
                    );
                }
                matched_target = true;
                break;
            }
            if !event.is_target
                && prediction.step >= event.start_step
                && prediction.step < event.end_step
            {
                in_distractor = true;
            }
        }

        if !matched_target {
            if in_distractor {
                score.false_positive_distractor =
                    score.false_positive_distractor.saturating_add(1);
            } else {
                score.false_positive_idle = score.false_positive_idle.saturating_add(1);
            }
        }
    }

    let event_slots: u64 = oracle.events
        .iter()
        .map(|event| {
            u64::from(event.end_step.saturating_sub(event.start_step))
                .saturating_add(u64::from(event.is_target))
        })
        .sum();
    let total_slots = u64::try_from(task.slots).unwrap_or(u64::MAX);
    score.quiet_exposure_us = total_slots
        .saturating_sub(event_slots)
        .saturating_mul(task.sample_interval_us);

    Ok(score)
}

fn validate_config(config: FastEventConfig) -> Result<(), FastEventGenerationError> {
    if config.slots < BLOCK_SIZE {
        return Err(FastEventGenerationError::TooFewSlots {
            requested: config.slots,
            minimum: BLOCK_SIZE,
        });
    }
    if config.slots > MAX_EVENT_SLOTS {
        return Err(FastEventGenerationError::TooManySlots {
            requested: config.slots,
            maximum: MAX_EVENT_SLOTS,
        });
    }
    if config.sample_interval_us == 0 {
        return Err(FastEventGenerationError::ZeroSampleInterval);
    }
    let duration = u128::from(config.sample_interval_us)
        .saturating_mul(u128::try_from(config.slots).unwrap_or(u128::MAX));
    if duration > 86_400_000_000_u128 {
        return Err(FastEventGenerationError::DurationExceedsLimit);
    }
    let available_blocks = config.slots / BLOCK_SIZE;
    let event_count = config.target_event_count.saturating_add(config.distractor_event_count);
    if event_count > available_blocks {
        return Err(FastEventGenerationError::TooManyEvents {
            requested: event_count,
            available_blocks,
        });
    }
    let maximum_distinct_codes = match config.similarity {
        EventDistractorSimilarity::High => 92,
        EventDistractorSimilarity::Low => 162,
    };
    if config.distractor_event_count > maximum_distinct_codes {
        return Err(FastEventGenerationError::TooManyDistinctDistractorCodes {
            requested: config.distractor_event_count,
            maximum: maximum_distinct_codes,
        });
    }
    Ok(())
}

fn fast_event_stream_id(config: FastEventConfig) -> u64 {
    let mut value = config.seed ^ (config.slots as u64).rotate_left(9);
    value ^= config.sample_interval_us.rotate_left(21);
    value ^= (config.target_event_count as u64).rotate_left(33);
    value ^= (config.distractor_event_count as u64).rotate_left(45);
    value ^= match config.similarity {
        EventDistractorSimilarity::High => 0x4849_4748_5F45_564E,
        EventDistractorSimilarity::Low => 0x4C4F_575F_4556_454E,
    };
    mix64(value)
}

fn unique_distractor_code(
    rng: &mut SplitMix64,
    similarity: EventDistractorSimilarity,
    used: &mut BTreeSet<u8>,
) -> u8 {
    loop {
        let distance = match similarity {
            EventDistractorSimilarity::High => 1 + (rng.next_u64() % 3),
            EventDistractorSimilarity::Low => 4 + (rng.next_u64() % 4),
        };
        let mask = random_mask8(rng, distance);
        let code = TARGET_CODE ^ mask;
        if used.insert(code) {
            return code;
        }
    }
}

fn random_mask8(rng: &mut SplitMix64, bit_count: u64) -> u8 {
    let mut mask = 0_u8;
    while mask.count_ones() < bit_count as u32 {
        mask |= 1_u8 << (rng.next_u64() % 8);
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_config_generates_a_long_stream_with_rare_events() {
        let config = FastEventConfig::reference(17);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        assert_eq!(task.observations.len(), 16_384);
        assert_eq!(oracle.target_event_count, 8);
        assert_eq!(oracle.distractor_event_count, 16);
        assert_eq!(oracle.events.len(), 24);
    }

    #[test]
    fn quiet_idle_control_contains_no_events() {
        let (task, oracle) = generate_fast_event_task(FastEventConfig::quiet_idle(18)).unwrap();
        assert!(oracle.events.is_empty());
        assert!(task.observations.iter().all(|row| row.sample.abs() <= MAX_NOISE));
    }

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let config = FastEventConfig::reference(19);
        assert_eq!(generate_fast_event_task(config), generate_fast_event_task(config));
    }

    #[test]
    fn generated_stream_and_oracle_pass_the_verifier() {
        let config = FastEventConfig::reference(20);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        assert_eq!(verify_fast_event_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn distractor_codes_obey_declared_similarity_band() {
        for similarity in [EventDistractorSimilarity::High, EventDistractorSimilarity::Low] {
            let mut config = FastEventConfig::reference(21);
            config.similarity = similarity;
            let (_, oracle) = generate_fast_event_task(config).unwrap();
            for event in oracle.events.iter().filter(|event| !event.is_target) {
                let distance = (event.code ^ TARGET_CODE).count_ones();
                match similarity {
                    EventDistractorSimilarity::High => assert!((1..=3).contains(&distance)),
                    EventDistractorSimilarity::Low => assert!((4..=7).contains(&distance)),
                }
            }
        }
    }

    #[test]
    fn events_never_overlap_or_share_a_block() {
        let config = FastEventConfig::reference(22);
        let (_, oracle) = generate_fast_event_task(config).unwrap();
        for pair in oracle.events.windows(2) {
            assert!(pair[1].start_step >= pair[0].end_step);
            assert_ne!(pair[0].start_step / BLOCK_SIZE as u32, pair[1].start_step / BLOCK_SIZE as u32);
        }
    }

    #[test]
    fn canonical_digests_cover_public_and_oracle_content() {
        let config = FastEventConfig::reference(23);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        assert_eq!(task.content_sha256, fast_event_content_digest(&task));
        assert_eq!(oracle.content_sha256, fast_event_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_public_stream_mutation() {
        let config = FastEventConfig::reference(24);
        let (mut task, oracle) = generate_fast_event_task(config).unwrap();
        task.observations[0].sample ^= 1;
        assert_eq!(
            verify_fast_event_trace(&task, &oracle),
            Err(FastEventVerificationError::PublicDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_oracle_mutation() {
        let config = FastEventConfig::reference(25);
        let (task, mut oracle) = generate_fast_event_task(config).unwrap();
        oracle.events[0].is_target = !oracle.events[0].is_target;
        assert_eq!(
            verify_fast_event_trace(&task, &oracle),
            Err(FastEventVerificationError::OracleDigestMismatch)
        );
    }

    #[test]
    fn perfect_detector_scores_all_targets_with_latency() {
        let config = FastEventConfig::reference(31);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        let predictions: Vec<_> = oracle
            .events
            .iter()
            .filter(|event| event.is_target)
            .map(|event| FastEventPrediction { step: event.end_step })
            .collect();
        let score = score_fast_event_predictions(&task, &oracle, &predictions).unwrap();
        assert_eq!(score.true_positives, config.target_event_count);
        assert_eq!(score.false_negatives, 0);
        assert_eq!(score.false_positives(), 0);
        assert_eq!(
            score.latency_percentile_us(99, 100),
            Some(u64::from(EVENT_WIDTH as u32) * config.sample_interval_us)
        );
    }

    #[test]
    fn quiet_idle_predictions_are_counted_as_false_triggers() {
        let config = FastEventConfig::quiet_idle(32);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        let predictions = [FastEventPrediction { step: 12 }];
        let score = score_fast_event_predictions(&task, &oracle, &predictions).unwrap();
        assert_eq!(score.false_positive_idle, 1);
        assert_eq!(score.true_positives, 0);
        assert_eq!(score.idle_false_triggers_per_hour(), Some(1.0 * 3_600_000_000.0 / score.quiet_exposure_us as f64));
    }

    #[test]
    fn distractor_detections_are_not_counted_as_idle_false_triggers() {
        let config = FastEventConfig::reference(33);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        let event = oracle.events.iter().find(|event| !event.is_target).unwrap();
        let score = score_fast_event_predictions(
            &task,
            &oracle,
            &[FastEventPrediction { step: event.start_step }],
        ).unwrap();
        assert_eq!(score.false_positive_distractor, 1);
        assert_eq!(score.false_positive_idle, 0);
    }

    #[test]
    fn second_activation_for_a_target_is_a_duplicate_not_an_extra_hit() {
        let config = FastEventConfig::reference(34);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        let event = oracle.events.iter().find(|event| event.is_target).unwrap();
        let predictions = [
            FastEventPrediction { step: event.start_step },
            FastEventPrediction { step: event.end_step },
        ];
        let score = score_fast_event_predictions(&task, &oracle, &predictions).unwrap();
        assert_eq!(score.true_positives, 1);
        assert_eq!(score.duplicate_target_activations, 1);
    }

    #[test]
    fn non_monotonic_candidate_predictions_are_rejected() {
        let config = FastEventConfig.reference(35);
        let (task, oracle) = generate_fast_event_task(config).unwrap();
        let predictions = [
            FastEventPrediction { step: 10 },
            FastEventPrediction { step: 9 },
        ];
        assert!(matches!(
            score_fast_event_predictions(&task, &oracle, &predictions),
            Err(FastEventScoringError::PredictionsNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn excessive_distinct_distractor_codes_are_rejected() {
        let mut config = FastEventConfig::reference(1);
        config.distractor_event_count = 93;
        assert!(matches!(
            generate_fast_event_task(config),
            Err(FastEventGenerationError::TooManyDistinctDistractorCodes {
                requested: 93,
                maximum: 92
            })
        ));
    }

    #[test]
    fn invalid_slot_count_is_rejected() {
        let mut config = FastEventConfig::reference(1);
        config.slots = 32;
        assert!(matches!(
            generate_fast_event_task(config),
            Err(FastEventGenerationError::TooFewSlots { .. })
        ));
    }

    #[test]
    fn excessive_event_count_is_rejected() {
        let mut config = FastEventConfig::reference(1);
        config.target_event_count = config.slots / BLOCK_SIZE + 1;
        assert!(matches!(
            generate_fast_event_task(config),
            Err(FastEventGenerationError::TooManyEvents { .. })
        ));
    }

    #[test]
    fn zero_sample_interval_is_rejected() {
        let mut config = FastEventConfig::reference(1);
        config.sample_interval_us = 0;
        assert!(matches!(
            generate_fast_event_task(config),
            Err(FastEventGenerationError::ZeroSampleInterval)
        ));
    }

    #[test]
    fn verifier_rejects_recommitted_event_code_outside_similarity_band() {
        let config = FastEventConfig::reference(26);
        let (task, mut oracle) = generate_fast_event_task(config).unwrap();
        if let Some(event) = oracle.events.iter_mut().find(|event| !event.is_target) {
            event.code = TARGET_CODE;
        }
        oracle.content_sha256 = fast_event_oracle_digest(&oracle);
        assert!(matches!(
            verify_fast_event_trace(&task, &oracle),
            Err(FastEventVerificationError::InvalidEventCode { .. })
        ));
    }

    #[test]
    fn verifier_rejects_recommitted_sample_outside_noise_bound() {
        let config = FastEventConfig::reference(27);
        let (mut task, oracle) = generate_fast_event_task(config).unwrap();
        task.observations[0].sample = MAX_NOISE.saturating_add(100);
        task.content_sha256 = fast_event_content_digest(&task);
        assert_eq!(
            verify_fast_event_trace(&task, &oracle),
            Err(FastEventVerificationError::NoiseBoundExceeded { step: 0 })
        );
    }

    #[test]
    fn verifier_rejects_timestamps_not_bound_to_the_sampling_interval() {
        let config = FastEventConfig::reference(28);
        let (mut task, oracle) = generate_fast_event_task(config).unwrap();
        task.observations[4].time_us += 1;
        task.content_sha256 = fast_event_content_digest(&task);
        assert_eq!(
            verify_fast_event_trace(&task, &oracle),
            Err(FastEventVerificationError::InvalidObservation { expected_step: 4 })
        );
    }
}
