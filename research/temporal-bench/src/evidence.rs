//! Deterministic sequential evidence-accumulation reference workload.
//!
//! This follows the BrainTrace public experiment generator's task structure:
//! seven cue periods, left/right cue populations, a delay, recall population,
//! and background-noise population. The binary target is the majority of
//! selected left/right cue assignments. The public event stream contains only
//! spikes; cue assignments and target are kept in the oracle record.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::fmt;

const MAX_EVIDENCE_STEPS: usize = 500_000;
const GROUP_SIZE: usize = 25;
const INPUT_CHANNELS: usize = GROUP_SIZE * 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvidenceConfig {
    pub seed: u64,
    pub dt_ms: u32,
    pub interval_ms: u32,
    pub cue_ms: u32,
    pub delay_ms: u32,
    pub recall_ms: u32,
    pub cue_count: usize,
    /// Threshold used by the source generator: p for positive ground truth,
    /// 1-p for negative ground truth; assignments are draws greater than it.
    pub cue_bias_probability: f64,
    pub left_rate_hz: f64,
    pub right_rate_hz: f64,
    pub recall_rate_hz: f64,
    pub noise_rate_hz: f64,
}

impl EvidenceConfig {
    pub fn braintrace_reference(seed: u64) -> Self {
        Self {
            seed,
            dt_ms: 1,
            interval_ms: 50,
            cue_ms: 100,
            delay_ms: 1_000,
            recall_ms: 150,
            cue_count: 7,
            cue_bias_probability: 0.3,
            left_rate_hz: 40.0,
            right_rate_hz: 40.0,
            recall_rate_hz: 40.0,
            noise_rate_hz: 10.0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceObservation {
    pub step: u32,
    pub active_channels: Vec<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceTask {
    pub stream_id: u64,
    pub dt_ms: u32,
    pub observations: Vec<EvidenceObservation>,
    pub recall_start_step: u32,
    pub recall_end_step: u32,
    pub interval_steps: u32,
    pub cue_steps: u32,
    pub delay_steps: u32,
    pub recall_steps: u32,
    pub cue_count: usize,
    pub input_channels: usize,
    pub cue_bias_probability_bits: u64,
    pub left_rate_hz_bits: u64,
    pub right_rate_hz_bits: u64,
    pub recall_rate_hz_bits: u64,
    pub noise_rate_hz_bits: u64,
    /// SHA-256 over the versioned canonical public evidence stream.
    pub content_sha256: [u8; 32],
}

/// Domain-separated canonical SHA-256 of the evidence-accumulation public stream.
pub fn evidence_content_digest(task: &EvidenceTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.evidence-accumulation.public.v1");
    hash.u64(task.stream_id);
    hash.u32(task.dt_ms);
    hash.u32(task.recall_start_step);
    hash.u32(task.recall_end_step);
    hash.u32(task.interval_steps);
    hash.u32(task.cue_steps);
    hash.u32(task.delay_steps);
    hash.u32(task.recall_steps);
    hash.usize(task.cue_count);
    hash.usize(task.input_channels);
    hash.u64(task.cue_bias_probability_bits);
    hash.u64(task.left_rate_hz_bits);
    hash.u64(task.right_rate_hz_bits);
    hash.u64(task.recall_rate_hz_bits);
    hash.u64(task.noise_rate_hz_bits);
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
pub struct EvidenceOracle {
    pub stream_id: u64,
    pub seed: u64,
    pub hidden_ground_truth: bool,
    pub cue_assignments: Vec<bool>,
    pub positive_cue_count: usize,
    pub expected_label: u8,
    /// SHA-256 over the versioned canonical oracle record.
    pub content_sha256: [u8; 32],
}

/// Domain-separated canonical SHA-256 of the evidence evaluator oracle.
pub fn evidence_oracle_digest(oracle: &EvidenceOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.evidence-accumulation.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u64(oracle.seed);
    hash.u8(u8::from(oracle.hidden_ground_truth));
    hash.usize(oracle.cue_assignments.len());
    for selected_left in &oracle.cue_assignments {
        hash.u8(u8::from(*selected_left));
    }
    hash.usize(oracle.positive_cue_count);
    hash.u8(oracle.expected_label);
    hash.finish()
}

#[derive(Clone, Debug, PartialEq)]
pub enum EvidenceGenerationError {
    ZeroTimeStep,
    ZeroCueDuration,
    ZeroRecallDuration,
    DurationNotDivisibleByStep { duration_ms: u32, dt_ms: u32 },
    InvalidCueCount { requested: usize, maximum: usize },
    InvalidCueBiasProbability,
    InvalidRate { name: &'static str },
    SpikeProbabilityOutOfRange { maximum_probability: f64 },
    TooManySteps { requested: usize, maximum: usize },
}

impl fmt::Display for EvidenceGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTimeStep => write!(f, "dt_ms must be greater than zero"),
            Self::ZeroCueDuration => write!(f, "cue_ms must be greater than zero"),
            Self::ZeroRecallDuration => write!(f, "recall_ms must be greater than zero"),
            Self::DurationNotDivisibleByStep { duration_ms, dt_ms } => write!(
                f,
                "duration {duration_ms} ms is not divisible by dt {dt_ms} ms"
            ),
            Self::InvalidCueCount { requested, maximum } => {
                write!(f, "cue_count {requested} is outside 1..={maximum}")
            }
            Self::InvalidCueBiasProbability => {
                write!(f, "cue_bias_probability must be finite and strictly between 0 and 0.5")
            }
            Self::InvalidRate { name } => write!(f, "{name} must be finite and non-negative"),
            Self::SpikeProbabilityOutOfRange { maximum_probability } => write!(
                f,
                "maximum spike probability {maximum_probability} exceeds 1"
            ),
            Self::TooManySteps { requested, maximum } => {
                write!(f, "evidence stream requires {requested} steps; maximum is {maximum}")
            }
        }
    }
}

impl std::error::Error for EvidenceGenerationError {}

/// Generate one sequential evidence-accumulation stream and its oracle.
///
/// The source task uses a random binary latent truth. Each of N cues selects
/// the left population for a positive assignment and the right population for
/// a negative assignment. A majority of positive assignments yields label 1.
/// The recall population is stimulated only in the final recall window; a
/// noise population is active throughout.
pub fn generate_evidence_task(
    config: EvidenceConfig,
) -> Result<(EvidenceTask, EvidenceOracle), EvidenceGenerationError> {
    validate_config(config)?;

    let stream_id = evidence_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x4556_4944_454E_4345);
    let hidden_ground_truth = rng.next_u64() & 1 == 1;
    let threshold = if hidden_ground_truth {
        config.cue_bias_probability
    } else {
        1.0 - config.cue_bias_probability
    };
    let mut cue_assignments = Vec::with_capacity(config.cue_count);
    for _ in 0..config.cue_count {
        cue_assignments.push(random_unit(&mut rng) > threshold);
    }
    let positive_cue_count = cue_assignments.iter().filter(|selected| **selected).count();
    let expected_label = u8::from(positive_cue_count * 2 > config.cue_count);

    let interval_steps = config.interval_ms / config.dt_ms;
    let cue_steps = config.cue_ms / config.dt_ms;
    let delay_steps = config.delay_ms / config.dt_ms;
    let recall_steps = config.recall_ms / config.dt_ms;
    let sequence_end = config.cue_count
        .saturating_mul(interval_steps as usize + cue_steps as usize);
    let recall_start = u32::try_from(sequence_end)
        .unwrap_or(u32::MAX)
        .saturating_add(delay_steps);
    let recall_end = recall_start.saturating_add(recall_steps);
    let total_steps = usize::try_from(recall_end).unwrap_or(usize::MAX);

    let dt_seconds = f64::from(config.dt_ms) / 1_000.0;
    let left_probability = config.left_rate_hz * dt_seconds;
    let right_probability = config.right_rate_hz * dt_seconds;
    let recall_probability = config.recall_rate_hz * dt_seconds;
    let noise_probability = config.noise_rate_hz * dt_seconds;

    let mut observations = Vec::with_capacity(total_steps);
    for step in 0..recall_end {
        let mut active_channels = Vec::new();
        let mut probabilities = [0.0_f64; INPUT_CHANNELS];

        // Background-noise population is active for the full trial.
        for channel in (GROUP_SIZE * 3)..INPUT_CHANNELS {
            probabilities[channel] = noise_probability;
        }

        // Cue k appears after its preceding inter-cue interval.
        for (cue_index, selected_left) in cue_assignments.iter().enumerate() {
            let cue_start = u32::try_from(
                interval_steps as usize
                    + cue_index * (interval_steps as usize + cue_steps as usize),
            )
            .unwrap_or(u32::MAX);
            let cue_end = cue_start.saturating_add(cue_steps);
            if step >= cue_start && step < cue_end {
                let range = if *selected_left {
                    0..GROUP_SIZE
                } else {
                    GROUP_SIZE..(GROUP_SIZE * 2)
                };
                let probability = if *selected_left {
                    left_probability
                } else {
                    right_probability
                };
                for channel in range {
                    probabilities[channel] = probability;
                }
            }
        }

        if step >= recall_start && step < recall_end {
            for channel in (GROUP_SIZE * 2)..(GROUP_SIZE * 3) {
                probabilities[channel] = recall_probability;
            }
        }

        for (channel, probability) in probabilities.into_iter().enumerate() {
            if random_unit(&mut rng) < probability {
                active_channels.push(u16::try_from(channel).unwrap_or(u16::MAX));
            }
        }
        observations.push(EvidenceObservation {
            step,
            active_channels,
        });
    }

    let mut public_task = EvidenceTask {
        stream_id,
        dt_ms: config.dt_ms,
        observations,
        recall_start_step: recall_start,
        recall_end_step: recall_end,
        interval_steps,
        cue_steps,
        delay_steps,
        recall_steps,
        cue_count: config.cue_count,
        input_channels: INPUT_CHANNELS,
        cue_bias_probability_bits: config.cue_bias_probability.to_bits(),
        left_rate_hz_bits: config.left_rate_hz.to_bits(),
        right_rate_hz_bits: config.right_rate_hz.to_bits(),
        recall_rate_hz_bits: config.recall_rate_hz.to_bits(),
        noise_rate_hz_bits: config.noise_rate_hz.to_bits(),
        content_sha256: [0; 32],
    };
    public_task.content_sha256 = evidence_content_digest(&public_task);

    Ok((
        public_task,
        {
            let mut oracle = EvidenceOracle {
                stream_id,
                hidden_ground_truth,
                cue_assignments,
                positive_cue_count,
                expected_label,
                content_sha256: [0; 32],
            };
            oracle.content_sha256 = evidence_oracle_digest(&oracle);
            oracle
        },
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceVerificationError {
    StreamIdMismatch,
    ContentDigestMismatch,
    OracleDigestMismatch,
    ReproducibilityMismatch,
    InvalidTaskWindow,
    InvalidOracle,
    InvalidObservation { expected_step: u32 },
    ChannelOutOfRange { step: u32, channel: u16 },
    ChannelsNotStrictlyOrdered { step: u32 },
    CueChannelOutsideAssignedWindow { step: u32, channel: u16 },
    RecallChannelOutsideRecallWindow { step: u32, channel: u16 },
}

impl fmt::Display for EvidenceVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "evidence task and oracle stream IDs differ"),
            Self::ContentDigestMismatch => {
                write!(f, "evidence public task SHA-256 digest does not match its contents")
            }
            Self::OracleDigestMismatch => {
                write!(f, "evidence oracle SHA-256 digest does not match its contents")
            }
            Self::ReproducibilityMismatch => {
                write!(f, "evidence public stream/oracle does not match the seeded generator")
            }
            Self::InvalidTaskWindow => {
                write!(f, "evidence task timing or input dimensions are invalid")
            }
            Self::InvalidOracle => {
                write!(f, "evidence oracle assignments or majority label are invalid")
            }
            Self::InvalidObservation { expected_step } => {
                write!(f, "evidence observation is missing or reordered at step {expected_step}")
            }
            Self::ChannelOutOfRange { step, channel } => {
                write!(f, "evidence event at step {step} contains out-of-range channel {channel}")
            }
            Self::ChannelsNotStrictlyOrdered { step } => {
                write!(f, "evidence active channels are duplicated or unordered at step {step}")
            }
            Self::CueChannelOutsideAssignedWindow { step, channel } => {
                write!(
                    f,
                    "cue channel {channel} is active outside its assigned cue window at step {step}"
                )
            }
            Self::RecallChannelOutsideRecallWindow { step, channel } => {
                write!(
                    f,
                    "recall channel {channel} is active outside the recall window at step {step}"
                )
            }
        }
    }
}

impl std::error::Error for EvidenceVerificationError {}

/// Evaluator-side oracle and event-structure validation for evidence trials.
pub fn verify_evidence_trace(
    task: &EvidenceTask,
    oracle: &EvidenceOracle,
) -> Result<(), EvidenceVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(EvidenceVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != evidence_content_digest(task) {
        return Err(EvidenceVerificationError::ContentDigestMismatch);
    }
    if oracle.content_sha256 != evidence_oracle_digest(oracle) {
        return Err(EvidenceVerificationError::OracleDigestMismatch);
    }
    let expected_sequence_end = u64::try_from(task.cue_count)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(task.interval_steps.saturating_add(task.cue_steps)));
    let expected_recall_start = expected_sequence_end.saturating_add(u64::from(task.delay_steps));
    if task.input_channels != INPUT_CHANNELS
        || task.dt_ms == 0
        || task.cue_count == 0
        || task.cue_count != oracle.cue_assignments.len()
        || u64::from(task.recall_start_step) != expected_recall_start
        || task.recall_start_step.saturating_add(task.recall_steps) != task.recall_end_step
        || usize::try_from(task.recall_end_step).unwrap_or(usize::MAX)
            != task.observations.len()
    {
        return Err(EvidenceVerificationError::InvalidTaskWindow);
    }

    let positive_count = oracle.cue_assignments.iter().filter(|value| **value).count();
    if positive_count != oracle.positive_cue_count
        || oracle.expected_label != u8::from(positive_count * 2 > task.cue_count)
        || oracle.expected_label > 1
    {
        return Err(EvidenceVerificationError::InvalidOracle);
    }

    for (index, observation) in task.observations.iter().enumerate() {
        let expected_step = u32::try_from(index).unwrap_or(u32::MAX);
        if observation.step != expected_step {
            return Err(EvidenceVerificationError::InvalidObservation { expected_step });
        }
        let mut previous = None;
        for channel in &observation.active_channels {
            let channel_index = usize::from(*channel);
            if channel_index >= task.input_channels {
                return Err(EvidenceVerificationError::ChannelOutOfRange {
                    step: observation.step,
                    channel: *channel,
                });
            }
            if previous.is_some_and(|prior| prior >= *channel) {
                return Err(EvidenceVerificationError::ChannelsNotStrictlyOrdered {
                    step: observation.step,
                });
            }
            previous = Some(*channel);

            if channel_index < GROUP_SIZE * 2 {
                let cue_index = (0..task.cue_count).find(|cue| {
                    let start = u64::from(task.interval_steps)
                        + u64::try_from(*cue).unwrap_or(u64::MAX)
                            * u64::from(task.interval_steps.saturating_add(task.cue_steps));
                    let end = start.saturating_add(u64::from(task.cue_steps));
                    u64::from(observation.step) >= start
                        && u64::from(observation.step) < end
                });
                let Some(cue_index) = cue_index else {
                    return Err(EvidenceVerificationError::CueChannelOutsideAssignedWindow {
                        step: observation.step,
                        channel: *channel,
                    });
                };
                let is_left_channel = channel_index < GROUP_SIZE;
                if is_left_channel != oracle.cue_assignments[cue_index] {
                    return Err(EvidenceVerificationError::CueChannelOutsideAssignedWindow {
                        step: observation.step,
                        channel: *channel,
                    });
                }
            } else if channel_index < GROUP_SIZE * 3
                && (observation.step < task.recall_start_step
                    || observation.step >= task.recall_end_step)
            {
                return Err(EvidenceVerificationError::RecallChannelOutsideRecallWindow {
                    step: observation.step,
                    channel: *channel,
                });
            }
        }
    }

    // Rebuild the exact public stream and hidden labels from the evaluator seed.
    // This detects recommitted but seed-inconsistent sample/oracle changes.
    let config = EvidenceConfig {
        seed: oracle.seed,
        dt_ms: task.dt_ms,
        interval_ms: task.interval_steps.saturating_mul(task.dt_ms),
        cue_ms: task.cue_steps.saturating_mul(task.dt_ms),
        delay_ms: task.delay_steps.saturating_mul(task.dt_ms),
        recall_ms: task.recall_steps.saturating_mul(task.dt_ms),
        cue_count: task.cue_count,
        cue_bias_probability: f64::from_bits(task.cue_bias_probability_bits),
        left_rate_hz: f64::from_bits(task.left_rate_hz_bits),
        right_rate_hz: f64::from_bits(task.right_rate_hz_bits),
        recall_rate_hz: f64::from_bits(task.recall_rate_hz_bits),
        noise_rate_hz: f64::from_bits(task.noise_rate_hz_bits),
    };
    let (expected_task, expected_oracle) =
    let (expected_task, expected_oracle) = generate_evidence_task(config)
        .map_err(|_| EvidenceVerificationError::ReproducibilityMismatch)?;
    if expected_task != *task || expected_oracle != *oracle {
        return Err(EvidenceVerificationError::ReproducibilityMismatch);
    }
    Ok(())
}

fn validate_config(config: EvidenceConfig) -> Result<(), EvidenceGenerationError> {
    if config.dt_ms == 0 {
        return Err(EvidenceGenerationError::ZeroTimeStep);
    }
    if config.cue_ms == 0 {
        return Err(EvidenceGenerationError::ZeroCueDuration);
    }
    if config.recall_ms == 0 {
        return Err(EvidenceGenerationError::ZeroRecallDuration);
    }
    for duration in [
        config.interval_ms,
        config.cue_ms,
        config.delay_ms,
        config.recall_ms,
    ] {
        if duration % config.dt_ms != 0 {
            return Err(EvidenceGenerationError::DurationNotDivisibleByStep {
                duration_ms: duration,
                dt_ms: config.dt_ms,
            });
        }
    }
    if !(1..=31).contains(&config.cue_count) {
        return Err(EvidenceGenerationError::InvalidCueCount {
            requested: config.cue_count,
            maximum: 31,
        });
    }
    if !config.cue_bias_probability.is_finite()
        || !(0.0..0.5).contains(&config.cue_bias_probability)
    {
        return Err(EvidenceGenerationError::InvalidCueBiasProbability);
    }
    for (name, rate) in [
        ("left_rate_hz", config.left_rate_hz),
        ("right_rate_hz", config.right_rate_hz),
        ("recall_rate_hz", config.recall_rate_hz),
        ("noise_rate_hz", config.noise_rate_hz),
    ] {
        if !rate.is_finite() || rate < 0.0 {
            return Err(EvidenceGenerationError::InvalidRate { name });
        }
        let probability = rate * f64::from(config.dt_ms) / 1_000.0;
        if probability > 1.0 {
            return Err(EvidenceGenerationError::SpikeProbabilityOutOfRange {
                maximum_probability: probability,
            });
        }
    }
    let interval_steps = config.interval_ms / config.dt_ms;
    let cue_steps = config.cue_ms / config.dt_ms;
    let delay_steps = config.delay_ms / config.dt_ms;
    let recall_steps = config.recall_ms / config.dt_ms;
    let total_steps = u64::try_from(config.cue_count)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(interval_steps.saturating_add(cue_steps)))
        .saturating_add(u64::from(delay_steps))
        .saturating_add(u64::from(recall_steps));
    if total_steps > MAX_EVIDENCE_STEPS as u64 {
        return Err(EvidenceGenerationError::TooManySteps {
            requested: usize::try_from(total_steps).unwrap_or(usize::MAX),
            maximum: MAX_EVIDENCE_STEPS,
        });
    }
    Ok(())
}

fn evidence_stream_id(config: EvidenceConfig) -> u64 {
    let mut value = config.seed ^ u64::from(config.dt_ms).rotate_left(3);
    value ^= u64::from(config.interval_ms).rotate_left(9);
    value ^= u64::from(config.cue_ms).rotate_left(15);
    value ^= u64::from(config.delay_ms).rotate_left(21);
    value ^= u64::from(config.recall_ms).rotate_left(27);
    value ^= (config.cue_count as u64).rotate_left(35);
    value ^= config.cue_bias_probability.to_bits().rotate_left(43);
    value ^= config.left_rate_hz.to_bits().rotate_left(5);
    value ^= config.right_rate_hz.to_bits().rotate_left(13);
    value ^= config.recall_rate_hz.to_bits().rotate_left(29);
    value ^= config.noise_rate_hz.to_bits().rotate_left(51);
    mix64(value)
}

fn random_unit(rng: &mut SplitMix64) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_config_matches_pinned_task_timing() {
        let config = EvidenceConfig::braintrace_reference(42);
        let (task, _) = generate_evidence_task(config).unwrap();
        assert_eq!(task.input_channels, 100);
        assert_eq!(task.observations.len(), 2_200);
        assert_eq!(task.recall_start_step, 2_050);
        assert_eq!(task.recall_steps, 150);
        assert_eq!(task.recall_end_step, 2_200);
    }

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let config = EvidenceConfig::braintrace_reference(77);
        assert_eq!(
            generate_evidence_task(config),
            generate_evidence_task(config)
        );
    }

    #[test]
    fn oracle_label_is_the_strict_majority_of_cue_assignments() {
        for seed in 0..128 {
            let config = EvidenceConfig::braintrace_reference(seed);
            let (_, oracle) = generate_evidence_task(config).unwrap();
            let count = oracle.cue_assignments.iter().filter(|value| **value).count();
            assert_eq!(oracle.positive_cue_count, count);
            assert_eq!(oracle.expected_label, u8::from(count * 2 > config.cue_count));
        }
    }

    #[test]
    fn observation_steps_and_channel_indices_are_valid() {
        let config = EvidenceConfig::braintrace_reference(13);
        let (task, _) = generate_evidence_task(config).unwrap();
        for (index, observation) in task.observations.iter().enumerate() {
            assert_eq!(observation.step as usize, index);
            assert!(observation
                .active_channels
                .iter()
                .all(|channel| usize::from(*channel) < task.input_channels));
        }
    }

    #[test]
    fn generated_evidence_trace_passes_oracle_verifier() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (task, oracle) = generate_evidence_task(config).unwrap();
        assert_eq!(verify_evidence_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn generated_oracle_has_valid_sha256() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (_, oracle) = generate_evidence_task(config).unwrap();
        assert_eq!(oracle.content_sha256, evidence_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_oracle_mutation_after_generation() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (task, mut oracle) = generate_evidence_task(config).unwrap();
        oracle.cue_assignments[0] = !oracle.cue_assignments[0];
        assert_eq!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::OracleDigestMismatch)
        );
    }

    #[test]
    fn generated_public_stream_has_valid_sha256() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (task, _) = generate_evidence_task(config).unwrap();
        assert_eq!(task.content_sha256, evidence_content_digest(&task));
    }

    #[test]
    fn verifier_rejects_post_generation_stream_mutation() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (mut task, oracle) = generate_evidence_task(config).unwrap();
        task.observations[0].step = 1;
        assert_eq!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::ContentDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_recall_duration_mismatch() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (mut task, oracle) = generate_evidence_task(config).unwrap();
        task.recall_steps = task.recall_steps.saturating_add(1);
        task.content_sha256 = evidence_content_digest(&task);
        assert_eq!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::InvalidTaskWindow)
        );
    }

    #[test]
    fn verifier_rejects_recommitted_seed_inconsistent_observation() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (mut task, oracle) = generate_evidence_task(config).unwrap();
        let observation = task
            .observations
            .iter_mut()
            .find(|observation| !observation.active_channels.is_empty())
            .unwrap();
        observation.active_channels.pop();
        task.content_sha256 = evidence_content_digest(&task);
        assert_eq!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::ReproducibilityMismatch)
        );
    }

    #[test]
    fn oracle_digest_binds_the_generator_seed() {
        let config = EvidenceConfig::braintrace_reference(322);
        let (_, mut oracle) = generate_evidence_task(config).unwrap();
        oracle.seed ^= 1;
        assert_ne!(oracle.content_sha256, evidence_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_mutated_majority_label() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (task, mut oracle) = generate_evidence_task(config).unwrap();
        oracle.expected_label ^= 1;
        assert_eq!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::InvalidOracle)
        );
    }

    #[test]
    fn verifier_rejects_recall_spikes_before_recall_window() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (mut task, oracle) = generate_evidence_task(config).unwrap();
        task.observations[0].active_channels.push(50);
        task.observations[0].active_channels.sort_unstable();
        task.content_sha256 = evidence_content_digest(&task);
        assert!(matches!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::RecallChannelOutsideRecallWindow { .. })
        ));
    }

    #[test]
    fn verifier_rejects_cue_spikes_outside_cue_windows() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (mut task, oracle) = generate_evidence_task(config).unwrap();
        task.observations[0].active_channels.push(0);
        task.observations[0].active_channels.sort_unstable();
        task.content_sha256 = evidence_content_digest(&task);
        assert!(matches!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::CueChannelOutsideAssignedWindow { .. })
        ));
    }

    #[test]
    fn verifier_rejects_stream_identity_mismatch() {
        let config = EvidenceConfig::braintrace_reference(321);
        let (task, mut oracle) = generate_evidence_task(config).unwrap();
        oracle.stream_id ^= 1;
        assert_eq!(
            verify_evidence_trace(&task, &oracle),
            Err(EvidenceVerificationError::StreamIdMismatch)
        );
    }

    #[test]
    fn invalid_cue_count_is_rejected() {
        let mut config = EvidenceConfig::braintrace_reference(1);
        config.cue_count = 0;
        assert!(matches!(
            generate_evidence_task(config),
            Err(EvidenceGenerationError::InvalidCueCount { .. })
        ));
    }

    #[test]
    fn zero_time_step_is_rejected() {
        let mut config = EvidenceConfig::braintrace_reference(1);
        config.dt_ms = 0;
        assert!(matches!(
            generate_evidence_task(config),
            Err(EvidenceGenerationError::ZeroTimeStep)
        ));
    }

    #[test]
    fn out_of_range_cue_bias_is_rejected() {
        let mut config = EvidenceConfig::braintrace_reference(1);
        config.cue_bias_probability = 0.8;
        assert!(matches!(
            generate_evidence_task(config),
            Err(EvidenceGenerationError::InvalidCueBiasProbability)
        ));
    }
}
