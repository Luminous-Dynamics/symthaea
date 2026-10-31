//! Deterministic synthetic controls for multi-timescale temporal benchmarks.
//!
//! This prototype deliberately has no model dependency and no runtime dependency.
//! It generates a public delayed-cue event stream and a separate oracle record.
//! Returning both artifacts from one process is NOT a sealed evaluator boundary;
//! qualification must move oracle ownership into a separately permissioned process.

use std::collections::BTreeSet;
use std::fmt;

use digest::CanonicalHasher;

pub mod digest;
pub mod dms;
pub mod evidence;
pub mod irregular;
pub mod temporal;

const MAX_DISTRACTORS: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistractorSimilarity {
    /// Distractor keys differ from the target key in 1-4 bits.
    High,
    /// Distractor keys differ from the target key in 28-36 bits.
    Low,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DelayedCueConfig {
    pub seed: u64,
    pub delay_steps: u64,
    pub distractor_count: usize,
    pub similarity: DistractorSimilarity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EventKind {
    Associate { key: u64, value: u64 },
    Probe { key: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimedEvent {
    /// Integer simulated time. This prototype uses unit-spaced time steps.
    pub step: u64,
    /// Stable order for multiple events sharing the same simulated timestamp.
    pub ordinal: u64,
    pub kind: EventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicTask {
    pub stream_id: u64,
    pub delay_steps: u64,
    pub distractor_count: usize,
    pub similarity: DistractorSimilarity,
    pub events: Vec<TimedEvent>,
    /// SHA-256 over the versioned canonical public-task encoding.
    pub content_sha256: [u8; 32],
}

/// Compute the domain-separated SHA-256 digest of a delayed-cue public task.
///
/// The digest covers all public task fields and ordered events, but not the
/// digest field itself. The 64-bit stream_id is included as a pairing key only.
pub fn delayed_cue_content_digest(task: &PublicTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.delayed-cue.public.v1");
    hash.u64(task.stream_id);
    hash.u64(task.delay_steps);
    hash.usize(task.distractor_count);
    hash.u8(match task.similarity {
        DistractorSimilarity::High => 0,
        DistractorSimilarity::Low => 1,
    });
    hash.usize(task.events.len());
    for event in &task.events {
        hash.u64(event.step);
        hash.u64(event.ordinal);
        match &event.kind {
            EventKind::Associate { key, value } => {
                hash.u8(0);
                hash.u64(*key);
                hash.u64(*value);
            }
            EventKind::Probe { key } => {
                hash.u8(1);
                hash.u64(*key);
            }
        }
    }
    hash.finish()
}

/// Check whether a delayed-cue task still matches its canonical content digest.
/// This detects accidental or post-commitment mutation when the digest itself
/// is bound into a separately protected receipt; it does not authenticate who
/// created the task or protect an oracle held by the same process.
pub fn delayed_cue_digest_valid(task: &PublicTask) -> bool {
    task.content_sha256 == delayed_cue_content_digest(task)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OracleRecord {
    pub stream_id: u64,
    pub target_key: u64,
    pub expected_value: u64,
    /// SHA-256 over the versioned canonical oracle record.
    pub content_sha256: [u8; 32],
}

/// Domain-separated canonical SHA-256 of a delayed-cue oracle record.
pub fn delayed_cue_oracle_digest(oracle: &OracleRecord) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.delayed-cue.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u64(oracle.target_key);
    hash.u64(oracle.expected_value);
    hash.finish()
}

pub fn delayed_cue_oracle_digest_valid(oracle: &OracleRecord) -> bool {
    oracle.content_sha256 == delayed_cue_oracle_digest(oracle)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DelayedCueVerificationError {
    StreamIdMismatch,
    PublicDigestMismatch,
    OracleDigestMismatch,
    InvalidEventCount,
    InvalidTargetBinding,
    InvalidProbe,
    InvalidDistractor { index: usize },
    InvalidEventOrder { index: usize },
}

impl fmt::Display for DelayedCueVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "delayed-cue task and oracle stream IDs differ"),
            Self::PublicDigestMismatch => write!(f, "delayed-cue public SHA-256 digest mismatch"),
            Self::OracleDigestMismatch => write!(f, "delayed-cue oracle SHA-256 digest mismatch"),
            Self::InvalidEventCount => write!(f, "delayed-cue event count is inconsistent"),
            Self::InvalidTargetBinding => write!(f, "target cue binding is missing or inconsistent"),
            Self::InvalidProbe => write!(f, "delayed-cue probe is invalid"),
            Self::InvalidDistractor { index } => {
                write!(f, "distractor {index} violates the declared schedule or similarity band")
            }
            Self::InvalidEventOrder { index } => {
                write!(f, "event {index} is not in strict temporal/ordinal order")
            }
        }
    }
}

impl std::error::Error for DelayedCueVerificationError {}

/// Check canonical digests and the complete delayed-cue task contract.
pub fn verify_delayed_cue_trace(
    task: &PublicTask,
    oracle: &OracleRecord,
) -> Result<(), DelayedCueVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(DelayedCueVerificationError::StreamIdMismatch);
    }
    if !delayed_cue_digest_valid(task) {
        return Err(DelayedCueVerificationError::PublicDigestMismatch);
    }
    if !delayed_cue_oracle_digest_valid(oracle) {
        return Err(DelayedCueVerificationError::OracleDigestMismatch);
    }
    if task.delay_steps == 0
        || task.distractor_count > MAX_DISTRACTORS
        || task.events.len() != task.distractor_count.saturating_add(2)
    {
        return Err(DelayedCueVerificationError::InvalidEventCount);
    }

    let first = task.events.first().ok_or(
        DelayedCueVerificationError::InvalidTargetBinding,
    )?;
    match &first.kind {
        EventKind::Associate { key, value }
            if first.step == 0
                && first.ordinal == 0
                && *key == oracle.target_key
                && *value == oracle.expected_value => {}
        _ => return Err(DelayedCueVerificationError::InvalidTargetBinding),
    }

    let last = task.events.last().ok_or(
        DelayedCueVerificationError::InvalidProbe,
    )?;
    let expected_probe_ordinal = u64::try_from(task.distractor_count)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    match &last.kind {
        EventKind::Probe { key }
            if last.step == task.delay_steps
                && last.ordinal == expected_probe_ordinal
                && *key == oracle.target_key => {}
        _ => return Err(DelayedCueVerificationError::InvalidProbe),
    }

    let mut keys = BTreeSet::new();
    let mut values = BTreeSet::new();
    keys.insert(oracle.target_key);
    values.insert(oracle.expected_value);
    for (index, event) in task.events[1..task.events.len() - 1].iter().enumerate() {
        let ordinal = u64::try_from(index).unwrap_or(u64::MAX).saturating_add(1);
        let expected_step = distractor_step(index, task.distractor_count, task.delay_steps);
        match &event.kind {
            EventKind::Associate { key, value } => {
                let distance = (*key ^ oracle.target_key).count_ones();
                let similarity_valid = match task.similarity {
                    DistractorSimilarity::High => (1..=4).contains(&distance),
                    DistractorSimilarity::Low => (28..=36).contains(&distance),
                };
                if event.step != expected_step
                    || event.ordinal != ordinal
                    || *key == oracle.target_key
                    || !similarity_valid
                    || !keys.insert(*key)
                    || !values.insert(*value)
                {
                    return Err(DelayedCueVerificationError::InvalidDistractor { index });
                }
            }
            EventKind::Probe { .. } => {
                return Err(DelayedCueVerificationError::InvalidDistractor { index });
            }
        }
    }

    for (index, pair) in task.events.windows(2).enumerate() {
        let previous = (pair[0].step, pair[0].ordinal);
        let current = (pair[1].step, pair[1].ordinal);
        if current <= previous {
            return Err(DelayedCueVerificationError::InvalidEventOrder { index: index + 1 });
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationError {
    ZeroDelay,
    TooManyDistractors { requested: usize, maximum: usize },
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDelay => write!(f, "delay_steps must be greater than zero"),
            Self::TooManyDistractors { requested, maximum } => {
                write!(
                    f,
                    "distractor_count {requested} exceeds the maximum {maximum}"
                )
            }
        }
    }
}

impl std::error::Error for GenerationError {}

/// Generate one delayed-cue association task.
///
/// The target association occurs at step 0. Distractors are ordered between
/// the target and probe; with a one-step delay they share the probe timestamp
/// but precede it by ordinal. Target and distractor keys have controlled
/// Hamming similarity. The returned oracle is a separate data structure, not
/// a security boundary: callers must not expose it to the candidate model.
pub fn generate_delayed_cue(
    config: DelayedCueConfig,
) -> Result<(PublicTask, OracleRecord), GenerationError> {
    if config.delay_steps == 0 {
        return Err(GenerationError::ZeroDelay);
    }
    if config.distractor_count > MAX_DISTRACTORS {
        return Err(GenerationError::TooManyDistractors {
            requested: config.distractor_count,
            maximum: MAX_DISTRACTORS,
        });
    }

    let stream_id = stream_id(config);
    let mut rng = SplitMix64::new(config.seed);
    let target_key = rng.next_u64();
    let target_value = rng.next_u64();

    let mut used_keys = BTreeSet::new();
    let mut used_values = BTreeSet::new();
    used_keys.insert(target_key);
    used_values.insert(target_value);

    let mut events = Vec::with_capacity(config.distractor_count.saturating_add(2));
    events.push(TimedEvent {
        step: 0,
        ordinal: 0,
        kind: EventKind::Associate {
            key: target_key,
            value: target_value,
        },
    });

    for index in 0..config.distractor_count {
        let key = unique_similar_key(&mut rng, target_key, config.similarity, &mut used_keys);
        let value = unique_value(&mut rng, &mut used_values);
        let step = distractor_step(index, config.distractor_count, config.delay_steps);
        let ordinal = u64::try_from(index).unwrap_or(u64::MAX).saturating_add(1);
        events.push(TimedEvent {
            step,
            ordinal,
            kind: EventKind::Associate { key, value },
        });
    }

    let probe_ordinal = u64::try_from(config.distractor_count)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    events.push(TimedEvent {
        step: config.delay_steps,
        ordinal: probe_ordinal,
        kind: EventKind::Probe { key: target_key },
    });

    let mut public_task = PublicTask {
        stream_id,
        delay_steps: config.delay_steps,
        distractor_count: config.distractor_count,
        similarity: config.similarity,
        events,
        content_sha256: [0; 32],
    };
    public_task.content_sha256 = delayed_cue_content_digest(&public_task);

    let mut oracle = OracleRecord {
        stream_id,
        target_key,
        expected_value: target_value,
        content_sha256: [0; 32],
    };
    oracle.content_sha256 = delayed_cue_oracle_digest(&oracle);

    Ok((public_task, oracle))
}

fn distractor_step(index: usize, count: usize, delay_steps: u64) -> u64 {
    if delay_steps == 1 {
        return 1;
    }
    let numerator = u128::try_from(index.saturating_add(1)).unwrap_or(u128::MAX);
    let denominator = u128::try_from(count.saturating_add(1)).unwrap_or(u128::MAX);
    let interior_steps = u128::from(delay_steps - 1);
    let offset = numerator.saturating_mul(interior_steps) / denominator;
    1_u64.saturating_add(u64::try_from(offset).unwrap_or(u64::MAX))
}

fn unique_similar_key(
    rng: &mut SplitMix64,
    target: u64,
    similarity: DistractorSimilarity,
    used: &mut BTreeSet<u64>,
) -> u64 {
    loop {
        let candidate = match similarity {
            DistractorSimilarity::High => {
                let bit_count = 1 + (rng.next_u64() % 4);
                target ^ random_mask(rng, bit_count)
            }
            DistractorSimilarity::Low => target ^ low_similarity_mask(rng),
        };
        if used.insert(candidate) {
            return candidate;
        }
    }
}

fn random_mask(rng: &mut SplitMix64, bit_count: u64) -> u64 {
    let mut mask = 0_u64;
    while mask.count_ones() < bit_count as u32 {
        mask |= 1_u64 << (rng.next_u64() % 64);
    }
    mask
}

fn low_similarity_mask(rng: &mut SplitMix64) -> u64 {
    loop {
        let mask = rng.next_u64();
        let distance = mask.count_ones();
        if (28..=36).contains(&distance) {
            return mask;
        }
    }
}

fn unique_value(rng: &mut SplitMix64, used: &mut BTreeSet<u64>) -> u64 {
    loop {
        let candidate = rng.next_u64();
        if used.insert(candidate) {
            return candidate;
        }
    }
}

fn stream_id(config: DelayedCueConfig) -> u64 {
    let mut value = config.seed ^ config.delay_steps.rotate_left(17);
    value ^= (config.distractor_count as u64).rotate_left(31);
    value ^= match config.similarity {
        DistractorSimilarity::High => 0x4849_4748_5F53_494D,
        DistractorSimilarity::Low => 0x4C4F_575F_5349_4D00,
    };
    mix64(value)
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(similarity: DistractorSimilarity) -> DelayedCueConfig {
        DelayedCueConfig {
            seed: 0xD1CE_BA5E_2026_1031,
            delay_steps: 64,
            distractor_count: 32,
            similarity,
        }
    }

    fn associations(task: &PublicTask) -> Vec<(u64, u64, u64)> {
        task.events
            .iter()
            .filter_map(|event| match &event.kind {
                EventKind::Associate { key, value } => Some((event.step, *key, *value)),
                EventKind::Probe { .. } => None,
            })
            .collect()
    }

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let first = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        let second = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn oracle_record_has_a_stable_sha256_digest() {
        let (_, oracle) = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        assert!(delayed_cue_oracle_digest_valid(&oracle));
    }

    #[test]
    fn oracle_digest_detects_target_value_mutation() {
        let (_, mut oracle) = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        oracle.expected_value ^= 1;
        assert!(!delayed_cue_oracle_digest_valid(&oracle));
    }

    #[test]
    fn generated_delayed_cue_task_passes_full_verifier() {
        let (task, oracle) = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        assert_eq!(verify_delayed_cue_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn verifier_rejects_recommitted_invalid_distractor() {
        let (mut task, oracle) = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        if let EventKind::Associate { key, .. } = &mut task.events[1].kind {
            *key = oracle.target_key;
        }
        task.content_sha256 = delayed_cue_content_digest(&task);
        assert!(matches!(
            verify_delayed_cue_trace(&task, &oracle),
            Err(DelayedCueVerificationError::InvalidDistractor { .. })
        ));
    }

    #[test]
    fn verifier_rejects_recommitted_wrong_probe() {
        let (mut task, oracle) = generate_delayed_cue(config(DistractorSimilarity::Low)).unwrap();
        if let Some(last) = task.events.last_mut() {
            if let EventKind::Probe { key } = &mut last.kind {
                *key ^= 1;
            }
        }
        task.content_sha256 = delayed_cue_content_digest(&task);
        assert_eq!(
            verify_delayed_cue_trace(&task, &oracle),
            Err(DelayedCueVerificationError::InvalidProbe)
        );
    }

    #[test]
    fn public_task_has_a_stable_sha256_digest() {
        let (task, _) = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        assert!(delayed_cue_digest_valid(&task));
        assert_eq!(
            task.content_sha256,
            delayed_cue_content_digest(&task)
        );
    }

    #[test]
    fn public_task_digest_detects_event_mutation() {
        let (mut task, _) = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        task.events[0].ordinal ^= 1;
        assert!(!delayed_cue_digest_valid(&task));
    }

    #[test]
    fn target_binding_precedes_probe_by_declared_delay() {
        let (task, oracle) =
            generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        let target = task
            .events
            .iter()
            .find(|event| {
                matches!(
                    &event.kind,
                    EventKind::Associate { key, value }
                        if *key == oracle.target_key && *value == oracle.expected_value
                )
            })
            .unwrap();
        let probe = task
            .events
            .iter()
            .find(|event| matches!(&event.kind, EventKind::Probe { .. }))
            .unwrap();
        assert_eq!(target.step, 0);
        assert_eq!(probe.step, task.delay_steps);
        assert_eq!(probe.step - target.step, task.delay_steps);
        assert!(matches!(
            &probe.kind,
            EventKind::Probe { key } if *key == oracle.target_key
        ));
    }

    #[test]
    fn high_similarity_distractors_are_within_four_bits() {
        let (task, oracle) =
            generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        let distractors = associations(&task)
            .into_iter()
            .filter(|(_, key, _)| *key != oracle.target_key);
        for (_, key, _) in distractors {
            let distance = (key ^ oracle.target_key).count_ones();
            assert!((1..=4).contains(&distance));
        }
    }

    #[test]
    fn low_similarity_distractors_are_between_twenty_eight_and_thirty_six_bits() {
        let (task, oracle) =
            generate_delayed_cue(config(DistractorSimilarity::Low)).unwrap();
        let distractors = associations(&task)
            .into_iter()
            .filter(|(_, key, _)| *key != oracle.target_key);
        for (_, key, _) in distractors {
            let distance = (key ^ oracle.target_key).count_ones();
            assert!((28..=36).contains(&distance));
        }
    }

    #[test]
    fn all_binding_keys_and_values_are_unique() {
        let (task, _) = generate_delayed_cue(config(DistractorSimilarity::High)).unwrap();
        let rows = associations(&task);
        let keys: BTreeSet<_> = rows.iter().map(|(_, key, _)| *key).collect();
        let values: BTreeSet<_> = rows.iter().map(|(_, _, value)| *value).collect();
        assert_eq!(keys.len(), rows.len());
        assert_eq!(values.len(), rows.len());
    }

    #[test]
    fn event_order_is_monotonic_and_probe_is_last() {
        let (task, _) = generate_delayed_cue(config(DistractorSimilarity::Low)).unwrap();
        let mut previous = (0_u64, 0_u64);
        for (index, event) in task.events.iter().enumerate() {
            let current = (event.step, event.ordinal);
            if index > 0 {
                assert!(current > previous);
            }
            previous = current;
        }
        assert!(matches!(
            task.events.last().map(|event| &event.kind),
            Some(EventKind::Probe { .. })
        ));
    }

    #[test]
    fn one_step_delay_keeps_distractors_before_probe_by_ordinal() {
        let mut cfg = config(DistractorSimilarity::High);
        cfg.delay_steps = 1;
        let (task, _) = generate_delayed_cue(cfg).unwrap();
        assert!(task.events[..task.events.len() - 1]
            .iter()
            .all(|event| event.step <= 1));
        assert!(matches!(
            task.events.last().map(|event| &event.kind),
            Some(EventKind::Probe { .. })
        ));
    }

    #[test]
    fn zero_delay_is_rejected() {
        let mut cfg = config(DistractorSimilarity::High);
        cfg.delay_steps = 0;
        assert_eq!(
            generate_delayed_cue(cfg),
            Err(GenerationError::ZeroDelay)
        );
    }

    #[test]
    fn excessive_distractor_count_is_rejected() {
        let mut cfg = config(DistractorSimilarity::High);
        cfg.distractor_count = MAX_DISTRACTORS + 1;
        assert_eq!(
            generate_delayed_cue(cfg),
            Err(GenerationError::TooManyDistractors {
                requested: MAX_DISTRACTORS + 1,
                maximum: MAX_DISTRACTORS,
            })
        );
    }
}
