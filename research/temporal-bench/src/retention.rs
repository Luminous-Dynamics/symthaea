//! Deterministic selective-retention and invalidation workload.
//!
//! A target association must survive distractor interference, then become
//! unavailable after an explicit invalidation event. The candidate sees the
//! store/invalidation/query stream; the oracle contains the correct response
//! for each query. Both are returned together for prototype use only.

use super::{digest::CanonicalHasher, mix64, DistractorSimilarity, SplitMix64};
use std::collections::BTreeSet;
use std::fmt;

const MAX_RETENTION_DISTRACTORS: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionConfig {
    pub seed: u64,
    pub distractor_count: usize,
    pub delay_steps: u64,
    pub similarity: DistractorSimilarity,
}

impl RetentionConfig {
    pub fn reference(seed: u64) -> Self {
        Self {
            seed,
            distractor_count: 64,
            delay_steps: 512,
            similarity: DistractorSimilarity::High,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetentionEventKind {
    Store {
        record_id: u64,
        key: u64,
        value: u64,
    },
    Invalidate {
        record_id: u64,
    },
    Query {
        query_id: u64,
        key: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionEvent {
    pub step: u64,
    pub ordinal: u32,
    pub kind: RetentionEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionTask {
    pub stream_id: u64,
    pub delay_steps: u64,
    pub distractor_count: usize,
    pub similarity: DistractorSimilarity,
    pub events: Vec<RetentionEvent>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionExpectedAnswer {
    pub query_id: u64,
    pub key: u64,
    pub expected_value: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionOracle {
    pub stream_id: u64,
    pub target_record_id: u64,
    pub target_key: u64,
    pub target_value: u64,
    pub invalidated_at_step: u64,
    pub answers: Vec<RetentionExpectedAnswer>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetentionGenerationError {
    ZeroDelay,
    TooManyDistractors { requested: usize, maximum: usize },
    DelayOverflow,
}

impl fmt::Display for RetentionGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDelay => write!(f, "delay_steps must be greater than zero"),
            Self::TooManyDistractors { requested, maximum } => {
                write!(f, "distractor_count {requested} exceeds maximum {maximum}")
            }
            Self::DelayOverflow => write!(f, "retention task timeline overflows u64"),
        }
    }
}

impl std::error::Error for RetentionGenerationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetentionVerificationError {
    StreamIdMismatch,
    PublicDigestMismatch,
    OracleDigestMismatch,
    InvalidEventCount,
    InvalidTargetStore,
    InvalidDistractor { index: usize },
    InvalidQuery { query_id: u64 },
    InvalidInvalidation,
    InvalidOracleAnswer { query_id: u64 },
    InvalidEventOrder { index: usize },
}

impl fmt::Display for RetentionVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "retention task and oracle stream IDs differ"),
            Self::PublicDigestMismatch => write!(f, "retention public SHA-256 digest mismatch"),
            Self::OracleDigestMismatch => write!(f, "retention oracle SHA-256 digest mismatch"),
            Self::InvalidEventCount => write!(f, "retention event count is inconsistent"),
            Self::InvalidTargetStore => write!(f, "target store event is missing or invalid"),
            Self::InvalidDistractor { index } => {
                write!(f, "retention distractor {index} violates the task contract")
            }
            Self::InvalidQuery { query_id } => {
                write!(f, "retention query {query_id} is missing or invalid")
            }
            Self::InvalidInvalidation => write!(f, "target invalidation is missing or mistimed"),
            Self::InvalidOracleAnswer { query_id } => {
                write!(f, "oracle answer for query {query_id} violates validity semantics")
            }
            Self::InvalidEventOrder { index } => {
                write!(f, "retention event {index} is not strictly ordered")
            }
        }
    }
}

impl std::error::Error for RetentionVerificationError {}

/// Generate a target/distractor/query/invalidate/query stream.
///
/// The first query occurs after one delay interval and must recall the stored
/// target. The target is then invalidated. The second query occurs after
/// invalidation and must return None, even though the original value appeared
/// earlier in the stream.
pub fn generate_retention_task(
    config: RetentionConfig,
) -> Result<(RetentionTask, RetentionOracle), RetentionGenerationError> {
    if config.delay_steps == 0 {
        return Err(RetentionGenerationError::ZeroDelay);
    }
    if config.distractor_count > MAX_RETENTION_DISTRACTORS {
        return Err(RetentionGenerationError::TooManyDistractors {
            requested: config.distractor_count,
            maximum: MAX_RETENTION_DISTRACTORS,
        });
    }
    let distractor_count =
        u64::try_from(config.distractor_count).unwrap_or(u64::MAX);
    let first_query_step = distractor_count
        .checked_add(1)
        .and_then(|step| step.checked_mul(config.delay_steps))
        .ok_or(RetentionGenerationError::DelayOverflow)?;
    let second_query_step = first_query_step
        .checked_add(config.delay_steps)
        .ok_or(RetentionGenerationError::DelayOverflow)?;
    let invalidated_at_step = second_query_step
        .checked_sub(1)
        .ok_or(RetentionGenerationError::DelayOverflow)?;

    let stream_id = retention_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x5245_5445_4E54_494F);
    let target_record_id = 1_u64;
    let target_key = rng.next_u64();
    let target_value = rng.next_u64();
    let mut used_keys = BTreeSet::new();
    let mut used_values = BTreeSet::new();
    used_keys.insert(target_key);
    used_values.insert(target_value);

    let mut events = Vec::with_capacity(config.distractor_count.saturating_add(4));
    events.push(RetentionEvent {
        step: 0,
        ordinal: 0,
        kind: RetentionEventKind::Store {
            record_id: target_record_id,
            key: target_key,
            value: target_value,
        },
    });

    for index in 0..config.distractor_count {
        let key = unique_distractor_key(
            &mut rng,
            target_key,
            config.similarity,
            &mut used_keys,
        );
        let value = unique_value(&mut rng, &mut used_values);
        let ordinal = u32::try_from(index).unwrap_or(u32::MAX);
        let step = u64::try_from(index)
            .unwrap_or(u64::MAX)
            .saturating_add(1)
            .saturating_mul(config.delay_steps);
        events.push(RetentionEvent {
            step,
            ordinal,
            kind: RetentionEventKind::Store {
                record_id: u64::try_from(index).unwrap_or(u64::MAX).saturating_add(2),
                key,
                value,
            },
        });
    }

    let invalidation_step = invalidated_at_step;
    let post_invalidation_query_step = second_query_step;
    events.push(RetentionEvent {
        step: first_query_step,
        ordinal: 0,
        kind: RetentionEventKind::Query {
            query_id: 1,
            key: target_key,
        },
    });
    events.push(RetentionEvent {
        step: invalidation_step,
        ordinal: 0,
        kind: RetentionEventKind::Invalidate {
            record_id: target_record_id,
        },
    });
    events.push(RetentionEvent {
        step: post_invalidation_query_step,
        ordinal: 0,
        kind: RetentionEventKind::Query {
            query_id: 2,
            key: target_key,
        },
    });

    // Stable chronological order; the source event ordinal disambiguates any
    // identical timestamps in future task variants.
    events.sort_by_key(|event| (event.step, event.ordinal));

    let mut task = RetentionTask {
        stream_id,
        delay_steps: config.delay_steps,
        distractor_count: config.distractor_count,
        similarity: config.similarity,
        events,
        content_sha256: [0; 32],
    };
    task.content_sha256 = retention_content_digest(&task);

    let mut oracle = RetentionOracle {
        stream_id,
        target_record_id,
        target_key,
        target_value,
        invalidated_at_step,
        answers: vec![
            RetentionExpectedAnswer {
                query_id: 1,
                key: target_key,
                expected_value: Some(target_value),
            },
            RetentionExpectedAnswer {
                query_id: 2,
                key: target_key,
                expected_value: None,
            },
        ],
        content_sha256: [0; 32],
    };
    oracle.content_sha256 = retention_oracle_digest(&oracle);

    Ok((task, oracle))
}

pub fn retention_content_digest(task: &RetentionTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.retention.public.v1");
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
        hash.u32(event.ordinal);
        match &event.kind {
            RetentionEventKind::Store {
                record_id,
                key,
                value,
            } => {
                hash.u8(0);
                hash.u64(*record_id);
                hash.u64(*key);
                hash.u64(*value);
            }
            RetentionEventKind::Invalidate { record_id } => {
                hash.u8(1);
                hash.u64(*record_id);
            }
            RetentionEventKind::Query { query_id, key } => {
                hash.u8(2);
                hash.u64(*query_id);
                hash.u64(*key);
            }
        }
    }
    hash.finish()
}

pub fn retention_oracle_digest(oracle: &RetentionOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.retention.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u64(oracle.target_record_id);
    hash.u64(oracle.target_key);
    hash.u64(oracle.target_value);
    hash.u64(oracle.invalidated_at_step);
    hash.usize(oracle.answers.len());
    for answer in &oracle.answers {
        hash.u64(answer.query_id);
        hash.u64(answer.key);
        match answer.expected_value {
            Some(value) => {
                hash.u8(1);
                hash.u64(value);
            }
            None => hash.u8(0),
        }
    }
    hash.finish()
}

pub fn verify_retention_trace(
    task: &RetentionTask,
    oracle: &RetentionOracle,
) -> Result<(), RetentionVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(RetentionVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != retention_content_digest(task) {
        return Err(RetentionVerificationError::PublicDigestMismatch);
    }
    if oracle.content_sha256 != retention_oracle_digest(oracle) {
        return Err(RetentionVerificationError::OracleDigestMismatch);
    }
    if task.events.len() != task.distractor_count.saturating_add(4)
        || task.delay_steps == 0
        || task.distractor_count > MAX_RETENTION_DISTRACTORS
        || oracle.target_record_id != 1
        || oracle.answers.len() != 2
    {
        return Err(RetentionVerificationError::InvalidEventCount);
    }

    let expected_first_query = u64::try_from(task.distractor_count)
        .unwrap_or(u64::MAX)
        .saturating_add(1)
        .checked_mul(task.delay_steps)
        .ok_or(RetentionVerificationError::InvalidEventCount)?;
    let expected_post_query = expected_first_query
        .checked_add(task.delay_steps)
        .ok_or(RetentionVerificationError::InvalidEventCount)?;
    let expected_invalidation = expected_post_query.saturating_sub(1);
    if oracle.invalidated_at_step != expected_invalidation {
        return Err(RetentionVerificationError::InvalidInvalidation);
    }

    let first = task.events.first().ok_or(
        RetentionVerificationError::InvalidTargetStore,
    )?;
    match &first.kind {
        RetentionEventKind::Store {
            record_id,
            key,
            value,
        } if first.step == 0
            && *record_id == oracle.target_record_id
            && *key == oracle.target_key
            && *value == oracle.target_value => {}
        _ => return Err(RetentionVerificationError::InvalidTargetStore),
    }

    let all_invalidation_events: Vec<_> = task
        .events
        .iter()
        .filter_map(|event| match &event.kind {
            RetentionEventKind::Invalidate { record_id } => {
                Some((event.step, *record_id))
            }
            _ => None,
        })
        .collect();
    let invalidation_count = all_invalidation_events
        .iter()
        .filter(|(_, record_id)| *record_id == oracle.target_record_id)
        .count();
    let actual_invalidation = task.events.iter().find_map(|event| match &event.kind {
        RetentionEventKind::Invalidate { record_id }
            if *record_id == oracle.target_record_id => Some(event.step),
        _ => None,
    });
    if invalidation_count != 1
        || all_invalidation_events.len() != 1
        || actual_invalidation != Some(expected_invalidation)
    {
        return Err(RetentionVerificationError::InvalidInvalidation);
    }

    let queries: Vec<_> = task
        .events
        .iter()
        .filter_map(|event| match &event.kind {
            RetentionEventKind::Query { query_id, key } => {
                Some((event.step, *query_id, *key))
            }
            _ => None,
        })
        .collect();
    if queries.len() != 2
        || queries[0] != (expected_first_query, 1, oracle.target_key)
        || queries[1] != (expected_post_query, 2, oracle.target_key)
    {
        return Err(RetentionVerificationError::InvalidQuery { query_id: 0 });
    }

    if oracle.answers[0]
        != (RetentionExpectedAnswer {
            query_id: 1,
            key: oracle.target_key,
            expected_value: Some(oracle.target_value),
        })
        || oracle.answers[1]
            != (RetentionExpectedAnswer {
                query_id: 2,
                key: oracle.target_key,
                expected_value: None,
            })
    {
        return Err(RetentionVerificationError::InvalidOracleAnswer { query_id: 0 });
    }

    let mut seen_ids = BTreeSet::new();
    let mut seen_keys = BTreeSet::new();
    let mut seen_values = BTreeSet::new();
    seen_ids.insert(oracle.target_record_id);
    seen_keys.insert(oracle.target_key);
    seen_values.insert(oracle.target_value);
    let mut distractor_index = 0_usize;
    for (index, event) in task.events.iter().enumerate().skip(1) {
        if index > 0 {
            let previous = &task.events[index - 1];
            if (event.step, event.ordinal) <= (previous.step, previous.ordinal) {
                return Err(RetentionVerificationError::InvalidEventOrder { index });
            }
        }
        if let RetentionEventKind::Store {
            record_id,
            key,
            value,
        } = &event.kind
        {
            let distance = (*key ^ oracle.target_key).count_ones();
            let similarity_valid = match task.similarity {
                DistractorSimilarity::High => (1..=4).contains(&distance),
                DistractorSimilarity::Low => (28..=36).contains(&distance),
            };
            let expected_step = u64::try_from(distractor_index)
                .unwrap_or(u64::MAX)
                .saturating_add(1)
                .saturating_mul(task.delay_steps);
            let expected_id = u64::try_from(distractor_index)
                .unwrap_or(u64::MAX)
                .saturating_add(2);
            if *record_id != expected_id
                || event.step != expected_step
                || event.step >= expected_first_query
                || !similarity_valid
                || !seen_ids.insert(*record_id)
                || !seen_keys.insert(*key)
                || !seen_values.insert(*value)
            {
                return Err(RetentionVerificationError::InvalidDistractor {
                    index: distractor_index,
                });
            }
            distractor_index = distractor_index.saturating_add(1);
        }
    }
    if distractor_index != task.distractor_count {
        return Err(RetentionVerificationError::InvalidDistractor {
            index: distractor_index,
        });
    }
    for (index, event) in task.events.iter().enumerate() {
        if index > 0 {
            let previous = &task.events[index - 1];
            if (event.step, event.ordinal) <= (previous.step, previous.ordinal) {
                return Err(RetentionVerificationError::InvalidEventOrder { index });
            }
        }
    }
    Ok(())
}

fn retention_stream_id(config: RetentionConfig) -> u64 {
    let mut value = config.seed ^ config.delay_steps.rotate_left(19);
    value ^= (config.distractor_count as u64).rotate_left(37);
    value ^= match config.similarity {
        DistractorSimilarity::High => 0x4849_4748_5F52_4554,
        DistractorSimilarity::Low => 0x4C4F_575F_5245_5400,
    };
    mix64(value)
}

fn unique_distractor_key(
    rng: &mut SplitMix64,
    target: u64,
    similarity: DistractorSimilarity,
    used: &mut BTreeSet<u64>,
) -> u64 {
    loop {
        let bit_count = match similarity {
            DistractorSimilarity::High => 1 + rng.next_u64() % 4,
            DistractorSimilarity::Low => 28 + rng.next_u64() % 9,
        };
        let mut mask = 0_u64;
        while mask.count_ones() < bit_count as u32 {
            mask |= 1_u64 << (rng.next_u64() % 64);
        }
        let candidate = target ^ mask;
        if used.insert(candidate) {
            return candidate;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config(similarity: DistractorSimilarity) -> RetentionConfig {
        RetentionConfig {
            seed: 0x5E1E_C71E_2026_1031,
            distractor_count: 32,
            delay_steps: 16,
            similarity,
        }
    }

    #[test]
    fn generation_is_deterministic_for_fixed_seed() {
        let cfg = config(DistractorSimilarity::High);
        assert_eq!(generate_retention_task(cfg), generate_retention_task(cfg));
    }

    #[test]
    fn generated_task_passes_full_retention_verifier() {
        let (task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        assert_eq!(verify_retention_trace(&task, &oracle), Ok(()));
    }

    #[test]
    fn first_query_retains_target_and_second_query_expires_it() {
        let (task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        assert_eq!(oracle.answers[0].expected_value, Some(oracle.target_value));
        assert_eq!(oracle.answers[1].expected_value, None);
        assert!(task.events.iter().any(|event| matches!(
            &event.kind,
            RetentionEventKind::Invalidate { record_id: 1 }
        )));
    }

    #[test]
    fn high_similarity_distractors_are_within_four_bits() {
        let (task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        for event in &task.events {
            if let RetentionEventKind::Store { key, record_id, .. } = &event.kind {
                if *record_id != oracle.target_record_id {
                    assert!((1..=4).contains(&(*key ^ oracle.target_key).count_ones()));
                }
            }
        }
    }

    #[test]
    fn low_similarity_distractors_are_28_to_36_bits_away() {
        let (task, oracle) =
            generate_retention_task(config(DistractorSimilarity::Low)).unwrap();
        for event in &task.events {
            if let RetentionEventKind::Store { key, record_id, .. } = &event.kind {
                if *record_id != oracle.target_record_id {
                    assert!((28..=36).contains(&(*key ^ oracle.target_key).count_ones()));
                }
            }
        }
    }

    #[test]
    fn public_and_oracle_digests_are_valid() {
        let (task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        assert_eq!(task.content_sha256, retention_content_digest(&task));
        assert_eq!(oracle.content_sha256, retention_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_post_generation_public_mutation() {
        let (mut task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        task.events[0].step = 1;
        assert_eq!(
            verify_retention_trace(&task, &oracle),
            Err(RetentionVerificationError::PublicDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_post_generation_oracle_mutation() {
        let (task, mut oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        oracle.answers[1].expected_value = Some(oracle.target_value);
        assert_eq!(
            verify_retention_trace(&task, &oracle),
            Err(RetentionVerificationError::OracleDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_recommitted_stale_recall_answer() {
        let (task, mut oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        oracle.answers[1].expected_value = Some(oracle.target_value);
        oracle.content_sha256 = retention_oracle_digest(&oracle);
        assert_eq!(
            verify_retention_trace(&task, &oracle),
            Err(RetentionVerificationError::InvalidOracleAnswer { query_id: 0 })
        );
    }

    #[test]
    fn verifier_rejects_recommitted_misplaced_invalidation() {
        let (mut task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        if let Some(event) = task.events.iter_mut().find(|event| matches!(
            &event.kind,
            RetentionEventKind::Invalidate { .. }
        )) {
            event.step = event.step.saturating_add(1);
        }
        task.content_sha256 = retention_content_digest(&task);
        assert_eq!(
            verify_retention_trace(&task, &oracle),
            Err(RetentionVerificationError::InvalidInvalidation)
        );
    }

    #[test]
    fn verifier_rejects_extra_invalidation_for_unknown_record() {
        let (mut task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        let insertion_step = task.events[task.events.len() - 1].step.saturating_add(1);
        task.events.insert(
            task.events.len() - 1,
            RetentionEvent {
                step: insertion_step,
                ordinal: 0,
                kind: RetentionEventKind::Invalidate { record_id: 999 },
            },
        );
        task.content_sha256 = retention_content_digest(&task);
        assert_eq!(
            verify_retention_trace(&task, &oracle),
            Err(RetentionVerificationError::InvalidEventCount)
        );
    }

    #[test]
    fn verifier_rejects_missing_invalidation() {
        let (mut task, oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        let index = task.events.iter().position(|event| matches!(
            &event.kind,
            RetentionEventKind::Invalidate { .. }
        )).unwrap();
        task.events.remove(index);
        task.content_sha256 = retention_content_digest(&task);
        assert_eq!(
            verify_retention_trace(&task, &oracle),
            Err(RetentionVerificationError::InvalidEventCount)
        );
    }

    #[test]
    fn verifier_rejects_wrong_first_query_value() {
        let (task, mut oracle) =
            generate_retention_task(config(DistractorSimilarity::High)).unwrap();
        oracle.answers[0].expected_value = None;
        oracle.content_sha256 = retention_oracle_digest(&oracle);
        assert_eq!(
            verify_retention_trace(&task, &oracle),
            Err(RetentionVerificationError::InvalidOracleAnswer { query_id: 0 })
        );
    }

    #[test]
    fn zero_delay_is_rejected() {
        let mut cfg = config(DistractorSimilarity::High);
        cfg.delay_steps = 0;
        assert_eq!(
            generate_retention_task(cfg),
            Err(RetentionGenerationError::ZeroDelay)
        );
    }

    #[test]
    fn excessive_distractor_count_is_rejected() {
        let mut cfg = config(DistractorSimilarity::High);
        cfg.distractor_count = MAX_RETENTION_DISTRACTORS + 1;
        assert_eq!(
            generate_retention_task(cfg),
            Err(RetentionGenerationError::TooManyDistractors {
                requested: MAX_RETENTION_DISTRACTORS + 1,
                maximum: MAX_RETENTION_DISTRACTORS
            })
        );
    }
}
