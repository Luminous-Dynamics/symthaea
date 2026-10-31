//! Bitemporal continual-knowledge stream with retroactive corrections.
//!
//! Each source revision has an observed/transaction time (when the candidate
//! receives it) and a valid-from time (when it applies in the modeled world).
//! Queries specify both valid_at and known_at. This is a synthetic benchmark
//! adapter to the bitemporal contracts in issues #9075/#9427, not a replacement
//! for the production proof-DAG or persistent ledger.

use super::{digest::CanonicalHasher, mix64, SplitMix64};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAX_KNOWLEDGE_DISTRACTORS: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgeStreamConfig {
    pub seed: u64,
    pub distractor_count: usize,
    pub interval_steps: u64,
}

impl KnowledgeStreamConfig {
    pub fn reference(seed: u64) -> Self {
        Self {
            seed,
            distractor_count: 64,
            interval_steps: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnowledgeOperation {
    Assert,
    Correct,
    Retract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeEventKind {
    Version {
        fact_id: u64,
        entity_id: u64,
        predicate_id: u32,
        revision: u32,
        valid_from: u64,
        operation: KnowledgeOperation,
        value: Option<i64>,
    },
    Query {
        query_id: u64,
        entity_id: u64,
        predicate_id: u32,
        valid_at: u64,
        known_at: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeEvent {
    /// Transaction/observation time: when this event reaches the candidate.
    pub observed_at: u64,
    /// Strict ordering for multiple events in the same transaction batch.
    pub ordinal: u32,
    pub kind: KnowledgeEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeStreamTask {
    pub stream_id: u64,
    pub target_entity_id: u64,
    pub target_predicate_id: u32,
    pub distractor_count: usize,
    pub interval_steps: u64,
    pub query_issued_at: u64,
    pub events: Vec<KnowledgeEvent>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeExpectedAnswer {
    pub query_id: u64,
    pub valid_at: u64,
    pub known_at: u64,
    pub expected_value: Option<i64>,
    pub expected_revision: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeStreamOracle {
    pub stream_id: u64,
    pub target_fact_id: u64,
    pub target_entity_id: u64,
    pub target_predicate_id: u32,
    pub initial_value: i64,
    pub corrected_value: i64,
    pub correction_observed_at: u64,
    pub correction_valid_from: u64,
    pub retraction_observed_at: u64,
    pub retraction_valid_from: u64,
    pub answers: Vec<KnowledgeExpectedAnswer>,
    pub content_sha256: [u8; 32],
}

/// Candidate-visible answer states. `NoFact` is a substantive answer;
/// `Abstain` means the candidate declined to provide an answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnowledgeCandidateAnswer {
    Value(i64),
    NoFact,
    Abstain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgePrediction {
    pub query_id: u64,
    pub answer: KnowledgeCandidateAnswer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeScoringError {
    DuplicatePrediction { query_id: u64 },
    UnknownQuery { query_id: u64 },
}

impl fmt::Display for KnowledgeScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePrediction { query_id } => {
                write!(f, "candidate submitted duplicate prediction for query {query_id}")
            }
            Self::UnknownQuery { query_id } => {
                write!(f, "candidate submitted prediction for unknown query {query_id}")
            }
        }
    }
}

impl std::error::Error for KnowledgeScoringError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeScore {
    pub query_count: usize,
    pub predictions_received: usize,
    pub exact_matches: usize,
    pub wrong_answers: usize,
    pub abstentions: usize,
    pub missing_predictions: usize,
    pub future_knowledge_leaks: usize,
    pub missed_corrections: usize,
    pub stale_answers_after_retraction: usize,
    pub historical_as_known_preserved: usize,
}

impl KnowledgeScore {
    pub fn exact_match_rate(&self) -> Option<f64> {
        (self.query_count > 0)
            .then(|| self.exact_matches as f64 / self.query_count as f64)
    }

    pub fn prediction_coverage(&self) -> Option<f64> {
        (self.query_count > 0)
            .then(|| self.predictions_received as f64 / self.query_count as f64)
    }
}

/// Score per-query outputs without collapsing the temporal failure modes.
///
/// A missing prediction is not silently converted into a correct abstention.
/// Returning `NoFact` is distinct from abstaining: after a known retraction it
/// is the correct result; for a still-valid fact it is an incorrect answer.
pub fn score_knowledge_predictions(
    oracle: &KnowledgeStreamOracle,
    predictions: &[KnowledgePrediction],
) -> Result<KnowledgeScore, KnowledgeScoringError> {
    let known_queries: BTreeSet<_> = oracle.answers.iter().map(|answer| answer.query_id).collect();
    let mut by_query = BTreeMap::new();
    for prediction in predictions {
        if !known_queries.contains(&prediction.query_id) {
            return Err(KnowledgeScoringError::UnknownQuery {
                query_id: prediction.query_id,
            });
        }
        if by_query.insert(prediction.query_id, prediction.answer).is_some() {
            return Err(KnowledgeScoringError::DuplicatePrediction {
                query_id: prediction.query_id,
            });
        }
    }

    let mut score = KnowledgeScore {
        query_count: oracle.answers.len(),
        predictions_received: by_query.len(),
        exact_matches: 0,
        wrong_answers: 0,
        abstentions: 0,
        missing_predictions: 0,
        future_knowledge_leaks: 0,
        missed_corrections: 0,
        stale_answers_after_retraction: 0,
        historical_as_known_preserved: 0,
    };

    for expected in &oracle.answers {
        let Some(prediction) = by_query.get(&expected.query_id).copied() else {
            score.missing_predictions = score.missing_predictions.saturating_add(1);
            continue;
        };

        match prediction {
            KnowledgeCandidateAnswer::Abstain => {
                score.abstentions = score.abstentions.saturating_add(1);
            }
            KnowledgeCandidateAnswer::NoFact => {
                if expected.expected_value.is_none() {
                    score.exact_matches = score.exact_matches.saturating_add(1);
                } else {
                    score.wrong_answers = score.wrong_answers.saturating_add(1);
                }
            }
            KnowledgeCandidateAnswer::Value(value) => {
                if expected.expected_value == Some(value) {
                    score.exact_matches = score.exact_matches.saturating_add(1);
                } else {
                    score.wrong_answers = score.wrong_answers.saturating_add(1);
                }

                if expected.known_at < oracle.correction_observed_at
                    && expected.valid_at >= oracle.correction_valid_from
                    && value == oracle.corrected_value
                {
                    score.future_knowledge_leaks =
                        score.future_knowledge_leaks.saturating_add(1);
                }
                if expected.known_at >= oracle.correction_observed_at
                    && expected.valid_at >= oracle.correction_valid_from
                    && value == oracle.initial_value
                {
                    score.missed_corrections = score.missed_corrections.saturating_add(1);
                }
                if expected.known_at >= oracle.retraction_observed_at
                    && expected.valid_at >= oracle.retraction_valid_from
                {
                    score.stale_answers_after_retraction =
                        score.stale_answers_after_retraction.saturating_add(1);
                }
                if expected.known_at >= oracle.correction_observed_at
                    && expected.valid_at < oracle.correction_valid_from
                    && value == oracle.initial_value
                {
                    score.historical_as_known_preserved =
                        score.historical_as_known_preserved.saturating_add(1);
                }
            }
        }
    }

    Ok(score)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeGenerationError {
    ZeroInterval,
    TooManyDistractors { requested: usize, maximum: usize },
    TimelineOverflow,
}

impl fmt::Display for KnowledgeGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInterval => write!(f, "interval_steps must be greater than zero"),
            Self::TooManyDistractors { requested, maximum } => {
                write!(f, "distractor_count {requested} exceeds maximum {maximum}")
            }
            Self::TimelineOverflow => write!(f, "knowledge-stream timeline overflows u64"),
        }
    }
}

impl std::error::Error for KnowledgeGenerationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeVerificationError {
    StreamIdMismatch,
    PublicDigestMismatch,
    OracleDigestMismatch,
    SeedReplayMismatch,
    InvalidEventOrder { index: usize },
    InvalidRevision { fact_id: u64, revision: u32 },
    InvalidOperation { fact_id: u64, revision: u32 },
    InvalidQuery { query_id: u64 },
    InvalidAnswer { query_id: u64 },
    InvalidTimeline,
}

impl fmt::Display for KnowledgeVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamIdMismatch => write!(f, "knowledge task/oracle stream IDs differ"),
            Self::PublicDigestMismatch => write!(f, "knowledge public SHA-256 digest mismatch"),
            Self::OracleDigestMismatch => write!(f, "knowledge oracle SHA-256 digest mismatch"),
            Self::SeedReplayMismatch => write!(f, "knowledge task/oracle differ from evaluator-held seed/configuration"),
            Self::InvalidEventOrder { index } => write!(f, "knowledge event {index} is not strictly ordered"),
            Self::InvalidRevision { fact_id, revision } => {
                write!(f, "fact {fact_id} has invalid or non-monotonic revision {revision}")
            }
            Self::InvalidOperation { fact_id, revision } => {
                write!(f, "fact {fact_id} revision {revision} has an invalid operation/value pair")
            }
            Self::InvalidQuery { query_id } => write!(f, "knowledge query {query_id} is invalid"),
            Self::InvalidAnswer { query_id } => write!(f, "bitemporal oracle answer for query {query_id} is incorrect"),
            Self::InvalidTimeline => write!(f, "knowledge stream timeline is invalid"),
        }
    }
}

impl std::error::Error for KnowledgeVerificationError {}

/// Build an OAKS-inspired fact stream with an assertion, late correction,
/// later retraction, distractor facts, and paired (valid_at, known_at) queries.
///
/// The correction and retraction arrive long after their valid-time boundary.
/// This exposes the key failure modes: backdating newly learned facts into
/// historical knowledge, failing to apply a retroactive correction, and using
/// a retracted fact after the retraction becomes known.
pub fn generate_knowledge_stream(
    config: KnowledgeStreamConfig,
) -> Result<(KnowledgeStreamTask, KnowledgeStreamOracle), KnowledgeGenerationError> {
    if config.interval_steps == 0 {
        return Err(KnowledgeGenerationError::ZeroInterval);
    }
    if config.distractor_count > MAX_KNOWLEDGE_DISTRACTORS {
        return Err(KnowledgeGenerationError::TooManyDistractors {
            requested: config.distractor_count,
            maximum: MAX_KNOWLEDGE_DISTRACTORS,
        });
    }
    let count = u64::try_from(config.distractor_count).unwrap_or(u64::MAX);
    let correction_observed_at = count
        .checked_add(1)
        .and_then(|value| value.checked_mul(config.interval_steps))
        .ok_or(KnowledgeGenerationError::TimelineOverflow)?;
    let retraction_observed_at = correction_observed_at
        .checked_add(config.interval_steps)
        .ok_or(KnowledgeGenerationError::TimelineOverflow)?;
    let query_issued_at = retraction_observed_at
        .checked_add(config.interval_steps)
        .ok_or(KnowledgeGenerationError::TimelineOverflow)?;
    let correction_valid_from = config.interval_steps
        .checked_mul(2)
        .ok_or(KnowledgeGenerationError::TimelineOverflow)?;
    let retraction_valid_from = config.interval_steps
        .checked_mul(3)
        .ok_or(KnowledgeGenerationError::TimelineOverflow)?;

    let stream_id = knowledge_stream_id(config);
    let mut rng = SplitMix64::new(config.seed ^ 0x4B4E_4F57_4C45_4447);
    let target_entity_id = rng.next_u64();
    let target_predicate_id = 7_u32;
    let target_fact_id = 1_u64;
    let initial_value = 10_001_i64
        .saturating_add(i64::try_from(rng.next_u64() % 1_000_000).unwrap_or(0));
    let corrected_value = initial_value.saturating_add(97);

    let mut events = Vec::with_capacity(config.distractor_count.saturating_add(9));
    events.push(KnowledgeEvent {
        observed_at: 0,
        ordinal: 0,
        kind: KnowledgeEventKind::Version {
            fact_id: target_fact_id,
            entity_id: target_entity_id,
            predicate_id: target_predicate_id,
            revision: 0,
            valid_from: 0,
            operation: KnowledgeOperation::Assert,
            value: Some(initial_value),
        },
    });

    let mut used_entities = BTreeSet::new();
    used_entities.insert(target_entity_id);
    for index in 0..config.distractor_count {
        let entity_id = unique_distractor_entity(
            &mut rng,
            target_entity_id,
            &mut used_entities,
        );
        let observed_at = u64::try_from(index)
            .unwrap_or(u64::MAX)
            .saturating_add(1)
            .saturating_mul(config.interval_steps);
        let distractor_value = i64::try_from(rng.next_u64() % 10_000_000).unwrap_or(0);
        events.push(KnowledgeEvent {
            observed_at,
            ordinal: 0,
            kind: KnowledgeEventKind::Version {
                fact_id: u64::try_from(index).unwrap_or(u64::MAX).saturating_add(2),
                entity_id,
                predicate_id: target_predicate_id,
                revision: 0,
                valid_from: 0,
                operation: KnowledgeOperation::Assert,
                value: Some(distractor_value),
            },
        });
    }

    events.push(KnowledgeEvent {
        observed_at: correction_observed_at,
        ordinal: 0,
        kind: KnowledgeEventKind::Version {
            fact_id: target_fact_id,
            entity_id: target_entity_id,
            predicate_id: target_predicate_id,
            revision: 1,
            valid_from: correction_valid_from,
            operation: KnowledgeOperation::Correct,
            value: Some(corrected_value),
        },
    });
    events.push(KnowledgeEvent {
        observed_at: retraction_observed_at,
        ordinal: 0,
        kind: KnowledgeEventKind::Version {
            fact_id: target_fact_id,
            entity_id: target_entity_id,
            predicate_id: target_predicate_id,
            revision: 2,
            valid_from: retraction_valid_from,
            operation: KnowledgeOperation::Retract,
            value: None,
        },
    });

    let query_specs = [
        (1_u64, correction_valid_from.saturating_add(1), correction_observed_at.saturating_sub(1)),
        (2_u64, correction_valid_from.saturating_add(1), correction_observed_at),
        (3_u64, retraction_valid_from.saturating_add(1), retraction_observed_at.saturating_sub(1)),
        (4_u64, retraction_valid_from.saturating_add(1), retraction_observed_at),
        (5_u64, config.interval_steps, query_issued_at),
    ];
    let mut answers = Vec::with_capacity(query_specs.len());
    for (query_id, valid_at, known_at) in query_specs {
        let (expected_value, expected_revision) = resolve_oracle_answer(
            &events,
            target_fact_id,
            target_entity_id,
            target_predicate_id,
            valid_at,
            known_at,
        );
        answers.push(KnowledgeExpectedAnswer {
            query_id,
            valid_at,
            known_at,
            expected_value,
            expected_revision,
        });
        events.push(KnowledgeEvent {
            observed_at: query_issued_at,
            ordinal: u32::try_from(query_id).unwrap_or(u32::MAX),
            kind: KnowledgeEventKind::Query {
                query_id,
                entity_id: target_entity_id,
                predicate_id: target_predicate_id,
                valid_at,
                known_at,
            },
        });
    }

    let mut task = KnowledgeStreamTask {
        stream_id,
        target_entity_id,
        target_predicate_id,
        distractor_count: config.distractor_count,
        interval_steps: config.interval_steps,
        query_issued_at,
        events,
        content_sha256: [0; 32],
    };
    task.events.sort_by_key(|event| (event.observed_at, event.ordinal));
    task.content_sha256 = knowledge_task_digest(&task);

    let mut oracle = KnowledgeStreamOracle {
        stream_id,
        target_fact_id,
        target_entity_id,
        target_predicate_id,
        initial_value,
        corrected_value,
        correction_observed_at,
        correction_valid_from,
        retraction_observed_at,
        retraction_valid_from,
        answers,
        content_sha256: [0; 32],
    };
    oracle.content_sha256 = knowledge_oracle_digest(&oracle);
    Ok((task, oracle))
}

fn resolve_oracle_answer(
    events: &[KnowledgeEvent],
    target_fact_id: u64,
    target_entity_id: u64,
    target_predicate_id: u32,
    valid_at: u64,
    known_at: u64,
) -> (Option<i64>, Option<u32>) {
    let mut selected: Option<(u32, KnowledgeOperation, Option<i64>)> = None;
    for event in events {
        if event.observed_at > known_at {
            continue;
        }
        if let KnowledgeEventKind::Version {
            fact_id,
            entity_id,
            predicate_id,
            revision,
            valid_from,
            operation,
            value,
        } = &event.kind
        {
            if *fact_id == target_fact_id
                && *entity_id == target_entity_id
                && *predicate_id == target_predicate_id
                && *valid_from <= valid_at
                && selected.is_none_or(|(current_revision, _, _)| *revision > current_revision)
            {
                selected = Some((*revision, *operation, *value));
            }
        }
    }
    match selected {
        Some((revision, KnowledgeOperation::Retract, _)) => (None, Some(revision)),
        Some((revision, _, value)) => (value, Some(revision)),
        None => (None, None),
    }
}

pub fn knowledge_task_digest(task: &KnowledgeStreamTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.knowledge-stream.public.v1");
    hash.u64(task.stream_id);
    hash.u64(task.target_entity_id);
    hash.u32(task.target_predicate_id);
    hash.usize(task.distractor_count);
    hash.u64(task.interval_steps);
    hash.u64(task.query_issued_at);
    hash.usize(task.events.len());
    for event in &task.events {
        hash.u64(event.observed_at);
        hash.u32(event.ordinal);
        match &event.kind {
            KnowledgeEventKind::Version {
                fact_id, entity_id, predicate_id, revision, valid_from, operation, value,
            } => {
                hash.u8(0);
                hash.u64(*fact_id);
                hash.u64(*entity_id);
                hash.u32(*predicate_id);
                hash.u32(*revision);
                hash.u64(*valid_from);
                hash.u8(match operation {
                    KnowledgeOperation::Assert => 0,
                    KnowledgeOperation::Correct => 1,
                    KnowledgeOperation::Retract => 2,
                });
                match value {
                    Some(value) => { hash.u8(1); hash.u64(*value as u64); }
                    None => hash.u8(0),
                }
            }
            KnowledgeEventKind::Query {
                query_id, entity_id, predicate_id, valid_at, known_at,
            } => {
                hash.u8(1);
                hash.u64(*query_id);
                hash.u64(*entity_id);
                hash.u32(*predicate_id);
                hash.u64(*valid_at);
                hash.u64(*known_at);
            }
        }
    }
    hash.finish()
}

pub fn knowledge_oracle_digest(oracle: &KnowledgeStreamOracle) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.knowledge-stream.oracle.v1");
    hash.u64(oracle.stream_id);
    hash.u64(oracle.target_fact_id);
    hash.u64(oracle.target_entity_id);
    hash.u32(oracle.target_predicate_id);
    hash.u64(oracle.initial_value as u64);
    hash.u64(oracle.corrected_value as u64);
    hash.u64(oracle.correction_observed_at);
    hash.u64(oracle.correction_valid_from);
    hash.u64(oracle.retraction_observed_at);
    hash.u64(oracle.retraction_valid_from);
    hash.usize(oracle.answers.len());
    for answer in &oracle.answers {
        hash.u64(answer.query_id);
        hash.u64(answer.valid_at);
        hash.u64(answer.known_at);
        match answer.expected_value {
            Some(value) => { hash.u8(1); hash.u64(value as u64); }
            None => hash.u8(0),
        }
        match answer.expected_revision {
            Some(revision) => { hash.u8(1); hash.u32(revision); }
            None => hash.u8(0),
        }
    }
    hash.finish()
}

pub fn verify_knowledge_stream(
    task: &KnowledgeStreamTask,
    oracle: &KnowledgeStreamOracle,
) -> Result<(), KnowledgeVerificationError> {
    if task.stream_id != oracle.stream_id {
        return Err(KnowledgeVerificationError::StreamIdMismatch);
    }
    if task.content_sha256 != knowledge_task_digest(task) {
        return Err(KnowledgeVerificationError::PublicDigestMismatch);
    }
    if oracle.content_sha256 != knowledge_oracle_digest(oracle) {
        return Err(KnowledgeVerificationError::OracleDigestMismatch);
    }
    if task.target_entity_id != oracle.target_entity_id
        || task.target_predicate_id != oracle.target_predicate_id
        || oracle.target_fact_id != 1
        || oracle.correction_valid_from >= oracle.retraction_valid_from
        || oracle.correction_observed_at >= oracle.retraction_observed_at
        || oracle.retraction_observed_at >= task.query_issued_at
        || oracle.answers.len() != 5
        || task.distractor_count > MAX_KNOWLEDGE_DISTRACTORS
        || task.interval_steps == 0
        || task.events.len() != task.distractor_count.saturating_add(8)
    {
        return Err(KnowledgeVerificationError::InvalidTimeline);
    }

    let mut previous_key = None;
    let mut revisions: std::collections::BTreeMap<u64, u32> = std::collections::BTreeMap::new();
    let mut fact_entities = BTreeSet::new();
    let mut fact_ids = BTreeSet::new();
    let mut query_answers = std::collections::BTreeMap::new();
    let mut query_count = 0_usize;
    let mut target_revision_count = 0_usize;
    let mut distractor_index = 0_usize;

    let count = u64::try_from(task.distractor_count).unwrap_or(u64::MAX);
    let expected_correction_observed = count.saturating_add(1).saturating_mul(task.interval_steps);
    let expected_retraction_observed = count.saturating_add(2).saturating_mul(task.interval_steps);
    let expected_query_issued = count.saturating_add(3).saturating_mul(task.interval_steps);
    if oracle.correction_valid_from != task.interval_steps.saturating_mul(2)
        || oracle.retraction_valid_from != task.interval_steps.saturating_mul(3)
        || oracle.correction_observed_at != expected_correction_observed
        || oracle.retraction_observed_at != expected_retraction_observed
        || task.query_issued_at != expected_query_issued
    {
        return Err(KnowledgeVerificationError::InvalidTimeline);
    }

    for (index, event) in task.events.iter().enumerate() {
        let key = (event.observed_at, event.ordinal);
        if previous_key.is_some_and(|previous| key <= previous) {
            return Err(KnowledgeVerificationError::InvalidEventOrder { index });
        }
        previous_key = Some(key);

        match &event.kind {
            KnowledgeEventKind::Version {
                fact_id, entity_id, predicate_id, revision, valid_from, operation, value,
            } => {
                let previous = revisions.get(fact_id).copied();
                let expected_revision = previous.map_or(0, |revision| revision.saturating_add(1));
                if *revision != expected_revision {
                    return Err(KnowledgeVerificationError::InvalidRevision {
                        fact_id: *fact_id, revision: *revision,
                    });
                }
                revisions.insert(*fact_id, *revision);
                if *fact_id == oracle.target_fact_id {
                    target_revision_count = target_revision_count.saturating_add(1);
                } else {
                    distractor_index = distractor_index.saturating_add(1);
                    let expected_fact_id =
                        u64::try_from(distractor_index).unwrap_or(u64::MAX).saturating_add(1);
                    let expected_observed_at = u64::try_from(distractor_index)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(task.interval_steps);
                    let distance = (*entity_id ^ oracle.target_entity_id).count_ones();
                    if *fact_id != expected_fact_id
                        || event.observed_at != expected_observed_at
                        || event.ordinal != 0
                        || *revision != 0
                        || *valid_from != 0
                        || *predicate_id != task.target_predicate_id
                        || *operation != KnowledgeOperation::Assert
                        || value.is_none()
                        || !(1..=4).contains(&distance)
                        || !fact_entities.insert(*entity_id)
                        || !fact_ids.insert(*fact_id)
                    {
                        return Err(KnowledgeVerificationError::InvalidRevision {
                            fact_id: *fact_id, revision: *revision,
                        });
                    }
                }
                match operation {
                    KnowledgeOperation::Assert if *revision == 0 && value.is_some() => {}
                    KnowledgeOperation::Correct if *revision > 0 && value.is_some() => {}
                    KnowledgeOperation::Retract if *revision > 0 && value.is_none() => {}
                    _ => return Err(KnowledgeVerificationError::InvalidOperation {
                        fact_id: *fact_id, revision: *revision,
                    }),
                }
                if *fact_id == oracle.target_fact_id {
                    let expected = match *revision {
                        0 => (0, KnowledgeOperation::Assert, Some(oracle.initial_value), 0),
                        1 => (
                            oracle.correction_valid_from,
                            KnowledgeOperation::Correct,
                            Some(oracle.corrected_value),
                            oracle.correction_observed_at,
                        ),
                        2 => (
                            oracle.retraction_valid_from,
                            KnowledgeOperation::Retract,
                            None,
                            oracle.retraction_observed_at,
                        ),
                        _ => return Err(KnowledgeVerificationError::InvalidRevision {
                            fact_id: *fact_id, revision: *revision,
                        }),
                    };
                    if *entity_id != oracle.target_entity_id
                        || *predicate_id != oracle.target_predicate_id
                        || *valid_from != expected.0
                        || *operation != expected.1
                        || *value != expected.2
                        || event.observed_at != expected.3
                        || event.ordinal != 0
                    {
                        return Err(KnowledgeVerificationError::InvalidRevision {
                            fact_id: *fact_id, revision: *revision,
                        });
                    }
                }
            }
            KnowledgeEventKind::Query {
                query_id, entity_id, predicate_id, valid_at, known_at,
            } => {
                if *entity_id != oracle.target_entity_id
                    || *predicate_id != oracle.target_predicate_id
                    || *known_at > event.observed_at
                    || event.observed_at != task.query_issued_at
                    || event.ordinal != u32::try_from(*query_id).unwrap_or(u32::MAX)
                {
                    return Err(KnowledgeVerificationError::InvalidQuery { query_id: *query_id });
                }
                query_count = query_count.saturating_add(1);
                if query_answers.insert(*query_id, (*valid_at, *known_at)).is_some() {
                    return Err(KnowledgeVerificationError::InvalidQuery { query_id: *query_id });
                }
            }
        }
    }

    if query_answers.len() != oracle.answers.len()
        || query_count != oracle.answers.len()
        || target_revision_count != 3
        || distractor_index != task.distractor_count
        || fact_ids.len() != task.distractor_count
    {
        return Err(KnowledgeVerificationError::InvalidTimeline);
    }

    // Independent verification: scan the public history for the highest
    // revision eligible at both valid_at and known_at, then compare the answer.
    let answer_ids: BTreeSet<_> = oracle.answers.iter().map(|answer| answer.query_id).collect();
    if answer_ids != BTreeSet::from([1_u64, 2, 3, 4, 5]) {
        return Err(KnowledgeVerificationError::InvalidTimeline);
    }
    let expected_query_coordinates = [
        (
            1_u64,
            oracle.correction_valid_from.saturating_add(1),
            oracle.correction_observed_at.saturating_sub(1),
        ),
        (
            2_u64,
            oracle.correction_valid_from.saturating_add(1),
            oracle.correction_observed_at,
        ),
        (
            3_u64,
            oracle.retraction_valid_from.saturating_add(1),
            oracle.retraction_observed_at.saturating_sub(1),
        ),
        (
            4_u64,
            oracle.retraction_valid_from.saturating_add(1),
            oracle.retraction_observed_at,
        ),
        (5_u64, oracle.correction_valid_from / 2, task.query_issued_at),
    ];
    for (query_id, expected_valid_at, expected_known_at) in expected_query_coordinates {
        if query_answers.get(&query_id).copied()
            != Some((expected_valid_at, expected_known_at))
        {
            return Err(KnowledgeVerificationError::InvalidQuery { query_id });
        }
    }

    for answer in &oracle.answers {
        let Some((valid_at, known_at)) = query_answers.get(&answer.query_id).copied() else {
            return Err(KnowledgeVerificationError::InvalidQuery { query_id: answer.query_id });
        };
        if valid_at != answer.valid_at || known_at != answer.known_at {
            return Err(KnowledgeVerificationError::InvalidAnswer { query_id: answer.query_id });
        }

        let mut selected_revision = None;
        let mut selected_operation = None;
        let mut selected_value = None;
        for event in &task.events {
            if event.observed_at > known_at { continue; }
            if let KnowledgeEventKind::Version {
                fact_id, entity_id, predicate_id, revision, valid_from, operation, value,
            } = &event.kind {
                if *fact_id == oracle.target_fact_id
                    && *entity_id == oracle.target_entity_id
                    && *predicate_id == oracle.target_predicate_id
                    && *valid_from <= valid_at
                    && selected_revision.is_none_or(|current| *revision > current)
                {
                    selected_revision = Some(*revision);
                    selected_operation = Some(*operation);
                    selected_value = *value;
                }
            }
        }
        let expected = if selected_operation == Some(KnowledgeOperation::Retract) {
            None
        } else {
            selected_value
        };
        if answer.expected_value != expected || answer.expected_revision != selected_revision {
            return Err(KnowledgeVerificationError::InvalidAnswer { query_id: answer.query_id });
        }
    }
    Ok(())
}

/// Replay a knowledge stream from evaluator-held seed and configuration.
pub fn verify_knowledge_stream_seeded(
    task: &KnowledgeStreamTask,
    oracle: &KnowledgeStreamOracle,
    config: KnowledgeStreamConfig,
) -> Result<(), KnowledgeVerificationError> {
    verify_knowledge_stream(task, oracle)?;
    let (expected_task, expected_oracle) = generate_knowledge_stream(config)
        .map_err(|_| KnowledgeVerificationError::SeedReplayMismatch)?;
    if task != &expected_task || oracle != &expected_oracle {
        return Err(KnowledgeVerificationError::SeedReplayMismatch);
    }
    Ok(())
}

fn knowledge_stream_id(config: KnowledgeStreamConfig) -> u64 {
    mix64(config.seed ^ (config.distractor_count as u64).rotate_left(17)
        ^ config.interval_steps.rotate_left(37))
}

fn unique_distractor_entity(
    rng: &mut SplitMix64,
    target_entity: u64,
    used: &mut BTreeSet<u64>,
) -> u64 {
    loop {
        let bit_count = 1 + rng.next_u64() % 4;
        let mut mask = 0_u64;
        while mask.count_ones() < bit_count as u32 {
            mask |= 1_u64 << (rng.next_u64() % 64);
        }
        let candidate = target_entity ^ mask;
        if used.insert(candidate) { return candidate; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> KnowledgeStreamConfig {
        KnowledgeStreamConfig {
            seed: 0x0A45_2026_1031,
            distractor_count: 32,
            interval_steps: 16,
        }
    }

    fn perfect_predictions(oracle: &KnowledgeStreamOracle) -> Vec<KnowledgePrediction> {
        oracle
            .answers
            .iter()
            .map(|answer| KnowledgePrediction {
                query_id: answer.query_id,
                answer: match answer.expected_value {
                    Some(value) => KnowledgeCandidateAnswer::Value(value),
                    None => KnowledgeCandidateAnswer::NoFact,
                },
            })
            .collect()
    }

    #[test]
    fn perfect_bitemporal_predictions_score_all_queries() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let score = score_knowledge_predictions(&oracle, &perfect_predictions(&oracle)).unwrap();
        assert_eq!(score.query_count, 5);
        assert_eq!(score.exact_matches, 5);
        assert_eq!(score.wrong_answers, 0);
        assert_eq!(score.future_knowledge_leaks, 0);
        assert_eq!(score.stale_answers_after_retraction, 0);
        assert_eq!(score.missed_corrections, 0);
        assert_eq!(score.historical_as_known_preserved, 1);
        assert_eq!(score.exact_match_rate(), Some(1.0));
        assert_eq!(score.prediction_coverage(), Some(1.0));
    }

    #[test]
    fn scorer_separately_counts_future_knowledge_leakage() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let mut predictions = perfect_predictions(&oracle);
        predictions[0].answer = KnowledgeCandidateAnswer::Value(oracle.corrected_value);
        let score = score_knowledge_predictions(&oracle, &predictions).unwrap();
        assert_eq!(score.future_knowledge_leaks, 1);
        assert_eq!(score.exact_matches, 4);
    }

    #[test]
    fn scorer_counts_stale_answer_after_known_retraction() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let mut predictions = perfect_predictions(&oracle);
        predictions[3].answer = KnowledgeCandidateAnswer::Value(oracle.corrected_value);
        let score = score_knowledge_predictions(&oracle, &predictions).unwrap();
        assert_eq!(score.stale_answers_after_retraction, 1);
        assert_eq!(score.wrong_answers, 1);
    }

    #[test]
    fn scorer_counts_missed_correction_and_preserved_history_separately() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let mut predictions = perfect_predictions(&oracle);
        predictions[1].answer = KnowledgeCandidateAnswer::Value(oracle.initial_value);
        let score = score_knowledge_predictions(&oracle, &predictions).unwrap();
        assert_eq!(score.missed_corrections, 1);
        assert_eq!(score.historical_as_known_preserved, 1);
    }

    #[test]
    fn scorer_distinguishes_abstention_from_missing_prediction_and_no_fact() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let predictions = [
            KnowledgePrediction { query_id: 1, answer: KnowledgeCandidateAnswer::Abstain },
            KnowledgePrediction { query_id: 2, answer: KnowledgeCandidateAnswer::NoFact },
            KnowledgePrediction { query_id: 3, answer: KnowledgeCandidateAnswer::Value(oracle.corrected_value) },
            KnowledgePrediction { query_id: 4, answer: KnowledgeCandidateAnswer::NoFact },
        ];
        let score = score_knowledge_predictions(&oracle, &predictions).unwrap();
        assert_eq!(score.abstentions, 1);
        assert_eq!(score.missing_predictions, 1);
        assert_eq!(score.wrong_answers, 1);
        assert_eq!(score.predictions_received, 4);
    }

    #[test]
    fn scorer_rejects_duplicate_and_unknown_query_predictions() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let duplicate = [
            KnowledgePrediction { query_id: 1, answer: KnowledgeCandidateAnswer::NoFact },
            KnowledgePrediction { query_id: 1, answer: KnowledgeCandidateAnswer::NoFact },
        ];
        assert_eq!(
            score_knowledge_predictions(&oracle, &duplicate),
            Err(KnowledgeScoringError::DuplicatePrediction { query_id: 1 })
        );
        let unknown = [KnowledgePrediction { query_id: 99, answer: KnowledgeCandidateAnswer::NoFact }];
        assert_eq!(
            score_knowledge_predictions(&oracle, &unknown),
            Err(KnowledgeScoringError::UnknownQuery { query_id: 99 })
        );
    }

    #[test]
    fn generation_is_deterministic_for_fixed_seed() {
        let cfg = config();
        assert_eq!(generate_knowledge_stream(cfg), generate_knowledge_stream(cfg));
    }

    #[test]
    fn generated_stream_passes_bitemporal_verifier() {
        let (task, oracle) = generate_knowledge_stream(config()).unwrap();
        assert_eq!(verify_knowledge_stream(&task, &oracle), Ok(()));
    }

    #[test]
    fn same_valid_time_changes_with_known_time_after_late_correction() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let before = oracle.answers.iter().find(|answer| answer.query_id == 1).unwrap();
        let after = oracle.answers.iter().find(|answer| answer.query_id == 2).unwrap();
        assert_eq!(before.valid_at, after.valid_at);
        assert_ne!(before.known_at, after.known_at);
        assert_eq!(before.expected_value, Some(oracle.initial_value));
        assert_eq!(after.expected_value, Some(oracle.corrected_value));
    }

    #[test]
    fn delayed_retraction_preserves_pre_retraction_known_state() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let before = oracle.answers.iter().find(|answer| answer.query_id == 3).unwrap();
        let after = oracle.answers.iter().find(|answer| answer.query_id == 4).unwrap();
        assert_eq!(before.valid_at, after.valid_at);
        assert_eq!(before.expected_value, Some(oracle.corrected_value));
        assert_eq!(after.expected_value, None);
    }

    #[test]
    fn retroactive_correction_does_not_rewrite_earlier_valid_interval() {
        let (_, oracle) = generate_knowledge_stream(config()).unwrap();
        let historical = oracle.answers.iter().find(|answer| answer.query_id == 5).unwrap();
        assert_eq!(historical.expected_value, Some(oracle.initial_value));
        assert_eq!(historical.expected_revision, Some(0));
    }

    #[test]
    fn query_knowledge_time_never_exceeds_issue_time() {
        let (task, _) = generate_knowledge_stream(config()).unwrap();
        for event in &task.events {
            if let KnowledgeEventKind::Query { known_at, .. } = &event.kind {
                assert!(*known_at <= event.observed_at);
            }
        }
    }

    #[test]
    fn target_revisions_have_independent_valid_and_transaction_times() {
        let (task, oracle) = generate_knowledge_stream(config()).unwrap();
        let revisions: Vec<_> = task.events.iter().filter_map(|event| match &event.kind {
            KnowledgeEventKind::Version { fact_id, revision, valid_from, .. }
                if *fact_id == oracle.target_fact_id => Some((event.observed_at, *valid_from, *revision)),
            _ => None,
        }).collect();
        assert_eq!(revisions, vec![
            (0, 0, 0),
            (oracle.correction_observed_at, oracle.correction_valid_from, 1),
            (oracle.retraction_observed_at, oracle.retraction_valid_from, 2),
        ]);
        assert!(revisions[1].0 > revisions[1].1);
        assert!(revisions[2].0 > revisions[2].1);
    }

    #[test]
    fn public_and_oracle_sha256_digests_are_valid() {
        let (task, oracle) = generate_knowledge_stream(config()).unwrap();
        assert_eq!(task.content_sha256, knowledge_task_digest(&task));
        assert_eq!(oracle.content_sha256, knowledge_oracle_digest(&oracle));
    }

    #[test]
    fn verifier_rejects_public_event_mutation() {
        let (mut task, oracle) = generate_knowledge_stream(config()).unwrap();
        task.events[0].ordinal ^= 1;
        assert_eq!(
            verify_knowledge_stream(&task, &oracle),
            Err(KnowledgeVerificationError::PublicDigestMismatch)
        );
    }

    #[test]
    fn verifier_rejects_recommitted_late_correction_time_mutation() {
        let (mut task, oracle) = generate_knowledge_stream(config()).unwrap();
        if let Some(event) = task.events.iter_mut().find(|event| matches!(
            &event.kind,
            KnowledgeEventKind::Version { revision: 1, .. }
        )) {
            event.observed_at = event.observed_at.saturating_sub(1);
        }
        task.content_sha256 = knowledge_task_digest(&task);
        assert!(matches!(
            verify_knowledge_stream(&task, &oracle),
            Err(KnowledgeVerificationError::InvalidRevision { revision: 1, .. })
        ));
    }

    #[test]
    fn verifier_rejects_recommitted_historical_answer_leakage() {
        let (task, mut oracle) = generate_knowledge_stream(config()).unwrap();
        oracle.answers[0].expected_value = Some(oracle.corrected_value);
        oracle.content_sha256 = knowledge_oracle_digest(&oracle);
        assert_eq!(
            verify_knowledge_stream(&task, &oracle),
            Err(KnowledgeVerificationError::InvalidAnswer { query_id: 1 })
        );
    }

    #[test]
    fn seeded_verifier_rejects_recommitted_task_mutation() {
        let cfg = config();
        let (mut task, oracle) = generate_knowledge_stream(cfg).unwrap();
        let distractor = task.events.iter_mut().find(|event| matches!(
            &event.kind,
            KnowledgeEventKind::Version { fact_id, .. } if *fact_id != oracle.target_fact_id
        )).unwrap();
        if let KnowledgeEventKind::Version { value, .. } = &mut distractor.kind {
            if let Some(value) = value {
                *value = value.saturating_add(1);
            }
        }
        task.content_sha256 = knowledge_task_digest(&task);
        assert_eq!(
            verify_knowledge_stream_seeded(&task, &oracle, cfg),
            Err(KnowledgeVerificationError::SeedReplayMismatch)
        );
    }

    #[test]
    fn zero_interval_is_rejected() {
        let mut cfg = config();
        cfg.interval_steps = 0;
        assert_eq!(
            generate_knowledge_stream(cfg),
            Err(KnowledgeGenerationError::ZeroInterval)
        );
    }

    #[test]
    fn excessive_distractor_count_is_rejected() {
        let mut cfg = config();
        cfg.distractor_count = MAX_KNOWLEDGE_DISTRACTORS + 1;
        assert!(matches!(
            generate_knowledge_stream(cfg),
            Err(KnowledgeGenerationError::TooManyDistractors { .. })
        ));
    }
}
