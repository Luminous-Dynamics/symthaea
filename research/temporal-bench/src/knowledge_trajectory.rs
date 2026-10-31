//! Repeated-checkpoint bitemporal knowledge trajectory for Temporal Error Debt.
//!
//! This module extends the five-anchor T10 knowledge stream without changing
//! its semantics. It emits current-valid-time and historical-as-known queries
//! at every knowledge-time checkpoint, then exposes a separate evaluator
//! window schedule for the TED scorer. The last checkpoint is held for one
//! additional declared interval; no wall-clock duration is inferred.

use super::digest::CanonicalHasher;
use super::knowledge_stream::{
    generate_knowledge_stream, knowledge_task_digest, verify_knowledge_stream,
    KnowledgeCandidateAnswer, KnowledgeEvent, KnowledgeEventKind, KnowledgeStreamConfig,
    KnowledgeStreamOracle, KnowledgeStreamTask, KnowledgeVerificationError,
};
use super::temporal_error_debt::KnowledgeExposureWindow;
use std::collections::BTreeMap;
use std::fmt;

pub const CURRENT_TRAJECTORY_QUERY_ID: u64 = 1;
pub const HISTORICAL_TRAJECTORY_QUERY_ID: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgeCheckpointBinding {
    pub checkpoint_index: u32,
    pub known_at: u64,
    pub current_valid_at: u64,
    pub historical_valid_at: u64,
    pub current_query_event_id: u64,
    pub historical_query_event_id: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeTrajectoryTask {
    /// Public append-only fact history plus repeated bitemporal query events.
    pub stream: KnowledgeStreamTask,
    /// Maps candidate-visible query events to stable current/historical lanes.
    pub checkpoints: Vec<KnowledgeCheckpointBinding>,
    /// Subdivision of the source update interval. Chosen to create about four
    /// observation opportunities between knowledge updates while exactly
    /// dividing the update interval.
    pub checkpoint_interval_steps: u64,
    /// SHA-256 over the public stream digest and the checkpoint bindings.
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeTrajectoryError {
    Generation(String),
    InvalidTaskDigest,
    InvalidCheckpointSchedule,
    InvalidQueryEvent { query_event_id: u64 },
    InvalidEventOrder { index: usize },
    InvalidBaseHistory(KnowledgeVerificationError),
    SeedReplayMismatch,
}

impl fmt::Display for KnowledgeTrajectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generation(message) => write!(f, "cannot generate knowledge trajectory: {message}"),
            Self::InvalidTaskDigest => write!(f, "knowledge trajectory SHA-256 digest mismatch"),
            Self::InvalidCheckpointSchedule => write!(f, "knowledge trajectory checkpoint schedule is invalid"),
            Self::InvalidQueryEvent { query_event_id } => {
                write!(f, "knowledge trajectory query event {query_event_id} is missing or invalid")
            }
            Self::InvalidEventOrder { index } => {
                write!(f, "knowledge trajectory event {index} is not strictly ordered")
            }
            Self::InvalidBaseHistory(error) => write!(f, "base bitemporal history is invalid: {error}"),
            Self::SeedReplayMismatch => {
                write!(f, "knowledge trajectory differs from evaluator-held seed/configuration")
            }
        }
    }
}

impl std::error::Error for KnowledgeTrajectoryError {}

fn checkpoint_interval(interval_steps: u64) -> u64 {
    if interval_steps <= 1 {
        return 1;
    }
    let desired = (interval_steps / 4).max(1);
    greatest_common_divisor(interval_steps, desired)
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

/// Generate repeated current and historical queries over the same append-only history.
///
/// There are two stable scorer lanes:
/// - lane 1 holds valid_at just after the retraction's valid-from time; its
///   correct answer transitions initial -> corrected -> NoFact as updates arrive;
/// - lane 2 holds valid_at before the correction's valid-from time; its
///   correct answer remains the initial value even after later updates arrive.
///
/// Each checkpoint is placed at 0, interval, 2*interval, ... through the
/// original T10 query-issue horizon. Each lane's output is held for one
/// interval, including the final checkpoint. The seed/config is evaluator-only.
pub fn generate_knowledge_trajectory(
    config: KnowledgeStreamConfig,
) -> Result<(KnowledgeTrajectoryTask, KnowledgeStreamOracle), KnowledgeTrajectoryError> {
    let (base_task, oracle) = generate_knowledge_stream(config)
        .map_err(|error| KnowledgeTrajectoryError::Generation(error.to_string()))?;

    let interval = config.interval_steps;
    let checkpoint_interval_steps = checkpoint_interval(interval);
    let query_issued_at = base_task.query_issued_at;
    let checkpoint_count_u64 = query_issued_at
        .checked_div(checkpoint_interval_steps)
        .and_then(|last| last.checked_add(1))
        .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
    let checkpoint_count = usize::try_from(checkpoint_count_u64)
        .map_err(|_| KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;

    let current_valid_at = oracle.retraction_valid_from
        .checked_add(1)
        .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
    // This is strictly before correction_valid_from in the base T10 contract.
    let historical_valid_at = interval;

    let mut stream = base_task.clone();
    stream.events.retain(|event| matches!(&event.kind, KnowledgeEventKind::Version { .. }));
    let mut checkpoints = Vec::with_capacity(checkpoint_count);

    for index in 0..checkpoint_count {
        let checkpoint_index = u32::try_from(index)
            .map_err(|_| KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
        let known_at = u64::try_from(index)
            .ok()
            .and_then(|step| step.checked_mul(checkpoint_interval_steps))
            .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
        let current_query_event_id = u64::try_from(index)
            .ok()
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(1))
            .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
        let historical_query_event_id = current_query_event_id
            .checked_add(1)
            .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;

        stream.events.push(KnowledgeEvent {
            observed_at: known_at,
            ordinal: 1,
            kind: KnowledgeEventKind::Query {
                query_id: current_query_event_id,
                entity_id: oracle.target_entity_id,
                predicate_id: oracle.target_predicate_id,
                valid_at: current_valid_at,
                known_at,
            },
        });
        stream.events.push(KnowledgeEvent {
            observed_at: known_at,
            ordinal: 2,
            kind: KnowledgeEventKind::Query {
                query_id: historical_query_event_id,
                entity_id: oracle.target_entity_id,
                predicate_id: oracle.target_predicate_id,
                valid_at: historical_valid_at,
                known_at,
            },
        });

        checkpoints.push(KnowledgeCheckpointBinding {
            checkpoint_index,
            known_at,
            current_valid_at,
            historical_valid_at,
            current_query_event_id,
            historical_query_event_id,
        });
    }

    stream.events.sort_by_key(|event| (event.observed_at, event.ordinal));
    stream.content_sha256 = knowledge_task_digest(&stream);

    let mut task = KnowledgeTrajectoryTask {
        stream,
        checkpoints,
        checkpoint_interval_steps,
        content_sha256: [0; 32],
    };
    task.content_sha256 = knowledge_trajectory_digest(&task);
    Ok((task, oracle))
}

/// Domain-separated canonical digest of the trajectory's public manifest.
pub fn knowledge_trajectory_digest(task: &KnowledgeTrajectoryTask) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.knowledge-trajectory.public.v1");
    hash.u64(task.stream.stream_id);
    hash.bytes(&task.stream.content_sha256);
    hash.u64(task.stream.query_issued_at);
    hash.u64(task.stream.interval_steps);
    hash.u64(task.checkpoint_interval_steps);
    hash.usize(task.checkpoints.len());
    for checkpoint in &task.checkpoints {
        hash.u32(checkpoint.checkpoint_index);
        hash.u64(checkpoint.known_at);
        hash.u64(checkpoint.current_valid_at);
        hash.u64(checkpoint.historical_valid_at);
        hash.u64(checkpoint.current_query_event_id);
        hash.u64(checkpoint.historical_query_event_id);
    }
    hash.finish()
}

/// Produce evaluator-side exposure windows. Each checkpoint state is held for
/// exactly one declared interval, including the final checkpoint.
pub fn knowledge_trajectory_windows(
    task: &KnowledgeTrajectoryTask,
) -> Vec<KnowledgeExposureWindow> {
    let duration = task.checkpoint_interval_steps;
    let mut windows = Vec::with_capacity(task.checkpoints.len().saturating_mul(2));
    for checkpoint in &task.checkpoints {
        windows.push(KnowledgeExposureWindow {
            query_id: CURRENT_TRAJECTORY_QUERY_ID,
            checkpoint_index: checkpoint.checkpoint_index,
            valid_at: checkpoint.current_valid_at,
            known_at: checkpoint.known_at,
            duration,
        });
        windows.push(KnowledgeExposureWindow {
            query_id: HISTORICAL_TRAJECTORY_QUERY_ID,
            checkpoint_index: checkpoint.checkpoint_index,
            valid_at: checkpoint.historical_valid_at,
            known_at: checkpoint.known_at,
            duration,
        });
    }
    windows
}

/// Reconstruct the original five-query T10 anchor task from version events and
/// verify it with the original independent bitemporal verifier.
fn reconstruct_anchor_task(
    task: &KnowledgeTrajectoryTask,
    oracle: &KnowledgeStreamOracle,
) -> KnowledgeStreamTask {
    let mut anchor = task.stream.clone();
    anchor.events.retain(|event| matches!(&event.kind, KnowledgeEventKind::Version { .. }));

    let anchor_queries = [
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
        (5_u64, oracle.correction_valid_from / 2, task.stream.query_issued_at),
    ];
    for (query_id, valid_at, known_at) in anchor_queries {
        anchor.events.push(KnowledgeEvent {
            observed_at: anchor.query_issued_at,
            ordinal: u32::try_from(query_id).unwrap_or(u32::MAX),
            kind: KnowledgeEventKind::Query {
                query_id,
                entity_id: oracle.target_entity_id,
                predicate_id: oracle.target_predicate_id,
                valid_at,
                known_at,
            },
        });
    }
    anchor.events.sort_by_key(|event| (event.observed_at, event.ordinal));
    anchor.content_sha256 = knowledge_task_digest(&anchor);
    anchor
}

/// Verify the repeated schedule, public digest, every query event, and the
/// original T10 fact/revision history.
pub fn verify_knowledge_trajectory(
    task: &KnowledgeTrajectoryTask,
    oracle: &KnowledgeStreamOracle,
) -> Result<(), KnowledgeTrajectoryError> {
    if task.stream.stream_id != oracle.stream_id
        || task.stream.content_sha256 != knowledge_task_digest(&task.stream)
        || task.content_sha256 != knowledge_trajectory_digest(task)
    {
        return Err(KnowledgeTrajectoryError::InvalidTaskDigest);
    }
    let interval = task.stream.interval_steps;
    let expected_checkpoint_interval = checkpoint_interval(interval);
    if interval == 0
        || task.checkpoint_interval_steps != expected_checkpoint_interval
        || task.checkpoints.is_empty()
    {
        return Err(KnowledgeTrajectoryError::InvalidCheckpointSchedule);
    }
    let expected_last = oracle.retraction_observed_at
        .checked_add(interval)
        .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
    if task.stream.query_issued_at != expected_last {
        return Err(KnowledgeTrajectoryError::InvalidCheckpointSchedule);
    }

    let expected_checkpoint_count = usize::try_from(
        expected_last
            .checked_div(task.checkpoint_interval_steps)
            .and_then(|value| value.checked_add(1))
            .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?,
    )
    .map_err(|_| KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
    if task.checkpoints.len() != expected_checkpoint_count {
        return Err(KnowledgeTrajectoryError::InvalidCheckpointSchedule);
    }

    let current_valid_at = oracle.retraction_valid_from
        .checked_add(1)
        .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
    let historical_valid_at = interval;
    let mut expected_queries = BTreeMap::new();

    for (index, checkpoint) in task.checkpoints.iter().enumerate() {
        let checkpoint_index = u32::try_from(index)
            .map_err(|_| KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
        let known_at = u64::try_from(index)
            .ok()
            .and_then(|step| step.checked_mul(task.checkpoint_interval_steps))
            .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
        let current_query_event_id = u64::try_from(index)
            .ok()
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(1))
            .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;
        let historical_query_event_id = current_query_event_id
            .checked_add(1)
            .ok_or(KnowledgeTrajectoryError::InvalidCheckpointSchedule)?;

        if checkpoint.checkpoint_index != checkpoint_index
            || checkpoint.known_at != known_at
            || checkpoint.current_valid_at != current_valid_at
            || checkpoint.historical_valid_at != historical_valid_at
            || checkpoint.current_query_event_id != current_query_event_id
            || checkpoint.historical_query_event_id != historical_query_event_id
        {
            return Err(KnowledgeTrajectoryError::InvalidCheckpointSchedule);
        }
        expected_queries.insert(
            current_query_event_id,
            (known_at, current_valid_at, 1_u32),
        );
        expected_queries.insert(
            historical_query_event_id,
            (known_at, historical_valid_at, 2_u32),
        );
    }

    let mut previous_event_key = None;
    for (index, event) in task.stream.events.iter().enumerate() {
        let key = (event.observed_at, event.ordinal);
        if previous_event_key.is_some_and(|previous| key <= previous) {
            return Err(KnowledgeTrajectoryError::InvalidEventOrder { index });
        }
        previous_event_key = Some(key);
    }

    let actual_queries = task.stream.events.iter().filter_map(|event| {
        if let KnowledgeEventKind::Query {
            query_id,
            entity_id,
            predicate_id,
            valid_at,
            known_at,
        } = &event.kind {
            Some((
                *query_id,
                (
                    event.observed_at,
                    event.ordinal,
                    *entity_id,
                    *predicate_id,
                    *valid_at,
                    *known_at,
                ),
            ))
        } else {
            None
        }
    }).collect::<BTreeMap<_, _>>();

    if actual_queries.len() != expected_queries.len() {
        return Err(KnowledgeTrajectoryError::InvalidQueryEvent { query_event_id: 0 });
    }
    for (query_event_id, (known_at, valid_at, ordinal)) in expected_queries {
        let expected = (
            known_at,
            ordinal,
            oracle.target_entity_id,
            oracle.target_predicate_id,
            valid_at,
            known_at,
        );
        if actual_queries.get(&query_event_id) != Some(&expected) {
            return Err(KnowledgeTrajectoryError::InvalidQueryEvent { query_event_id });
        }
    }

    let anchor = reconstruct_anchor_task(task, oracle);
    verify_knowledge_stream(&anchor, oracle)
        .map_err(KnowledgeTrajectoryError::InvalidBaseHistory)?;

    Ok(())
}

/// Replay the full trajectory using evaluator-held seed/configuration.
pub fn verify_knowledge_trajectory_seeded(
    task: &KnowledgeTrajectoryTask,
    oracle: &KnowledgeStreamOracle,
    config: KnowledgeStreamConfig,
) -> Result<(), KnowledgeTrajectoryError> {
    verify_knowledge_trajectory(task, oracle)?;
    let (expected_task, expected_oracle) = generate_knowledge_trajectory(config)?;
    if task != &expected_task || oracle != &expected_oracle {
        return Err(KnowledgeTrajectoryError::SeedReplayMismatch);
    }
    Ok(())
}

/// Build perfect candidate predictions for tests and reference baselines. This
/// helper is intended for evaluator/tests only; production candidate processes
/// must not receive the oracle.
pub fn perfect_knowledge_trajectory_predictions(
    oracle: &KnowledgeStreamOracle,
    task: &KnowledgeTrajectoryTask,
) -> Vec<super::temporal_error_debt::KnowledgeTimedPrediction> {
    use super::temporal_error_debt::KnowledgeTimedPrediction;
    let mut predictions = Vec::with_capacity(task.checkpoints.len().saturating_mul(2));
    for checkpoint in &task.checkpoints {
        for (query_id, valid_at) in [
            (CURRENT_TRAJECTORY_QUERY_ID, checkpoint.current_valid_at),
            (HISTORICAL_TRAJECTORY_QUERY_ID, checkpoint.historical_valid_at),
        ] {
            let answer = super::temporal_error_debt::resolve_expected_value(
                oracle,
                valid_at,
                checkpoint.known_at,
            );
            predictions.push(KnowledgeTimedPrediction {
                query_id,
                checkpoint_index: checkpoint.checkpoint_index,
                answer: match answer {
                    Some(value) => KnowledgeCandidateAnswer::Value(value),
                    None => KnowledgeCandidateAnswer::NoFact,
                },
            });
        }
    }
    predictions
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::temporal_error_debt::{
        score_temporal_error_debt, KnowledgeCandidateAnswer, KnowledgeTimedPrediction,
    };

    fn config() -> KnowledgeStreamConfig {
        KnowledgeStreamConfig {
            seed: 0x9492_2026_1031,
            distractor_count: 4,
            interval_steps: 16,
        }
    }

    #[test]
    fn generated_trajectory_has_repeated_current_and_historical_query_checkpoints() {
        let (task, oracle) = generate_knowledge_trajectory(config()).unwrap();
        assert_eq!(verify_knowledge_trajectory(&task, &oracle), Ok(()));
        assert_eq!(task.checkpoints.len(), 29);
        assert_eq!(task.checkpoint_interval_steps, 4);
        assert_eq!(task.checkpoints.first().unwrap().known_at, 0);
        assert_eq!(task.checkpoints.last().unwrap().known_at, task.stream.query_issued_at);
        let windows = knowledge_trajectory_windows(&task);
        assert_eq!(windows.len(), task.checkpoints.len() * 2);
        assert_eq!(windows.first().unwrap().query_id, CURRENT_TRAJECTORY_QUERY_ID);
        assert_eq!(windows.get(1).unwrap().query_id, HISTORICAL_TRAJECTORY_QUERY_ID);
    }

    #[test]
    fn perfect_event_aligned_trajectory_has_zero_temporal_error_debt() {
        let (task, oracle) = generate_knowledge_trajectory(config()).unwrap();
        let windows = knowledge_trajectory_windows(&task);
        let predictions = perfect_knowledge_trajectory_predictions(&oracle, &task);
        let score = score_temporal_error_debt(&oracle, &windows, &predictions).unwrap();
        assert_eq!(score.wrong_belief_exposure, 0);
        assert_eq!(score.abstention_exposure, 0);
        assert_eq!(score.missing_exposure, 0);
        assert_eq!(score.correction_latency, Some(0));
        assert_eq!(score.retraction_latency, Some(0));
    }

    #[test]
    fn one_checkpoint_correction_delay_adds_exactly_one_interval_of_debt() {
        let (task, oracle) = generate_knowledge_trajectory(config()).unwrap();
        let windows = knowledge_trajectory_windows(&task);
        let mut predictions = perfect_knowledge_trajectory_predictions(&oracle, &task);
        let correction_index = usize::try_from(oracle.correction_observed_at / task.checkpoint_interval_steps)
            .unwrap();
        let delayed = predictions.iter_mut().find(|prediction| {
            prediction.query_id == CURRENT_TRAJECTORY_QUERY_ID
                && prediction.checkpoint_index == u32::try_from(correction_index).unwrap()
        }).unwrap();
        delayed.answer = KnowledgeCandidateAnswer::Value(oracle.initial_value);
        let score = score_temporal_error_debt(&oracle, &windows, &predictions).unwrap();
        assert_eq!(score.wrong_belief_exposure, u128::from(task.checkpoint_interval_steps));
        assert_eq!(score.missed_correction_exposure, u128::from(task.checkpoint_interval_steps));
        assert_eq!(score.correction_latency, Some(task.checkpoint_interval_steps));
        assert_eq!(score.retraction_latency, Some(0));
    }

    #[test]
    fn delayed_retraction_response_adds_exactly_one_interval_of_stale_debt() {
        let (task, oracle) = generate_knowledge_trajectory(config()).unwrap();
        let windows = knowledge_trajectory_windows(&task);
        let mut predictions = perfect_knowledge_trajectory_predictions(&oracle, &task);
        let retraction_index = usize::try_from(oracle.retraction_observed_at / task.checkpoint_interval_steps)
            .unwrap();
        let delayed = predictions.iter_mut().find(|prediction| {
            prediction.query_id == CURRENT_TRAJECTORY_QUERY_ID
                && prediction.checkpoint_index == u32::try_from(retraction_index).unwrap()
        }).unwrap();
        delayed.answer = KnowledgeCandidateAnswer::Value(oracle.corrected_value);
        let score = score_temporal_error_debt(&oracle, &windows, &predictions).unwrap();
        assert_eq!(score.wrong_belief_exposure, u128::from(task.checkpoint_interval_steps));
        assert_eq!(score.stale_after_retraction_exposure, u128::from(task.checkpoint_interval_steps));
        assert_eq!(score.retraction_latency, Some(task.checkpoint_interval_steps));
    }

    #[test]
    fn historical_as_known_lane_preserves_initial_value_after_every_update() {
        let (task, oracle) = generate_knowledge_trajectory(config()).unwrap();
        let predictions = perfect_knowledge_trajectory_predictions(&oracle, &task);
        for prediction in predictions.iter().filter(|prediction| prediction.query_id == HISTORICAL_TRAJECTORY_QUERY_ID) {
            assert_eq!(prediction.answer, KnowledgeCandidateAnswer::Value(oracle.initial_value));
        }
    }

    #[test]
    fn abstention_and_missing_checkpoint_have_separate_exposure() {
        let (task, oracle) = generate_knowledge_trajectory(config()).unwrap();
        let windows = knowledge_trajectory_windows(&task);
        let mut predictions = perfect_knowledge_trajectory_predictions(&oracle, &task);
        let interval_index = 1_u32;
        predictions.iter_mut().find(|prediction| {
            prediction.query_id == CURRENT_TRAJECTORY_QUERY_ID
                && prediction.checkpoint_index == interval_index
        }).unwrap().answer = KnowledgeCandidateAnswer::Abstain;
        predictions.retain(|prediction| !(
            prediction.query_id == HISTORICAL_TRAJECTORY_QUERY_ID
                && prediction.checkpoint_index == interval_index
        ));
        let score = score_temporal_error_debt(&oracle, &windows, &predictions).unwrap();
        assert_eq!(score.abstention_exposure, u128::from(task.checkpoint_interval_steps));
        assert_eq!(score.missing_exposure, u128::from(task.checkpoint_interval_steps));
        assert_eq!(score.wrong_belief_exposure, 0);
    }

    #[test]
    fn recommitted_query_event_mutation_is_rejected() {
        let (mut task, oracle) = generate_knowledge_trajectory(config()).unwrap();
        if let Some(event) = task.stream.events.iter_mut().find(|event| {
            matches!(&event.kind, KnowledgeEventKind::Query { .. })
        }) {
            event.observed_at = event.observed_at.saturating_add(1);
        }
        task.stream.content_sha256 = knowledge_task_digest(&task.stream);
        task.content_sha256 = knowledge_trajectory_digest(&task);
        assert!(matches!(
            verify_knowledge_trajectory(&task, &oracle),
            Err(KnowledgeTrajectoryError::InvalidQueryEvent { .. })
        ));
    }

    #[test]
    fn seeded_verifier_rejects_recommitted_history_mutation() {
        let cfg = config();
        let (mut task, oracle) = generate_knowledge_trajectory(cfg).unwrap();
        let event = task.stream.events.iter_mut().find(|event| matches!(
            &event.kind,
            KnowledgeEventKind::Version { revision: 1, .. }
        )).unwrap();
        if let KnowledgeEventKind::Version { value, .. } = &mut event.kind {
            *value = Some(value.unwrap_or(0).saturating_add(1));
        }
        task.stream.content_sha256 = knowledge_task_digest(&task.stream);
        task.content_sha256 = knowledge_trajectory_digest(&task);
        assert_eq!(
            verify_knowledge_trajectory_seeded(&task, &oracle, cfg),
            Err(KnowledgeTrajectoryError::SeedReplayMismatch)
        );
    }
}
