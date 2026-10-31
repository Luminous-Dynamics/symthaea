//! Time-integrated stale-belief, abstention, and missing-output exposure.
//!
//! This scorer evaluates repeated bitemporal query trajectories. Each window
//! declares the query's valid-time coordinate, its knowledge-time checkpoint,
//! and the duration for which the candidate output is held. Durations use the
//! dataset's declared unit (seconds, milliseconds, or chunk-time units).
//! The scorer never fabricates wall-clock time.
//!
//! Call only after the task/oracle have passed the knowledge-stream semantic and
//! seed-replay verifier. Candidate processes receive public checkpoints only;
//! this module consumes evaluator-owned oracle data.

use super::knowledge_stream::{KnowledgeCandidateAnswer, KnowledgeStreamOracle};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One evaluator-defined interval in a single query's answer trajectory.
///
/// For each query_id, checkpoint_index starts at zero and increments by one.
/// Intervals must be contiguous in knowledge time: the next checkpoint's
/// known_at must equal this interval's known_at + duration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgeExposureWindow {
    pub query_id: u64,
    pub checkpoint_index: u32,
    pub valid_at: u64,
    pub known_at: u64,
    pub duration: u64,
}

/// Candidate output for one query checkpoint.
///
/// A missing prediction is represented by the absence of a record for the
/// corresponding (query_id, checkpoint_index) pair, not by Abstain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgeTimedPrediction {
    pub query_id: u64,
    pub checkpoint_index: u32,
    pub answer: KnowledgeCandidateAnswer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemporalErrorDebtError {
    EmptyWindows,
    InvalidOracleTimeline,
    ZeroDuration { query_id: u64, checkpoint_index: u32 },
    InvalidWindowOrder { query_id: u64, checkpoint_index: u32 },
    NonContiguousTimeline { query_id: u64, checkpoint_index: u32 },
    ValidTimeChanged { query_id: u64, checkpoint_index: u32 },
    TimelineOverflow { query_id: u64, checkpoint_index: u32 },
    DuplicatePrediction { query_id: u64, checkpoint_index: u32 },
    UnknownPrediction { query_id: u64, checkpoint_index: u32 },
}

impl fmt::Display for TemporalErrorDebtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyWindows => write!(f, "temporal error debt requires at least one exposure window"),
            Self::InvalidOracleTimeline => write!(f, "knowledge oracle correction/retraction timeline is invalid"),
            Self::ZeroDuration { query_id, checkpoint_index } => write!(
                f,
                "query {query_id} checkpoint {checkpoint_index} has zero exposure duration"
            ),
            Self::InvalidWindowOrder { query_id, checkpoint_index } => write!(
                f,
                "query {query_id} checkpoint {checkpoint_index} is duplicated or out of order"
            ),
            Self::NonContiguousTimeline { query_id, checkpoint_index } => write!(
                f,
                "query {query_id} checkpoint {checkpoint_index} does not begin when the prior interval ends"
            ),
            Self::ValidTimeChanged { query_id, checkpoint_index } => write!(
                f,
                "query {query_id} changes valid_at within its trajectory at checkpoint {checkpoint_index}"
            ),
            Self::TimelineOverflow { query_id, checkpoint_index } => write!(
                f,
                "query {query_id} checkpoint {checkpoint_index} interval end overflows"
            ),
            Self::DuplicatePrediction { query_id, checkpoint_index } => write!(
                f,
                "candidate submitted duplicate prediction for query {query_id} checkpoint {checkpoint_index}"
            ),
            Self::UnknownPrediction { query_id, checkpoint_index } => write!(
                f,
                "candidate submitted a prediction for unknown query {query_id} checkpoint {checkpoint_index}"
            ),
        }
    }
}

impl std::error::Error for TemporalErrorDebtError {}

/// Time-integrated error and output-state exposure.
///
/// All exposure values use the exact integer duration units supplied by the
/// windows. Wrong-belief exposure includes incorrect Value/NoFact answers.
/// Abstention and missing output are separate components, not correct answers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalErrorDebtScore {
    pub expected_windows: usize,
    pub predictions_received: usize,
    pub missing_predictions: usize,
    pub correct_windows: usize,
    pub wrong_belief_windows: usize,
    pub abstention_windows: usize,
    pub wrong_belief_exposure: u128,
    pub correct_exposure: u128,
    pub abstention_exposure: u128,
    pub missing_exposure: u128,
    pub future_knowledge_leak_windows: usize,
    pub future_knowledge_leak_exposure: u128,
    pub missed_correction_windows: usize,
    pub missed_correction_exposure: u128,
    pub stale_after_retraction_windows: usize,
    pub stale_after_retraction_exposure: u128,
    /// Knowledge-time lag from correction arrival to first correct corrected value.
    pub correction_latency: Option<u64>,
    /// Knowledge-time lag from retraction arrival to first correct NoFact response.
    pub retraction_latency: Option<u64>,
}

impl TemporalErrorDebtScore {
    pub fn total_observed_exposure(&self) -> u128 {
        self.wrong_belief_exposure
            .saturating_add(self.correct_exposure)
            .saturating_add(self.abstention_exposure)
            .saturating_add(self.missing_exposure)
    }

    pub fn full_stream_prediction_coverage_ppm(&self) -> u32 {
        if self.expected_windows == 0 {
            return 0;
        }
        u32::try_from(
            (u128::try_from(self.predictions_received).unwrap_or(u128::MAX) * 1_000_000)
                / u128::try_from(self.expected_windows).unwrap_or(u128::MAX),
        )
        .unwrap_or(u32::MAX)
    }
}

/// Score repeated bitemporal predictions as an exposure integral.
///
/// A query's valid_at coordinate is held constant while known_at advances.
/// The expected answer is independently resolved from the evaluator-owned
/// revision timeline, not copied from the oracle's precomputed query labels.
///
/// Per interval:
/// - incorrect Value/NoFact -> wrong-belief exposure;
/// - Abstain -> abstention exposure;
/// - absent prediction -> missing exposure;
/// - correct Value/NoFact -> correct exposure.
///
/// Future-leak, missed-correction, and stale-after-retraction exposures are
/// subsets of wrong-belief exposure and are also reported separately.
pub fn score_temporal_error_debt(
    oracle: &KnowledgeStreamOracle,
    windows: &[KnowledgeExposureWindow],
    predictions: &[KnowledgeTimedPrediction],
) -> Result<TemporalErrorDebtScore, TemporalErrorDebtError> {
    if windows.is_empty() {
        return Err(TemporalErrorDebtError::EmptyWindows);
    }
    if oracle.correction_valid_from >= oracle.retraction_valid_from
        || oracle.correction_observed_at >= oracle.retraction_observed_at
        || oracle.target_fact_id == 0
    {
        return Err(TemporalErrorDebtError::InvalidOracleTimeline);
    }

    let mut window_keys = BTreeSet::new();
    let mut previous_by_query: BTreeMap<u64, KnowledgeExposureWindow> = BTreeMap::new();
    for window in windows {
        let key = (window.query_id, window.checkpoint_index);
        if !window_keys.insert(key) {
            return Err(TemporalErrorDebtError::InvalidWindowOrder {
                query_id: window.query_id,
                checkpoint_index: window.checkpoint_index,
            });
        }
        if window.duration == 0 {
            return Err(TemporalErrorDebtError::ZeroDuration {
                query_id: window.query_id,
                checkpoint_index: window.checkpoint_index,
            });
        }
        window.known_at.checked_add(window.duration).ok_or(
            TemporalErrorDebtError::TimelineOverflow {
                query_id: window.query_id,
                checkpoint_index: window.checkpoint_index,
            },
        )?;

        if let Some(previous) = previous_by_query.get(&window.query_id).copied() {
            if window.checkpoint_index != previous.checkpoint_index.saturating_add(1) {
                return Err(TemporalErrorDebtError::InvalidWindowOrder {
                    query_id: window.query_id,
                    checkpoint_index: window.checkpoint_index,
                });
            }
            if window.valid_at != previous.valid_at {
                return Err(TemporalErrorDebtError::ValidTimeChanged {
                    query_id: window.query_id,
                    checkpoint_index: window.checkpoint_index,
                });
            }
            let previous_end = previous.known_at.checked_add(previous.duration).ok_or(
                TemporalErrorDebtError::TimelineOverflow {
                    query_id: window.query_id,
                    checkpoint_index: previous.checkpoint_index,
                },
            )?;
            if window.known_at != previous_end {
                return Err(TemporalErrorDebtError::NonContiguousTimeline {
                    query_id: window.query_id,
                    checkpoint_index: window.checkpoint_index,
                });
            }
        } else if window.checkpoint_index != 0 {
            return Err(TemporalErrorDebtError::InvalidWindowOrder {
                query_id: window.query_id,
                checkpoint_index: window.checkpoint_index,
            });
        }
        previous_by_query.insert(window.query_id, *window);
    }

    let mut window_by_key = BTreeMap::new();
    for window in windows {
        window_by_key.insert((window.query_id, window.checkpoint_index), *window);
    }

    let mut predictions_by_key = BTreeMap::new();
    for prediction in predictions {
        let key = (prediction.query_id, prediction.checkpoint_index);
        if !window_by_key.contains_key(&key) {
            return Err(TemporalErrorDebtError::UnknownPrediction {
                query_id: prediction.query_id,
                checkpoint_index: prediction.checkpoint_index,
            });
        }
        if predictions_by_key.insert(key, prediction.answer).is_some() {
            return Err(TemporalErrorDebtError::DuplicatePrediction {
                query_id: prediction.query_id,
                checkpoint_index: prediction.checkpoint_index,
            });
        }
    }

    let mut score = TemporalErrorDebtScore {
        expected_windows: windows.len(),
        predictions_received: predictions_by_key.len(),
        missing_predictions: 0,
        correct_windows: 0,
        wrong_belief_windows: 0,
        abstention_windows: 0,
        wrong_belief_exposure: 0,
        correct_exposure: 0,
        abstention_exposure: 0,
        missing_exposure: 0,
        future_knowledge_leak_windows: 0,
        future_knowledge_leak_exposure: 0,
        missed_correction_windows: 0,
        missed_correction_exposure: 0,
        stale_after_retraction_windows: 0,
        stale_after_retraction_exposure: 0,
        correction_latency: None,
        retraction_latency: None,
    };

    for window in windows {
        let duration = u128::from(window.duration);
        let prediction = predictions_by_key
            .get(&(window.query_id, window.checkpoint_index))
            .copied();
        let expected_value = resolve_expected_value(oracle, window.valid_at, window.known_at);

        let Some(prediction) = prediction else {
            score.missing_predictions = score.missing_predictions.saturating_add(1);
            score.missing_exposure = score.missing_exposure.saturating_add(duration);
            continue;
        };

        match prediction {
            KnowledgeCandidateAnswer::Abstain => {
                score.abstention_windows = score.abstention_windows.saturating_add(1);
                score.abstention_exposure = score.abstention_exposure.saturating_add(duration);
            }
            KnowledgeCandidateAnswer::NoFact => {
                if expected_value.is_none() {
                    score.correct_windows = score.correct_windows.saturating_add(1);
                    score.correct_exposure = score.correct_exposure.saturating_add(duration);
                    update_retraction_latency(oracle, *window, &mut score);
                } else {
                    score.wrong_belief_windows = score.wrong_belief_windows.saturating_add(1);
                    score.wrong_belief_exposure =
                        score.wrong_belief_exposure.saturating_add(duration);
                    classify_wrong_nofact(oracle, *window, &mut score);
                }
            }
            KnowledgeCandidateAnswer::Value(value) => {
                if expected_value == Some(value) {
                    score.correct_windows = score.correct_windows.saturating_add(1);
                    score.correct_exposure = score.correct_exposure.saturating_add(duration);
                    update_correction_latency(oracle, *window, value, &mut score);
                } else {
                    score.wrong_belief_windows = score.wrong_belief_windows.saturating_add(1);
                    score.wrong_belief_exposure =
                        score.wrong_belief_exposure.saturating_add(duration);
                    classify_wrong_value(oracle, *window, value, &mut score);
                }
            }
        }
    }
    Ok(score)
}

/// Resolve the target fact using both valid time and knowledge/transaction time.
pub fn resolve_expected_value(
    oracle: &KnowledgeStreamOracle,
    valid_at: u64,
    known_at: u64,
) -> Option<i64> {
    let mut selected_revision: Option<(u32, KnowledgeOperationRef, Option<i64>)> = None;
    let revisions = [
        (
            0_u32,
            0_u64,
            0_u64,
            Some(oracle.initial_value),
            KnowledgeOperationRef::Assert,
        ),
        (
            1_u32,
            oracle.correction_valid_from,
            oracle.correction_observed_at,
            Some(oracle.corrected_value),
            KnowledgeOperationRef::Correct,
        ),
        (
            2_u32,
            oracle.retraction_valid_from,
            oracle.retraction_observed_at,
            None,
            KnowledgeOperationRef::Retract,
        ),
    ];
    for (revision, valid_from, observed_at, value, operation) in revisions {
        if valid_from <= valid_at
            && observed_at <= known_at
            && selected_revision.is_none_or(|(current, _, _)| revision > current)
        {
            selected_revision = Some((revision, operation, value));
        }
    }
    match selected_revision {
        Some((_, KnowledgeOperationRef::Retract, _)) => None,
        Some((_, _, value)) => value,
        None => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KnowledgeOperationRef {
    Assert,
    Correct,
    Retract,
}

fn classify_wrong_value(
    oracle: &KnowledgeStreamOracle,
    window: KnowledgeExposureWindow,
    value: i64,
    score: &mut TemporalErrorDebtScore,
) {
    let duration = u128::from(window.duration);
    if window.known_at < oracle.correction_observed_at
        && window.valid_at >= oracle.correction_valid_from
        && value == oracle.corrected_value
    {
        score.future_knowledge_leak_windows = score.future_knowledge_leak_windows.saturating_add(1);
        score.future_knowledge_leak_exposure =
            score.future_knowledge_leak_exposure.saturating_add(duration);
    }

    if window.known_at >= oracle.correction_observed_at
        && window.known_at < oracle.retraction_observed_at
        && window.valid_at >= oracle.correction_valid_from
        && value == oracle.initial_value
    {
        score.missed_correction_windows = score.missed_correction_windows.saturating_add(1);
        score.missed_correction_exposure =
            score.missed_correction_exposure.saturating_add(duration);
    }

    if window.known_at >= oracle.retraction_observed_at
        && window.valid_at >= oracle.retraction_valid_from
    {
        score.stale_after_retraction_windows =
            score.stale_after_retraction_windows.saturating_add(1);
        score.stale_after_retraction_exposure =
            score.stale_after_retraction_exposure.saturating_add(duration);
    }
}

fn classify_wrong_nofact(
    oracle: &KnowledgeStreamOracle,
    window: KnowledgeExposureWindow,
    score: &mut TemporalErrorDebtScore,
) {
    if window.known_at >= oracle.correction_observed_at
        && window.known_at < oracle.retraction_observed_at
        && window.valid_at >= oracle.correction_valid_from
    {
        score.missed_correction_windows = score.missed_correction_windows.saturating_add(1);
        score.missed_correction_exposure = score
            .missed_correction_exposure
            .saturating_add(u128::from(window.duration));
    }
}

fn update_correction_latency(
    oracle: &KnowledgeStreamOracle,
    window: KnowledgeExposureWindow,
    value: i64,
    score: &mut TemporalErrorDebtScore,
) {
    if value != oracle.corrected_value
        || window.known_at < oracle.correction_observed_at
        || window.known_at >= oracle.retraction_observed_at
        || window.valid_at < oracle.correction_valid_from
    {
        return;
    }
    let latency = window.known_at.saturating_sub(oracle.correction_observed_at);
    score.correction_latency =
        Some(score.correction_latency.map_or(latency, |current| current.min(latency)));
}

fn update_retraction_latency(
    oracle: &KnowledgeStreamOracle,
    window: KnowledgeExposureWindow,
    score: &mut TemporalErrorDebtScore,
) {
    if window.known_at < oracle.retraction_observed_at
        || window.valid_at < oracle.retraction_valid_from
    {
        return;
    }
    let latency = window.known_at.saturating_sub(oracle.retraction_observed_at);
    score.retraction_latency =
        Some(score.retraction_latency.map_or(latency, |current| current.min(latency)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_stream::{generate_knowledge_stream, KnowledgeStreamConfig};

    fn fixture() -> (KnowledgeStreamOracle, Vec<KnowledgeExposureWindow>) {
        let config = KnowledgeStreamConfig {
            seed: 0x5EED_9492,
            distractor_count: 4,
            interval_steps: 16,
        };
        let (_, oracle) = generate_knowledge_stream(config).unwrap();
        let windows = vec![
            KnowledgeExposureWindow {
                query_id: 77,
                checkpoint_index: 0,
                valid_at: oracle.retraction_valid_from + 1,
                known_at: 0,
                duration: oracle.correction_observed_at,
            },
            KnowledgeExposureWindow {
                query_id: 77,
                checkpoint_index: 1,
                valid_at: oracle.retraction_valid_from + 1,
                known_at: oracle.correction_observed_at,
                duration: 8,
            },
            KnowledgeExposureWindow {
                query_id: 77,
                checkpoint_index: 2,
                valid_at: oracle.retraction_valid_from + 1,
                known_at: oracle.correction_observed_at + 8,
                duration: oracle.retraction_observed_at - oracle.correction_observed_at - 8,
            },
            KnowledgeExposureWindow {
                query_id: 77,
                checkpoint_index: 3,
                valid_at: oracle.retraction_valid_from + 1,
                known_at: oracle.retraction_observed_at,
                duration: 10,
            },
        ];
        (oracle, windows)
    }

    fn predictions(
        first: KnowledgeCandidateAnswer,
        after_correction_early: KnowledgeCandidateAnswer,
        after_correction_late: KnowledgeCandidateAnswer,
        after_retraction: KnowledgeCandidateAnswer,
    ) -> Vec<KnowledgeTimedPrediction> {
        vec![
            KnowledgeTimedPrediction { query_id: 77, checkpoint_index: 0, answer: first },
            KnowledgeTimedPrediction { query_id: 77, checkpoint_index: 1, answer: after_correction_early },
            KnowledgeTimedPrediction { query_id: 77, checkpoint_index: 2, answer: after_correction_late },
            KnowledgeTimedPrediction { query_id: 77, checkpoint_index: 3, answer: after_retraction },
        ]
    }

    #[test]
    fn event_aligned_candidate_has_zero_temporal_error_debt() {
        let (oracle, windows) = fixture();
        let candidate = predictions(
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::NoFact,
        );
        let score = score_temporal_error_debt(&oracle, &windows, &candidate).unwrap();
        assert_eq!(score.wrong_belief_exposure, 0);
        assert_eq!(score.abstention_exposure, 0);
        assert_eq!(score.missing_exposure, 0);
        assert_eq!(score.future_knowledge_leak_exposure, 0);
        assert_eq!(score.missed_correction_exposure, 0);
        assert_eq!(score.stale_after_retraction_exposure, 0);
        assert_eq!(score.correction_latency, Some(0));
        assert_eq!(score.retraction_latency, Some(0));
        assert_eq!(score.full_stream_prediction_coverage_ppm(), 1_000_000);
    }

    #[test]
    fn same_final_state_but_late_update_accumulates_wrong_belief_exposure() {
        let (oracle, windows) = fixture();
        let candidate = predictions(
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::NoFact,
        );
        let score = score_temporal_error_debt(&oracle, &windows, &candidate).unwrap();
        assert_eq!(score.wrong_belief_exposure, 8);
        assert_eq!(score.missed_correction_exposure, 8);
        assert_eq!(score.stale_after_retraction_exposure, 0);
        assert_eq!(score.correction_latency, Some(8));
        assert_eq!(score.retraction_latency, Some(0));
    }

    #[test]
    fn no_fact_during_corrected_valid_interval_counts_as_missed_correction() {
        let (oracle, windows) = fixture();
        let candidate = predictions(
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::NoFact,
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::NoFact,
        );
        let score = score_temporal_error_debt(&oracle, &windows, &candidate).unwrap();
        assert_eq!(score.wrong_belief_exposure, 8);
        assert_eq!(score.missed_correction_exposure, 8);
        assert_eq!(score.correction_latency, Some(8));
    }

    #[test]
    fn stale_value_after_known_retraction_accumulates_its_own_exposure() {
        let (oracle, windows) = fixture();
        let candidate = predictions(
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
        );
        let score = score_temporal_error_debt(&oracle, &windows, &candidate).unwrap();
        assert_eq!(score.stale_after_retraction_exposure, 10);
        assert_eq!(score.wrong_belief_exposure, 10);
        assert_eq!(score.retraction_latency, None);
    }

    #[test]
    fn abstention_and_missing_output_are_integrated_separately() {
        let (oracle, windows) = fixture();
        let candidate = vec![
            KnowledgeTimedPrediction {
                query_id: 77,
                checkpoint_index: 0,
                answer: KnowledgeCandidateAnswer::Value(oracle.initial_value),
            },
            KnowledgeTimedPrediction {
                query_id: 77,
                checkpoint_index: 1,
                answer: KnowledgeCandidateAnswer::Abstain,
            },
            KnowledgeTimedPrediction {
                query_id: 77,
                checkpoint_index: 2,
                answer: KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            },
        ];
        let score = score_temporal_error_debt(&oracle, &windows, &candidate).unwrap();
        assert_eq!(score.abstention_exposure, 8);
        assert_eq!(score.missing_exposure, 10);
        assert_eq!(score.wrong_belief_exposure, 0);
        assert_eq!(score.missing_predictions, 1);
        assert_eq!(score.predictions_received, 3);
        assert_eq!(score.full_stream_prediction_coverage_ppm(), 750_000);
    }

    #[test]
    fn future_correction_answer_before_known_time_is_counted_as_leakage() {
        let (oracle, windows) = fixture();
        let candidate = predictions(
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::NoFact,
        );
        let score = score_temporal_error_debt(&oracle, &windows, &candidate).unwrap();
        assert_eq!(score.future_knowledge_leak_exposure, u128::from(oracle.correction_observed_at));
        assert_eq!(score.wrong_belief_exposure, u128::from(oracle.correction_observed_at));
    }

    #[test]
    fn query_trajectory_must_be_contiguous() {
        let (oracle, mut windows) = fixture();
        windows[2].known_at = windows[2].known_at.saturating_add(1);
        assert!(matches!(
            score_temporal_error_debt(&oracle, &windows, &[]),
            Err(TemporalErrorDebtError::NonContiguousTimeline { .. })
        ));
    }

    #[test]
    fn query_valid_time_cannot_change_within_a_trajectory() {
        let (oracle, mut windows) = fixture();
        windows[2].valid_at = windows[2].valid_at.saturating_sub(1);
        assert!(matches!(
            score_temporal_error_debt(&oracle, &windows, &[]),
            Err(TemporalErrorDebtError::ValidTimeChanged { .. })
        ));
    }

    #[test]
    fn zero_duration_windows_fail_closed() {
        let (oracle, mut windows) = fixture();
        windows[1].duration = 0;
        assert!(matches!(
            score_temporal_error_debt(&oracle, &windows, &[]),
            Err(TemporalErrorDebtError::ZeroDuration { .. })
        ));
    }

    #[test]
    fn duplicate_and_unknown_predictions_are_rejected() {
        let (oracle, windows) = fixture();
        let duplicate = [
            KnowledgeTimedPrediction { query_id: 77, checkpoint_index: 0, answer: KnowledgeCandidateAnswer::Abstain },
            KnowledgeTimedPrediction { query_id: 77, checkpoint_index: 0, answer: KnowledgeCandidateAnswer::Abstain },
        ];
        assert!(matches!(
            score_temporal_error_debt(&oracle, &windows, &duplicate),
            Err(TemporalErrorDebtError::DuplicatePrediction { .. })
        ));
        let unknown = [KnowledgeTimedPrediction {
            query_id: 77,
            checkpoint_index: 99,
            answer: KnowledgeCandidateAnswer::Abstain,
        }];
        assert!(matches!(
            score_temporal_error_debt(&oracle, &windows, &unknown),
            Err(TemporalErrorDebtError::UnknownPrediction { .. })
        ));
    }

    #[test]
    fn same_final_answer_can_hide_different_time_integrated_debt() {
        let (oracle, windows) = fixture();
        let immediate = predictions(
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::NoFact,
        );
        let delayed = predictions(
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::Value(oracle.initial_value),
            KnowledgeCandidateAnswer::Value(oracle.corrected_value),
            KnowledgeCandidateAnswer::NoFact,
        );
        let a = score_temporal_error_debt(&oracle, &windows, &immediate).unwrap();
        let b = score_temporal_error_debt(&oracle, &windows, &delayed).unwrap();
        assert_eq!(immediate.last().unwrap().answer, delayed.last().unwrap().answer);
        assert_eq!(a.correct_windows, 4);
        assert_eq!(b.correct_windows, 3);
        assert!(a.wrong_belief_exposure < b.wrong_belief_exposure);
    }
}
