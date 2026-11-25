//! WorldTest/AutumnBench reward-free exploration trace diagnostics.
//!
//! This module scores exploration behaviour around explicit environment resets.
//! It is representation-agnostic and evaluator-side. It does not score query
//! correctness, infer a causal model, or authenticate source observations.
//!
//! The replay statistic is a Symthaea-defined diagnostic: the longest-common-
//! subsequence length of consecutive nonempty exploration segments divided by
//! the longer segment length. It is not asserted to reproduce the exact
//! normalization of the external paper's LCS figure.
//!
//! Reset count and replay similarity are descriptors, not objectives. A system
//! must not be rewarded for resetting more often. The downstream query result,
//! interaction budget, task cost, and exploration/test separation remain the
//! primary evaluation contract.

use std::fmt;

pub const WORLDTEST_TRACE_MAX_EVENTS: usize = 16_384;
pub const WORLDTEST_SEGMENT_MAX_ACTIONS: usize = 512;
pub const WORLDTEST_RATIO_PPM: u32 = 1_000_000;
const MAX_ACTION_ID_BYTES: usize = 128;

/// Events visible in an exploration trace.
/// BeginTest is a terminal phase boundary: no further exploration events
/// may appear in this record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExplorationEvent {
    /// Any candidate-issued environment action, including noop.
    Action(String),
    /// Explicit native environment reset.
    Reset,
    /// End reward-free exploration and enter the separately scored test phase.
    BeginTest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplorationTrace {
    pub events: Vec<ExplorationEvent>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResetReplayPair {
    pub before_action_count: u64,
    pub after_action_count: u64,
    pub lcs_length: u64,
    /// LCS length divided by max(before_action_count, after_action_count),
    /// multiplied by 1,000,000 and rounded down.
    pub replay_lcs_ratio_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTestExplorationCertificate {
    pub exploration_action_count: u64,
    pub reset_count: u64,
    /// reset_count / (exploration_action_count + reset_count), in PPM.
    /// Reset events remain charged in the interaction denominator.
    pub reset_share_ppm: u32,
    /// Number of nonempty action segments separated by reset events.
    pub nonempty_segment_count: u64,
    pub reset_replay_pair_count: u64,
    pub mean_reset_replay_lcs_ratio_ppm: u32,
    pub median_reset_replay_lcs_ratio_ppm: u32,
    pub reset_replay_pairs: Vec<ResetReplayPair>,
    pub test_phase_started: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldTestExplorationError {
    EmptyTrace,
    TooManyEvents { requested: usize, maximum: usize },
    EmptyActionId,
    ActionIdTooLong { bytes: usize, maximum: usize },
    TooManyActionsInSegment { requested: usize, maximum: usize },
    DuplicateTestTransition,
    MissingTestTransition,
    EventAfterTestTransition { index: usize },
    ArithmeticOverflow,
}

impl fmt::Display for WorldTestExplorationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTrace => write!(formatter, "WorldTest exploration trace must not be empty"),
            Self::TooManyEvents { requested, maximum } => write!(
                formatter,
                "WorldTest trace event count {requested} exceeds the bounded maximum {maximum}"
            ),
            Self::EmptyActionId => write!(formatter, "exploration action IDs must be nonempty and trimmed"),
            Self::ActionIdTooLong { bytes, maximum } => write!(
                formatter,
                "exploration action ID is {bytes} bytes; maximum is {maximum}"
            ),
            Self::TooManyActionsInSegment { requested, maximum } => write!(
                formatter,
                "exploration segment contains {requested} actions; maximum is {maximum}"
            ),
            Self::DuplicateTestTransition => write!(formatter, "WorldTest trace contains multiple test-phase transitions"),
            Self::MissingTestTransition => write!(formatter, "WorldTest trace is missing its test-phase transition"),
            Self::EventAfterTestTransition { index } => write!(
                formatter,
                "exploration event at index {index} occurs after the test-phase transition"
            ),
            Self::ArithmeticOverflow => write!(formatter, "WorldTest exploration score arithmetic overflowed"),
        }
    }
}

impl std::error::Error for WorldTestExplorationError {}

fn push_nonempty_segment(
    current: &mut Vec<String>,
    segments: &mut Vec<Vec<String>>,
) -> Result<(), WorldTestExplorationError> {
    if !current.is_empty() {
        if current.len() > WORLDTEST_SEGMENT_MAX_ACTIONS {
            return Err(WorldTestExplorationError::TooManyActionsInSegment {
                requested: current.len(),
                maximum: WORLDTEST_SEGMENT_MAX_ACTIONS,
            });
        }
        segments.push(std::mem::take(current));
    }
    Ok(())
}

/// Return the LCS length using two rows of dynamic-programming memory.
fn longest_common_subsequence_length(
    left: &[String],
    right: &[String],
) -> Result<usize, WorldTestExplorationError> {
    let (outer, inner) = if left.len() >= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let mut previous = vec![0_usize; inner.len() + 1];
    let mut current = vec![0_usize; inner.len() + 1];

    for outer_action in outer {
        current[0] = 0;
        for (index, inner_action) in inner.iter().enumerate() {
            current[index + 1] = if outer_action == inner_action {
                previous[index]
                    .checked_add(1)
                    .ok_or(WorldTestExplorationError::ArithmeticOverflow)?
            } else {
                current[index].max(previous[index + 1])
            };
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok(previous[inner.len()])
}

fn replay_pair(
    before: &[String],
    after: &[String],
) -> Result<ResetReplayPair, WorldTestExplorationError> {
    let maximum_length = before.len().max(after.len());
    if maximum_length == 0 {
        return Err(WorldTestExplorationError::ArithmeticOverflow);
    }
    let lcs = longest_common_subsequence_length(before, after)?;
    let numerator = u128::try_from(lcs)
        .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?
        .checked_mul(u128::from(WORLDTEST_RATIO_PPM))
        .ok_or(WorldTestExplorationError::ArithmeticOverflow)?;
    let ratio = numerator
        / u128::try_from(maximum_length)
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?;
    Ok(ResetReplayPair {
        before_action_count: u64::try_from(before.len())
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?,
        after_action_count: u64::try_from(after.len())
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?,
        lcs_length: u64::try_from(lcs)
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?,
        replay_lcs_ratio_ppm: u32::try_from(ratio)
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?,
    })
}

/// Validate and summarize one complete reward-free exploration trace.
///
/// Every Reset increments the reset count, even if the current segment is
/// empty; the reset is a charged interaction, not free metadata. Only nonempty
/// action segments contribute to replay similarity. Replay similarity is
/// computed between adjacent nonempty segments. Exactly one terminal BeginTest
/// event is required. Actions and resets after BeginTest are rejected.
pub fn evaluate_worldtest_exploration(
    trace: &ExplorationTrace,
) -> Result<WorldTestExplorationCertificate, WorldTestExplorationError> {
    if trace.events.is_empty() {
        return Err(WorldTestExplorationError::EmptyTrace);
    }
    if trace.events.len() > WORLDTEST_TRACE_MAX_EVENTS {
        return Err(WorldTestExplorationError::TooManyEvents {
            requested: trace.events.len(),
            maximum: WORLDTEST_TRACE_MAX_EVENTS,
        });
    }

    let mut current_segment = Vec::<String>::new();
    let mut segments = Vec::<Vec<String>>::new();
    let mut action_count = 0_u64;
    let mut reset_count = 0_u64;
    let mut test_started = false;

    for (index, event) in trace.events.iter().enumerate() {
        if test_started {
            if matches!(event, ExplorationEvent::BeginTest) {
                return Err(WorldTestExplorationError::DuplicateTestTransition);
            }
            return Err(WorldTestExplorationError::EventAfterTestTransition { index });
        }
        match event {
            ExplorationEvent::Action(action_id) => {
                if action_id.trim().is_empty() || action_id.trim() != action_id.as_str() {
                    return Err(WorldTestExplorationError::EmptyActionId);
                }
                if action_id.len() > MAX_ACTION_ID_BYTES {
                    return Err(WorldTestExplorationError::ActionIdTooLong {
                        bytes: action_id.len(),
                        maximum: MAX_ACTION_ID_BYTES,
                    });
                }
                if current_segment.len() >= WORLDTEST_SEGMENT_MAX_ACTIONS {
                    return Err(WorldTestExplorationError::TooManyActionsInSegment {
                        requested: current_segment.len() + 1,
                        maximum: WORLDTEST_SEGMENT_MAX_ACTIONS,
                    });
                }
                current_segment.push(action_id.clone());
                action_count = action_count
                    .checked_add(1)
                    .ok_or(WorldTestExplorationError::ArithmeticOverflow)?;
            }
            ExplorationEvent::Reset => {
                reset_count = reset_count
                    .checked_add(1)
                    .ok_or(WorldTestExplorationError::ArithmeticOverflow)?;
                push_nonempty_segment(&mut current_segment, &mut segments)?;
            }
            ExplorationEvent::BeginTest => {
                push_nonempty_segment(&mut current_segment, &mut segments)?;
                test_started = true;
            }
        }
    }

    if !test_started {
        return Err(WorldTestExplorationError::MissingTestTransition);
    }

    let mut replay_pairs = Vec::with_capacity(segments.len().saturating_sub(1));
    for pair in segments.windows(2) {
        replay_pairs.push(replay_pair(&pair[0], &pair[1])?);
    }

    let total_interactions = u128::from(action_count)
        .checked_add(u128::from(reset_count))
        .ok_or(WorldTestExplorationError::ArithmeticOverflow)?;
    let reset_share_ppm = if total_interactions == 0 {
        0
    } else {
        let numerator = u128::from(reset_count)
            .checked_mul(u128::from(WORLDTEST_RATIO_PPM))
            .ok_or(WorldTestExplorationError::ArithmeticOverflow)?;
        u32::try_from(numerator / total_interactions)
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?
    };

    let mut ratios: Vec<u32> = replay_pairs
        .iter()
        .map(|pair| pair.replay_lcs_ratio_ppm)
        .collect();
    ratios.sort_unstable();
    let (mean, median) = if ratios.is_empty() {
        (0, 0)
    } else {
        let sum = ratios.iter().try_fold(0_u128, |total, ratio| {
            total
                .checked_add(u128::from(*ratio))
                .ok_or(WorldTestExplorationError::ArithmeticOverflow)
        })?;
        let mean = u32::try_from(sum / ratios.len() as u128)
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?;
        let middle = ratios.len() / 2;
        let median = if ratios.len() % 2 == 0 {
            let pair_sum = u64::from(ratios[middle - 1])
                .checked_add(u64::from(ratios[middle]))
                .ok_or(WorldTestExplorationError::ArithmeticOverflow)?;
            u32::try_from(pair_sum / 2)
                .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?
        } else {
            ratios[middle]
        };
        (mean, median)
    };

    Ok(WorldTestExplorationCertificate {
        exploration_action_count: action_count,
        reset_count,
        reset_share_ppm,
        nonempty_segment_count: u64::try_from(segments.len())
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?,
        reset_replay_pair_count: u64::try_from(replay_pairs.len())
            .map_err(|_| WorldTestExplorationError::ArithmeticOverflow)?,
        mean_reset_replay_lcs_ratio_ppm: mean,
        median_reset_replay_lcs_ratio_ppm: median,
        reset_replay_pairs: replay_pairs,
        test_phase_started: test_started,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(value: &str) -> ExplorationEvent {
        ExplorationEvent::Action(value.to_owned())
    }

    fn trace(events: Vec<ExplorationEvent>) -> ExplorationTrace {
        ExplorationTrace { events }
    }

    #[test]
    fn empty_trace_is_rejected() {
        assert_eq!(
            evaluate_worldtest_exploration(&trace(Vec::new())),
            Err(WorldTestExplorationError::EmptyTrace)
        );
    }

    #[test]
    fn exact_replay_after_reset_scores_one_million_ppm() {
        let result = evaluate_worldtest_exploration(&trace(vec![
            action("left"),
            action("right"),
            action("noop"),
            ExplorationEvent::Reset,
            action("left"),
            action("right"),
            action("noop"),
            ExplorationEvent::BeginTest,
        ]))
        .unwrap();
        assert_eq!(result.exploration_action_count, 6);
        assert_eq!(result.reset_count, 1);
        assert_eq!(result.reset_share_ppm, 142_857);
        assert_eq!(result.nonempty_segment_count, 2);
        assert_eq!(result.reset_replay_pair_count, 1);
        assert_eq!(result.mean_reset_replay_lcs_ratio_ppm, 1_000_000);
        assert_eq!(result.median_reset_replay_lcs_ratio_ppm, 1_000_000);
        assert_eq!(result.reset_replay_pairs[0].lcs_length, 3);
    }

    #[test]
    fn ordered_subsequence_not_set_overlap_controls_replay_similarity() {
        let result = evaluate_worldtest_exploration(&trace(vec![
            action("left"),
            action("right"),
            action("up"),
            ExplorationEvent::Reset,
            action("left"),
            action("up"),
            ExplorationEvent::BeginTest,
        ]))
        .unwrap();
        assert_eq!(result.reset_replay_pairs[0].lcs_length, 2);
        assert_eq!(result.mean_reset_replay_lcs_ratio_ppm, 666_666);
    }

    #[test]
    fn disjoint_replay_sequences_score_zero() {
        let result = evaluate_worldtest_exploration(&trace(vec![
            action("left"),
            action("right"),
            ExplorationEvent::Reset,
            action("click:1:1"),
            action("click:2:2"),
            ExplorationEvent::BeginTest,
        ]))
        .unwrap();
        assert_eq!(result.mean_reset_replay_lcs_ratio_ppm, 0);
    }

    #[test]
    fn no_reset_has_zero_replay_similarity_not_a_perfect_score() {
        let result = evaluate_worldtest_exploration(&trace(vec![
            action("left"),
            action("noop"),
            ExplorationEvent::BeginTest,
        ]))
        .unwrap();
        assert_eq!(result.reset_count, 0);
        assert_eq!(result.reset_share_ppm, 0);
        assert_eq!(result.reset_replay_pair_count, 0);
        assert_eq!(result.mean_reset_replay_lcs_ratio_ppm, 0);
    }

    #[test]
    fn reset_cost_remains_in_interaction_denominator() {
        let result = evaluate_worldtest_exploration(&trace(vec![
            action("left"),
            ExplorationEvent::Reset,
            action("left"),
            ExplorationEvent::Reset,
            ExplorationEvent::BeginTest,
        ]))
        .unwrap();
        assert_eq!(result.exploration_action_count, 2);
        assert_eq!(result.reset_count, 2);
        assert_eq!(result.reset_share_ppm, 500_000);
        assert_eq!(result.reset_replay_pair_count, 1);
        assert_eq!(result.mean_reset_replay_lcs_ratio_ppm, 1_000_000);
    }

    #[test]
    fn empty_resets_are_counted_but_do_not_create_fake_replay_segments() {
        let result = evaluate_worldtest_exploration(&trace(vec![
            ExplorationEvent::Reset,
            action("left"),
            ExplorationEvent::Reset,
            ExplorationEvent::Reset,
            action("right"),
            ExplorationEvent::BeginTest,
        ]))
        .unwrap();
        assert_eq!(result.reset_count, 3);
        assert_eq!(result.nonempty_segment_count, 2);
        assert_eq!(result.reset_replay_pair_count, 1);
        assert_eq!(result.mean_reset_replay_lcs_ratio_ppm, 0);
    }

    #[test]
    fn median_aggregates_multiple_resets_without_pooling_action_rows() {
        let result = evaluate_worldtest_exploration(&trace(vec![
            action("a"),
            action("b"),
            ExplorationEvent::Reset,
            action("a"),
            action("c"),
            ExplorationEvent::Reset,
            action("a"),
            ExplorationEvent::Reset,
            action("b"),
            ExplorationEvent::BeginTest,
        ]))
        .unwrap();
        assert_eq!(result.reset_replay_pair_count, 3);
        assert_eq!(result.reset_replay_pairs[0].replay_lcs_ratio_ppm, 500_000);
        assert_eq!(result.reset_replay_pairs[1].replay_lcs_ratio_ppm, 500_000);
        assert_eq!(result.reset_replay_pairs[2].replay_lcs_ratio_ppm, 0);
        assert_eq!(result.mean_reset_replay_lcs_ratio_ppm, 333_333);
        assert_eq!(result.median_reset_replay_lcs_ratio_ppm, 500_000);
    }

    #[test]
    fn missing_test_transition_is_rejected() {
        assert_eq!(
            evaluate_worldtest_exploration(&trace(vec![action("left")])),
            Err(WorldTestExplorationError::MissingTestTransition)
        );
    }

    #[test]
    fn duplicate_test_transition_is_rejected() {
        assert_eq!(
            evaluate_worldtest_exploration(&trace(vec![
                action("left"),
                ExplorationEvent::BeginTest,
                ExplorationEvent::BeginTest,
            ])),
            Err(WorldTestExplorationError::DuplicateTestTransition)
        );
    }

    #[test]
    fn actions_after_test_transition_are_rejected() {
        assert_eq!(
            evaluate_worldtest_exploration(&trace(vec![
                action("left"),
                ExplorationEvent::BeginTest,
                action("right"),
            ])),
            Err(WorldTestExplorationError::EventAfterTestTransition { index: 2 })
        );
    }

    #[test]
    fn whitespace_only_or_untrimmed_action_ids_are_rejected() {
        for action_id in ["", "  ", " left"] {
            assert_eq!(
                evaluate_worldtest_exploration(&trace(vec![
                    action(action_id),
                    ExplorationEvent::BeginTest,
                ])),
                Err(WorldTestExplorationError::EmptyActionId)
            );
        }
    }

    #[test]
    fn oversized_action_ids_are_rejected() {
        let long_id = "x".repeat(MAX_ACTION_ID_BYTES + 1);
        assert_eq!(
            evaluate_worldtest_exploration(&trace(vec![
                action(&long_id),
                ExplorationEvent::BeginTest,
            ])),
            Err(WorldTestExplorationError::ActionIdTooLong {
                bytes: MAX_ACTION_ID_BYTES + 1,
                maximum: MAX_ACTION_ID_BYTES,
            })
        );
    }

    #[test]
    fn exact_limit_is_accepted_and_one_over_is_rejected() {
        let mut events = vec![action("noop"); WORLDTEST_SEGMENT_MAX_ACTIONS];
        events.push(ExplorationEvent::BeginTest);
        assert!(evaluate_worldtest_exploration(&trace(events)).is_ok());

        let mut oversized = vec![action("noop"); WORLDTEST_SEGMENT_MAX_ACTIONS + 1];
        oversized.push(ExplorationEvent::BeginTest);
        assert!(matches!(
            evaluate_worldtest_exploration(&trace(oversized)),
            Err(WorldTestExplorationError::TooManyActionsInSegment { .. })
        ));
    }

    #[test]
    fn trace_event_bound_is_enforced() {
        let mut events = vec![action("noop"); WORLDTEST_TRACE_MAX_EVENTS];
        events.push(ExplorationEvent::BeginTest);
        assert!(matches!(
            evaluate_worldtest_exploration(&trace(events)),
            Err(WorldTestExplorationError::TooManyEvents { .. })
        ));
    }
}
