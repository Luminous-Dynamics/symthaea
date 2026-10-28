//! Time-resolved features from XY-to-Ising trajectory histories for MIS search.
//!
//! This module consumes sampled in-phase amplitudes x_i(t), not raw hardware.
//! It records spin-flip activity, late-run flips, stabilization, and terminal
//! cross-run disagreement. These are candidate ordering features only: they are
//! not calibrated probabilities of optimal-set membership and never certify
//! reductions, fix variables, prune exact-search branches, or prove an optimum.
//!
//! Samples within the declared zero tolerance are ambiguous for that vertex and
//! are excluded from adjacent-transition calculations. A flip is counted only
//! when both endpoints of an adjacent sampled interval project to valid opposite
//! signs. Failed and timed-out histories remain in the attempt-level accounting.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// One sampled state with time normalized to the planned run.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryFrame {
    /// Normalized time in [0, 1] from the planned start.
    pub normalized_time: f64,
    /// In-phase quadrature x_i in canonical vertex order.
    pub in_phase_x: Vec<f64>,
}

/// Terminal outcome for one attempted trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryOutcome {
    /// Solver reached its declared end condition and returned a terminal frame.
    Completed,
    /// Projection could not produce a complete spin state at some point.
    AmbiguousProjection,
    /// Dynamics encountered a numerical failure.
    NumericalFailure,
    /// Execution ended before the declared run budget.
    TimedOut,
    /// Mapping or decoding failed.
    MappingFailure,
}

/// Full sampled trajectory history. Failed histories may contain a partial prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryHistory {
    /// Stable, non-empty identifier unique within the aggregation input.
    pub attempt_id: String,
    /// Explicit terminal outcome.
    pub outcome: HistoryOutcome,
    /// Recorded frames, possibly a partial prefix for non-completed outcomes.
    pub frames: Vec<TrajectoryFrame>,
}

/// Candidate time-series ordering policy for a downstream exact-search solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryOrdering {
    /// Ignore trajectory features; deterministic degree-descending order.
    DegreeOnly,
    /// Prioritize vertices whose observed flips concentrate late in the planned run.
    LateActivityFirst,
    /// Prioritize vertices whose last observed spin flip occurred later.
    RecentFlipFirst,
    /// Prioritize vertices with more terminal disagreement across completed runs.
    TerminalDisagreementFirst,
}

/// Per-vertex time-resolved features aggregated across trajectory attempts.
#[derive(Debug, Clone, PartialEq)]
pub struct VertexHistoryFeatures {
    /// Canonical zero-based vertex index.
    pub vertex: usize,
    /// Number of sampled x-values outside the zero tolerance.
    pub projected_frame_count: usize,
    /// Number of sampled x-values within the zero tolerance.
    pub ambiguous_frame_count: usize,
    /// Number of adjacent frame pairs with both endpoint signs observed.
    pub observed_transition_count: usize,
    /// Number of observed adjacent pairs whose projected spins differed.
    pub flip_count: usize,
    /// Number of histories with at least one observed transition.
    pub traces_with_observed_transitions: usize,
    /// Number of histories with at least one observed spin flip.
    pub traces_with_flips: usize,
    /// Mean per-history flip fraction among histories with observed transitions.
    pub mean_flip_rate: Option<f64>,
    /// Mean fraction of observed transitions that flip at normalized time >= 0.75.
    /// Uses the planned-run clock; it does not renormalize timed-out traces.
    pub mean_late_flip_rate: Option<f64>,
    /// Mean normalized time of the last observed flip among histories that flipped.
    /// Partial histories contribute and are right-censored observations.
    pub mean_last_flip_time: Option<f64>,
    /// Mean stable-suffix fraction on completed histories only. No flips means 1.
    pub mean_completed_stable_suffix_fraction: Option<f64>,
    /// Count of completed runs with a valid terminal projection for this vertex.
    pub terminal_sample_count: usize,
    /// Selected (+1) terminal count among valid terminal projections.
    pub terminal_selected_count: usize,
    /// Empirical terminal selection frequency; not an optimum-membership probability.
    pub terminal_selection_frequency: Option<f64>,
    /// 1 - |2 f - 1| for terminal frequency f; descriptive disagreement only.
    pub terminal_disagreement: Option<f64>,
}

/// Aggregated history features, including failure and missingness counts.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryHistorySummary {
    /// Number of attempted trajectories, including failures and timeouts.
    pub attempted_count: usize,
    /// Number of completed trajectories.
    pub completed_count: usize,
    /// Number of ambiguous-projection outcomes.
    pub ambiguous_projection_count: usize,
    /// Number of numerical-failure outcomes.
    pub numerical_failure_count: usize,
    /// Number of timed-out outcomes.
    pub timeout_count: usize,
    /// Number of mapping/decoding-failure outcomes.
    pub mapping_failure_count: usize,
    /// Number of attempts containing at least two frames.
    pub histories_with_transitions: usize,
    /// Histories with at least two frames divided by all attempts.
    pub history_yield: f64,
    /// Fixed zero tolerance used for all projections in this summary.
    pub zero_tolerance: f64,
    /// Per-vertex features in canonical vertex order.
    pub vertices: Vec<VertexHistoryFeatures>,
}

/// Input or validation error in history aggregation.
#[derive(Debug, Clone, PartialEq)]
pub enum HistoryFeatureError {
    /// At least one attempted history is required.
    NoAttempts,
    /// Attempt identifier is empty or whitespace-only.
    EmptyAttemptId,
    /// Attempt identifier occurs more than once.
    DuplicateAttemptId(String),
    /// Zero tolerance is negative or non-finite.
    InvalidZeroTolerance(f64),
    /// A completed trajectory must contain at least one frame.
    CompletedWithoutFrames(String),
    /// A frame time is outside [0, 1] or non-finite.
    InvalidTime { attempt_id: String, frame_index: usize, value: f64 },
    /// Frame times are not strictly increasing.
    NonIncreasingTime { attempt_id: String, frame_index: usize },
    /// A frame has the wrong number of x values.
    VertexCountMismatch {
        attempt_id: String,
        frame_index: usize,
        expected: usize,
        actual: usize,
    },
    /// A frame contains a non-finite amplitude.
    NonFiniteAmplitude { attempt_id: String, frame_index: usize, vertex: usize },
    /// Degree vector has the wrong number of entries.
    DegreeCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for HistoryFeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAttempts => write!(f, "at least one trajectory history is required"),
            Self::EmptyAttemptId => write!(f, "trajectory history ID must not be empty"),
            Self::DuplicateAttemptId(id) => write!(f, "duplicate trajectory history ID: {id}"),
            Self::InvalidZeroTolerance(value) => write!(f, "invalid zero tolerance: {value}"),
            Self::CompletedWithoutFrames(id) => write!(f, "completed history {id} has no frames"),
            Self::InvalidTime { attempt_id, frame_index, value } => write!(
                f, "history {attempt_id} frame {frame_index} has invalid time {value}"
            ),
            Self::NonIncreasingTime { attempt_id, frame_index } => write!(
                f, "history {attempt_id} frame {frame_index} does not increase in time"
            ),
            Self::VertexCountMismatch { attempt_id, frame_index, expected, actual } => write!(
                f, "history {attempt_id} frame {frame_index} has {actual} vertices; expected {expected}"
            ),
            Self::NonFiniteAmplitude { attempt_id, frame_index, vertex } => write!(
                f, "history {attempt_id} frame {frame_index} has non-finite x at vertex {vertex}"
            ),
            Self::DegreeCountMismatch { expected, actual } => write!(
                f, "degree vector has {actual} entries; expected {expected}"
            ),
        }
    }
}

impl Error for HistoryFeatureError {}

impl TrajectoryHistorySummary {
    /// Aggregate histories using one fixed absolute tolerance for spin projection.
    ///
    /// Every frame must have finite amplitudes and strictly increasing normalized
    /// times. A non-completed history may be empty or contain a valid partial
    /// prefix; a completed history must have at least one frame.
    pub fn from_histories(
        vertex_count: usize,
        histories: &[TrajectoryHistory],
        zero_tolerance: f64,
    ) -> Result<Self, HistoryFeatureError> {
        if histories.is_empty() {
            return Err(HistoryFeatureError::NoAttempts);
        }
        if !zero_tolerance.is_finite() || zero_tolerance < 0.0 {
            return Err(HistoryFeatureError::InvalidZeroTolerance(zero_tolerance));
        }

        let mut ids = HashSet::with_capacity(histories.len());
        let (mut completed_count, mut ambiguous_projection_count) = (0, 0);
        let (mut numerical_failure_count, mut timeout_count, mut mapping_failure_count) = (0, 0, 0);
        let mut histories_with_transitions = 0_usize;

        let mut projected_frame_count = vec![0_usize; vertex_count];
        let mut ambiguous_frame_count = vec![0_usize; vertex_count];
        let mut observed_transition_count = vec![0_usize; vertex_count];
        let mut flip_count = vec![0_usize; vertex_count];
        let mut traces_with_observed_transitions = vec![0_usize; vertex_count];
        let mut traces_with_flips = vec![0_usize; vertex_count];
        let mut flip_rate_sum = vec![0.0_f64; vertex_count];
        let mut late_flip_rate_sum = vec![0.0_f64; vertex_count];
        let mut last_flip_time_sum = vec![0.0_f64; vertex_count];
        let mut stable_suffix_sum = vec![0.0_f64; vertex_count];
        let mut stable_suffix_count = vec![0_usize; vertex_count];
        let mut terminal_sample_count = vec![0_usize; vertex_count];
        let mut terminal_selected_count = vec![0_usize; vertex_count];

        for history in histories {
            if history.attempt_id.trim().is_empty() {
                return Err(HistoryFeatureError::EmptyAttemptId);
            }
            if !ids.insert(history.attempt_id.as_str()) {
                return Err(HistoryFeatureError::DuplicateAttemptId(history.attempt_id.clone()));
            }
            if history.outcome == HistoryOutcome::Completed && history.frames.is_empty() {
                return Err(HistoryFeatureError::CompletedWithoutFrames(history.attempt_id.clone()));
            }

            match history.outcome {
                HistoryOutcome::Completed => completed_count += 1,
                HistoryOutcome::AmbiguousProjection => ambiguous_projection_count += 1,
                HistoryOutcome::NumericalFailure => numerical_failure_count += 1,
                HistoryOutcome::TimedOut => timeout_count += 1,
                HistoryOutcome::MappingFailure => mapping_failure_count += 1,
            }

            let mut previous_time: Option<f64> = None;
            for (frame_index, frame) in history.frames.iter().enumerate() {
                if !frame.normalized_time.is_finite()
                    || !(0.0..=1.0).contains(&frame.normalized_time)
                {
                    return Err(HistoryFeatureError::InvalidTime {
                        attempt_id: history.attempt_id.clone(),
                        frame_index,
                        value: frame.normalized_time,
                    });
                }
                if previous_time.is_some_and(|t| frame.normalized_time <= t) {
                    return Err(HistoryFeatureError::NonIncreasingTime {
                        attempt_id: history.attempt_id.clone(),
                        frame_index,
                    });
                }
                previous_time = Some(frame.normalized_time);

                if frame.in_phase_x.len() != vertex_count {
                    return Err(HistoryFeatureError::VertexCountMismatch {
                        attempt_id: history.attempt_id.clone(),
                        frame_index,
                        expected: vertex_count,
                        actual: frame.in_phase_x.len(),
                    });
                }
                for (vertex, &x) in frame.in_phase_x.iter().enumerate() {
                    if !x.is_finite() {
                        return Err(HistoryFeatureError::NonFiniteAmplitude {
                            attempt_id: history.attempt_id.clone(),
                            frame_index,
                            vertex,
                        });
                    }
                    if x.abs() <= zero_tolerance {
                        ambiguous_frame_count[vertex] += 1;
                    } else {
                        projected_frame_count[vertex] += 1;
                    }
                }
            }

            if history.frames.len() >= 2 {
                histories_with_transitions += 1;
            }

            for vertex in 0..vertex_count {
                let mut seen_pairs = 0_usize;
                let mut flips = 0_usize;
                let mut late_flips = 0_usize;
                let mut last_flip: Option<f64> = None;

                for pair in history.frames.windows(2) {
                    let left_x = pair[0].in_phase_x[vertex];
                    let right_x = pair[1].in_phase_x[vertex];
                    if left_x.abs() <= zero_tolerance || right_x.abs() <= zero_tolerance {
                        continue;
                    }
                    seen_pairs += 1;
                    if left_x.is_sign_positive() != right_x.is_sign_positive() {
                        flips += 1;
                        last_flip = Some(pair[1].normalized_time);
                        if pair[1].normalized_time >= 0.75 {
                            late_flips += 1;
                        }
                    }
                }

                observed_transition_count[vertex] += seen_pairs;
                flip_count[vertex] += flips;
                if seen_pairs > 0 {
                    traces_with_observed_transitions[vertex] += 1;
                    flip_rate_sum[vertex] += flips as f64 / seen_pairs as f64;
                    late_flip_rate_sum[vertex] += late_flips as f64 / seen_pairs as f64;
                }
                if let Some(last_flip_time) = last_flip {
                    traces_with_flips[vertex] += 1;
                    last_flip_time_sum[vertex] += last_flip_time;
                }

                if history.outcome == HistoryOutcome::Completed && seen_pairs > 0 {
                    // Measure only a contiguous, observed same-sign suffix ending
                    // at the last frame. If the terminal observation is ambiguous,
                    // or an ambiguous sample interrupts the suffix, leave this
                    // feature unavailable instead of treating the gap as stability.
                    let last_index = history.frames.len() - 1;
                    let terminal_x = history.frames[last_index].in_phase_x[vertex];
                    if terminal_x.abs() > zero_tolerance {
                        let mut suffix_start = last_index;
                        while suffix_start > 0 {
                            let previous_x =
                                history.frames[suffix_start - 1].in_phase_x[vertex];
                            let current_x = history.frames[suffix_start].in_phase_x[vertex];
                            if previous_x.abs() <= zero_tolerance
                                || current_x.abs() <= zero_tolerance
                                || previous_x.is_sign_positive() != current_x.is_sign_positive()
                            {
                                break;
                            }
                            suffix_start -= 1;
                        }

                        let first_time = history.frames.first().expect("history has frames").normalized_time;
                        let end_time = history.frames[last_index].normalized_time;
                        let observed_duration = end_time - first_time;
                        if suffix_start < last_index && observed_duration > 0.0 {
                            let stable_suffix = ((end_time
                                - history.frames[suffix_start].normalized_time)
                                / observed_duration)
                                .clamp(0.0, 1.0);
                            stable_suffix_sum[vertex] += stable_suffix;
                            stable_suffix_count[vertex] += 1;
                        }
                    }
                }
            }

            if history.outcome == HistoryOutcome::Completed {
                let terminal = history.frames.last().expect("completed history checked above");
                for (vertex, &x) in terminal.in_phase_x.iter().enumerate() {
                    if x.abs() <= zero_tolerance {
                        continue;
                    }
                    terminal_sample_count[vertex] += 1;
                    if x > 0.0 {
                        terminal_selected_count[vertex] += 1;
                    }
                }
            }
        }

        let attempted_count = histories.len();
        let vertices = (0..vertex_count)
            .map(|vertex| {
                let terminal_n = terminal_sample_count[vertex];
                let terminal_frequency = if terminal_n > 0 {
                    Some(terminal_selected_count[vertex] as f64 / terminal_n as f64)
                } else {
                    None
                };
                let terminal_disagreement =
                    terminal_frequency.map(|f| 1.0 - (2.0 * f - 1.0).abs());
                let transition_traces = traces_with_observed_transitions[vertex];
                let flip_traces = traces_with_flips[vertex];
                let stable_traces = stable_suffix_count[vertex];

                VertexHistoryFeatures {
                    vertex,
                    projected_frame_count: projected_frame_count[vertex],
                    ambiguous_frame_count: ambiguous_frame_count[vertex],
                    observed_transition_count: observed_transition_count[vertex],
                    flip_count: flip_count[vertex],
                    traces_with_observed_transitions: transition_traces,
                    traces_with_flips: flip_traces,
                    mean_flip_rate: if transition_traces > 0 {
                        Some(flip_rate_sum[vertex] / transition_traces as f64)
                    } else {
                        None
                    },
                    mean_late_flip_rate: if transition_traces > 0 {
                        Some(late_flip_rate_sum[vertex] / transition_traces as f64)
                    } else {
                        None
                    },
                    mean_last_flip_time: if flip_traces > 0 {
                        Some(last_flip_time_sum[vertex] / flip_traces as f64)
                    } else {
                        None
                    },
                    mean_completed_stable_suffix_fraction: if stable_traces > 0 {
                        Some(stable_suffix_sum[vertex] / stable_traces as f64)
                    } else {
                        None
                    },
                    terminal_sample_count: terminal_n,
                    terminal_selected_count: terminal_selected_count[vertex],
                    terminal_selection_frequency: terminal_frequency,
                    terminal_disagreement,
                }
            })
            .collect();

        Ok(Self {
            attempted_count,
            completed_count,
            ambiguous_projection_count,
            numerical_failure_count,
            timeout_count,
            mapping_failure_count,
            histories_with_transitions,
            history_yield: histories_with_transitions as f64 / attempted_count as f64,
            zero_tolerance,
            vertices,
        })
    }

    /// Produce a deterministic ordering suggestion. Every vertex remains in the
    /// permutation; this function cannot authorize variable fixing or branch pruning.
    pub fn ordered_vertices(
        &self,
        degrees: &[usize],
        ordering: HistoryOrdering,
    ) -> Result<Vec<usize>, HistoryFeatureError> {
        if degrees.len() != self.vertices.len() {
            return Err(HistoryFeatureError::DegreeCountMismatch {
                expected: self.vertices.len(),
                actual: degrees.len(),
            });
        }
        let score = |vertex: usize| -> f64 {
            match ordering {
                HistoryOrdering::DegreeOnly => 0.0,
                HistoryOrdering::LateActivityFirst =>
                    self.vertices[vertex].mean_late_flip_rate.unwrap_or(-1.0),
                HistoryOrdering::RecentFlipFirst =>
                    self.vertices[vertex].mean_last_flip_time.unwrap_or(-1.0),
                HistoryOrdering::TerminalDisagreementFirst =>
                    self.vertices[vertex].terminal_disagreement.unwrap_or(-1.0),
            }
        };

        let mut order: Vec<usize> = (0..self.vertices.len()).collect();
        order.sort_by(|left, right| {
            score(*right)
                .total_cmp(&score(*left))
                .then_with(|| degrees[*right].cmp(&degrees[*left]))
                .then_with(|| left.cmp(right))
        });
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(time: f64, x: &[f64]) -> TrajectoryFrame {
        TrajectoryFrame { normalized_time: time, in_phase_x: x.to_vec() }
    }

    fn history(id: &str, outcome: HistoryOutcome, frames: Vec<TrajectoryFrame>) -> TrajectoryHistory {
        TrajectoryHistory { attempt_id: id.to_owned(), outcome, frames }
    }

    #[test]
    fn summarizes_flip_activity_stability_and_terminal_disagreement() {
        let histories = vec![
            history("a", HistoryOutcome::Completed, vec![
                frame(0.0, &[-1.0, 1.0]), frame(0.25, &[1.0, 1.0]),
                frame(0.75, &[-1.0, -1.0]), frame(1.0, &[-1.0, 1.0]),
            ]),
            history("b", HistoryOutcome::Completed, vec![
                frame(0.0, &[1.0, -1.0]), frame(0.25, &[1.0, -1.0]),
                frame(0.75, &[1.0, -1.0]), frame(1.0, &[1.0, -1.0]),
            ]),
            history("c-timeout", HistoryOutcome::TimedOut, vec![
                frame(0.0, &[-1.0, 1.0]), frame(0.5, &[1.0, 1.0]),
            ]),
        ];
        let summary = TrajectoryHistorySummary::from_histories(2, &histories, 0.01).unwrap();
        assert_eq!(summary.attempted_count, 3);
        assert_eq!(summary.completed_count, 2);
        assert_eq!(summary.timeout_count, 1);
        assert_eq!(summary.histories_with_transitions, 3);
        assert_eq!(summary.history_yield, 1.0);

        let v0 = &summary.vertices[0];
        assert_eq!(v0.flip_count, 3);
        assert_eq!(v0.traces_with_flips, 2);
        assert_eq!(v0.terminal_sample_count, 2);
        assert_eq!(v0.terminal_selected_count, 1);
        assert_eq!(v0.terminal_selection_frequency, Some(0.5));
        assert_eq!(v0.terminal_disagreement, Some(1.0));
        assert_eq!(v0.mean_last_flip_time, Some(0.625));
        assert!(v0.mean_completed_stable_suffix_fraction.is_some());
    }

    #[test]
    fn temporal_features_distinguish_trajectories_with_identical_terminal_frequency() {
        let early = vec![
            history("early-a", HistoryOutcome::Completed, vec![
                frame(0.0, &[-1.0]),
                frame(0.25, &[1.0]),
                frame(0.75, &[1.0]),
                frame(1.0, &[1.0]),
            ]),
            history("early-b", HistoryOutcome::Completed, vec![
                frame(0.0, &[-1.0]),
                frame(0.25, &[1.0]),
                frame(0.75, &[1.0]),
                frame(1.0, &[1.0]),
            ]),
        ];
        let late = vec![
            history("late-a", HistoryOutcome::Completed, vec![
                frame(0.0, &[-1.0]),
                frame(0.25, &[-1.0]),
                frame(0.75, &[-1.0]),
                frame(1.0, &[1.0]),
            ]),
            history("late-b", HistoryOutcome::Completed, vec![
                frame(0.0, &[-1.0]),
                frame(0.25, &[-1.0]),
                frame(0.75, &[-1.0]),
                frame(1.0, &[1.0]),
            ]),
        ];
        let early_summary = TrajectoryHistorySummary::from_histories(1, &early, 0.0).unwrap();
        let late_summary = TrajectoryHistorySummary::from_histories(1, &late, 0.0).unwrap();
        let early_vertex = &early_summary.vertices[0];
        let late_vertex = &late_summary.vertices[0];

        // Final-spin priors cannot distinguish these cohorts: both select the
        // only vertex in every completed run. The time-resolved feature can.
        assert_eq!(early_vertex.terminal_selection_frequency, Some(1.0));
        assert_eq!(late_vertex.terminal_selection_frequency, Some(1.0));
        assert_eq!(early_vertex.mean_late_flip_rate, Some(0.0));
        assert_eq!(late_vertex.mean_late_flip_rate, Some(1.0 / 3.0));
        assert_eq!(early_vertex.mean_last_flip_time, Some(0.25));
        assert_eq!(late_vertex.mean_last_flip_time, Some(1.0));
    }

    #[test]
    fn ambiguous_samples_are_not_imputed_into_spurious_flips() {
        let histories = vec![history("ambiguous-middle", HistoryOutcome::Completed, vec![
            frame(0.0, &[-1.0]), frame(0.5, &[0.0]), frame(1.0, &[1.0]),
        ])];
        let summary = TrajectoryHistorySummary::from_histories(1, &histories, 1e-9).unwrap();
        let vertex = &summary.vertices[0];
        assert_eq!(vertex.ambiguous_frame_count, 1);
        assert_eq!(vertex.projected_frame_count, 2);
        assert_eq!(vertex.observed_transition_count, 0);
        assert_eq!(vertex.flip_count, 0);
        assert_eq!(vertex.mean_flip_rate, None);
        assert_eq!(vertex.terminal_selection_frequency, Some(1.0));
    }

    #[test]
    fn ambiguous_terminal_sample_does_not_extend_stability_window() {
        let histories = vec![history("terminal-ambiguous", HistoryOutcome::Completed, vec![
            frame(0.0, &[-1.0]),
            frame(0.5, &[1.0]),
            frame(0.75, &[1.0]),
            frame(1.0, &[0.0]),
        ])];
        let summary = TrajectoryHistorySummary::from_histories(1, &histories, 1e-9).unwrap();
        assert_eq!(
            summary.vertices[0].mean_completed_stable_suffix_fraction,
            None
        );
        assert_eq!(summary.vertices[0].terminal_sample_count, 0);
        assert_eq!(summary.vertices[0].terminal_selection_frequency, None);
    }

    #[test]
    fn preserves_partial_failures_and_excludes_them_from_terminal_frequency() {
        let histories = vec![
            history("complete", HistoryOutcome::Completed, vec![frame(0.0, &[-1.0]), frame(1.0, &[1.0])]),
            history("failed-partial", HistoryOutcome::NumericalFailure, vec![frame(0.0, &[1.0]), frame(0.4, &[-1.0])]),
            history("failed-empty", HistoryOutcome::TimedOut, vec![]),
            history("ambiguous", HistoryOutcome::AmbiguousProjection, vec![]),
            history("mapping", HistoryOutcome::MappingFailure, vec![]),
        ];
        let summary = TrajectoryHistorySummary::from_histories(1, &histories, 0.0).unwrap();
        assert_eq!(summary.attempted_count, 5);
        assert_eq!(summary.completed_count, 1);
        assert_eq!(summary.numerical_failure_count, 1);
        assert_eq!(summary.timeout_count, 1);
        assert_eq!(summary.ambiguous_projection_count, 1);
        assert_eq!(summary.mapping_failure_count, 1);
        assert_eq!(summary.history_yield, 0.4);
        assert_eq!(summary.vertices[0].flip_count, 2);
        assert_eq!(summary.vertices[0].terminal_sample_count, 1);
        assert_eq!(summary.vertices[0].terminal_selection_frequency, Some(1.0));
    }

    #[test]
    fn rejects_invalid_input_and_duplicate_ids() {
        assert!(matches!(TrajectoryHistorySummary::from_histories(1, &[], 0.0), Err(HistoryFeatureError::NoAttempts)));
        assert!(matches!(
            TrajectoryHistorySummary::from_histories(1, &[history("x", HistoryOutcome::Completed, vec![])], 0.0),
            Err(HistoryFeatureError::CompletedWithoutFrames(_))
        ));
        assert!(matches!(
            TrajectoryHistorySummary::from_histories(1, &[history("x", HistoryOutcome::Completed, vec![frame(0.0, &[f64::NAN])])], 0.0),
            Err(HistoryFeatureError::NonFiniteAmplitude { .. })
        ));
        assert!(matches!(
            TrajectoryHistorySummary::from_histories(1, &[history("x", HistoryOutcome::Completed, vec![frame(0.5, &[1.0]), frame(0.5, &[1.0])])], 0.0),
            Err(HistoryFeatureError::NonIncreasingTime { .. })
        ));
        let duplicate = vec![
            history("same", HistoryOutcome::Completed, vec![frame(0.0, &[1.0])]),
            history("same", HistoryOutcome::Completed, vec![frame(0.0, &[-1.0])]),
        ];
        assert!(matches!(
            TrajectoryHistorySummary::from_histories(1, &duplicate, 0.0),
            Err(HistoryFeatureError::DuplicateAttemptId(_))
        ));
    }

    #[test]
    fn ordering_is_deterministic_and_never_drops_vertices() {
        let histories = vec![
            history("a", HistoryOutcome::Completed, vec![
                frame(0.0, &[-1.0, -1.0, 1.0]), frame(0.25, &[1.0, -1.0, 1.0]),
                frame(0.75, &[-1.0, -1.0, -1.0]), frame(1.0, &[-1.0, 1.0, -1.0]),
            ]),
            history("b", HistoryOutcome::Completed, vec![
                frame(0.0, &[1.0, -1.0, 1.0]), frame(0.25, &[1.0, -1.0, -1.0]),
                frame(0.75, &[1.0, 1.0, 1.0]), frame(1.0, &[1.0, 1.0, 1.0]),
            ]),
        ];
        let summary = TrajectoryHistorySummary::from_histories(3, &histories, 0.0).unwrap();
        let degrees = [2, 5, 5];
        for policy in [HistoryOrdering::DegreeOnly, HistoryOrdering::LateActivityFirst,
            HistoryOrdering::RecentFlipFirst, HistoryOrdering::TerminalDisagreementFirst] {
            let order = summary.ordered_vertices(&degrees, policy).unwrap();
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![0, 1, 2]);
            assert_eq!(order, summary.ordered_vertices(&degrees, policy).unwrap());
        }
        assert_eq!(summary.ordered_vertices(&degrees, HistoryOrdering::DegreeOnly).unwrap(), vec![1, 2, 0]);
        assert!(matches!(
            summary.ordered_vertices(&[1], HistoryOrdering::LateActivityFirst),
            Err(HistoryFeatureError::DegreeCountMismatch { .. })
        ));
    }

    #[test]
    fn rejects_invalid_tolerance_and_time_range() {
        assert!(matches!(
            TrajectoryHistorySummary::from_histories(1, &[history("x", HistoryOutcome::TimedOut, vec![])], f64::NAN),
            Err(HistoryFeatureError::InvalidZeroTolerance(_))
        ));
        assert!(matches!(
            TrajectoryHistorySummary::from_histories(1, &[history("x", HistoryOutcome::TimedOut, vec![frame(1.1, &[1.0])])], 0.0),
            Err(HistoryFeatureError::InvalidTime { .. })
        ));
    }
}
