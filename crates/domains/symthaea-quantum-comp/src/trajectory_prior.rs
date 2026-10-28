//! Conservative summaries of photonic-inspired trajectory samples for MIS search.
//!
//! This module summarizes already-projected Ising spins. It does not simulate
//! optics, infer membership in an optimal solution, fix variables, or certify
//! reductions. Its outputs may only influence ordering in a downstream algorithm
//! whose correctness does not depend on the prior.
//!
//! An empirical inclusion frequency describes the observed trajectory generator
//! conditional on a valid spin projection. It is not, by itself, a calibrated
//! probability that a vertex belongs to an optimal independent set. Wilson
//! intervals are exposed only when the caller explicitly declares independent
//! restarts; even then, they describe conditional valid-projection frequency,
//! not optimal-set membership. Projection yield must be reported separately
//! because ambiguous or failed projections may not be missing at random.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// One complete attempted trajectory and its spin-projection outcome.
///
/// Every attempt—including ambiguous projections and failures—must be retained
/// so that projection yield is not hidden by filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryAttempt {
    /// Stable, non-empty identifier unique within one summary.
    pub attempt_id: String,
    /// Caller-provided seed or run identifier. Uniqueness does not prove
    /// statistical independence; the independent-restart declaration is separate.
    pub seed: u64,
    /// Explicit outcome of the trajectory and spin projection.
    pub outcome: TrajectoryOutcome,
}

/// Outcome recorded for an attempted trajectory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryOutcome {
    /// Successfully projected spins, with +1 meaning selected and -1 unselected.
    ProjectedSpins(Vec<i8>),
    /// The in-phase amplitudes did not produce a valid discrete spin assignment.
    AmbiguousProjection,
    /// Numerical state or feedback became invalid/non-finite.
    NumericalFailure,
    /// The trajectory did not finish within its declared execution budget.
    TimedOut,
    /// Projection or decoding failed independently of trajectory stability.
    MappingFailure,
}

/// A Wilson interval for a per-vertex empirical selection frequency.
///
/// These bounds apply only to the Bernoulli sample formed by valid projections
/// when the caller declares independent restarts. They do not estimate the
/// probability that a vertex belongs to an optimal MIS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonInterval {
    /// Number of valid projected samples used.
    pub sample_count: usize,
    /// Number of those samples where the vertex was selected.
    pub selected_count: usize,
    /// Lower endpoint of the two-sided 95% Wilson interval.
    pub lower: f64,
    /// Upper endpoint of the two-sided 95% Wilson interval.
    pub upper: f64,
}

/// Per-vertex descriptive statistics from valid spin projections.
#[derive(Debug, Clone, PartialEq)]
pub struct VertexTrajectorySummary {
    /// Zero-based vertex index in the canonical input graph.
    pub vertex: usize,
    /// Number of valid projected samples where spin +1 selected this vertex.
    pub selected_count: usize,
    /// Number of valid projected samples used for this vertex.
    pub valid_sample_count: usize,
    /// Empirical selection frequency among valid projections; absent when none
    /// of the attempted trajectories yielded a valid spin vector.
    pub empirical_inclusion_frequency: Option<f64>,
    /// Absolute magnetization score |2 f - 1|, in [0, 1], when f is observed.
    /// This is a prioritization feature, not an optimal-membership probability.
    pub bias_magnitude: Option<f64>,
    /// Complementary ambiguity score 1 - |2 f - 1|, in [0, 1].
    /// This is a prioritization feature, not an optimal-membership probability.
    pub uncertainty_score: Option<f64>,
    /// Wilson interval for conditional empirical selection frequency. Present
    /// only when independent restarts were explicitly declared and valid samples
    /// exist; not a confidence interval for optimal-set membership.
    pub wilson_95: Option<WilsonInterval>,
}

/// Deterministic ordering policy for downstream exact search or proof-query work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorOrdering {
    /// Ignore trajectory statistics; sort by descending graph degree, then index.
    DegreeOnly,
    /// Prefer vertices with more extreme observed selection frequencies.
    /// This heuristic did not show a robust benefit in the existing synthetic pilot.
    BiasFirst,
    /// Prefer vertices with less extreme observed selection frequencies.
    /// This is a challenger, not a presumed improvement.
    UncertaintyFirst,
}

/// Validated aggregate of repeated trajectory attempts.
///
/// No method in this type can make a graph reduction or certify a vertex. Its only
/// policy operation is a deterministic ordering suggestion.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryPriorSummary {
    /// Number of attempted trajectories, including failures and ambiguous results.
    pub attempted_count: usize,
    /// Number of valid projected spin samples.
    pub valid_sample_count: usize,
    /// Number of ambiguous projections.
    pub ambiguous_projection_count: usize,
    /// Number of numerical failures.
    pub numerical_failure_count: usize,
    /// Number of timed-out trajectories.
    pub timeout_count: usize,
    /// Number of mapping/decoding failures.
    pub mapping_failure_count: usize,
    /// Valid projected samples divided by all attempted trajectories.
    pub projection_yield: f64,
    /// Whether the caller declared restarts independent. This declaration is
    /// recorded, not independently established by this module.
    pub independent_restarts_declared: bool,
    /// Per-vertex descriptive statistics in canonical vertex order.
    pub vertices: Vec<VertexTrajectorySummary>,
}

/// Input or validation error for trajectory prior aggregation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryPriorError {
    /// At least one attempted trajectory is required.
    NoAttempts,
    /// An attempt identifier is empty or whitespace-only.
    EmptyAttemptId,
    /// An attempt identifier appears more than once.
    DuplicateAttemptId(String),
    /// A projected spin vector has the wrong number of vertices.
    SpinCountMismatch {
        /// Attempt identifier whose output has the wrong length.
        attempt_id: String,
        /// Expected number of vertices.
        expected: usize,
        /// Actual number of spins.
        actual: usize,
    },
    /// A projected spin is not -1 or +1.
    InvalidSpin {
        /// Attempt identifier containing the invalid spin.
        attempt_id: String,
        /// Vertex index where the invalid spin occurred.
        vertex: usize,
        /// Invalid spin value.
        value: i8,
    },
    /// Degree vector has the wrong length for the summary.
    DegreeCountMismatch {
        /// Expected number of degrees.
        expected: usize,
        /// Actual number of degrees.
        actual: usize,
    },
}

impl fmt::Display for TrajectoryPriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAttempts => write!(f, "at least one trajectory attempt is required"),
            Self::EmptyAttemptId => write!(f, "trajectory attempt ID must not be empty"),
            Self::DuplicateAttemptId(id) => write!(f, "duplicate trajectory attempt ID: {id}"),
            Self::SpinCountMismatch { attempt_id, expected, actual } => write!(
                f,
                "attempt {attempt_id} has {actual} spins; expected {expected}"
            ),
            Self::InvalidSpin { attempt_id, vertex, value } => write!(
                f,
                "attempt {attempt_id} has invalid spin {value} at vertex {vertex}; expected -1 or +1"
            ),
            Self::DegreeCountMismatch { expected, actual } => write!(
                f,
                "degree vector has {actual} entries; expected {expected}"
            ),
        }
    }
}

impl Error for TrajectoryPriorError {}

impl TrajectoryPriorSummary {
    /// Aggregate every attempt, keeping invalid and ambiguous attempts in the
    /// denominator for projection-yield accounting.
    ///
    /// vertex_count is the canonical graph's vertex count. independent_restarts
    /// must be set only when the experiment design supports that assumption. It
    /// does not imply that trajectory samples are IID at the physical level; any
    /// temporal or device correlations still need to be characterized separately.
    pub fn from_attempts(
        vertex_count: usize,
        attempts: &[TrajectoryAttempt],
        independent_restarts: bool,
    ) -> Result<Self, TrajectoryPriorError> {
        if attempts.is_empty() {
            return Err(TrajectoryPriorError::NoAttempts);
        }

        let mut ids = HashSet::with_capacity(attempts.len());
        let mut selected_counts = vec![0_usize; vertex_count];
        let mut valid_sample_count = 0_usize;
        let mut ambiguous_projection_count = 0_usize;
        let mut numerical_failure_count = 0_usize;
        let mut timeout_count = 0_usize;
        let mut mapping_failure_count = 0_usize;

        for attempt in attempts {
            if attempt.attempt_id.trim().is_empty() {
                return Err(TrajectoryPriorError::EmptyAttemptId);
            }
            if !ids.insert(attempt.attempt_id.as_str()) {
                return Err(TrajectoryPriorError::DuplicateAttemptId(
                    attempt.attempt_id.clone(),
                ));
            }

            match &attempt.outcome {
                TrajectoryOutcome::ProjectedSpins(spins) => {
                    if spins.len() != vertex_count {
                        return Err(TrajectoryPriorError::SpinCountMismatch {
                            attempt_id: attempt.attempt_id.clone(),
                            expected: vertex_count,
                            actual: spins.len(),
                        });
                    }
                    for (vertex, &spin) in spins.iter().enumerate() {
                        if spin != -1 && spin != 1 {
                            return Err(TrajectoryPriorError::InvalidSpin {
                                attempt_id: attempt.attempt_id.clone(),
                                vertex,
                                value: spin,
                            });
                        }
                        if spin == 1 {
                            selected_counts[vertex] += 1;
                        }
                    }
                    valid_sample_count += 1;
                }
                TrajectoryOutcome::AmbiguousProjection => ambiguous_projection_count += 1,
                TrajectoryOutcome::NumericalFailure => numerical_failure_count += 1,
                TrajectoryOutcome::TimedOut => timeout_count += 1,
                TrajectoryOutcome::MappingFailure => mapping_failure_count += 1,
            }
        }

        let attempted_count = attempts.len();
        let projection_yield = valid_sample_count as f64 / attempted_count as f64;
        let vertices = selected_counts
            .into_iter()
            .enumerate()
            .map(|(vertex, selected_count)| {
                let empirical_inclusion_frequency = if valid_sample_count > 0 {
                    Some(selected_count as f64 / valid_sample_count as f64)
                } else {
                    None
                };
                let bias_magnitude = empirical_inclusion_frequency
                    .map(|frequency| (2.0 * frequency - 1.0).abs());
                let uncertainty_score = bias_magnitude.map(|bias| 1.0 - bias);
                let wilson_95 = if independent_restarts && valid_sample_count > 0 {
                    Some(wilson_interval_95(selected_count, valid_sample_count))
                } else {
                    None
                };
                VertexTrajectorySummary {
                    vertex,
                    selected_count,
                    valid_sample_count,
                    empirical_inclusion_frequency,
                    bias_magnitude,
                    uncertainty_score,
                    wilson_95,
                }
            })
            .collect();

        Ok(Self {
            attempted_count,
            valid_sample_count,
            ambiguous_projection_count,
            numerical_failure_count,
            timeout_count,
            mapping_failure_count,
            projection_yield,
            independent_restarts_declared: independent_restarts,
            vertices,
        })
    }

    /// Return a deterministic variable order suggestion.
    ///
    /// Every vertex remains in the returned permutation. This method only sorts;
    /// callers must still explore all branches required by their exact algorithm
    /// and cannot use the scores to fix variables, prune branches, or claim
    /// optimality. Ties use descending degree, then ascending vertex index.
    pub fn ordered_vertices(
        &self,
        degrees: &[usize],
        ordering: PriorOrdering,
    ) -> Result<Vec<usize>, TrajectoryPriorError> {
        if degrees.len() != self.vertices.len() {
            return Err(TrajectoryPriorError::DegreeCountMismatch {
                expected: self.vertices.len(),
                actual: degrees.len(),
            });
        }

        let mut order: Vec<usize> = (0..self.vertices.len()).collect();
        order.sort_by(|&left, &right| {
            let prior_order = match ordering {
                PriorOrdering::DegreeOnly => std::cmp::Ordering::Equal,
                PriorOrdering::BiasFirst => {
                    let l = self.vertices[left].bias_magnitude.unwrap_or(0.0);
                    let r = self.vertices[right].bias_magnitude.unwrap_or(0.0);
                    r.total_cmp(&l)
                }
                PriorOrdering::UncertaintyFirst => {
                    let l = self.vertices[left].uncertainty_score.unwrap_or(0.0);
                    let r = self.vertices[right].uncertainty_score.unwrap_or(0.0);
                    r.total_cmp(&l)
                }
            };
            prior_order
                .then_with(|| degrees[right].cmp(&degrees[left]))
                .then_with(|| left.cmp(&right))
        });
        Ok(order)
    }
}

fn wilson_interval_95(successes: usize, trials: usize) -> WilsonInterval {
    const Z: f64 = 1.959_963_984_540_054;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z * Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin = Z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    WilsonInterval {
        sample_count: trials,
        selected_count: successes,
        lower: (center - margin).max(0.0),
        upper: (center + margin).min(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attempt(id: &str, seed: u64, outcome: TrajectoryOutcome) -> TrajectoryAttempt {
        TrajectoryAttempt {
            attempt_id: id.to_owned(),
            seed,
            outcome,
        }
    }

    #[test]
    fn includes_failed_and_ambiguous_attempts_in_projection_yield() {
        let attempts = vec![
            attempt("a", 1, TrajectoryOutcome::ProjectedSpins(vec![1, -1])),
            attempt("b", 2, TrajectoryOutcome::ProjectedSpins(vec![1, 1])),
            attempt("c", 3, TrajectoryOutcome::AmbiguousProjection),
            attempt("d", 4, TrajectoryOutcome::NumericalFailure),
            attempt("e", 5, TrajectoryOutcome::TimedOut),
            attempt("f", 6, TrajectoryOutcome::MappingFailure),
        ];
        let summary = TrajectoryPriorSummary::from_attempts(2, &attempts, false).unwrap();
        assert_eq!(summary.attempted_count, 6);
        assert_eq!(summary.valid_sample_count, 2);
        assert_eq!(summary.ambiguous_projection_count, 1);
        assert_eq!(summary.numerical_failure_count, 1);
        assert_eq!(summary.timeout_count, 1);
        assert_eq!(summary.mapping_failure_count, 1);
        assert!((summary.projection_yield - (2.0 / 6.0)).abs() < 1e-12);
        assert_eq!(summary.vertices[0].selected_count, 2);
        assert_eq!(summary.vertices[0].empirical_inclusion_frequency, Some(1.0));
        assert_eq!(summary.vertices[1].selected_count, 1);
        assert_eq!(summary.vertices[1].empirical_inclusion_frequency, Some(0.5));
        assert!(summary.vertices.iter().all(|v| v.wilson_95.is_none()));
    }

    #[test]
    fn wilson_interval_is_only_emitted_when_independence_is_declared() {
        let attempts: Vec<_> = (0..100)
            .map(|i| {
                let spin = if i < 50 { 1 } else { -1 };
                attempt(&format!("trial-{i}"), i, TrajectoryOutcome::ProjectedSpins(vec![spin]))
            })
            .collect();
        let dependent = TrajectoryPriorSummary::from_attempts(1, &attempts, false).unwrap();
        assert!(dependent.vertices[0].wilson_95.is_none());

        let declared = TrajectoryPriorSummary::from_attempts(1, &attempts, true).unwrap();
        let interval = declared.vertices[0].wilson_95.unwrap();
        assert_eq!(interval.sample_count, 100);
        assert_eq!(interval.selected_count, 50);
        assert!((interval.lower - 0.4038).abs() < 0.001);
        assert!((interval.upper - 0.5962).abs() < 0.001);
    }

    #[test]
    fn zero_valid_projections_do_not_turn_into_zero_frequency() {
        let attempts = vec![
            attempt("a", 1, TrajectoryOutcome::AmbiguousProjection),
            attempt("b", 2, TrajectoryOutcome::TimedOut),
        ];
        let summary = TrajectoryPriorSummary::from_attempts(3, &attempts, true).unwrap();
        assert_eq!(summary.projection_yield, 0.0);
        assert!(summary.vertices.iter().all(|v| v.empirical_inclusion_frequency.is_none()));
        assert!(summary.vertices.iter().all(|v| v.wilson_95.is_none()));
        assert_eq!(
            summary.ordered_vertices(&[1, 3, 2], PriorOrdering::BiasFirst).unwrap(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn rejects_duplicate_ids_bad_dimensions_and_invalid_spins() {
        assert_eq!(
            TrajectoryPriorSummary::from_attempts(1, &[], false).unwrap_err(),
            TrajectoryPriorError::NoAttempts
        );
        let duplicate = vec![
            attempt("same", 1, TrajectoryOutcome::ProjectedSpins(vec![1])),
            attempt("same", 2, TrajectoryOutcome::ProjectedSpins(vec![-1])),
        ];
        assert!(matches!(
            TrajectoryPriorSummary::from_attempts(1, &duplicate, false),
            Err(TrajectoryPriorError::DuplicateAttemptId(_))
        ));
        let mismatch = vec![attempt("a", 1, TrajectoryOutcome::ProjectedSpins(vec![1, -1]))];
        assert!(matches!(
            TrajectoryPriorSummary::from_attempts(1, &mismatch, false),
            Err(TrajectoryPriorError::SpinCountMismatch { .. })
        ));
        let invalid = vec![attempt("a", 1, TrajectoryOutcome::ProjectedSpins(vec![0]))];
        assert!(matches!(
            TrajectoryPriorSummary::from_attempts(1, &invalid, false),
            Err(TrajectoryPriorError::InvalidSpin { .. })
        ));
        let empty_id = vec![attempt("   ", 1, TrajectoryOutcome::AmbiguousProjection)];
        assert_eq!(
            TrajectoryPriorSummary::from_attempts(1, &empty_id, false).unwrap_err(),
            TrajectoryPriorError::EmptyAttemptId
        );
    }

    #[test]
    fn ordering_is_a_permutation_and_ties_are_deterministic() {
        let attempts = vec![
            attempt("a", 1, TrajectoryOutcome::ProjectedSpins(vec![1, 1, -1])),
            attempt("b", 2, TrajectoryOutcome::ProjectedSpins(vec![1, -1, -1])),
            attempt("c", 3, TrajectoryOutcome::ProjectedSpins(vec![1, 1, 1])),
        ];
        let summary = TrajectoryPriorSummary::from_attempts(3, &attempts, true).unwrap();
        let degrees = [2, 5, 5];
        let order = summary.ordered_vertices(&degrees, PriorOrdering::BiasFirst).unwrap();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
        assert_eq!(order, summary.ordered_vertices(&degrees, PriorOrdering::BiasFirst).unwrap());
        assert_eq!(
            summary.ordered_vertices(&degrees, PriorOrdering::DegreeOnly).unwrap(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn no_prior_mode_does_not_depend_on_sample_statistics() {
        let first = vec![
            attempt("a", 1, TrajectoryOutcome::ProjectedSpins(vec![1, 1, -1])),
            attempt("b", 2, TrajectoryOutcome::ProjectedSpins(vec![1, 1, -1])),
        ];
        let second = vec![
            attempt("a", 1, TrajectoryOutcome::ProjectedSpins(vec![-1, -1, 1])),
            attempt("b", 2, TrajectoryOutcome::ProjectedSpins(vec![-1, -1, 1])),
        ];
        let left = TrajectoryPriorSummary::from_attempts(3, &first, false).unwrap();
        let right = TrajectoryPriorSummary::from_attempts(3, &second, false).unwrap();
        let degrees = [2, 5, 5];
        assert_eq!(
            left.ordered_vertices(&degrees, PriorOrdering::DegreeOnly).unwrap(),
            right.ordered_vertices(&degrees, PriorOrdering::DegreeOnly).unwrap()
        );
    }

    #[test]
    fn rejects_mismatched_degree_vector() {
        let attempts = vec![attempt("a", 1, TrajectoryOutcome::ProjectedSpins(vec![1, -1]))];
        let summary = TrajectoryPriorSummary::from_attempts(2, &attempts, false).unwrap();
        assert!(matches!(
            summary.ordered_vertices(&[2], PriorOrdering::BiasFirst),
            Err(TrajectoryPriorError::DegreeCountMismatch { .. })
        ));
    }
}
