//! Budget trajectory acquisition using downstream ranking stability.
//!
//! This controller decides only whether to request another batch of trajectories
//! or stop sampling. It does not certify that a ranking is useful, that a vertex
//! is persistent, that a solution is optimal, or that more samples cannot help.
//! A ranking can stabilize at the wrong answer. Downstream exact search,
//! certification, and independent original-graph verification remain authoritative.
//!
//! Stop decisions require minimum total attempts, minimum valid projections,
//! minimum cumulative projection yield, and a predeclared number of consecutive
//! identical top-k rankings. A missing ranking resets the stability streak.
//! Reaching the cap without satisfying all stop conditions returns
//! BudgetExhausted, not a successful stable-stop decision.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// Predeclared stopping thresholds for the trajectory acquisition controller.
#[derive(Debug, Clone, PartialEq)]
pub struct AcquisitionBudgetConfig {
    /// Number of vertices in the canonical graph.
    pub vertex_count: usize,
    /// Number of leading vertices whose entire order must stabilize.
    pub top_k: usize,
    /// Minimum cumulative attempted trajectories before stable-stop can fire.
    pub min_attempts: usize,
    /// Minimum cumulative valid spin projections before stable-stop can fire.
    pub min_valid_projections: usize,
    /// Minimum cumulative valid/attempted yield required before stable-stop.
    pub min_projection_yield: f64,
    /// Consecutive batches with the same ordered top-k needed for stable-stop.
    /// Must be at least two.
    pub stable_batches_required: usize,
    /// Maximum cumulative attempted trajectories across all batches.
    pub max_attempts: usize,
}

/// One incoming batch record. Yield is derived from counts rather than provided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcquisitionBatch {
    /// Stable, non-empty ID unique within this controller instance.
    pub batch_id: String,
    /// Number of trajectories attempted in this batch; must be positive.
    pub attempted_runs: usize,
    /// Number of valid spin projections in this batch; must not exceed attempts.
    pub valid_projections: usize,
    /// Optional complete vertex ordering. None means there was no admissible
    /// recommendation for this batch (e.g. all projections were invalid).
    pub ordering: Option<Vec<usize>>,
}

/// Status returned after observing a batch.
#[derive(Debug, Clone, PartialEq)]
pub enum AcquisitionDecision {
    /// More acquisition is warranted or the stability thresholds are not met.
    Continue(AcquisitionSnapshot),
    /// The declared thresholds are met and the top-k ranking has stabilized.
    ///
    /// This means only that the ordering recommendation was stable for the
    /// configured number of batches. It is not a quality, uncertainty, or
    /// optimality certificate.
    StableRanking(AcquisitionSnapshot),
    /// Maximum attempted-run budget was reached without satisfying all stable-stop
    /// criteria. The caller must retain this as budget exhaustion, not success.
    BudgetExhausted(AcquisitionSnapshot),
}

/// Diagnostics needed to replay why the controller continued or stopped.
#[derive(Debug, Clone, PartialEq)]
pub struct AcquisitionSnapshot {
    /// Cumulative trajectory attempts, including failures.
    pub attempted_total: usize,
    /// Cumulative valid spin projections.
    pub valid_projection_total: usize,
    /// Cumulative valid/attempted ratio, or 0 when no attempts have been recorded.
    pub projection_yield: f64,
    /// Consecutive batch count with an identical ordered top-k recommendation.
    pub stable_batch_count: usize,
    /// Current ordered top-k signature, if the last batch supplied a recommendation.
    pub current_top_k: Option<Vec<usize>>,
    /// Whether the minimum attempted-run threshold has been met.
    pub minimum_attempts_met: bool,
    /// Whether the minimum valid-projection threshold has been met.
    pub minimum_valid_projections_met: bool,
    /// Whether the minimum cumulative projection-yield threshold has been met.
    pub minimum_projection_yield_met: bool,
    /// Whether the stability streak threshold has been met.
    pub ranking_stability_met: bool,
}

/// Validation error in the acquisition budget contract.
#[derive(Debug, Clone, PartialEq)]
pub enum AcquisitionBudgetError {
    /// Vertex count must be positive.
    EmptyGraph,
    /// Top-k must lie in 1..=vertex_count.
    InvalidTopK { top_k: usize, vertex_count: usize },
    /// A configured threshold is inconsistent with the maximum budget.
    InvalidAttemptThresholds {
        min_attempts: usize,
        min_valid_projections: usize,
        max_attempts: usize,
    },
    /// Minimum projection yield must be finite and lie in [0, 1].
    InvalidMinimumYield(f64),
    /// At least two stable batches are required.
    InvalidStabilityThreshold(usize),
    /// At least one trajectory batch is required.
    EmptyBatchId,
    /// Batch ID has already been observed.
    DuplicateBatchId(String),
    /// Batch attempted count must be nonzero.
    ZeroBatchAttempts(String),
    /// A batch cannot have more valid projections than attempts.
    ValidExceedsAttempts {
        batch_id: String,
        attempted_runs: usize,
        valid_projections: usize,
    },
    /// Batch input contained a complete ranking with the wrong vertex count.
    RankingLengthMismatch {
        batch_id: String,
        expected: usize,
        actual: usize,
    },
    /// Ranking contains a vertex outside the canonical range.
    RankingVertexOutOfRange {
        batch_id: String,
        vertex: usize,
        vertex_count: usize,
    },
    /// Ranking contains a vertex more than once.
    DuplicateRankingVertex { batch_id: String, vertex: usize },
    /// Cumulative counters would overflow usize.
    CounterOverflow,
}

impl fmt::Display for AcquisitionBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGraph => write!(f, "vertex count must be positive"),
            Self::InvalidTopK { top_k, vertex_count } => write!(
                f,
                "top-k value {top_k} is invalid for {vertex_count} vertices"
            ),
            Self::InvalidAttemptThresholds {
                min_attempts,
                min_valid_projections,
                max_attempts,
            } => write!(
                f,
                "inconsistent trajectory thresholds: min attempts {min_attempts}, min valid {min_valid_projections}, max attempts {max_attempts}"
            ),
            Self::InvalidMinimumYield(value) => {
                write!(f, "minimum projection yield must lie in [0, 1], got {value}")
            }
            Self::InvalidStabilityThreshold(value) => write!(
                f,
                "stable batches required must be at least 2, got {value}"
            ),
            Self::EmptyBatchId => write!(f, "batch ID must not be empty"),
            Self::DuplicateBatchId(id) => write!(f, "duplicate acquisition batch ID: {id}"),
            Self::ZeroBatchAttempts(id) => write!(f, "batch {id} has zero attempts"),
            Self::ValidExceedsAttempts {
                batch_id,
                attempted_runs,
                valid_projections,
            } => write!(
                f,
                "batch {batch_id} has {valid_projections} valid projections but only {attempted_runs} attempts"
            ),
            Self::RankingLengthMismatch { batch_id, expected, actual } => write!(
                f,
                "batch {batch_id} ranking has {actual} vertices; expected {expected}"
            ),
            Self::RankingVertexOutOfRange { batch_id, vertex, vertex_count } => write!(
                f,
                "batch {batch_id} ranking vertex {vertex} is outside 0..{vertex_count}"
            ),
            Self::DuplicateRankingVertex { batch_id, vertex } => write!(
                f,
                "batch {batch_id} ranking repeats vertex {vertex}"
            ),
            Self::CounterOverflow => write!(f, "cumulative acquisition counters overflowed"),
        }
    }
}

impl Error for AcquisitionBudgetError {}

/// Stateful deterministic gate for adaptive trajectory acquisition.
#[derive(Debug, Clone)]
pub struct TrajectoryAcquisitionGate {
    config: AcquisitionBudgetConfig,
    seen_batch_ids: HashSet<String>,
    attempted_total: usize,
    valid_projection_total: usize,
    previous_top_k: Option<Vec<usize>>,
    stable_batch_count: usize,
}

impl TrajectoryAcquisitionGate {
    /// Construct a gate from a frozen configuration.
    pub fn new(config: AcquisitionBudgetConfig) -> Result<Self, AcquisitionBudgetError> {
        if config.vertex_count == 0 {
            return Err(AcquisitionBudgetError::EmptyGraph);
        }
        if config.top_k == 0 || config.top_k > config.vertex_count {
            return Err(AcquisitionBudgetError::InvalidTopK {
                top_k: config.top_k,
                vertex_count: config.vertex_count,
            });
        }
        if config.min_attempts == 0
            || config.min_valid_projections == 0
            || config.min_attempts > config.max_attempts
            || config.min_valid_projections > config.max_attempts
        {
            return Err(AcquisitionBudgetError::InvalidAttemptThresholds {
                min_attempts: config.min_attempts,
                min_valid_projections: config.min_valid_projections,
                max_attempts: config.max_attempts,
            });
        }
        if !config.min_projection_yield.is_finite()
            || !(0.0..=1.0).contains(&config.min_projection_yield)
        {
            return Err(AcquisitionBudgetError::InvalidMinimumYield(
                config.min_projection_yield,
            ));
        }
        if config.stable_batches_required < 2 {
            return Err(AcquisitionBudgetError::InvalidStabilityThreshold(
                config.stable_batches_required,
            ));
        }

        Ok(Self {
            config,
            seen_batch_ids: HashSet::new(),
            attempted_total: 0,
            valid_projection_total: 0,
            previous_top_k: None,
            stable_batch_count: 0,
        })
    }

    /// Observe one batch and return the only permitted control decision.
    ///
    /// Invalid batch input is rejected before controller state is mutated.
    pub fn observe(
        &mut self,
        batch: AcquisitionBatch,
    ) -> Result<AcquisitionDecision, AcquisitionBudgetError> {
        if batch.batch_id.trim().is_empty() {
            return Err(AcquisitionBudgetError::EmptyBatchId);
        }
        if self.seen_batch_ids.contains(&batch.batch_id) {
            return Err(AcquisitionBudgetError::DuplicateBatchId(batch.batch_id));
        }
        if batch.attempted_runs == 0 {
            return Err(AcquisitionBudgetError::ZeroBatchAttempts(batch.batch_id));
        }
        if batch.valid_projections > batch.attempted_runs {
            return Err(AcquisitionBudgetError::ValidExceedsAttempts {
                batch_id: batch.batch_id,
                attempted_runs: batch.attempted_runs,
                valid_projections: batch.valid_projections,
            });
        }

        let top_k = match batch.ordering.as_ref() {
            Some(ordering) => {
                self.validate_ordering(&batch.batch_id, ordering)?;
                Some(ordering[..self.config.top_k].to_vec())
            }
            None => None,
        };

        let attempted_total = self
            .attempted_total
            .checked_add(batch.attempted_runs)
            .ok_or(AcquisitionBudgetError::CounterOverflow)?;
        let valid_projection_total = self
            .valid_projection_total
            .checked_add(batch.valid_projections)
            .ok_or(AcquisitionBudgetError::CounterOverflow)?;

        if attempted_total > self.config.max_attempts {
            // Do not silently discard excess work. The caller must split batches
            // to honor the budget and make the gate's accounting auditable.
            return Err(AcquisitionBudgetError::InvalidAttemptThresholds {
                min_attempts: self.config.min_attempts,
                min_valid_projections: self.config.min_valid_projections,
                max_attempts: self.config.max_attempts,
            });
        }

        self.seen_batch_ids.insert(batch.batch_id);
        self.attempted_total = attempted_total;
        self.valid_projection_total = valid_projection_total;

        if let Some(current_top_k) = top_k {
            if self.previous_top_k.as_ref() == Some(&current_top_k) {
                self.stable_batch_count = self.stable_batch_count.saturating_add(1);
            } else {
                self.stable_batch_count = 1;
            }
            self.previous_top_k = Some(current_top_k);
        } else {
            self.previous_top_k = None;
            self.stable_batch_count = 0;
        }

        let snapshot = self.snapshot();
        let all_sampling_thresholds_met = snapshot.minimum_attempts_met
            && snapshot.minimum_valid_projections_met
            && snapshot.minimum_projection_yield_met;
        if all_sampling_thresholds_met && snapshot.ranking_stability_met {
            Ok(AcquisitionDecision::StableRanking(snapshot))
        } else if self.attempted_total >= self.config.max_attempts {
            Ok(AcquisitionDecision::BudgetExhausted(snapshot))
        } else {
            Ok(AcquisitionDecision::Continue(snapshot))
        }
    }

    /// Current cumulative diagnostic snapshot.
    pub fn current_snapshot(&self) -> AcquisitionSnapshot {
        self.snapshot()
    }

    fn validate_ordering(
        &self,
        batch_id: &str,
        ordering: &[usize],
    ) -> Result<(), AcquisitionBudgetError> {
        if ordering.len() != self.config.vertex_count {
            return Err(AcquisitionBudgetError::RankingLengthMismatch {
                batch_id: batch_id.to_owned(),
                expected: self.config.vertex_count,
                actual: ordering.len(),
            });
        }
        let mut seen = vec![false; self.config.vertex_count];
        for &vertex in ordering {
            if vertex >= self.config.vertex_count {
                return Err(AcquisitionBudgetError::RankingVertexOutOfRange {
                    batch_id: batch_id.to_owned(),
                    vertex,
                    vertex_count: self.config.vertex_count,
                });
            }
            if seen[vertex] {
                return Err(AcquisitionBudgetError::DuplicateRankingVertex {
                    batch_id: batch_id.to_owned(),
                    vertex,
                });
            }
            seen[vertex] = true;
        }
        Ok(())
    }

    fn snapshot(&self) -> AcquisitionSnapshot {
        let projection_yield = if self.attempted_total > 0 {
            self.valid_projection_total as f64 / self.attempted_total as f64
        } else {
            0.0
        };
        let minimum_attempts_met = self.attempted_total >= self.config.min_attempts;
        let minimum_valid_projections_met =
            self.valid_projection_total >= self.config.min_valid_projections;
        let minimum_projection_yield_met =
            projection_yield >= self.config.min_projection_yield;
        let ranking_stability_met =
            self.stable_batch_count >= self.config.stable_batches_required;

        AcquisitionSnapshot {
            attempted_total: self.attempted_total,
            valid_projection_total: self.valid_projection_total,
            projection_yield,
            stable_batch_count: self.stable_batch_count,
            current_top_k: self.previous_top_k.clone(),
            minimum_attempts_met,
            minimum_valid_projections_met,
            minimum_projection_yield_met,
            ranking_stability_met,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AcquisitionBudgetConfig {
        AcquisitionBudgetConfig {
            vertex_count: 4,
            top_k: 2,
            min_attempts: 6,
            min_valid_projections: 4,
            min_projection_yield: 0.5,
            stable_batches_required: 3,
            max_attempts: 10,
        }
    }

    fn batch(id: &str, attempts: usize, valid: usize, ordering: Option<Vec<usize>>) -> AcquisitionBatch {
        AcquisitionBatch {
            batch_id: id.to_owned(),
            attempted_runs: attempts,
            valid_projections: valid,
            ordering,
        }
    }

    #[test]
    fn stops_only_after_all_thresholds_and_consecutive_top_k_are_met() {
        let mut gate = TrajectoryAcquisitionGate::new(config()).unwrap();
        let order = vec![2, 1, 0, 3];
        assert!(matches!(
            gate.observe(batch("b1", 2, 1, Some(order.clone()))).unwrap(),
            AcquisitionDecision::Continue(_)
        ));
        assert!(matches!(
            gate.observe(batch("b2", 2, 2, Some(order.clone()))).unwrap(),
            AcquisitionDecision::Continue(_)
        ));
        let decision = gate.observe(batch("b3", 2, 2, Some(order.clone()))).unwrap();
        match decision {
            AcquisitionDecision::StableRanking(snapshot) => {
                assert_eq!(snapshot.attempted_total, 6);
                assert_eq!(snapshot.valid_projection_total, 5);
                assert!((snapshot.projection_yield - (5.0 / 6.0)).abs() < 1e-12);
                assert_eq!(snapshot.stable_batch_count, 3);
                assert_eq!(snapshot.current_top_k, Some(vec![2, 1]));
            }
            other => panic!("expected stable ranking, got {other:?}"),
        }
    }

    #[test]
    fn changed_ranking_resets_the_stability_streak() {
        let mut gate = TrajectoryAcquisitionGate::new(config()).unwrap();
        gate.observe(batch("b1", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        gate.observe(batch("b2", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        let snapshot = match gate.observe(batch("b3", 2, 2, Some(vec![1, 0, 2, 3]))).unwrap() {
            AcquisitionDecision::Continue(snapshot) => snapshot,
            other => panic!("expected continue after ranking changed, got {other:?}"),
        };
        assert_eq!(snapshot.stable_batch_count, 1);
        assert_eq!(snapshot.current_top_k, Some(vec![1, 0]));
    }

    #[test]
    fn unavailable_ranking_clears_stability_and_stays_in_attempt_denominator() {
        let mut gate = TrajectoryAcquisitionGate::new(config()).unwrap();
        gate.observe(batch("b1", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        gate.observe(batch("b2", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        let snapshot = match gate.observe(batch("b3-invalid", 2, 0, None)).unwrap() {
            AcquisitionDecision::Continue(snapshot) => snapshot,
            other => panic!("expected continue when ranking is unavailable, got {other:?}"),
        };
        assert_eq!(snapshot.attempted_total, 6);
        assert_eq!(snapshot.valid_projection_total, 4);
        assert!((snapshot.projection_yield - (4.0 / 6.0)).abs() < 1e-12);
        assert_eq!(snapshot.stable_batch_count, 0);
        assert_eq!(snapshot.current_top_k, None);
    }

    #[test]
    fn budget_exhaustion_is_not_reported_as_stable_stop() {
        let mut c = config();
        c.max_attempts = 6;
        c.min_valid_projections = 6;
        let mut gate = TrajectoryAcquisitionGate::new(c).unwrap();
        gate.observe(batch("b1", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        gate.observe(batch("b2", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        let decision = gate.observe(batch("b3", 2, 1, Some(vec![0, 1, 2, 3]))).unwrap();
        assert!(matches!(decision, AcquisitionDecision::BudgetExhausted(_)));
    }

    #[test]
    fn fixed_but_wrong_ranking_can_stabilize_without_becoming_a_certificate() {
        let mut gate = TrajectoryAcquisitionGate::new(config()).unwrap();
        let deliberately_arbitrary = vec![3, 2, 1, 0];
        gate.observe(batch("b1", 2, 2, Some(deliberately_arbitrary.clone()))).unwrap();
        gate.observe(batch("b2", 2, 2, Some(deliberately_arbitrary.clone()))).unwrap();
        let decision = gate.observe(batch("b3", 2, 2, Some(deliberately_arbitrary)).unwrap();
        assert!(matches!(decision, AcquisitionDecision::StableRanking(_)));
        // The result type contains no optimum, persistency, or quality certificate.
    }

    #[test]
    fn rejects_duplicate_ids_bad_counts_and_invalid_permutations_without_mutation() {
        let mut gate = TrajectoryAcquisitionGate::new(config()).unwrap();
        assert!(matches!(
            gate.observe(batch("bad-count", 2, 3, Some(vec![0, 1, 2, 3]))),
            Err(AcquisitionBudgetError::ValidExceedsAttempts { .. })
        ));
        assert_eq!(gate.current_snapshot().attempted_total, 0);
        assert!(matches!(
            gate.observe(batch("bad-order", 2, 2, Some(vec![0, 1, 1, 3]))),
            Err(AcquisitionBudgetError::DuplicateRankingVertex { .. })
        ));
        assert_eq!(gate.current_snapshot().attempted_total, 0);
        gate.observe(batch("ok", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        assert!(matches!(
            gate.observe(batch("ok", 2, 2, Some(vec![0, 1, 2, 3]))),
            Err(AcquisitionBudgetError::DuplicateBatchId(_))
        ));
        assert_eq!(gate.current_snapshot().attempted_total, 2);
    }

    #[test]
    fn validates_configuration_and_rejects_batches_that_exceed_cap() {
        let mut c = config();
        c.stable_batches_required = 1;
        assert!(matches!(
            TrajectoryAcquisitionGate::new(c),
            Err(AcquisitionBudgetError::InvalidStabilityThreshold(_))
        ));

        let mut gate = TrajectoryAcquisitionGate::new(config()).unwrap();
        assert!(matches!(
            gate.observe(batch("oversized", 11, 11, Some(vec![0, 1, 2, 3]))),
            Err(AcquisitionBudgetError::ValidExceedsAttempts { .. })
        ));
        assert_eq!(gate.current_snapshot().attempted_total, 0);

        let mut gate = TrajectoryAcquisitionGate::new(config()).unwrap();
        gate.observe(batch("first", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        gate.observe(batch("second", 2, 2, Some(vec![0, 1, 2, 3]))).unwrap();
        assert!(matches!(
            gate.observe(batch("over-cap", 7, 7, Some(vec![0, 1, 2, 3]))),
            Err(AcquisitionBudgetError::InvalidAttemptThresholds { .. })
        ));
        assert_eq!(gate.current_snapshot().attempted_total, 4);
    }
}
