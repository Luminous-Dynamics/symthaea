//! Validated, graph-bound inclusion-frequency signals for MIS reduction experiments.
//!
//! A signal describes how often each vertex appeared in a sample population. It
//! does not identify the process that generated the samples and makes no claim
//! about quantum hardware. Source provenance and evidence class belong in the
//! experiment manifest. An absent sample must never be confused with a measured
//! zero inclusion frequency.

use core::fmt;
use crate::mis_graph_identity::MisGraphIdentity;

/// Direction in which an empirical inclusion-frequency signal is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisSelectionMode {
    /// Select a vertex with the highest observed inclusion count.
    PreferInSet,
    /// Select a vertex with the lowest observed inclusion count.
    PreferOutSet,
}

/// A validated empirical inclusion-frequency signal tied to one graph instance.
///
/// The opaque instance identifier must match the frozen benchmark manifest.
/// A caller must validate that identifier and the vertex ordering are bound to the
/// same graph before passing this signal to a reducer. This type is not a
/// cryptographic provenance receipt and does not establish the signal's origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisInformerSignal {
    instance_id: String,
    /// SHA-256 of the canonical graph artifact/manifest used to produce the signal.
    graph_sha256: [u8; 32],
    vertex_count: usize,
    sample_count: u64,
    inclusion_counts: Vec<u64>,
}

/// A recommendation whose empirical frequency was separated from its nearest rival.
#[derive(Debug, Clone, PartialEq)]
pub struct MisConfidenceRecommendation {
    /// Selected vertex and raw observed frequency.
    pub recommendation: MisVertexRecommendation,
    /// Empirical inclusion rate for the selected vertex.
    pub empirical_rate: f64,
    /// Lower simultaneous confidence bound for the selected vertex rate.
    pub lower_bound: f64,
    /// Upper simultaneous confidence bound for the selected vertex rate.
    pub upper_bound: f64,
    /// Upper bound for the best rival when preferring inclusion, or lower bound
    /// for the best rival when preferring exclusion.
    pub competitor_bound: f64,
    /// Family-wise error probability supplied by the caller.
    pub familywise_error_probability: f64,
}

/// Why the confidence-aware informer declined to make a frequency-based choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisConfidenceAbstention {
    /// The most extreme empirical frequency is tied across multiple candidates.
    EmpiricalTie,
    /// Simultaneous bounds overlap, so the score ordering is not resolved.
    ConfidenceIntervalsOverlap,
    /// A single eligible vertex leaves no competing signal score to compare.
    OnlyOneCandidate,
}

/// Confidence-aware informer result. Abstention is a valid outcome, not an error.
#[derive(Debug, Clone, PartialEq)]
pub enum MisConfidenceDecision {
    /// The selected score is separated from the strongest competing candidate.
    Selected(MisConfidenceRecommendation),
    /// The informer declines to use its frequency score for this decision.
    Abstain {
        /// Reason the empirical signal did not resolve the decision.
        reason: MisConfidenceAbstention,
        /// Number of eligible vertices considered.
        eligible_vertex_count: usize,
        /// Number of source samples summarized by the signal.
        sample_count: u64,
    },
}

/// Errors returned when constructing or using an MIS informer signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisSignalError {
    /// The graph instance identifier was empty or contained control characters.
    InvalidInstanceId,
    /// No sample population exists; missing evidence cannot mean zero frequency.
    NoSamples,
    /// Signal count vector does not cover exactly the graph's vertices.
    VertexCountMismatch {
        /// Expected number of vertex counts.
        expected: usize,
        /// Actual number of vertex counts.
        actual: usize,
    },
    /// A per-vertex inclusion count exceeds the declared number of samples.
    InclusionCountExceedsSamples {
        /// Vertex index of the invalid count.
        vertex: usize,
        /// Reported number of appearances.
        inclusion_count: u64,
        /// Declared number of samples.
        sample_count: u64,
    },
    /// The signal belongs to a different benchmark instance.
    InstanceIdMismatch,
    /// The signal's source graph digest differs from the expected graph artifact.
    GraphDigestMismatch,
    /// The requested family-wise error probability must be finite and strictly in (0, 1).
    InvalidConfidenceLevel,
    /// Confidence-aware selection requires a non-empty candidate set.
    EmptyEligibleSet,
    /// A candidate vertex appears more than once in the eligible set.
    DuplicateEligibleVertex {
        /// Repeated vertex index.
        vertex: usize,
    },
    /// A candidate vertex is not part of the graph.
    EligibleVertexOutOfRange {
        /// Out-of-range candidate index.
        vertex: usize,
        /// Number of vertices in the graph.
        vertex_count: usize,
    },
    /// The signal's graph size differs from the reducer's graph.
    GraphSizeMismatch {
        /// Vertex count declared by the signal.
        signal_vertices: usize,
        /// Actual vertex count expected by the reducer.
        expected_vertices: usize,
    },
}

/// Selected vertex and the observed frequency that motivated its selection.
///
/// This is a heuristic recommendation only. It is not a proof that the vertex
/// belongs to, or is excluded from, an optimal independent set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisVertexRecommendation {
    /// Chosen zero-based vertex index.
    pub vertex: usize,
    /// Number of source samples in which the vertex appeared.
    pub inclusion_count: u64,
    /// Total number of source samples in the signal.
    pub sample_count: u64,
    /// Number of vertices tied at the chosen extreme score.
    pub tie_count: usize,
    /// Interpretation used to choose this vertex.
    pub mode: MisSelectionMode,
}

impl fmt::Display for MisSignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInstanceId => {
                write!(f, "instance identifier must be non-empty and contain no control characters")
            }
            Self::NoSamples => write!(f, "informer signal has no samples"),
            Self::VertexCountMismatch { expected, actual } => write!(
                f,
                "informer vector length mismatch: expected {expected}, got {actual}"
            ),
            Self::InclusionCountExceedsSamples {
                vertex,
                inclusion_count,
                sample_count,
            } => write!(
                f,
                "vertex {vertex} appears {inclusion_count} times in only {sample_count} samples"
            ),
            Self::InstanceIdMismatch => {
                write!(f, "informer signal belongs to a different graph instance")
            }
            Self::GraphDigestMismatch => {
                write!(f, "informer signal graph digest does not match the expected graph")
            }
            Self::InvalidConfidenceLevel => {
                write!(f, "family-wise error probability must be finite and strictly between 0 and 1")
            }
            Self::EmptyEligibleSet => write!(f, "confidence-aware selection requires at least one eligible vertex"),
            Self::DuplicateEligibleVertex { vertex } => {
                write!(f, "eligible vertex {vertex} appears more than once")
            }
            Self::EligibleVertexOutOfRange { vertex, vertex_count } => write!(
                f,
                "eligible vertex {vertex} is outside graph with {vertex_count} vertices"
            ),
            Self::GraphSizeMismatch {
                signal_vertices,
                expected_vertices,
            } => write!(
                f,
                "informer graph size mismatch: signal has {signal_vertices}, expected {expected_vertices}"
            ),
        }
    }
}

impl std::error::Error for MisSignalError {}

impl MisInformerSignal {
    /// Construct a signal using a caller-supplied digest that this method does not verify.
    ///
    /// Prefer `from_graph` for normal use. The vector index is the vertex index.
    /// Every count must be at most the total sample count. At least one sample is
    /// required: missing evidence is not evidence of zero inclusion probability.
    pub fn from_unverified_digest(
        instance_id: impl Into<String>,
        graph_sha256: [u8; 32],
        vertex_count: usize,
        sample_count: u64,
        inclusion_counts: Vec<u64>,
    ) -> Result<Self, MisSignalError> {
        let instance_id = instance_id.into();
        if instance_id.trim().is_empty() || instance_id.chars().any(char::is_control) {
            return Err(MisSignalError::InvalidInstanceId);
        }
        if sample_count == 0 {
            return Err(MisSignalError::NoSamples);
        }
        if inclusion_counts.len() != vertex_count {
            return Err(MisSignalError::VertexCountMismatch {
                expected: vertex_count,
                actual: inclusion_counts.len(),
            });
        }
        for (vertex, &inclusion_count) in inclusion_counts.iter().enumerate() {
            if inclusion_count > sample_count {
                return Err(MisSignalError::InclusionCountExceedsSamples {
                    vertex,
                    inclusion_count,
                    sample_count,
                });
            }
        }

        Ok(Self {
            instance_id,
            graph_sha256,
            vertex_count,
            sample_count,
            inclusion_counts,
        })
    }

    /// Build a signal from a computed canonical graph identity.
    ///
    /// Prefer this constructor over supplying a raw digest: the identity's digest
    /// was computed from a canonical edge list, rather than asserted by the caller.
    /// This still does not attest who produced the sample population.
    pub fn from_graph(
        instance_id: impl Into<String>,
        graph: &MisGraphIdentity,
        sample_count: u64,
        inclusion_counts: Vec<u64>,
    ) -> Result<Self, MisSignalError> {
        Self::from_unverified_digest(
            instance_id,
            graph.sha256(),
            graph.vertex_count(),
            sample_count,
            inclusion_counts,
        )
    }

    /// Recommend a vertex only when this signal matches a computed graph identity.
    pub fn recommend_for_graph(
        &self,
        expected_instance_id: &str,
        graph: &MisGraphIdentity,
        mode: MisSelectionMode,
    ) -> Result<MisVertexRecommendation, MisSignalError> {
        self.recommend_vertex_for_unverified_digest(
            expected_instance_id,
            graph.sha256(),
            graph.vertex_count(),
            mode,
        )
    }

    /// Make a confidence-aware choice only when one candidate's observed rate is separated.
    ///
    /// Uses a Hoeffding/union-bound interval radius
    /// sqrt(ln(2*m/alpha)/(2*n)), where m is the number of eligible vertices,
    /// n is the sample count, and alpha is the requested family-wise error
    /// probability. For the stated bound to be meaningful, the sample population
    /// must be compatible with independent, identically distributed Bernoulli
    /// observations for each vertex. This routine cannot verify that assumption;
    /// the caller must document the sampling design in the run manifest.
    ///
    /// This is uncertainty about empirical inclusion-rate ordering, not a guarantee
    /// that a selected vertex belongs to an optimal independent set. Tied or
    /// overlapping scores produce an explicit abstention so callers can take more
    /// samples or use a classical fallback.
    pub fn recommend_confidently_for_graph(
        &self,
        expected_instance_id: &str,
        graph: &MisGraphIdentity,
        eligible_vertices: &[usize],
        mode: MisSelectionMode,
        familywise_error_probability: f64,
    ) -> Result<MisConfidenceDecision, MisSignalError> {
        self.validate_graph_binding(
            expected_instance_id,
            graph.sha256(),
            graph.vertex_count(),
        )?;

        if !familywise_error_probability.is_finite()
            || familywise_error_probability <= 0.0
            || familywise_error_probability >= 1.0
        {
            return Err(MisSignalError::InvalidConfidenceLevel);
        }
        if eligible_vertices.is_empty() {
            return Err(MisSignalError::EmptyEligibleSet);
        }

        let mut seen = vec![false; self.vertex_count];
        for &vertex in eligible_vertices {
            if vertex >= self.vertex_count {
                return Err(MisSignalError::EligibleVertexOutOfRange {
                    vertex,
                    vertex_count: self.vertex_count,
                });
            }
            if seen[vertex] {
                return Err(MisSignalError::DuplicateEligibleVertex { vertex });
            }
            seen[vertex] = true;
        }

        let abstain = |reason| {
            Ok(MisConfidenceDecision::Abstain {
                reason,
                eligible_vertex_count: eligible_vertices.len(),
                sample_count: self.sample_count,
            })
        };
        if eligible_vertices.len() == 1 {
            return abstain(MisConfidenceAbstention::OnlyOneCandidate);
        }

        let selected_vertex = eligible_vertices.iter().copied().reduce(|best, candidate| {
            let best_count = self.inclusion_counts[best];
            let candidate_count = self.inclusion_counts[candidate];
            match mode {
                MisSelectionMode::PreferInSet
                    if candidate_count > best_count
                        || (candidate_count == best_count && candidate < best) =>
                {
                    candidate
                }
                MisSelectionMode::PreferOutSet
                    if candidate_count < best_count
                        || (candidate_count == best_count && candidate < best) =>
                {
                    candidate
                }
                _ => best,
            }
        }).expect("eligible vertices checked non-empty");

        let selected_count = self.inclusion_counts[selected_vertex];
        let tie_count = eligible_vertices
            .iter()
            .filter(|&&vertex| self.inclusion_counts[vertex] == selected_count)
            .count();
        if tie_count > 1 {
            return abstain(MisConfidenceAbstention::EmpiricalTie);
        }

        let competitor_vertex = eligible_vertices.iter().copied()
            .filter(|&vertex| vertex != selected_vertex)
            .reduce(|best, candidate| {
                let best_count = self.inclusion_counts[best];
                let candidate_count = self.inclusion_counts[candidate];
                match mode {
                    MisSelectionMode::PreferInSet if candidate_count > best_count => candidate,
                    MisSelectionMode::PreferOutSet if candidate_count < best_count => candidate,
                    _ => best,
                }
            }).expect("at least two eligible vertices checked above");

        let eligible_count = eligible_vertices.len() as f64;
        let samples = self.sample_count as f64;
        let radius =
            ((2.0 * eligible_count / familywise_error_probability).ln() / (2.0 * samples)).sqrt();
        let selected_rate = selected_count as f64 / samples;
        let competitor_rate = self.inclusion_counts[competitor_vertex] as f64 / samples;
        let lower_bound = (selected_rate - radius).max(0.0);
        let upper_bound = (selected_rate + radius).min(1.0);
        let competitor_lower = (competitor_rate - radius).max(0.0);
        let competitor_upper = (competitor_rate + radius).min(1.0);
        let separated = match mode {
            MisSelectionMode::PreferInSet => lower_bound > competitor_upper,
            MisSelectionMode::PreferOutSet => upper_bound < competitor_lower,
        };
        if !separated {
            return abstain(MisConfidenceAbstention::ConfidenceIntervalsOverlap);
        }

        Ok(MisConfidenceDecision::Selected(MisConfidenceRecommendation {
            recommendation: MisVertexRecommendation {
                vertex: selected_vertex,
                inclusion_count: selected_count,
                sample_count: self.sample_count,
                tie_count,
                mode,
            },
            empirical_rate: selected_rate,
            lower_bound,
            upper_bound,
            competitor_bound: match mode {
                MisSelectionMode::PreferInSet => competitor_upper,
                MisSelectionMode::PreferOutSet => competitor_lower,
            },
            familywise_error_probability,
        }))
    }

    fn validate_graph_binding(
        &self,
        expected_instance_id: &str,
        expected_graph_sha256: [u8; 32],
        expected_vertex_count: usize,
    ) -> Result<(), MisSignalError> {
        if self.instance_id != expected_instance_id {
            return Err(MisSignalError::InstanceIdMismatch);
        }
        if self.graph_sha256 != expected_graph_sha256 {
            return Err(MisSignalError::GraphDigestMismatch);
        }
        if self.vertex_count != expected_vertex_count {
            return Err(MisSignalError::GraphSizeMismatch {
                signal_vertices: self.vertex_count,
                expected_vertices: expected_vertex_count,
            });
        }
        Ok(())
    }

    /// Frozen benchmark-instance identifier attached to this signal.
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// Vertex count declared by this signal.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Total source sample count.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Per-vertex number of sample appearances in vertex-index order.
    pub fn inclusion_counts(&self) -> &[u64] {
        &self.inclusion_counts
    }

    /// Low-level selection using a caller-supplied digest and vertex count.
    ///
    /// Prefer `recommend_for_graph` so the expected identity is computed from
    /// canonical graph edges. Ties are resolved by the lowest index. This heuristic
    /// is not proof of optimal membership; provenance and final verification remain
    /// the caller's responsibility.
    pub fn recommend_vertex_for_unverified_digest(
        &self,
        expected_instance_id: &str,
        expected_graph_sha256: [u8; 32],
        expected_vertex_count: usize,
        mode: MisSelectionMode,
    ) -> Result<MisVertexRecommendation, MisSignalError> {
        if self.instance_id != expected_instance_id {
            return Err(MisSignalError::InstanceIdMismatch);
        }
        if self.graph_sha256 != expected_graph_sha256 {
            return Err(MisSignalError::GraphDigestMismatch);
        }
        if self.vertex_count != expected_vertex_count {
            return Err(MisSignalError::GraphSizeMismatch {
                signal_vertices: self.vertex_count,
                expected_vertices: expected_vertex_count,
            });
        }
        if self.vertex_count == 0 {
            return Err(MisSignalError::VertexCountMismatch {
                expected: 1,
                actual: 0,
            });
        }

        let extreme = match mode {
            MisSelectionMode::PreferInSet => self.inclusion_counts.iter().copied().max(),
            MisSelectionMode::PreferOutSet => self.inclusion_counts.iter().copied().min(),
        }
        .expect("non-empty vertex counts checked above");

        let tie_count = self
            .inclusion_counts
            .iter()
            .filter(|&&count| count == extreme)
            .count();

        // position() makes ties deterministic by selecting the lowest vertex index.
        let vertex = self
            .inclusion_counts
            .iter()
            .position(|&count| count == extreme)
            .expect("extreme count must occur in a non-empty vector");

        Ok(MisVertexRecommendation {
            vertex,
            inclusion_count: extreme,
            sample_count: self.sample_count,
            tie_count,
            mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_or_control_character_instance_ids() {
        assert_eq!(
            MisInformerSignal::from_unverified_digest("  ", [0; 32], 2, 1, vec![0, 0]).expect_err("blank id is invalid"),
            MisSignalError::InvalidInstanceId
        );
        assert_eq!(
            MisInformerSignal::from_unverified_digest("graph\nwrong", [0; 32], 2, 1, vec![0, 0])
                .expect_err("control character is invalid"),
            MisSignalError::InvalidInstanceId
        );
    }

    #[test]
    fn rejects_missing_samples_instead_of_inventing_zero_frequency() {
        assert_eq!(
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 2, 0, vec![0, 0])
                .expect_err("no samples are not negative evidence"),
            MisSignalError::NoSamples
        );
    }

    #[test]
    fn rejects_vector_length_mismatch() {
        assert_eq!(
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 3, 4, vec![0, 1])
                .expect_err("one vertex is missing"),
            MisSignalError::VertexCountMismatch {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn rejects_inclusion_count_greater_than_sample_count() {
        assert_eq!(
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 2, 4, vec![2, 5])
                .expect_err("count cannot exceed sample total"),
            MisSignalError::InclusionCountExceedsSamples {
                vertex: 1,
                inclusion_count: 5,
                sample_count: 4,
            }
        );
    }

    #[test]
    fn recommend_in_set_picks_highest_frequency_and_breaks_ties_by_index() {
        let signal =
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 4, 10, vec![2, 8, 8, 4]).expect("valid signal");
        let result = signal
            .recommend_vertex_for_unverified_digest("mis-case-a", [1; 32], 4, MisSelectionMode::PreferInSet)
            .expect("matching graph identity");
        assert_eq!(result.vertex, 1);
        assert_eq!(result.inclusion_count, 8);
        assert_eq!(result.sample_count, 10);
        assert_eq!(result.tie_count, 2);
    }

    #[test]
    fn recommend_out_set_picks_lowest_frequency_and_breaks_ties_by_index() {
        let signal =
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 4, 10, vec![0, 8, 0, 4]).expect("valid signal");
        let result = signal
            .recommend_vertex_for_unverified_digest("mis-case-a", [1; 32], 4, MisSelectionMode::PreferOutSet)
            .expect("matching graph identity");
        assert_eq!(result.vertex, 0);
        assert_eq!(result.inclusion_count, 0);
        assert_eq!(result.tie_count, 2);
    }

    #[test]
    fn rejects_signal_from_wrong_instance() {
        let signal =
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 2, 5, vec![4, 1]).expect("valid signal");
        assert_eq!(
            signal
                .recommend_vertex_for_unverified_digest("mis-case-b", [1; 32], 2, MisSelectionMode::PreferInSet)
                .expect_err("signal is stale or belongs to another graph"),
            MisSignalError::InstanceIdMismatch
        );
    }

    #[test]
    fn rejects_signal_from_different_graph_digest_even_if_instance_id_and_size_match() {
        let signal =
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 2, 5, vec![4, 1])
                .expect("valid signal");
        assert_eq!(
            signal
                .recommend_vertex_for_unverified_digest("mis-case-a", [9; 32], 2, MisSelectionMode::PreferInSet)
                .expect_err("graph content digest differs"),
            MisSignalError::GraphDigestMismatch
        );
    }

    #[test]
    fn rejects_wrong_graph_size() {
        let signal =
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 3, 5, vec![4, 1, 2]).expect("valid signal");
        assert_eq!(
            signal
                .recommend_vertex_for_unverified_digest("mis-case-a", [1; 32], 4, MisSelectionMode::PreferOutSet)
                .expect_err("graph size does not match"),
            MisSignalError::GraphSizeMismatch {
                signal_vertices: 3,
                expected_vertices: 4,
            }
        );
    }

    #[test]
    fn zero_frequency_is_a_real_observation_when_samples_exist() {
        let signal =
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 3, 100, vec![0, 1, 0]).expect("valid signal");
        let result = signal
            .recommend_vertex_for_unverified_digest("mis-case-a", [1; 32], 3, MisSelectionMode::PreferOutSet)
            .expect("matching graph identity");
        assert_eq!(result.sample_count, 100);
        assert_eq!(result.inclusion_count, 0);
        assert_eq!(result.tie_count, 2);
    }

    #[test]
    fn confidence_aware_selector_abstains_on_overlapping_intervals() {
        let graph = MisGraphIdentity::from_edges(2, &[(0, 1)]).expect("valid graph");
        let signal = MisInformerSignal::from_graph("case", &graph, 10, vec![6, 5])
            .expect("valid signal");
        assert_eq!(
            signal.recommend_confidently_for_graph(
                "case",
                &graph,
                &[0, 1],
                MisSelectionMode::PreferInSet,
                0.05,
            ).expect("valid confidence request"),
            MisConfidenceDecision::Abstain {
                reason: MisConfidenceAbstention::ConfidenceIntervalsOverlap,
                eligible_vertex_count: 2,
                sample_count: 10,
            }
        );
    }

    #[test]
    fn confidence_aware_selector_abstains_on_empirical_tie() {
        let graph = MisGraphIdentity::from_edges(2, &[(0, 1)]).expect("valid graph");
        let signal = MisInformerSignal::from_graph("case", &graph, 1000, vec![500, 500])
            .expect("valid signal");
        assert!(matches!(
            signal.recommend_confidently_for_graph(
                "case", &graph, &[0, 1], MisSelectionMode::PreferOutSet, 0.05
            ).expect("valid confidence request"),
            MisConfidenceDecision::Abstain {
                reason: MisConfidenceAbstention::EmpiricalTie,
                ..
            }
        ));
    }

    #[test]
    fn confidence_aware_selector_selects_only_after_separation() {
        let graph = MisGraphIdentity::from_edges(2, &[(0, 1)]).expect("valid graph");
        let signal = MisInformerSignal::from_graph("case", &graph, 1000, vec![950, 20])
            .expect("valid signal");
        let decision = signal.recommend_confidently_for_graph(
            "case", &graph, &[0, 1], MisSelectionMode::PreferInSet, 0.05
        ).expect("valid confidence request");
        match decision {
            MisConfidenceDecision::Selected(result) => {
                assert_eq!(result.recommendation.vertex, 0);
                assert!(result.lower_bound > result.competitor_bound);
                assert!(result.lower_bound <= result.empirical_rate);
                assert!(result.empirical_rate <= result.upper_bound);
            }
            MisConfidenceDecision::Abstain { reason, .. } => panic!("expected separated scores, got {reason:?}"),
        }
    }

    #[test]
    fn confidence_aware_selector_validates_alpha_and_candidate_set() {
        let graph = MisGraphIdentity::from_edges(2, &[(0, 1)]).expect("valid graph");
        let signal = MisInformerSignal::from_graph("case", &graph, 100, vec![90, 10])
            .expect("valid signal");
        assert_eq!(
            signal.recommend_confidently_for_graph("case", &graph, &[0, 1], MisSelectionMode::PreferInSet, 1.0)
                .expect_err("alpha must be less than one"),
            MisSignalError::InvalidConfidenceLevel
        );
        assert_eq!(
            signal.recommend_confidently_for_graph("case", &graph, &[], MisSelectionMode::PreferInSet, 0.05)
                .expect_err("candidate set must not be empty"),
            MisSignalError::EmptyEligibleSet
        );
        assert_eq!(
            signal.recommend_confidently_for_graph("case", &graph, &[0, 0], MisSelectionMode::PreferInSet, 0.05)
                .expect_err("duplicate candidate is invalid"),
            MisSignalError::DuplicateEligibleVertex { vertex: 0 }
        );
    }

    #[test]
    fn computed_graph_identity_binds_signal_to_canonical_graph() {
        let graph = MisGraphIdentity::from_edges(3, &[(0, 1), (1, 2)]).expect("valid graph");
        let equivalent = MisGraphIdentity::from_edges(3, &[(2, 1), (1, 0), (0, 1)])
            .expect("canonical equivalent graph");
        let signal = MisInformerSignal::from_graph("case-graph-a", &graph, 20, vec![18, 1, 17])
            .expect("valid graph-bound signal");
        assert_eq!(
            signal
                .recommend_for_graph("case-graph-a", &equivalent, MisSelectionMode::PreferInSet)
                .expect("equivalent canonical graph"),
            MisVertexRecommendation {
                vertex: 0,
                inclusion_count: 18,
                sample_count: 20,
                tie_count: 1,
                mode: MisSelectionMode::PreferInSet,
            }
        );
    }

    #[test]
    fn computed_graph_identity_rejects_structurally_different_graph() {
        let graph = MisGraphIdentity::from_edges(3, &[(0, 1), (1, 2)]).expect("valid graph");
        let wrong_graph = MisGraphIdentity::from_edges(3, &[(0, 1), (0, 2)])
            .expect("other valid graph");
        let signal = MisInformerSignal::from_graph("case-graph-a", &graph, 20, vec![18, 1, 17])
            .expect("valid graph-bound signal");
        assert_eq!(
            signal
                .recommend_for_graph("case-graph-a", &wrong_graph, MisSelectionMode::PreferOutSet)
                .expect_err("graph content must match"),
            MisSignalError::GraphDigestMismatch
        );
    }

    #[test]
    fn all_vertices_tied_is_reported_as_tie_not_hidden_confidence() {
        let signal =
            MisInformerSignal::from_unverified_digest("mis-case-a", [1; 32], 3, 7, vec![4, 4, 4]).expect("valid signal");
        let result = signal
            .recommend_vertex_for_unverified_digest("mis-case-a", [1; 32], 3, MisSelectionMode::PreferInSet)
            .expect("matching graph identity");
        assert_eq!(result.vertex, 0);
        assert_eq!(result.tie_count, 3);
    }
}
