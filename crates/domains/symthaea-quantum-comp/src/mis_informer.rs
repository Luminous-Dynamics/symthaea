//! Validated, graph-bound inclusion-frequency signals for MIS reduction experiments.
//!
//! A signal describes how often each vertex appeared in a sample population. It
//! does not identify the process that generated the samples and makes no claim
//! about quantum hardware. Source provenance and evidence class belong in the
//! experiment manifest. An absent sample must never be confused with a measured
//! zero inclusion frequency.

use core::fmt;

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
    /// Construct a signal from per-vertex inclusion counts.
    ///
    /// The vector index is the vertex index. Every count must be at most the total
    /// sample count. At least one sample is required: an empty sample population
    /// is missing evidence, not evidence that every vertex has probability zero.
    pub fn new(
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

    /// Select an extreme-frequency vertex after checking the graph binding.
    ///
    /// The instance identifier, graph digest, and vertex count are supplied by
    /// the caller's validated graph manifest. Ties are resolved by the lowest index,
    /// making this recommendation deterministic. The caller remains responsible
    /// for recording signal provenance and verifying the final solution.
    pub fn recommend_vertex(
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
            MisInformerSignal::new("  ", [0; 32], 2, 1, vec![0, 0]).expect_err("blank id is invalid"),
            MisSignalError::InvalidInstanceId
        );
        assert_eq!(
            MisInformerSignal::new("graph\nwrong", [0; 32], 2, 1, vec![0, 0])
                .expect_err("control character is invalid"),
            MisSignalError::InvalidInstanceId
        );
    }

    #[test]
    fn rejects_missing_samples_instead_of_inventing_zero_frequency() {
        assert_eq!(
            MisInformerSignal::new("mis-case-a", [1; 32], 2, 0, vec![0, 0])
                .expect_err("no samples are not negative evidence"),
            MisSignalError::NoSamples
        );
    }

    #[test]
    fn rejects_vector_length_mismatch() {
        assert_eq!(
            MisInformerSignal::new("mis-case-a", [1; 32], 3, 4, vec![0, 1])
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
            MisInformerSignal::new("mis-case-a", [1; 32], 2, 4, vec![2, 5])
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
            MisInformerSignal::new("mis-case-a", [1; 32], 4, 10, vec![2, 8, 8, 4]).expect("valid signal");
        let result = signal
            .recommend_vertex("mis-case-a", [1; 32], 4, MisSelectionMode::PreferInSet)
            .expect("matching graph identity");
        assert_eq!(result.vertex, 1);
        assert_eq!(result.inclusion_count, 8);
        assert_eq!(result.sample_count, 10);
        assert_eq!(result.tie_count, 2);
    }

    #[test]
    fn recommend_out_set_picks_lowest_frequency_and_breaks_ties_by_index() {
        let signal =
            MisInformerSignal::new("mis-case-a", [1; 32], 4, 10, vec![0, 8, 0, 4]).expect("valid signal");
        let result = signal
            .recommend_vertex("mis-case-a", [1; 32], 4, MisSelectionMode::PreferOutSet)
            .expect("matching graph identity");
        assert_eq!(result.vertex, 0);
        assert_eq!(result.inclusion_count, 0);
        assert_eq!(result.tie_count, 2);
    }

    #[test]
    fn rejects_signal_from_wrong_instance() {
        let signal =
            MisInformerSignal::new("mis-case-a", [1; 32], 2, 5, vec![4, 1]).expect("valid signal");
        assert_eq!(
            signal
                .recommend_vertex("mis-case-b", [1; 32], 2, MisSelectionMode::PreferInSet)
                .expect_err("signal is stale or belongs to another graph"),
            MisSignalError::InstanceIdMismatch
        );
    }

    #[test]
    fn rejects_signal_from_different_graph_digest_even_if_instance_id_and_size_match() {
        let signal =
            MisInformerSignal::new("mis-case-a", [1; 32], 2, 5, vec![4, 1])
                .expect("valid signal");
        assert_eq!(
            signal
                .recommend_vertex("mis-case-a", [9; 32], 2, MisSelectionMode::PreferInSet)
                .expect_err("graph content digest differs"),
            MisSignalError::GraphDigestMismatch
        );
    }

    #[test]
    fn rejects_wrong_graph_size() {
        let signal =
            MisInformerSignal::new("mis-case-a", [1; 32], 3, 5, vec![4, 1, 2]).expect("valid signal");
        assert_eq!(
            signal
                .recommend_vertex("mis-case-a", [1; 32], 4, MisSelectionMode::PreferOutSet)
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
            MisInformerSignal::new("mis-case-a", [1; 32], 3, 100, vec![0, 1, 0]).expect("valid signal");
        let result = signal
            .recommend_vertex("mis-case-a", [1; 32], 3, MisSelectionMode::PreferOutSet)
            .expect("matching graph identity");
        assert_eq!(result.sample_count, 100);
        assert_eq!(result.inclusion_count, 0);
        assert_eq!(result.tie_count, 2);
    }

    #[test]
    fn all_vertices_tied_is_reported_as_tie_not_hidden_confidence() {
        let signal =
            MisInformerSignal::new("mis-case-a", [1; 32], 3, 7, vec![4, 4, 4]).expect("valid signal");
        let result = signal
            .recommend_vertex("mis-case-a", [1; 32], 3, MisSelectionMode::PreferInSet)
            .expect("matching graph identity");
        assert_eq!(result.vertex, 0);
        assert_eq!(result.tie_count, 3);
    }
}
