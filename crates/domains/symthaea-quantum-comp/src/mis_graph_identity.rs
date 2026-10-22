//! Canonical cryptographic identity for MIS graph instances.
//!
//! The fingerprint is computed from the graph structure, not supplied by a caller.
//! Vertex numbering is significant: callers must preserve the benchmark's frozen
//! vertex ordering when building an identity and interpreting sample counts.

use core::fmt;
use sha2::{Digest, Sha256};

const DOMAIN_SEPARATOR: &[u8] = b"symthaea:mis:canonical-simple-undirected-graph:v1\0";

/// A validated identity for a simple, undirected MIS graph.
///
/// Edges are endpoint-normalized, sorted, and deduplicated before hashing, so edge
/// iteration order, endpoint direction, and duplicate edge records do not change
/// the identity. Vertex count and vertex numbering are part of the hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisGraphIdentity {
    vertex_count: usize,
    graph_sha256: [u8; 32],
}

/// Failure to construct a canonical graph identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisGraphIdentityError {
    /// An edge references a vertex outside the declared graph.
    VertexOutOfRange { vertex: usize, vertex_count: usize },
    /// A simple graph cannot contain an edge from a vertex to itself.
    SelfLoop { vertex: usize },
    /// The vertex count cannot be represented by the canonical encoding.
    VertexCountTooLarge,
    /// The edge count cannot be represented by the canonical encoding.
    EdgeCountTooLarge,
}

impl fmt::Display for MisGraphIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexOutOfRange { vertex, vertex_count } => write!(
                f,
                "edge references vertex {vertex}, outside graph with {vertex_count} vertices"
            ),
            Self::SelfLoop { vertex } => write!(f, "self-loop at vertex {vertex} is not a simple MIS edge"),
            Self::VertexCountTooLarge => write!(f, "vertex count cannot be represented in canonical graph encoding"),
            Self::EdgeCountTooLarge => write!(f, "edge count cannot be represented in canonical graph encoding"),
        }
    }
}

impl std::error::Error for MisGraphIdentityError {}

impl MisGraphIdentity {
    /// Compute a domain-separated SHA-256 fingerprint from a graph edge list.
    ///
    /// The input is interpreted as a simple, undirected graph. Repeated edges and
    /// reversed duplicate edges are canonicalized away. Self-loops and out-of-range
    /// vertices are rejected instead of silently repairing malformed graph input.
    /// The caller must retain the same vertex-to-variable mapping used by its
    /// benchmark instance; the identity does not prove dataset provenance.
    pub fn from_edges(
        vertex_count: usize,
        edges: &[(usize, usize)],
    ) -> Result<Self, MisGraphIdentityError> {
        let vertex_count_encoded =
            u64::try_from(vertex_count).map_err(|_| MisGraphIdentityError::VertexCountTooLarge)?;

        let mut canonical_edges = Vec::with_capacity(edges.len());
        for &(left, right) in edges {
            if left >= vertex_count {
                return Err(MisGraphIdentityError::VertexOutOfRange {
                    vertex: left,
                    vertex_count,
                });
            }
            if right >= vertex_count {
                return Err(MisGraphIdentityError::VertexOutOfRange {
                    vertex: right,
                    vertex_count,
                });
            }
            if left == right {
                return Err(MisGraphIdentityError::SelfLoop { vertex: left });
            }
            canonical_edges.push(if left < right {
                (left, right)
            } else {
                (right, left)
            });
        }
        canonical_edges.sort_unstable();
        canonical_edges.dedup();

        let edge_count_encoded = u64::try_from(canonical_edges.len())
            .map_err(|_| MisGraphIdentityError::EdgeCountTooLarge)?;

        // Fixed-width little-endian integers and a versioned domain separator make
        // this encoding unambiguous and stable across host architectures.
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_SEPARATOR);
        hasher.update(vertex_count_encoded.to_le_bytes());
        hasher.update(edge_count_encoded.to_le_bytes());
        for (left, right) in canonical_edges {
            // These conversions are guaranteed by the checked vertex_count above
            // and the endpoint-range checks.
            let left = u64::try_from(left).map_err(|_| MisGraphIdentityError::VertexCountTooLarge)?;
            let right = u64::try_from(right).map_err(|_| MisGraphIdentityError::VertexCountTooLarge)?;
            hasher.update(left.to_le_bytes());
            hasher.update(right.to_le_bytes());
        }

        let digest: [u8; 32] = hasher.finalize().into();
        Ok(Self {
            vertex_count,
            graph_sha256: digest,
        })
    }

    /// Number of vertices included in this graph identity.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// SHA-256 fingerprint of the canonical graph encoding.
    pub fn sha256(&self) -> [u8; 32] {
        self.graph_sha256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_digest_ignores_edge_order_direction_and_duplicates() {
        let first = MisGraphIdentity::from_edges(4, &[(0, 1), (2, 1), (1, 0)]).expect("valid graph");
        let second = MisGraphIdentity::from_edges(4, &[(1, 2), (0, 1)]).expect("same simple graph");
        assert_eq!(first.sha256(), second.sha256());
    }

    #[test]
    fn digest_changes_when_graph_structure_changes() {
        let first = MisGraphIdentity::from_edges(4, &[(0, 1), (1, 2)]).expect("valid graph");
        let changed_edge = MisGraphIdentity::from_edges(4, &[(0, 1), (1, 3)]).expect("valid graph");
        let changed_size = MisGraphIdentity::from_edges(5, &[(0, 1), (1, 2)]).expect("valid graph");
        assert_ne!(first.sha256(), changed_edge.sha256());
        assert_ne!(first.sha256(), changed_size.sha256());
    }

    #[test]
    fn rejects_self_loops() {
        assert_eq!(
            MisGraphIdentity::from_edges(2, &[(1, 1)]).expect_err("self-loop is invalid"),
            MisGraphIdentityError::SelfLoop { vertex: 1 }
        );
    }

    #[test]
    fn rejects_out_of_range_vertices() {
        assert_eq!(
            MisGraphIdentity::from_edges(2, &[(0, 2)]).expect_err("endpoint is outside graph"),
            MisGraphIdentityError::VertexOutOfRange {
                vertex: 2,
                vertex_count: 2,
            }
        );
    }
}
