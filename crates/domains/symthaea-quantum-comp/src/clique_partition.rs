//! Independent verification of clique-partition upper bounds for unweighted MIS.
//!
//! A clique partition into k blocks proves alpha(G) <= k. This module verifies the
//! certificate against the exact canonical graph bytes and derives k from the blocks;
//! it never treats a caller-supplied bound or digest as authority.

use super::{canonical_mis_graph_vertex_count, sha256_hex};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const GRAPH_HEADER: &str = "SYMTHEA-MIS-GRAPH-V1";
const CERTIFICATE_HEADER: &str = "SYMTHEA-MIS-CLIQUE-PARTITION-V1";

/// Successful independent verification of a canonical clique-partition certificate.
/// Fields are private so callers cannot construct a certified upper bound from metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCliquePartition {
    graph_sha256: String,
    certificate_sha256: String,
    vertex_count: usize,
    block_count: usize,
}

impl VerifiedCliquePartition {
    /// SHA-256 recomputed from the exact canonical graph bytes.
    pub fn graph_sha256(&self) -> &str { &self.graph_sha256 }

    /// SHA-256 recomputed from the exact certificate bytes.
    pub fn certificate_sha256(&self) -> &str { &self.certificate_sha256 }

    /// Vertex count parsed from and validated against the canonical graph bytes.
    pub fn vertex_count(&self) -> usize { self.vertex_count }

    /// Certified MIS upper bound, derived from independently verified block count.
    pub fn upper_bound(&self) -> usize { self.block_count }

    /// Number of nonempty cliques in the verified partition.
    pub fn block_count(&self) -> usize { self.block_count }
}

/// Failure while parsing or independently checking a clique-partition certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliquePartitionError {
    /// The graph is invalid under the existing canonical MIS graph parser.
    InvalidGraph(String),
    /// Graph or certificate is not valid UTF-8.
    InvalidUtf8,
    /// Bytes do not use the required LF-only, final-newline encoding.
    NonCanonicalLineEndings,
    /// Certificate header is absent or unsupported.
    InvalidCertificateHeader,
    /// A count header is absent, malformed, or not canonical.
    InvalidCountLine(&'static str),
    /// A decimal token is empty, malformed, overflowing, or has leading zeroes.
    InvalidDecimal(&'static str),
    /// Certificate vertex count disagrees with the exact graph.
    GraphVertexCountMismatch { graph_vertices: usize, certificate_vertices: usize },
    /// The declared number of blocks disagrees with the encoded block lines.
    BlockCountMismatch { declared: usize, encoded: usize },
    /// A block line is malformed or has an inconsistent member count.
    InvalidBlockLine(usize),
    /// A block is empty; clique partitions require nonempty blocks.
    EmptyBlock(usize),
    /// A vertex identifier is outside the graph.
    VertexOutOfRange(usize),
    /// Vertex identifiers inside a block are not strictly increasing.
    VerticesNotStrictlySorted(usize),
    /// Blocks are not in strict lexicographic order.
    BlocksNotStrictlySorted(usize),
    /// A vertex appears in more than one block.
    DuplicateVertex(usize),
    /// The partition does not cover every graph vertex exactly once.
    MissingVertices { expected: usize, actual: usize },
    /// At least one pair in a proposed clique is not an edge of the exact graph.
    NonCliquePair { block: usize, u: usize, v: usize },
}

impl fmt::Display for CliquePartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(message) => write!(f, "invalid canonical MIS graph: {message}"),
            Self::InvalidUtf8 => write!(f, "graph or certificate is not UTF-8"),
            Self::NonCanonicalLineEndings => write!(f, "expected LF-only lines and a final newline"),
            Self::InvalidCertificateHeader => write!(f, "invalid clique-partition certificate header"),
            Self::InvalidCountLine(field) => write!(f, "invalid count line: {field}"),
            Self::InvalidDecimal(field) => write!(f, "invalid canonical decimal token: {field}"),
            Self::GraphVertexCountMismatch { graph_vertices, certificate_vertices } =>
                write!(f, "graph has {graph_vertices} vertices but certificate declares {certificate_vertices}"),
            Self::BlockCountMismatch { declared, encoded } =>
                write!(f, "certificate declares {declared} blocks but encodes {encoded}"),
            Self::InvalidBlockLine(line) => write!(f, "invalid block line at 1-based line {line}"),
            Self::EmptyBlock(block) => write!(f, "block {block} is empty"),
            Self::VertexOutOfRange(vertex) => write!(f, "vertex {vertex} is out of range"),
            Self::VerticesNotStrictlySorted(block) =>
                write!(f, "vertices in block {block} are not strictly sorted"),
            Self::BlocksNotStrictlySorted(block) =>
                write!(f, "block {block} is not in canonical lexicographic order"),
            Self::DuplicateVertex(vertex) => write!(f, "vertex {vertex} appears more than once"),
            Self::MissingVertices { expected, actual } =>
                write!(f, "partition covers {actual} distinct vertices; expected {expected}"),
            Self::NonCliquePair { block, u, v } =>
                write!(f, "vertices {u} and {v} in block {block} are not adjacent"),
        }
    }
}

impl Error for CliquePartitionError {}

fn parse_decimal(value: &str, field: &'static str) -> Result<usize, CliquePartitionError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CliquePartitionError::InvalidDecimal(field));
    }
    let parsed = value.parse::<usize>().map_err(|_| CliquePartitionError::InvalidDecimal(field))?;
    if parsed.to_string() != value {
        return Err(CliquePartitionError::InvalidDecimal(field));
    }
    Ok(parsed)
}

fn parse_count_line(line: &str, prefix: &'static str) -> Result<usize, CliquePartitionError> {
    let value = line.strip_prefix(prefix).ok_or(CliquePartitionError::InvalidCountLine(prefix))?;
    if value.contains(' ') {
        return Err(CliquePartitionError::InvalidCountLine(prefix));
    }
    parse_decimal(value, prefix)
}

fn parse_graph_edges(graph_bytes: &[u8]) -> Result<(usize, HashSet<(usize, usize)>), CliquePartitionError> {
    let vertex_count = canonical_mis_graph_vertex_count(graph_bytes)
        .map_err(|error| CliquePartitionError::InvalidGraph(error.to_string()))?;
    let text = std::str::from_utf8(graph_bytes).map_err(|_| CliquePartitionError::InvalidUtf8)?;
    // canonical_mis_graph_vertex_count already validates header, counts, endpoints,
    // strict edge order, line endings, and the exact canonical graph grammar.
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    if lines.first().copied() != Some(GRAPH_HEADER) {
        return Err(CliquePartitionError::InvalidGraph("unexpected graph header".to_owned()));
    }
    let mut edges = HashSet::new();
    for line in &lines[3..] {
        let mut parts = line.split(' ');
        let u = parts.next().ok_or_else(|| CliquePartitionError::InvalidGraph("missing edge endpoint".to_owned()))?
            .parse::<usize>().map_err(|_| CliquePartitionError::InvalidGraph("invalid edge endpoint".to_owned()))?;
        let v = parts.next().ok_or_else(|| CliquePartitionError::InvalidGraph("missing edge endpoint".to_owned()))?
            .parse::<usize>().map_err(|_| CliquePartitionError::InvalidGraph("invalid edge endpoint".to_owned()))?;
        if parts.next().is_some() {
            return Err(CliquePartitionError::InvalidGraph("invalid edge line".to_owned()));
        }
        edges.insert((u, v));
    }
    Ok((vertex_count, edges))
}

/// Verify a canonical clique partition against exact canonical graph bytes.
///
/// Certificate format:
///
/// SYMTHEA-MIS-CLIQUE-PARTITION-V1
/// vertices N
/// blocks K
/// block COUNT V0 V1 ...
///
/// Each block must be nonempty; its vertices are strictly increasing; blocks are
/// strictly lexicographically ordered; and the blocks form a disjoint partition
/// of every vertex. Every pair inside a block must be an edge in the graph.
/// The returned upper bound is derived from the verified block count.
pub fn verify_clique_partition_certificate(
    graph_bytes: &[u8],
    certificate_bytes: &[u8],
) -> Result<VerifiedCliquePartition, CliquePartitionError> {
    let (graph_vertices, edges) = parse_graph_edges(graph_bytes)?;
    let text = std::str::from_utf8(certificate_bytes).map_err(|_| CliquePartitionError::InvalidUtf8)?;
    if !text.ends_with('\n') || text.contains('\r') {
        return Err(CliquePartitionError::NonCanonicalLineEndings);
    }
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    if lines.len() < 3 || lines[0] != CERTIFICATE_HEADER {
        return Err(CliquePartitionError::InvalidCertificateHeader);
    }
    let certificate_vertices = parse_count_line(lines[1], "vertices ")?;
    if graph_vertices != certificate_vertices {
        return Err(CliquePartitionError::GraphVertexCountMismatch {
            graph_vertices,
            certificate_vertices,
        });
    }
    let declared_blocks = parse_count_line(lines[2], "blocks ")?;
    let encoded_blocks = lines.len() - 3;
    if declared_blocks != encoded_blocks {
        return Err(CliquePartitionError::BlockCountMismatch {
            declared: declared_blocks,
            encoded: encoded_blocks,
        });
    }

    let mut covered = HashSet::new();
    let mut previous_block: Option<Vec<usize>> = None;
    for (block_index, line) in lines[3..].iter().enumerate() {
        let tokens: Vec<&str> = line.split(' ').collect();
        if tokens.len() < 3 || tokens[0] != "block" || tokens.iter().any(|token| token.is_empty()) {
            return Err(CliquePartitionError::InvalidBlockLine(block_index + 4));
        }
        let member_count = parse_decimal(tokens[1], "block member count")?;
        if member_count == 0 {
            return Err(CliquePartitionError::EmptyBlock(block_index));
        }
        if tokens.len() != member_count.saturating_add(2) {
            return Err(CliquePartitionError::InvalidBlockLine(block_index + 4));
        }
        let mut block = Vec::with_capacity(member_count);
        let mut previous_vertex = None;
        for token in &tokens[2..] {
            let vertex = parse_decimal(token, "vertex identifier")?;
            if vertex >= graph_vertices {
                return Err(CliquePartitionError::VertexOutOfRange(vertex));
            }
            if previous_vertex.is_some_and(|previous| previous >= vertex) {
                return Err(CliquePartitionError::VerticesNotStrictlySorted(block_index));
            }
            if !covered.insert(vertex) {
                return Err(CliquePartitionError::DuplicateVertex(vertex));
            }
            previous_vertex = Some(vertex);
            block.push(vertex);
        }
        if previous_block.as_ref().is_some_and(|previous| previous.as_slice() >= block.as_slice()) {
            return Err(CliquePartitionError::BlocksNotStrictlySorted(block_index));
        }

        for left in 0..block.len() {
            for right in (left + 1)..block.len() {
                if !edges.contains(&(block[left], block[right])) {
                    return Err(CliquePartitionError::NonCliquePair {
                        block: block_index,
                        u: block[left],
                        v: block[right],
                    });
                }
            }
        }
        previous_block = Some(block);
    }

    if covered.len() != graph_vertices {
        return Err(CliquePartitionError::MissingVertices {
            expected: graph_vertices,
            actual: covered.len(),
        });
    }

    Ok(VerifiedCliquePartition {
        graph_sha256: sha256_hex(graph_bytes),
        certificate_sha256: sha256_hex(certificate_bytes),
        vertex_count: graph_vertices,
        block_count: declared_blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAPH: &[u8] = b"SYMTHEA-MIS-GRAPH-V1\nvertices 6\nedges 3\n0 1\n2 3\n4 5\n";

    fn certificate(blocks: &[&[usize]]) -> Vec<u8> {
        let mut text = format!(
            "{CERTIFICATE_HEADER}\nvertices 6\nblocks {}\n",
            blocks.len()
        );
        for block in blocks {
            text.push_str(&format!("block {}", block.len()));
            for vertex in *block {
                text.push_str(&format!(" {vertex}"));
            }
            text.push('\n');
        }
        text.into_bytes()
    }

    #[test]
    fn verifies_nontrivial_partition_and_derives_bound_from_block_count() {
        let bytes = certificate(&[&[0, 1], &[2, 3], &[4, 5]]);
        let verified = verify_clique_partition_certificate(GRAPH, &bytes).unwrap();
        assert_eq!(verified.vertex_count(), 6);
        assert_eq!(verified.block_count(), 3);
        assert_eq!(verified.upper_bound(), 3);
        assert_eq!(verified.graph_sha256(), sha256_hex(GRAPH));
        assert_eq!(verified.certificate_sha256(), sha256_hex(&bytes));
    }

    #[test]
    fn singleton_partition_is_the_trivial_vertex_count_bound() {
        let bytes = certificate(&[&[0], &[1], &[2], &[3], &[4], &[5]]);
        let verified = verify_clique_partition_certificate(GRAPH, &bytes).unwrap();
        assert_eq!(verified.upper_bound(), 6);
    }

    #[test]
    fn rejects_missing_duplicate_out_of_range_and_nonclique_vertices() {
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, &certificate(&[&[0, 1], &[2, 3], &[4]])),
            Err(CliquePartitionError::MissingVertices { .. })
        ));
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, &certificate(&[&[0, 1], &[2, 3], &[4], &[4, 5]])),
            Err(CliquePartitionError::DuplicateVertex(4))
        ));
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, &certificate(&[&[0, 1], &[2, 3], &[4, 6]])),
            Err(CliquePartitionError::VertexOutOfRange(6))
        ));
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, &certificate(&[&[0, 2], &[1, 3], &[4, 5]])),
            Err(CliquePartitionError::NonCliquePair { .. })
        ));
    }

    #[test]
    fn rejects_noncanonical_order_counts_and_line_endings() {
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, &certificate(&[&[2, 3], &[0, 1], &[4, 5]])),
            Err(CliquePartitionError::BlocksNotStrictlySorted(_))
        ));
        let mut bad_count = certificate(&[&[0, 1], &[2, 3], &[4, 5]]);
        let last = bad_count.iter().position(|byte| *byte == b'3').unwrap();
        bad_count[last] = b'4';
        assert!(verify_clique_partition_certificate(GRAPH, &bad_count).is_err());
        let mut crlf = certificate(&[&[0, 1], &[2, 3], &[4, 5]]);
        crlf = String::from_utf8(crlf).unwrap().replace('\n', "\r\n").into_bytes();
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, &crlf),
            Err(CliquePartitionError::NonCanonicalLineEndings)
        ));
        let leading_zero = format!("{CERTIFICATE_HEADER}\nvertices 06\nblocks 1\nblock 6 0 1 2 3 4 5\n");
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, leading_zero.as_bytes()),
            Err(CliquePartitionError::InvalidDecimal("vertices "))
        ));
    }

    #[test]
    fn rejects_empty_blocks_and_unsorted_vertices() {
        let empty = format!("{CERTIFICATE_HEADER}\nvertices 6\nblocks 1\nblock 0\n");
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, empty.as_bytes()),
            Err(CliquePartitionError::InvalidBlockLine(_)) | Err(CliquePartitionError::EmptyBlock(_))
        ));
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, &certificate(&[&[1, 0], &[2, 3], &[4, 5]])),
            Err(CliquePartitionError::VerticesNotStrictlySorted(0))
        ));
    }

    #[test]
    fn rejects_graph_and_certificate_header_mismatch() {
        let cert = certificate(&[&[0, 1], &[2, 3], &[4, 5]]);
        assert!(matches!(
            verify_clique_partition_certificate(
                b"SYMTHEA-MIS-GRAPH-V1\nvertices 6\nedges 3\n0 1\n2 4\n4 5\n",
                &cert
            ),
            Err(CliquePartitionError::NonCliquePair { .. })
                | Err(CliquePartitionError::InvalidGraph(_))
        ));
        let wrong_graph_size = format!("{CERTIFICATE_HEADER}\nvertices 5\nblocks 1\nblock 5 0 1 2 3 4\n");
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, wrong_graph_size.as_bytes()),
            Err(CliquePartitionError::GraphVertexCountMismatch { .. })
        ));
        assert!(matches!(
            verify_clique_partition_certificate(GRAPH, b"BAD-HEADER\nvertices 6\nblocks 0\n"),
            Err(CliquePartitionError::InvalidCertificateHeader)
        ));
    }

    fn canonical_graph(n: usize, edges: &[(usize, usize)]) -> Vec<u8> {
        let mut graph = format!("SYMTHEA-MIS-GRAPH-V1\nvertices {n}\nedges {}\n", edges.len());
        for &(u, v) in edges {
            graph.push_str(&format!("{u} {v}\n"));
        }
        graph.into_bytes()
    }

    fn all_partitions(n: usize) -> Vec<Vec<Vec<usize>>> {
        fn visit(
            next: usize,
            n: usize,
            blocks: &mut Vec<Vec<usize>>,
            out: &mut Vec<Vec<Vec<usize>>>,
        ) {
            if next == n {
                out.push(blocks.clone());
                return;
            }
            for index in 0..blocks.len() {
                blocks[index].push(next);
                visit(next + 1, n, blocks, out);
                blocks[index].pop();
            }
            blocks.push(vec![next]);
            visit(next + 1, n, blocks, out);
            blocks.pop();
        }
        let mut out = Vec::new();
        visit(0, n, &mut Vec::new(), &mut out);
        out
    }

    fn serialize_partition(n: usize, blocks: &[Vec<usize>]) -> Vec<u8> {
        let mut text = format!("{CERTIFICATE_HEADER}\nvertices {n}\nblocks {}\n", blocks.len());
        for block in blocks {
            text.push_str(&format!("block {}", block.len()));
            for vertex in block {
                text.push_str(&format!(" {vertex}"));
            }
            text.push('\n');
        }
        text.into_bytes()
    }

    fn exact_mis_size(n: usize, edges: &HashSet<(usize, usize)>) -> usize {
        let mut best = 0;
        for mask in 0usize..(1usize << n) {
            let selected: Vec<usize> = (0..n).filter(|vertex| (mask & (1 << vertex)) != 0).collect();
            let feasible = selected.iter().enumerate().all(|(i, &u)| {
                selected.iter().skip(i + 1).all(|&v| !edges.contains(&(u.min(v), u.max(v))))
            });
            if feasible {
                best = best.max(selected.len());
            }
        }
        best
    }

    #[test]
    fn exhaustive_partitions_through_four_vertices_never_underestimate_exact_mis() {
        for n in 1usize..=4 {
            let possible_edges: Vec<(usize, usize)> = (0..n)
                .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
                .collect();
            let partitions = all_partitions(n);
            for edge_mask in 0usize..(1usize << possible_edges.len()) {
                let edges: Vec<(usize, usize)> = possible_edges.iter().enumerate()
                    .filter_map(|(bit, &edge)| ((edge_mask >> bit) & 1 == 1).then_some(edge))
                    .collect();
                let graph = canonical_graph(n, &edges);
                let edge_set: HashSet<(usize, usize)> = edges.iter().copied().collect();
                let alpha = exact_mis_size(n, &edge_set);
                for blocks in &partitions {
                    let expected_valid = blocks.iter().all(|block| {
                        block.iter().enumerate().all(|(i, &u)| {
                            block.iter().skip(i + 1).all(|&v| edge_set.contains(&(u.min(v), u.max(v))))
                        })
                    });
                    let cert = serialize_partition(n, blocks);
                    match verify_clique_partition_certificate(&graph, &cert) {
                        Ok(verified) => {
                            assert!(expected_valid, "accepted non-clique partition: n={n}, edges={edges:?}, blocks={blocks:?}");
                            assert!(verified.upper_bound() >= alpha, "understated bound: n={n}, edges={edges:?}, blocks={blocks:?}, alpha={alpha}, U={}", verified.upper_bound());
                        }
                        Err(error) => {
                            assert!(!expected_valid, "rejected valid partition: n={n}, edges={edges:?}, blocks={blocks:?}, error={error}");
                        }
                    }
                }
            }
        }
    }
}
