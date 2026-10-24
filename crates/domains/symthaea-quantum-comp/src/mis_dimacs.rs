//! Strict, dependency-free parser for the DIMACS edge format used by QOBLIB MIS.
//!
//! The parser preserves one-based source IDs and converts them to the zero-based
//! indices used by MisGraph. Malformed records, duplicate edges, self-loops, invalid
//! endpoints and declared/observed edge-count mismatches fail closed.

use core::fmt;
use crate::mis_verifier::{MisGraph, MisGraphError};

/// A parsed DIMACS stable-set instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedMisDimacs {
    /// Canonical zero-based graph used by the independent verifier.
    pub graph: MisGraph,
    /// Vertex count declared by the source problem line.
    pub declared_vertex_count: usize,
    /// Edge count declared by the source problem line.
    pub declared_edge_count: usize,
    /// Number of edge records present in the source file.
    pub observed_edge_records: usize,
}

/// Strict DIMACS instance or QOBLIB solution-format error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisDimacsError {
    /// No valid problem line appeared in the file.
    MissingProblemLine,
    /// A second problem line was found.
    DuplicateProblemLine { line: usize },
    /// The problem line was not exactly p edge vertices edges.
    MalformedProblemLine { line: usize },
    /// A DIMACS problem type other than edge was encountered.
    UnsupportedProblemType { line: usize, problem_type: String },
    /// An edge record appeared before the problem line.
    EdgeBeforeProblemLine { line: usize },
    /// An edge line was not exactly e u v.
    MalformedEdgeLine { line: usize },
    /// A required integer token did not parse as usize.
    InvalidInteger { line: usize, field: &'static str },
    /// An unrecognized non-comment record was found.
    UnexpectedRecord { line: usize, record: String },
    /// A source edge referred to a vertex outside the declared one-based range.
    VertexOutOfRange { line: usize, vertex: usize, vertex_count: usize },
    /// The source graph contained a self-loop.
    SelfLoop { line: usize, vertex: usize },
    /// An undirected edge appeared more than once, including reversed duplicates.
    DuplicateEdge { line: usize, u: usize, v: usize },
    /// The number of edge records differed from the declaration.
    DeclaredEdgeCountMismatch { declared: usize, observed: usize },
    /// Source text exceeded the configured byte limit.
    InputByteLimitExceeded { actual: usize, maximum: usize },
    /// Declared vertex count exceeded the configured allocation limit.
    VertexLimitExceeded { declared: usize, maximum: usize },
    /// Declared edge count exceeded the configured storage limit.
    EdgeLimitExceeded { declared: usize, maximum: usize },
    /// Observed edge records exceeded the configured storage limit.
    EdgeRecordLimitExceeded { line: usize, maximum: usize },
    /// Construction of the canonical graph failed.
    GraphConstruction(MisGraphError),
    /// A solution contained a token other than a vertex index.
    InvalidSolutionToken { line: usize, token: String },
    /// Solution vertex zero is invalid in the one-based QOBLIB format.
    ZeroSolutionVertex { line: usize },
    /// A solution vertex exceeded the graph's one-based range.
    SolutionVertexOutOfRange { line: usize, vertex: usize, vertex_count: usize },
    /// The solution had no vertex indices; the upstream QOBLIB checker rejects this.
    EmptySolution,
}

impl fmt::Display for MisDimacsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProblemLine => write!(f, "missing DIMACS problem line"),
            Self::DuplicateProblemLine { line } => write!(f, "duplicate problem line at line {line}"),
            Self::MalformedProblemLine { line } => write!(f, "malformed problem line at line {line}; expected p edge vertices edges"),
            Self::UnsupportedProblemType { line, problem_type } => write!(f, "unsupported problem type {problem_type:?} at line {line}"),
            Self::EdgeBeforeProblemLine { line } => write!(f, "edge record precedes problem line at line {line}"),
            Self::MalformedEdgeLine { line } => write!(f, "malformed edge line at line {line}; expected e u v"),
            Self::InvalidInteger { line, field } => write!(f, "invalid integer for {field} at line {line}"),
            Self::UnexpectedRecord { line, record } => write!(f, "unexpected DIMACS record {record:?} at line {line}"),
            Self::VertexOutOfRange { line, vertex, vertex_count } => write!(f, "vertex {vertex} at line {line} is outside 1..={vertex_count}"),
            Self::SelfLoop { line, vertex } => write!(f, "self-loop at vertex {vertex} on line {line}"),
            Self::DuplicateEdge { line, u, v } => write!(f, "duplicate undirected edge ({u}, {v}) on line {line}"),
            Self::DeclaredEdgeCountMismatch { declared, observed } => write!(f, "problem line declares {declared} edges but file contains {observed} edge records"),
            Self::InputByteLimitExceeded { actual, maximum } => write!(f, "input has {actual} bytes, exceeding the configured {maximum}-byte limit"),
            Self::VertexLimitExceeded { declared, maximum } => write!(f, "problem line declares {declared} vertices, exceeding the configured limit of {maximum}"),
            Self::EdgeLimitExceeded { declared, maximum } => write!(f, "problem line declares {declared} edges, exceeding the configured limit of {maximum}"),
            Self::EdgeRecordLimitExceeded { line, maximum } => write!(f, "edge record at line {line} exceeds the configured limit of {maximum}"),
            Self::GraphConstruction(error) => write!(f, "failed to construct graph: {error}"),
            Self::InvalidSolutionToken { line, token } => write!(f, "invalid solution token {token:?} on line {line}"),
            Self::ZeroSolutionVertex { line } => write!(f, "solution vertex zero is invalid at line {line}"),
            Self::SolutionVertexOutOfRange { line, vertex, vertex_count } => write!(f, "solution vertex {vertex} at line {line} is outside 1..={vertex_count}"),
            Self::DuplicateSolutionVertex { line, vertex } => write!(f, "solution repeats vertex {vertex} on line {line}"),
            Self::EmptySolution => write!(f, "solution has no vertex indices; QOBLIB rejects empty solution files"),
        }
    }
}

impl std::error::Error for MisDimacsError {}

/// Parse a DIMACS edge-format graph used by QOBLIB's stable-set/MIS class.
///
/// Supported records are blank lines, comment lines beginning with c, exactly one
/// problem line p edge n m, and exactly m edge lines e u v. Nodes are one-based.
/// Repeated undirected edges, self-loops, and out-of-range nodes are rejected.
pub fn parse_dimacs_mis(input: &str) -> Result<ParsedMisDimacs, MisDimacsError> {
    parse_dimacs_mis_with_limits(input, MisDimacsLimits::default())
}

/// Resource limits for parsing untrusted or externally sourced DIMACS graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisDimacsLimits {
    /// Maximum UTF-8 source size in bytes. Default: 64 MiB, enough for the pinned QOBLIB C4000-5 file.
    pub max_input_bytes: usize,
    /// Maximum declared/allocated vertex count. Default: 100,000.
    pub max_vertices: usize,
    /// Maximum declared/stored edge count. Default: 5,000,000.
    pub max_edges: usize,
}

impl Default for MisDimacsLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 64 * 1024 * 1024,
            max_vertices: 100_000,
            max_edges: 5_000_000,
        }
    }
}

/// Parse DIMACS with explicit resource limits. The function rejects input sizes and
/// graph declarations before allocation, and stops if observed edge records exceed
/// the configured edge budget. Raise limits only when the execution environment is
/// provisioned for the corresponding memory footprint.
pub fn parse_dimacs_mis_with_limits(
    input: &str,
    limits: MisDimacsLimits,
) -> Result<ParsedMisDimacs, MisDimacsError> {
    if input.len() > limits.max_input_bytes {
        return Err(MisDimacsError::InputByteLimitExceeded {
            actual: input.len(),
            maximum: limits.max_input_bytes,
        });
    }
    let mut problem: Option<(usize, usize)> = None;
    // Keep source line with each canonical edge so duplicates can be rejected after
    // one sort. This avoids a HashSet allocation proportional to every input edge,
    // important for the dense 45 MB C4000-5 QOBLIB source instance.
    let mut source_edges: Vec<(usize, usize, usize)> = Vec::new();

    for (zero_line, raw_line) in input.lines().enumerate() {
        let line = zero_line + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.first() == Some(&"c") {
            continue;
        }

        match fields[0] {
            "p" => {
                if problem.is_some() {
                    return Err(MisDimacsError::DuplicateProblemLine { line });
                }
                if fields.len() != 4 {
                    return Err(MisDimacsError::MalformedProblemLine { line });
                }
                if fields[1] != "edge" {
                    return Err(MisDimacsError::UnsupportedProblemType {
                        line,
                        problem_type: fields[1].to_owned(),
                    });
                }
                let vertex_count = fields[2].parse::<usize>()
                    .map_err(|_| MisDimacsError::InvalidInteger { line, field: "vertex_count" })?;
                let edge_count = fields[3].parse::<usize>()
                    .map_err(|_| MisDimacsError::InvalidInteger { line, field: "edge_count" })?;
                if vertex_count > limits.max_vertices {
                    return Err(MisDimacsError::VertexLimitExceeded {
                        declared: vertex_count,
                        maximum: limits.max_vertices,
                    });
                }
                if edge_count > limits.max_edges {
                    return Err(MisDimacsError::EdgeLimitExceeded {
                        declared: edge_count,
                        maximum: limits.max_edges,
                    });
                }
                problem = Some((vertex_count, edge_count));
            }
            "e" => {
                let (vertex_count, _) =
                    problem.ok_or(MisDimacsError::EdgeBeforeProblemLine { line })?;
                if fields.len() != 3 {
                    return Err(MisDimacsError::MalformedEdgeLine { line });
                }
                let u = fields[1].parse::<usize>()
                    .map_err(|_| MisDimacsError::InvalidInteger { line, field: "edge_endpoint_u" })?;
                let v = fields[2].parse::<usize>()
                    .map_err(|_| MisDimacsError::InvalidInteger { line, field: "edge_endpoint_v" })?;
                for vertex in [u, v] {
                    if vertex == 0 || vertex > vertex_count {
                        return Err(MisDimacsError::VertexOutOfRange { line, vertex, vertex_count });
                    }
                }
                if u == v {
                    return Err(MisDimacsError::SelfLoop { line, vertex: u });
                }
                if source_edges.len() >= limits.max_edges {
                    return Err(MisDimacsError::EdgeRecordLimitExceeded {
                        line,
                        maximum: limits.max_edges,
                    });
                }
                let canonical = if u < v { (u - 1, v - 1) } else { (v - 1, u - 1) };
                source_edges.push((canonical.0, canonical.1, line));
            }
            record => return Err(MisDimacsError::UnexpectedRecord { line, record: record.to_owned() }),
        }
    }

    let (vertex_count, declared_edge_count) =
        problem.ok_or(MisDimacsError::MissingProblemLine)?;
    if source_edges.len() != declared_edge_count {
        return Err(MisDimacsError::DeclaredEdgeCountMismatch {
            declared: declared_edge_count,
            observed: source_edges.len(),
        });
    }

    // Sort once and detect a reversed/repeated undirected edge by checking adjacent
    // canonical pairs. Avoid a large per-edge hash table on dense QOBLIB inputs.
    source_edges.sort_unstable_by_key(|&(u, v, line)| (u, v, line));
    for pair in source_edges.windows(2) {
        let (u, v, first_line) = pair[0];
        let (next_u, next_v, next_line) = pair[1];
        if u == next_u && v == next_v {
            return Err(MisDimacsError::DuplicateEdge {
                line: first_line.max(next_line),
                u: u + 1,
                v: v + 1,
            });
        }
    }

    let observed_edge_records = source_edges.len();
    let graph = MisGraph::new(
        vertex_count,
        source_edges.into_iter().map(|(u, v, _line)| (u, v)),
    ).map_err(MisDimacsError::GraphConstruction)?;
    Ok(ParsedMisDimacs {
        graph,
        declared_vertex_count: vertex_count,
        declared_edge_count,
        observed_edge_records,
    })
}

/// Parse a QOBLIB-style solution index list into zero-based graph vertex indices.
///
/// Whole lines beginning with # are comments. Other whitespace-separated tokens must
/// be unique one-based vertices in 1..=vertex_count. The published QOBLIB checker
/// rejects solution files containing no indices; this parser conforms to that rule.
/// Objective comments are not trusted: count vertices and call verify_solution.
pub fn parse_qoblib_solution(
    input: &str,
    vertex_count: usize,
) -> Result<Vec<usize>, MisDimacsError> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    for (zero_line, raw_line) in input.lines().enumerate() {
        let line = zero_line + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        for token in trimmed.split_whitespace() {
            let vertex = token.parse::<usize>().map_err(|_| {
                MisDimacsError::InvalidSolutionToken { line, token: token.to_owned() }
            })?;
            if vertex == 0 {
                return Err(MisDimacsError::ZeroSolutionVertex { line });
            }
            if vertex > vertex_count {
                return Err(MisDimacsError::SolutionVertexOutOfRange { line, vertex, vertex_count });
            }
            if !seen.insert(vertex) {
                return Err(MisDimacsError::DuplicateSolutionVertex { line, vertex });
            }
            selected.push(vertex - 1);
        }
    }
    if selected.is_empty() {
        return Err(MisDimacsError::EmptySolution);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mis_verifier::MisViolation;

    #[test]
    fn parses_comments_whitespace_and_one_based_edges() {
        let parsed = parse_dimacs_mis("c sample graph\n\n  p edge 4 2\n e 1 4\n e 2 3\n").unwrap();
        assert_eq!(parsed.declared_vertex_count, 4);
        assert_eq!(parsed.declared_edge_count, 2);
        assert_eq!(parsed.observed_edge_records, 2);
        assert_eq!(parsed.graph.edges(), &[(0, 3), (1, 2)]);
    }

    #[test]
    fn preserves_isolated_vertices_and_edgeless_graphs() {
        let parsed = parse_dimacs_mis("p edge 3 0\n").unwrap();
        assert_eq!(parsed.graph.vertex_count(), 3);
        assert_eq!(parsed.graph.edges(), &[]);
        assert_eq!(parsed.graph.exact_maximum_independent_set().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn rejects_missing_or_duplicate_problem_lines() {
        assert_eq!(parse_dimacs_mis("c only a comment\n"), Err(MisDimacsError::MissingProblemLine));
        assert_eq!(parse_dimacs_mis("p edge 1 0\np edge 1 0\n"), Err(MisDimacsError::DuplicateProblemLine { line: 2 }));
    }

    #[test]
    fn rejects_edge_before_header_and_malformed_records() {
        assert_eq!(parse_dimacs_mis("e 1 2\np edge 2 1\n"), Err(MisDimacsError::EdgeBeforeProblemLine { line: 1 }));
        assert_eq!(parse_dimacs_mis("p edge 2\n"), Err(MisDimacsError::MalformedProblemLine { line: 1 }));
        assert_eq!(parse_dimacs_mis("p edge 2 1\ne 1 2 3\n"), Err(MisDimacsError::MalformedEdgeLine { line: 2 }));
    }

    #[test]
    fn rejects_out_of_range_self_loop_and_duplicate_edges() {
        assert_eq!(parse_dimacs_mis("p edge 2 1\ne 0 2\n"), Err(MisDimacsError::VertexOutOfRange { line: 2, vertex: 0, vertex_count: 2 }));
        assert_eq!(parse_dimacs_mis("p edge 2 1\ne 2 2\n"), Err(MisDimacsError::SelfLoop { line: 2, vertex: 2 }));
        assert_eq!(parse_dimacs_mis("p edge 2 2\ne 1 2\ne 2 1\n"), Err(MisDimacsError::DuplicateEdge { line: 3, u: 1, v: 2 }));
    }

    #[test]
    fn enforces_declared_edge_count() {
        assert_eq!(parse_dimacs_mis("p edge 3 2\ne 1 2\n"), Err(MisDimacsError::DeclaredEdgeCountMismatch { declared: 2, observed: 1 }));
    }

    #[test]
    fn parses_and_independently_verifies_qoblib_solution() {
        let parsed = parse_dimacs_mis("p edge 3 1\ne 1 2\n").unwrap();
        let selected = parse_qoblib_solution("# Objective value = 2\n1\n3\n", 3).unwrap();
        let report = parsed.graph.verify_solution(&selected, Some(selected.len()));
        assert!(report.feasible);
        assert_eq!(report.computed_objective, 2);
    }

    #[test]
    fn rejects_invalid_qoblib_solutions() {
        assert_eq!(parse_qoblib_solution("# no vertices\n", 3), Err(MisDimacsError::EmptySolution));
        assert_eq!(parse_qoblib_solution("0\n", 3), Err(MisDimacsError::ZeroSolutionVertex { line: 1 }));
        assert_eq!(parse_qoblib_solution("4\n", 3), Err(MisDimacsError::SolutionVertexOutOfRange { line: 1, vertex: 4, vertex_count: 3 }));
        assert_eq!(parse_qoblib_solution("1\n1\n", 3), Err(MisDimacsError::DuplicateSolutionVertex { line: 2, vertex: 1 }));
        assert_eq!(parse_qoblib_solution("1\nx#1 0\n", 3), Err(MisDimacsError::InvalidSolutionToken { line: 2, token: "x#1".to_owned() }));
    }

    #[test]
    fn parsed_infeasible_candidate_is_not_silently_accepted() {
        let parsed = parse_dimacs_mis("p edge 2 1\ne 1 2\n").unwrap();
        let selected = parse_qoblib_solution("1\n2\n", 2).unwrap();
        let report = parsed.graph.verify_solution(&selected, Some(2));
        assert!(!report.feasible);
        assert!(report.violations.contains(&MisViolation::AdjacentSelectedVertices { u: 0, v: 1 }));
    }

    #[test]
    fn pinned_qoblib_farm_fixture_matches_published_best_known_value() {
        // QOBLIB commit d61f234b9fc816deb3d4e1dea4c031595b35acd7,
        // 07-independentset/instances/farm.gph (Git blob ae103bba88413c1a8eafdeddd07ba5fe69fddc8d).
        // Its ParallelILS submission reports objective 10 on 10/10 trials; the record
        // is best-known, not a proof by itself. The bounded exact oracle checks this
        // 17-vertex source fixture independently.
        let source = "c Undirected Graph\n\
p edge 17 39\n\
e 7 17\n\
e 6 4\n\
e 7 5\n\
e 1 6\n\
e 2 6\n\
e 3 6\n\
e 1 7\n\
e 2 7\n\
e 3 7\n\
e 4 7\n\
e 1 8\n\
e 2 8\n\
e 1 9\n\
e 2 9\n\
e 4 9\n\
e 1 10\n\
e 2 10\n\
e 5 10\n\
e 1 11\n\
e 2 11\n\
e 4 11\n\
e 1 12\n\
e 2 12\n\
e 3 12\n\
e 6 12\n\
e 1 13\n\
e 2 13\n\
e 4 13\n\
e 6 13\n\
e 1 14\n\
e 2 14\n\
e 3 14\n\
e 7 14\n\
e 1 15\n\
e 2 15\n\
e 4 15\n\
e 7 15\n\
e 3 16\n\
e 4 17\n";
        let parsed = parse_dimacs_mis(source).expect("pinned QOBLIB source fixture parses");
        assert_eq!(parsed.declared_vertex_count, 17);
        assert_eq!(parsed.declared_edge_count, 39);
        assert_eq!(parsed.observed_edge_records, 39);
        assert_eq!(parsed.graph.edges().len(), 39);

        let exact = parsed
            .graph
            .exact_maximum_independent_set()
            .expect("17-vertex graph is within the exact oracle limit");
        assert_eq!(exact.len(), 10, "exact optimum should match QOBLIB best-known objective");
        assert!(parsed.graph.verify_solution(&exact, Some(10)).feasible);
    }


    #[test]
    fn parser_limits_reject_input_and_graph_declarations_before_allocation() {
        assert_eq!(
            parse_dimacs_mis_with_limits(
                "p edge 2 1\\ne 1 2\\n",
                MisDimacsLimits { max_input_bytes: 4, ..MisDimacsLimits::default() }
            ),
            Err(MisDimacsError::InputByteLimitExceeded {
                actual: 17,
                maximum: 4,
            })
        );
        assert_eq!(
            parse_dimacs_mis_with_limits(
                "p edge 2 1\\ne 1 2\\n",
                MisDimacsLimits { max_vertices: 1, ..MisDimacsLimits::default() }
            ),
            Err(MisDimacsError::VertexLimitExceeded { declared: 2, maximum: 1 })
        );
        assert_eq!(
            parse_dimacs_mis_with_limits(
                "p edge 2 1\\ne 1 2\\n",
                MisDimacsLimits { max_edges: 0, ..MisDimacsLimits::default() }
            ),
            Err(MisDimacsError::EdgeLimitExceeded { declared: 1, maximum: 0 })
        );
        assert_eq!(
            parse_dimacs_mis_with_limits(
                "p edge 2 1\\ne 1 2\\ne 1 2\\n",
                MisDimacsLimits { max_edges: 1, ..MisDimacsLimits::default() }
            ),
            Err(MisDimacsError::EdgeRecordLimitExceeded { line: 3, maximum: 1 })
        );
    }

    #[test]
    fn parses_large_sparse_instance_with_linear_edge_storage() {
        // Exercise parser behavior on a larger input than the tiny semantic fixtures.
        // The 100k-vertex path is still sparse, so test memory stays bounded.
        let vertices = 100_000usize;
        let mut source = format!("p edge {} {}\n", vertices, vertices - 1);
        for u in 1..vertices {
            source.push_str(&format!("e {} {}\n", u, u + 1));
        }
        let parsed = parse_dimacs_mis(&source).expect("large sparse DIMACS graph parses");
        assert_eq!(parsed.declared_vertex_count, vertices);
        assert_eq!(parsed.declared_edge_count, vertices - 1);
        assert_eq!(parsed.graph.vertex_count(), vertices);
        assert_eq!(parsed.graph.edges().len(), vertices - 1);
    }

}
