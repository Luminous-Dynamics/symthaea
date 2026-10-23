//! Run a deterministic MIS smoke baseline and bounded exact reference on one DIMACS file.
//!
//! This binary produces one JSON record on stdout. It is a reproducibility smoke test,
//! not a competitive solver benchmark and not a quantum backend. Redirect stdout to a
//! file to preserve the raw receipt.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use symthaea_quantum_comp::mis_verifier::{MisGraph, EXACT_ORACLE_MAX_VERTICES};
use symthaea_quantum_comp::parse_dimacs_mis;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch <= '\u{001f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn json_vertices(vertices: &[usize]) -> String {
    let body = vertices.iter()
        .map(|vertex| (vertex + 1).to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn greedy_min_residual_degree(graph: &MisGraph) -> Vec<usize> {
    let mut adjacency = vec![Vec::new(); graph.vertex_count()];
    for &(u, v) in graph.edges() {
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }

    let mut active = vec![true; graph.vertex_count()];
    let mut selected = Vec::new();
    loop {
        let next = (0..graph.vertex_count())
            .filter(|&vertex| active[vertex])
            .min_by_key(|&vertex| {
                let residual_degree = adjacency[vertex]
                    .iter()
                    .filter(|&&neighbor| active[neighbor])
                    .count();
                (residual_degree, vertex)
            });
        let Some(vertex) = next else {
            break;
        };
        active[vertex] = false;
        selected.push(vertex);
        for &neighbor in &adjacency[vertex] {
            active[neighbor] = false;
        }
    }
    selected.sort_unstable();
    selected
}

fn run(path: &str) -> Result<(), String> {
    let end_to_end_start = Instant::now();
    let source = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {path:?}: {error}"))?;
    let source_hash = fnv1a64(source.as_bytes());

    let parse_start = Instant::now();
    let parsed = parse_dimacs_mis(&source)
        .map_err(|error| format!("failed to parse {path:?}: {error}"))?;
    let parse_ns = parse_start.elapsed().as_nanos();

    let greedy_start = Instant::now();
    let greedy = greedy_min_residual_degree(&parsed.graph);
    let greedy_ns = greedy_start.elapsed().as_nanos();
    let greedy_report = parsed.graph.verify_solution(&greedy, Some(greedy.len()));
    if !greedy_report.feasible {
        return Err("internal error: greedy baseline failed independent verification".to_owned());
    }

    let exact_start = Instant::now();
    let exact: Option<Vec<usize>> = if parsed.graph.vertex_count() <= EXACT_ORACLE_MAX_VERTICES {
        Some(
            parsed.graph.exact_maximum_independent_set()
                .map_err(|error| format!("exact reference failed: {error}"))?,
        )
    } else {
        None
    };
    let exact_ns = exact.as_ref().map(|_| exact_start.elapsed().as_nanos());
    let exact_report = exact.as_ref().map(|solution| {
        parsed.graph.verify_solution(solution, Some(solution.len()))
    });
    if exact_report.as_ref().is_some_and(|report| !report.feasible) {
        return Err("internal error: exact reference failed independent verification".to_owned());
    }

    // Hash a deterministic normalized result body (not runtime fields) so repeated runs
    // can compare outputs even though measured elapsed time naturally changes.
    let normalized = format!(
        "v1|{}|{}|{}|greedy-min-residual-degree|{}|exact={}",
        format!("{source_hash:016x}"),
        parsed.declared_vertex_count,
        parsed.declared_edge_count,
        greedy.iter().map(|vertex| (vertex + 1).to_string()).collect::<Vec<_>>().join(","),
        exact.as_ref().map(|vertices| vertices.iter()
            .map(|vertex| (vertex + 1).to_string()).collect::<Vec<_>>().join(","))
            .unwrap_or_else(|| "skipped-over-limit".to_owned())
    );
    let normalized_hash = fnv1a64(normalized.as_bytes());
    let end_to_end_ns = end_to_end_start.elapsed().as_nanos();

    println!(
        concat!(
            "{{\n",
            "  \"schema\": \"org.luminousdynamics.symthaea.qcomp.mis-smoke-run.v1\",\n",
            "  \"benchmark_track\": \"correctness-smoke-not-competitive\",\n",
            "  \"source_path\": \"{}\",\n",
            "  \"source_bytes\": {},\n",
            "  \"source_fnv1a64\": \"{:016x}\",\n",
            "  \"vertices\": {},\n",
            "  \"edges\": {},\n",
            "  \"greedy\": {{\"policy\": \"minimum-residual-degree-then-lowest-index\", \"objective\": {}, \"feasible\": {}, \"vertices_one_based\": {}, \"elapsed_ns\": {}}},\n",
            "  \"exact_oracle\": {{\"status\": \"{}\", \"objective\": {}, \"feasible\": {}, \"vertices_one_based\": {}, \"elapsed_ns\": {}}},\n",
            "  \"timing_ns\": {{\"parse\": {}, \"end_to_end\": {}}},\n",
            "  \"normalized_result_fnv1a64\": \"{:016x}\",\n",
            "  \"limitations\": [\"FNV-1a is a non-cryptographic integrity checksum\", \"wall-clock timings are local diagnostics and are not comparable across machines without a controlled protocol\", \"the bounded exact oracle is a correctness reference, not a competitive solver\", \"this run does not execute a quantum backend\"]\n",
            "}}"
        ),
        json_escape(path),
        source.len(),
        source_hash,
        parsed.declared_vertex_count,
        parsed.declared_edge_count,
        greedy_report.computed_objective,
        greedy_report.feasible,
        json_vertices(&greedy),
        greedy_ns,
        if exact.is_some() { "ran" } else { "skipped-vertex-limit" },
        exact_report.as_ref().map(|report| report.computed_objective.to_string()).unwrap_or_else(|| "null".to_owned()),
        exact_report.as_ref().map(|report| report.feasible.to_string()).unwrap_or_else(|| "null".to_owned()),
        exact.as_ref().map(|vertices| json_vertices(vertices)).unwrap_or_else(|| "null".to_owned()),
        exact_ns.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned()),
        parse_ns,
        end_to_end_ns,
        normalized_hash
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: qoblib-mis-smoke <path-to-DIMACS-edge-graph>");
        eprintln!("Emits a JSON smoke receipt with a deterministic greedy solution and, for at most {EXACT_ORACLE_MAX_VERTICES} vertices, the tiny exact reference.");
        return if args.len() == 2 { ExitCode::SUCCESS } else { ExitCode::from(2) };
    }
    match run(&args[1]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_degree_greedy_is_deterministic_and_feasible_on_a_path() {
        let graph = MisGraph::new(4, [(0, 1), (1, 2), (2, 3)]).unwrap();
        let first = greedy_min_residual_degree(&graph);
        let second = greedy_min_residual_degree(&graph);
        assert_eq!(first, vec![0, 2]);
        assert_eq!(first, second);
        assert!(graph.verify_solution(&first, Some(first.len())).feasible);
    }

    #[test]
    fn residual_degree_greedy_keeps_all_leaves_of_a_star() {
        let graph = MisGraph::new(4, [(0, 1), (0, 2), (0, 3)]).unwrap();
        let selected = greedy_min_residual_degree(&graph);
        assert_eq!(selected, vec![1, 2, 3]);
        assert!(graph.verify_solution(&selected, Some(3)).feasible);
    }
}
