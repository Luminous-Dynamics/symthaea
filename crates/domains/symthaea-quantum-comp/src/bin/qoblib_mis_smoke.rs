//! Run a deterministic MIS smoke baseline and bounded exact reference on one DIMACS file.
//!
//! This binary produces one JSON record on stdout. It is a reproducibility smoke test,
//! not a competitive solver benchmark and not a quantum backend. Redirect stdout to a
//! file to preserve the raw receipt.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use symthaea_quantum_comp::mis_verifier::{MisGraph, EXACT_ORACLE_MAX_VERTICES};
use symthaea_quantum_comp::rng::XorShift64;
use symthaea_quantum_comp::{MisDimacsLimits, parse_dimacs_mis_with_limits, parse_qoblib_solution};

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

fn json_numbers(values: &[usize]) -> String {
    let body = values.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    format!("[{body}]")
}

fn qoblib_run_points_json(trace: &[(u128, usize)]) -> String {
    let points = trace.iter()
        .map(|(elapsed_ns, incumbent)| {
            format!(
                "{{\"Time\": {:.9}, \"Incumbent\": {}}}",
                *elapsed_ns as f64 / 1_000_000_000.0,
                incumbent
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{points}]")
}

fn qoblib_many_runs_series_json(runs: &[Vec<(u128, usize)>]) -> String {
    let body = runs.iter()
        .map(|trace| qoblib_run_points_json(trace))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn time_to_target_nanos(trace: &[(u128, usize)], target_objective: usize) -> Option<u128> {
    trace.iter()
        .find(|point| point.1 >= target_objective)
        .map(|point| point.0)
}

/// Summarize fixed-target attainment across independent, relative-time traces.
fn target_trial_summary_json(
    target_objective: usize,
    runs: &[Vec<(u128, usize)>],
) -> String {
    let hit_times = runs.iter()
        .map(|trace| time_to_target_nanos(trace, target_objective))
        .collect::<Vec<_>>();
    let target_hits = hit_times.iter().filter(|time| time.is_some()).count();
    let trial_count = hit_times.len();
    let empirical_hit_rate = if trial_count == 0 {
        0.0
    } else {
        target_hits as f64 / trial_count as f64
    };
    let times = hit_times.iter()
        .map(|time| match time {
            Some(nanos) => format!("{:.9}", *nanos as f64 / 1_000_000_000.0),
            None => "null".to_owned(),
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"trial_count\": {trial_count}, \"target_hits\": {target_hits}, \"empirical_hit_rate\": {empirical_hit_rate:.6}, \"time_to_target_seconds_by_trial\": [{times}]}}"
    )
}

/// Build optional target-hit summaries from the same incumbent traces emitted in the receipt.
/// Stage-local time for local improvement is kept separate from full greedy-run time.
fn target_analysis_json(
    target_objective: Option<usize>,
    deterministic_trace: &[(u128, usize)],
    portfolio_traces: &[Vec<(u128, usize)>],
    local_improvement_trace: Option<&[(u128, usize)]>,
) -> String {
    let Some(target) = target_objective else {
        return "null".to_owned();
    };
    let deterministic_runs = vec![deterministic_trace.to_vec()];
    let deterministic = target_trial_summary_json(target, &deterministic_runs);
    let portfolio = target_trial_summary_json(target, portfolio_traces);
    let local = local_improvement_trace
        .map(|trace| target_trial_summary_json(target, &[trace.to_vec()]))
        .unwrap_or_else(|| "null".to_owned());
    format!(
        "{{\"target_objective\": {target}, \"deterministic_greedy\": {deterministic}, \"multi_start_greedy_per_trial\": {portfolio}, \"local_improvement_stage\": {local}}}"
    )
}

const MULTISTART_GREEDY_RUNS: usize = 16;
const MULTISTART_GREEDY_SEED_BASE: u64 = 0x5143_4f4d_505f_3031;

/// Build an independent set with minimum residual degree and a deterministic vertex tie break.
fn greedy_min_residual_degree(graph: &MisGraph) -> Vec<usize> {
    greedy_min_residual_degree_with_seed(graph, None)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GreedyRunResult {
    vertices: Vec<usize>,
    /// Incumbent objective observations in nanoseconds from the start of this run.
    objective_time_series: Vec<(u128, usize)>,
    elapsed_nanos: u128,
}

/// Compatibility wrapper for callers that only need the final candidate set.
fn greedy_min_residual_degree_with_seed(graph: &MisGraph, seed: Option<u64>) -> Vec<usize> {
    greedy_min_residual_degree_with_seed_trace(graph, seed).vertices
}

/// Build an independent set with minimum residual degree and preserve every incumbent.
///
/// With a seed, ties among equal residual degrees are selected using a reproducible
/// local RNG. A lazy min-heap avoids rescanning all adjacency lists after every
/// selection. Each selected prefix is a feasible independent set, so cardinality
/// monotonically increases and can be emitted as a QOBLIB incumbent time series.
fn greedy_min_residual_degree_with_seed_trace(
    graph: &MisGraph,
    seed: Option<u64>,
) -> GreedyRunResult {
    let run_start = Instant::now();
    let vertex_count = graph.vertex_count();
    let mut adjacency = vec![Vec::new(); vertex_count];
    for &(u, v) in graph.edges() {
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }

    let mut active = vec![true; vertex_count];
    let mut residual_degree: Vec<usize> = adjacency.iter().map(Vec::len).collect();
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for (vertex, &degree) in residual_degree.iter().enumerate() {
        heap.push(Reverse((degree, vertex)));
    }
    let mut rng = seed.map(XorShift64::new);
    let mut selected = Vec::new();
    let mut objective_time_series = vec![(0_u128, 0_usize)];

    loop {
        // Discard stale degree entries left behind by neighbor removals.
        while let Some(Reverse((degree, vertex))) = heap.peek().copied() {
            if !active[vertex] || degree != residual_degree[vertex] {
                heap.pop();
            } else {
                break;
            }
        }

        let Some(Reverse((minimum_degree, _))) = heap.peek().copied() else {
            break;
        };

        // Gather every currently valid vertex tied at the minimum degree. Reinsert
        // unchosen ties so their heap entries remain live in later iterations.
        let mut ties = Vec::new();
        while let Some(Reverse((degree, vertex))) = heap.peek().copied() {
            if degree != minimum_degree {
                break;
            }
            heap.pop();
            if active[vertex] && degree == residual_degree[vertex] {
                ties.push(vertex);
            }
        }
        if ties.is_empty() {
            continue;
        }
        let tie_index = rng
            .as_mut()
            .and_then(|state| state.next_usize(ties.len()))
            .unwrap_or(0);
        let chosen = ties[tie_index];
        for &vertex in &ties {
            if vertex != chosen {
                heap.push(Reverse((residual_degree[vertex], vertex)));
            }
        }

        // The chosen vertex enters the independent set; its closed neighborhood is
        // removed. Then decrement residual degrees only for still-active neighbors.
        let mut removed = Vec::with_capacity(adjacency[chosen].len() + 1);
        active[chosen] = false;
        removed.push(chosen);
        for &neighbor in &adjacency[chosen] {
            if active[neighbor] {
                active[neighbor] = false;
                removed.push(neighbor);
            }
        }
        for &gone in &removed {
            for &neighbor in &adjacency[gone] {
                if active[neighbor] {
                    residual_degree[neighbor] -= 1;
                    heap.push(Reverse((residual_degree[neighbor], neighbor)));
                }
            }
        }
        selected.push(chosen);
        objective_time_series.push((run_start.elapsed().as_nanos(), selected.len()));
    }

    selected.sort_unstable();
    let elapsed_nanos = run_start.elapsed().as_nanos();
    let last_time = objective_time_series.last().map(|point| point.0).unwrap_or(0);
    objective_time_series.push((elapsed_nanos.max(last_time), selected.len()));
    GreedyRunResult {
        vertices: selected,
        objective_time_series,
        elapsed_nanos,
    }
}

/// A bounded, reproducible multi-start classical baseline. It selects the best
/// feasible greedy result; equal-size results use lexicographically smaller vertices.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MultiStartGreedyResult {
    best_vertices: Vec<usize>,
    best_seed: u64,
    objectives: Vec<usize>,
    distinct_solutions: usize,
    /// One QOBLIB-compatible incumbent time series per independently seeded run.
    objective_time_series: Vec<Vec<(u128, usize)>>,
}

fn multi_start_greedy(graph: &MisGraph, repeats: usize, seed_base: u64) -> MultiStartGreedyResult {
    let repeats = repeats.max(1);
    let mut best_vertices = Vec::new();
    let mut best_seed = seed_base;
    let mut objectives = Vec::with_capacity(repeats);
    let mut objective_time_series = Vec::with_capacity(repeats);
    let mut unique_solutions = BTreeSet::new();

    for run_index in 0..repeats {
        let seed = seed_base.wrapping_add(run_index as u64);
        let run = greedy_min_residual_degree_with_seed_trace(graph, Some(seed));
        let vertices = run.vertices;
        objectives.push(vertices.len());
        objective_time_series.push(run.objective_time_series);
        unique_solutions.insert(vertices.clone());
        if vertices.len() > best_vertices.len()
            || (vertices.len() == best_vertices.len() && vertices < best_vertices)
        {
            best_vertices = vertices;
            best_seed = seed;
        }
    }

    MultiStartGreedyResult {
        best_vertices,
        best_seed,
        objectives,
        distinct_solutions: unique_solutions.len(),
        objective_time_series,
    }
}


const LOCAL_SEARCH_MAX_VERTICES: usize = 250;
const LOCAL_SEARCH_MAX_PAIR_CHECKS: usize = 2_000_000;

/// Result of deterministic feasible 1-for-2 improvement of an independent set.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalImprovementResult {
    vertices: Vec<usize>,
    exchanges: usize,
    pair_checks: usize,
    locally_optimal: bool,
    /// QOBLIB-compatible incumbent observations relative to this stage's start.
    objective_time_series: Vec<(u128, usize)>,
}

/// Repeatedly replace one selected vertex by two mutually nonadjacent compatible
/// unselected vertices. Candidate pairs are scanned in ascending vertex order, making
/// the result deterministic. A pair-check budget makes the diagnostic bounded; if it
/// expires the current feasible solution is returned but not labelled locally optimal.
fn improve_one_for_two(
    graph: &MisGraph,
    initial: &[usize],
    pair_check_budget: usize,
) -> LocalImprovementResult {
    let search_start = Instant::now();
    let vertex_count = graph.vertex_count();
    let mut adjacency = vec![Vec::new(); vertex_count];
    for &(u, v) in graph.edges() {
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }

    let mut selected = vec![false; vertex_count];
    for &vertex in initial {
        if vertex < vertex_count {
            selected[vertex] = true;
        }
    }
    let mut exchanges = 0usize;
    let mut pair_checks = 0usize;
    let initial_objective = selected.iter().filter(|&&keep| keep).count();
    let mut objective_time_series = vec![(0_u128, initial_objective)];

    loop {
        let mut selected_neighbor_count = vec![0usize; vertex_count];
        for vertex in 0..vertex_count {
            if selected[vertex] {
                for &neighbor in &adjacency[vertex] {
                    selected_neighbor_count[neighbor] += 1;
                }
            }
        }

        let mut found_move: Option<(usize, usize, usize)> = None;
        'removed: for removed in 0..vertex_count {
            if !selected[removed] {
                continue;
            }

            // A candidate is compatible with every selected vertex except possibly the
            // one removed. The count test avoids scanning the whole selected set for
            // every candidate; if its one selected neighbor is removed, it is allowed.
            let candidates: Vec<usize> = (0..vertex_count)
                .filter(|&candidate| {
                    !selected[candidate]
                        && (selected_neighbor_count[candidate] == 0
                            || (selected_neighbor_count[candidate] == 1
                                && adjacency[candidate].binary_search(&removed).is_ok()))
                })
                .collect();

            for left_index in 0..candidates.len() {
                let left = candidates[left_index];
                for &right in &candidates[(left_index + 1)..] {
                    if pair_checks >= pair_check_budget {
                        let vertices = selected.iter().enumerate()
                            .filter_map(|(vertex, &keep)| keep.then_some(vertex))
                            .collect();
                        let elapsed = search_start.elapsed().as_nanos();
                        let last_time = objective_time_series.last().map(|point| point.0).unwrap_or(0);
                        objective_time_series.push((elapsed.max(last_time), vertices.len()));
                        return LocalImprovementResult {
                            vertices, exchanges, pair_checks,
                            locally_optimal: false,
                            objective_time_series,
                        };
                    }
                    pair_checks += 1;
                    if adjacency[left].binary_search(&right).is_err() {
                        found_move = Some((removed, left, right));
                        break 'removed;
                    }
                }
            }
        }

        let Some((removed, left, right)) = found_move else {
            let vertices = selected.iter().enumerate()
                .filter_map(|(vertex, &keep)| keep.then_some(vertex))
                .collect();
            let elapsed = search_start.elapsed().as_nanos();
            let last_time = objective_time_series.last().map(|point| point.0).unwrap_or(0);
            objective_time_series.push((elapsed.max(last_time), vertices.len()));
            return LocalImprovementResult {
                vertices, exchanges, pair_checks, locally_optimal: true,
                objective_time_series,
            };
        };
        selected[removed] = false;
        selected[left] = true;
        selected[right] = true;
        exchanges += 1;
        let objective = selected.iter().filter(|&&keep| keep).count();
        objective_time_series.push((search_start.elapsed().as_nanos(), objective));
    }
}


#[derive(Debug)]
struct RunFailure {
    kind: &'static str,
    message: String,
    source_bytes: Option<usize>,
    source_fnv1a64: Option<u64>,
    elapsed_ns: u128,
}

fn failure_receipt_json(path: &str, failure: &RunFailure, seed_base: u64, target_objective: Option<usize>) -> String {
    let source_bytes = failure.source_bytes
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "null".to_owned());
    let source_hash = failure.source_fnv1a64
        .map(|hash| format!("\"{hash:016x}\""))
        .unwrap_or_else(|| "null".to_owned());
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"org.luminousdynamics.symthaea.qcomp.mis-smoke-run.v5\",\n",
            "  \"benchmark_track\": \"correctness-smoke-not-competitive\",\n",
            "  \"return_status\": \"failed\",\n",
            "  \"source_path\": \"{}\",\n",
            "  \"source_bytes\": {},\n",
            "  \"source_fnv1a64\": {},\n",
            "  \"error_kind\": \"{}\",\n",
            "  \"error\": \"{}\",\n",
            "  \"software\": {{\"package_version\": \"{}\", \"target_os\": \"{}\", \"target_arch\": \"{}\", \"rustc_version\": \"{}\"}},\n",
            "  \"determinism\": \"not-attested: failed run; inspect error_kind and source identity\",\n",
            "  \"target_objective\": {},\n",
            "  \"randomness\": {{\"seed_base\": \"{}\", \"restarts\": {}, \"policy\": \"XorShift64 seeded tie-breaks\"}},\n",
            "  \"resource_limits\": {{\"time_limit_ns\": null, \"memory_limit_bytes\": null, \"max_input_bytes\": {}, \"max_vertices\": {}, \"max_edges\": {}}},\n",
            "  \"timing_ns\": {{\"end_to_end\": {}}},\n",
            "  \"limitations\": [\"FNV-1a is a non-cryptographic integrity checksum\", \"no runtime limit or memory limit was configured\", \"failure receipt records a read or parse failure rather than silently dropping the run\", \"this run does not execute a quantum backend\"]\n",
            "}}"
        ),
        json_escape(path),
        source_bytes,
        source_hash,
        failure.kind,
        json_escape(&failure.message),
        env!("CARGO_PKG_VERSION"),
        env::consts::OS,
        env::consts::ARCH,
        option_env!("RUSTC_VERSION").unwrap_or("not-recorded-at-build"),
        target_objective.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned()),
        seed_base.to_string(),
        MULTISTART_GREEDY_RUNS,
        MisDimacsLimits::default().max_input_bytes,
        MisDimacsLimits::default().max_vertices,
        MisDimacsLimits::default().max_edges,
        failure.elapsed_ns
    )
}

fn run(path: &str, seed_base: u64, target_objective: Option<usize>) -> Result<(), RunFailure> {
    let end_to_end_start = Instant::now();
    let limits = MisDimacsLimits::default();
    let metadata = fs::metadata(path).map_err(|error| RunFailure {
        kind: "source_read",
        message: format!("failed to stat {path:?}: {error}"),
        source_bytes: None,
        source_fnv1a64: None,
        elapsed_ns: end_to_end_start.elapsed().as_nanos(),
    })?;
    if metadata.len() > limits.max_input_bytes as u64 {
        let actual = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
        return Err(RunFailure {
            kind: "source_size_limit",
            message: format!(
                "source is {actual} bytes, exceeding the configured {}-byte limit",
                limits.max_input_bytes
            ),
            source_bytes: Some(actual),
            source_fnv1a64: None,
            elapsed_ns: end_to_end_start.elapsed().as_nanos(),
        });
    }
    let source = fs::read_to_string(path).map_err(|error| RunFailure {
        kind: "source_read",
        message: format!("failed to read {path:?}: {error}"),
        source_bytes: None,
        source_fnv1a64: None,
        elapsed_ns: end_to_end_start.elapsed().as_nanos(),
    })?;
    let source_hash = fnv1a64(source.as_bytes());

    let parse_start = Instant::now();
    let parsed = parse_dimacs_mis_with_limits(&source, limits).map_err(|error| RunFailure {
        kind: "dimacs_parse",
        message: format!("failed to parse {path:?}: {error}"),
        source_bytes: Some(source.len()),
        source_fnv1a64: Some(source_hash),
        elapsed_ns: end_to_end_start.elapsed().as_nanos(),
    })?;
    let parse_ns = parse_start.elapsed().as_nanos();

    if target_objective.is_some_and(|target| target > parsed.declared_vertex_count) {
        return Err(RunFailure {
            kind: "target_out_of_range",
            message: format!(
                "target objective {} exceeds vertex count {}",
                target_objective.unwrap_or(0),
                parsed.declared_vertex_count
            ),
            source_bytes: Some(source.len()),
            source_fnv1a64: Some(source_hash),
            elapsed_ns: end_to_end_start.elapsed().as_nanos(),
        });
    }

    let greedy_run = greedy_min_residual_degree_with_seed_trace(&parsed.graph, None);
    let greedy = &greedy_run.vertices;
    let greedy_ns = greedy_run.elapsed_nanos;
    let greedy_report = parsed.graph.verify_solution(greedy, Some(greedy.len()));
    if !greedy_report.feasible {
        return Err(RunFailure {
            kind: "greedy_verification",
            message: "internal error: greedy baseline failed independent verification".to_owned(),
            source_bytes: Some(source.len()),
            source_fnv1a64: Some(source_hash),
            elapsed_ns: end_to_end_start.elapsed().as_nanos(),
        });
    }

    let portfolio_start = Instant::now();
    let portfolio = multi_start_greedy(&parsed.graph, MULTISTART_GREEDY_RUNS, seed_base);
    let portfolio_ns = portfolio_start.elapsed().as_nanos();
    let portfolio_report = parsed.graph.verify_solution(
        &portfolio.best_vertices,
        Some(portfolio.best_vertices.len()),
    );
    if !portfolio_report.feasible {
        return Err(RunFailure {
            kind: "portfolio_verification",
            message: "internal error: multi-start greedy portfolio failed independent verification".to_owned(),
            source_bytes: Some(source.len()),
            source_fnv1a64: Some(source_hash),
            elapsed_ns: end_to_end_start.elapsed().as_nanos(),
        });
    }

    let local_start = Instant::now();
    let local: Option<LocalImprovementResult> = if parsed.graph.vertex_count() <= LOCAL_SEARCH_MAX_VERTICES {
        Some(improve_one_for_two(&parsed.graph, greedy, LOCAL_SEARCH_MAX_PAIR_CHECKS))
    } else {
        None
    };
    let local_ns = local.as_ref().map(|_| local_start.elapsed().as_nanos());
    let local_report = local.as_ref().map(|result| {
        parsed.graph.verify_solution(&result.vertices, Some(result.vertices.len()))
    });
    if local_report.as_ref().is_some_and(|report| !report.feasible) {
        return Err(RunFailure {
            kind: "local_verification",
            message: "internal error: 1-for-2 local improvement failed independent verification".to_owned(),
            source_bytes: Some(source.len()),
            source_fnv1a64: Some(source_hash),
            elapsed_ns: end_to_end_start.elapsed().as_nanos(),
        });
    }

    let exact_start = Instant::now();
    let exact: Option<Vec<usize>> = if parsed.graph.vertex_count() <= EXACT_ORACLE_MAX_VERTICES {
        Some(
            parsed.graph.exact_maximum_independent_set()
                .map_err(|error| RunFailure {
                kind: "exact_oracle",
                message: format!("exact reference failed: {error}"),
                source_bytes: Some(source.len()),
                source_fnv1a64: Some(source_hash),
                elapsed_ns: end_to_end_start.elapsed().as_nanos(),
            })?,
        )
    } else {
        None
    };
    let exact_ns = exact.as_ref().map(|_| exact_start.elapsed().as_nanos());
    let exact_report = exact.as_ref().map(|solution| {
        parsed.graph.verify_solution(solution, Some(solution.len()))
    });
    if exact_report.as_ref().is_some_and(|report| !report.feasible) {
        return Err(RunFailure {
            kind: "exact_verification",
            message: "internal error: exact reference failed independent verification".to_owned(),
            source_bytes: Some(source.len()),
            source_fnv1a64: Some(source_hash),
            elapsed_ns: end_to_end_start.elapsed().as_nanos(),
        });
    }

    // Hash a deterministic normalized result body (not runtime fields) so repeated runs
    // can compare outputs even though measured elapsed time naturally changes.
    let local_normalized = local.as_ref().map(|result| {
        format!(
            "locally_optimal={}|exchanges={}|pair_checks={}|vertices={}",
            result.locally_optimal,
            result.exchanges,
            result.pair_checks,
            result.vertices.iter().map(|vertex| (vertex + 1).to_string()).collect::<Vec<_>>().join(",")
        )
    }).unwrap_or_else(|| "skipped-vertex-limit".to_owned());
    let portfolio_normalized = format!(
        "seed_base={}|best_seed={}|objectives={}|distinct={}|vertices={}",
        seed_base,
        portfolio.best_seed,
        portfolio.objectives.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
        portfolio.distinct_solutions,
        portfolio.best_vertices.iter().map(|vertex| (vertex + 1).to_string()).collect::<Vec<_>>().join(",")
    );
    let normalized = format!(
        "v5|source={}|vertices={}|edges={}|seed_base={}|restarts={}|target={}|greedy-min-residual-degree|{}|portfolio={}|local_max_vertices={}|local_pair_budget={}|local={}|exact_max_vertices={}|exact={}",
        format!("{source_hash:016x}"),
        parsed.declared_vertex_count,
        parsed.declared_edge_count,
        seed_base,
        MULTISTART_GREEDY_RUNS,
        target_objective.map(|value| value.to_string()).unwrap_or_else(|| "none".to_owned()),
        greedy.iter().map(|vertex| (vertex + 1).to_string()).collect::<Vec<_>>().join(","),
        portfolio_normalized,
        LOCAL_SEARCH_MAX_VERTICES,
        LOCAL_SEARCH_MAX_PAIR_CHECKS,
        local_normalized,
        EXACT_ORACLE_MAX_VERTICES,
        exact.as_ref().map(|vertices| vertices.iter()
            .map(|vertex| (vertex + 1).to_string()).collect::<Vec<_>>().join(","))
            .unwrap_or_else(|| "skipped-over-limit".to_owned())
    );
    let normalized_hash = fnv1a64(normalized.as_bytes());
    let end_to_end_ns = end_to_end_start.elapsed().as_nanos();

    println!(
        concat!(
            "{{\n",
            "  \"schema\": \"org.luminousdynamics.symthaea.qcomp.mis-smoke-run.v5\",\n",
            "  \"benchmark_track\": \"correctness-smoke-not-competitive\",\n",
            "  \"return_status\": \"completed\",\n",
            "  \"source_path\": \"{}\",\n",
            "  \"source_bytes\": {},\n",
            "  \"source_fnv1a64\": \"{:016x}\",\n",
            "  \"vertices\": {},\n",
            "  \"edges\": {},\n",
            "  \"target_objective\": {},\n",
            "  \"greedy\": {{\"policy\": \"minimum-residual-degree-then-lowest-index\", \"objective\": {}, \"feasible\": {}, \"vertices_one_based\": {}, \"elapsed_ns\": {}}},\n",
            "  \"multi_start_greedy\": {{\"policy\": \"minimum-residual-degree-seeded-ties\", \"runs\": {}, \"best_seed\": \"{}\", \"best_objective\": {}, \"feasible\": {}, \"distinct_solutions\": {}, \"objectives_by_run\": {}, \"vertices_one_based\": {}, \"elapsed_ns\": {}}},\n",
            "  \"local_improvement\": {{\"neighborhood\": \"one-selected-for-two-unselected\", \"status\": \"{}\", \"objective\": {}, \"feasible\": {}, \"vertices_one_based\": {}, \"exchanges\": {}, \"pair_checks\": {}, \"elapsed_ns\": {}}},\n",
            "  \"exact_oracle\": {{\"status\": \"{}\", \"objective\": {}, \"feasible\": {}, \"vertices_one_based\": {}, \"elapsed_ns\": {}}},\n",
            "  \"timing_ns\": {{\"parse\": {}, \"end_to_end\": {}}},\n",
            "  \"software\": {{\"package_version\": \"{}\", \"target_os\": \"{}\", \"target_arch\": \"{}\", \"rustc_version\": \"{}\"}},\n",
            "  \"determinism\": \"same input, toolchain and seed replay algorithm outcomes; elapsed fields vary\",\n",
            "  \"resource_limits\": {{\"time_limit_ns\": null, \"memory_limit_bytes\": null, \"max_input_bytes\": {}, \"max_vertices\": {}, \"max_edges\": {}}},\n",
            "  \"normalized_result_fnv1a64\": \"{:016x}\",\n",
            "  \"qoblib_objective_time_series\": {{\"deterministic_greedy\": {}, \"multi_start_greedy_per_trial\": {}, \"local_improvement_stage\": {}}},\n",
            "  \"target_analysis\": {},\n",
            "  \"randomness\": {{\"seed_base\": \"{}\", \"restarts\": {}, \"policy\": \"XorShift64 seeded tie-breaks\"}},\n",
            "  \"limitations\": [\"FNV-1a is a non-cryptographic integrity checksum\", \"wall-clock timings are local diagnostics and are not comparable across machines without a controlled protocol\", \"the bounded exact oracle is a correctness reference, not a competitive solver\", \"this run does not execute a quantum backend\"]\n",
            "}}"
        ),
        json_escape(path),
        source.len(),
        source_hash,
        parsed.declared_vertex_count,
        parsed.declared_edge_count,
        target_objective.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned()),
        greedy_report.computed_objective,
        greedy_report.feasible,
        json_vertices(greedy),
        greedy_ns,
        MULTISTART_GREEDY_RUNS,
        portfolio.best_seed,
        portfolio_report.computed_objective,
        portfolio_report.feasible,
        portfolio.distinct_solutions,
        json_numbers(&portfolio.objectives),
        json_vertices(&portfolio.best_vertices),
        portfolio_ns,
        local.as_ref().map(|result| if result.locally_optimal { "locally-optimal-1-for-2" } else { "pair-check-budget-exhausted" }).unwrap_or("skipped-vertex-limit"),
        local_report.as_ref().map(|report| report.computed_objective.to_string()).unwrap_or_else(|| "null".to_owned()),
        local_report.as_ref().map(|report| report.feasible.to_string()).unwrap_or_else(|| "null".to_owned()),
        local.as_ref().map(|result| json_vertices(&result.vertices)).unwrap_or_else(|| "null".to_owned()),
        local.as_ref().map(|result| result.exchanges.to_string()).unwrap_or_else(|| "null".to_owned()),
        local.as_ref().map(|result| result.pair_checks.to_string()).unwrap_or_else(|| "null".to_owned()),
        local_ns.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned()),
        if exact.is_some() { "ran" } else { "skipped-vertex-limit" },
        exact_report.as_ref().map(|report| report.computed_objective.to_string()).unwrap_or_else(|| "null".to_owned()),
        exact_report.as_ref().map(|report| report.feasible.to_string()).unwrap_or_else(|| "null".to_owned()),
        exact.as_ref().map(|vertices| json_vertices(vertices)).unwrap_or_else(|| "null".to_owned()),
        exact_ns.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned()),
        parse_ns,
        end_to_end_ns,
        env!("CARGO_PKG_VERSION"),
        env::consts::OS,
        env::consts::ARCH,
        option_env!("RUSTC_VERSION").unwrap_or("not-recorded-at-build"),
        limits.max_input_bytes,
        limits.max_vertices,
        limits.max_edges,
        normalized_hash,
        qoblib_many_runs_series_json(&[greedy_run.objective_time_series.clone()]),
        qoblib_many_runs_series_json(&portfolio.objective_time_series),
        local.as_ref()
            .map(|result| qoblib_many_runs_series_json(&[result.objective_time_series.clone()]))
            .unwrap_or_else(|| "null".to_owned()),
        target_analysis_json(
            target_objective,
            &greedy_run.objective_time_series,
            &portfolio.objective_time_series,
            local.as_ref().map(|result| result.objective_time_series.as_slice()),
        ),
        seed_base.to_string(),
        MULTISTART_GREEDY_RUNS
    );
    Ok(())
}


fn verify_solution_text(
    graph_source: &str,
    solution_source: &str,
    reported_objective: Option<usize>,
) -> Result<
    (
        symthaea_quantum_comp::ParsedMisDimacs,
        Vec<usize>,
        symthaea_quantum_comp::mis_verifier::MisVerificationReport,
    ),
    String,
> {
    let limits = MisDimacsLimits::default();
    let parsed = parse_dimacs_mis_with_limits(graph_source, limits)
        .map_err(|error| format!("DIMACS parse error: {error}"))?;
    let selected = parse_qoblib_solution(solution_source, parsed.graph.vertex_count())
        .map_err(|error| format!("QOBLIB solution parse error: {error}"))?;
    let report = parsed.graph.verify_solution(&selected, reported_objective);
    Ok((parsed, selected, report))
}

fn run_verify_solution(
    graph_path: &str,
    solution_path: &str,
    reported_objective: Option<usize>,
) -> Result<bool, String> {
    let graph_source = fs::read_to_string(graph_path)
        .map_err(|error| format!("failed to read graph {graph_path:?}: {error}"))?;
    let solution_source = fs::read_to_string(solution_path)
        .map_err(|error| format!("failed to read solution {solution_path:?}: {error}"))?;
    let (parsed, selected, report) = verify_solution_text(
        &graph_source,
        &solution_source,
        reported_objective,
    )?;
    let violations = report.violations.iter()
        .map(|violation| format!("\"{}\"", json_escape(&format!("{violation:?}"))))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        concat!(
            "{\n",
            "  \"schema\": \"org.luminousdynamics.symthaea.qcomp.qoblib-solution-verification.v1\",\n",
            "  \"benchmark_track\": \"published-solution-feasibility-audit\",\n",
            "  \"graph_path\": \"{}\",\n",
            "  \"graph_bytes\": {},\n",
            "  \"graph_fnv1a64\": \"{:016x}\",\n",
            "  \"solution_path\": \"{}\",\n",
            "  \"solution_bytes\": {},\n",
            "  \"solution_fnv1a64\": \"{:016x}\",\n",
            "  \"vertices\": {},\n",
            "  \"edges\": {},\n",
            "  \"selected_count\": {},\n",
            "  \"computed_objective\": {},\n",
            "  \"reported_objective\": {},\n",
            "  \"feasible\": {},\n",
            "  \"violations\": [{}],\n",
            "  \"limitations\": [\"FNV-1a is a non-cryptographic integrity checksum\", \"feasibility does not prove optimality\", \"this command verifies a solution file but does not reproduce the submitted algorithm or its performance measurements\"]\n",
            "}"
        ),
        json_escape(graph_path),
        graph_source.len(),
        fnv1a64(graph_source.as_bytes()),
        json_escape(solution_path),
        solution_source.len(),
        fnv1a64(solution_source.as_bytes()),
        parsed.declared_vertex_count,
        parsed.declared_edge_count,
        selected.len(),
        report.computed_objective,
        reported_objective.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned()),
        report.feasible,
        violations
    );
    Ok(report.feasible)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--verify-solution") {
        if args.len() != 4 && args.len() != 5 {
            eprintln!("Usage: qoblib-mis-smoke --verify-solution <graph-path> <solution-path> [reported-objective]");
            return ExitCode::from(2);
        }
        let reported_objective = if args.len() == 5 {
            match args[4].parse::<usize>() {
                Ok(value) => Some(value),
                Err(_) => {
                    eprintln!("reported-objective must be a nonnegative integer");
                    return ExitCode::from(2);
                }
            }
        } else {
            None
        };
        return match run_verify_solution(&args[2], &args[3], reported_objective) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        };
    }
    let mut path: Option<String> = None;
    let mut seed_base = MULTISTART_GREEDY_SEED_BASE;
    let mut target_objective: Option<usize> = None;
    let mut seed_base_seen = false;
    let mut target_objective_seen = false;
    let mut index = 1usize;

    while index < args.len() {
        match args[index].as_str() {
            "--help" | "-h" => {
                eprintln!("Usage: qoblib-mis-smoke [--seed-base <u64>] [--target-objective <usize>] <path-to-DIMACS-edge-graph>\n       qoblib-mis-smoke --verify-solution <graph-path> <solution-path> [reported-objective]");
                eprintln!("Runs one deterministic greedy control, 16 seeded randomized greedy trials, bounded local improvement, and the tiny exact reference when within its vertex limit.");
                return ExitCode::SUCCESS;
            }
            "--seed-base" => {
                if seed_base_seen {
                    eprintln!("--seed-base may be supplied only once");
                    return ExitCode::from(2);
                }
                seed_base_seen = true;
                if index + 1 >= args.len() {
                    eprintln!("--seed-base requires an unsigned 64-bit integer");
                    return ExitCode::from(2);
                }
                match args[index + 1].parse::<u64>() {
                    Ok(seed) => seed_base = seed,
                    Err(_) => {
                        eprintln!("invalid --seed-base value: {:?}", args[index + 1]);
                        return ExitCode::from(2);
                    }
                }
                index += 2;
            }
            "--target-objective" => {
                if target_objective_seen {
                    eprintln!("--target-objective may be supplied only once");
                    return ExitCode::from(2);
                }
                target_objective_seen = true;
                if index + 1 >= args.len() {
                    eprintln!("--target-objective requires a nonnegative integer");
                    return ExitCode::from(2);
                }
                match args[index + 1].parse::<usize>() {
                    Ok(target) => target_objective = Some(target),
                    Err(_) => {
                        eprintln!("invalid --target-objective value: {:?}", args[index + 1]);
                        return ExitCode::from(2);
                    }
                }
                index += 2;
            }
            argument if argument.starts_with("--") => {
                eprintln!("unknown option: {argument}");
                eprintln!("Usage: qoblib-mis-smoke [--seed-base <u64>] [--target-objective <usize>] <path-to-DIMACS-edge-graph>");
                return ExitCode::from(2);
            }
            argument => {
                if path.is_some() {
                    eprintln!("only one DIMACS graph path may be supplied");
                    return ExitCode::from(2);
                }
                path = Some(argument.to_owned());
                index += 1;
            }
        }
    }

    let Some(path) = path else {
        eprintln!("Usage: qoblib-mis-smoke [--seed-base <u64>] [--target-objective <usize>] <path-to-DIMACS-edge-graph>");
        return ExitCode::from(2);
    };

    match run(&path, seed_base, target_objective) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}", error.message);
            println!("{}", failure_receipt_json(&path, &error, seed_base, target_objective));
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_solution_mode_checks_feasibility_and_reported_objective() {
        let graph = "p edge 4 2\n e 1 2\n e 3 4\n";
        let good = "# Objective value = 2\n1\n3\n";
        let (parsed, selected, report) = verify_solution_text(graph, good, Some(2)).unwrap();
        assert_eq!(parsed.declared_vertex_count, 4);
        assert_eq!(selected, vec![0, 2]);
        assert!(report.feasible);
        assert_eq!(report.computed_objective, 2);

        let mismatch = verify_solution_text(graph, good, Some(3)).unwrap();
        assert!(!mismatch.2.feasible);
        assert!(mismatch.2.violations.iter().any(|v| matches!(
            v,
            symthaea_quantum_comp::mis_verifier::MisViolation::ObjectiveMismatch {
                reported: 3,
                computed: 2
            }
        )));
    }

    #[test]
    fn published_solution_mode_rejects_adjacent_selected_vertices() {
        let graph = "p edge 2 1\ne 1 2\n";
        let bad = "1\n2\n";
        let (_, _, report) = verify_solution_text(graph, bad, Some(2)).unwrap();
        assert!(!report.feasible);
        assert!(report.violations.iter().any(|v| matches!(
            v,
            symthaea_quantum_comp::mis_verifier::MisViolation::AdjacentSelectedVertices { .. }
        )));
    }

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

    #[test]
    fn multi_start_greedy_portfolio_is_seeded_replayable_and_feasible() {
        let graph = MisGraph::new(7, [
            (0, 3), (0, 4), (1, 3), (1, 5), (2, 4), (2, 6), (3, 5), (4, 6),
        ]).unwrap();
        let first = multi_start_greedy(&graph, MULTISTART_GREEDY_RUNS, MULTISTART_GREEDY_SEED_BASE);
        let second = multi_start_greedy(&graph, MULTISTART_GREEDY_RUNS, MULTISTART_GREEDY_SEED_BASE);
        assert_eq!(first.best_vertices, second.best_vertices);
        assert_eq!(first.best_seed, second.best_seed);
        assert_eq!(first.objectives, second.objectives);
        assert_eq!(first.distinct_solutions, second.distinct_solutions);
        assert_eq!(first.objectives.len(), MULTISTART_GREEDY_RUNS);
        assert_eq!(first.objective_time_series.len(), MULTISTART_GREEDY_RUNS);
        for (trace, objective) in first.objective_time_series.iter().zip(first.objectives.iter()) {
            assert_eq!(trace.first().map(|point| point.1), Some(0));
            assert_eq!(trace.last().map(|point| point.1), Some(*objective));
            assert!(trace.windows(2).all(|pair| pair[0].1 <= pair[1].1));
            assert!(trace.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        }
        assert!(first.best_vertices.len() <= graph.vertex_count());
        assert!(graph.verify_solution(&first.best_vertices, Some(first.best_vertices.len())).feasible);
        assert!(first.distinct_solutions >= 1);
    }

    #[test]
    fn one_for_two_exchange_improves_a_feasible_greedy_set() {
        // Vertices 0 and 1 are the deterministic greedy result; removing 0 permits
        // adding mutually nonadjacent vertices 4 and 5.
        let graph = MisGraph::new(6, [
            (0, 4), (0, 5), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5),
        ]).unwrap();
        let greedy = greedy_min_residual_degree(&graph);
        assert_eq!(greedy, vec![0, 1]);
        let improved = improve_one_for_two(&graph, &greedy, LOCAL_SEARCH_MAX_PAIR_CHECKS);
        assert!(improved.locally_optimal);
        assert_eq!(improved.vertices, vec![1, 4, 5]);
        assert_eq!(improved.exchanges, 1);
        assert_eq!(improved.objective_time_series.first().map(|point| point.1), Some(2));
        assert_eq!(improved.objective_time_series.last().map(|point| point.1), Some(3));
        assert!(improved.objective_time_series.windows(2).all(|pair| pair[0].1 <= pair[1].1));
        assert!(improved.objective_time_series.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        assert!(graph.verify_solution(&improved.vertices, Some(3)).feasible);
    }

    #[test]
    fn local_improvement_budget_exhaustion_is_explicit_and_keeps_feasibility() {
        let graph = MisGraph::new(6, [
            (0, 4), (0, 5), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5),
        ]).unwrap();
        let greedy = greedy_min_residual_degree(&graph);
        let improved = improve_one_for_two(&graph, &greedy, 0);
        assert!(!improved.locally_optimal);
        assert_eq!(improved.exchanges, 0);
        assert_eq!(improved.pair_checks, 0);
        assert!(graph.verify_solution(&improved.vertices, Some(improved.vertices.len())).feasible);
    }


    #[test]
    fn failure_receipt_preserves_parse_failure_and_source_identity() {
        let failure = RunFailure {
            kind: "dimacs_parse",
            message: "bad \"header\"\nline".to_owned(),
            source_bytes: Some(42),
            source_fnv1a64: Some(0x0123_4567_89ab_cdef),
            elapsed_ns: 17,
        };
        let receipt = failure_receipt_json("graphs/farm\n.gph", &failure, 991, Some(10));
        assert!(receipt.contains("\"return_status\": \"failed\""));
        assert!(receipt.contains("\"error_kind\": \"dimacs_parse\""));
        assert!(receipt.contains("\"source_bytes\": 42"));
        assert!(receipt.contains("\"source_fnv1a64\": \"0123456789abcdef\""));
        assert!(receipt.contains("\"error\": \"bad \\\"header\\\"\\nline\""));
        assert!(receipt.contains("\"source_path\": \"graphs/farm\\n.gph\""));
        assert!(receipt.contains("\"time_limit_ns\": null"));
        assert!(receipt.contains("\"seed_base\": \"991\""));
        assert!(receipt.contains("\"target_objective\": 10"));
        assert!(receipt.contains("\"restarts\": 16"));
        assert!(receipt.contains("\"determinism\": \"not-attested: failed run; inspect error_kind and source identity\""));
    }


    #[test]
    fn qoblib_time_series_serializer_emits_runs_and_seconds_schema() {
        let json = qoblib_many_runs_series_json(&[vec![(0_u128, 0_usize), (1_500_000_000, 1)]]);
        assert_eq!(
            json,
            r#"[[{"Time": 0.000000000, "Incumbent": 0},{"Time": 1.500000000, "Incumbent": 1}]]"#
        );
    }


    #[test]
    fn fixed_target_summary_preserves_hits_and_unreached_trials() {
        let runs = vec![
            vec![(0_u128, 0_usize), (1_000, 1), (2_000, 2)],
            vec![(0_u128, 0_usize), (500, 1)],
        ];
        assert_eq!(
            target_trial_summary_json(2, &runs),
            r#"{"trial_count": 2, "target_hits": 1, "empirical_hit_rate": 0.500000, "time_to_target_seconds_by_trial": [0.000002000,null]}"#
        );
        let analysis = target_analysis_json(Some(2), &runs[0], &runs, None);
        assert!(analysis.contains("\"target_objective\": 2"));
        assert!(analysis.contains("\"time_to_target_seconds_by_trial\": [0.000002000,null]"));
        assert!(analysis.ends_with("\"local_improvement_stage\": null}"));
        assert_eq!(target_analysis_json(None, &runs[0], &runs, None), "null");
    }

}
