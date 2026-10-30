//! Counterfactual query-impact scorer for human review prioritization.
//!
//! Each candidate is evaluated independently against one immutable baseline graph.
//! Scores are sensitivity diagnostics, not proof that a proposed edit is correct.
//! The probability input must be calibrated elsewhere; this binary does not train
//! or calibrate it, authorize edits, or compose interacting proposals.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-query-impact",
    version,
    about = "Rank proposed graph corrections by counterfactual scientific-query impact per review minute"
)]
struct Args {
    /// Frozen graph, query bundle, and candidate mutations in one JSON document.
    #[arg(long)]
    input: PathBuf,

    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    dataset_id: String,
    dataset_revision_commit_sha: String,
    source_graph_sha256: String,
    query_bundle_sha256: String,
    graph: Vec<Edge>,
    queries: Vec<Query>,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    source: u64,
    target: u64,
    weight: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    weight: f64,
    normalization_scale: f64,
    #[serde(flatten)]
    metric: QueryMetric,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum QueryMetric {
    EdgeWeight { source: u64, target: u64 },
    InDegree { node: u64 },
    OutDegree { node: u64 },
    PathExists { source: u64, target: u64 },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    candidate_id: String,
    estimated_error_probability: f64,
    review_minutes: f64,
    evidence_sha256: String,
    changes: Vec<EdgeChange>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum EdgeChange {
    AddWeight { source: u64, target: u64, weight: f64 },
    RemoveWeight { source: u64, target: u64, weight: f64 },
}

impl EdgeChange {
    fn edge_key(&self) -> (u64, u64) {
        match self {
            Self::AddWeight { source, target, .. }
            | Self::RemoveWeight { source, target, .. } => (*source, *target),
        }
    }

    fn weight(&self) -> f64 {
        match self {
            Self::AddWeight { weight, .. } | Self::RemoveWeight { weight, .. } => *weight,
        }
    }
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    input_sha256: String,
    dataset_id: String,
    dataset_revision_commit_sha: String,
    source_graph_sha256: String,
    query_bundle_sha256: String,
    graph_edge_count: usize,
    query_count: usize,
    candidate_count: usize,
    scoring_semantics: &'static str,
    candidate_evaluations: Vec<CandidateEvaluation>,
    warnings: Vec<String>,
    status: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct CandidateEvaluation {
    rank: usize,
    candidate_id: String,
    estimated_error_probability: f64,
    review_minutes: f64,
    weighted_query_impact: f64,
    expected_weighted_query_impact: f64,
    expected_query_impact_per_review_minute: f64,
    query_impacts: Vec<QueryImpact>,
    overlaps_with_candidates: Vec<String>,
    recommendation: &'static str,
}

#[derive(Debug, Serialize)]
struct QueryImpact {
    query_id: String,
    baseline_value: f64,
    counterfactual_value: f64,
    delta: f64,
    normalized_absolute_delta: f64,
    weighted_contribution: f64,
}

type Graph = BTreeMap<(u64, u64), f64>;

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_input(bytes: &[u8]) -> Result<Input, String> {
    let input = serde_json::from_slice::<Input>(bytes)
        .map_err(|error| format!("input is invalid or contains an unknown field: {error}"))?;
    validate_input(&input)?;
    Ok(input)
}

fn validate_input(input: &Input) -> Result<(), String> {
    if input.schema_version != 1 {
        return Err("schema_version must be 1".into());
    }
    if input.dataset_id.trim().is_empty() {
        return Err("dataset_id must be non-empty".into());
    }
    if !is_lower_hex(&input.dataset_revision_commit_sha, 40) {
        return Err("dataset_revision_commit_sha must be 40 lowercase hexadecimal characters".into());
    }
    for (field, value) in [
        ("source_graph_sha256", input.source_graph_sha256.as_str()),
        ("query_bundle_sha256", input.query_bundle_sha256.as_str()),
    ] {
        if !is_lower_hex(value, 64) {
            return Err(format!("{field} must be 64 lowercase hexadecimal characters"));
        }
    }
    if input.graph.is_empty() {
        return Err("graph must contain at least one edge".into());
    }
    if input.queries.is_empty() {
        return Err("queries must contain at least one declared scientific query".into());
    }
    if input.candidates.is_empty() {
        return Err("candidates must contain at least one proposed correction".into());
    }

    let mut graph_edges = BTreeSet::new();
    for edge in &input.graph {
        if edge.source == edge.target {
            return Err("graph self-edges are not supported by schema v1".into());
        }
        if !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err("graph edge weights must be finite and positive".into());
        }
        if !graph_edges.insert((edge.source, edge.target)) {
            return Err(format!(
                "duplicate graph edge {} -> {}; aggregate it before scoring",
                edge.source, edge.target
            ));
        }
    }

    let mut query_ids = BTreeSet::new();
    let mut total_query_weight = 0.0;
    for query in &input.queries {
        if query.query_id.trim().is_empty() || !query_ids.insert(query.query_id.clone()) {
            return Err("query IDs must be non-empty and unique".into());
        }
        if !query.weight.is_finite() || query.weight < 0.0 {
            return Err(format!("query {} weight must be finite and non-negative", query.query_id));
        }
        if !query.normalization_scale.is_finite() || query.normalization_scale <= 0.0 {
            return Err(format!("query {} normalization_scale must be finite and positive", query.query_id));
        }
        total_query_weight += query.weight;
        if !total_query_weight.is_finite() {
            return Err("sum of query weights overflowed".into());
        }
    }
    if total_query_weight <= 0.0 {
        return Err("at least one query must have a positive weight".into());
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in &input.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !candidate_ids.insert(candidate.candidate_id.clone())
        {
            return Err("candidate IDs must be non-empty and unique".into());
        }
        if !candidate.estimated_error_probability.is_finite()
            || !(0.0..=1.0).contains(&candidate.estimated_error_probability)
        {
            return Err(format!(
                "candidate {} estimated_error_probability must be in [0, 1]",
                candidate.candidate_id
            ));
        }
        if !candidate.review_minutes.is_finite() || candidate.review_minutes <= 0.0 {
            return Err(format!("candidate {} review_minutes must be finite and positive", candidate.candidate_id));
        }
        if !is_lower_hex(&candidate.evidence_sha256, 64) {
            return Err(format!("candidate {} evidence_sha256 must be 64 lowercase hexadecimal characters", candidate.candidate_id));
        }
        if candidate.changes.is_empty() {
            return Err(format!("candidate {} must declare at least one graph change", candidate.candidate_id));
        }
        for change in &candidate.changes {
            let (source, target) = change.edge_key();
            if source == target {
                return Err(format!("candidate {} contains an unsupported self-edge mutation", candidate.candidate_id));
            }
            if !change.weight().is_finite() || change.weight() <= 0.0 {
                return Err(format!("candidate {} mutation weights must be finite and positive", candidate.candidate_id));
            }
        }
    }

    Ok(())
}

fn graph_from_edges(edges: &[Edge]) -> Graph {
    edges
        .iter()
        .map(|edge| ((edge.source, edge.target), edge.weight))
        .collect()
}

fn apply_candidate(graph: &Graph, candidate: &Candidate) -> Result<Graph, String> {
    let mut counterfactual = graph.clone();
    for change in &candidate.changes {
        let key = change.edge_key();
        match change {
            EdgeChange::AddWeight { weight, .. } => {
                let next = counterfactual.get(&key).copied().unwrap_or(0.0) + weight;
                if !next.is_finite() {
                    return Err(format!("candidate {} causes edge-weight overflow", candidate.candidate_id));
                }
                counterfactual.insert(key, next);
            }
            EdgeChange::RemoveWeight { weight, .. } => {
                let current = counterfactual.get(&key).copied().unwrap_or(0.0);
                if *weight > current + 1e-9 {
                    return Err(format!(
                        "candidate {} removes weight {} from edge {} -> {} with only {} available",
                        candidate.candidate_id, weight, key.0, key.1, current
                    ));
                }
                let remaining = current - weight;
                if remaining <= 1e-9 {
                    counterfactual.remove(&key);
                } else {
                    counterfactual.insert(key, remaining);
                }
            }
        }
    }
    Ok(counterfactual)
}

fn path_exists(graph: &Graph, source: u64, target: u64) -> bool {
    if source == target {
        return true;
    }
    let mut adjacency = BTreeMap::<u64, Vec<u64>>::new();
    for ((from, to), weight) in graph {
        if *weight > 0.0 {
            adjacency.entry(*from).or_default().push(*to);
        }
    }
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::from([source]);
    visited.insert(source);
    while let Some(node) = queue.pop_front() {
        if let Some(neighbours) = adjacency.get(&node) {
            for neighbour in neighbours {
                if *neighbour == target {
                    return true;
                }
                if visited.insert(*neighbour) {
                    queue.push_back(*neighbour);
                }
            }
        }
    }
    false
}

fn query_value(graph: &Graph, metric: &QueryMetric) -> f64 {
    match metric {
        QueryMetric::EdgeWeight { source, target } => {
            graph.get(&(*source, *target)).copied().unwrap_or(0.0)
        }
        QueryMetric::InDegree { node } => graph
            .iter()
            .filter(|((_, target), _)| target == node)
            .map(|(_, weight)| *weight)
            .sum(),
        QueryMetric::OutDegree { node } => graph
            .iter()
            .filter(|((source, _), _)| source == node)
            .map(|(_, weight)| *weight)
            .sum(),
        QueryMetric::PathExists { source, target } => {
            if path_exists(graph, *source, *target) { 1.0 } else { 0.0 }
        }
    }
}

fn changed_edges(candidate: &Candidate) -> BTreeSet<(u64, u64)> {
    candidate.changes.iter().map(EdgeChange::edge_key).collect()
}

fn overlap_index(candidates: &[Candidate]) -> BTreeMap<String, Vec<String>> {
    let mut by_edge = BTreeMap::<(u64, u64), Vec<String>>::new();
    for candidate in candidates {
        for edge in changed_edges(candidate) {
            by_edge.entry(edge).or_default().push(candidate.candidate_id.clone());
        }
    }
    let mut overlaps = BTreeMap::<String, BTreeSet<String>>::new();
    for candidate_ids in by_edge.values().filter(|ids| ids.len() > 1) {
        for candidate_id in candidate_ids {
            for other in candidate_ids {
                if candidate_id != other {
                    overlaps
                        .entry(candidate_id.clone())
                        .or_default()
                        .insert(other.clone());
                }
            }
        }
    }
    overlaps
        .into_iter()
        .map(|(candidate, others)| (candidate, others.into_iter().collect()))
        .collect()
}

fn evaluate_candidate(
    graph: &Graph,
    queries: &[Query],
    candidate: &Candidate,
    total_query_weight: f64,
    overlaps: &[String],
) -> Result<CandidateEvaluation, String> {
    let counterfactual = apply_candidate(graph, candidate)?;
    let mut query_impacts = Vec::with_capacity(queries.len());
    let mut weighted_sum = 0.0;
    for query in queries {
        let baseline_value = query_value(graph, &query.metric);
        let counterfactual_value = query_value(&counterfactual, &query.metric);
        let delta = counterfactual_value - baseline_value;
        if !delta.is_finite() {
            return Err(format!("candidate {} produces a non-finite query delta", candidate.candidate_id));
        }
        let normalized_absolute_delta = (delta.abs() / query.normalization_scale).min(1.0);
        let weighted_contribution = query.weight * normalized_absolute_delta;
        weighted_sum += weighted_contribution;
        query_impacts.push(QueryImpact {
            query_id: query.query_id.clone(),
            baseline_value,
            counterfactual_value,
            delta,
            normalized_absolute_delta,
            weighted_contribution,
        });
    }
    let weighted_query_impact = weighted_sum / total_query_weight;
    let expected_weighted_query_impact =
        candidate.estimated_error_probability * weighted_query_impact;
    let expected_query_impact_per_review_minute =
        expected_weighted_query_impact / candidate.review_minutes;
    if !expected_query_impact_per_review_minute.is_finite() {
        return Err(format!("candidate {} produced a non-finite priority score", candidate.candidate_id));
    }
    Ok(CandidateEvaluation {
        rank: 0,
        candidate_id: candidate.candidate_id.clone(),
        estimated_error_probability: candidate.estimated_error_probability,
        review_minutes: candidate.review_minutes,
        weighted_query_impact,
        expected_weighted_query_impact,
        expected_query_impact_per_review_minute,
        query_impacts,
        overlaps_with_candidates: overlaps.to_vec(),
        recommendation: "HUMAN_REVIEW_ONLY",
    })
}

fn score(input: &Input, input_hash: String) -> Result<Report, String> {
    let graph = graph_from_edges(&input.graph);
    let total_query_weight: f64 = input.queries.iter().map(|query| query.weight).sum();
    let overlap_map = overlap_index(&input.candidates);
    let mut evaluations = Vec::with_capacity(input.candidates.len());
    for candidate in &input.candidates {
        let overlaps = overlap_map.get(&candidate.candidate_id).map(Vec::as_slice).unwrap_or(&[]);
        evaluations.push(evaluate_candidate(
            &graph,
            &input.queries,
            candidate,
            total_query_weight,
            overlaps,
        )?);
    }
    evaluations.sort_by(|left, right| {
        right
            .expected_query_impact_per_review_minute
            .total_cmp(&left.expected_query_impact_per_review_minute)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    for (index, evaluation) in evaluations.iter_mut().enumerate() {
        evaluation.rank = index + 1;
    }

    let mut warnings = vec![
        "Candidate probabilities are caller-supplied and are not calibrated by this tool.".to_string(),
        "Each candidate is evaluated independently against the same baseline graph; scores are not additive and interacting edits are not jointly simulated.".to_string(),
        "Query weights and normalization scales are scientific-policy inputs and must be preregistered before evaluation.".to_string(),
        "A large query delta measures sensitivity of the declared query suite, not biological correctness or the truth of the proposed edit.".to_string(),
    ];
    if evaluations.iter().any(|evaluation| !evaluation.overlaps_with_candidates.is_empty()) {
        warnings.push("At least two candidates touch the same baseline edge; review their interaction before treating queue scores as independent.".to_string());
    }
    warnings.sort();

    Ok(Report {
        schema_version: 1,
        tool: "connectome-query-impact",
        input_sha256: input_hash,
        dataset_id: input.dataset_id.clone(),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        source_graph_sha256: input.source_graph_sha256.clone(),
        query_bundle_sha256: input.query_bundle_sha256.clone(),
        graph_edge_count: input.graph.len(),
        query_count: input.queries.len(),
        candidate_count: input.candidates.len(),
        scoring_semantics: "probability_weighted_mean_normalized_query_delta_per_review_minute",
        candidate_evaluations: evaluations,
        warnings,
        status: "RANKED_FOR_HUMAN_REVIEW_ONLY",
        scope_note: "This tool evaluates predicted weighted-edge mutations against an explicit directed-graph query bundle. It does not infer corrections, calibrate probabilities, validate evidence, model morphology/segmentation geometry, evaluate actual post-review outcomes, or authorize annotation writes. Query impact is conditional on the frozen graph, query definitions, weights, scales, and predicted mutation. Every candidate remains human-review-only.",
    })
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    let bytes = fs::read(&args.input)
        .map_err(|error| format!("failed to read {}: {error}", args.input.display()))?;
    let input_hash = sha256(&bytes);
    let input = parse_input(&bytes)?;
    let report = score(&input, input_hash)?;
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))? + "\n";
    if let Some(output) = args.output {
        fs::write(&output, json.as_bytes())
            .map_err(|error| format!("failed to write {}: {error}", output.display()))?;
    } else {
        print!("{json}");
    }
    Ok(0)
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(code) => ExitCode::from(code as u8),
        Err(error) => {
            eprintln!("connectome-query-impact: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Input {
        serde_json::from_value(json!({
            "schema_version": 1,
            "dataset_id": "synthetic-connectome",
            "dataset_revision_commit_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "source_graph_sha256": "b".repeat(64),
            "query_bundle_sha256": "c".repeat(64),
            "graph": [
                {"source": 1, "target": 2, "weight": 4.0},
                {"source": 2, "target": 3, "weight": 3.0},
                {"source": 1, "target": 3, "weight": 1.0}
            ],
            "queries": [
                {"query_id":"direct_1_to_3","weight":5.0,"normalization_scale":1.0,"kind":"edge_weight","source":1,"target":3},
                {"query_id":"input_3","weight":2.0,"normalization_scale":3.0,"kind":"in_degree","node":3},
                {"query_id":"output_1","weight":1.0,"normalization_scale":4.0,"kind":"out_degree","node":1},
                {"query_id":"reach_1_to_3","weight":3.0,"normalization_scale":1.0,"kind":"path_exists","source":1,"target":3}
            ],
            "candidates": [
                {
                    "candidate_id":"remove_direct_edge",
                    "estimated_error_probability":0.8,
                    "review_minutes":2.0,
                    "evidence_sha256":"d".repeat(64),
                    "changes":[{"operation":"remove_weight","source":1,"target":3,"weight":1.0}]
                },
                {
                    "candidate_id":"remove_second_leg",
                    "estimated_error_probability":0.6,
                    "review_minutes":1.5,
                    "evidence_sha256":"e".repeat(64),
                    "changes":[{"operation":"remove_weight","source":2,"target":3,"weight":3.0}]
                }
            ]
        })).expect("fixture deserializes")
    }

    #[test]
    fn fixture_passes_schema_and_scoring() {
        let input = fixture();
        validate_input(&input).expect("fixture validates");
        let report = score(&input, "f".repeat(64)).expect("fixture scores");
        assert_eq!(report.status, "RANKED_FOR_HUMAN_REVIEW_ONLY");
        assert_eq!(report.candidate_count, 2);
        assert_eq!(report.candidate_evaluations[0].rank, 1);
    }

    #[test]
    fn direct_edge_removal_changes_edge_weight_and_out_degree() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let candidate = &input.candidates[0];
        let after = apply_candidate(&graph, candidate).unwrap();
        assert_eq!(query_value(&graph, &input.queries[0].metric), 1.0);
        assert_eq!(query_value(&after, &input.queries[0].metric), 0.0);
        assert_eq!(query_value(&graph, &input.queries[2].metric), 5.0);
        assert_eq!(query_value(&after, &input.queries[2].metric), 4.0);
    }

    #[test]
    fn path_query_is_unchanged_when_an_alternate_route_remains() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let after = apply_candidate(&graph, &input.candidates[0]).unwrap();
        assert_eq!(query_value(&graph, &input.queries[3].metric), 1.0);
        assert_eq!(query_value(&after, &input.queries[3].metric), 1.0);
    }

    #[test]
    fn second_leg_removal_changes_reachability() {
        let mut input = fixture();
        input.graph.retain(|edge| !(edge.source == 1 && edge.target == 3));
        let graph = graph_from_edges(&input.graph);
        let candidate = &input.candidates[1];
        let after = apply_candidate(&graph, candidate).unwrap();
        assert_eq!(query_value(&graph, &input.queries[3].metric), 1.0);
        assert_eq!(query_value(&after, &input.queries[3].metric), 0.0);
    }

    #[test]
    fn irrelevant_edge_mutation_has_zero_query_impact() {
        let mut input = fixture();
        input.candidates[0].changes = vec![EdgeChange::AddWeight {
            source: 8, target: 9, weight: 2.0,
        }];
        let graph = graph_from_edges(&input.graph);
        let total_weight: f64 = input.queries.iter().map(|query| query.weight).sum();
        let evaluation = evaluate_candidate(&graph, &input.queries, &input.candidates[0], total_weight, &[]).unwrap();
        assert_eq!(evaluation.weighted_query_impact, 0.0);
        assert_eq!(evaluation.expected_query_impact_per_review_minute, 0.0);
    }

    #[test]
    fn higher_probability_increases_expected_impact_when_other_inputs_match() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let total_weight: f64 = input.queries.iter().map(|query| query.weight).sum();
        let mut lower = input.candidates[0].clone();
        lower.estimated_error_probability = 0.2;
        let mut higher = lower.clone();
        higher.estimated_error_probability = 0.8;
        let a = evaluate_candidate(&graph, &input.queries, &lower, total_weight, &[]).unwrap();
        let b = evaluate_candidate(&graph, &input.queries, &higher, total_weight, &[]).unwrap();
        assert!(b.expected_weighted_query_impact > a.expected_weighted_query_impact);
    }

    #[test]
    fn shorter_review_time_increases_value_per_minute() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let total_weight: f64 = input.queries.iter().map(|query| query.weight).sum();
        let mut fast = input.candidates[0].clone();
        fast.review_minutes = 1.0;
        let mut slow = fast.clone();
        slow.review_minutes = 2.0;
        let a = evaluate_candidate(&graph, &input.queries, &fast, total_weight, &[]).unwrap();
        let b = evaluate_candidate(&graph, &input.queries, &slow, total_weight, &[]).unwrap();
        assert_eq!(a.expected_query_impact_per_review_minute, 2.0 * b.expected_query_impact_per_review_minute);
    }

    #[test]
    fn query_normalization_caps_each_contribution_at_one() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let after = apply_candidate(&graph, &input.candidates[0]).unwrap();
        let query = &input.queries[0];
        let delta = (query_value(&after, &query.metric) - query_value(&graph, &query.metric)).abs();
        assert_eq!((delta / query.normalization_scale).min(1.0), 1.0);
    }

    #[test]
    fn rejects_probability_outside_unit_interval() {
        let mut input = fixture();
        input.candidates[0].estimated_error_probability = 1.01;
        assert!(validate_input(&input).unwrap_err().contains("estimated_error_probability"));
    }

    #[test]
    fn rejects_duplicate_candidate_ids() {
        let mut input = fixture();
        input.candidates[1].candidate_id = input.candidates[0].candidate_id.clone();
        assert!(validate_input(&input).unwrap_err().contains("candidate IDs"));
    }

    #[test]
    fn rejects_duplicate_query_ids() {
        let mut input = fixture();
        input.queries[1].query_id = input.queries[0].query_id.clone();
        assert!(validate_input(&input).unwrap_err().contains("query IDs"));
    }

    #[test]
    fn rejects_duplicate_graph_edges() {
        let mut input = fixture();
        input.graph.push(input.graph[0].clone());
        assert!(validate_input(&input).unwrap_err().contains("duplicate graph edge"));
    }

    #[test]
    fn rejects_removal_that_would_make_edge_weight_negative() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let mut candidate = input.candidates[0].clone();
        candidate.changes = vec![EdgeChange::RemoveWeight {
            source: 1, target: 3, weight: 2.0,
        }];
        assert!(apply_candidate(&graph, &candidate).unwrap_err().contains("only 1 available"));
    }

    #[test]
    fn rejects_removal_of_nonexistent_edge() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let mut candidate = input.candidates[0].clone();
        candidate.changes = vec![EdgeChange::RemoveWeight {
            source: 8, target: 9, weight: 1.0,
        }];
        assert!(apply_candidate(&graph, &candidate).unwrap_err().contains("only 0 available"));
    }

    #[test]
    fn reports_candidate_conflicts_on_shared_edges() {
        let mut input = fixture();
        input.candidates[1].changes = vec![EdgeChange::RemoveWeight {
            source: 1, target: 3, weight: 0.5,
        }];
        let overlaps = overlap_index(&input.candidates);
        assert_eq!(overlaps["remove_direct_edge"], vec!["remove_second_leg"]);
        assert_eq!(overlaps["remove_second_leg"], vec!["remove_direct_edge"]);
    }

    #[test]
    fn deterministic_tie_break_uses_candidate_id() {
        let mut input = fixture();
        for candidate in &mut input.candidates {
            candidate.estimated_error_probability = 0.0;
        }
        let report = score(&input, "f".repeat(64)).unwrap();
        assert_eq!(report.candidate_evaluations[0].candidate_id, "remove_direct_edge");
        assert_eq!(report.candidate_evaluations[1].candidate_id, "remove_second_leg");
    }

    #[test]
    fn rejects_zero_total_query_weight() {
        let mut input = fixture();
        for query in &mut input.queries {
            query.weight = 0.0;
        }
        assert!(validate_input(&input).unwrap_err().contains("positive weight"));
    }

    #[test]
    fn unknown_top_level_fields_fail_closed() {
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["unreviewed_metadata"] = serde_json::json!(true);
        assert!(serde_json::from_value::<Input>(value).unwrap_err().to_string().contains("unknown field"));
    }

    #[test]
    fn candidate_probabilities_are_not_inferred_from_query_impact() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let total_weight: f64 = input.queries.iter().map(|query| query.weight).sum();
        let mut candidate = input.candidates[0].clone();
        candidate.estimated_error_probability = 0.0;
        let evaluation = evaluate_candidate(&graph, &input.queries, &candidate, total_weight, &[]).unwrap();
        assert!(evaluation.weighted_query_impact > 0.0);
        assert_eq!(evaluation.expected_weighted_query_impact, 0.0);
    }
}
