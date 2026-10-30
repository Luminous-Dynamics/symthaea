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

#[derive(Debug, Deserialize, Serialize)]
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
    review_budgets_minutes: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    source: u64,
    target: u64,
    weight: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    weight: f64,
    normalization_scale: f64,
    #[serde(default)]
    reference_value: Option<f64>,
    metric: QueryMetric,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum QueryMetric {
    EdgeWeight { source: u64, target: u64 },
    InDegree { node: u64 },
    OutDegree { node: u64 },
    PathExists { source: u64, target: u64 },
    TriadMotifCount { motif: TriadMotif },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum TriadMotif {
    FeedForwardLoop,
    DirectedThreeCycle,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    baseline_graph_canonical_sha256: String,
    query_bundle_sha256: String,
    graph_edge_count: usize,
    query_count: usize,
    candidate_count: usize,
    scoring_semantics: &'static str,
    candidate_evaluations: Vec<CandidateEvaluation>,
    policy_budget_results: Vec<PolicyBudgetResult>,
    warnings: Vec<String>,
    status: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct PolicyBudgetResult {
    policy: &'static str,
    review_budget_minutes: f64,
    actual_review_minutes: f64,
    reviewed_candidate_count: usize,
    applied_candidate_count: usize,
    rejected_conflict_count: usize,
    reviewed_candidate_ids: Vec<String>,
    applied_candidate_ids: Vec<String>,
    rejected_conflict_candidate_ids: Vec<String>,
    baseline_weighted_query_error: Option<f64>,
    final_weighted_query_error: Option<f64>,
    weighted_query_error_reduction: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CandidateEvaluation {
    rank: usize,
    candidate_id: String,
    evidence_sha256: String,
    estimated_error_probability: f64,
    review_minutes: f64,
    weighted_query_sensitivity: f64,
    reference_query_count: usize,
    weighted_query_error_reduction: Option<f64>,
    probability_weighted_query_sensitivity: f64,
    probability_weighted_query_sensitivity_per_review_minute: f64,
    counterfactual_graph_canonical_sha256: String,
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
    reference_value: Option<f64>,
    baseline_absolute_error: Option<f64>,
    counterfactual_absolute_error: Option<f64>,
    normalized_query_error_reduction: Option<f64>,
}

type Graph = BTreeMap<(u64, u64), f64>;

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn canonical_graph_sha256(graph: &Graph) -> String {
    let mut hasher = Sha256::new();
    for ((source, target), weight) in graph {
        hasher.update(source.to_be_bytes());
        hasher.update(target.to_be_bytes());
        hasher.update(weight.to_bits().to_be_bytes());
    }
    hex::encode(hasher.finalize())
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
    if input.queries.iter().any(|query| matches!(&query.metric, QueryMetric::TriadMotifCount { .. }))
        && input.graph.len() > 4096
    {
        return Err("triad motif queries are limited to frozen subgraphs with at most 4096 directed edges in schema v1".into());
    }
    if input.queries.is_empty() {
        return Err("queries must contain at least one declared scientific query".into());
    }
    if input.candidates.is_empty() {
        return Err("candidates must contain at least one proposed correction".into());
    }
    if input.review_budgets_minutes.is_empty() {
        return Err("review_budgets_minutes must contain at least one matched-budget evaluation".into());
    }
    let mut unique_budgets = BTreeSet::new();
    for budget in &input.review_budgets_minutes {
        if !budget.is_finite() || *budget <= 0.0 {
            return Err("review_budgets_minutes values must be finite and positive".into());
        }
        if !unique_budgets.insert(budget.to_bits()) {
            return Err("review_budgets_minutes values must be unique".into());
        }
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
        if query.reference_value.is_some_and(|value| !value.is_finite()) {
            return Err(format!("query {} reference_value must be finite when provided", query.query_id));
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

fn weighted_triad_motif_count(graph: &Graph, motif: &TriadMotif) -> f64 {
    let mut adjacency = BTreeMap::<u64, Vec<(u64, f64)>>::new();
    for ((source, target), weight) in graph {
        adjacency.entry(*source).or_default().push((*target, *weight));
    }
    let mut total = 0.0;
    for ((source, middle), first_weight) in graph {
        if let Some(second_edges) = adjacency.get(middle) {
            for (target, second_weight) in second_edges {
                let closing_edge = match motif {
                    TriadMotif::FeedForwardLoop => (*source, *target),
                    TriadMotif::DirectedThreeCycle => (*target, *source),
                };
                if let Some(closing_weight) = graph.get(&closing_edge) {
                    total += *first_weight * *second_weight * *closing_weight;
                }
            }
        }
    }
    if matches!(motif, TriadMotif::DirectedThreeCycle) {
        total / 3.0
    } else {
        total
    }
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
        QueryMetric::TriadMotifCount { motif } => weighted_triad_motif_count(graph, motif),
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
    let mut reference_weighted_sum = 0.0;
    let mut reference_weight_total = 0.0;
    let mut reference_query_count = 0;
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
        let (baseline_absolute_error, counterfactual_absolute_error, normalized_query_error_reduction) =
            if let Some(reference) = query.reference_value {
                let baseline_error = (baseline_value - reference).abs();
                let counterfactual_error = (counterfactual_value - reference).abs();
                let normalized_reduction =
                    ((baseline_error - counterfactual_error) / query.normalization_scale).clamp(-1.0, 1.0);
                reference_query_count += 1;
                reference_weight_total += query.weight;
                reference_weighted_sum += query.weight * normalized_reduction;
                (Some(baseline_error), Some(counterfactual_error), Some(normalized_reduction))
            } else {
                (None, None, None)
            };
        query_impacts.push(QueryImpact {
            query_id: query.query_id.clone(),
            baseline_value,
            counterfactual_value,
            delta,
            normalized_absolute_delta,
            weighted_contribution,
            reference_value: query.reference_value,
            baseline_absolute_error,
            counterfactual_absolute_error,
            normalized_query_error_reduction,
        });
    }
    let weighted_query_sensitivity = weighted_sum / total_query_weight;
    let probability_weighted_query_sensitivity =
        candidate.estimated_error_probability * weighted_query_sensitivity;
    let probability_weighted_query_sensitivity_per_review_minute =
        probability_weighted_query_sensitivity / candidate.review_minutes;
    if !probability_weighted_query_sensitivity_per_review_minute.is_finite() {
        return Err(format!("candidate {} produced a non-finite priority score", candidate.candidate_id));
    }
    Ok(CandidateEvaluation {
        rank: 0,
        candidate_id: candidate.candidate_id.clone(),
        evidence_sha256: candidate.evidence_sha256.clone(),
        estimated_error_probability: candidate.estimated_error_probability,
        review_minutes: candidate.review_minutes,
        weighted_query_sensitivity,
        reference_query_count,
        weighted_query_error_reduction: (reference_weight_total > 0.0)
            .then_some(reference_weighted_sum / reference_weight_total),
        probability_weighted_query_sensitivity,
        probability_weighted_query_sensitivity_per_review_minute,
        counterfactual_graph_canonical_sha256: canonical_graph_sha256(&counterfactual),
        query_impacts,
        overlaps_with_candidates: overlaps.to_vec(),
        recommendation: "HUMAN_REVIEW_ONLY",
    })
}

#[derive(Debug, Clone, Copy)]
enum RankingPolicy {
    QuerySensitivity,
    ErrorProbability,
    Uncertainty,
    LocalDegreeDelta,
    SensitivityOnly,
    DeterministicRandom,
}

impl RankingPolicy {
    fn id(self) -> &'static str {
        match self {
            Self::QuerySensitivity => "query_sensitivity_per_minute",
            Self::ErrorProbability => "error_probability_per_minute",
            Self::Uncertainty => "uncertainty_per_minute",
            Self::LocalDegreeDelta => "local_weighted_degree_delta_per_minute",
            Self::SensitivityOnly => "query_sensitivity_only_per_minute",
            Self::DeterministicRandom => "deterministic_random",
        }
    }
}

fn deterministic_random_score(input_hash: &str, candidate_id: &str) -> u64 {
    let digest = Sha256::digest(format!("{input_hash}:{candidate_id}").as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 has at least eight bytes"))
}

fn weighted_degree_signature(graph: &Graph) -> Result<BTreeMap<u64, (f64, f64)>, String> {
    let mut degrees = BTreeMap::<u64, (f64, f64)>::new();
    for ((source, target), weight) in graph {
        degrees.entry(*source).or_default().1 += *weight;
        degrees.entry(*target).or_default().0 += *weight;
    }
    if degrees.values().any(|(in_degree, out_degree)| !in_degree.is_finite() || !out_degree.is_finite()) {
        return Err("weighted degree accumulation overflowed".into());
    }
    Ok(degrees)
}

fn local_weighted_degree_delta(graph: &Graph, candidate: &Candidate) -> Result<f64, String> {
    let counterfactual = apply_candidate(graph, candidate)?;
    let before = weighted_degree_signature(graph)?;
    let after = weighted_degree_signature(&counterfactual)?;
    let nodes: BTreeSet<u64> = before.keys().chain(after.keys()).copied().collect();
    let mut total_delta = 0.0;
    for node in nodes {
        let before_degree = before.get(&node).copied().unwrap_or_default();
        let after_degree = after.get(&node).copied().unwrap_or_default();
        total_delta += (after_degree.0 - before_degree.0).abs();
        total_delta += (after_degree.1 - before_degree.1).abs();
    }
    if !total_delta.is_finite() {
        return Err("local weighted-degree delta overflowed".into());
    }
    Ok(total_delta)
}

fn policy_score(
    graph: &Graph,
    policy: RankingPolicy,
    candidate: &Candidate,
    evaluation: &CandidateEvaluation,
    input_hash: &str,
) -> Result<f64, String> {
    let score = match policy {
        RankingPolicy::QuerySensitivity => {
            evaluation.probability_weighted_query_sensitivity_per_review_minute
        }
        RankingPolicy::ErrorProbability => {
            candidate.estimated_error_probability / candidate.review_minutes
        }
        RankingPolicy::Uncertainty => {
            (1.0 - (2.0 * candidate.estimated_error_probability - 1.0).abs())
                / candidate.review_minutes
        }
        RankingPolicy::LocalDegreeDelta => {
            local_weighted_degree_delta(graph, candidate)? / candidate.review_minutes
        }
        RankingPolicy::SensitivityOnly => {
            evaluation.weighted_query_sensitivity / candidate.review_minutes
        }
        RankingPolicy::DeterministicRandom => {
            deterministic_random_score(input_hash, &candidate.candidate_id) as f64
        }
    };
    if !score.is_finite() {
        return Err(format!("policy {} produced a non-finite score for {}", policy.id(), candidate.candidate_id));
    }
    Ok(score)
}

fn ranked_candidate_indices(
    graph: &Graph,
    policy: RankingPolicy,
    candidates: &[Candidate],
    evaluations: &[CandidateEvaluation],
    input_hash: &str,
) -> Result<Vec<usize>, String> {
    let evaluation_by_id: BTreeMap<&str, &CandidateEvaluation> = evaluations
        .iter()
        .map(|evaluation| (evaluation.candidate_id.as_str(), evaluation))
        .collect();
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    let mut scored = Vec::with_capacity(candidates.len());
    for index in indices.drain(..) {
        let candidate = &candidates[index];
        let evaluation = evaluation_by_id
            .get(candidate.candidate_id.as_str())
            .ok_or_else(|| format!("candidate {} has no sensitivity evaluation", candidate.candidate_id))?;
        scored.push((
            index,
            policy_score(graph, policy, candidate, evaluation, input_hash)?,
            candidate.candidate_id.as_str(),
        ));
    }
    scored.sort_by(|left, right| {
        right.1.total_cmp(&left.1).then_with(|| left.2.cmp(right.2))
    });
    Ok(scored.into_iter().map(|(index, _, _)| index).collect())
}

fn weighted_query_error(graph: &Graph, queries: &[Query]) -> Result<Option<f64>, String> {
    let mut weighted_error = 0.0;
    let mut total_weight = 0.0;
    for query in queries {
        if let Some(reference) = query.reference_value {
            let actual = query_value(graph, &query.metric);
            if !actual.is_finite() {
                return Err(format!("query {} produced a non-finite oracle-evaluation value", query.query_id));
            }
            let normalized_error = ((actual - reference).abs() / query.normalization_scale).min(1.0);
            weighted_error += query.weight * normalized_error;
            total_weight += query.weight;
        }
    }
    if total_weight <= 0.0 {
        Ok(None)
    } else {
        Ok(Some(weighted_error / total_weight))
    }
}

fn replay_policy_budget(
    graph: &Graph,
    queries: &[Query],
    candidates: &[Candidate],
    evaluations: &[CandidateEvaluation],
    policy: RankingPolicy,
    budget: f64,
    input_hash: &str,
) -> Result<PolicyBudgetResult, String> {
    let order = ranked_candidate_indices(graph, policy, candidates, evaluations, input_hash)?;
    let baseline_error = weighted_query_error(graph, queries)?;
    let mut current_graph = graph.clone();
    let mut actual_minutes = 0.0;
    let mut reviewed_ids = Vec::new();
    let mut applied_ids = Vec::new();
    let mut rejected_ids = Vec::new();

    for index in order {
        let candidate = &candidates[index];
        if actual_minutes + candidate.review_minutes > budget + 1e-9 {
            continue;
        }
        actual_minutes += candidate.review_minutes;
        reviewed_ids.push(candidate.candidate_id.clone());
        match apply_candidate(&current_graph, candidate) {
            Ok(next_graph) => {
                current_graph = next_graph;
                applied_ids.push(candidate.candidate_id.clone());
            }
            Err(_) => rejected_ids.push(candidate.candidate_id.clone()),
        }
    }

    let final_error = weighted_query_error(&current_graph, queries)?;
    let error_reduction = match (baseline_error, final_error) {
        (Some(before), Some(after)) => Some(before - after),
        _ => None,
    };
    Ok(PolicyBudgetResult {
        policy: policy.id(),
        review_budget_minutes: budget,
        actual_review_minutes: actual_minutes,
        reviewed_candidate_count: reviewed_ids.len(),
        applied_candidate_count: applied_ids.len(),
        rejected_conflict_count: rejected_ids.len(),
        reviewed_candidate_ids: reviewed_ids,
        applied_candidate_ids: applied_ids,
        rejected_conflict_candidate_ids: rejected_ids,
        baseline_weighted_query_error: baseline_error,
        final_weighted_query_error: final_error,
        weighted_query_error_reduction: error_reduction,
    })
}

fn evaluate_policy_budgets(
    graph: &Graph,
    queries: &[Query],
    candidates: &[Candidate],
    evaluations: &[CandidateEvaluation],
    budgets: &[f64],
    input_hash: &str,
) -> Result<Vec<PolicyBudgetResult>, String> {
    let policies = [
        RankingPolicy::QuerySensitivity,
        RankingPolicy::ErrorProbability,
        RankingPolicy::Uncertainty,
        RankingPolicy::LocalDegreeDelta,
        RankingPolicy::SensitivityOnly,
        RankingPolicy::DeterministicRandom,
    ];
    let mut results = Vec::with_capacity(budgets.len() * policies.len());
    for budget in budgets {
        for policy in policies {
            results.push(replay_policy_budget(
                graph, queries, candidates, evaluations, policy, *budget, input_hash,
            )?);
        }
    }
    Ok(results)
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
            .probability_weighted_query_sensitivity_per_review_minute
            .total_cmp(&left.probability_weighted_query_sensitivity_per_review_minute)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    for (index, evaluation) in evaluations.iter_mut().enumerate() {
        evaluation.rank = index + 1;
    }
    let policy_budget_results = evaluate_policy_budgets(
        &graph,
        &input.queries,
        &input.candidates,
        &evaluations,
        &input.review_budgets_minutes,
        &input_hash,
    )?;

    let mut warnings = vec![
        "Candidate probabilities are caller-supplied and are not calibrated by this tool.".to_string(),
        "Single-candidate sensitivity scores are computed independently against the same baseline graph and are not additive; matched-budget replay separately applies selected valid proposals sequentially on the evolving graph.".to_string(),
        "Query weights and normalization scales are scientific-policy inputs and must be preregistered before evaluation.".to_string(),
        "A large query delta measures sensitivity of the declared query suite, not biological correctness or the truth of the proposed edit.".to_string(),
    ];
    if evaluations.iter().any(|evaluation| !evaluation.overlaps_with_candidates.is_empty()) {
        warnings.push("At least two candidates touch the same baseline edge; review their interaction before treating queue scores as independent.".to_string());
    }
    if !input.queries.iter().any(|query| query.reference_value.is_some_and(|_| query.weight > 0.0)) {
        warnings.push("No positive-weight query oracle values are present; matched-budget policy replay is descriptive only and query-error reductions are not estimable.".to_string());
    }
    warnings.push("Matched-budget replay is retrospective and assumes every selected proposal that remains valid on the evolving graph is accepted; rejected conflicts still consume review time. It does not model human rejection or evidence acquired during review.".to_string());
    warnings.sort();

    Ok(Report {
        schema_version: 1,
        tool: "connectome-query-impact",
        input_sha256: input_hash,
        dataset_id: input.dataset_id.clone(),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        source_graph_sha256: input.source_graph_sha256.clone(),
        baseline_graph_canonical_sha256: canonical_graph_sha256(&graph),
        query_bundle_sha256: input.query_bundle_sha256.clone(),
        graph_edge_count: input.graph.len(),
        query_count: input.queries.len(),
        candidate_count: input.candidates.len(),
        scoring_semantics: "probability_weighted_mean_normalized_query_sensitivity_per_review_minute",
        candidate_evaluations: evaluations,
        policy_budget_results,
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
                {"query_id":"direct_1_to_3","weight":5.0,"normalization_scale":1.0,"metric":{"kind":"edge_weight","source":1,"target":3}},
                {"query_id":"input_3","weight":2.0,"normalization_scale":3.0,"metric":{"kind":"in_degree","node":3}},
                {"query_id":"output_1","weight":1.0,"normalization_scale":4.0,"metric":{"kind":"out_degree","node":1}},
                {"query_id":"reach_1_to_3","weight":3.0,"normalization_scale":1.0,"metric":{"kind":"path_exists","source":1,"target":3}}
            ],
            "review_budgets_minutes": [2.0, 3.5, 10.0],
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
                },
                {
                    "candidate_id":"remove_part_of_second_leg",
                    "estimated_error_probability":0.4,
                    "review_minutes":1.0,
                    "evidence_sha256":"f".repeat(64),
                    "changes":[{"operation":"remove_weight","source":2,"target":3,"weight":1.0}]
                }
            ]
        })).expect("fixture deserializes")
    }

    fn fixture_with_oracle() -> Input {
        let mut input = fixture();
        input.queries[0].reference_value = Some(0.0);
        input.queries[1].reference_value = Some(3.0);
        input.queries[2].reference_value = Some(4.0);
        input.queries[3].reference_value = Some(1.0);
        if input.queries.len() > 4 {
            input.queries[4].reference_value = Some(12.0);
        }
        input
    }

    #[test]
    fn fixture_passes_schema_and_scoring() {
        let input = fixture();
        validate_input(&input).expect("fixture validates");
        let report = score(&input, "f".repeat(64)).expect("fixture scores");
        assert_eq!(report.status, "RANKED_FOR_HUMAN_REVIEW_ONLY");
        assert_eq!(report.candidate_count, 3);
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
        assert_eq!(evaluation.weighted_query_sensitivity, 0.0);
        assert_eq!(evaluation.probability_weighted_query_sensitivity_per_review_minute, 0.0);
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
        assert!(b.probability_weighted_query_sensitivity > a.probability_weighted_query_sensitivity);
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
        assert_eq!(a.probability_weighted_query_sensitivity_per_review_minute, 2.0 * b.probability_weighted_query_sensitivity_per_review_minute);
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
        assert_eq!(report.candidate_evaluations[1].candidate_id, "remove_part_of_second_leg");
        assert_eq!(report.candidate_evaluations[2].candidate_id, "remove_second_leg");
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
        assert!(evaluation.weighted_query_sensitivity > 0.0);
        assert_eq!(evaluation.probability_weighted_query_sensitivity, 0.0);
    }
    #[test]
    fn checked_in_query_impact_fixture_is_scored_and_flags_interacting_candidates() {
        let input = parse_input(include_str!("../../../../docs/neuroscience/examples/CONNECTOME_QUERY_IMPACT_SYNTHETIC.example.json").as_bytes())
            .expect("checked-in query-impact fixture parses");
        let report = score(&input, "0".repeat(64)).expect("query-impact fixture scores");
        assert_eq!(report.candidate_count, 3);
        assert!(input.queries.iter().any(|query| matches!(&query.metric, QueryMetric::TriadMotifCount { motif: TriadMotif::FeedForwardLoop })));
        assert!(report.candidate_evaluations.iter().any(|item| item.candidate_id == "remove_second_leg"
            && item.overlaps_with_candidates == vec!["remove_part_of_second_leg".to_string()]));
        assert_eq!(report.status, "RANKED_FOR_HUMAN_REVIEW_ONLY");
    }

    #[test]
    fn canonical_graph_digest_is_independent_of_input_edge_order() {
        let input = fixture();
        let forward = graph_from_edges(&input.graph);
        let mut reversed_edges = input.graph.clone();
        reversed_edges.reverse();
        let reversed = graph_from_edges(&reversed_edges);
        assert_eq!(canonical_graph_sha256(&forward), canonical_graph_sha256(&reversed));
    }

    #[test]
    fn counterfactual_graph_digest_changes_when_a_mutation_changes_graph() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let after = apply_candidate(&graph, &input.candidates[0]).unwrap();
        assert_ne!(canonical_graph_sha256(&graph), canonical_graph_sha256(&after));
    }

    #[test]
    fn oracle_separates_query_improvement_from_query_sensitivity() {
        let input = fixture_with_oracle();
        let graph = graph_from_edges(&input.graph);
        let total_weight: f64 = input.queries.iter().map(|query| query.weight).sum();
        let helpful = evaluate_candidate(&graph, &input.queries, &input.candidates[0], total_weight, &[]).unwrap();
        let harmful = evaluate_candidate(&graph, &input.queries, &input.candidates[1], total_weight, &[]).unwrap();
        assert_eq!(helpful.reference_query_count, 4);
        assert!(helpful.weighted_query_sensitivity > 0.0);
        assert!(helpful.weighted_query_error_reduction.unwrap() > 0.0);
        assert!(harmful.weighted_query_sensitivity > 0.0);
        assert!(harmful.weighted_query_error_reduction.unwrap() < 0.0);
    }

    #[test]
    fn absence_of_oracle_values_is_explicitly_not_estimable() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let total_weight: f64 = input.queries.iter().map(|query| query.weight).sum();
        let evaluation = evaluate_candidate(&graph, &input.queries, &input.candidates[0], total_weight, &[]).unwrap();
        assert_eq!(evaluation.reference_query_count, 0);
        assert_eq!(evaluation.weighted_query_error_reduction, None);
        assert!(evaluation.query_impacts.iter().all(|impact| impact.normalized_query_error_reduction.is_none()));
    }

    #[test]
    fn oracle_reference_values_do_not_change_candidate_priority() {
        let without_oracle = fixture();
        let with_oracle = fixture_with_oracle();
        let plain = score(&without_oracle, "a".repeat(64)).unwrap();
        let evaluated = score(&with_oracle, "b".repeat(64)).unwrap();
        let plain_order: Vec<_> = plain.candidate_evaluations.iter().map(|row| (&row.candidate_id, row.probability_weighted_query_sensitivity_per_review_minute)).collect();
        let oracle_order: Vec<_> = evaluated.candidate_evaluations.iter().map(|row| (&row.candidate_id, row.probability_weighted_query_sensitivity_per_review_minute)).collect();
        assert_eq!(plain_order, oracle_order);
    }

    #[test]
    fn rejects_non_finite_query_reference_values() {
        let mut input = fixture();
        input.queries[0].reference_value = Some(f64::INFINITY);
        assert!(validate_input(&input).unwrap_err().contains("reference_value"));
    }

    #[test]
    fn weighted_feed_forward_loop_count_is_product_of_edge_weights() {
        let graph = BTreeMap::from([
            ((1, 2), 2.0),
            ((2, 3), 3.0),
            ((1, 3), 4.0),
        ]);
        assert_eq!(
            weighted_triad_motif_count(&graph, &TriadMotif::FeedForwardLoop),
            24.0
        );
    }

    #[test]
    fn directed_three_cycle_count_corrects_three_rotations() {
        let graph = BTreeMap::from([
            ((1, 2), 2.0),
            ((2, 3), 3.0),
            ((3, 1), 4.0),
        ]);
        assert_eq!(
            weighted_triad_motif_count(&graph, &TriadMotif::DirectedThreeCycle),
            24.0
        );
    }

    #[test]
    fn feed_forward_loop_query_changes_when_a_closing_edge_is_removed() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let metric = QueryMetric::TriadMotifCount { motif: TriadMotif::FeedForwardLoop };
        assert_eq!(query_value(&graph, &metric), 12.0);
        let after = apply_candidate(&graph, &input.candidates[0]).unwrap();
        assert_eq!(query_value(&after, &metric), 0.0);
    }

    #[test]
    fn rejects_oversized_subgraphs_for_triad_motif_queries() {
        let mut input = fixture();
        input.queries.push(Query {
            query_id: "motif".into(),
            weight: 1.0,
            normalization_scale: 1.0,
            reference_value: None,
            metric: QueryMetric::TriadMotifCount { motif: TriadMotif::FeedForwardLoop },
        });
        input.graph = (0..4097).map(|index| Edge {
            source: index * 2,
            target: index * 2 + 1,
            weight: 1.0,
        }).collect();
        assert!(validate_input(&input).unwrap_err().contains("4096"));
    }

    #[test]
    fn bidirected_three_node_graph_counts_both_cycle_orientations_and_six_feedforward_orders() {
        let graph = BTreeMap::from([
            ((1, 2), 1.0),
            ((2, 1), 1.0),
            ((1, 3), 1.0),
            ((3, 1), 1.0),
            ((2, 3), 1.0),
            ((3, 2), 1.0),
        ]);
        assert_eq!(
            weighted_triad_motif_count(&graph, &TriadMotif::DirectedThreeCycle),
            2.0
        );
        assert_eq!(
            weighted_triad_motif_count(&graph, &TriadMotif::FeedForwardLoop),
            6.0
        );
    }

    #[test]
    fn matched_budget_policy_replay_never_exceeds_review_minutes() {
        let input = fixture_with_oracle();
        let report = score(&input, "a".repeat(64)).unwrap();
        assert_eq!(report.policy_budget_results.len(), 18);
        for result in &report.policy_budget_results {
            assert!(result.actual_review_minutes <= result.review_budget_minutes + 1e-9);
            assert_eq!(
                result.reviewed_candidate_count,
                result.applied_candidate_count + result.rejected_conflict_count
            );
        }
    }

    #[test]
    fn sequential_policy_replay_rejects_conflicting_edit_but_charges_review_time() {
        let input = fixture_with_oracle();
        let report = score(&input, "b".repeat(64)).unwrap();
        let replay = report.policy_budget_results.iter()
            .find(|result| result.policy == "query_sensitivity_per_minute"
                && result.review_budget_minutes == 10.0)
            .expect("query sensitivity 10-minute replay");
        assert_eq!(replay.reviewed_candidate_count, 3);
        assert_eq!(replay.applied_candidate_count, 2);
        assert_eq!(replay.rejected_conflict_count, 1);
        assert_eq!(replay.actual_review_minutes, 4.5);
        assert!(replay.weighted_query_error_reduction.is_some());
    }

    #[test]
    fn matched_budget_outcomes_are_not_estimable_without_oracle_values() {
        let input = fixture();
        let report = score(&input, "c".repeat(64)).unwrap();
        assert!(report.policy_budget_results.iter().all(|result|
            result.weighted_query_error_reduction.is_none()
                && result.baseline_weighted_query_error.is_none()
                && result.final_weighted_query_error.is_none()
        ));
        assert!(report.warnings.iter().any(|warning| warning.contains("not estimable")));
    }

    #[test]
    fn deterministic_random_policy_replays_for_frozen_input_and_seed() {
        let input = fixture_with_oracle();
        let first = score(&input, "random-seed".into()).unwrap();
        let second = score(&input, "random-seed".into()).unwrap();
        let first_random: Vec<_> = first.policy_budget_results.iter()
            .filter(|result| result.policy == "deterministic_random")
            .map(|result| (result.review_budget_minutes, result.reviewed_candidate_ids.clone()))
            .collect();
        let second_random: Vec<_> = second.policy_budget_results.iter()
            .filter(|result| result.policy == "deterministic_random")
            .map(|result| (result.review_budget_minutes, result.reviewed_candidate_ids.clone()))
            .collect();
        assert_eq!(first_random, second_random);
    }

    #[test]
    fn review_budgets_must_be_positive_and_unique() {
        let mut input = fixture();
        input.review_budgets_minutes = vec![1.0, 1.0];
        assert!(validate_input(&input).unwrap_err().contains("unique"));
        input.review_budgets_minutes = vec![0.0];
        assert!(validate_input(&input).unwrap_err().contains("positive"));
    }

    #[test]
    fn local_weighted_degree_delta_counts_source_and_target_effects() {
        let input = fixture();
        let graph = graph_from_edges(&input.graph);
        let delta = local_weighted_degree_delta(&graph, &input.candidates[0]).unwrap();
        assert_eq!(delta, 2.0);
    }

    #[test]
    fn local_weighted_degree_policy_is_separate_from_query_sensitivity() {
        let input = fixture_with_oracle();
        let graph = graph_from_edges(&input.graph);
        let evaluations = input.candidates.iter().map(|candidate| {
            evaluate_candidate(
                &graph,
                &input.queries,
                candidate,
                input.queries.iter().map(|query| query.weight).sum(),
                &[],
            ).unwrap()
        }).collect::<Vec<_>>();
        let query_order = ranked_candidate_indices(
            &graph, RankingPolicy::QuerySensitivity, &input.candidates, &evaluations, "seed"
        ).unwrap();
        let degree_order = ranked_candidate_indices(
            &graph, RankingPolicy::LocalDegreeDelta, &input.candidates, &evaluations, "seed"
        ).unwrap();
        assert_eq!(query_order.len(), 3);
        assert_eq!(degree_order.len(), 3);
    }

    #[test]
    fn replay_report_discloses_that_valid_proposals_are_assumed_accepted() {
        let input = fixture_with_oracle();
        let report = score(&input, "d".repeat(64)).unwrap();
        assert!(report.warnings.iter().any(|warning|
            warning.contains("assumes") && warning.contains("valid") && warning.contains("accepted")
        ));
    }

}
