//! Exact small-instance benchmark for interacting connectome corrections.
//!
//! This is an evaluation tool, not a production queue or edit applier. It enumerates
//! candidate subsets (up to 20 proposals), applies each selected portfolio in a
//! deterministic canonical order, and measures held-out query error against an
//! evaluator-only reference. Candidate ranking baselines never consume references.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-portfolio-audit",
    version,
    about = "Evaluate joint proofreading portfolios against exact small-instance optima"
)]
struct Args {
    /// Frozen graph, query references, candidate mutations and review budgets.
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
    reference_value: f64,
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
    candidate_count: usize,
    subset_count_evaluated: u64,
    baseline_weighted_query_error: f64,
    pairwise_interactions: Vec<PairwiseInteraction>,
    budget_results: Vec<BudgetResult>,
    status: &'static str,
    ranking_information_flow: &'static str,
    optimization_scope: &'static str,
}

#[derive(Debug, Serialize)]
struct PairwiseInteraction {
    candidate_a: String,
    candidate_b: String,
    joint_utility: Option<f64>,
    utility_a: Option<f64>,
    utility_b: Option<f64>,
    interaction: Option<f64>,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct BudgetResult {
    review_budget_minutes: f64,
    exact_optimum: PortfolioResult,
    ranked_policies: Vec<PortfolioResult>,
}

#[derive(Debug, Serialize)]
struct PortfolioResult {
    policy: &'static str,
    selected_candidate_ids: Vec<String>,
    canonical_application_order: Vec<String>,
    actual_review_minutes: f64,
    baseline_weighted_query_error: f64,
    final_weighted_query_error: Option<f64>,
    utility: Option<f64>,
    regret_to_exact_optimum: Option<f64>,
    reverse_order_weighted_query_error: Option<f64>,
    order_sensitivity_absolute_error_delta: Option<f64>,
    status: &'static str,
}

type Graph = BTreeMap<(u64, u64), f64>;

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn graph_from_edges(edges: &[Edge]) -> Graph {
    edges
        .iter()
        .map(|edge| ((edge.source, edge.target), edge.weight))
        .collect()
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

fn validate(input: &Input) -> Result<(), String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if input.dataset_id.trim().is_empty() {
        return Err("dataset_id must be non-empty".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40) {
        return Err("dataset_revision_commit_sha must be 40 lowercase hexadecimal characters".into());
    }
    if !valid_hex(&input.source_graph_sha256, 64) {
        return Err("source_graph_sha256 must be 64 lowercase hexadecimal characters".into());
    }
    if input.graph.is_empty() {
        return Err("graph must contain at least one edge".into());
    }
    if input.queries.is_empty() {
        return Err("queries must contain at least one held-out reference query".into());
    }
    if input.candidates.is_empty() || input.candidates.len() > 20 {
        return Err("candidates must contain between 1 and 20 proposals for exact enumeration".into());
    }
    if input.review_budgets_minutes.is_empty() {
        return Err("review_budgets_minutes must be non-empty".into());
    }

    let mut graph_keys = BTreeSet::new();
    for edge in &input.graph {
        if edge.source == edge.target || !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err("graph edges must be non-self, finite, and strictly positive".into());
        }
        if !graph_keys.insert((edge.source, edge.target)) {
            return Err(format!("duplicate graph edge {} -> {}", edge.source, edge.target));
        }
    }

    let mut query_ids = BTreeSet::new();
    let mut total_weight = 0.0_f64;
    for query in &input.queries {
        if query.query_id.trim().is_empty() || !query_ids.insert(query.query_id.clone()) {
            return Err("query IDs must be non-empty and unique".into());
        }
        if !query.weight.is_finite() || query.weight < 0.0
            || !query.normalization_scale.is_finite() || query.normalization_scale <= 0.0
            || !query.reference_value.is_finite()
        {
            return Err(format!("query {} has invalid weight, scale, or reference", query.query_id));
        }
        total_weight += query.weight;
        if !total_weight.is_finite() {
            return Err("sum of query weights overflowed".into());
        }
    }
    if total_weight <= 0.0 {
        return Err("at least one query must have positive weight".into());
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
            return Err(format!("candidate {} probability must be in [0, 1]", candidate.candidate_id));
        }
        if !candidate.review_minutes.is_finite() || candidate.review_minutes <= 0.0 {
            return Err(format!("candidate {} review_minutes must be finite and positive", candidate.candidate_id));
        }
        if !valid_hex(&candidate.evidence_sha256, 64) || candidate.changes.is_empty() {
            return Err(format!("candidate {} needs an evidence SHA-256 and non-empty mutations", candidate.candidate_id));
        }
        let mut touched = BTreeSet::new();
        for change in &candidate.changes {
            let (source, target) = change.edge_key();
            if source == target || !change.weight().is_finite() || change.weight() <= 0.0 {
                return Err(format!("candidate {} has an invalid mutation", candidate.candidate_id));
            }
            if !touched.insert((source, target)) {
                return Err(format!("candidate {} mutates edge {} -> {} more than once", candidate.candidate_id, source, target));
            }
        }
    }

    let mut budgets = BTreeSet::new();
    for budget in &input.review_budgets_minutes {
        if !budget.is_finite() || *budget <= 0.0 || !budgets.insert(budget.to_bits()) {
            return Err("review budgets must be finite, positive, and unique".into());
        }
    }
    Ok(())
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

fn weighted_query_error(input: &Input, graph: &Graph) -> Result<f64, String> {
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for query in &input.queries {
        let value = query_value(graph, &query.metric);
        let normalized_error = (value - query.reference_value).abs() / query.normalization_scale;
        numerator += query.weight * normalized_error;
        denominator += query.weight;
    }
    let error = numerator / denominator;
    if !error.is_finite() {
        return Err("weighted query error overflowed".into());
    }
    Ok(error)
}

fn apply_candidate(graph: &Graph, candidate: &Candidate) -> Result<Graph, String> {
    let mut next_graph = graph.clone();
    for change in &candidate.changes {
        let key = change.edge_key();
        match change {
            EdgeChange::AddWeight { weight, .. } => {
                let next = next_graph.get(&key).copied().unwrap_or(0.0) + weight;
                if !next.is_finite() {
                    return Err(format!("candidate {} overflows edge weight", candidate.candidate_id));
                }
                next_graph.insert(key, next);
            }
            EdgeChange::RemoveWeight { weight, .. } => {
                let current = next_graph.get(&key).copied().unwrap_or(0.0);
                if *weight > current + 1.0e-9 {
                    return Err(format!("candidate {} removes unavailable edge weight", candidate.candidate_id));
                }
                let remaining = current - weight;
                if remaining <= 1.0e-9 {
                    next_graph.remove(&key);
                } else {
                    next_graph.insert(key, remaining);
                }
            }
        }
    }
    Ok(next_graph)
}

fn canonical_indices(input: &Input, selected: &[usize], reverse: bool) -> Vec<usize> {
    let mut indices = selected.to_vec();
    indices.sort_by(|left, right| {
        input.candidates[*left].candidate_id.cmp(&input.candidates[*right].candidate_id)
    });
    if reverse {
        indices.reverse();
    }
    indices
}

fn simulate(input: &Input, selected: &[usize], reverse: bool) -> Result<(Graph, Vec<String>), String> {
    let mut graph = graph_from_edges(&input.graph);
    let order = canonical_indices(input, selected, reverse);
    let mut ids = Vec::with_capacity(order.len());
    for index in order {
        let candidate = &input.candidates[index];
        graph = apply_candidate(&graph, candidate)?;
        ids.push(candidate.candidate_id.clone());
    }
    Ok((graph, ids))
}

fn utility(input: &Input, baseline_error: f64, selected: &[usize]) -> Result<f64, String> {
    let (graph, _) = simulate(input, selected, false)?;
    Ok(baseline_error - weighted_query_error(input, &graph)?)
}

fn marginal_sensitivity(input: &Input, graph: &Graph, candidate: &Candidate) -> f64 {
    let Ok(next_graph) = apply_candidate(graph, candidate) else {
        return f64::NEG_INFINITY;
    };
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for query in &input.queries {
        let before = query_value(graph, &query.metric);
        let after = query_value(&next_graph, &query.metric);
        numerator += query.weight * ((after - before).abs() / query.normalization_scale).min(1.0);
        denominator += query.weight;
    }
    numerator / denominator
}

fn rank_candidates(input: &Input, input_sha: &str, policy: &str) -> Vec<usize> {
    let baseline = graph_from_edges(&input.graph);
    let mut ranked: Vec<(usize, f64, String)> = input
        .candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let probability = candidate.estimated_error_probability;
            let score = match policy {
                "probability_per_minute" => probability / candidate.review_minutes,
                "uncertainty_per_minute" => {
                    (1.0 - (2.0 * probability - 1.0).abs()) / candidate.review_minutes
                }
                "probability_weighted_query_sensitivity_per_minute" => {
                    probability * marginal_sensitivity(input, &baseline, candidate)
                        / candidate.review_minutes
                }
                "deterministic_random" => {
                    let digest = Sha256::digest(
                        format!("{input_sha}:{}", candidate.candidate_id).as_bytes()
                    );
                    let mut first_eight = [0_u8; 8];
                    first_eight.copy_from_slice(&digest[..8]);
                    u64::from_be_bytes(first_eight) as f64 / u64::MAX as f64
                }
                _ => 0.0,
            };
            (index, score, candidate.candidate_id.clone())
        })
        .collect();
    ranked.sort_by(|left, right| {
        right.1.total_cmp(&left.1).then_with(|| left.2.cmp(&right.2))
    });
    ranked.into_iter().map(|(index, _, _)| index).collect()
}

fn portfolio_for_ranking(
    input: &Input,
    ranking: &[usize],
    budget: f64,
) -> (Vec<usize>, f64) {
    let mut selected = Vec::new();
    let mut spent = 0.0;
    for index in ranking {
        let candidate = &input.candidates[*index];
        if spent + candidate.review_minutes > budget + 1.0e-9 {
            continue;
        }
        let mut proposed = selected.clone();
        proposed.push(*index);
        if simulate(input, &proposed, false).is_ok() {
            selected.push(*index);
            spent += candidate.review_minutes;
        }
    }
    (selected, spent)
}

fn exact_optimum(
    input: &Input,
    baseline_error: f64,
    budget: f64,
) -> Result<(Vec<usize>, f64, f64), String> {
    let count = input.candidates.len();
    let total = 1_u64 << count;
    let mut best_indices = Vec::new();
    let mut best_utility = 0.0_f64;
    let mut best_cost = 0.0_f64;

    for mask in 1..total {
        let mut selected = Vec::new();
        let mut cost = 0.0;
        for index in 0..count {
            if (mask & (1_u64 << index)) != 0 {
                selected.push(index);
                cost += input.candidates[index].review_minutes;
            }
        }
        if cost > budget + 1.0e-9 {
            continue;
        }
        let Ok(candidate_utility) = utility(input, baseline_error, &selected) else {
            continue;
        };
        let ids = canonical_indices(input, &selected, false)
            .iter()
            .map(|index| input.candidates[*index].candidate_id.clone())
            .collect::<Vec<_>>();
        let best_ids = canonical_indices(input, &best_indices, false)
            .iter()
            .map(|index| input.candidates[*index].candidate_id.clone())
            .collect::<Vec<_>>();
        let better = candidate_utility > best_utility + 1.0e-12
            || ((candidate_utility - best_utility).abs() <= 1.0e-12
                && (cost < best_cost - 1.0e-12
                    || ((cost - best_cost).abs() <= 1.0e-12 && ids < best_ids)));
        if better {
            best_indices = selected;
            best_utility = candidate_utility;
            best_cost = cost;
        }
    }
    Ok((best_indices, best_utility, best_cost))
}

fn make_result(
    input: &Input,
    baseline_error: f64,
    policy: &'static str,
    selected: &[usize],
    spent: f64,
    exact_utility: f64,
) -> PortfolioResult {
    let ids = selected
        .iter()
        .map(|index| input.candidates[*index].candidate_id.clone())
        .collect::<Vec<_>>();
    let canonical_order = canonical_indices(input, selected, false)
        .iter()
        .map(|index| input.candidates[*index].candidate_id.clone())
        .collect::<Vec<_>>();
    let forward = simulate(input, selected, false);
    let reverse = simulate(input, selected, true);
    match forward {
        Ok((graph, _)) => {
            let final_error = weighted_query_error(input, &graph).unwrap_or(f64::NAN);
            let gain = baseline_error - final_error;
            let reverse_error = reverse
                .ok()
                .and_then(|(reverse_graph, _)| weighted_query_error(input, &reverse_graph).ok());
            PortfolioResult {
                policy,
                selected_candidate_ids: ids,
                canonical_application_order: canonical_order,
                actual_review_minutes: spent,
                baseline_weighted_query_error: baseline_error,
                final_weighted_query_error: Some(final_error),
                utility: Some(gain),
                regret_to_exact_optimum: Some((exact_utility - gain).max(0.0)),
                reverse_order_weighted_query_error: reverse_error,
                order_sensitivity_absolute_error_delta: reverse_error.map(|value| (value - final_error).abs()),
                status: "EVALUATED",
            }
        }
        Err(_) => PortfolioResult {
            policy,
            selected_candidate_ids: ids,
            canonical_application_order: canonical_order,
            actual_review_minutes: spent,
            baseline_weighted_query_error: baseline_error,
            final_weighted_query_error: None,
            utility: None,
            regret_to_exact_optimum: None,
            reverse_order_weighted_query_error: None,
            order_sensitivity_absolute_error_delta: None,
            status: "INCOMPATIBLE_PORTFOLIO",
        },
    }
}

fn pairwise_interactions(input: &Input, baseline_error: f64) -> Vec<PairwiseInteraction> {
    let mut rows = Vec::new();
    for left in 0..input.candidates.len() {
        for right in (left + 1)..input.candidates.len() {
            let pair = vec![left, right];
            let utility_a = utility(input, baseline_error, &[left]).ok();
            let utility_b = utility(input, baseline_error, &[right]).ok();
            let joint = utility(input, baseline_error, &pair).ok();
            let interaction = match (joint, utility_a, utility_b) {
                (Some(joint_value), Some(a), Some(b)) => Some(joint_value - a - b),
                _ => None,
            };
            rows.push(PairwiseInteraction {
                candidate_a: input.candidates[left].candidate_id.clone(),
                candidate_b: input.candidates[right].candidate_id.clone(),
                joint_utility: joint,
                utility_a,
                utility_b,
                interaction,
                status: if joint.is_some() { "EVALUATED" } else { "INCOMPATIBLE" },
            });
        }
    }
    rows
}

fn run(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    validate(input)?;
    let input_sha = sha256(input_bytes);
    let baseline_graph = graph_from_edges(&input.graph);
    let baseline_error = weighted_query_error(input, &baseline_graph)?;
    let input_sha_str = input_sha.as_str();

    let policy_names = [
        "probability_weighted_query_sensitivity_per_minute",
        "probability_per_minute",
        "uncertainty_per_minute",
        "deterministic_random",
    ];
    let rankings = policy_names
        .iter()
        .map(|name| (*name, rank_candidates(input, input_sha_str, name)))
        .collect::<Vec<_>>();

    let mut budget_results = Vec::new();
    for budget in &input.review_budgets_minutes {
        let (optimal, optimal_utility, optimal_cost) =
            exact_optimum(input, baseline_error, *budget)?;
        let exact = make_result(
            input,
            baseline_error,
            "exact_oracle_optimum",
            &optimal,
            optimal_cost,
            optimal_utility,
        );
        let mut ranked_results = Vec::new();
        for (name, ranking) in &rankings {
            let (selected, spent) = portfolio_for_ranking(input, ranking, *budget);
            ranked_results.push(make_result(
                input,
                baseline_error,
                name,
                &selected,
                spent,
                optimal_utility,
            ));
        }
        budget_results.push(BudgetResult {
            review_budget_minutes: *budget,
            exact_optimum: exact,
            ranked_policies: ranked_results,
        });
    }

    Ok(Report {
        schema_version: 1,
        tool: "connectome-portfolio-audit",
        input_sha256: input_sha,
        dataset_id: input.dataset_id.clone(),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        source_graph_sha256: input.source_graph_sha256.clone(),
        baseline_graph_canonical_sha256: canonical_graph_sha256(&baseline_graph),
        candidate_count: input.candidates.len(),
        subset_count_evaluated: ((1_u64 << input.candidates.len()) - 1) * input.review_budgets_minutes.len() as u64,
        baseline_weighted_query_error: baseline_error,
        pairwise_interactions: pairwise_interactions(input, baseline_error),
        budget_results,
        status: "EVALUATED_NOT_PROMOTED",
        ranking_information_flow: "Candidate ranking uses predicted error probability, review cost, baseline query sensitivity, and deterministic input hashing only. Evaluator reference values are used only for utility, interactions, regret, and the exact oracle optimum.",
        optimization_scope: "Exact optimum is restricted to the supplied candidate pool, review budget, declared graph mutations, canonical candidate-ID application order, and supplied query reference bundle. It is not a whole-connectome optimum or biological truth certificate.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("could not read {}: {error}", args.input.display());
            return ExitCode::from(2);
        }
    };
    let input = match serde_json::from_slice::<Input>(&bytes) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("invalid portfolio JSON: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match run(&input, &bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("portfolio audit failed: {error}");
            return ExitCode::from(1);
        }
    };
    let output = match serde_json::to_string_pretty(&report) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("could not serialize report: {error}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(&path, output) {
            eprintln!("could not write {}: {error}", path.display());
            return ExitCode::from(2);
        }
    } else {
        println!("{output}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(char: char) -> String {
        char.to_string().repeat(64)
    }

    fn base_input(candidates: Vec<Candidate>, queries: Vec<Query>, graph: Vec<Edge>) -> Input {
        Input {
            schema_version: 1,
            dataset_id: "synthetic-connectome".into(),
            dataset_revision_commit_sha: "a".repeat(40),
            source_graph_sha256: "b".repeat(64),
            graph,
            queries,
            candidates,
            review_budgets_minutes: vec![2.0, 4.0],
        }
    }

    fn candidate(id: &str, p: f64, minutes: f64, mutation: EdgeChange) -> Candidate {
        Candidate {
            candidate_id: id.into(),
            estimated_error_probability: p,
            review_minutes: minutes,
            evidence_sha256: h('c'),
            changes: vec![mutation],
        }
    }

    #[test]
    fn exact_optimum_finds_two_edit_synergy_that_single_edit_ranking_misses() {
        let candidates = vec![
            candidate("add-first-leg", 0.8, 1.0, EdgeChange::AddWeight { source: 1, target: 2, weight: 1.0 }),
            candidate("add-second-leg", 0.8, 1.0, EdgeChange::AddWeight { source: 2, target: 3, weight: 1.0 }),
        ];
        let queries = vec![Query {
            query_id: "reachability".into(),
            weight: 1.0,
            normalization_scale: 1.0,
            reference_value: 1.0,
            metric: QueryMetric::PathExists { source: 1, target: 3 },
        }];
        let graph = vec![Edge { source: 9, target: 10, weight: 1.0 }];
        let input = base_input(candidates, queries, graph);
        let baseline = weighted_query_error(&input, &graph_from_edges(&input.graph)).unwrap();
        let joint = utility(&input, baseline, &[0, 1]).unwrap();
        let single = utility(&input, baseline, &[0]).unwrap();
        assert!((joint - 1.0).abs() < 1.0e-12);
        assert!(single.abs() < 1.0e-12);
        let pairs = pairwise_interactions(&input, baseline);
        assert!((pairs[0].interaction.unwrap() - 1.0).abs() < 1.0e-12);
        let (best, gain, _) = exact_optimum(&input, baseline, 2.0).unwrap();
        assert_eq!(best.len(), 2);
        assert!((gain - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn exact_optimum_respects_budget_and_never_reports_oracle_as_ranking_input() {
        let candidates = vec![
            candidate("expensive-correct", 0.99, 3.0, EdgeChange::AddWeight { source: 1, target: 2, weight: 1.0 }),
            candidate("cheap-correct", 0.7, 1.0, EdgeChange::AddWeight { source: 1, target: 2, weight: 1.0 }),
        ];
        let queries = vec![Query {
            query_id: "edge".into(), weight: 1.0, normalization_scale: 1.0,
            reference_value: 1.0, metric: QueryMetric::EdgeWeight { source: 1, target: 2 },
        }];
        let input = base_input(candidates, queries, vec![Edge { source: 9, target: 10, weight: 1.0 }]);
        let ranking = rank_candidates(&input, &"d".repeat(64), "probability_per_minute");
        assert_eq!(input.candidates[ranking[0]].candidate_id, "cheap-correct");
        let baseline = weighted_query_error(&input, &graph_from_edges(&input.graph)).unwrap();
        let (best, _, cost) = exact_optimum(&input, baseline, 1.0).unwrap();
        assert_eq!(best, vec![1]);
        assert!((cost - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn reverse_order_reports_order_dependent_validity() {
        let add = candidate("a-add", 0.9, 1.0, EdgeChange::AddWeight { source: 1, target: 2, weight: 0.5 });
        let remove = candidate("b-remove", 0.9, 1.0, EdgeChange::RemoveWeight { source: 1, target: 2, weight: 1.2 });
        let query = Query {
            query_id: "edge".into(), weight: 1.0, normalization_scale: 2.0,
            reference_value: 0.3, metric: QueryMetric::EdgeWeight { source: 1, target: 2 },
        };
        let input = base_input(vec![add, remove], vec![query], vec![Edge { source: 1, target: 2, weight: 1.0 }]);
        assert!(simulate(&input, &[0, 1], false).is_ok());
        assert!(simulate(&input, &[0, 1], true).is_err());
    }

    #[test]
    fn invalid_overdrawn_portfolio_is_excluded_from_exact_search() {
        let candidates = vec![
            candidate("remove-most", 0.9, 1.0, EdgeChange::RemoveWeight { source: 1, target: 2, weight: 0.8 }),
            candidate("remove-again", 0.9, 1.0, EdgeChange::RemoveWeight { source: 1, target: 2, weight: 0.8 }),
        ];
        let query = Query {
            query_id: "edge".into(), weight: 1.0, normalization_scale: 1.0,
            reference_value: 0.0, metric: QueryMetric::EdgeWeight { source: 1, target: 2 },
        };
        let input = base_input(candidates, vec![query], vec![Edge { source: 1, target: 2, weight: 1.0 }]);
        let baseline = weighted_query_error(&input, &graph_from_edges(&input.graph)).unwrap();
        assert!(utility(&input, baseline, &[0, 1]).is_err());
        let (best, _, _) = exact_optimum(&input, baseline, 2.0).unwrap();
        assert_eq!(best.len(), 1);
    }

    #[test]
    fn canonical_graph_hash_is_independent_of_input_order() {
        let a = graph_from_edges(&[
            Edge { source: 1, target: 2, weight: 0.5 },
            Edge { source: 2, target: 3, weight: 1.0 },
        ]);
        let b = graph_from_edges(&[
            Edge { source: 2, target: 3, weight: 1.0 },
            Edge { source: 1, target: 2, weight: 0.5 },
        ]);
        assert_eq!(canonical_graph_sha256(&a), canonical_graph_sha256(&b));
    }

    #[test]
    fn schema_rejects_more_than_twenty_candidates() {
        let many = (0..21)
            .map(|index| candidate(&format!("c{index}"), 0.5, 1.0, EdgeChange::AddWeight {
                source: 100 + index, target: 200 + index, weight: 1.0,
            }))
            .collect::<Vec<_>>();
        let query = Query {
            query_id: "edge".into(), weight: 1.0, normalization_scale: 1.0,
            reference_value: 0.0, metric: QueryMetric::EdgeWeight { source: 1, target: 2 },
        };
        let input = base_input(many, vec![query], vec![Edge { source: 9, target: 10, weight: 1.0 }]);
        assert!(validate(&input).is_err());
    }

    #[test]
    fn duplicate_edge_mutations_inside_one_candidate_are_rejected() {
        let mut item = candidate("duplicate", 0.5, 1.0, EdgeChange::AddWeight {
            source: 1, target: 2, weight: 1.0,
        });
        item.changes.push(EdgeChange::RemoveWeight { source: 1, target: 2, weight: 0.5 });
        let query = Query {
            query_id: "edge".into(), weight: 1.0, normalization_scale: 1.0,
            reference_value: 0.0, metric: QueryMetric::EdgeWeight { source: 1, target: 2 },
        };
        let input = base_input(vec![item], vec![query], vec![Edge { source: 9, target: 10, weight: 1.0 }]);
        assert!(validate(&input).is_err());
    }

    #[test]
    fn zero_query_weight_and_nonfinite_reference_are_rejected() {
        let item = candidate("candidate", 0.5, 1.0, EdgeChange::AddWeight {
            source: 1, target: 2, weight: 1.0,
        });
        let mut query = Query {
            query_id: "edge".into(), weight: 0.0, normalization_scale: 1.0,
            reference_value: 0.0, metric: QueryMetric::EdgeWeight { source: 1, target: 2 },
        };
        let mut input = base_input(vec![item.clone()], vec![query.clone()], vec![Edge { source: 9, target: 10, weight: 1.0 }]);
        assert!(validate(&input).is_err());
        query.weight = 1.0;
        query.reference_value = f64::NAN;
        input.queries = vec![query];
        assert!(validate(&input).is_err());
    }

    #[test]
    fn deterministic_random_ranking_is_stable_for_exact_input_digest() {
        let candidates = vec![
            candidate("left", 0.5, 1.0, EdgeChange::AddWeight { source: 1, target: 2, weight: 1.0 }),
            candidate("right", 0.5, 1.0, EdgeChange::AddWeight { source: 2, target: 3, weight: 1.0 }),
        ];
        let query = Query {
            query_id: "path".into(), weight: 1.0, normalization_scale: 1.0,
            reference_value: 1.0, metric: QueryMetric::PathExists { source: 1, target: 3 },
        };
        let input = base_input(candidates, vec![query], vec![Edge { source: 9, target: 10, weight: 1.0 }]);
        assert_eq!(
            rank_candidates(&input, &"e".repeat(64), "deterministic_random"),
            rank_candidates(&input, &"e".repeat(64), "deterministic_random")
        );
    }

    #[test]
    fn report_keeps_reference_information_out_of_candidate_ranking() {
        let candidates = vec![candidate("only", 0.7, 1.0, EdgeChange::AddWeight {
            source: 1, target: 2, weight: 1.0,
        })];
        let query = Query {
            query_id: "edge".into(), weight: 1.0, normalization_scale: 1.0,
            reference_value: 1.0, metric: QueryMetric::EdgeWeight { source: 1, target: 2 },
        };
        let input = base_input(candidates, vec![query], vec![Edge { source: 9, target: 10, weight: 1.0 }]);
        let ranking_a = rank_candidates(&input, &"f".repeat(64), "probability_weighted_query_sensitivity_per_minute");
        let mut changed_reference = input;
        changed_reference.queries[0].reference_value = 99.0;
        let ranking_b = rank_candidates(&changed_reference, &"f".repeat(64), "probability_weighted_query_sensitivity_per_minute");
        assert_eq!(ranking_a, ranking_b);
    }
}
