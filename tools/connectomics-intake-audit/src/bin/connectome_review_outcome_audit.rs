//! Post-hoc reviewer-outcome evaluator for frozen connectome review policies.
//!
//! This binary is intentionally separate from connectome-query-impact. It consumes
//! the frozen proposal input and ranking report plus a separate adjudication/
//! reviewer-outcome file. The ranker never reads evaluator-only outcomes, preventing
//! reviewer decisions and hidden correctness labels from leaking into policy scores.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-review-outcome-audit",
    version,
    about = "Evaluate frozen connectome review rankings against independent adjudication and actual reviewer outcomes"
)]
struct Args {
    /// Original proposal input passed to connectome-query-impact.
    #[arg(long)]
    proposals: PathBuf,

    /// Frozen JSON report emitted by connectome-query-impact.
    #[arg(long)]
    ranking_report: PathBuf,

    /// Separate evaluation-only adjudication and reviewer decision file.
    #[arg(long)]
    outcomes: PathBuf,

    /// Optional deterministic JSON evaluation report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProposalInput {
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
    #[serde(default)]
    reference_value: Option<f64>,
    metric: QueryMetric,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum QueryMetric {
    EdgeWeight { source: u64, target: u64 },
    InDegree { node: u64 },
    OutDegree { node: u64 },
    PathExists { source: u64, target: u64 },
    TriadMotifCount { motif: TriadMotif },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TriadMotif {
    FeedForwardLoop,
    DirectedThreeCycle,
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

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum EdgeChange {
    AddWeight { source: u64, target: u64, weight: f64 },
    RemoveWeight { source: u64, target: u64, weight: f64 },
}

impl EdgeChange {
    fn key(&self) -> (u64, u64) {
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenRankingReport {
    schema_version: u32,
    input_sha256: String,
    baseline_graph_canonical_sha256: String,
    policy_budget_results: Vec<FrozenPolicyOrder>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenPolicyOrder {
    policy: String,
    review_budget_minutes: f64,
    policy_order_candidate_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeFile {
    schema_version: u32,
    source_input_sha256: String,
    source_ranking_report_sha256: String,
    adjudication_protocol_sha256: String,
    candidate_outcomes: Vec<CandidateOutcome>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateOutcome {
    candidate_id: String,
    adjudication: Adjudication,
    reviewer_decision: ReviewerDecision,
    actual_review_minutes: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Adjudication {
    Correct,
    Incorrect,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReviewerDecision {
    Accept,
    Reject,
    Abstain,
}

#[derive(Debug, Default, Serialize)]
struct OutcomeCounts {
    reviewed_correct: usize,
    reviewed_incorrect: usize,
    reviewed_indeterminate: usize,
    accepted_correct: usize,
    accepted_incorrect: usize,
    accepted_indeterminate: usize,
    rejected_correct: usize,
    rejected_incorrect: usize,
    rejected_indeterminate: usize,
    abstained_correct: usize,
    abstained_incorrect: usize,
    abstained_indeterminate: usize,
    applied_correct: usize,
    applied_incorrect: usize,
    applied_indeterminate: usize,
}

#[derive(Debug, Serialize)]
struct EvaluatedPolicyBudget {
    policy: String,
    review_budget_minutes: f64,
    actual_review_minutes: f64,
    outcome_mode: &'static str,
    reviewed_candidate_count: usize,
    applied_candidate_count: usize,
    stale_conflict_count: usize,
    reviewed_candidate_ids: Vec<String>,
    applied_candidate_ids: Vec<String>,
    stale_conflict_candidate_ids: Vec<String>,
    outcome_counts: OutcomeCounts,
    candidate_pool_correct_count: usize,
    candidate_pool_incorrect_count: usize,
    accepted_decision_precision: Option<f64>,
    applied_correction_precision: Option<f64>,
    correct_proposal_review_recall: Option<f64>,
    correct_proposal_acceptance_recall: Option<f64>,
    incorrect_proposal_acceptance_rate: Option<f64>,
    baseline_weighted_query_error: Option<f64>,
    final_weighted_query_error: Option<f64>,
    weighted_query_error_reduction: Option<f64>,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    schema_version: u32,
    tool: &'static str,
    proposal_input_sha256: String,
    ranking_report_sha256: String,
    outcomes_sha256: String,
    adjudication_protocol_sha256: String,
    dataset_id: String,
    dataset_revision_commit_sha: String,
    source_graph_sha256: String,
    baseline_graph_canonical_sha256: String,
    query_bundle_sha256: String,
    candidate_count: usize,
    evaluated_policy_budget_count: usize,
    policy_budget_results: Vec<EvaluatedPolicyBudget>,
    warnings: Vec<String>,
    status: &'static str,
    scope_note: &'static str,
}

type Graph = BTreeMap<(u64, u64), f64>;

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn validate_proposals(input: &ProposalInput) -> Result<(), String> {
    if input.schema_version != 1 {
        return Err("proposal schema_version must be 1".into());
    }
    if input.dataset_id.trim().is_empty() {
        return Err("proposal dataset_id must be non-empty".into());
    }
    if !is_lower_hex(&input.dataset_revision_commit_sha, 40)
        || !is_lower_hex(&input.source_graph_sha256, 64)
        || !is_lower_hex(&input.query_bundle_sha256, 64)
    {
        return Err("proposal dataset/source/query digests have invalid format".into());
    }
    if input.graph.is_empty() || input.queries.is_empty() || input.candidates.is_empty() {
        return Err("proposal graph, queries, and candidates must all be non-empty".into());
    }
    if input.review_budgets_minutes.is_empty()
        || input.review_budgets_minutes.iter().any(|budget| !budget.is_finite() || *budget <= 0.0)
    {
        return Err("proposal review budgets must be finite, positive, and non-empty".into());
    }
    let mut edge_ids = BTreeSet::new();
    for edge in &input.graph {
        if edge.source == edge.target || !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err("proposal graph contains invalid self-edge or edge weight".into());
        }
        if !edge_ids.insert((edge.source, edge.target)) {
            return Err(format!("proposal graph has duplicate edge {} -> {}", edge.source, edge.target));
        }
    }
    let mut candidate_ids = BTreeSet::new();
    for candidate in &input.candidates {
        if candidate.candidate_id.trim().is_empty() || !candidate_ids.insert(candidate.candidate_id.clone()) {
            return Err("proposal candidate IDs must be non-empty and unique".into());
        }
        if !candidate.estimated_error_probability.is_finite()
            || !(0.0..=1.0).contains(&candidate.estimated_error_probability)
            || !candidate.review_minutes.is_finite()
            || candidate.review_minutes <= 0.0
            || !is_lower_hex(&candidate.evidence_sha256, 64)
            || candidate.changes.is_empty()
        {
            return Err(format!("proposal candidate {} has invalid score/evidence/cost/change fields", candidate.candidate_id));
        }
        for change in &candidate.changes {
            let (source, target) = change.key();
            if source == target || !change.weight().is_finite() || change.weight() <= 0.0 {
                return Err(format!("proposal candidate {} has invalid edge mutation", candidate.candidate_id));
            }
        }
    }
    for query in &input.queries {
        if query.query_id.trim().is_empty()
            || !query.weight.is_finite()
            || query.weight < 0.0
            || !query.normalization_scale.is_finite()
            || query.normalization_scale <= 0.0
            || query.reference_value.is_some_and(|value| !value.is_finite())
        {
            return Err(format!("proposal query {} has invalid metric/weight/reference", query.query_id));
        }
    }
    Ok(())
}

fn validate_outcomes(
    outcomes: &OutcomeFile,
    input: &ProposalInput,
    input_hash: &str,
    ranking_hash: &str,
) -> Result<BTreeMap<String, CandidateOutcome>, String> {
    if outcomes.schema_version != 1 {
        return Err("outcomes schema_version must be 1".into());
    }
    if !is_lower_hex(&outcomes.source_input_sha256, 64)
        || !is_lower_hex(&outcomes.source_ranking_report_sha256, 64)
        || !is_lower_hex(&outcomes.adjudication_protocol_sha256, 64)
    {
        return Err("outcome provenance digests have invalid format".into());
    }
    if outcomes.source_input_sha256 != input_hash {
        return Err("outcomes source_input_sha256 does not match proposal input bytes".into());
    }
    if outcomes.source_ranking_report_sha256 != ranking_hash {
        return Err("outcomes source_ranking_report_sha256 does not match frozen ranking report bytes".into());
    }

    let expected: BTreeSet<String> = input.candidates.iter().map(|candidate| candidate.candidate_id.clone()).collect();
    let mut by_id = BTreeMap::new();
    for outcome in &outcomes.candidate_outcomes {
        if outcome.candidate_id.trim().is_empty() || by_id.contains_key(&outcome.candidate_id) {
            return Err("outcome candidate IDs must be non-empty and unique".into());
        }
        if !outcome.actual_review_minutes.is_finite() || outcome.actual_review_minutes <= 0.0 {
            return Err(format!("outcome {} actual_review_minutes must be finite and positive", outcome.candidate_id));
        }
        by_id.insert(outcome.candidate_id.clone(), outcome.clone());
    }
    let actual: BTreeSet<String> = by_id.keys().cloned().collect();
    if actual != expected {
        return Err("outcomes must contain exactly one record for every proposal candidate".into());
    }
    Ok(by_id)
}

fn graph_from_edges(edges: &[Edge]) -> Graph {
    edges.iter().map(|edge| ((edge.source, edge.target), edge.weight)).collect()
}

fn apply_candidate(graph: &Graph, candidate: &Candidate) -> Result<Graph, String> {
    let mut result = graph.clone();
    for change in &candidate.changes {
        let key = change.key();
        match change {
            EdgeChange::AddWeight { weight, .. } => {
                let next = result.get(&key).copied().unwrap_or(0.0) + weight;
                if !next.is_finite() {
                    return Err(format!("candidate {} edge addition overflow", candidate.candidate_id));
                }
                result.insert(key, next);
            }
            EdgeChange::RemoveWeight { weight, .. } => {
                let current = result.get(&key).copied().unwrap_or(0.0);
                if *weight > current + 1e-9 {
                    return Err(format!("candidate {} conflicts with current graph", candidate.candidate_id));
                }
                let remaining = current - weight;
                if remaining <= 1e-9 {
                    result.remove(&key);
                } else {
                    result.insert(key, remaining);
                }
            }
        }
    }
    Ok(result)
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
    let mut visited = BTreeSet::from([source]);
    let mut queue = VecDeque::from([source]);
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

fn motif_count(graph: &Graph, motif: &TriadMotif) -> f64 {
    let mut adjacency = BTreeMap::<u64, Vec<(u64, f64)>>::new();
    for ((source, target), weight) in graph {
        adjacency.entry(*source).or_default().push((*target, *weight));
    }
    let mut total = 0.0;
    for ((source, middle), first_weight) in graph {
        if let Some(next_edges) = adjacency.get(middle) {
            for (target, second_weight) in next_edges {
                let closing = match motif {
                    TriadMotif::FeedForwardLoop => (*source, *target),
                    TriadMotif::DirectedThreeCycle => (*target, *source),
                };
                if let Some(closing_weight) = graph.get(&closing) {
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
        QueryMetric::EdgeWeight { source, target } => graph.get(&(*source, *target)).copied().unwrap_or(0.0),
        QueryMetric::InDegree { node } => graph.iter().filter(|((_, target), _)| target == node).map(|(_, weight)| *weight).sum(),
        QueryMetric::OutDegree { node } => graph.iter().filter(|((source, _), _)| source == node).map(|(_, weight)| *weight).sum(),
        QueryMetric::PathExists { source, target } => if path_exists(graph, *source, *target) { 1.0 } else { 0.0 },
        QueryMetric::TriadMotifCount { motif } => motif_count(graph, motif),
    }
}

fn weighted_query_error(graph: &Graph, queries: &[Query]) -> Result<Option<f64>, String> {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for query in queries {
        if let Some(reference) = query.reference_value {
            let actual = query_value(graph, &query.metric);
            if !actual.is_finite() {
                return Err(format!("query {} produced non-finite output", query.query_id));
            }
            let normalized = ((actual - reference).abs() / query.normalization_scale).min(1.0);
            weighted += query.weight * normalized;
            total += query.weight;
        }
    }
    if total <= 0.0 { Ok(None) } else { Ok(Some(weighted / total)) }
}

fn increment_decision_counts(counts: &mut OutcomeCounts, outcome: &CandidateOutcome) {
    match outcome.adjudication {
        Adjudication::Correct => counts.reviewed_correct += 1,
        Adjudication::Incorrect => counts.reviewed_incorrect += 1,
        Adjudication::Indeterminate => counts.reviewed_indeterminate += 1,
    }
    match (outcome.reviewer_decision, outcome.adjudication) {
        (ReviewerDecision::Accept, Adjudication::Correct) => counts.accepted_correct += 1,
        (ReviewerDecision::Accept, Adjudication::Incorrect) => counts.accepted_incorrect += 1,
        (ReviewerDecision::Accept, Adjudication::Indeterminate) => counts.accepted_indeterminate += 1,
        (ReviewerDecision::Reject, Adjudication::Correct) => counts.rejected_correct += 1,
        (ReviewerDecision::Reject, Adjudication::Incorrect) => counts.rejected_incorrect += 1,
        (ReviewerDecision::Reject, Adjudication::Indeterminate) => counts.rejected_indeterminate += 1,
        (ReviewerDecision::Abstain, Adjudication::Correct) => counts.abstained_correct += 1,
        (ReviewerDecision::Abstain, Adjudication::Incorrect) => counts.abstained_incorrect += 1,
        (ReviewerDecision::Abstain, Adjudication::Indeterminate) => counts.abstained_indeterminate += 1,
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn evaluate_policy_budget(
    graph: &Graph,
    queries: &[Query],
    candidates: &[Candidate],
    outcomes: &BTreeMap<String, CandidateOutcome>,
    policy_order: &FrozenPolicyOrder,
    pool_correct: usize,
    pool_incorrect: usize,
) -> Result<EvaluatedPolicyBudget, String> {
    let candidate_map: BTreeMap<&str, &Candidate> = candidates.iter().map(|candidate| (candidate.candidate_id.as_str(), candidate)).collect();
    let baseline_error = weighted_query_error(graph, queries)?;
    let mut current_graph = graph.clone();
    let mut consumed = 0.0;
    let mut counts = OutcomeCounts::default();
    let mut reviewed = Vec::new();
    let mut applied = Vec::new();
    let mut stale_conflicts = Vec::new();

    for candidate_id in &policy_order.policy_order_candidate_ids {
        let outcome = outcomes.get(candidate_id).ok_or_else(|| format!("missing outcome for {candidate_id}"))?;
        if consumed + outcome.actual_review_minutes > policy_order.review_budget_minutes + 1e-9 {
            continue;
        }
        consumed += outcome.actual_review_minutes;
        reviewed.push(candidate_id.clone());
        increment_decision_counts(&mut counts, outcome);

        if matches!(outcome.reviewer_decision, ReviewerDecision::Accept) {
            let candidate = candidate_map.get(candidate_id.as_str()).ok_or_else(|| format!("ranking contains unknown candidate {candidate_id}"))?;
            match apply_candidate(&current_graph, candidate) {
                Ok(next_graph) => {
                    current_graph = next_graph;
                    applied.push(candidate_id.clone());
                    match outcome.adjudication {
                        Adjudication::Correct => counts.applied_correct += 1,
                        Adjudication::Incorrect => counts.applied_incorrect += 1,
                        Adjudication::Indeterminate => counts.applied_indeterminate += 1,
                    }
                }
                Err(_) => stale_conflicts.push(candidate_id.clone()),
            }
        }
    }

    let final_error = weighted_query_error(&current_graph, queries)?;
    let error_reduction = match (baseline_error, final_error) {
        (Some(before), Some(after)) => Some(before - after),
        _ => None,
    };
    let accept_adjudicated = counts.accepted_correct + counts.accepted_incorrect;
    let applied_adjudicated = counts.applied_correct + counts.applied_incorrect;
    Ok(EvaluatedPolicyBudget {
        policy: policy_order.policy.clone(),
        review_budget_minutes: policy_order.review_budget_minutes,
        actual_review_minutes: consumed,
        outcome_mode: "observed_reviewer_decisions_and_independent_adjudication",
        reviewed_candidate_count: reviewed.len(),
        applied_candidate_count: applied.len(),
        stale_conflict_count: stale_conflicts.len(),
        reviewed_candidate_ids: reviewed,
        applied_candidate_ids: applied,
        stale_conflict_candidate_ids: stale_conflicts,
        outcome_counts: counts,
        candidate_pool_correct_count: pool_correct,
        candidate_pool_incorrect_count: pool_incorrect,
        accepted_decision_precision: ratio(counts.accepted_correct, accept_adjudicated),
        applied_correction_precision: ratio(counts.applied_correct, applied_adjudicated),
        correct_proposal_review_recall: ratio(counts.reviewed_correct, pool_correct),
        correct_proposal_acceptance_recall: ratio(counts.accepted_correct, pool_correct),
        incorrect_proposal_acceptance_rate: ratio(counts.accepted_incorrect, pool_incorrect),
        baseline_weighted_query_error: baseline_error,
        final_weighted_query_error: final_error,
        weighted_query_error_reduction: error_reduction,
    })
}

fn evaluate(
    proposals: &ProposalInput,
    ranking: &FrozenRankingReport,
    outcomes: &OutcomeFile,
    proposal_hash: String,
    ranking_hash: String,
    outcomes_hash: String,
) -> Result<EvaluationReport, String> {
    if ranking.schema_version != 1 {
        return Err("ranking report schema_version must be 1".into());
    }
    if ranking.input_sha256 != proposal_hash {
        return Err("ranking report input_sha256 does not match proposal input bytes".into());
    }
    let graph = graph_from_edges(&proposals.graph);
    let graph_digest = canonical_graph_sha256(&graph);
    if ranking.baseline_graph_canonical_sha256 != graph_digest {
        return Err("ranking report baseline graph digest does not match proposal graph".into());
    }
    let outcome_map = validate_outcomes(outcomes, proposals, &proposal_hash, &ranking_hash)?;
    let mut seen_orders = BTreeSet::new();
    for policy_order in &ranking.policy_budget_results {
        if !policy_order.review_budget_minutes.is_finite() || policy_order.review_budget_minutes <= 0.0 {
            return Err("ranking report contains invalid review budget".into());
        }
        let expected: BTreeSet<String> = proposals.candidates.iter().map(|candidate| candidate.candidate_id.clone()).collect();
        let observed: BTreeSet<String> = policy_order.policy_order_candidate_ids.iter().cloned().collect();
        if observed != expected || observed.len() != policy_order.policy_order_candidate_ids.len() {
            return Err(format!("frozen policy order {} is not a permutation of all candidates", policy_order.policy));
        }
        let key = (policy_order.policy.clone(), policy_order.review_budget_minutes.to_bits());
        if !seen_orders.insert(key) {
            return Err("ranking report contains duplicate policy/budget rows".into());
        }
    }
    if ranking.policy_budget_results.is_empty() {
        return Err("ranking report has no policy/budget replay rows".into());
    }

    let pool_correct = outcomes.candidate_outcomes.iter().filter(|outcome| matches!(outcome.adjudication, Adjudication::Correct)).count();
    let pool_incorrect = outcomes.candidate_outcomes.iter().filter(|outcome| matches!(outcome.adjudication, Adjudication::Incorrect)).count();
    let mut results = Vec::with_capacity(ranking.policy_budget_results.len());
    for policy_order in &ranking.policy_budget_results {
        results.push(evaluate_policy_budget(
            &graph,
            &proposals.queries,
            &proposals.candidates,
            &outcome_map,
            policy_order,
            pool_correct,
            pool_incorrect,
        )?);
    }
    results.sort_by(|left, right| {
        left.review_budget_minutes.total_cmp(&right.review_budget_minutes)
            .then_with(|| left.policy.cmp(&right.policy))
    });

    Ok(EvaluationReport {
        schema_version: 1,
        tool: "connectome-review-outcome-audit",
        proposal_input_sha256: proposal_hash,
        ranking_report_sha256: ranking_hash,
        outcomes_sha256: outcomes_hash,
        adjudication_protocol_sha256: outcomes.adjudication_protocol_sha256.clone(),
        dataset_id: proposals.dataset_id.clone(),
        dataset_revision_commit_sha: proposals.dataset_revision_commit_sha.clone(),
        source_graph_sha256: proposals.source_graph_sha256.clone(),
        baseline_graph_canonical_sha256: graph_digest,
        query_bundle_sha256: proposals.query_bundle_sha256.clone(),
        candidate_count: proposals.candidates.len(),
        evaluated_policy_budget_count: results.len(),
        policy_budget_results: results,
        warnings: vec![
            "Reviewer outcomes and adjudication labels are read only by this post-hoc evaluator; they are not inputs to the query-impact ranker.".to_string(),
            "Candidate correctness is defined by the separately versioned adjudication protocol and may not be inferred from model confidence or query sensitivity.".to_string(),
            "Actual review minutes determine which candidates fit each matched budget; policy ordering remains the frozen ranking report order.".to_string(),
            "An accepted incorrect proposal is applied to the in-memory graph so its downstream query harm is measured; stale conflicting proposals are recorded and not applied.".to_string(),
            "Candidate-pool recall is conditional on the frozen candidate pool and does not measure errors never proposed by any policy.".to_string(),
        ],
        status: "POST_HOC_REVIEW_OUTCOMES_EVALUATED",
        scope_note: "This evaluator measures a frozen proposal policy against separate reviewer decisions and independent candidate adjudication. It is a retrospective policy evaluation, not prospective causal evidence; candidate-pool recall is not whole-connectome error recall. It never modifies authoritative annotations.",
    })
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    let proposal_bytes = fs::read(&args.proposals).map_err(|error| format!("failed to read proposals: {error}"))?;
    let ranking_bytes = fs::read(&args.ranking_report).map_err(|error| format!("failed to read ranking report: {error}"))?;
    let outcome_bytes = fs::read(&args.outcomes).map_err(|error| format!("failed to read outcomes: {error}"))?;
    let proposal_hash = sha256(&proposal_bytes);
    let ranking_hash = sha256(&ranking_bytes);
    let outcome_hash = sha256(&outcome_bytes);

    let proposals: ProposalInput = serde_json::from_slice(&proposal_bytes)
        .map_err(|error| format!("proposal input parse error: {error}"))?;
    validate_proposals(&proposals)?;
    let ranking: FrozenRankingReport = serde_json::from_slice(&ranking_bytes)
        .map_err(|error| format!("ranking report parse error: {error}"))?;
    let outcomes: OutcomeFile = serde_json::from_slice(&outcome_bytes)
        .map_err(|error| format!("outcome file parse error: {error}"))?;

    let report = evaluate(&proposals, &ranking, &outcomes, proposal_hash, ranking_hash, outcome_hash)?;
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize evaluation report: {error}"))? + "\n";
    if let Some(output) = args.output {
        fs::write(&output, json.as_bytes()).map_err(|error| format!("failed to write report: {error}"))?;
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
            eprintln!("connectome-review-outcome-audit: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn proposals() -> ProposalInput {
        serde_json::from_value(json!({
            "schema_version": 1,
            "dataset_id": "synthetic-outcome-eval",
            "dataset_revision_commit_sha": "a".repeat(40),
            "source_graph_sha256": "b".repeat(64),
            "query_bundle_sha256": "c".repeat(64),
            "graph": [{"source":1,"target":2,"weight":2.0}],
            "queries": [{
                "query_id":"edge_1_2",
                "weight":1.0,
                "normalization_scale":2.0,
                "reference_value":0.0,
                "metric":{"kind":"edge_weight","source":1,"target":2}
            }],
            "review_budgets_minutes":[1.0,2.0,3.0],
            "candidates":[
                {
                    "candidate_id":"remove_one",
                    "estimated_error_probability":0.9,
                    "review_minutes":1.0,
                    "evidence_sha256":"d".repeat(64),
                    "changes":[{"operation":"remove_weight","source":1,"target":2,"weight":1.0}]
                },
                {
                    "candidate_id":"add_one_wrong",
                    "estimated_error_probability":0.8,
                    "review_minutes":1.0,
                    "evidence_sha256":"e".repeat(64),
                    "changes":[{"operation":"add_weight","source":1,"target":2,"weight":1.0}]
                },
                {
                    "candidate_id":"remove_two",
                    "estimated_error_probability":0.7,
                    "review_minutes":1.0,
                    "evidence_sha256":"f".repeat(64),
                    "changes":[{"operation":"remove_weight","source":1,"target":2,"weight":2.0}]
                }
            ]
        })).unwrap()
    }

    fn ranking(proposals: &ProposalInput, proposal_hash: &str) -> FrozenRankingReport {
        let graph = graph_from_edges(&proposals.graph);
        let ids = vec!["remove_one".to_string(),"add_one_wrong".to_string(),"remove_two".to_string()];
        let reverse = vec!["add_one_wrong".to_string(),"remove_one".to_string(),"remove_two".to_string()];
        serde_json::from_value(json!({
            "schema_version":1,
            "input_sha256":proposal_hash,
            "baseline_graph_canonical_sha256":canonical_graph_sha256(&graph),
            "policy_budget_results":[
                {"policy":"policy_a","review_budget_minutes":1.0,"policy_order_candidate_ids":ids},
                {"policy":"policy_a","review_budget_minutes":2.0,"policy_order_candidate_ids":ids},
                {"policy":"policy_b","review_budget_minutes":1.0,"policy_order_candidate_ids":reverse},
                {"policy":"policy_b","review_budget_minutes":2.0,"policy_order_candidate_ids":reverse}
            ]
        })).unwrap()
    }

    fn outcomes(proposals: &ProposalInput, proposal_hash: &str, ranking_hash: &str) -> OutcomeFile {
        serde_json::from_value(json!({
            "schema_version":1,
            "source_input_sha256":proposal_hash,
            "source_ranking_report_sha256":ranking_hash,
            "adjudication_protocol_sha256":"9".repeat(64),
            "candidate_outcomes":[
                {"candidate_id":"remove_one","adjudication":"correct","reviewer_decision":"accept","actual_review_minutes":1.0},
                {"candidate_id":"add_one_wrong","adjudication":"incorrect","reviewer_decision":"accept","actual_review_minutes":1.0},
                {"candidate_id":"remove_two","adjudication":"correct","reviewer_decision":"reject","actual_review_minutes":2.0}
            ]
        })).unwrap()
    }

    fn evaluate_fixture() -> EvaluationReport {
        let proposals = proposals();
        let proposal_bytes = serde_json::to_vec(&proposals).unwrap();
        let proposal_hash = sha256(&proposal_bytes);
        let ranking = ranking(&proposals, &proposal_hash);
        let ranking_bytes = serde_json::to_vec(&ranking).unwrap();
        let ranking_hash = sha256(&ranking_bytes);
        let outcomes = outcomes(&proposals, &proposal_hash, &ranking_hash);
        let outcome_hash = sha256(&serde_json::to_vec(&outcomes).unwrap());
        evaluate(&proposals,&ranking,&outcomes,proposal_hash,ranking_hash,outcome_hash).unwrap()
    }

    #[test]
    fn evaluates_observed_reviewer_decisions_separately_from_proposal_ranker() {
        let report = evaluate_fixture();
        assert_eq!(report.status,"POST_HOC_REVIEW_OUTCOMES_EVALUATED");
        assert_eq!(report.candidate_count,3);
        assert_eq!(report.evaluated_policy_budget_count,4);
    }

    #[test]
    fn actual_reviewer_minutes_control_matched_budget_selection() {
        let report = evaluate_fixture();
        let one_minute = report.policy_budget_results.iter()
            .find(|row| row.policy=="policy_a" && row.review_budget_minutes==1.0).unwrap();
        assert_eq!(one_minute.reviewed_candidate_ids,vec!["remove_one"]);
        assert_eq!(one_minute.actual_review_minutes,1.0);
    }

    #[test]
    fn accepted_incorrect_proposal_is_applied_and_query_harm_is_measured() {
        let report = evaluate_fixture();
        let two_minutes = report.policy_budget_results.iter()
            .find(|row| row.policy=="policy_a" && row.review_budget_minutes==2.0).unwrap();
        assert_eq!(two_minutes.outcome_counts.accepted_correct,1);
        assert_eq!(two_minutes.outcome_counts.accepted_incorrect,1);
        assert_eq!(two_minutes.outcome_counts.applied_correct,1);
        assert_eq!(two_minutes.outcome_counts.applied_incorrect,1);
        assert!(two_minutes.weighted_query_error_reduction.unwrap() < 0.0);
    }

    #[test]
    fn rejected_correct_proposal_counts_as_missed_correction() {
        let report = evaluate_fixture();
        let two_minutes = report.policy_budget_results.iter()
            .find(|row| row.policy=="policy_a" && row.review_budget_minutes==2.0).unwrap();
        assert_eq!(two_minutes.outcome_counts.rejected_correct,0);
        // The rejected correct proposal is not selected under this 2-minute budget.
        assert_eq!(two_minutes.outcome_counts.reviewed_correct,1);
    }

    #[test]
    fn candidate_pool_recall_and_acceptance_precision_have_distinct_denominators() {
        let report = evaluate_fixture();
        let row = report.policy_budget_results.iter()
            .find(|row| row.policy=="policy_a" && row.review_budget_minutes==2.0).unwrap();
        assert_eq!(row.candidate_pool_correct_count,2);
        assert_eq!(row.candidate_pool_incorrect_count,1);
        assert_eq!(row.accepted_decision_precision,Some(0.5));
        assert_eq!(row.correct_proposal_review_recall,Some(0.5));
        assert_eq!(row.correct_proposal_acceptance_recall,Some(0.5));
        assert_eq!(row.incorrect_proposal_acceptance_rate,Some(1.0));
    }

    #[test]
    fn reviewer_rejection_prevents_application_but_does_not_erase_adjudication() {
        let report = evaluate_fixture();
        let row = report.policy_budget_results.iter()
            .find(|row| row.policy=="policy_a" && row.review_budget_minutes==3.0).unwrap();
        assert_eq!(row.outcome_counts.rejected_correct,1);
        assert!(!row.applied_candidate_ids.contains(&"remove_two".to_string()));
    }

    #[test]
    fn conflicting_accepted_proposal_is_recorded_as_stale_not_applied() {
        let mut proposals = proposals();
        proposals.candidates.truncate(2);
        let proposal_hash = sha256(&serde_json::to_vec(&proposals).unwrap());
        let ids = vec!["remove_one".to_string(),"remove_one_again".to_string()];
        let mut second = proposals.candidates[0].clone();
        second.candidate_id = "remove_one_again".into();
        proposals.candidates.push(second);
        let proposal_hash = sha256(&serde_json::to_vec(&proposals).unwrap());
        let ranking = ranking(&proposals,&proposal_hash);
        let ranking_bytes = serde_json::to_vec(&ranking).unwrap();
        let ranking_hash = sha256(&ranking_bytes);
        let outcomes: OutcomeFile = serde_json::from_value(json!({
            "schema_version":1,
            "source_input_sha256":proposal_hash,
            "source_ranking_report_sha256":ranking_hash,
            "adjudication_protocol_sha256":"9".repeat(64),
            "candidate_outcomes":[
                {"candidate_id":"remove_one","adjudication":"correct","reviewer_decision":"accept","actual_review_minutes":1.0},
                {"candidate_id":"add_one_wrong","adjudication":"incorrect","reviewer_decision":"reject","actual_review_minutes":1.0},
                {"candidate_id":"remove_two","adjudication":"correct","reviewer_decision":"reject","actual_review_minutes":1.0},
                {"candidate_id":"remove_one_again","adjudication":"correct","reviewer_decision":"accept","actual_review_minutes":1.0}
            ]
        })).unwrap();
        let output=evaluate(&proposals,&ranking,&outcomes,proposal_hash,ranking_hash,"8".repeat(64)).unwrap();
        let row=output.policy_budget_results.iter()
            .find(|row|row.policy=="policy_a"&&row.review_budget_minutes==3.0).unwrap();
        assert_eq!(row.stale_conflict_count,1);
        assert!(row.stale_conflict_candidate_ids.contains(&"remove_one_again".to_string()));
    }

    #[test]
    fn rejects_outcome_file_bound_to_different_proposal_bytes() {
        let report=evaluate_fixture();
        let proposals=proposals();
        let ranking=ranking(&proposals,&report.proposal_input_sha256);
        let outcomes=outcomes(&proposals,&"0".repeat(64),&report.ranking_report_sha256);
        assert!(validate_outcomes(&outcomes,&proposals,&report.proposal_input_sha256,&report.ranking_report_sha256)
            .unwrap_err().contains("source_input_sha256"));
    }

    #[test]
    fn rejects_outcome_file_bound_to_different_ranking_report() {
        let report=evaluate_fixture();
        let proposals=proposals();
        let outcomes=outcomes(&proposals,&report.proposal_input_sha256,&"0".repeat(64));
        assert!(validate_outcomes(&outcomes,&proposals,&report.proposal_input_sha256,&report.ranking_report_sha256)
            .unwrap_err().contains("source_ranking_report_sha256"));
    }

    #[test]
    fn rejects_missing_or_duplicate_candidate_outcomes() {
        let proposals=proposals();
        let input_hash="1".repeat(64);
        let ranking_hash="2".repeat(64);
        let mut outcomes=outcomes(&proposals,&input_hash,&ranking_hash);
        outcomes.candidate_outcomes.pop();
        assert!(validate_outcomes(&outcomes,&proposals,&input_hash,&ranking_hash).unwrap_err().contains("exactly one"));
        outcomes.candidate_outcomes=outcomes(&proposals,&input_hash,&ranking_hash).candidate_outcomes;
        outcomes.candidate_outcomes.push(outcomes.candidate_outcomes[0].clone());
        assert!(validate_outcomes(&outcomes,&proposals,&input_hash,&ranking_hash).unwrap_err().contains("unique"));
    }

    #[test]
    fn rejects_incomplete_or_duplicate_frozen_policy_orders() {
        let proposals=proposals();
        let input_hash=sha256(&serde_json::to_vec(&proposals).unwrap());
        let mut ranking=ranking(&proposals,&input_hash);
        ranking.policy_budget_results[0].policy_order_candidate_ids.pop();
        let outcomes=outcomes(&proposals,&input_hash,&"3".repeat(64));
        assert!(evaluate(&proposals,&ranking,&outcomes,input_hash,"3".repeat(64),"4".repeat(64))
            .unwrap_err().contains("permutation"));
    }

    #[test]
    fn indeterminate_adjudications_are_excluded_from_precision_denominators() {
        let mut proposals=proposals();
        proposals.candidates.truncate(1);
        let input_hash=sha256(&serde_json::to_vec(&proposals).unwrap());
        let ranking = FrozenRankingReport {
            schema_version:1,
            input_sha256:input_hash.clone(),
            baseline_graph_canonical_sha256:canonical_graph_sha256(&graph_from_edges(&proposals.graph)),
            policy_budget_results:vec![FrozenPolicyOrder {
                policy:"single".into(),review_budget_minutes:2.0,policy_order_candidate_ids:vec!["remove_one".into()]
            }]
        };
        let ranking_hash="5".repeat(64);
        let outcomes:OutcomeFile=serde_json::from_value(json!({
            "schema_version":1,
            "source_input_sha256":input_hash,
            "source_ranking_report_sha256":ranking_hash,
            "adjudication_protocol_sha256":"9".repeat(64),
            "candidate_outcomes":[
                {"candidate_id":"remove_one","adjudication":"indeterminate","reviewer_decision":"accept","actual_review_minutes":1.0}
            ]
        })).unwrap();
        let report=evaluate(&proposals,&ranking,&outcomes,"6".repeat(64),ranking_hash,"7".repeat(64)).unwrap();
        let row=&report.policy_budget_results[0];
        assert_eq!(row.accepted_decision_precision,None);
        assert_eq!(row.outcome_counts.accepted_indeterminate,1);
        assert_eq!(row.outcome_counts.applied_indeterminate,1);
    }
}
