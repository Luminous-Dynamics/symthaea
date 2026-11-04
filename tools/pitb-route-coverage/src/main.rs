use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

const RETRIEVAL_SCHEMA: &str = "pitb.retrieval-query-receipt.v0";
const SUPPORT_REPORT_SCHEMA: &str = "pitb.support-antichain-report.v0";
const MAPPING_SCHEMA: &str = "pitb.retrieval-declaration-mapping.v0";
const REPORT_SCHEMA: &str = "pitb.retrieval-support-route-coverage.v0";
const COVERAGE_K: [usize; 6] = [1, 5, 10, 32, 64, 100];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: String,
    evidence_class: String,
    scheduler_admission: bool,
    retriever: Retriever,
    benchmark: Benchmark,
    query: Query,
    graph: GraphBinding,
    candidate_pool: CandidatePool,
    ranking: Ranking,
    budget: Budget,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Retriever {
    kind: String,
    name: String,
    version: String,
    source_repository_commit: String,
    implementation_sha256: String,
    configuration_sha256: String,
    checkpoint_sha256: Option<String>,
    query_formulator_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Benchmark {
    benchmark_id: String,
    benchmark_version: String,
    split_id: String,
    split_role: String,
    split_manifest_sha256: String,
    label_set_sha256: Option<String>,
    labels_sequestered: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    query_text_sha256: String,
    target_statement_sha256: Option<String>,
    target_environment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphBinding {
    snapshot_id: String,
    repository_commit: String,
    environment_sha256: String,
    graph_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidatePool {
    candidate_count: usize,
    requested_top_k: usize,
    candidate_node_ids: Vec<String>,
    node_id_set_sha256: String,
    candidate_policy_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Ranking {
    score_metric: String,
    score_direction: String,
    calibration: String,
    candidates: Vec<RankedCandidate>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RankedCandidate {
    rank: usize,
    node_id: String,
    score: Option<f64>,
    upstream_candidate_id: Option<String>,
    source_relation_label: Option<String>,
    evidence_artifact_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Budget {
    query_wall_ms: u64,
    retrieval_cpu_ms: u64,
    retrieval_gpu_ms: u64,
    tokens: u64,
    cost_microusd: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Mapping {
    schema_version: String,
    graph_manifest_sha256: String,
    node_id: String,
    mapping_status: String,
    declaration_id: Option<String>,
    mapping_method: String,
    mapping_evidence_sha256: String,
    reviewer_attestation_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SupportReport {
    schema_version: String,
    groups: Vec<SupportGroup>,
}

#[derive(Debug, Clone, Deserialize)]
struct SupportGroup {
    group_id: String,
    study_id: String,
    component_id: String,
    target_id: String,
    analysis_endpoint: String,
    target_statement_sha256: String,
    environment_sha256: String,
    toolchain_sha256: String,
    budget_sha256: String,
    solver_policy_sha256: String,
    mutable_declaration_universe: Vec<String>,
    minimal_sufficient_support_antichain: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
struct KCoverage {
    k: usize,
    effective_k: usize,
    complete_route_covered: bool,
    minimal_routes_covered: usize,
    minimal_route_coverage: f64,
    best_route_completion_rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct RouteAssessment {
    query_id: String,
    group_id: String,
    study_id: String,
    component_id: String,
    target_id: String,
    target_statement_sha256: String,
    environment_sha256: String,
    toolchain_sha256: String,
    benchmark_id: String,
    benchmark_version: String,
    split_id: String,
    retriever_kind: String,
    retriever_name: String,
    retriever_version: String,
    receipt_sha256: String,
    graph_manifest_sha256: String,
    requested_top_k: usize,
    returned_candidate_count: usize,
    mapped_mutable_candidate_count: usize,
    mapped_out_of_universe_count: usize,
    unresolved_or_ambiguous_candidate_count: usize,
    useful_route_candidate_count: usize,
    useful_route_premise_rate: Option<f64>,
    necessary_declaration_count: usize,
    necessary_declaration_recall: Option<f64>,
    minimal_route_count: usize,
    minimal_routes_covered_at_returned_k: usize,
    minimal_route_coverage_at_returned_k: f64,
    any_complete_route_covered_at_returned_k: bool,
    any_exact_minimal_route_recovered: bool,
    mean_partial_route_fraction: f64,
    best_route_completion_rank: Option<usize>,
    cost_microusd: u64,
    retrieval_wall_ms: u64,
    retrieval_cpu_ms: u64,
    retrieval_gpu_ms: u64,
    tokens: u64,
    coverage_at_k: Vec<KCoverage>,
    interpretation_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonSummary {
    retriever_kind: String,
    retriever_name: String,
    retriever_version: String,
    matched_query_count: usize,
    matched_group_count: usize,
    any_complete_route_covered_count: usize,
    any_complete_route_coverage: f64,
    mean_minimal_route_coverage: f64,
    mean_useful_route_premise_rate: Option<f64>,
    mean_necessary_declaration_recall: Option<f64>,
    total_cost_microusd: u64,
    total_retrieval_wall_ms: u64,
    total_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
struct PairedComparison {
    reference_kind: String,
    reference_name: String,
    reference_version: String,
    challenger_kind: String,
    challenger_name: String,
    challenger_version: String,
    matched_assessment_count: usize,
    matched_query_count: usize,
    reference_only_route_count: usize,
    challenger_only_route_count: usize,
    both_route_count: usize,
    neither_route_count: usize,
    reference_complete_route_coverage: f64,
    challenger_complete_route_coverage: f64,
    complete_route_coverage_delta: f64,
    mean_minimal_route_coverage_delta: f64,
    component_cluster_count: usize,
    cluster_bootstrap_replicates: usize,
    cluster_bootstrap_ci_low: Option<f64>,
    cluster_bootstrap_ci_high: Option<f64>,
    exact_mcnemar_p_value: f64,
    holm_adjusted_mcnemar_p_value: Option<f64>,
    interpretation_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    support_report_sha256: String,
    retrieval_receipts_jsonl_sha256: String,
    declaration_mappings_jsonl_sha256: String,
    support_group_count: usize,
    retrieval_receipt_count: usize,
    declaration_mapping_count: usize,
    matched_assessment_count: usize,
    unmatched_receipt_count: usize,
    unmapped_candidate_count: usize,
    route_assessments: Vec<RouteAssessment>,
    comparison_summaries: Vec<ComparisonSummary>,
    paired_comparisons: Vec<PairedComparison>,
    admission_boundary: String,
    interpretation_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 64) {
        Ok(())
    } else {
        Err(format!("{label} must be lowercase SHA-256 hex"))
    }
}

fn commit(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 40) {
        Ok(())
    } else {
        Err(format!("{label} must be a 40-character lowercase Git SHA-1"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn sorted_unique(values: &[String], label: &str) -> Result<(), String> {
    if values.iter().any(|x| x.trim().is_empty()) {
        return Err(format!("{label} contains an empty ID"));
    }
    if !values.windows(2).all(|w| w[0] < w[1]) {
        return Err(format!("{label} must be sorted and unique"));
    }
    Ok(())
}

fn jsonl<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<Vec<T>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("{label} is not UTF-8: {e}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        rows.push(serde_json::from_str(line)
            .map_err(|e| format!("{label} line {}: {e}", index + 1))?);
    }
    if rows.is_empty() {
        return Err(format!("{label} must contain at least one row"));
    }
    Ok(rows)
}

fn validate_support_group(group: &SupportGroup, index: usize) -> Result<(), String> {
    let label = format!("support group {}", index + 1);
    for (field, value) in [
        ("group_id", group.group_id.as_str()),
        ("study_id", group.study_id.as_str()),
        ("component_id", group.component_id.as_str()),
        ("target_id", group.target_id.as_str()),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    for (field, value) in [
        ("group_id", group.group_id.as_str()),
        ("target_statement_sha256", group.target_statement_sha256.as_str()),
        ("environment_sha256", group.environment_sha256.as_str()),
        ("toolchain_sha256", group.toolchain_sha256.as_str()),
        ("budget_sha256", group.budget_sha256.as_str()),
        ("solver_policy_sha256", group.solver_policy_sha256.as_str()),
    ] {
        digest(value, &format!("{label}.{field}"))?;
    }
    if group.analysis_endpoint != "exact_target_replan" {
        return Err(format!("{label} is not an exact_target_replan group"));
    }
    sorted_unique(&group.mutable_declaration_universe, &format!("{label}.mutable_declaration_universe"))?;
    if group.minimal_sufficient_support_antichain.is_empty() {
        return Err(format!("{label} has no minimal sufficient support routes"));
    }
    let universe: BTreeSet<&str> = group.mutable_declaration_universe.iter().map(String::as_str).collect();
    let mut routes = BTreeSet::<Vec<String>>::new();
    for (route_index, route) in group.minimal_sufficient_support_antichain.iter().enumerate() {
        sorted_unique(route, &format!("{label}.route[{}]", route_index + 1))?;
        if route.is_empty() {
            return Err(format!("{label} has an empty minimal route"));
        }
        if route.iter().any(|declaration| !universe.contains(declaration.as_str())) {
            return Err(format!("{label} route contains declaration outside mutable universe"));
        }
        if !routes.insert(route.clone()) {
            return Err(format!("{label} contains duplicate minimal routes"));
        }
    }
    for (i, a) in group.minimal_sufficient_support_antichain.iter().enumerate() {
        for (j, b) in group.minimal_sufficient_support_antichain.iter().enumerate() {
            if i != j && a.len() <= b.len() && a.iter().all(|x| b.binary_search(x).is_ok()) {
                return Err(format!("{label} routes are not an inclusion-minimal antichain"));
            }
        }
    }
    Ok(())
}

fn validate_receipt(receipt: &Receipt, index: usize) -> Result<(), String> {
    let label = format!("retrieval receipt {}", index + 1);
    if receipt.schema_version != RETRIEVAL_SCHEMA
        || receipt.evidence_class != "external_candidate_run"
        || receipt.scheduler_admission
    {
        return Err(format!("{label} schema/evidence/admission boundary invalid"));
    }
    for (field, value) in [
        ("retriever.kind", receipt.retriever.kind.as_str()),
        ("retriever.name", receipt.retriever.name.as_str()),
        ("retriever.version", receipt.retriever.version.as_str()),
        ("benchmark_id", receipt.benchmark.benchmark_id.as_str()),
        ("benchmark_version", receipt.benchmark.benchmark_version.as_str()),
        ("split_id", receipt.benchmark.split_id.as_str()),
        ("query_id", receipt.query.query_id.as_str()),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    if !receipt.benchmark.labels_sequestered {
        return Err(format!("{label} evaluation labels are not sequestered"));
    }
    if receipt.ranking.calibration != "uncalibrated" {
        return Err(format!("{label} external ranking scores must remain uncalibrated"));
    }
    digest(&receipt.query.query_text_sha256, &format!("{label}.query_text_sha256"))?;
    digest(&receipt.query.target_environment_sha256, &format!("{label}.target_environment_sha256"))?;
    let target = receipt.query.target_statement_sha256.as_ref()
        .ok_or_else(|| format!("{label} has no exact target statement digest"))?;
    digest(target, &format!("{label}.target_statement_sha256"))?;
    digest(&receipt.graph.environment_sha256, &format!("{label}.graph.environment_sha256"))?;
    digest(&receipt.graph.graph_manifest_sha256, &format!("{label}.graph.graph_manifest_sha256"))?;
    commit(&receipt.graph.repository_commit, &format!("{label}.graph.repository_commit"))?;
    if receipt.query.target_environment_sha256 != receipt.graph.environment_sha256 {
        return Err(format!("{label} target and graph environments differ"));
    }
    digest(&receipt.candidate_pool.node_id_set_sha256, &format!("{label}.candidate_pool.node_id_set_sha256"))?;
    digest(&receipt.candidate_pool.candidate_policy_sha256, &format!("{label}.candidate_pool.candidate_policy_sha256"))?;
    if receipt.candidate_pool.candidate_count != receipt.candidate_pool.candidate_node_ids.len() {
        return Err(format!("{label} candidate pool count differs from node IDs"));
    }
    sorted_unique(&receipt.candidate_pool.candidate_node_ids, &format!("{label}.candidate_pool.candidate_node_ids"))?;
    let candidate_ids: BTreeSet<String> = receipt.candidate_pool.candidate_node_ids.iter().cloned().collect();
    let candidate_id_bytes = serde_json::to_vec(&candidate_ids).map_err(|e| format!("serialize candidate node ID set: {e}"))?;
    if sha256(&candidate_id_bytes) != receipt.candidate_pool.node_id_set_sha256 {
        return Err(format!("{label} candidate node-set digest does not match exact candidate IDs"));
    }
    if receipt.candidate_pool.requested_top_k == 0
        || receipt.candidate_pool.requested_top_k > receipt.candidate_pool.candidate_count
        || receipt.ranking.candidates.len() > receipt.candidate_pool.requested_top_k
    {
        return Err(format!("{label} top-k is outside candidate pool"));
    }
    let candidate_set: BTreeSet<&str> = receipt.candidate_pool.candidate_node_ids.iter().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    let mut previous_score: Option<f64> = None;
    for (position, candidate) in receipt.ranking.candidates.iter().enumerate() {
        if candidate.rank != position + 1 {
            return Err(format!("{label} ranks must be contiguous and 1-based"));
        }
        if !seen.insert(candidate.node_id.as_str()) || !candidate_set.contains(candidate.node_id.as_str()) {
            return Err(format!("{label} contains duplicate or out-of-pool ranked candidate"));
        }
        digest(&candidate.evidence_artifact_sha256, &format!("{label}.candidate[{}].evidence", position + 1))?;
        match (receipt.ranking.score_metric.as_str(), receipt.ranking.score_direction.as_str(), candidate.score) {
            ("rank_only", "rank_only", None) => {}
            ("rank_only", _, _) => return Err(format!("{label} rank_only score semantics are inconsistent")),
            (_, "higher_better", Some(score)) if score.is_finite() => {
                if previous_score.is_some_and(|previous| previous < score) {
                    return Err(format!("{label} scores contradict higher_better ordering"));
                }
                previous_score = Some(score);
            }
            (_, "lower_better", Some(score)) if score.is_finite() => {
                if previous_score.is_some_and(|previous| previous > score) {
                    return Err(format!("{label} scores contradict lower_better ordering"));
                }
                previous_score = Some(score);
            }
            _ => return Err(format!("{label} has unsupported score/direction combination")),
        }
    }
    Ok(())
}

fn validate_mapping(mapping: &Mapping, index: usize) -> Result<(), String> {
    let label = format!("candidate mapping {}", index + 1);
    if mapping.schema_version != MAPPING_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    digest(&mapping.graph_manifest_sha256, &format!("{label}.graph_manifest_sha256"))?;
    digest(&mapping.mapping_evidence_sha256, &format!("{label}.mapping_evidence_sha256"))?;
    non_empty(&mapping.node_id, &format!("{label}.node_id"))?;
    if !matches!(mapping.mapping_status.as_str(), "exact" | "unresolved" | "ambiguous") {
        return Err(format!("{label} has unsupported mapping_status"));
    }
    if !matches!(mapping.mapping_method.as_str(), "exact_formal_name" | "exact_source_identity" | "human_reviewed") {
        return Err(format!("{label} has unsupported mapping_method"));
    }
    match mapping.mapping_status.as_str() {
        "exact" => {
            let id = mapping.declaration_id.as_ref().ok_or_else(|| format!("{label} exact mapping has no declaration_id"))?;
            non_empty(id, &format!("{label}.declaration_id"))?;
        }
        "unresolved" | "ambiguous" => {
            if mapping.declaration_id.is_some() {
                return Err(format!("{label} non-exact mapping cannot carry declaration_id"));
            }
        }
        _ => unreachable!(),
    }
    if mapping.mapping_method == "human_reviewed" {
        let reviewer = mapping.reviewer_attestation_sha256.as_ref()
            .ok_or_else(|| format!("{label} human mapping lacks reviewer attestation"))?;
        digest(reviewer, &format!("{label}.reviewer_attestation_sha256"))?;
    } else if mapping.reviewer_attestation_sha256.is_some() {
        return Err(format!("{label} non-human mapping cannot carry reviewer attestation"));
    }
    Ok(())
}

fn route_covered(route: &[String], retrieved: &BTreeSet<String>) -> bool {
    route.iter().all(|declaration| retrieved.contains(declaration))
}

fn partial_fraction(route: &[String], retrieved: &BTreeSet<String>) -> f64 {
    route.iter().filter(|declaration| retrieved.contains(*declaration)).count() as f64 / route.len() as f64
}

fn declaration_first_rank(
    declaration: &str,
    candidates: &[RankedCandidate],
    graph_sha: &str,
    mappings: &BTreeMap<(String, String), &Mapping>,
    limit: usize,
) -> Option<usize> {
    candidates.iter().take(limit).filter_map(|candidate| {
        let key = (graph_sha.to_owned(), candidate.node_id.clone());
        mappings.get(&key)
            .filter(|mapping| mapping.mapping_status == "exact")
            .and_then(|mapping| mapping.declaration_id.as_deref())
            .filter(|id| *id == declaration)
            .map(|_| candidate.rank)
    }).min()
}

fn route_completion_rank(
    route: &[String],
    candidates: &[RankedCandidate],
    graph_sha: &str,
    mappings: &BTreeMap<(String, String), &Mapping>,
    limit: usize,
) -> Option<usize> {
    let mut ranks = Vec::with_capacity(route.len());
    for declaration in route {
        ranks.push(declaration_first_rank(declaration, candidates, graph_sha, mappings, limit)?);
    }
    ranks.into_iter().max()
}

fn build_coverage(
    receipt: &Receipt,
    group: &SupportGroup,
    mappings: &BTreeMap<(String, String), &Mapping>,
) -> Result<RouteAssessment, String> {
    let graph_sha = &receipt.graph.graph_manifest_sha256;
    let universe: BTreeSet<String> = group.mutable_declaration_universe.iter().cloned().collect();
    let mut retrieved = BTreeSet::<String>::new();
    let mut mapped_outside = 0usize;
    let mut unresolved = 0usize;
    for candidate in &receipt.ranking.candidates {
        let key = (graph_sha.clone(), candidate.node_id.clone());
        let mapping = mappings.get(&key)
            .ok_or_else(|| format!(
                "no explicit declaration crosswalk for graph node {} in query {}",
                candidate.node_id, receipt.query.query_id
            ))?;
        match mapping.mapping_status.as_str() {
            "exact" => {
                let declaration = mapping.declaration_id.as_ref().ok_or("exact mapping lost declaration ID")?;
                if universe.contains(declaration) {
                    retrieved.insert(declaration.clone());
                } else {
                    mapped_outside += 1;
                }
            }
            "unresolved" | "ambiguous" => unresolved += 1,
            _ => return Err("unsupported mapping state".into()),
        }
    }
    let route_union: BTreeSet<String> = group.minimal_sufficient_support_antichain
        .iter().flatten().cloned().collect();
    let useful = retrieved.intersection(&route_union).count();
    let useful_rate = if retrieved.is_empty() { None } else { Some(useful as f64 / retrieved.len() as f64) };
    let necessary: BTreeSet<String> = if group.minimal_sufficient_support_antichain.is_empty() {
        BTreeSet::new()
    } else {
        group.minimal_sufficient_support_antichain[0].iter()
            .filter(|id| group.minimal_sufficient_support_antichain.iter().all(|route| route.binary_search(id).is_ok()))
            .cloned()
            .collect()
    };
    let necessary_recall = if necessary.is_empty() {
        None
    } else {
        Some(necessary.intersection(&retrieved).count() as f64 / necessary.len() as f64)
    };
    let mut complete = 0usize;
    let mut exact = false;
    let mut best = None::<usize>;
    let mut partial_sum = 0.0;
    for route in &group.minimal_sufficient_support_antichain {
        partial_sum += partial_fraction(route, &retrieved);
        if route_covered(route, &retrieved) {
            complete += 1;
            if let Some(completion_rank) = route_completion_rank(
                route,
                &receipt.ranking.candidates,
                graph_sha,
                mappings,
                receipt.ranking.candidates.len(),
            ) {
                best = Some(best.map_or(completion_rank, |current| current.min(completion_rank)));
            }
            let route_set: BTreeSet<String> = route.iter().cloned().collect();
            if route_set == retrieved {
                exact = true;
            }
        }
    }
    let route_count = group.minimal_sufficient_support_antichain.len();
    let actual_coverage = complete as f64 / route_count as f64;
    let mean_partial = partial_sum / route_count as f64;
    let mut coverage_at_k = Vec::new();
    for k in COVERAGE_K {
        let effective_k = k.min(receipt.ranking.candidates.len());
        let mut prefix_set = BTreeSet::<String>::new();
        for candidate in receipt.ranking.candidates.iter().take(effective_k) {
            let mapping = mappings.get(&(graph_sha.clone(), candidate.node_id.clone()))
                .ok_or_else(|| format!("missing mapping at prefix rank {}", candidate.rank))?;
            if mapping.mapping_status == "exact" {
                if let Some(declaration) = &mapping.declaration_id {
                    if universe.contains(declaration) {
                        prefix_set.insert(declaration.clone());
                    }
                }
            }
        }
        let covered = group.minimal_sufficient_support_antichain.iter()
            .filter(|route| route_covered(route, &prefix_set)).count();
        let best_rank = group.minimal_sufficient_support_antichain.iter()
            .filter_map(|route| route_completion_rank(
                route,
                &receipt.ranking.candidates,
                graph_sha,
                mappings,
                effective_k,
            ))
            .min();
        coverage_at_k.push(KCoverage {
            k,
            effective_k,
            complete_route_covered: covered > 0,
            minimal_routes_covered: covered,
            minimal_route_coverage: covered as f64 / route_count as f64,
            best_route_completion_rank: best_rank,
        });
    }
    let receipt_bytes = serde_json::to_vec(receipt).map_err(|e| format!("serialize receipt: {e}"))?;
    Ok(RouteAssessment {
        query_id: receipt.query.query_id.clone(),
        group_id: group.group_id.clone(),
        study_id: group.study_id.clone(),
        component_id: group.component_id.clone(),
        target_id: group.target_id.clone(),
        target_statement_sha256: group.target_statement_sha256.clone(),
        environment_sha256: group.environment_sha256.clone(),
        toolchain_sha256: group.toolchain_sha256.clone(),
        benchmark_id: receipt.benchmark.benchmark_id.clone(),
        benchmark_version: receipt.benchmark.benchmark_version.clone(),
        split_id: receipt.benchmark.split_id.clone(),
        retriever_kind: receipt.retriever.kind.clone(),
        retriever_name: receipt.retriever.name.clone(),
        retriever_version: receipt.retriever.version.clone(),
        receipt_sha256: sha256(&receipt_bytes),
        graph_manifest_sha256: graph_sha.clone(),
        requested_top_k: receipt.candidate_pool.requested_top_k,
        returned_candidate_count: receipt.ranking.candidates.len(),
        mapped_mutable_candidate_count: retrieved.len(),
        mapped_out_of_universe_count: mapped_outside,
        unresolved_or_ambiguous_candidate_count: unresolved,
        useful_route_candidate_count: useful,
        useful_route_premise_rate: useful_rate,
        necessary_declaration_count: necessary.len(),
        necessary_declaration_recall: necessary_recall,
        minimal_route_count: route_count,
        minimal_routes_covered_at_returned_k: complete,
        minimal_route_coverage_at_returned_k: actual_coverage,
        any_complete_route_covered_at_returned_k: complete > 0,
        any_exact_minimal_route_recovered: exact,
        mean_partial_route_fraction: mean_partial,
        best_route_completion_rank: best,
        cost_microusd: receipt.budget.cost_microusd,
        retrieval_wall_ms: receipt.budget.query_wall_ms,
        retrieval_cpu_ms: receipt.budget.retrieval_cpu_ms,
        retrieval_gpu_ms: receipt.budget.retrieval_gpu_ms,
        tokens: receipt.budget.tokens,
        coverage_at_k,
        interpretation_boundary: "Candidate coverage only. A complete retrieved route was independently shown sufficient under the support experiment's frozen target, fixed context, toolchain, policy and budget. This join does not prove the retrieved node-to-declaration crosswalk is semantically correct, does not replay a proof, and does not authorize scheduler admission.".into(),
    })
}

fn summarize(assessments: &[RouteAssessment]) -> Vec<ComparisonSummary> {
    let mut groups = BTreeMap::<(String, String, String), Vec<&RouteAssessment>>::new();
    for assessment in assessments {
        groups.entry((
            assessment.retriever_kind.clone(),
            assessment.retriever_name.clone(),
            assessment.retriever_version.clone(),
        )).or_default().push(assessment);
    }
    groups.into_iter().map(|((kind, name, version), rows)| {
        let covered = rows.iter().filter(|row| row.any_complete_route_covered_at_returned_k).count();
        let mean_route = rows.iter().map(|row| row.minimal_route_coverage_at_returned_k).sum::<f64>() / rows.len() as f64;
        let useful_values: Vec<f64> = rows.iter().filter_map(|row| row.useful_route_premise_rate).collect();
        let necessary_values: Vec<f64> = rows.iter().filter_map(|row| row.necessary_declaration_recall).collect();
        let mut unique_receipts = BTreeMap::<String, &RouteAssessment>::new();
        for row in &rows {
            unique_receipts.entry(row.receipt_sha256.clone()).or_insert(row);
        }
        ComparisonSummary {
            retriever_kind: kind,
            retriever_name: name,
            retriever_version: version,
            matched_query_count: rows.iter().map(|x| x.query_id.as_str()).collect::<BTreeSet<_>>().len(),
            matched_group_count: rows.len(),
            any_complete_route_covered_count: covered,
            any_complete_route_coverage: covered as f64 / rows.len() as f64,
            mean_minimal_route_coverage: mean_route,
            mean_useful_route_premise_rate: if useful_values.is_empty() { None } else { Some(useful_values.iter().sum::<f64>() / useful_values.len() as f64) },
            mean_necessary_declaration_recall: if necessary_values.is_empty() { None } else { Some(necessary_values.iter().sum::<f64>() / necessary_values.len() as f64) },
            total_cost_microusd: unique_receipts.values().map(|x| x.cost_microusd).sum(),
            total_retrieval_wall_ms: unique_receipts.values().map(|x| x.retrieval_wall_ms).sum(),
            total_tokens: unique_receipts.values().map(|x| x.tokens).sum(),
        }
    }).collect()
}

type RetrieverKey = (String, String, String);
type AssessmentKey = (String, String, String, String, String);

fn retriever_key(row: &RouteAssessment) -> RetrieverKey {
    (row.retriever_kind.clone(), row.retriever_name.clone(), row.retriever_version.clone())
}

fn assessment_key(row: &RouteAssessment) -> AssessmentKey {
    (
        row.benchmark_id.clone(),
        row.benchmark_version.clone(),
        row.split_id.clone(),
        row.query_id.clone(),
        row.group_id.clone(),
    )
}

fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn deterministic_seed(reference: &RetrieverKey, challenger: &RetrieverKey) -> u64 {
    let label = format!(
        "{}|{}|{}||{}|{}|{}",
        reference.0, reference.1, reference.2,
        challenger.0, challenger.1, challenger.2
    );
    let hash = sha256(label.as_bytes());
    u64::from_str_radix(&hash[..16], 16).unwrap_or(0x9e3779b97f4a7c15).max(1)
}

fn percentile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() { return None; }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        Some(sorted[low])
    } else {
        let fraction = position - low as f64;
        Some(sorted[low] * (1.0 - fraction) + sorted[high] * fraction)
    }
}

fn cluster_bootstrap_interval(
    cluster_deltas: &BTreeMap<String, Vec<f64>>,
    seed: u64,
    replicates: usize,
) -> (Option<f64>, Option<f64>) {
    if cluster_deltas.len() < 2 || replicates == 0 {
        return (None, None);
    }
    let clusters: Vec<&Vec<f64>> = cluster_deltas.values().collect();
    let mut state = seed.max(1);
    let mut values = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = 0.0;
        let mut count = 0usize;
        for _ in 0..clusters.len() {
            let index = (xorshift_next(&mut state) as usize) % clusters.len();
            for delta in clusters[index] {
                sum += delta;
                count += 1;
            }
        }
        if count > 0 {
            values.push(sum / count as f64);
        }
    }
    values.sort_by(f64::total_cmp);
    (percentile(&values, 0.025), percentile(&values, 0.975))
}

fn exact_mcnemar_p_value(challenger_only: usize, reference_only: usize) -> f64 {
    let n = challenger_only + reference_only;
    if n == 0 {
        return 1.0;
    }
    let k = challenger_only.min(reference_only);
    let mut probability = 2.0_f64.powi(-(n as i32));
    let mut lower_tail = probability;
    for i in 1..=k {
        probability *= (n - i + 1) as f64 / i as f64;
        lower_tail += probability;
    }
    (2.0 * lower_tail).min(1.0)
}

fn build_paired_comparisons(assessments: &[RouteAssessment]) -> Vec<PairedComparison> {
    let mut systems = BTreeMap::<RetrieverKey, BTreeMap<AssessmentKey, &RouteAssessment>>::new();
    for row in assessments {
        systems.entry(retriever_key(row)).or_default().insert(assessment_key(row), row);
    }
    let keys: Vec<RetrieverKey> = systems.keys().cloned().collect();
    let mut comparisons = Vec::new();
    for reference_index in 0..keys.len() {
        for challenger_index in (reference_index + 1)..keys.len() {
            let reference_key = &keys[reference_index];
            let challenger_key = &keys[challenger_index];
            let reference_rows = &systems[reference_key];
            let challenger_rows = &systems[challenger_key];
            let mut matched = Vec::<(&RouteAssessment, &RouteAssessment)>::new();
            for (key, reference) in reference_rows {
                if let Some(challenger) = challenger_rows.get(key) {
                    if reference.graph_manifest_sha256 != challenger.graph_manifest_sha256
                        || reference.target_statement_sha256 != challenger.target_statement_sha256
                        || reference.environment_sha256 != challenger.environment_sha256
                        || reference.component_id != challenger.component_id
                    {
                        // Same query/group identity but different graph or target is not a matched trial.
                        continue;
                    }
                    matched.push((reference, *challenger));
                }
            }
            if matched.is_empty() { continue; }
            let mut reference_only = 0usize;
            let mut challenger_only = 0usize;
            let mut both = 0usize;
            let mut neither = 0usize;
            let mut route_delta_sum = 0.0;
            let mut clusters = BTreeMap::<String, Vec<f64>>::new();
            let mut matched_queries = BTreeSet::<String>::new();
            for (reference, challenger) in &matched {
                let r = reference.any_complete_route_covered_at_returned_k;
                let c = challenger.any_complete_route_covered_at_returned_k;
                match (r, c) {
                    (true, true) => both += 1,
                    (true, false) => reference_only += 1,
                    (false, true) => challenger_only += 1,
                    (false, false) => neither += 1,
                }
                let delta = challenger.minimal_route_coverage_at_returned_k
                    - reference.minimal_route_coverage_at_returned_k;
                route_delta_sum += delta;
                clusters.entry(reference.component_id.clone()).or_default().push(delta);
                matched_queries.insert(reference.query_id.clone());
            }
            let n = matched.len();
            let reference_coverage = (reference_only + both) as f64 / n as f64;
            let challenger_coverage = (challenger_only + both) as f64 / n as f64;
            let (ci_low, ci_high) = cluster_bootstrap_interval(
                &clusters,
                deterministic_seed(reference_key, challenger_key),
                2000,
            );
            comparisons.push(PairedComparison {
                reference_kind: reference_key.0.clone(),
                reference_name: reference_key.1.clone(),
                reference_version: reference_key.2.clone(),
                challenger_kind: challenger_key.0.clone(),
                challenger_name: challenger_key.1.clone(),
                challenger_version: challenger_key.2.clone(),
                matched_assessment_count: n,
                matched_query_count: matched_queries.len(),
                reference_only_route_count: reference_only,
                challenger_only_route_count: challenger_only,
                both_route_count: both,
                neither_route_count: neither,
                reference_complete_route_coverage: reference_coverage,
                challenger_complete_route_coverage: challenger_coverage,
                complete_route_coverage_delta: challenger_coverage - reference_coverage,
                mean_minimal_route_coverage_delta: route_delta_sum / n as f64,
                component_cluster_count: clusters.len(),
                cluster_bootstrap_replicates: if clusters.len() >= 2 { 2000 } else { 0 },
                cluster_bootstrap_ci_low: ci_low,
                cluster_bootstrap_ci_high: ci_high,
                exact_mcnemar_p_value: exact_mcnemar_p_value(challenger_only, reference_only),
                holm_adjusted_mcnemar_p_value: None,
                interpretation_boundary: "Paired descriptive comparison on identical query/support-group keys and graph/target/environment identity. Confidence intervals resample component clusters, not individual rows. P-values are exact two-sided McNemar tests and must be corrected for multiplicity across the comparison family.".into(),
            });
        }
    }
    let mut order: Vec<usize> = (0..comparisons.len()).collect();
    order.sort_by(|a, b| comparisons[*a].exact_mcnemar_p_value.total_cmp(&comparisons[*b].exact_mcnemar_p_value));
    let family_size = order.len();
    let mut adjusted_floor = 0.0_f64;
    for (rank, index) in order.into_iter().enumerate() {
        let multiplier = (family_size - rank) as f64;
        adjusted_floor = adjusted_floor.max(
            (comparisons[index].exact_mcnemar_p_value * multiplier).min(1.0)
        );
        comparisons[index].holm_adjusted_mcnemar_p_value = Some(adjusted_floor);
    }
    comparisons
}

fn analyze(
    support_bytes: &[u8],
    retrieval_bytes: &[u8],
    mapping_bytes: &[u8],
) -> Result<Report, String> {
    let support: SupportReport = serde_json::from_slice(support_bytes)
        .map_err(|e| format!("parse support antichain report: {e}"))?;
    if support.schema_version != SUPPORT_REPORT_SCHEMA {
        return Err("unsupported support antichain report schema".into());
    }
    if support.groups.is_empty() {
        return Err("support antichain report has no groups".into());
    }
    let mut groups_by_target = BTreeMap::<(String, String), Vec<&SupportGroup>>::new();
    let mut seen_group_ids = BTreeSet::new();
    for (index, group) in support.groups.iter().enumerate() {
        validate_support_group(group, index)?;
        if !seen_group_ids.insert(group.group_id.clone()) {
            return Err(format!("duplicate support group id {}", group.group_id));
        }
        groups_by_target.entry((
            group.target_statement_sha256.clone(),
            group.environment_sha256.clone(),
        )).or_default().push(group);
    }
    let receipts: Vec<Receipt> = jsonl(retrieval_bytes, "retrieval receipts JSONL")?;
    let mappings: Vec<Mapping> = jsonl(mapping_bytes, "declaration mapping JSONL")?;
    let mut mapping_index = BTreeMap::<(String, String), &Mapping>::new();
    for (index, mapping) in mappings.iter().enumerate() {
        validate_mapping(mapping, index)?;
        let key = (mapping.graph_manifest_sha256.clone(), mapping.node_id.clone());
        if mapping_index.insert(key.clone(), mapping).is_some() {
            return Err(format!("duplicate declaration mapping for graph {} node {}", key.0, key.1));
        }
    }
    let mut used_mappings = BTreeSet::<(String, String)>::new();
    let mut seen_runs = BTreeSet::new();
    let mut assessments = Vec::new();
    let mut unmatched = 0usize;
    for (index, receipt) in receipts.iter().enumerate() {
        validate_receipt(receipt, index)?;
        let target_sha = receipt.query.target_statement_sha256.as_ref().unwrap();
        let key = (target_sha.clone(), receipt.query.target_environment_sha256.clone());
        let target_groups = groups_by_target.get(&key)
            .ok_or_else(|| format!("retrieval query {} has no matching exact target/environment support group", receipt.query.query_id))?;
        let run_key = format!(
            "{}|{}|{}|{}|{}|{}",
            receipt.benchmark.benchmark_id,
            receipt.benchmark.split_id,
            receipt.retriever.kind,
            receipt.retriever.name,
            receipt.retriever.version,
            receipt.query.query_id
        );
        if !seen_runs.insert(run_key.clone()) {
            return Err(format!("duplicate retrieval run identity {run_key}"));
        }
        if receipt.ranking.candidates.is_empty() {
            unmatched += 1;
            continue;
        }
        for candidate in &receipt.ranking.candidates {
            let map_key = (receipt.graph.graph_manifest_sha256.clone(), candidate.node_id.clone());
            if !mapping_index.contains_key(&map_key) {
                return Err(format!("missing mapping for returned candidate node {}", candidate.node_id));
            }
            used_mappings.insert(map_key);
        }
        for group in target_groups {
            assessments.push(build_coverage(&receipt, group, &mapping_index)?);
        }
    }
    if assessments.is_empty() {
        return Err("no retrieval query could be joined to a non-empty ranked candidate list".into());
    }
    let unused_mapping_count = mapping_index.keys().filter(|key| !used_mappings.contains(*key)).count();
    if unused_mapping_count > 0 {
        return Err(format!("{unused_mapping_count} declaration mappings are not referenced by any returned candidate"));
    }
    let total_unmapped = assessments.iter().map(|x| x.unresolved_or_ambiguous_candidate_count).sum();
    let summaries = summarize(&assessments);
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        support_report_sha256: sha256(support_bytes),
        retrieval_receipts_jsonl_sha256: sha256(retrieval_bytes),
        declaration_mappings_jsonl_sha256: sha256(mapping_bytes),
        support_group_count: support.groups.len(),
        retrieval_receipt_count: receipts.len(),
        declaration_mapping_count: mappings.len(),
        matched_assessment_count: assessments.len(),
        unmatched_receipt_count: unmatched,
        unmapped_candidate_count: total_unmapped,
        paired_comparisons: build_paired_comparisons(&assessments),
        route_assessments: assessments,
        comparison_summaries: summaries,
        admission_boundary: "All graph nodes remain retrieval candidates. The join never promotes external candidates to typed observed dependencies or scheduler-admissible proof context.".into(),
        interpretation_boundary: "The report measures whether explicit exact-name/source-identity/manual-review node-to-declaration crosswalks cover independently measured minimal sufficient routes. It does not establish theorem-level logical necessity, authenticate mapping evidence bytes, verify model correctness, or show that a model used the candidates. Complete-route coverage is an offline diagnostic and must not be presented as a proof solve.".into(),
    })
}

fn synthetic_support_report() -> Value {
    json!({
        "schema_version": SUPPORT_REPORT_SCHEMA,
        "groups": [{
            "group_id": sha256(b"group-a"),
            "study_id": "study-a",
            "component_id": "component-a",
            "target_id": "target-a",
            "analysis_endpoint": "exact_target_replan",
            "target_statement_sha256": sha256(b"target statement"),
            "environment_sha256": sha256(b"environment"),
            "toolchain_sha256": sha256(b"toolchain"),
            "budget_sha256": sha256(b"budget"),
            "solver_policy_sha256": sha256(b"policy"),
            "mutable_declaration_universe": ["h1","h2","h3"],
            "minimal_sufficient_support_antichain": [["h1","h2"],["h3"]]
        }]
    })
}

fn synthetic_receipt() -> Receipt {
    let target = sha256(b"target statement");
    let env = sha256(b"environment");
    let graph_sha = sha256(b"graph manifest");
    let candidates = vec![
        RankedCandidate {
            rank: 1, node_id: "node-h1".into(), score: Some(0.9),
            upstream_candidate_id: Some("u1".into()), source_relation_label: Some("USES_DEFINITION".into()),
            evidence_artifact_sha256: sha256(b"evidence-1")
        },
        RankedCandidate {
            rank: 2, node_id: "node-h2".into(), score: Some(0.8),
            upstream_candidate_id: Some("u2".into()), source_relation_label: Some("USES_CLASS".into()),
            evidence_artifact_sha256: sha256(b"evidence-2")
        },
    ];
    let mut ids = vec!["node-h1".to_string(),"node-h2".to_string(),"node-h3".to_string()];
    ids.sort();
    let set_bytes = serde_json::to_vec(&ids).unwrap();
    Receipt {
        schema_version: RETRIEVAL_SCHEMA.into(),
        evidence_class: "external_candidate_run".into(),
        scheduler_admission: false,
        retriever: Retriever {
            kind: "bm25".into(), name: "BM25".into(), version: "v0".into(),
            source_repository_commit: "1".repeat(40),
            implementation_sha256: sha256(b"implementation"),
            configuration_sha256: sha256(b"configuration"),
            checkpoint_sha256: None,
            query_formulator_sha256: sha256(b"query formulator")
        },
        benchmark: Benchmark {
            benchmark_id: "synthetic-benchmark".into(), benchmark_version: "v0".into(),
            split_id: "test".into(), split_role: "test".into(),
            split_manifest_sha256: sha256(b"split"), label_set_sha256: None, labels_sequestered: true
        },
        query: Query {
            query_id: "query-a".into(), query_text_sha256: sha256(b"query"),
            target_statement_sha256: Some(target), target_environment_sha256: env.clone()
        },
        graph: GraphBinding {
            snapshot_id: "graph-a".into(), repository_commit: "2".repeat(40),
            environment_sha256: env, graph_manifest_sha256: graph_sha.clone()
        },
        candidate_pool: CandidatePool {
            candidate_count: 3, requested_top_k: 2, candidate_node_ids: ids,
            node_id_set_sha256: sha256(&set_bytes), candidate_policy_sha256: sha256(b"policy")
        },
        ranking: Ranking {
            score_metric: "dot_product".into(), score_direction: "higher_better".into(),
            calibration: "uncalibrated".into(), candidates
        },
        budget: Budget {
            query_wall_ms: 12, retrieval_cpu_ms: 3, retrieval_gpu_ms: 8, tokens: 12, cost_microusd: 0
        }
    }
}

fn synthetic_mapping(node: &str, declaration: &str) -> Mapping {
    Mapping {
        schema_version: MAPPING_SCHEMA.into(),
        graph_manifest_sha256: sha256(b"graph manifest"),
        node_id: node.into(),
        mapping_status: "exact".into(),
        declaration_id: Some(declaration.into()),
        mapping_method: "exact_formal_name".into(),
        mapping_evidence_sha256: sha256(format!("{node}:{declaration}").as_bytes()),
        reviewer_attestation_sha256: None
    }
}

fn run_self_tests() -> Result<(), String> {
    let support = synthetic_support_report();
    let support_bytes = serde_json::to_vec_pretty(&support).map_err(|e| e.to_string())?;
    let receipt = synthetic_receipt();
    let receipts_bytes = [serde_json::to_vec(&receipt).map_err(|e| e.to_string())?, b"\n".to_vec()].concat();
    let mappings = vec![
        synthetic_mapping("node-h1","h1"),
        synthetic_mapping("node-h2","h2")
    ];
    let mappings_bytes = mappings.iter().flat_map(|m| {
        let mut b=serde_json::to_vec(m).unwrap(); b.push(b'\n'); b
    }).collect::<Vec<_>>();
    let report = analyze(&support_bytes, &receipts_bytes, &mappings_bytes)?;
    if report.matched_assessment_count != 1
        || report.route_assessments[0].any_complete_route_covered_at_returned_k != true
        || report.route_assessments[0].minimal_routes_covered_at_returned_k != 1
        || report.route_assessments[0].best_route_completion_rank != Some(2)
        || report.route_assessments[0].useful_route_premise_rate != Some(1.0)
    {
        return Err("retrieval route coverage positive control failed".into());
    }
    println!("SELF_TEST_PASS a retrieved AND-bundle covers its exact minimal support route");

    let mut only_one = receipt.clone();
    only_one.ranking.candidates.pop();
    let one_bytes = [serde_json::to_vec(&only_one).map_err(|e| e.to_string())?, b"\n".to_vec()].concat();
    let one_mappings = vec![synthetic_mapping("node-h1","h1")];
    let one_mapping_bytes = [serde_json::to_vec(&one_mappings[0]).map_err(|e| e.to_string())?, b"\n".to_vec()].concat();
    let partial = analyze(&support_bytes, &one_bytes, &one_mapping_bytes)?;
    if partial.route_assessments[0].any_complete_route_covered_at_returned_k
        || partial.route_assessments[0].mean_partial_route_fraction <= 0.0
    {
        return Err("partial AND-route must not be reported as complete".into());
    }
    println!("SELF_TEST_PASS partial AND-bundle is not counted as complete route");

    let mut ambiguous = synthetic_mapping("node-h2","h2");
    ambiguous.mapping_status = "ambiguous".into();
    ambiguous.declaration_id = None;
    let amb_bytes = [
        serde_json::to_vec(&mappings[0]).map_err(|e| e.to_string())?, b"\n".to_vec(),
        serde_json::to_vec(&ambiguous).map_err(|e| e.to_string())?, b"\n".to_vec()
    ].concat();
    let no_false = analyze(&support_bytes, &receipts_bytes, &amb_bytes)?;
    if no_false.route_assessments[0].any_complete_route_covered_at_returned_k
        || no_false.unmapped_candidate_count != 1
    {
        return Err("ambiguous candidate mapping was counted as an exact declaration".into());
    }
    println!("SELF_TEST_PASS ambiguous node-to-declaration mapping cannot create route coverage");

    let mut wrong = synthetic_mapping("node-h2","not-in-universe");
    let wrong_bytes = [
        serde_json::to_vec(&mappings[0]).map_err(|e| e.to_string())?, b"\n".to_vec(),
        serde_json::to_vec(&wrong).map_err(|e| e.to_string())?, b"\n".to_vec()
    ].concat();
    let outside = analyze(&support_bytes, &receipts_bytes, &wrong_bytes)?;
    if outside.route_assessments[0].mapped_out_of_universe_count != 1
        || outside.route_assessments[0].any_complete_route_covered_at_returned_k
    {
        return Err("declaration outside mutable support universe was counted".into());
    }
    println!("SELF_TEST_PASS out-of-universe declaration is reported but not counted");

    let mut bad = support.clone();
    bad["groups"][0]["minimal_sufficient_support_antichain"] = json!([["h1"],["h1","h2"]]);
    let bad_support = serde_json::to_vec(&bad).map_err(|e| e.to_string())?;
    if analyze(&bad_support,&receipts_bytes,&mappings_bytes).is_ok() {
        return Err("non-antichain support routes were accepted".into());
    }
    println!("SELF_TEST_PASS non-minimal support route lattice is rejected");

    let mut wrong_target = receipt.clone();
    wrong_target.query.target_statement_sha256 = Some(sha256(b"wrong target"));
    let wrong_bytes = [serde_json::to_vec(&wrong_target).map_err(|e| e.to_string())?,b"\n".to_vec()].concat();
    if analyze(&support_bytes,&wrong_bytes,&mappings_bytes).is_ok() {
        return Err("retrieval receipt matched to different target statement".into());
    }
    println!("SELF_TEST_PASS target statement mismatch is rejected");

    let mut bad_mapping = synthetic_mapping("node-h2","h2");
    bad_mapping.graph_manifest_sha256 = sha256(b"different graph");
    let bad_bytes = [
        serde_json::to_vec(&mappings[0]).map_err(|e| e.to_string())?,b"\n".to_vec(),
        serde_json::to_vec(&bad_mapping).map_err(|e| e.to_string())?,b"\n".to_vec()
    ].concat();
    if analyze(&support_bytes,&receipts_bytes,&bad_bytes).is_ok() {
        return Err("stale graph mapping was accepted".into());
    }
    println!("SELF_TEST_PASS declaration crosswalk binds exact graph manifest digest");
    Ok(())
}

fn synthetic_report() -> Result<Report, String> {
    let support = synthetic_support_report();
    let support_bytes = serde_json::to_vec_pretty(&support).map_err(|e| e.to_string())?;
    let receipt = synthetic_receipt();
    let receipts_bytes = [serde_json::to_vec(&receipt).map_err(|e| e.to_string())?, b"\n".to_vec()].concat();
    let mappings = vec![
        synthetic_mapping("node-h1","h1"),
        synthetic_mapping("node-h2","h2")
    ];
    let mut mapping_bytes = Vec::new();
    for mapping in &mappings {
        mapping_bytes.extend(serde_json::to_vec(mapping).map_err(|e| e.to_string())?);
        mapping_bytes.push(b'\n');
    }
    analyze(&support_bytes, &receipts_bytes, &mapping_bytes)
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_ROUTE_COVERAGE_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let report = synthetic_report()?;
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize synthetic route report: {e}"))?;
        fs::write(&args[2], bytes).map_err(|e| format!("write synthetic route report: {e}"))?;
        println!("PITB_ROUTE_COVERAGE_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() != 6 || args[1] != "--analyze" {
        return Err(format!(
            "usage: {} --analyze <support-antichain-report.json> <retrieval-runs.jsonl> <declaration-mappings.jsonl> <report.json> | --self-test | --self-test-report <report.json>",
            args.first().map(String::as_str).unwrap_or("pitb-route-coverage")
        ));
    }
    let support_bytes = fs::read(&args[2]).map_err(|e| format!("read support antichain report: {e}"))?;
    let retrieval_bytes = fs::read(&args[3]).map_err(|e| format!("read retrieval receipts: {e}"))?;
    let mapping_bytes = fs::read(&args[4]).map_err(|e| format!("read declaration mappings: {e}"))?;
    let report = analyze(&support_bytes,&retrieval_bytes,&mapping_bytes)?;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize route coverage report: {e}"))?;
    fs::write(&args[5],bytes).map_err(|e| format!("write route coverage report: {e}"))?;
    println!("PITB_ROUTE_COVERAGE_ANALYSIS_OK assessments={} retrievers={}", report.matched_assessment_count, report.comparison_summaries.len());
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("INVALID: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_route_coverage_controls_pass() {
        run_self_tests().expect("all route coverage controls must pass");
    }

    #[test]
    fn empty_ranked_receipts_do_not_create_false_coverage() {
        let support = synthetic_support_report();
        let support_bytes = serde_json::to_vec(&support).unwrap();
        let mut receipt = synthetic_receipt();
        receipt.ranking.candidates.clear();
        let receipt_bytes = [serde_json::to_vec(&receipt).unwrap(), b"\n".to_vec()].concat();
        assert!(analyze(&support_bytes,&receipt_bytes,b"{}\n").is_err());
    }
}
