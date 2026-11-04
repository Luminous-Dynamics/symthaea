use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const GRAPH_SCHEMA: &str = "pitb.proof-graph-evidence.v0";
const RUN_SCHEMA: &str = "pitb.retrieval-query-receipt.v0";
const REPORT_SCHEMA: &str = "pitb.retrieval-evidence-report.v0";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    query_text_sha256: String,
    target_statement_sha256: Option<String>,
    target_environment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct GraphBinding {
    snapshot_id: String,
    repository_commit: String,
    environment_sha256: String,
    graph_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CandidatePool {
    candidate_count: usize,
    requested_top_k: usize,
    candidate_node_ids: Vec<String>,
    node_id_set_sha256: String,
    candidate_policy_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Ranking {
    score_metric: String,
    score_direction: String,
    calibration: String,
    candidates: Vec<RankedCandidate>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RankedCandidate {
    rank: usize,
    node_id: String,
    score: Option<f64>,
    upstream_candidate_id: Option<String>,
    source_relation_label: Option<String>,
    evidence_artifact_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Budget {
    query_wall_ms: u64,
    retrieval_cpu_ms: u64,
    retrieval_gpu_ms: u64,
    tokens: u64,
    cost_microusd: u64,
}

#[derive(Debug, Clone, Serialize)]
struct RunSummary {
    query_id: String,
    retriever_kind: String,
    candidate_count: usize,
    requested_top_k: usize,
    returned_candidates: usize,
    score_metric: String,
    graph_manifest_sha256: String,
    query_wall_ms: u64,
    retrieval_cpu_ms: u64,
    retrieval_gpu_ms: u64,
    tokens: u64,
    cost_microusd: u64,
    receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    graph_manifest_sha256: String,
    graph_snapshot_id: String,
    graph_node_count: usize,
    receipt_jsonl_sha256: String,
    receipt_count: usize,
    run_summaries: Vec<RunSummary>,
    scheduler_admission: bool,
    score_interpretation: String,
    trust_boundary: String,
}

fn fail(message: impl AsRef<str>) -> ExitCode {
    eprintln!("INVALID: {}", message.as_ref());
    ExitCode::FAILURE
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest_field(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 64) {
        Ok(())
    } else {
        Err(format!("{label} must be 64 lowercase hexadecimal characters"))
    }
}

fn git_commit(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 40) {
        Ok(())
    } else {
        Err(format!("{label} must be a 40-character lowercase Git commit"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} cannot be empty"))
    } else {
        Ok(())
    }
}

fn canonical_node_id_set_digest(ids: &BTreeSet<String>) -> Result<String, String> {
    let bytes = serde_json::to_vec(ids).map_err(|e| format!("encode candidate node ID set: {e}"))?;
    Ok(sha256(&bytes))
}

fn load_graph(bytes: &[u8]) -> Result<(String, String, String, BTreeSet<String>), String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("graph manifest JSON invalid: {e}"))?;
    if value.get("schema_version").and_then(Value::as_str) != Some(GRAPH_SCHEMA) {
        return Err("unsupported PITB graph manifest schema".into());
    }
    let evidence_class = value.get("evidence_class").and_then(Value::as_str)
        .ok_or("graph manifest evidence_class missing")?;
    if !matches!(evidence_class, "synthetic_contract_fixture" | "measured_snapshot" | "external_candidate_snapshot") {
        return Err("graph manifest evidence_class is unsupported".into());
    }
    let snapshot = value.get("snapshot").ok_or("graph manifest snapshot missing")?;
    let snapshot_id = snapshot.get("snapshot_id").and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty()).ok_or("graph snapshot_id missing")?.to_owned();
    let environment = snapshot.get("environment_sha256").and_then(Value::as_str)
        .ok_or("graph environment_sha256 missing")?.to_owned();
    digest_field(&environment, "graph environment_sha256")?;
    let repository_commit = snapshot.get("repository_commit").and_then(Value::as_str)
        .ok_or("graph repository_commit missing")?.to_owned();
    git_commit(&repository_commit, "graph repository_commit")?;
    let nodes = value.get("nodes").and_then(Value::as_array)
        .ok_or("graph nodes must be an array")?;
    let mut ids = BTreeSet::new();
    for (index, node) in nodes.iter().enumerate() {
        let id = node.get("node_id").and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| format!("graph node {index} has no node_id"))?;
        if !ids.insert(id.to_owned()) {
            return Err(format!("duplicate graph node_id {id}"));
        }
    }
    if ids.is_empty() {
        return Err("graph manifest cannot be empty".into());
    }
    Ok((snapshot_id, repository_commit, environment, ids))
}

fn validate_retriever(retriever: &Retriever) -> Result<(), String> {
    if !matches!(
        retriever.kind.as_str(),
        "daps_selector" | "leangraphsearch" | "leansearch_v2" | "theoremgraph" | "bm25" | "pitb_union" | "other"
    ) {
        return Err(format!("unsupported retriever kind {}", retriever.kind));
    }
    non_empty(&retriever.name, "retriever.name")?;
    non_empty(&retriever.version, "retriever.version")?;
    git_commit(&retriever.source_repository_commit, "retriever.source_repository_commit")?;
    digest_field(&retriever.implementation_sha256, "retriever.implementation_sha256")?;
    digest_field(&retriever.configuration_sha256, "retriever.configuration_sha256")?;
    digest_field(&retriever.query_formulator_sha256, "retriever.query_formulator_sha256")?;
    if let Some(checkpoint) = &retriever.checkpoint_sha256 {
        digest_field(checkpoint, "retriever.checkpoint_sha256")?;
    }
    if retriever.kind == "daps_selector" && retriever.checkpoint_sha256.is_none() {
        return Err("DAPS selector receipt requires exact model checkpoint SHA-256".into());
    }
    Ok(())
}

fn validate_receipt(
    receipt: &Receipt,
    index: usize,
    graph_snapshot_id: &str,
    graph_repository_commit: &str,
    graph_environment: &str,
    graph_sha256: &str,
    graph_node_ids: &BTreeSet<String>,
) -> Result<(), String> {
    let label = format!("retrieval receipt {index}");
    if receipt.schema_version != RUN_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    if receipt.evidence_class != "external_candidate_run" {
        return Err(format!("{label} must be labeled external_candidate_run"));
    }
    if receipt.scheduler_admission {
        return Err(format!("{label} cannot claim scheduler admission"));
    }
    validate_retriever(&receipt.retriever)?;
    for (field, value) in [
        ("benchmark_id", receipt.benchmark.benchmark_id.as_str()),
        ("benchmark_version", receipt.benchmark.benchmark_version.as_str()),
        ("split_id", receipt.benchmark.split_id.as_str()),
        ("query_id", receipt.query.query_id.as_str()),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    if !matches!(receipt.benchmark.split_role.as_str(), "train" | "validation" | "test" | "development" | "hidden_test") {
        return Err(format!("{label} has unsupported split_role"));
    }
    digest_field(&receipt.benchmark.split_manifest_sha256, &format!("{label}.split_manifest_sha256"))?;
    if let Some(labels) = &receipt.benchmark.label_set_sha256 {
        digest_field(labels, &format!("{label}.label_set_sha256"))?;
    }
    if !receipt.benchmark.labels_sequestered {
        return Err(format!("{label} must declare labels_sequestered=true"));
    }
    digest_field(&receipt.query.query_text_sha256, &format!("{label}.query_text_sha256"))?;
    if let Some(target) = &receipt.query.target_statement_sha256 {
        digest_field(target, &format!("{label}.target_statement_sha256"))?;
    }
    digest_field(&receipt.query.target_environment_sha256, &format!("{label}.target_environment_sha256"))?;
    if receipt.query.target_environment_sha256 != graph_environment {
        return Err(format!("{label} query environment differs from the graph environment"));
    }

    non_empty(&receipt.graph.snapshot_id, &format!("{label}.graph.snapshot_id"))?;
    if receipt.graph.snapshot_id != graph_snapshot_id
        || receipt.graph.repository_commit != graph_repository_commit
        || receipt.graph.environment_sha256 != graph_environment
        || receipt.graph.graph_manifest_sha256 != graph_sha256
    {
        return Err(format!("{label} graph identity/digest differs from the supplied graph manifest"));
    }

    let candidate_ids: BTreeSet<String> =
        receipt.candidate_pool.candidate_node_ids.iter().cloned().collect();
    if receipt.candidate_pool.candidate_node_ids.len() != candidate_ids.len()
        || receipt.candidate_pool.candidate_node_ids.iter().any(|id| id.trim().is_empty())
        || !receipt.candidate_pool.candidate_node_ids.windows(2).all(|w| w[0] < w[1])
    {
        return Err(format!("{label} candidate_node_ids must be sorted, unique, non-empty IDs"));
    }
    if receipt.candidate_pool.candidate_count != candidate_ids.len() {
        return Err(format!("{label} candidate_count differs from candidate_node_ids length"));
    }
    if candidate_ids.iter().any(|id| !graph_node_ids.contains(id)) {
        return Err(format!("{label} candidate pool contains a node outside the graph snapshot"));
    }
    if receipt.candidate_pool.requested_top_k == 0
        || receipt.candidate_pool.requested_top_k > receipt.candidate_pool.candidate_count
    {
        return Err(format!("{label} requested_top_k is outside the candidate pool"));
    }
    digest_field(&receipt.candidate_pool.node_id_set_sha256, &format!("{label}.candidate_pool.node_id_set_sha256"))?;
    digest_field(&receipt.candidate_pool.candidate_policy_sha256, &format!("{label}.candidate_pool.candidate_policy_sha256"))?;
    if receipt.candidate_pool.node_id_set_sha256 != canonical_node_id_set_digest(&candidate_ids)? {
        return Err(format!("{label} candidate node-set digest differs from declared candidate pool"));
    }

    let (metric, direction) = (receipt.ranking.score_metric.as_str(), receipt.ranking.score_direction.as_str());
    if !matches!(metric, "cosine_similarity" | "dot_product" | "logit" | "distance" | "rank_only") {
        return Err(format!("{label} has unsupported score_metric {metric}"));
    }
    if !matches!(direction, "higher_better" | "lower_better" | "rank_only") {
        return Err(format!("{label} has unsupported score_direction {direction}"));
    }
    if receipt.ranking.calibration != "uncalibrated" {
        return Err(format!("{label} external retrieval scores must be declared uncalibrated"));
    }
    if (metric == "rank_only") != (direction == "rank_only") {
        return Err(format!("{label} rank_only metric and direction must be used together"));
    }
    if receipt.ranking.candidates.len() > receipt.candidate_pool.requested_top_k {
        return Err(format!("{label} returns more candidates than requested_top_k"));
    }
    let mut seen_nodes = BTreeSet::new();
    let mut previous_score: Option<f64> = None;
    for (rank_index, candidate) in receipt.ranking.candidates.iter().enumerate() {
        let expected_rank = rank_index + 1;
        if candidate.rank != expected_rank {
            return Err(format!("{label} ranking must have contiguous 1-based ranks"));
        }
        if !seen_nodes.insert(candidate.node_id.clone()) {
            return Err(format!("{label} contains duplicate candidate node {}", candidate.node_id));
        }
        if !candidate_ids.contains(&candidate.node_id) {
            return Err(format!("{label} references node {} outside the declared candidate pool", candidate.node_id));
        }
        digest_field(&candidate.evidence_artifact_sha256, &format!("{label}.candidate[{expected_rank}].evidence_artifact_sha256"))?;
        if let Some(upstream_id) = &candidate.upstream_candidate_id {
            non_empty(upstream_id, &format!("{label}.candidate[{expected_rank}].upstream_candidate_id"))?;
        }
        match (metric, candidate.score) {
            ("rank_only", None) => {}
            ("rank_only", Some(_)) => return Err(format!("{label} rank_only candidate cannot contain a numeric score")),
            (_, None) => return Err(format!("{label} scored candidate is missing numeric score")),
            (_, Some(score)) if !score.is_finite() => return Err(format!("{label} candidate score is not finite")),
            (_, Some(score)) => {
                if let Some(previous) = previous_score {
                    let ordered = if direction == "higher_better" {
                        previous >= score
                    } else {
                        previous <= score
                    };
                    if !ordered {
                        return Err(format!("{label} scores do not agree with declared ranking direction"));
                    }
                }
                previous_score = Some(score);
            }
        }
    }
    Ok(())
}

fn load_receipts(bytes: &[u8]) -> Result<Vec<Receipt>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("retrieval receipt JSONL is not UTF-8: {e}"))?;
    let mut receipts = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        receipts.push(serde_json::from_str(line)
            .map_err(|e| format!("retrieval receipt JSONL line {}: {e}", index + 1))?);
    }
    if receipts.is_empty() {
        return Err("retrieval receipt JSONL must contain at least one query".into());
    }
    Ok(receipts)
}

fn validate_all(graph_bytes: &[u8], receipt_bytes: &[u8]) -> Result<Report, String> {
    let graph_sha256 = sha256(graph_bytes);
    let (snapshot_id, repository_commit, environment, node_ids) = load_graph(graph_bytes)?;
    let receipts = load_receipts(receipt_bytes)?;
    let mut seen_query_retriever_pairs = BTreeSet::new();
    let mut summaries = Vec::new();
    for (index, receipt) in receipts.iter().enumerate() {
        validate_receipt(
            receipt,
            index + 1,
            &snapshot_id,
            &repository_commit,
            &environment,
            &graph_sha256,
            &node_ids,
        )?;
        let run_identity = format!(
            "{}|{}|{}|{}|{}|{}|{}",
            receipt.benchmark.benchmark_id,
            receipt.benchmark.benchmark_version,
            receipt.benchmark.split_id,
            receipt.retriever.kind,
            receipt.retriever.name,
            receipt.retriever.version,
            receipt.query.query_id
        );
        if !seen_query_retriever_pairs.insert(run_identity) {
            return Err(format!(
                "duplicate query receipt for retriever {} / query {}",
                receipt.retriever.name, receipt.query.query_id
            ));
        }
        summaries.push(RunSummary {
            query_id: receipt.query.query_id.clone(),
            retriever_kind: receipt.retriever.kind.clone(),
            candidate_count: receipt.candidate_pool.candidate_count,
            requested_top_k: receipt.candidate_pool.requested_top_k,
            returned_candidates: receipt.ranking.candidates.len(),
            score_metric: receipt.ranking.score_metric.clone(),
            graph_manifest_sha256: graph_sha256.clone(),
            query_wall_ms: receipt.budget.query_wall_ms,
            retrieval_cpu_ms: receipt.budget.retrieval_cpu_ms,
            retrieval_gpu_ms: receipt.budget.retrieval_gpu_ms,
            tokens: receipt.budget.tokens,
            cost_microusd: receipt.budget.cost_microusd,
            receipt_sha256: sha256(&serde_json::to_vec(receipt).map_err(|e| format!("serialize receipt: {e}"))?),
        });
    }
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        graph_manifest_sha256: graph_sha256,
        graph_snapshot_id: snapshot_id,
        graph_node_count: node_ids.len(),
        receipt_jsonl_sha256: sha256(receipt_bytes),
        receipt_count: summaries.len(),
        run_summaries: summaries,
        scheduler_admission: false,
        score_interpretation: "Scores are retriever-specific uncalibrated ranking values and must not be interpreted as probabilities or comparable across retrieval systems.".into(),
        trust_boundary: "Receipts bind ranked candidate IDs to a graph snapshot, query, split, retriever implementation/checkpoint, candidate policy and measured budget. They do not prove candidate relevance, target sufficiency, logical necessity or permission for scheduler admission. Verify the receipt and upstream artifact bytes before benchmark scoring.".into(),
    })
}

fn synthetic_graph() -> Value {
    let mk_node = |id: &str, name: &str| json!({
        "node_id": id,
        "kind": "formal_declaration",
        "label": name,
        "statement_sha256": sha256(name.as_bytes()),
        "statement_artifact_id": format!("stmt-{id}"),
        "source_locator": format!("Demo/{name}.lean:1-1"),
        "formal_name": name
    });
    json!({
        "schema_version": GRAPH_SCHEMA,
        "evidence_class": "synthetic_contract_fixture",
        "snapshot": {
            "snapshot_id": "synthetic-graph",
            "repository_commit": "1".repeat(40),
            "environment_sha256": sha256(b"lean-env"),
            "extractor_id": "synthetic",
            "extractor_version": "0.0",
            "extractor_config_sha256": sha256(b"extractor-config")
        },
        "nodes": [
            mk_node("node-h1","Demo.h1"),
            mk_node("node-h2","Demo.h2"),
            mk_node("node-d","Demo.distractor")
        ],
        "edges": [],
        "artifacts": [],
        "upstream_provenance": null
    })
}

fn synthetic_receipt(graph: &Value, graph_bytes: &[u8]) -> Receipt {
    let nodes = graph["nodes"].as_array().unwrap();
    let mut ids: Vec<String> = nodes.iter().map(|x| x["node_id"].as_str().unwrap().to_owned()).collect();
    ids.sort();
    let candidate_ids: BTreeSet<String> = [ids[0].clone(), ids[1].clone()].into_iter().collect();
    let mut candidate_iter = candidate_ids.iter();
    let candidate_one = candidate_iter.next().unwrap().clone();
    let candidate_two = candidate_iter.next().unwrap().clone();
    let environment = graph["snapshot"]["environment_sha256"].as_str().unwrap().to_owned();
    let repository_commit = graph["snapshot"]["repository_commit"].as_str().unwrap().to_owned();
    let snapshot_id = graph["snapshot"]["snapshot_id"].as_str().unwrap().to_owned();
    Receipt {
        schema_version: RUN_SCHEMA.into(),
        evidence_class: "external_candidate_run".into(),
        scheduler_admission: false,
        retriever: Retriever {
            kind: "daps_selector".into(),
            name: "DAPS Phase 2".into(),
            version: "pinned-demo".into(),
            source_repository_commit: "2".repeat(40),
            implementation_sha256: sha256(b"daps-eval-script"),
            configuration_sha256: sha256(b"daps-selector-config"),
            checkpoint_sha256: Some(sha256(b"fake-checkpoint-digest-for-contract-only")),
            query_formulator_sha256: sha256(b"exact-formal-query"),
        },
        benchmark: Benchmark {
            benchmark_id: "synthetic-benchmark".into(),
            benchmark_version: "0.1".into(),
            split_id: "test-heldout".into(),
            split_role: "test".into(),
            split_manifest_sha256: sha256(b"split-manifest"),
            label_set_sha256: Some(sha256(b"gold-labels-separate")),
            labels_sequestered: true,
        },
        query: Query {
            query_id: "query-1".into(),
            query_text_sha256: sha256(b"theorem statement"),
            target_statement_sha256: Some(sha256(b"target statement")),
            target_environment_sha256: environment.clone(),
        },
        graph: GraphBinding {
            snapshot_id,
            repository_commit,
            environment_sha256: environment,
            graph_manifest_sha256: sha256(graph_bytes),
        },
        candidate_pool: CandidatePool {
            candidate_count: candidate_ids.len(),
            requested_top_k: 2,
            candidate_node_ids: candidate_ids.iter().cloned().collect(),
            node_id_set_sha256: canonical_node_id_set_digest(&candidate_ids).unwrap(),
            candidate_policy_sha256: sha256(b"all-formal-declarations"),
        },
        ranking: Ranking {
            score_metric: "dot_product".into(),
            score_direction: "higher_better".into(),
            calibration: "uncalibrated".into(),
            candidates: vec![
                RankedCandidate {
                    rank: 1,
                    node_id: candidate_one,
                    score: Some(0.93),
                    upstream_candidate_id: Some("17".into()),
                    source_relation_label: Some("USES_DEFINITION".into()),
                    evidence_artifact_sha256: sha256(b"candidate-1"),
                },
                RankedCandidate {
                    rank: 2,
                    node_id: candidate_two,
                    score: Some(0.87),
                    upstream_candidate_id: Some("21".into()),
                    source_relation_label: Some("USES_CLASS".into()),
                    evidence_artifact_sha256: sha256(b"candidate-2"),
                },
            ],
        },
        budget: Budget {
            query_wall_ms: 14,
            retrieval_cpu_ms: 3,
            retrieval_gpu_ms: 9,
            tokens: 80,
            cost_microusd: 0,
        },
    }
}

fn run_self_tests() -> Result<(), String> {
    let graph = synthetic_graph();
    let graph_bytes = serde_json::to_vec(&graph).map_err(|e| e.to_string())?;
    let receipt = synthetic_receipt(&graph, &graph_bytes);
    validate_receipt(
        &receipt, 1,
        graph["snapshot"]["snapshot_id"].as_str().unwrap(),
        graph["snapshot"]["repository_commit"].as_str().unwrap(),
        graph["snapshot"]["environment_sha256"].as_str().unwrap(),
        &sha256(&graph_bytes),
        &graph["nodes"].as_array().unwrap().iter().map(|x| x["node_id"].as_str().unwrap().to_owned()).collect(),
    )?;
    println!("SELF_TEST_PASS exact query, graph, split, retriever and budget identity");

    let mut bad = receipt.clone();
    bad.scheduler_admission = true;
    if validate_receipt(&bad, 1, "synthetic-graph", bad.graph.repository_commit.as_str(), bad.graph.environment_sha256.as_str(), &bad.graph.graph_manifest_sha256, &graph["nodes"].as_array().unwrap().iter().map(|x| x["node_id"].as_str().unwrap().to_owned()).collect()).is_ok() {
        return Err("scheduler-admission promotion was accepted".into());
    }
    println!("SELF_TEST_PASS candidate receipts cannot claim scheduler admission");

    let node_ids: BTreeSet<String> = graph["nodes"].as_array().unwrap().iter().map(|x| x["node_id"].as_str().unwrap().to_owned()).collect();
    let mut bad = receipt.clone();
    bad.ranking.candidates[0].node_id = "not-in-graph".into();
    if validate_receipt(&bad,1,"synthetic-graph",&bad.graph.repository_commit,&bad.graph.environment_sha256,&bad.graph.graph_manifest_sha256,&node_ids).is_ok() {
        return Err("candidate outside graph snapshot was accepted".into());
    }
    println!("SELF_TEST_PASS candidate node must belong to exact graph snapshot");

    let mut bad = receipt.clone();
    bad.ranking.candidates[0].rank = 2;
    if validate_receipt(&bad,1,"synthetic-graph",&bad.graph.repository_commit,&bad.graph.environment_sha256,&bad.graph.graph_manifest_sha256,&node_ids).is_ok() {
        return Err("non-contiguous ranking was accepted".into());
    }
    println!("SELF_TEST_PASS rank positions are contiguous and 1-based");

    let mut bad = receipt.clone();
    bad.ranking.candidates[1].score = Some(0.99);
    if validate_receipt(&bad,1,"synthetic-graph",&bad.graph.repository_commit,&bad.graph.environment_sha256,&bad.graph.graph_manifest_sha256,&node_ids).is_ok() {
        return Err("score order contradicting declared direction was accepted".into());
    }
    println!("SELF_TEST_PASS numeric scores follow declared ranking direction");

    let mut bad = receipt.clone();
    bad.ranking.calibration = "calibrated_probability".into();
    if validate_receipt(&bad,1,"synthetic-graph",&bad.graph.repository_commit,&bad.graph.environment_sha256,&bad.graph.graph_manifest_sha256,&node_ids).is_ok() {
        return Err("retrieval score was promoted to calibrated probability".into());
    }
    println!("SELF_TEST_PASS raw retrieval scores remain uncalibrated");

    let mut bad = receipt.clone();
    bad.retriever.checkpoint_sha256 = None;
    if validate_receipt(&bad,1,"synthetic-graph",&bad.graph.repository_commit,&bad.graph.environment_sha256,&bad.graph.graph_manifest_sha256,&node_ids).is_ok() {
        return Err("DAPS selector without checkpoint identity was accepted".into());
    }
    println!("SELF_TEST_PASS learned DAPS selector requires checkpoint digest");

    let mut bad = receipt.clone();
    bad.benchmark.labels_sequestered = false;
    if validate_receipt(&bad,1,"synthetic-graph",&bad.graph.repository_commit,&bad.graph.environment_sha256,&bad.graph.graph_manifest_sha256,&node_ids).is_ok() {
        return Err("non-sequestered labels were accepted".into());
    }
    println!("SELF_TEST_PASS evaluation labels must remain sequestered");

    let graph_bytes = serde_json::to_vec(&graph).map_err(|e| e.to_string())?;
    let base = synthetic_receipt(&graph, &graph_bytes);
    let second = {
        let mut receipt = base.clone();
        receipt.retriever.kind = "bm25".into();
        receipt.retriever.name = "BM25".into();
        receipt.retriever.version = "v1".into();
        receipt.retriever.checkpoint_sha256 = None;
        receipt
    };
    let receipt_bytes = [
        serde_json::to_vec(&base).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
        serde_json::to_vec(&second).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
    ].concat();
    if validate_all(&graph_bytes, &receipt_bytes).is_err() {
        return Err("matched retrieval systems could not score the same query identity".into());
    }
    println!("SELF_TEST_PASS matched retrievers can emit receipts for the same benchmark query");

    let duplicate_bytes = [
        serde_json::to_vec(&base).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
        serde_json::to_vec(&base).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
    ].concat();
    if validate_all(&graph_bytes, &duplicate_bytes).is_ok() {
        return Err("duplicate query receipt for the same retriever was accepted".into());
    }
    println!("SELF_TEST_PASS duplicate query receipt for one retriever is rejected");

    Ok(())
}

fn write_synthetic_run(
    graph_path: &PathBuf,
    receipt_path: &PathBuf,
    report_path: &PathBuf,
) -> Result<(), String> {
    let graph_bytes = fs::read(graph_path).map_err(|e| format!("read synthetic graph: {e}"))?;
    let graph: Value = serde_json::from_slice(&graph_bytes)
        .map_err(|e| format!("parse synthetic graph: {e}"))?;
    let (snapshot_id, repository_commit, environment, node_ids) = load_graph(&graph_bytes)?;
    let nodes = graph["nodes"].as_array().ok_or("synthetic graph nodes missing")?;
    if nodes.len() < 2 {
        return Err("synthetic retrieval graph must contain at least two nodes".into());
    }
    let candidate_node_ids: BTreeSet<String> = nodes.iter().take(2)
        .map(|node| node["node_id"].as_str().ok_or("synthetic node_id missing").map(str::to_owned))
        .collect::<Result<_, _>>()?;
    let mut receipt = synthetic_receipt(&graph, &graph_bytes);
    receipt.graph.snapshot_id = snapshot_id.clone();
    receipt.graph.repository_commit = repository_commit.clone();
    receipt.graph.environment_sha256 = environment.clone();
    receipt.graph.graph_manifest_sha256 = sha256(&graph_bytes);
    receipt.query.target_environment_sha256 = environment.clone();
    receipt.candidate_pool.candidate_count = candidate_node_ids.len();
    receipt.candidate_pool.requested_top_k = candidate_node_ids.len();
    receipt.candidate_pool.candidate_node_ids = candidate_node_ids.iter().cloned().collect();
    receipt.candidate_pool.node_id_set_sha256 = canonical_node_id_set_digest(&candidate_node_ids)?;
    receipt.ranking.candidates = receipt.ranking.candidates.into_iter().take(candidate_node_ids.len()).collect();
    for (rank, candidate) in receipt.ranking.candidates.iter_mut().enumerate() {
        candidate.rank = rank + 1;
        candidate.node_id = receipt.candidate_pool.candidate_node_ids[rank].clone();
    }
    validate_receipt(
        &receipt,
        1,
        &snapshot_id,
        &repository_commit,
        &environment,
        &sha256(&graph_bytes),
        &node_ids,
    )?;
    let receipt_line = serde_json::to_vec(&receipt).map_err(|e| format!("serialize synthetic retrieval receipt: {e}"))?;
    let receipt_jsonl = [receipt_line, b"\n".to_vec()].concat();
    let report = validate_all(&graph_bytes, &receipt_jsonl)?;
    fs::write(receipt_path, receipt_jsonl).map_err(|e| format!("write synthetic receipt JSONL: {e}"))?;
    let report_bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize synthetic retrieval report: {e}"))?;
    fs::write(report_path, report_bytes).map_err(|e| format!("write synthetic retrieval report: {e}"))?;
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_RETRIEVAL_EVIDENCE_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 5 && args[1] == "--self-test-report" {
        run_self_tests()?;
        write_synthetic_run(
            &PathBuf::from(&args[2]),
            &PathBuf::from(&args[3]),
            &PathBuf::from(&args[4]),
        )?;
        println!("PITB_RETRIEVAL_EVIDENCE_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() != 5 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <graph.json> <retrieval-runs.jsonl> <report.json> OR --self-test OR --self-test-report <graph.json> <receipt.jsonl> <report.json>",
            args.first().map(String::as_str).unwrap_or("pitb-retrieval-evidence")
        ));
    }
    let graph_bytes = fs::read(&args[2]).map_err(|e| format!("read graph: {e}"))?;
    let receipt_bytes = fs::read(&args[3]).map_err(|e| format!("read retrieval receipts: {e}"))?;
    let report = validate_all(&graph_bytes, &receipt_bytes)?;
    let output = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize retrieval report: {e}"))?;
    fs::write(&args[4], output).map_err(|e| format!("write retrieval report: {e}"))?;
    println!("PITB_RETRIEVAL_EVIDENCE_OK");
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
    fn registered_retrieval_receipt_controls_pass() {
        run_self_tests().expect("all retrieval receipt controls must pass");
    }

    #[test]
    fn exact_graph_manifest_and_query_environment_are_bound() {
        let graph = synthetic_graph();
        let bytes = serde_json::to_vec(&graph).unwrap();
        let receipt = synthetic_receipt(&graph, &bytes);
        let (snapshot, commit, environment, nodes) = load_graph(&bytes).unwrap();
        assert!(validate_receipt(&receipt,1,&snapshot,&commit,&environment,&sha256(&bytes),&nodes).is_ok());
        let wrong_environment = sha256(b"wrong environment");
        let mut changed = receipt;
        changed.query.target_environment_sha256 = wrong_environment;
        assert!(validate_receipt(&changed,1,&snapshot,&commit,&environment,&sha256(&bytes),&nodes).is_err());
    }
}
