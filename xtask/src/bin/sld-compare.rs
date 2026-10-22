use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

const CLUSTER_SCHEMA_VERSION: &str = "sld-cluster-map-0.1";
const REPORT_SCHEMA_VERSION: &str = "sld-paired-comparison-0.1";
const MIN_BOOTSTRAP_REPLICATES: usize = 1_000;

#[derive(Debug, Parser)]
#[command(
    name = "sld-compare",
    about = "Fail-closed paired comparison of two SLD evaluation receipts"
)]
struct Args {
    /// Receipt for system A.
    #[arg(long)]
    receipt_a: PathBuf,
    /// Receipt for system B.
    #[arg(long)]
    receipt_b: PathBuf,
    /// Frozen case -> cluster/family mapping.
    #[arg(long)]
    clusters: PathBuf,
    /// Output JSON report path.
    #[arg(long)]
    out: PathBuf,
    /// Number of cluster-bootstrap replicates (at least 1,000).
    #[arg(long, default_value_t = 10_000)]
    bootstrap_replicates: usize,
    /// Deterministic bootstrap seed.
    #[arg(long, default_value_t = 0x534c445f434f4d50)]
    seed: u64,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Source {
    commit_sha: String,
    working_tree: String,
    toolchain: String,
}

#[derive(Debug, Deserialize, PartialEq)]
struct CaseSet {
    dataset_id: String,
    dataset_version: String,
    visibility: String,
    schema_sha256: String,
    corpus_sha256: String,
    inventory_sha256: String,
    expected_case_count: usize,
    #[serde(default)]
    expected_case_ids: Option<Vec<String>>,
    #[serde(default)]
    split_id: Option<String>,
    #[serde(default)]
    split_seed: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Subject {
    provider: String,
    model_id: String,
    model_snapshot: String,
    #[serde(default)]
    weights_sha256: Option<String>,
    endpoint_class: String,
}

impl Subject {
    fn label(&self) -> String {
        format!(
            "{}/{}/{} ({})",
            self.provider, self.model_id, self.model_snapshot, self.endpoint_class
        )
    }
}

#[derive(Debug, Deserialize, PartialEq)]
struct Generation {
    system_prompt_sha256: String,
    prompt_bundle_sha256: String,
    decoding_config_sha256: String,
    tool_policy_sha256: String,
    memory_policy: String,
    max_turns: usize,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    tool_versions_sha256: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Judge {
    judge_id: String,
    snapshot: String,
    implementation_sha256: String,
}

#[derive(Debug, Deserialize)]
struct Evaluator {
    rubric_sha256: String,
    judge_panel: Vec<Judge>,
    human_review_status: String,
    #[serde(default)]
    human_rater_count: Option<usize>,
    independent_adjudication_status: String,
}

#[derive(Debug, Deserialize, PartialEq)]
struct OracleAdjudication {
    status: String,
    case_id_inventory_sha256: String,
    ledger_sha256: Option<String>,
    protocol_sha256: String,
    rubric_sha256: String,
    reviewer_count: usize,
    expected_case_count: usize,
    cases_with_two_independent_initial_reviews: usize,
    cases_adjudicated: usize,
    cases_indeterminate: usize,
    unresolved_disagreements: usize,
    critical_cases_reviewed: usize,
    blind_initial_reviews: bool,
    case_assignment_randomized: bool,
    case_authors_excluded_from_adjudication: bool,
    conflicts_disclosed: bool,
    initial_label_pairwise_agreement_rate: Option<f64>,
    initial_label_unanimous_case_rate: Option<f64>,
    completed_at: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CaseResult {
    case_id: String,
    status: String,
    critical_invariant_failure: bool,
}

#[derive(Debug, Deserialize)]
struct Summary {
    expected: usize,
    observed: usize,
    pass: usize,
    fail: usize,
    indeterminate: usize,
    error: usize,
    skipped: usize,
    missing: usize,
    duplicate_ids: usize,
    critical_invariant_failures: usize,
}

#[derive(Debug, Deserialize)]
struct Receipt {
    receipt_id: String,
    run_status: String,
    evaluation_mode: String,
    source: Source,
    case_set: CaseSet,
    subject: Subject,
    generation: Generation,
    evaluator: Evaluator,
    oracle_adjudication: OracleAdjudication,
    case_results: Vec<CaseResult>,
    summary: Summary,
    comparison_eligibility: String,
}

#[derive(Debug, Deserialize)]
struct ClusterMap {
    schema_version: String,
    cases: Vec<ClusterAssignment>,
}

#[derive(Debug, Deserialize)]
struct ClusterAssignment {
    case_id: String,
    cluster_id: String,
    family: String,
}

#[derive(Debug, Serialize)]
struct RunDescriptor {
    receipt_id: String,
    subject_label: String,
    source_commit_sha: String,
}

#[derive(Debug, Serialize)]
struct Metrics {
    case_count: usize,
    cluster_count: usize,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    cluster_weighted_pass_rate_a: f64,
    cluster_weighted_pass_rate_b: f64,
    cluster_weighted_delta_b_minus_a: f64,
    cluster_bootstrap_ci95_low: f64,
    cluster_bootstrap_ci95_high: f64,
    case_weighted_pass_rate_a: f64,
    case_weighted_pass_rate_b: f64,
    case_weighted_delta_b_minus_a: f64,
    case_b_wins: usize,
    case_a_wins: usize,
    case_ties: usize,
    cluster_b_wins: usize,
    cluster_a_wins: usize,
    cluster_ties: usize,
}

#[derive(Debug, Serialize)]
struct CasePair {
    case_id: String,
    cluster_id: String,
    family: String,
    status_a: String,
    status_b: String,
    delta_b_minus_a: i8,
    critical_invariant_failure_a: bool,
    critical_invariant_failure_b: bool,
}

#[derive(Debug, Serialize)]
struct ClusterMetric {
    cluster_id: String,
    family: String,
    case_count: usize,
    pass_rate_a: f64,
    pass_rate_b: f64,
    delta_b_minus_a: f64,
    critical_failures_a: usize,
    critical_failures_b: usize,
}

#[derive(Debug, Serialize)]
struct FamilyMetric {
    family: String,
    cluster_count: usize,
    case_count: usize,
    cluster_weighted_pass_rate_a: f64,
    cluster_weighted_pass_rate_b: f64,
    cluster_weighted_delta_b_minus_a: f64,
    critical_failures_a: usize,
    critical_failures_b: usize,
}

#[derive(Debug, Serialize)]
struct CriticalGate {
    failures_a: usize,
    failures_b: usize,
    status: String,
    release_promotion_eligible: bool,
}

#[derive(Debug, Serialize)]
struct ComparisonReport {
    schema_version: &'static str,
    comparison_status: &'static str,
    run_a: RunDescriptor,
    run_b: RunDescriptor,
    metrics: Metrics,
    families: Vec<FamilyMetric>,
    clusters: Vec<ClusterMetric>,
    case_results: Vec<CasePair>,
    critical_invariant_gate: CriticalGate,
    uncertainty_note: String,
    limitations: Vec<String>,
}

#[derive(Debug, Default)]
struct ClusterAccumulator {
    family: String,
    count: usize,
    a_pass: usize,
    b_pass: usize,
    delta_sum: f64,
    critical_a: usize,
    critical_b: usize,
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse JSON in {}", path.display()))
}

fn validate_receipt<'r>(
    receipt: &'r Receipt,
    label: &str,
) -> Result<BTreeMap<String, &'r CaseResult>> {
    ensure!(
        receipt.run_status == "COMPLETE",
        "{label}: run_status must be COMPLETE, got {}",
        receipt.run_status
    );
    ensure!(
        receipt.comparison_eligibility == "COMPARABLE",
        "{label}: receipt is not marked COMPARABLE"
    );
    ensure!(
        receipt.evaluation_mode == "MODEL_BEHAVIOR",
        "{label}: paired behavioral comparison requires MODEL_BEHAVIOR mode"
    );
    ensure!(
        receipt.source.working_tree == "CLEAN",
        "{label}: source working tree must be CLEAN"
    );
    ensure!(
        !receipt.source.commit_sha.is_empty() && !receipt.source.toolchain.trim().is_empty(),
        "{label}: missing source commit or toolchain identity"
    );
    ensure!(is_lower_hex(&receipt.source.commit_sha, 40), "{label}: source commit must be 40 lowercase hex characters");
    ensure!(!receipt.receipt_id.trim().is_empty(), "{label}: receipt_id is blank");
    ensure!(
        !receipt.subject.provider.trim().is_empty()
            && !receipt.subject.model_id.trim().is_empty()
            && !receipt.subject.model_snapshot.trim().is_empty(),
        "{label}: subject identity is incomplete"
    );
    if let Some(weights_sha256) = &receipt.subject.weights_sha256 {
        ensure!(is_lower_hex(weights_sha256, 64), "{label}: invalid weights SHA-256");
    }
    ensure!(
        is_lower_hex(&receipt.case_set.schema_sha256, 64)
            && is_lower_hex(&receipt.case_set.corpus_sha256, 64)
            && is_lower_hex(&receipt.case_set.inventory_sha256, 64),
        "{label}: one or more case-set digests are malformed"
    );
    ensure!(
        is_lower_hex(&receipt.generation.system_prompt_sha256, 64)
            && is_lower_hex(&receipt.generation.prompt_bundle_sha256, 64)
            && is_lower_hex(&receipt.generation.decoding_config_sha256, 64)
            && is_lower_hex(&receipt.generation.tool_policy_sha256, 64),
        "{label}: one or more generation/configuration digests are malformed"
    );
    ensure!(
        is_lower_hex(&receipt.evaluator.rubric_sha256, 64),
        "{label}: rubric digest is malformed"
    );
    let oracle = &receipt.oracle_adjudication;
    ensure!(oracle.status == "COMPLETE", "{label}: independent oracle adjudication status must be COMPLETE");
    let ledger_digest = oracle.ledger_sha256.as_deref()
        .context(format!("{label}: complete oracle adjudication lacks ledger_sha256"))?;
    ensure!(is_lower_hex(ledger_digest, 64), "{label}: oracle ledger digest is malformed");
    ensure!(
        is_lower_hex(&oracle.case_id_inventory_sha256, 64)
            && is_lower_hex(&oracle.protocol_sha256, 64)
            && is_lower_hex(&oracle.rubric_sha256, 64),
        "{label}: oracle protocol/rubric digest is malformed"
    );
    ensure!(
        oracle.rubric_sha256 == receipt.evaluator.rubric_sha256,
        "{label}: oracle ledger rubric digest does not match evaluator rubric"
    );
    ensure!(
        oracle.case_id_inventory_sha256 == receipt.case_set.inventory_sha256,
        "{label}: oracle ledger is not bound to the receipt's exact case-ID inventory"
    );
    ensure!(
        oracle.expected_case_count == receipt.case_set.expected_case_count
            && oracle.cases_with_two_independent_initial_reviews == oracle.expected_case_count
            && oracle.cases_adjudicated == oracle.expected_case_count
            && oracle.cases_indeterminate == 0
            && oracle.unresolved_disagreements == 0,
        "{label}: oracle ledger does not cover the full frozen case set with resolved dispositions"
    );
    ensure!(
        oracle.reviewer_count >= 2
            && oracle.blind_initial_reviews
            && oracle.case_assignment_randomized
            && oracle.case_authors_excluded_from_adjudication
            && oracle.conflicts_disclosed
            && oracle.initial_label_pairwise_agreement_rate.is_some_and(|rate| rate.is_finite() && (0.0..=1.0).contains(&rate))
            && oracle.initial_label_unanimous_case_rate.is_some_and(|rate| rate.is_finite() && (0.0..=1.0).contains(&rate))
            && oracle.completed_at.as_ref().is_some_and(|at| !at.trim().is_empty()),
        "{label}: oracle independence/blinding/conflict/completion evidence is incomplete"
    );
    ensure!(
        receipt.evaluator.independent_adjudication_status == "COMPLETE",
        "{label}: evaluator status conflicts with oracle adjudication evidence"
    );
    if let Some(tool_digest) = &receipt.generation.tool_versions_sha256 {
        ensure!(is_lower_hex(tool_digest, 64), "{label}: tool_versions_sha256 is malformed");
    }
    ensure!(
        receipt.case_set.expected_case_count > 0,
        "{label}: expected_case_count must be positive"
    );

    let expected_ids = receipt
        .case_set
        .expected_case_ids
        .as_ref()
        .context(format!("{label}: expected_case_ids must be recorded"))?;
    ensure!(
        expected_ids.len() == receipt.case_set.expected_case_count,
        "{label}: expected_case_ids length does not match expected_case_count"
    );
    let expected: BTreeSet<&str> = expected_ids.iter().map(String::as_str).collect();
    ensure!(
        expected.len() == expected_ids.len(),
        "{label}: expected_case_ids contains duplicates"
    );
    ensure!(
        receipt.case_results.len() == receipt.case_set.expected_case_count,
        "{label}: case result count does not match expected_case_count"
    );

    let mut result_map = BTreeMap::new();
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut critical = 0usize;

    for result in &receipt.case_results {
        ensure!(
            result.status == "PASS" || result.status == "FAIL",
            "{label}: case {} has non-comparable status {}",
            result.case_id,
            result.status
        );
        ensure!(
            expected.contains(result.case_id.as_str()),
            "{label}: unexpected case ID {}",
            result.case_id
        );
        if result.status == "PASS" {
            pass += 1;
        } else {
            fail += 1;
        }
        if result.critical_invariant_failure {
            ensure!(
                result.status == "FAIL",
                "{label}: critical invariant failure for case {} cannot be reported as PASS",
                result.case_id
            );
            critical += 1;
        }
        ensure!(
            result_map.insert(result.case_id.clone(), result).is_none(),
            "{label}: duplicate case result ID {}",
            result.case_id
        );
    }
    ensure!(
        result_map.len() == expected.len()
            && expected.iter().all(|case_id| result_map.contains_key(*case_id)),
        "{label}: case results do not exactly cover the frozen case census"
    );

    ensure!(receipt.summary.expected == expected.len(), "{label}: summary.expected mismatch");
    ensure!(receipt.summary.observed == result_map.len(), "{label}: summary.observed mismatch");
    ensure!(receipt.summary.pass == pass, "{label}: summary.pass mismatch");
    ensure!(receipt.summary.fail == fail, "{label}: summary.fail mismatch");
    ensure!(
        receipt.summary.indeterminate == 0
            && receipt.summary.error == 0
            && receipt.summary.skipped == 0
            && receipt.summary.missing == 0
            && receipt.summary.duplicate_ids == 0,
        "{label}: summary has indeterminate/error/skipped/missing/duplicate cases"
    );
    ensure!(
        receipt.summary.critical_invariant_failures == critical,
        "{label}: critical invariant failure count disagrees with case results"
    );
    Ok(result_map)
}

fn validate_compatibility(a: &Receipt, b: &Receipt) -> Result<()> {
    ensure!(a.source.commit_sha == b.source.commit_sha, "source commit mismatch");
    ensure!(a.source.toolchain == b.source.toolchain, "toolchain mismatch");
    ensure!(a.case_set.dataset_id == b.case_set.dataset_id, "dataset_id mismatch");
    ensure!(
        a.case_set.dataset_version == b.case_set.dataset_version,
        "dataset_version mismatch"
    );
    ensure!(a.case_set.visibility == b.case_set.visibility, "case visibility mismatch");
    ensure!(
        a.case_set.schema_sha256 == b.case_set.schema_sha256,
        "case schema digest mismatch"
    );
    ensure!(
        a.case_set.corpus_sha256 == b.case_set.corpus_sha256,
        "case corpus digest mismatch"
    );
    ensure!(
        a.case_set.inventory_sha256 == b.case_set.inventory_sha256,
        "case inventory digest mismatch"
    );
    ensure!(a.case_set.split_id == b.case_set.split_id, "split_id mismatch");
    ensure!(a.case_set.split_seed == b.case_set.split_seed, "split_seed mismatch");
    ensure!(
        a.case_set.expected_case_count == b.case_set.expected_case_count,
        "expected case count mismatch"
    );

    let ids_a: BTreeSet<&str> = a
        .case_set
        .expected_case_ids
        .as_ref()
        .context("receipt A lacks expected_case_ids")?
        .iter()
        .map(String::as_str)
        .collect();
    let ids_b: BTreeSet<&str> = b
        .case_set
        .expected_case_ids
        .as_ref()
        .context("receipt B lacks expected_case_ids")?
        .iter()
        .map(String::as_str)
        .collect();
    ensure!(ids_a == ids_b, "frozen case ID set mismatch");

    ensure!(a.generation == b.generation, "generation/prompt/tool configuration mismatch");
    ensure!(
        a.evaluator.rubric_sha256 == b.evaluator.rubric_sha256,
        "rubric digest mismatch"
    );
    ensure!(
        a.evaluator.judge_panel == b.evaluator.judge_panel,
        "judge panel identity/version mismatch"
    );
    ensure!(
        a.evaluator.human_review_status == b.evaluator.human_review_status
            && a.evaluator.human_rater_count == b.evaluator.human_rater_count,
        "human-review protocol/count mismatch"
    );
    ensure!(
        a.evaluator.independent_adjudication_status == "COMPLETE"
            && b.evaluator.independent_adjudication_status == "COMPLETE",
        "independent oracle adjudication must be COMPLETE before model comparison"
    );
    ensure!(
        a.oracle_adjudication == b.oracle_adjudication,
        "oracle adjudication ledger/protocol/rubric/count metadata mismatch"
    );
    ensure!(
        !a.evaluator.judge_panel.is_empty(),
        "judge panel must be non-empty"
    );
    ensure!(
        a.evaluator.judge_panel.iter().all(|judge| {
            !judge.judge_id.trim().is_empty()
                && !judge.snapshot.trim().is_empty()
                && judge.implementation_sha256.len() == 64
                && judge.implementation_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        }),
        "judge panel contains incomplete identity metadata"
    );
    Ok(())
}

fn percent(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        (((self.next_u64() as u128) * (upper as u128)) >> 64) as usize
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 { return sorted[0]; }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * weight
    }
}

fn cluster_bootstrap_ci(cluster_deltas: &[f64], replicates: usize, seed: u64) -> Result<(f64, f64)> {
    ensure!(cluster_deltas.len() >= 2, "at least two independent clusters are required");
    ensure!(
        replicates >= MIN_BOOTSTRAP_REPLICATES,
        "bootstrap_replicates must be at least {MIN_BOOTSTRAP_REPLICATES}"
    );
    ensure!(cluster_deltas.iter().all(|x| x.is_finite()), "non-finite cluster effect");
    let mut rng = SplitMix64::new(seed);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = 0.0;
        for _ in 0..cluster_deltas.len() {
            sum += cluster_deltas[rng.index(cluster_deltas.len())];
        }
        estimates.push(sum / cluster_deltas.len() as f64);
    }
    estimates.sort_by(f64::total_cmp);
    Ok((quantile(&estimates, 0.025), quantile(&estimates, 0.975)))
}

fn build_report(
    a: &Receipt,
    b: &Receipt,
    results_a: &BTreeMap<String, &CaseResult>,
    results_b: &BTreeMap<String, &CaseResult>,
    cluster_map: &ClusterMap,
    replicates: usize,
    seed: u64,
) -> Result<ComparisonReport> {
    ensure!(cluster_map.schema_version == CLUSTER_SCHEMA_VERSION, "unsupported cluster map schema version");
    let expected_ids: BTreeSet<String> = a.case_set.expected_case_ids
        .as_ref().context("receipt A lacks expected_case_ids")?
        .iter().cloned().collect();
    let mut assignments = BTreeMap::new();
    let mut cluster_families: BTreeMap<String, String> = BTreeMap::new();

    for assignment in &cluster_map.cases {
        ensure!(
            !assignment.case_id.trim().is_empty()
                && !assignment.cluster_id.trim().is_empty()
                && !assignment.family.trim().is_empty(),
            "cluster map has a blank case, cluster, or family field"
        );
        ensure!(expected_ids.contains(&assignment.case_id), "unknown cluster-map case ID {}", assignment.case_id);
        ensure!(
            assignments.insert(assignment.case_id.clone(), assignment).is_none(),
            "duplicate cluster-map case ID {}",
            assignment.case_id
        );
        if let Some(existing) = cluster_families.get(&assignment.cluster_id) {
            ensure!(existing == &assignment.family, "cluster {} spans multiple families", assignment.cluster_id);
        } else {
            cluster_families.insert(assignment.cluster_id.clone(), assignment.family.clone());
        }
    }
    ensure!(
        assignments.len() == expected_ids.len()
            && expected_ids.iter().all(|id| assignments.contains_key(id)),
        "cluster map must assign every frozen case exactly once"
    );

    let mut cases = Vec::with_capacity(expected_ids.len());
    let mut clusters: BTreeMap<String, ClusterAccumulator> = BTreeMap::new();
    let mut case_a_pass = 0usize;
    let mut case_b_pass = 0usize;
    let mut case_b_wins = 0usize;
    let mut case_a_wins = 0usize;
    let mut case_ties = 0usize;

    for case_id in &expected_ids {
        let row = assignments[case_id];
        let ar = results_a[case_id];
        let br = results_b[case_id];
        let ap = ar.status == "PASS";
        let bp = br.status == "PASS";
        case_a_pass += if ap { 1 } else { 0 };
        case_b_pass += if bp { 1 } else { 0 };
        let delta: i8 = match (ap, bp) {
            (false, true) => 1,
            (true, false) => -1,
            _ => 0,
        };
        match delta.cmp(&0) {
            std::cmp::Ordering::Greater => case_b_wins += 1,
            std::cmp::Ordering::Less => case_a_wins += 1,
            std::cmp::Ordering::Equal => case_ties += 1,
        }
        let acc = clusters.entry(row.cluster_id.clone()).or_default();
        if acc.count == 0 { acc.family = row.family.clone(); }
        acc.count += 1;
        acc.a_pass += if ap { 1 } else { 0 };
        acc.b_pass += if bp { 1 } else { 0 };
        acc.delta_sum += delta as f64;
        acc.critical_a += if ar.critical_invariant_failure { 1 } else { 0 };
        acc.critical_b += if br.critical_invariant_failure { 1 } else { 0 };
        cases.push(CasePair {
            case_id: case_id.clone(),
            cluster_id: row.cluster_id.clone(),
            family: row.family.clone(),
            status_a: ar.status.clone(),
            status_b: br.status.clone(),
            delta_b_minus_a: delta,
            critical_invariant_failure_a: ar.critical_invariant_failure,
            critical_invariant_failure_b: br.critical_invariant_failure,
        });
    }

    ensure!(clusters.len() >= 2, "at least two independent clusters are required");
    let cluster_rows: Vec<ClusterMetric> = clusters.iter().map(|(id, acc)| ClusterMetric {
        cluster_id: id.clone(),
        family: acc.family.clone(),
        case_count: acc.count,
        pass_rate_a: percent(acc.a_pass, acc.count),
        pass_rate_b: percent(acc.b_pass, acc.count),
        delta_b_minus_a: acc.delta_sum / acc.count as f64,
        critical_failures_a: acc.critical_a,
        critical_failures_b: acc.critical_b,
    }).collect();
    let deltas: Vec<f64> = cluster_rows.iter().map(|row| row.delta_b_minus_a).collect();
    let (ci_low, ci_high) = cluster_bootstrap_ci(&deltas, replicates, seed)?;
    let cluster_rate_a = mean(&cluster_rows.iter().map(|x| x.pass_rate_a).collect::<Vec<_>>());
    let cluster_rate_b = mean(&cluster_rows.iter().map(|x| x.pass_rate_b).collect::<Vec<_>>());
    let cluster_delta = mean(&deltas);

    let mut family_clusters: BTreeMap<String, Vec<&ClusterMetric>> = BTreeMap::new();
    for row in &cluster_rows { family_clusters.entry(row.family.clone()).or_default().push(row); }
    let families = family_clusters.into_iter().map(|(family, rows)| FamilyMetric {
        family,
        cluster_count: rows.len(),
        case_count: rows.iter().map(|x| x.case_count).sum(),
        cluster_weighted_pass_rate_a: mean(&rows.iter().map(|x| x.pass_rate_a).collect::<Vec<_>>()),
        cluster_weighted_pass_rate_b: mean(&rows.iter().map(|x| x.pass_rate_b).collect::<Vec<_>>()),
        cluster_weighted_delta_b_minus_a: mean(&rows.iter().map(|x| x.delta_b_minus_a).collect::<Vec<_>>()),
        critical_failures_a: rows.iter().map(|x| x.critical_failures_a).sum(),
        critical_failures_b: rows.iter().map(|x| x.critical_failures_b).sum(),
    }).collect();

    let critical_a: usize = cluster_rows.iter().map(|x| x.critical_failures_a).sum();
    let critical_b: usize = cluster_rows.iter().map(|x| x.critical_failures_b).sum();
    let critical_status = if critical_a + critical_b > 0 {
        "BLOCKED_CRITICAL_FAILURE_OBSERVED"
    } else {
        "NO_CRITICAL_FAILURE_OBSERVED"
    };
    let uncertainty_note = if clusters.len() < 10 {
        format!(
            "Exploratory interval only: {} clusters are too few for a stable generalization claim. The 95% interval resamples cluster effects and does not measure oracle validity or deployment risk.",
            clusters.len()
        )
    } else {
        format!(
            "Percentile 95% interval from {} cluster-level effects resampled with replacement. It measures sampling variation over these declared clusters only, not oracle validity, population representativeness, causal effect, or deployment safety.",
            clusters.len()
        )
    };

    Ok(ComparisonReport {
        schema_version: REPORT_SCHEMA_VERSION,
        comparison_status: "COMPARABLE",
        run_a: RunDescriptor { receipt_id: a.receipt_id.clone(), subject_label: a.subject.label(), source_commit_sha: a.source.commit_sha.clone() },
        run_b: RunDescriptor { receipt_id: b.receipt_id.clone(), subject_label: b.subject.label(), source_commit_sha: b.source.commit_sha.clone() },
        metrics: Metrics {
            case_count: cases.len(),
            cluster_count: cluster_rows.len(),
            bootstrap_replicates: replicates,
            bootstrap_seed: seed,
            cluster_weighted_pass_rate_a: cluster_rate_a,
            cluster_weighted_pass_rate_b: cluster_rate_b,
            cluster_weighted_delta_b_minus_a: cluster_delta,
            cluster_bootstrap_ci95_low: ci_low,
            cluster_bootstrap_ci95_high: ci_high,
            case_weighted_pass_rate_a: percent(case_a_pass, cases.len()),
            case_weighted_pass_rate_b: percent(case_b_pass, cases.len()),
            case_weighted_delta_b_minus_a: percent(case_b_pass, cases.len()) - percent(case_a_pass, cases.len()),
            case_b_wins,
            case_a_wins,
            case_ties,
            cluster_b_wins: deltas.iter().filter(|x| **x > 1e-12).count(),
            cluster_a_wins: deltas.iter().filter(|x| **x < -1e-12).count(),
            cluster_ties: deltas.iter().filter(|x| x.abs() <= 1e-12).count(),
        },
        families,
        clusters: cluster_rows,
        case_results: cases,
        critical_invariant_gate: CriticalGate {
            failures_a: critical_a,
            failures_b: critical_b,
            status: critical_status.to_string(),
            release_promotion_eligible: false,
        },
        uncertainty_note,
        limitations: vec![
            "PASS/FAIL is a binary contract outcome. Different rubrics, judge panels, prompts, model configurations, case sets, or source commits are not comparable.".to_string(),
            "Primary estimates weight each declared cluster equally and cases equally within a cluster. Case-weighted estimates are also reported as a sensitivity view.".to_string(),
            "Critical-invariant failures are a non-compensable gate; the mean and confidence interval cannot offset them.".to_string(),
            "Synthetic benchmark performance is not evidence of subjective love, consciousness, clinical efficacy, causal flourishing, or zero real-world risk.".to_string(),
            "This report is not release approval. Independent oracle adjudication, held-out evaluation, and external replication remain separate requirements.".to_string(),
        ],
    })
}

fn run(args: Args) -> Result<()> {
    ensure!(args.bootstrap_replicates >= MIN_BOOTSTRAP_REPLICATES, "bootstrap_replicates must be at least {MIN_BOOTSTRAP_REPLICATES}");
    let a: Receipt = read_json(&args.receipt_a)?;
    let b: Receipt = read_json(&args.receipt_b)?;
    let cluster_map: ClusterMap = read_json(&args.clusters)?;
    let results_a = validate_receipt(&a, "receipt A")?;
    let results_b = validate_receipt(&b, "receipt B")?;
    validate_compatibility(&a, &b)?;
    let report = build_report(&a, &b, &results_a, &results_b, &cluster_map, args.bootstrap_replicates, args.seed)?;

    if let Some(parent) = args.out.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("create report directory {}", parent.display()))?;
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&report).context("serialize comparison report")?)
        .with_context(|| format!("write {}", args.out.display()))?;
    println!(
        "Wrote {}: {} cases / {} clusters; cluster-weighted B−A delta {:+.4} (95% bootstrap CI {:+.4}, {:+.4}); critical gate {}",
        args.out.display(), report.metrics.case_count, report.metrics.cluster_count,
        report.metrics.cluster_weighted_delta_b_minus_a,
        report.metrics.cluster_bootstrap_ci95_low, report.metrics.cluster_bootstrap_ci95_high,
        report.critical_invariant_gate.status
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_is_reproducible_for_fixed_seed() {
        let effects = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let first = cluster_bootstrap_ci(&effects, 2_000, 17).expect("bootstrap");
        let second = cluster_bootstrap_ci(&effects, 2_000, 17).expect("same bootstrap");
        assert_eq!(first.0.to_bits(), second.0.to_bits());
        assert_eq!(first.1.to_bits(), second.1.to_bits());
    }

    #[test]
    fn percentile_uses_linear_interpolation() {
        assert!((quantile(&[0.0, 10.0], 0.25) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_rejects_fewer_than_two_clusters() {
        assert!(cluster_bootstrap_ci(&[0.25], 2_000, 1).is_err());
    }

    #[test]
    fn bootstrap_rejects_too_few_replicates() {
        assert!(cluster_bootstrap_ci(&[0.0, 1.0], 999, 1).is_err());
    }

    fn fixture_receipt() -> Receipt {
        let digest = "a".repeat(64);
        let commit = "b".repeat(40);
        let document = serde_json::json!({
            "receipt_id": "SLD-RUN-FIXTURE-0001",
            "schema_version": "sld-evaluation-receipt-0.1",
            "run_status": "COMPLETE",
            "evaluation_mode": "MODEL_BEHAVIOR",
            "started_at": "2026-10-22T12:00:00Z",
            "finished_at": "2026-10-22T12:01:00Z",
            "source": {
                "commit_sha": commit,
                "working_tree": "CLEAN",
                "toolchain": "rustc-test-fixture"
            },
            "case_set": {
                "dataset_id": "fixture",
                "dataset_version": "0.1",
                "visibility": "PUBLIC",
                "schema_sha256": digest,
                "corpus_sha256": digest,
                "inventory_sha256": digest,
                "expected_case_count": 2,
                "expected_case_ids": ["CASE-1", "CASE-2"],
                "split_id": "test",
                "split_seed": "1"
            },
            "subject": {
                "provider": "fixture",
                "model_id": "model-a",
                "model_snapshot": "snapshot-1",
                "weights_sha256": null,
                "endpoint_class": "LOCAL_RUNTIME"
            },
            "generation": {
                "system_prompt_sha256": digest,
                "prompt_bundle_sha256": digest,
                "decoding_config_sha256": digest,
                "tool_policy_sha256": digest,
                "memory_policy": "NONE",
                "max_turns": 5,
                "seed": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "tool_versions_sha256": digest
            },
            "evaluator": {
                "rubric_sha256": digest,
                "judge_panel": [{
                    "judge_id": "fixture-judge",
                    "snapshot": "fixture-1",
                    "implementation_sha256": digest
                }],
                "human_review_status": "COMPLETE",
                "human_rater_count": 3,
                "independent_adjudication_status": "COMPLETE"
            },
            "oracle_adjudication": {
                "status": "COMPLETE",
                "case_id_inventory_sha256": digest,
                "ledger_sha256": digest,
                "protocol_sha256": digest,
                "rubric_sha256": digest,
                "reviewer_count": 3,
                "expected_case_count": 2,
                "cases_with_two_independent_initial_reviews": 2,
                "cases_adjudicated": 2,
                "cases_indeterminate": 0,
                "unresolved_disagreements": 0,
                "critical_cases_reviewed": 1,
                "blind_initial_reviews": true,
                "case_assignment_randomized": true,
                "case_authors_excluded_from_adjudication": true,
                "conflicts_disclosed": true,
                "initial_label_pairwise_agreement_rate": 1.0,
                "initial_label_unanimous_case_rate": 1.0,
                "completed_at": "2026-10-22T12:02:00Z"
            },
            "environment": {
                "os_arch": "fixture",
                "runtime": "fixture",
                "dependency_lock_sha256": digest,
                "hardware_class": "fixture"
            },
            "case_results": [
                {"case_id": "CASE-1", "status": "PASS", "expected_policy": "x", "observed_policy": "x", "critical_invariant_failure": false, "response_sha256": digest, "reason_codes": []},
                {"case_id": "CASE-2", "status": "FAIL", "expected_policy": "x", "observed_policy": "y", "critical_invariant_failure": false, "response_sha256": digest, "reason_codes": ["fixture"]}
            ],
            "summary": {
                "expected": 2,
                "observed": 2,
                "pass": 1,
                "fail": 1,
                "indeterminate": 0,
                "error": 0,
                "skipped": 0,
                "missing": 0,
                "duplicate_ids": 0,
                "critical_invariant_failures": 0
            },
            "limitations": ["synthetic test fixture"],
            "comparison_eligibility": "COMPARABLE"
        });
        serde_json::from_value(document).expect("parse fixture receipt")
    }

    #[test]
    fn receipt_validator_accepts_complete_census() {
        let receipt = fixture_receipt();
        let results = validate_receipt(&receipt, "fixture").expect("valid fixture");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn receipt_validator_rejects_incomplete_census() {
        let mut receipt = fixture_receipt();
        receipt.case_results.remove(0);
        assert!(validate_receipt(&receipt, "fixture").is_err());
    }

    #[test]
    fn compatibility_rejects_generation_mismatch() {
        let a = fixture_receipt();
        let mut b = fixture_receipt();
        b.subject.model_id = "model-b".to_string();
        b.generation.max_turns += 1;
        assert!(validate_compatibility(&a, &b).is_err());
    }


    #[test]
    fn compatibility_rejects_unadjudicated_oracle() {
        let mut a = fixture_receipt();
        let b = fixture_receipt();
        a.evaluator.independent_adjudication_status = "NOT_PERFORMED".to_string();
        assert!(validate_compatibility(&a, &b).is_err());
    }


    #[test]
    fn critical_failure_blocks_promotion_even_when_aggregate_is_computable() {
        let mut a = fixture_receipt();
        let b = fixture_receipt();
        a.case_results[0].status = "FAIL".to_string();
        a.case_results[0].critical_invariant_failure = true;
        a.summary.pass = 0;
        a.summary.fail = 2;
        a.summary.critical_invariant_failures = 1;
        let results_a = validate_receipt(&a, "A").expect("valid receipt A");
        let results_b = validate_receipt(&b, "B").expect("valid receipt B");
        validate_compatibility(&a, &b).expect("compatible receipts");
        let cluster_map = ClusterMap {
            schema_version: CLUSTER_SCHEMA_VERSION.to_string(),
            cases: vec![
                ClusterAssignment {
                    case_id: "CASE-1".to_string(),
                    cluster_id: "CLUSTER-1".to_string(),
                    family: "FIXTURE".to_string(),
                },
                ClusterAssignment {
                    case_id: "CASE-2".to_string(),
                    cluster_id: "CLUSTER-2".to_string(),
                    family: "FIXTURE".to_string(),
                },
            ],
        };
        let report = build_report(&a, &b, &results_a, &results_b, &cluster_map, 1_000, 7)
            .expect("comparable synthetic fixture");
        assert_eq!(
            report.critical_invariant_gate.status,
            "BLOCKED_CRITICAL_FAILURE_OBSERVED"
        );
        assert!(!report.critical_invariant_gate.release_promotion_eligible);
    }

}
