use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const COHORT_SCHEMA: &str = "pitb.matched-benchmark-cohort.v0";
const RECORD_SCHEMA: &str = "pitb.matched-benchmark-run.v0";
const REPORT_SCHEMA: &str = "pitb.matched-benchmark-report.v0";
const MAX_BOOTSTRAP_REPLICATES: usize = 100_000;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    arm_id: String,
    model_id: String,
    model_version: String,
    scaffold_sha256: String,
    implementation_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Cohort {
    schema_version: String,
    benchmark_id: String,
    snapshot_id: String,
    split_id: String,
    split_role: String,
    split_manifest_sha256: String,
    expected_task_count: usize,
    expected_cluster_count: usize,
    task_id_set_sha256: String,
    budget_policy_sha256: String,
    max_attempts_per_task: u32,
    max_wall_ms_per_task: u64,
    max_cost_microusd_per_task: u64,
    require_source_reproduction: bool,
    require_specification_audit: bool,
    require_vacuity_audit: bool,
    require_assumption_audit: bool,
    bootstrap_replicates: usize,
    bootstrap_seed_sha256: String,
    arms: Vec<Arm>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunRecord {
    schema_version: String,
    benchmark_id: String,
    snapshot_id: String,
    split_id: String,
    split_manifest_sha256: String,
    task_id: String,
    cluster_id: String,
    repository: String,
    repository_commit: String,
    lean_version: String,
    source_file: String,
    target_statement_sha256: String,
    environment_sha256: String,
    budget_policy_sha256: String,
    arm_id: String,
    model_id: String,
    model_version: String,
    scaffold_sha256: String,
    implementation_sha256: String,
    outcome: String,
    attempts: u32,
    wall_ms: u64,
    cost_microusd: u64,
    tokens: u64,
    kernel_verified: bool,
    source_reproduction_status: String,
    source_reproduction_receipt_sha256: Option<String>,
    goal_alignment: String,
    goal_alignment_receipt_sha256: Option<String>,
    specification_audit_status: String,
    specification_audit_receipt_sha256: Option<String>,
    vacuity_audit_status: String,
    vacuity_audit_receipt_sha256: Option<String>,
    assumption_audit_status: String,
    assumption_audit_receipt_sha256: Option<String>,
    run_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArmMetrics {
    arm_id: String,
    model_id: String,
    model_version: String,
    task_count: usize,
    attempted_task_count: usize,
    did_not_compete_count: usize,
    raw_kernel_verified_successes: usize,
    source_reproduced_successes: usize,
    assurance_qualified_successes: usize,
    assurance_disqualified_successes: usize,
    assurance_unreviewed_successes: usize,
    raw_kernel_success_rate: f64,
    source_reproduced_success_rate: f64,
    assurance_qualified_success_rate: f64,
    attempt_coverage: f64,
    total_attempts: u64,
    total_wall_ms: u64,
    total_cost_microusd: u64,
    total_tokens: u64,
    cost_per_raw_kernel_success_microusd: Option<f64>,
    cost_per_source_reproduced_success_microusd: Option<f64>,
    cost_per_assurance_qualified_success_microusd: Option<f64>,
    median_cost_per_task_microusd: f64,
    p95_cost_per_task_microusd: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PairComparison {
    reference_arm_id: String,
    challenger_arm_id: String,
    paired_task_count: usize,
    repository_cluster_count: usize,
    reference_assurance_qualified_successes: usize,
    challenger_assurance_qualified_successes: usize,
    challenger_wins: usize,
    reference_wins: usize,
    ties: usize,
    paired_assurance_success_difference: f64,
    cluster_bootstrap_95_percentile_lower: f64,
    cluster_bootstrap_95_percentile_upper: f64,
    bootstrap_replicates: usize,
    bootstrap_seed_sha256: String,
    interpretation_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    benchmark_id: String,
    snapshot_id: String,
    split_id: String,
    split_role: String,
    split_manifest_sha256: String,
    expected_task_count: usize,
    expected_cluster_count: usize,
    task_id_set_sha256: String,
    results_jsonl_sha256: String,
    observed_task_count: usize,
    observed_cluster_count: usize,
    arm_count: usize,
    budget_policy_sha256: String,
    source_reproduction_required: bool,
    specification_audit_required: bool,
    vacuity_audit_required: bool,
    assumption_audit_required: bool,
    arm_metrics: Vec<ArmMetrics>,
    pairwise_comparisons: Vec<PairComparison>,
    missing_receipt_count: usize,
    scheduler_admission: bool,
    primary_metric: String,
    statistical_interpretation_boundary: String,
    receipt_authentication_boundary: String,
}

#[derive(Debug, Clone)]
struct TaskIdentity {
    cluster_id: String,
    repository: String,
    repository_commit: String,
    lean_version: String,
    source_file: String,
    target_statement_sha256: String,
    environment_sha256: String,
    budget_policy_sha256: String,
}

#[derive(Debug, Clone, Copy)]
struct TaskScore {
    raw_success: bool,
    source_reproduced_success: bool,
    assurance_qualified_success: bool,
    assurance_disqualified_success: bool,
    assurance_unreviewed_success: bool,
}

#[derive(Debug, Clone)]
struct ClusterDiff {
    task_differences: Vec<f64>,
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

fn digest(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 64) {
        Ok(())
    } else {
        Err(format!("{label} must be 64 lowercase SHA-256 hex characters"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

fn optional_cost_ratio(cost: u64, successes: usize) -> Option<f64> {
    (successes > 0).then_some(cost as f64 / successes as f64)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn canonical_task_id_set_digest(ids: &[&str]) -> String {
    let sorted: BTreeSet<String> = ids.iter().map(|id| (*id).to_owned()).collect();
    let bytes = serde_json::to_vec(&sorted).expect("task ID set serializes deterministically");
    sha256(&bytes)
}

fn validate_cohort(cohort: &Cohort) -> Result<(), String> {
    if cohort.schema_version != COHORT_SCHEMA {
        return Err("unsupported cohort schema_version".into());
    }
    for (label, value) in [
        ("benchmark_id", cohort.benchmark_id.as_str()),
        ("snapshot_id", cohort.snapshot_id.as_str()),
        ("split_id", cohort.split_id.as_str()),
    ] {
        non_empty(value, &format!("cohort.{label}"))?;
    }
    if cohort.split_role != "test" {
        return Err("matched comparisons must use a test split, not train/validation/development".into());
    }
    digest(&cohort.split_manifest_sha256, "cohort.split_manifest_sha256")?;
    digest(&cohort.task_id_set_sha256, "cohort.task_id_set_sha256")?;
    digest(&cohort.budget_policy_sha256, "cohort.budget_policy_sha256")?;
    digest(&cohort.bootstrap_seed_sha256, "cohort.bootstrap_seed_sha256")?;
    if cohort.expected_task_count == 0 || cohort.expected_cluster_count == 0 {
        return Err("cohort expected task and cluster counts must be positive".into());
    }
    if cohort.max_attempts_per_task == 0 || cohort.max_wall_ms_per_task == 0 || cohort.max_cost_microusd_per_task == 0 {
        return Err("cohort hard resource limits must be positive".into());
    }
    if !(1_000..=MAX_BOOTSTRAP_REPLICATES).contains(&cohort.bootstrap_replicates) {
        return Err(format!(
            "bootstrap_replicates must be between 1000 and {MAX_BOOTSTRAP_REPLICATES}"
        ));
    }
    if cohort.arms.len() < 2 {
        return Err("a matched comparison requires at least two arms".into());
    }
    let mut arm_ids = BTreeSet::new();
    for arm in &cohort.arms {
        non_empty(&arm.arm_id, "arm.arm_id")?;
        non_empty(&arm.model_id, "arm.model_id")?;
        non_empty(&arm.model_version, "arm.model_version")?;
        digest(&arm.scaffold_sha256, "arm.scaffold_sha256")?;
        digest(&arm.implementation_sha256, "arm.implementation_sha256")?;
        if !arm_ids.insert(arm.arm_id.clone()) {
            return Err(format!("duplicate arm_id {}", arm.arm_id));
        }
    }
    Ok(())
}

fn validate_status(status: &str, allowed: &[&str], label: &str) -> Result<(), String> {
    if allowed.contains(&status) {
        Ok(())
    } else {
        Err(format!("{label} has unsupported status {status}"))
    }
}

fn validate_optional_receipt(
    status: &str,
    receipt: &Option<String>,
    passed_or_finding: &[&str],
    label: &str,
) -> Result<(), String> {
    if passed_or_finding.contains(&status) {
        let value = receipt.as_deref().ok_or_else(|| format!("{label} status requires a receipt digest"))?;
        digest(value, &format!("{label} receipt"))?;
    } else if receipt.is_some() {
        return Err(format!("{label} receipt digest must be null when status is not_run/not_applicable"));
    }
    Ok(())
}

fn validate_run(
    row: &RunRecord,
    cohort: &Cohort,
    task_identity: Option<&TaskIdentity>,
) -> Result<(TaskIdentity, TaskScore), String> {
    if row.schema_version != RECORD_SCHEMA {
        return Err("run record has unsupported schema_version".into());
    }
    for (field, actual, expected) in [
        ("benchmark_id", row.benchmark_id.as_str(), cohort.benchmark_id.as_str()),
        ("snapshot_id", row.snapshot_id.as_str(), cohort.snapshot_id.as_str()),
        ("split_id", row.split_id.as_str(), cohort.split_id.as_str()),
        ("split_manifest_sha256", row.split_manifest_sha256.as_str(), cohort.split_manifest_sha256.as_str()),
        ("budget_policy_sha256", row.budget_policy_sha256.as_str(), cohort.budget_policy_sha256.as_str()),
    ] {
        if actual != expected {
            return Err(format!("run {field} differs from frozen cohort"));
        }
    }
    for (label, value) in [
        ("task_id", row.task_id.as_str()),
        ("cluster_id", row.cluster_id.as_str()),
        ("repository", row.repository.as_str()),
        ("repository_commit", row.repository_commit.as_str()),
        ("lean_version", row.lean_version.as_str()),
        ("source_file", row.source_file.as_str()),
        ("arm_id", row.arm_id.as_str()),
        ("model_id", row.model_id.as_str()),
        ("model_version", row.model_version.as_str()),
    ] {
        non_empty(value, &format!("run.{label}"))?;
    }
    if !(valid_hex(&row.repository_commit, 40) || valid_hex(&row.repository_commit, 64)) {
        return Err("run.repository_commit must be a full lowercase Git revision".into());
    }
    non_empty(&row.lean_version, "run.lean_version")?;
    for (label, value) in [
        ("target_statement_sha256", row.target_statement_sha256.as_str()),
        ("environment_sha256", row.environment_sha256.as_str()),
        ("scaffold_sha256", row.scaffold_sha256.as_str()),
        ("implementation_sha256", row.implementation_sha256.as_str()),
        ("run_receipt_sha256", row.run_receipt_sha256.as_str()),
    ] {
        digest(value, &format!("run.{label}"))?;
    }
    if row.attempts > cohort.max_attempts_per_task
        || row.wall_ms > cohort.max_wall_ms_per_task
        || row.cost_microusd > cohort.max_cost_microusd_per_task
    {
        return Err(format!("run {} / {} exceeds the frozen hard resource envelope", row.task_id, row.arm_id));
    }
    validate_status(
        &row.outcome,
        &["solved", "unsolved", "timeout", "invalid_proof", "build_failure", "did_not_compete"],
        "run.outcome",
    )?;
    validate_status(
        &row.source_reproduction_status,
        &["passed", "failed", "not_run", "not_applicable"],
        "run.source_reproduction_status",
    )?;
    validate_status(
        &row.goal_alignment,
        &["exact", "normalized", "reviewed", "mismatch", "not_applicable"],
        "run.goal_alignment",
    )?;
    validate_status(
        &row.specification_audit_status,
        &["passed", "finding", "not_run", "not_applicable"],
        "run.specification_audit_status",
    )?;
    validate_status(
        &row.vacuity_audit_status,
        &["passed", "finding", "not_run", "not_applicable"],
        "run.vacuity_audit_status",
    )?;
    validate_status(
        &row.assumption_audit_status,
        &["passed", "finding", "not_run", "not_applicable"],
        "run.assumption_audit_status",
    )?;
    validate_optional_receipt(
        &row.source_reproduction_status,
        &row.source_reproduction_receipt_sha256,
        &["passed", "failed"],
        "source_reproduction",
    )?;
    validate_optional_receipt(
        &row.goal_alignment,
        &row.goal_alignment_receipt_sha256,
        &["normalized", "reviewed", "mismatch"],
        "goal_alignment",
    )?;
    validate_optional_receipt(
        &row.specification_audit_status,
        &row.specification_audit_receipt_sha256,
        &["passed", "finding"],
        "specification_audit",
    )?;
    validate_optional_receipt(
        &row.vacuity_audit_status,
        &row.vacuity_audit_receipt_sha256,
        &["passed", "finding"],
        "vacuity_audit",
    )?;
    validate_optional_receipt(
        &row.assumption_audit_status,
        &row.assumption_audit_receipt_sha256,
        &["passed", "finding"],
        "assumption_audit",
    )?;

    if row.outcome == "solved" && !row.kernel_verified {
        return Err(format!("run {} claims solved without independent kernel verification", row.task_id));
    }
    if row.outcome != "solved" && row.kernel_verified {
        return Err(format!("run {} claims kernel verification but outcome is not solved", row.task_id));
    }
    if row.outcome == "did_not_compete"
        && (row.attempts != 0 || row.wall_ms != 0 || row.cost_microusd != 0 || row.tokens != 0 || row.kernel_verified)
    {
        return Err(format!("did_not_compete row {} must have zero attempts, resources and no proof", row.task_id));
    }
    if row.outcome != "did_not_compete" && row.attempts == 0 {
        return Err(format!("attempted row {} must record at least one attempt", row.task_id));
    }
    if row.goal_alignment == "exact" && row.goal_alignment_receipt_sha256.is_some() {
        return Err("exact goal alignment does not require a normalization/review receipt".into());
    }

    let arm = cohort.arms.iter().find(|arm| arm.arm_id == row.arm_id)
        .ok_or_else(|| format!("run references undeclared arm {}", row.arm_id))?;
    if row.model_id != arm.model_id
        || row.model_version != arm.model_version
        || row.scaffold_sha256 != arm.scaffold_sha256
        || row.implementation_sha256 != arm.implementation_sha256
    {
        return Err(format!("run {} arm identity differs from cohort manifest", row.arm_id));
    }

    let identity = TaskIdentity {
        cluster_id: row.cluster_id.clone(),
        repository: row.repository.clone(),
        repository_commit: row.repository_commit.clone(),
        lean_version: row.lean_version.clone(),
        source_file: row.source_file.clone(),
        target_statement_sha256: row.target_statement_sha256.clone(),
        environment_sha256: row.environment_sha256.clone(),
        budget_policy_sha256: row.budget_policy_sha256.clone(),
    };
    if let Some(expected) = task_identity {
        if expected.cluster_id != identity.cluster_id
            || expected.repository != identity.repository
            || expected.repository_commit != identity.repository_commit
            || expected.lean_version != identity.lean_version
            || expected.source_file != identity.source_file
            || expected.target_statement_sha256 != identity.target_statement_sha256
            || expected.environment_sha256 != identity.environment_sha256
            || expected.budget_policy_sha256 != identity.budget_policy_sha256
        {
            return Err(format!("matched task {} differs across arms", row.task_id));
        }
    }

    let goal_ok = row.goal_alignment != "mismatch";
    let source_reproduced = row.source_reproduction_status == "passed"
        && matches!(row.goal_alignment.as_str(), "exact" | "normalized" | "reviewed");
    let source_ok = !cohort.require_source_reproduction || source_reproduced;
    let specification_ok = !cohort.require_specification_audit || row.specification_audit_status == "passed";
    let vacuity_ok = !cohort.require_vacuity_audit || row.vacuity_audit_status == "passed";
    let assumptions_ok = !cohort.require_assumption_audit || row.assumption_audit_status == "passed";
    let raw_success = row.outcome == "solved" && row.kernel_verified;
    let source_reproduced_success = raw_success && source_reproduced && goal_ok;
    let assurance_qualified_success =
        raw_success && goal_ok && source_ok && specification_ok && vacuity_ok && assumptions_ok;
    let assurance_disqualified_success = raw_success
        && (row.goal_alignment == "mismatch"
            || row.specification_audit_status == "finding"
            || row.vacuity_audit_status == "finding"
            || row.assumption_audit_status == "finding");
    let assurance_unreviewed_success = raw_success
        && !assurance_qualified_success
        && !assurance_disqualified_success;

    Ok((
        identity,
        TaskScore {
            raw_success,
            source_reproduced_success,
            assurance_qualified_success,
            assurance_disqualified_success,
            assurance_unreviewed_success,
        },
    ))
}

fn median_p95_cost(rows: &[&RunRecord]) -> (f64, f64) {
    let mut values: Vec<f64> = rows.iter().map(|row| row.cost_microusd as f64).collect();
    values.sort_by(f64::total_cmp);
    (quantile(&values, 0.50), quantile(&values, 0.95))
}

fn arm_metrics(arm: &Arm, rows: &[&RunRecord], scores: &[TaskScore]) -> ArmMetrics {
    let task_count = rows.len();
    let attempted_task_count = rows.iter().filter(|row| row.outcome != "did_not_compete").count();
    let did_not_compete_count = task_count - attempted_task_count;
    let raw = scores.iter().filter(|score| score.raw_success).count();
    let source = scores.iter().filter(|score| score.source_reproduced_success).count();
    let qualified = scores.iter().filter(|score| score.assurance_qualified_success).count();
    let disqualified = scores.iter().filter(|score| score.assurance_disqualified_success).count();
    let unreviewed = scores.iter().filter(|score| score.assurance_unreviewed_success).count();
    let attempts = rows.iter().map(|row| row.attempts as u64).sum();
    let wall = rows.iter().map(|row| row.wall_ms).sum();
    let cost = rows.iter().map(|row| row.cost_microusd).sum();
    let tokens = rows.iter().map(|row| row.tokens).sum();
    let (median, p95) = median_p95_cost(rows);
    ArmMetrics {
        arm_id: arm.arm_id.clone(),
        model_id: arm.model_id.clone(),
        model_version: arm.model_version.clone(),
        task_count,
        attempted_task_count,
        did_not_compete_count,
        raw_kernel_verified_successes: raw,
        source_reproduced_successes: source,
        assurance_qualified_successes: qualified,
        assurance_disqualified_successes: disqualified,
        assurance_unreviewed_successes: unreviewed,
        raw_kernel_success_rate: ratio(raw, task_count),
        source_reproduced_success_rate: ratio(source, task_count),
        assurance_qualified_success_rate: ratio(qualified, task_count),
        attempt_coverage: ratio(attempted_task_count, task_count),
        total_attempts: attempts,
        total_wall_ms: wall,
        total_cost_microusd: cost,
        total_tokens: tokens,
        cost_per_raw_kernel_success_microusd: optional_cost_ratio(cost, raw),
        cost_per_source_reproduced_success_microusd: optional_cost_ratio(cost, source),
        cost_per_assurance_qualified_success_microusd: optional_cost_ratio(cost, qualified),
        median_cost_per_task_microusd: median,
        p95_cost_per_task_microusd: p95,
    }
}

#[derive(Debug, Clone)]
struct XorShift64(u64);

impl XorShift64 {
    fn from_seed(seed_hex: &str) -> Self {
        let seed = u64::from_str_radix(&seed_hex[..16], 16).unwrap_or(1);
        Self(if seed == 0 { 0x9e3779b97f4a7c15 } else { seed })
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }
}

fn cluster_bootstrap_interval(
    clusters: &BTreeMap<String, ClusterDiff>,
    replicates: usize,
    seed: &str,
) -> (f64, f64) {
    let cluster_values: Vec<&ClusterDiff> = clusters.values().collect();
    let mut rng = XorShift64::from_seed(seed);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut difference_sum = 0.0;
        let mut task_weight = 0usize;
        for _ in 0..cluster_values.len() {
            let selected = cluster_values[rng.index(cluster_values.len())];
            difference_sum += selected.task_differences.iter().sum::<f64>();
            task_weight += selected.task_differences.len();
        }
        estimates.push(if task_weight == 0 { 0.0 } else { difference_sum / task_weight as f64 });
    }
    estimates.sort_by(f64::total_cmp);
    (quantile(&estimates, 0.025), quantile(&estimates, 0.975))
}

fn compare_pair(
    reference: &Arm,
    challenger: &Arm,
    task_scores: &BTreeMap<String, BTreeMap<String, TaskScore>>,
    task_clusters: &BTreeMap<String, String>,
    replicates: usize,
    seed: &str,
) -> Result<PairComparison, String> {
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut ties = 0usize;
    let mut reference_successes = 0usize;
    let mut challenger_successes = 0usize;
    let mut differences = Vec::new();
    let mut clusters = BTreeMap::<String, ClusterDiff>::new();
    for (task_id, arm_scores) in task_scores {
        let baseline = arm_scores.get(&reference.arm_id)
            .ok_or_else(|| format!("missing reference arm {} for {task_id}", reference.arm_id))?;
        let candidate = arm_scores.get(&challenger.arm_id)
            .ok_or_else(|| format!("missing challenger arm {} for {task_id}", challenger.arm_id))?;
        let baseline_value = usize::from(baseline.assurance_qualified_success);
        let candidate_value = usize::from(candidate.assurance_qualified_success);
        reference_successes += baseline_value;
        challenger_successes += candidate_value;
        let difference = candidate_value as f64 - baseline_value as f64;
        differences.push(difference);
        if difference > 0.0 {
            wins += 1;
        } else if difference < 0.0 {
            losses += 1;
        } else {
            ties += 1;
        }
        let cluster_id = task_clusters.get(task_id)
            .ok_or_else(|| format!("missing cluster identity for {task_id}"))?;
        clusters.entry(cluster_id.clone())
            .or_insert_with(|| ClusterDiff { task_differences: Vec::new() })
            .task_differences.push(difference);
    }
    let paired_task_count = differences.len();
    let mean_difference = if paired_task_count == 0 {
        0.0
    } else {
        differences.iter().sum::<f64>() / paired_task_count as f64
    };
    let (lower, upper) = cluster_bootstrap_interval(&clusters, replicates, seed);
    Ok(PairComparison {
        reference_arm_id: reference.arm_id.clone(),
        challenger_arm_id: challenger.arm_id.clone(),
        paired_task_count,
        repository_cluster_count: clusters.len(),
        reference_assurance_qualified_successes: reference_successes,
        challenger_assurance_qualified_successes: challenger_successes,
        challenger_wins: wins,
        reference_wins: losses,
        ties,
        paired_assurance_success_difference: mean_difference,
        cluster_bootstrap_95_percentile_lower: lower,
        cluster_bootstrap_95_percentile_upper: upper,
        bootstrap_replicates: replicates,
        bootstrap_seed_sha256: seed.to_owned(),
        interpretation_boundary: "The interval resamples repository clusters, not individual task rows. It estimates the paired task-weighted success-rate difference for this frozen split and budget policy; it does not establish project-disjoint or temporal generalization.".into(),
    })
}

fn analyze(cohort: &Cohort, rows: &[RunRecord], input_sha256: String) -> Result<Report, String> {
    validate_cohort(cohort)?;
    let expected_arms: BTreeSet<String> = cohort.arms.iter().map(|arm| arm.arm_id.clone()).collect();
    let mut by_task = BTreeMap::<String, BTreeMap<String, &RunRecord>>::new();
    let mut task_identities = BTreeMap::<String, TaskIdentity>::new();
    let mut task_scores = BTreeMap::<String, BTreeMap<String, TaskScore>>::new();
    let mut task_clusters = BTreeMap::<String, String>::new();
    for (index, row) in rows.iter().enumerate() {
        let arm = cohort.arms.iter().find(|arm| arm.arm_id == row.arm_id)
            .ok_or_else(|| format!("row {} references undeclared arm {}", index + 1, row.arm_id))?;
        let current_identity = task_identities.get(&row.task_id);
        let (identity, score) = validate_run(row, cohort, current_identity)?;
        if let Some(existing) = task_identities.get(&row.task_id) {
            if existing.cluster_id != identity.cluster_id {
                return Err(format!("task {} changed repository cluster", row.task_id));
            }
        } else {
            task_identities.insert(row.task_id.clone(), identity.clone());
            task_clusters.insert(row.task_id.clone(), identity.cluster_id.clone());
        }
        let task_rows = by_task.entry(row.task_id.clone()).or_default();
        if task_rows.insert(arm.arm_id.clone(), row).is_some() {
            return Err(format!("duplicate row for task {} / arm {}", row.task_id, arm.arm_id));
        }
        task_scores.entry(row.task_id.clone()).or_default().insert(arm.arm_id.clone(), score);
    }
    if by_task.len() != cohort.expected_task_count {
        return Err(format!(
            "observed {} task IDs; cohort pins {}",
            by_task.len(), cohort.expected_task_count
        ));
    }
    for (task_id, arm_rows) in &by_task {
        let actual: BTreeSet<String> = arm_rows.keys().cloned().collect();
        if actual != expected_arms {
            let missing: Vec<_> = expected_arms.difference(&actual).cloned().collect();
            return Err(format!("task {task_id} does not contain exactly one row per arm; missing {missing:?}"));
        }
    }
    let observed_clusters: BTreeSet<String> = task_clusters.values().cloned().collect();
    let observed_task_ids: Vec<&str> = by_task.keys().map(String::as_str).collect();
    if canonical_task_id_set_digest(&observed_task_ids) != cohort.task_id_set_sha256 {
        return Err("observed task IDs do not match the frozen cohort task_id_set_sha256".into());
    }
    if observed_clusters.len() != cohort.expected_cluster_count {
        return Err(format!(
            "observed {} clusters; cohort pins {}",
            observed_clusters.len(), cohort.expected_cluster_count
        ));
    }

    let mut arm_metrics_list = Vec::new();
    for arm in &cohort.arms {
        let mut arm_rows = Vec::with_capacity(cohort.expected_task_count);
        let mut arm_scores = Vec::with_capacity(cohort.expected_task_count);
        for task in by_task.values() {
            let row = task.get(&arm.arm_id).ok_or_else(|| format!("missing arm {}", arm.arm_id))?;
            arm_rows.push(*row);
            arm_scores.push(*task_scores.get(&row.task_id).and_then(|scores| scores.get(&arm.arm_id)).unwrap());
        }
        arm_metrics_list.push(arm_metrics(arm, &arm_rows, &arm_scores));
    }

    let mut pairwise = Vec::new();
    for i in 0..cohort.arms.len() {
        for j in 0..cohort.arms.len() {
            if i == j { continue; }
            let pair_seed = sha256(format!(
                "{}|{}|{}|{}",
                cohort.bootstrap_seed_sha256,
                cohort.benchmark_id,
                cohort.arms[i].arm_id,
                cohort.arms[j].arm_id
            ).as_bytes());
            pairwise.push(compare_pair(
                &cohort.arms[i],
                &cohort.arms[j],
                &task_scores,
                &task_clusters,
                cohort.bootstrap_replicates,
                &pair_seed,
            )?);
        }
    }

    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        benchmark_id: cohort.benchmark_id.clone(),
        snapshot_id: cohort.snapshot_id.clone(),
        split_id: cohort.split_id.clone(),
        split_role: cohort.split_role.clone(),
        split_manifest_sha256: cohort.split_manifest_sha256.clone(),
        expected_task_count: cohort.expected_task_count,
        expected_cluster_count: cohort.expected_cluster_count,
        task_id_set_sha256: cohort.task_id_set_sha256.clone(),
        results_jsonl_sha256: input_sha256,
        observed_task_count: by_task.len(),
        observed_cluster_count: observed_clusters.len(),
        arm_count: cohort.arms.len(),
        budget_policy_sha256: cohort.budget_policy_sha256.clone(),
        source_reproduction_required: cohort.require_source_reproduction,
        specification_audit_required: cohort.require_specification_audit,
        vacuity_audit_required: cohort.require_vacuity_audit,
        assumption_audit_required: cohort.require_assumption_audit,
        arm_metrics: arm_metrics_list,
        pairwise_comparisons: pairwise,
        missing_receipt_count: 0,
        scheduler_admission: false,
        primary_metric: "assurance-qualified exact-target completion rate under the same frozen benchmark split, task set, resource envelope and pinned proof-checking policy".into(),
        statistical_interpretation_boundary: "All comparisons are paired on identical task IDs. Confidence intervals resample repository clusters to account for within-project dependence. The reported point estimate is task-weighted for the frozen cohort. Do not treat tasks as IID, compare different benchmark snapshots as if paired, or infer project-disjoint generalization from a task-level split.".into(),
        receipt_authentication_boundary: format!(
            "The input JSONL digest is {input_sha256}. Run and audit receipt digests are syntax-checked and bound to each result row; this analyzer does not open or authenticate their referenced bytes. Authenticate those artifacts before publishing assurance-qualified scores."
        ),
    })
}

fn load_jsonl(bytes: &[u8]) -> Result<Vec<RunRecord>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("run JSONL is not UTF-8: {e}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let row = serde_json::from_str(line)
            .map_err(|e| format!("run JSONL line {}: {e}", index + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("run JSONL must contain at least one record".into());
    }
    Ok(rows)
}

fn fake_digest(label: &str) -> String {
    sha256(label.as_bytes())
}

fn synthetic_cohort() -> Cohort {
    Cohort {
        schema_version: COHORT_SCHEMA.into(),
        benchmark_id: "synthetic-sorrydb".into(),
        snapshot_id: "synthetic-2026-11".into(),
        split_id: "test".into(),
        split_role: "test".into(),
        split_manifest_sha256: fake_digest("split-manifest"),
        expected_task_count: 4,
        expected_cluster_count: 2,
        task_id_set_sha256: canonical_task_id_set_digest(&["task-1", "task-2", "task-3", "task-4"]),
        budget_policy_sha256: fake_digest("fixed-budget-policy"),
        max_attempts_per_task: 32,
        max_wall_ms_per_task: 300_000,
        max_cost_microusd_per_task: 5_000_000,
        require_source_reproduction: true,
        require_specification_audit: true,
        require_vacuity_audit: true,
        require_assumption_audit: true,
        bootstrap_replicates: 1_000,
        bootstrap_seed_sha256: fake_digest("seed"),
        arms: vec![
            Arm {
                arm_id: "baseline".into(),
                model_id: "model-a".into(),
                model_version: "v1".into(),
                scaffold_sha256: fake_digest("scaffold-a"),
                implementation_sha256: fake_digest("impl-a"),
            },
            Arm {
                arm_id: "challenger".into(),
                model_id: "model-b".into(),
                model_version: "v1".into(),
                scaffold_sha256: fake_digest("scaffold-b"),
                implementation_sha256: fake_digest("impl-b"),
            },
        ],
    }
}

fn synthetic_row(cohort: &Cohort, arm: &Arm, task_id: &str, cluster: &str, success: bool) -> RunRecord {
    let pass = |label: &str| Some(fake_digest(label));
    RunRecord {
        schema_version: RECORD_SCHEMA.into(),
        benchmark_id: cohort.benchmark_id.clone(),
        snapshot_id: cohort.snapshot_id.clone(),
        split_id: cohort.split_id.clone(),
        split_manifest_sha256: cohort.split_manifest_sha256.clone(),
        task_id: task_id.into(),
        cluster_id: cluster.into(),
        repository: format!("https://github.com/example/{cluster}"),
        repository_commit: fake_digest(&format!("commit:{cluster}"))[..40].to_owned(),
        lean_version: "v4.22.0".into(),
        source_file: format!("{cluster}/src/{task_id}.lean"),
        target_statement_sha256: fake_digest(&format!("target:{task_id}")),
        environment_sha256: fake_digest("environment"),
        budget_policy_sha256: cohort.budget_policy_sha256.clone(),
        arm_id: arm.arm_id.clone(),
        model_id: arm.model_id.clone(),
        model_version: arm.model_version.clone(),
        scaffold_sha256: arm.scaffold_sha256.clone(),
        implementation_sha256: arm.implementation_sha256.clone(),
        outcome: if success { "solved" } else { "timeout" }.into(),
        attempts: if success { 2 } else { 32 },
        wall_ms: if success { 4_000 } else { 300_000 },
        cost_microusd: if success { 120_000 } else { 1_000_000 },
        tokens: if success { 3_000 } else { 30_000 },
        kernel_verified: success,
        source_reproduction_status: "passed".into(),
        source_reproduction_receipt_sha256: pass(&format!("source:{task_id}")),
        goal_alignment: "exact".into(),
        goal_alignment_receipt_sha256: None,
        specification_audit_status: "passed".into(),
        specification_audit_receipt_sha256: pass(&format!("spec:{task_id}")),
        vacuity_audit_status: "passed".into(),
        vacuity_audit_receipt_sha256: pass(&format!("vacuity:{task_id}")),
        assumption_audit_status: "passed".into(),
        assumption_audit_receipt_sha256: pass(&format!("assumption:{task_id}")),
        run_receipt_sha256: fake_digest(&format!("run:{}:{task_id}", arm.arm_id)),
    }
}

fn synthetic_rows(cohort: &Cohort) -> Vec<RunRecord> {
    let tasks = [
        ("task-1", "repo-a", true, true),
        ("task-2", "repo-a", false, true),
        ("task-3", "repo-b", false, true),
        ("task-4", "repo-b", false, false),
    ];
    let mut rows = Vec::new();
    for (task, cluster, baseline, challenger) in tasks {
        rows.push(synthetic_row(cohort, &cohort.arms[0], task, cluster, baseline));
        rows.push(synthetic_row(cohort, &cohort.arms[1], task, cluster, challenger));
    }
    // The challenger has a valid kernel proof but only an unreviewed semantic
    // status for task-4; the kernel rate and assurance-qualified rate diverge.
    if let Some(row) = rows.iter_mut().find(|row| row.arm_id == "challenger" && row.task_id == "task-4") {
        row.specification_audit_status = "not_run".into();
        row.specification_audit_receipt_sha256 = None;
    }
    rows
}

fn run_self_tests() -> Result<(), String> {
    let cohort = synthetic_cohort();
    let rows = synthetic_rows(&cohort);
    let report = analyze(&cohort, &rows, fake_digest("synthetic-input"))?;
    if report.observed_task_count != 4 || report.observed_cluster_count != 2 {
        return Err("synthetic cohort denominators changed".into());
    }
    let baseline = report.arm_metrics.iter().find(|arm| arm.arm_id == "baseline").unwrap();
    let challenger = report.arm_metrics.iter().find(|arm| arm.arm_id == "challenger").unwrap();
    if baseline.raw_kernel_verified_successes != 1
        || baseline.assurance_qualified_successes != 1
        || challenger.raw_kernel_verified_successes != 4
        || challenger.assurance_qualified_successes != 3
        || challenger.assurance_unreviewed_successes != 1
    {
        return Err("raw kernel and assurance-qualified success were incorrectly conflated".into());
    }
    println!("SELF_TEST_PASS raw kernel success is separate from assurance-qualified success");

    let pair = report.pairwise_comparisons.iter()
        .find(|p| p.reference_arm_id == "baseline" && p.challenger_arm_id == "challenger")
        .ok_or("synthetic pair comparison missing")?;
    if pair.paired_task_count != 4
        || pair.repository_cluster_count != 2
        || pair.challenger_wins != 2
        || pair.reference_wins != 0
        || pair.ties != 2
        || pair.paired_assurance_success_difference != 0.5
    {
        return Err("paired task comparison or cluster accounting changed".into());
    }
    println!("SELF_TEST_PASS matched comparison reports paired wins, losses, ties and cluster bootstrap");

    let mut bad = rows.clone();
    bad[1].budget_policy_sha256 = fake_digest("other-budget");
    if analyze(&cohort, &bad, fake_digest("bad-budget")).is_ok() {
        return Err("mixed budget policies were accepted".into());
    }
    println!("SELF_TEST_PASS mixed budget policy rejected");

    let mut bad = rows.clone();
    bad.pop();
    if analyze(&cohort, &bad, fake_digest("missing-row")).is_ok() {
        return Err("missing task-arm result was accepted".into());
    }
    println!("SELF_TEST_PASS every task requires one result per arm");

    let mut bad = rows.clone();
    bad[1].environment_sha256 = fake_digest("other-environment");
    if analyze(&cohort, &bad, fake_digest("environment-drift")).is_ok() {
        return Err("matched task environment drift was accepted".into());
    }
    println!("SELF_TEST_PASS matched task identity and environment drift rejected");

    let mut bad = rows.clone();
    bad[0].outcome = "solved".into();
    bad[0].kernel_verified = false;
    if analyze(&cohort, &bad, fake_digest("non-kernel")).is_ok() {
        return Err("solved outcome without kernel check was accepted".into());
    }
    println!("SELF_TEST_PASS solved means independent kernel-verified proof");

    let mut bad = rows.clone();
    bad[0].attempts = cohort.max_attempts_per_task + 1;
    if analyze(&cohort, &bad, fake_digest("budget-overrun")).is_ok() {
        return Err("attempt count beyond the frozen envelope was accepted".into());
    }
    println!("SELF_TEST_PASS hard attempt budget enforced");

    let mut bad = rows.clone();
    bad[0].source_reproduction_status = "passed".into();
    bad[0].source_reproduction_receipt_sha256 = None;
    if analyze(&cohort, &bad, fake_digest("missing-source-receipt")).is_ok() {
        return Err("source reproduction without evidence receipt was accepted".into());
    }
    println!("SELF_TEST_PASS a passed gate requires its evidence digest");

    let mut bad = rows.clone();
    bad[0].outcome = "did_not_compete".into();
    bad[0].attempts = 1;
    if analyze(&cohort, &bad, fake_digest("invalid-dnc")).is_ok() {
        return Err("did_not_compete with attempted resources was accepted".into());
    }
    println!("SELF_TEST_PASS did-not-compete is a distinct zero-resource outcome");

    // Deterministic bootstrap: same cohort/rows/seed must produce byte-stable bounds.
    let again = analyze(&cohort, &rows, fake_digest("synthetic-input"))?;
    if report.pairwise_comparisons[0].cluster_bootstrap_95_percentile_lower
        != again.pairwise_comparisons[0].cluster_bootstrap_95_percentile_lower
        || report.pairwise_comparisons[0].cluster_bootstrap_95_percentile_upper
            != again.pairwise_comparisons[0].cluster_bootstrap_95_percentile_upper
    {
        return Err("cluster bootstrap is not deterministic".into());
    }
    println!("SELF_TEST_PASS repository-cluster bootstrap is deterministic");

    Ok(())
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    if args.len() != 5 || args[1] != "--compare" {
        return Err(format!(
            "usage: {} --compare <cohort.json> <run-results.jsonl> <report.json> OR --self-test",
            args.first().map(String::as_str).unwrap_or("pitb-matched-benchmark-comparison")
        ));
    }
    Ok((PathBuf::from(&args[2]), PathBuf::from(&args[3]), PathBuf::from(&args[4])))
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_MATCHED_BENCHMARK_COMPARISON_SELF_TEST_PASS");
        return Ok(());
    }
    let (cohort_path, results_path, report_path) = parse_args(&args)?;
    let cohort_bytes = fs::read(&cohort_path).map_err(|e| format!("read cohort manifest: {e}"))?;
    let results_bytes = fs::read(&results_path).map_err(|e| format!("read run results: {e}"))?;
    let cohort: Cohort = serde_json::from_slice(&cohort_bytes)
        .map_err(|e| format!("parse cohort manifest: {e}"))?;
    let rows = load_jsonl(&results_bytes)?;
    let report = analyze(&cohort, &rows, sha256(&results_bytes))?;
    let output = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize comparison report: {e}"))?;
    fs::write(&report_path, output).map_err(|e| format!("write comparison report: {e}"))?;
    println!("PITB_MATCHED_BENCHMARK_COMPARISON_OK");
    println!(
        "benchmark={} tasks={} clusters={} arms={} pairs={}",
        report.benchmark_id,
        report.observed_task_count,
        report.observed_cluster_count,
        report.arm_count,
        report.pairwise_comparisons.len()
    );
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
    fn registered_matched_comparison_controls_pass() {
        run_self_tests().expect("all matched benchmark comparison controls must pass");
    }

    #[test]
    fn assurance_requirement_switch_changes_only_qualification() {
        let mut cohort = synthetic_cohort();
        let rows = synthetic_rows(&cohort);
        let gated = analyze(&cohort, &rows, fake_digest("gated")).unwrap();
        cohort.require_specification_audit = false;
        let ungated = analyze(&cohort, &rows, fake_digest("ungated")).unwrap();
        let gated_arm = gated.arm_metrics.iter().find(|x| x.arm_id == "challenger").unwrap();
        let ungated_arm = ungated.arm_metrics.iter().find(|x| x.arm_id == "challenger").unwrap();
        assert_eq!(gated_arm.raw_kernel_verified_successes, ungated_arm.raw_kernel_verified_successes);
        assert_eq!(gated_arm.assurance_qualified_successes + 1, ungated_arm.assurance_qualified_successes);
    }

    #[test]
    fn source_reproduction_metric_never_claims_unperformed_checks() {
        let mut cohort = synthetic_cohort();
        cohort.require_source_reproduction = false;
        let mut rows = synthetic_rows(&cohort);
        for row in &mut rows {
            row.source_reproduction_status = "not_applicable".into();
            row.source_reproduction_receipt_sha256 = None;
        }
        let report = analyze(&cohort, &rows, fake_digest("source-not-required")).unwrap();
        for arm in &report.arm_metrics {
            assert_eq!(arm.source_reproduced_successes, 0);
            assert_eq!(arm.source_reproduced_success_rate, 0.0);
        }
        let challenger = report.arm_metrics.iter().find(|x| x.arm_id == "challenger").unwrap();
        assert_eq!(challenger.assurance_qualified_successes, 3);
    }

    #[test]
    fn goal_mismatch_always_blocks_assurance_qualified_success() {
        let mut cohort = synthetic_cohort();
        cohort.require_source_reproduction = false;
        let mut rows = synthetic_rows(&cohort);
        let row = rows.iter_mut().find(|r| r.arm_id == "challenger" && r.task_id == "task-1").unwrap();
        row.goal_alignment = "mismatch".into();
        row.goal_alignment_receipt_sha256 = Some(fake_digest("mismatch"));
        row.source_reproduction_status = "not_applicable".into();
        row.source_reproduction_receipt_sha256 = None;
        let report = analyze(&cohort, &rows, fake_digest("goal-mismatch")).unwrap();
        let challenger = report.arm_metrics.iter().find(|x| x.arm_id == "challenger").unwrap();
        assert_eq!(challenger.raw_kernel_verified_successes, 4);
        assert_eq!(challenger.assurance_qualified_successes, 2);
        assert_eq!(challenger.assurance_disqualified_successes, 1);
    }
}
