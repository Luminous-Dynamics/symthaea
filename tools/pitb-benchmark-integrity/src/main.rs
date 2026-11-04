use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const COHORT_SCHEMA: &str = "pitb.sorrydb-cohort-manifest.v0";
const RECEIPT_SCHEMA: &str = "pitb.proof-replacement-receipt.v0";
const REPORT_SCHEMA: &str = "pitb.benchmark-integrity-report.v0";
const SORRYDB_BENCHMARK: &str = "SorryDB_2601_1000_evaluation_split";
const SORRYDB_SOURCE_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const SORRYDB_SOURCE_BLOB: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const SORRYDB_SOURCE_SHA256: &str = "26bfd11099f6b8db7f80d49bdea5d9f8ea0d23ba65d44d93f67ed42ba1530062";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CohortManifest {
    schema_version: String,
    benchmark_id: String,
    provenance: Provenance,
    statistics: DeclaredStatistics,
    tasks: Vec<Task>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    repository: String,
    repository_commit: String,
    source_path: String,
    source_blob_sha1: String,
    source_file_bytes: usize,
    source_file_sha256: String,
    source_license: String,
    paper: String,
    database_spec: String,
    snapshot_description: String,
    task_goal_text_distributed: bool,
    blame_email_hash_distributed: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DeclaredStatistics {
    task_count: usize,
    distinct_repository_count: usize,
    distinct_repository_commit_lean_toolchain_count: usize,
    unique_task_id_count: usize,
    unique_goal_hash_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Task {
    task_id: String,
    repository_url: String,
    repository_branch: String,
    repository_commit: String,
    lean_toolchain: String,
    source_path: String,
    start_line: u32,
    start_column: u32,
    end_line: u32,
    end_column: u32,
    goal_sha256: String,
    source_url: String,
}

#[derive(Debug, Clone, Serialize)]
struct CohortReport {
    schema_version: String,
    benchmark_id: String,
    cohort_manifest_sha256: String,
    source_repository_commit: String,
    source_blob_sha1: String,
    source_file_sha256: String,
    task_count: usize,
    distinct_repository_count: usize,
    distinct_repository_commit_lean_toolchain_count: usize,
    unique_task_id_count: usize,
    unique_goal_hash_count: usize,
    lean_toolchain_distribution: BTreeMap<String, usize>,
    integrity_scope: String,
    proof_execution_scope: String,
    trust_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProofReceipt {
    schema_version: String,
    cohort_manifest_sha256: String,
    task_id: String,
    repository_url: String,
    repository_commit: String,
    lean_toolchain: String,
    source_path: String,
    start_line: u32,
    start_column: u32,
    goal_sha256: String,
    environment_sha256: String,
    lake_manifest_sha256: String,
    source_file_sha256: String,
    baseline_tree_sha256: String,
    final_tree_sha256: String,
    patch_sha256: String,
    proof_sha256: String,
    declaration_signature_before_sha256: String,
    declaration_signature_after_sha256: String,
    actual_axioms_sha256: String,
    axiom_policy_sha256: String,
    execution_receipt_sha256: String,
    clean_checkout: bool,
    baseline_reproduction_passed: bool,
    debug_goal_matched: bool,
    target_is_prop: bool,
    target_replaced_exactly_once: bool,
    lake_build_exit_code: i32,
    kernel_replay_passed: bool,
    permitted_axioms_passed: bool,
    target_sorry_count: u32,
    semantic_review: SemanticReview,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SemanticReview {
    source_correspondence: String,
    hypothesis_fidelity: String,
    non_vacuity: String,
    reviewer_attestation_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ProofResult {
    task_id: String,
    qualification_tier: String,
    semantic_review_claimed: bool,
    proof_sha256: String,
    execution_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProofSummary {
    receipt_count: usize,
    kernel_verified_count: usize,
    semantic_review_claim_count: usize,
    kernel_verified_rate_of_cohort: f64,
    semantic_review_claim_rate_of_cohort: f64,
    results: Vec<ProofResult>,
    interpretation_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct IntegrityReport {
    schema_version: String,
    benchmark_id: String,
    cohort_manifest_sha256: String,
    source_repository_commit: String,
    source_blob_sha1: String,
    source_file_sha256: String,
    task_count: usize,
    distinct_repository_count: usize,
    distinct_repository_commit_lean_toolchain_count: usize,
    unique_task_id_count: usize,
    unique_goal_hash_count: usize,
    lean_toolchain_distribution: BTreeMap<String, usize>,
    proof_summary: Option<ProofSummary>,
    integrity_scope: String,
    proof_execution_scope: String,
    trust_boundary: String,
}

#[derive(Debug, Clone)]
struct LoadedCohort {
    cohort: CohortManifest,
    manifest_sha256: String,
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
        Err(format!("{label} must be 64 lowercase hexadecimal characters"))
    }
}

fn commit(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 40) {
        Ok(())
    } else {
        Err(format!("{label} must be 40 lowercase hexadecimal characters"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} cannot be empty"))
    } else {
        Ok(())
    }
}

fn valid_lean_toolchain(value: &str) -> bool {
    let rest = match value.strip_prefix('v') {
        Some(v) => v,
        None => return false,
    };
    let version = rest.split('-').next().unwrap_or_default();
    let parts: Vec<&str> = version.split('.').collect();
    parts.len() == 3 && parts.iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

fn valid_repository_url(value: &str) -> bool {
    let rest = match value.strip_prefix("https://github.com/") {
        Some(v) => v,
        None => return false,
    };
    let parts: Vec<&str> = rest.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| !p.is_empty())
        && !rest.contains(char::is_whitespace)
        && !value.ends_with(".git")
}

fn valid_source_path(value: &str) -> bool {
    if value.is_empty() || value.starts_with('/') || !value.ends_with(".lean") {
        return false;
    }
    value.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

fn validate_task(task: &Task, index: usize) -> Result<(), String> {
    let label = format!("task {}", index + 1);
    digest(&task.task_id, &format!("{label}.task_id"))?;
    digest(&task.goal_sha256, &format!("{label}.goal_sha256"))?;
    non_empty(&task.repository_branch, &format!("{label}.repository_branch"))?;
    if !valid_repository_url(&task.repository_url) {
        return Err(format!("{label}.repository_url is not an HTTPS GitHub owner/repository URL"));
    }
    commit(&task.repository_commit, &format!("{label}.repository_commit"))?;
    if !valid_lean_toolchain(&task.lean_toolchain) {
        return Err(format!("{label}.lean_toolchain is not a version-tag such as v4.26.0"));
    }
    if !valid_source_path(&task.source_path) {
        return Err(format!("{label}.source_path must be a normalized relative .lean path"));
    }
    if task.start_line == 0 || task.end_line < task.start_line {
        return Err(format!("{label} has invalid source line coordinates"));
    }
    if task.end_line == task.start_line && task.end_column < task.start_column {
        return Err(format!("{label} has an end column before its start column"));
    }
    let expected_url = format!(
        "{}/blob/{}/{ }#L{}",
        task.repository_url, task.repository_commit, task.source_path, task.start_line
    );
    if task.source_url != expected_url {
        return Err(format!("{label}.source_url does not match the pinned repository commit and location"));
    }
    Ok(())
}

fn validate_cohort(cohort: &CohortManifest, manifest_sha256: &str) -> Result<CohortReport, String> {
    if cohort.schema_version != COHORT_SCHEMA {
        return Err("unsupported cohort schema_version".into());
    }
    non_empty(&cohort.benchmark_id, "benchmark_id")?;
    let provenance = &cohort.provenance;
    if provenance.repository != "SorryDB/SorryDB" {
        return Err("SorryDB cohort must cite the SorryDB/SorryDB source repository".into());
    }
    commit(&provenance.repository_commit, "source repository commit")?;
    non_empty(&provenance.source_path, "source_path")?;
    commit(&provenance.source_blob_sha1, "source_blob_sha1")?;
    digest(&provenance.source_file_sha256, "source_file_sha256")?;
    if provenance.source_file_bytes == 0 {
        return Err("source_file_bytes must be positive".into());
    }
    if !provenance.source_license.contains("Apache-2.0") {
        return Err("SorryDB repository license attribution is missing".into());
    }
    if provenance.task_goal_text_distributed || provenance.blame_email_hash_distributed {
        return Err("cohort manifest must not redistribute goal text or blame-email hashes".into());
    }
    if cohort.benchmark_id == SORRYDB_BENCHMARK {
        if provenance.repository_commit != SORRYDB_SOURCE_COMMIT
            || provenance.source_blob_sha1 != SORRYDB_SOURCE_BLOB
            || provenance.source_file_sha256 != SORRYDB_SOURCE_SHA256
        {
            return Err("SorryDB_2601 evaluation split source identity changed".into());
        }
    }
    if cohort.tasks.is_empty() {
        return Err("cohort cannot be empty".into());
    }
    let mut task_ids = BTreeSet::new();
    let mut goal_hashes = BTreeSet::new();
    let mut task_locations = BTreeSet::new();
    let mut repositories = BTreeSet::new();
    let mut repo_commit_toolchains = BTreeSet::new();
    let mut toolchain_counts = BTreeMap::<String, usize>::new();

    for (index, task) in cohort.tasks.iter().enumerate() {
        validate_task(task, index)?;
        if !task_ids.insert(task.task_id.clone()) {
            return Err(format!("duplicate task_id {}", task.task_id));
        }
        if !goal_hashes.insert(task.goal_sha256.clone()) {
            return Err(format!("duplicate goal hash {}", task.goal_sha256));
        }
        let location_key = format!(
            "{}@{}@{}:{}:{}-{}:{}",
            task.repository_url, task.repository_commit, task.source_path,
            task.start_line, task.start_column, task.end_line, task.end_column
        );
        if !task_locations.insert(location_key) {
            return Err(format!("duplicate source location in cohort at row {}", index + 1));
        }
        repositories.insert(task.repository_url.clone());
        repo_commit_toolchains.insert(format!(
            "{}@{}@{}", task.repository_url, task.repository_commit, task.lean_toolchain
        ));
        *toolchain_counts.entry(task.lean_toolchain.clone()).or_default() += 1;
    }

    let actual = DeclaredStatistics {
        task_count: cohort.tasks.len(),
        distinct_repository_count: repositories.len(),
        distinct_repository_commit_lean_toolchain_count: repo_commit_toolchains.len(),
        unique_task_id_count: task_ids.len(),
        unique_goal_hash_count: goal_hashes.len(),
    };
    if serde_json::to_value(&actual).map_err(|e| e.to_string())?
        != serde_json::to_value(&cohort.statistics).map_err(|e| e.to_string())?
    {
        return Err("declared cohort statistics do not match actual task records".into());
    }
    if cohort.benchmark_id == SORRYDB_BENCHMARK
        && (actual.task_count != 1000
            || actual.distinct_repository_count != 78
            || actual.distinct_repository_commit_lean_toolchain_count != 107)
    {
        return Err("pinned SorryDB_2601 cohort cardinalities changed".into());
    }

    Ok(CohortReport {
        schema_version: REPORT_SCHEMA.into(),
        benchmark_id: cohort.benchmark_id.clone(),
        cohort_manifest_sha256: manifest_sha256.to_owned(),
        source_repository_commit: provenance.repository_commit.clone(),
        source_blob_sha1: provenance.source_blob_sha1.clone(),
        source_file_sha256: provenance.source_file_sha256.clone(),
        task_count: actual.task_count,
        distinct_repository_count: actual.distinct_repository_count,
        distinct_repository_commit_lean_toolchain_count: actual.distinct_repository_commit_lean_toolchain_count,
        unique_task_id_count: actual.unique_task_id_count,
        unique_goal_hash_count: actual.unique_goal_hash_count,
        lean_toolchain_distribution: toolchain_counts,
        integrity_scope: "Cohort identity, provenance, task cardinality, source coordinates and goal digests are structurally validated against a frozen SorryDB source blob.".into(),
        proof_execution_scope: "No source repository has been cloned or rebuilt by this cohort-only validation.".into(),
        trust_boundary: "The task index intentionally omits goal text. Goal hashes are bound to the pinned SorryDB source blob but are not independently re-extracted from each upstream Lean repository by this validator.".into(),
    })
}

fn receipt_semantic_claim(receipt: &ProofReceipt) -> Result<bool, String> {
    for (label, value) in [
        ("source_correspondence", receipt.semantic_review.source_correspondence.as_str()),
        ("hypothesis_fidelity", receipt.semantic_review.hypothesis_fidelity.as_str()),
        ("non_vacuity", receipt.semantic_review.non_vacuity.as_str()),
    ] {
        if !matches!(value, "pass" | "fail" | "not_run" | "inconclusive") {
            return Err(format!("semantic_review.{label} has invalid status {value}"));
        }
    }
    if let Some(attestation) = &receipt.semantic_review.reviewer_attestation_sha256 {
        digest(attestation, "semantic_review.reviewer_attestation_sha256")?;
    }
    let all_pass = receipt.semantic_review.source_correspondence == "pass"
        && receipt.semantic_review.hypothesis_fidelity == "pass"
        && receipt.semantic_review.non_vacuity == "pass";
    Ok(all_pass && receipt.semantic_review.reviewer_attestation_sha256.is_some())
}

fn validate_proof_receipt(
    receipt: &ProofReceipt,
    cohort: &CohortManifest,
    cohort_sha256: &str,
    task_map: &BTreeMap<&str, &Task>,
    index: usize,
) -> Result<ProofResult, String> {
    let label = format!("proof receipt {}", index + 1);
    if receipt.schema_version != RECEIPT_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    digest(&receipt.cohort_manifest_sha256, &format!("{label}.cohort_manifest_sha256"))?;
    if receipt.cohort_manifest_sha256 != cohort_sha256 {
        return Err(format!("{label} is bound to a different cohort manifest"));
    }
    let task = task_map.get(receipt.task_id.as_str())
        .ok_or_else(|| format!("{label} references unknown task {}", receipt.task_id))?;
    for (field, actual, expected) in [
        ("repository_url", receipt.repository_url.as_str(), task.repository_url.as_str()),
        ("repository_commit", receipt.repository_commit.as_str(), task.repository_commit.as_str()),
        ("lean_toolchain", receipt.lean_toolchain.as_str(), task.lean_toolchain.as_str()),
        ("source_path", receipt.source_path.as_str(), task.source_path.as_str()),
        ("goal_sha256", receipt.goal_sha256.as_str(), task.goal_sha256.as_str()),
    ] {
        if actual != expected {
            return Err(format!("{label}.{field} differs from the pinned cohort task"));
        }
    }
    if receipt.start_line != task.start_line || receipt.start_column != task.start_column {
        return Err(format!("{label} source coordinates differ from the pinned cohort task"));
    }
    for (field, value) in [
        ("environment_sha256", &receipt.environment_sha256),
        ("lake_manifest_sha256", &receipt.lake_manifest_sha256),
        ("source_file_sha256", &receipt.source_file_sha256),
        ("baseline_tree_sha256", &receipt.baseline_tree_sha256),
        ("final_tree_sha256", &receipt.final_tree_sha256),
        ("patch_sha256", &receipt.patch_sha256),
        ("proof_sha256", &receipt.proof_sha256),
        ("declaration_signature_before_sha256", &receipt.declaration_signature_before_sha256),
        ("declaration_signature_after_sha256", &receipt.declaration_signature_after_sha256),
        ("actual_axioms_sha256", &receipt.actual_axioms_sha256),
        ("axiom_policy_sha256", &receipt.axiom_policy_sha256),
        ("execution_receipt_sha256", &receipt.execution_receipt_sha256),
    ] {
        digest(value, &format!("{label}.{field}"))?;
    }
    if receipt.declaration_signature_before_sha256 != receipt.declaration_signature_after_sha256 {
        return Err(format!("{label} changes the target declaration signature"));
    }
    if !receipt.clean_checkout
        || !receipt.baseline_reproduction_passed
        || !receipt.debug_goal_matched
        || !receipt.target_is_prop
        || !receipt.target_replaced_exactly_once
        || receipt.lake_build_exit_code != 0
        || !receipt.kernel_replay_passed
        || !receipt.permitted_axioms_passed
        || receipt.target_sorry_count != 0
    {
        return Err(format!("{label} does not satisfy all kernel-verification gates"));
    }
    let semantic_review_claimed = receipt_semantic_claim(receipt)?;
    Ok(ProofResult {
        task_id: receipt.task_id.clone(),
        qualification_tier: if semantic_review_claimed {
            "kernel_verified_with_semantic_review_claim".into()
        } else {
            "kernel_verified_semantic_review_unqualified".into()
        },
        semantic_review_claimed,
        proof_sha256: receipt.proof_sha256.clone(),
        execution_receipt_sha256: receipt.execution_receipt_sha256.clone(),
    })
}

fn load_cohort(path: &PathBuf) -> Result<LoadedCohort, String> {
    let bytes = fs::read(path).map_err(|e| format!("read cohort manifest: {e}"))?;
    let cohort: CohortManifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse cohort manifest: {e}"))?;
    Ok(LoadedCohort {
        cohort,
        manifest_sha256: sha256(&bytes),
    })
}

fn load_receipts(path: &PathBuf) -> Result<Vec<ProofReceipt>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read proof receipts: {e}"))?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("proof receipts are not UTF-8: {e}"))?;
    let mut receipts = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        receipts.push(serde_json::from_str(line)
            .map_err(|e| format!("proof receipt line {}: {e}", index + 1))?);
    }
    Ok(receipts)
}

fn build_integrity_report(
    cohort_report: CohortReport,
    proof_summary: Option<ProofSummary>,
) -> IntegrityReport {
    IntegrityReport {
        schema_version: REPORT_SCHEMA.into(),
        benchmark_id: cohort_report.benchmark_id,
        cohort_manifest_sha256: cohort_report.cohort_manifest_sha256,
        source_repository_commit: cohort_report.source_repository_commit,
        source_blob_sha1: cohort_report.source_blob_sha1,
        source_file_sha256: cohort_report.source_file_sha256,
        task_count: cohort_report.task_count,
        distinct_repository_count: cohort_report.distinct_repository_count,
        distinct_repository_commit_lean_toolchain_count: cohort_report.distinct_repository_commit_lean_toolchain_count,
        unique_task_id_count: cohort_report.unique_task_id_count,
        unique_goal_hash_count: cohort_report.unique_goal_hash_count,
        lean_toolchain_distribution: cohort_report.lean_toolchain_distribution,
        proof_summary,
        integrity_scope: cohort_report.integrity_scope,
        proof_execution_scope: cohort_report.proof_execution_scope,
        trust_boundary: cohort_report.trust_boundary,
    }
}

fn verify_cohort_file(manifest_path: &PathBuf) -> Result<IntegrityReport, String> {
    let loaded = load_cohort(manifest_path)?;
    let report = validate_cohort(&loaded.cohort, &loaded.manifest_sha256)?;
    Ok(build_integrity_report(report, None))
}

fn verify_proof_file(
    manifest_path: &PathBuf,
    receipts_path: &PathBuf,
) -> Result<IntegrityReport, String> {
    let loaded = load_cohort(manifest_path)?;
    let cohort_report = validate_cohort(&loaded.cohort, &loaded.manifest_sha256)?;
    let receipts = load_receipts(receipts_path)?;
    let task_map: BTreeMap<&str, &Task> = loaded.cohort.tasks.iter()
        .map(|task| (task.task_id.as_str(), task))
        .collect();
    let mut seen = BTreeSet::new();
    let mut results = Vec::new();
    for (index, receipt) in receipts.iter().enumerate() {
        if !seen.insert(receipt.task_id.clone()) {
            return Err(format!("duplicate proof receipt for task {}", receipt.task_id));
        }
        results.push(validate_proof_receipt(
            receipt,
            &loaded.cohort,
            &loaded.manifest_sha256,
            &task_map,
            index,
        )?);
    }
    let kernel_count = results.len();
    let semantic_count = results.iter().filter(|x| x.semantic_review_claimed).count();
    let denom = loaded.cohort.tasks.len() as f64;
    let summary = ProofSummary {
        receipt_count: results.len(),
        kernel_verified_count: kernel_count,
        semantic_review_claim_count: semantic_count,
        kernel_verified_rate_of_cohort: if denom == 0.0 { 0.0 } else { kernel_count as f64 / denom },
        semantic_review_claim_rate_of_cohort: if denom == 0.0 { 0.0 } else { semantic_count as f64 / denom },
        results,
        interpretation_boundary: "Kernel qualification requires exact cohort binding, original source/toolchain identity, baseline reproduction, unchanged declaration signature, target Prop status, one exact hole replacement, successful build, kernel replay and permitted axioms. Semantic-review claims are not cryptographically authenticated by this tool.".into(),
    };
    Ok(build_integrity_report(cohort_report, Some(summary)))
}

fn synthetic_cohort() -> CohortManifest {
    let task_a = Task {
        task_id: sha256(b"synthetic-task-a"),
        repository_url: "https://github.com/example/project".into(),
        repository_branch: "main".into(),
        repository_commit: "1".repeat(40),
        lean_toolchain: "v4.26.0".into(),
        source_path: "Demo/A.lean".into(),
        start_line: 10,
        start_column: 2,
        end_line: 10,
        end_column: 7,
        goal_sha256: sha256(b"goal-a"),
        source_url: "https://github.com/example/project/blob/1111111111111111111111111111111111111111/Demo/A.lean#L10".into(),
    };
    let task_b = Task {
        task_id: sha256(b"synthetic-task-b"),
        repository_branch: "feature/b".into(),
        repository_url: "https://github.com/example/project".into(),
        repository_commit: "2".repeat(40),
        lean_toolchain: "v4.27.0-rc1".into(),
        source_path: "Demo/B.lean".into(),
        start_line: 22,
        start_column: 0,
        end_line: 22,
        end_column: 5,
        goal_sha256: sha256(b"goal-b"),
        source_url: "https://github.com/example/project/blob/2222222222222222222222222222222222222222/Demo/B.lean#L22".into(),
    };
    CohortManifest {
        schema_version: COHORT_SCHEMA.into(),
        benchmark_id: "synthetic-sorrydb-cohort".into(),
        provenance: Provenance {
            repository: "SorryDB/SorryDB".into(),
            repository_commit: "3".repeat(40),
            source_path: "fixtures/sorrydb.json".into(),
            source_blob_sha1: "4".repeat(40),
            source_file_bytes: 42,
            source_file_sha256: sha256(b"synthetic source bytes"),
            source_license: "Apache-2.0".into(),
            paper: "https://proceedings.mlr.press/v306/letson26a.html".into(),
            database_spec: "https://github.com/SorryDB/SorryDB/blob/master/doc/DATABASE.md".into(),
            snapshot_description: "Synthetic negative and positive controls".into(),
            task_goal_text_distributed: false,
            blame_email_hash_distributed: false,
        },
        statistics: DeclaredStatistics {
            task_count: 2,
            distinct_repository_count: 1,
            distinct_repository_commit_lean_toolchain_count: 2,
            unique_task_id_count: 2,
            unique_goal_hash_count: 2,
        },
        tasks: vec![task_a, task_b],
    }
}

fn synthetic_receipt(task: &Task, cohort_sha256: &str) -> ProofReceipt {
    let statement = sha256(b"declaration signature");
    ProofReceipt {
        schema_version: RECEIPT_SCHEMA.into(),
        cohort_manifest_sha256: cohort_sha256.into(),
        task_id: task.task_id.clone(),
        repository_url: task.repository_url.clone(),
        repository_commit: task.repository_commit.clone(),
        lean_toolchain: task.lean_toolchain.clone(),
        source_path: task.source_path.clone(),
        start_line: task.start_line,
        start_column: task.start_column,
        goal_sha256: task.goal_sha256.clone(),
        environment_sha256: sha256(b"environment"),
        lake_manifest_sha256: sha256(b"lake manifest"),
        source_file_sha256: sha256(b"source file"),
        baseline_tree_sha256: sha256(b"baseline tree"),
        final_tree_sha256: sha256(b"final tree"),
        patch_sha256: sha256(b"patch"),
        proof_sha256: sha256(b"proof"),
        declaration_signature_before_sha256: statement.clone(),
        declaration_signature_after_sha256: statement,
        actual_axioms_sha256: sha256(b"axioms"),
        axiom_policy_sha256: sha256(b"axiom policy"),
        execution_receipt_sha256: sha256(b"execution receipt"),
        clean_checkout: true,
        baseline_reproduction_passed: true,
        debug_goal_matched: true,
        target_is_prop: true,
        target_replaced_exactly_once: true,
        lake_build_exit_code: 0,
        kernel_replay_passed: true,
        permitted_axioms_passed: true,
        target_sorry_count: 0,
        semantic_review: SemanticReview {
            source_correspondence: "not_run".into(),
            hypothesis_fidelity: "not_run".into(),
            non_vacuity: "not_run".into(),
            reviewer_attestation_sha256: None,
        },
    }
}

fn run_self_tests() -> Result<(), String> {
    let cohort = synthetic_cohort();
    let cohort_bytes = serde_json::to_vec(&cohort).map_err(|e| e.to_string())?;
    let cohort_sha = sha256(&cohort_bytes);
    let report = validate_cohort(&cohort, &cohort_sha)?;
    if report.task_count != 2 || report.distinct_repository_count != 1 {
        return Err("synthetic cohort cardinality control failed".into());
    }
    println!("SELF_TEST_PASS synthetic cohort manifest and identity counts");

    let task_map: BTreeMap<&str, &Task> = cohort.tasks.iter().map(|task| (task.task_id.as_str(), task)).collect();
    let mut receipt = synthetic_receipt(&cohort.tasks[0], &cohort_sha);
    let result = validate_proof_receipt(&receipt, &cohort, &cohort_sha, &task_map, 0)?;
    if result.qualification_tier != "kernel_verified_semantic_review_unqualified" {
        return Err("kernel-only receipt was given semantic qualification".into());
    }
    println!("SELF_TEST_PASS kernel verification does not imply semantic review");

    receipt.semantic_review.source_correspondence = "pass".into();
    receipt.semantic_review.hypothesis_fidelity = "pass".into();
    receipt.semantic_review.non_vacuity = "pass".into();
    receipt.semantic_review.reviewer_attestation_sha256 = Some(sha256(b"human review"));
    let result = validate_proof_receipt(&receipt, &cohort, &cohort_sha, &task_map, 0)?;
    if !result.semantic_review_claimed {
        return Err("complete semantic review claim was not recorded".into());
    }
    println!("SELF_TEST_PASS semantic review is separately and explicitly claimed");

    let mut bad = receipt.clone();
    bad.repository_commit = "9".repeat(40);
    if validate_proof_receipt(&bad, &cohort, &cohort_sha, &task_map, 0).is_ok() {
        return Err("proof receipt with changed repository commit was accepted".into());
    }
    println!("SELF_TEST_PASS proof receipt cannot drift from source commit");

    let mut bad = receipt.clone();
    bad.goal_sha256 = sha256(b"changed goal");
    if validate_proof_receipt(&bad, &cohort, &cohort_sha, &task_map, 0).is_ok() {
        return Err("proof receipt with changed target goal was accepted".into());
    }
    println!("SELF_TEST_PASS proof receipt cannot change target goal");

    let mut bad = receipt.clone();
    bad.declaration_signature_after_sha256 = sha256(b"changed theorem signature");
    if validate_proof_receipt(&bad, &cohort, &cohort_sha, &task_map, 0).is_ok() {
        return Err("proof receipt that changes declaration signature was accepted".into());
    }
    println!("SELF_TEST_PASS target declaration signature must be preserved");

    let mut bad = receipt.clone();
    bad.kernel_replay_passed = false;
    if validate_proof_receipt(&bad, &cohort, &cohort_sha, &task_map, 0).is_ok() {
        return Err("proof receipt without kernel replay was accepted".into());
    }
    println!("SELF_TEST_PASS build/verification failures cannot be scored as proof completion");

    let mut bad_cohort = cohort.clone();
    bad_cohort.tasks[1].task_id = bad_cohort.tasks[0].task_id.clone();
    if validate_cohort(&bad_cohort, &cohort_sha).is_ok() {
        return Err("duplicate task ID was accepted".into());
    }
    println!("SELF_TEST_PASS duplicate task identities are rejected");

    let mut bad_cohort = cohort.clone();
    bad_cohort.tasks[0].source_path = "../escape.lean".into();
    if validate_cohort(&bad_cohort, &cohort_sha).is_ok() {
        return Err("path traversal source locator was accepted".into());
    }
    println!("SELF_TEST_PASS unsafe source paths are rejected");
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_BENCHMARK_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    match args.get(1).map(String::as_str) {
        Some("--verify-cohort") if args.len() == 4 => {
            let report = verify_cohort_file(&PathBuf::from(&args[2]))?;
            let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
            fs::write(&args[3], bytes).map_err(|e| format!("write integrity report: {e}"))?;
            println!("PITB_COHORT_INTEGRITY_OK tasks={}", report.task_count);
        }
        Some("--verify-proofs") if args.len() == 5 => {
            let report = verify_proof_file(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]))?;
            let count = report.proof_summary.as_ref().map(|x| x.kernel_verified_count).unwrap_or(0);
            let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
            fs::write(&args[4], bytes).map_err(|e| format!("write proof report: {e}"))?;
            println!("PITB_PROOF_RECEIPTS_OK kernel_verified={count}");
        }
        _ => {
            return Err(format!(
                "usage: {} --verify-cohort <cohort.json> <report.json> | --verify-proofs <cohort.json> <proof-receipts.jsonl> <report.json> | --self-test",
                args.first().map(String::as_str).unwrap_or("pitb-benchmark-integrity")
            ));
        }
    }
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
    fn benchmark_integrity_contract_controls_pass() {
        run_self_tests().expect("benchmark integrity controls must pass");
    }

    #[test]
    fn duplicate_task_receipts_are_not_accepted() {
        let cohort = synthetic_cohort();
        let bytes = serde_json::to_vec(&cohort).unwrap();
        let cohort_sha = sha256(&bytes);
        let map: BTreeMap<&str, &Task> = cohort.tasks.iter().map(|task| (task.task_id.as_str(), task)).collect();
        let first = synthetic_receipt(&cohort.tasks[0], &cohort_sha);
        let duplicate = first.clone();
        let results = [first, duplicate];
        let mut seen = BTreeSet::new();
        assert!(results.iter().all(|receipt| seen.insert(receipt.task_id.clone())) == false);
        assert!(validate_proof_receipt(&results[0], &cohort, &cohort_sha, &map, 0).is_ok());
    }
}
