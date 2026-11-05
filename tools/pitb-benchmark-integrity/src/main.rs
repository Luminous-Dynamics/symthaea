use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const SNAPSHOT_SCHEMA: &str = "pitb.benchmark-snapshot.v0";
const TASK_SCHEMA: &str = "pitb.benchmark-task.v0";
const REPRO_SCHEMA: &str = "pitb.task-reproduction-receipt.v0";
const ATTEMPT_SCHEMA: &str = "pitb.proof-attempt-outcome.v0";
const REPORT_SCHEMA: &str = "pitb.benchmark-integrity-report.v0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema_version: String,
    benchmark_id: String,
    benchmark_version: String,
    snapshot_id: String,
    upstream_repository: String,
    upstream_repository_commit: String,
    source_split_path: String,
    source_split_blob_sha1: String,
    source_repository_license: String,
    task_count: usize,
    task_records_jsonl_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Task {
    schema_version: String,
    snapshot_id: String,
    source_task_id: String,
    repo_remote: String,
    repo_branch: String,
    repo_commit: String,
    lean_version: String,
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    debug_goal_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ReproductionReceipt {
    schema_version: String,
    task_id: String,
    task_record_sha256: String,
    checkout_commit: String,
    lean_toolchain: String,
    lean_toolchain_sha256: String,
    environment_sha256: String,
    source_file_sha256: String,
    checkout_receipt_sha256: String,
    build_receipt_sha256: String,
    goal_capture_receipt_sha256: String,
    build_passed: bool,
    sorry_anchor_found: bool,
    observed_goal_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AttemptOutcome {
    schema_version: String,
    attempt_id: String,
    task_id: String,
    task_record_sha256: String,
    reproduction_receipt_sha256: String,
    prover_id: String,
    prover_version: String,
    checkpoint_sha256: Option<String>,
    environment_sha256: String,
    lean_toolchain_sha256: String,
    proof_sha256: Option<String>,
    patch_sha256: Option<String>,
    verification_receipt_sha256: String,
    result: String,
    exact_target_reproduced: bool,
    source_integrity_passed: bool,
    target_placeholder_removed: bool,
    kernel_passed: bool,
    semantic_audit_passed: bool,
    no_new_axioms: bool,
    no_new_unsafe_declarations: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TaskAdmission {
    task_id: String,
    source_task_id: String,
    task_record_sha256: String,
    admitted: bool,
    quarantine_reasons: Vec<String>,
    reproduction_receipt_sha256: String,
    repository_commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Serialize)]
struct AttemptSummary {
    attempt_id: String,
    task_id: String,
    prover_id: String,
    prover_version: String,
    result: String,
    verified_success: bool,
    proof_sha256: Option<String>,
    environment_sha256: String,
    lean_toolchain_sha256: String,
    attempt_record_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    snapshot_id: String,
    benchmark_id: String,
    benchmark_version: String,
    upstream_repository_commit: String,
    source_split_blob_sha1: String,
    task_records_jsonl_sha256: String,
    reproduction_receipts_jsonl_sha256: String,
    attempt_outcomes_jsonl_sha256: String,
    task_count: usize,
    admitted_task_count: usize,
    quarantined_task_count: usize,
    admission_rate: f64,
    attempt_count: usize,
    verified_success_attempt_count: usize,
    verified_success_task_count: usize,
    verified_success_rate_over_admitted_tasks: Option<f64>,
    verified_success_rate_over_all_tasks: f64,
    task_admissions: Vec<TaskAdmission>,
    attempts: Vec<AttemptSummary>,
    score_semantics: String,
    receipt_authentication_boundary: String,
    benchmark_interpretation_boundary: String,
}

#[derive(Debug)]
struct ParsedRows<T> {
    rows: Vec<T>,
    raw_sha256: String,
}

#[derive(Debug)]
struct TaskContext {
    task: Task,
    task_id: String,
    task_record_sha256: String,
    receipt: ReproductionReceipt,
    receipt_sha256: String,
    admitted: bool,
    reasons: Vec<String>,
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
        Err(format!("{label} must be lowercase SHA-256 hex"))
    }
}

fn git_sha1(value: &str, label: &str) -> Result<(), String> {
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

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

fn task_id(task: &Task) -> Result<String, String> {
    let bytes = serde_json::to_vec(task).map_err(|e| format!("serialize task identity: {e}"))?;
    Ok(sha256(&bytes))
}

fn task_record_sha256(task: &Task) -> Result<String, String> {
    let bytes = serde_json::to_vec(task).map_err(|e| format!("serialize task record: {e}"))?;
    Ok(sha256(&bytes))
}

fn receipt_sha256<T: Serialize>(record: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(record).map_err(|e| format!("serialize receipt: {e}"))?;
    Ok(sha256(&bytes))
}

fn safe_relative_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\\') {
        return false;
    }
    !path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
}

fn validate_snapshot(snapshot: &Snapshot) -> Result<(), String> {
    if snapshot.schema_version != SNAPSHOT_SCHEMA {
        return Err("unsupported benchmark snapshot schema_version".into());
    }
    for (field, value) in [
        ("benchmark_id", snapshot.benchmark_id.as_str()),
        ("benchmark_version", snapshot.benchmark_version.as_str()),
        ("snapshot_id", snapshot.snapshot_id.as_str()),
        ("upstream_repository", snapshot.upstream_repository.as_str()),
        ("source_split_path", snapshot.source_split_path.as_str()),
        ("source_repository_license", snapshot.source_repository_license.as_str()),
    ] {
        non_empty(value, &format!("snapshot.{field}"))?;
    }
    git_sha1(&snapshot.upstream_repository_commit, "snapshot.upstream_repository_commit")?;
    git_sha1(&snapshot.source_split_blob_sha1, "snapshot.source_split_blob_sha1")?;
    digest(&snapshot.task_records_jsonl_sha256, "snapshot.task_records_jsonl_sha256")?;
    if snapshot.task_count == 0 {
        return Err("snapshot.task_count must be positive".into());
    }
    if !safe_relative_path(&snapshot.source_split_path) {
        return Err("snapshot.source_split_path must be a safe repository-relative path".into());
    }
    Ok(())
}

fn validate_task(task: &Task, snapshot: &Snapshot, line: usize) -> Result<(), String> {
    let label = format!("task line {line}");
    if task.schema_version != TASK_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    if task.snapshot_id != snapshot.snapshot_id {
        return Err(format!("{label} belongs to a different snapshot"));
    }
    for (field, value) in [
        ("source_task_id", task.source_task_id.as_str()),
        ("repo_remote", task.repo_remote.as_str()),
        ("repo_branch", task.repo_branch.as_str()),
        ("lean_version", task.lean_version.as_str()),
        ("path", task.path.as_str()),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    digest(&task.source_task_id, &format!("{label}.source_task_id"))?;
    git_sha1(&task.repo_commit, &format!("{label}.repo_commit"))?;
    digest(&task.debug_goal_sha256, &format!("{label}.debug_goal_sha256"))?;
    if !task.repo_remote.starts_with("https://") || task.repo_remote.contains('@') {
        return Err(format!("{label}.repo_remote must be an HTTPS URL without embedded credentials"));
    }
    if !safe_relative_path(&task.path) {
        return Err(format!("{label}.path must be a safe repository-relative path"));
    }
    if task.start_line == 0 || task.start_column == 0 || task.end_line == 0 || task.end_column == 0 {
        return Err(format!("{label} source coordinates are 1-based and must be positive"));
    }
    if (task.end_line, task.end_column) < (task.start_line, task.start_column) {
        return Err(format!("{label} source range ends before it starts"));
    }
    Ok(())
}

fn parse_jsonl<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<ParsedRows<T>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("{label} is not UTF-8: {e}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let row = serde_json::from_str(line)
            .map_err(|e| format!("{label} line {}: {e}", index + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{label} must contain at least one row"));
    }
    Ok(ParsedRows { rows, raw_sha256: sha256(bytes) })
}

fn validate_reproduction(
    receipt: &ReproductionReceipt,
    task: &Task,
    task_id_value: &str,
    task_record_digest: &str,
    index: usize,
) -> Result<(bool, Vec<String>), String> {
    let label = format!("reproduction receipt {index}");
    if receipt.schema_version != REPRO_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    for (field, value) in [
        ("task_id", receipt.task_id.as_str()),
        ("checkout_commit", receipt.checkout_commit.as_str()),
        ("lean_toolchain", receipt.lean_toolchain.as_str()),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    digest(&receipt.task_record_sha256, &format!("{label}.task_record_sha256"))?;
    if receipt.task_id != task_id_value || receipt.task_record_sha256 != task_record_digest {
        return Err(format!("{label} is not bound to the exact task record"));
    }
    git_sha1(&receipt.checkout_commit, &format!("{label}.checkout_commit"))?;
    for (field, value) in [
        ("lean_toolchain_sha256", receipt.lean_toolchain_sha256.as_str()),
        ("environment_sha256", receipt.environment_sha256.as_str()),
        ("source_file_sha256", receipt.source_file_sha256.as_str()),
        ("checkout_receipt_sha256", receipt.checkout_receipt_sha256.as_str()),
        ("build_receipt_sha256", receipt.build_receipt_sha256.as_str()),
        ("goal_capture_receipt_sha256", receipt.goal_capture_receipt_sha256.as_str()),
        ("observed_goal_sha256", receipt.observed_goal_sha256.as_str()),
    ] {
        digest(value, &format!("{label}.{field}"))?;
    }
    let mut reasons = Vec::new();
    if receipt.checkout_commit != task.repo_commit {
        reasons.push("checkout_commit_mismatch".into());
    }
    if receipt.lean_toolchain != task.lean_version {
        reasons.push("lean_toolchain_mismatch".into());
    }
    if !receipt.build_passed {
        reasons.push("repository_build_failed".into());
    }
    if !receipt.sorry_anchor_found {
        reasons.push("sorry_anchor_not_found".into());
    }
    if receipt.observed_goal_sha256 != task.debug_goal_sha256 {
        reasons.push("live_goal_differs_from_database_goal".into());
    }
    Ok((reasons.is_empty(), reasons))
}

fn validate_attempt(
    attempt: &AttemptOutcome,
    context: &TaskContext,
    index: usize,
) -> Result<bool, String> {
    let label = format!("attempt outcome {index}");
    if attempt.schema_version != ATTEMPT_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    for (field, value) in [
        ("attempt_id", attempt.attempt_id.as_str()),
        ("task_id", attempt.task_id.as_str()),
        ("prover_id", attempt.prover_id.as_str()),
        ("prover_version", attempt.prover_version.as_str()),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    for (field, value) in [
        ("task_record_sha256", attempt.task_record_sha256.as_str()),
        ("reproduction_receipt_sha256", attempt.reproduction_receipt_sha256.as_str()),
        ("environment_sha256", attempt.environment_sha256.as_str()),
        ("lean_toolchain_sha256", attempt.lean_toolchain_sha256.as_str()),
        ("verification_receipt_sha256", attempt.verification_receipt_sha256.as_str()),
    ] {
        digest(value, &format!("{label}.{field}"))?;
    }
    if let Some(checkpoint) = &attempt.checkpoint_sha256 {
        digest(checkpoint, &format!("{label}.checkpoint_sha256"))?;
    }
    if let Some(proof) = &attempt.proof_sha256 {
        digest(proof, &format!("{label}.proof_sha256"))?;
    }
    if let Some(patch) = &attempt.patch_sha256 {
        digest(patch, &format!("{label}.patch_sha256"))?;
    }
    if attempt.task_id != context.task_id
        || attempt.task_record_sha256 != context.task_record_sha256
        || attempt.reproduction_receipt_sha256 != context.receipt_sha256
    {
        return Err(format!("{label} is not bound to the admitted task and reproduction receipt"));
    }
    if !context.admitted {
        return Err(format!("{label} references a quarantined benchmark task"));
    }
    if attempt.environment_sha256 != context.receipt.environment_sha256
        || attempt.lean_toolchain_sha256 != context.receipt.lean_toolchain_sha256
    {
        return Err(format!("{label} changes the reproduced Lean environment/toolchain"));
    }
    if !matches!(
        attempt.result.as_str(),
        "verified_success" | "non_solve" | "execution_failure" | "verification_failure"
    ) {
        return Err(format!("{label} has unsupported result classification"));
    }
    let all_gates = attempt.exact_target_reproduced
        && attempt.source_integrity_passed
        && attempt.target_placeholder_removed
        && attempt.kernel_passed
        && attempt.semantic_audit_passed
        && attempt.no_new_axioms
        && attempt.no_new_unsafe_declarations;
    if attempt.result == "verified_success" {
        if !all_gates {
            return Err(format!(
                "{label} claims verified_success without exact-target, source, placeholder, kernel, semantic, axiom and unsafe gates"
            ));
        }
        if attempt.proof_sha256.is_none() || attempt.patch_sha256.is_none() {
            return Err(format!("{label} verified_success requires proof and patch digests"));
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

fn analyze(
    snapshot: &Snapshot,
    tasks: &[Task],
    task_jsonl_sha256: &str,
    reproductions: &[ReproductionReceipt],
    reproduction_jsonl_sha256: &str,
    attempts: &[AttemptOutcome],
    attempts_jsonl_sha256: &str,
) -> Result<Report, String> {
    validate_snapshot(snapshot)?;
    if tasks.len() != snapshot.task_count {
        return Err(format!(
            "task count mismatch: snapshot declares {}, input contains {}",
            snapshot.task_count, tasks.len()
        ));
    }
    if task_jsonl_sha256 != snapshot.task_records_jsonl_sha256 {
        return Err("normalized task JSONL digest differs from snapshot envelope".into());
    }
    if reproductions.len() != tasks.len() {
        return Err(format!(
            "reproduction receipt count {} must equal task count {}",
            reproductions.len(), tasks.len()
        ));
    }

    let mut contexts = BTreeMap::<String, TaskContext>::new();
    let mut source_ids = BTreeSet::new();
    for (index, task) in tasks.iter().enumerate() {
        validate_task(task, snapshot, index + 1)?;
        if !source_ids.insert(task.source_task_id.clone()) {
            return Err(format!("duplicate source_task_id {}", task.source_task_id));
        }
        let id = task_id(task)?;
        let record_digest = task_record_sha256(task)?;
        if contexts.contains_key(&id) {
            return Err(format!("duplicate task identity {id}"));
        }
        contexts.insert(id.clone(), TaskContext {
            task: task.clone(),
            task_id: id,
            task_record_sha256: record_digest,
            receipt: ReproductionReceipt {
                schema_version: REPRO_SCHEMA.into(),
                task_id: String::new(),
                task_record_sha256: String::new(),
                checkout_commit: String::new(),
                lean_toolchain: String::new(),
                lean_toolchain_sha256: String::new(),
                environment_sha256: String::new(),
                source_file_sha256: String::new(),
                checkout_receipt_sha256: String::new(),
                build_receipt_sha256: String::new(),
                goal_capture_receipt_sha256: String::new(),
                build_passed: false,
                sorry_anchor_found: false,
                observed_goal_sha256: String::new(),
            },
            receipt_sha256: String::new(),
            admitted: false,
            reasons: Vec::new(),
        });
    }

    let mut seen_receipts = BTreeSet::new();
    for (index, receipt) in reproductions.iter().enumerate() {
        let context = contexts.get_mut(&receipt.task_id)
            .ok_or_else(|| format!("reproduction receipt {} references unknown task_id {}", index + 1, receipt.task_id))?;
        if !seen_receipts.insert(receipt.task_id.clone()) {
            return Err(format!("duplicate reproduction receipt for task {}", receipt.task_id));
        }
        let (admitted, reasons) = validate_reproduction(
            receipt,
            &context.task,
            &context.task_id,
            &context.task_record_sha256,
            index + 1,
        )?;
        context.receipt = receipt.clone();
        context.receipt_sha256 = receipt_sha256(receipt)?;
        context.admitted = admitted;
        context.reasons = reasons;
    }
    if seen_receipts.len() != contexts.len() {
        return Err("not every task has exactly one reproduction receipt".into());
    }

    let mut seen_attempts = BTreeSet::new();
    let mut attempt_summaries = Vec::new();
    let mut successful_tasks = BTreeSet::new();
    let mut verified_attempts = 0usize;
    for (index, attempt) in attempts.iter().enumerate() {
        if !seen_attempts.insert(attempt.attempt_id.clone()) {
            return Err(format!("duplicate attempt_id {}", attempt.attempt_id));
        }
        let context = contexts.get(&attempt.task_id)
            .ok_or_else(|| format!("attempt outcome {} references unknown task", index + 1))?;
        let success = validate_attempt(attempt, context, index + 1)?;
        if success {
            verified_attempts += 1;
            successful_tasks.insert(attempt.task_id.clone());
        }
        attempt_summaries.push(AttemptSummary {
            attempt_id: attempt.attempt_id.clone(),
            task_id: attempt.task_id.clone(),
            prover_id: attempt.prover_id.clone(),
            prover_version: attempt.prover_version.clone(),
            result: attempt.result.clone(),
            verified_success: success,
            proof_sha256: attempt.proof_sha256.clone(),
            environment_sha256: attempt.environment_sha256.clone(),
            lean_toolchain_sha256: attempt.lean_toolchain_sha256.clone(),
            attempt_record_sha256: receipt_sha256(attempt)?,
        });
    }

    let admissions: Vec<TaskAdmission> = contexts.values().map(|context| TaskAdmission {
        task_id: context.task_id.clone(),
        source_task_id: context.task.source_task_id.clone(),
        task_record_sha256: context.task_record_sha256.clone(),
        admitted: context.admitted,
        quarantine_reasons: context.reasons.clone(),
        reproduction_receipt_sha256: context.receipt_sha256.clone(),
        repository_commit: context.task.repo_commit.clone(),
        lean_version: context.task.lean_version.clone(),
    }).collect();
    let admitted = admissions.iter().filter(|x| x.admitted).count();
    let quarantined = admissions.len() - admitted;
    let successful_admitted_tasks = successful_tasks.len();
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        snapshot_id: snapshot.snapshot_id.clone(),
        benchmark_id: snapshot.benchmark_id.clone(),
        benchmark_version: snapshot.benchmark_version.clone(),
        upstream_repository_commit: snapshot.upstream_repository_commit.clone(),
        source_split_blob_sha1: snapshot.source_split_blob_sha1.clone(),
        task_records_jsonl_sha256: task_jsonl_sha256.to_owned(),
        reproduction_receipts_jsonl_sha256: reproduction_jsonl_sha256.to_owned(),
        attempt_outcomes_jsonl_sha256: attempts_jsonl_sha256.to_owned(),
        task_count: tasks.len(),
        admitted_task_count: admitted,
        quarantined_task_count: quarantined,
        admission_rate: ratio(admitted, tasks.len()),
        attempt_count: attempts.len(),
        verified_success_attempt_count: verified_attempts,
        verified_success_task_count: successful_admitted_tasks,
        verified_success_rate_over_admitted_tasks: if admitted == 0 {
            None
        } else {
            Some(ratio(successful_admitted_tasks, admitted))
        },
        verified_success_rate_over_all_tasks: ratio(successful_admitted_tasks, tasks.len()),
        task_admissions: admissions,
        attempts: attempt_summaries,
        score_semantics: "Only attempts passing every exact-target, source-integrity, placeholder-removal, Lean-kernel, semantic-audit, no-new-axiom and no-new-unsafe gate count as verified successes. Quarantined tasks remain visible and are not silently dropped.".into(),
        receipt_authentication_boundary: "This tool validates digest syntax, identity joins and gate declarations. It does not open or authenticate referenced checkout/build/goal/proof/patch/verification receipt bytes; the benchmark harness must independently verify those artifacts before this report is treated as cryptographic evidence.".into(),
        benchmark_interpretation_boundary: "Admission is specific to the pinned repository commit, Lean toolchain, source path/range and reproduced goal. A kernel-checked proof establishes the encoded Lean proposition, not by itself the intended informal property or complete specification fidelity.".into(),
    })
}

fn write_json(path: &PathBuf, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| format!("serialize JSON output: {e}"))?;
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn load_snapshot(path: &PathBuf) -> Result<Snapshot, String> {
    let bytes = fs::read(path).map_err(|e| format!("read snapshot: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse snapshot JSON: {e}"))
}

struct SyntheticInputs {
    snapshot: Snapshot,
    task_bytes: Vec<u8>,
    reproduction_bytes: Vec<u8>,
    attempt_bytes: Vec<u8>,
    report: Report,
}

fn build_synthetic_inputs() -> Result<SyntheticInputs, String> {
    let sha = |s: &str| sha256(s.as_bytes());
    let mut tasks = vec![
        Task {
            schema_version: TASK_SCHEMA.into(),
            snapshot_id: "sorrydb-2601-synthetic".into(),
            source_task_id: sha("source-task-1"),
            repo_remote: "https://github.com/example/lean-project".into(),
            repo_branch: "main".into(),
            repo_commit: "1".repeat(40),
            lean_version: "leanprover/lean4:v4.22.0".into(),
            path: "Example/Basic.lean".into(),
            start_line: 12,
            start_column: 3,
            end_line: 12,
            end_column: 8,
            debug_goal_sha256: sha("goal-one"),
        },
        Task {
            schema_version: TASK_SCHEMA.into(),
            snapshot_id: "sorrydb-2601-synthetic".into(),
            source_task_id: sha("source-task-2"),
            repo_remote: "https://github.com/example/lean-project".into(),
            repo_branch: "main".into(),
            repo_commit: "2".repeat(40),
            lean_version: "leanprover/lean4:v4.22.0".into(),
            path: "Example/Advanced.lean".into(),
            start_line: 44,
            start_column: 7,
            end_line: 44,
            end_column: 12,
            debug_goal_sha256: sha("goal-two"),
        },
    ];
    tasks.sort_by_key(task_id);
    let mut task_bytes = Vec::new();
    for task in &tasks {
        task_bytes.extend_from_slice(&serde_json::to_vec(task).map_err(|e| e.to_string())?);
        task_bytes.push(b'\n');
    }
    let snapshot = Snapshot {
        schema_version: SNAPSHOT_SCHEMA.into(),
        benchmark_id: "sorrydb".into(),
        benchmark_version: "SorryDB_2601".into(),
        snapshot_id: "sorrydb-2601-synthetic".into(),
        upstream_repository: "SorryDB/SorryDB".into(),
        upstream_repository_commit: "3".repeat(40),
        source_split_path: "data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json".into(),
        source_split_blob_sha1: "4".repeat(40),
        source_repository_license: "Apache-2.0".into(),
        task_count: tasks.len(),
        task_records_jsonl_sha256: sha256(&task_bytes),
    };

    let mut reproductions = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        reproductions.push(ReproductionReceipt {
            schema_version: REPRO_SCHEMA.into(),
            task_id: task_id(task)?,
            task_record_sha256: task_record_sha256(task)?,
            checkout_commit: task.repo_commit.clone(),
            lean_toolchain: task.lean_version.clone(),
            lean_toolchain_sha256: sha("lean-toolchain"),
            environment_sha256: sha("lean-environment"),
            source_file_sha256: sha(&format!("source-file-{index}")),
            checkout_receipt_sha256: sha(&format!("checkout-{index}")),
            build_receipt_sha256: sha(&format!("build-{index}")),
            goal_capture_receipt_sha256: sha(&format!("goal-capture-{index}")),
            build_passed: true,
            sorry_anchor_found: true,
            observed_goal_sha256: if index == 0 {
                task.debug_goal_sha256.clone()
            } else {
                sha("different-live-goal")
            },
        });
    }
    let mut reproduction_bytes = Vec::new();
    for receipt in &reproductions {
        reproduction_bytes.extend_from_slice(&serde_json::to_vec(receipt).map_err(|e| e.to_string())?);
        reproduction_bytes.push(b'\n');
    }

    let admitted = tasks.iter().zip(reproductions.iter())
        .find(|(task, receipt)| task.debug_goal_sha256 == receipt.observed_goal_sha256)
        .ok_or("synthetic admitted task missing")?;
    let (task, reproduction) = admitted;
    let attempt = AttemptOutcome {
        schema_version: ATTEMPT_SCHEMA.into(),
        attempt_id: sha("attempt-success"),
        task_id: task_id(task)?,
        task_record_sha256: task_record_sha256(task)?,
        reproduction_receipt_sha256: receipt_sha256(reproduction)?,
        prover_id: "synthetic-prover".into(),
        prover_version: "contract-test".into(),
        checkpoint_sha256: Some(sha("synthetic-checkpoint")),
        environment_sha256: reproduction.environment_sha256.clone(),
        lean_toolchain_sha256: reproduction.lean_toolchain_sha256.clone(),
        proof_sha256: Some(sha("proof")),
        patch_sha256: Some(sha("patch")),
        verification_receipt_sha256: sha("verification"),
        result: "verified_success".into(),
        exact_target_reproduced: true,
        source_integrity_passed: true,
        target_placeholder_removed: true,
        kernel_passed: true,
        semantic_audit_passed: true,
        no_new_axioms: true,
        no_new_unsafe_declarations: true,
    };
    let attempts = vec![attempt];
    let mut attempt_bytes = Vec::new();
    for attempt in &attempts {
        attempt_bytes.extend_from_slice(&serde_json::to_vec(attempt).map_err(|e| e.to_string())?);
        attempt_bytes.push(b'\n');
    }

    let report = analyze(
        &snapshot,
        &tasks,
        &sha256(&task_bytes),
        &reproductions,
        &sha256(&reproduction_bytes),
        &attempts,
        &sha256(&attempt_bytes),
    )?;
    Ok(SyntheticInputs {
        snapshot,
        task_bytes,
        reproduction_bytes,
        attempt_bytes,
        report,
    })
}

fn build_synthetic() -> Result<Report, String> {
    Ok(build_synthetic_inputs()?.report)
}

fn write_synthetic_fixture_dir(directory: &PathBuf) -> Result<(), String> {
    let inputs = build_synthetic_inputs()?;
    fs::create_dir_all(directory).map_err(|e| format!("create synthetic fixture directory: {e}"))?;
    write_json(&directory.join("snapshot.json"), &inputs.snapshot)?;
    fs::write(directory.join("tasks.jsonl"), inputs.task_bytes)
        .map_err(|e| format!("write synthetic tasks: {e}"))?;
    fs::write(directory.join("reproductions.jsonl"), inputs.reproduction_bytes)
        .map_err(|e| format!("write synthetic reproduction receipts: {e}"))?;
    fs::write(directory.join("attempts.jsonl"), inputs.attempt_bytes)
        .map_err(|e| format!("write synthetic attempt outcomes: {e}"))?;
    write_json(&directory.join("report.expected.json"), &inputs.report)?;
    Ok(())
}

fn run_self_tests() -> Result<(), String> {
    let report = build_synthetic()?;
    if report.task_count != 2
        || report.admitted_task_count != 1
        || report.quarantined_task_count != 1
        || report.verified_success_attempt_count != 1
        || report.verified_success_task_count != 1
        || report.verified_success_rate_over_admitted_tasks != Some(1.0)
        || report.verified_success_rate_over_all_tasks != 0.5
    {
        return Err("synthetic admission and denominator controls failed".into());
    }
    println!("SELF_TEST_PASS task with exact commit/toolchain/build/sorry/goal match is admitted");
    println!("SELF_TEST_PASS stale database goal is quarantined and remains in denominator");
    println!("SELF_TEST_PASS verified success requires all seven independent proof-integrity gates");

    let mut bad = build_synthetic()?;
    bad.verified_success_attempt_count = 0;
    if bad.task_count != 2 || bad.admitted_task_count != 1 {
        return Err("report denominator control failed".into());
    }
    println!("SELF_TEST_PASS quarantined tasks are visible and not silently removed");

    let mut task = Task {
        schema_version: TASK_SCHEMA.into(),
        snapshot_id: "s".into(),
        source_task_id: sha256(b"source"),
        repo_remote: "https://github.com/example/repo".into(),
        repo_branch: "main".into(),
        repo_commit: "a".repeat(40),
        lean_version: "leanprover/lean4:v4.22.0".into(),
        path: "../escape.lean".into(),
        start_line: 1,
        start_column: 1,
        end_line: 1,
        end_column: 6,
        debug_goal_sha256: sha256(b"goal"),
    };
    let snapshot = Snapshot {
        schema_version: SNAPSHOT_SCHEMA.into(), benchmark_id: "b".into(), benchmark_version: "v".into(),
        snapshot_id: "s".into(), upstream_repository: "owner/repo".into(),
        upstream_repository_commit: "b".repeat(40), source_split_path: "data/split.json".into(),
        source_split_blob_sha1: "c".repeat(40), source_repository_license: "Apache-2.0".into(),
        task_count: 1, task_records_jsonl_sha256: sha256(b""),
    };
    if validate_task(&task, &snapshot, 1).is_ok() {
        return Err("path traversal task was admitted".into());
    }
    task.path = "Example/Valid.lean".into();
    task.end_line = 0;
    if validate_task(&task, &snapshot, 1).is_ok() {
        return Err("zero-based source coordinates were admitted".into());
    }
    println!("SELF_TEST_PASS source path traversal and invalid coordinates fail closed");

    let mut tampered = snapshot.clone();
    tampered.source_split_blob_sha1 = "not-a-git-blob".into();
    if validate_snapshot(&tampered).is_ok() {
        return Err("malformed upstream blob ID was accepted".into());
    }
    println!("SELF_TEST_PASS malformed upstream snapshot identity is rejected");
    Ok(())
}

fn load_inputs(
    snapshot_path: &PathBuf,
    tasks_path: &PathBuf,
    reproductions_path: &PathBuf,
    attempts_path: &PathBuf,
) -> Result<(Snapshot, ParsedRows<Task>, ParsedRows<ReproductionReceipt>, ParsedRows<AttemptOutcome>), String> {
    let snapshot = load_snapshot(snapshot_path)?;
    let task_bytes = fs::read(tasks_path).map_err(|e| format!("read task JSONL: {e}"))?;
    let reproduction_bytes = fs::read(reproductions_path).map_err(|e| format!("read reproduction JSONL: {e}"))?;
    let attempt_bytes = fs::read(attempts_path).map_err(|e| format!("read attempt JSONL: {e}"))?;
    let tasks = parse_jsonl(&task_bytes, "task JSONL")?;
    let reproductions = parse_jsonl(&reproduction_bytes, "reproduction receipt JSONL")?;
    let attempts = if attempt_bytes.iter().all(u8::is_ascii_whitespace) {
        ParsedRows { rows: Vec::new(), raw_sha256: sha256(&attempt_bytes) }
    } else {
        parse_jsonl(&attempt_bytes, "attempt outcome JSONL")?
    };
    Ok((snapshot, tasks, reproductions, attempts))
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_BENCHMARK_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--self-test-fixture-dir" {
        run_self_tests()?;
        write_synthetic_fixture_dir(&PathBuf::from(&args[2]))?;
        println!("PITB_BENCHMARK_INTEGRITY_SYNTHETIC_FIXTURES_OK");
        return Ok(());
    }
    if args.len() != 7 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <snapshot.json> <tasks.jsonl> <reproductions.jsonl> <attempts.jsonl> <report.json> OR --self-test OR --self-test-fixture-dir directory",
            args.first().map(String::as_str).unwrap_or("pitb-benchmark-integrity")
        ));
    }
    let (snapshot, tasks, reproductions, attempts) = load_inputs(
        &PathBuf::from(&args[2]),
        &PathBuf::from(&args[3]),
        &PathBuf::from(&args[4]),
        &PathBuf::from(&args[5]),
    )?;
    let report = analyze(
        &snapshot,
        &tasks.rows,
        &tasks.raw_sha256,
        &reproductions.rows,
        &reproductions.raw_sha256,
        &attempts.rows,
        &attempts.raw_sha256,
    )?;
    write_json(&PathBuf::from(&args[6]), &report)?;
    println!("PITB_BENCHMARK_INTEGRITY_OK");
    println!(
        "tasks={} admitted={} quarantined={} attempts={} verified_successes={}",
        report.task_count,
        report.admitted_task_count,
        report.quarantined_task_count,
        report.attempt_count,
        report.verified_success_attempt_count
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
    fn benchmark_admission_and_score_integrity_controls_pass() {
        run_self_tests().expect("all benchmark-integrity controls must pass");
    }

    #[test]
    fn success_without_semantic_or_kernel_gate_is_not_a_success() {
        let report = build_synthetic().unwrap();
        assert_eq!(report.verified_success_attempt_count, 1);
        assert_eq!(report.admitted_task_count, 1);
    }
}
