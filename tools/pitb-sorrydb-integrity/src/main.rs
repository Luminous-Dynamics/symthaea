use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

const PROVENANCE_SCHEMA: &str = "pitb.sorrydb-snapshot-provenance.v0";
const RECEIPT_SCHEMA: &str = "pitb.sorrydb-task-reproduction.v0";
const REPORT_SCHEMA: &str = "pitb.sorrydb-integrity-report.v0";

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Database {
    sorries: Vec<SorryRecord>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
struct SorryRecord {
    repo: RepoInfo,
    location: Location,
    debug_info: DebugInfo,
    metadata: Value,
    id: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
struct RepoInfo {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
struct Location {
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
struct DebugInfo {
    goal: String,
    url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    schema_version: String,
    snapshot_id: String,
    upstream_repository: String,
    upstream_repository_commit: String,
    database_git_blob_sha1: String,
    evaluation_split_git_blob_sha1: String,
    expected_database_task_count: usize,
    documented_database_task_count: usize,
    expected_evaluation_task_count: usize,
    evaluation_split_role: String,
    license: String,
    paper_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TaskReceipt {
    schema_version: String,
    task_id: String,
    repo_remote: String,
    repo_commit: String,
    lean_version: String,
    source_path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    debug_goal_sha256: String,
    source_file_sha256: String,
    source_range_sha256: String,
    elaborated_goal_sha256: String,
    lean_toolchain_sha256: String,
    lake_manifest_sha256: Option<String>,
    source_location_reproduced: bool,
    sorry_marker_verified: bool,
    goal_extracted: bool,
    goal_is_prop: bool,
    project_build_passed: bool,
    goal_alignment: String,
    goal_alignment_review_sha256: Option<String>,
    environment_receipt_sha256: String,
    source_receipt_sha256: String,
    semantic_audit_status: String,
    semantic_audit_receipt_sha256: Option<String>,
    vacuity_audit_status: String,
    vacuity_audit_receipt_sha256: Option<String>,
    assumption_audit_status: String,
    assumption_audit_receipt_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    snapshot_id: String,
    upstream_repository: String,
    upstream_repository_commit: String,
    database_git_blob_sha1: String,
    evaluation_split_git_blob_sha1: String,
    database_raw_sha256: String,
    evaluation_split_raw_sha256: String,
    database_task_count: usize,
    documented_database_task_count: usize,
    full_count_documentation_drift: bool,
    evaluation_task_count: usize,
    repository_commit_count: usize,
    receipt_count: usize,
    reproducible_task_count: usize,
    goal_alignment_counts: BTreeMap<String, usize>,
    semantic_audit_counts: BTreeMap<String, usize>,
    vacuity_audit_counts: BTreeMap<String, usize>,
    assumption_audit_counts: BTreeMap<String, usize>,
    unqualified_task_ids: Vec<String>,
    scheduler_admission: bool,
    receipt_authentication_boundary: String,
    semantic_integrity_boundary: String,
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

fn git_blob_sha1(bytes: &[u8], label: &str) -> Result<String, String> {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("start git hash-object for {label}: {e}"))?;
    child.stdin.take().ok_or("git hash-object stdin unavailable")?
        .write_all(bytes).map_err(|e| format!("write {label} to git hash-object: {e}"))?;
    let output = child.wait_with_output().map_err(|e| format!("wait for git hash-object: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "git hash-object failed for {label}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|e| format!("git hash-object output is not UTF-8: {e}"))?
        .trim()
        .to_owned();
    if !valid_hex(&value, 40) {
        return Err(format!("git hash-object returned malformed SHA-1 for {label}"));
    }
    Ok(value)
}

fn validate_provenance(
    provenance: &Provenance,
    database_bytes: &[u8],
    split_bytes: &[u8],
    database: &Database,
    split: &Database,
) -> Result<(), String> {
    if provenance.schema_version != PROVENANCE_SCHEMA {
        return Err("unsupported SorryDB provenance schema".into());
    }
    for (name, value) in [
        ("snapshot_id", provenance.snapshot_id.as_str()),
        ("upstream_repository", provenance.upstream_repository.as_str()),
        ("paper_url", provenance.paper_url.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("provenance.{name} cannot be empty"));
        }
    }
    if !provenance.upstream_repository.starts_with("https://github.com/") {
        return Err("upstream_repository must be an HTTPS GitHub URL".into());
    }
    if !valid_hex(&provenance.upstream_repository_commit, 40) {
        return Err("upstream_repository_commit must be a 40-character Git commit".into());
    }
    if !valid_hex(&provenance.database_git_blob_sha1, 40)
        || !valid_hex(&provenance.evaluation_split_git_blob_sha1, 40)
    {
        return Err("SorryDB dataset Git blob IDs must be 40-character lowercase SHA-1 values".into());
    }
    if provenance.evaluation_split_role != "test" {
        return Err("frozen SorryDB evaluation split must be declared as test".into());
    }
    if provenance.license != "Apache-2.0" {
        return Err("SorryDB source repository license identity changed; require explicit review".into());
    }
    let actual_database_blob = git_blob_sha1(database_bytes, "full SorryDB database")?;
    let actual_split_blob = git_blob_sha1(split_bytes, "SorryDB evaluation split")?;
    if actual_database_blob != provenance.database_git_blob_sha1 {
        return Err(format!(
            "full database Git blob mismatch: expected {}, got {}",
            provenance.database_git_blob_sha1, actual_database_blob
        ));
    }
    if actual_split_blob != provenance.evaluation_split_git_blob_sha1 {
        return Err(format!(
            "evaluation split Git blob mismatch: expected {}, got {}",
            provenance.evaluation_split_git_blob_sha1, actual_split_blob
        ));
    }
    if provenance.documented_database_task_count == 0 {
        return Err("documented_database_task_count must be non-zero".into());
    }
    if database.sorries.len() != provenance.expected_database_task_count {
        return Err(format!(
            "full database task count mismatch: expected {}, got {}",
            provenance.expected_database_task_count, database.sorries.len()
        ));
    }
    if split.sorries.len() != provenance.expected_evaluation_task_count {
        return Err(format!(
            "evaluation task count mismatch: expected {}, got {}",
            provenance.expected_evaluation_task_count, split.sorries.len()
        ));
    }
    Ok(())
}

fn validate_record(record: &SorryRecord, label: &str) -> Result<(), String> {
    if record.id.trim().is_empty() {
        return Err(format!("{label}.id cannot be empty"));
    }
    if !record.repo.remote.starts_with("https://github.com/") {
        return Err(format!("{label}.repo.remote must be an HTTPS GitHub URL"));
    }
    if record.repo.branch.trim().is_empty() || record.repo.lean_version.trim().is_empty() {
        return Err(format!("{label} branch and Lean version must be non-empty"));
    }
    if !valid_hex(&record.repo.commit, 40) {
        return Err(format!("{label}.repo.commit must be a 40-character lowercase Git commit"));
    }
    // SorryDB line numbers are 1-based; columns are 0-based character offsets.
    // Preserve the upstream convention exactly, including valid start_column=0 rows.
    if record.location.path.trim().is_empty()
        || record.location.start_line == 0
        || record.location.end_line == 0
    {
        return Err(format!("{label} has an invalid source path or line coordinate"));
    }
    if (record.location.end_line, record.location.end_column)
        < (record.location.start_line, record.location.start_column)
    {
        return Err(format!("{label} source range ends before it begins"));
    }
    if record.debug_info.goal.trim().is_empty() || record.debug_info.url.trim().is_empty() {
        return Err(format!("{label} debug goal and source URL must be non-empty"));
    }
    if !record.debug_info.url.contains(&record.repo.commit)
        || !record.debug_info.url.contains(&record.location.path)
    {
        return Err(format!("{label} debug URL does not bind the pinned commit and source path"));
    }
    Ok(())
}

fn validate_receipt(
    receipt: &TaskReceipt,
    record: &SorryRecord,
    index: usize,
) -> Result<bool, String> {
    let label = format!("task receipt {index}");
    if receipt.schema_version != RECEIPT_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    if receipt.task_id != record.id
        || receipt.repo_remote != record.repo.remote
        || receipt.repo_commit != record.repo.commit
        || receipt.lean_version != record.repo.lean_version
        || receipt.source_path != record.location.path
        || receipt.start_line != record.location.start_line
        || receipt.start_column != record.location.start_column
        || receipt.end_line != record.location.end_line
        || receipt.end_column != record.location.end_column
    {
        return Err(format!("{label} source identity differs from the pinned SorryDB record"));
    }
    for (name, value) in [
        ("debug_goal_sha256", &receipt.debug_goal_sha256),
        ("source_file_sha256", &receipt.source_file_sha256),
        ("source_range_sha256", &receipt.source_range_sha256),
        ("elaborated_goal_sha256", &receipt.elaborated_goal_sha256),
        ("lean_toolchain_sha256", &receipt.lean_toolchain_sha256),
        ("environment_receipt_sha256", &receipt.environment_receipt_sha256),
        ("source_receipt_sha256", &receipt.source_receipt_sha256),
    ] {
        digest(value, &format!("{label}.{name}"))?;
    }
    let expected_goal = sha256(record.debug_info.goal.as_bytes());
    if receipt.debug_goal_sha256 != expected_goal {
        return Err(format!("{label} debug goal digest differs from the pinned SorryDB record"));
    }
    if let Some(value) = &receipt.lake_manifest_sha256 {
        digest(value, &format!("{label}.lake_manifest_sha256"))?;
    }
    if let Some(value) = &receipt.goal_alignment_review_sha256 {
        digest(value, &format!("{label}.goal_alignment_review_sha256"))?;
    }
    if let Some(value) = &receipt.semantic_audit_receipt_sha256 {
        digest(value, &format!("{label}.semantic_audit_receipt_sha256"))?;
    }
    if let Some(value) = &receipt.vacuity_audit_receipt_sha256 {
        digest(value, &format!("{label}.vacuity_audit_receipt_sha256"))?;
    }
    if let Some(value) = &receipt.assumption_audit_receipt_sha256 {
        digest(value, &format!("{label}.assumption_audit_receipt_sha256"))?;
    }
    if !matches!(receipt.goal_alignment.as_str(), "exact" | "normalized" | "reviewed" | "mismatch") {
        return Err(format!("{label} has unsupported goal_alignment"));
    }
    if receipt.goal_alignment == "reviewed" && receipt.goal_alignment_review_sha256.is_none() {
        return Err(format!("{label} reviewed goal alignment requires a review receipt"));
    }
    if receipt.goal_alignment == "mismatch" {
        // A mismatch is an auditable finding and disqualifies the task; it is not
        // silently accepted as a reproduction success.
    }
    for (name, status, evidence) in [
        ("semantic_audit", &receipt.semantic_audit_status, &receipt.semantic_audit_receipt_sha256),
        ("vacuity_audit", &receipt.vacuity_audit_status, &receipt.vacuity_audit_receipt_sha256),
        ("assumption_audit", &receipt.assumption_audit_status, &receipt.assumption_audit_receipt_sha256),
    ] {
        if !matches!(status.as_str(), "passed" | "finding" | "not_run") {
            return Err(format!("{label}.{name}_status is unsupported"));
        }
        if status != "not_run" && evidence.is_none() {
            return Err(format!("{label}.{name} non-default status requires an evidence receipt"));
        }
    }

    let reproduced = receipt.source_location_reproduced
        && receipt.sorry_marker_verified
        && receipt.goal_extracted
        && receipt.goal_is_prop
        && receipt.project_build_passed
        && receipt.goal_alignment != "mismatch";
    Ok(reproduced)
}

fn parse_json<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<T, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("parse {label}: {e}"))
}

fn load_receipts(bytes: &[u8]) -> Result<Vec<TaskReceipt>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("receipts are not UTF-8: {e}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(line)
            .map_err(|e| format!("receipt JSONL line {}: {e}", index + 1))?);
    }
    Ok(rows)
}

fn status_counts<I: IntoIterator<Item = String>>(values: I) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for value in values {
        *out.entry(value).or_insert(0) += 1;
    }
    out
}

fn verify(
    database_bytes: &[u8],
    split_bytes: &[u8],
    provenance_bytes: &[u8],
    receipts_bytes: &[u8],
) -> Result<Report, String> {
    let database: Database = parse_json(database_bytes, "full SorryDB database")?;
    let split: Database = parse_json(split_bytes, "SorryDB evaluation split")?;
    let provenance: Provenance = parse_json(provenance_bytes, "SorryDB provenance")?;
    validate_provenance(&provenance, database_bytes, split_bytes, &database, &split)?;

    let mut full_by_id = BTreeMap::<String, &SorryRecord>::new();
    for (index, record) in database.sorries.iter().enumerate() {
        validate_record(record, &format!("database.sorries[{index}]"))?;
        if full_by_id.insert(record.id.clone(), record).is_some() {
            return Err(format!("duplicate task ID {} in full SorryDB database", record.id));
        }
    }
    let mut split_ids = BTreeSet::new();
    for (index, record) in split.sorries.iter().enumerate() {
        validate_record(record, &format!("split.sorries[{index}]"))?;
        if !split_ids.insert(record.id.clone()) {
            return Err(format!("duplicate task ID {} in evaluation split", record.id));
        }
        let original = full_by_id.get(&record.id)
            .ok_or_else(|| format!("evaluation task {} is absent from the full snapshot", record.id))?;
        if **original != *record {
            return Err(format!("evaluation task {} differs from its full-snapshot record", record.id));
        }
    }

    let receipts = load_receipts(receipts_bytes)?;
    let mut receipt_by_id = BTreeMap::<String, &TaskReceipt>::new();
    let mut reproducible = 0usize;
    let mut unqualified = Vec::new();
    let mut goal_alignments = Vec::new();
    let mut semantic = Vec::new();
    let mut vacuity = Vec::new();
    let mut assumptions = Vec::new();
    for (index, receipt) in receipts.iter().enumerate() {
        let record = full_by_id.get(&receipt.task_id)
            .ok_or_else(|| format!("receipt references unknown SorryDB task {}", receipt.task_id))?;
        if !split_ids.contains(&receipt.task_id) {
            return Err(format!("receipt {} is outside the frozen evaluation split", receipt.task_id));
        }
        if receipt_by_id.insert(receipt.task_id.clone(), receipt).is_some() {
            return Err(format!("duplicate receipt for task {}", receipt.task_id));
        }
        let pass = validate_receipt(receipt, record, index + 1)?;
        if pass {
            reproducible += 1;
        } else {
            unqualified.push(receipt.task_id.clone());
        }
        goal_alignments.push(receipt.goal_alignment.clone());
        semantic.push(receipt.semantic_audit_status.clone());
        vacuity.push(receipt.vacuity_audit_status.clone());
        assumptions.push(receipt.assumption_audit_status.clone());
    }
    let missing: Vec<_> = split_ids.difference(&receipt_by_id.keys().cloned().collect())
        .take(20).cloned().collect();
    if !missing.is_empty() || receipt_by_id.len() != split_ids.len() {
        return Err(format!("evaluation split does not have exactly one reproduction receipt per task; missing {missing:?}"));
    }

    let unique_repositories: BTreeSet<_> = split.sorries.iter()
        .map(|r| (r.repo.remote.clone(), r.repo.commit.clone()))
        .collect();
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        snapshot_id: provenance.snapshot_id,
        upstream_repository: provenance.upstream_repository,
        upstream_repository_commit: provenance.upstream_repository_commit,
        database_git_blob_sha1: provenance.database_git_blob_sha1,
        evaluation_split_git_blob_sha1: provenance.evaluation_split_git_blob_sha1,
        database_raw_sha256: sha256(database_bytes),
        evaluation_split_raw_sha256: sha256(split_bytes),
        database_task_count: database.sorries.len(),
        documented_database_task_count: provenance.documented_database_task_count,
        full_count_documentation_drift: database.sorries.len() != provenance.documented_database_task_count,
        evaluation_task_count: split.sorries.len(),
        repository_commit_count: unique_repositories.len(),
        receipt_count: receipts.len(),
        reproducible_task_count: reproducible,
        goal_alignment_counts: status_counts(goal_alignments),
        semantic_audit_counts: status_counts(semantic),
        vacuity_audit_counts: status_counts(vacuity),
        assumption_audit_counts: status_counts(assumptions),
        unqualified_task_ids: unqualified,
        scheduler_admission: false,
        receipt_authentication_boundary: "Receipt hashes are syntax-checked and bound to task identities; the referenced source files, toolchains, build logs and review bytes must be independently opened and authenticated by the evidence-byte verifier before leaderboard admission.".into(),
        semantic_integrity_boundary: "A reproduced sorry and kernel-checked replacement prove only the pinned Lean proposition. Semantic, vacuity and assumption audits are separate signals; a passed build is not evidence that the proposition faithfully encodes an informal intent.".into(),
    })
}

fn fake_hash(label: &str) -> String {
    sha256(label.as_bytes())
}

fn synthetic_record(id: &str, commit: &str) -> SorryRecord {
    SorryRecord {
        repo: RepoInfo {
            remote: "https://github.com/example/lean-project".into(),
            branch: "main".into(),
            commit: commit.into(),
            lean_version: "v4.18.0".into(),
        },
        location: Location {
            path: "Example/Theorem.lean".into(),
            start_line: 12,
            start_column: 1,
            end_line: 12,
            end_column: 6,
        },
        debug_info: DebugInfo {
            goal: "⊢ True".into(),
            url: format!("https://github.com/example/lean-project/blob/{commit}/Example/Theorem.lean#L12"),
        },
        metadata: serde_json::json!({
            "blame_email_hash":"aabbccdd",
            "blame_date":"2025-01-01T00:00:00+00:00",
            "inclusion_date":"2025-02-01T00:00:00+00:00"
        }),
        id: id.into(),
    }
}

fn synthetic_receipt(record: &SorryRecord) -> TaskReceipt {
    TaskReceipt {
        schema_version: RECEIPT_SCHEMA.into(),
        task_id: record.id.clone(),
        repo_remote: record.repo.remote.clone(),
        repo_commit: record.repo.commit.clone(),
        lean_version: record.repo.lean_version.clone(),
        source_path: record.location.path.clone(),
        start_line: record.location.start_line,
        start_column: record.location.start_column,
        end_line: record.location.end_line,
        end_column: record.location.end_column,
        debug_goal_sha256: sha256(record.debug_info.goal.as_bytes()),
        source_file_sha256: fake_hash("source-file"),
        source_range_sha256: fake_hash("source-range"),
        elaborated_goal_sha256: fake_hash("elaborated-goal"),
        lean_toolchain_sha256: fake_hash("lean-toolchain"),
        lake_manifest_sha256: Some(fake_hash("lake-manifest")),
        source_location_reproduced: true,
        sorry_marker_verified: true,
        goal_extracted: true,
        goal_is_prop: true,
        project_build_passed: true,
        goal_alignment: "exact".into(),
        goal_alignment_review_sha256: None,
        environment_receipt_sha256: fake_hash("environment"),
        source_receipt_sha256: fake_hash("source"),
        semantic_audit_status: "not_run".into(),
        semantic_audit_receipt_sha256: None,
        vacuity_audit_status: "not_run".into(),
        vacuity_audit_receipt_sha256: None,
        assumption_audit_status: "not_run".into(),
        assumption_audit_receipt_sha256: None,
    }
}

fn run_self_tests() -> Result<(), String> {
    let full = Database { sorries: vec![
        synthetic_record("task-a", &"1".repeat(40)),
        synthetic_record("task-b", &"2".repeat(40)),
    ]};
    let split = Database { sorries: vec![full.sorries[0].clone()] };
    let receipts = vec![synthetic_receipt(&split.sorries[0])];
    let full_bytes = serde_json::to_vec(&full).map_err(|e| e.to_string())?;
    let split_bytes = serde_json::to_vec(&split).map_err(|e| e.to_string())?;
    let provenance = Provenance {
        schema_version: PROVENANCE_SCHEMA.into(),
        snapshot_id: "SorryDB-synthetic".into(),
        upstream_repository: "https://github.com/SorryDB/SorryDB".into(),
        upstream_repository_commit: "3".repeat(40),
        database_git_blob_sha1: git_blob_sha1(&full_bytes, "synthetic full database")?,
        evaluation_split_git_blob_sha1: git_blob_sha1(&split_bytes, "synthetic split")?,
        expected_database_task_count: 2,
        documented_database_task_count: 2,
        expected_evaluation_task_count: 1,
        evaluation_split_role: "test".into(),
        license: "Apache-2.0".into(),
        paper_url: "https://proceedings.mlr.press/v306/letson26a.html".into(),
    };
    let provenance_bytes = serde_json::to_vec(&provenance).map_err(|e| e.to_string())?;
    let receipt_bytes = [
        serde_json::to_vec(&receipts[0]).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
    ].concat();
    let report = verify(&full_bytes, &split_bytes, &provenance_bytes, &receipt_bytes)?;
    if report.reproducible_task_count != 1 || report.evaluation_task_count != 1 {
        return Err("synthetic SorryDB identity and reproduction control failed".into());
    }
    println!("SELF_TEST_PASS pinned Git blob IDs and exact split membership");

    let mut bad_receipt = receipts[0].clone();
    bad_receipt.repo_commit = "4".repeat(40);
    if validate_receipt(&bad_receipt, &split.sorries[0], 1).is_ok() {
        return Err("receipt with wrong source repository commit was accepted".into());
    }
    println!("SELF_TEST_PASS task receipts bind exact source commit and coordinates");

    let mut bad_receipt = receipts[0].clone();
    bad_receipt.debug_goal_sha256 = fake_hash("wrong debug goal");
    if validate_receipt(&bad_receipt, &split.sorries[0], 1).is_ok() {
        return Err("receipt with wrong debug-goal digest was accepted".into());
    }
    println!("SELF_TEST_PASS receipt binds exact upstream debug-goal bytes");

    let mut bad_receipt = receipts[0].clone();
    bad_receipt.goal_is_prop = false;
    if validate_receipt(&bad_receipt, &split.sorries[0], 1).unwrap() {
        return Err("non-Prop target was treated as a qualified SorryDB theorem".into());
    }
    println!("SELF_TEST_PASS non-Prop targets remain unqualified");

    let mut bad_receipt = receipts[0].clone();
    bad_receipt.goal_alignment = "mismatch".into();
    if validate_receipt(&bad_receipt, &split.sorries[0], 1).unwrap() {
        return Err("mismatched extracted goal was treated as qualified".into());
    }
    println!("SELF_TEST_PASS mismatched elaborated goal remains unqualified");

    let mut duplicate = receipts.clone();
    duplicate.push(receipts[0].clone());
    let duplicate_bytes = [
        serde_json::to_vec(&duplicate[0]).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
        serde_json::to_vec(&duplicate[1]).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
    ].concat();
    if verify(&full_bytes, &split_bytes, &provenance_bytes, &duplicate_bytes).is_ok() {
        return Err("duplicate task receipts were accepted".into());
    }
    println!("SELF_TEST_PASS duplicate task receipts are rejected");

    let missing = b"";
    if verify(&full_bytes, &split_bytes, &provenance_bytes, missing).is_ok() {
        return Err("missing per-task reproduction receipt was accepted".into());
    }
    println!("SELF_TEST_PASS every evaluation task requires one reproduction receipt");

    Ok(())
}

fn synthetic_report_and_receipt() -> Result<(Report, TaskReceipt), String> {
    let full = Database { sorries: vec![
        synthetic_record("task-a", &"1".repeat(40)),
        synthetic_record("task-b", &"2".repeat(40)),
    ]};
    let split = Database { sorries: vec![full.sorries[0].clone()] };
    let receipt = synthetic_receipt(&split.sorries[0]);
    let full_bytes = serde_json::to_vec(&full).map_err(|e| e.to_string())?;
    let split_bytes = serde_json::to_vec(&split).map_err(|e| e.to_string())?;
    let provenance = Provenance {
        schema_version: PROVENANCE_SCHEMA.into(),
        snapshot_id: "SorryDB-synthetic".into(),
        upstream_repository: "https://github.com/SorryDB/SorryDB".into(),
        upstream_repository_commit: "3".repeat(40),
        database_git_blob_sha1: git_blob_sha1(&full_bytes, "synthetic full database")?,
        evaluation_split_git_blob_sha1: git_blob_sha1(&split_bytes, "synthetic split")?,
        expected_database_task_count: 2,
        expected_evaluation_task_count: 1,
        evaluation_split_role: "test".into(),
        license: "Apache-2.0".into(),
        paper_url: "https://proceedings.mlr.press/v306/letson26a.html".into(),
    };
    let provenance_bytes = serde_json::to_vec(&provenance).map_err(|e| e.to_string())?;
    let receipt_bytes = [
        serde_json::to_vec(&receipt).map_err(|e| e.to_string())?,
        b"\n".to_vec(),
    ].concat();
    let report = verify(&full_bytes, &split_bytes, &provenance_bytes, &receipt_bytes)?;
    Ok((report, receipt))
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SORRYDB_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 4 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let (report, receipt) = synthetic_report_and_receipt()?;
        let report_bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize synthetic report: {e}"))?;
        fs::write(&args[2], report_bytes).map_err(|e| format!("write synthetic report: {e}"))?;
        let receipt_bytes = [
            serde_json::to_vec(&receipt).map_err(|e| format!("serialize synthetic receipt: {e}"))?,
            b"\n".to_vec(),
        ].concat();
        fs::write(&args[3], receipt_bytes).map_err(|e| format!("write synthetic receipt: {e}"))?;
        println!("PITB_SORRYDB_INTEGRITY_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() != 7 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <full-database.json> <evaluation-split.json> <provenance.json> <task-receipts.jsonl> <report.json> OR --self-test",
            args.first().map(String::as_str).unwrap_or("pitb-sorrydb-integrity")
        ));
    }
    let database_path = PathBuf::from(&args[2]);
    let split_path = PathBuf::from(&args[3]);
    let provenance_path = PathBuf::from(&args[4]);
    let receipts_path = PathBuf::from(&args[5]);
    let report_path = PathBuf::from(&args[6]);
    let database_bytes = fs::read(&database_path).map_err(|e| format!("read full database: {e}"))?;
    let split_bytes = fs::read(&split_path).map_err(|e| format!("read evaluation split: {e}"))?;
    let provenance_bytes = fs::read(&provenance_path).map_err(|e| format!("read provenance: {e}"))?;
    let receipts_bytes = fs::read(&receipts_path).map_err(|e| format!("read task receipts: {e}"))?;
    let report = verify(&database_bytes, &split_bytes, &provenance_bytes, &receipts_bytes)?;
    let output = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?;
    fs::write(&report_path, output).map_err(|e| format!("write report: {e}"))?;
    println!("PITB_SORRYDB_INTEGRITY_OK");
    println!("snapshot={} tasks={} reproducible={}", report.snapshot_id, report.evaluation_task_count, report.reproducible_task_count);
    Ok(())
}

fn main() -> std::process::ExitCode {
    match execute() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("INVALID: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_sorrydb_integrity_controls_pass() {
        run_self_tests().expect("all SorryDB integrity controls must pass");
    }

    #[test]
    fn semantic_audit_finding_does_not_change_source_reproduction() {
        let record = synthetic_record("task", &"1".repeat(40));
        let mut receipt = synthetic_receipt(&record);
        receipt.semantic_audit_status = "finding".into();
        receipt.semantic_audit_receipt_sha256 = Some(fake_hash("finding"));
        assert!(validate_receipt(&receipt, &record, 1).unwrap());
    }
}
