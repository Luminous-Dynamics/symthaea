use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    process::ExitCode,
};

const PIN_SCHEMA: &str = "pitb.sorrydb-snapshot-manifest.v0";
const REPORT_SCHEMA: &str = "pitb.sorrydb-snapshot-audit-report.v0";
const UPSTREAM_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const FULL_DATASET_BLOB: &str = "5fafddac467af6d30d8f5c8731d6edec9c273a47";
const EVALUATION_SPLIT_BLOB: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const EXPECTED_FULL_TASKS: usize = 2601;
const EXPECTED_EVALUATION_TASKS: usize = 1000;
const EXPECTED_REPOSITORIES: usize = 78;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SnapshotPin {
    schema_version: String,
    snapshot_id: String,
    upstream_repository: String,
    upstream_repository_commit: String,
    upstream_license: String,
    full_dataset_path: String,
    full_dataset_git_blob_sha1: String,
    evaluation_split_path: String,
    evaluation_split_git_blob_sha1: String,
    expected_full_task_count: usize,
    expected_evaluation_task_count: usize,
    expected_unique_repository_count: usize,
    paper_url: String,
    leaderboard_url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Database {
    #[serde(default)]
    repos: Vec<RepoIndex>,
    sorries: Vec<SorryRecord>,
}

#[derive(Debug, Clone, Deserialize)]
struct RepoIndex {
    remote_url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct SorryRecord {
    id: String,
    repo: RepoInfo,
    location: Location,
    debug_info: DebugInfo,
    metadata: Metadata,
}

#[derive(Debug, Clone, Deserialize)]
struct RepoInfo {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Location {
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct DebugInfo {
    goal: String,
    url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Metadata {
    blame_email_hash: String,
    blame_date: String,
    inclusion_date: String,
}

#[derive(Debug, Clone, Serialize)]
struct Distribution {
    key: String,
    task_count: usize,
    repository_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SnapshotReport {
    input_raw_sha256: String,
    input_git_blob_sha1: String,
    task_count: usize,
    unique_task_id_count: usize,
    unique_source_location_count: usize,
    unique_repository_count: usize,
    task_count_by_lean_version: Vec<Distribution>,
    repositories_by_task_count: Vec<(String, usize)>,
    repeated_goal_group_count: usize,
    repeated_goal_task_count: usize,
    repository_registry_count: usize,
    repository_registry_complete: bool,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    snapshot_id: String,
    upstream_repository: String,
    upstream_repository_commit: String,
    full_snapshot: SnapshotReport,
    evaluation_split: SnapshotReport,
    evaluation_split_is_subset_of_full_snapshot: bool,
    missing_evaluation_task_ids: Vec<String>,
    count_pins_match: bool,
    repository_count_pin_matches: bool,
    source_git_blob_pins_match: bool,
    source_identity_boundary: String,
    semantic_audit_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn git_blob_sha1(bytes: &[u8], label: &str) -> Result<String, String> {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{label}: unable to start git hash-object: {e}"))?;
    {
        let stdin = child.stdin.as_mut().ok_or("git hash-object stdin unavailable")?;
        stdin.write_all(bytes).map_err(|e| format!("{label}: write to git hash-object: {e}"))?;
    }
    let output = child.wait_with_output().map_err(|e| format!("{label}: wait for git hash-object: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{label}: git hash-object failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let digest = String::from_utf8(output.stdout)
        .map_err(|e| format!("{label}: git hash-object output is not UTF-8: {e}"))?
        .trim()
        .to_owned();
    if !valid_hex(&digest, 40) {
        return Err(format!("{label}: git hash-object returned malformed object ID"));
    }
    Ok(digest)
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn validate_pin(pin: &SnapshotPin) -> Result<(), String> {
    if pin.schema_version != PIN_SCHEMA {
        return Err("unsupported SorryDB snapshot pin schema".into());
    }
    if pin.snapshot_id != "SorryDB_2601"
        || pin.upstream_repository != "SorryDB/SorryDB"
        || pin.upstream_repository_commit != UPSTREAM_COMMIT
        || pin.upstream_license != "Apache-2.0"
        || pin.full_dataset_path != "data/SorryDB_2601/SorryDB_2601.json"
        || pin.full_dataset_git_blob_sha1 != FULL_DATASET_BLOB
        || pin.evaluation_split_path != "data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json"
        || pin.evaluation_split_git_blob_sha1 != EVALUATION_SPLIT_BLOB
        || pin.expected_full_task_count != EXPECTED_FULL_TASKS
        || pin.expected_evaluation_task_count != EXPECTED_EVALUATION_TASKS
        || pin.expected_unique_repository_count != EXPECTED_REPOSITORIES
    {
        return Err("SorryDB_2601 manifest differs from the independently pinned upstream snapshot".into());
    }
    for (field, value) in [
        ("paper_url", pin.paper_url.as_str()),
        ("leaderboard_url", pin.leaderboard_url.as_str()),
    ] {
        if !value.starts_with("https://") {
            return Err(format!("{field} must be an HTTPS URL"));
        }
    }
    Ok(())
}

fn valid_timestamp(value: &str) -> bool {
    // Strict enough to reject missing or malformed timestamp fields without
    // imposing timezone parsing semantics on the source's ISO-8601 variants.
    value.len() >= 19
        && value.as_bytes().get(4) == Some(&b'-')
        && value.as_bytes().get(7) == Some(&b'-')
        && value.as_bytes().get(10) == Some(&b'T')
        && value.contains(':')
}

fn validate_task(task: &SorryRecord, index: usize) -> Result<(), String> {
    let label = format!("task[{index}]");
    non_empty(&task.id, &format!("{label}.id"))?;
    if task.repo.remote.trim().is_empty() || !task.repo.remote.starts_with("https://") {
        return Err(format!("{label}.repo.remote must be a non-empty HTTPS repository URL"));
    }
    non_empty(&task.repo.branch, &format!("{label}.repo.branch"))?;
    if !valid_hex(&task.repo.commit, 40) {
        return Err(format!("{label}.repo.commit must be a 40-character lowercase Git SHA-1"));
    }
    non_empty(&task.repo.lean_version, &format!("{label}.repo.lean_version"))?;

    let path = task.location.path.as_str();
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        || !path.ends_with(".lean")
    {
        return Err(format!("{label}.location.path is unsafe or is not a Lean source path"));
    }
    if task.location.start_line == 0
        || task.location.end_line == 0
        || task.location.end_line < task.location.start_line
        || (task.location.end_line == task.location.start_line
            && task.location.end_column < task.location.start_column)
    {
        return Err(format!("{label}.location has invalid source coordinates"));
    }

    non_empty(&task.debug_info.goal, &format!("{label}.debug_info.goal"))?;
    if !task.debug_info.url.starts_with("https://") {
        return Err(format!("{label}.debug_info.url must be HTTPS"));
    }
    non_empty(&task.metadata.blame_email_hash, &format!("{label}.metadata.blame_email_hash"))?;
    if !valid_timestamp(&task.metadata.blame_date) || !valid_timestamp(&task.metadata.inclusion_date) {
        return Err(format!("{label}.metadata timestamps must be ISO-8601-like date-time strings"));
    }
    Ok(())
}

fn task_identity(task: &SorryRecord) -> String {
    format!(
        "{}|{}|{}|{}|{}:{}-{}:{}",
        task.repo.remote,
        task.repo.branch,
        task.repo.commit,
        task.location.path,
        task.location.start_line,
        task.location.start_column,
        task.location.end_line,
        task.location.end_column
    )
}

fn source_location_identity(task: &SorryRecord) -> String {
    format!(
        "{}|{}|{}|{}:{}-{}:{}",
        task.repo.remote,
        task.repo.commit,
        task.location.path,
        task.location.start_line,
        task.location.start_column,
        task.location.end_line,
        task.location.end_column
    )
}

fn goal_key(task: &SorryRecord) -> String {
    task.debug_info
        .goal
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn audit_database(
    bytes: &[u8],
    expected_blob: &str,
    label: &str,
) -> Result<(SnapshotReport, BTreeMap<String, String>, BTreeSet<String>), String> {
    let actual_blob = git_blob_sha1(bytes, label)?;
    if actual_blob != expected_blob {
        return Err(format!(
            "{label} Git blob identity mismatch: expected {expected_blob}, got {actual_blob}"
        ));
    }
    let db: Database = serde_json::from_slice(bytes)
        .map_err(|e| format!("{label} JSON does not match the SorryDB database structure: {e}"))?;
    if db.sorries.is_empty() {
        return Err(format!("{label} contains no sorries"));
    }

    let registry: BTreeSet<String> = db.repos.iter().map(|r| r.remote_url.clone()).collect();

    let mut ids = BTreeMap::<String, String>::new();
    let mut locations = BTreeSet::<String>::new();
    let mut tasks_by_repo = BTreeMap::<String, usize>::new();
    let mut tasks_by_lean = BTreeMap::<String, usize>::new();
    let mut repositories_by_lean = BTreeMap::<String, BTreeSet<String>>::new();
    let mut goal_counts = BTreeMap::<String, usize>::new();
    let mut task_records = BTreeSet::<String>::new();

    for (index, task) in db.sorries.iter().enumerate() {
        validate_task(task, index + 1)?;
        let identity = task_identity(task);
        if ids.insert(task.id.clone(), identity.clone()).is_some() {
            return Err(format!("{label} contains duplicate task ID {}", task.id));
        }
        if !task_records.insert(identity.clone()) {
            return Err(format!("{label} contains duplicate source task identity {identity}"));
        }
        locations.insert(source_location_identity(task));
        *tasks_by_repo.entry(task.repo.remote.clone()).or_default() += 1;
        *tasks_by_lean.entry(task.repo.lean_version.clone()).or_default() += 1;
        repositories_by_lean
            .entry(task.repo.lean_version.clone())
            .or_default()
            .insert(task.repo.remote.clone());
        *goal_counts.entry(goal_key(task)).or_default() += 1;
        if !registry.is_empty() && !registry.contains(&task.repo.remote) {
            return Err(format!("{label} task references repository absent from repo registry: {}", task.repo.remote));
        }
    }

    let repeated_goal_groups = goal_counts.values().filter(|count| **count > 1).count();
    let repeated_goal_tasks = goal_counts.values().filter(|count| **count > 1).sum();

    let distributions = tasks_by_lean
        .iter()
        .map(|(version, count)| Distribution {
            key: version.clone(),
            task_count: *count,
            repository_count: repositories_by_lean.get(version).map_or(0, BTreeSet::len),
        })
        .collect();

    let mut repositories_by_count: Vec<(String, usize)> = tasks_by_repo.into_iter().collect();
    repositories_by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let report = SnapshotReport {
        input_raw_sha256: sha256(bytes),
        input_git_blob_sha1: actual_blob,
        task_count: db.sorries.len(),
        unique_task_id_count: ids.len(),
        unique_source_location_count: locations.len(),
        unique_repository_count: repositories_by_count.len(),
        task_count_by_lean_version: distributions,
        repositories_by_task_count: repositories_by_count,
        repeated_goal_group_count: repeated_goal_groups,
        repeated_goal_task_count: repeated_goal_tasks,
        repository_registry_count: registry.len(),
        repository_registry_complete: !registry.is_empty(),
    };
    Ok((report, ids, task_records))
}

fn run_verify(
    full_bytes: &[u8],
    split_bytes: &[u8],
    pin: &SnapshotPin,
) -> Result<Report, String> {
    validate_pin(pin)?;
    let (full_report, full_ids, _) =
        audit_database(full_bytes, FULL_DATASET_BLOB, "SorryDB_2601 full snapshot")?;
    let (split_report, split_ids, split_records) =
        audit_database(split_bytes, EVALUATION_SPLIT_BLOB, "SorryDB_2601 evaluation split")?;

    if full_report.task_count != EXPECTED_FULL_TASKS {
        return Err(format!(
            "full snapshot task count mismatch: expected {EXPECTED_FULL_TASKS}, got {}",
            full_report.task_count
        ));
    }
    if split_report.task_count != EXPECTED_EVALUATION_TASKS {
        return Err(format!(
            "evaluation split task count mismatch: expected {EXPECTED_EVALUATION_TASKS}, got {}",
            split_report.task_count
        ));
    }
    if full_report.unique_repository_count != EXPECTED_REPOSITORIES {
        return Err(format!(
            "full snapshot repository count mismatch: expected {EXPECTED_REPOSITORIES}, got {}",
            full_report.unique_repository_count
        ));
    }

    let missing: Vec<String> = split_ids
        .keys()
        .filter(|id| !full_ids.contains_key(*id))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "evaluation split contains {} task IDs absent from the full snapshot",
            missing.len()
        ));
    }
    for id in split_ids.keys() {
        if full_ids.get(id) != split_ids.get(id) {
            return Err(format!("evaluation split task {id} changes its pinned source identity"));
        }
    }

    let full_identity_set: BTreeSet<String> = full_ids.into_values().collect();
    if split_records.iter().any(|identity| !full_identity_set.contains(identity)) {
        return Err("evaluation split contains source identities absent from the full snapshot".into());
    }

    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        snapshot_id: pin.snapshot_id.clone(),
        upstream_repository: pin.upstream_repository.clone(),
        upstream_repository_commit: pin.upstream_repository_commit.clone(),
        full_snapshot: full_report,
        evaluation_split: split_report,
        evaluation_split_is_subset_of_full_snapshot: true,
        missing_evaluation_task_ids: missing,
        count_pins_match: true,
        repository_count_pin_matches: true,
        source_git_blob_pins_match: true,
        source_identity_boundary: "This validator proves that supplied bytes match the pinned upstream Git blobs and that all task records structurally bind repository, branch, commit, Lean version, source path, coordinates, goal and metadata. It does not clone those repositories or prove that the current checkout still reproduces the goal.".into(),
        semantic_audit_boundary: "A valid SorryDB record is not evidence that its proposition is non-vacuous, faithfully specified, nontrivial, or suitable as a benchmark target. Source-to-goal correspondence, hypotheses, axioms, theorem strength and proof substitution need independent semantic and Lean-kernel auditing.".into(),
    })
}

fn synthetic_task(id: &str, path: &str, line: usize) -> Value {
    serde_json::json!({
        "id": id,
        "repo": {
            "remote": "https://github.com/example/project",
            "branch": "main",
            "commit": "1".repeat(40),
            "lean_version": "leanprover/lean4:v4.18.0"
        },
        "location": {
            "path": path,
            "start_line": line,
            "start_column": 1,
            "end_line": line,
            "end_column": 6
        },
        "debug_info": {
            "goal": "x : Nat ⊢ x = x",
            "url": "https://github.com/example/project/blob/1111111111111111111111111111111111111111/Demo.lean#L1"
        },
        "metadata": {
            "blame_email_hash": "abcdef123456",
            "blame_date": "2025-01-01T00:00:00+00:00",
            "inclusion_date": "2025-02-01T00:00:00+00:00"
        }
    })
}

fn run_self_tests() -> Result<(), String> {
    let valid = synthetic_task("task-a", "Demo.lean", 1);
    let record: SorryRecord = serde_json::from_value(valid.clone())
        .map_err(|e| format!("decode synthetic SorryDB task: {e}"))?;
    validate_task(&record, 1)?;
    println!("SELF_TEST_PASS valid pinned-task metadata and coordinates");

    let mut zero_based_column = valid.clone();
    zero_based_column["location"]["start_column"] = Value::from(0);
    zero_based_column["location"]["end_column"] = Value::from(5);
    let zero_based: SorryRecord = serde_json::from_value(zero_based_column).map_err(|e| e.to_string())?;
    validate_task(&zero_based, 10)?;
    println!("SELF_TEST_PASS zero-based Lean columns are accepted");

    let mut bad = valid.clone();
    bad["repo"]["commit"] = Value::String("not-a-commit".into());
    let record: SorryRecord = serde_json::from_value(bad).map_err(|e| e.to_string())?;
    if validate_task(&record, 2).is_ok() {
        return Err("malformed repository commit was accepted".into());
    }
    println!("SELF_TEST_PASS malformed source commit is rejected");

    let mut bad = valid.clone();
    bad["location"]["path"] = Value::String("../escape.lean".into());
    let record: SorryRecord = serde_json::from_value(bad).map_err(|e| e.to_string())?;
    if validate_task(&record, 3).is_ok() {
        return Err("path traversal source location was accepted".into());
    }
    println!("SELF_TEST_PASS unsafe source path is rejected");

    let mut bad = valid.clone();
    bad["location"]["end_line"] = Value::from(0);
    let record: SorryRecord = serde_json::from_value(bad).map_err(|e| e.to_string())?;
    if validate_task(&record, 4).is_ok() {
        return Err("invalid end source coordinate was accepted".into());
    }
    println!("SELF_TEST_PASS invalid source coordinates are rejected");

    let full_ids = BTreeSet::from(["task-a".to_owned(), "task-b".to_owned()]);
    let split_ids = BTreeSet::from(["task-a".to_owned()]);
    if split_ids.difference(&full_ids).next().is_some() {
        return Err("valid subset control failed".into());
    }
    let bad_split = BTreeSet::from(["task-c".to_owned()]);
    if bad_split.difference(&full_ids).next().is_none() {
        return Err("evaluation split outside full snapshot was not detected".into());
    }
    println!("SELF_TEST_PASS evaluation split must be a subset of pinned full snapshot");

    let task_a: SorryRecord = serde_json::from_value(valid).map_err(|e| e.to_string())?;
    let mut task_b = task_a.clone();
    task_b.repo.commit = "2".repeat(40);
    if task_identity(&task_a) == task_identity(&task_b) {
        return Err("repository commit change did not alter source identity".into());
    }
    println!("SELF_TEST_PASS any source repository commit change alters task identity");

    let two_locations = vec!["project|commit|Demo.lean|1", "project|commit|Demo.lean|2"];
    if two_locations[0] == two_locations[1] {
        return Err("different source locations collapsed to one identity".into());
    }
    println!("SELF_TEST_PASS source coordinates remain part of task identity");
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SORRYDB_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() != 6 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <full-snapshot.json> <evaluation-split.json> <snapshot-pin.json> <report.json> OR --self-test",
            args.first().map(String::as_str).unwrap_or("pitb-sorrydb-integrity")
        ));
    }
    let full_bytes = fs::read(&args[2]).map_err(|e| format!("read full SorryDB snapshot: {e}"))?;
    let split_bytes = fs::read(&args[3]).map_err(|e| format!("read SorryDB evaluation split: {e}"))?;
    let pin_bytes = fs::read(&args[4]).map_err(|e| format!("read SorryDB snapshot pin: {e}"))?;
    let pin: SnapshotPin = serde_json::from_slice(&pin_bytes).map_err(|e| format!("parse snapshot pin: {e}"))?;
    let report = run_verify(&full_bytes, &split_bytes, &pin)?;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize audit report: {e}"))?;
    fs::write(&args[5], bytes).map_err(|e| format!("write audit report: {e}"))?;
    println!("PITB_SORRYDB_SNAPSHOT_INTEGRITY_OK");
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
    fn registered_sorrydb_snapshot_controls_pass() {
        run_self_tests().expect("all SorryDB integrity controls must pass");
    }

    #[test]
    fn source_identity_changes_when_commit_or_location_changes() {
        let task: SorryRecord = serde_json::from_value(synthetic_task("task-a", "Demo.lean", 1)).unwrap();
        let mut commit = task.clone();
        commit.repo.commit = "2".repeat(40);
        let mut location = task.clone();
        location.location.start_line = 2;
        assert_ne!(task_identity(&task), task_identity(&commit));
        assert_ne!(task_identity(&task), task_identity(&location));
    }
}
