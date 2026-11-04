use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const REPORT_SCHEMA: &str = "pitb.sorrydb-snapshot-integrity-report.v0";

#[derive(Debug, Clone, Serialize)]
struct RepositorySummary {
    remote: String,
    task_count: usize,
    evaluation_task_count: usize,
    commits: Vec<String>,
    lean_versions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    database_sha256: String,
    evaluation_split_sha256: String,
    database_task_count: usize,
    evaluation_task_count: usize,
    training_pool_task_count: usize,
    repository_count: usize,
    duplicate_task_id_count: usize,
    duplicate_source_location_count: usize,
    evaluation_ids_sha256: String,
    repositories: Vec<RepositorySummary>,
    source_identity_status: String,
    lean_replay_status: String,
    semantic_specification_status: String,
    proof_verification_status: String,
    trust_boundary: String,
    interpretation: String,
}

#[derive(Debug, Clone)]
struct TaskIdentity {
    id: String,
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    goal: String,
    url: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn require_string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("{context}.{key} must be a non-empty string"))
}

fn require_u64(value: &Value, key: &str, context: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("{context}.{key} must be a positive integer"))
}

fn valid_iso8601(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || bytes.get(10) != Some(&b'T')
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
        || !bytes[..19].iter().enumerate().all(|(i, b)| {
            matches!(i, 4 | 7) && *b == b'-'
                || i == 10 && *b == b'T'
                || matches!(i, 13 | 16) && *b == b':'
                || !matches!(i, 4 | 7 | 10 | 13 | 16) && b.is_ascii_digit()
        })
    {
        return false;
    }
    let tail = &value[19..];
    if tail.is_empty() {
        return false;
    }
    let zone_start = tail
        .rfind('+')
        .or_else(|| tail.rfind('-'))
        .or_else(|| tail.rfind('Z'));
    let Some(index) = zone_start else { return false };
    if tail[index..] == "Z" {
        return true;
    }
    let zone = &tail[index..];
    zone.len() == 6
        && (zone.starts_with('+') || zone.starts_with('-'))
        && zone.as_bytes()[3] == b':'
        && zone.as_bytes()[1..3].iter().all(u8::is_ascii_digit)
        && zone.as_bytes()[4..6].iter().all(u8::is_ascii_digit)
}

fn canonical_github_remote(remote: &str) -> Result<String, String> {
    let suffix = if let Some(rest) = remote.strip_prefix("https://github.com/") {
        rest
    } else if let Some(rest) = remote.strip_prefix("http://github.com/") {
        rest
    } else if let Some(rest) = remote.strip_prefix("git@github.com:") {
        rest
    } else {
        return Err(format!("unsupported repository remote {remote}; expected GitHub HTTPS or SSH"));
    };
    let clean = suffix.trim_end_matches(".git").trim_end_matches('/');
    let parts: Vec<&str> = clean.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|p| p.is_empty() || *p == "." || *p == "..")
        || clean.contains('?')
        || clean.contains('#')
    {
        return Err(format!("repository remote {remote} is not a canonical owner/repository URL"));
    }
    Ok(format!("https://github.com/{clean}"))
}

fn validate_task(row: &Value, index: usize) -> Result<TaskIdentity, String> {
    let label = format!("sorries[{index}]");
    let id = require_string(row, "id", &label)?;
    if id.len() > 128 || id.chars().any(char::is_whitespace) {
        return Err(format!("{label}.id must be a compact stable identifier"));
    }
    let repo = row.get("repo").ok_or_else(|| format!("{label}.repo missing"))?;
    let remote = require_string(repo, "remote", &format!("{label}.repo"))?.to_owned();
    let canonical_remote = canonical_github_remote(&remote)?;
    let branch = require_string(repo, "branch", &format!("{label}.repo"))?.to_owned();
    if branch.starts_with('-') || branch.contains('\n') {
        return Err(format!("{label}.repo.branch is malformed"));
    }
    let commit = require_string(repo, "commit", &format!("{label}.repo"))?.to_owned();
    if !valid_hex(&commit, 40) {
        return Err(format!("{label}.repo.commit must be a 40-character lowercase Git commit"));
    }
    let lean_version = require_string(repo, "lean_version", &format!("{label}.repo"))?.to_owned();
    if lean_version.chars().any(char::is_whitespace) {
        return Err(format!("{label}.repo.lean_version contains whitespace"));
    }

    let location = row.get("location").ok_or_else(|| format!("{label}.location missing"))?;
    let path = require_string(location, "path", &format!("{label}.location"))?.to_owned();
    if path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        || !path.ends_with(".lean")
    {
        return Err(format!("{label}.location.path must be a normalized relative .lean path"));
    }
    let start_line = require_u64(location, "start_line", &format!("{label}.location"))?;
    let start_column = require_u64(location, "start_column", &format!("{label}.location"))?;
    let end_line = require_u64(location, "end_line", &format!("{label}.location"))?;
    let end_column = require_u64(location, "end_column", &format!("{label}.location"))?;
    if end_line < start_line || (end_line == start_line && end_column <= start_column) {
        return Err(format!("{label}.location end position must follow its start position"));
    }

    let debug = row.get("debug_info").ok_or_else(|| format!("{label}.debug_info missing"))?;
    let goal = require_string(debug, "goal", &format!("{label}.debug_info"))?.to_owned();
    let url = require_string(debug, "url", &format!("{label}.debug_info"))?.to_owned();
    let expected_url = format!(
        "{canonical_remote}/blob/{commit}/{path}#L{start_line}"
    );
    if url != expected_url {
        return Err(format!(
            "{label}.debug_info.url does not resolve to the declared remote, commit, path and line"
        ));
    }

    let metadata = row.get("metadata").ok_or_else(|| format!("{label}.metadata missing"))?;
    require_string(metadata, "blame_email_hash", &format!("{label}.metadata"))?;
    let blame_date = require_string(metadata, "blame_date", &format!("{label}.metadata"))?;
    let inclusion_date = require_string(metadata, "inclusion_date", &format!("{label}.metadata"))?;
    if !valid_iso8601(blame_date) || !valid_iso8601(inclusion_date) {
        return Err(format!("{label}.metadata dates must be timezone-qualified ISO-8601 timestamps"));
    }

    Ok(TaskIdentity {
        id: id.to_owned(),
        remote,
        branch,
        commit,
        lean_version,
        path,
        start_line,
        start_column,
        end_line,
        end_column,
        goal,
        url,
    })
}

fn records(root: &Value, label: &str) -> Result<&Vec<Value>, String> {
    if let Some(rows) = root.as_array() {
        return Ok(rows);
    }
    root.get("sorries")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} must be an array or an object with a sorries array"))
}

fn source_location_key(task: &TaskIdentity) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}",
        task.remote,
        task.commit,
        task.lean_version,
        task.path,
        task.start_line,
        task.start_column,
        task.end_column
    )
}

fn audit(database_bytes: &[u8], split_bytes: &[u8]) -> Result<Report, String> {
    let database: Value = serde_json::from_slice(database_bytes)
        .map_err(|e| format!("database JSON invalid: {e}"))?;
    let split: Value = serde_json::from_slice(split_bytes)
        .map_err(|e| format!("evaluation split JSON invalid: {e}"))?;
    let db_rows = records(&database, "database")?;
    let split_rows = records(&split, "evaluation split")?;
    if db_rows.is_empty() || split_rows.is_empty() {
        return Err("database and evaluation split must both contain tasks".into());
    }

    let mut by_id = BTreeMap::<String, (TaskIdentity, Value)>::new();
    let mut seen_locations = BTreeSet::new();
    let mut repo_tasks = BTreeMap::<String, (usize, BTreeSet<String>, BTreeSet<String>)>::new();
    let mut duplicate_location_count = 0usize;
    for (index, row) in db_rows.iter().enumerate() {
        let task = validate_task(row, index)?;
        let location_key = source_location_key(&task);
        if !seen_locations.insert(location_key) {
            duplicate_location_count += 1;
            return Err(format!(
                "duplicate source location at task {} (remote, commit, Lean version and coordinates match)",
                task.id
            ));
        }
        if by_id.insert(task.id.clone(), (task.clone(), row.clone())).is_some() {
            return Err(format!("duplicate task id {}", task.id));
        }
        let entry = repo_tasks
            .entry(task.remote.clone())
            .or_insert_with(|| (0, BTreeSet::new(), BTreeSet::new()));
        entry.0 += 1;
        entry.1.insert(task.commit.clone());
        entry.2.insert(task.lean_version.clone());
    }

    let mut evaluation_ids = BTreeSet::new();
    let mut evaluation_repo_counts = BTreeMap::<String, usize>::new();
    for (index, row) in split_rows.iter().enumerate() {
        let task = validate_task(row, index)?;
        if !evaluation_ids.insert(task.id.clone()) {
            return Err(format!("duplicate evaluation task id {}", task.id));
        }
        let Some((database_task, database_row)) = by_id.get(&task.id) else {
            return Err(format!(
                "evaluation task {} is absent from the frozen full database",
                task.id
            ));
        };
        if database_task.commit != task.commit
            || database_task.path != task.path
            || database_task.start_line != task.start_line
            || database_task.start_column != task.start_column
            || database_task.end_line != task.end_line
            || database_task.end_column != task.end_column
            || database_task.goal != task.goal
            || database_task.url != task.url
            || database_task.lean_version != task.lean_version
            || database_task.remote != task.remote
            || database_row != row
        {
            return Err(format!(
                "evaluation task {} differs from the exact full-database source record",
                task.id
            ));
        }
        *evaluation_repo_counts.entry(task.remote).or_default() += 1;
    }

    let repositories = repo_tasks
        .into_iter()
        .map(|(remote, (task_count, commits, versions))| RepositorySummary {
            evaluation_task_count: *evaluation_repo_counts.get(&remote).unwrap_or(&0),
            remote,
            task_count,
            commits: commits.into_iter().collect(),
            lean_versions: versions.into_iter().collect(),
        })
        .collect::<Vec<_>>();
    let mut ids_for_hash = evaluation_ids.iter().cloned().collect::<Vec<_>>();
    ids_for_hash.sort();
    let id_bytes = serde_json::to_vec(&ids_for_hash)
        .map_err(|e| format!("encode evaluation id set: {e}"))?;
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        database_sha256: sha256(database_bytes),
        evaluation_split_sha256: sha256(split_bytes),
        database_task_count: db_rows.len(),
        evaluation_task_count: split_rows.len(),
        training_pool_task_count: db_rows.len() - evaluation_ids.len(),
        repository_count: repositories.len(),
        duplicate_task_id_count: 0,
        duplicate_source_location_count: duplicate_location_count,
        evaluation_ids_sha256: sha256(&id_bytes),
        repositories,
        source_identity_status: "database_and_evaluation_split_consistent".into(),
        lean_replay_status: "not_performed".into(),
        semantic_specification_status: "not_performed".into(),
        proof_verification_status: "not_performed".into(),
        trust_boundary: "This report verifies JSON structure, pinned source identities, task IDs, task locations and exact evaluation-split membership. It does not clone repositories, replay Lean goals, establish that a goal has type Prop, audit axioms, prove specification fidelity, or verify candidate proofs. Those require independent source-replay and proof-verification receipts.".into(),
        interpretation: "A task is reproducible only relative to its exact repository commit, Lean toolchain, source path and coordinates. Any change to those inputs creates a different task identity. The evaluation split is sealed by its exact byte digest and task-ID-set digest.".into(),
    })
}

fn fixture_task(id: &str, remote: &str, commit: &str, path: &str, line: u64, goal: &str) -> Value {
    json!({
        "repo": {
            "remote": remote,
            "branch": "main",
            "commit": commit,
            "lean_version": "v4.22.0"
        },
        "location": {
            "path": path,
            "start_line": line,
            "start_column": 2,
            "end_line": line,
            "end_column": 7
        },
        "debug_info": {
            "goal": goal,
            "url": format!("{remote}/blob/{commit}/{path}#L{line}")
        },
        "metadata": {
            "blame_email_hash": "synthetic-blame-hash",
            "blame_date": "2026-01-01T00:00:00+00:00",
            "inclusion_date": "2026-01-02T00:00:00+00:00"
        },
        "id": id
    })
}

fn synthetic_inputs() -> (Vec<u8>, Vec<u8>) {
    let a = fixture_task(
        "task-a",
        "https://github.com/example/LeanProject",
        &"1".repeat(40),
        "Project/A.lean",
        10,
        "⊢ True",
    );
    let b = fixture_task(
        "task-b",
        "https://github.com/example/LeanProject",
        &"1".repeat(40),
        "Project/B.lean",
        20,
        "n : Nat\n⊢ n = n",
    );
    let c = fixture_task(
        "task-c",
        "https://github.com/example/OtherProject",
        &"2".repeat(40),
        "Other/C.lean",
        30,
        "⊢ 1 = 1",
    );
    let database = json!({"repos": [], "sorries": [a.clone(), b.clone(), c.clone()]});
    let split = json!({"sorries": [a, c]});
    (
        serde_json::to_vec_pretty(&database).unwrap(),
        serde_json::to_vec_pretty(&split).unwrap(),
    )
}

fn run_self_tests() -> Result<(), String> {
    let (db, split) = synthetic_inputs();
    let report = audit(&db, &split)?;
    if report.database_task_count != 3
        || report.evaluation_task_count != 2
        || report.training_pool_task_count != 1
        || report.repository_count != 2
        || report.source_identity_status != "database_and_evaluation_split_consistent"
    {
        return Err("synthetic snapshot and split accounting failed".into());
    }
    println!("SELF_TEST_PASS exact repository, commit, Lean version and source coordinates");

    let mut changed_split: Value = serde_json::from_slice(&split).unwrap();
    changed_split["sorries"][0]["debug_info"]["goal"] = Value::String("⊢ False".into());
    let changed = serde_json::to_vec(&changed_split).unwrap();
    if audit(&db, &changed).is_ok() {
        return Err("changed evaluation goal under same task ID was accepted".into());
    }
    println!("SELF_TEST_PASS evaluation goal drift under a stable task ID is rejected");

    let mut missing_split: Value = serde_json::from_slice(&split).unwrap();
    missing_split["sorries"][0]["id"] = Value::String("not-in-database".into());
    let missing = serde_json::to_vec(&missing_split).unwrap();
    if audit(&db, &missing).is_ok() {
        return Err("evaluation task absent from full snapshot was accepted".into());
    }
    println!("SELF_TEST_PASS evaluation tasks must be exact members of the full snapshot");

    let mut bad_location: Value = serde_json::from_slice(&db).unwrap();
    bad_location["sorries"][0]["location"]["path"] = Value::String("../escape.lean".into());
    let bad = serde_json::to_vec(&bad_location).unwrap();
    if audit(&bad, &split).is_ok() {
        return Err("path traversal in task location was accepted".into());
    }
    println!("SELF_TEST_PASS source path traversal is rejected");

    let mut bad_commit: Value = serde_json::from_slice(&db).unwrap();
    bad_commit["sorries"][0]["repo"]["commit"] = Value::String("moving-branch".into());
    let bad = serde_json::to_vec(&bad_commit).unwrap();
    if audit(&bad, &split).is_ok() {
        return Err("moving branch was accepted as a source commit".into());
    }
    println!("SELF_TEST_PASS task source requires immutable Git commit identity");

    let mut bad_url: Value = serde_json::from_slice(&db).unwrap();
    bad_url["sorries"][0]["debug_info"]["url"] =
        Value::String("https://github.com/example/LeanProject/blob/main/Project/A.lean#L10".into());
    let bad = serde_json::to_vec(&bad_url).unwrap();
    if audit(&bad, &split).is_ok() {
        return Err("source URL pointing at a branch rather than pinned commit was accepted".into());
    }
    println!("SELF_TEST_PASS debug URL must resolve to the pinned commit and coordinates");

    let mut duplicate: Value = serde_json::from_slice(&db).unwrap();
    duplicate["sorries"][1]["id"] = Value::String("task-a".into());
    let bad = serde_json::to_vec(&duplicate).unwrap();
    if audit(&bad, &split).is_ok() {
        return Err("duplicate full-snapshot task ID was accepted".into());
    }
    println!("SELF_TEST_PASS duplicate task identities are rejected");

    Ok(())
}

fn synthetic_report() -> Result<Report, String> {
    let (db, split) = synthetic_inputs();
    audit(&db, &split)
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SORRYDB_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let report = synthetic_report()?;
        let bytes = serde_json::to_vec_pretty(&report)
            .map_err(|e| format!("serialize synthetic report: {e}"))?;
        fs::write(&args[2], bytes).map_err(|e| format!("write synthetic report: {e}"))?;
        println!("PITB_SORRYDB_INTEGRITY_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() != 5 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <sorrydb.json> <evaluation-split.json> <report.json> OR --self-test OR --self-test-report <report.json>",
            args.first().map(String::as_str).unwrap_or("pitb-sorrydb-integrity")
        ));
    }
    let database_bytes =
        fs::read(&args[2]).map_err(|e| format!("read SorryDB database: {e}"))?;
    let split_bytes =
        fs::read(&args[3]).map_err(|e| format!("read evaluation split: {e}"))?;
    let report = audit(&database_bytes, &split_bytes)?;
    let output = serde_json::to_vec_pretty(&report)
        .map_err(|e| format!("serialize integrity report: {e}"))?;
    fs::write(&args[4], output).map_err(|e| format!("write integrity report: {e}"))?;
    println!("PITB_SORRYDB_INTEGRITY_OK");
    println!(
        "database_tasks={} evaluation_tasks={} repositories={}",
        report.database_task_count, report.evaluation_task_count, report.repository_count
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
    fn registered_sorrydb_integrity_controls_pass() {
        run_self_tests().expect("SorryDB integrity controls must pass");
    }

    #[test]
    fn evaluation_split_digest_changes_when_membership_changes() {
        let (db, split) = synthetic_inputs();
        let original = audit(&db, &split).unwrap();
        let mut changed: Value = serde_json::from_slice(&split).unwrap();
        changed["sorries"].as_array_mut().unwrap().pop();
        let changed = serde_json::to_vec(&changed).unwrap();
        let revised = audit(&db, &changed).unwrap();
        assert_ne!(original.evaluation_split_sha256, revised.evaluation_split_sha256);
        assert_ne!(original.evaluation_ids_sha256, revised.evaluation_ids_sha256);
    }
}
