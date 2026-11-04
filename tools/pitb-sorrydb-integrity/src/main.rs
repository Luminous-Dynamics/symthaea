use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Component, Path, PathBuf},
    process::{Command, ExitCode},
};

const TASK_SCHEMA: &str = "pitb.sorrydb-task-identity.v0";
const REPORT_SCHEMA: &str = "pitb.sorrydb-integrity-report.v0";
const UPSTREAM_REPOSITORY: &str = "SorryDB/SorryDB";
const UPSTREAM_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const FULL_SNAPSHOT_BLOB: &str = "5fafddac467af6d30d8f5c8731d6edec9c273a47";
const EVAL_SPLIT_BLOB: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";

#[derive(Debug, Clone, Deserialize)]
struct Database {
    sorries: Vec<Value>,
    #[serde(default)]
    repos: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TaskIdentity {
    schema_version: String,
    task_id: String,
    source_database_id: Option<String>,
    repository_remote: String,
    repository_branch: String,
    repository_commit: String,
    lean_version: String,
    source_path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    debug_goal_sha256: String,
    debug_url: Option<String>,
    source_goal_trust: String,
    prop_parent_type_trust: String,
}

#[derive(Debug, Clone, Serialize)]
struct SnapshotProvenance {
    upstream_repository: String,
    upstream_repository_commit: String,
    snapshot_path: String,
    upstream_git_blob_sha1: String,
    raw_input_sha256: String,
    snapshot_kind: String,
    goal_field_policy: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    provenance: SnapshotProvenance,
    task_count: usize,
    unique_task_count: usize,
    unique_repository_commit_count: usize,
    lean_version_count: usize,
    source_database_id_count: usize,
    duplicate_source_database_ids: Vec<String>,
    task_records: Vec<TaskIdentity>,
    scoring_admission: bool,
    interpretation_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct CheckoutVerification {
    schema_version: String,
    task_id: String,
    checked_out_commit: String,
    checked_out_remote: String,
    lean_toolchain_file: String,
    normalized_lean_version: String,
    source_file_sha256: String,
    source_span: String,
    source_span_verified_as_sorry: bool,
    scoring_admission: bool,
    limitation: String,
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

fn sha1_digest(input: &[u8]) -> [u8; 20] {
    let mut data = input.to_vec();
    let bit_length = (data.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_length.to_be_bytes());

    let mut h0 = 0x67452301_u32;
    let mut h1 = 0xefcdab89_u32;
    let mut h2 = 0x98badcfe_u32;
    let mut h3 = 0x10325476_u32;
    let mut h4 = 0xc3d2e1f0_u32;

    for block in data.chunks_exact(64) {
        let mut words = [0_u32; 80];
        for (index, chunk) in block.chunks_exact(4).enumerate().take(16) {
            words[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..80 {
            words[index] = (words[index - 3]
                ^ words[index - 8]
                ^ words[index - 14]
                ^ words[index - 16])
                .rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (index, word) in words.iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5a827999_u32),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1_u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc_u32),
                _ => (b ^ c ^ d, 0xca62c1d6_u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut output = [0_u8; 20];
    for (chunk, word) in output.chunks_exact_mut(4).zip([h0, h1, h2, h3, h4]) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    output
}

fn git_blob_sha1(raw: &[u8]) -> String {
    let header = format!("blob {}\0", raw.len());
    let mut object = Vec::with_capacity(header.len() + raw.len());
    object.extend_from_slice(header.as_bytes());
    object.extend_from_slice(raw);
    sha1_digest(&object)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn require_nonempty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_sha1(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 40) {
        Ok(())
    } else {
        Err(format!("{label} must be a 40-character lowercase Git SHA-1"))
    }
}

fn safe_repo_path(value: &str) -> Result<(), String> {
    require_nonempty(value, "location.path")?;
    if value.starts_with('/') || value.starts_with('\\') || value.contains('\\') {
        return Err("location.path must be a repository-relative POSIX path".into());
    }
    let path = Path::new(value);
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => return Err(format!("location.path contains unsafe component: {value}")),
        }
    }
    if value.split('/').any(|part| part.is_empty() || part == "." || part == "..") {
        return Err(format!("location.path is not canonical: {value}"));
    }
    Ok(())
}

fn normalize_lean_version(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    require_nonempty(trimmed, "repo.lean_version")?;
    let normalized = trimmed
        .rsplit_once(':')
        .map(|(_, suffix)| suffix)
        .unwrap_or(trimmed)
        .trim();
    if !normalized.starts_with('v') || normalized.len() < 2 {
        return Err(format!("unsupported Lean version string {value:?}"));
    }
    if normalized.contains('/') || normalized.contains(' ') {
        return Err(format!("malformed Lean version string {value:?}"));
    }
    Ok(normalized.to_owned())
}

fn canonical_task_id(fields: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(fields)
        .map_err(|e| format!("serialize canonical task identity: {e}"))?;
    Ok(sha256(&bytes))
}

fn get_str<'a>(object: &'a Value, path: &[&str], label: &str) -> Result<&'a str, String> {
    let mut current = object;
    for key in path {
        current = current.get(*key)
            .ok_or_else(|| format!("{label} missing field {key}"))?;
    }
    current.as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn get_usize(object: &Value, key: &str, label: &str) -> Result<usize, String> {
    object.get(key).and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{label}.{key} must be a non-negative integer"))
}

fn build_task(raw: &Value, index: usize) -> Result<TaskIdentity, String> {
    let label = format!("sorries[{index}]");
    let remote = get_str(raw, &["repo", "remote"], &label)?.to_owned();
    if !remote.starts_with("https://github.com/") && !remote.starts_with("git@github.com:") {
        return Err(format!("{label}.repo.remote must be a GitHub HTTPS or SSH remote"));
    }
    if remote.contains('?') || remote.contains('#') {
        return Err(format!("{label}.repo.remote cannot contain query or fragment"));
    }
    let branch = get_str(raw, &["repo", "branch"], &label)?.to_owned();
    let commit = get_str(raw, &["repo", "commit"], &label)?.to_owned();
    require_sha1(&commit, &format!("{label}.repo.commit"))?;
    let lean_version = normalize_lean_version(get_str(raw, &["repo", "lean_version"], &label)?)?;

    let location = raw.get("location")
        .ok_or_else(|| format!("{label}.location missing"))?;
    let source_path = get_str(location, &["path"], &label)?.to_owned();
    safe_repo_path(&source_path)?;
    let start_line = get_usize(location, "start_line", &label)?;
    let start_column = get_usize(location, "start_column", &label)?;
    let end_line = get_usize(location, "end_line", &label)?;
    let end_column = get_usize(location, "end_column", &label)?;
    if start_line == 0 || end_line == 0 || end_line < start_line {
        return Err(format!("{label} has invalid 1-based line coordinates"));
    }
    if end_line == start_line && end_column <= start_column {
        return Err(format!("{label} has an empty or reversed source span"));
    }

    let goal = get_str(raw, &["debug_info", "goal"], &label)?;
    let goal_digest = sha256(goal.as_bytes());
    let debug_url = raw.get("debug_info").and_then(|x| x.get("url"))
        .and_then(Value::as_str).map(str::to_owned);
    let source_database_id = raw.get("id").and_then(Value::as_str).map(str::to_owned);

    let identity_value = json!({
        "repository_remote": remote,
        "repository_branch": branch,
        "repository_commit": commit,
        "lean_version": lean_version,
        "source_path": source_path,
        "start_line": start_line,
        "start_column": start_column,
        "end_line": end_line,
        "end_column": end_column
    });
    let task_id = canonical_task_id(&identity_value)?;

    Ok(TaskIdentity {
        schema_version: TASK_SCHEMA.into(),
        task_id,
        source_database_id,
        repository_remote: identity_value["repository_remote"].as_str().unwrap().to_owned(),
        repository_branch: identity_value["repository_branch"].as_str().unwrap().to_owned(),
        repository_commit: identity_value["repository_commit"].as_str().unwrap().to_owned(),
        lean_version: identity_value["lean_version"].as_str().unwrap().to_owned(),
        source_path: identity_value["source_path"].as_str().unwrap().to_owned(),
        start_line,
        start_column,
        end_line,
        end_column,
        debug_goal_sha256: goal_digest,
        debug_url,
        source_goal_trust: "debug_hint_not_reproduced".into(),
        prop_parent_type_trust: "database_inclusion_claim_not_independently_rechecked".into(),
    })
}

fn validate_database(raw_bytes: &[u8], snapshot_path: &str, blob_sha1: &str) -> Result<Report, String> {
    require_sha1(blob_sha1, "upstream snapshot Git blob SHA-1")?;
    let computed_blob = git_blob_sha1(raw_bytes);
    if computed_blob != blob_sha1 {
        return Err(format!(
            "raw snapshot bytes do not match declared Git blob SHA-1: expected {blob_sha1}, computed {computed_blob}"
        ));
    }
    if snapshot_path.ends_with("SorryDB_2601.json") && blob_sha1 != FULL_SNAPSHOT_BLOB {
        return Err("pinned SorryDB_2601 blob SHA-1 mismatch".into());
    }
    if snapshot_path.ends_with("SorryDB_2601_1000_evaluation_split.json") && blob_sha1 != EVAL_SPLIT_BLOB {
        return Err("pinned SorryDB_2601 evaluation split blob SHA-1 mismatch".into());
    }
    let database: Database = serde_json::from_slice(raw_bytes)
        .map_err(|e| format!("SorryDB snapshot JSON invalid: {e}"))?;
    if database.sorries.is_empty() {
        return Err("SorryDB snapshot contains no sorries".into());
    }

    let mut tasks = Vec::with_capacity(database.sorries.len());
    let mut task_ids = BTreeSet::new();
    let mut source_ids = BTreeMap::<String, usize>::new();
    let mut repo_commits = BTreeSet::new();
    let mut lean_versions = BTreeSet::new();
    for (index, raw) in database.sorries.iter().enumerate() {
        let task = build_task(raw, index)?;
        if !task_ids.insert(task.task_id.clone()) {
            return Err(format!("duplicate canonical task identity {}", task.task_id));
        }
        if let Some(source_id) = &task.source_database_id {
            *source_ids.entry(source_id.clone()).or_default() += 1;
        }
        repo_commits.insert((task.repository_remote.clone(), task.repository_commit.clone()));
        lean_versions.insert(task.lean_version.clone());
        tasks.push(task);
    }
    tasks.sort_by(|a, b| a.task_id.cmp(&b.task_id));
    let duplicates = source_ids.iter()
        .filter(|(_, count)| **count > 1)
        .map(|(id, _)| id.clone())
        .collect::<Vec<_>>();
    let source_id_count = source_ids.len();
    let source_kind = if snapshot_path.ends_with("SorryDB_2601_1000_evaluation_split.json") {
        "frozen_evaluation_split"
    } else if snapshot_path.ends_with("SorryDB_2601.json") {
        "frozen_paper_snapshot"
    } else {
        "dynamic_or_other_snapshot"
    };

    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        provenance: SnapshotProvenance {
            upstream_repository: UPSTREAM_REPOSITORY.into(),
            upstream_repository_commit: UPSTREAM_COMMIT.into(),
            snapshot_path: snapshot_path.to_owned(),
            upstream_git_blob_sha1: blob_sha1.to_owned(),
            raw_input_sha256: sha256(raw_bytes),
            snapshot_kind: source_kind.into(),
            goal_field_policy: "debug_info.goal is retained only as a SHA-256 debugging hint; it is not an authoritative source-level goal until independently reproduced at the pinned source span".into(),
        },
        task_count: tasks.len(),
        unique_task_count: task_ids.len(),
        unique_repository_commit_count: repo_commits.len(),
        lean_version_count: lean_versions.len(),
        source_database_id_count: source_id_count,
        duplicate_source_database_ids: duplicates,
        task_records: tasks,
        scoring_admission: false,
        interpretation_boundary: "Snapshot structure and task coordinates are validated; no repository was cloned, no Lean toolchain was run, no Prop goal was independently reproduced, and no source-to-informal-semantic audit was completed. Tasks remain non-scoring until checkout and proof/audit receipts are independently verified.".into(),
    })
}

fn run_git_bytes(repo: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git").args(args).current_dir(repo).output()
        .map_err(|e| format!("run git {:?}: {e}", args))?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output.stdout)
}

fn run_git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let bytes = run_git_bytes(repo, args)?;
    String::from_utf8(bytes)
        .map(|value| value.trim().to_owned())
        .map_err(|e| format!("git {:?} returned non-UTF-8 output: {e}", args))
}

fn position_to_byte_index(content: &str, line: usize, column: usize) -> Result<usize, String> {
    if line == 0 {
        return Err("source line coordinates are 1-based".into());
    }
    let lines: Vec<&str> = content.split('\n').collect();
    let selected = lines.get(line - 1)
        .ok_or_else(|| format!("source line {line} is out of range"))?;
    if column > selected.chars().count() {
        return Err(format!("source column {column} exceeds line {line} length {}", selected.chars().count()));
    }
    let byte_column = selected.char_indices().nth(column)
        .map(|(byte, _)| byte)
        .unwrap_or(selected.len());
    let prefix_bytes = lines.iter().take(line - 1).map(|s| s.len() + 1).sum::<usize>();
    Ok(prefix_bytes + byte_column)
}

fn normalize_remote(remote: &str) -> String {
    let mut value = remote.trim().to_owned();
    if let Some(rest) = value.strip_prefix("git@github.com:") {
        value = format!("https://github.com/{rest}");
    }
    if value.ends_with(".git") {
        value.truncate(value.len() - 4);
    }
    value.trim_end_matches('/').to_ascii_lowercase()
}

fn verify_checkout(task: &TaskIdentity, repo_root: &Path) -> Result<CheckoutVerification, String> {
    let commit = run_git(repo_root, &["rev-parse", "HEAD"])?;
    if commit != task.repository_commit {
        return Err(format!("checkout commit mismatch: expected {}, got {commit}", task.repository_commit));
    }
    let remote = run_git(repo_root, &["remote", "get-url", "origin"])?;
    if normalize_remote(&remote) != normalize_remote(&task.repository_remote) {
        return Err(format!("checkout remote mismatch: expected {}, got {remote}", task.repository_remote));
    }
    let toolchain_bytes = run_git_bytes(repo_root, &["show", "HEAD:lean-toolchain"])?;
    let toolchain = std::str::from_utf8(&toolchain_bytes)
        .map_err(|e| format!("committed lean-toolchain is not UTF-8: {e}"))?;
    let normalized = normalize_lean_version(toolchain)?;
    if normalized != task.lean_version {
        return Err(format!("Lean toolchain mismatch: expected {}, got {}", task.lean_version, normalized));
    }

    let source_spec = format!("HEAD:{}", task.source_path);
    let source_bytes = run_git_bytes(repo_root, &["show", &source_spec])?;
    let source = std::str::from_utf8(&source_bytes)
        .map_err(|e| format!("committed source file is not UTF-8: {e}"))?;
    let start = position_to_byte_index(source, task.start_line, task.start_column)?;
    let end = position_to_byte_index(source, task.end_line, task.end_column)?;
    if end <= start {
        return Err("source span is empty or reversed".into());
    }
    let span = source.get(start..end).ok_or("source span is not on UTF-8 boundaries")?;
    if span != "sorry" {
        return Err(format!("pinned source span is not exactly sorry: {span:?}"));
    }

    Ok(CheckoutVerification {
        schema_version: "pitb.sorrydb-checkout-verification.v0".into(),
        task_id: task.task_id.clone(),
        checked_out_commit: commit,
        checked_out_remote: remote,
        lean_toolchain_file: toolchain.trim().to_owned(),
        normalized_lean_version: normalized,
        source_file_sha256: sha256(&source_bytes),
        source_span: span.to_owned(),
        source_span_verified_as_sorry: true,
        scoring_admission: false,
        limitation: "This verifies Git identity, Lean toolchain identity and the source token span only. It does not run Lake/Lean, reproduce the elaborated Prop goal, verify a proof, or audit whether the formal statement faithfully captures the intended informal mathematics.".into(),
    })
}

fn validate_task_identity(task: &TaskIdentity) -> Result<(), String> {
    if task.schema_version != TASK_SCHEMA {
        return Err("unsupported SorryDB task identity schema".into());
    }
    require_nonempty(&task.repository_remote, "repository_remote")?;
    if !task.repository_remote.starts_with("https://github.com/")
        && !task.repository_remote.starts_with("git@github.com:")
    {
        return Err("repository_remote must be a GitHub HTTPS or SSH remote".into());
    }
    require_nonempty(&task.repository_branch, "repository_branch")?;
    require_sha1(&task.repository_commit, "repository_commit")?;
    if normalize_lean_version(&task.lean_version)? != task.lean_version {
        return Err("lean_version must be normalized".into());
    }
    safe_repo_path(&task.source_path)?;
    if task.start_line == 0 || task.end_line == 0 || task.end_line < task.start_line {
        return Err("task source line coordinates are invalid".into());
    }
    if task.end_line == task.start_line && task.end_column <= task.start_column {
        return Err("task source span is empty or reversed".into());
    }
    if !valid_hex(&task.debug_goal_sha256, 64) {
        return Err("debug_goal_sha256 must be lowercase SHA-256".into());
    }
    let identity_value = json!({
        "repository_remote": task.repository_remote,
        "repository_branch": task.repository_branch,
        "repository_commit": task.repository_commit,
        "lean_version": task.lean_version,
        "source_path": task.source_path,
        "start_line": task.start_line,
        "start_column": task.start_column,
        "end_line": task.end_line,
        "end_column": task.end_column
    });
    let expected = canonical_task_id(&identity_value)?;
    if task.task_id != expected {
        return Err(format!("task_id does not match canonical identity: expected {expected}"));
    }
    Ok(())
}

fn parse_task_file(path: &Path) -> Result<TaskIdentity, String> {
    let raw = fs::read(path).map_err(|e| format!("read task identity JSON: {e}"))?;
    if let Ok(task) = serde_json::from_slice::<TaskIdentity>(&raw) {
        return Ok(task);
    }
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|e| format!("task identity JSON invalid: {e}"))?;
    build_task(&value, 0)
}

fn synthetic_database() -> Value {
    json!({
        "repos": [],
        "sorries": [{
            "repo": {
                "remote": "https://github.com/example/math",
                "branch": "main",
                "commit": "1".repeat(40),
                "lean_version": "v4.18.0-rc1"
            },
            "location": {
                "path": "Math/Example.lean",
                "start_line": 3,
                "start_column": 6,
                "end_line": 3,
                "end_column": 11
            },
            "debug_info": {
                "goal": "x : Nat\n⊢ x = x",
                "url": "https://github.com/example/math/blob/1111111111111111111111111111111111111111/Math/Example.lean#L3"
            },
            "metadata": {
                "blame_email_hash": "0123456789ab",
                "blame_date": "2026-01-01T00:00:00+00:00",
                "inclusion_date": "2026-02-01T00:00:00+00:00"
            },
            "id": "source-db-id"
        }]
    })
}

fn run_self_tests() -> Result<(), String> {
    let database = synthetic_database();
    let bytes = serde_json::to_vec(&database).map_err(|e| e.to_string())?;
    let synthetic_blob = git_blob_sha1(&bytes);
    if git_blob_sha1(b"abc") != "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f" {
        return Err("Git blob SHA-1 implementation self-test failed".into());
    }
    let report = validate_database(&bytes, "synthetic.json", &synthetic_blob)?;
    if report.task_count != 1 || report.unique_task_count != 1
        || report.task_records[0].source_goal_trust != "debug_hint_not_reproduced"
        || report.scoring_admission
    {
        return Err("synthetic task manifest or trust-boundary control failed".into());
    }
    println!("SELF_TEST_PASS task identity binds repo commit, Lean version and exact source coordinates");

    for path in ["/etc/passwd", "../escape.lean", "Math/../Escape.lean", "Math\\\\Escape.lean"] {
        if safe_repo_path(path).is_ok() {
            return Err(format!("unsafe source path was accepted: {path}"));
        }
    }
    println!("SELF_TEST_PASS absolute, traversal and non-POSIX paths are rejected");

    let mut malformed = database.clone();
    malformed["sorries"][0]["location"]["start_line"] = Value::from(0);
    let malformed_bytes = serde_json::to_vec(&malformed).unwrap();
    if validate_database(&malformed_bytes, "synthetic.json", &git_blob_sha1(&malformed_bytes)).is_ok() {
        return Err("zero-based line in SorryDB input was accepted".into());
    }
    println!("SELF_TEST_PASS line coordinates must be 1-based");

    let mut reversed = database.clone();
    reversed["sorries"][0]["location"]["end_column"] = Value::from(5);
    let reversed_bytes = serde_json::to_vec(&reversed).unwrap();
    if validate_database(&reversed_bytes, "synthetic.json", &git_blob_sha1(&reversed_bytes)).is_ok() {
        return Err("reversed source range was accepted".into());
    }
    println!("SELF_TEST_PASS empty/reversed source range is rejected");

    let mut duplicate = database.clone();
    duplicate["sorries"].push(duplicate["sorries"][0].clone());
    let duplicate_bytes = serde_json::to_vec(&duplicate).unwrap();
    if validate_database(&duplicate_bytes, "synthetic.json", &git_blob_sha1(&duplicate_bytes)).is_ok() {
        return Err("duplicate canonical task identity was accepted".into());
    }
    println!("SELF_TEST_PASS duplicate task identity is rejected");

    let sample = "αβ\n  sorry\n";
    let start = position_to_byte_index(sample, 2, 2)?;
    let end = position_to_byte_index(sample, 2, 7)?;
    if sample.get(start..end) != Some("sorry") {
        return Err("Unicode character columns were not converted to byte offsets correctly".into());
    }
    println!("SELF_TEST_PASS Unicode source coordinates follow SorryDB character-index semantics");

    Ok(())
}

fn write_synthetic_report(report_path: &Path, task_path: &Path) -> Result<(), String> {
    let database = synthetic_database();
    let bytes = serde_json::to_vec(&database).map_err(|e| format!("serialize synthetic database: {e}"))?;
    let blob = git_blob_sha1(&bytes);
    let report = validate_database(&bytes, "synthetic.json", &blob)?;
    let report_bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| format!("serialize synthetic report: {e}"))?;
    fs::write(report_path, report_bytes).map_err(|e| format!("write synthetic report: {e}"))?;
    let task = report.task_records.first().ok_or("synthetic task manifest is empty")?;
    let task_bytes = serde_json::to_vec_pretty(task)
        .map_err(|e| format!("serialize synthetic task identity: {e}"))?;
    fs::write(task_path, task_bytes).map_err(|e| format!("write synthetic task identity: {e}"))?;
    Ok(())
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
        write_synthetic_report(Path::new(&args[2]), Path::new(&args[3]))?;
        println!("PITB_SORRYDB_INTEGRITY_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() == 5 && args[1] == "--validate" {
        let source = fs::read(&args[2]).map_err(|e| format!("read SorryDB snapshot: {e}"))?;
        let report = validate_database(&source, &args[2], &args[3])?;
        let output = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize integrity report: {e}"))?;
        fs::write(&args[4], output).map_err(|e| format!("write integrity report: {e}"))?;
        println!("PITB_SORRYDB_INTEGRITY_OK");
        println!("tasks={} repositories={} lean_versions={}", report.task_count, report.unique_repository_commit_count, report.lean_version_count);
        return Ok(());
    }
    if args.len() == 4 && args[1] == "--verify-checkout" {
        let task = parse_task_file(Path::new(&args[2]))?;
        validate_task_identity(&task)?;
        let result = verify_checkout(&task, Path::new(&args[3]))?;
        println!("{}", serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?);
        println!("PITB_SORRYDB_CHECKOUT_IDENTITY_OK");
        return Ok(());
    }
    Err(format!(
        "usage: {} --self-test | --self-test-report <report.json> <task.json> | --validate <sorrydb.json> <upstream-blob-sha1> <report.json> | --verify-checkout <task.json> <repo-root>",
        args.first().map(String::as_str).unwrap_or("pitb-sorrydb-integrity")
    ))
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
    fn sorrydb_integrity_controls_pass() {
        run_self_tests().expect("all SorryDB integrity controls must pass");
    }

    #[test]
    fn canonical_task_identity_changes_when_repository_commit_changes() {
        let task = build_task(&synthetic_database()["sorries"][0], 0).unwrap();
        let mut changed = synthetic_database();
        changed["sorries"][0]["repo"]["commit"] = Value::String("2".repeat(40));
        let changed_task = build_task(&changed["sorries"][0], 0).unwrap();
        assert_ne!(task.task_id, changed_task.task_id);
    }

    #[test]
    fn debug_goal_is_not_treated_as_authoritative_source_goal() {
        let task = build_task(&synthetic_database()["sorries"][0], 0).unwrap();
        assert_eq!(task.source_goal_trust, "debug_hint_not_reproduced");
        assert_eq!(
            task.prop_parent_type_trust,
            "database_inclusion_claim_not_independently_rechecked"
        );
    }
}
