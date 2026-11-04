use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

const REPORT_SCHEMA: &str = "pitb.sorrydb-snapshot-integrity-report.v0";
const SORRYDB_REPOSITORY_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const SORRYDB_SNAPSHOT_ID: &str = "SorryDB-2601";
const SNAPSHOT_BLOB_SHA1: &str = "5fafddac467af6d30d8f5c8731d6edec9c273a47";
const EVALUATION_SPLIT_BLOB_SHA1: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const EXPECTED_SNAPSHOT_RECORDS: usize = 2601;
const EXPECTED_EVALUATION_RECORDS: usize = 1000;

#[derive(Debug, Clone, Serialize)]
struct RecordStats {
    record_count: usize,
    unique_id_count: usize,
    computed_id_match_count: usize,
    unique_repository_count: usize,
    unique_goal_count: usize,
    unique_source_location_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    benchmark: String,
    snapshot_id: String,
    upstream_repository: String,
    upstream_repository_commit: String,
    snapshot_file_path: String,
    snapshot_git_blob_sha1: String,
    snapshot_raw_sha256: String,
    evaluation_split_file_path: String,
    evaluation_split_git_blob_sha1: String,
    evaluation_split_raw_sha256: String,
    sorrydb_id_algorithm: String,
    full_snapshot: RecordStats,
    evaluation_split: RecordStats,
    evaluation_ids_subset_of_snapshot: bool,
    repository_registry_count: usize,
    split_repositories_subset_of_snapshot: bool,
    structural_integrity_passed: bool,
    source_checkout_reproduction_verified: bool,
    goal_to_source_semantic_fidelity_verified: bool,
    proof_solution_integrity_verified: bool,
    score_baseline: BTreeMap<String, String>,
    trust_boundary: String,
    next_qualification_requirements: Vec<String>,
}

fn fail(message: impl AsRef<str>) -> ExitCode {
    eprintln!("INVALID: {}", message.as_ref());
    ExitCode::FAILURE
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn sha1(bytes: &[u8]) -> [u32; 5] {
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let padded_len = ((bytes.len() + 9 + 63) / 64) * 64;
    let mut padded = vec![0_u8; padded_len];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded[bytes.len()] = 0x80;
    padded[padded_len - 8..].copy_from_slice(&bit_len.to_be_bytes());
    let mut state = [0x67452301_u32, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];

    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 80];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset], chunk[offset + 1], chunk[offset + 2], chunk[offset + 3],
            ]);
        }
        for index in 16..80 {
            words[index] = (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = state;
        for (index, word) in words.iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5a827999),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let temp = a.rotate_left(5).wrapping_add(function).wrapping_add(e)
                .wrapping_add(constant).wrapping_add(*word);
            e = d; d = c; c = b.rotate_left(30); b = a; a = temp;
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
    }
    state
}

fn git_blob_sha1(bytes: &[u8]) -> String {
    let header = format!("blob {}\0", bytes.len());
    let mut object = Vec::with_capacity(header.len() + bytes.len());
    object.extend_from_slice(header.as_bytes());
    object.extend_from_slice(bytes);
    let digest = sha1(&object);
    let mut raw = [0_u8; 20];
    for (index, word) in digest.iter().enumerate() {
        raw[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    hex(&raw)
}

fn python_json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch if (ch as u32) < 0x20 => output.push_str(&format!("\\u{:04x}", ch as u32)),
            ch if (ch as u32) >= 0x7f && (ch as u32) <= 0xffff => output.push_str(&format!("\\u{:04x}", ch as u32)),
            ch if (ch as u32) > 0xffff => {
                let scalar = ch as u32 - 0x1_0000;
                let high = 0xd800 + ((scalar >> 10) & 0x3ff);
                let low = 0xdc00 + (scalar & 0x3ff);
                output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}

// Mirrors Python json.dumps(sort_keys=True) defaults: ensure_ascii=true and
// spaces after separators. SorryDB hashes the dataclass without id and inclusion_date.
fn python_json_dumps(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".into()),
        Value::Bool(value) => Ok(if *value { "true" } else { "false" }.into()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => Ok(python_json_string(value)),
        Value::Array(values) => {
            let mut encoded = Vec::with_capacity(values.len());
            for item in values { encoded.push(python_json_dumps(item)?); }
            Ok(format!("[{}]", encoded.join(", ")))
        }
        Value::Object(object) => {
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            let mut encoded = Vec::with_capacity(keys.len());
            for key in keys {
                let item = object.get(key).ok_or("object key disappeared")?;
                encoded.push(format!("{}: {}", python_json_string(key), python_json_dumps(item)?));
            }
            Ok(format!("{{{}}}", encoded.join(", ")))
        }
    }
}

fn sorry_id(record: &Value) -> Result<String, String> {
    // Reconstruct the exact dataclass payload, ignoring any non-dataclass fields.
    let metadata = &record["metadata"];
    let hash_value = serde_json::json!({
        "repo": record["repo"],
        "location": record["location"],
        "debug_info": record["debug_info"],
        "metadata": {
            "blame_email_hash": metadata["blame_email_hash"],
            "blame_date": metadata["blame_date"]
        }
    });
    let encoded = python_json_dumps(&hash_value)?;
    Ok(sha256(encoded.as_bytes()))
}

fn is_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn non_empty<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value.as_str().filter(|s| !s.trim().is_empty()).ok_or_else(|| format!("{label} must be a non-empty string"))
}
fn valid_iso8601(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20 || !bytes.get(0..4).is_some_and(|s| s.iter().all(u8::is_ascii_digit))
        || bytes.get(4) != Some(&b'-') || bytes.get(7) != Some(&b'-')
        || !matches!(bytes.get(10), Some(&b'T') | Some(&b't') | Some(&b' '))
        || bytes.get(13) != Some(&b':') || bytes.get(16) != Some(&b':') { return false; }
    for (start, end) in [(5,7),(8,10),(11,13),(14,16),(17,19)] {
        if !bytes.get(start..end).is_some_and(|p| p.iter().all(u8::is_ascii_digit)) { return false; }
    }
    let month = value[5..7].parse::<u32>().unwrap_or(0);
    let day = value[8..10].parse::<u32>().unwrap_or(0);
    let hour = value[11..13].parse::<u32>().unwrap_or(99);
    let minute = value[14..16].parse::<u32>().unwrap_or(99);
    let second = value[17..19].parse::<u32>().unwrap_or(99);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 { return false; }
    let suffix = &value[19..];
    if suffix == "Z" || suffix == "z" { return true; }
    let suffix = suffix.strip_prefix('.').unwrap_or(suffix);
    let suffix = if let Some(i) = suffix.find('+') { &suffix[i..] } else if let Some(i) = suffix.find('-') { &suffix[i..] } else { return false; };
    suffix.len() == 6 && (suffix.as_bytes()[0] == b'+' || suffix.as_bytes()[0] == b'-')
        && suffix.as_bytes()[3] == b':'
        && suffix.as_bytes()[1..3].iter().all(u8::is_ascii_digit)
        && suffix.as_bytes()[4..6].iter().all(u8::is_ascii_digit)
}

fn validate_record(record: &Value, index: usize) -> Result<(String, String, String), String> {
    let label = format!("sorries[{index}]");
    let obj = record.as_object().ok_or_else(|| format!("{label} must be an object"))?;
    for key in ["repo","location","debug_info","metadata","id"] {
        if !obj.contains_key(key) { return Err(format!("{label} missing {key}")); }
    }
    let repo = &record["repo"];
    let remote = non_empty(&repo["remote"], &format!("{label}.repo.remote"))?;
    if !(remote.starts_with("https://github.com/") || remote.starts_with("https://gitlab.com/") || remote.starts_with("git@github.com:")) {
        return Err(format!("{label}.repo.remote is not a supported public Git remote"));
    }
    let branch = non_empty(&repo["branch"], &format!("{label}.repo.branch"))?;
    let commit = non_empty(&repo["commit"], &format!("{label}.repo.commit"))?;
    if !is_sha1(commit) { return Err(format!("{label}.repo.commit must be a full lowercase 40-character Git SHA-1")); }
    let lean_version = non_empty(&repo["lean_version"], &format!("{label}.repo.lean_version"))?;

    let location = &record["location"];
    let path = non_empty(&location["path"], &format!("{label}.location.path"))?;
    if path.starts_with('/') || path.contains('\\') || path.split('/').any(|p| p == "." || p == "..") {
        return Err(format!("{label}.location.path must be a normalized relative POSIX path"));
    }
    let start_line = location["start_line"].as_u64().ok_or_else(|| format!("{label}.location.start_line must be an integer"))?;
    let start_column = location["start_column"].as_u64().ok_or_else(|| format!("{label}.location.start_column must be non-negative"))?;
    let end_line = location["end_line"].as_u64().ok_or_else(|| format!("{label}.location.end_line must be an integer"))?;
    let end_column = location["end_column"].as_u64().ok_or_else(|| format!("{label}.location.end_column must be non-negative"))?;
    if start_line == 0 || end_line < start_line || (end_line == start_line && end_column < start_column) {
        return Err(format!("{label}.location source range is invalid"));
    }

    let goal = non_empty(&record["debug_info"]["goal"], &format!("{label}.debug_info.goal"))?;
    let url = non_empty(&record["debug_info"]["url"], &format!("{label}.debug_info.url"))?;
    if !url.contains(commit) || !url.contains(path) || !url.ends_with(&format!("#L{start_line}")) {
        return Err(format!("{label}.debug_info.url does not bind commit, path and start line"));
    }
    let metadata = &record["metadata"];
    non_empty(&metadata["blame_email_hash"], &format!("{label}.metadata.blame_email_hash"))?;
    let blame_date = non_empty(&metadata["blame_date"], &format!("{label}.metadata.blame_date"))?;
    let inclusion_date = non_empty(&metadata["inclusion_date"], &format!("{label}.metadata.inclusion_date"))?;
    if !valid_iso8601(blame_date) || !valid_iso8601(inclusion_date) {
        return Err(format!("{label}.metadata dates must be timezone-aware ISO-8601"));
    }
    let id = non_empty(&record["id"], &format!("{label}.id"))?;
    if !is_sha256(id) { return Err(format!("{label}.id must be 64-character lowercase SHA-256")); }
    let computed = sorry_id(record)?;
    if id != computed { return Err(format!("{label}.id does not match SorryDB canonical record hash")); }

    let repo_key = format!("{remote}|{branch}|{commit}|{lean_version}");
    let goal_digest = sha256(goal.as_bytes());
    let location_key = format!("{remote}|{commit}|{path}|{start_line}|{start_column}|{end_line}|{end_column}");
    Ok((repo_key, goal_digest, location_key))
}

fn records_array<'a>(value: &'a Value, label: &str) -> Result<&'a Vec<Value>, String> {
    if let Some(array) = value.as_array() { return Ok(array); }
    value.get("sorries").and_then(Value::as_array).ok_or_else(|| format!("{label} must be array or object with sorries"))
}

fn validate_records(records: &[Value], label: &str) -> Result<(RecordStats, BTreeSet<String>), String> {
    if records.is_empty() { return Err(format!("{label} contains no sorry records")); }
    let mut ids = BTreeSet::new();
    let mut repos = BTreeSet::new();
    let mut goals = BTreeSet::new();
    let mut locations = BTreeSet::new();
    for (index, record) in records.iter().enumerate() {
        let (repo, goal, location) = validate_record(record, index)?;
        let id = record["id"].as_str().ok_or("validated record id changed type")?.to_owned();
        if !ids.insert(id.clone()) { return Err(format!("{label} contains duplicate sorry ID {id}")); }
        repos.insert(repo); goals.insert(goal); locations.insert(location);
    }
    let count = records.len();
    let stats = RecordStats {
        record_count: count,
        unique_id_count: ids.len(),
        computed_id_match_count: ids.len(),
        unique_repository_count: repos.len(),
        unique_goal_count: goals.len(),
        unique_source_location_count: locations.len(),
    };
    Ok((stats, ids))
}

fn validate_database(database_bytes: &[u8], evaluation_bytes: &[u8]) -> Result<Report, String> {
    let database_blob = git_blob_sha1(database_bytes);
    let evaluation_blob = git_blob_sha1(evaluation_bytes);
    if database_blob != SNAPSHOT_BLOB_SHA1 {
        return Err(format!("SorryDB snapshot Git blob mismatch: expected {SNAPSHOT_BLOB_SHA1}, got {database_blob}"));
    }
    if evaluation_blob != EVALUATION_SPLIT_BLOB_SHA1 {
        return Err(format!("SorryDB evaluation split Git blob mismatch: expected {EVALUATION_SPLIT_BLOB_SHA1}, got {evaluation_blob}"));
    }
    let database: Value = serde_json::from_slice(database_bytes).map_err(|e| format!("SorryDB database JSON invalid: {e}"))?;
    let evaluation: Value = serde_json::from_slice(evaluation_bytes).map_err(|e| format!("evaluation JSON invalid: {e}"))?;
    let full = database["sorries"].as_array().ok_or("database must contain sorries array")?;
    let split = records_array(&evaluation, "evaluation split")?;
    if full.len() != EXPECTED_SNAPSHOT_RECORDS || split.len() != EXPECTED_EVALUATION_RECORDS {
        return Err(format!("expected {EXPECTED_SNAPSHOT_RECORDS} full tasks and {EXPECTED_EVALUATION_RECORDS} evaluation tasks; found {} and {}", full.len(), split.len()));
    }
    let repo_registry_count = database["repos"].as_array().ok_or("database must contain repos array")?.len();
    let (full_stats, full_ids) = validate_records(full, "full snapshot")?;
    let (split_stats, split_ids) = validate_records(split, "evaluation split")?;
    let split_ids_subset = split_ids.iter().all(|id| full_ids.contains(id));
    if !split_ids_subset { return Err("evaluation split contains IDs absent from full snapshot".into()); }

    let full_repositories: BTreeSet<String> = full.iter().map(|r| format!("{}|{}", r["repo"]["remote"].as_str().unwrap_or(""), r["repo"]["commit"].as_str().unwrap_or(""))).collect();
    let split_repositories: BTreeSet<String> = split.iter().map(|r| format!("{}|{}", r["repo"]["remote"].as_str().unwrap_or(""), r["repo"]["commit"].as_str().unwrap_or(""))).collect();
    let split_repositories_subset = split_repositories.iter().all(|key| full_repositories.contains(key));
    if !split_repositories_subset { return Err("evaluation split includes repository commits absent from full snapshot".into()); }

    let mut baseline = BTreeMap::new();
    baseline.insert("trivial_pass_at_1".into(), "2.1%".into());
    baseline.insert("tactics_pass_at_1".into(), "8.4%".into());
    baseline.insert("gpt_5_2_pass_at_1".into(), "6.2%".into());
    baseline.insert("gemini_flash_3_agentic_pass_at_1".into(), "30.3%".into());
    baseline.insert("combined_pass_at_1".into(), "35.7%".into());

    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        benchmark: "SorryDB".into(),
        snapshot_id: SORRYDB_SNAPSHOT_ID.into(),
        upstream_repository: "SorryDB/SorryDB".into(),
        upstream_repository_commit: SORRYDB_REPOSITORY_COMMIT.into(),
        snapshot_file_path: "data/SorryDB_2601/SorryDB_2601.json".into(),
        snapshot_git_blob_sha1: database_blob,
        snapshot_raw_sha256: sha256(database_bytes),
        evaluation_split_file_path: "data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json".into(),
        evaluation_split_git_blob_sha1: evaluation_blob,
        evaluation_split_raw_sha256: sha256(evaluation_bytes),
        sorrydb_id_algorithm: "sha256(python-json-dumps-sort-keys-ensure-ascii-default-separators(record excluding id and metadata.inclusion_date))".into(),
        full_snapshot: full_stats,
        evaluation_split: split_stats,
        evaluation_ids_subset_of_snapshot: split_ids_subset,
        repository_registry_count: repo_registry_count,
        split_repositories_subset_of_snapshot: split_repositories_subset,
        structural_integrity_passed: true,
        source_checkout_reproduction_verified: false,
        goal_to_source_semantic_fidelity_verified: false,
        proof_solution_integrity_verified: false,
        score_baseline: baseline,
        trust_boundary: "This checks exact source-file Git blob identities, record structure, task hashes, uniqueness and evaluation-split membership. It does not build each Lean repository, reproduce the goal at source coordinates, audit vacuity/specification fidelity or verify generated proofs.".into(),
        next_qualification_requirements: vec![
            "Materialize each source repository at the exact recorded commit and Lean toolchain; run the pinned Lake build.".into(),
            "Use Lean REPL or an independent extractor to reproduce the sorry range and recorded goal.".into(),
            "Record the fully qualified declaration type, target statement AST, local context, imports and environment digest.".into(),
            "Audit non-vacuity, hypothesis sanity, sorryAx and unsound axioms.".into(),
            "Replace only the designated sorry span and independently replay the exact target and clean repository build.".into(),
            "Separate by repository and temporal cohort to test transfer beyond known projects and prior public tasks.".into(),
        ],
    })
}

fn synthetic_record() -> Result<Value, String> {
    let commit = "1".repeat(40);
    let mut record = serde_json::json!({
        "repo": {"remote":"https://github.com/example/formal-project","branch":"main","commit":commit,"lean_version":"v4.22.0"},
        "location": {"path":"Example/Basic.lean","start_line":12,"start_column":4,"end_line":12,"end_column":9},
        "debug_info": {"goal":"n : Nat\n⊢ n = n","url":format!("https://github.com/example/formal-project/blob/{commit}/Example/Basic.lean#L12")},
        "metadata": {"blame_email_hash":"0123456789ab","blame_date":"2025-01-01T12:00:00+00:00","inclusion_date":"2025-02-01T00:00:00+00:00"},
        "id":""
    });
    record["id"] = Value::String(sorry_id(&record)?);
    Ok(record)
}

fn run_self_tests() -> Result<(), String> {
    let record = synthetic_record()?;
    if record["id"].as_str() != Some("3370c6575a3bb3575b5467327c63e27b70736d45feed637d6e3557e85f52650d") {
        return Err("Python-compatible SorryDB ID calculation failed".into());
    }
    validate_record(&record, 0)?;
    println!("SELF_TEST_PASS Python-compatible task ID with Unicode goal escaping");

    let mut inclusion = record.clone();
    inclusion["metadata"]["inclusion_date"] = Value::String("2026-11-04T00:00:00+00:00".into());
    if sorry_id(&inclusion)? != record["id"].as_str().unwrap() { return Err("inclusion_date changed stable SorryDB ID".into()); }
    println!("SELF_TEST_PASS inclusion_date excluded from identity");

    let mut goal = record.clone();
    goal["debug_info"]["goal"] = Value::String("⊢ False".into());
    if sorry_id(&goal)? == record["id"].as_str().unwrap() { return Err("goal mutation did not change task ID".into()); }
    println!("SELF_TEST_PASS goal change mutates task identity");

    let mut bad_url = record.clone();
    bad_url["debug_info"]["url"] = Value::String("https://github.com/example/formal-project/blob/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Example/Basic.lean#L12".into());
    if validate_record(&bad_url, 0).is_ok() { return Err("wrong source commit URL was accepted".into()); }
    println!("SELF_TEST_PASS source URL binds commit and coordinates");

    let mut bad_path = record.clone();
    bad_path["location"]["path"] = Value::String("../outside.lean".into());
    if validate_record(&bad_path, 0).is_ok() { return Err("path traversal was accepted".into()); }
    println!("SELF_TEST_PASS path traversal rejected");

    let mut bad_position = record.clone();
    bad_position["location"]["start_line"] = Value::Number(0.into());
    if validate_record(&bad_position, 0).is_ok() { return Err("zero source line was accepted".into()); }
    println!("SELF_TEST_PASS invalid source coordinate rejected");

    if git_blob_sha1(b"") != "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        || git_blob_sha1(b"hello\n") != "ce013625030ba8dba906f756967f9e9ca394464a" {
        return Err("Git blob SHA-1 implementation failed known vectors".into());
    }
    println!("SELF_TEST_PASS Git blob hashing matches known vectors");
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SORRYDB_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() != 5 || args[1] != "--validate" {
        return Err(format!("usage: {} --validate <snapshot.json> <evaluation-split.json> <report.json> OR --self-test", args.first().map(String::as_str).unwrap_or("pitb-sorrydb-integrity")));
    }
    let snapshot_bytes = fs::read(&args[2]).map_err(|e| format!("read SorryDB snapshot: {e}"))?;
    let split_bytes = fs::read(&args[3]).map_err(|e| format!("read evaluation split: {e}"))?;
    let report = validate_database(&snapshot_bytes, &split_bytes)?;
    let output = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?;
    fs::write(&args[4], output).map_err(|e| format!("write report: {e}"))?;
    println!("PITB_SORRYDB_SNAPSHOT_INTEGRITY_OK");
    println!("full_snapshot={} evaluation_split={} repos={}", report.full_snapshot.record_count, report.evaluation_split.record_count, report.repository_registry_count);
    Ok(())
}
fn main() -> ExitCode {
    match execute() { Ok(()) => ExitCode::SUCCESS, Err(message) => fail(message) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registered_sorrydb_integrity_controls_pass() {
        run_self_tests().expect("SorryDB integrity controls must pass");
    }
    #[test]
    fn duplicate_task_ids_are_rejected() {
        let record = synthetic_record().unwrap();
        assert!(validate_records(&[record.clone(), record], "synthetic").is_err());
    }
    #[test]
    fn identity_excludes_inclusion_date_but_includes_goal() {
        let record = synthetic_record().unwrap();
        let id = record["id"].as_str().unwrap().to_owned();
        let mut changed = record.clone();
        changed["metadata"]["inclusion_date"] = Value::String("2030-01-01T00:00:00+00:00".into());
        assert_eq!(sorry_id(&changed).unwrap(), id);
        changed["debug_info"]["goal"] = Value::String("⊢ False".into());
        assert_ne!(sorry_id(&changed).unwrap(), id);
    }
}
