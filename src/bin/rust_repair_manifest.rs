//! Fail-closed auditor for SWE-bench-Live's Rust task-validation output.
//!
//! Recomputes FAIL_TO_PASS and PASS_TO_PASS from observed base/gold test statuses
//! instead of trusting the upstream derived lists. It also requires the three
//! gold-patch run markers from the pinned upstream validator.
//!
//! This tool does not re-run Docker tests, prove source-transformation provenance,
//! or authenticate the host. Preserve the input, parquet and raw logs separately.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const PROTOCOL_ID: &str = "symthaea-rust-repair";
const MANIFEST_VERSION: u32 = 1;
const GOLD_RUN_MARKERS: [&str; 3] = ["eval No.0", "eval No.1", "eval No.2"];

#[derive(Debug)]
struct Cli {
    candidate_jsonl: PathBuf,
    validation_dir: PathBuf,
    source_parquet: PathBuf,
    expected_parquet_sha256: String,
    dataset_revision: String,
    upstream_code_revision: String,
    expected_candidate_count: usize,
    manifest_out: PathBuf,
    report_out: PathBuf,
}

#[derive(Debug, Serialize)]
struct TaskManifestRecord {
    instance_id: String,
    repo: String,
    base_commit: String,
    docker_image: String,
    problem_statement_sha256: String,
    gold_patch_sha256: String,
    test_patch_sha256: String,
    validation_status_sha256: String,
    baseline_validation_log_sha256: String,
    gold_validation_log_sha256: String,
    fail_to_pass_count: usize,
    pass_to_pass_count: usize,
}

#[derive(Debug, Serialize)]
struct LockedTaskManifest {
    format_version: u32,
    protocol_id: String,
    dataset_repository: String,
    dataset_revision: String,
    upstream_evaluator_repository: String,
    upstream_evaluator_revision: String,
    source_parquet_path: String,
    source_parquet_sha256: String,
    candidate_jsonl_sha256: String,
    candidate_row_count: usize,
    eligible_task_count: usize,
    rejected_task_count: usize,
    candidate_ids_sha256: String,
    eligible_task_ids_sha256: String,
    task_records_sha256: String,
    digest_encoding: String,
    tasks: Vec<TaskManifestRecord>,
}

#[derive(Debug, Serialize)]
struct TaskAudit {
    instance_id: String,
    eligible: bool,
    unresolved: bool,
    reasons: Vec<String>,
    baseline_test_count: usize,
    gold_test_count: usize,
    baseline_pass_count: usize,
    explicit_fail_to_pass_count: usize,
    gold_nonpassing_test_count: usize,
    gold_run_marker_counts: Vec<usize>,
    candidate_row_sha256: String,
    validation_status_sha256: Option<String>,
    baseline_validation_log_sha256: Option<String>,
    gold_validation_log_sha256: Option<String>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    format_version: u32,
    protocol_id: String,
    dataset_revision: String,
    upstream_evaluator_revision: String,
    source_parquet_sha256: String,
    expected_source_parquet_sha256: String,
    candidate_jsonl_sha256: String,
    candidate_row_count: usize,
    eligible_task_count: usize,
    rejected_task_count: usize,
    unresolved_task_count: usize,
    candidate_ids_sha256: String,
    task_records_sha256: Option<String>,
    lockable: bool,
    limitation: String,
    tasks: Vec<TaskAudit>,
}

struct TaskOutcome {
    audit: TaskAudit,
    manifest_record: Option<TaskManifestRecord>,
}

fn usage() -> &'static str {
    "Usage:
  cargo run --locked --bin rust-repair-manifest --features rust-repair-manifest -- \\
    --candidate-jsonl <full-rust-candidate-split.jsonl> \\
    --validation-dir <upstream-validation-output-dir> \\
    --source-parquet <pinned-rust.parquet> \\
    --expected-parquet-sha256 <64-hex> \\
    --dataset-revision <40-hex-commit> \\
    --upstream-code-revision <40-hex-commit> \\
    --expected-candidate-count <count> \\
    --manifest-out <eligible-task-manifest.json> \\
    --report-out <all-task-audit.json>

Input JSONL must be the complete candidate split, not the already-filtered
validated_instances.jsonl. The validation directory must contain
<instance_id>/status.json and post_patch_log.txt for each candidate. The source
Parquet SHA-256 and row count must be pinned before execution."
}

fn parse_cli(args: &[String]) -> Result<Option<Cli>, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", usage());
        return Ok(None);
    }
    let mut values = BTreeMap::<String, String>::new();
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        if !flag.starts_with("--") {
            return Err(format!("unexpected argument: {flag}\n{}", usage()));
        }
        if index + 1 >= args.len() {
            return Err(format!("missing value for {flag}\n{}", usage()));
        }
        if values.insert(flag.clone(), args[index + 1].clone()).is_some() {
            return Err(format!("duplicate argument: {flag}"));
        }
        index += 2;
    }

    let take = |key: &str| {
        values
            .get(key)
            .cloned()
            .ok_or_else(|| format!("missing required argument {key}\n{}", usage()))
    };
    let expected_candidate_count = take("--expected-candidate-count")?
        .parse::<usize>()
        .map_err(|_| "--expected-candidate-count must be a positive integer".to_string())?;
    if expected_candidate_count == 0 {
        return Err("--expected-candidate-count must be greater than zero".to_string());
    }
    let expected_parquet_sha256 = take("--expected-parquet-sha256")?.to_ascii_lowercase();
    if !is_hex(&expected_parquet_sha256, 64) {
        return Err("--expected-parquet-sha256 must be 64 hex characters".to_string());
    }
    let dataset_revision = take("--dataset-revision")?;
    let upstream_code_revision = take("--upstream-code-revision")?;
    if !is_hex(&dataset_revision, 40) {
        return Err("--dataset-revision must be a full 40-character commit SHA".to_string());
    }
    if !is_hex(&upstream_code_revision, 40) {
        return Err("--upstream-code-revision must be a full 40-character commit SHA".to_string());
    }
    let allowed = [
        "--candidate-jsonl",
        "--validation-dir",
        "--source-parquet",
        "--expected-parquet-sha256",
        "--dataset-revision",
        "--upstream-code-revision",
        "--expected-candidate-count",
        "--manifest-out",
        "--report-out",
    ];
    if let Some(unknown) = values.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unknown argument: {unknown}"));
    }
    Ok(Some(Cli {
        candidate_jsonl: PathBuf::from(take("--candidate-jsonl")?),
        validation_dir: PathBuf::from(take("--validation-dir")?),
        source_parquet: PathBuf::from(take("--source-parquet")?),
        expected_parquet_sha256,
        dataset_revision,
        upstream_code_revision,
        expected_candidate_count,
        manifest_out: PathBuf::from(take("--manifest-out")?),
        report_out: PathBuf::from(take("--report-out")?),
    }))
}

fn is_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn required_string<'a>(row: &'a Value, field: &str) -> Result<&'a str, String> {
    row.get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing or empty field {field}"))
}

fn safe_instance_id(instance_id: &str) -> bool {
    !instance_id.is_empty()
        && instance_id != "."
        && instance_id != ".."
        && !instance_id.contains("..")
        && instance_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn parse_status_map(value: &Value, field: &str) -> Result<BTreeMap<String, String>, String> {
    let object = value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{field} is missing or is not an object"))?;
    if object.is_empty() {
        return Err(format!("{field} is empty"));
    }
    let mut statuses = BTreeMap::new();
    for (test_id, raw_status) in object {
        let status = raw_status
            .as_str()
            .ok_or_else(|| format!("{field}.{test_id} is not a string"))?
            .trim()
            .to_ascii_lowercase();
        if !matches!(status.as_str(), "pass" | "fail" | "skip") {
            return Err(format!("{field}.{test_id} has unknown status {status:?}"));
        }
        statuses.insert(test_id.clone(), status);
    }
    Ok(statuses)
}

fn parse_test_id_set(value: &Value, field: &str) -> Result<BTreeSet<String>, String> {
    let array = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} is missing or is not an array"))?;
    let mut result = BTreeSet::new();
    for item in array {
        let id = item
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{field} contains a non-string or empty test id"))?;
        if !result.insert(id.to_string()) {
            return Err(format!("{field} contains duplicate test id {id:?}"));
        }
    }
    Ok(result)
}

fn transition_sets(
    baseline: &BTreeMap<String, String>,
    gold: &BTreeMap<String, String>,
) -> (BTreeSet<String>, BTreeSet<String>) {
    let baseline_pass = baseline
        .iter()
        .filter(|(_, status)| status.as_str() == "pass")
        .map(|(test_id, _)| test_id.clone())
        .collect::<BTreeSet<_>>();
    let explicit_fail_to_pass = baseline
        .iter()
        .filter(|(test_id, status)| {
            status.as_str() == "fail" && gold.get(*test_id).is_some_and(|post| post.as_str() == "pass")
        })
        .map(|(test_id, _)| test_id.clone())
        .collect::<BTreeSet<_>>();
    (baseline_pass, explicit_fail_to_pass)
}

fn run_marker_counts(log: &str) -> Vec<usize> {
    GOLD_RUN_MARKERS
        .iter()
        .map(|marker| log.match_indices(marker).count())
        .collect()
}

fn hash_field(row: &Value, field: &str) -> Result<String, String> {
    let value = required_string(row, field)?;
    Ok(sha256_bytes(value.as_bytes()))
}

fn audit_candidate(
    row: &Value,
    instance_id: &str,
    validation_dir: &Path,
    candidate_row_sha256: String,
) -> TaskOutcome {
    let mut reasons = Vec::<String>::new();
    let mut unresolved = false;

    let repo = match required_string(row, "repo") {
        Ok(value) if value.split('/').count() == 2 => Some(value.to_string()),
        Ok(_) => {
            reasons.push("repo must have owner/name form".to_string());
            None
        }
        Err(error) => {
            reasons.push(error);
            None
        }
    };
    let base_commit = match required_string(row, "base_commit") {
        Ok(value) if is_hex(value, 40) => Some(value.to_ascii_lowercase()),
        Ok(_) => {
            reasons.push("base_commit must be a full 40-character SHA".to_string());
            None
        }
        Err(error) => {
            reasons.push(error);
            None
        }
    };
    let docker_image = match required_string(row, "docker_image") {
        Ok(value) => Some(value.to_string()),
        Err(error) => {
            reasons.push(error);
            None
        }
    };
    for field in ["problem_statement", "patch", "test_patch"] {
        if required_string(row, field).is_err() {
            reasons.push(format!("{field} is missing or empty"));
        }
    }
    let problem_statement_sha256 = hash_field(row, "problem_statement").ok();
    let gold_patch_sha256 = hash_field(row, "patch").ok();
    let test_patch_sha256 = hash_field(row, "test_patch").ok();

    let task_dir = validation_dir.join(instance_id);
    let status_path = task_dir.join("status.json");
    let baseline_log_path = task_dir.join("pre_patch_log.txt");
    let log_path = task_dir.join("post_patch_log.txt");
    let status_bytes = match fs::read(&status_path) {
        Ok(bytes) => Some(bytes),
        Err(error) => {
            unresolved = true;
            reasons.push(format!("cannot read validation status: {error}"));
            None
        }
    };
    let baseline_log_bytes = match fs::read(&baseline_log_path) {
        Ok(bytes) if !bytes.is_empty() => Some(bytes),
        Ok(_) => {
            unresolved = true;
            reasons.push("baseline validation log is empty".to_string());
            None
        }
        Err(error) => {
            unresolved = true;
            reasons.push(format!("cannot read baseline validation log: {error}"));
            None
        }
    };
    let log_bytes = match fs::read(&log_path) {
        Ok(bytes) if !bytes.is_empty() => Some(bytes),
        Ok(_) => {
            unresolved = true;
            reasons.push("gold validation log is empty".to_string());
            None
        }
        Err(error) => {
            unresolved = true;
            reasons.push(format!("cannot read gold validation log: {error}"));
            None
        }
    };

    let status_json_sha256 = status_bytes.as_ref().map(|bytes| sha256_bytes(bytes));
    let baseline_log_sha256 = baseline_log_bytes.as_ref().map(|bytes| sha256_bytes(bytes));
    let log_sha256 = log_bytes.as_ref().map(|bytes| sha256_bytes(bytes));
    let mut baseline_count = 0;
    let mut gold_count = 0;
    let mut baseline_pass_count = 0;
    let mut explicit_fail_to_pass_count = 0;
    let mut gold_nonpassing_count = 0;
    let mut marker_counts = vec![0; GOLD_RUN_MARKERS.len()];

    if let Some(bytes) = status_bytes.as_ref() {
        match serde_json::from_slice::<Value>(bytes) {
            Ok(status) => {
                if status.get("instance_id").and_then(Value::as_str) != Some(instance_id) {
                    unresolved = true;
                    reasons.push("status.json instance_id does not match candidate".to_string());
                }
                let baseline = match parse_status_map(&status, "pre_patch_status") {
                    Ok(map) => Some(map),
                    Err(error) => {
                        unresolved = true;
                        reasons.push(error);
                        None
                    }
                };
                let gold = match parse_status_map(&status, "post_patch_status") {
                    Ok(map) => Some(map),
                    Err(error) => {
                        unresolved = true;
                        reasons.push(error);
                        None
                    }
                };
                let listed_f2p = match parse_test_id_set(&status, "FAIL_TO_PASS") {
                    Ok(set) => Some(set),
                    Err(error) => {
                        unresolved = true;
                        reasons.push(error);
                        None
                    }
                };
                let listed_p2p = match parse_test_id_set(&status, "PASS_TO_PASS") {
                    Ok(set) => Some(set),
                    Err(error) => {
                        unresolved = true;
                        reasons.push(error);
                        None
                    }
                };

                if let (Some(baseline), Some(gold)) = (baseline.as_ref(), gold.as_ref()) {
                    baseline_count = baseline.len();
                    gold_count = gold.len();
                    if baseline.keys().collect::<BTreeSet<_>>()
                        != gold.keys().collect::<BTreeSet<_>>()
                    {
                        unresolved = true;
                        reasons.push("baseline and gold test-ID sets differ".to_string());
                    }
                    baseline_pass_count = baseline
                        .values()
                        .filter(|state| state.as_str() == "pass")
                        .count();
                    gold_nonpassing_count = gold
                        .values()
                        .filter(|state| state.as_str() != "pass")
                        .count();
                    let (baseline_pass, explicit_f2p) = transition_sets(baseline, gold);
                    explicit_fail_to_pass_count = explicit_f2p.len();

                    if explicit_f2p.is_empty() {
                        reasons.push("no explicit fail-to-pass transition (fail -> pass)".to_string());
                    }
                    if let Some(listed) = listed_f2p.as_ref() {
                        if *listed != explicit_f2p {
                            reasons.push(
                                "upstream FAIL_TO_PASS differs from explicit fail -> pass transitions"
                                    .to_string(),
                            );
                        }
                    }
                    if let Some(listed) = listed_p2p.as_ref() {
                        if *listed != baseline_pass {
                            reasons.push(
                                "upstream PASS_TO_PASS differs from baseline-passing tests; baseline passes must not be dropped"
                                    .to_string(),
                            );
                        }
                    }
                    let regressed = baseline_pass
                        .iter()
                        .filter(|test_id| {
                            gold.get(*test_id).map(String::as_str) != Some("pass")
                        })
                        .count();
                    if regressed > 0 {
                        reasons.push(format!(
                            "{regressed} baseline-passing test(s) fail or are missing after the gold patch"
                        ));
                    }
                    if gold_nonpassing_count > 0 {
                        reasons.push(format!(
                            "{gold_nonpassing_count} gold-patch test status(es) are fail or skip in the three-run aggregate"
                        ));
                    }
                    if baseline.values().any(|state| state == "skip") {
                        reasons.push("baseline contains skipped or missing test status".to_string());
                    }
                }
            }
            Err(error) => {
                unresolved = true;
                reasons.push(format!("status.json is malformed: {error}"));
            }
        }
    }

    if let Some(bytes) = log_bytes.as_ref() {
        let log = String::from_utf8_lossy(bytes);
        marker_counts = run_marker_counts(&log);
        for (index, count) in marker_counts.iter().enumerate() {
            if *count != 1 {
                unresolved = true;
                reasons.push(format!(
                    "gold validation log marker {:?} occurs {count} times; expected exactly once",
                    GOLD_RUN_MARKERS[index]
                ));
            }
        }
    }

    let eligible = !unresolved && reasons.is_empty();
    let audit = TaskAudit {
        instance_id: instance_id.to_string(),
        eligible,
        unresolved,
        reasons,
        baseline_test_count: baseline_count,
        gold_test_count: gold_count,
        baseline_pass_count,
        explicit_fail_to_pass_count,
        gold_nonpassing_test_count: gold_nonpassing_count,
        gold_run_marker_counts: marker_counts,
        candidate_row_sha256,
        validation_status_sha256: status_json_sha256.clone(),
        baseline_validation_log_sha256: baseline_log_sha256.clone(),
        gold_validation_log_sha256: log_sha256.clone(),
    };

    let manifest_record = if eligible {
        Some(TaskManifestRecord {
            instance_id: instance_id.to_string(),
            repo: repo.expect("eligible task has repo"),
            base_commit: base_commit.expect("eligible task has base commit"),
            docker_image: docker_image.expect("eligible task has docker image"),
            problem_statement_sha256: problem_statement_sha256.expect("eligible task has prompt"),
            gold_patch_sha256: gold_patch_sha256.expect("eligible task has gold patch"),
            test_patch_sha256: test_patch_sha256.expect("eligible task has test patch"),
            validation_status_sha256: status_json_sha256.expect("eligible task has status file"),
            baseline_validation_log_sha256: baseline_log_sha256.expect("eligible task has baseline log"),
            gold_validation_log_sha256: log_sha256.expect("eligible task has gold validation log"),
            fail_to_pass_count: explicit_fail_to_pass_count,
            pass_to_pass_count: baseline_pass_count,
        })
    } else {
        None
    };

    TaskOutcome {
        audit,
        manifest_record,
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize output JSON: {error}"))?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn canonical_id_digest(ids: &[String]) -> String {
    let mut canonical = ids.join("\n");
    if !canonical.is_empty() {
        canonical.push('\n');
    }
    sha256_bytes(canonical.as_bytes())
}

fn run(cli: Cli) -> Result<(), String> {
    let actual_parquet_sha256 = sha256_file(&cli.source_parquet)?;
    if actual_parquet_sha256 != cli.expected_parquet_sha256 {
        return Err(format!(
            "source Parquet SHA-256 mismatch: expected {}, observed {}",
            cli.expected_parquet_sha256, actual_parquet_sha256
        ));
    }

    let candidate_bytes = fs::read(&cli.candidate_jsonl)
        .map_err(|error| format!("cannot read {}: {error}", cli.candidate_jsonl.display()))?;
    let candidate_jsonl_sha256 = sha256_bytes(&candidate_bytes);
    let candidate_text = std::str::from_utf8(&candidate_bytes)
        .map_err(|error| format!("candidate JSONL is not UTF-8: {error}"))?;

    let mut candidates = Vec::<(String, Value, String)>::new();
    let mut seen = HashSet::<String>::new();
    for (line_index, raw_line) in candidate_text.lines().enumerate() {
        if raw_line.trim().is_empty() {
            return Err(format!("candidate JSONL line {} is empty", line_index + 1));
        }
        let row: Value = serde_json::from_str(raw_line).map_err(|error| {
            format!("candidate JSONL line {} is invalid JSON: {error}", line_index + 1)
        })?;
        let instance_id = required_string(&row, "instance_id")
            .map_err(|error| format!("candidate JSONL line {}: {error}", line_index + 1))?
            .to_string();
        if !safe_instance_id(&instance_id) {
            return Err(format!(
                "unsafe instance_id at candidate JSONL line {}: {instance_id:?}",
                line_index + 1
            ));
        }
        if !seen.insert(instance_id.clone()) {
            return Err(format!("duplicate candidate instance_id: {instance_id}"));
        }
        candidates.push((instance_id, row, sha256_bytes(raw_line.as_bytes())));
    }

    if candidates.len() != cli.expected_candidate_count {
        return Err(format!(
            "candidate row count mismatch: expected {}, observed {}",
            cli.expected_candidate_count,
            candidates.len()
        ));
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    let candidate_ids = candidates.iter().map(|(id, _, _)| id.clone()).collect::<Vec<_>>();
    let candidate_ids_sha256 = canonical_id_digest(&candidate_ids);

    let mut outcomes = Vec::with_capacity(candidates.len());
    let mut manifest_tasks = Vec::new();
    for (instance_id, row, row_sha256) in candidates {
        let outcome = audit_candidate(&row, &instance_id, &cli.validation_dir, row_sha256);
        if let Some(record) = outcome.manifest_record.as_ref() {
            manifest_tasks.push(TaskManifestRecord {
                instance_id: record.instance_id.clone(),
                repo: record.repo.clone(),
                base_commit: record.base_commit.clone(),
                docker_image: record.docker_image.clone(),
                problem_statement_sha256: record.problem_statement_sha256.clone(),
                gold_patch_sha256: record.gold_patch_sha256.clone(),
                test_patch_sha256: record.test_patch_sha256.clone(),
                validation_status_sha256: record.validation_status_sha256.clone(),
                baseline_validation_log_sha256: record.baseline_validation_log_sha256.clone(),
                gold_validation_log_sha256: record.gold_validation_log_sha256.clone(),
                fail_to_pass_count: record.fail_to_pass_count,
                pass_to_pass_count: record.pass_to_pass_count,
            });
        }
        outcomes.push(outcome.audit);
    }
    manifest_tasks.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));

    let eligible_ids = manifest_tasks
        .iter()
        .map(|task| task.instance_id.clone())
        .collect::<Vec<_>>();
    let eligible_task_ids_sha256 = canonical_id_digest(&eligible_ids);
    let serialized_tasks = serde_json::to_vec(&manifest_tasks)
        .map_err(|error| format!("cannot serialize task records for digest: {error}"))?;
    let task_records_sha256 = sha256_bytes(&serialized_tasks);
    let rejected_count = outcomes.iter().filter(|task| !task.eligible && !task.unresolved).count();
    let unresolved_count = outcomes.iter().filter(|task| task.unresolved).count();
    let lockable = unresolved_count == 0 && !manifest_tasks.is_empty();

    let report = AuditReport {
        format_version: MANIFEST_VERSION,
        protocol_id: PROTOCOL_ID.to_string(),
        dataset_revision: cli.dataset_revision.clone(),
        upstream_evaluator_revision: cli.upstream_code_revision.clone(),
        source_parquet_sha256: actual_parquet_sha256.clone(),
        expected_source_parquet_sha256: cli.expected_parquet_sha256.clone(),
        candidate_jsonl_sha256: candidate_jsonl_sha256.clone(),
        candidate_row_count: outcomes.len(),
        eligible_task_count: manifest_tasks.len(),
        rejected_task_count: rejected_count,
        unresolved_task_count: unresolved_count,
        candidate_ids_sha256: candidate_ids_sha256.clone(),
        task_records_sha256: if lockable {
            Some(task_records_sha256.clone())
        } else {
            None
        },
        lockable,
        limitation: "This audit recomputes label transitions and checks provenance consistency. It does not execute Docker tests, prove the candidate JSONL was derived from the pinned Parquet by a trusted loader, or authenticate the runner. Preserve source bytes and raw logs for independent replication.".to_string(),
        tasks: outcomes,
    };
    write_json(&cli.report_out, &report)?;

    if unresolved_count > 0 {
        return Err(format!(
            "audit is not lockable: {unresolved_count} task(s) lack complete validation evidence; see {}. No manifest was written.",
            cli.report_out.display()
        ));
    }
    if manifest_tasks.is_empty() {
        return Err(format!(
            "no task passed the strict audit; see {}. No manifest was written.",
            cli.report_out.display()
        ));
    }

    let manifest = LockedTaskManifest {
        format_version: MANIFEST_VERSION,
        protocol_id: PROTOCOL_ID.to_string(),
        dataset_repository: "SWE-bench-Live/MultiLang".to_string(),
        dataset_revision: cli.dataset_revision,
        upstream_evaluator_repository: "microsoft/SWE-bench-Live".to_string(),
        upstream_evaluator_revision: cli.upstream_code_revision,
        source_parquet_path: "data/rust-00000-of-00001.parquet".to_string(),
        source_parquet_sha256: actual_parquet_sha256,
        candidate_jsonl_sha256,
        candidate_row_count: report.candidate_row_count,
        eligible_task_count: manifest_tasks.len(),
        rejected_task_count: rejected_count,
        candidate_ids_sha256,
        eligible_task_ids_sha256,
        task_records_sha256,
        digest_encoding: "task_records_sha256 = SHA256(serde_json compact serialization of tasks sorted by instance_id); *_task_ids_sha256 = SHA256 of sorted UTF-8 IDs joined by LF with a final LF".to_string(),
        tasks: manifest_tasks,
    };
    write_json(&cli.manifest_out, &manifest)?;
    println!(
        "RUST REPAIR MANIFEST LOCKED: {} eligible / {} candidate tasks; task records SHA-256 {}",
        manifest.eligible_task_count, manifest.candidate_row_count, manifest.task_records_sha256
    );
    println!("Audit report: {}", cli.report_out.display());
    println!("Task manifest: {}", cli.manifest_out.display());
    Ok(())
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let cli = match parse_cli(&args) {
        Ok(Some(cli)) => cli,
        Ok(None) => return,
        Err(error) => {
            eprintln!("ERROR: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(cli) {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status_map(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(id, status)| (id.to_string(), status.to_string()))
            .collect()
    }

    #[test]
    fn fail_to_pass_requires_explicit_baseline_failure() {
        let baseline = status_map(&[("existing", "pass")]);
        let gold = status_map(&[("existing", "pass"), ("appeared-after", "pass")]);
        let (baseline_pass, explicit_f2p) = transition_sets(&baseline, &gold);
        assert_eq!(baseline_pass, BTreeSet::from(["existing".to_string()]));
        assert!(explicit_f2p.is_empty());
        assert_ne!(
            baseline.keys().collect::<BTreeSet<_>>(),
            gold.keys().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn baseline_passes_are_retained_even_if_gold_patch_regresses_them() {
        let baseline = status_map(&[("must-stay-green", "pass"), ("bug", "fail")]);
        let gold = status_map(&[("must-stay-green", "fail"), ("bug", "pass")]);
        let (baseline_pass, explicit_f2p) = transition_sets(&baseline, &gold);
        assert_eq!(
            baseline_pass,
            BTreeSet::from(["must-stay-green".to_string()])
        );
        assert_eq!(explicit_f2p, BTreeSet::from(["bug".to_string()]));
        let stable_baseline_passes = baseline_pass
            .iter()
            .filter(|test_id| gold.get(*test_id).map(String::as_str) == Some("pass"))
            .cloned()
            .collect::<BTreeSet<_>>();
        assert_ne!(baseline_pass, stable_baseline_passes);
    }

    #[test]
    fn three_gold_run_markers_must_be_present_exactly_once() {
        let valid = "eval No.0 \npass\neval No.1 \npass\neval No.2 \npass";
        assert_eq!(run_marker_counts(valid), vec![1, 1, 1]);
        assert_eq!(run_marker_counts("eval No.0 \neval No.1 \neval No.1"), vec![1, 2, 0]);
    }

    #[test]
    fn canonical_ids_are_order_independent_after_sorting() {
        let mut ids_a = vec!["b".to_string(), "a".to_string()];
        ids_a.sort();
        let ids_b = vec!["a".to_string(), "b".to_string()];
        assert_eq!(canonical_id_digest(&ids_a), canonical_id_digest(&ids_b));
    }

    #[test]
    fn sha256_matches_standard_known_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
