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
    agent_prompts_out: PathBuf,
    controller_manifest_out: PathBuf,
}

#[derive(Debug, Serialize)]
struct TaskManifestRecord {
    instance_id: String,
    candidate_row_sha256: String,
    repo: String,
    base_commit: String,
    repo_head_sha: String,
    runtime_platform: String,
    docker_image: String,
    problem_statement_sha256: String,
    gold_patch_sha256: String,
    test_patch_sha256: String,
    validation_status_sha256: String,
    baseline_validation_log_sha256: String,
    gold_validation_log_sha256: String,
    runtime_evidence_sha256: String,
    container_image_id: String,
    network_mode: String,
    workspace_clean: bool,
    container_ids: Vec<String>,
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
struct AgentPromptRecord {
    /// This is the only task-specific field serialized into the agent-facing input.
    problem_statement: String,
}

#[derive(Debug, Serialize)]
struct ControllerTaskRecord {
    run_index: usize,
    instance_id: String,
    repo_head_sha: String,
    runtime_platform: String,
    docker_image_reference: String,
    container_image_id: String,
    network_mode: String,
    workspace_clean: bool,
}

#[derive(Debug, Serialize)]
struct ControllerManifest {
    format_version: u32,
    protocol_id: String,
    dataset_revision: String,
    upstream_evaluator_revision: String,
    source_parquet_sha256: String,
    candidate_jsonl_sha256: String,
    eligible_task_ids_sha256: String,
    agent_prompt_projection_sha256: String,
    task_count: usize,
    tasks: Vec<ControllerTaskRecord>,
    visibility_warning: String,
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
    runtime_evidence_sha256: Option<String>,
    container_image_id: Option<String>,
    repo_head_sha: Option<String>,
    runtime_platform: Option<String>,
    network_mode: Option<String>,
    workspace_clean: Option<bool>,
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
    agent_prompt_projection_sha256: Option<String>,
    controller_manifest_sha256: Option<String>,
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
    --report-out <all-task-audit.json> \\
    --agent-prompts-out <agent-visible-prompts.jsonl> \\
    --controller-manifest-out <evaluator-only-controller.json>

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
        if values
            .insert(flag.clone(), args[index + 1].clone())
            .is_some()
        {
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
        "--agent-prompts-out",
        "--controller-manifest-out",
    ];
    if let Some(unknown) = values.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unknown argument: {unknown}"));
    }
    let output_keys = [
        "--manifest-out",
        "--report-out",
        "--agent-prompts-out",
        "--controller-manifest-out",
    ];
    let mut output_paths = HashSet::<String>::new();
    for key in output_keys {
        let output = values
            .get(key)
            .expect("all output paths are required")
            .clone();
        if !output_paths.insert(output.clone()) {
            return Err(format!(
                "output paths must be distinct; {key} reuses {output:?}"
            ));
        }
    }
    let source_paths = ["--candidate-jsonl", "--source-parquet"];
    for output_key in output_keys {
        let output = values.get(output_key).expect("output exists");
        for source_key in source_paths {
            if values
                .get(source_key)
                .is_some_and(|source| source == output)
            {
                return Err(format!(
                    "output path {output:?} must not overwrite input {source_key}"
                ));
            }
        }
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
        agent_prompts_out: PathBuf::from(take("--agent-prompts-out")?),
        controller_manifest_out: PathBuf::from(take("--controller-manifest-out")?),
    }))
}

fn is_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file =
        fs::File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
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
            status.as_str() == "fail"
                && gold
                    .get(*test_id)
                    .is_some_and(|post| post.as_str() == "pass")
        })
        .map(|(test_id, _)| test_id.clone())
        .collect::<BTreeSet<_>>();
    (baseline_pass, explicit_fail_to_pass)
}

fn transition_contract_errors(
    baseline: &BTreeMap<String, String>,
    gold: &BTreeMap<String, String>,
    listed_f2p: &BTreeSet<String>,
    listed_p2p: &BTreeSet<String>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let (baseline_pass, explicit_f2p) = transition_sets(baseline, gold);

    if explicit_f2p.is_empty() {
        errors.push("no explicit fail-to-pass transition (fail -> pass)".to_string());
    }
    if *listed_f2p != explicit_f2p {
        errors.push(
            "upstream FAIL_TO_PASS differs from explicit fail -> pass transitions".to_string(),
        );
    }
    if *listed_p2p != baseline_pass {
        errors.push(
            "upstream PASS_TO_PASS differs from baseline-passing tests; baseline passes must not be dropped"
                .to_string(),
        );
    }

    let regressed = baseline_pass
        .iter()
        .filter(|test_id| gold.get(*test_id).map(String::as_str) != Some("pass"))
        .count();
    if regressed > 0 {
        errors.push(format!(
            "{regressed} baseline-passing test(s) fail or are missing after the gold patch"
        ));
    }
    let gold_nonpassing = gold
        .values()
        .filter(|state| state.as_str() != "pass")
        .count();
    if gold_nonpassing > 0 {
        errors.push(format!(
            "{gold_nonpassing} gold-patch test status(es) are fail or skip in the three-run aggregate"
        ));
    }
    if baseline.values().any(|state| state == "skip") {
        errors.push("baseline contains skipped or missing test status".to_string());
    }
    errors
}

fn run_marker_counts(log: &str) -> Vec<usize> {
    GOLD_RUN_MARKERS
        .iter()
        .map(|marker| log.match_indices(marker).count())
        .collect()
}

#[derive(Debug)]
struct RuntimeEvidenceSummary {
    container_image_id: String,
    network_mode: String,
    repo_head_sha: String,
    runtime_platform: String,
    workspace_clean: bool,
    container_ids: Vec<String>,
}

fn command_exit_code(run: &Value, key: &str) -> Result<i64, String> {
    run.get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("{key} missing or not an integer"))
}

fn validate_container_identity(
    run: &Value,
    expected_image_reference: &str,
) -> Result<(String, String), String> {
    let identity = run
        .get("container_identity")
        .and_then(Value::as_object)
        .ok_or_else(|| "container_identity missing or not an object".to_string())?;
    if identity
        .get("git_metadata_sanitization_exit_code")
        .and_then(Value::as_i64)
        != Some(0)
    {
        return Err("Git metadata sanitization was not confirmed".to_string());
    }
    let runtime_platform = identity
        .get("runtime_platform")
        .and_then(Value::as_str)
        .ok_or_else(|| "container_identity.runtime_platform missing".to_string())?;
    if runtime_platform != "linux" {
        return Err(format!(
            "strict Rust repair evaluation requires Linux runtime; observed {runtime_platform:?}"
        ));
    }
    let network_mode = identity
        .get("network_mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "container_identity.network_mode missing".to_string())?;
    if network_mode != "none" {
        return Err(format!(
            "container_identity.network_mode must be 'none' for strict evaluation; observed {network_mode:?}"
        ));
    }
    let container_id = identity
        .get("container_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "container_identity.container_id missing".to_string())?
        .to_ascii_lowercase();
    if !is_hex(&container_id, 64) {
        return Err("container_identity.container_id must be a 64-character Docker ID".to_string());
    }

    let configured_image = identity
        .get("configured_image")
        .and_then(Value::as_str)
        .ok_or_else(|| "container_identity.configured_image missing".to_string())?;
    if configured_image != expected_image_reference {
        return Err(format!(
            "configured image reference differs: expected {expected_image_reference:?}, got {configured_image:?}"
        ));
    }

    let image_id = identity
        .get("container_image_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "container_identity.container_image_id missing".to_string())?
        .to_ascii_lowercase();
    if !image_id.starts_with("sha256:") || !is_hex(&image_id[7..], 64) {
        return Err(
            "container_image_id must be an immutable sha256:<64-hex> Docker image ID".to_string(),
        );
    }
    Ok((container_id, image_id))
}

fn verify_logged_run(
    run: &Value,
    expected_image_reference: &str,
    expected_base_commit: &str,
    expected_role: &str,
    expected_index: Option<usize>,
    expected_test_ids: &BTreeSet<String>,
    expected_log_path: &Path,
    gold: bool,
) -> Result<(BTreeMap<String, String>, String, String), String> {
    if run.get("role").and_then(Value::as_str) != Some(expected_role) {
        return Err(format!("expected run role {expected_role:?}"));
    }
    if let Some(index) = expected_index {
        if run.get("run_index").and_then(Value::as_u64) != Some(index as u64) {
            return Err(format!("run_index must be {index}"));
        }
    }
    if run.get("test_patch_applied").and_then(Value::as_bool) != Some(true) {
        return Err("test patch application did not return true".to_string());
    }
    if gold && run.get("solution_patch_applied").and_then(Value::as_bool) != Some(true) {
        return Err("gold solution patch application did not return true".to_string());
    }
    if command_exit_code(run, "workspace_clean_exit_code")? != 0 {
        return Err("repository worktree is dirty or cleanliness inspection failed".to_string());
    }
    if command_exit_code(run, "repo_head_exit_code")? != 0 {
        return Err("repository HEAD inspection did not exit zero".to_string());
    }
    let observed_repo_head = run
        .get("repo_head_sha")
        .and_then(Value::as_str)
        .ok_or_else(|| "repo_head_sha missing or not a string".to_string())?
        .to_ascii_lowercase();
    if !(is_hex(&observed_repo_head, 40) || is_hex(&observed_repo_head, 64)) {
        return Err("repo_head_sha must be a 40- or 64-character Git SHA".to_string());
    }
    if observed_repo_head != expected_base_commit.to_ascii_lowercase() {
        return Err(format!(
            "container repository HEAD differs from task base_commit: expected {}, observed {}",
            expected_base_commit, observed_repo_head
        ));
    }
    if command_exit_code(run, "rebuild_exit_code")? != 0 {
        return Err("rebuild command did not exit zero".to_string());
    }

    let test_exit = command_exit_code(run, "test_exit_code")?;
    if test_exit == 124 || test_exit < 0 {
        return Err(format!(
            "test command is unresolved (exit code {test_exit})"
        ));
    }
    if command_exit_code(run, "print_exit_code")? != 0 {
        return Err("print/log command did not exit zero".to_string());
    }

    let (container_id, image_id) = validate_container_identity(run, expected_image_reference)?;
    let parsed = parse_status_map(run, "parsed_status")?;
    let observed_ids = parsed.keys().cloned().collect::<BTreeSet<_>>();
    if &observed_ids != expected_test_ids {
        return Err("per-run parsed test-ID set differs from the baseline test-ID set".to_string());
    }
    if gold && test_exit != 0 && parsed.values().all(|state| state == "pass") {
        return Err(format!(
            "gold test command exited {test_exit}, but parser reports every test passing"
        ));
    }

    let log_sha = run
        .get("log_sha256")
        .and_then(Value::as_str)
        .filter(|value| is_hex(value, 64))
        .ok_or_else(|| "log_sha256 missing or malformed".to_string())?
        .to_ascii_lowercase();
    let actual_log_sha = sha256_file(expected_log_path)?;
    if log_sha != actual_log_sha {
        return Err(format!(
            "raw run log SHA-256 mismatch for {}",
            expected_log_path.display()
        ));
    }
    Ok((parsed, container_id, image_id))
}

fn validate_runtime_evidence(
    bytes: &[u8],
    row: &Value,
    instance_id: &str,
    validation_dir: &Path,
    status: &Value,
) -> Result<RuntimeEvidenceSummary, Vec<String>> {
    let evidence: Value = serde_json::from_slice(bytes)
        .map_err(|error| vec![format!("validation_evidence.json is malformed: {error}")])?;
    let mut errors = Vec::new();
    if evidence.get("schema_version").and_then(Value::as_u64) != Some(1) {
        errors.push("validation evidence schema_version must be 1".to_string());
    }
    if evidence.get("instance_id").and_then(Value::as_str) != Some(instance_id) {
        errors.push("validation evidence instance_id does not match candidate".to_string());
    }

    let expected_image_reference = match required_string(row, "docker_image") {
        Ok(value) => value,
        Err(error) => return Err(vec![error]),
    };
    let expected_base_commit = match required_string(row, "base_commit") {
        Ok(value) if is_hex(value, 40) || is_hex(value, 64) => value.to_ascii_lowercase(),
        Ok(_) => {
            return Err(vec![
                "candidate base_commit is not a valid Git SHA".to_string(),
            ]);
        }
        Err(error) => return Err(vec![error]),
    };
    if evidence.get("image_reference").and_then(Value::as_str) != Some(expected_image_reference) {
        errors.push(
            "validation evidence image_reference differs from candidate metadata".to_string(),
        );
    }

    let baseline_expected = parse_status_map(status, "pre_patch_status")
        .map_err(|error| vec![format!("status.json {error}")])?;
    let gold_expected = parse_status_map(status, "post_patch_status")
        .map_err(|error| vec![format!("status.json {error}")])?;
    let expected_test_ids = baseline_expected.keys().cloned().collect::<BTreeSet<_>>();
    let baseline = match evidence.get("baseline") {
        Some(value) => value,
        None => {
            return Err(vec![
                "validation evidence has no baseline record".to_string(),
            ]);
        }
    };
    let baseline_log_path = validation_dir.join(instance_id).join("pre_patch_log.txt");
    let mut run_ids = Vec::new();
    let mut image_ids = Vec::new();

    match verify_logged_run(
        baseline,
        expected_image_reference,
        &expected_base_commit,
        "baseline",
        None,
        &expected_test_ids,
        &baseline_log_path,
        false,
    ) {
        Ok((baseline_status, container_id, image_id)) => {
            if baseline_status != baseline_expected {
                errors.push(
                    "baseline parsed statuses differ from status.json pre_patch_status".to_string(),
                );
            }
            run_ids.push(container_id);
            image_ids.push(image_id);
        }
        Err(error) => errors.push(format!("baseline evidence: {error}")),
    }

    let gold_runs = match evidence.get("gold_runs").and_then(Value::as_array) {
        Some(runs) if runs.len() == 3 => runs,
        Some(runs) => {
            return Err(vec![format!(
                "validation evidence has {} gold runs; exactly 3 required",
                runs.len()
            )]);
        }
        None => {
            return Err(vec![
                "validation evidence gold_runs missing or not an array".to_string(),
            ]);
        }
    };
    let mut parsed_gold_runs = Vec::with_capacity(3);
    for index in 0..3 {
        let log_path = validation_dir
            .join(instance_id)
            .join(format!("gold_run_{index}.log"));
        match verify_logged_run(
            &gold_runs[index],
            expected_image_reference,
            &expected_base_commit,
            "gold",
            Some(index),
            &expected_test_ids,
            &log_path,
            true,
        ) {
            Ok((statuses, container_id, image_id)) => {
                parsed_gold_runs.push(statuses);
                run_ids.push(container_id);
                image_ids.push(image_id);
            }
            Err(error) => errors.push(format!("gold run {index} evidence: {error}")),
        }
    }

    if run_ids.len() == 4 && run_ids.iter().collect::<HashSet<_>>().len() != 4 {
        errors.push("container IDs are not unique across baseline and gold runs".to_string());
    }
    if image_ids.len() == 4 && image_ids.iter().collect::<HashSet<_>>().len() != 1 {
        errors.push("container image ID differs across baseline and gold runs".to_string());
    }
    if parsed_gold_runs.len() == 3 {
        let mut all_ids = BTreeSet::new();
        for run in &parsed_gold_runs {
            all_ids.extend(run.keys().cloned());
        }

        let mut aggregate = BTreeMap::new();
        for test_id in all_ids {
            let states = parsed_gold_runs
                .iter()
                .map(|run| run.get(&test_id).map(String::as_str).unwrap_or("skip"))
                .collect::<Vec<_>>();
            let state = if states.contains(&"fail") {
                "fail"
            } else if states.contains(&"skip") {
                "skip"
            } else {
                "pass"
            };
            aggregate.insert(test_id, state.to_string());
        }
        if aggregate != gold_expected {
            errors.push(
                "per-run parsed statuses do not reproduce status.json post_patch_status"
                    .to_string(),
            );
        }
    } else {
        errors.push("one or more gold-run parsed statuses could not be validated".to_string());
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(RuntimeEvidenceSummary {
        container_image_id: image_ids[0].clone(),
        network_mode: "none".to_string(),
        repo_head_sha: expected_base_commit,
        runtime_platform: "linux".to_string(),
        workspace_clean: true,
        container_ids: run_ids,
    })
}

fn hash_field(row: &Value, field: &str) -> Result<String, String> {
    let value = row
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("missing or empty field {field}"))?;
    // Hash the exact field bytes. Leading/trailing whitespace in task text or
    // patches is semantically relevant and must not be normalized away.
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
    let runtime_evidence_path = task_dir.join("validation_evidence.json");
    let runtime_evidence_bytes = match fs::read(&runtime_evidence_path) {
        Ok(bytes) => Some(bytes),
        Err(error) => {
            unresolved = true;
            reasons.push(format!("cannot read runtime evidence: {error}"));
            None
        }
    };
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
    let runtime_evidence_sha256 = runtime_evidence_bytes
        .as_ref()
        .map(|bytes| sha256_bytes(bytes));
    let baseline_log_sha256 = baseline_log_bytes.as_ref().map(|bytes| sha256_bytes(bytes));
    let log_sha256 = log_bytes.as_ref().map(|bytes| sha256_bytes(bytes));
    let mut baseline_count = 0;
    let mut gold_count = 0;
    let mut baseline_pass_count = 0;
    let mut explicit_fail_to_pass_count = 0;
    let mut gold_nonpassing_count = 0;
    let mut marker_counts = vec![0; GOLD_RUN_MARKERS.len()];
    let mut runtime_summary: Option<RuntimeEvidenceSummary> = None;

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

                if let Some(evidence_bytes) = runtime_evidence_bytes.as_ref() {
                    match validate_runtime_evidence(
                        evidence_bytes,
                        row,
                        instance_id,
                        validation_dir,
                        &status,
                    ) {
                        Ok(summary) => runtime_summary = Some(summary),
                        Err(evidence_errors) => {
                            unresolved = true;
                            reasons.extend(
                                evidence_errors
                                    .into_iter()
                                    .map(|error| format!("runtime evidence audit: {error}")),
                            );
                        }
                    }
                }
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
                    let (_, explicit_f2p) = transition_sets(baseline, gold);
                    explicit_fail_to_pass_count = explicit_f2p.len();
                    if let (Some(listed_f2p), Some(listed_p2p)) =
                        (listed_f2p.as_ref(), listed_p2p.as_ref())
                    {
                        reasons.extend(transition_contract_errors(
                            baseline, gold, listed_f2p, listed_p2p,
                        ));
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
        candidate_row_sha256: candidate_row_sha256.clone(),
        validation_status_sha256: status_json_sha256.clone(),
        baseline_validation_log_sha256: baseline_log_sha256.clone(),
        gold_validation_log_sha256: log_sha256.clone(),
        runtime_evidence_sha256: runtime_evidence_sha256.clone(),
        container_image_id: runtime_summary
            .as_ref()
            .map(|summary| summary.container_image_id.clone()),
        repo_head_sha: runtime_summary
            .as_ref()
            .map(|summary| summary.repo_head_sha.clone()),
        runtime_platform: runtime_summary
            .as_ref()
            .map(|summary| summary.runtime_platform.clone()),
        network_mode: runtime_summary
            .as_ref()
            .map(|summary| summary.network_mode.clone()),
        workspace_clean: runtime_summary
            .as_ref()
            .map(|summary| summary.workspace_clean),
    };

    let manifest_record = if eligible {
        Some(TaskManifestRecord {
            instance_id: instance_id.to_string(),
            candidate_row_sha256: candidate_row_sha256.clone(),
            repo: repo.expect("eligible task has repo"),
            base_commit: base_commit.expect("eligible task has base commit"),
            repo_head_sha: runtime_summary
                .as_ref()
                .expect("eligible task has runtime summary")
                .repo_head_sha
                .clone(),
            runtime_platform: runtime_summary
                .as_ref()
                .expect("eligible task has runtime summary")
                .runtime_platform
                .clone(),
            docker_image: docker_image.expect("eligible task has docker image"),
            problem_statement_sha256: problem_statement_sha256.expect("eligible task has prompt"),
            gold_patch_sha256: gold_patch_sha256.expect("eligible task has gold patch"),
            test_patch_sha256: test_patch_sha256.expect("eligible task has test patch"),
            validation_status_sha256: status_json_sha256.expect("eligible task has status file"),
            baseline_validation_log_sha256: baseline_log_sha256
                .expect("eligible task has baseline log"),
            gold_validation_log_sha256: log_sha256.expect("eligible task has gold validation log"),
            runtime_evidence_sha256: runtime_evidence_sha256
                .expect("eligible task has runtime evidence"),
            container_image_id: runtime_summary
                .as_ref()
                .expect("eligible task has runtime summary")
                .container_image_id
                .clone(),
            network_mode: runtime_summary
                .as_ref()
                .expect("eligible task has runtime summary")
                .network_mode
                .clone(),
            workspace_clean: runtime_summary
                .as_ref()
                .expect("eligible task has runtime summary")
                .workspace_clean,
            container_ids: runtime_summary
                .as_ref()
                .expect("eligible task has runtime summary")
                .container_ids
                .clone(),
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
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
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
            format!(
                "candidate JSONL line {} is invalid JSON: {error}",
                line_index + 1
            )
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
    let candidate_ids = candidates
        .iter()
        .map(|(id, _, _)| id.clone())
        .collect::<Vec<_>>();
    let candidate_ids_sha256 = canonical_id_digest(&candidate_ids);

    let mut outcomes = Vec::with_capacity(candidates.len());
    let mut manifest_tasks = Vec::new();
    let mut agent_prompts_by_id = BTreeMap::<String, String>::new();
    for (instance_id, row, row_sha256) in candidates {
        let outcome = audit_candidate(&row, &instance_id, &cli.validation_dir, row_sha256);
        if let Some(record) = outcome.manifest_record.as_ref() {
            let problem_statement = row
                .get("problem_statement")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("eligible task {instance_id} lost its problem statement"))?;
            agent_prompts_by_id.insert(instance_id.clone(), problem_statement.to_string());
            manifest_tasks.push(TaskManifestRecord {
                instance_id: record.instance_id.clone(),
                candidate_row_sha256: record.candidate_row_sha256.clone(),
                repo: record.repo.clone(),
                base_commit: record.base_commit.clone(),
                repo_head_sha: record.repo_head_sha.clone(),
                runtime_platform: record.runtime_platform.clone(),
                docker_image: record.docker_image.clone(),
                problem_statement_sha256: record.problem_statement_sha256.clone(),
                gold_patch_sha256: record.gold_patch_sha256.clone(),
                test_patch_sha256: record.test_patch_sha256.clone(),
                validation_status_sha256: record.validation_status_sha256.clone(),
                baseline_validation_log_sha256: record.baseline_validation_log_sha256.clone(),
                gold_validation_log_sha256: record.gold_validation_log_sha256.clone(),
                runtime_evidence_sha256: record.runtime_evidence_sha256.clone(),
                container_image_id: record.container_image_id.clone(),
                network_mode: record.network_mode.clone(),
                workspace_clean: record.workspace_clean,
                container_ids: record.container_ids.clone(),
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
    let rejected_count = outcomes
        .iter()
        .filter(|task| !task.eligible && !task.unresolved)
        .count();
    let unresolved_count = outcomes.iter().filter(|task| task.unresolved).count();
    let lockable = unresolved_count == 0 && !manifest_tasks.is_empty();

    let (
        agent_prompt_bytes,
        controller_manifest_bytes,
        agent_prompt_projection_sha256,
        controller_manifest_sha256,
    ) = if lockable {
        let mut prompt_bytes = Vec::new();
        for task in &manifest_tasks {
            let problem_statement =
                agent_prompts_by_id.get(&task.instance_id).ok_or_else(|| {
                    format!(
                        "missing agent prompt for eligible task {}",
                        task.instance_id
                    )
                })?;
            let row = serde_json::to_vec(&AgentPromptRecord {
                problem_statement: problem_statement.clone(),
            })
            .map_err(|error| format!("cannot serialize safe agent prompt: {error}"))?;
            prompt_bytes.extend_from_slice(&row);
            prompt_bytes.push(b'\n');
        }
        let prompt_sha256 = sha256_bytes(&prompt_bytes);
        let controller = ControllerManifest {
                format_version: MANIFEST_VERSION,
                protocol_id: PROTOCOL_ID.to_string(),
                dataset_revision: cli.dataset_revision.clone(),
                upstream_evaluator_revision: cli.upstream_code_revision.clone(),
                source_parquet_sha256: actual_parquet_sha256.clone(),
                candidate_jsonl_sha256: candidate_jsonl_sha256.clone(),
                eligible_task_ids_sha256: eligible_task_ids_sha256.clone(),
                agent_prompt_projection_sha256: prompt_sha256.clone(),
                task_count: manifest_tasks.len(),
                tasks: manifest_tasks
                    .iter()
                    .enumerate()
                    .map(|(run_index, task)| ControllerTaskRecord {
                        run_index,
                        instance_id: task.instance_id.clone(),
                        repo_head_sha: task.repo_head_sha.clone(),
                        runtime_platform: task.runtime_platform.clone(),
                        docker_image_reference: task.docker_image.clone(),
                        container_image_id: task.container_image_id.clone(),
                        network_mode: task.network_mode.clone(),
                        workspace_clean: task.workspace_clean,
                    })
                    .collect(),
                visibility_warning: "EVALUATOR-ONLY: never expose this controller manifest to the coding agent. Agent-visible JSONL contains only problem_statement. Retain the configured image reference and use container_image_id as the observed immutable runtime identity when reproducing the task environment.".to_string(),
            };
        let mut controller_bytes = serde_json::to_vec_pretty(&controller)
            .map_err(|error| format!("cannot serialize controller manifest: {error}"))?;
        controller_bytes.push(b'\n');
        let controller_sha256 = sha256_bytes(&controller_bytes);
        (
            Some(prompt_bytes),
            Some(controller_bytes),
            Some(prompt_sha256),
            Some(controller_sha256),
        )
    } else {
        (None, None, None, None)
    };

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
        agent_prompt_projection_sha256: agent_prompt_projection_sha256.clone(),
        controller_manifest_sha256: controller_manifest_sha256.clone(),
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
    fs::write(
        &cli.agent_prompts_out,
        agent_prompt_bytes.expect("lockable report has safe agent prompts"),
    )
    .map_err(|error| format!("cannot write {}: {error}", cli.agent_prompts_out.display()))?;
    fs::write(
        &cli.controller_manifest_out,
        controller_manifest_bytes.expect("lockable report has controller manifest"),
    )
    .map_err(|error| {
        format!(
            "cannot write {}: {error}",
            cli.controller_manifest_out.display()
        )
    })?;
    println!(
        "RUST REPAIR MANIFEST LOCKED: {} eligible / {} candidate tasks; task records SHA-256 {}",
        manifest.eligible_task_count, manifest.candidate_row_count, manifest.task_records_sha256
    );
    println!("Audit report: {}", cli.report_out.display());
    println!("Task manifest: {}", cli.manifest_out.display());
    println!(
        "Agent prompts (problem_statement only): {}",
        cli.agent_prompts_out.display()
    );
    println!(
        "Controller manifest (evaluator-only): {}",
        cli.controller_manifest_out.display()
    );
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
    fn missing_baseline_test_is_not_a_fail_to_pass_transition() {
        let baseline = status_map(&[("existing", "pass")]);
        let gold = status_map(&[("existing", "pass"), ("appeared-after", "pass")]);
        let (baseline_pass, explicit_f2p) = transition_sets(&baseline, &gold);
        assert_eq!(baseline_pass, BTreeSet::from(["existing".to_string()]));
        assert!(explicit_f2p.is_empty());
        let errors = transition_contract_errors(
            &baseline,
            &gold,
            &BTreeSet::from(["appeared-after".to_string()]),
            &baseline_pass,
        );
        assert!(errors.iter().any(|error| error.contains("FAIL_TO_PASS")));
    }

    #[test]
    fn baseline_passes_cannot_be_dropped_after_a_gold_patch_regression() {
        let baseline = status_map(&[("must-stay-green", "pass"), ("bug", "fail")]);
        let gold = status_map(&[("must-stay-green", "fail"), ("bug", "pass")]);
        let (baseline_pass, explicit_f2p) = transition_sets(&baseline, &gold);
        assert_eq!(
            baseline_pass,
            BTreeSet::from(["must-stay-green".to_string()])
        );
        assert_eq!(explicit_f2p, BTreeSet::from(["bug".to_string()]));
        let errors = transition_contract_errors(&baseline, &gold, &explicit_f2p, &BTreeSet::new());
        assert!(errors.iter().any(|error| error.contains("PASS_TO_PASS")));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("baseline-passing test"))
        );
    }

    #[test]
    fn valid_explicit_transitions_have_no_contract_errors() {
        let baseline = status_map(&[("regression-test", "pass"), ("fix-me", "fail")]);
        let gold = status_map(&[("regression-test", "pass"), ("fix-me", "pass")]);
        let f2p = BTreeSet::from(["fix-me".to_string()]);
        let p2p = BTreeSet::from(["regression-test".to_string()]);
        assert!(transition_contract_errors(&baseline, &gold, &f2p, &p2p).is_empty());
    }

    #[test]
    fn three_gold_run_markers_must_be_present_exactly_once() {
        let valid = "eval No.0 \npass\neval No.1 \npass\neval No.2 \npass";
        assert_eq!(run_marker_counts(valid), vec![1, 1, 1]);
        assert_eq!(
            run_marker_counts("eval No.0 \neval No.1 \neval No.1"),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn canonical_ids_are_order_independent_after_sorting() {
        let mut ids_a = vec!["b".to_string(), "a".to_string()];
        ids_a.sort();
        let ids_b = vec!["a".to_string(), "b".to_string()];
        assert_eq!(canonical_id_digest(&ids_a), canonical_id_digest(&ids_b));
    }

    fn write_candidate_artifacts(
        root: &Path,
        instance_id: &str,
        baseline: &[(&str, &str)],
        gold: &[(&str, &str)],
        listed_f2p: &[&str],
        listed_p2p: &[&str],
        gold_log: &str,
    ) -> Value {
        let task_dir = root.join(instance_id);
        fs::create_dir_all(&task_dir).unwrap();
        let baseline_status = baseline
            .iter()
            .map(|(id, state)| ((*id).to_string(), (*state).to_string()))
            .collect::<BTreeMap<_, _>>();
        let gold_status = gold
            .iter()
            .map(|(id, state)| ((*id).to_string(), (*state).to_string()))
            .collect::<BTreeMap<_, _>>();
        let status = serde_json::json!({
            "instance_id": instance_id,
            "pre_patch_status": baseline_status,
            "post_patch_status": gold_status,
            "FAIL_TO_PASS": listed_f2p,
            "PASS_TO_PASS": listed_p2p
        });
        fs::write(
            task_dir.join("status.json"),
            serde_json::to_vec(&status).unwrap(),
        )
        .unwrap();

        let baseline_log = "baseline: complete test log\n";
        fs::write(task_dir.join("pre_patch_log.txt"), baseline_log).unwrap();
        fs::write(task_dir.join("post_patch_log.txt"), gold_log).unwrap();
        let image_reference = "test-image@sha256:deadbeef";
        let image_id = format!("sha256:{}", "b".repeat(64));
        let gold_test_exit_code = if gold.iter().all(|(_, state)| *state == "pass") {
            0
        } else {
            1
        };

        let mut gold_runs = Vec::new();
        for run_index in 0..3 {
            let run_log = format!("gold run {run_index} complete test log\n");
            fs::write(task_dir.join(format!("gold_run_{run_index}.log")), &run_log).unwrap();
            gold_runs.push(serde_json::json!({
                "role": "gold",
                "run_index": run_index,
                "test_patch_applied": true,
                "solution_patch_applied": true,
                "workspace_clean_exit_code": 0,
                "repo_head_exit_code": 0,
                "repo_head_sha": "a".repeat(40),
                "rebuild_exit_code": 0,
                "test_exit_code": gold_test_exit_code,
                "print_exit_code": 0,
                "container_identity": {
                    "container_id": format!("{:064x}", run_index + 2),
                    "configured_image": image_reference,
                    "container_image_id": image_id,
                    "runtime_platform": "linux",
                    "network_mode": "none",
                    "git_metadata_sanitization_exit_code": 0,
                    "repo_digests": []
                },
                "log_sha256": sha256_bytes(run_log.as_bytes()),
                "parsed_status": gold_status
            }));
        }
        let evidence = serde_json::json!({
            "schema_version": 1,
            "instance_id": instance_id,
            "image_reference": image_reference,
            "baseline": {
                "role": "baseline",
                "test_patch_applied": true,
                "workspace_clean_exit_code": 0,
                "repo_head_exit_code": 0,
                "repo_head_sha": "a".repeat(40),
                "rebuild_exit_code": 0,
                "test_exit_code": 1,
                "print_exit_code": 0,
                "container_identity": {
                    "container_id": format!("{:064x}", 1),
                    "configured_image": image_reference,
                    "container_image_id": image_id,
                    "runtime_platform": "linux",
                    "network_mode": "none",
                    "git_metadata_sanitization_exit_code": 0,
                    "repo_digests": []
                },
                "log_sha256": sha256_bytes(baseline_log.as_bytes()),
                "parsed_status": baseline_status
            },
            "gold_runs": gold_runs
        });
        fs::write(
            task_dir.join("validation_evidence.json"),
            serde_json::to_vec(&evidence).unwrap(),
        )
        .unwrap();

        serde_json::json!({
            "instance_id": instance_id,
            "repo": "owner/repo",
            "base_commit": "a".repeat(40),
            "docker_image": image_reference,
            "problem_statement": "Fix the documented bug.",
            "patch": "diff --git a/src/lib.rs b/src/lib.rs\n+fn fixed() {}",
            "test_patch": "diff --git a/tests/lib.rs b/tests/lib.rs\n+assert!(fixed());"
        })
    }

    #[test]
    fn valid_task_artifacts_produce_hashed_manifest_record() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-issue-1";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail"), ("regression-guard", "pass")],
            &[("bug-fixed", "pass"), ("regression-guard", "pass")],
            &["bug-fixed"],
            &["regression-guard"],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"candidate row"),
        );
        assert!(result.audit.eligible, "{:?}", result.audit.reasons);
        let record = result.manifest_record.expect("eligible task has manifest");
        assert_eq!(record.candidate_row_sha256, sha256_bytes(b"candidate row"));
        assert_eq!(record.fail_to_pass_count, 1);
        assert_eq!(record.pass_to_pass_count, 1);
        assert_eq!(
            record.baseline_validation_log_sha256,
            sha256_bytes(b"baseline: complete test log\n")
        );
        assert_eq!(
            record.gold_validation_log_sha256,
            sha256_bytes(b"eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n")
        );
        assert_eq!(
            record.container_image_id,
            format!("sha256:{}", "b".repeat(64))
        );
        assert_eq!(record.container_ids.len(), 4);
        let evidence_bytes = fs::read(
            root.path()
                .join(instance_id)
                .join("validation_evidence.json"),
        )
        .unwrap();
        assert_eq!(
            record.runtime_evidence_sha256,
            sha256_bytes(&evidence_bytes)
        );
    }

    #[test]
    fn gold_patch_regression_is_rejected_even_if_upstream_drops_the_guard() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-regression";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail"), ("must-remain-green", "pass")],
            &[("bug-fixed", "pass"), ("must-remain-green", "fail")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"regression row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.manifest_record.is_none());
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("PASS_TO_PASS"))
        );
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("baseline-passing test"))
        );
    }

    #[test]
    fn incomplete_gold_run_log_is_unresolved_and_cannot_be_locked() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-incomplete";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n",
        );
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"incomplete row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(result.manifest_record.is_none());
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("eval No.2"))
        );
    }

    #[test]
    fn missing_baseline_test_cannot_be_misclassified_as_fixed() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-missing-baseline";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("stable", "pass")],
            &[("stable", "pass"), ("appeared-after", "pass")],
            &["appeared-after"],
            &["stable"],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"missing baseline row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(result.manifest_record.is_none());
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("test-ID sets differ"))
        );
    }

    #[test]
    fn non_linux_runtime_cannot_be_locked_under_linux_only_isolation() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-windows-runtime";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&bytes).unwrap();
        evidence["gold_runs"][0]["container_identity"]["runtime_platform"] =
            Value::String("windows".to_string());
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"wrong runtime platform row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(result.audit.reasons.iter().any(|reason| {
            reason.contains("strict Rust repair evaluation requires Linux runtime")
        }));
    }

    #[test]
    fn unsanitized_git_metadata_cannot_be_locked() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-unsanitized-git";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&bytes).unwrap();
        evidence["gold_runs"][1]["container_identity"]["git_metadata_sanitization_exit_code"] =
            Value::from(86);
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"unsanitized git metadata row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| { reason.contains("Git metadata sanitization was not confirmed") })
        );
    }

    #[test]
    fn dirty_pre_patch_worktree_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-dirty-worktree";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&bytes).unwrap();
        evidence["baseline"]["workspace_clean_exit_code"] = Value::from(1);
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"dirty worktree row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(result.audit.reasons.iter().any(|reason| {
            reason.contains("worktree is dirty or cleanliness inspection failed")
        }));
    }

    #[test]
    fn container_checkout_must_match_task_base_commit() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-wrong-checkout";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&bytes).unwrap();
        evidence["gold_runs"][1]["repo_head_sha"] = Value::String("b".repeat(40));
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"wrong checkout row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| { reason.contains("repository HEAD differs from task base_commit") })
        );
    }

    #[test]
    fn bridged_network_container_is_rejected_for_strict_trials() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-network-enabled";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&bytes).unwrap();
        evidence["gold_runs"][1]["container_identity"]["network_mode"] =
            Value::String("bridge".to_string());
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"network enabled row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("network_mode must be 'none'"))
        );
    }

    #[test]
    fn tampered_per_run_log_invalidates_runtime_evidence() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-log-tamper";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        fs::write(
            root.path().join(instance_id).join("gold_run_1.log"),
            "tampered log\n",
        )
        .unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"tampered log row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("raw run log SHA-256 mismatch"))
        );
    }

    #[test]
    fn different_actual_image_ids_across_gold_runs_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-image-drift";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let evidence_bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&evidence_bytes).unwrap();
        evidence["gold_runs"][2]["container_identity"]["container_image_id"] =
            Value::String(format!("sha256:{}", "c".repeat(64)));
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"image drift row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("image ID differs"))
        );
    }

    #[test]
    fn baseline_timeout_cannot_be_treated_as_a_valid_oracle() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-baseline-timeout";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let evidence_bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&evidence_bytes).unwrap();
        evidence["baseline"]["test_exit_code"] = Value::from(124);
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"baseline timeout row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("test command is unresolved"))
        );
    }

    #[test]
    fn unapplied_gold_patch_is_not_accepted_as_validation() {
        let root = tempfile::tempdir().unwrap();
        let instance_id = "owner__repo-patch-rejected";
        let row = write_candidate_artifacts(
            root.path(),
            instance_id,
            &[("bug-fixed", "fail")],
            &[("bug-fixed", "pass")],
            &["bug-fixed"],
            &[],
            "eval No.0\npass\n eval No.1\npass\n eval No.2\npass\n",
        );
        let evidence_path = root
            .path()
            .join(instance_id)
            .join("validation_evidence.json");
        let evidence_bytes = fs::read(&evidence_path).unwrap();
        let mut evidence: Value = serde_json::from_slice(&evidence_bytes).unwrap();
        evidence["gold_runs"][1]["solution_patch_applied"] = Value::Bool(false);
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let result = audit_candidate(
            &row,
            instance_id,
            root.path(),
            sha256_bytes(b"patch rejected row"),
        );
        assert!(!result.audit.eligible);
        assert!(result.audit.unresolved);
        assert!(
            result
                .audit
                .reasons
                .iter()
                .any(|reason| reason.contains("gold solution patch application"))
        );
    }

    #[test]
    fn task_field_hash_preserves_leading_and_trailing_whitespace() {
        let row = serde_json::json!({"problem_statement": "  exact bytes  "});
        assert_eq!(
            hash_field(&row, "problem_statement").unwrap(),
            sha256_bytes(b"  exact bytes  ")
        );
        assert_ne!(
            hash_field(&row, "problem_statement").unwrap(),
            sha256_bytes(b"exact bytes")
        );
    }

    #[test]
    fn agent_prompt_projection_contains_only_problem_statement() {
        let raw_row = serde_json::json!({
            "instance_id": "owner__repo-issue-1",
            "problem_statement": "Fix the bug.",
            "patch": "SECRET_GOLD_PATCH",
            "test_patch": "SECRET_TEST_PATCH",
            "hints_text": "SECRET_HINT",
            "all_hints_text": "SECRET_ALL_HINTS",
            "FAIL_TO_PASS": ["secret-test"],
            "PASS_TO_PASS": ["secret-regression"],
            "commit_url": "https://example.invalid/secret"
        });
        let prompt = AgentPromptRecord {
            problem_statement: required_string(&raw_row, "problem_statement")
                .unwrap()
                .to_string(),
        };
        let projected = serde_json::to_value(prompt).unwrap();
        assert_eq!(
            projected,
            serde_json::json!({"problem_statement": "Fix the bug."})
        );
        let serialized = projected.to_string();
        for forbidden in [
            "SECRET_GOLD_PATCH",
            "SECRET_TEST_PATCH",
            "SECRET_HINT",
            "SECRET_ALL_HINTS",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "commit_url",
            "owner__repo-issue-1",
        ] {
            assert!(!serialized.contains(forbidden), "leaked {forbidden}");
        }
    }

    #[test]
    fn controller_projection_contains_no_patch_or_test_contents() {
        let controller = ControllerTaskRecord {
            run_index: 0,
            instance_id: "owner__repo-issue-1".to_string(),
            repo_head_sha: "a".repeat(40),
            runtime_platform: "linux".to_string(),
            docker_image_reference: "image@sha256:deadbeef".to_string(),
            container_image_id: format!("sha256:{}", "b".repeat(64)),
            network_mode: "none".to_string(),
            workspace_clean: true,
        };
        let projected = serde_json::to_value(controller).unwrap();
        assert_eq!(
            projected,
            serde_json::json!({
                "run_index": 0,
                "instance_id": "owner__repo-issue-1",
                "repo_head_sha": "a".repeat(40),
                "runtime_platform": "linux",
                "docker_image_reference": "image@sha256:deadbeef",
                "container_image_id": format!("sha256:{}", "b".repeat(64)),
                "network_mode": "none",
                "workspace_clean": true
            })
        );
        let serialized = projected.to_string();
        assert!(!serialized.contains("patch"));
        assert!(!serialized.contains("FAIL_TO_PASS"));
        assert!(!serialized.contains("PASS_TO_PASS"));
    }

    #[test]
    fn sha256_matches_standard_known_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
