//! Independent pass@1 scorer for SWE-bench-Live-style prediction-run evidence.
//!
//! This audits *scored agent runs*, not task/gold validation. It joins the pinned
//! task manifest, original candidate JSONL, prediction map, task validation outputs,
//! and per-task scored evaluation sidecars. It does not execute containers or
//! authenticate the machine that produced the evidence.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PROTOCOL_ID: &str = "symthaea-rust-repair";
const REPORT_VERSION: u32 = 1;

#[derive(Debug)]
struct Cli {
    task_manifest: PathBuf,
    expected_task_manifest_sha256: String,
    candidate_jsonl: PathBuf,
    predictions_file: PathBuf,
    task_validation_dir: PathBuf,
    evaluation_dir: PathBuf,
    report_out: PathBuf,
}

#[derive(Debug, Serialize)]
struct TaskRecordForDigest {
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

fn task_record_for_digest(value: &Value) -> Result<TaskRecordForDigest, String> {
    let string = |key: &str| required_string(value, key).map(str::to_string);
    let count = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| format!("task manifest {key} missing or invalid"))
    };
    let container_ids = value
        .get("container_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "task manifest container_ids missing or invalid".to_string())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| "task manifest container_ids contains a non-string".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TaskRecordForDigest {
        instance_id: string("instance_id")?,
        candidate_row_sha256: string("candidate_row_sha256")?,
        repo: string("repo")?,
        base_commit: string("base_commit")?,
        repo_head_sha: string("repo_head_sha")?,
        runtime_platform: string("runtime_platform")?,
        docker_image: string("docker_image")?,
        problem_statement_sha256: string("problem_statement_sha256")?,
        gold_patch_sha256: string("gold_patch_sha256")?,
        test_patch_sha256: string("test_patch_sha256")?,
        validation_status_sha256: string("validation_status_sha256")?,
        baseline_validation_log_sha256: string("baseline_validation_log_sha256")?,
        gold_validation_log_sha256: string("gold_validation_log_sha256")?,
        runtime_evidence_sha256: string("runtime_evidence_sha256")?,
        container_image_id: string("container_image_id")?,
        network_mode: string("network_mode")?,
        workspace_clean: value
            .get("workspace_clean")
            .and_then(Value::as_bool)
            .ok_or_else(|| "task manifest workspace_clean missing or invalid".to_string())?,
        container_ids,
        fail_to_pass_count: count("fail_to_pass_count")?,
        pass_to_pass_count: count("pass_to_pass_count")?,
    })
}

fn classify_scored_patch(
    required_failed: usize,
    patch_applied: Option<bool>,
    test_exit_code: i64,
    non_required_failure_count: usize,
) -> &'static str {
    if required_failed > 0 || patch_applied != Some(true) {
        "fail"
    } else if test_exit_code != 0 && non_required_failure_count == 0 {
        "unresolved"
    } else {
        // Match the task oracle: unrelated baseline-failing tests are not F2P/P2P
        // obligations. Preserve their count and the test exit code in the score report.
        "pass"
    }
}

fn headline_pass_at_1(
    task_count: usize,
    success_count: usize,
    failure_count: usize,
    unresolved_count: usize,
) -> Option<f64> {
    if task_count > 0 && unresolved_count == 0 && success_count + failure_count == task_count {
        Some(success_count as f64 / task_count as f64)
    } else {
        None
    }
}

#[derive(Debug, Serialize)]
struct TaskScore {
    instance_id: String,
    outcome: String,
    reasons: Vec<String>,
    prediction_patch_sha256: Option<String>,
    scored_log_sha256: Option<String>,
    evaluation_evidence_sha256: Option<String>,
    container_image_id: Option<String>,
    required_test_count: usize,
    required_tests_passing: usize,
    runtime_platform_observed: Option<String>,
    workspace_clean_exit_code: Option<i64>,
    network_mode_observed: Option<String>,
    repo_head_sha_observed: Option<String>,
    scored_test_exit_code: Option<i64>,
    non_required_failure_count: usize,
}

#[derive(Debug, Serialize)]
struct ScoreReport {
    format_version: u32,
    protocol_id: String,
    task_manifest_sha256: String,
    candidate_jsonl_sha256: String,
    predictions_sha256: String,
    task_count: usize,
    scored_count: usize,
    success_count: usize,
    failure_count: usize,
    unresolved_count: usize,
    lockable: bool,
    pass_at_1: Option<f64>,
    completed_run_success_rate: Option<f64>,
    interpretation: String,
    tasks: Vec<TaskScore>,
}

fn usage() -> &'static str {
    r#"Usage:
  cargo run --locked --bin rust-repair-score-audit --features rust-repair-manifest -- \
    --task-manifest <locked-task-manifest.json> \
    --expected-task-manifest-sha256 <64-hex-pinned-externally> \
    --candidate-jsonl <full-pinned-rust-candidate-split.jsonl> \
    --predictions-file <prediction-map.json> \
    --task-validation-dir <gold-validation-output-dir> \
    --evaluation-dir <scored-agent-output-dir> \
    --report-out <score-audit-report.json>

Predictions may be an ID-keyed JSON object, standard JSONL records, or a JSON array
of records. Record objects use the fields "instance_id" and "model_patch".
The scored evaluation directory must contain <instance_id>/evaluation_evidence.json,
status.json, and post_patch_log.txt. Missing or unresolved runs stay in the
assigned denominator and prevent a lockable pass@1 result."#
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
    let allowed = [
        "--task-manifest",
        "--expected-task-manifest-sha256",
        "--candidate-jsonl",
        "--predictions-file",
        "--task-validation-dir",
        "--evaluation-dir",
        "--report-out",
    ];
    if let Some(unknown) = values.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unknown argument: {unknown}"));
    }
    let expected = take("--expected-task-manifest-sha256")?.to_ascii_lowercase();
    if !is_hex(&expected, 64) {
        return Err("--expected-task-manifest-sha256 must be 64 hex characters".to_string());
    }
    let output = take("--report-out")?;
    for input_key in ["--task-manifest", "--candidate-jsonl", "--predictions-file"] {
        if values.get(input_key).is_some_and(|path| path == &output) {
            return Err(format!(
                "report output must not overwrite input {input_key}"
            ));
        }
    }
    Ok(Some(Cli {
        task_manifest: PathBuf::from(take("--task-manifest")?),
        expected_task_manifest_sha256: expected,
        candidate_jsonl: PathBuf::from(take("--candidate-jsonl")?),
        predictions_file: PathBuf::from(take("--predictions-file")?),
        task_validation_dir: PathBuf::from(take("--task-validation-dir")?),
        evaluation_dir: PathBuf::from(take("--evaluation-dir")?),
        report_out: PathBuf::from(output),
    }))
}

fn is_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("malformed JSON in {}: {error}", path.display()))
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}

fn optional_i64(value: &Value, key: &str) -> Result<Option<i64>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(number) => number
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("{key} is not an integer or null")),
    }
}

fn parse_status_map(value: &Value) -> Result<BTreeMap<String, String>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "test status must be a JSON object".to_string())?;
    if object.is_empty() {
        return Err("test status object is empty".to_string());
    }
    let mut out = BTreeMap::new();
    for (test_id, raw_status) in object {
        let status = raw_status
            .as_str()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| format!("test status for {test_id:?} is not a string"))?;
        if !matches!(status.as_str(), "pass" | "fail" | "skip") {
            return Err(format!("unknown test status {status:?} for {test_id:?}"));
        }
        out.insert(test_id.clone(), status);
    }
    Ok(out)
}

fn candidate_rows(
    path: &Path,
    expected_sha256: &str,
) -> Result<(BTreeMap<String, (Value, String)>, String), String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let digest = sha256_bytes(&bytes);
    if digest != expected_sha256 {
        return Err(format!(
            concat!(
                "candidate JSONL SHA-256 does not match the locked task ",
                "manifest: expected {}, observed {}"
            ),
            expected_sha256,
            digest
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("candidate JSONL is not UTF-8: {error}"))?;
    let mut rows = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("candidate JSONL line {} is empty", index + 1));
        }
        let row: Value = serde_json::from_str(line).map_err(|error| {
            format!(
                "candidate JSONL line {} is invalid JSON: {error}",
                index + 1
            )
        })?;
        let id = required_string(&row, "instance_id")
            .map_err(|error| format!("candidate JSONL line {}: {error}", index + 1))?
            .to_string();
        let row_sha256 = sha256_bytes(line.as_bytes());
        if rows.insert(id.clone(), (row, row_sha256)).is_some() {
            return Err(format!("duplicate candidate instance_id {id:?}"));
        }
    }
    Ok((rows, digest))
}

fn insert_prediction_record(
    out: &mut BTreeMap<String, Option<String>>,
    record: &Value,
) -> Result<(), String> {
    let instance_id = required_string(record, "instance_id")?.to_string();
    let patch = record
        .get("model_patch")
        .and_then(Value::as_str)
        .map(str::to_string);
    if out.insert(instance_id.clone(), patch).is_some() {
        return Err(format!("duplicate prediction instance_id {instance_id:?}"));
    }
    Ok(())
}

fn predictions_map(path: &Path) -> Result<(BTreeMap<String, Option<String>>, String), String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let digest = sha256_bytes(&bytes);
    let mut out = BTreeMap::new();

    if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
        match value {
            Value::Object(object) if object.contains_key("instance_id") => {
                insert_prediction_record(&mut out, &Value::Object(object))?;
            }
            Value::Object(object) => {
                for (instance_id, prediction) in object {
                    // Missing/malformed patch text remains a task-level unresolved
                    // attempt, preserving the locked benchmark denominator.
                    let patch = prediction
                        .get("model_patch")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    out.insert(instance_id, patch);
                }
            }
            Value::Array(records) => {
                for record in records {
                    insert_prediction_record(&mut out, &record)?;
                }
            }
            _ => {
                return Err(
                    "prediction JSON must be an ID-keyed map, an array of records, or JSONL"
                        .to_string(),
                );
            }
        }
    } else {
        let text = std::str::from_utf8(&bytes)
            .map_err(|error| format!("prediction JSONL is not UTF-8: {error}"))?;
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                return Err(format!("prediction JSONL line {} is empty", index + 1));
            }
            let record: Value = serde_json::from_str(line).map_err(|error| {
                format!(
                    "prediction JSONL line {} is invalid JSON: {error}",
                    index + 1
                )
            })?;
            insert_prediction_record(&mut out, &record)
                .map_err(|error| format!("prediction JSONL line {}: {error}", index + 1))?;
        }
    }

    if out.is_empty() {
        return Err("prediction input contains zero task records".to_string());
    }
    Ok((out, digest))
}

fn validate_locked_runtime_logs(
    task: &Value,
    candidate: &Value,
    status: &Value,
    task_validation_dir: &Path,
    errors: &mut Vec<String>,
) {
    let instance_id = match required_string(task, "instance_id") {
        Ok(id) => id,
        Err(error) => {
            errors.push(error);
            return;
        }
    };
    let task_dir = task_validation_dir.join(instance_id);
    let evidence_path = task_dir.join("validation_evidence.json");
    let evidence = match read_json(&evidence_path) {
        Ok(value) => value,
        Err(error) => {
            errors.push(error);
            return;
        }
    };
    let expected_image_reference = candidate
        .get("docker_image")
        .and_then(Value::as_str)
        .unwrap_or("");
    if evidence.get("image_reference").and_then(Value::as_str) != Some(expected_image_reference) {
        errors.push(
            "task validation runtime evidence image reference differs from candidate".to_string(),
        );
    }
    let expected_image_id = task
        .get("container_image_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    let expected_repo_head = task
        .get("repo_head_sha")
        .and_then(Value::as_str)
        .unwrap_or("");
    let candidate_base_commit = candidate
        .get("base_commit")
        .and_then(Value::as_str)
        .unwrap_or("");
    if expected_repo_head.is_empty() || expected_repo_head != candidate_base_commit {
        errors.push("locked task repository HEAD differs from candidate base_commit".to_string());
    }
    if task.get("runtime_platform").and_then(Value::as_str) != Some("linux") {
        errors.push("locked task manifest runtime_platform must be 'linux'".to_string());
    }
    if task.get("network_mode").and_then(Value::as_str) != Some("none") {
        errors.push("locked task manifest network_mode must be 'none'".to_string());
    }
    if task.get("workspace_clean").and_then(Value::as_bool) != Some(true) {
        errors.push("locked task manifest workspace_clean must be true".to_string());
    }
    let expected_container_ids = task
        .get("container_ids")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if expected_container_ids.len() != 4 {
        errors.push("locked task manifest must contain four validation container IDs".to_string());
    }

    let baseline = evidence.get("baseline").unwrap_or(&Value::Null);
    if baseline.get("role").and_then(Value::as_str) != Some("baseline") {
        errors.push("runtime sidecar baseline role is invalid".to_string());
    }
    if !workspace_is_clean(baseline) {
        errors.push("baseline worktree was dirty or cleanliness inspection failed".to_string());
    }
    if !repo_head_matches(
        expected_repo_head,
        baseline.get("repo_head_sha").and_then(Value::as_str),
        baseline.get("repo_head_exit_code").and_then(Value::as_i64),
    ) {
        errors.push(
            "baseline repository HEAD differs from locked task base commit or inspection failed"
                .to_string(),
        );
    }
    let baseline_identity = baseline.get("container_identity").unwrap_or(&Value::Null);
    if baseline_identity
        .get("runtime_platform")
        .and_then(Value::as_str)
        != Some("linux")
    {
        errors.push("baseline validation runtime platform must be 'linux'".to_string());
    }
    if baseline_identity
        .get("git_metadata_sanitization_exit_code")
        .and_then(Value::as_i64)
        != Some(0)
    {
        errors.push("baseline Git metadata sanitization was not confirmed".to_string());
    }
    if baseline_identity
        .get("network_mode")
        .and_then(Value::as_str)
        != Some("none")
    {
        errors.push("baseline validation container network_mode must be 'none'".to_string());
    }
    let baseline_id = baseline_identity
        .get("container_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    if expected_container_ids.first().and_then(Value::as_str) != Some(baseline_id) {
        errors.push("baseline container ID differs from locked task manifest".to_string());
    }
    if baseline_identity
        .get("container_image_id")
        .and_then(Value::as_str)
        != Some(expected_image_id)
    {
        errors.push("baseline container image ID differs from locked task manifest".to_string());
    }
    if baseline.get("test_patch_applied").and_then(Value::as_bool) != Some(true) {
        errors
            .push("baseline runtime evidence does not confirm test patch application".to_string());
    }
    for field in ["rebuild_exit_code", "print_exit_code"] {
        match optional_i64(baseline, field) {
            Ok(Some(0)) => {}
            Ok(value) => errors.push(format!("baseline {field} unresolved or nonzero: {value:?}")),
            Err(error) => errors.push(error),
        }
    }
    let baseline_test_exit = optional_i64(baseline, "test_exit_code");
    match baseline_test_exit {
        Ok(Some(code)) if code != 124 && code >= 0 => {}
        Ok(other) => errors.push(format!(
            "baseline test command has unresolved exit code: {other:?}"
        )),
        Err(error) => errors.push(error),
    }
    let baseline_log = task_dir.join("pre_patch_log.txt");
    match sha256_file(&baseline_log) {
        Ok(actual) => {
            if baseline.get("log_sha256").and_then(Value::as_str) != Some(actual.as_str()) {
                errors.push("baseline raw log digest differs from runtime evidence".to_string());
            }
        }
        Err(error) => errors.push(error),
    }
    let baseline_expected = status
        .get("pre_patch_status")
        .and_then(|value| parse_status_map(value).ok())
        .unwrap_or_default();
    let baseline_observed = baseline
        .get("parsed_status")
        .and_then(|value| parse_status_map(value).ok())
        .unwrap_or_default();
    if baseline_expected != baseline_observed {
        errors.push("baseline runtime parsed statuses differ from status.json".to_string());
    }

    let gold_runs = match evidence.get("gold_runs").and_then(Value::as_array) {
        Some(runs) if runs.len() == 3 => runs,
        Some(runs) => {
            errors.push(format!(
                "task validation runtime evidence has {} gold runs; expected 3",
                runs.len()
            ));
            return;
        }
        None => {
            errors.push("task validation runtime evidence gold_runs missing".to_string());
            return;
        }
    };
    let expected_test_ids = baseline_expected.keys().cloned().collect::<BTreeSet<_>>();
    let mut parsed_runs = Vec::new();
    for index in 0..3 {
        let run = &gold_runs[index];
        if run.get("role").and_then(Value::as_str) != Some("gold")
            || run.get("run_index").and_then(Value::as_u64) != Some(index as u64)
        {
            errors.push(format!("gold runtime run {index} role or index is invalid"));
        }
        if run.get("test_patch_applied").and_then(Value::as_bool) != Some(true)
            || run.get("solution_patch_applied").and_then(Value::as_bool) != Some(true)
        {
            errors.push(format!(
                "gold runtime run {index} did not confirm both patches applied"
            ));
        }
        for field in ["rebuild_exit_code", "test_exit_code", "print_exit_code"] {
            match optional_i64(run, field) {
                Ok(Some(0)) => {}
                Ok(value) => errors.push(format!(
                    "gold runtime run {index} {field} unresolved or nonzero: {value:?}"
                )),
                Err(error) => errors.push(error),
            }
        }

        if !workspace_is_clean(run) {
            errors.push(format!(
                "gold runtime run {index} worktree was dirty or cleanliness inspection failed"
            ));
        }
        if !repo_head_matches(
            expected_repo_head,
            run.get("repo_head_sha").and_then(Value::as_str),
            run.get("repo_head_exit_code").and_then(Value::as_i64),
        ) {
            errors.push(format!(
                "gold runtime run {index} repository HEAD differs from locked task base commit or inspection failed"
            ));
        }
        let identity = run.get("container_identity").unwrap_or(&Value::Null);
        if identity.get("runtime_platform").and_then(Value::as_str) != Some("linux") {
            errors.push(format!(
                "gold runtime run {index} runtime platform must be 'linux'"
            ));
        }
        if identity
            .get("git_metadata_sanitization_exit_code")
            .and_then(Value::as_i64)
            != Some(0)
        {
            errors.push(format!(
                "gold runtime run {index} Git metadata sanitization was not confirmed"
            ));
        }
        if identity.get("network_mode").and_then(Value::as_str) != Some("none") {
            errors.push(format!(
                "gold runtime run {index} network_mode must be 'none'"
            ));
        }
        let container_id = identity
            .get("container_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        if expected_container_ids
            .get(index + 1)
            .and_then(Value::as_str)
            != Some(container_id)
        {
            errors.push(format!(
                "gold runtime run {index} container ID differs from locked task manifest"
            ));
        }
        if identity.get("container_image_id").and_then(Value::as_str) != Some(expected_image_id) {
            errors.push(format!(
                "gold runtime run {index} image ID differs from locked task manifest"
            ));
        }

        let log_path = task_dir.join(format!("gold_run_{index}.log"));
        match sha256_file(&log_path) {
            Ok(actual) => {
                if run.get("log_sha256").and_then(Value::as_str) != Some(actual.as_str()) {
                    errors.push(format!("gold runtime run {index} raw log digest mismatch"));
                }
            }
            Err(error) => errors.push(error),
        }
        let statuses = run
            .get("parsed_status")
            .and_then(|value| parse_status_map(value).ok())
            .unwrap_or_default();
        let ids = statuses.keys().cloned().collect::<BTreeSet<_>>();
        if ids != expected_test_ids {
            errors.push(format!(
                "gold runtime run {index} test-ID set differs from baseline"
            ));
        }
        if statuses.values().any(|state| state != "pass") {
            errors.push(format!(
                "gold runtime run {index} contains a failing or skipped test"
            ));
        }
        parsed_runs.push(statuses);
    }
    if parsed_runs.len() == 3 {
        let mut aggregate = BTreeMap::new();
        for test_id in &expected_test_ids {
            let states = parsed_runs
                .iter()
                .map(|run| run.get(test_id).map(String::as_str).unwrap_or("skip"))
                .collect::<Vec<_>>();
            let state = if states.contains(&"fail") {
                "fail"
            } else if states.contains(&"skip") {
                "skip"
            } else {
                "pass"
            };
            aggregate.insert(test_id.clone(), state.to_string());
        }
        let recorded_gold = status
            .get("post_patch_status")
            .and_then(|value| parse_status_map(value).ok())
            .unwrap_or_default();
        if aggregate != recorded_gold {
            errors.push(
                "recomputed per-run gold statuses differ from recorded aggregate".to_string(),
            );
        }
    }
}

fn validate_task_oracle(
    task: &Value,
    candidate: &Value,
    candidate_row_sha256: &str,
    task_validation_dir: &Path,
) -> Result<(BTreeSet<String>, BTreeSet<String>, BTreeSet<String>), Vec<String>> {
    let id = required_string(task, "instance_id").map_err(|error| vec![error])?;
    let mut errors = Vec::new();
    if task.get("candidate_row_sha256").and_then(Value::as_str) != Some(candidate_row_sha256) {
        errors.push("candidate JSONL row digest differs from locked task manifest".to_string());
    }
    let task_dir = task_validation_dir.join(id);
    let status_path = task_dir.join("status.json");
    let evidence_path = task_dir.join("validation_evidence.json");
    let baseline_log_path = task_dir.join("pre_patch_log.txt");
    let gold_log_path = task_dir.join("post_patch_log.txt");

    for (path, expected_key) in [
        (&status_path, "validation_status_sha256"),
        (&baseline_log_path, "baseline_validation_log_sha256"),
        (&gold_log_path, "gold_validation_log_sha256"),
        (&evidence_path, "runtime_evidence_sha256"),
    ] {
        let expected = task.get(expected_key).and_then(Value::as_str).unwrap_or("");
        match sha256_file(path) {
            Ok(actual) if !expected.is_empty() && actual == expected => {}
            Ok(actual) => errors.push(format!(
                concat!(
                    "task validation artifact {} digest differs from manifest ",
                    "(expected {}, observed {})"
                ),
                path.display(),
                expected,
                actual
            )),
            Err(error) => errors.push(error),
        }
    }
    let status = match read_json(&status_path) {
        Ok(value) => value,
        Err(error) => return Err(vec![error]),
    };
    validate_locked_runtime_logs(task, candidate, &status, task_validation_dir, &mut errors);

    let baseline = match status
        .get("pre_patch_status")
        .and_then(|v| parse_status_map(v).ok())
    {
        Some(map) => map,
        None => {
            return Err(vec![
                "task validation pre_patch_status missing or malformed".to_string(),
            ]);
        }
    };
    let gold = match status
        .get("post_patch_status")
        .and_then(|v| parse_status_map(v).ok())
    {
        Some(map) => map,
        None => {
            return Err(vec![
                "task validation post_patch_status missing or malformed".to_string(),
            ]);
        }
    };
    let baseline_ids = baseline.keys().cloned().collect::<BTreeSet<_>>();
    let gold_ids = gold.keys().cloned().collect::<BTreeSet<_>>();
    if baseline_ids != gold_ids {
        errors.push("task validation baseline and gold test-ID sets differ".to_string());
    }

    let baseline_pass = baseline
        .iter()
        .filter(|(_, state)| state.as_str() == "pass")
        .map(|(test_id, _)| test_id.clone())
        .collect::<BTreeSet<_>>();
    let explicit_f2p = baseline
        .iter()
        .filter(|(test_id, state)| {
            state.as_str() == "fail"
                && gold
                    .get(*test_id)
                    .is_some_and(|gold_state| gold_state == "pass")
        })
        .map(|(test_id, _)| test_id.clone())
        .collect::<BTreeSet<_>>();
    if explicit_f2p.is_empty() {
        errors.push(
            "task has no explicit fail-to-pass test after independent recomputation".to_string(),
        );
    }
    if task.get("fail_to_pass_count").and_then(Value::as_u64) != Some(explicit_f2p.len() as u64) {
        errors.push("manifest fail_to_pass_count disagrees with recomputed oracle".to_string());
    }
    if task.get("pass_to_pass_count").and_then(Value::as_u64) != Some(baseline_pass.len() as u64) {
        errors
            .push("manifest pass_to_pass_count disagrees with baseline passing tests".to_string());
    }
    let test_patch = candidate
        .get("test_patch")
        .and_then(Value::as_str)
        .ok_or_else(|| vec!["candidate test_patch missing".to_string()])?;
    let test_patch_sha = sha256_bytes(test_patch.as_bytes());
    if task.get("test_patch_sha256").and_then(Value::as_str) != Some(test_patch_sha.as_str()) {
        errors.push("candidate test_patch digest differs from locked task manifest".to_string());
    }
    if errors.is_empty() {
        Ok((baseline_ids, explicit_f2p, baseline_pass))
    } else {
        Err(errors)
    }
}

fn repo_head_matches(
    expected_base_commit: &str,
    observed_repo_head: Option<&str>,
    command_exit_code: Option<i64>,
) -> bool {
    (is_hex(expected_base_commit, 40) || is_hex(expected_base_commit, 64))
        && command_exit_code == Some(0)
        && observed_repo_head
            .is_some_and(|observed| observed.eq_ignore_ascii_case(expected_base_commit))
}

fn workspace_is_clean(run: &Value) -> bool {
    run.get("workspace_clean_exit_code").and_then(Value::as_i64) == Some(0)
}

fn canonical_id_digest(ids: &BTreeSet<String>) -> String {
    let mut canonical = ids.iter().cloned().collect::<Vec<_>>().join("\n");
    if !canonical.is_empty() {
        canonical.push('\n');
    }
    sha256_bytes(canonical.as_bytes())
}

fn validate_scored_container_identity(
    task: &Value,
    candidate: &Value,
    evidence: &Value,
) -> Result<(String, String), Vec<String>> {
    let mut errors = Vec::new();
    let identity = evidence
        .get("container_identity")
        .and_then(Value::as_object);
    let identity = match identity {
        Some(identity) => identity,
        None => {
            return Err(vec![
                "scored container_identity is missing or malformed".to_string(),
            ]);
        }
    };
    if identity
        .get("git_metadata_sanitization_exit_code")
        .and_then(Value::as_i64)
        != Some(0)
    {
        errors.push("Git metadata sanitization was not confirmed".to_string());
    }
    let runtime_platform = identity
        .get("runtime_platform")
        .and_then(Value::as_str)
        .unwrap_or("");
    if task.get("runtime_platform").and_then(Value::as_str) != Some("linux")
        || runtime_platform != "linux"
    {
        errors.push("strict Rust repair scoring requires Linux runtime platform".to_string());
    }
    let configured_image = identity
        .get("configured_image")
        .and_then(Value::as_str)
        .unwrap_or("");
    let expected_reference = candidate
        .get("docker_image")
        .and_then(Value::as_str)
        .unwrap_or("");
    if identity.get("network_mode").and_then(Value::as_str) != Some("none") {
        errors.push("scored container network_mode must be 'none'".to_string());
    }
    if configured_image.is_empty() || configured_image != expected_reference {
        errors.push(
            "Docker-reported configured image differs from the candidate image reference"
                .to_string(),
        );
    }

    let container_id = identity
        .get("container_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if !is_hex(&container_id, 64) {
        errors.push("scored container ID must be 64 hexadecimal characters".to_string());
    }
    let image_id = identity
        .get("container_image_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if !image_id.starts_with("sha256:")
        || !is_hex(image_id.strip_prefix("sha256:").unwrap_or(""), 64)
    {
        errors.push("scored container image ID must be sha256:<64 hex characters>".to_string());
    }
    let expected_image_id = task
        .get("container_image_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if task.get("network_mode").and_then(Value::as_str) != Some("none") {
        errors.push("locked task manifest network_mode must be 'none'".to_string());
    }
    if identity.get("network_mode").and_then(Value::as_str) != Some("none") {
        errors.push("scored Docker container network_mode must be 'none'".to_string());
    }
    if image_id != expected_image_id {
        errors.push(
            "scored run actual Docker image ID differs from task-validation image ID".to_string(),
        );
    }

    let validation_ids = task
        .get("container_ids")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if validation_ids
        .iter()
        .any(|id| id.as_str() == Some(container_id.as_str()))
    {
        errors.push("scored container ID was reused from a task-validation run".to_string());
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok((image_id, container_id))
}

fn audit_scored_task(
    task: &Value,
    candidate: &Value,
    candidate_row_sha256: &str,
    prediction_patch: Option<&str>,
    task_validation_dir: &Path,
    evaluation_dir: &Path,
) -> TaskScore {
    let instance_id = task
        .get("instance_id")
        .and_then(Value::as_str)
        .unwrap_or("<missing-id>")
        .to_string();
    let mut reasons = Vec::new();
    let patch_sha = prediction_patch.map(|patch| sha256_bytes(patch.as_bytes()));
    let mut scored_log_sha = None;
    let mut evaluation_evidence_sha = None;
    let mut image_id_observed = None;
    let mut workspace_clean_exit_code = None;
    let mut network_mode_observed = None;
    let mut repo_head_sha_observed = None;
    let mut runtime_platform_observed = None;

    let (all_test_ids, explicit_f2p, baseline_pass) =
        match validate_task_oracle(task, candidate, candidate_row_sha256, task_validation_dir) {
            Ok(sets) => sets,
            Err(errors) => {
                return TaskScore {
                    instance_id,
                    outcome: "unresolved".to_string(),
                    reasons: errors,
                    prediction_patch_sha256: patch_sha,
                    scored_log_sha256: None,
                    evaluation_evidence_sha256: None,
                    container_image_id: None,
                    network_mode_observed: None,
                    repo_head_sha_observed: None,
                    runtime_platform_observed: None,
                    required_test_count: 0,
                    required_tests_passing: 0,
                    workspace_clean_exit_code: None,
                    scored_test_exit_code: None,
                    non_required_failure_count: 0,
                };
            }
        };
    let required_tests = explicit_f2p
        .union(&baseline_pass)
        .cloned()
        .collect::<BTreeSet<_>>();
    if prediction_patch.is_none() {
        return TaskScore {
            instance_id,
            outcome: "unresolved".to_string(),
            reasons: vec!["no prediction entry for an assigned task".to_string()],
            prediction_patch_sha256: None,
            scored_log_sha256: None,
            evaluation_evidence_sha256: None,
            container_image_id: None,
            network_mode_observed: None,
            repo_head_sha_observed: None,
            runtime_platform_observed: None,
            required_test_count: required_tests.len(),
            required_tests_passing: 0,
            workspace_clean_exit_code: None,
            scored_test_exit_code: None,
            non_required_failure_count: 0,
        };
    }

    let task_dir = evaluation_dir.join(&instance_id);
    let evidence_path = task_dir.join("evaluation_evidence.json");
    let log_path = task_dir.join("post_patch_log.txt");
    let status_path = task_dir.join("status.json");
    let evidence_bytes = match fs::read(&evidence_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return TaskScore {
                instance_id,
                outcome: "unresolved".to_string(),
                reasons: vec![format!("cannot read scored evaluation evidence: {error}")],
                prediction_patch_sha256: patch_sha,
                scored_log_sha256: None,
                evaluation_evidence_sha256: None,
                container_image_id: None,
                network_mode_observed: None,
                repo_head_sha_observed: None,
                runtime_platform_observed: None,
                required_test_count: required_tests.len(),
                required_tests_passing: 0,
                workspace_clean_exit_code: None,
                scored_test_exit_code: None,
                non_required_failure_count: 0,
            };
        }
    };
    evaluation_evidence_sha = Some(sha256_bytes(&evidence_bytes));
    let evidence: Value = match serde_json::from_slice(&evidence_bytes) {
        Ok(value) => value,
        Err(error) => {
            return TaskScore {
                instance_id,
                outcome: "unresolved".to_string(),
                reasons: vec![format!("scored evidence JSON malformed: {error}")],
                prediction_patch_sha256: patch_sha,
                scored_log_sha256: None,
                evaluation_evidence_sha256: evaluation_evidence_sha,
                container_image_id: None,
                network_mode_observed: None,
                repo_head_sha_observed: None,
                runtime_platform_observed: None,
                required_test_count: required_tests.len(),
                required_tests_passing: 0,
                workspace_clean_exit_code: None,
                scored_test_exit_code: None,
                non_required_failure_count: 0,
            };
        }
    };

    workspace_clean_exit_code = evidence
        .get("workspace_clean_exit_code")
        .and_then(Value::as_i64);
    if workspace_clean_exit_code != Some(0) {
        reasons.push("scored worktree was dirty or cleanliness inspection failed".to_string());
    }
    network_mode_observed = evidence
        .get("container_identity")
        .and_then(|identity| identity.get("network_mode"))
        .and_then(Value::as_str)
        .map(str::to_string);
    repo_head_sha_observed = evidence
        .get("repo_head_sha")
        .and_then(Value::as_str)
        .map(str::to_string);
    runtime_platform_observed = evidence
        .get("container_identity")
        .and_then(|identity| identity.get("runtime_platform"))
        .and_then(Value::as_str)
        .map(str::to_string);

    if evidence.get("schema_version").and_then(Value::as_u64) != Some(1) {
        reasons.push("scored evidence schema_version must be 1".to_string());
    }
    if evidence.get("role").and_then(Value::as_str) != Some("scored_agent_patch") {
        reasons.push("scored evidence role must be scored_agent_patch".to_string());
    }
    if evidence.get("instance_id").and_then(Value::as_str) != Some(instance_id.as_str()) {
        reasons.push("scored evidence instance_id differs from assigned task".to_string());
    }
    let expected_base_commit = candidate
        .get("base_commit")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !repo_head_matches(
        expected_base_commit,
        repo_head_sha_observed.as_deref(),
        evidence.get("repo_head_exit_code").and_then(Value::as_i64),
    ) || task.get("repo_head_sha").and_then(Value::as_str) != Some(expected_base_commit)
    {
        reasons.push(
            "scored repository HEAD differs from locked task base_commit or inspection failed"
                .to_string(),
        );
    }
    if evidence
        .get("prediction_patch_sha256")
        .and_then(Value::as_str)
        != patch_sha.as_deref()
    {
        reasons.push("scored prediction patch digest differs from prediction file".to_string());
    }
    let candidate_test_patch = candidate
        .get("test_patch")
        .and_then(Value::as_str)
        .unwrap_or("");
    let expected_test_patch_sha256 = sha256_bytes(candidate_test_patch.as_bytes());
    if evidence.get("test_patch_sha256").and_then(Value::as_str)
        != Some(expected_test_patch_sha256.as_str())
    {
        reasons.push("scored test-patch digest differs from pinned task patch".to_string());
    }
    let configured_image = candidate
        .get("docker_image")
        .and_then(Value::as_str)
        .unwrap_or("");
    if evidence.get("image_reference").and_then(Value::as_str) != Some(configured_image) {
        reasons.push("scored configured image reference differs from candidate".to_string());
    }

    match validate_scored_container_identity(task, candidate, &evidence) {
        Ok((actual_image_id, _container_id)) => {
            image_id_observed = Some(actual_image_id);
        }
        Err(identity_errors) => reasons.extend(identity_errors),
    }

    match sha256_file(&log_path) {
        Ok(actual) => {
            scored_log_sha = Some(actual.clone());
            if evidence.get("log_sha256").and_then(Value::as_str) != Some(actual.as_str()) {
                reasons.push(
                    "scored raw log digest differs from evaluation_evidence.json".to_string(),
                );
            }
        }
        Err(error) => reasons.push(error),
    }

    let scored_status_value = match read_json(&status_path) {
        Ok(value) => value,
        Err(error) => {
            reasons.push(error);
            Value::Null
        }
    };
    let sidecar_status_value = evidence
        .get("parsed_status")
        .cloned()
        .unwrap_or(Value::Null);
    let scored_status = match parse_status_map(&scored_status_value) {
        Ok(map) => map,
        Err(error) => {
            reasons.push(format!("scored status.json is invalid: {error}"));
            BTreeMap::new()
        }
    };
    let sidecar_status = match parse_status_map(&sidecar_status_value) {
        Ok(map) => map,
        Err(error) => {
            reasons.push(format!("scored evidence parsed_status is invalid: {error}"));
            BTreeMap::new()
        }
    };
    if scored_status != sidecar_status {
        reasons.push("scored status.json differs from the embedded parsed_status".to_string());
    }
    if scored_status.keys().cloned().collect::<BTreeSet<_>>() != all_test_ids {
        reasons.push("scored status test-ID set differs from the audited task oracle".to_string());
    }

    let test_patch_applied = evidence.get("test_patch_applied").and_then(Value::as_bool);
    if test_patch_applied != Some(true) {
        reasons.push("scored run did not confirm successful test_patch application".to_string());
    }
    let rebuild_code = optional_i64(&evidence, "rebuild_exit_code");
    let test_code = optional_i64(&evidence, "test_exit_code");
    let print_code = optional_i64(&evidence, "print_exit_code");
    let candidate_has_rebuild = candidate
        .get("rebuild_cmds")
        .and_then(Value::as_array)
        .is_some_and(|cmds| !cmds.is_empty());
    match rebuild_code {
        Ok(Some(0)) => {}
        Ok(None) if !candidate_has_rebuild => {}
        Ok(other) => reasons.push(format!(
            "scored rebuild exit code is unresolved or nonzero: {other:?}"
        )),
        Err(error) => reasons.push(error),
    }
    match &test_code {
        Ok(Some(code)) if *code == 124 || *code < 0 => {
            reasons.push(format!(
                "scored test command timed out or was terminated (exit code {code})"
            ));
        }
        Ok(None) => reasons.push("scored test command has no exit code".to_string()),
        Ok(Some(_)) => {}
        Err(error) => reasons.push(error.clone()),
    }
    match print_code {
        Ok(Some(0)) => {}
        Ok(other) => reasons.push(format!(
            "scored log/print command did not exit zero: {other:?}"
        )),
        Err(error) => reasons.push(error),
    }

    let patch_applied = evidence
        .get("prediction_patch_applied")
        .and_then(Value::as_bool);
    let mut required_passing = 0;
    let mut required_failed = 0;
    let mut required_skipped = 0;
    let non_required_failure_count = scored_status
        .iter()
        .filter(|(test_id, state)| !required_tests.contains(*test_id) && state.as_str() == "fail")
        .count();
    let scored_test_exit_code = test_code.as_ref().ok().copied().flatten();
    for test_id in &required_tests {
        match scored_status.get(test_id).map(String::as_str) {
            Some("pass") => required_passing += 1,
            Some("fail") => required_failed += 1,
            Some("skip") | None => required_skipped += 1,
            _ => required_skipped += 1,
        }
    }
    if required_skipped > 0 {
        reasons.push(format!(
            "{required_skipped} required test(s) are skipped or missing from the scored run"
        ));
    }
    if !reasons.is_empty() {
        return TaskScore {
            instance_id,
            outcome: "unresolved".to_string(),
            reasons,
            prediction_patch_sha256: patch_sha,
            scored_log_sha256: scored_log_sha,
            evaluation_evidence_sha256: evaluation_evidence_sha,
            container_image_id: image_id_observed.clone(),
            network_mode_observed: network_mode_observed.clone(),
            repo_head_sha_observed: repo_head_sha_observed.clone(),
            runtime_platform_observed: runtime_platform_observed.clone(),
            required_test_count: required_tests.len(),
            required_tests_passing: required_passing,
            workspace_clean_exit_code,
            scored_test_exit_code,
            non_required_failure_count,
        };
    }

    let exit_code = test_code.as_ref().ok().copied().flatten().unwrap_or(-1);
    match classify_scored_patch(
        required_failed,
        patch_applied,
        exit_code,
        non_required_failure_count,
    ) {
        "fail" => TaskScore {
            instance_id,
            outcome: "fail".to_string(),
            reasons: vec![if patch_applied != Some(true) {
                "prediction patch was not applied".to_string()
            } else {
                format!("{required_failed} required regression/fail-to-pass test(s) failed")
            }],
            prediction_patch_sha256: patch_sha,
            scored_log_sha256: scored_log_sha,
            evaluation_evidence_sha256: evaluation_evidence_sha,
            container_image_id: image_id_observed.clone(),
            network_mode_observed: network_mode_observed.clone(),
            repo_head_sha_observed: repo_head_sha_observed.clone(),
            runtime_platform_observed: runtime_platform_observed.clone(),
            required_test_count: required_tests.len(),
            required_tests_passing: required_passing,
            workspace_clean_exit_code,
            scored_test_exit_code,
            non_required_failure_count,
        },
        "unresolved" => TaskScore {
            instance_id,
            outcome: "unresolved".to_string(),
            reasons: vec![
                "test command exited nonzero but no parsed test failure explains the exit"
                    .to_string(),
            ],
            prediction_patch_sha256: patch_sha,
            scored_log_sha256: scored_log_sha,
            evaluation_evidence_sha256: evaluation_evidence_sha,
            container_image_id: image_id_observed.clone(),
            network_mode_observed: network_mode_observed.clone(),
            repo_head_sha_observed: repo_head_sha_observed.clone(),
            runtime_platform_observed: runtime_platform_observed.clone(),
            required_test_count: required_tests.len(),
            required_tests_passing: required_passing,
            workspace_clean_exit_code,
            scored_test_exit_code,
            non_required_failure_count,
        },
        _ => TaskScore {
            instance_id,
            outcome: "pass".to_string(),
            reasons: vec![],
            prediction_patch_sha256: patch_sha,
            scored_log_sha256: scored_log_sha,
            evaluation_evidence_sha256: evaluation_evidence_sha,
            container_image_id: image_id_observed.clone(),
            network_mode_observed: network_mode_observed.clone(),
            repo_head_sha_observed: repo_head_sha_observed.clone(),
            runtime_platform_observed: runtime_platform_observed.clone(),
            required_test_count: required_tests.len(),
            required_tests_passing: required_passing,
            workspace_clean_exit_code,
            scored_test_exit_code,
            non_required_failure_count,
        },
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize report: {error}"))?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn run(cli: Cli) -> Result<(), String> {
    let task_manifest_bytes = fs::read(&cli.task_manifest)
        .map_err(|error| format!("cannot read task manifest: {error}"))?;
    let task_manifest_sha256 = sha256_bytes(&task_manifest_bytes);
    if task_manifest_sha256 != cli.expected_task_manifest_sha256 {
        return Err(format!(
            "task manifest digest mismatch: expected {}, observed {}",
            cli.expected_task_manifest_sha256, task_manifest_sha256
        ));
    }
    let task_manifest: Value = serde_json::from_slice(&task_manifest_bytes)
        .map_err(|error| format!("task manifest JSON invalid: {error}"))?;
    if task_manifest.get("format_version").and_then(Value::as_u64) != Some(1)
        || task_manifest.get("protocol_id").and_then(Value::as_str) != Some(PROTOCOL_ID)
    {
        return Err("unsupported task manifest format/protocol".to_string());
    }
    let expected_candidate_sha =
        required_string(&task_manifest, "candidate_jsonl_sha256")?.to_string();
    let (candidate_rows, candidate_jsonl_sha256) =
        candidate_rows(&cli.candidate_jsonl, &expected_candidate_sha)?;
    let expected_source_sha = required_string(&task_manifest, "source_parquet_sha256")?;
    if !is_hex(expected_source_sha, 64) {
        return Err("locked task manifest source_parquet_sha256 is malformed".to_string());
    }
    required_string(&task_manifest, "dataset_revision")?;
    required_string(&task_manifest, "upstream_evaluator_revision")?;
    let observed_candidate_ids = candidate_rows.keys().cloned().collect::<BTreeSet<_>>();
    if task_manifest
        .get("candidate_ids_sha256")
        .and_then(Value::as_str)
        != Some(canonical_id_digest(&observed_candidate_ids).as_str())
    {
        return Err("candidate ID digest differs from locked task manifest".to_string());
    }
    let (predictions, predictions_sha256) = predictions_map(&cli.predictions_file)?;

    let task_list = task_manifest
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "task manifest missing tasks array".to_string())?;
    if task_list.is_empty() {
        return Err("task manifest has zero eligible tasks".to_string());
    }
    if task_manifest
        .get("eligible_task_count")
        .and_then(Value::as_u64)
        != Some(task_list.len() as u64)
    {
        return Err("task manifest eligible_task_count differs from its task array".to_string());
    }
    let expected_candidate_rows = task_manifest
        .get("candidate_row_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| "task manifest missing candidate_row_count".to_string())?;
    if candidate_rows.len() as u64 != expected_candidate_rows {
        return Err(format!(
            concat!(
                "candidate row count differs from locked manifest: expected ",
                "{expected_candidate_rows}, observed {}"
            ),
            candidate_rows.len()
        ));
    }
    let mut tasks_by_id = BTreeMap::<String, Value>::new();
    for task in task_list {
        let id = required_string(task, "instance_id")?.to_string();
        if tasks_by_id.insert(id.clone(), task.clone()).is_some() {
            return Err(format!("duplicate task manifest instance_id {id:?}"));
        }
        if !candidate_rows.contains_key(&id) {
            return Err(format!(
                "manifest task {id:?} is absent from pinned candidate JSONL"
            ));
        }
    }
    let manifest_ids = tasks_by_id.keys().cloned().collect::<BTreeSet<_>>();
    if task_manifest
        .get("eligible_task_ids_sha256")
        .and_then(Value::as_str)
        != Some(canonical_id_digest(&manifest_ids).as_str())
    {
        return Err("eligible task ID digest differs from locked task manifest".to_string());
    }
    let mut canonical_tasks = task_list
        .iter()
        .map(task_record_for_digest)
        .collect::<Result<Vec<_>, _>>()?;
    canonical_tasks.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    let canonical_task_bytes = serde_json::to_vec(&canonical_tasks)
        .map_err(|error| format!("cannot serialize locked tasks for digest: {error}"))?;
    if task_manifest
        .get("task_records_sha256")
        .and_then(Value::as_str)
        != Some(sha256_bytes(&canonical_task_bytes).as_str())
    {
        return Err("task_records_sha256 differs from canonical task records".to_string());
    }
    let prediction_ids = predictions.keys().cloned().collect::<BTreeSet<_>>();
    let extra_predictions = prediction_ids
        .difference(&manifest_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !extra_predictions.is_empty() {
        return Err(format!(
            "prediction file contains {} task ID(s) outside the locked manifest; first is {:?}",
            extra_predictions.len(),
            extra_predictions.first()
        ));
    }

    let mut results = Vec::new();
    for (id, task) in &tasks_by_id {
        let (candidate, candidate_row_sha256) =
            candidate_rows.get(id).expect("manifest ID checked above");
        results.push(audit_scored_task(
            task,
            candidate,
            candidate_row_sha256,
            predictions.get(id).and_then(Option::as_deref),
            &cli.task_validation_dir,
            &cli.evaluation_dir,
        ));
    }
    let task_count = results.len();
    let success_count = results
        .iter()
        .filter(|result| result.outcome == "pass")
        .count();
    let failure_count = results
        .iter()
        .filter(|result| result.outcome == "fail")
        .count();
    let unresolved_count = results
        .iter()
        .filter(|result| result.outcome == "unresolved")
        .count();
    let scored_count = success_count + failure_count;
    let lockable = unresolved_count == 0 && scored_count == task_count;
    let report = ScoreReport {
        format_version: REPORT_VERSION,
        protocol_id: PROTOCOL_ID.to_string(),
        task_manifest_sha256,
        candidate_jsonl_sha256,
        predictions_sha256,
        task_count,
        scored_count,
        success_count,
        failure_count,
        unresolved_count,
        lockable,
        pass_at_1: headline_pass_at_1(task_count, success_count, failure_count, unresolved_count),
        completed_run_success_rate: if scored_count > 0 {
            Some(success_count as f64 / scored_count as f64)
        } else {
            None
        },
        interpretation: if lockable {
            concat!(
                "Complete scored-run audit. pass_at_1 uses the full locked task manifest ",
                "denominator and is eligible for external review, not automatically an SOTA claim."
            )
            .to_string()
        } else {
            concat!(
                "NOT LOCKABLE: unresolved or missing scored-run evidence remains in the ",
                "denominator. completed_run_success_rate is diagnostic only and must not be ",
                "reported as pass@1."
            )
            .to_string()
        },
        tasks: results,
    };
    write_json(&cli.report_out, &report)?;
    if !lockable {
        return Err(format!(
            "scored audit not lockable: {} unresolved task(s); report written to {}",
            unresolved_count,
            cli.report_out.display()
        ));
    }
    println!(
        "SCORED RUN AUDIT: {}/{} pass@1 ({} failures); report {}",
        success_count,
        task_count,
        failure_count,
        cli.report_out.display()
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

    #[test]
    fn sha256_standard_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn required_tests_include_fail_to_pass_and_all_baseline_passes() {
        let f2p = BTreeSet::from(["bug".to_string()]);
        let p2p = BTreeSet::from(["regression".to_string()]);
        let required = f2p.union(&p2p).cloned().collect::<BTreeSet<_>>();
        assert_eq!(
            required,
            BTreeSet::from(["bug".to_string(), "regression".to_string()])
        );
    }

    #[test]
    fn no_headline_pass_at_1_when_any_task_is_unresolved() {
        let task_count = 5;
        let success_count = 3;
        let unresolved_count = 1;
        let lockable = unresolved_count == 0;
        let pass_at_1 = if lockable {
            Some(success_count as f64 / task_count as f64)
        } else {
            None
        };
        assert_eq!(pass_at_1, None);
    }

    #[test]
    fn candidate_rows_bind_manifest_hashes_to_exact_jsonl_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("candidates.jsonl");
        let line = r#"{"instance_id":"task-a","problem_statement":"Fix it"}"#;
        let contents = format!("{line}\n");
        fs::write(&path, &contents).unwrap();
        let (rows, digest) = candidate_rows(&path, &sha256_bytes(contents.as_bytes())).unwrap();
        assert_eq!(digest, sha256_bytes(contents.as_bytes()));
        let (row, row_digest) = rows.get("task-a").unwrap();
        assert_eq!(row["problem_statement"], "Fix it");
        assert_eq!(*row_digest, sha256_bytes(line.as_bytes()));
    }

    #[test]
    fn standard_swebench_jsonl_predictions_are_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("predictions.jsonl");
        let contents = concat!(
            "{\"instance_id\":\"task-a\",\"model_patch\":\"diff --git a/a b/a\\n+x\\n\"}\n",
            "{\"instance_id\":\"task-b\",\"model_patch\":\"diff --git a/b b/b\\n+y\\n\"}\n"
        );
        fs::write(&path, contents).unwrap();
        let (predictions, digest) = predictions_map(&path).unwrap();
        assert_eq!(predictions.len(), 2);
        assert!(
            predictions
                .get("task-a")
                .unwrap()
                .as_ref()
                .unwrap()
                .contains("+x")
        );
        assert!(
            predictions
                .get("task-b")
                .unwrap()
                .as_ref()
                .unwrap()
                .contains("+y")
        );
        assert_eq!(digest, sha256_bytes(contents.as_bytes()));
    }

    #[test]
    fn duplicate_jsonl_prediction_ids_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duplicate.jsonl");
        let contents = concat!(
            "{\"instance_id\":\"task-a\",\"model_patch\":\"one\"}\n",
            "{\"instance_id\":\"task-a\",\"model_patch\":\"two\"}\n"
        );
        fs::write(&path, contents).unwrap();
        assert!(
            predictions_map(&path)
                .unwrap_err()
                .contains("duplicate prediction instance_id")
        );
    }

    #[test]
    fn missing_patch_payload_is_preserved_as_a_task_level_unresolved_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("predictions.json");
        fs::write(
            &path,
            r#"{"task-present": {"model_patch": "diff --git ..."}, "task-missing": {}}"#,
        )
        .unwrap();
        let (predictions, digest) = predictions_map(&path).unwrap();
        assert!(is_hex(&digest, 64));
        assert_eq!(
            predictions.get("task-present"),
            Some(&Some("diff --git ...".to_string()))
        );
        assert_eq!(predictions.get("task-missing"), Some(&None));
        assert!(!predictions.contains_key("task-never-emitted"));
    }

    #[test]
    fn scored_container_rejects_unsanitized_git_metadata() {
        let image_id = format!("sha256:{}", "c".repeat(64));
        let task = serde_json::json!({
            "container_image_id": image_id,
            "network_mode": "none",
            "runtime_platform": "linux",
            "repo_head_sha": "a".repeat(40),
            "container_ids": [format!("{:064x}", 1), format!("{:064x}", 2)]
        });
        let candidate = serde_json::json!({"docker_image": "example@sha256:deadbeef"});
        let evidence = serde_json::json!({
            "container_identity": {
                "container_id": format!("{:064x}", 9),
                "configured_image": "example@sha256:deadbeef",
                "container_image_id": image_id,
                "network_mode": "none",
                "git_metadata_sanitization_exit_code": 86
            }
        });
        let errors = validate_scored_container_identity(&task, &candidate, &evidence).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| { error.contains("Git metadata sanitization was not confirmed") })
        );
    }

    #[test]
    fn scored_container_rejects_non_linux_platform() {
        let image_id = format!("sha256:{}", "c".repeat(64));
        let task = serde_json::json!({
            "container_image_id": image_id,
            "network_mode": "none",
            "runtime_platform": "linux",
            "repo_head_sha": "a".repeat(40),
            "container_ids": [format!("{:064x}", 1), format!("{:064x}", 2)]
        });
        let candidate = serde_json::json!({"docker_image": "example@sha256:deadbeef"});
        let evidence = serde_json::json!({
            "container_identity": {
                "container_id": format!("{:064x}", 9),
                "configured_image": "example@sha256:deadbeef",
                "container_image_id": image_id,
                "network_mode": "none",
                "runtime_platform": "windows",
                "git_metadata_sanitization_exit_code": 0
            }
        });
        let errors = validate_scored_container_identity(&task, &candidate, &evidence).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.contains("strict Rust repair scoring requires Linux runtime platform")
        }));
    }

    #[test]
    fn scored_container_rejects_a_bridge_network() {
        let image_id = format!("sha256:{}", "c".repeat(64));
        let task = serde_json::json!({
            "container_image_id": image_id,
            "container_ids": [format!("{:064x}", 1), format!("{:064x}", 2)],
            "network_mode": "none"
        });
        let candidate = serde_json::json!({"docker_image": "example@sha256:deadbeef"});
        let evidence = serde_json::json!({
            "container_identity": {
                "container_id": format!("{:064x}", 9),
                "configured_image": "example@sha256:deadbeef",
                "container_image_id": image_id,
                "network_mode": "bridge"
            }
        });
        let errors = validate_scored_container_identity(&task, &candidate, &evidence).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("network_mode must be 'none'"))
        );
    }

    #[test]
    fn scored_container_requires_valid_ids_and_the_locked_image() {
        let image_id = format!("sha256:{}", "c".repeat(64));
        let task = serde_json::json!({
            "container_image_id": image_id,
            "network_mode": "none",
            "repo_head_sha": "a".repeat(40),
            "container_ids": [format!("{:064x}", 1), format!("{:064x}", 2)]
        });
        let candidate = serde_json::json!({"docker_image": "example@sha256:deadbeef"});
        let evidence = serde_json::json!({
            "container_identity": {
                "container_id": format!("{:064x}", 9),
                "configured_image": "example@sha256:deadbeef",
                "container_image_id": image_id,
                "network_mode": "none",
                "runtime_platform": "linux",
                "git_metadata_sanitization_exit_code": 0
            }
        });
        assert!(validate_scored_container_identity(&task, &candidate, &evidence).is_ok());

        let mut bad_image = evidence.clone();
        bad_image["container_identity"]["container_image_id"] =
            Value::String(format!("sha256:{}", "d".repeat(64)));
        assert!(
            validate_scored_container_identity(&task, &candidate, &bad_image)
                .unwrap_err()
                .iter()
                .any(|error| error.contains("differs from task-validation"))
        );

        let mut reused = evidence.clone();
        reused["container_identity"]["container_id"] = Value::String(format!("{:064x}", 2));
        assert!(
            validate_scored_container_identity(&task, &candidate, &reused)
                .unwrap_err()
                .iter()
                .any(|error| error.contains("reused"))
        );
    }

    #[test]
    fn pass_at_1_is_suppressed_when_any_assigned_task_is_unresolved() {
        assert_eq!(headline_pass_at_1(10, 7, 2, 1), None);
        assert_eq!(headline_pass_at_1(10, 7, 3, 0), Some(0.7));
        assert_eq!(headline_pass_at_1(0, 0, 0, 0), None);
    }

    #[test]
    fn workspace_cleanliness_is_fail_closed() {
        assert!(workspace_is_clean(&serde_json::json!({
            "workspace_clean_exit_code": 0
        })));
        assert!(!workspace_is_clean(&serde_json::json!({
            "workspace_clean_exit_code": 1
        })));
        assert!(!workspace_is_clean(&serde_json::json!({})));
        assert!(!workspace_is_clean(&serde_json::json!({
            "workspace_clean_exit_code": null
        })));
    }

    #[test]
    fn repository_head_must_match_pinned_task_commit() {
        let base = "a".repeat(40);
        let wrong = "b".repeat(40);
        assert!(repo_head_matches(&base, Some(base.as_str()), Some(0)));
        assert!(!repo_head_matches(&base, Some(wrong.as_str()), Some(0)));
        assert!(!repo_head_matches(&base, Some(base.as_str()), Some(1)));
        assert!(!repo_head_matches(&base, None, Some(0)));
    }

    #[test]
    fn required_oracle_failures_fail_even_when_other_tests_pass() {
        assert_eq!(classify_scored_patch(1, Some(true), 1, 0), "fail");
    }

    #[test]
    fn unrelated_baseline_failure_does_not_invalidate_the_required_oracle() {
        assert_eq!(classify_scored_patch(0, Some(true), 1, 1), "pass");
    }

    #[test]
    fn unexplained_nonzero_test_exit_is_unresolved() {
        assert_eq!(classify_scored_patch(0, Some(true), 1, 0), "unresolved");
    }

    #[test]
    fn unapplied_prediction_patch_is_a_failure() {
        assert_eq!(classify_scored_patch(0, Some(false), 0, 0), "fail");
    }

    #[test]
    fn canonical_task_id_digest_is_order_independent() {
        let first = BTreeSet::from(["task-b".to_string(), "task-a".to_string()]);
        let second = BTreeSet::from(["task-a".to_string(), "task-b".to_string()]);
        assert_eq!(canonical_id_digest(&first), canonical_id_digest(&second));
        assert_ne!(
            canonical_id_digest(&first),
            canonical_id_digest(&BTreeSet::from(["task-a".to_string()]))
        );
    }

    #[test]
    fn score_evidence_is_not_the_gold_validation_evidence() {
        let evidence = serde_json::json!({
            "schema_version": 1,
            "role": "scored_agent_patch",
            "prediction_patch_sha256": "a".repeat(64),
            "test_patch_sha256": "b".repeat(64),
            "container_identity": { "container_image_id": format!("sha256:{}", "c".repeat(64)) }
        });
        assert_eq!(evidence["role"], "scored_agent_patch");
        assert!(evidence.get("gold_runs").is_none());
    }
}
