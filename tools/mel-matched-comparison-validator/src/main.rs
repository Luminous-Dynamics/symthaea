use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const SCHEMA_V1: &str = "mel-matched-comparison-run-v1";
const DEFAULT_FILE: &str = "matched-comparison-run-v1.example.json";

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be a JSON object"))
}

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{context} must be a JSON array"))
}

fn string<'a>(value: &'a Value, context: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{context} must be a non-empty string"))
}

fn sha256(value: &Value, context: &str) -> Result<String, String> {
    let value = string(value, context)?;
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(value.to_owned())
}

fn unique_id(value: &Value, context: &str) -> Result<String, String> {
    let value = string(value, context)?;
    if value.len() > 120
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || !value.as_bytes()[0].is_ascii_lowercase()
        || !value.as_bytes()[value.len() - 1].is_ascii_lowercase()
            && !value.as_bytes()[value.len() - 1].is_ascii_digit()
    {
        return Err(format!("{context} is not a stable lowercase identifier"));
    }
    Ok(value.to_owned())
}

fn integer(value: &Value, context: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{context} must be a non-negative integer"))
}

fn exact_keys(
    value: &Value,
    required: &[&str],
    optional: &[&str],
    context: &str,
) -> Result<(), String> {
    let map = object(value, context)?;
    for key in required {
        if !map.contains_key(*key) {
            return Err(format!("{context} is missing required field {key}"));
        }
    }
    let allowed: BTreeSet<&str> = required.iter().chain(optional.iter()).copied().collect();
    for key in map.keys() {
        if !allowed.contains(key.as_str()) {
            return Err(format!("{context} has unknown field {key}"));
        }
    }
    Ok(())
}

fn field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("{context} is missing field {key}"))
}

fn validate_run(run: &Value) -> Result<(), Vec<String>> {
    match validate_run_inner(run) {
        Ok(()) => Ok(()),
        Err(error) => Err(vec![error]),
    }
}

fn validate_run_inner(run: &Value) -> Result<(), String> {
    exact_keys(
        run,
        &[
            "schema_version",
            "run_id",
            "run_status",
            "frozen_at",
            "benchmark",
            "dataset",
            "environment",
            "retry_policy",
            "arms",
            "cases",
            "judgments",
            "metrics",
            "claim_ceiling",
        ],
        &["executed_at", "analyzed_at"],
        "run",
    )?;

    if string(field(run, "schema_version", "run")?, "schema_version")? != SCHEMA_V1 {
        return Err(format!("unsupported schema version; expected {SCHEMA_V1}"));
    }
    let _ = unique_id(field(run, "run_id", "run")?, "run_id")?;
    let status = string(field(run, "run_status", "run")?, "run_status")?;
    if !["frozen_plan", "executed", "analyzed"].contains(&status) {
        return Err(format!("unsupported run_status: {status}"));
    }
    let _ = string(field(run, "frozen_at", "run")?, "frozen_at")?;
    if status == "executed" || status == "analyzed" {
        let _ = string(field(run, "executed_at", "run")?, "executed_at")?;
    }
    if status == "analyzed" {
        let _ = string(field(run, "analyzed_at", "run")?, "analyzed_at")?;
    }

    let benchmark = field(run, "benchmark", "run")?;
    exact_keys(
        benchmark,
        &[
            "benchmark_id",
            "benchmark_version",
            "task_family",
            "task_definition_sha256",
            "scorer_sha256",
            "comparability_group",
        ],
        &["official_split_id", "official_split_sha256"],
        "benchmark",
    )?;
    for key in ["benchmark_id", "benchmark_version", "task_family", "comparability_group"] {
        let _ = string(field(benchmark, key, "benchmark")?, key)?;
    }
    let _ = sha256(field(benchmark, "task_definition_sha256", "benchmark")?, "task_definition_sha256")?;
    let _ = sha256(field(benchmark, "scorer_sha256", "benchmark")?, "scorer_sha256")?;
    if let Some(split_sha) = benchmark.get("official_split_sha256") {
        let _ = sha256(split_sha, "official_split_sha256")?;
    }

    let dataset = field(run, "dataset", "run")?;
    exact_keys(
        dataset,
        &[
            "dataset_id",
            "dataset_version",
            "split_manifest_sha256",
            "rights_clearance_receipt_sha256",
            "permitted_uses",
        ],
        &[],
        "dataset",
    )?;
    for key in ["dataset_id", "dataset_version"] {
        let _ = string(field(dataset, key, "dataset")?, key)?;
    }
    let _ = sha256(field(dataset, "split_manifest_sha256", "dataset")?, "split_manifest_sha256")?;
    let _ = sha256(
        field(dataset, "rights_clearance_receipt_sha256", "dataset")?,
        "rights_clearance_receipt_sha256",
    )?;
    let uses = array(field(dataset, "permitted_uses", "dataset")?, "permitted_uses")?;
    if uses.is_empty() {
        return Err("permitted_uses must not be empty".into());
    }
    let mut seen_uses = BTreeSet::new();
    for item in uses {
        let use_name = string(item, "permitted_uses[]")?;
        if ![
            "evaluation",
            "research_display",
            "model_training",
            "public_release",
            "redistribution",
        ]
        .contains(&use_name)
        {
            return Err(format!("unsupported permitted use: {use_name}"));
        }
        if !seen_uses.insert(use_name) {
            return Err(format!("duplicate permitted use: {use_name}"));
        }
    }

    let environment = field(run, "environment", "run")?;
    exact_keys(
        environment,
        &[
            "source_revision",
            "rust_toolchain",
            "host_os",
            "host_arch",
            "runner_version",
            "environment_sha256",
        ],
        &["renderer_id", "renderer_version", "renderer_config_sha256"],
        "environment",
    )?;
    let revision = string(field(environment, "source_revision", "environment")?, "source_revision")?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("source_revision must be a 40-character lowercase Git SHA".into());
    }
    for key in ["rust_toolchain", "host_os", "host_arch", "runner_version"] {
        let _ = string(field(environment, key, "environment")?, key)?;
    }
    let _ = sha256(field(environment, "environment_sha256", "environment")?, "environment_sha256")?;
    if let Some(renderer_sha) = environment.get("renderer_config_sha256") {
        let _ = sha256(renderer_sha, "renderer_config_sha256")?;
    }

    let retry = field(run, "retry_policy", "run")?;
    exact_keys(
        retry,
        &["max_attempts_per_arm_case", "retry_on", "failed_attempts_consume_budget"],
        &[],
        "retry_policy",
    )?;
    let max_attempts = integer(
        field(retry, "max_attempts_per_arm_case", "retry_policy")?,
        "max_attempts_per_arm_case",
    )?;
    if !(1..=32).contains(&max_attempts) {
        return Err("max_attempts_per_arm_case must be in 1..=32".into());
    }
    if field(retry, "failed_attempts_consume_budget", "retry_policy")?.as_bool() != Some(true) {
        return Err("failed_attempts_consume_budget must be true".into());
    }

    let arms = array(field(run, "arms", "run")?, "arms")?;
    if arms.len() < 2 {
        return Err("at least two comparison arms are required".into());
    }
    let mut arm_ids = BTreeSet::new();
    for arm in arms {
        exact_keys(
            arm,
            &[
                "arm_id",
                "system_name",
                "system_version",
                "adapter_version",
                "modality",
                "execution_surface",
                "model_identity_state",
                "identity_evidence_sha256",
                "control_surface_sha256",
                "inference_budget",
            ],
            &["provider_model_id", "immutable_artifact_sha256"],
            "arm",
        )?;
        let arm_id = unique_id(field(arm, "arm_id", "arm")?, "arm_id")?;
        if !arm_ids.insert(arm_id.clone()) {
            return Err(format!("duplicate arm_id: {arm_id}"));
        }
        for key in [
            "system_name",
            "system_version",
            "adapter_version",
            "modality",
            "execution_surface",
        ] {
            let _ = string(field(arm, key, "arm")?, key)?;
        }
        let identity_state = string(field(arm, "model_identity_state", "arm")?, "model_identity_state")?;
        if ![
            "immutable_verified",
            "provider_declared",
            "local_build_only",
            "unavailable",
        ]
        .contains(&identity_state)
        {
            return Err(format!("unsupported model_identity_state: {identity_state}"));
        }
        let _ = sha256(field(arm, "identity_evidence_sha256", "arm")?, "identity_evidence_sha256")?;
        let _ = sha256(field(arm, "control_surface_sha256", "arm")?, "control_surface_sha256")?;
        if identity_state == "immutable_verified" {
            let immutable = field(arm, "immutable_artifact_sha256", "arm")
                .map_err(|_| format!("arm {arm_id} claims immutable_verified without immutable_artifact_sha256"))?;
            let _ = sha256(immutable, "immutable_artifact_sha256")?;
        }
        let budget = field(arm, "inference_budget", "arm")?;
        exact_keys(
            budget,
            &["max_wall_time_ms", "max_calls", "max_retries", "cost_accounting"],
            &["max_output_tokens", "max_cost_microusd"],
            "inference_budget",
        )?;
        for key in ["max_wall_time_ms", "max_calls"] {
            if integer(field(budget, key, "inference_budget")?, key)? == 0 {
                return Err(format!("{key} must be positive"));
            }
        }
        let _ = integer(field(budget, "max_retries", "inference_budget")?, "max_retries")?;
        let _ = string(field(budget, "cost_accounting", "inference_budget")?, "cost_accounting")?;
    }

    let cases = array(field(run, "cases", "run")?, "cases")?;
    if cases.is_empty() {
        return Err("at least one matched case is required".into());
    }
    let mut case_ids = BTreeSet::new();
    let mut case_outputs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for case in cases {
        exact_keys(
            case,
            &[
                "case_id",
                "shared_input_sha256",
                "task_instance_sha256",
                "prompt_or_instruction_sha256",
                "seed",
                "attempts",
            ],
            &["source_artifact_sha256"],
            "case",
        )?;
        let case_id = unique_id(field(case, "case_id", "case")?, "case_id")?;
        if !case_ids.insert(case_id.clone()) {
            return Err(format!("duplicate case_id: {case_id}"));
        }
        let _ = sha256(field(case, "shared_input_sha256", "case")?, "shared_input_sha256")?;
        let _ = sha256(field(case, "task_instance_sha256", "case")?, "task_instance_sha256")?;
        let _ = sha256(
            field(case, "prompt_or_instruction_sha256", "case")?,
            "prompt_or_instruction_sha256",
        )?;
        let _ = integer(field(case, "seed", "case")?, "seed")?;
        if let Some(source_sha) = case.get("source_artifact_sha256") {
            let _ = sha256(source_sha, "source_artifact_sha256")?;
        }
        let attempts = array(field(case, "attempts", "case")?, "attempts")?;
        if status == "frozen_plan" && !attempts.is_empty() {
            return Err(format!("frozen_plan case {case_id} must not contain execution attempts"));
        }

        let mut attempt_keys = BTreeSet::new();
        let mut arms_attempted = BTreeSet::new();
        let mut outputs = BTreeSet::new();
        for attempt in attempts {
            exact_keys(
                attempt,
                &["arm_id", "attempt_index", "status", "started_at", "elapsed_ms"],
                &[
                    "output_sha256",
                    "output_bytes",
                    "reported_cost_microusd",
                    "provider_request_id",
                    "error_code",
                    "error_detail_sha256",
                ],
                "attempt",
            )?;
            let arm_id = string(field(attempt, "arm_id", "attempt")?, "arm_id")?.to_owned();
            if !arm_ids.contains(&arm_id) {
                return Err(format!("case {case_id} attempt references unknown arm {arm_id}"));
            }
            let attempt_index = integer(field(attempt, "attempt_index", "attempt")?, "attempt_index")?;
            if attempt_index == 0 || attempt_index > max_attempts {
                return Err(format!("case {case_id} attempt index exceeds frozen retry policy"));
            }
            if !attempt_keys.insert((arm_id.clone(), attempt_index)) {
                return Err(format!("case {case_id} has duplicate attempt {arm_id}#{attempt_index}"));
            }
            arms_attempted.insert(arm_id.clone());
            let attempt_status = string(field(attempt, "status", "attempt")?, "status")?;
            let _ = string(field(attempt, "started_at", "attempt")?, "started_at")?;
            let _ = integer(field(attempt, "elapsed_ms", "attempt")?, "elapsed_ms")?;
            if attempt_status == "completed" {
                let output = sha256(field(attempt, "output_sha256", "attempt")?, "output_sha256")?;
                let _ = integer(field(attempt, "output_bytes", "attempt")?, "output_bytes")?;
                outputs.insert(output);
            } else {
                if ![
                    "timeout",
                    "error",
                    "abstained",
                    "unavailable",
                    "invalid_output",
                ]
                .contains(&attempt_status)
                {
                    return Err(format!("case {case_id} has unsupported attempt status {attempt_status}"));
                }
                let _ = string(field(attempt, "error_code", "attempt")?, "error_code")?;
                if let Some(output) = attempt.get("output_sha256") {
                    let _ = sha256(output, "output_sha256")?;
                }
                if let Some(output_bytes) = attempt.get("output_bytes") {
                    let _ = integer(output_bytes, "output_bytes")?;
                }
            }
            if let Some(cost) = attempt.get("reported_cost_microusd") {
                let _ = integer(cost, "reported_cost_microusd")?;
            }
            if let Some(error_sha) = attempt.get("error_detail_sha256") {
                let _ = sha256(error_sha, "error_detail_sha256")?;
            }
        }
        if status != "frozen_plan" {
            for arm_id in &arm_ids {
                if !arms_attempted.contains(arm_id) {
                    return Err(format!("executed case {case_id} is missing an attempt for arm {arm_id}"));
                }
            }
        }
        case_outputs.insert(case_id, outputs);
    }

    let judgments = array(field(run, "judgments", "run")?, "judgments")?;
    let mut judgment_ids = BTreeSet::new();
    for judgment in judgments {
        exact_keys(
            judgment,
            &[
                "judgment_id",
                "case_id",
                "left_output_sha256",
                "right_output_sha256",
                "display_order_seed",
                "judge_pseudonym_sha256",
                "choice",
                "judgment_protocol_sha256",
            ],
            &[],
            "judgment",
        )?;
        let judgment_id = string(field(judgment, "judgment_id", "judgment")?, "judgment_id")?.to_owned();
        if !judgment_ids.insert(judgment_id.clone()) {
            return Err(format!("duplicate judgment_id: {judgment_id}"));
        }
        let case_id = string(field(judgment, "case_id", "judgment")?, "case_id")?;
        let outputs = case_outputs
            .get(case_id)
            .ok_or_else(|| format!("judgment {judgment_id} references unknown case {case_id}"))?;
        let left = sha256(field(judgment, "left_output_sha256", "judgment")?, "left_output_sha256")?;
        let right = sha256(field(judgment, "right_output_sha256", "judgment")?, "right_output_sha256")?;
        if left == right {
            return Err(format!("judgment {judgment_id} compares an output with itself"));
        }
        if !outputs.contains(&left) || !outputs.contains(&right) {
            return Err(format!("judgment {judgment_id} references output bytes not completed in case {case_id}"));
        }
        let _ = integer(field(judgment, "display_order_seed", "judgment")?, "display_order_seed")?;
        let _ = sha256(field(judgment, "judge_pseudonym_sha256", "judgment")?, "judge_pseudonym_sha256")?;
        let choice = string(field(judgment, "choice", "judgment")?, "choice")?;
        if !["left", "right", "tie", "neither", "invalid", "skip"].contains(&choice) {
            return Err(format!("judgment {judgment_id} has unsupported choice {choice}"));
        }
        let _ = sha256(
            field(judgment, "judgment_protocol_sha256", "judgment")?,
            "judgment_protocol_sha256",
        )?;
    }

    let metrics = array(field(run, "metrics", "run")?, "metrics")?;
    if status == "frozen_plan" && (!judgments.is_empty() || !metrics.is_empty()) {
        return Err("frozen_plan must not contain judgments or aggregate metrics".into());
    }
    if status == "analyzed" && metrics.is_empty() {
        return Err("analyzed run must contain at least one metric".into());
    }
    let mut metric_ids = BTreeSet::new();
    for metric in metrics {
        exact_keys(
            metric,
            &[
                "metric_id",
                "name",
                "value",
                "unit",
                "numerator",
                "denominator",
                "cohort_scope",
            ],
            &["lower_bound", "upper_bound", "confidence_level"],
            "metric",
        )?;
        let metric_id = string(field(metric, "metric_id", "metric")?, "metric_id")?.to_owned();
        if !metric_ids.insert(metric_id.clone()) {
            return Err(format!("duplicate metric_id: {metric_id}"));
        }
        let _ = string(field(metric, "name", "metric")?, "name")?;
        let value = field(metric, "value", "metric")?
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("metric {metric_id} value must be finite"))?;
        let _ = value;
        let _ = string(field(metric, "unit", "metric")?, "unit")?;
        let numerator = integer(field(metric, "numerator", "metric")?, "numerator")?;
        let denominator = integer(field(metric, "denominator", "metric")?, "denominator")?;
        if denominator == 0 || numerator > denominator {
            return Err(format!("metric {metric_id} has invalid numerator/denominator"));
        }
        let _ = string(field(metric, "cohort_scope", "metric")?, "cohort_scope")?;
        if let (Some(lower), Some(upper)) = (metric.get("lower_bound"), metric.get("upper_bound")) {
            let lower = lower.as_f64().filter(|value| value.is_finite()).ok_or_else(|| {
                format!("metric {metric_id} lower_bound must be finite")
            })?;
            let upper = upper.as_f64().filter(|value| value.is_finite()).ok_or_else(|| {
                format!("metric {metric_id} upper_bound must be finite")
            })?;
            if lower > upper {
                return Err(format!("metric {metric_id} lower_bound exceeds upper_bound"));
            }
        } else if metric.get("lower_bound").is_some() != metric.get("upper_bound").is_some() {
            return Err(format!("metric {metric_id} must supply both interval bounds or neither"));
        }
        if let Some(level) = metric.get("confidence_level") {
            let level = level.as_f64().filter(|value| value.is_finite()).ok_or_else(|| {
                format!("metric {metric_id} confidence_level must be finite")
            })?;
            if !(0.0..1.0).contains(&level) {
                return Err(format!("metric {metric_id} confidence_level must be in (0,1)"));
            }
        }
    }

    let claims = array(field(run, "claim_ceiling", "run")?, "claim_ceiling")?;
    if claims.is_empty() || claims.iter().any(|claim| claim.as_str().is_none_or(|text| text.trim().is_empty())) {
        return Err("claim_ceiling must contain non-empty statements".into());
    }
    Ok(())
}

fn default_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/mel")
        .join(DEFAULT_FILE)
}

fn run() -> Result<(), String> {
    let mut args = env::args_os().skip(1);
    let path = args.next().map(PathBuf::from).unwrap_or_else(default_path);
    if args.next().is_some() {
        return Err("usage: mel-matched-comparison-validator [run.json]".into());
    }
    let bytes = fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid JSON at {}: {error}", path.display()))?;
    validate_run(&value).map_err(|errors| errors.join("\n"))?;
    let run_id = string(field(&value, "run_id", "run")?, "run_id")?;
    let run_status = string(field(&value, "run_status", "run")?, "run_status")?;
    let cases = array(field(&value, "cases", "run")?, "cases")?;
    let arms = array(field(&value, "arms", "run")?, "arms")?;
    let attempts = cases
        .iter()
        .filter_map(|case| case.get("attempts").and_then(Value::as_array))
        .map(Vec::len)
        .sum::<usize>();
    println!(
        "valid: run_id={run_id} status={run_status} arms={} cases={} attempts={attempts}",
        arms.len(),
        cases.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mel-matched-comparison-validator: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn attempt(arm: &str, output: &str) -> Value {
        serde_json::json!({
            "arm_id": arm,
            "attempt_index": 1,
            "status": "completed",
            "started_at": "2026-11-21T10:00:00Z",
            "elapsed_ms": 1200,
            "output_sha256": output,
            "output_bytes": 4096
        })
    }

    fn fixture() -> Value {
        let arm = |arm_id: &str| {
            serde_json::json!({
                "arm_id": arm_id,
                "system_name": arm_id,
                "system_version": "1",
                "adapter_version": "1",
                "modality": "audio",
                "execution_surface": "local",
                "model_identity_state": "immutable_verified",
                "identity_evidence_sha256": digest('1'),
                "immutable_artifact_sha256": digest('2'),
                "control_surface_sha256": digest('3'),
                "inference_budget": {
                    "max_wall_time_ms": 10000,
                    "max_calls": 1,
                    "max_retries": 0,
                    "cost_accounting": "measured_local"
                }
            })
        };
        let output_a = digest('a');
        let output_b = digest('b');
        serde_json::json!({
            "schema_version": SCHEMA_V1,
            "run_id": "fixture-run-v1",
            "run_status": "analyzed",
            "frozen_at": "2026-11-21T09:00:00Z",
            "executed_at": "2026-11-21T10:00:00Z",
            "analyzed_at": "2026-11-21T11:00:00Z",
            "benchmark": {
                "benchmark_id": "matched-audio-preference",
                "benchmark_version": "1",
                "task_family": "audio_preference",
                "task_definition_sha256": digest('4'),
                "scorer_sha256": digest('5'),
                "comparability_group": "same-prompts-same-audio-settings"
            },
            "dataset": {
                "dataset_id": "rights-cleared-fixture",
                "dataset_version": "1",
                "split_manifest_sha256": digest('6'),
                "rights_clearance_receipt_sha256": digest('7'),
                "permitted_uses": ["evaluation"]
            },
            "environment": {
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
                "rust_toolchain": "1.96.0",
                "host_os": "linux",
                "host_arch": "x86_64",
                "runner_version": "1",
                "environment_sha256": digest('8')
            },
            "retry_policy": {
                "max_attempts_per_arm_case": 1,
                "retry_on": [],
                "failed_attempts_consume_budget": true
            },
            "arms": [arm("system-a"), arm("system-b")],
            "cases": [{
                "case_id": "case-001",
                "shared_input_sha256": digest('9'),
                "task_instance_sha256": digest('c'),
                "prompt_or_instruction_sha256": digest('d'),
                "seed": 42,
                "attempts": [attempt("system-a", &output_a), attempt("system-b", &output_b)]
            }],
            "judgments": [{
                "judgment_id": "judgment-001",
                "case_id": "case-001",
                "left_output_sha256": output_a,
                "right_output_sha256": output_b,
                "display_order_seed": 314,
                "judge_pseudonym_sha256": digest('e'),
                "choice": "left",
                "judgment_protocol_sha256": digest('f')
            }],
            "metrics": [{
                "metric_id": "preference-rate",
                "name": "left-choice-share",
                "value": 1.0,
                "unit": "proportion",
                "numerator": 1,
                "denominator": 1,
                "lower_bound": 1.0,
                "upper_bound": 1.0,
                "confidence_level": 0.95,
                "cohort_scope": "fixture-only"
            }],
            "claim_ceiling": ["synthetic validator fixture only"]
        })
    }

    #[test]
    fn valid_matched_run_passes() {
        validate_run(&fixture()).expect("valid fixture");
    }

    #[test]
    fn executed_case_missing_an_arm_attempt_fails() {
        let mut run = fixture();
        run["cases"][0]["attempts"].as_array_mut().unwrap().pop();
        assert!(validate_run(&run).is_err());
    }

    #[test]
    fn timeout_without_error_code_fails_instead_of_disappearing() {
        let mut run = fixture();
        run["cases"][0]["attempts"][0] = serde_json::json!({
            "arm_id": "system-a",
            "attempt_index": 1,
            "status": "timeout",
            "started_at": "2026-11-21T10:00:00Z",
            "elapsed_ms": 10000
        });
        assert!(validate_run(&run).is_err());
    }

    #[test]
    fn judgment_must_reference_an_observed_completed_output() {
        let mut run = fixture();
        run["judgments"][0]["left_output_sha256"] = Value::String(digest('0'));
        assert!(validate_run(&run).is_err());
    }

    #[test]
    fn identity_claim_requires_immutable_artifact_digest() {
        let mut run = fixture();
        run["arms"][0].as_object_mut().unwrap().remove("immutable_artifact_sha256");
        assert!(validate_run(&run).is_err());
    }

    #[test]
    fn retry_attempt_identity_must_be_unique_and_bounded() {
        let mut run = fixture();
        let duplicate = run["cases"][0]["attempts"][0].clone();
        run["cases"][0]["attempts"].as_array_mut().unwrap().push(duplicate);
        assert!(validate_run(&run).is_err());
    }

    #[test]
    fn frozen_plan_cannot_publish_results_or_attempts() {
        let mut run = fixture();
        run["run_status"] = Value::String("frozen_plan".into());
        run.as_object_mut().unwrap().remove("executed_at");
        run.as_object_mut().unwrap().remove("analyzed_at");
        assert!(validate_run(&run).is_err());
    }
}
