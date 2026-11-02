#![forbid(unsafe_code)]

//! Offline structural-accounting and integrity validator for benchmark receipts.
//!
//! This tool checks the receipt object against core schema/accounting
//! invariants, recomputes the RFC 8785 JCS payload digest with the top-level
//! integrity field omitted, and reports signature verification as unavailable.
//! It never authenticates a runner or grader and never establishes correctness.

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

const MAX_RECEIPT_BYTES: usize = 16 * 1024 * 1024;
const TERMINAL: [&str; 8] = [
    "passed", "failed", "refused", "abstained",
    "unsupported", "timed_out", "tool_failed", "ungraded",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidationError(String);

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for ValidationError {}

fn error(message: impl Into<String>) -> ValidationError {
    ValidationError(message.into())
}

fn required<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value, ValidationError> {
    value.get(key).ok_or_else(|| error(format!("{context}.{key} is required")))
}

fn string<'a>(value: &'a Value, context: &str) -> Result<&'a str, ValidationError> {
    value.as_str().ok_or_else(|| error(format!("{context} must be a string")))
}

fn nonempty_string<'a>(value: &'a Value, context: &str) -> Result<&'a str, ValidationError> {
    let value = string(value, context)?;
    if value.is_empty() {
        return Err(error(format!("{context} must not be empty")));
    }
    Ok(value)
}

fn require_keys(
    value: &Value,
    required: &[&str],
    optional: &[&str],
    context: &str,
) -> Result<(), ValidationError> {
    let object = value.as_object().ok_or_else(|| error(format!("{context} must be an object")))?;
    for key in required {
        if !object.contains_key(*key) {
            return Err(error(format!("{context}.{key} is required")));
        }
    }
    for key in object.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return Err(error(format!("{context} contains unknown field '{key}'")));
        }
    }
    Ok(())
}

fn enum_string<'a>(
    value: &'a Value,
    choices: &[&str],
    context: &str,
) -> Result<&'a str, ValidationError> {
    let value = string(value, context)?;
    if !choices.contains(&value) {
        return Err(error(format!("{context} has unsupported value '{value}'")));
    }
    Ok(value)
}

fn validate_measurement(value: &Value, context: &str) -> Result<(), ValidationError> {
    let status = enum_string(
        required(value, "status", context)?,
        &["measured", "unavailable", "not_applicable"],
        &format!("{context}.status"),
    )?;
    if status == "measured" {
        require_keys(value, &["status", "value", "unit"], &[], context)?;
        let measurement = required(value, "value", context)?.as_f64()
            .ok_or_else(|| error(format!("{context}.value must be numeric")))?;
        if !measurement.is_finite() || measurement < 0.0 {
            return Err(error(format!("{context}.value must be finite and non-negative")));
        }
        nonempty_string(required(value, "unit", context)?, &format!("{context}.unit"))?;
    } else {
        require_keys(value, &["status", "reason"], &[], context)?;
        nonempty_string(required(value, "reason", context)?, &format!("{context}.reason"))?;
    }
    Ok(())
}

fn validate_budget_cap(value: &Value, context: &str) -> Result<(), ValidationError> {
    let status = enum_string(
        required(value, "status", context)?,
        &["bounded", "unbounded", "not_applicable"],
        &format!("{context}.status"),
    )?;
    if status == "bounded" {
        require_keys(value, &["status", "maximum", "unit"], &[], context)?;
        let maximum = required(value, "maximum", context)?.as_f64()
            .ok_or_else(|| error(format!("{context}.maximum must be numeric")))?;
        if !maximum.is_finite() || maximum < 0.0 {
            return Err(error(format!("{context}.maximum must be finite and non-negative")));
        }
        nonempty_string(required(value, "unit", context)?, &format!("{context}.unit"))?;
    } else {
        require_keys(value, &["status", "reason"], &[], context)?;
        nonempty_string(required(value, "reason", context)?, &format!("{context}.reason"))?;
    }
    Ok(())
}

fn validate_structure(receipt: &Value) -> Result<(), ValidationError> {
    require_exact_keys(
        receipt,
        &[
            "schema", "run_id", "created_at", "run_kind", "benchmark", "system",
            "environment", "budget_caps", "actual_resources", "task_accounting",
            "task_outcomes", "excluded_tasks", "official_score", "derived_metrics",
            "comparison", "provenance", "integrity", "claim_ceiling",
        ],
        "receipt",
    )?;
    nonempty_string(required(receipt, "run_id", "receipt")?, "run_id")?;
    nonempty_string(required(receipt, "created_at", "receipt")?, "created_at")?;
    enum_string(
        required(receipt, "run_kind", "receipt")?,
        &["synthetic_fixture", "internal_holdout", "public_benchmark", "private_challenge", "production_evaluation"],
        "run_kind",
    )?;

    let benchmark = required(receipt, "benchmark", "receipt")?;
    require_keys(
        benchmark,
        &[
            "benchmark_id", "version", "revision", "access_class",
            "dataset_manifest_sha256", "eligible_task_set_sha256", "split_id",
            "scoring_protocol_id", "scoring_protocol_version",
        ],
        &["licence_or_access_terms", "official_documentation_uri"],
        "benchmark",
    )?;
    for key in ["benchmark_id", "version", "split_id", "scoring_protocol_id", "scoring_protocol_version"] {
        nonempty_string(required(benchmark, key, "benchmark")?, &format!("benchmark.{key}"))?;
    }
    enum_string(
        required(benchmark, "access_class", "benchmark")?,
        &["internal", "public", "authorized_private", "restricted", "synthetic"],
        "benchmark.access_class",
    )?;
    for key in ["dataset_manifest_sha256", "eligible_task_set_sha256"] {
        valid_sha256(required(benchmark, key, "benchmark")?, &format!("benchmark.{key}"))?;
    }

    let system = required(receipt, "system", "receipt")?;
    require_keys(
        system,
        &[
            "arm_id", "arm_kind", "system_id", "system_version", "source_revision",
            "configuration_sha256", "prompt_or_policy_sha256", "model", "tools",
        ],
        &["adapter_revision"],
        "system",
    )?;
    for key in ["arm_id", "system_id", "system_version"] {
        nonempty_string(required(system, key, "system")?, &format!("system.{key}"))?;
    }
    enum_string(
        required(system, "arm_kind", "system")?,
        &["rea_only", "symthaea_only", "rea_plus_symthaea", "specialist_tool", "specialist_stack", "human_expert", "other"],
        "system.arm_kind",
    )?;
    for key in ["configuration_sha256", "prompt_or_policy_sha256"] {
        valid_sha256(required(system, key, "system")?, &format!("system.{key}"))?;
    }
    let model = required(system, "model", "system")?;
    if !model.is_null() {
        require_exact_keys(model, &["provider", "model_id", "model_version", "configuration_sha256"], "system.model")?;
        for key in ["provider", "model_id", "model_version"] {
            nonempty_string(required(model, key, "system.model")?, &format!("system.model.{key}"))?;
        }
        valid_sha256(required(model, "configuration_sha256", "system.model")?, "system.model.configuration_sha256")?;
    }
    let tools = required(system, "tools", "system")?.as_array()
        .ok_or_else(|| error("system.tools must be an array"))?;
    let mut tool_ids = HashSet::new();
    for (index, tool) in tools.iter().enumerate() {
        let context = format!("system.tools[{index}]");
        require_exact_keys(tool, &["tool_id", "version", "executable_sha256", "configuration_sha256"], &context)?;
        let id = nonempty_string(required(tool, "tool_id", &context)?, &format!("{context}.tool_id"))?;
        if !tool_ids.insert(id.to_owned()) {
            return Err(error(format!("duplicate tool ID in system catalog: {id}")));
        }
        nonempty_string(required(tool, "version", &context)?, &format!("{context}.version"))?;
        for key in ["executable_sha256", "configuration_sha256"] {
            valid_sha256(required(tool, key, &context)?, &format!("{context}.{key}"))?;
        }
    }

    let environment = required(receipt, "environment", "receipt")?;
    require_exact_keys(
        environment,
        &["host_os", "host_architecture", "runtime", "container_image_sha256", "network_policy", "instrumentation", "target_cohort", "hardware"],
        "environment",
    )?;
    for key in ["host_os", "host_architecture", "runtime", "target_cohort"] {
        nonempty_string(required(environment, key, "environment")?, &format!("environment.{key}"))?;
    }
    enum_string(
        required(environment, "network_policy", "environment")?,
        &["disabled", "allowlisted", "benchmark_defined", "unrestricted", "unknown"],
        "environment.network_policy",
    )?;
    let instrumentation = required(environment, "instrumentation", "environment")?
        .as_array().ok_or_else(|| error("environment.instrumentation must be an array"))?;
    for (index, item) in instrumentation.iter().enumerate() {
        nonempty_string(item, &format!("environment.instrumentation[{index}]"))?;
    }
    let container_digest = required(environment, "container_image_sha256", "environment")?;
    if !container_digest.is_null() {
        valid_sha256(container_digest, "environment.container_image_sha256")?;
    }
    let hardware = required(environment, "hardware", "environment")?;
    if !hardware.is_null() {
        require_exact_keys(hardware, &["cpu", "gpu", "memory_bytes", "device_class"], "environment.hardware")?;
        if let Some(memory) = hardware.get("memory_bytes").filter(|v| !v.is_null()) {
            count(memory, "environment.hardware.memory_bytes")?;
        }
    }

    const RESOURCE_KEYS: [&str; 10] = [
        "wall_time_ms", "cpu_time_ms", "gpu_time_ms", "peak_rss_bytes",
        "model_calls", "input_tokens", "output_tokens", "tool_calls",
        "cost_microusd", "energy_microjoules",
    ];
    let budget_caps = required(receipt, "budget_caps", "receipt")?;
    require_exact_keys(budget_caps, &RESOURCE_KEYS, "budget_caps")?;
    for key in RESOURCE_KEYS {
        validate_budget_cap(required(budget_caps, key, "budget_caps")?, &format!("budget_caps.{key}"))?;
    }
    let actual_resources = required(receipt, "actual_resources", "receipt")?;
    require_exact_keys(actual_resources, &RESOURCE_KEYS, "actual_resources")?;
    for key in RESOURCE_KEYS {
        validate_measurement(required(actual_resources, key, "actual_resources")?, &format!("actual_resources.{key}"))?;
    }

    let accounting = required(receipt, "task_accounting", "receipt")?;
    require_exact_keys(
        accounting,
        &[
            "universe", "eligible", "excluded", "attempted", "not_attempted",
            "in_progress", "completed", "graded", "terminal_dispositions",
        ],
        "task_accounting",
    )?;
    let terminal = required(accounting, "terminal_dispositions", "task_accounting")?;
    require_exact_keys(
        terminal,
        &["passed", "failed", "refused", "abstained", "unsupported", "timed_out", "tool_failed", "ungraded"],
        "terminal_dispositions",
    )?;
    for key in [
        "universe", "eligible", "excluded", "attempted", "not_attempted",
        "in_progress", "completed", "graded",
    ] {
        count_at(accounting, key, "task_accounting")?;
    }
    for key in TERMINAL {
        count_at(terminal, key, "terminal_dispositions")?;
    }

    let outcomes = required(receipt, "task_outcomes", "receipt")?.as_array()
        .ok_or_else(|| error("task_outcomes must be an array"))?;
    for (index, row) in outcomes.iter().enumerate() {
        let context = format!("task_outcomes[{index}]");
        require_exact_keys(
            row,
            &["task_id", "input_sha256", "disposition", "attempted", "graded", "grade", "reason_code", "grader_result_sha256", "evidence_ids", "resources"],
            &context,
        )?;
        nonempty_string(required(row, "task_id", &context)?, &format!("{context}.task_id"))?;
        valid_sha256(required(row, "input_sha256", &context)?, &format!("{context}.input_sha256"))?;
        enum_string(
            required(row, "disposition", &context)?,
            &["passed", "failed", "refused", "abstained", "unsupported", "timed_out", "tool_failed", "ungraded", "not_attempted", "in_progress"],
            &format!("{context}.disposition"),
        )?;
        for key in ["attempted", "graded"] {
            if required(row, key, &context)?.as_bool().is_none() {
                return Err(error(format!("{context}.{key} must be boolean")));
            }
        }
        let grade = required(row, "grade", &context)?;
        if !grade.is_null() && grade.as_f64().is_none() {
            return Err(error(format!("{context}.grade must be numeric or null")));
        }
        let reason = required(row, "reason_code", &context)?;
        if !reason.is_null() && reason.as_str().is_none() {
            return Err(error(format!("{context}.reason_code must be string or null")));
        }
        let grader = required(row, "grader_result_sha256", &context)?;
        if !grader.is_null() {
            valid_sha256(grader, &format!("{context}.grader_result_sha256"))?;
        }
        let evidence_ids = required(row, "evidence_ids", &context)?.as_array()
            .ok_or_else(|| error(format!("{context}.evidence_ids must be an array")))?;
        let mut evidence_set = HashSet::new();
        for evidence_id in evidence_ids {
            let id = string(evidence_id, &format!("{context}.evidence_ids[]"))?;
            if id.len() != 67 || !id.starts_with("ev_") || !id[3..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return Err(error(format!("{context} has invalid Evidence ID")));
            }
            if !evidence_set.insert(id.to_owned()) {
                return Err(error(format!("{context} has duplicate Evidence ID")));
            }
        }
        let resources = required(row, "resources", &context)?;
        require_exact_keys(resources, &["wall_time_ms", "model_calls", "input_tokens", "output_tokens", "tool_calls", "cost_microusd"], &format!("{context}.resources"))?;
        for key in ["wall_time_ms", "model_calls", "input_tokens", "output_tokens", "tool_calls", "cost_microusd"] {
            validate_measurement(required(resources, key, &format!("{context}.resources"))?, &format!("{context}.resources.{key}"))?;
        }
    }

    let excluded = required(receipt, "excluded_tasks", "receipt")?.as_array()
        .ok_or_else(|| error("excluded_tasks must be an array"))?;
    for (index, row) in excluded.iter().enumerate() {
        let context = format!("excluded_tasks[{index}]");
        require_exact_keys(row, &["task_id", "input_sha256", "reason_code", "predeclared", "approval_reference"], &context)?;
        nonempty_string(required(row, "task_id", &context)?, &format!("{context}.task_id"))?;
        valid_sha256(required(row, "input_sha256", &context)?, &format!("{context}.input_sha256"))?;
        nonempty_string(required(row, "reason_code", &context)?, &format!("{context}.reason_code"))?;
        if required(row, "predeclared", &context)?.as_bool().is_none() {
            return Err(error(format!("{context}.predeclared must be boolean")));
        }
        let approval = required(row, "approval_reference", &context)?;
        if !approval.is_null() && approval.as_str().is_none() {
            return Err(error(format!("{context}.approval_reference must be string or null")));
        }
    }

    let official = required(receipt, "official_score", "receipt")?;
    let status = enum_string(
        required(official, "status", "official_score")?,
        &["not_applicable", "reported_by_official_grader", "unavailable"],
        "official_score.status",
    )?;
    match status {
        "reported_by_official_grader" => {
            require_exact_keys(official, &["status", "metric_id", "formula_version", "value", "unit", "numerator", "denominator", "grader_result_sha256"], "official_score")?;
            nonempty_string(required(official, "metric_id", "official_score")?, "official_score.metric_id")?;
            nonempty_string(required(official, "formula_version", "official_score")?, "official_score.formula_version")?;
            if required(official, "value", "official_score")?.as_f64().is_none() {
                return Err(error("official_score.value must be numeric"));
            }
            nonempty_string(required(official, "unit", "official_score")?, "official_score.unit")?;
            for key in ["numerator", "denominator"] {
                let value = required(official, key, "official_score")?;
                if !value.is_null() && value.as_f64().is_none() {
                    return Err(error(format!("official_score.{key} must be numeric or null")));
                }
            }
            valid_sha256(required(official, "grader_result_sha256", "official_score")?, "official_score.grader_result_sha256")?;
        }
        "not_applicable" | "unavailable" => {
            require_exact_keys(official, &["status", "reason"], "official_score")?;
            nonempty_string(required(official, "reason", "official_score")?, "official_score.reason")?;
        }
        _ => unreachable!(),
    }

    let metrics = required(receipt, "derived_metrics", "receipt")?.as_array()
        .ok_or_else(|| error("derived_metrics must be an array"))?;
    for (index, metric) in metrics.iter().enumerate() {
        let context = format!("derived_metrics[{index}]");
        require_exact_keys(metric, &["metric_id", "classification", "formula", "value", "numerator", "denominator", "zero_denominator_reason"], &context)?;
        nonempty_string(required(metric, "metric_id", &context)?, &format!("{context}.metric_id"))?;
        if required(metric, "classification", &context)? != "derived_non_official" {
            return Err(error(format!("{context}.classification must be derived_non_official")));
        }
        nonempty_string(required(metric, "formula", &context)?, &format!("{context}.formula"))?;
        for key in ["value", "numerator", "denominator"] {
            let value = required(metric, key, &context)?;
            if !value.is_null() && value.as_f64().is_none() {
                return Err(error(format!("{context}.{key} must be numeric or null")));
            }
        }
        let reason = required(metric, "zero_denominator_reason", &context)?;
        if !reason.is_null() && reason.as_str().is_none() {
            return Err(error(format!("{context}.zero_denominator_reason must be string or null")));
        }
    }

    let comparison = required(receipt, "comparison", "receipt")?;
    require_exact_keys(comparison, &["comparison_group_id", "matched", "comparable", "eligible_task_set_sha256", "budget_protocol_sha256", "matched_arm_ids", "non_comparability_reasons"], "comparison")?;
    nonempty_string(required(comparison, "comparison_group_id", "comparison")?, "comparison.comparison_group_id")?;
    for key in ["matched", "comparable"] {
        if required(comparison, key, "comparison")?.as_bool().is_none() {
            return Err(error(format!("comparison.{key} must be boolean")));
        }
    }
    for key in ["eligible_task_set_sha256", "budget_protocol_sha256"] {
        valid_sha256(required(comparison, key, "comparison")?, &format!("comparison.{key}"))?;
    }
    let arms = required(comparison, "matched_arm_ids", "comparison")?.as_array()
        .ok_or_else(|| error("comparison.matched_arm_ids must be an array"))?;
    for (index, arm) in arms.iter().enumerate() {
        nonempty_string(arm, &format!("comparison.matched_arm_ids[{index}]"))?;
    }
    let reasons = required(comparison, "non_comparability_reasons", "comparison")?.as_array()
        .ok_or_else(|| error("comparison.non_comparability_reasons must be an array"))?;
    for (index, reason) in reasons.iter().enumerate() {
        nonempty_string(reason, &format!("comparison.non_comparability_reasons[{index}]"))?;
    }

    let provenance = required(receipt, "provenance", "receipt")?;
    require_keys(
        provenance,
        &["runner_id", "runner_revision", "grader_id", "grader_version", "grader_sha256", "result_artifact_sha256", "reproduction_recipe_sha256"],
        &["preregistration_uri", "notes"],
        "provenance",
    )?;
    for key in ["runner_id", "grader_id", "grader_version"] {
        nonempty_string(required(provenance, key, "provenance")?, &format!("provenance.{key}"))?;
    }
    for key in ["grader_sha256", "result_artifact_sha256", "reproduction_recipe_sha256"] {
        valid_sha256(required(provenance, key, "provenance")?, &format!("provenance.{key}"))?;
    }
    if let Some(notes) = provenance.get("notes") {
        let notes = notes.as_array().ok_or_else(|| error("provenance.notes must be an array"))?;
        for (index, note) in notes.iter().enumerate() {
            if note.as_str().is_none() {
                return Err(error(format!("provenance.notes[{index}] must be a string")));
            }
        }
    }

    let integrity = required(receipt, "integrity", "receipt")?;
    require_keys(integrity, &["canonicalization", "payload_sha256_excludes_integrity"], &["signature"], "integrity")?;
    if required(integrity, "canonicalization", "integrity")? != "RFC8785-JCS" {
        return Err(error("unsupported receipt canonicalization"));
    }
    valid_sha256(required(integrity, "payload_sha256_excludes_integrity", "integrity")?, "integrity.payload_sha256_excludes_integrity")?;
    if let Some(signature) = integrity.get("signature").filter(|v| !v.is_null()) {
        require_exact_keys(signature, &["algorithm", "key_id", "signature_base64"], "integrity.signature")?;
        nonempty_string(required(signature, "algorithm", "integrity.signature")?, "integrity.signature.algorithm")?;
        nonempty_string(required(signature, "key_id", "integrity.signature")?, "integrity.signature.key_id")?;
        nonempty_string(required(signature, "signature_base64", "integrity.signature")?, "integrity.signature.signature_base64")?;
    }

    let ceiling = required(receipt, "claim_ceiling", "receipt")?;
    require_exact_keys(ceiling, &["synthetic_fixture", "official_score_is_official", "grader_independence_verified", "system_superiority_established"], "claim_ceiling")?;
    for key in ["synthetic_fixture", "official_score_is_official", "grader_independence_verified", "system_superiority_established"] {
        if required(ceiling, key, "claim_ceiling")?.as_bool().is_none() {
            return Err(error(format!("claim_ceiling.{key} must be boolean")));
        }
    }
    Ok(())
}

fn count(value: &Value, context: &str) -> Result<u64, ValidationError> {
    value.as_u64().ok_or_else(|| error(format!("{context} must be a non-negative integer")))
}

fn count_at(value: &Value, key: &str, context: &str) -> Result<u64, ValidationError> {
    count(required(value, key, context)?, &format!("{context}.{key}"))
}

fn require_exact_keys(value: &Value, expected: &[&str], context: &str) -> Result<(), ValidationError> {
    let object = value.as_object().ok_or_else(|| error(format!("{context} must be an object")))?;
    for key in expected {
        if !object.contains_key(*key) {
            return Err(error(format!("{context}.{key} is required")));
        }
    }
    for key in object.keys() {
        if !expected.contains(&key.as_str()) {
            return Err(error(format!("{context} contains unknown field '{key}'")));
        }
    }
    Ok(())
}

fn checked_add(left: u64, right: u64, context: &str) -> Result<u64, ValidationError> {
    left.checked_add(right).ok_or_else(|| error(format!("{context} count overflow")))
}

fn valid_sha256(value: &Value, context: &str) -> Result<(), ValidationError> {
    let digest = string(value, context)?;
    if digest.len() != 64
        || !digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(error(format!("{context} must be 64 lowercase hexadecimal characters")));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn canonical_payload_digest(receipt: &Value) -> Result<String, ValidationError> {
    let mut payload = receipt.clone();
    let object = payload.as_object_mut().ok_or_else(|| error("benchmark receipt must be an object"))?;
    object.remove("integrity");
    let canonical = serde_json_canonicalizer::to_vec(&payload)
        .map_err(|cause| error(format!("JCS canonicalization failed: {cause}")))?;
    Ok(sha256_hex(&canonical))
}

fn validate_accounting(receipt: &Value) -> Result<Value, ValidationError> {
    require_exact_keys(
        receipt,
        &[
            "schema", "run_id", "created_at", "run_kind", "benchmark", "system",
            "environment", "budget_caps", "actual_resources", "task_accounting",
            "task_outcomes", "excluded_tasks", "official_score", "derived_metrics",
            "comparison", "provenance", "integrity", "claim_ceiling",
        ],
        "receipt",
    )?;
    let accounting = required(receipt, "task_accounting", "receipt")?;
    require_exact_keys(
        accounting,
        &[
            "universe", "eligible", "excluded", "attempted", "not_attempted",
            "in_progress", "completed", "graded", "terminal_dispositions",
        ],
        "task_accounting",
    )?;
    let universe = count_at(accounting, "universe", "task_accounting")?;
    let eligible = count_at(accounting, "eligible", "task_accounting")?;
    let excluded = count_at(accounting, "excluded", "task_accounting")?;
    let attempted = count_at(accounting, "attempted", "task_accounting")?;
    let not_attempted = count_at(accounting, "not_attempted", "task_accounting")?;
    let in_progress = count_at(accounting, "in_progress", "task_accounting")?;
    let completed = count_at(accounting, "completed", "task_accounting")?;
    let graded = count_at(accounting, "graded", "task_accounting")?;

    if universe != checked_add(eligible, excluded, "universe")? {
        return Err(error(format!("universe accounting mismatch: {universe} != {eligible} + {excluded}")));
    }
    if eligible != checked_add(attempted, not_attempted, "eligible")? {
        return Err(error(format!("eligibility accounting mismatch: {eligible} != {attempted} + {not_attempted}")));
    }
    if attempted != checked_add(completed, in_progress, "attempted")? {
        return Err(error(format!("attempt accounting mismatch: {attempted} != {completed} + {in_progress}")));
    }

    let terminal = required(accounting, "terminal_dispositions", "task_accounting")?;
    require_exact_keys(
        terminal,
        &[
            "passed", "failed", "refused", "abstained", "unsupported",
            "timed_out", "tool_failed", "ungraded",
        ],
        "terminal_dispositions",
    )?;
    let mut terminal_counts = HashMap::new();
    let mut terminal_total = 0u64;
    for disposition in TERMINAL {
        let value = count_at(terminal, disposition, "terminal_dispositions")?;
        terminal_total = terminal_total.checked_add(value).ok_or_else(|| error("terminal count overflow"))?;
        terminal_counts.insert(disposition, value);
    }
    if terminal_total != completed {
        return Err(error(format!("completed count {completed} does not equal terminal disposition total {terminal_total}")));
    }
    let passed = terminal_counts["passed"];
    let failed = terminal_counts["failed"];
    if graded != checked_add(passed, failed, "graded")? {
        return Err(error(format!("binary grading mismatch: graded {graded} != passed {passed} + failed {failed}")));
    }

    let outcomes = required(receipt, "task_outcomes", "receipt")?
        .as_array().ok_or_else(|| error("task_outcomes must be an array"))?;
    if outcomes.len() as u64 != eligible {
        return Err(error(format!("task_outcomes has {} rows but eligible count is {eligible}", outcomes.len())));
    }
    let excluded_rows = required(receipt, "excluded_tasks", "receipt")?
        .as_array().ok_or_else(|| error("excluded_tasks must be an array"))?;
    if excluded_rows.len() as u64 != excluded {
        return Err(error(format!("excluded_tasks has {} rows but excluded count is {excluded}", excluded_rows.len())));
    }

    let mut task_ids = HashSet::new();
    let mut row_counts = HashMap::<&str, u64>::new();
    let mut row_not_attempted = 0u64;
    let mut row_in_progress = 0u64;
    let mut row_graded = 0u64;

    for (index, row) in outcomes.iter().enumerate() {
        let context = format!("task_outcomes[{index}]");
        require_exact_keys(
            row,
            &[
                "task_id", "input_sha256", "disposition", "attempted", "graded",
                "grade", "reason_code", "grader_result_sha256", "evidence_ids", "resources",
            ],
            &context,
        )?;
        let task_id = nonempty_string(required(row, "task_id", &context)?, &format!("{context}.task_id"))?;
        if !task_ids.insert(task_id.to_owned()) {
            return Err(error(format!("duplicate task ID: {task_id}")));
        }
        valid_sha256(required(row, "input_sha256", &context)?, &format!("{context}.input_sha256"))?;

        let disposition = nonempty_string(required(row, "disposition", &context)?, &format!("{context}.disposition"))?;
        let attempted_flag = required(row, "attempted", &context)?.as_bool()
            .ok_or_else(|| error(format!("{context}.attempted must be boolean")))?;
        let graded_flag = required(row, "graded", &context)?.as_bool()
            .ok_or_else(|| error(format!("{context}.graded must be boolean")))?;
        let grade = required(row, "grade", &context)?;
        let grader_digest = required(row, "grader_result_sha256", &context)?;

        if disposition == "not_attempted" {
            if attempted_flag || graded_flag || !grade.is_null() || !grader_digest.is_null() {
                return Err(error(format!("{context} not_attempted row cannot be attempted or graded")));
            }
            row_not_attempted += 1;
        } else if disposition == "in_progress" {
            if !attempted_flag || graded_flag || !grade.is_null() || !grader_digest.is_null() {
                return Err(error(format!("{context} in_progress row must be attempted and ungraded")));
            }
            row_in_progress += 1;
        } else if TERMINAL.contains(&disposition) {
            if !attempted_flag {
                return Err(error(format!("{context} terminal task must be attempted")));
            }
            *row_counts.entry(disposition).or_default() += 1;
            if disposition == "passed" || disposition == "failed" {
                if !graded_flag || !grade.is_number() || grader_digest.is_null() {
                    return Err(error(format!("{context} passed/failed task must have a grade and grader digest")));
                }
                valid_sha256(grader_digest, &format!("{context}.grader_result_sha256"))?;
                row_graded += 1;
            } else if graded_flag || !grade.is_null() {
                return Err(error(format!("{context} {disposition} task cannot be binary graded")));
            }
        } else {
            return Err(error(format!("{context} has unknown disposition '{disposition}'")));
        }
    }

    for disposition in TERMINAL {
        let expected = terminal_counts[disposition];
        let observed = row_counts.get(disposition).copied().unwrap_or(0);
        if observed != expected {
            return Err(error(format!("task row count for '{disposition}' is {observed}, accounting declares {expected}")));
        }
    }
    if row_not_attempted != not_attempted {
        return Err(error(format!("not_attempted row count {row_not_attempted} does not match declared {not_attempted}")));
    }
    if row_in_progress != in_progress {
        return Err(error(format!("in_progress row count {row_in_progress} does not match declared {in_progress}")));
    }
    if row_graded != graded {
        return Err(error(format!("graded task rows {row_graded} do not match declared graded count {graded}")));
    }

    for (index, row) in excluded_rows.iter().enumerate() {
        require_exact_keys(
            row,
            &["task_id", "input_sha256", "reason_code", "predeclared", "approval_reference"],
            &format!("excluded_tasks[{index}]"),
        )?;
        let context = format!("excluded_tasks[{index}]");
        let task_id = nonempty_string(required(row, "task_id", &context)?, &format!("{context}.task_id"))?;
        if !task_ids.insert(task_id.to_owned()) {
            return Err(error(format!("duplicate task ID across eligible/excluded rows: {task_id}")));
        }
        valid_sha256(required(row, "input_sha256", &context)?, &format!("{context}.input_sha256"))?;
        nonempty_string(required(row, "reason_code", &context)?, &format!("{context}.reason_code"))?;
        if required(row, "predeclared", &context)?.as_bool() != Some(true) {
            return Err(error(format!("{context} exclusion was not predeclared")));
        }
    }

    let score_protocol = string(
        required(required(receipt, "benchmark", "receipt")?, "scoring_protocol_id", "benchmark")?,
        "benchmark.scoring_protocol_id",
    )?;
    if score_protocol == "binary-pass-fail" && graded != passed + failed {
        return Err(error("binary pass/fail grading denominator is inconsistent"));
    }

    Ok(json!({
        "universe": universe, "eligible": eligible, "excluded": excluded,
        "attempted": attempted, "not_attempted": not_attempted, "in_progress": in_progress,
        "completed": completed, "graded": graded, "passed": passed, "failed": failed,
        "refused": terminal_counts["refused"], "abstained": terminal_counts["abstained"],
        "unsupported": terminal_counts["unsupported"], "timed_out": terminal_counts["timed_out"],
        "tool_failed": terminal_counts["tool_failed"], "ungraded": terminal_counts["ungraded"]
    }))
}

fn validate_budgets(receipt: &Value) -> Result<(), ValidationError> {
    let caps = required(receipt, "budget_caps", "receipt")?
        .as_object().ok_or_else(|| error("budget_caps must be an object"))?;
    let actual = required(receipt, "actual_resources", "receipt")?
        .as_object().ok_or_else(|| error("actual_resources must be an object"))?;
    for (resource, cap) in caps {
        let cap_context = format!("budget_caps.{resource}");
        let actual_context = format!("actual_resources.{resource}");
        let status = string(required(cap, "status", &cap_context)?, &format!("{cap_context}.status"))?;
        if status == "bounded" {
            let maximum = required(cap, "maximum", &cap_context)?.as_f64()
                .ok_or_else(|| error(format!("{cap_context}.maximum must be numeric")))?;
            let actual_value = actual.get(resource)
                .ok_or_else(|| error(format!("{actual_context} is missing")))?;
            let actual_status = string(required(actual_value, "status", &actual_context)?, &format!("{actual_context}.status"))?;
            if actual_status == "measured" {
                let value = required(actual_value, "value", &actual_context)?.as_f64()
                    .ok_or_else(|| error(format!("{actual_context}.value must be numeric")))?;
                if value > maximum {
                    return Err(error(format!("budget exceeded for {resource}: measured {value}, cap {maximum}")));
                }
            }
        }
    }
    Ok(())
}

fn validate_derived_metrics(receipt: &Value, accounting: &Value) -> Result<(), ValidationError> {
    let metrics = required(receipt, "derived_metrics", "receipt")?
        .as_array().ok_or_else(|| error("derived_metrics must be an array"))?;
    let eligible = count_at(accounting, "eligible", "task_accounting")?;
    let attempted = count_at(accounting, "attempted", "task_accounting")?;
    let graded = count_at(accounting, "graded", "task_accounting")?;
    let passed = count_at(
        required(accounting, "terminal_dispositions", "task_accounting")?,
        "passed", "terminal_dispositions",
    )?;

    for (index, metric) in metrics.iter().enumerate() {
        let context = format!("derived_metrics[{index}]");
        if required(metric, "classification", &context)? != "derived_non_official" {
            return Err(error(format!("{context} must be labelled derived_non_official")));
        }
        let numerator_value = required(metric, "numerator", &context)?;
        let denominator_value = required(metric, "denominator", &context)?;
        let value = required(metric, "value", &context)?;
        let reason = required(metric, "zero_denominator_reason", &context)?;
        if denominator_value.as_f64() == Some(0.0) {
            if !value.is_null() || nonempty_string(reason, &format!("{context}.zero_denominator_reason")).is_err() {
                return Err(error(format!("{context} must use null and a reason for zero denominator")));
            }
        } else {
            if !reason.is_null() {
                return Err(error(format!("{context} nonzero denominator must not have zero-denominator reason")));
            }
            let numerator = numerator_value.as_f64()
                .ok_or_else(|| error(format!("{context}.numerator must be numeric or null")))?;
            let denominator = denominator_value.as_f64()
                .ok_or_else(|| error(format!("{context}.denominator must be numeric")))?;
            let observed = value.as_f64()
                .ok_or_else(|| error(format!("{context}.value must be numeric")))?;
            let expected = numerator / denominator;
            if !observed.is_finite() || (observed - expected).abs() > 1e-12 {
                return Err(error(format!("{context}.value does not match numerator/denominator")));
            }
        }

        let metric_id = string(required(metric, "metric_id", &context)?, &format!("{context}.metric_id"))?;
        let expected = match metric_id {
            "eligible_coverage" => Some((attempted as f64, eligible as f64)),
            "graded_coverage" => Some((graded as f64, eligible as f64)),
            "coverage_adjusted_task_yield" => Some((passed as f64, eligible as f64)),
            _ => None,
        };
        if let Some((expected_num, expected_den)) = expected {
            let given_num = numerator_value.as_f64().ok_or_else(|| error(format!("{context}.numerator must be numeric")))?;
            let given_den = denominator_value.as_f64().ok_or_else(|| error(format!("{context}.denominator must be numeric")))?;
            if given_num != expected_num || given_den != expected_den {
                return Err(error(format!("{context} numerator/denominator do not match task accounting")));
            }
        }
    }
    Ok(())
}

fn validate_official_score(receipt: &Value) -> Result<(), ValidationError> {
    let run_kind = string(required(receipt, "run_kind", "receipt")?, "run_kind")?;
    let official = required(receipt, "official_score", "receipt")?;
    let status = string(required(official, "status", "official_score")?, "official_score.status")?;
    let ceiling = required(receipt, "claim_ceiling", "receipt")?;
    let is_official = required(ceiling, "official_score_is_official", "claim_ceiling")?
        .as_bool().ok_or_else(|| error("claim_ceiling.official_score_is_official must be boolean"))?;

    if run_kind == "synthetic_fixture" {
        if status != "not_applicable" || is_official {
            return Err(error("synthetic fixture cannot declare an official benchmark score"));
        }
        if required(ceiling, "synthetic_fixture", "claim_ceiling")?.as_bool() != Some(true) {
            return Err(error("synthetic run must set claim_ceiling.synthetic_fixture=true"));
        }
    } else if required(ceiling, "synthetic_fixture", "claim_ceiling")?.as_bool() != Some(false) {
        return Err(error("non-synthetic run cannot set synthetic_fixture=true"));
    }

    if status == "reported_by_official_grader" {
        if run_kind == "synthetic_fixture" || !is_official {
            return Err(error("official grader score requires a non-synthetic run and official claim flag"));
        }
        let denominator = required(official, "denominator", "official_score")?.as_f64()
            .ok_or_else(|| error("official_score.denominator must be numeric"))?;
        if denominator <= 0.0 {
            return Err(error("official score denominator must be positive"));
        }
        let grader_digest = required(official, "grader_result_sha256", "official_score")?;
        valid_sha256(grader_digest, "official_score.grader_result_sha256")?;
        let provenance = required(receipt, "provenance", "receipt")?;
        if grader_digest != required(provenance, "grader_sha256", "provenance")? {
            return Err(error("official score grader digest does not match provenance.grader_sha256"));
        }
    } else if is_official {
        return Err(error("official score claim flag is true but no official score is present"));
    }

    if required(ceiling, "system_superiority_established", "claim_ceiling")?.as_bool() != Some(false) {
        return Err(error("a run receipt cannot by itself establish system superiority"));
    }
    Ok(())
}

fn validate_comparison(receipt: &Value) -> Result<(), ValidationError> {
    let comparison = required(receipt, "comparison", "receipt")?;
    let matched = required(comparison, "matched", "comparison")?.as_bool()
        .ok_or_else(|| error("comparison.matched must be boolean"))?;
    let comparable = required(comparison, "comparable", "comparison")?.as_bool()
        .ok_or_else(|| error("comparison.comparable must be boolean"))?;
    let arms = required(comparison, "matched_arm_ids", "comparison")?
        .as_array().ok_or_else(|| error("comparison.matched_arm_ids must be an array"))?;
    let reasons = required(comparison, "non_comparability_reasons", "comparison")?
        .as_array().ok_or_else(|| error("comparison.non_comparability_reasons must be an array"))?;

    if comparable {
        if !matched || arms.len() < 2 || !reasons.is_empty() {
            return Err(error("comparable=true requires matched=true, at least two arms, and no non-comparability reasons"));
        }
        let benchmark = required(receipt, "benchmark", "receipt")?;
        if required(comparison, "eligible_task_set_sha256", "comparison")?
            != required(benchmark, "eligible_task_set_sha256", "benchmark")?
        {
            return Err(error("comparison task-set digest differs from benchmark eligible task-set digest"));
        }
        let arm_id = required(required(receipt, "system", "receipt")?, "arm_id", "system")?;
        if !arms.iter().any(|value| value == arm_id) {
            return Err(error("current system arm is absent from comparison.matched_arm_ids"));
        }
    } else if reasons.is_empty() {
        return Err(error("non-comparable run must explain why it is not comparable"));
    }
    Ok(())
}

fn validate_receipt(value: &Value) -> Result<Value, ValidationError> {
    validate_structure(value)?;
    if required(value, "schema", "receipt")? != "luminous.benchmark-run-receipt.v1" {
        return Err(error("unsupported benchmark receipt schema"));
    }

    let accounting = validate_accounting(value)?;
    validate_budgets(value)?;
    validate_derived_metrics(value, required(value, "task_accounting", "receipt")?)?;
    validate_official_score(value)?;
    validate_comparison(value)?;

    let integrity = required(value, "integrity", "receipt")?;
    let integrity_map = integrity.as_object().ok_or_else(|| error("integrity must be an object"))?;
    for key in ["canonicalization", "payload_sha256_excludes_integrity"] {
        if !integrity_map.contains_key(key) {
            return Err(error(format!("integrity.{key} is required")));
        }
    }
    for key in integrity_map.keys() {
        if !["canonicalization", "payload_sha256_excludes_integrity", "signature"].contains(&key.as_str()) {
            return Err(error(format!("integrity contains unknown field '{key}'")));
        }
    }
    if required(integrity, "canonicalization", "integrity")? != "RFC8785-JCS" {
        return Err(error("unsupported receipt canonicalization"));
    }
    let expected_digest = string(
        required(integrity, "payload_sha256_excludes_integrity", "integrity")?,
        "integrity.payload_sha256_excludes_integrity",
    )?;
    let actual_digest = canonical_payload_digest(value)?;
    if actual_digest != expected_digest {
        return Err(error(format!("receipt payload digest mismatch: expected {expected_digest}, computed {actual_digest}")));
    }

    Ok(json!({
        "valid": true,
        "schema": "luminous.benchmark-run-receipt.v1",
        "run_id": required(value, "run_id", "receipt")?,
        "accounting": accounting,
        "official_score_status": required(required(value, "official_score", "receipt")?, "status", "official_score")?,
        "derived_metric_count": required(value, "derived_metrics", "receipt")?.as_array().map_or(0, |metrics| metrics.len()),
        "payload_sha256": actual_digest,
        "signature_present": !required(integrity, "signature", "integrity")?.is_null(),
        "signature_verified": false,
        "claim_ceiling": {
            "producer_authenticated": false,
            "grader_authenticated": false,
            "task_correctness_established": false,
            "system_superiority_established": false
        }
    }))
}

struct StrictJson(Value);

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object keys")
    }
    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where E: serde::de::Error { Ok(StrictJson(Value::Null)) }
    fn visit_none<E>(self) -> Result<Self::Value, E>
    where E: serde::de::Error { Ok(StrictJson(Value::Null)) }
    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where E: serde::de::Error { Ok(StrictJson(Value::Bool(value))) }
    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where E: serde::de::Error { Ok(StrictJson(Value::Number(Number::from(value)))) }
    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where E: serde::de::Error { Ok(StrictJson(Value::Number(Number::from(value)))) }
    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where E: serde::de::Error {
        let number = Number::from_f64(value).ok_or_else(|| E::custom("non-finite JSON number"))?;
        Ok(StrictJson(Value::Number(number)))
    }
    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where E: serde::de::Error { Ok(StrictJson(Value::String(value.to_owned()))) }
    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where E: serde::de::Error { Ok(StrictJson(Value::String(value))) }
    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where A: SeqAccess<'de> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJson>()? {
            values.push(value.0);
        }
        Ok(StrictJson(Value::Array(values)))
    }
    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where A: MapAccess<'de> {
        let mut values = Map::new();
        let mut keys = HashSet::new();
        while let Some((key, value)) = object.next_entry::<String, StrictJson>()? {
            if !keys.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!("duplicate JSON object key '{key}'")));
            }
            values.insert(key, value.0);
        }
        Ok(StrictJson(Value::Object(values)))
    }
}

fn parse_unique(bytes: &[u8]) -> Result<Value, ValidationError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = StrictJson::deserialize(&mut deserializer)
        .map_err(|cause| error(format!("invalid or duplicate-key JSON: {cause}")))?;
    deserializer.end().map_err(|cause| error(format!("trailing JSON content: {cause}")))?;
    Ok(parsed.0)
}

fn read_bounded(path: &PathBuf) -> Result<Vec<u8>, ValidationError> {
    let file = File::open(path).map_err(|cause| error(format!("cannot open receipt: {cause}")))?;
    let metadata = file.metadata().map_err(|cause| error(format!("cannot stat receipt: {cause}")))?;
    if metadata.len() > MAX_RECEIPT_BYTES as u64 {
        return Err(error(format!("receipt exceeds the {} byte limit", MAX_RECEIPT_BYTES)));
    }
    let mut bytes = Vec::new();
    file.take((MAX_RECEIPT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|cause| error(format!("cannot read receipt: {cause}")))?;
    if bytes.len() > MAX_RECEIPT_BYTES {
        return Err(error(format!("receipt grew beyond the {} byte limit", MAX_RECEIPT_BYTES)));
    }
    Ok(bytes)
}

fn verify_path(path: &PathBuf) -> Result<Value, ValidationError> {
    let bytes = read_bounded(path)?;
    let receipt = parse_unique(&bytes)?;
    validate_receipt(&receipt)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("Usage: benchmark-run-receipt-verify <receipt.json>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("Only one receipt path is accepted.");
        return ExitCode::from(2);
    }

    match verify_path(&PathBuf::from(path)) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).expect("report JSON"));
            ExitCode::SUCCESS
        }
        Err(problem) => {
            let report = json!({
                "valid": false,
                "error": problem.to_string(),
                "claim_ceiling": {
                    "producer_authenticated": false,
                    "grader_authenticated": false,
                    "task_correctness_established": false,
                    "system_superiority_established": false
                }
            });
            println!("{}", serde_json::to_string_pretty(&report).expect("error report JSON"));
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        parse_unique(include_bytes!("../examples/benchmark-run-receipt-v1.synthetic.json")).unwrap()
    }

    fn reseal(value: &mut Value) {
        let digest = canonical_payload_digest(value).unwrap();
        value["integrity"]["payload_sha256_excludes_integrity"] = Value::String(digest);
    }

    fn validate_mutation(mut value: Value) -> Result<Value, ValidationError> {
        reseal(&mut value);
        validate_receipt(&value)
    }

    #[test]
    fn accepts_synthetic_example_and_reports_non_official_ceiling() {
        let report = validate_receipt(&fixture()).unwrap();
        assert_eq!(report["valid"], true);
        assert_eq!(report["accounting"]["eligible"], 3);
        assert_eq!(report["accounting"]["refused"], 1);
        assert_eq!(report["accounting"]["not_attempted"], 1);
        assert_eq!(report["signature_verified"], false);
        assert_eq!(report["claim_ceiling"]["system_superiority_established"], false);
    }

    #[test]
    fn rejects_hidden_refusal_in_terminal_denominator() {
        let mut value = fixture();
        value["task_accounting"]["terminal_dispositions"]["refused"] = Value::from(0);
        value["task_accounting"]["terminal_dispositions"]["ungraded"] = Value::from(1);
        assert!(validate_mutation(value).unwrap_err().to_string().contains("refused"));
    }

    #[test]
    fn rejects_eligible_task_row_drop() {
        let mut value = fixture();
        value["task_outcomes"].as_array_mut().unwrap().pop();
        assert!(validate_mutation(value).unwrap_err().to_string().contains("task_outcomes"));
    }

    #[test]
    fn rejects_duplicate_task_ids() {
        let mut value = fixture();
        let first_id = value["task_outcomes"][0]["task_id"].clone();
        value["task_outcomes"][1]["task_id"] = first_id;
        assert!(validate_mutation(value).unwrap_err().to_string().contains("duplicate task ID"));
    }

    #[test]
    fn rejects_budget_overrun() {
        let mut value = fixture();
        value["actual_resources"]["wall_time_ms"]["value"] = Value::from(20000);
        assert!(validate_mutation(value).unwrap_err().to_string().contains("budget exceeded"));
    }

    #[test]
    fn rejects_coverage_metric_that_drops_refused_cases() {
        let mut value = fixture();
        value["derived_metrics"][0]["value"] = Value::from(1.0);
        value["derived_metrics"][0]["numerator"] = Value::from(1);
        value["derived_metrics"][0]["denominator"] = Value::from(1);
        assert!(validate_mutation(value).unwrap_err().to_string().contains("eligible_coverage"));
    }

    #[test]
    fn rejects_synthetic_run_claiming_official_score() {
        let mut value = fixture();
        value["official_score"] = json!({
            "status":"reported_by_official_grader",
            "metric_id":"official",
            "formula_version":"1",
            "value":1.0,
            "unit":"fraction",
            "numerator":1,
            "denominator":1,
            "grader_result_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        });
        value["claim_ceiling"]["official_score_is_official"] = Value::Bool(true);
        assert!(validate_mutation(value).unwrap_err().to_string().contains("synthetic"));
    }

    #[test]
    fn rejects_single_arm_claimed_as_comparable() {
        let mut value = fixture();
        value["comparison"]["matched"] = Value::Bool(true);
        value["comparison"]["comparable"] = Value::Bool(true);
        value["comparison"]["non_comparability_reasons"] = json!([]);
        assert!(validate_mutation(value).unwrap_err().to_string().contains("at least two arms"));
    }

    #[test]
    fn rejects_integrity_digest_mismatch() {
        let mut value = fixture();
        value["comparison"]["non_comparability_reasons"] = json!(["changed after sealing"]);
        assert!(validate_receipt(&value).unwrap_err().to_string().contains("payload digest mismatch"));
    }

    #[test]
    fn rejects_unaccounted_top_level_fields() {
        let mut value = fixture();
        value["unreviewed_metric"] = Value::Bool(true);
        assert!(validate_receipt(&value).unwrap_err().to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_duplicate_json_object_keys() {
        assert!(parse_unique(br#"{"same":1,"same":2}"#).unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn rejects_post_hoc_exclusions() {
        let mut value = fixture();
        value["task_accounting"]["universe"] = Value::from(4);
        value["task_accounting"]["excluded"] = Value::from(1);
        value["excluded_tasks"] = json!([{
            "task_id":"post-hoc",
            "input_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "reason_code":"removed_after_results",
            "predeclared":false
        }]);
        assert!(validate_mutation(value).unwrap_err().to_string().contains("predeclared"));
    }
}
