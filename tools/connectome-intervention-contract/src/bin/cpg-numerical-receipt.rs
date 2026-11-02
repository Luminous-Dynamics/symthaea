//! Fail-closed verifier for CPG numerical run receipts.
//!
//! A COMPLETE result requires the raw pre-postprocessing floating-point state
//! artifact. The verifier checks its byte count, SHA-256, finiteness and
//! summary statistics. It does not establish biological validity.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

const EXAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/neuroscience/CPG_NUMERICAL_RUN_RECEIPT_EXAMPLE_V1.json"
);
const NUMERIC_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq)]
struct Report {
    disposition: &'static str,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl Report {
    fn as_json(&self) -> Value {
        json!({
            "disposition": self.disposition,
            "errors": self.errors,
            "warnings": self.warnings,
            "biological_validation": false
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ScanSummary {
    count: usize,
    non_finite_count: usize,
    minimum: f64,
    maximum: f64,
    maximum_absolute: f64,
}

fn at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cursor = value;
    for part in path {
        cursor = cursor.get(*part)?;
    }
    Some(cursor)
}

fn str_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(value, path)?.as_str()
}

fn number_at(value: &Value, path: &[&str]) -> Option<f64> {
    at(value, path)?.as_f64()
}

fn uint_at(value: &Value, path: &[&str]) -> Option<u64> {
    at(value, path)?.as_u64()
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sha256(value: &str) -> bool {
    is_hex(value, 64)
}

fn is_git_sha(value: &str) -> bool {
    is_hex(value, 40)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn near(left: f64, right: f64) -> bool {
    (left - right).abs() <= NUMERIC_TOLERANCE * left.abs().max(right.abs()).max(1.0)
}

fn push_missing_hash(value: &Value, path: &[&str], label: &str, errors: &mut Vec<String>) {
    match str_at(value, path) {
        Some(hash) if is_sha256(hash) => {}
        Some(_) => errors.push(format!("{label}: expected lowercase 64-hex SHA-256")),
        None => errors.push(format!("{label}: missing SHA-256 digest")),
    }
}

fn scan_raw_state(bytes: &[u8], encoding: &str) -> Result<ScanSummary, String> {
    let item_size = match encoding {
        "f32le" => 4,
        "f64le" => 8,
        other => return Err(format!("unsupported raw-state encoding: {other}")),
    };
    if bytes.is_empty() {
        return Err("raw-state artifact is empty".to_owned());
    }
    if bytes.len() % item_size != 0 {
        return Err(format!(
            "raw-state byte length {} is not divisible by {item_size}",
            bytes.len()
        ));
    }

    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut maximum_absolute: f64 = 0.0;
    let mut non_finite_count = 0usize;
    let count = bytes.len() / item_size;

    for chunk in bytes.chunks_exact(item_size) {
        let value = if encoding == "f32le" {
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64
        } else {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        };
        if !value.is_finite() {
            non_finite_count += 1;
            continue;
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        maximum_absolute = maximum_absolute.max(value.abs());
    }

    if non_finite_count == count {
        return Err("raw-state artifact contains no finite values".to_owned());
    }

    Ok(ScanSummary {
        count,
        non_finite_count,
        minimum,
        maximum,
        maximum_absolute,
    })
}

fn verify_noise_channel(value: &Value, path: &[&str], label: &str, errors: &mut Vec<String>) {
    let requested_mean = number_at(value, &[path[0], path[1], "requested_mean"]);
    let requested_stddev = number_at(value, &[path[0], path[1], "requested_stddev"]);
    let applied = at(value, &[path[0], path[1], "applied"]).and_then(Value::as_bool);
    let realized_mean = number_at(value, &[path[0], path[1], "realized_mean"]);
    let realized_stddev = number_at(value, &[path[0], path[1], "realized_stddev"]);
    let sample = str_at(value, &[path[0], path[1], "sample_sha256"]);

    let (
        Some(requested_mean),
        Some(requested_stddev),
        Some(applied),
        Some(realized_mean),
        Some(realized_stddev),
    ) = (
        requested_mean,
        requested_stddev,
        applied,
        realized_mean,
        realized_stddev,
    )
    else {
        errors.push(format!("{label}: missing or non-numeric noise declaration"));
        return;
    };

    if !requested_mean.is_finite()
        || !requested_stddev.is_finite()
        || requested_stddev < 0.0
        || !realized_mean.is_finite()
        || !realized_stddev.is_finite()
        || realized_stddev < 0.0
    {
        errors.push(format!(
            "{label}: noise statistics must be finite and standard deviations non-negative"
        ));
        return;
    }

    let requested = requested_mean != 0.0 || requested_stddev > 0.0;
    if requested && !applied {
        errors.push(format!(
            "{label}: non-zero requested noise is marked unapplied"
        ));
    }
    if requested {
        match sample {
            Some(hash) if is_sha256(hash) => {}
            _ => errors.push(format!("{label}: applied noise requires a sample SHA-256")),
        }
        if requested_stddev > 0.0 && realized_stddev <= 0.0 {
            errors.push(format!("{label}: requested non-zero standard deviation has zero realized standard deviation"));
        }
    } else if applied || realized_mean != 0.0 || realized_stddev != 0.0 || sample.is_some() {
        errors.push(format!(
            "{label}: zero-noise request must not report an applied sample"
        ));
    }
}

fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if path.is_absolute() {
        return Err("raw-state path must be relative to the receipt".to_owned());
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(
                "raw-state path may not contain '.', '..', root or prefix components".to_owned(),
            );
        }
    }
    if value.trim().is_empty() {
        return Err("raw-state relative path is empty".to_owned());
    }
    Ok(path.to_path_buf())
}

/// Resolve an artifact relative to the receipt and enforce the canonical
/// parent boundary after symlink resolution. Lexically safe paths alone do
/// not prevent a directory symlink from escaping the receipt's folder.
fn resolve_artifact_path(manifest_path: &Path, relative: &str) -> Result<PathBuf, String> {
    let relative_path = safe_relative_path(relative)?;
    let parent = manifest_path
        .parent()
        .ok_or_else(|| "manifest path has no parent directory".to_owned())?;
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|error| format!("cannot canonicalize manifest directory: {error}"))?;
    let target = canonical_parent.join(relative_path);
    let canonical_target = fs::canonicalize(&target)
        .map_err(|error| format!("cannot canonicalize raw-state artifact: {error}"))?;
    if !canonical_target.starts_with(&canonical_parent) {
        return Err("raw-state artifact resolves outside the receipt directory".to_owned());
    }
    let metadata = fs::metadata(&canonical_target)
        .map_err(|error| format!("cannot inspect raw-state artifact: {error}"))?;
    if !metadata.is_file() {
        return Err("raw-state artifact is not a regular file".to_owned());
    }
    Ok(canonical_target)
}

fn verify_complete(value: &Value, raw_state_bytes: Option<&[u8]>) -> Vec<String> {
    let mut errors = Vec::new();

    let tier = str_at(value, &["evidence_tier"]);
    let ceiling = str_at(value, &["claim_ceiling"]);
    let source_kind = str_at(value, &["source", "source_kind"]);
    match tier {
        Some("synthetic_known_answer") => {
            if ceiling != Some("synthetic_only") || source_kind != Some("synthetic") {
                errors.push(
                    "synthetic evidence requires synthetic source and synthetic_only claim ceiling"
                        .to_owned(),
                );
            }
            if str_at(value, &["source", "rights_state"]) != Some("synthetic") {
                errors.push("synthetic source must declare rights_state=synthetic".to_owned());
            }
        }
        Some("in_silico_intervention") => {
            if ceiling != Some("model_counterfactual_only") || source_kind != Some("in_silico") {
                errors.push(
                    "in-silico evidence requires model_counterfactual_only claim ceiling and in_silico source"
                        .to_owned(),
                );
            }
            if str_at(value, &["source", "rights_state"]) != Some("documented_permitted") {
                errors.push("in-silico source must declare documented_permitted rights".to_owned());
            }
            push_missing_hash(
                value,
                &["source", "data_manifest_sha256"],
                "source.data_manifest_sha256",
                &mut errors,
            );
            push_missing_hash(
                value,
                &["source", "permission_receipt_sha256"],
                "source.permission_receipt_sha256",
                &mut errors,
            );
        }
        _ => errors.push("evidence_tier: unsupported".to_owned()),
    }

    match str_at(value, &["source", "code_commit_sha"]) {
        Some(commit) if is_git_sha(commit) => {}
        _ => errors.push(
            "source.code_commit_sha: missing or malformed immutable source commit".to_owned(),
        ),
    }
    push_missing_hash(
        value,
        &["source", "source_file_sha256"],
        "source.source_file_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["simulator", "equation_manifest_sha256"],
        "simulator.equation_manifest_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["simulator", "parameter_manifest_sha256"],
        "simulator.parameter_manifest_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["solver", "dependency_lock_sha256"],
        "solver.dependency_lock_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["numerics", "raw_output_manifest_sha256"],
        "numerics.raw_output_manifest_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["perturbation", "protocol_sha256"],
        "perturbation.protocol_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["perturbation", "waveform_sha256"],
        "perturbation.waveform_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["artifacts", "resolved_config_sha256"],
        "artifacts.resolved_config_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["artifacts", "environment_sha256"],
        "artifacts.environment_sha256",
        &mut errors,
    );
    push_missing_hash(
        value,
        &["artifacts", "logs_sha256"],
        "artifacts.logs_sha256",
        &mut errors,
    );

    if at(value, &["perturbation", "stimulus_applied"]).and_then(Value::as_bool) != Some(true) {
        errors.push("perturbation.stimulus_applied: complete run requires the declared stimulus to be applied".to_owned());
    }
    verify_noise_channel(
        value,
        &["perturbation", "input_noise"],
        "perturbation.input_noise",
        &mut errors,
    );
    verify_noise_channel(
        value,
        &["perturbation", "weight_noise"],
        "perturbation.weight_noise",
        &mut errors,
    );

    let solver_status = str_at(value, &["simulation", "solver_status"]);
    if solver_status != Some("success") {
        errors.push(format!(
            "simulation.solver_status: expected success, found {solver_status:?}"
        ));
    }
    let start = number_at(value, &["simulation", "start_time_seconds"]);
    let end = number_at(value, &["simulation", "end_time_seconds"]);
    if start
        .zip(end)
        .is_none_or(|(a, b)| !a.is_finite() || !b.is_finite() || b <= a)
    {
        errors.push("simulation time interval must be finite and end after start".to_owned());
    }
    let expected_saved = uint_at(value, &["simulation", "expected_saved_timepoints"]);
    let actual_saved = uint_at(value, &["simulation", "actual_saved_timepoints"]);
    if expected_saved.is_none() || actual_saved != expected_saved {
        errors.push(
            "simulation.actual_saved_timepoints must equal expected_saved_timepoints".to_owned(),
        );
    }
    let state_dimension = uint_at(value, &["simulation", "state_dimension"]);
    if state_dimension.is_none_or(|count| count == 0) {
        errors.push("simulation.state_dimension must be positive".to_owned());
    }
    let attempted = uint_at(value, &["simulation", "attempted_steps"]);
    let accepted = uint_at(value, &["simulation", "accepted_steps"]);
    let rejected = uint_at(value, &["simulation", "rejected_steps"]);
    if attempted.is_none_or(|n| n == 0) {
        errors.push("simulation.attempted_steps must be positive".to_owned());
    }
    if accepted.is_none_or(|n| n == 0) {
        errors.push("simulation.accepted_steps must be positive".to_owned());
    }
    if let (Some(attempted), Some(accepted), Some(rejected)) = (attempted, accepted, rejected) {
        if accepted.checked_add(rejected) != Some(attempted) {
            errors.push(
                "simulation.attempted_steps must equal accepted_steps + rejected_steps".to_owned(),
            );
        }
    }
    if at(value, &["solver", "throw_on_failure"]).and_then(Value::as_bool) != Some(true) {
        errors.push("solver.throw_on_failure must be true for a complete run".to_owned());
    }
    if uint_at(value, &["simulation", "function_evaluations"]).is_none_or(|n| n == 0) {
        errors.push("simulation.function_evaluations must be positive".to_owned());
    }
    if number_at(value, &["solver", "rtol"]).is_none_or(|n| !n.is_finite() || n <= 0.0)
        || number_at(value, &["solver", "atol"]).is_none_or(|n| !n.is_finite() || n <= 0.0)
    {
        errors.push("solver.rtol and solver.atol must be finite and positive".to_owned());
    }
    if uint_at(value, &["solver", "max_steps"]).is_none_or(|n| n == 0) {
        errors.push("solver.max_steps must be positive".to_owned());
    }

    let dtype = str_at(value, &["solver", "dtype"]);
    let artifact = value
        .get("numerics")
        .and_then(|v| v.get("raw_state_artifact"));
    let encoding = artifact
        .and_then(|v| v.get("encoding"))
        .and_then(Value::as_str);
    match (dtype, encoding) {
        (Some("float32"), Some("f32le")) | (Some("float64"), Some("f64le")) => {}
        _ => errors.push("solver.dtype and numerics.raw_state_artifact.encoding must agree (float32/f32le or float64/f64le)".to_owned()),
    }
    if str_at(value, &["numerics", "raw_state_artifact", "capture_stage"])
        != Some("direct_solver_output_before_postprocessing")
    {
        errors.push(
            "raw_state_artifact.capture_stage must be direct_solver_output_before_postprocessing"
                .to_owned(),
        );
    }
    if !str_at(value, &["numerics", "raw_state_artifact", "relative_path"])
        .is_some_and(|p| safe_relative_path(p).is_ok())
    {
        errors.push("numerics.raw_state_artifact.relative_path is missing or unsafe".to_owned());
    }

    let Some(bytes) = raw_state_bytes else {
        errors.push(
            "raw pre-postprocessing state artifact bytes were not supplied to the verifier"
                .to_owned(),
        );
        return errors;
    };

    let artifact_bytes = uint_at(value, &["numerics", "raw_state_artifact", "byte_length"]);
    if artifact_bytes != Some(bytes.len() as u64) {
        errors.push("raw_state_artifact.byte_length does not match supplied bytes".to_owned());
    }
    let computed_sha = sha256_hex(bytes);
    for path in [
        &["numerics", "raw_state_artifact", "sha256"][..],
        &["numerics", "raw_state_sha256"][..],
    ] {
        if str_at(value, path) != Some(computed_sha.as_str()) {
            errors.push(format!(
                "{}: digest does not match raw state bytes",
                path.join(".")
            ));
        }
    }

    if let (Some(state_dim), Some(timepoints), Some(encoding)) =
        (state_dimension, actual_saved, encoding)
    {
        let item_size = if encoding == "f32le" {
            4u64
        } else if encoding == "f64le" {
            8u64
        } else {
            0u64
        };
        let expected_bytes = state_dim
            .checked_mul(timepoints)
            .and_then(|n| n.checked_mul(item_size));
        if expected_bytes != Some(bytes.len() as u64) {
            errors.push("raw-state byte count does not equal state_dimension × actual_saved_timepoints × bytes_per_value".to_owned());
        }
    }

    if let Some(encoding) = encoding {
        match scan_raw_state(bytes, encoding) {
            Ok(scan) => {
                if uint_at(value, &["numerics", "non_finite_value_count"])
                    != Some(scan.non_finite_count as u64)
                {
                    errors.push(
                        "numerics.non_finite_value_count does not match raw-state scan".to_owned(),
                    );
                }
                if scan.non_finite_count != 0 {
                    errors.push(format!(
                        "raw state contains {} non-finite values",
                        scan.non_finite_count
                    ));
                }
                if at(value, &["numerics", "raw_state_finite_scan_performed"])
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    errors.push("numerics.raw_state_finite_scan_performed must be true".to_owned());
                }
                if !number_at(value, &["numerics", "minimum_raw_value"])
                    .is_some_and(|x| near(x, scan.minimum))
                {
                    errors.push(
                        "numerics.minimum_raw_value does not match raw-state scan".to_owned(),
                    );
                }
                if !number_at(value, &["numerics", "maximum_raw_value"])
                    .is_some_and(|x| near(x, scan.maximum))
                {
                    errors.push(
                        "numerics.maximum_raw_value does not match raw-state scan".to_owned(),
                    );
                }
                if !number_at(value, &["numerics", "maximum_absolute_raw_value"])
                    .is_some_and(|x| near(x, scan.maximum_absolute))
                {
                    errors.push(
                        "numerics.maximum_absolute_raw_value does not match raw-state scan"
                            .to_owned(),
                    );
                }
                if let (Some(state_dim), Some(timepoints)) = (state_dimension, actual_saved) {
                    if scan.count as u64 != state_dim.saturating_mul(timepoints) {
                        errors.push("decoded raw-state value count does not match declared state dimensions".to_owned());
                    }
                }
            }
            Err(error) => errors.push(error),
        }
    }

    for field in [
        "post_solver_sanitized_value_count",
        "post_solver_clipped_value_count",
        "post_solver_transformed_value_count",
    ] {
        if uint_at(value, &["numerics", field]) != Some(0) {
            errors.push(format!("numerics.{field}: complete run requires zero"));
        }
    }

    if number_at(value, &["resources", "wall_time_seconds"])
        .is_none_or(|n| !n.is_finite() || n <= 0.0)
    {
        errors.push("resources.wall_time_seconds must be positive".to_owned());
    }
    if uint_at(value, &["resources", "peak_memory_bytes"]).is_none_or(|n| n == 0) {
        errors.push("resources.peak_memory_bytes must be positive".to_owned());
    }

    errors
}

fn verify_manifest(value: &Value, raw_state_bytes: Option<&[u8]>) -> Report {
    if str_at(value, &["schema_version"]) != Some("CPG-NUMERICAL-RUN-RECEIPT-V1") {
        return Report {
            disposition: "INVALID",
            errors: vec!["schema_version: unsupported".to_owned()],
            warnings: Vec::new(),
        };
    }
    match str_at(value, &["run_state"]) {
        Some("incomplete" | "planned" | "running") => {
            let tier = str_at(value, &["evidence_tier"]);
            let ceiling = str_at(value, &["claim_ceiling"]);
            let correct = matches!(
                (tier, ceiling),
                (Some("synthetic_known_answer"), Some("synthetic_only"))
                    | (
                        Some("in_silico_intervention"),
                        Some("model_counterfactual_only")
                    )
            );
            if !correct {
                return Report {
                    disposition: "INVALID",
                    errors: vec!["evidence tier and claim ceiling are inconsistent".to_owned()],
                    warnings: Vec::new(),
                };
            }
            Report {
                disposition: "INCOMPLETE",
                errors: Vec::new(),
                warnings: vec![
                    "Incomplete solver receipt is not a numerical pass or biological result."
                        .to_owned(),
                ],
            }
        }
        Some("failed" | "invalid") => Report {
            disposition: "NOT_COMPLETE",
            errors: Vec::new(),
            warnings: vec!["Receipt explicitly records a failed or invalid run.".to_owned()],
        },
        Some("complete") => {
            let mut errors = verify_complete(value, raw_state_bytes);
            errors.sort();
            errors.dedup();
            if errors.is_empty() {
                Report {
                    disposition: "COMPLETE",
                    errors,
                    warnings: vec!["Numerical execution contract passed; model and biological validity are not established.".to_owned()],
                }
            } else {
                Report {
                    disposition: "INVALID",
                    errors,
                    warnings: Vec::new(),
                }
            }
        }
        _ => Report {
            disposition: "INVALID",
            errors: vec!["run_state: missing or unsupported".to_owned()],
            warnings: Vec::new(),
        },
    }
}

fn main() {
    let default_manifest = PathBuf::from(EXAMPLE);
    let manifest_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or(default_manifest);
    let manifest_text = match fs::read_to_string(&manifest_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot read {}: {error}", manifest_path.display());
            std::process::exit(2);
        }
    };
    let value: Value = match serde_json::from_str(&manifest_text) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid JSON in {}: {error}", manifest_path.display());
            std::process::exit(2);
        }
    };

    let raw_state_bytes = if str_at(&value, &["run_state"]) == Some("complete") {
        let relative = match str_at(&value, &["numerics", "raw_state_artifact", "relative_path"]) {
            Some(path) => path,
            None => {
                eprintln!("complete receipt is missing raw-state artifact path");
                std::process::exit(2);
            }
        };
        let resolved = match resolve_artifact_path(&manifest_path, relative) {
            Ok(path) => path,
            Err(error) => {
                eprintln!("unsafe or unavailable raw-state artifact: {error}");
                std::process::exit(2);
            }
        };
        match fs::read(&resolved) {
            Ok(bytes) => Some(bytes),
            Err(error) => {
                eprintln!("cannot read raw-state artifact: {error}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    let report = verify_manifest(&value, raw_state_bytes.as_deref());
    println!("{}", report.as_json());
    if report.disposition == "INVALID" {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64_bytes(values: &[f64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn complete_fixture(values: &[f64]) -> Value {
        let bytes = f64_bytes(values);
        let hash = sha256_hex(&bytes);
        json!({
            "schema_version":"CPG-NUMERICAL-RUN-RECEIPT-V1",
            "run_id":"synthetic-cpg-complete-001",
            "run_state":"complete",
            "evidence_tier":"synthetic_known_answer",
            "claim_ceiling":"synthetic_only",
            "created_at_utc":"2026-11-02T00:00:00Z",
            "source":{
                "source_kind":"synthetic","repository":null,"immutable_code_ref":null,
                "code_commit_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "source_file_path":"synthetic://rate-equation.rs","source_file_sha256":digest("source"),
                "data_revision":null,"data_manifest_sha256":null,"rights_state":"synthetic",
                "permission_receipt_sha256":null,"provenance_notes":"synthetic verifier fixture"
            },
            "simulator":{
                "model_family":"rate_network","equation_manifest_sha256":digest("equations"),
                "parameter_manifest_sha256":digest("parameters"),
                "state_semantics":"Synthetic three-state bounded rate system.","observation_model_id":"identity"
            },
            "solver":{
                "solver_id":"Dopri5","library":"reference-test-integrator","library_version":"1.0",
                "dependency_lock_sha256":digest("lock"),"backend":"cpu","dtype":"float64",
                "rtol":1.0e-8,"atol":1.0e-10,"initial_step_seconds":0.001,"max_steps":10000,
                "throw_on_failure":true,"nonfinite_event_detection_enabled":true
            },
            "simulation":{
                "start_time_seconds":0.0,"end_time_seconds":0.01,"nominal_save_interval_seconds":0.01,
                "state_dimension":3,"expected_saved_timepoints":2,"actual_saved_timepoints":2,
                "solver_status":"success","attempted_steps":1,"accepted_steps":1,"rejected_steps":0,
                "function_evaluations":7,"random_seed":7,"prng_id":"synthetic-counter-v1","deterministic_mode":true
            },
            "perturbation":{
                "protocol_sha256":digest("protocol"),"waveform_sha256":digest("waveform"),"stimulus_applied":true,
                "input_noise":{"requested_mean":0.0,"requested_stddev":0.0,"applied":false,"realized_mean":0.0,"realized_stddev":0.0,"sample_sha256":null},
                "weight_noise":{"requested_mean":0.0,"requested_stddev":0.0,"applied":false,"realized_mean":0.0,"realized_stddev":0.0,"sample_sha256":null}
            },
            "numerics":{
                "raw_state_sha256":hash,"raw_output_manifest_sha256":digest("raw-output-manifest"),
                "raw_state_finite_scan_performed":true,"non_finite_value_count":0,
                "post_solver_sanitized_value_count":0,"post_solver_clipped_value_count":0,"post_solver_transformed_value_count":0,
                "minimum_raw_value":0.1,"maximum_raw_value":0.6,"maximum_absolute_raw_value":0.6,
                "raw_state_artifact":{
                    "relative_path":"raw/state.f64le","encoding":"f64le","byte_length":bytes.len(),
                    "sha256":hash,"capture_stage":"direct_solver_output_before_postprocessing",
                    "layout":"state_major_time_minor"
                }
            },
            "resources":{"wall_time_seconds":0.01,"peak_memory_bytes":4096,"energy_joules":null},
            "artifacts":{"resolved_config_sha256":digest("config"),"environment_sha256":digest("environment"),"logs_sha256":digest("logs")},
            "limitations":["synthetic numerical-verifier fixture only"]
        })
    }

    #[test]
    fn finite_complete_synthetic_receipt_passes_with_actual_raw_bytes() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let report = verify_manifest(&complete_fixture(&values), Some(&bytes));
        assert_eq!(report.disposition, "COMPLETE", "{:?}", report.errors);
    }

    #[test]
    fn incomplete_public_example_is_not_promoted() {
        let example: Value = serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/CPG_NUMERICAL_RUN_RECEIPT_EXAMPLE_V1.json"
        ))
        .unwrap();
        let report = verify_manifest(&example, None);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.errors.is_empty());
    }

    #[test]
    fn actual_nan_is_rejected_even_when_receipt_claims_zero_non_finite_values() {
        let values = [0.1, 0.2, f64::NAN, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["numerics"]["non_finite_value_count"] = json!(0);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("non-finite"))
        );
    }

    #[test]
    fn post_solver_sanitization_is_rejected() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["numerics"]["post_solver_sanitized_value_count"] = json!(1);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("post_solver_sanitized"))
        );
    }

    #[test]
    fn post_solver_clipping_is_rejected() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["numerics"]["post_solver_clipped_value_count"] = json!(1);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("post_solver_clipped"))
        );
    }

    #[test]
    fn non_success_solver_status_is_rejected() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["simulation"]["solver_status"] = json!("max_steps_reached");
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("solver_status"))
        );
    }

    #[test]
    fn attempted_steps_must_reconcile_with_accepted_and_rejected_steps() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["simulation"]["attempted_steps"] = json!(2);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("attempted_steps must equal"))
        );
    }

    #[test]
    fn complete_run_requires_solver_to_throw_on_failure() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["solver"]["throw_on_failure"] = json!(false);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("throw_on_failure"))
        );
    }

    #[test]
    fn requested_input_noise_cannot_be_silently_ignored() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["perturbation"]["input_noise"]["requested_stddev"] = json!(0.1);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("non-zero requested noise"))
        );
    }

    #[test]
    fn requested_weight_noise_cannot_be_silently_ignored() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["perturbation"]["weight_noise"]["requested_mean"] = json!(0.1);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("non-zero requested noise"))
        );
    }

    #[test]
    fn raw_state_digest_must_match_actual_bytes() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["numerics"]["raw_state_artifact"]["sha256"] = json!(digest("wrong"));
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("digest does not match"))
        );
    }

    #[test]
    fn raw_state_shape_must_match_dimensions_and_saved_timepoints() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["simulation"]["state_dimension"] = json!(4);
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("state_dimension"))
        );
    }

    #[test]
    fn unsafe_relative_paths_are_rejected() {
        assert!(safe_relative_path("../state.f64le").is_err());
        assert!(safe_relative_path("/tmp/state.f64le").is_err());
        assert!(safe_relative_path("raw/state.f64le").is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_from_receipt_directory_is_rejected() {
        use std::os::unix::fs::symlink;

        let root =
            std::env::temp_dir().join(format!("cpg-receipt-path-boundary-{}", std::process::id()));
        let receipt_dir = root.join("receipt");
        let outside_dir = root.join("outside");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(receipt_dir.join("artifacts")).unwrap();
        fs::create_dir_all(&outside_dir).unwrap();
        fs::write(outside_dir.join("state.f64le"), [0u8; 8]).unwrap();
        symlink(&outside_dir, receipt_dir.join("artifacts/escape")).unwrap();

        let manifest = receipt_dir.join("receipt.json");
        let rejected = resolve_artifact_path(&manifest, "artifacts/escape/state.f64le");
        assert!(rejected.is_err());
        let allowed = resolve_artifact_path(&manifest, "artifacts");
        assert!(allowed.is_err()); // a directory is not a raw-state file

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn float_encoding_must_match_solver_dtype() {
        let values = [0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["numerics"]["raw_state_artifact"]["encoding"] = json!("f32le");
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("dtype and"))
        );
    }

    #[test]
    fn measured_or_in_silico_claim_ceiling_cannot_exceed_evidence() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bytes = f64_bytes(&values);
        let mut fixture = complete_fixture(&values);
        fixture["evidence_tier"] = json!("in_silico_intervention");
        fixture["claim_ceiling"] = json!("measured_intervention_prediction");
        let report = verify_manifest(&fixture, Some(&bytes));
        assert_eq!(report.disposition, "INVALID");
    }
}
