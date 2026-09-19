// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1C result-integrity qualification gate.
//!
//! This program performs no cosmology. It verifies that an A1C result is bound
//! to the exact current subject, authorization, A1Q/A1E/contract prerequisites,
//! executor implementation, and frozen fixed-point decision rule.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_JSON_BYTES: u64 = 8 * 1024 * 1024;
const PROTOCOL: &str = "DE-001A1C-RESULT-QUALIFICATION-v1";
const AUTHORITY: &str = "a1c-result-integrity-only";
const RESULT_PROTOCOL: &str = "DE-001A1C-COBAYA-RESULT-v1";
const AUTH_PROTOCOL: &str = "DE-001A1C-EXECUTION-AUTHORIZATION-v1";
const A1Q_PROTOCOL: &str = "DE-001A1Q-BUNDLE-INTEGRITY-v1";
const A1E_PROTOCOL: &str = "DE-001A1E-ENVIRONMENT-EVIDENCE-v1";
const CONTRACT_PROTOCOL: &str = "DE-001A1C-CONTRACT-CONSISTENCY-v1";
const EXECUTOR_PROTOCOL: &str = "DE-001A1C-COBAYA-EXECUTOR-v1";
const SCIENTIFIC_CLAIM: &str = "NONE";

const OMEGA_M: f64 = 0.297_177_87;
const H_R_D_MPC: f64 = 101.547_86;
const REFERENCE_CHI2: f64 = 10.282_299;
const ABSOLUTE_TOLERANCE: f64 = 0.01;
const ROW_COUNT: usize = 13;
const MEAN_SHA256: &str = "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585";
const COV_SHA256: &str = "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509";
const LIKELIHOOD_YAML_SHA256: &str =
    "fd7e9bf2dcf5ffee90a9a30b18227f4337d6d5c1978782c63513cbe0d8280daa";
const COBAYA_VERSION: &str = "3.6.2";
const COBAYA_COMMIT: &str = "899f30a49f85de610dac321e91a1af50018e56aa";

const SPEC_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/references/de001a_a1c_result_qualification_v1.json";
const EXECUTOR_SPEC_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_a1c_executor_v1.json";
const A1C_MANIFEST_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/references/de001a_a1c_cobaya_fixed_point_v1.json";

const RESULT_KEYS: &[&str] = &[
    "protocol",
    "verdict",
    "scientific_claim",
    "authority",
    "subject_head",
    "subject_tree",
    "authorization_receipt_sha256",
    "authorization_spec_sha256",
    "executor_spec_sha256",
    "executor_script_sha256",
    "a1c_manifest_sha256",
    "a1q_receipt_sha256",
    "a1e_receipt_sha256",
    "a1c_contract_receipt_sha256",
    "a0_receipt_sha256",
    "environment_receipt_sha256",
    "environment_versions_sha256",
    "point_manifest_sha256",
    "cobaya_version",
    "cobaya_source_commit",
    "installed_likelihood_yaml_sha256",
    "installed_likelihood_class_source_sha256",
    "dataset_mean_sha256",
    "dataset_covariance_sha256",
    "omega_m",
    "h_r_d_mpc",
    "rdrag_gauge_mpc",
    "h0_gauge_km_s_mpc",
    "likelihood_call_budget",
    "likelihood_call_count",
    "logp_bao",
    "chi2_bao",
    "reference_chi2_bao",
    "absolute_delta_chi2",
    "absolute_tolerance",
    "internal_chi2_tolerance",
    "prediction_vector",
    "prediction_vector_sha256",
    "a1q_reproduction_verdict",
];

const AUTH_KEYS: &[&str] = &[
    "protocol",
    "verdict",
    "scientific_claim",
    "authority",
    "a1c_execution_authorized",
    "subject_head",
    "subject_tree",
    "authorization_spec_sha256",
    "a1c_manifest_sha256",
    "a1q_receipt_sha256",
    "a1e_receipt_sha256",
    "a1c_contract_receipt_sha256",
    "a1q_reproduction_verdict",
    "environment_definition_sha256",
    "likelihood_call_budget",
    "sampler_allowed",
    "minimizer_allowed",
    "optimization_allowed",
    "parameter_mutation_allowed",
    "network_allowed",
    "camb_allowed",
    "runtime_package_installation_allowed",
    "authorization_scope",
    "stateful_global_single_use_claimed",
];

#[derive(Debug, Serialize)]
struct QualificationReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    a1x_comparison_authorized: bool,
    a2_execution_authorized: bool,
    subject_head: String,
    subject_tree: String,
    qualification_spec_sha256: String,
    a1c_result_sha256: String,
    authorization_receipt_sha256: String,
    a1q_receipt_sha256: String,
    a1e_receipt_sha256: String,
    a1c_contract_receipt_sha256: String,
    executor_spec_sha256: String,
    executor_script_sha256: String,
    a1c_manifest_sha256: String,
    a0_receipt_sha256: String,
    environment_receipt_sha256: String,
    point_manifest_sha256: String,
    reproduction_verdict: String,
    chi2_bao: f64,
    prediction_vector_sha256: String,
    likelihood_call_count: u64,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    a1x_comparison_authorized: bool,
    a2_execution_authorized: bool,
    error: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > MAX_JSON_BYTES {
        return Err(format!("{}: file too large", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn parse_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("invalid {label} JSON: {error}"))
}

fn run_text(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|error| format!("{program} output was not UTF-8: {error}"))
}

fn repository_identity() -> Result<(PathBuf, String, String), String> {
    let root = run_text("git", &["rev-parse", "--show-toplevel"])?;
    let head = run_text("git", &["rev-parse", "HEAD"])?;
    let tree = run_text("git", &["rev-parse", "HEAD^{tree}"])?;
    if root.is_empty() || head.is_empty() || tree.is_empty() {
        return Err("git returned an empty repository identity".into());
    }
    Ok((PathBuf::from(root), head, tree))
}

fn require_exact_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let wanted: BTreeSet<_> = expected.iter().copied().collect();
    if actual != wanted {
        return Err(format!("{label} field set differs from the frozen schema"));
    }
    Ok(())
}

fn field_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field: {key}"))
}

fn field_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or non-boolean field: {key}"))
}

fn field_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or non-u64 field: {key}"))
}

fn field_f64(value: &Value, key: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("missing or non-finite numeric field: {key}"))
}

fn nested<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing nested field: {key}"))
}

fn exact_f64(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

fn valid_lane_verdict(value: &str) -> bool {
    matches!(value, "PASS" | "NEGATIVE")
}

fn rederive_verdict(chi2: f64, reference: f64, tolerance: f64) -> &'static str {
    if (chi2 - reference).abs() <= tolerance {
        "PASS"
    } else {
        "NEGATIVE"
    }
}

fn ordered_strings_equal(value: &Value, expected: &[&str]) -> bool {
    let Some(array) = value.as_array() else {
        return false;
    };
    array.len() == expected.len()
        && array
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.as_str() == Some(*expected))
}

fn validate_spec(spec: &Value) -> Result<(), String> {
    if spec.get("schema_version").and_then(Value::as_u64) != Some(1)
        || field_str(spec, "protocol")? != PROTOCOL
        || field_str(spec, "status")? != "preregistered-integrity-gate"
        || field_str(spec, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(spec, "authority")? != AUTHORITY
        || field_str(spec, "accepted_result_protocol")? != RESULT_PROTOCOL
        || !ordered_strings_equal(
            nested(spec, "accepted_reproduction_verdicts")?,
            &["PASS", "NEGATIVE"],
        )
        || field_str(spec, "required_authorization_protocol")? != AUTH_PROTOCOL
        || field_str(spec, "required_a1q_protocol")? != A1Q_PROTOCOL
        || field_str(spec, "required_a1e_protocol")? != A1E_PROTOCOL
        || field_str(spec, "required_a1c_contract_protocol")? != CONTRACT_PROTOCOL
        || field_str(spec, "required_executor_protocol")? != EXECUTOR_PROTOCOL
    {
        return Err("A1C result-qualification specification identity drifted".into());
    }

    let subject = nested(spec, "subject")?;
    if !exact_f64(field_f64(subject, "omega_m")?, OMEGA_M)
        || !exact_f64(field_f64(subject, "h_r_d_mpc")?, H_R_D_MPC)
        || !exact_f64(field_f64(subject, "reference_chi2_bao")?, REFERENCE_CHI2)
        || !exact_f64(
            field_f64(subject, "absolute_tolerance")?,
            ABSOLUTE_TOLERANCE,
        )
        || field_u64(subject, "row_count")? != ROW_COUNT as u64
        || field_str(subject, "mean_sha256")? != MEAN_SHA256
        || field_str(subject, "covariance_sha256")? != COV_SHA256
    {
        return Err("A1C result-qualification subject drifted".into());
    }

    let requirements = nested(spec, "integrity_requirements")?;
    for key in [
        "current_head_tree_must_equal_result_subject",
        "authorization_hash_must_match_result",
        "a1q_hash_must_match_result_and_authorization",
        "a1e_hash_must_match_result_and_authorization",
        "a1c_contract_hash_must_match_result_and_authorization",
        "executor_spec_must_match_current_checkout",
        "executor_script_must_match_current_checkout",
        "likelihood_call_count_must_equal_one",
        "stored_delta_and_verdict_must_rederive",
        "prediction_vector_must_have_13_finite_entries",
    ] {
        if !field_bool(requirements, key)? {
            return Err(format!("qualification requirement {key} is not frozen true"));
        }
    }

    let promotion = nested(spec, "promotion")?;
    let excluded = [
        "A1X agreement",
        "optimizer reproduction",
        "LambdaCDM validity",
        "dynamic dark energy",
        "observational anomaly",
        "physical mechanism",
    ];
    if field_str(promotion, "pass_authority")?
        != "qualified A1C fixed-point result eligible for A1X comparison only"
        || !field_bool(promotion, "a1x_comparison_authorized")?
        || field_bool(promotion, "a2_execution_authorized")?
        || !ordered_strings_equal(nested(promotion, "does_not_establish")?, &excluded)
    {
        return Err("A1C result-qualification promotion boundary drifted".into());
    }
    Ok(())
}

fn validate_hash(value: &Value, key: &str) -> Result<&str, String> {
    let digest = field_str(value, key)?;
    if !is_sha256(digest) {
        return Err(format!("malformed SHA-256 field: {key}"));
    }
    Ok(digest)
}

fn execute(
    result_path: &Path,
    authorization_path: &Path,
    a1q_path: &Path,
    a1e_path: &Path,
    contract_path: &Path,
) -> Result<QualificationReceipt, String> {
    let (root, head, tree) = repository_identity()?;
    let spec_bytes = read_regular_file(&root.join(SPEC_RELATIVE))?;
    let executor_spec_bytes = read_regular_file(&root.join(EXECUTOR_SPEC_RELATIVE))?;
    let manifest_bytes = read_regular_file(&root.join(A1C_MANIFEST_RELATIVE))?;
    let result_bytes = read_regular_file(result_path)?;
    let authorization_bytes = read_regular_file(authorization_path)?;
    let a1q_bytes = read_regular_file(a1q_path)?;
    let a1e_bytes = read_regular_file(a1e_path)?;
    let contract_bytes = read_regular_file(contract_path)?;

    let spec = parse_json(&spec_bytes, "qualification specification")?;
    let executor_spec = parse_json(&executor_spec_bytes, "executor specification")?;
    let result = parse_json(&result_bytes, "A1C result")?;
    let authorization = parse_json(&authorization_bytes, "A1C authorization receipt")?;
    let a1q = parse_json(&a1q_bytes, "A1Q receipt")?;
    let a1e = parse_json(&a1e_bytes, "A1E receipt")?;
    let contract = parse_json(&contract_bytes, "A1C contract receipt")?;

    validate_spec(&spec)?;
    require_exact_keys(&result, RESULT_KEYS, "A1C result")?;
    require_exact_keys(&authorization, AUTH_KEYS, "A1C authorization receipt")?;

    if field_str(&result, "protocol")? != RESULT_PROTOCOL
        || field_str(&result, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&result, "authority")?
            != "released-likelihood-fixed-point-reproduction-only"
        || !valid_lane_verdict(field_str(&result, "verdict")?)
        || field_str(&result, "subject_head")? != head
        || field_str(&result, "subject_tree")? != tree
    {
        return Err("A1C result does not describe the current exact qualified subject".into());
    }

    if field_str(&authorization, "protocol")? != AUTH_PROTOCOL
        || field_str(&authorization, "verdict")? != "PASS"
        || field_str(&authorization, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&authorization, "authority")? != "fixed-point-execution-authorization-only"
        || !field_bool(&authorization, "a1c_execution_authorized")?
        || field_str(&authorization, "subject_head")? != head
        || field_str(&authorization, "subject_tree")? != tree
    {
        return Err("A1C authorization does not qualify the current exact subject".into());
    }

    let authorization_sha256 = sha256_hex(&authorization_bytes);
    let a1q_sha256 = sha256_hex(&a1q_bytes);
    let a1e_sha256 = sha256_hex(&a1e_bytes);
    let contract_sha256 = sha256_hex(&contract_bytes);

    if field_str(&result, "authorization_receipt_sha256")? != authorization_sha256
        || field_str(&result, "a1q_receipt_sha256")? != a1q_sha256
        || field_str(&result, "a1e_receipt_sha256")? != a1e_sha256
        || field_str(&result, "a1c_contract_receipt_sha256")? != contract_sha256
        || field_str(&authorization, "a1q_receipt_sha256")? != a1q_sha256
        || field_str(&authorization, "a1e_receipt_sha256")? != a1e_sha256
        || field_str(&authorization, "a1c_contract_receipt_sha256")? != contract_sha256
    {
        return Err("A1C result prerequisite hashes do not match the authorization lineage".into());
    }

    if field_str(&a1q, "protocol")? != A1Q_PROTOCOL
        || field_str(&a1q, "verdict")? != "PASS"
        || field_str(&a1q, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&a1q, "authority")? != "evidence-bundle-integrity-only"
        || field_str(&a1q, "subject_head")? != head
        || field_str(&a1q, "subject_tree")? != tree
    {
        return Err("A1Q prerequisite does not qualify the current subject".into());
    }
    if field_str(&a1e, "protocol")? != A1E_PROTOCOL
        || field_str(&a1e, "verdict")? != "PASS"
        || field_str(&a1e, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&a1e, "authority")? != "environment-evidence-reuse-only"
        || !field_bool(&a1e, "environment_reuse_authorized")?
        || field_bool(&a1e, "a1c_execution_authorized")?
    {
        return Err("A1E prerequisite is not a qualified reusable environment".into());
    }
    if field_str(&contract, "protocol")? != CONTRACT_PROTOCOL
        || field_str(&contract, "verdict")? != "PASS"
        || field_str(&contract, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&contract, "authority")? != "contract-consistency-only"
        || field_bool(&contract, "execution_authorized")?
    {
        return Err("A1C contract prerequisite is invalid".into());
    }

    let current_executor_spec_sha256 = sha256_hex(&executor_spec_bytes);
    let current_manifest_sha256 = sha256_hex(&manifest_bytes);
    if field_str(&result, "executor_spec_sha256")? != current_executor_spec_sha256
        || field_str(&result, "a1c_manifest_sha256")? != current_manifest_sha256
        || field_str(&authorization, "a1c_manifest_sha256")? != current_manifest_sha256
        || field_str(&contract, "a1c_manifest_sha256")? != current_manifest_sha256
    {
        return Err("current A1C implementation/manifest does not match the result lineage".into());
    }

    if field_str(&executor_spec, "protocol")? != EXECUTOR_PROTOCOL {
        return Err("current executor specification protocol drifted".into());
    }
    let script_relative = field_str(&executor_spec, "script_path")?;
    let script_bytes = read_regular_file(&root.join(script_relative))?;
    let script_sha256 = sha256_hex(&script_bytes);
    if field_str(&executor_spec, "script_sha256")? != script_sha256
        || field_str(&result, "executor_script_sha256")? != script_sha256
    {
        return Err("current executor script does not match the A1C result".into());
    }

    if field_str(&result, "cobaya_version")? != COBAYA_VERSION
        || field_str(&result, "cobaya_source_commit")? != COBAYA_COMMIT
        || field_str(&result, "installed_likelihood_yaml_sha256")? != LIKELIHOOD_YAML_SHA256
        || field_str(&result, "dataset_mean_sha256")? != MEAN_SHA256
        || field_str(&result, "dataset_covariance_sha256")? != COV_SHA256
        || !exact_f64(field_f64(&result, "omega_m")?, OMEGA_M)
        || !exact_f64(field_f64(&result, "h_r_d_mpc")?, H_R_D_MPC)
        || !exact_f64(field_f64(&result, "reference_chi2_bao")?, REFERENCE_CHI2)
        || !exact_f64(field_f64(&result, "absolute_tolerance")?, ABSOLUTE_TOLERANCE)
        || field_u64(&result, "likelihood_call_budget")? != 1
        || field_u64(&result, "likelihood_call_count")? != 1
    {
        return Err("A1C result fixed subject/software identity drifted".into());
    }

    if field_str(&result, "a0_receipt_sha256")? != field_str(&a1q, "a0_receipt_sha256")?
        || field_str(&result, "environment_receipt_sha256")?
            != field_str(&a1e, "qualification_receipt_sha256")?
        || field_str(&result, "environment_versions_sha256")? != field_str(&a1e, "versions_sha256")?
        || field_str(&result, "point_manifest_sha256")?
            != field_str(&contract, "a1r_manifest_sha256")?
    {
        return Err("A1C result transitive evidence bindings diverged".into());
    }

    let a1q_reproduction = field_str(&a1q, "reproduction_verdict")?;
    if !valid_lane_verdict(a1q_reproduction)
        || field_str(&authorization, "a1q_reproduction_verdict")? != a1q_reproduction
        || field_str(&result, "a1q_reproduction_verdict")? != a1q_reproduction
    {
        return Err("A1Q reproduction verdict is inconsistent across the A1C lineage".into());
    }

    let logp = field_f64(&result, "logp_bao")?;
    let chi2 = field_f64(&result, "chi2_bao")?;
    let stored_delta = field_f64(&result, "absolute_delta_chi2")?;
    let derived_delta = (chi2 - REFERENCE_CHI2).abs();
    if !exact_f64(chi2, -2.0 * logp)
        || !exact_f64(stored_delta, derived_delta)
        || field_str(&result, "verdict")?
            != rederive_verdict(chi2, REFERENCE_CHI2, ABSOLUTE_TOLERANCE)
    {
        return Err("A1C stored chi2/delta/verdict does not rederive exactly".into());
    }

    let predictions = result
        .get("prediction_vector")
        .and_then(Value::as_array)
        .ok_or_else(|| "A1C prediction_vector is not an array".to_owned())?;
    if predictions.len() != ROW_COUNT
        || predictions
            .iter()
            .any(|value| value.as_f64().is_none_or(|number| !number.is_finite()))
    {
        return Err("A1C prediction vector is not exactly 13 finite values".into());
    }

    for key in [
        "authorization_receipt_sha256",
        "authorization_spec_sha256",
        "executor_spec_sha256",
        "executor_script_sha256",
        "a1c_manifest_sha256",
        "a1q_receipt_sha256",
        "a1e_receipt_sha256",
        "a1c_contract_receipt_sha256",
        "a0_receipt_sha256",
        "environment_receipt_sha256",
        "environment_versions_sha256",
        "point_manifest_sha256",
        "installed_likelihood_yaml_sha256",
        "installed_likelihood_class_source_sha256",
        "dataset_mean_sha256",
        "dataset_covariance_sha256",
        "prediction_vector_sha256",
    ] {
        validate_hash(&result, key)?;
    }

    Ok(QualificationReceipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: SCIENTIFIC_CLAIM,
        authority: AUTHORITY,
        a1x_comparison_authorized: true,
        a2_execution_authorized: false,
        subject_head: head,
        subject_tree: tree,
        qualification_spec_sha256: sha256_hex(&spec_bytes),
        a1c_result_sha256: sha256_hex(&result_bytes),
        authorization_receipt_sha256: authorization_sha256,
        a1q_receipt_sha256: a1q_sha256,
        a1e_receipt_sha256: a1e_sha256,
        a1c_contract_receipt_sha256: contract_sha256,
        executor_spec_sha256: current_executor_spec_sha256,
        executor_script_sha256: script_sha256,
        a1c_manifest_sha256: current_manifest_sha256,
        a0_receipt_sha256: field_str(&result, "a0_receipt_sha256")?.to_owned(),
        environment_receipt_sha256: field_str(&result, "environment_receipt_sha256")?.to_owned(),
        point_manifest_sha256: field_str(&result, "point_manifest_sha256")?.to_owned(),
        reproduction_verdict: field_str(&result, "verdict")?.to_owned(),
        chi2_bao: chi2,
        prediction_vector_sha256: field_str(&result, "prediction_vector_sha256")?.to_owned(),
        likelihood_call_count: 1,
    })
}

fn write_json<T: Serialize>(value: &T) -> Result<(), String> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer_pretty(&mut out, value)
        .map_err(|error| format!("failed to serialize qualification receipt: {error}"))?;
    writeln!(out).map_err(|error| format!("failed to write receipt: {error}"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        let invalid = InvalidReceipt {
            protocol: PROTOCOL,
            verdict: "INVALID",
            scientific_claim: SCIENTIFIC_CLAIM,
            authority: AUTHORITY,
            a1x_comparison_authorized: false,
            a2_execution_authorized: false,
            error: "usage: de001a-a1c-result-qualify RESULT.json AUTHORIZATION.json A1Q.json A1E.json A1C-CONTRACT.json".into(),
        };
        let _ = write_json(&invalid);
        std::process::exit(2);
    }

    match execute(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        Path::new(&args[5]),
    ) {
        Ok(receipt) => {
            if let Err(error) = write_json(&receipt) {
                eprintln!("{error}");
                std::process::exit(2);
            }
        }
        Err(error) => {
            let invalid = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: SCIENTIFIC_CLAIM,
                authority: AUTHORITY,
                a1x_comparison_authorized: false,
                a2_execution_authorized: false,
                error,
            };
            let _ = write_json(&invalid);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_reproduction_results_are_both_qualifiable() {
        assert!(valid_lane_verdict("PASS"));
        assert!(valid_lane_verdict("NEGATIVE"));
        assert!(!valid_lane_verdict("INVALID"));
    }

    #[test]
    fn frozen_reproduction_boundary_is_inclusive() {
        assert_eq!(rederive_verdict(REFERENCE_CHI2, REFERENCE_CHI2, ABSOLUTE_TOLERANCE), "PASS");
        assert_eq!(
            rederive_verdict(REFERENCE_CHI2 + ABSOLUTE_TOLERANCE, REFERENCE_CHI2, ABSOLUTE_TOLERANCE),
            "PASS"
        );
        assert_eq!(
            rederive_verdict(REFERENCE_CHI2 + ABSOLUTE_TOLERANCE + 1.0e-6, REFERENCE_CHI2, ABSOLUTE_TOLERANCE),
            "NEGATIVE"
        );
    }
}
