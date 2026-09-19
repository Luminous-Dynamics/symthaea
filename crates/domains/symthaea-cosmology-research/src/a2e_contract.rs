// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A2E raw optimizer execution contract verifier.
//!
//! This program performs no cosmology, likelihood evaluation, sampler
//! construction, minimization, or optimization. It validates that a qualified
//! A2 authorization receipt is eligible for a future exact raw optimizer
//! executor under a frozen execution-evidence contract.

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
const PROTOCOL: &str = "DE-001A2E-EXECUTION-CONTRACT-v1";
const AUTHORITY: &str = "raw-optimizer-execution-contract-only";
const A2_AUTH_PROTOCOL: &str = "DE-001A2-EXECUTION-AUTHORIZATION-v1";
const A2_AUTH_AUTHORITY: &str = "exact-optimizer-execution-authorization-only";
const RAW_RESULT_PROTOCOL: &str = "DE-001A2E-RAW-OPTIMIZER-EXECUTION-v1";
const CONTRACT_SPEC_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_a2e_execution_contract_v1.json";
const AUTH_SPEC_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_a2_authorization_v1.json";

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    optimizer_execution_contract_valid: bool,
    optimizer_execution_authorized_from_upstream: bool,
    executor_implementation_authorized: bool,
    reproduction_verdict: &'static str,
    a2q_execution_authorized: bool,
    contract_head: String,
    contract_tree: String,
    scientific_subject_head: String,
    scientific_subject_tree: String,
    contract_spec_sha256: String,
    authorization_spec_sha256: String,
    authorization_receipt_sha256: String,
    effective_sampler_sha256: String,
    sampled_coordinate_order: Vec<String>,
    optimizer_invocation_budget: u64,
    required_raw_result_protocol: &'static str,
    automatic_retry_allowed: bool,
    normalization_or_migration_allowed: bool,
    network_allowed: bool,
    runtime_package_installation_allowed: bool,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    optimizer_execution_contract_valid: bool,
    executor_implementation_authorized: bool,
    reproduction_verdict: &'static str,
    a2q_execution_authorized: bool,
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

fn is_git_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
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
        return Err(format!("{}: file exceeds {MAX_JSON_BYTES} bytes", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
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
    if root.is_empty() || !is_git_oid(&head) || !is_git_oid(&tree) {
        return Err("invalid contract Git identity".into());
    }
    Ok((PathBuf::from(root), head, tree))
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

fn nested<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing nested field: {key}"))
}

fn require_exact_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let required: BTreeSet<_> = expected.iter().copied().collect();
    if actual != required {
        return Err(format!("{label} field set differs from the frozen schema"));
    }
    Ok(())
}

fn ordered_strings(value: &Value, label: &str) -> Result<Vec<String>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    if array.is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    let mut result = Vec::with_capacity(array.len());
    let mut seen = BTreeSet::new();
    for item in array {
        let text = item
            .as_str()
            .filter(|text| !text.is_empty())
            .ok_or_else(|| format!("{label} contains a non-string or empty item"))?;
        if !seen.insert(text) {
            return Err(format!("{label} contains duplicate item {text}"));
        }
        result.push(text.to_owned());
    }
    Ok(result)
}

fn expect_false(value: &Value, key: &str, label: &str) -> Result<(), String> {
    if field_bool(value, key)? {
        return Err(format!("{label}.{key} must remain false"));
    }
    Ok(())
}

fn expect_true(value: &Value, key: &str, label: &str) -> Result<(), String> {
    if !field_bool(value, key)? {
        return Err(format!("{label}.{key} must remain true"));
    }
    Ok(())
}

fn validate_spec(spec: &Value) -> Result<(), String> {
    if spec.get("schema_version").and_then(Value::as_u64) != Some(1)
        || field_str(spec, "protocol")? != PROTOCOL
        || field_str(spec, "status")? != "preregistered-executor-contract-only"
        || field_str(spec, "scientific_claim")? != "NONE"
        || field_str(spec, "authority")? != AUTHORITY
    {
        return Err("A2E execution-contract identity drifted".into());
    }

    let authorization = nested(spec, "required_authorization")?;
    if field_str(authorization, "protocol")? != A2_AUTH_PROTOCOL
        || field_str(authorization, "verdict")? != "PASS"
        || field_str(authorization, "authority")? != A2_AUTH_AUTHORITY
        || field_u64(authorization, "optimizer_invocation_budget")? != 1
    {
        return Err("A2E required authorization drifted".into());
    }
    for key in [
        "optimizer_execution_authorized",
        "a2_execution_authorized",
        "exact_effective_sampler_required",
        "exact_sampled_coordinate_order_required",
    ] {
        expect_true(authorization, key, "required_authorization")?;
    }
    for key in [
        "normalization_or_migration_authorized",
        "backend_substitution_allowed",
        "method_substitution_allowed",
        "option_rewrite_allowed",
        "network_allowed",
        "runtime_package_installation_allowed",
    ] {
        expect_false(authorization, key, "required_authorization")?;
    }

    let policy = nested(spec, "execution_policy")?;
    if field_u64(policy, "optimizer_invocations_per_executor_process")? != 1 {
        return Err("A2E invocation budget drifted".into());
    }
    for key in [
        "retry_requires_new_authorization_receipt",
        "exact_effective_sampler_sha256_required",
        "exact_sampled_coordinate_order_required",
        "exact_authenticated_configuration_bytes_required",
    ] {
        expect_true(policy, key, "execution_policy")?;
    }
    for key in [
        "preflight_optimizer_invocation_allowed",
        "dry_run_optimizer_invocation_allowed",
        "automatic_retry_allowed",
        "configuration_mutation_allowed",
        "backend_substitution_allowed",
        "method_substitution_allowed",
        "option_addition_allowed",
        "option_deletion_allowed",
        "option_rewrite_allowed",
        "coordinate_addition_allowed",
        "coordinate_deletion_allowed",
        "coordinate_reordering_allowed",
        "normalization_or_migration_allowed",
        "network_allowed",
        "runtime_package_installation_allowed",
    ] {
        expect_false(policy, key, "execution_policy")?;
    }

    let raw = nested(spec, "required_raw_result")?;
    if field_str(raw, "protocol")? != RAW_RESULT_PROTOCOL
        || field_str(raw, "reproduction_verdict")? != "UNASSESSED"
    {
        return Err("A2E raw-result identity drifted".into());
    }
    let states = nested(raw, "allowed_execution_states")?
        .as_array()
        .ok_or_else(|| "allowed_execution_states must be an array".to_owned())?;
    let expected = ["EXECUTED", "EXECUTION_ERROR", "INVALID"];
    if states.len() != expected.len()
        || states
            .iter()
            .zip(expected)
            .any(|(actual, expected)| actual.as_str() != Some(expected))
    {
        return Err("A2E raw execution-state vocabulary drifted".into());
    }
    for key in [
        "must_bind_authorization_receipt_sha256",
        "must_bind_execution_contract_receipt_sha256",
        "must_bind_effective_sampler_sha256",
        "must_bind_sampled_coordinate_order",
        "must_record_actual_optimizer_invocation_count",
        "must_record_process_exit_code",
        "must_record_command_argv_sha256",
        "must_record_stdout_sha256",
        "must_record_stderr_sha256",
        "must_record_result_file_manifest",
        "must_record_postflight_identity",
    ] {
        expect_true(raw, key, "required_raw_result")?;
    }

    let boundary = nested(spec, "interpretation_boundary")?;
    for key in [
        "executor_may_classify_reproduction",
        "executor_may_change_historical_tolerances",
        "executor_may_select_success_metric",
        "a2q_execution_authorized",
    ] {
        expect_false(boundary, key, "interpretation_boundary")?;
    }
    Ok(())
}

fn validate_authorization(
    value: &Value,
    expected_authorization_spec_sha256: &str,
) -> Result<(String, String, String, Vec<String>), String> {
    require_exact_keys(
        value,
        &[
            "protocol",
            "verdict",
            "scientific_claim",
            "authority",
            "optimizer_execution_authorized",
            "a2_execution_authorized",
            "normalization_or_migration_authorized",
            "authorizer_head",
            "authorizer_tree",
            "scientific_subject_head",
            "scientific_subject_tree",
            "authorization_spec_sha256",
            "a1x_receipt_sha256",
            "a1x_capsule_sha256",
            "a2m_receipt_sha256",
            "a2m_capsule_sha256",
            "a2r_qualification_capsule_sha256",
            "effective_sampler_sha256",
            "sampled_coordinate_order",
            "optimizer_invocation_budget",
            "exact_effective_sampler_required",
            "exact_sampled_coordinate_order_required",
            "backend_substitution_allowed",
            "method_substitution_allowed",
            "option_rewrite_allowed",
            "network_allowed",
            "runtime_package_installation_allowed",
            "authorization_scope",
            "stateful_global_single_use_claimed",
        ],
        "A2 authorization receipt",
    )?;
    if field_str(value, "protocol")? != A2_AUTH_PROTOCOL
        || field_str(value, "verdict")? != "PASS"
        || field_str(value, "scientific_claim")? != "NONE"
        || field_str(value, "authority")? != A2_AUTH_AUTHORITY
        || field_str(value, "authorization_spec_sha256")? != expected_authorization_spec_sha256
        || field_u64(value, "optimizer_invocation_budget")? != 1
        || field_str(value, "authorization_scope")?
            != "one-exact-optimizer-invocation-per-executor-process"
    {
        return Err("A2 authorization receipt identity or scope drifted".into());
    }
    for key in [
        "optimizer_execution_authorized",
        "a2_execution_authorized",
        "exact_effective_sampler_required",
        "exact_sampled_coordinate_order_required",
    ] {
        expect_true(value, key, "A2 authorization receipt")?;
    }
    for key in [
        "normalization_or_migration_authorized",
        "backend_substitution_allowed",
        "method_substitution_allowed",
        "option_rewrite_allowed",
        "network_allowed",
        "runtime_package_installation_allowed",
        "stateful_global_single_use_claimed",
    ] {
        expect_false(value, key, "A2 authorization receipt")?;
    }

    let authorizer_head = field_str(value, "authorizer_head")?;
    let authorizer_tree = field_str(value, "authorizer_tree")?;
    let subject_head = field_str(value, "scientific_subject_head")?;
    let subject_tree = field_str(value, "scientific_subject_tree")?;
    if !is_git_oid(authorizer_head)
        || !is_git_oid(authorizer_tree)
        || !is_git_oid(subject_head)
        || !is_git_oid(subject_tree)
    {
        return Err("A2 authorization receipt contains invalid Git identity".into());
    }
    for key in [
        "authorization_spec_sha256",
        "a1x_receipt_sha256",
        "a1x_capsule_sha256",
        "a2m_receipt_sha256",
        "a2m_capsule_sha256",
        "a2r_qualification_capsule_sha256",
        "effective_sampler_sha256",
    ] {
        if !is_sha256(field_str(value, key)?) {
            return Err(format!("A2 authorization receipt has invalid SHA-256 field {key}"));
        }
    }
    let coordinates = ordered_strings(nested(value, "sampled_coordinate_order")?, "sampled_coordinate_order")?;
    Ok((
        subject_head.to_owned(),
        subject_tree.to_owned(),
        field_str(value, "effective_sampler_sha256")?.to_owned(),
        coordinates,
    ))
}

fn execute(authorization_path: &Path) -> Result<Receipt, String> {
    let (root, contract_head, contract_tree) = repository_identity()?;
    let contract_spec_bytes = read_regular_file(&root.join(CONTRACT_SPEC_RELATIVE))?;
    let contract_spec: Value = serde_json::from_slice(&contract_spec_bytes)
        .map_err(|error| format!("invalid A2E execution-contract spec JSON: {error}"))?;
    validate_spec(&contract_spec)?;

    let authorization_spec_bytes = read_regular_file(&root.join(AUTH_SPEC_RELATIVE))?;
    let expected_authorization_spec_sha256 = sha256_hex(&authorization_spec_bytes);

    let authorization_bytes = read_regular_file(authorization_path)?;
    let authorization: Value = serde_json::from_slice(&authorization_bytes)
        .map_err(|error| format!("invalid A2 authorization JSON: {error}"))?;
    let (subject_head, subject_tree, sampler_sha, coordinates) =
        validate_authorization(&authorization, &expected_authorization_spec_sha256)?;

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: "NONE",
        authority: AUTHORITY,
        optimizer_execution_contract_valid: true,
        optimizer_execution_authorized_from_upstream: true,
        executor_implementation_authorized: false,
        reproduction_verdict: "UNASSESSED",
        a2q_execution_authorized: false,
        contract_head,
        contract_tree,
        scientific_subject_head: subject_head,
        scientific_subject_tree: subject_tree,
        contract_spec_sha256: sha256_hex(&contract_spec_bytes),
        authorization_spec_sha256: expected_authorization_spec_sha256,
        authorization_receipt_sha256: sha256_hex(&authorization_bytes),
        effective_sampler_sha256: sampler_sha,
        sampled_coordinate_order: coordinates,
        optimizer_invocation_budget: 1,
        required_raw_result_protocol: RAW_RESULT_PROTOCOL,
        automatic_retry_allowed: false,
        normalization_or_migration_allowed: false,
        network_allowed: false,
        runtime_package_installation_allowed: false,
    })
}

fn write_json<T: Serialize>(value: &T) -> Result<(), String> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer(&mut lock, value)
        .map_err(|error| format!("failed to serialize receipt: {error}"))?;
    lock.write_all(b"\n")
        .map_err(|error| format!("failed to write receipt: {error}"))
}

fn main() {
    let args: Vec<_> = env::args_os().collect();
    let result = if args.len() == 2 {
        execute(Path::new(&args[1]))
    } else {
        Err("usage: de001a-a2e-contract <a2-authorization.json>".into())
    };

    match result {
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
                scientific_claim: "NONE",
                authority: AUTHORITY,
                optimizer_execution_contract_valid: false,
                executor_implementation_authorized: false,
                reproduction_verdict: "UNASSESSED",
                a2q_execution_authorized: false,
                error,
            };
            if let Err(write_error) = write_json(&invalid) {
                eprintln!("{write_error}");
            }
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{is_git_oid, is_sha256};

    #[test]
    fn identity_helpers_reject_malformed_values() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256(&"A".repeat(64)));
        assert!(is_git_oid(&"b".repeat(40)));
        assert!(!is_git_oid("not-a-git-object"));
    }
}
