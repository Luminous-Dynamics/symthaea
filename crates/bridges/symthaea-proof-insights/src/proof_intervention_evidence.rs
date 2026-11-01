// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict, identity-bound receipt validation for paired proof interventions.
//! Benchmark semantics and reviewer signatures are verified by benchmark_integrity.rs.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy)]
pub struct ExpectedIdentity<'a> {
    pub trial_id: &'a str,
    pub pair_id: &'a str,
    pub arm_id: &'a str,
    pub run_id: &'a str,
    pub task_id: &'a str,
    pub component_id: &'a str,
    pub repository_commit: &'a str,
    pub lean_toolchain: &'a str,
    pub environment_sha256: &'a str,
    pub initial_proof_state_sha256: &'a str,
    pub target_statement_sha256: &'a str,
    pub model_seed_sha256: &'a str,
    pub evaluation_horizon_id: &'a str,
    pub continuation_policy_sha256: &'a str,
    pub execution_order_index: u64,
}

const COMMON_FIELDS: [&str; 14] = [
    "evidence_class",
    "evidence_kind",
    "trial_id",
    "pair_id",
    "arm_id",
    "run_id",
    "task_id",
    "component_id",
    "repository_commit",
    "lean_toolchain",
    "environment_sha256",
    "initial_proof_state_sha256",
    "target_statement_sha256",
    "model_seed_sha256",
];

fn get_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("receipt field {key} must be a string"))
}

fn get_u64(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("receipt field {key} must be a nonnegative integer"))
}

fn get_bool(value: &Value, key: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .with_context(|| format!("receipt field {key} must be boolean"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn check_sha256(value: &str, label: &str) -> Result<()> {
    if !is_sha256(value) {
        bail!("{label} must be a 64-character lowercase SHA-256 digest");
    }
    Ok(())
}

fn exact_keys(value: &Value, expected: &[&str], label: &str) -> Result<()> {
    let object = value
        .as_object()
        .with_context(|| format!("{label} must be a JSON object"))?;
    let expected_set: std::collections::BTreeSet<&str> = expected.iter().copied().collect();
    if object.len() != expected_set.len()
        || object.keys().any(|key| !expected_set.contains(key.as_str()))
    {
        bail!("{label} has missing or unknown fields");
    }
    Ok(())
}

fn parse_receipt(bytes: &[u8], expected_sha256: &str, label: &str) -> Result<Value> {
    check_sha256(expected_sha256, label)?;
    let actual = hex::encode(Sha256::digest(bytes));
    if actual != expected_sha256 {
        bail!("{label} bytes do not match the paired outcome SHA-256");
    }
    serde_json::from_slice(bytes).with_context(|| format!("parse {label} JSON"))
}

fn validate_common(
    value: &Value,
    schema_version: &str,
    evidence_kind: &str,
    expected: ExpectedIdentity<'_>,
) -> Result<()> {
    if get_str(value, "schema_version")? != schema_version
        || get_str(value, "evidence_class")? != "measured_evidence"
        || get_str(value, "evidence_kind")? != evidence_kind
    {
        bail!("receipt schema/evidence identity is not the measured {evidence_kind} v0 contract");
    }
    let expected_pairs = [
        ("trial_id", expected.trial_id),
        ("pair_id", expected.pair_id),
        ("arm_id", expected.arm_id),
        ("run_id", expected.run_id),
        ("task_id", expected.task_id),
        ("component_id", expected.component_id),
        ("repository_commit", expected.repository_commit),
        ("lean_toolchain", expected.lean_toolchain),
        ("environment_sha256", expected.environment_sha256),
        ("initial_proof_state_sha256", expected.initial_proof_state_sha256),
        ("target_statement_sha256", expected.target_statement_sha256),
        ("model_seed_sha256", expected.model_seed_sha256),
    ];
    for (key, expected_value) in expected_pairs {
        if get_str(value, key)? != expected_value {
            bail!("{evidence_kind} receipt does not bind exact {key}");
        }
    }
    Ok(())
}

fn parse_sorted_string_list(value: &Value, key: &str) -> Result<Vec<String>> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("receipt field {key} must be an array"))?;
    let mut result = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let name = item
            .as_str()
            .with_context(|| format!("{key}[{index}] must be a string"))?;
        if name.trim().is_empty() {
            bail!("{key}[{index}] cannot be empty");
        }
        result.push(name.to_owned());
    }
    if result.windows(2).any(|pair| pair[0] >= pair[1]) {
        bail!("{key} must be sorted and unique");
    }
    Ok(result)
}

fn digest_string_list(values: &[String]) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(values)?)))
}

pub fn verify_checker_receipt(
    bytes: &[u8],
    expected_receipt_sha256: &str,
    expected: ExpectedIdentity<'_>,
    outcome_status: &str,
    proof_sha256: Option<&str>,
    checked_target_sha256: Option<&str>,
) -> Result<()> {
    const KEYS: [&str; 25] = [
        "schema_version",
        "evidence_class",
        "evidence_kind",
        "trial_id",
        "pair_id",
        "arm_id",
        "run_id",
        "task_id",
        "component_id",
        "repository_commit",
        "lean_toolchain",
        "environment_sha256",
        "initial_proof_state_sha256",
        "target_statement_sha256",
        "model_seed_sha256",
        "checker_id",
        "checker_version",
        "checker_binary_sha256",
        "status",
        "kernel_passed",
        "checked_target_statement_sha256",
        "proof_sha256",
        "proof_contains_sorry_ax",
        "used_axioms",
        "used_axioms_manifest_sha256",
    ];
    const ALLOWLIST_FIELDS: [&str; 2] = ["allowed_axioms", "allowlist_manifest_sha256"];
    const EXTRA_FIELDS: [&str; 1] = ["unexpected_axioms"];
    let value = parse_receipt(bytes, expected_receipt_sha256, "checker receipt")?;
    let all_fields: Vec<&str> = KEYS
        .into_iter()
        .chain(ALLOWLIST_FIELDS)
        .chain(EXTRA_FIELDS)
        .collect();
    exact_keys(&value, &all_fields, "checker receipt")?;
    validate_common(
        &value,
        "pitb.proof-intervention-checker-receipt.v0",
        "checker",
        expected,
    )?;
    if get_str(&value, "checker_id")?.trim().is_empty()
        || get_str(&value, "checker_version")?.trim().is_empty()
    {
        bail!("checker identity/version cannot be empty");
    }
    check_sha256(get_str(&value, "checker_binary_sha256")?, "checker binary")?;
    let used = parse_sorted_string_list(&value, "used_axioms")?;
    let allowed = parse_sorted_string_list(&value, "allowed_axioms")?;
    if digest_string_list(&used)? != get_str(&value, "used_axioms_manifest_sha256")?
        || digest_string_list(&allowed)? != get_str(&value, "allowlist_manifest_sha256")?
    {
        bail!("checker receipt axiom list digest mismatch");
    }
    let allowed_set: std::collections::BTreeSet<&str> =
        allowed.iter().map(String::as_str).collect();
    let unexpected = used
        .iter()
        .filter(|axiom| !allowed_set.contains(axiom.as_str()))
        .count() as u64;
    if unexpected != get_u64(&value, "unexpected_axioms")? {
        bail!("checker receipt unexpected_axioms count does not reconcile");
    }

    let status = get_str(&value, "status")?;
    if !matches!(status, "passed" | "failed" | "not_run" | "unknown") {
        bail!("checker receipt contains unknown checker status");
    }
    let kernel_passed = get_bool(&value, "kernel_passed")?;
    let no_sorry = !get_bool(&value, "proof_contains_sorry_ax")?;
    let checked = match value.get("checked_target_statement_sha256") {
        Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        _ => bail!("checker checked-target digest must be a digest or null"),
    };
    let proof = match value.get("proof_sha256") {
        Some(Value::Null) => None,
        Some(Value::String(value)) => {
            check_sha256(value, "checker proof")?;
            Some(value.as_str())
        }
        _ => bail!("checker proof digest must be a digest or null"),
    };
    if outcome_status == "verified_success" {
        if status != "passed"
            || !kernel_passed
            || !no_sorry
            || unexpected != 0
            || checked != Some(expected.target_statement_sha256)
            || proof != proof_sha256
            || proof_sha256.is_none()
            || checked_target_sha256 != Some(expected.target_statement_sha256)
        {
            bail!("checker receipt does not validate the exact proof/target with the registered axiom policy");
        }
    } else if status == "passed"
        && (!kernel_passed || !no_sorry || unexpected != 0 || checked != Some(expected.target_statement_sha256))
    {
        bail!("checker receipt claims a pass without a clean exact-target kernel result");
    }
    Ok(())
}

pub fn verify_run_receipt(
    bytes: &[u8],
    expected_receipt_sha256: &str,
    expected: ExpectedIdentity<'_>,
    outcome: &Value,
    planned_budget: &Value,
) -> Result<()> {
    const KEYS: [&str; 28] = [
        "schema_version",
        "evidence_class",
        "evidence_kind",
        "trial_id",
        "pair_id",
        "arm_id",
        "run_id",
        "task_id",
        "component_id",
        "repository_commit",
        "lean_toolchain",
        "environment_sha256",
        "initial_proof_state_sha256",
        "target_statement_sha256",
        "model_seed_sha256",
        "evaluation_horizon_id",
        "continuation_policy_sha256",
        "execution_order_index",
        "status",
        "actual_resources",
        "resource_budget",
        "proof_sha256",
        "checker_receipt_sha256",
        "semantic_gate_receipt_sha256",
        "successor_receipt_sha256",
        "run_config_sha256",
        "started_from_fresh_process",
        "mutable_caches_isolated",
    ];
    let value = parse_receipt(bytes, expected_receipt_sha256, "run receipt")?;
    exact_keys(&value, &KEYS, "run receipt")?;
    validate_common(
        &value,
        "pitb.proof-intervention-run-receipt.v0",
        "run",
        expected,
    )?;
    if get_str(&value, "evaluation_horizon_id")? != expected.evaluation_horizon_id
        || get_str(&value, "continuation_policy_sha256")? != expected.continuation_policy_sha256
        || get_u64(&value, "execution_order_index")? != expected.execution_order_index
    {
        bail!("run receipt horizon, continuation policy or execution order differs from the paired plan");
    }
    check_sha256(get_str(&value, "run_config_sha256")?, "run configuration")?;
    if !get_bool(&value, "started_from_fresh_process")?
        || !get_bool(&value, "mutable_caches_isolated")?
    {
        bail!("run receipt did not isolate the Lean process and mutable caches");
    }
    if get_str(&value, "status")? != get_str(outcome, "/status")? {
        bail!("run receipt status differs from the outcome row");
    }
    let actual = value.get("actual_resources").context("run receipt actual_resources missing")?;
    let row_actual = outcome.get("actual_resources").context("outcome actual_resources missing")?;
    let budget = value.get("resource_budget").context("run receipt resource_budget missing")?;
    if actual != row_actual || budget != planned_budget {
        bail!("run receipt resource accounting differs from the measured outcome or frozen budget");
    }
    for (receipt_key, outcome_pointer) in [
        ("proof_sha256", "/proof_sha256"),
        ("checker_receipt_sha256", "/checker_receipt_sha256"),
        ("semantic_gate_receipt_sha256", "/semantic_gate_receipt_sha256"),
        ("successor_receipt_sha256", "/successor_gate_receipt_sha256"),
    ] {
        let receipt_value = value.get(receipt_key).context("run receipt reference missing")?;
        let outcome_value = outcome
            .pointer(outcome_pointer)
            .with_context(|| format!("outcome reference {outcome_pointer} missing"))?;
        if receipt_value != outcome_value {
            bail!("run receipt {receipt_key} differs from outcome row");
        }
    }
    Ok(())
}

pub fn verify_successor_receipt(
    bytes: &[u8],
    expected_receipt_sha256: &str,
    expected: ExpectedIdentity<'_>,
    proof_sha256: &str,
) -> Result<()> {
    const KEYS: [&str; 20] = [
        "schema_version",
        "evidence_class",
        "evidence_kind",
        "trial_id",
        "pair_id",
        "arm_id",
        "run_id",
        "task_id",
        "component_id",
        "repository_commit",
        "lean_toolchain",
        "environment_sha256",
        "initial_proof_state_sha256",
        "target_statement_sha256",
        "model_seed_sha256",
        "status",
        "proof_sha256",
        "successor_suite_id",
        "successor_suite_sha256",
        "passed_check_ids",
    ];
    const FAILED: [&str; 1] = ["failed_check_ids"];
    let value = parse_receipt(bytes, expected_receipt_sha256, "successor receipt")?;
    let all_fields: Vec<&str> = KEYS.into_iter().chain(FAILED).collect();
    exact_keys(&value, &all_fields, "successor receipt")?;
    validate_common(
        &value,
        "pitb.proof-intervention-successor-receipt.v0",
        "successor",
        expected,
    )?;
    if get_str(&value, "status")? != "passed"
        || get_str(&value, "proof_sha256")? != proof_sha256
    {
        bail!("successor receipt does not pass for the exact accepted proof");
    }
    if get_str(&value, "successor_suite_id")?.trim().is_empty() {
        bail!("successor receipt suite identity cannot be empty");
    }
    check_sha256(get_str(&value, "successor_suite_sha256")?, "successor suite")?;
    let passed = parse_sorted_string_list(&value, "passed_check_ids")?;
    let failed = parse_sorted_string_list(&value, "failed_check_ids")?;
    if passed.is_empty() || !failed.is_empty() {
        bail!("successor receipt requires passed checks and zero failed checks");
    }
    Ok(())
}
