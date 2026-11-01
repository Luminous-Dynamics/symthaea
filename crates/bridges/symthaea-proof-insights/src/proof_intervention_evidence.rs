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
    pub initial_state_receipt_sha256: &'a str,
    pub target_statement_sha256: &'a str,
    pub model_seed_sha256: &'a str,
    pub model_id: &'a str,
    pub model_version: &'a str,
    pub arm_policy_sha256: &'a str,
    pub arm_config_sha256: &'a str,
    pub evaluation_horizon_id: &'a str,
    pub continuation_policy_sha256: &'a str,
    pub execution_order_index: u64,
}


fn member<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    if key.starts_with('/') {
        value.pointer(key)
    } else {
        value.get(key)
    }
}

fn get_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    member(value, key)
        .and_then(Value::as_str)
        .with_context(|| format!("receipt field {key} must be a string"))
}

fn get_u64(value: &Value, key: &str) -> Result<u64> {
    member(value, key)
        .and_then(Value::as_u64)
        .with_context(|| format!("receipt field {key} must be a nonnegative integer"))
}

fn get_bool(value: &Value, key: &str) -> Result<bool> {
    member(value, key)
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
        ("initial_state_receipt_sha256", expected.initial_state_receipt_sha256),
        ("target_statement_sha256", expected.target_statement_sha256),
        ("model_seed_sha256", expected.model_seed_sha256),
        ("model_id", expected.model_id),
        ("model_version", expected.model_version),
        ("arm_policy_sha256", expected.arm_policy_sha256),
        ("arm_config_sha256", expected.arm_config_sha256),
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
    const KEYS: [&str; 30] = [
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
        "initial_state_receipt_sha256",
        "target_statement_sha256",
        "model_seed_sha256",
        "model_id",
        "model_version",
        "arm_policy_sha256",
        "arm_config_sha256",
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
    const KEYS: [&str; 33] = [
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
        "initial_state_receipt_sha256",
        "target_statement_sha256",
        "model_seed_sha256",
        "model_id",
        "model_version",
        "arm_policy_sha256",
        "arm_config_sha256",
        "evaluation_horizon_id",
        "continuation_policy_sha256",
        "execution_order_index",
        "status",
        "actual_resources",
        "resource_budget",
        "proof_sha256",
        "checker_receipt_sha256",
        "target_integrity_receipt_sha256",
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
    if get_str(&value, "run_config_sha256")? != expected.arm_config_sha256 {
        bail!("run receipt configuration digest differs from the frozen arm config");
    }
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
        ("target_integrity_receipt_sha256", "/target_integrity_receipt_sha256"),
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
        "initial_state_receipt_sha256",
        "target_statement_sha256",
        "model_seed_sha256",
        "model_id",
        "model_version",
        "arm_policy_sha256",
        "arm_config_sha256",
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const SHA_G: &str = "abababababababababababababababababababababababababababababababab";
    const SHA_H: &str = "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";
    const SHA_I: &str = "edededededededededededededededededededededededededededededededed";
    const TASK: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const COMMIT: &str = "2222222222222222222222222222222222222222";
    const COMPONENT: &str = "project-test";

    fn identity() -> ExpectedIdentity<'static> {
        ExpectedIdentity {
            trial_id: "trial-test",
            pair_id: SHA_A,
            arm_id: "arm-test",
            run_id: "run-test",
            task_id: TASK,
            component_id: COMPONENT,
            repository_commit: COMMIT,
            lean_toolchain: "leanprover/lean4:v4.26.0",
            environment_sha256: SHA_B,
            initial_proof_state_sha256: SHA_C,
            initial_state_receipt_sha256: SHA_G,
            target_statement_sha256: SHA_D,
            model_seed_sha256: SHA_E,
            model_id: "synthetic-model",
            model_version: "synthetic-version",
            arm_policy_sha256: SHA_H,
            arm_config_sha256: SHA_I,
            evaluation_horizon_id: "horizon-test",
            continuation_policy_sha256: SHA_F,
            execution_order_index: 1,
        }
    }

    fn common(kind: &str) -> Value {
        json!({
            "evidence_class":"measured_evidence",
            "evidence_kind":kind,
            "trial_id":"trial-test",
            "pair_id":SHA_A,
            "arm_id":"arm-test",
            "run_id":"run-test",
            "task_id":TASK,
            "component_id":COMPONENT,
            "repository_commit":COMMIT,
            "lean_toolchain":"leanprover/lean4:v4.26.0",
            "environment_sha256":SHA_B,
            "initial_proof_state_sha256":SHA_C,
            "initial_state_receipt_sha256":SHA_G,
            "target_statement_sha256":SHA_D,
            "model_seed_sha256":SHA_E,
            "model_id":"synthetic-model",
            "model_version":"synthetic-version",
            "arm_policy_sha256":SHA_H,
            "arm_config_sha256":SHA_I
        })
    }

    fn digest_value(value: &Value) -> String {
        hex::encode(Sha256::digest(serde_json::to_vec(value).unwrap()))
    }

    fn checker_receipt() -> Value {
        let mut receipt = common("checker");
        receipt["schema_version"] = json!("pitb.proof-intervention-checker-receipt.v0");
        receipt["checker_id"] = json!("lean-kernel");
        receipt["checker_version"] = json!("4.26.0");
        receipt["checker_binary_sha256"] = json!(SHA_F);
        receipt["status"] = json!("passed");
        receipt["kernel_passed"] = Value::Bool(true);
        receipt["checked_target_statement_sha256"] = json!(SHA_D);
        receipt["proof_sha256"] = json!(SHA_E);
        receipt["proof_contains_sorry_ax"] = Value::Bool(false);
        receipt["used_axioms"] = json!(["propext"]);
        receipt["used_axioms_manifest_sha256"] = json!(digest_value(&json!(["propext"])));
        receipt["allowed_axioms"] = json!(["propext"]);
        receipt["allowlist_manifest_sha256"] = json!(digest_value(&json!(["propext"])));
        receipt["unexpected_axioms"] = json!(0);
        receipt
    }

    fn run_receipt(checker_sha: &str) -> Value {
        let mut receipt = common("run");
        receipt["schema_version"] = json!("pitb.proof-intervention-run-receipt.v0");
        receipt["evaluation_horizon_id"] = json!("horizon-test");
        receipt["continuation_policy_sha256"] = json!(SHA_F);
        receipt["execution_order_index"] = json!(1);
        receipt["status"] = json!("verified_success");
        let resources = json!({
            "tokens":1000,"model_calls":1,"tool_calls":2,
            "wall_ms":1000,"lean_cpu_ms":500,"cost_microusd":200
        });
        receipt["actual_resources"] = resources.clone();
        receipt["resource_budget"] = json!({
            "tokens":40000,"model_calls":8,"tool_calls":40,
            "wall_ms":300000,"lean_cpu_ms":180000,"cost_microusd":2000000
        });
        receipt["proof_sha256"] = json!(SHA_E);
        receipt["checker_receipt_sha256"] = json!(checker_sha);
        receipt["target_integrity_receipt_sha256"] = json!(SHA_A);
        receipt["successor_receipt_sha256"] = json!(SHA_B);
        receipt["run_config_sha256"] = json!(SHA_I);
        receipt["started_from_fresh_process"] = Value::Bool(true);
        receipt["mutable_caches_isolated"] = Value::Bool(true);
        receipt
    }

    fn successor_receipt() -> Value {
        let mut receipt = common("successor");
        receipt["schema_version"] = json!("pitb.proof-intervention-successor-receipt.v0");
        receipt["status"] = json!("passed");
        receipt["proof_sha256"] = json!(SHA_E);
        receipt["successor_suite_id"] = json!("suite-v0");
        receipt["successor_suite_sha256"] = json!(SHA_F);
        receipt["passed_check_ids"] = json!(["downstream-lemma-a", "downstream-lemma-b"]);
        receipt["failed_check_ids"] = json!([]);
        receipt
    }

    #[test]
    fn checker_run_and_successor_receipts_join_exact_pair_identity() {
        let identity = identity();
        let checker = checker_receipt();
        let checker_bytes = serde_json::to_vec(&checker).unwrap();
        let checker_sha = hex::encode(Sha256::digest(&checker_bytes));
        verify_checker_receipt(
            &checker_bytes,
            &checker_sha,
            identity,
            "verified_success",
            Some(SHA_E),
            Some(SHA_D),
        )
        .unwrap();

        let run = run_receipt(&checker_sha);
        let run_bytes = serde_json::to_vec(&run).unwrap();
        let run_sha = hex::encode(Sha256::digest(&run_bytes));
        let outcome = json!({
            "status":"verified_success",
            "proof_sha256":SHA_E,
            "checker_receipt_sha256":checker_sha,
            "target_integrity_receipt_sha256":SHA_A,
            "successor_gate_receipt_sha256":SHA_B,
            "actual_resources":run["actual_resources"]
        });
        let budget = json!({
            "tokens":40000,"model_calls":8,"tool_calls":40,
            "wall_ms":300000,"lean_cpu_ms":180000,"cost_microusd":2000000
        });
        verify_run_receipt(&run_bytes, &run_sha, identity, &outcome, &budget).unwrap();

        let successor = successor_receipt();
        let successor_bytes = serde_json::to_vec(&successor).unwrap();
        let successor_sha = hex::encode(Sha256::digest(&successor_bytes));
        verify_successor_receipt(&successor_bytes, &successor_sha, identity, SHA_E).unwrap();
    }

    #[test]
    fn checker_receipt_rejects_proof_substitution_even_with_new_file_digest() {
        let mut receipt = checker_receipt();
        receipt["proof_sha256"] = json!(SHA_F);
        let bytes = serde_json::to_vec(&receipt).unwrap();
        let digest = hex::encode(Sha256::digest(&bytes));
        assert!(verify_checker_receipt(
            &bytes,
            &digest,
            identity(),
            "verified_success",
            Some(SHA_E),
            Some(SHA_D),
        )
        .is_err());
    }

    #[test]
    fn run_receipt_rejects_resource_accounting_drift() {
        let checker = checker_receipt();
        let checker_sha = digest_value(&checker);
        let mut run = run_receipt(&checker_sha);
        run["actual_resources"]["tokens"] = json!(999);
        let bytes = serde_json::to_vec(&run).unwrap();
        let digest = hex::encode(Sha256::digest(&bytes));
        let outcome = json!({
            "status":"verified_success",
            "proof_sha256":SHA_E,
            "checker_receipt_sha256":checker_sha,
            "target_integrity_receipt_sha256":SHA_A,
            "successor_gate_receipt_sha256":SHA_B,
            "actual_resources":{
                "tokens":1000,"model_calls":1,"tool_calls":2,
                "wall_ms":1000,"lean_cpu_ms":500,"cost_microusd":200
            }
        });
        let budget = json!({
            "tokens":40000,"model_calls":8,"tool_calls":40,
            "wall_ms":300000,"lean_cpu_ms":180000,"cost_microusd":2000000
        });
        assert!(verify_run_receipt(&bytes, &digest, identity(), &outcome, &budget).is_err());
    }

    #[test]
    fn successor_receipt_rejects_failed_downstream_checks() {
        let mut receipt = successor_receipt();
        receipt["failed_check_ids"] = json!(["downstream-lemma-c"]);
        let bytes = serde_json::to_vec(&receipt).unwrap();
        let digest = hex::encode(Sha256::digest(&bytes));
        assert!(verify_successor_receipt(&bytes, &digest, identity(), SHA_E).is_err());
    }
}
