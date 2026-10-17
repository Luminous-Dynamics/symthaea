// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent receipt checks for the synthetic language/code smoke corpus.
//!
//! This example verifies receipt invariants and the frozen negative controls.
//! It does not run a model, compile generated candidates, or report capability
//! scores. The digest contract for result_payload is SHA-256 over compact
//! serde_json serialization; interoperability with another canonical JSON
//! implementation requires an explicitly versioned canonicalization contract.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::process;

const CORPUS: &str =
    include_str!("../fixtures/language-code-capability/smoke-v1.json");

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty_string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
}

/// Validates the receipt's claimed outcome. Returns the first fail-closed reason.
///
/// Specific tampering/negative-control conditions are checked before generic
/// required fields so each frozen control tests the intended invariant.
fn validate_receipt(receipt: &Value) -> Result<(), String> {
    let status = receipt.get("status").and_then(Value::as_str).unwrap_or("");

    let dataset_digest = receipt
        .get("dataset_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_required_dataset_digest".to_owned())?;
    if !is_lower_hex(dataset_digest, 64) {
        return Err("missing_required_dataset_digest".to_owned());
    }

    if status == "passed"
        && receipt.get("test_lane_status").and_then(Value::as_str) == Some("skipped")
    {
        return Err("skipped_lane_cannot_claim_pass".to_owned());
    }

    if let Some(expected_source) = receipt.get("expected_source_sha").and_then(Value::as_str) {
        let actual_source = receipt
            .get("source_sha")
            .and_then(Value::as_str)
            .unwrap_or("");
        if actual_source != expected_source {
            return Err("source_revision_mismatch".to_owned());
        }
    }

    if status == "passed"
        && receipt.get("test_status").and_then(Value::as_str) == Some("timeout")
    {
        return Err("timeout_cannot_claim_pass".to_owned());
    }

    if let Some(declared_digest) = receipt.get("declared_artifact_sha256") {
        let declared = declared_digest.as_str().unwrap_or("");
        let artifact = receipt
            .get("artifact_bytes")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !is_lower_hex(declared, 64) || sha256_hex(artifact.as_bytes()) != declared {
            return Err("artifact_digest_mismatch".to_owned());
        }
    }

    if status == "passed"
        && receipt.get("test_exit_code").and_then(Value::as_i64).is_some_and(|code| code != 0)
    {
        return Err("test_exit_status_conflicts_with_pass".to_owned());
    }

    let tests_executed = receipt
        .get("tests_executed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let compile_only_policy = nonempty_string(receipt, "compile_only_policy");
    if status == "passed"
        && receipt.get("claim").and_then(Value::as_str) == Some("functional_correctness")
        && tests_executed == 0
        && compile_only_policy.is_none()
    {
        return Err("functional_claim_without_tests".to_owned());
    }

    if let Some(payload) = receipt.get("result_payload") {
        let payload_status = payload.get("status").and_then(Value::as_str).unwrap_or("");
        let serialized = serde_json::to_vec(payload)
            .map_err(|_| "result_payload_digest_or_status_mismatch".to_owned())?;
        let recomputed_digest = sha256_hex(&serialized);
        let declared = receipt
            .get("declared_result_sha256")
            .and_then(Value::as_str)
            .unwrap_or("");
        if payload_status != status
            || !is_lower_hex(declared, 64)
            || recomputed_digest != declared
        {
            return Err("result_payload_digest_or_status_mismatch".to_owned());
        }
    }

    let run_id = nonempty_string(receipt, "run_id")
        .ok_or_else(|| "missing_run_id".to_owned())?;
    let _ = run_id;

    let source_sha = nonempty_string(receipt, "source_sha")
        .ok_or_else(|| "missing_source_revision".to_owned())?;
    if !is_lower_hex(source_sha, 40) {
        return Err("invalid_source_revision".to_owned());
    }

    match status {
        "passed" | "failed" | "indeterminate" => {}
        _ => return Err("invalid_status".to_owned()),
    }

    if status == "passed" {
        if receipt.get("compile_exit_code").and_then(Value::as_i64) != Some(0) {
            return Err("compile_exit_status_conflicts_with_pass".to_owned());
        }
        if receipt.get("test_exit_code").and_then(Value::as_i64) != Some(0) {
            return Err("test_exit_status_conflicts_with_pass".to_owned());
        }
        if tests_executed == 0 && compile_only_policy.is_none() {
            return Err("passed_receipt_without_tests_or_compile_only_policy".to_owned());
        }

        let artifact = receipt
            .get("artifact_bytes")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing_artifact_bytes".to_owned())?;
        let declared = receipt
            .get("declared_artifact_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing_artifact_digest".to_owned())?;
        if !is_lower_hex(declared, 64) || sha256_hex(artifact.as_bytes()) != declared {
            return Err("artifact_digest_mismatch".to_owned());
        }
    }

    Ok(())
}

fn run_negative_controls() -> Result<Value, String> {
    let corpus: Value = serde_json::from_str(CORPUS)
        .map_err(|error| format!("failed to parse smoke corpus: {error}"))?;
    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "smoke corpus has no cases array".to_owned())?;

    let mut examined = 0usize;
    let mut rejected_as_expected = 0usize;
    let mut outcomes = Vec::new();

    for case in cases
        .iter()
        .filter(|case| case.get("lane").and_then(Value::as_str) == Some("verifier_integrity"))
    {
        let id = nonempty_string(case, "id")
            .ok_or_else(|| "verifier case is missing its id".to_owned())?;
        let expected_reason = nonempty_string(case, "expected_reason")
            .ok_or_else(|| format!("{id}: missing expected_reason"))?;
        let receipt = case
            .get("receipt_fixture")
            .ok_or_else(|| format!("{id}: missing receipt_fixture"))?;

        let actual = validate_receipt(receipt);
        match actual {
            Err(ref reason) if reason == expected_reason => {
                rejected_as_expected += 1;
                outcomes.push(json!({
                    "case_id": id,
                    "result": "REJECTED_AS_EXPECTED",
                    "reason": reason
                }));
            }
            Err(reason) => {
                outcomes.push(json!({
                    "case_id": id,
                    "result": "REJECTED_WRONG_REASON",
                    "expected_reason": expected_reason,
                    "actual_reason": reason
                }));
            }
            Ok(()) => {
                outcomes.push(json!({
                    "case_id": id,
                    "result": "FALSE_PASS",
                    "expected_reason": expected_reason
                }));
            }
        }
        examined += 1;
    }

    if examined != 8 || rejected_as_expected != examined {
        return Err(format!(
            "negative-control gate failed: {rejected_as_expected}/{examined} controls rejected for the expected reason"
        ));
    }

    Ok(json!({
        "status": "NEGATIVE_CONTROLS_REJECTED",
        "benchmark_evaluated": false,
        "synthetic": true,
        "controls_examined": examined,
        "controls_rejected_as_expected": rejected_as_expected,
        "outcomes": outcomes,
        "claim_ceiling": "The receipt checker rejected the frozen synthetic negative controls; this is not a model task score or a complete security audit."
    }))
}

fn main() {
    match run_negative_controls() {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("failed to serialize receipt audit: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("receipt integrity gate failed: {error}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_receipt() -> Value {
        let artifact = "fn answer() -> u8 { 42 }";
        let payload = json!({
            "status": "passed",
            "tests_failed": 0,
            "tests_passed": 1
        });
        let payload_bytes = serde_json::to_vec(&payload).expect("serialize payload");
        json!({
            "run_id": "synthetic-valid-receipt-v1",
            "status": "passed",
            "source_sha": "a".repeat(40),
            "dataset_sha256": "b".repeat(64),
            "compile_exit_code": 0,
            "test_exit_code": 0,
            "tests_executed": 1,
            "artifact_bytes": artifact,
            "declared_artifact_sha256": sha256_hex(artifact.as_bytes()),
            "result_payload": payload,
            "declared_result_sha256": sha256_hex(&payload_bytes)
        })
    }

    #[test]
    fn all_frozen_negative_controls_are_rejected_for_expected_reason() {
        let report = run_negative_controls().expect("all controls must be caught");
        assert_eq!(report["controls_examined"], 8);
        assert_eq!(report["controls_rejected_as_expected"], 8);
        assert_eq!(report["benchmark_evaluated"], false);
    }

    #[test]
    fn accepts_consistent_receipt_with_recomputed_digests() {
        assert_eq!(validate_receipt(&valid_receipt()), Ok(()));
    }

    #[test]
    fn rejects_missing_dataset_digest() {
        let mut receipt = valid_receipt();
        receipt.as_object_mut().unwrap().remove("dataset_sha256");
        assert_eq!(
            validate_receipt(&receipt),
            Err("missing_required_dataset_digest".to_owned())
        );
    }

    #[test]
    fn rejects_tampered_artifact_bytes() {
        let mut receipt = valid_receipt();
        receipt["artifact_bytes"] = json!("different code bytes");
        assert_eq!(
            validate_receipt(&receipt),
            Err("artifact_digest_mismatch".to_owned())
        );
    }

    #[test]
    fn rejects_tampered_result_payload() {
        let mut receipt = valid_receipt();
        receipt["result_payload"]["tests_failed"] = json!(1);
        assert_eq!(
            validate_receipt(&receipt),
            Err("result_payload_digest_or_status_mismatch".to_owned())
        );
    }

    #[test]
    fn rejects_a_passed_claim_with_failed_tests() {
        let mut receipt = valid_receipt();
        receipt["test_exit_code"] = json!(1);
        assert_eq!(
            validate_receipt(&receipt),
            Err("test_exit_status_conflicts_with_pass".to_owned())
        );
    }

    #[test]
    fn rejects_a_functional_claim_without_tests() {
        let mut receipt = valid_receipt();
        receipt["claim"] = json!("functional_correctness");
        receipt["tests_executed"] = json!(0);
        assert_eq!(
            validate_receipt(&receipt),
            Err("functional_claim_without_tests".to_owned())
        );
    }

    #[test]
    fn rejects_skipped_tests_disguised_as_passed() {
        let mut receipt = valid_receipt();
        receipt["test_lane_status"] = json!("skipped");
        assert_eq!(
            validate_receipt(&receipt),
            Err("skipped_lane_cannot_claim_pass".to_owned())
        );
    }
}
