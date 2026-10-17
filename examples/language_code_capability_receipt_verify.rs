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
use std::path::{Component, Path};
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
fn validate_receipt(
    receipt: &Value,
    expected_source_sha: Option<&str>,
    artifact_bytes_override: Option<&[u8]>,
) -> Result<(), String> {
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

    let actual_source = receipt
        .get("source_sha")
        .and_then(Value::as_str)
        .unwrap_or("");
    if let Some(pinned_source) = expected_source_sha {
        if !is_lower_hex(pinned_source, 40) || actual_source != pinned_source {
            return Err("source_revision_mismatch".to_owned());
        }
    }
    if let Some(expected_source) = receipt.get("expected_source_sha").and_then(Value::as_str) {
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
        let embedded_artifact = receipt.get("artifact_bytes").and_then(Value::as_str);
        let artifact_bytes = artifact_bytes_override
            .or_else(|| embedded_artifact.map(str::as_bytes))
            .ok_or_else(|| "missing_artifact_bytes".to_owned())?;
        if !is_lower_hex(declared, 64) || sha256_hex(artifact_bytes) != declared {
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
    let compile_only_policy = receipt
        .get("compile_only_policy")
        .and_then(Value::as_str)
        .filter(|policy| *policy == "compile_only");
    if status == "passed"
        && receipt.get("claim").and_then(Value::as_str) == Some("functional_correctness")
        && tests_executed == 0
        && compile_only_policy.is_none()
    {
        return Err("functional_claim_without_tests".to_owned());
    }

    let payload = receipt
        .get("result_payload")
        .ok_or_else(|| "missing_result_payload".to_owned())?;
    let payload_status = payload.get("status").and_then(Value::as_str).unwrap_or("");
    let serialized = serde_json::to_vec(payload)
        .map_err(|_| "result_payload_digest_or_status_mismatch".to_owned())?;
    let recomputed_digest = sha256_hex(&serialized);
    let declared_result = receipt
        .get("declared_result_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_result_payload_digest".to_owned())?;
    if payload_status != status
        || !is_lower_hex(declared_result, 64)
        || recomputed_digest != declared_result
    {
        return Err("result_payload_digest_or_status_mismatch".to_owned());
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


    }

    // Every receipt status must bind the actual artifact bytes. A failed or
    // indeterminate run may be valid evidence of failure, but it cannot omit
    // artifact identity and still be promoted as an internally valid receipt.
    let embedded_artifact = receipt.get("artifact_bytes").and_then(Value::as_str);
    let artifact_bytes = artifact_bytes_override
        .or_else(|| embedded_artifact.map(str::as_bytes))
        .ok_or_else(|| "missing_artifact_bytes".to_owned())?;
    let declared_artifact = receipt
        .get("declared_artifact_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_artifact_digest".to_owned())?;
    if !is_lower_hex(declared_artifact, 64)
        || sha256_hex(artifact_bytes) != declared_artifact
    {
        return Err("artifact_digest_mismatch".to_owned());
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

        let actual = validate_receipt(receipt, None, None);
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
            concat!(
                "negative-control gate failed: ",
                "{rejected_as_expected}/{examined} controls rejected for the expected reason"
            )
        ));
    }

    Ok(json!({
        "status": "NEGATIVE_CONTROLS_REJECTED",
        "benchmark_evaluated": false,
        "synthetic": true,
        "controls_examined": examined,
        "controls_rejected_as_expected": rejected_as_expected,
        "outcomes": outcomes,
        "claim_ceiling": concat!(
            "The receipt checker rejected the frozen synthetic negative controls; ",
            "this is not a model task score or a complete security audit."
        )
    }))
}

fn resolve_artifact_under_root(root: &Path, relative_path: &str) -> Result<Vec<u8>, String> {
    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || !relative.components().all(|component| matches!(component, Component::Normal(_)))
    {
        return Err("artifact_path_must_be_relative_and_cannot_traverse".to_owned());
    }

    let canonical_root = root
        .canonicalize()
        .map_err(|_| "artifact_root_unavailable".to_owned())?;
    let candidate = canonical_root.join(relative);
    let canonical_candidate = candidate
        .canonicalize()
        .map_err(|_| "artifact_file_unavailable".to_owned())?;
    if !canonical_candidate.starts_with(&canonical_root) || !canonical_candidate.is_file() {
        return Err("artifact_path_escapes_root_or_is_not_a_file".to_owned());
    }

    std::fs::read(canonical_candidate).map_err(|_| "artifact_file_unreadable".to_owned())
}

fn verify_external_receipt(args: &[String]) -> Result<(Value, bool), String> {
    let mut receipt_path: Option<&str> = None;
    let mut artifact_root: Option<&str> = None;
    let mut expected_source: Option<&str> = None;
    let mut index = 0usize;
    while index < args.len() {
        let key = args[index].as_str();
        let value = args
            .get(index + 1)
            .map(String::as_str)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--receipt" => receipt_path = Some(value),
            "--artifact-root" => artifact_root = Some(value),
            "--expected-source-sha" => expected_source = Some(value),
            _ => return Err(format!("unknown argument: {key}")),
        }
        index += 2;
    }

    let receipt_path = receipt_path.ok_or_else(|| "--receipt is required".to_owned())?;
    let artifact_root = artifact_root.ok_or_else(|| "--artifact-root is required".to_owned())?;
    let expected_source = expected_source
        .ok_or_else(|| "--expected-source-sha is required to pin the expected head".to_owned())?;
    if !is_lower_hex(expected_source, 40) {
        return Err("--expected-source-sha must be 40 lowercase hexadecimal characters".to_owned());
    }

    let raw = std::fs::read(receipt_path).map_err(|_| "receipt_file_unavailable".to_owned())?;
    let receipt: Value = serde_json::from_slice(&raw)
        .map_err(|_| "receipt_file_is_not_valid_json".to_owned())?;
    let relative_artifact = receipt
        .get("artifact_path")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_artifact_path".to_owned())?;
    let artifact_bytes = resolve_artifact_under_root(Path::new(artifact_root), relative_artifact)?;

    match validate_receipt(&receipt, Some(expected_source), Some(&artifact_bytes)) {
        Ok(()) => Ok((
            json!({
                "status": "RECEIPT_VALID",
                "benchmark_evaluated": false,
                "source_sha": expected_source,
                "artifact_path": relative_artifact,
                "artifact_sha256": sha256_hex(&artifact_bytes),
                "claim_ceiling": concat!(
                    "Receipt metadata and referenced artifact identity are internally consistent; ",
                    "this does not independently rerun the declared tests."
                )
            }),
            true,
        )),
        Err(reason) => Ok((
            json!({
                "status": "RECEIPT_REJECTED",
                "benchmark_evaluated": false,
                "reason": reason,
                "claim_ceiling": "The receipt did not satisfy the independent integrity contract."
            }),
            false,
        )),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() && args[0] == "--receipt" {
        match verify_external_receipt(&args) {
            Ok((report, accepted)) => {
                let rendered =
                    serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into());
                println!("{rendered}");
                if !accepted {
                    process::exit(2);
                }
            }
            Err(error) => {
                eprintln!("receipt verifier could not evaluate input: {error}");
                process::exit(3);
            }
        }
        return;
    }

    if !args.is_empty() {
        eprintln!(
            concat!(
                "usage: language_code_capability_receipt_verify ",
                "[--receipt FILE --artifact-root DIR --expected-source-sha SHA40]"
            )
        );
        process::exit(64);
    }

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
        let pinned_source = "a".repeat(40);
        let receipt = valid_receipt();
        assert_eq!(
            validate_receipt(&receipt, Some(pinned_source.as_str()), None),
            Ok(())
        );
    }

    #[test]
    fn rejects_missing_dataset_digest() {
        let mut receipt = valid_receipt();
        receipt.as_object_mut().unwrap().remove("dataset_sha256");
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("missing_required_dataset_digest".to_owned())
        );
    }

    #[test]
    fn rejects_tampered_artifact_bytes() {
        let mut receipt = valid_receipt();
        receipt["artifact_bytes"] = json!("different code bytes");
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("artifact_digest_mismatch".to_owned())
        );
    }

    #[test]
    fn rejects_tampered_result_payload() {
        let mut receipt = valid_receipt();
        receipt["result_payload"]["tests_failed"] = json!(1);
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("result_payload_digest_or_status_mismatch".to_owned())
        );
    }

    #[test]
    fn rejects_artifact_path_traversal_before_filesystem_access() {
        assert_eq!(
            resolve_artifact_under_root(Path::new("."), "../Cargo.toml"),
            Err("artifact_path_must_be_relative_and_cannot_traverse".to_owned())
        );
        assert_eq!(
            resolve_artifact_under_root(Path::new("."), "/etc/passwd"),
            Err("artifact_path_must_be_relative_and_cannot_traverse".to_owned())
        );
    }

    #[test]
    fn rejects_receipt_for_an_unexpected_source_revision() {
        let mut receipt = valid_receipt();
        receipt["source_sha"] = json!("c".repeat(40));
        let pinned_source = "a".repeat(40);
        assert_eq!(
            validate_receipt(&receipt, Some(pinned_source.as_str()), None),
            Err("source_revision_mismatch".to_owned())
        );
    }

    #[test]
    fn uses_external_artifact_bytes_when_verifying_digest() {
        let receipt = valid_receipt();
        let different_bytes = b"not the artifact embedded in the test receipt";
        assert_eq!(
            validate_receipt(&receipt, Some(&"a".repeat(40)), Some(different_bytes)),
            Err("artifact_digest_mismatch".to_owned())
        );
    }

    #[test]
    fn failed_receipt_still_requires_result_payload() {
        let mut receipt = valid_receipt();
        receipt["status"] = json!("failed");
        receipt.as_object_mut().unwrap().remove("result_payload");
        receipt.as_object_mut().unwrap().remove("declared_result_sha256");
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("missing_result_payload".to_owned())
        );
    }

    #[test]
    fn failed_receipt_still_requires_artifact_identity() {
        let mut receipt = valid_receipt();
        receipt["status"] = json!("failed");
        receipt["result_payload"]["status"] = json!("failed");
        let payload_bytes =
            serde_json::to_vec(&receipt["result_payload"]).expect("serialize payload");
        receipt["declared_result_sha256"] = json!(sha256_hex(&payload_bytes));
        receipt.as_object_mut().unwrap().remove("declared_artifact_sha256");
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("missing_artifact_digest".to_owned())
        );
    }

    #[test]
    fn failed_receipt_must_hash_the_referenced_artifact_bytes() {
        let mut receipt = valid_receipt();
        receipt["status"] = json!("failed");
        receipt["result_payload"]["status"] = json!("failed");
        let payload_bytes =
            serde_json::to_vec(&receipt["result_payload"]).expect("serialize payload");
        receipt["declared_result_sha256"] = json!(sha256_hex(&payload_bytes));
        receipt["artifact_bytes"] = json!("changed after failure");
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("artifact_digest_mismatch".to_owned())
        );
    }

    #[test]
    fn rejects_a_passed_claim_with_failed_tests() {
        let mut receipt = valid_receipt();
        receipt["test_exit_code"] = json!(1);
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("test_exit_status_conflicts_with_pass".to_owned())
        );
    }

    #[test]
    fn rejects_a_functional_claim_without_tests() {
        let mut receipt = valid_receipt();
        receipt["claim"] = json!("functional_correctness");
        receipt["tests_executed"] = json!(0);
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("functional_claim_without_tests".to_owned())
        );
    }

    #[test]
    fn rejects_skipped_tests_disguised_as_passed() {
        let mut receipt = valid_receipt();
        receipt["test_lane_status"] = json!("skipped");
        assert_eq!(
            validate_receipt(&receipt, None, None),
            Err("skipped_lane_cannot_claim_pass".to_owned())
        );
    }
}
