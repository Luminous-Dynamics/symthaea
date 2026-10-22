// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent consistency verifier for a coding-oracle mutation receipt.
//!
//! This tool recomputes fixture hashes, test counts, outcome classifications,
//! and summary totals from the receipt's self-contained task records. It does
//! not rerun rustc or authenticate who produced the receipt.

use std::path::PathBuf;

use serde_json::Value;

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing field: {key}"))
}

fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("field {key} must be a string"))
}

fn bool_field(value: &Value, key: &str) -> Result<bool, String> {
    field(value, key)?
        .as_bool()
        .ok_or_else(|| format!("field {key} must be a boolean"))
}

fn usize_field(value: &Value, key: &str) -> Result<usize, String> {
    field(value, key)?
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| format!("field {key} must be a non-negative integer"))
}

fn digest(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

fn count_expected_tests(test_source: &str) -> usize {
    test_source
        .lines()
        .filter(|line| line.trim_start().starts_with("#[test]"))
        .count()
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn verify_content_digest(
    task: &Value,
    digest_key: &str,
    content_key: &str,
) -> Result<String, String> {
    let contents = string_field(task, content_key)?;
    let recorded = string_field(task, digest_key)?;
    if !valid_digest(recorded) {
        return Err(format!("field {digest_key} is not a 64-character hex digest"));
    }
    let computed = digest(contents);
    if computed != recorded {
        return Err(format!("digest mismatch for {content_key}"));
    }
    Ok(recorded.to_string())
}

fn derived_status(variant: &Value, tests_expected: usize) -> Result<&'static str, String> {
    let compiled = bool_field(variant, "compiled")?;
    let simulated = bool_field(variant, "simulated")?;
    let reported_expected = usize_field(variant, "tests_expected")?;
    let executed = usize_field(variant, "tests_executed")?;
    let passed = usize_field(variant, "tests_passed")?;
    let failed = usize_field(variant, "tests_failed")?;

    if reported_expected != tests_expected {
        return Err("variant expected-test count differs from task fixture".to_string());
    }
    if executed != passed + failed {
        return Err("executed test count does not equal passed + failed".to_string());
    }
    if executed > tests_expected {
        return Err("variant executed more tests than its oracle declares".to_string());
    }

    let expected_status = if simulated {
        "simulated_unverified"
    } else if !compiled {
        "compile_failed"
    } else if tests_expected == 0 {
        "missing_oracle"
    } else if executed == 0 {
        "no_tests_executed"
    } else if executed != tests_expected {
        "test_count_mismatch"
    } else if failed > 0 {
        "failed"
    } else if passed == tests_expected {
        "passed"
    } else {
        "unverified"
    };
    let reported_status = string_field(variant, "status")?;
    if reported_status != expected_status {
        return Err(format!(
            "status mismatch: receipt says {reported_status}, counts imply {expected_status}"
        ));
    }
    Ok(expected_status)
}

fn verify(receipt: &Value) -> Result<(usize, usize, usize, bool), String> {
    if usize_field(receipt, "schema_version")? != 1 {
        return Err("unsupported schema_version".to_string());
    }
    if string_field(receipt, "schema_id")? != "symthaea.coding-oracle-mutation.v1" {
        return Err("unexpected schema_id".to_string());
    }
    if string_field(receipt, "benchmark")? != "symthaea_coding_oracle_mutation" {
        return Err("unexpected benchmark identifier".to_string());
    }

    let tasks = field(receipt, "tasks")?
        .as_array()
        .ok_or_else(|| "tasks must be an array".to_string())?;
    if tasks.is_empty() {
        return Err("tasks array must not be empty".to_string());
    }
    if usize_field(field(receipt, "summary")?, "task_count")? != tasks.len() {
        return Err("summary task_count differs from task record count".to_string());
    }

    let mut task_ids = std::collections::HashSet::new();
    let mut aggregate = String::new();
    let mut expected_assertions = 0usize;
    let mut reference_passes = 0usize;
    let mut mutants_killed = 0usize;

    for task in tasks {
        let task_id = string_field(task, "task_id")?;
        if task_id.is_empty() || !task_ids.insert(task_id.to_string()) {
            return Err(format!("empty or duplicate task_id: {task_id:?}"));
        }

        let test_source = string_field(task, "test_source")?;
        let tests_expected = count_expected_tests(test_source);
        if tests_expected == 0 || tests_expected != usize_field(task, "tests_expected")? {
            return Err(format!("{task_id}: missing oracle or fixture test-count mismatch"));
        }
        let test_digest = verify_content_digest(task, "fixture_digest_blake3", "test_source")?;
        let reference_digest =
            verify_content_digest(task, "reference_source_digest_blake3", "reference_source")?;
        let mutant_digest =
            verify_content_digest(task, "mutant_source_digest_blake3", "mutant_source")?;

        aggregate.push_str(task_id);
        aggregate.push('\n');
        aggregate.push_str(&reference_digest);
        aggregate.push('\n');
        aggregate.push_str(&mutant_digest);
        aggregate.push('\n');
        aggregate.push_str(&test_digest);
        aggregate.push('\n');

        expected_assertions += tests_expected;
        let reference = field(task, "reference")?;
        let mutant = field(task, "mutant")?;
        let reference_status = derived_status(reference, tests_expected)?;
        let mutant_status = derived_status(mutant, tests_expected)?;
        let reference_passed = reference_status == "passed";
        let mutation_killed = reference_passed
            && mutant_status == "failed"
            && bool_field(mutant, "compiled")?
            && !bool_field(mutant, "simulated")?
            && usize_field(mutant, "tests_executed")? == tests_expected
            && usize_field(mutant, "tests_failed")? > 0;

        if bool_field(task, "reference_control_passed")? != reference_passed {
            return Err(format!("{task_id}: reference_control_passed contradicts result"));
        }
        if bool_field(task, "mutation_killed")? != mutation_killed {
            return Err(format!("{task_id}: mutation_killed contradicts result"));
        }
        if reference_passed {
            reference_passes += 1;
        }
        if mutation_killed {
            mutants_killed += 1;
        }
    }

    let aggregate_digest = string_field(receipt, "fixture_digest_blake3")?;
    if !valid_digest(aggregate_digest) || digest(&aggregate) != aggregate_digest {
        return Err("aggregate fixture digest mismatch".to_string());
    }

    let summary = field(receipt, "summary")?;
    if usize_field(summary, "expected_test_functions")? != expected_assertions {
        return Err("summary expected_test_functions differs from task records".to_string());
    }
    if usize_field(summary, "reference_controls_passed")? != reference_passes {
        return Err("summary reference_controls_passed differs from task records".to_string());
    }
    if usize_field(summary, "mutants_killed")? != mutants_killed {
        return Err("summary mutants_killed differs from task records".to_string());
    }
    let reported_rate = field(summary, "mutation_kill_rate")?
        .as_f64()
        .ok_or_else(|| "mutation_kill_rate must be a number".to_string())?;
    let actual_rate = mutants_killed as f64 / tasks.len() as f64;
    if (reported_rate - actual_rate).abs() > 1e-12 {
        return Err("summary mutation_kill_rate differs from task records".to_string());
    }

    let errors = field(summary, "qualification_errors")?
        .as_array()
        .ok_or_else(|| "qualification_errors must be an array".to_string())?;
    if errors.iter().any(|value| value.as_str().is_none()) {
        return Err("qualification_errors must contain only strings".to_string());
    }
    let expected_qualified =
        errors.is_empty() && reference_passes == tasks.len() && mutants_killed == tasks.len();
    let reported_qualified = bool_field(summary, "qualified")?;
    if reported_qualified != expected_qualified {
        return Err("summary qualified flag contradicts the task results".to_string());
    }

    Ok((tasks.len(), reference_passes, mutants_killed, reported_qualified))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match (args.next(), args.next()) {
        (Some(path), None) => PathBuf::from(path),
        _ => {
            eprintln!("usage: cargo run --example coding_oracle_mutation_verify -- <receipt.json>");
            std::process::exit(2);
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {}: {error}", path.display());
            std::process::exit(2);
        }
    };
    let receipt: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid JSON receipt: {error}");
            std::process::exit(2);
        }
    };

    match verify(&receipt) {
        Ok((tasks, references, mutants, qualified)) => {
            println!(
                "Receipt consistency verified: {tasks} tasks; {references} reference controls passed; {mutants} mutants killed; qualified={qualified}"
            );
            if !qualified {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("receipt verification failed: {error}");
            std::process::exit(1);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::{digest, verify};
    use serde_json::{json, Value};

    fn valid_receipt() -> Value {
        let task_id = "signed_addition";
        let reference_source = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let mutant_source = "fn add(a: i32, b: i32) -> i32 { a - b }";
        let test_source = "#[test]\nfn addition() { assert_eq!(add(2, 3), 5); }\n";
        let reference_digest = digest(reference_source);
        let mutant_digest = digest(mutant_source);
        let fixture_digest = digest(test_source);
        let aggregate = format!(
            "{task_id}\n{reference_digest}\n{mutant_digest}\n{fixture_digest}\n"
        );
        json!({
            "schema_version": 1,
            "schema_id": "symthaea.coding-oracle-mutation.v1",
            "benchmark": "symthaea_coding_oracle_mutation",
            "source_revision": "test-revision",
            "fixture_digest_blake3": digest(&aggregate),
            "runtime": {
                "rustc_version": "test",
                "os": "test",
                "arch": "test",
                "debug_assertions": true,
                "code_generation_feature": true,
                "executor": "CodeExecutor::with_real_execution"
            },
            "summary": {
                "task_count": 1,
                "expected_test_functions": 1,
                "reference_controls_passed": 1,
                "mutants_killed": 1,
                "mutation_kill_rate": 1.0,
                "qualified": true,
                "elapsed_ms": 1,
                "qualification_errors": []
            },
            "tasks": [{
                "task_id": task_id,
                "fixture_digest_blake3": fixture_digest,
                "reference_source_digest_blake3": reference_digest,
                "mutant_source_digest_blake3": mutant_digest,
                "reference_source": reference_source,
                "mutant_source": mutant_source,
                "test_source": test_source,
                "tests_expected": 1,
                "reference": {
                    "status": "passed", "compiled": true, "simulated": false,
                    "tests_expected": 1, "tests_executed": 1, "tests_passed": 1,
                    "tests_failed": 0, "elapsed_ms": 1, "compile_errors": [],
                    "runtime_error": null
                },
                "mutant": {
                    "status": "failed", "compiled": true, "simulated": false,
                    "tests_expected": 1, "tests_executed": 1, "tests_passed": 0,
                    "tests_failed": 1, "elapsed_ms": 1, "compile_errors": [],
                    "runtime_error": null
                },
                "reference_control_passed": true,
                "mutation_killed": true
            }]
        })
    }

    #[test]
    fn accepts_consistent_receipt() {
        assert_eq!(verify(&valid_receipt()).unwrap(), (1, 1, 1, true));
    }

    #[test]
    fn rejects_source_tampering_without_updated_digest() {
        let mut receipt = valid_receipt();
        receipt["tasks"][0]["reference_source"] = json!("fn add(_: i32, _: i32) -> i32 { 0 }");
        assert!(verify(&receipt).unwrap_err().contains("digest mismatch"));
    }

    #[test]
    fn rejects_incomplete_oracle_counts_even_if_status_claims_failure() {
        let mut receipt = valid_receipt();
        receipt["tasks"][0]["mutant"]["tests_executed"] = json!(0);
        receipt["tasks"][0]["mutant"]["tests_passed"] = json!(0);
        receipt["tasks"][0]["mutant"]["tests_failed"] = json!(0);
        assert!(verify(&receipt).unwrap_err().contains("status mismatch"));
    }

    #[test]
    fn rejects_false_mutation_kill_flag() {
        let mut receipt = valid_receipt();
        receipt["tasks"][0]["mutation_killed"] = json!(false);
        assert!(verify(&receipt).unwrap_err().contains("mutation_killed contradicts"));
    }

    #[test]
    fn rejects_inconsistent_summary_totals() {
        let mut receipt = valid_receipt();
        receipt["summary"]["mutants_killed"] = json!(0);
        assert!(verify(&receipt).unwrap_err().contains("summary mutants_killed"));
    }

    #[test]
    fn rejects_duplicate_task_ids() {
        let mut receipt = valid_receipt();
        let duplicate = receipt["tasks"][0].clone();
        receipt["tasks"].as_array_mut().unwrap().push(duplicate);
        receipt["summary"]["task_count"] = json!(2);
        assert!(verify(&receipt).unwrap_err().contains("duplicate task_id"));
    }

    #[test]
    fn rejects_corrupt_aggregate_digest() {
        let mut receipt = valid_receipt();
        receipt["fixture_digest_blake3"] = json!("0".repeat(64));
        assert!(verify(&receipt).unwrap_err().contains("aggregate fixture digest"));
    }
}
