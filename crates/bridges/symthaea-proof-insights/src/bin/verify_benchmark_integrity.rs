// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact-byte verifier for PITB benchmark target integrity receipts.

#[path = "../benchmark_integrity.rs"]
mod benchmark_integrity;

use std::{
    env,
    fs,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

enum Mode {
    Help,
    Verify {
        path: PathBuf,
        expected_sha256: Option<String>,
    },
    SelfTest(PathBuf),
}

fn digest_list(value: &Value, path: &str) -> Result<String> {
    let list = value
        .pointer(path)
        .and_then(Value::as_array)
        .with_context(|| format!("{path} must be an array"))?;
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(list)?)))
}

fn make_measured_eligible(mut value: Value) -> Result<Value> {
    value["evidence_class"] = json!("measured_audit");
    value["elaboration"]["status"] = json!("passed");
    value["elaboration"]["receipt_sha256"] = json!("c".repeat(64));
    value["counterexample_audit"]["status"] = json!("not_found");
    value["counterexample_audit"]["method_id"] = json!("synthetic-counterexample-probe-v0");
    value["counterexample_audit"]["receipt_sha256"] = json!("d".repeat(64));
    value["vacuity_audit"]["status"] = json!("no_counterexample_found");
    value["vacuity_audit"]["method_id"] = json!("synthetic-vacuity-probe-v0");
    value["vacuity_audit"]["receipt_sha256"] = json!("e".repeat(64));
    value["semantic_review"]["status"] = json!("approved");
    value["semantic_review"]["review_scope"] = json!("combined");
    value["semantic_review"]["reviewer_identity"] = json!("synthetic-reviewer-not-real");
    value["semantic_review"]["reviewer_attestation_sha256"] = json!("f".repeat(64));
    let used_digest = digest_list(&value, "/axioms/used_axioms")?;
    let allowed_digest = digest_list(&value, "/axioms/allowed_axioms")?;
    value["axioms"]["used_axioms_manifest_sha256"] = json!(used_digest);
    value["axioms"]["allowlist_manifest_sha256"] = json!(allowed_digest);
    value["integrity_status"] = json!("eligible");
    Ok(value)
}

fn must_reject(label: &str, value: Value) -> Result<()> {
    let bytes = serde_json::to_vec(&value)?;
    if benchmark_integrity::verify_bytes(&bytes, None).is_ok() {
        bail!("benchmark integrity verifier accepted negative control: {label}");
    }
    println!("PITB_BENCHMARK_INTEGRITY_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(path: &PathBuf) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("read fixture {}", path.display()))?;
    let summary = benchmark_integrity::verify_bytes(&bytes, None)?;
    if summary.integrity_status != "not_attempted" {
        bail!("checked-in benchmark integrity fixture must remain a no-run example");
    }
    println!("PITB_BENCHMARK_INTEGRITY_FIXTURE_PASS: synthetic no-run receipt");

    let base: Value = serde_json::from_slice(&bytes)?;
    let eligible = make_measured_eligible(base.clone())?;
    let eligible_bytes = serde_json::to_vec(&eligible)?;
    let eligible_summary = benchmark_integrity::verify_bytes(&eligible_bytes, None)?;
    if eligible_summary.integrity_status != "eligible" {
        bail!("fully qualified synthetic semantic review did not derive eligible status");
    }
    println!("PITB_BENCHMARK_INTEGRITY_ELIGIBILITY_PASS: derived status from elaboration, axioms, probes and semantic review");

    let mut bad = eligible.clone();
    bad["target"]["statement_sha256"] = json!("0".repeat(64));
    must_reject("elaborated statement identity drift", bad)?;

    let mut bad = eligible.clone();
    bad["elaboration"]["contains_sorry_ax"] = Value::Bool(true);
    bad["integrity_status"] = json!("ineligible");
    must_reject("sorryAx in elaborated target", bad)?;

    let mut bad = eligible.clone();
    bad["elaboration"]["metavariable_count"] = json!(1);
    bad["integrity_status"] = json!("ineligible");
    must_reject("unresolved metavariable", bad)?;

    let mut bad = eligible.clone();
    bad["axioms"]["used_axioms"] = json!(["Unsafe.assumption"]);
    bad["axioms"]["used_axioms_manifest_sha256"] = json!(digest_list(&bad, "/axioms/used_axioms")?);
    bad["axioms"]["unexpected_axioms"] = json!(1);
    bad["integrity_status"] = json!("ineligible");
    must_reject("unapproved axiom in target dependency closure", bad)?;

    let mut bad = eligible.clone();
    bad["counterexample_audit"]["status"] = json!("counterexample_found");
    bad["integrity_status"] = json!("ineligible");
    must_reject("counterexample found for the formal statement", bad)?;

    let mut bad = eligible.clone();
    bad["vacuity_audit"]["status"] = json!("counterexample_found");
    bad["integrity_status"] = json!("ineligible");
    must_reject("vacuity probe exposes a counterexample", bad)?;

    let mut pending_review = eligible.clone();
    pending_review["semantic_review"]["status"] = json!("needs_revision");
    pending_review["integrity_status"] = json!("pending");
    let pending_bytes = serde_json::to_vec(&pending_review)?;
    let pending_summary = benchmark_integrity::verify_bytes(&pending_bytes, None)?;
    if pending_summary.integrity_status != "pending" {
        bail!("target without semantic approval was not held pending");
    }
    println!("PITB_BENCHMARK_INTEGRITY_PENDING_PASS: compiler success without semantic approval is not eligible");


    let mut bad = eligible.clone();
    bad["semantic_review"]["review_scope"] = json!("statement_to_source");
    bad["integrity_status"] = json!("pending");
    let scope_bytes = serde_json::to_vec(&bad)?;
    let scope_summary = benchmark_integrity::verify_bytes(&scope_bytes, None)?;
    if scope_summary.integrity_status != "pending" {
        bail!("source-only review incorrectly approved semantic target meaning");
    }
    println!("PITB_BENCHMARK_INTEGRITY_PENDING_PASS: source-coordinate review alone is not semantic approval");

    let mut bad = eligible.clone();
    bad["target"]["expected_informal_statement_sha256"] = Value::Null;
    bad["integrity_status"] = json!("pending");
    let mapping_bytes = serde_json::to_vec(&bad)?;
    let mapping_summary = benchmark_integrity::verify_bytes(&mapping_bytes, None)?;
    if mapping_summary.integrity_status != "pending" {
        bail!("combined semantic review without intended informal statement digest was approved");
    }
    println!("PITB_BENCHMARK_INTEGRITY_PENDING_PASS: combined review requires intended informal statement identity");

    let mut bad = eligible.clone();
    bad["semantic_review"]["reviewer_attestation_sha256"] = Value::Null;
    must_reject("approved semantic review has no attestation", bad)?;

    let mut bad = eligible.clone();
    bad["axioms"]["used_axioms_manifest_sha256"] = json!("0".repeat(64));
    must_reject("axiom closure manifest digest substitution", bad)?;

    let mut bad = eligible.clone();
    bad["target"]["source_path"] = json!("../../outside.lean");
    must_reject("source path traversal", bad)?;

    let mut bad = eligible.clone();
    bad["integrity_status"] = json!("pending");
    must_reject("caller downgrades or falsifies derived integrity status", bad)?;

    let expected = hex::encode(Sha256::digest(&bytes));
    benchmark_integrity::verify_bytes(&bytes, Some(&expected))?;
    if benchmark_integrity::verify_bytes(&bytes, Some(&"0".repeat(64))).is_ok() {
        bail!("benchmark integrity verifier accepted a mismatched receipt digest");
    }
    println!("PITB_BENCHMARK_INTEGRITY_BYTE_IDENTITY_PASS: exact receipt hash and substitution rejection");
    Ok(())
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let Some(action) = args.next() else {
        bail!("expected --verify FILE [--expected-sha256 HEX], --self-test FILE, or --help");
    };
    if matches!(action.as_str(), "--help" | "-h") {
        return Ok(Mode::Help);
    }
    let path = PathBuf::from(args.next().context("missing receipt path")?);
    match action.as_str() {
        "--self-test" => {
            if args.next().is_some() {
                bail!("--self-test accepts exactly one fixture path");
            }
            Ok(Mode::SelfTest(path))
        }
        "--verify" => {
            let mut expected_sha256 = None;
            while let Some(flag) = args.next() {
                if flag != "--expected-sha256" || expected_sha256.is_some() {
                    bail!("unknown or duplicate argument {flag}");
                }
                expected_sha256 = Some(args.next().context("missing expected SHA-256")?);
            }
            Ok(Mode::Verify { path, expected_sha256 })
        }
        _ => bail!("unknown action {action}"),
    }
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: verify_benchmark_integrity --verify FILE [--expected-sha256 HEX] | --self-test FILE");
            Ok(())
        }
        Mode::SelfTest(path) => self_test(&path),
        Mode::Verify { path, expected_sha256 } => {
            let summary = benchmark_integrity::verify_file(&path, expected_sha256.as_deref())?;
            println!(
                "PITB_BENCHMARK_INTEGRITY_PASS: task={} statement={} status={} used_axioms={} allowlisted_axioms={}",
                summary.task_id,
                summary.statement_sha256,
                summary.integrity_status,
                summary.used_axiom_count,
                summary.allowed_axiom_count
            );
            println!("PITB_BENCHMARK_INTEGRITY_SHA256: {}", summary.receipt_sha256);
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_BENCHMARK_INTEGRITY_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}
