// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent verifier for PITB R5 post-elaboration snapshot receipts.

#[path = "../snapshot_branch_receipt.rs"]
mod snapshot_branch_receipt;

use std::{
    env,
    fs,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde_json::Value;

enum Mode {
    Help,
    Verify {
        path: PathBuf,
        expected_sha256: Option<String>,
    },
    SelfTest(PathBuf),
}

fn must_reject(label: &str, value: Value) -> Result<()> {
    let bytes = serde_json::to_vec(&value).context("serialize negative snapshot receipt")?;
    if snapshot_branch_receipt::verify_bytes(&bytes, None).is_ok() {
        bail!("snapshot receipt verifier accepted negative control: {label}");
    }
    println!("PITB_SNAPSHOT_RECEIPT_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(path: &PathBuf) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("read fixture {}", path.display()))?;
    let summary = snapshot_branch_receipt::verify_bytes(&bytes, None)?;
    if summary.run_status != "not_attempted" || summary.branch_count != 0 {
        bail!("checked-in snapshot fixture must remain a no-run contract fixture");
    }
    println!("PITB_SNAPSHOT_RECEIPT_FIXTURE_PASS: synthetic no-run receipt verified");

    let mut measured = serde_json::from_slice::<Value>(&bytes)?;
    measured["evidence_class"] = Value::from("measured_run");
    measured["run_status"] = Value::from("measured");
    measured["snapshot"]["capture_status"] = Value::from("captured");
    measured["snapshot"]["snapshot_id_sha256"] = Value::from("a".repeat(64));
    measured["measurements"]["capture_wall_ms"] = Value::from(10);
    measured["measurements"]["branch_count"] = Value::from(1);
    measured["measurements"]["branch_restore_p50_ms"] = Value::from(3);
    measured["measurements"]["branch_restore_p95_ms"] = Value::from(5);
    measured["measurements"]["elapsed_wall_ms"] = Value::from(100);
    measured["branches"] = serde_json::json!([{
        "branch_id":"branch-1",
        "base_snapshot_sha256":"a".repeat(64),
        "mutable_context_isolated":true,
        "cache_policy":"discard_and_reinitialize_per_branch",
        "status":"candidate_available",
        "candidate_sha256":"b".repeat(64)
    }]);
    let measured_bytes = serde_json::to_vec(&measured)?;
    let measured_summary = snapshot_branch_receipt::verify_bytes(&measured_bytes, None)?;
    if measured_summary.run_status != "measured" || measured_summary.branch_count != 1 {
        bail!("synthetic measured-shape control returned inconsistent metadata");
    }
    println!("PITB_SNAPSHOT_RECEIPT_MEASURED_SHAPE_PASS: synthetic only; no real snapshot or Lean branch");

    let mut bad = measured.clone();
    bad["branches"][0]["mutable_context_isolated"] = Value::Bool(false);
    must_reject("mutable MetavarContext shared across branches", bad)?;

    let mut bad = measured.clone();
    bad["branches"][0]["base_snapshot_sha256"] = Value::String("c".repeat(64));
    must_reject("branch descends from a different snapshot", bad)?;

    let mut bad = measured.clone();
    bad["execution_contract"]["typeclass_cache_policy"] =
        Value::String("share_without_generation_validation".into());
    must_reject("unqualified mutable typeclass cache reuse", bad)?;

    let mut bad = measured.clone();
    bad["measurements"]["stale_cache_observations"] = Value::from(1);
    must_reject("stale cache observation", bad)?;

    let mut bad = measured.clone();
    bad["measurements"]["theorem_reelaboration_count"] = Value::from(1);
    must_reject("theorem body re-elaborated after capture", bad)?;

    let mut bad = measured.clone();
    bad["measurements"]["branch_count"] = Value::from(2);
    must_reject("branch count does not match branch receipts", bad)?;

    let mut bad = measured.clone();
    bad["measurements"]["branch_restore_p50_ms"] = Value::from(6);
    must_reject("restore p50 exceeds p95", bad)?;

    let mut bad = measured.clone();
    bad["branches"][0]["base_snapshot_sha256"] = Value::from("f".repeat(64));
    must_reject("snapshot identity substitution", bad)?;

    let expected = hex::encode(sha2::Sha256::digest(&bytes));
    if snapshot_branch_receipt::verify_bytes(&bytes, Some(&expected)).is_err() {
        bail!("receipt verifier rejected matching exact-byte digest");
    }
    if snapshot_branch_receipt::verify_bytes(&bytes, Some(&"f".repeat(64))).is_ok() {
        bail!("receipt verifier accepted a mismatched exact-byte digest");
    }
    println!("PITB_SNAPSHOT_RECEIPT_BYTE_IDENTITY_PASS: exact SHA-256 binding and substitution rejection");
    Ok(())
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let Some(action) = args.next() else {
        bail!("expected --verify FILE, --self-test FILE, or --help");
    };
    if matches!(action.as_str(), "--help" | "-h") {
        return Ok(Mode::Help);
    }
    let path = PathBuf::from(args.next().context("expected a receipt path")?);
    match action.as_str() {
        "--self-test" => {
            if args.next().is_some() {
                bail!("--self-test accepts exactly one receipt path");
            }
            Ok(Mode::SelfTest(path))
        }
        "--verify" => {
            let mut expected_sha256 = None;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--expected-sha256" => {
                        if expected_sha256.is_some() {
                            bail!("--expected-sha256 was supplied more than once");
                        }
                        expected_sha256 = Some(args.next().context("missing expected SHA-256")?);
                    }
                    _ => bail!("unknown argument: {flag}"),
                }
            }
            Ok(Mode::Verify { path, expected_sha256 })
        }
        _ => bail!("unknown action {action}"),
    }
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: verify_snapshot_branch_receipt --verify FILE [--expected-sha256 HEX] | --self-test FILE");
            Ok(())
        }
        Mode::SelfTest(path) => self_test(&path),
        Mode::Verify { path, expected_sha256 } => {
            let summary = snapshot_branch_receipt::verify_file(
                &path,
                expected_sha256.as_deref(),
            )?;
            println!(
                "PITB_SNAPSHOT_RECEIPT_VERIFY_PASS: task={} commit={} environment={} proof_state={} snapshot={} branches={} status={}",
                summary.task_id,
                summary.repository_commit,
                summary.environment_sha256,
                summary.proof_state_sha256,
                summary.snapshot_id_sha256.as_deref().unwrap_or("none"),
                summary.branch_count,
                summary.run_status
            );
            println!("PITB_SNAPSHOT_RECEIPT_SHA256: {}", summary.receipt_sha256);
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_SNAPSHOT_RECEIPT_VERIFY_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn lower_hex_is_case_sensitive_and_fixed_width() {
        assert_eq!(hex::encode(sha2::Sha256::digest(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    use sha2::Digest;
}
