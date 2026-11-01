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
    Verify { run_path: PathBuf, integrity_path: PathBuf },
    SelfTest { run_path: PathBuf, integrity_path: PathBuf },
}

fn string<'a>(value: &'a Value, path: &str) -> Result<&'a str> {
    value
        .pointer(path)
        .and_then(Value::as_str)
        .with_context(|| format!("{path} must be a string"))
}

fn integer(value: &Value, path: &str) -> Result<u64> {
    value
        .pointer(path)
        .and_then(Value::as_u64)
        .with_context(|| format!("{path} must be a nonnegative integer"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn verify_binding(run: &Value, integrity: &benchmark_integrity::IntegritySummary) -> Result<()> {
    if string(run, "/schema_version")? != "pitb.sorrydb.run-receipt.v0" {
        bail!("unsupported SorryDB run receipt version");
    }
    if string(run, "/evidence_class")? != "measured_run" {
        bail!("task-integrity binding is only meaningful for measured_run receipts");
    }
    if integrity.integrity_status != "eligible" {
        bail!("benchmark target is not eligible: {}", integrity.integrity_status);
    }
    if string(run, "/task_integrity/integrity_status")? != "eligible" {
        bail!("measured SorryDB run does not declare an eligible task-integrity status");
    }
    if string(run, "/task_integrity/receipt_sha256")? != integrity.receipt_sha256 {
        bail!("run task-integrity receipt digest does not match exact receipt bytes");
    }
    if string(run, "/task_integrity/task_id_sha256")? != integrity.task_id {
        bail!("run task-integrity task ID digest differs from the receipt");
    }

    if string(run, "/dataset/snapshot_id")? != integrity.snapshot_id {
        bail!("SorryDB snapshot ID differs from the integrity receipt");
    }
    if string(run, "/task/sorry_id")? != integrity.source_task_id {
        bail!("SorryDB source task ID differs from the integrity receipt");
    }
    if string(run, "/task/repository_remote")? != integrity.repository_remote {
        bail!("repository remote differs from the integrity receipt");
    }
    if string(run, "/task/repository_branch")? != integrity.repository_branch {
        bail!("repository branch differs from the integrity receipt");
    }
    if string(run, "/task/repository_commit")? != integrity.repository_commit
        || string(run, "/environment/repository_commit")? != integrity.repository_commit
    {
        bail!("repository commit differs between run, task and integrity receipt");
    }
    if string(run, "/task/source_path")? != integrity.source_path {
        bail!("source path differs from the integrity receipt");
    }
    for (path, expected) in [
        ("/task/start_line", integrity.start_line),
        ("/task/start_column", integrity.start_column),
        ("/task/end_line", integrity.end_line),
        ("/task/end_column", integrity.end_column),
    ] {
        if integer(run, path)? != expected {
            bail!("source coordinate {path} differs from the integrity receipt");
        }
    }
    if string(run, "/task/lean_version")? != integrity.lean_toolchain
        || string(run, "/environment/lean_version")? != integrity.lean_toolchain
    {
        bail!("Lean toolchain differs from the integrity receipt");
    }
    if string(run, "/environment/lake_manifest_sha256")? != integrity.lake_manifest_sha256 {
        bail!("Lake manifest digest differs from the integrity receipt");
    }
    Ok(())
}

fn verify_pair(run_bytes: &[u8], integrity_bytes: &[u8]) -> Result<benchmark_integrity::IntegritySummary> {
    let run: Value = serde_json::from_slice(run_bytes).context("parse SorryDB run receipt")?;
    let expected = string(&run, "/task_integrity/receipt_sha256")?;
    let integrity =
        benchmark_integrity::verify_bytes(integrity_bytes, Some(expected))?;
    verify_binding(&run, &integrity)?;
    Ok(integrity)
}

fn digest_string_list(rows: &Value, path: &str) -> Result<String> {
    let items = rows
        .pointer(path)
        .and_then(Value::as_array)
        .with_context(|| format!("{path} must be an array"))?;
    Ok(sha256_hex(&serde_json::to_vec(items)?))
}

fn make_eligible_integrity(mut value: Value, run: &Value) -> Result<Value> {
    let sorry_id = string(run, "/task/sorry_id")?.to_owned();
    let repository_commit = string(run, "/task/repository_commit")?.to_owned();
    let repository_remote = string(run, "/task/repository_remote")?.to_owned();
    let repository_branch = string(run, "/task/repository_branch")?.to_owned();
    let source_path = string(run, "/task/source_path")?.to_owned();
    let lean_version = string(run, "/task/lean_version")?.to_owned();

    value["evidence_class"] = json!("measured_audit");
    value["dataset"]["cohort_id"] = json!("SorryDB_2601");
    value["dataset"]["snapshot_id"] = run["dataset"]["snapshot_id"].clone();
    value["dataset"]["source_task_id"] = json!(sorry_id);
    value["dataset"]["task_id"] = json!(sha256_hex(sorry_id.as_bytes()));

    value["target"]["repository_remote"] = json!(repository_remote);
    value["target"]["repository_branch"] = json!(repository_branch);
    value["target"]["repository_commit"] = json!(repository_commit);
    value["target"]["source_path"] = json!(source_path);
    value["target"]["start_line"] = run["task"]["start_line"].clone();
    value["target"]["start_column"] = run["task"]["start_column"].clone();
    value["target"]["end_line"] = run["task"]["end_line"].clone();
    value["target"]["end_column"] = run["task"]["end_column"].clone();

    value["environment"]["lean_toolchain"] = json!(lean_version);
    value["environment"]["lake_manifest_sha256"] = run["environment"]["lake_manifest_sha256"].clone();

    value["elaboration"]["status"] = json!("passed");
    value["elaboration"]["checked_statement_sha256"] = value["target"]["statement_sha256"].clone();
    value["elaboration"]["receipt_sha256"] = json!("c".repeat(64));

    // Keep all axiom manifests canonical and sorted, even in the synthetic example.
    value["axioms"]["used_axioms"] = json!([]);
    value["axioms"]["allowed_axioms"] = json!(["Classical.choice", "Quot.sound", "propext"]);
    value["axioms"]["used_axioms_manifest_sha256"] =
        json!(digest_string_list(&value, "/axioms/used_axioms")?);
    value["axioms"]["allowlist_manifest_sha256"] =
        json!(digest_string_list(&value, "/axioms/allowed_axioms")?);
    value["axioms"]["unexpected_axioms"] = json!(0);

    value["counterexample_audit"]["status"] = json!("not_found");
    value["counterexample_audit"]["method_id"] = json!("synthetic-counterexample-check");
    value["counterexample_audit"]["receipt_sha256"] = json!("d".repeat(64));
    value["vacuity_audit"]["status"] = json!("no_counterexample_found");
    value["vacuity_audit"]["method_id"] = json!("synthetic-vacuity-check");
    value["vacuity_audit"]["receipt_sha256"] = json!("e".repeat(64));
    value["semantic_review"]["status"] = json!("approved");
    value["semantic_review"]["review_scope"] = json!("domain_expert_review");
    value["semantic_review"]["reviewer_identity"] = json!("synthetic-reviewer-not-real");
    value["semantic_review"]["reviewer_attestation_sha256"] = json!("f".repeat(64));
    value["integrity_status"] = json!("eligible");
    Ok(value)
}

fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn must_reject(label: &str, run: Value, integrity_bytes: &[u8]) -> Result<()> {
    let run_bytes = json_bytes(&run)?;
    if verify_pair(&run_bytes, integrity_bytes).is_ok() {
        bail!("SorryDB integrity join accepted negative control: {label}");
    }
    println!("PITB_SORRYDB_INTEGRITY_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(run_path: &PathBuf, integrity_path: &PathBuf) -> Result<()> {
    let run_bytes = fs::read(run_path).with_context(|| format!("read run fixture {}", run_path.display()))?;
    let integrity_bytes =
        fs::read(integrity_path).with_context(|| format!("read integrity fixture {}", integrity_path.display()))?;
    let mut run: Value = serde_json::from_slice(&run_bytes).context("parse synthetic run fixture")?;
    let integrity: Value =
        serde_json::from_slice(&integrity_bytes).context("parse synthetic integrity fixture")?;

    run["task_integrity"] = Value::Null;
    if verify_pair(&run_bytes, &integrity_bytes).is_ok() {
        bail!("unbound synthetic receipts unexpectedly passed as a measured run");
    }
    println!("PITB_SORRYDB_INTEGRITY_NEGATIVE_PASS: no-run fixture is not an eligible measured task");

    let eligible = make_eligible_integrity(integrity, &run)?;
    let integrity_bytes = json_bytes(&eligible)?;
    let integrity_summary = benchmark_integrity::verify_bytes(&integrity_bytes, None)?;
    if integrity_summary.integrity_status != "eligible" {
        bail!("synthetic eligible integrity fixture did not derive eligible status");
    }
    run["evidence_class"] = json!("measured_run");
    run["task_integrity"] = json!({
        "receipt_sha256":sha256_hex(&integrity_bytes),
        "task_id_sha256":integrity_summary.task_id,
        "integrity_status":"eligible"
    });
    let run_bytes = json_bytes(&run)?;
    verify_pair(&run_bytes, &integrity_bytes)?;
    println!("PITB_SORRYDB_INTEGRITY_JOIN_PASS: exact integrity bytes joined to task, source span and Lean environment");

    let mut bad = run.clone();
    bad["task"]["repository_commit"] = json!("0".repeat(40));
    bad["environment"]["repository_commit"] = json!("0".repeat(40));
    must_reject("repository commit substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["task"]["repository_branch"] = json!("other-branch");
    must_reject("repository branch substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["task"]["source_path"] = json!("Synthetic/Neighbor.lean");
    must_reject("source file substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["task"]["start_line"] = json!(13);
    must_reject("source span substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["task"]["lean_version"] = json!("leanprover/lean4:v4.other");
    bad["environment"]["lean_version"] = json!("leanprover/lean4:v4.other");
    must_reject("Lean toolchain substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["environment"]["lake_manifest_sha256"] = json!("0".repeat(64));
    must_reject("Lake manifest substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["task"]["sorry_id"] = json!("different-sorry-id");
    must_reject("task ID substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["task_integrity"]["receipt_sha256"] = json!("0".repeat(64));
    must_reject("integrity receipt digest substitution", bad, &integrity_bytes)?;

    let mut bad = run.clone();
    bad["task_integrity"]["integrity_status"] = json!("pending");
    must_reject("pending target integrity", bad, &integrity_bytes)?;

    println!("PITB_SORRYDB_INTEGRITY_SELF_TEST_PASS");
    Ok(())
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let Some(action) = args.next() else {
        bail!("expected --verify RUN_RECEIPT INTEGRITY_RECEIPT, --self-test RUN_FIXTURE INTEGRITY_FIXTURE, or --help");
    };
    if matches!(action.as_str(), "--help" | "-h") {
        return Ok(Mode::Help);
    }
    let run_path = PathBuf::from(args.next().context("missing SorryDB run receipt path")?);
    let integrity_path = PathBuf::from(args.next().context("missing benchmark integrity receipt path")?);
    if args.next().is_some() {
        bail!("too many command arguments");
    }
    match action.as_str() {
        "--verify" => Ok(Mode::Verify { run_path, integrity_path }),
        "--self-test" => Ok(Mode::SelfTest { run_path, integrity_path }),
        _ => bail!("unknown action {action}"),
    }
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: verify_sorrydb_run_integrity --verify RUN_RECEIPT INTEGRITY_RECEIPT | --self-test RUN_FIXTURE INTEGRITY_FIXTURE");
            Ok(())
        }
        Mode::SelfTest { run_path, integrity_path } => self_test(&run_path, &integrity_path),
        Mode::Verify { run_path, integrity_path } => {
            let run_bytes = fs::read(&run_path).with_context(|| format!("read run receipt {}", run_path.display()))?;
            let integrity_bytes =
                fs::read(&integrity_path).with_context(|| format!("read integrity receipt {}", integrity_path.display()))?;
            let summary = verify_pair(&run_bytes, &integrity_bytes)?;
            println!(
                "PITB_SORRYDB_INTEGRITY_PASS: task={} snapshot={} commit={} status={}",
                summary.source_task_id, summary.snapshot_id, summary.repository_commit, summary.integrity_status
            );
            println!("PITB_SORRYDB_INTEGRITY_SHA256: {}", summary.receipt_sha256);
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_SORRYDB_INTEGRITY_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}
