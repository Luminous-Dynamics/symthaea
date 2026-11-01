#[path = "../benchmark_integrity.rs"]
mod benchmark_integrity;

use std::{
    env,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use ed25519_dalek::{Signer, SigningKey};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

enum Mode {
    Help,
    Verify { run_path: PathBuf, integrity_path: PathBuf, evidence_root: PathBuf, reviewer_registry_path: PathBuf, expected_registry_sha256: String },
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
    if !integrity.reviewer_signature_verified {
        bail!("benchmark target has no verified Ed25519 reviewer attestation");
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

    if integrity.cohort_id != "SorryDB_2601" {
        bail!("SorryDB run cannot bind an integrity receipt from another cohort");
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

fn verify_pair(
    run_bytes: &[u8],
    integrity_bytes: &[u8],
    evidence_root: &Path,
    reviewer_registry_bytes: &[u8],
    expected_registry_sha256: &str,
) -> Result<benchmark_integrity::IntegritySummary> {
    let run: Value = serde_json::from_slice(run_bytes).context("parse SorryDB run receipt")?;
    let expected = string(&run, "/task_integrity/receipt_sha256")?;
    let run_registry_sha = string(&run, "/task_integrity/reviewer_key_registry_sha256")?;
    if run_registry_sha != expected_registry_sha256 {
        bail!("run receipt reviewer registry pin differs from the externally supplied registry digest");
    }
    let integrity_value: Value =
        serde_json::from_slice(integrity_bytes).context("parse benchmark integrity receipt")?;
    if string(&integrity_value, "/reviewer_key_registry_sha256")? != expected_registry_sha256 {
        bail!("benchmark integrity receipt reviewer registry pin differs from run receipt");
    }
    let integrity = benchmark_integrity::verify_bytes_with_evidence_root_and_reviewer_registry(
        integrity_bytes,
        Some(expected),
        evidence_root,
        reviewer_registry_bytes,
        expected_registry_sha256,
    )?;
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

fn write_evidence_artifact(
    evidence_root: &Path,
    relative_path: &str,
    bytes: &[u8],
) -> Result<Value> {
    let path = evidence_root.join(relative_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create evidence directory {}", parent.display()))?;
    }
    fs::write(&path, bytes).with_context(|| format!("write evidence artifact {}", path.display()))?;
    Ok(json!({
        "path":relative_path,
        "sha256":sha256_hex(bytes)
    }))
}

fn make_eligible_integrity(
    mut value: Value,
    run: &Value,
    evidence_root: &Path,
    signing_key: &SigningKey,
    reviewer_registry_sha256: &str,
) -> Result<Value> {
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

    let mut source_bytes = vec![b'\n'; 11];
    source_bytes.extend_from_slice(b"     sorry\n");
    let informal_bytes = b"A synthetic proposition used only to test the evidence bundle.\n";
    let lean_bytes = b"synthetic-lean-binary-v0";
    let lake_bytes = b"{\"version\":1}\n";
    let root_module_bytes = b"synthetic-root-module-v0";
    let elaboration_bytes = br#"{"status":"passed","target":"synthetic"}"#;
    let used_axioms_bytes = b"[]";
    let allowlist_bytes = br#"["Classical.choice","Quot.sound","propext"]"#;
    let counterexample_bytes = br#"{"status":"not_found","method":"synthetic"}"#;
    let vacuity_bytes = br#"{"status":"no_counterexample_found","method":"synthetic"}"#;

    value["target"]["source_file_sha256"] = json!(sha256_hex(&source_bytes));
    value["target"]["source_span_sha256"] = json!(sha256_hex(b"sorry"));
    value["target"]["expected_informal_statement_sha256"] = json!(sha256_hex(informal_bytes));
    value["environment"]["lean_binary_sha256"] = json!(sha256_hex(lean_bytes));
    value["environment"]["lake_manifest_sha256"] = json!(sha256_hex(lake_bytes));
    value["environment"]["root_module_sha256"] = json!(sha256_hex(root_module_bytes));

    value["elaboration"]["status"] = json!("passed");
    value["elaboration"]["checked_statement_sha256"] = value["target"]["statement_sha256"].clone();
    value["elaboration"]["receipt_sha256"] = json!(sha256_hex(elaboration_bytes));
    value["counterexample_audit"]["receipt_sha256"] = json!(sha256_hex(counterexample_bytes));
    value["vacuity_audit"]["receipt_sha256"] = json!(sha256_hex(vacuity_bytes));
    value["semantic_review"]["reviewer_attestation_sha256"] = json!(sha256_hex(reviewer_bytes));

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
    value["vacuity_audit"]["status"] = json!("no_counterexample_found");
    value["vacuity_audit"]["method_id"] = json!("synthetic-vacuity-check");
    value["semantic_review"]["status"] = json!("approved");
    value["semantic_review"]["review_scope"] = json!("domain_expert_review");
    value["semantic_review"]["reviewer_identity"] = json!("synthetic-reviewer-not-real");
    value["reviewer_key_registry_sha256"] = json!(reviewer_registry_sha256);

    let mut attestation = json!({
        "schema_version":"pitb.reviewer-attestation.v0",
        "reviewer_identity":"synthetic-reviewer-not-real",
        "key_id":"synthetic-reviewer-key-v0",
        "audit_id":value["audit_id"],
        "task_id_sha256":value["dataset"]["task_id"],
        "statement_sha256":value["target"]["statement_sha256"],
        "expected_informal_statement_sha256":value["target"]["expected_informal_statement_sha256"],
        "review_scope":"domain_expert_review",
        "decision":"approved"
    });
    let signed_payload = serde_json::to_vec(&attestation)?;
    let signature = signing_key.sign(&signed_payload);
    attestation["signature_ed25519_hex"] = json!(hex::encode(signature.to_bytes()));
    let reviewer_bytes = json_bytes(&attestation)?;
    value["semantic_review"]["reviewer_attestation_sha256"] = json!(sha256_hex(&reviewer_bytes));

    value["evidence_artifacts"] = json!({
        "source_file":write_evidence_artifact(evidence_root,"source/Synthetic/Example.lean",&source_bytes)?,
        "informal_statement":write_evidence_artifact(evidence_root,"spec/problem.txt",informal_bytes)?,
        "lean_binary":write_evidence_artifact(evidence_root,"toolchain/lean",lean_bytes)?,
        "lake_manifest":write_evidence_artifact(evidence_root,"project/lake-manifest.json",lake_bytes)?,
        "root_module":write_evidence_artifact(evidence_root,"project/Synthetic.olean",root_module_bytes)?,
        "elaboration_receipt":write_evidence_artifact(evidence_root,"receipts/elaboration.json",elaboration_bytes)?,
        "used_axioms_manifest":write_evidence_artifact(evidence_root,"manifests/used-axioms.json",used_axioms_bytes)?,
        "allowlist_manifest":write_evidence_artifact(evidence_root,"manifests/allowlist.json",allowlist_bytes)?,
        "counterexample_receipt":write_evidence_artifact(evidence_root,"receipts/counterexample.json",counterexample_bytes)?,
        "vacuity_receipt":write_evidence_artifact(evidence_root,"receipts/vacuity.json",vacuity_bytes)?,
        "reviewer_attestation":write_evidence_artifact(evidence_root,"review/reviewer-attestation.json",&reviewer_bytes)?
    });
    value["integrity_status"] = json!("eligible");
    Ok(value)
}

fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn must_reject(label: &str, run: Value, integrity_bytes: &[u8], evidence_root: &Path, registry_bytes: &[u8], registry_sha256: &str) -> Result<()> {
    let run_bytes = json_bytes(&run)?;
    if verify_pair(&run_bytes, integrity_bytes, evidence_root, registry_bytes, registry_sha256).is_ok() {
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
    let evidence_root = env::temp_dir().join(format!("pitb-sorrydb-evidence-{}", std::process::id()));
    if evidence_root.exists() {
        fs::remove_dir_all(&evidence_root)
            .with_context(|| format!("clear stale evidence root {}", evidence_root.display()))?;
    }
    fs::create_dir_all(&evidence_root)
        .with_context(|| format!("create evidence root {}", evidence_root.display()))?;

    let signing_key = SigningKey::from_bytes(&[42_u8; 32]);
    let registry = json!({
        "schema_version":"pitb.reviewer-key-registry.v0",
        "registry_id":"synthetic-test-registry-v0",
        "keys":[{
            "key_id":"synthetic-reviewer-key-v0",
            "reviewer_identity":"synthetic-reviewer-not-real",
            "public_key_ed25519_hex":hex::encode(signing_key.verifying_key().to_bytes()),
            "status":"active"
        }]
    });
    let registry_bytes = json_bytes(&registry)?;
    let registry_sha256 = sha256_hex(&registry_bytes);
    let registry_path = evidence_root.join("reviewer-key-registry.json");
    fs::write(&registry_path, &registry_bytes)
        .with_context(|| format!("write test reviewer registry {}", registry_path.display()))?;

    run["task_integrity"] = Value::Null;
    if verify_pair(&run_bytes, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256).is_ok() {
        bail!("unbound synthetic receipts unexpectedly passed as a measured run");
    }
    println!("PITB_SORRYDB_INTEGRITY_NEGATIVE_PASS: no-run fixture is not an eligible measured task");

    let eligible = make_eligible_integrity(integrity, &run, &evidence_root, &signing_key, &registry_sha256)?;
    let integrity_bytes = json_bytes(&eligible)?;
    let integrity_summary = benchmark_integrity::verify_bytes_with_evidence_root_and_reviewer_registry(
        &integrity_bytes,
        None,
        &evidence_root,
        &registry_bytes,
        &registry_sha256,
    )?;
    run["environment"]["lake_manifest_sha256"] =
        eligible["environment"]["lake_manifest_sha256"].clone();
    if integrity_summary.integrity_status != "eligible" {
        bail!("synthetic eligible integrity fixture did not derive eligible status");
    }
    if !integrity_summary.reviewer_signature_verified
        || integrity_summary.verified_evidence_artifact_count != 11
    {
        bail!("synthetic signed integrity fixture did not verify all eleven artifacts and reviewer signature");
    }
    run["evidence_class"] = json!("measured_run");
    run["task_integrity"] = json!({
        "receipt_sha256":sha256_hex(&integrity_bytes),
        "task_id_sha256":integrity_summary.task_id,
        "integrity_status":"eligible",
        "reviewer_key_registry_sha256":registry_sha256
    });
    let run_bytes = json_bytes(&run)?;
    verify_pair(&run_bytes, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;
    println!("PITB_SORRYDB_INTEGRITY_JOIN_PASS: exact integrity receipt, eleven evidence artifacts and Ed25519 reviewer signature joined");

    // Mutate the reviewer signature, update both content-addressed parent
    // references, and require the cryptographic verifier itself to reject it.
    let reviewer_artifact_path = evidence_root.join("review/reviewer-attestation.json");
    let original_attestation_bytes = fs::read(&reviewer_artifact_path)?;
    let mut bad_attestation: Value = serde_json::from_slice(&original_attestation_bytes)?;
    bad_attestation["signature_ed25519_hex"] = json!("00".repeat(64));
    let bad_attestation_bytes = json_bytes(&bad_attestation)?;
    fs::write(&reviewer_artifact_path, &bad_attestation_bytes)?;
    let mut bad_integrity: Value = serde_json::from_slice(&integrity_bytes)?;
    bad_integrity["semantic_review"]["reviewer_attestation_sha256"] =
        json!(sha256_hex(&bad_attestation_bytes));
    bad_integrity["evidence_artifacts"]["reviewer_attestation"]["sha256"] =
        json!(sha256_hex(&bad_attestation_bytes));
    let bad_integrity_bytes = json_bytes(&bad_integrity)?;
    let mut bad_run: Value = serde_json::from_slice(&run_bytes)?;
    bad_run["task_integrity"]["receipt_sha256"] = json!(sha256_hex(&bad_integrity_bytes));
    let bad_run_bytes = json_bytes(&bad_run)?;
    if verify_pair(&bad_run_bytes, &bad_integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256).is_ok() {
        bail!("signature mutation with updated content hashes unexpectedly passed reviewer authentication");
    }
    fs::write(&reviewer_artifact_path, &original_attestation_bytes)?;
    println!("PITB_SORRYDB_REVIEWER_NEGATIVE_PASS: invalid Ed25519 signature rejected after all parent hashes were updated");

    // Revocation is applied through the out-of-band registry. Updating the
    // receipt and run pins to the new registry must not reactivate a revoked key.
    let mut revoked_registry: Value = serde_json::from_slice(&registry_bytes)?;
    revoked_registry["keys"][0]["status"] = json!("revoked");
    let revoked_registry_bytes = json_bytes(&revoked_registry)?;
    let revoked_registry_sha256 = sha256_hex(&revoked_registry_bytes);
    let mut revoked_integrity: Value = serde_json::from_slice(&integrity_bytes)?;
    revoked_integrity["reviewer_key_registry_sha256"] = json!(revoked_registry_sha256);
    let revoked_integrity_bytes = json_bytes(&revoked_integrity)?;
    let mut revoked_run: Value = serde_json::from_slice(&run_bytes)?;
    revoked_run["task_integrity"]["receipt_sha256"] = json!(sha256_hex(&revoked_integrity_bytes));
    revoked_run["task_integrity"]["reviewer_key_registry_sha256"] = json!(revoked_registry_sha256);
    let revoked_run_bytes = json_bytes(&revoked_run)?;
    if verify_pair(&revoked_run_bytes, &revoked_integrity_bytes, &evidence_root, &revoked_registry_bytes, &revoked_registry_sha256).is_ok() {
        bail!("revoked reviewer key unexpectedly passed benchmark integrity admission");
    }
    println!("PITB_SORRYDB_REVIEWER_NEGATIVE_PASS: revoked reviewer key rejected despite updated registry and receipt pins");

    if verify_pair(&run_bytes, &integrity_bytes, &evidence_root, &registry_bytes, &"0".repeat(64)).is_ok() {
        bail!("mismatched external reviewer registry pin unexpectedly passed");
    }
    println!("PITB_SORRYDB_REVIEWER_NEGATIVE_PASS: externally pinned reviewer registry digest mismatch");


    let source_artifact = evidence_root.join("source/Synthetic/Example.lean");
    let original_source = fs::read(&source_artifact)?;
    fs::write(&source_artifact, b"tampered after receipt creation")?;
    if verify_pair(&run_bytes, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256).is_ok() {
        bail!("tampered source artifact unexpectedly passed evidence bundle verification");
    }
    fs::write(&source_artifact, &original_source)?;
    println!("PITB_SORRYDB_INTEGRITY_NEGATIVE_PASS: referenced source artifact byte tampering");

    let mut span_mutation: Value = serde_json::from_slice(&integrity_bytes)?;
    span_mutation["target"]["start_column"] = json!(4);
    let span_mutation_bytes = json_bytes(&span_mutation)?;
    let mut run_span_mutation: Value = serde_json::from_slice(&run_bytes)?;
    run_span_mutation["task_integrity"]["receipt_sha256"] = json!(sha256_hex(&span_mutation_bytes));
    let run_span_mutation_bytes = json_bytes(&run_span_mutation)?;
    if verify_pair(&run_span_mutation_bytes, &span_mutation_bytes, &evidence_root, &registry_bytes, &registry_sha256).is_ok() {
        bail!("source coordinate mutation did not invalidate the exact source-span digest");
    }
    println!("PITB_SORRYDB_INTEGRITY_NEGATIVE_PASS: source coordinates must select exact recorded source-span bytes");

    let escaped_path = evidence_root.join("source/escape.lean");
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let outside = evidence_root.parent().unwrap_or(&evidence_root).join("pitb-integrity-outside");
        fs::write(&outside, b"outside evidence root")?;
        symlink(&outside, &escaped_path)?;
        let mut escaped: Value = serde_json::from_slice(&integrity_bytes)?;
        escaped["evidence_artifacts"]["source_file"]["path"] = json!("source/escape.lean");
        let escaped_bytes = json_bytes(&escaped)?;
        if benchmark_integrity::verify_bytes_with_evidence_root_and_reviewer_registry(&escaped_bytes, None, &evidence_root, &registry_bytes, &registry_sha256).is_ok() {
            bail!("evidence bundle verifier accepted a symlink escaping the evidence root");
        }
        fs::remove_file(&escaped_path)?;
        fs::remove_file(&outside)?;
        println!("PITB_SORRYDB_INTEGRITY_NEGATIVE_PASS: evidence-root symlink escape");
    }

    let mut bad = run.clone();
    bad["task"]["repository_commit"] = json!("0".repeat(40));
    bad["environment"]["repository_commit"] = json!("0".repeat(40));
    must_reject("repository commit substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["task"]["repository_branch"] = json!("other-branch");
    must_reject("repository branch substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["task"]["source_path"] = json!("Synthetic/Neighbor.lean");
    must_reject("source file substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["task"]["start_line"] = json!(13);
    must_reject("source span substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["task"]["lean_version"] = json!("leanprover/lean4:v4.other");
    bad["environment"]["lean_version"] = json!("leanprover/lean4:v4.other");
    must_reject("Lean toolchain substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["environment"]["lake_manifest_sha256"] = json!("0".repeat(64));
    must_reject("Lake manifest substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["task"]["sorry_id"] = json!("different-sorry-id");
    must_reject("task ID substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["task_integrity"]["receipt_sha256"] = json!("0".repeat(64));
    must_reject("integrity receipt digest substitution", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    let mut bad = run.clone();
    bad["task_integrity"]["integrity_status"] = json!("pending");
    must_reject("pending target integrity", bad, &integrity_bytes, &evidence_root, &registry_bytes, &registry_sha256)?;

    println!("PITB_SORRYDB_INTEGRITY_SELF_TEST_PASS");
    fs::remove_dir_all(&evidence_root)
        .with_context(|| format!("remove evidence root {}", evidence_root.display()))?;
    Ok(())
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let Some(action) = args.next() else {
        bail!("expected --verify RUN_RECEIPT INTEGRITY_RECEIPT EVIDENCE_ROOT REVIEWER_REGISTRY EXPECTED_REGISTRY_SHA256, --self-test RUN_FIXTURE INTEGRITY_FIXTURE, or --help");
    };
    if matches!(action.as_str(), "--help" | "-h") {
        return Ok(Mode::Help);
    }
    let run_path = PathBuf::from(args.next().context("missing SorryDB run receipt path")?);
    let integrity_path = PathBuf::from(args.next().context("missing benchmark integrity receipt path")?);
    match action.as_str() {
        "--verify" => {
            let evidence_root = PathBuf::from(args.next().context("missing evidence root directory")?);
            let reviewer_registry_path = PathBuf::from(args.next().context("missing reviewer registry path")?);
            let expected_registry_sha256 = args.next().context("missing expected reviewer registry SHA-256")?;
            if args.next().is_some() {
                bail!("too many --verify arguments");
            }
            Ok(Mode::Verify { run_path, integrity_path, evidence_root, reviewer_registry_path, expected_registry_sha256 })
        }
        "--self-test" => {
            if args.next().is_some() {
                bail!("--self-test accepts only run and integrity fixture paths");
            }
            Ok(Mode::SelfTest { run_path, integrity_path })
        }
        _ => bail!("unknown action {action}"),
    }
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: verify_sorrydb_run_integrity --verify RUN_RECEIPT INTEGRITY_RECEIPT EVIDENCE_ROOT REVIEWER_REGISTRY EXPECTED_REGISTRY_SHA256 | --self-test RUN_FIXTURE INTEGRITY_FIXTURE");
            Ok(())
        }
        Mode::SelfTest { run_path, integrity_path } => self_test(&run_path, &integrity_path),
        Mode::Verify { run_path, integrity_path, evidence_root, reviewer_registry_path, expected_registry_sha256 } => {
            let run_bytes = fs::read(&run_path).with_context(|| format!("read run receipt {}", run_path.display()))?;
            let integrity_bytes =
                fs::read(&integrity_path).with_context(|| format!("read integrity receipt {}", integrity_path.display()))?;
            let registry_bytes = fs::read(&reviewer_registry_path)
                .with_context(|| format!("read reviewer registry {}", reviewer_registry_path.display()))?;
            let actual_registry_sha256 = sha256_hex(&registry_bytes);
            if actual_registry_sha256 != expected_registry_sha256 {
                bail!("reviewer registry file does not match externally pinned SHA-256");
            }
            let summary = verify_pair(&run_bytes, &integrity_bytes, &evidence_root, &registry_bytes, &expected_registry_sha256)?;
            println!(
                "PITB_SORRYDB_INTEGRITY_PASS: task={} snapshot={} commit={} status={}",
                summary.source_task_id, summary.snapshot_id, summary.repository_commit, summary.integrity_status
            );
            println!("PITB_SORRYDB_INTEGRITY_SHA256: {}", summary.receipt_sha256);
            println!("PITB_SORRYDB_EVIDENCE_ARTIFACTS_PASS: {}", summary.verified_evidence_artifact_count);
            println!("PITB_SORRYDB_REVIEWER_SIGNATURE_PASS: {}", summary.reviewer_signature_verified);
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
