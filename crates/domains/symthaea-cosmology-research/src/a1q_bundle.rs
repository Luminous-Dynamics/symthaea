// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1Q evidence-bundle integrity qualifier.
//!
//! This program performs no cosmology. It proves that the fixed-point evidence
//! artifacts produced by A0/A1P/A1R/A1N and the exact software subject belong
//! to one coherent execution DAG. Bundle integrity is orthogonal to whether
//! the frozen reproduction verdict is PASS or NEGATIVE.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_JSON_BYTES: u64 = 4 * 1024 * 1024;
const MAX_BINARY_BYTES: u64 = 256 * 1024 * 1024;
const PROTOCOL: &str = "DE-001A1Q-BUNDLE-INTEGRITY-v1";
const AUTHORITY: &str = "evidence-bundle-integrity-only";
const NIXPKGS_REVISION: &str = "9ae611a455b90cf061d8f332b977e387bda8e1ca";
const CHECKOUT_COMMIT: &str = "11bd71901bbe5b1630ceea73d27597364c9af683";
const RUST_TOOLCHAIN_COMMIT: &str = "ebb3d1676050bfd0971c36c1e215b5751473994d";
const INSTALL_NIX_COMMIT: &str = "8aa03977d8d733052d78f4e008a241fd1dbf36b3";
const UPLOAD_ARTIFACT_COMMIT: &str = "ea165f8d65b6e75b540449e92b4886f43607fa02";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    schema_version: u32,
    protocol: String,
    scientific_claim: String,
    authority: String,
    subject_head: String,
    subject_tree: String,
    base: String,
    rust: String,
    nixpkgs_revision: String,
    action_commits: ActionCommits,
    cargo_lock_sha256: String,
    binary_sha256: BinaryHashes,
    a0_artifact_store: A0Store,
    receipt_sha256: ReceiptHashes,
    manifest_sha256: ManifestHashes,
    workflow: WorkflowState,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionCommits {
    checkout: String,
    rust_toolchain: String,
    install_nix: String,
    upload_artifact: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BinaryHashes {
    a0: String,
    a1p: String,
    a1r: String,
    a1n: String,
    a1q: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct A0Store {
    path: String,
    nar_hash: String,
    closure_size: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptHashes {
    a0: String,
    a1p: String,
    a1r_primary: String,
    a1r_refined: String,
    a1n: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestHashes {
    a1r_primary: String,
    a1r_refined: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkflowState {
    run_id: String,
    attempt: String,
    software_outcome: String,
    artifact_outcome: String,
    chain_outcome: String,
    chain_exit_code: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Prediction {
    z: f64,
    observable: String,
    observed: f64,
    predicted: f64,
    residual: f64,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    bundle_sha256: String,
    subject_head: String,
    subject_tree: String,
    reproduction_verdict: String,
    cargo_lock_sha256: String,
    binary_sha256: BinaryHashes,
    a0_nar_hash: String,
    a0_receipt_sha256: String,
    a1p_receipt_sha256: String,
    a1r_primary_receipt_sha256: String,
    a1r_refined_receipt_sha256: String,
    a1n_receipt_sha256: String,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    error: String,
}

struct Inputs {
    bundle: PathBuf,
    a0: PathBuf,
    a1p: PathBuf,
    primary_manifest: PathBuf,
    refined_manifest: PathBuf,
    primary_receipt: PathBuf,
    refined_receipt: PathBuf,
    a1n: PathBuf,
    cargo_lock: PathBuf,
    a0_binary: PathBuf,
    a1p_binary: PathBuf,
    a1r_binary: PathBuf,
    a1n_binary: PathBuf,
    a1q_binary: PathBuf,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_regular_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{}: size {} exceeds limit {max_bytes}",
            path.display(),
            metadata.len()
        ));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: size changed while reading", path.display()));
    }
    Ok(bytes)
}

fn hash_regular_file(path: &Path, max_bytes: u64) -> Result<String, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{}: size {} exceeds limit {max_bytes}",
            path.display(),
            metadata.len()
        ));
    }
    let mut file = File::open(path).map_err(|error| format!("{}: open failed: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("{}: read failed: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        total += read as u64;
        if total > max_bytes {
            return Err(format!("{}: file grew beyond size limit", path.display()));
        }
        hasher.update(&buffer[..read]);
    }
    if total != metadata.len() {
        return Err(format!("{}: size changed while hashing", path.display()));
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn parse_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("invalid {label} JSON: {error}"))
}

fn require_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field {key:?}"))
}

fn require_array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or non-array field {key:?}"))
}

fn require_protocol(
    value: &Value,
    protocol: &str,
    verdicts: &[&str],
    authority: Option<&str>,
) -> Result<(), String> {
    if require_str(value, "protocol")? != protocol
        || require_str(value, "scientific_claim")? != "NONE"
        || !verdicts.contains(&require_str(value, "verdict")?)
    {
        return Err(format!("unexpected identity for protocol {protocol}"));
    }
    if let Some(expected) = authority {
        if require_str(value, "authority")? != expected {
            return Err(format!("unexpected authority for protocol {protocol}"));
        }
    }
    Ok(())
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
    let text = String::from_utf8(output.stdout)
        .map_err(|error| format!("{program} output is not UTF-8: {error}"))?;
    Ok(text.trim().to_owned())
}

fn validate_bundle_identity(bundle: &Bundle) -> Result<(), String> {
    if bundle.schema_version != 1
        || bundle.protocol != "DE-001A1Q-PREQUALIFICATION-BUNDLE-v1"
        || bundle.scientific_claim != "NONE"
        || bundle.authority != AUTHORITY
    {
        return Err("invalid prequalification bundle identity".into());
    }
    if !is_lower_hex(&bundle.subject_head, 40) || !is_lower_hex(&bundle.subject_tree, 40) {
        return Err("bundle subject HEAD/TREE must be 40-character lowercase Git object IDs".into());
    }
    if !bundle.base.is_empty() && !is_lower_hex(&bundle.base, 40) {
        return Err("bundle base must be empty or a 40-character lowercase Git object ID".into());
    }
    if bundle.rust != "1.96.0" || bundle.nixpkgs_revision != NIXPKGS_REVISION {
        return Err("bundle toolchain identity is not the frozen DE-001A identity".into());
    }
    if bundle.action_commits.checkout != CHECKOUT_COMMIT
        || bundle.action_commits.rust_toolchain != RUST_TOOLCHAIN_COMMIT
        || bundle.action_commits.install_nix != INSTALL_NIX_COMMIT
        || bundle.action_commits.upload_artifact != UPLOAD_ARTIFACT_COMMIT
    {
        return Err("bundle GitHub Action identities are not frozen as expected".into());
    }
    for (name, digest) in [
        ("cargo_lock_sha256", bundle.cargo_lock_sha256.as_str()),
        ("binary.a0", bundle.binary_sha256.a0.as_str()),
        ("binary.a1p", bundle.binary_sha256.a1p.as_str()),
        ("binary.a1r", bundle.binary_sha256.a1r.as_str()),
        ("binary.a1n", bundle.binary_sha256.a1n.as_str()),
        ("binary.a1q", bundle.binary_sha256.a1q.as_str()),
        ("receipt.a0", bundle.receipt_sha256.a0.as_str()),
        ("receipt.a1p", bundle.receipt_sha256.a1p.as_str()),
        ("receipt.a1r_primary", bundle.receipt_sha256.a1r_primary.as_str()),
        ("receipt.a1r_refined", bundle.receipt_sha256.a1r_refined.as_str()),
        ("receipt.a1n", bundle.receipt_sha256.a1n.as_str()),
        ("manifest.a1r_primary", bundle.manifest_sha256.a1r_primary.as_str()),
        ("manifest.a1r_refined", bundle.manifest_sha256.a1r_refined.as_str()),
    ] {
        if !is_lower_hex(digest, 64) {
            return Err(format!("{name} must be lowercase SHA-256"));
        }
    }
    if !bundle.a0_artifact_store.path.starts_with("/nix/store/")
        || !bundle.a0_artifact_store.nar_hash.starts_with("sha256-")
        || bundle.a0_artifact_store.nar_hash.len() <= "sha256-".len()
        || bundle.a0_artifact_store.closure_size.parse::<u64>().ok().filter(|size| *size > 0).is_none()
    {
        return Err("invalid A0 Nix store identity".into());
    }
    if bundle.workflow.run_id.parse::<u64>().ok().filter(|value| *value > 0).is_none()
        || bundle.workflow.attempt.parse::<u64>().ok().filter(|value| *value > 0).is_none()
        || bundle.workflow.software_outcome != "success"
        || bundle.workflow.artifact_outcome != "success"
        || !matches!(bundle.workflow.chain_exit_code.as_str(), "0" | "1")
    {
        return Err("invalid workflow state in prequalification bundle".into());
    }
    Ok(())
}

fn validate_hash(expected: &str, path: &Path, max_bytes: u64, label: &str) -> Result<(), String> {
    let actual = hash_regular_file(path, max_bytes)?;
    if actual != expected {
        return Err(format!("{label} SHA-256 mismatch: expected {expected}, got {actual}"));
    }
    Ok(())
}

fn validate_git_subject(bundle: &Bundle) -> Result<(), String> {
    let head = run_text("git", &["rev-parse", "HEAD"])?;
    let tree = run_text("git", &["rev-parse", "HEAD^{tree}"])?;
    if head != bundle.subject_head || tree != bundle.subject_tree {
        return Err(format!(
            "bundle subject does not match live checkout: head={head} tree={tree}"
        ));
    }
    Ok(())
}

fn validate_nix_store(bundle: &Bundle) -> Result<(), String> {
    let path = bundle.a0_artifact_store.path.as_str();
    let nar_output = run_text("nix", &["path-info", "--nar-hash", path])?;
    let nar_hash = nar_output
        .split_whitespace()
        .last()
        .ok_or_else(|| "nix path-info --nar-hash returned no fields".to_owned())?;
    if nar_hash != bundle.a0_artifact_store.nar_hash {
        return Err(format!(
            "A0 NAR hash mismatch: expected {}, got {nar_hash}",
            bundle.a0_artifact_store.nar_hash
        ));
    }
    let size_output = run_text("nix", &["path-info", "-S", path])?;
    let closure_size = size_output
        .split_whitespace()
        .last()
        .ok_or_else(|| "nix path-info -S returned no fields".to_owned())?;
    if closure_size != bundle.a0_artifact_store.closure_size {
        return Err(format!(
            "A0 closure-size mismatch: expected {}, got {closure_size}",
            bundle.a0_artifact_store.closure_size
        ));
    }
    Ok(())
}

fn prediction_digest(receipt: &Value) -> Result<String, String> {
    let predictions: Vec<Prediction> = serde_json::from_value(
        receipt
            .get("predictions")
            .cloned()
            .ok_or_else(|| "A1R receipt missing predictions".to_owned())?,
    )
    .map_err(|error| format!("invalid A1R predictions: {error}"))?;
    let bytes = serde_json::to_vec(&predictions)
        .map_err(|error| format!("prediction serialization failed: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn validate_a0(a0: &Value) -> Result<String, String> {
    require_protocol(a0, "DE-001A0-BYTE-INTEGRITY-v1", &["PASS"], None)?;
    if !require_array(a0, "errors")?.is_empty() {
        return Err("A0 PASS receipt contains errors".into());
    }
    let artifacts = require_array(a0, "artifacts")?;
    let matching: Vec<_> = artifacts
        .iter()
        .filter(|artifact| artifact.get("role").and_then(Value::as_str) == Some("reference-bestfit-text"))
        .collect();
    if matching.len() != 1 {
        return Err("A0 receipt must contain exactly one reference-bestfit-text artifact".into());
    }
    let artifact = matching[0];
    if artifact.get("status").and_then(Value::as_str) != Some("PASS") {
        return Err("A0 best-fit artifact did not PASS".into());
    }
    let expected = artifact
        .get("expected_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "A0 best-fit artifact lacks expected_sha256".to_owned())?;
    let actual = artifact
        .get("actual_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "A0 best-fit artifact lacks actual_sha256".to_owned())?;
    if !is_lower_hex(expected, 64) || expected != actual {
        return Err("A0 best-fit artifact digest is invalid or inconsistent".into());
    }
    Ok(actual.to_owned())
}

fn validate_chain_outcome(reproduction_verdict: &str, workflow: &WorkflowState) -> Result<(), String> {
    match reproduction_verdict {
        "PASS" if workflow.chain_exit_code == "0" && workflow.chain_outcome == "success" => Ok(()),
        "NEGATIVE" if workflow.chain_exit_code == "1" && workflow.chain_outcome == "failure" => Ok(()),
        "PASS" | "NEGATIVE" => Err(format!(
            "workflow chain outcome/exit code does not match reproduction verdict {reproduction_verdict:?}"
        )),
        other => Err(format!("unsupported reproduction verdict {other:?}")),
    }
}

fn execute(inputs: &Inputs) -> Result<Receipt, String> {
    let bundle_bytes = read_regular_file(&inputs.bundle, MAX_JSON_BYTES)?;
    let bundle_sha256 = sha256_hex(&bundle_bytes);
    let bundle: Bundle = serde_json::from_slice(&bundle_bytes)
        .map_err(|error| format!("invalid prequalification bundle JSON: {error}"))?;
    validate_bundle_identity(&bundle)?;
    validate_git_subject(&bundle)?;

    validate_hash(&bundle.cargo_lock_sha256, &inputs.cargo_lock, MAX_JSON_BYTES, "Cargo.lock")?;
    for (expected, path, label) in [
        (&bundle.binary_sha256.a0, &inputs.a0_binary, "A0 binary"),
        (&bundle.binary_sha256.a1p, &inputs.a1p_binary, "A1P binary"),
        (&bundle.binary_sha256.a1r, &inputs.a1r_binary, "A1R binary"),
        (&bundle.binary_sha256.a1n, &inputs.a1n_binary, "A1N binary"),
        (&bundle.binary_sha256.a1q, &inputs.a1q_binary, "A1Q binary"),
    ] {
        validate_hash(expected, path, MAX_BINARY_BYTES, label)?;
    }
    validate_nix_store(&bundle)?;

    let a0_bytes = read_regular_file(&inputs.a0, MAX_JSON_BYTES)?;
    let a1p_bytes = read_regular_file(&inputs.a1p, MAX_JSON_BYTES)?;
    let primary_manifest_bytes = read_regular_file(&inputs.primary_manifest, MAX_JSON_BYTES)?;
    let refined_manifest_bytes = read_regular_file(&inputs.refined_manifest, MAX_JSON_BYTES)?;
    let primary_receipt_bytes = read_regular_file(&inputs.primary_receipt, MAX_JSON_BYTES)?;
    let refined_receipt_bytes = read_regular_file(&inputs.refined_receipt, MAX_JSON_BYTES)?;
    let a1n_bytes = read_regular_file(&inputs.a1n, MAX_JSON_BYTES)?;

    let a0_hash = sha256_hex(&a0_bytes);
    let a1p_hash = sha256_hex(&a1p_bytes);
    let primary_manifest_hash = sha256_hex(&primary_manifest_bytes);
    let refined_manifest_hash = sha256_hex(&refined_manifest_bytes);
    let primary_receipt_hash = sha256_hex(&primary_receipt_bytes);
    let refined_receipt_hash = sha256_hex(&refined_receipt_bytes);
    let a1n_hash = sha256_hex(&a1n_bytes);

    for (expected, actual, label) in [
        (bundle.receipt_sha256.a0.as_str(), a0_hash.as_str(), "A0 receipt"),
        (bundle.receipt_sha256.a1p.as_str(), a1p_hash.as_str(), "A1P receipt"),
        (
            bundle.receipt_sha256.a1r_primary.as_str(),
            primary_receipt_hash.as_str(),
            "primary A1R receipt",
        ),
        (
            bundle.receipt_sha256.a1r_refined.as_str(),
            refined_receipt_hash.as_str(),
            "refined A1R receipt",
        ),
        (bundle.receipt_sha256.a1n.as_str(), a1n_hash.as_str(), "A1N receipt"),
        (
            bundle.manifest_sha256.a1r_primary.as_str(),
            primary_manifest_hash.as_str(),
            "primary A1R manifest",
        ),
        (
            bundle.manifest_sha256.a1r_refined.as_str(),
            refined_manifest_hash.as_str(),
            "refined A1R manifest",
        ),
    ] {
        if expected != actual {
            return Err(format!("{label} digest disagrees with prequalification bundle"));
        }
    }

    let a0 = parse_json(&a0_bytes, "A0 receipt")?;
    let a1p = parse_json(&a1p_bytes, "A1P receipt")?;
    let primary = parse_json(&primary_receipt_bytes, "primary A1R receipt")?;
    let refined = parse_json(&refined_receipt_bytes, "refined A1R receipt")?;
    let a1n = parse_json(&a1n_bytes, "A1N receipt")?;

    let bestfit_sha256 = validate_a0(&a0)?;

    require_protocol(
        &a1p,
        "DE-001A1P-POINT-BINDING-v1",
        &["PASS"],
        Some("parameter-provenance-binding-only"),
    )?;
    if require_str(&a1p, "a1r_manifest_sha256")? != primary_manifest_hash
        || require_str(&a1p, "a0_receipt_sha256")? != a0_hash
        || require_str(&a1p, "bestfit_sha256")? != bestfit_sha256
        || require_str(&a1p, "bestfit_role")? != "reference-bestfit-text"
        || require_array(&a1p, "bindings")?.iter().any(|binding| {
            binding.get("exact_f64_match").and_then(Value::as_bool) != Some(true)
        })
    {
        return Err("A1P receipt is not coherently bound to A0 and the primary A1R manifest".into());
    }

    for (receipt, manifest_hash, label) in [
        (&primary, primary_manifest_hash.as_str(), "primary"),
        (&refined, refined_manifest_hash.as_str(), "refined"),
    ] {
        require_protocol(
            receipt,
            "DE-001A1R-RUST-ORACLE-v1",
            &["PASS", "NEGATIVE"],
            Some("fixed-point-reproduction-sanity-only"),
        )?;
        if require_str(receipt, "point_manifest_sha256")? != manifest_hash
            || require_str(receipt, "a0_receipt_sha256")? != a0_hash
        {
            return Err(format!("{label} A1R receipt is not bound to the supplied manifest/A0 receipt"));
        }
    }
    let reproduction_verdict = require_str(&primary, "verdict")?.to_owned();
    if require_str(&refined, "verdict")? != reproduction_verdict {
        return Err("primary/refined A1R reproduction verdicts disagree".into());
    }
    validate_chain_outcome(&reproduction_verdict, &bundle.workflow)?;

    require_protocol(
        &a1n,
        "DE-001A1N-NUMERICAL-CONVERGENCE-v1",
        &["PASS"],
        Some("numerical-convergence-qualification-only"),
    )?;
    if require_str(&a1n, "primary_manifest_sha256")? != primary_manifest_hash
        || require_str(&a1n, "refined_manifest_sha256")? != refined_manifest_hash
        || require_str(&a1n, "primary_receipt_sha256")? != primary_receipt_hash
        || require_str(&a1n, "refined_receipt_sha256")? != refined_receipt_hash
        || require_str(&a1n, "a0_receipt_sha256")? != a0_hash
    {
        return Err("A1N receipt does not bind the supplied A1R manifests/receipts/A0 subject".into());
    }
    if require_str(&a1n, "primary_prediction_sha256")? != prediction_digest(&primary)?
        || require_str(&a1n, "refined_prediction_sha256")? != prediction_digest(&refined)?
    {
        return Err("A1N prediction-vector digests do not match the bound A1R receipts".into());
    }

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: "NONE",
        authority: AUTHORITY,
        bundle_sha256,
        subject_head: bundle.subject_head,
        subject_tree: bundle.subject_tree,
        reproduction_verdict,
        cargo_lock_sha256: bundle.cargo_lock_sha256,
        binary_sha256: bundle.binary_sha256,
        a0_nar_hash: bundle.a0_artifact_store.nar_hash,
        a0_receipt_sha256: a0_hash,
        a1p_receipt_sha256: a1p_hash,
        a1r_primary_receipt_sha256: primary_receipt_hash,
        a1r_refined_receipt_sha256: refined_receipt_hash,
        a1n_receipt_sha256: a1n_hash,
    })
}

fn write_json<T: Serialize>(value: &T) -> io::Result<()> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    writeln!(&mut lock)?;
    Ok(())
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 14 {
        return Err("usage: de001a-a1q-bundle PREQUAL.json A0.json A1P.json PRIMARY_POINT.json REFINED_POINT.json PRIMARY_A1R.json REFINED_A1R.json A1N.json Cargo.lock A0_BIN A1P_BIN A1R_BIN A1N_BIN A1Q_BIN".into());
    }
    let inputs = Inputs {
        bundle: PathBuf::from(&args[0]),
        a0: PathBuf::from(&args[1]),
        a1p: PathBuf::from(&args[2]),
        primary_manifest: PathBuf::from(&args[3]),
        refined_manifest: PathBuf::from(&args[4]),
        primary_receipt: PathBuf::from(&args[5]),
        refined_receipt: PathBuf::from(&args[6]),
        a1n: PathBuf::from(&args[7]),
        cargo_lock: PathBuf::from(&args[8]),
        a0_binary: PathBuf::from(&args[9]),
        a1p_binary: PathBuf::from(&args[10]),
        a1r_binary: PathBuf::from(&args[11]),
        a1n_binary: PathBuf::from(&args[12]),
        a1q_binary: PathBuf::from(&args[13]),
    };
    let receipt = execute(&inputs)?;
    write_json(&receipt).map_err(|error| format!("failed to write A1Q receipt: {error}"))?;
    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            let receipt = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: "NONE",
                authority: AUTHORITY,
                error,
            };
            let _ = write_json(&receipt);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workflow(outcome: &str, code: &str) -> WorkflowState {
        WorkflowState {
            run_id: "1".into(),
            attempt: "1".into(),
            software_outcome: "success".into(),
            artifact_outcome: "success".into(),
            chain_outcome: outcome.into(),
            chain_exit_code: code.into(),
        }
    }

    #[test]
    fn digest_validation_is_lowercase_and_exact_width() {
        assert!(is_lower_hex(&"a".repeat(64), 64));
        assert!(!is_lower_hex(&"A".repeat(64), 64));
        assert!(!is_lower_hex(&"a".repeat(63), 64));
    }

    #[test]
    fn pass_and_negative_have_distinct_workflow_outcomes() {
        assert!(validate_chain_outcome("PASS", &workflow("success", "0")).is_ok());
        assert!(validate_chain_outcome("NEGATIVE", &workflow("failure", "1")).is_ok());
        assert!(validate_chain_outcome("PASS", &workflow("failure", "1")).is_err());
        assert!(validate_chain_outcome("NEGATIVE", &workflow("success", "0")).is_err());
    }
}
