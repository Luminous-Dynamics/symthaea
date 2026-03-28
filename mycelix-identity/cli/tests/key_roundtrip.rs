// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for CLI key file round-trip: keygen → write → read → sign → verify.
//!
//! These tests exercise the full file I/O pipeline using tempfiles,
//! ensuring the JSON key format is stable across all PQC algorithms.

use std::path::Path;
use std::process::Command;

fn cli_bin() -> String {
    // Build path to the CLI binary (cargo test builds it in the same target dir)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let target_dir = Path::new(manifest_dir).join("target").join("release");
    target_dir.join("mycelix-credential").to_string_lossy().to_string()
}

fn build_cli() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let status = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(manifest_dir)
        .status()
        .expect("Failed to build CLI");
    assert!(status.success(), "CLI build failed");
}

fn run_cli(args: &[&str]) -> std::process::Output {
    let bin = cli_bin();
    Command::new(&bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {} {:?}: {}", bin, args, e))
}

fn keygen_sign_verify(algorithm: &str, label: &str) {
    let dir = tempfile::tempdir().unwrap();
    let key_path = dir.path().join(format!("{}-key.json", label));
    let cred_path = dir.path().join("credential.json");
    let signed_path = dir.path().join(format!("{}-signed.json", label));

    // Write a test credential
    std::fs::write(&cred_path, r#"{"type":"TestCredential","value":42}"#).unwrap();

    // keygen
    let output = run_cli(&[
        "keygen",
        "--algorithm", algorithm,
        "--output", key_path.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "keygen failed for {}: {}",
        label, String::from_utf8_lossy(&output.stderr));
    assert!(key_path.exists(), "Key file not created for {}", label);

    // Verify key file is valid JSON with expected fields
    let key_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&key_path).unwrap()
    ).unwrap();
    assert!(key_json.get("algorithm").is_some(), "Missing 'algorithm' field");
    assert!(key_json.get("algorithm_id").is_some(), "Missing 'algorithm_id' field");
    assert!(key_json.get("public_key").is_some(), "Missing 'public_key' field");
    assert!(key_json.get("public_key_bytes").is_some(), "Missing 'public_key_bytes' field");
    assert!(key_json.get("secret_key_bytes").is_some(), "Missing 'secret_key_bytes' field");
    assert!(key_json.get("created_at").is_some(), "Missing 'created_at' field");

    // sign
    let output = run_cli(&[
        "sign",
        "--key", key_path.to_str().unwrap(),
        "--credential", cred_path.to_str().unwrap(),
        "--output", signed_path.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "sign failed for {}: {}",
        label, String::from_utf8_lossy(&output.stderr));
    assert!(signed_path.exists(), "Signed file not created for {}", label);

    // Verify signed credential has proof block
    let signed_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&signed_path).unwrap()
    ).unwrap();
    let proof = signed_json.get("proof").expect("Missing 'proof' in signed credential");
    assert!(proof.get("proofValue").is_some(), "Missing 'proofValue' in proof");
    assert!(proof.get("cryptosuite").is_some(), "Missing 'cryptosuite' in proof");

    // verify (full cryptographic verification with --key)
    let output = run_cli(&[
        "pqc-verify",
        "--credential", signed_path.to_str().unwrap(),
        "--key", key_path.to_str().unwrap(),
        "--verbose",
    ]);
    assert!(output.status.success(), "pqc-verify failed for {}: {}",
        label, String::from_utf8_lossy(&output.stderr));

    // verify (structure-only, no --key)
    let output = run_cli(&[
        "pqc-verify",
        "--credential", signed_path.to_str().unwrap(),
        "--verbose",
    ]);
    assert!(output.status.success(), "pqc-verify (structure-only) failed for {}: {}",
        label, String::from_utf8_lossy(&output.stderr));
}

// Build once before all tests
static BUILD_ONCE: std::sync::Once = std::sync::Once::new();

fn ensure_built() {
    BUILD_ONCE.call_once(|| build_cli());
}

#[test]
fn ed25519_key_roundtrip() {
    ensure_built();
    keygen_sign_verify("ed25519", "ed25519");
}

#[test]
fn mldsa65_key_roundtrip() {
    ensure_built();
    keygen_sign_verify("ml-dsa-65", "mldsa65");
}

#[test]
fn mldsa87_key_roundtrip() {
    ensure_built();
    keygen_sign_verify("ml-dsa-87", "mldsa87");
}

#[test]
fn hybrid_key_roundtrip() {
    ensure_built();
    keygen_sign_verify("hybrid-ed25519-ml-dsa-65", "hybrid");
}

#[test]
fn key_file_deterministic_fields() {
    ensure_built();
    let dir = tempfile::tempdir().unwrap();
    let key1 = dir.path().join("key1.json");
    let key2 = dir.path().join("key2.json");

    // Generate two keys with same algorithm
    run_cli(&["keygen", "--algorithm", "ed25519", "--output", key1.to_str().unwrap()]);
    run_cli(&["keygen", "--algorithm", "ed25519", "--output", key2.to_str().unwrap()]);

    let j1: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&key1).unwrap()).unwrap();
    let j2: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&key2).unwrap()).unwrap();

    // Same algorithm fields
    assert_eq!(j1["algorithm"], j2["algorithm"]);
    assert_eq!(j1["algorithm_id"], j2["algorithm_id"]);

    // Different keys (randomness)
    assert_ne!(j1["public_key"], j2["public_key"]);
    assert_ne!(j1["secret_key_bytes"], j2["secret_key_bytes"]);
}

#[test]
fn slh_dsa_key_roundtrip() {
    ensure_built();
    keygen_sign_verify("slh-dsa-sha2-128s", "slhdsa");
}

#[test]
fn hybrid_dual_key_verify() {
    ensure_built();
    let dir = tempfile::tempdir().unwrap();
    let ed_key = dir.path().join("ed-key.json");
    let pqc_key = dir.path().join("pqc-key.json");
    let cred_path = dir.path().join("credential.json");
    let signed_path = dir.path().join("hybrid-signed.json");

    std::fs::write(&cred_path, r#"{"type":"DualKeyTest","value":99}"#).unwrap();

    // Generate separate Ed25519 and ML-DSA-65 keys
    let output = run_cli(&["keygen", "--algorithm", "ed25519", "--output", ed_key.to_str().unwrap()]);
    assert!(output.status.success(), "Ed25519 keygen failed: {}", String::from_utf8_lossy(&output.stderr));

    let output = run_cli(&["keygen", "--algorithm", "ml-dsa-65", "--output", pqc_key.to_str().unwrap()]);
    assert!(output.status.success(), "ML-DSA-65 keygen failed: {}", String::from_utf8_lossy(&output.stderr));

    // Sign with hybrid-sign using both keys
    let output = run_cli(&[
        "hybrid-sign",
        "--ed25519-key", ed_key.to_str().unwrap(),
        "--pqc-key", pqc_key.to_str().unwrap(),
        "--credential", cred_path.to_str().unwrap(),
        "--output", signed_path.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "hybrid-sign failed: {}", String::from_utf8_lossy(&output.stderr));

    // Verify with dual-key path (--ed25519-key + --pqc-key)
    let output = run_cli(&[
        "pqc-verify",
        "--credential", signed_path.to_str().unwrap(),
        "--ed25519-key", ed_key.to_str().unwrap(),
        "--pqc-key", pqc_key.to_str().unwrap(),
        "--verbose",
    ]);
    assert!(output.status.success(), "Dual-key verify failed: {}", String::from_utf8_lossy(&output.stderr));

    // Also verify with single --key using the Ed25519 key (should detect hybrid and handle)
    let output = run_cli(&[
        "pqc-verify",
        "--credential", signed_path.to_str().unwrap(),
        "--verbose",
    ]);
    assert!(output.status.success(), "Structure-only verify of hybrid-signed failed: {}",
        String::from_utf8_lossy(&output.stderr));
}

#[test]
fn tampered_signature_rejected() {
    ensure_built();
    let dir = tempfile::tempdir().unwrap();
    let key_path = dir.path().join("key.json");
    let cred_path = dir.path().join("cred.json");
    let signed_path = dir.path().join("signed.json");

    std::fs::write(&cred_path, r#"{"test":"data"}"#).unwrap();
    run_cli(&["keygen", "--algorithm", "ed25519", "--output", key_path.to_str().unwrap()]);
    run_cli(&[
        "sign", "--key", key_path.to_str().unwrap(),
        "--credential", cred_path.to_str().unwrap(),
        "--output", signed_path.to_str().unwrap(),
    ]);

    // Tamper with the signed credential (change the payload)
    let mut signed: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&signed_path).unwrap()
    ).unwrap();
    signed["credential"]["test"] = serde_json::Value::String("TAMPERED".into());
    std::fs::write(&signed_path, serde_json::to_string_pretty(&signed).unwrap()).unwrap();

    // Verify should fail (non-zero exit code)
    let output = run_cli(&[
        "pqc-verify",
        "--credential", signed_path.to_str().unwrap(),
        "--key", key_path.to_str().unwrap(),
    ]);
    assert!(!output.status.success(), "Tampered credential should fail verification");
}
