use serde_json::Value;
use std::fs::{self, File};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

fn temp_file(suffix: &str) -> PathBuf {
    let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rea-evidence-verify-{}-{sequence}.{suffix}",
        std::process::id()
    ))
}

fn run_json(path: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rea-evidence-verify"))
        .arg("--json")
        .arg(path)
        .output()
        .expect("execute offline Evidence CLI")
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    ))
    .unwrap()
}

#[test]
fn json_success_separates_digests_and_redacts_paths_and_payloads() {
    let path = temp_file("json");
    let raw = serde_json::to_vec(&fixture()).unwrap();
    fs::write(&path, &raw).unwrap();

    let output = run_json(&path);
    assert_eq!(output.status.code(), Some(0));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["result"], "semantic_consistency_verified");
    assert_eq!(
        report["evidence_id"],
        "ev_a4e49bdca4cbf80c2ec8e4d7eba116af403bc33a4937ffb7fca00b6aab6590c4"
    );
    assert_ne!(report["raw_file_sha256"], report["semantic_sha256"]);
    assert_eq!(report["claim_ceiling"]["producer_authenticated"], false);

    let visible = String::from_utf8(output.stdout).unwrap();
    assert!(!visible.contains("/tmp/sample.apk"));
    assert!(!visible.contains("classes.dex"));
    assert!(!visible.contains("raw_result"));
    assert!(!visible.contains("normalized_result"));
    assert!(!visible.contains("local_path"));

    fs::remove_file(path).unwrap();
}

#[test]
fn text_success_has_separate_identities_and_explicit_claim_ceiling() {
    let path = temp_file("json");
    fs::write(
        &path,
        include_bytes!(
            "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_rea-evidence-verify"))
        .arg("--text")
        .arg(&path)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));
    let visible = String::from_utf8(output.stdout).unwrap();
    assert!(visible.contains("Evidence ID: ev_"));
    assert!(visible.contains("Semantic SHA-256:"));
    assert!(visible.contains("Raw file SHA-256:"));
    assert!(visible.contains("NOT established"));
    assert!(!visible.contains("/tmp/sample.apk"));

    fs::remove_file(path).unwrap();
}

#[test]
fn duplicate_keys_return_invalid_json_exit_code() {
    let path = temp_file("json");
    fs::write(&path, br#"{"same":1,"same":2}"#).unwrap();

    let output = run_json(&path);
    assert_eq!(output.status.code(), Some(4));
    let report: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(report["result"], "rejected");
    assert_eq!(report["error"]["code"], "invalid_json");

    fs::remove_file(path).unwrap();
}

#[test]
fn semantic_tampering_returns_identity_exit_code() {
    let path = temp_file("json");
    let mut evidence = fixture();
    evidence["normalized_result"]["entries"][0]["size"] = Value::from(99);
    fs::write(&path, serde_json::to_vec(&evidence).unwrap()).unwrap();

    let output = run_json(&path);
    assert_eq!(output.status.code(), Some(6));
    let report: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(report["error"]["code"], "identity_mismatch");

    fs::remove_file(path).unwrap();
}

#[test]
fn input_over_limit_returns_size_exit_code_without_parsing() {
    const LIMIT: u64 = 64 * 1024 * 1024;
    let path = temp_file("oversized");
    let file = File::create(&path).unwrap();
    file.set_len(LIMIT + 1).unwrap();
    drop(file);

    let output = run_json(&path);
    assert_eq!(output.status.code(), Some(7));
    let report: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(report["error"]["code"], "input_too_large");

    fs::remove_file(path).unwrap();
}

#[test]
fn missing_file_returns_io_exit_code_without_echoing_local_path() {
    let path = temp_file("does-not-exist");
    let output = run_json(&path);
    assert_eq!(output.status.code(), Some(3));
    let report: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(report["error"]["code"], "io_error");
    assert!(
        !String::from_utf8(output.stderr)
            .unwrap()
            .contains(path.to_string_lossy().as_ref())
    );
}
