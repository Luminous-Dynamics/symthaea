use std::{path::PathBuf, process::Command};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/research/marine/fixtures")
        .join(name)
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_marine-full-envelope-validate"))
        .args(args)
        .output()
        .expect("full-envelope validation CLI must launch")
}

fn assert_valid_fixture(name: &str) {
    let path = fixture_path(name);
    let path = path.to_str().expect("fixture path must be UTF-8");
    let output = run_cli(&[path]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "CLI rejected {name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit JSON report");
    assert_eq!(report["disposition"], "valid-input-contract");
    assert_eq!(report["issues"], serde_json::json!([]));
    assert_eq!(
        report["source_spec_digest_blake3"].as_str().map(str::len),
        Some(64)
    );
}

#[test]
fn cli_validates_symmetric_transom_fixture_and_emits_digest() {
    assert_valid_fixture("marine-full-envelope-box-001.json");
}

#[test]
fn cli_validates_explicitly_asymmetric_fixture_without_mirroring() {
    assert_valid_fixture("marine-full-envelope-asymmetric-001.json");
}

#[test]
fn cli_returns_documented_error_code_for_unknown_options() {
    let output = run_cli(&["--unknown"]);
    assert_eq!(output.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unknown option"),
        "unexpected CLI stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_help_is_successful_and_names_claim_boundary() {
    let output = run_cli(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("does not generate a surface"));
    assert!(stdout.contains("valid input contract"));
}
