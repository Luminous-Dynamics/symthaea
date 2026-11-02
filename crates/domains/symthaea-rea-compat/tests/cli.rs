use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rea-evidence-verify"))
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json")
}

fn temporary_file(label: &str, bytes: &[u8]) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "rea-evidence-verify-{}-{nonce}-{label}.json",
        std::process::id()
    ));
    fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn json_output_reports_distinct_semantic_and_raw_file_identities() {
    let path = fixture_path();
    let bytes = fs::read(&path).unwrap();
    let output = binary().arg("--json").arg(&path).output().unwrap();
    assert!(output.status.success());

    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["status"], "semantic_identity_verified");
    assert_eq!(
        document["evidence_id"],
        "ev_a4e49bdca4cbf80c2ec8e4d7eba116af403bc33a4937ffb7fca00b6aab6590c4"
    );
    assert_ne!(document["raw_file_sha256"], document["semantic_sha256"]);
    assert_eq!(document["input_bytes"], bytes.len() as u64);
    assert_eq!(document["claim_ceiling"]["producer_authenticated"], false);
    assert_eq!(
        document["claim_ceiling"]["observation_truth_verified"],
        false
    );

    let rendered = String::from_utf8(output.stdout).unwrap();
    assert!(!rendered.contains("/tmp/sample.apk"));
    assert!(!rendered.contains("classes.dex"));
    assert!(!rendered.contains("normalized_result"));
    assert!(!rendered.contains("raw_result"));
}

#[test]
fn text_and_json_outputs_report_the_same_evidence_id() {
    let path = fixture_path();
    let json_output = binary().arg("--json").arg(&path).output().unwrap();
    let text_output = binary().arg("--text").arg(&path).output().unwrap();
    assert!(json_output.status.success());
    assert!(text_output.status.success());

    let json: Value = serde_json::from_slice(&json_output.stdout).unwrap();
    let text = String::from_utf8(text_output.stdout).unwrap();
    assert!(text.contains(json["evidence_id"].as_str().unwrap()));
    assert!(text.contains(json["semantic_sha256"].as_str().unwrap()));
    assert!(text.contains(json["raw_file_sha256"].as_str().unwrap()));
    assert!(text.contains("Producer authentication: not performed"));
}

#[test]
fn duplicate_keys_and_trailing_json_are_rejected_with_stable_categories() {
    for (label, bytes, expected) in [
        (
            "duplicate",
            br#"{"same":1,"same":2}"#.as_slice(),
            "invalid_json",
        ),
        ("trailing", br#"{} {}"#.as_slice(), "invalid_json"),
    ] {
        let path = temporary_file(label, bytes);
        let output = binary().arg("--json").arg(&path).output().unwrap();
        let _ = fs::remove_file(&path);
        assert!(!output.status.success());

        let document: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["status"], "rejected");
        assert_eq!(document["error"]["category"], expected);
    }
}

#[test]
fn tampered_semantic_identity_is_rejected_without_leaking_payload() {
    let bytes = fs::read(fixture_path()).unwrap();
    let mut evidence: Value = serde_json::from_slice(&bytes).unwrap();
    evidence["normalized_result"]["entries"][0]["size"] = Value::from(43);
    let path = temporary_file(
        "tampered",
        serde_json::to_string(&evidence).unwrap().as_bytes(),
    );

    let output = binary().arg("--json").arg(&path).output().unwrap();
    let _ = fs::remove_file(&path);
    assert!(!output.status.success());

    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["status"], "rejected");
    assert_eq!(document["error"]["category"], "identity_mismatch");
    assert!(
        !output
            .stdout
            .windows(b"classes.dex".len())
            .any(|window| window == b"classes.dex")
    );
}

#[test]
fn oversized_input_is_stopped_at_the_read_boundary() {
    const LIMIT: usize = 64 * 1024 * 1024;
    let path = temporary_file("oversized", &vec![b' '; LIMIT + 1]);
    let output = binary().arg("--json").arg(&path).output().unwrap();
    let _ = fs::remove_file(&path);
    assert!(!output.status.success());

    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["error"]["category"], "input_too_large");
}
