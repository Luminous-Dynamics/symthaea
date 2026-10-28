use std::fs;
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_vleb-cobol-index")
}

#[test]
fn fixed_format_fixture_reports_source_anchors_without_claiming_semantics() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let source = format!("{manifest_dir}/fixtures/settlement.cbl");
    let output = Command::new(binary())
        .args(["--format", "fixed", &source])
        .output()
        .expect("run COBOL indexer");

    assert!(output.status.success(), "stderr={}", String::from_utf8_lossy(&output.stderr));
    let json = String::from_utf8(output.stdout).expect("JSON is UTF-8");
    assert!(json.contains("\"schema_version\": \"vleb-cobol-index-v1\""));
    assert!(json.contains("\"program_id\": {\"name\":\"SETTLE01\""));
    assert!(json.contains("\"name\":\"BANK-REC\""));
    assert!(json.contains("\"name\":\"AMOUNT-REC\""));
    assert!(json.contains("\"has_replacing_clause\":true"));
    assert!(json.contains("\"name\":\"COMP-3\"") || json.contains("\"usage\":\"COMP-3\""));
    assert!(json.contains("\"picture\":\"S9(7)V99\""));
    assert!(json.contains("\"redefines\":\"WS-AMOUNT\""));
    assert!(json.contains("\"kind\":\"SQL\""));
    assert!(json.contains("\"FD:SETTLEMENT-IN\""));
    assert!(json.contains("\"diagnostics\": []"));
}

#[test]
fn source_format_must_be_explicit() {
    let output = Command::new(binary())
        .output()
        .expect("run COBOL indexer");
    assert_eq!(output.status.code(), Some(64));
    assert!(String::from_utf8_lossy(&output.stderr).contains("source format must be declared explicitly"));
}

#[test]
fn free_format_is_rejected_instead_of_guessed() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let source = format!("{manifest_dir}/fixtures/settlement.cbl");
    let output = Command::new(binary())
        .args(["--format", "free", &source])
        .output()
        .expect("run COBOL indexer");
    assert_eq!(output.status.code(), Some(64));
    assert!(String::from_utf8_lossy(&output.stderr).contains("only explicit fixed format is supported"));
}

#[test]
fn uncertain_fixed_format_record_produces_json_diagnostics_and_nonzero_status() {
    let path = std::env::temp_dir().join(format!(
        "vleb-cobol-index-{}-bad.cbl",
        std::process::id()
    ));
    fs::write(
        &path,
        "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. BADFIX.\n       DISPLAY CAFÉ.\n",
    )
    .expect("write temporary COBOL sample");

    let output = Command::new(binary())
        .args(["--format", "fixed", path.to_str().expect("temp path is UTF-8")])
        .output()
        .expect("run COBOL indexer");
    let _ = fs::remove_file(path);

    assert_eq!(output.status.code(), Some(2));
    let json = String::from_utf8(output.stdout).expect("diagnostic JSON is UTF-8");
    assert!(json.contains("\"code\":\"non_ascii_fixed_format_record\""));
}
