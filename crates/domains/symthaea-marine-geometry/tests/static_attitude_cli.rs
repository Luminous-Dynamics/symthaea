use std::{fs, path::PathBuf, process::Command};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/research/marine/fixtures/marine-static-attitude-box-001.json")
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_marine-static-attitude-evaluate"))
        .args(args)
        .output()
        .expect("static-attitude CLI must launch")
}

fn run_cli_with_request(request: &serde_json::Value) -> std::process::Output {
    let path = std::env::temp_dir().join(format!(
        "symthaea-static-attitude-{}-request.json",
        std::process::id()
    ));
    fs::write(
        &path,
        serde_json::to_vec(request).expect("request must serialize"),
    )
    .expect("temporary request must be written");
    let output = run_cli(&[path.to_str().expect("temporary path must be UTF-8")]);
    fs::remove_file(&path).expect("temporary request must be removed");
    output
}

#[test]
fn cli_evaluates_upright_box_and_emits_digest_bound_residuals() {
    let fixture = fixture_path();
    let output = run_cli(&[fixture.to_str().expect("fixture path must be UTF-8")]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "CLI rejected fixture: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit JSON");
    assert_eq!(report["status"], "evaluated");
    let receipt = &report["receipt"];
    assert!((receipt["displacement_volume_m3"].as_f64().unwrap() - 12_000.0).abs() < 1.0e-8);
    assert!((receipt["displaced_mass_tonnes"].as_f64().unwrap() - 12_300.0).abs() < 1.0e-8);
    assert!(
        receipt["residuals"]["mass_residual_tonnes"]
            .as_f64()
            .unwrap()
            .abs()
            < 1.0e-8
    );
    assert!(
        receipt["residuals"]["moment_residual_1_nm"]
            .as_f64()
            .unwrap()
            .abs()
            < 1.0e-7
    );
    assert_eq!(
        receipt["receipt_digest_blake3"].as_str().map(str::len),
        Some(64)
    );
}

#[test]
fn cli_recomputes_a_saved_receipt_against_the_exact_request() {
    let fixture = fixture_path();
    let fixture_str = fixture.to_str().expect("fixture path must be UTF-8");
    let generated = run_cli(&[fixture_str]);
    assert_eq!(generated.status.code(), Some(0));
    let report: serde_json::Value =
        serde_json::from_slice(&generated.stdout).expect("CLI must emit JSON");
    let receipt_path = std::env::temp_dir().join(format!(
        "symthaea-static-attitude-{}-receipt.json",
        std::process::id()
    ));
    fs::write(
        &receipt_path,
        serde_json::to_vec(&report["receipt"]).expect("receipt must serialize"),
    )
    .expect("receipt file must be written");
    let verified = run_cli(&[
        fixture_str,
        "--verify",
        receipt_path.to_str().expect("receipt path must be UTF-8"),
    ]);
    fs::remove_file(&receipt_path).expect("receipt file must be removed");
    assert_eq!(
        verified.status.code(),
        Some(0),
        "CLI failed fresh receipt verification: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let verification: serde_json::Value =
        serde_json::from_slice(&verified.stdout).expect("verification must emit JSON");
    assert_eq!(verification["status"], "verified");
}

#[test]
fn cli_rejects_an_out_of_domain_pitch_with_a_typed_report() {
    let fixture = fixture_path();
    let bytes = fs::read(fixture).expect("fixture must be readable");
    let mut request: serde_json::Value =
        serde_json::from_slice(&bytes).expect("fixture must be JSON");
    request["attitude"]["pitch_rad"] = serde_json::json!(1.0);
    let output = run_cli_with_request(&request);
    assert_eq!(output.status.code(), Some(1));
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit typed rejection JSON");
    assert_eq!(report["status"], "rejected");
    assert!(
        report["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "attitude_domain")
    );
}

#[test]
fn cli_help_describes_the_non_approval_claim_boundary() {
    let output = run_cli(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("not an equilibrium solver"));
    assert!(help.contains("not a stability approval"));
}
