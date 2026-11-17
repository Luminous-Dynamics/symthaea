use std::{fs, path::PathBuf, process::Command};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/research/marine/fixtures/marine-static-attitude-equilibrium-box-001.json")
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_marine-static-attitude-equilibrium-solve"))
        .args(args)
        .output()
        .expect("equilibrium CLI must launch")
}

#[test]
fn cli_solves_the_box_fixture_and_emits_a_reproducible_trace() {
    let fixture = fixture_path();
    let output = run_cli(&[fixture.to_str().unwrap()]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "equilibrium CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit a receipt");
    assert_eq!(receipt["disposition"], "converged-candidate");
    assert_eq!(receipt["attempts"].as_array().unwrap().len(), 1);
    assert_eq!(
        receipt["attempts"][0]["stop_reason"],
        "converged"
    );
    assert!(receipt["attempts"][0]["trace"].as_array().unwrap().len() >= 2);
    assert_eq!(
        receipt["receipt_digest_blake3"].as_str().map(str::len),
        Some(64)
    );
    let final_attitude = &receipt["converged_roots"][0]["attitude"];
    assert!(final_attitude["waterplane_offset_m"].as_f64().unwrap().abs() < 1.0e-6);
}

#[test]
fn cli_recomputes_exact_solver_receipt() {
    let fixture = fixture_path();
    let fixture_arg = fixture.to_str().unwrap();
    let output = run_cli(&[fixture_arg]);
    assert_eq!(output.status.code(), Some(0));
    let receipt: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit JSON");
    let receipt_path = std::env::temp_dir().join(format!(
        "symthaea-static-equilibrium-{}-receipt.json",
        std::process::id()
    ));
    fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    let verified = run_cli(&[
        fixture_arg,
        "--verify",
        receipt_path.to_str().unwrap(),
    ]);
    fs::remove_file(&receipt_path).unwrap();
    assert_eq!(
        verified.status.code(),
        Some(0),
        "receipt verification failed: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&verified.stdout).expect("verification must emit JSON");
    assert_eq!(report["status"], "verified");
}
