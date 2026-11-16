use std::{fs, path::PathBuf, process::Command};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/research/marine/fixtures")
        .join(name)
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_marine-upright-mass-balance"))
        .args(args)
        .output()
        .expect("upright mass-balance CLI must launch")
}

#[test]
fn cli_solves_and_replays_mass_balance_receipt() {
    let request = fixture_path("marine-upright-mass-balance-box-001.json");
    let request_arg = request.to_str().expect("fixture path must be UTF-8");
    let solved = run_cli(&[request_arg]);
    assert_eq!(
        solved.status.code(),
        Some(0),
        "mass-balance CLI failed: {}",
        String::from_utf8_lossy(&solved.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&solved.stdout).expect("CLI must emit a JSON report");
    assert_eq!(report["status"], "solved");
    assert_eq!(
        report["receipt"]["algorithm"],
        "upright-mass-balance-bisection-v2"
    );
    assert!(
        (report["receipt"]["equilibrium"]["displaced_mass_tonnes"]
            .as_f64()
            .unwrap()
            - 12_300.0)
            .abs()
            < 1.0e-8
    );
    assert_eq!(
        report["receipt"]["receipt_digest_blake3"]
            .as_str()
            .map(str::len),
        Some(64)
    );

    let receipt_path = std::env::temp_dir().join(format!(
        "symthaea-upright-mass-balance-receipt-{}.json",
        std::process::id()
    ));
    fs::write(
        &receipt_path,
        serde_json::to_vec(&report["receipt"]).expect("receipt must serialize"),
    )
    .expect("receipt must be written");
    let receipt_arg = receipt_path.to_str().expect("receipt path must be UTF-8");
    let verified = run_cli(&[request_arg, "--verify", receipt_arg]);
    fs::remove_file(&receipt_path).expect("temporary receipt must be removed");

    assert_eq!(
        verified.status.code(),
        Some(0),
        "mass-balance receipt verification failed: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let verification: serde_json::Value =
        serde_json::from_slice(&verified.stdout).expect("verification must emit JSON");
    assert_eq!(verification["status"], "verified");
    assert_eq!(verification["receipt_schema_version"], 2);
}

#[test]
fn cli_rejects_a_mass_balance_target_outside_the_bracket() {
    let request = fixture_path("marine-upright-mass-balance-box-001.json");
    let bytes = fs::read(&request).expect("fixture must be readable");
    let mut value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("fixture must be valid JSON");
    value["target_mass_tonnes"] = serde_json::json!(100_000.0);

    let request_path = std::env::temp_dir().join(format!(
        "symthaea-upright-mass-balance-unbracketed-{}.json",
        std::process::id()
    ));
    fs::write(
        &request_path,
        serde_json::to_vec(&value).expect("mutated request must serialize"),
    )
    .expect("mutated request must be written");
    let request_arg = request_path.to_str().expect("temporary path must be UTF-8");
    let rejected = run_cli(&[request_arg]);
    fs::remove_file(&request_path).expect("temporary request must be removed");

    assert_eq!(rejected.status.code(), Some(1));
    let report: serde_json::Value =
        serde_json::from_slice(&rejected.stdout).expect("rejection must emit JSON");
    assert_eq!(report["status"], "rejected");
    assert!(
        report["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "mass_balance_not_bracketed")
    );
}
