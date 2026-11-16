use std::{fs, path::PathBuf, process::Command};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/research/marine/fixtures")
        .join(name)
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!(
        "CARGO_BIN_EXE_marine-sectional-hydrostatics-reference"
    ))
    .args(args)
    .output()
    .expect("sectional reference CLI must launch")
}

fn run_cli_with_request(name: &str, request: &serde_json::Value) -> std::process::Output {
    let path = std::env::temp_dir().join(format!(
        "symthaea-sectional-reference-{}-{name}.json",
        std::process::id()
    ));
    fs::write(
        &path,
        serde_json::to_vec(request).expect("loading-sweep request must serialize"),
    )
    .expect("temporary loading-sweep request must be written");
    let output = run_cli(&[path.to_str().expect("temporary path must be UTF-8")]);
    fs::remove_file(&path).expect("temporary loading-sweep request must be removed");
    output
}

fn run_fixture(name: &str) -> std::process::Output {
    let path = fixture_path(name);
    run_cli(&[path.to_str().expect("fixture path must be UTF-8")])
}

#[test]
fn cli_evaluates_box_loading_sweep_and_emits_digest_bound_receipt() {
    let output = run_fixture("marine-full-envelope-loading-sweep-001.json");
    assert_eq!(
        output.status.code(),
        Some(0),
        "CLI rejected box sweep: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit a JSON report");
    assert_eq!(report["status"], "evaluated");
    let receipt = &report["receipt"];
    assert_eq!(receipt["algorithm"], "sectional-area-trapezoid-v1");
    assert_eq!(
        receipt["source_spec_digest_blake3"].as_str().map(str::len),
        Some(64)
    );
    assert_eq!(receipt["results"].as_array().map(Vec::len), Some(3));
    let design = receipt["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|result| result["loading_condition_id"] == "DESIGN-WL-Z0")
        .unwrap();
    assert!((design["displacement_volume_m3"].as_f64().unwrap() - 12_000.0).abs() < 1.0e-9);
    assert!((design["waterplane_area_m2"].as_f64().unwrap() - 2_000.0).abs() < 1.0e-9);
    assert_eq!(
        design["result_digest_blake3"].as_str().map(str::len),
        Some(64)
    );
    assert_eq!(
        receipt["sweep_digest_blake3"].as_str().map(str::len),
        Some(64)
    );
}

#[test]
fn cli_evaluates_tapered_loading_sweep_and_off_midship_centres() {
    let output = run_fixture("marine-full-envelope-tapered-loading-sweep-001.json");
    assert_eq!(
        output.status.code(),
        Some(0),
        "CLI rejected tapered sweep: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit a JSON report");
    assert_eq!(report["status"], "evaluated");
    let results = report["receipt"]["results"].as_array().unwrap();
    let design = results
        .iter()
        .find(|result| result["loading_condition_id"] == "TAPER-DESIGN-Z0")
        .unwrap();
    assert!((design["displacement_volume_m3"].as_f64().unwrap() - 9_000.0).abs() < 1.0e-9);
    assert!((design["waterplane_area_m2"].as_f64().unwrap() - 1_500.0).abs() < 1.0e-9);
    assert!(
        (design["longitudinal_center_of_buoyancy_m"]
            .as_f64()
            .unwrap()
            - 500_000.0 / 9_000.0)
            .abs()
            < 1.0e-9
    );
}

#[test]
fn cli_returns_typed_rejection_for_non_positive_density() {
    let bytes = fs::read(fixture_path("marine-full-envelope-loading-sweep-001.json"))
        .expect("box fixture must be readable");
    let mut request: serde_json::Value =
        serde_json::from_slice(&bytes).expect("box fixture must be valid JSON");
    request["loading_conditions"][1]["water_density_kg_m3"] = serde_json::json!(0.0);
    let output = run_cli_with_request("invalid-density", &request);
    assert_eq!(output.status.code(), Some(1));
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit a rejection report");
    assert_eq!(report["status"], "rejected");
    assert!(
        report["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "water_density")
    );
}

#[test]
fn cli_help_documents_the_method_boundary() {
    let output = run_cli(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("upright horizontal waterplanes"));
    assert!(stdout.contains("does not generate displacement meshes"));
    assert!(stdout.contains("does not approve vessel stability"));
}
