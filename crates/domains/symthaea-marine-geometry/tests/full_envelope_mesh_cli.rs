use std::{fs, path::PathBuf, process::Command};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/research/marine/fixtures")
        .join(name)
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_marine-full-envelope-mesh"))
        .args(args)
        .output()
        .expect("full-envelope mesh CLI must launch")
}

#[test]
fn cli_generates_closed_box_envelope_and_exact_mesh_volume() {
    let spec = fixture_path("marine-full-envelope-box-001.json");
    let profile = fixture_path("marine-full-envelope-mesh-admission-profile-001.json");
    let output = run_cli(&[
        spec.to_str().unwrap(),
        profile.to_str().unwrap(),
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "mesh generation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit JSON artifact");
    assert_eq!(report["status"], "generated");
    assert_eq!(report["artifact"]["admission_report"]["disposition"], "admissible-for-profile");
    let volume = report["artifact"]["admission_report"]["metrics"]["signed_volume_m3"]
        .as_f64()
        .unwrap();
    assert!((volume - 20_500.0).abs() < 1.0e-8);
}

#[test]
fn cli_writes_artifact_and_replays_exact_source_profile_and_manifest() {
    let spec = fixture_path("marine-full-envelope-box-001.json");
    let profile = fixture_path("marine-full-envelope-mesh-admission-profile-001.json");
    let artifact = std::env::temp_dir().join(format!(
        "symthaea-full-envelope-mesh-{}.json",
        std::process::id()
    ));
    let output = run_cli(&[
        spec.to_str().unwrap(),
        profile.to_str().unwrap(),
        "--output",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(0));
    let verify = run_cli(&[
        "--verify",
        spec.to_str().unwrap(),
        profile.to_str().unwrap(),
        artifact.to_str().unwrap(),
    ]);
    fs::remove_file(&artifact).unwrap();
    assert_eq!(
        verify.status.code(),
        Some(0),
        "mesh replay failed: {}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&verify.stdout).expect("verification CLI must emit JSON");
    assert_eq!(report["status"], "verified");
}

#[test]
fn cli_returns_incomplete_for_unimplemented_opening_geometry() {
    let spec = fixture_path("marine-full-envelope-box-001.json");
    let profile = fixture_path("marine-full-envelope-mesh-admission-profile-001.json");
    let bytes = fs::read(&spec).unwrap();
    let mut source: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    source["boundary_semantics"]["exterior_shell"] =
        serde_json::json!("EXPLICIT_BOUNDARY_LOOPS");
    source["boundary_semantics"]["openings"] = serde_json::json!([{
        "opening_id": "HATCH-001",
        "boundary_point_refs": ["BOW-KEEL", "BOW-WL-S", "BOW-SHEER-S"],
        "status": "SUPPORTED"
    }]);
    let temporary = std::env::temp_dir().join(format!(
        "symthaea-full-envelope-open-{}.json",
        std::process::id()
    ));
    fs::write(&temporary, serde_json::to_vec(&source).unwrap()).unwrap();
    let output = run_cli(&[
        temporary.to_str().unwrap(),
        profile.to_str().unwrap(),
    ]);
    fs::remove_file(&temporary).unwrap();
    assert_eq!(output.status.code(), Some(1));
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("CLI must emit rejection report");
    assert_eq!(report["status"], "rejected");
}

#[test]
fn cli_help_names_closed_shell_scope_and_claim_boundary() {
    let output = run_cli(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("closed external hull mesh"));
    assert!(stdout.contains("does not calculate vessel stability"));
}
