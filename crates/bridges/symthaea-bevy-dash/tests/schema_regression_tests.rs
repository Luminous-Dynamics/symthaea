use std::process::Command;

#[test]
fn test_schema_v0_2_regression() {
    // 1. Run headless export to generate the latest v0.2 time_waterfall_export.json bundle
    let export_status = Command::new("cargo")
        .args(&[
            "run",
            "--example",
            "time_waterfall",
            "-p",
            "symthaea-bevy-dash",
            "--",
            "--export-only",
        ])
        .status()
        .expect("Failed to execute cargo run --example time_waterfall -- --export-only");

    assert!(export_status.success(), "Headless export run failed");

    // 2. Run the headless schema validation CLI on the generated export file
    let analyze_output = Command::new("cargo")
        .args(&[
            "run",
            "-p",
            "symthaea-bevy-dash",
            "--bin",
            "time_waterfall_analyze",
            "--",
            "/srv/luminous-dynamics/symthaea/time_waterfall_export.json",
        ])
        .output()
        .expect("Failed to execute time_waterfall_analyze CLI");

    let stdout = String::from_utf8_lossy(&analyze_output.stdout);
    let stderr = String::from_utf8_lossy(&analyze_output.stderr);

    println!("STDOUT:\n{}", stdout);
    println!("STDERR:\n{}", stderr);

    assert!(
        analyze_output.status.success(),
        "Headless analyzer CLI validation reported schema errors"
    );

    // Verify key validation milestones are present in the output
    assert!(
        stdout.contains("✓ Headless schema validation passed cleanly."),
        "Headless schema validation success marker not found in output"
    );
    assert!(
        stdout.contains("schema: time_waterfall_export_v0.2"),
        "V0.2 schema version marker not found"
    );
    assert!(
        stdout.contains("classification: recovered_after_false_green"),
        "Expected recovery classification not detected"
    );
}
