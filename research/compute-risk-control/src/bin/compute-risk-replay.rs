use std::env;
use serde_json::json;
use symthaea_compute_risk_control::{
    calibrate_joint_candidate_grid, generate_synthetic_validation_panel, load_builtin_fixture,
    sha256_hex, verify_compute_risk_report, CalibrationDesign,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source_commit = env::var("GIT_COMMIT_SHA")
        .map_err(|_| "set GIT_COMMIT_SHA to the exact source commit before emitting a receipt")?;
    if source_commit.len() != 40
        || !source_commit.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("GIT_COMMIT_SHA must be a 40-character lowercase hexadecimal commit SHA".into());
    }

    let fixture = load_builtin_fixture()?;
    let traces = generate_synthetic_validation_panel(&fixture)?;
    let report = calibrate_joint_candidate_grid(
        &fixture.candidates,
        &traces,
        CalibrationDesign::SyntheticFixedPanel,
        fixture.risk_upper_budget_ppm,
        fixture.risk_lower_budget_ppm,
        fixture.risk_output_error_budget_ppm,
        fixture.confidence_delta_ppm,
    )?;
    verify_compute_risk_report(&report)?;

    let payload = json!({
        "schema":"symthaea.compute_risk_joint_receipt.v0.1",
        "status":"synthetic_stratified_panel_diagnostic_not_a_confidence_claim",
        "source_commit":source_commit,
        "fixture_sha256":sha256_hex(include_str!("../../fixtures/compute_risk_joint_grid_v0.1.json").as_bytes()),
        "validation_panel_sha256":report.validation_panel_sha256,
        "candidate_grid_sha256":report.candidate_grid_sha256,
        "report":report,
        "claim_ceiling":"fixed synthetic panel replay only; simultaneous Hoeffding UCB arithmetic is reported as a diagnostic and is not a population guarantee for this stratified panel"
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
