use std::env;
use serde_json::json;
use symthaea_compute_risk_simultaneous_ucb::{evaluate_candidate_grid, load_builtin_fixture, sha256_hex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let code_commit = env::var("GIT_COMMIT_SHA")
        .map_err(|_| "set GIT_COMMIT_SHA to the exact source commit before emitting a run receipt")?;
    if code_commit.len() != 40
        || !code_commit.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("GIT_COMMIT_SHA must be a 40-character lowercase hexadecimal commit SHA".into());
    }
    let fixture = load_builtin_fixture()?;
    let report = evaluate_candidate_grid(&fixture)?;
    let trajectories = serde_json::to_vec(&fixture.calibration_trajectories)?;
    let candidates = serde_json::to_vec(&fixture.candidate_policies)?;
    let receipt = json!({
        "schema":"symthaea.compute_risk_simultaneous_ucb_run_receipt.v0.1",
        "status":"synthetic_joint_grid_reference_not_a_model_or_distribution_shift_claim",
        "code_commit":code_commit,
        "fixture_sha256":sha256_hex(include_str!("../../fixtures/dual_threshold_compute_risk_v0.1.json").as_bytes()),
        "candidate_grid_sha256":sha256_hex(&candidates),
        "calibration_trajectory_sha256":sha256_hex(&trajectories),
        "candidate_count":fixture.candidate_policies.len(),
        "calibration_trajectory_count":fixture.calibration_trajectories.len(),
        "selected_candidate_id":report.selected_candidate_id,
        "report":report,
        "claim_ceiling":"simultaneous finite-grid Hoeffding bounds only under iid/exchangeable whole-trajectory calibration units and a frozen candidate grid; no distribution-shift, subgroup, or anytime guarantee"
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
