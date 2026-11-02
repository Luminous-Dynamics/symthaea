use std::env;
use symthaea_conformal_policy_control::{evaluate_beta_grid,load_builtin_fixture,sha256_hex};
use serde_json::json;

fn main()->Result<(),Box<dyn std::error::Error>>{
    let code_commit=env::var("GIT_COMMIT_SHA").map_err(|_|"set GIT_COMMIT_SHA to exact source commit before emitting receipt")?;
    if code_commit.len()!=40||!code_commit.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)){
        return Err("GIT_COMMIT_SHA must be 40-character lowercase hexadecimal".into());
    }
    let fixture=load_builtin_fixture()?;let evaluations=evaluate_beta_grid(&fixture)?;
    let receipt=json!({
        "schema":"symthaea.risk_qualified_policy_run_receipt.v0.1",
        "status":"synthetic_finite_support_reference_not_a_formal_guarantee",
        "scenario_id":fixture.scenario_id,"code_commit":code_commit,
        "fixture_sha256":sha256_hex(include_str!("../../fixtures/risk_qualified_policy_v0.1.json").as_bytes()),
        "risk_budget_ppm":fixture.risk_budget_ppm,
        "beta_grid_selection_is_formally_qualified":false,
        "beta_evaluations":evaluations,
        "claim_ceiling":"finite-support exact arithmetic and synthetic evaluator diagnostics only; no CPC theorem or AC-CP guarantee claimed"
    });
    println!("{}",serde_json::to_string_pretty(&receipt)?);Ok(())
}
