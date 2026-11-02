use std::env;
use serde_json::json;
use symthaea_compositional_risk_budget::{
    append_risk_evidence, compose_risk_budget, create_risk_budget_ledger,
    load_builtin_plan, sha256_hex, verify_risk_budget_ledger,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let code_commit = env::var("GIT_COMMIT_SHA")
        .map_err(|_| "set GIT_COMMIT_SHA to the exact source commit before emitting a receipt")?;
    if code_commit.len() != 40
        || !code_commit.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("GIT_COMMIT_SHA must be a 40-character lowercase hexadecimal commit SHA".into());
    }

    let plan = load_builtin_plan()?;
    let contracts = plan.contracts.clone();
    let mut ledger = create_risk_budget_ledger(plan)?;
    for contract in &contracts {
        // Deterministic synthetic hashes make a replayable topology fixture.
        // They are not evidence that any paper theorem or calibration is valid.
        let evidence_sha = sha256_hex(format!("synthetic-evidence:{}", contract.claim_id).as_bytes());
        let method_receipt_sha = sha256_hex(format!("synthetic-method-receipt:{}", contract.claim_id).as_bytes());
        append_risk_evidence(
            &mut ledger,
            &contract.claim_id,
            &evidence_sha,
            &method_receipt_sha,
            contract.source_receipt_sha256.as_deref(),
        )?;
    }
    verify_risk_budget_ledger(&ledger)?;
    let report = compose_risk_budget(&ledger)?;
    let receipt = json!({
        "schema":"symthaea.compositional_risk_run_receipt.v0.1",
        "status":"synthetic_topology_only_component_guarantees_unverified",
        "code_commit":code_commit,
        "fixture_sha256":sha256_hex(include_str!("../../fixtures/compositional_risk_plan_v0.1.json").as_bytes()),
        "plan_sha256":ledger.plan.plan_sha256,
        "ledger_sha256":ledger.ledger_sha256,
        "contract_count":ledger.plan.contracts.len(),
        "evidence_receipt_count":ledger.receipts.len(),
        "report":report,
        "claim_ceiling":"exact budget arithmetic, type compatibility, action-partition completeness, and receipt replay only; component theorem validity and real-world risk are not verified"
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
