use std::env;
use symthaea_adaptive_temporal_replay::{
    build_feedback_events, generate_outcomes, load_builtin_registry, replay_feedback_events,
    RevisionKind, REGISTRY_JSON,
};
use sha2::{Digest, Sha256};

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let scenario_id = args.get(0).map(String::as_str).unwrap_or("abrupt_location_shift");
    let seed = args.get(1).map(|x| x.parse::<u64>()).transpose()?.unwrap_or(20_261_102);
    let code_commit = env::var("GIT_COMMIT_SHA")
        .map_err(|_| "set GIT_COMMIT_SHA to the exact source commit before generating a receipt")?;
    if code_commit.len() != 40
        || !code_commit.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("GIT_COMMIT_SHA must be a 40-character lowercase hexadecimal commit SHA".into());
    }
    let (seeds, scenarios) = load_builtin_registry()?;
    if !seeds.contains(&seed) {
        return Err(format!("seed {seed} is not preregistered").into());
    }
    let scenario = scenarios.iter().find(|s| s.id == scenario_id)
        .ok_or_else(|| format!("unknown scenario: {scenario_id}"))?;
    let outcomes = generate_outcomes(scenario, seed)?;
    let events = build_feedback_events(scenario, &outcomes)?;
    let ledger = replay_feedback_events(&events)?;

    let min_residual = outcomes.iter().map(|x| x.residual_milli).min().unwrap_or(0);
    let max_residual = outcomes.iter().map(|x| x.residual_milli).max().unwrap_or(0);
    let max_feedback_lag = outcomes.iter()
        .map(|x| x.feedback_available_step.saturating_sub(x.target_step)).max().unwrap_or(0);
    let max_source_lag = outcomes.iter()
        .map(|x| x.source_available_step.saturating_sub(x.target_step)).max().unwrap_or(0);
    let revisions = events.iter().filter(|x| x.revision > 1).count();
    let retractions = events.iter().filter(|x| x.kind == RevisionKind::Retract).count();
    let payload = serde_json::json!({
        "schema": "symthaea.adaptive_temporal_calibration_run_receipt.v0.1",
        "status": "synthetic_only_not_a_real_data_or_coverage_claim",
        "scenario_id": scenario.id,
        "scenario_kind": scenario.kind,
        "seed_id": seed,
        "generator_id": "splitmix64-uniform-residual-v1",
        "fixture_sha256": hex_sha256(REGISTRY_JSON.as_bytes()),
        "code_commit": code_commit,
        "outcome_count": outcomes.len(),
        "feedback_event_count": events.len(),
        "revision_count": revisions,
        "retraction_count": retractions,
        "maximum_feedback_lag_steps": max_feedback_lag,
        "maximum_source_availability_lag_steps": max_source_lag,
        "residual_min_milli": min_residual,
        "residual_max_milli": max_residual,
        "label_ledger_sha256": ledger.sha256_hex(),
        "label_ledger_verified": true
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
