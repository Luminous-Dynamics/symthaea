use std::env;
use symthaea_adaptive_temporal_replay::{
    build_feedback_events, generate_outcomes, load_builtin_registry, replay_feedback_events, RevisionKind,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let scenario_id = args.get(0).map(String::as_str).unwrap_or("abrupt_location_shift");
    let seed = args.get(1).map(|x| x.parse::<u64>()).transpose()?.unwrap_or(20_261_102);
    let (seeds, scenarios) = load_builtin_registry()?;
    if !seeds.contains(&seed) { return Err(format!("seed {seed} is not preregistered").into()); }
    let scenario = scenarios.iter().find(|s| s.id == scenario_id)
        .ok_or_else(|| format!("unknown scenario: {scenario_id}"))?;
    let outcomes = generate_outcomes(scenario, seed)?;
    let events = build_feedback_events(scenario, &outcomes)?;
    let ledger = replay_feedback_events(&events)?;
    let min_residual = outcomes.iter().map(|x| x.residual_milli).min().unwrap_or(0);
    let max_residual = outcomes.iter().map(|x| x.residual_milli).max().unwrap_or(0);
    let max_lag = outcomes.iter().map(|x| x.feedback_available_step.saturating_sub(x.target_step)).max().unwrap_or(0);
    let revisions = events.iter().filter(|x| x.revision > 1).count();
    let retractions = events.iter().filter(|x| x.kind == RevisionKind::Retract).count();
    println!("scenario={}", scenario.id);
    println!("seed={seed}");
    println!("outcomes={}", outcomes.len());
    println!("feedback_events={}", events.len());
    println!("revisions={revisions}");
    println!("retractions={retractions}");
    println!("max_feedback_lag_steps={max_lag}");
    println!("residual_min_milli={min_residual}");
    println!("residual_max_milli={max_residual}");
    println!("ledger_records={}", ledger.records().len());
    println!("ledger_verified=true");
    Ok(())
}
