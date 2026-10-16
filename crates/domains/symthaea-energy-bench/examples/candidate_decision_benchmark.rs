//! Evaluate candidate-superiority false-positive behavior on synthetic negative controls.
//!
//! cargo run -p symthaea-energy-bench --example candidate_decision_benchmark > candidate-decision-benchmark.json
//!
//! This is an adversarial software test. It does not produce physical evidence.

use std::error::Error;
use symthaea_energy_bench::candidate_decision_benchmark::{
    candidate_decision_report_json, generate_candidate_decision_scenarios,
    run_candidate_decision_benchmark,
};

fn main() -> Result<(), Box<dyn Error>> {
    let scenarios = generate_candidate_decision_scenarios();
    let report = run_candidate_decision_benchmark(&scenarios)?;
    println!("{}", candidate_decision_report_json(&report)?);
    Ok(())
}
