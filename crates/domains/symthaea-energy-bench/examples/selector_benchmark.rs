//! Run the oracle-backed synthetic energy-conversion selector benchmark.
//!
//! From the repository root:
//! cargo run -p symthaea-energy-bench --example selector_benchmark > energy-selector-benchmark.json
//!
//! The output is a synthetic benchmark receipt, not physical performance evidence.
//! Keep the exact source revision with the receipt.

use std::error::Error;
use symthaea_energy_bench::experiment_benchmark::{
    benchmark_report_json, generate_synthetic_scenarios, run_selection_benchmark,
    ExperimentSelectionStrategy,
};
use symthaea_energy_bench::experiment_selection::{ResourceBasis, SelectionPolicy};

fn main() -> Result<(), Box<dyn Error>> {
    let policy = SelectionPolicy {
        policy_id: "energy-selector-synthetic-v1".into(),
        registered_before_queue_generation: true,
        resource_basis: ResourceBasis::ComputeTime,
        resource_unit_id: "synthetic-cost-unit".into(),
    };

    let scenarios = generate_synthetic_scenarios(1_000, 0x51ec7, &policy);
    let mut strategies = vec![
        ExperimentSelectionStrategy::PhaseA,
        ExperimentSelectionStrategy::CheapestFirst,
        ExperimentSelectionStrategy::SpaceFilling,
        ExperimentSelectionStrategy::OracleUpperBound,
    ];
    strategies.extend((0..30).map(|seed| ExperimentSelectionStrategy::Random { seed }));

    let report = run_selection_benchmark(&policy, &scenarios, &strategies)?;
    println!("{}", benchmark_report_json(&report)?);
    Ok(())
}
