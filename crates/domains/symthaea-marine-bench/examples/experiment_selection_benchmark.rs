// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;

use symthaea_marine_bench::experiment_selection::{
    run_reference_benchmark, verify_receipt, DEFAULT_BUDGET_UNITS, DEFAULT_PAIRED_SEEDS,
    DEFAULT_SCENARIOS,
};

fn main() -> Result<(), Box<dyn Error>> {
    let seeds: Vec<u64> = (0..DEFAULT_PAIRED_SEEDS as u64).collect();
    let receipt = run_reference_benchmark(DEFAULT_SCENARIOS, &seeds, DEFAULT_BUDGET_UNITS)?;
    if !verify_receipt(&receipt)? {
        return Err("benchmark receipt digest verification failed".into());
    }
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
