// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emit matched selective-prediction baseline reports for the frozen v0.1 fixture.

use symthaea_psych_bench::benchmarks::metacognition::selective_prediction_task;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reports = selective_prediction_task::run_reference_suite()?;
    for report in reports {
        println!("{}", serde_json::to_string(&report)?);
    }
    Ok(())
}
