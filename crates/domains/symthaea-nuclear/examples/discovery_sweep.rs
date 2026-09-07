// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Run: cargo run -p symthaea-nuclear --example discovery_sweep
//
// Historical filename retained for command compatibility. This example performs
// a heuristic SEMF + shell-model exploration of the superheavy region. It does
// not claim discovery, calibrated uncertainty, isotope existence, or stability.

use symthaea_nuclear::discovery::NuclearExplorationEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Nuclear Exploration — Superheavy Region");
    println!("=======================================\n");

    let mut engine = NuclearExplorationEngine::new();
    let measured = engine.add_ame2020_measured_reference()?;
    println!("Measured AME2020 residual-calibration entries admitted: {measured}");
    println!("Estimated/extrapolated AME entries are excluded from empirical residual calibration.\n");

    println!("Exploring Z=110-120, N=170-190 with heuristic proxies...\n");
    let report = engine.explore(110, 120, 170, 190)?;
    println!("{}", NuclearExplorationEngine::format_report(&report));
    println!("\nNON-CLAIM: this ranking is hypothesis generation only; it is not evidence that any unmeasured coordinate corresponds to an existing or long-lived nucleus.");
    Ok(())
}
