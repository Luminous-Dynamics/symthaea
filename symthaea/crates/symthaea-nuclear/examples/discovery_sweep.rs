// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Run: cargo run -p symthaea-nuclear --example discovery_sweep
//
// Performs a nuclear discovery analysis of the superheavy region,
// calibrated against known binding energies, and identifies
// isotopes where new physics discoveries are most likely.

use symthaea_nuclear::discovery::NuclearDiscoveryEngine;

fn main() {
    println!("Nuclear Discovery Engine — Superheavy Region Analysis");
    println!("=====================================================\n");

    let mut engine = NuclearDiscoveryEngine::new();
    engine.add_reference_nuclei();

    println!("Running discovery sweep: Z=110-120, N=170-190...\n");
    let report = engine.discover(110, 120, 170, 190);
    println!("{}", NuclearDiscoveryEngine::print_report(&report));
}
