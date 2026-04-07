// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Run: cargo run -p symthaea-nuclear --example superheavy_sweep
//
// Performs a systematic sweep of superheavy elements Z=110-120
// and prints a discovery report showing predicted stability.

use symthaea_nuclear::*;

fn main() {
    println!("Starting superheavy isotope discovery sweep...");
    println!("Z=110-120, N=170-190 ({} isotopes)\n", 11 * 21);

    let landscape = StabilityLandscape::default();
    let report = landscape.find_stable_isotopes(110, 120, 170, 190);

    println!("{}", StabilityLandscape::print_report(&report));
}
