// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prints the exotic-matter cost of the shift-vector field this crate's simulator
//! moves a craft through, for a few illustrative speeds — from "as fast as
//! anything humans have ever radar-tracked" up to a meaningful fraction of c.
//!
//! Run: `cargo run -p symthaea-gravcraft --example feasibility_report`

use symthaea_gravcraft::energetics::{
    casimir_energy_density_si, exotic_mass_kg, feasibility_gap_orders_of_magnitude,
    peak_energy_density_si,
};
use symthaea_gravcraft::types::GravcraftConfig;

const JUPITER_MASS_KG: f64 = 1.898e27;
const SUN_MASS_KG: f64 = 1.989e30;
const C: f64 = 299_792_458.0;

fn main() {
    let cfg = GravcraftConfig::default();
    println!(
        "Bubble radius: {} m, wall sigma: {} 1/m\n",
        cfg.bubble_radius, cfg.wall_sigma
    );
    println!(
        "Most extreme negative energy density ever measured (Casimir, 1nm gap): {:.3e} J/m^3\n",
        casimir_energy_density_si(1e-9)
    );

    let cases: &[(&str, f64)] = &[
        (
            "Mach 25 (~8,500 m/s, upper end of reported hypersonic UAP tracks)",
            8_500.0 / C,
        ),
        ("1% of c", 0.01),
        ("10% of c", 0.10),
    ];

    for (label, v_frac) in cases {
        let mass = exotic_mass_kg(cfg.bubble_radius, cfg.wall_sigma, *v_frac);
        let peak_rho = peak_energy_density_si(cfg.bubble_radius, cfg.wall_sigma, *v_frac);
        let gap = feasibility_gap_orders_of_magnitude(cfg.bubble_radius, cfg.wall_sigma, *v_frac);
        println!("--- {label} (v/c = {v_frac:.3e}) ---");
        println!("  required exotic mass:  {mass:.3e} kg");
        println!(
            "    = {:.3e} x Jupiter, {:.3e} x the Sun",
            mass / JUPITER_MASS_KG,
            mass / SUN_MASS_KG
        );
        println!("  peak negative energy density: {peak_rho:.3e} J/m^3");
        println!("  vs. most extreme ever measured: {gap:.1} orders of magnitude larger\n");
    }
}
