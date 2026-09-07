// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Run: cargo run -p symthaea-nuclear --example novel_isotope_search
//
// Historical filename retained for command compatibility. This is an exploratory
// coordinate-ranking example, not a novel-isotope discovery or stability claim.

use symthaea_nuclear::discovery::*;
use symthaea_nuclear::mass_formula::SemiEmpiricalMassFormula;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Symthaea Nuclear Exploration: Superheavy Coordinates");
    println!("Heuristic proxies only -- no discovery/stability claim\n");

    let mut engine = NuclearExplorationEngine::new();
    let measured = engine.add_ame2020_measured_reference()?;
    println!("Measured AME2020 entries admitted for residual diagnostics: {measured}");
    println!("Estimated/extrapolated AME entries are excluded from residual calibration.\n");

    println!(
        "Exploring Z=104-130, N=150-200 ({} coordinates)...\n",
        (130 - 104 + 1) * (200 - 150 + 1)
    );
    let report = engine.explore(104, 130, 150, 200)?;

    println!("=== Residual Fit Summary ===");
    println!(
        "  Measured comparisons: {}",
        report.residual_fit_summary.n_measured_comparisons
    );
    println!(
        "  Mean residual:        {:.2} MeV",
        report.residual_fit_summary.mean_residual_mev
    );
    println!(
        "  RMS residual:         {:.2} MeV",
        report.residual_fit_summary.rms_residual_mev
    );
    println!(
        "  Max |residual|:       {:.2} MeV",
        report.residual_fit_summary.max_abs_residual_mev
    );
    println!(
        "  Within local spread proxy: {:.0}%\n",
        report
            .residual_fit_summary
            .fraction_within_local_dispersion_proxy
            * 100.0
    );

    println!("=== Top {} Exploration Coordinates ===", report.candidates.len());
    println!(
        "{:<5} {:<5} {:<5} | {:<9} | {:<12} | {:<8} | {:<12} | {}",
        "Z", "N", "A", "Priority", "Spread proxy", "Support", "Shell proxy", "Reason"
    );
    println!("{}", "-".repeat(116));
    for candidate in &report.candidates {
        println!(
            "{:<5} {:<5} {:<5} | {:<9.3} | {:<12.2} | {:<8} | {:<12.2} | {}",
            candidate.z,
            candidate.n,
            candidate.a,
            candidate.exploration_priority_proxy,
            candidate.local_residual_dispersion_proxy_mev,
            candidate.local_measured_support_count,
            candidate.shell_correction_mev,
            candidate.reason
        );
    }

    println!("\n=== Optional SEMF Diagnostics for Highest-Priority Coordinates ===");
    let semf = SemiEmpiricalMassFormula::default();
    for candidate in report.candidates.iter().take(10) {
        let ba_proxy = semf.binding_energy_per_nucleon(candidate.a, candidate.z);
        let q_alpha_proxy = semf.alpha_decay_q(candidate.a, candidate.z);
        println!(
            "  Z={}, N={}, A={}: priority={:.3}, SEMF B/A proxy={:.3} MeV, SEMF Q-alpha proxy={:.2} MeV, GN half-life proxy={}",
            candidate.z,
            candidate.n,
            candidate.a,
            candidate.exploration_priority_proxy,
            ba_proxy,
            q_alpha_proxy,
            format_half_life_proxy(candidate.semf_geiger_nuttall_half_life_proxy_seconds)
        );
    }

    println!("\n=== Interpretation Boundary ===");
    println!("  {:?}", report.interpretation);
    println!("  - Local residual spread is not calibrated predictive uncertainty.");
    println!("  - Residual/spread ratios are not significance tests.");
    println!("  - Shell, B/A, Q-alpha, and Geiger-Nuttall outputs here are simple-model proxies.");
    println!("  - Ranking does not establish nuclear existence, stability, lifetime, synthesizability, or discovery probability.");
    println!("  - Strong candidates should advance to qualified mass-model, fission-barrier, decay-chain, and experimental-feasibility evidence rather than being promoted directly.");

    Ok(())
}

fn format_half_life_proxy(seconds: Option<f64>) -> String {
    match seconds {
        Some(value) if value.is_finite() && value >= 0.0 => format!("{value:.3e} s"),
        _ => "not produced".to_string(),
    }
}
