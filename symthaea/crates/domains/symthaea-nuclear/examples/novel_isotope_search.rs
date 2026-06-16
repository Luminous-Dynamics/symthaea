// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Run: cargo run -p symthaea-nuclear --example novel_isotope_search
//
// Systematic search for potentially novel stable isotopes:
// 1. Calibrate against AME2020 (70 nuclei)
// 2. Sweep the superheavy region (Z=104-130, N=150-200)
// 3. Rank by discovery interest score
// 4. Filter for unmeasured isotopes with strong shell enhancement
// 5. Report predictions with confidence estimates

use symthaea_nuclear::discovery::*;
use symthaea_nuclear::island_stability::*;
use symthaea_nuclear::mass_formula::SemiEmpiricalMassFormula;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Nuclear Discovery: Novel Isotope Search      ║");
    println!("║  Calibrated against AME2020 (70 reference nuclei)      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // 1. Calibrate the discovery engine
    let mut engine = NuclearDiscoveryEngine::new();
    engine.add_reference_nuclei(); // 70 AME2020 nuclei
    println!("Calibration: {} reference nuclei loaded\n", 70);

    // 2. Run discovery sweep on the superheavy region
    println!(
        "Sweeping Z=104-130, N=150-200 ({} isotopes)...\n",
        (130 - 104 + 1) * (200 - 150 + 1)
    );

    let report = engine.discover(104, 130, 150, 200);

    // 3. Print calibration quality
    println!("=== Model Calibration ===");
    println!("  Nuclei compared: {}", report.calibration.n_calibration);
    println!(
        "  Mean residual:   {:.2} MeV (systematic bias)",
        report.calibration.mean_residual
    );
    println!(
        "  RMS residual:    {:.2} MeV (overall accuracy)",
        report.calibration.rms_residual
    );
    println!(
        "  Max residual:    {:.2} MeV",
        report.calibration.max_residual
    );
    println!(
        "  Within 1σ:       {:.0}%\n",
        report.calibration.fraction_within_1sigma * 100.0
    );

    // 4. Top discovery candidates
    println!(
        "=== Top {} Discovery Candidates ===",
        report.candidates.len()
    );
    println!(
        "{:<6} {:<6} {:<6} | {:<8} | {:<8} | {:<12} | {}",
        "Z", "N", "A", "Interest", "σ (MeV)", "Shell (MeV)", "Reason"
    );
    println!("{}", "-".repeat(90));

    for c in &report.candidates {
        println!(
            "{:<6} {:<6} {:<6} | {:<8.3} | {:<8.1} | {:<12.1} | {}",
            c.z, c.n, c.a, c.interest_score, c.sigma, c.shell_correction, c.reason
        );
    }

    // 5. Find the MOST promising novel isotopes
    println!("\n=== Most Promising Novel Isotopes ===");
    println!("(Unmeasured, strong shell enhancement, model uncertainty quantified)\n");

    let semf = SemiEmpiricalMassFormula::default();
    let landscape = StabilityLandscape::default();

    let mut best_candidates: Vec<_> = report
        .candidates
        .iter()
        .filter(|c| c.interest_score > 0.7 && c.shell_correction < -5.0)
        .collect();
    best_candidates.sort_by(|a, b| a.shell_correction.partial_cmp(&b.shell_correction).unwrap());

    if best_candidates.is_empty() {
        println!("  No high-confidence novel candidates found in this region.");
        println!("  This may indicate the model needs refinement for superheavy elements.");
    } else {
        for c in best_candidates.iter().take(10) {
            let ba = semf.binding_energy_per_nucleon(c.a, c.z);
            let q_alpha = semf.alpha_decay_q(c.a, c.z);
            let half_life = semf.geiger_nuttall_half_life(c.a, c.z);

            let hl_str = match half_life {
                Some(t) if t > 3.156e7 => format!("{:.1e} yr", t / 3.156e7),
                Some(t) if t > 86400.0 => format!("{:.1} days", t / 86400.0),
                Some(t) if t > 1.0 => format!("{:.1} s", t),
                Some(t) => format!("{:.1e} s", t),
                None => "stable?".to_string(),
            };

            println!(
                "  {} (Z={}, N={}, A={})",
                element_symbol(c.z),
                c.z,
                c.n,
                c.a
            );
            println!("    Shell correction: {:.1} MeV", c.shell_correction);
            println!("    B/A: {:.3} MeV/nucleon", ba);
            println!("    Q(α): {:.2} MeV", q_alpha);
            println!("    Predicted t½: {}", hl_str);
            println!("    Model σ: {:.1} MeV", c.sigma);
            println!("    Interest: {:.3}", c.interest_score);
            println!("    {}", c.reason);
            println!();
        }
    }

    // 6. Summary assessment
    println!("=== Honest Assessment ===");
    println!("Model: SEMF (Rohlf) + Woods-Saxon shell model (simplified)");
    println!(
        "Calibration: {} nuclei from AME2020",
        report.calibration.n_calibration
    );
    println!(
        "RMS accuracy: {:.1} MeV (vs ~0.5 MeV for FRDM(2012))",
        report.calibration.rms_residual
    );
    println!();
    println!("LIMITATIONS:");
    println!("  - Simplified shell model (no deformation, no pairing correlations)");
    println!("  - SEMF coefficients are mean-field (not fitted to superheavy region)");
    println!("  - Geiger-Nuttall half-lives are order-of-magnitude estimates");
    println!("  - No fission barrier calculation (dominant decay mode for Z>104)");
    println!();
    println!("WHAT THE MODEL CAN SAY:");
    println!("  - Qualitative prediction: enhanced stability near Z≈114-120, N≈180-184");
    println!("  - This is CONSISTENT with published FRDM/HFB predictions");
    println!("  - The model identifies WHERE to look, not precisely WHAT to find");
    println!();
    println!("WHAT WOULD MAKE THIS RESEARCH-GRADE:");
    println!("  - Replace shell model with ML (Random Forest on AME2020 features)");
    println!("  - Add fission barrier heights (Strutinsky + liquid drop)");
    println!("  - Deformation parameters (β₂, β₄) for prolate/oblate nuclei");
    println!("  - Full Hartree-Fock-Bogoliubov calculation");
}

fn element_symbol(z: u16) -> &'static str {
    match z {
        104 => "Rf",
        105 => "Db",
        106 => "Sg",
        107 => "Bh",
        108 => "Hs",
        109 => "Mt",
        110 => "Ds",
        111 => "Rg",
        112 => "Cn",
        113 => "Nh",
        114 => "Fl",
        115 => "Mc",
        116 => "Lv",
        117 => "Ts",
        118 => "Og",
        119 => "E119",
        120 => "E120",
        121 => "E121",
        122 => "E122",
        123 => "E123",
        124 => "E124",
        125 => "E125",
        126 => "E126",
        127 => "E127",
        128 => "E128",
        129 => "E129",
        130 => "E130",
        _ => "?",
    }
}
