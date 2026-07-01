// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spark Engine CLI
//!
//! Command-line interface for LCF anomaly investigation.
//!
//! ## Usage
//!
//! ```bash
//! # Analyze an observation
//! spark analyze --rate 1000 --material Pd --trigger xray
//!
//! # List literature database
//! spark literature --filter neutrons --min-credibility 7
//!
//! # Compare hypotheses
//! spark hypotheses
//!
//! # Design experiments
//! spark experiments --phase validation
//! ```

use std::env;

use spark_engine::constants::*;
use spark_engine::experimental_design::ExperimentDesigner;
use spark_engine::hypothesis_models::HypothesisComparison;
use spark_engine::literature::{HostMaterial as LitMaterial, LiteratureDatabase, TriggerMethod};
use spark_engine::physics::GamowIntegration;
use spark_engine::prelude::*;
use spark_engine::rate_gap::{
    ExperimentalConditions, HostMaterial, RateGapCalculator, TriggerType,
};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "analyze" => cmd_analyze(&args[2..]),
        "literature" | "lit" => cmd_literature(&args[2..]),
        "hypotheses" | "hyp" => cmd_hypotheses(&args[2..]),
        "experiments" | "exp" => cmd_experiments(&args[2..]),
        "gamow" => cmd_gamow(&args[2..]),
        "help" | "--help" | "-h" => print_help(),
        "version" | "--version" | "-V" => println!("spark-engine {}", spark_engine::VERSION),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_help();
        }
    }
}

fn print_help() {
    println!("spark-engine - LCF Anomaly Investigation Tool");
    println!();
    println!("USAGE:");
    println!("    spark <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    analyze     Analyze observed rate against Gamow predictions");
    println!("    literature  Query the literature database");
    println!("    hypotheses  Compare anomaly hypotheses");
    println!("    experiments Design experimental program");
    println!("    gamow       Calculate Gamow fusion rates");
    println!("    help        Show this help message");
    println!("    version     Show version");
    println!();
    println!("EXAMPLES:");
    println!("    spark analyze --rate 1000 --material Er");
    println!("    spark literature --filter neutrons");
    println!("    spark hypotheses");
    println!("    spark gamow --temp 1000 --screening 309");
}

fn cmd_analyze(args: &[String]) {
    let mut rate = 1000.0;
    let mut material = HostMaterial::Palladium;
    let mut trigger = TriggerType::XRay;
    let mut temp = 300.0;
    let mut loading = 0.7;
    let mut volume = 0.01;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rate" | "-r" => {
                i += 1;
                if i < args.len() {
                    rate = args[i].parse().unwrap_or(1000.0);
                }
            }
            "--material" | "-m" => {
                i += 1;
                if i < args.len() {
                    material = match args[i].to_lowercase().as_str() {
                        "pd" | "palladium" => HostMaterial::Palladium,
                        "ti" | "titanium" => HostMaterial::Titanium,
                        "ni" | "nickel" => HostMaterial::Nickel,
                        "er" | "erbium" => HostMaterial::Erbium,
                        _ => HostMaterial::Palladium,
                    };
                }
            }
            "--trigger" | "-t" => {
                i += 1;
                if i < args.len() {
                    trigger = match args[i].to_lowercase().as_str() {
                        "xray" | "x-ray" => TriggerType::XRay,
                        "gamma" => TriggerType::Gamma,
                        "acoustic" => TriggerType::Acoustic,
                        "electrolysis" | "elec" => TriggerType::Electrochemical,
                        "thermal" => TriggerType::Thermal,
                        "none" => TriggerType::None,
                        _ => TriggerType::XRay,
                    };
                }
            }
            "--temp" | "-T" => {
                i += 1;
                if i < args.len() {
                    temp = args[i].parse().unwrap_or(300.0);
                }
            }
            "--loading" | "-l" => {
                i += 1;
                if i < args.len() {
                    loading = args[i].parse().unwrap_or(0.7);
                }
            }
            "--volume" | "-v" => {
                i += 1;
                if i < args.len() {
                    volume = args[i].parse().unwrap_or(0.01);
                }
            }
            "--help" | "-h" => {
                println!("spark analyze - Analyze observed rate against Gamow predictions");
                println!();
                println!("OPTIONS:");
                println!("    --rate, -r <N>       Observed neutron rate (n/s) [default: 1000]");
                println!("    --material, -m <M>   Host material: Pd, Ti, Ni, Er [default: Pd]");
                println!(
                    "    --trigger, -t <T>    Trigger: xray, gamma, acoustic, elec [default: xray]"
                );
                println!("    --temp, -T <K>       Temperature (K) [default: 300]");
                println!("    --loading, -l <R>    D/M loading ratio [default: 0.7]");
                println!("    --volume, -v <V>     Active volume (cm³) [default: 0.01]");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let conditions = ExperimentalConditions {
        temperature_k: temp,
        host_material: material,
        loading_ratio: loading,
        active_volume_cm3: volume,
        trigger,
        trigger_intensity: 1.0,
    };

    let analysis = RateGapCalculator::analyze(&conditions, rate);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              RATE GAP ANALYSIS                             ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Input Conditions:");
    println!("  • Material:     {:?}", material);
    println!("  • Screening:    {:.0} eV", material.screening_ev());
    println!("  • Temperature:  {:.0} K", temp);
    println!("  • Loading:      {:.2}", loading);
    println!("  • Volume:       {:.4} cm³", volume);
    println!("  • Trigger:      {:?}", trigger);
    println!();
    println!("Results:");
    println!("  • Observed rate:    {:.2e} n/s", analysis.observed_rate);
    println!(
        "  • Predicted rate:   {:.2e} n/s",
        analysis.gamow_predicted_rate
    );
    println!("  • Gap factor:       {:.2e}", analysis.gap_factor);
    println!("  • Gap orders:       {:.1}", analysis.gap_orders);
    println!(
        "  • Classification:   {:?}",
        analysis.assessment.classification
    );
    println!();
    println!("Gap Breakdown:");
    println!(
        "  • Screening:        {:.2e}×",
        analysis.gap_breakdown.screening_enhancement
    );
    println!(
        "  • Phonons:          {:.2e}×",
        analysis.gap_breakdown.phonon_enhancement
    );
    println!(
        "  • Hot spots:        {:.2e}×",
        analysis.gap_breakdown.hot_spot_enhancement
    );
    println!(
        "  • Unexplained:      {:.2e}× ({:.0} orders)",
        analysis.gap_breakdown.unexplained_factor, analysis.gap_breakdown.unexplained_orders
    );
    println!();
    println!("Recommendations:");
    for rec in &analysis.assessment.recommendations {
        println!("  • {}", rec);
    }
}

fn cmd_literature(args: &[String]) {
    let mut filter_neutrons = false;
    let mut filter_heat = false;
    let mut min_credibility = 0.0;
    let mut material_filter: Option<LitMaterial> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--filter" | "-f" => {
                i += 1;
                if i < args.len() {
                    match args[i].to_lowercase().as_str() {
                        "neutrons" | "n" => filter_neutrons = true,
                        "heat" | "h" => filter_heat = true,
                        _ => {}
                    }
                }
            }
            "--min-credibility" | "-c" => {
                i += 1;
                if i < args.len() {
                    min_credibility = args[i].parse().unwrap_or(0.0);
                }
            }
            "--material" | "-m" => {
                i += 1;
                if i < args.len() {
                    material_filter = Some(match args[i].to_lowercase().as_str() {
                        "pd" | "palladium" => LitMaterial::Palladium,
                        "ti" | "titanium" => LitMaterial::Titanium,
                        "ni" | "nickel" => LitMaterial::Nickel,
                        _ => LitMaterial::Palladium,
                    });
                }
            }
            "--help" | "-h" => {
                println!("spark literature - Query the literature database");
                println!();
                println!("OPTIONS:");
                println!("    --filter, -f <TYPE>      Filter by effect: neutrons, heat");
                println!("    --min-credibility, -c <N> Minimum credibility score [default: 0]");
                println!("    --material, -m <M>       Filter by material: Pd, Ti, Ni");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let db = LiteratureDatabase::canonical();

    let observations: Vec<_> = if filter_neutrons {
        db.with_neutrons()
    } else if filter_heat {
        db.with_excess_heat()
    } else if let Some(mat) = material_filter {
        db.by_material(mat)
    } else if min_credibility > 0.0 {
        db.by_credibility(min_credibility)
    } else {
        db.all().iter().collect()
    };

    let filtered: Vec<_> = observations
        .into_iter()
        .filter(|o| o.credibility.score >= min_credibility)
        .collect();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              LITERATURE DATABASE                           ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Found {} observations", filtered.len());
    println!();

    for obs in filtered {
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!(
            "│ {} ({})                                      ",
            obs.author, obs.year
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Venue:       {}", obs.venue);
        println!("│ Material:    {:?}", obs.host_material);
        println!("│ Trigger:     {:?}", obs.trigger);
        println!("│ Credibility: {:.1}/10", obs.credibility.score);
        if !obs.effects.is_empty() {
            println!("│ Effects:");
            for effect in &obs.effects {
                match effect {
                    spark_engine::literature::ObservedEffect::Neutrons {
                        rate_per_s,
                        energy_mev,
                    } => {
                        println!(
                            "│   • Neutrons: {:.0} n/s, {} MeV",
                            rate_per_s,
                            energy_mev
                                .map(|e| format!("{:.2}", e))
                                .unwrap_or("?".to_string())
                        );
                    }
                    spark_engine::literature::ObservedEffect::ExcessHeat { power_w, .. } => {
                        println!("│   • Excess heat: {:.1} W", power_w);
                    }
                    _ => {}
                }
            }
        }
        println!(
            "│ Notes: {}",
            obs.notes.chars().take(55).collect::<String>()
        );
        println!("└─────────────────────────────────────────────────────────────┘");
        println!();
    }
}

fn cmd_hypotheses(_args: &[String]) {
    let comparison = HypothesisComparison::compare_all();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              HYPOTHESIS COMPARISON                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    println!("Target: Explain NASA observation of ~10³ n/s");
    println!();

    println!("1. HOT SPOTS (transient high-T from X-ray absorption)");
    println!(
        "   • Predicted rate:     {:.2e} n/s",
        comparison.hot_spot.predicted_rate
    );
    println!(
        "   • Gap from observed:  {:.2e}×",
        comparison.hot_spot.gap_factor
    );
    println!(
        "   • Required T:         {:.2e} K",
        comparison.hot_spot.required_temperature_k
    );
    println!(
        "   • Plausible:          {}",
        if comparison.hot_spot.plausible {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    println!("2. PHONON CASCADE (coherent phonon energy addition)");
    println!(
        "   • Predicted (3 modes): {:.2e} n/s",
        comparison.phonon_cascade.predicted_rate_n3
    );
    println!(
        "   • Gap from observed:   {:.2e}×",
        comparison.phonon_cascade.gap_factor
    );
    println!(
        "   • Phonons needed:      {}",
        comparison.phonon_cascade.phonons_needed
    );
    println!(
        "   • Plausible:           {}",
        if comparison.phonon_cascade.plausible {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    println!("3. SUPER-SCREENING (enhanced Ue beyond measured)");
    println!(
        "   • Predicted rate:     {:.2e} n/s",
        comparison.super_screening.predicted_rate
    );
    println!(
        "   • Gap from observed:  {:.2e}×",
        comparison.super_screening.gap_factor
    );
    println!(
        "   • Screening needed:   {:.0} eV",
        comparison.super_screening.screening_needed_ev
    );
    println!(
        "   • Plausible:          {}",
        if comparison.super_screening.plausible {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("BEST FIT: {}", comparison.best_fit);
    println!("═══════════════════════════════════════════════════════════════");
}

fn cmd_experiments(args: &[String]) {
    let mut phase_filter: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--phase" | "-p" => {
                i += 1;
                if i < args.len() {
                    phase_filter = match args[i].to_lowercase().as_str() {
                        "1" | "validation" => Some(0),
                        "2" | "mechanism" => Some(1),
                        "3" | "optimization" => Some(2),
                        _ => None,
                    };
                }
            }
            "--help" | "-h" => {
                println!("spark experiments - Design experimental program");
                println!();
                println!("OPTIONS:");
                println!(
                    "    --phase, -p <N>   Show specific phase: 1/validation, 2/mechanism, 3/optimization"
                );
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let program = ExperimentDesigner::design_program();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              EXPERIMENTAL PROGRAM                          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Program: {}", program.name);
    println!("Total Budget: ${:.0}K", program.total_cost_usd / 1000.0);
    println!(
        "Total Duration: {:.0} months",
        program.total_duration_months
    );
    println!();

    let phases_to_show: Vec<_> = if let Some(idx) = phase_filter {
        vec![&program.phases[idx]]
    } else {
        program.phases.iter().collect()
    };

    for (i, phase) in phases_to_show.iter().enumerate() {
        let phase_num = phase_filter.unwrap_or(i) + 1;
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!(
            "│ PHASE {}: {}                                      ",
            phase_num,
            phase.name.to_uppercase()
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Goal: {}", phase.goal);
        println!(
            "│ Budget: ${:.0}K  Duration: {} months",
            phase.cost_usd / 1000.0,
            phase.duration_months
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ EXPERIMENTS:");
        for exp in &phase.experiments {
            println!(
                "│   [{:.0}%] {} - ${:.0}K",
                exp.priority * 100.0,
                exp.name,
                exp.estimated_cost_usd / 1000.0
            );
        }
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ SUCCESS CRITERIA:");
        for criterion in &phase.success_criteria {
            println!("│   ✓ {}", criterion);
        }
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ GO/NO-GO: {}", phase.go_nogo);
        println!("└─────────────────────────────────────────────────────────────┘");
        println!();
    }
}

fn cmd_gamow(args: &[String]) {
    let mut temp = 300.0;
    let mut screening = SCREENING_PD_EV;
    let mut phonons = 0u32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" | "-T" => {
                i += 1;
                if i < args.len() {
                    temp = args[i].parse().unwrap_or(300.0);
                }
            }
            "--screening" | "-s" => {
                i += 1;
                if i < args.len() {
                    screening = args[i].parse().unwrap_or(SCREENING_PD_EV);
                }
            }
            "--phonons" | "-p" => {
                i += 1;
                if i < args.len() {
                    phonons = args[i].parse().unwrap_or(0);
                }
            }
            "--help" | "-h" => {
                println!("spark gamow - Calculate Gamow fusion rates");
                println!();
                println!("OPTIONS:");
                println!("    --temp, -T <K>       Temperature (K) [default: 300]");
                println!("    --screening, -s <eV> Screening energy (eV) [default: 309]");
                println!("    --phonons, -p <N>    Coherent phonon modes [default: 0]");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let gamow = GamowIntegration::dd_rate(temp, screening, phonons);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              GAMOW CALCULATION                             ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Input Parameters:");
    println!("  • Temperature:     {:.1} K", temp);
    println!("  • Screening Ue:    {:.1} eV", screening);
    println!("  • Phonon modes:    {}", phonons);
    println!();
    println!("Results:");
    println!("  • <σv>:            {:.3e} cm³/s", gamow.sigma_v_cm3_s);
    println!("  • Gamow peak:      {:.4} keV", gamow.gamow_peak_kev);
    println!("  • Gamow width:     {:.4} keV", gamow.gamow_width_kev);
    println!("  • Screening factor: {:.2e}", gamow.screening_enhancement);
    println!("  • Phonon factor:   {:.2e}", gamow.phonon_enhancement);
    println!();

    // Q factor
    let q_params = QFactorParams::lcf_typical();
    let (q, achievable) = QFactor::compute(&gamow, &q_params);
    println!("Q Factor (LCF typical conditions):");
    println!("  • Q = {:.2e}", q);
    println!(
        "  • Energy gain: {}",
        if achievable {
            "POSSIBLE"
        } else {
            "NOT ACHIEVABLE"
        }
    );
}
