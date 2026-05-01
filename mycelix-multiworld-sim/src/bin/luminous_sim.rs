// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # luminous-sim — Unified Civilizational Simulator CLI
//!
//! Run simulations from a unified TOML config, export standardized reports,
//! and compare scenarios with A/B testing.
//!
//! ## Usage
//!
//! ```bash
//! # Run with default config
//! cargo run --bin luminous_sim
//!
//! # Run with custom config
//! cargo run --bin luminous_sim -- --config scenarios/unified_default.toml
//!
//! # Override specific parameters
//! cargo run --bin luminous_sim -- --years 1000 --seed 123
//!
//! # Export full documented default config
//! cargo run --bin luminous_sim -- --export-config
//!
//! # Run sensitivity analysis on top-5 parameters
//! cargo run --bin luminous_sim -- --sensitivity 5
//! ```

use luminous_sim_core::{
    module_trait::ReportSection,
    report::{ExecutiveSummary, ReproducibilityInfo, SimulationOutcome},
    HonestAssessment, StandardizedReport, UnifiedConfig,
};
use mycelix_multiworld_sim::{
    config::SimulationConfig, unified_config_bridge, MultiWorldSimulator,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --export-config: dump the documented default config
    if args.iter().any(|a| a == "--export-config") {
        let config = UnifiedConfig::default();
        match config.to_toml() {
            Ok(s) => println!("{}", s),
            Err(e) => eprintln!("Failed to serialize config: {e}"),
        }
        return;
    }

    // --help
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // Parse config source
    let config_path = get_arg(&args, "--config");
    let override_years = get_arg(&args, "--years").and_then(|s| s.parse::<u32>().ok());
    let override_seed = get_arg(&args, "--seed").and_then(|s| s.parse::<u64>().ok());
    let sensitivity_n = get_arg(&args, "--sensitivity").and_then(|s| s.parse::<usize>().ok());
    let format = get_arg(&args, "--format").unwrap_or_else(|| "text".into());

    // Load config
    let unified = if let Some(path) = &config_path {
        match std::fs::read_to_string(path) {
            Ok(toml_str) => match UnifiedConfig::from_toml(&toml_str) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to parse config {path}: {e}");
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Failed to read {path}: {e}");
                std::process::exit(1);
            }
        }
    } else {
        UnifiedConfig::default()
    };

    // Convert to existing config types
    let (mut sim_config, policy, _params) =
        match unified_config_bridge::from_unified_toml(&unified.to_toml().unwrap_or_default()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Config conversion failed: {e}");
                std::process::exit(1);
            }
        };

    // Apply CLI overrides
    if let Some(years) = override_years {
        sim_config.total_ticks = years * 12;
    }
    if let Some(seed) = override_seed {
        sim_config.seed = seed;
    }
    sim_config.policy = policy;

    let duration_years = sim_config.total_ticks / 12;
    let seed = sim_config.seed;

    eprintln!("=== luminous-sim v0.1.0 ===");
    eprintln!(
        "Duration: {} years ({} ticks)",
        duration_years, sim_config.total_ticks
    );
    eprintln!("Seed: {}", seed);
    if let Some(ref path) = config_path {
        eprintln!("Config: {}", path);
    }
    eprintln!("Feedback loops: disaster_trauma={}, harmony_policy={}, affect_behavior={}, consciousness_governance={}, cultural_memory={}",
        unified.feedback_loops.disaster_trauma,
        unified.feedback_loops.harmony_policy,
        unified.feedback_loops.affect_behavior,
        unified.feedback_loops.consciousness_governance,
        unified.feedback_loops.cultural_memory,
    );
    eprintln!();

    // Run simulation
    let start = std::time::Instant::now();
    let mut sim = MultiWorldSimulator::new(sim_config);
    let report = sim.run();
    let wall_time = start.elapsed().as_secs_f64();

    // Sensitivity analysis (if requested)
    if let Some(n) = sensitivity_n {
        eprintln!("Running sensitivity analysis (top {} parameters)...", n);
        run_sensitivity_analysis(n, &unified, duration_years, seed);
        return;
    }

    // Build standardized report
    let std_report = build_standardized_report(&report, &sim, &unified, seed, wall_time);

    // Output
    match format.as_str() {
        "json" => match std_report.to_json() {
            Ok(json) => println!("{}", json),
            Err(e) => eprintln!("JSON export failed: {e}"),
        },
        "markdown" | "md" => {
            println!("{}", std_report.to_markdown());
        }
        _ => {
            // Default: human-readable text (existing format)
            if report.duration_years >= 300.0 {
                println!("{}", report.summary_extended());
            } else {
                println!("{}", report.summary());
            }

            // Epoch snapshots
            println!("\n=== EPOCH SNAPSHOTS ===");
            for snap in &report.epoch_snapshots {
                println!(
                    "  Yr {:>5.1} | Pop {:>6} | Phi {:.3} | CVS {:.3}",
                    snap.tick as f64 / 12.0,
                    snap.total_population,
                    snap.mean_phi,
                    snap.civilization_viability_score
                );
            }

            // Narrative
            if !sim.narrative_engine.events.is_empty() {
                println!("\n{}", sim.narrative_engine.format_chronicle());
            }

            // Honest assessment
            println!("\n=== HONEST ASSESSMENT ===");
            for lim in &std_report.honest_assessment.limitations {
                println!("  - {}", lim);
            }
            println!("\nWhat this simulation CANNOT predict:");
            for item in &std_report.honest_assessment.cannot_predict {
                println!("  - {}", item);
            }
        }
    }

    eprintln!("\n=== Completed in {:.1}s ===", wall_time);
    if report.survived {
        eprintln!(
            "RESULT: Civilization survived {} years (CVS: {:.4})",
            duration_years, report.final_cvs
        );
    } else {
        eprintln!("RESULT: Civilization collapsed");
    }
}

fn build_standardized_report(
    report: &mycelix_multiworld_sim::report::CivilizationReport,
    sim: &MultiWorldSimulator,
    _unified: &UnifiedConfig,
    seed: u64,
    wall_time: f64,
) -> StandardizedReport {
    let outcome = if report.final_cvs > 0.7 {
        SimulationOutcome::Thrived {
            peak_cvs: report.final_cvs,
        }
    } else if report.survived {
        SimulationOutcome::Survived {
            final_cvs: report.final_cvs,
        }
    } else {
        SimulationOutcome::Collapsed {
            at_year: report.duration_years,
        }
    };
    let worlds_surviving = sim.worlds.iter().filter(|w| w.population() > 0).count();

    let mut key_findings = Vec::new();
    key_findings.push(format!(
        "Final population: {} across {} worlds ({} surviving)",
        report.final_population,
        sim.worlds.len(),
        worlds_surviving
    ));
    key_findings.push(format!(
        "CVS: {:.4} ({})",
        report.final_cvs,
        if report.final_cvs > 0.7 {
            "thriving"
        } else if report.final_cvs > 0.3 {
            "surviving"
        } else {
            "struggling"
        }
    ));
    if report.tech_milestones_achieved > 0 {
        key_findings.push(format!(
            "{} technology milestones achieved",
            report.tech_milestones_achieved
        ));
    }
    key_findings.push(format!(
        "{} total disasters endured",
        report.total_disasters
    ));

    // Viability report section
    let viability_section = ReportSection {
        module_id: "viability".into(),
        module_name: "Viability Engine (Phase 0)".into(),
        summary: format!(
            "Thermodynamic axioms enforced. West-Bettencourt scaling active. {} viability violations recorded.",
            sim.viability_engine.violations.len()
        ),
        metrics: vec![
            ("worlds_tracked".into(), sim.viability_engine.ledgers.len() as f64),
        ],
        warnings: if !sim.viability_engine.violations.is_empty() {
            vec![format!("{} viability violations detected", sim.viability_engine.violations.len())]
        } else {
            vec![]
        },
    };

    // Earth population section
    let earth_section = if let Some(ref pop_model) = sim.earth_pop_model {
        Some(ReportSection {
            module_id: "earth_population".into(),
            module_name: "Earth Population Model (Phase 2)".into(),
            summary: format!(
                "12-region cohort model. Total Earth population: {:.0}M. Climate temp anomaly: {:.2}°C.",
                pop_model.total_population,
                pop_model.climate.global_temp_anomaly,
            ),
            metrics: vec![
                ("total_population_millions".into(), pop_model.total_population),
                ("global_temp_anomaly_c".into(), pop_model.climate.global_temp_anomaly),
                ("annual_emissions_gtco2".into(), pop_model.climate.annual_emissions),
                ("carbon_intensity_kg_per_usd".into(), pop_model.climate.carbon_intensity),
            ],
            warnings: if pop_model.climate.global_temp_anomaly > 3.0 {
                vec!["Global temperature anomaly exceeds 3°C — severe climate damage active".into()]
            } else {
                vec![]
            },
        })
    } else {
        None
    };

    let mut sections = vec![viability_section];
    if let Some(s) = earth_section {
        sections.push(s);
    }

    StandardizedReport {
        executive_summary: ExecutiveSummary {
            title: "Luminous Dynamics Civilization Simulation".into(),
            outcome: outcome.to_string(),
            key_findings,
            wall_time_seconds: wall_time,
            final_cvs: report.final_cvs,
            final_population: report.final_population,
            worlds_surviving,
            critical_events: report.total_disasters, // Map total disasters as critical events
        },
        module_reports: sections,
        honest_assessment: HonestAssessment {
            limitations: vec![
                "Simulation abstracts many real-world dynamics".into(),
                "Parameter calibration against historical data is incomplete".into(),
            ],
            cannot_predict: vec![
                "Black swan geopolitical events".into(),
                "Technological breakthroughs outside trained priors".into(),
            ],
            confidence_level: "exploratory".into(),
        },
        reproducibility: ReproducibilityInfo {
            seed,
            total_ticks: (report.duration_years * 12.0) as u32,
            version: "0.1.0".into(),
            config_hash: format!("{:x}", seed),
        },
    }
}

fn run_sensitivity_analysis(top_n: usize, base_config: &UnifiedConfig, years: u32, base_seed: u64) {
    // Key parameters to test
    let params = [
        ("stress_baseline", 0.002, 0.008),
        ("trauma_disaster_scale", 0.1, 0.5),
        ("spoilage_energy", 0.05, 0.25),
        ("radiation_mars_sv_month", 0.010, 0.030),
        ("pair_bond_rate", 0.01, 0.04),
        ("trade_openness", 0.3, 1.0),
        ("defense_spending", 0.0, 0.15),
    ];

    let n = top_n.min(params.len());
    println!("Sensitivity Analysis: testing {} parameters\n", n);
    println!(
        "{:<30} {:>10} {:>10} {:>10} {:>10}",
        "Parameter", "Low", "Default", "High", "|ΔCVS|"
    );
    println!("{}", "-".repeat(75));

    for &(name, low, high) in params.iter().take(n) {
        // Run baseline
        let base_cvs = run_quick_sim(base_config, years, base_seed);

        // Run low variant
        let mut low_config = base_config.clone();
        set_param(&mut low_config, name, low);
        let low_cvs = run_quick_sim(&low_config, years, base_seed);

        // Run high variant
        let mut high_config = base_config.clone();
        set_param(&mut high_config, name, high);
        let high_cvs = run_quick_sim(&high_config, years, base_seed);

        let delta = (high_cvs - low_cvs).abs();
        println!(
            "{:<30} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            name, low_cvs, base_cvs, high_cvs, delta
        );
    }
}

fn run_quick_sim(config: &UnifiedConfig, years: u32, seed: u64) -> f64 {
    let toml_str = config.to_toml().unwrap_or_default();
    let (mut sim_config, policy, _) = unified_config_bridge::from_unified_toml(&toml_str)
        .unwrap_or_else(|_| {
            let c = SimulationConfig::default_150_year();
            (
                c.clone(),
                c.policy,
                mycelix_multiworld_sim::constants::SimulationParams::default(),
            )
        });
    sim_config.total_ticks = years.min(150) * 12; // Cap at 150yr for speed
    sim_config.seed = seed;
    sim_config.policy = policy;

    let mut sim = MultiWorldSimulator::new(sim_config);
    let report = sim.run();
    report.final_cvs
}

fn set_param(config: &mut UnifiedConfig, name: &str, value: f64) {
    match name {
        "stress_baseline" => config.params.stress_baseline = Some(value),
        "trauma_disaster_scale" => config.params.trauma_disaster_scale = Some(value),
        "spoilage_energy" => config.params.spoilage_energy = Some(value),
        "radiation_mars_sv_month" => config.params.radiation_mars_sv_month = Some(value),
        "pair_bond_rate" => config.policy.pair_bond_rate = Some(value),
        "trade_openness" => config.policy.trade_openness = Some(value),
        "defense_spending" => config.policy.defense_spending = Some(value),
        _ => {}
    }
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn print_help() {
    eprintln!(
        "luminous-sim — Unified Civilizational Simulator

USAGE:
    luminous-sim [OPTIONS]

OPTIONS:
    --config <path>       Load unified TOML config file
    --years <n>           Override simulation duration (years)
    --seed <n>            Override RNG seed
    --format <fmt>        Output format: text (default), json, markdown
    --sensitivity <n>     Run sensitivity analysis on top-N parameters
    --export-config       Print the full documented default config
    --help                Show this help

EXAMPLES:
    luminous-sim --config scenarios/unified_default.toml
    luminous-sim --years 1000 --seed 123 --format json
    luminous-sim --sensitivity 5
    luminous-sim --export-config > my_scenario.toml
"
    );
}
