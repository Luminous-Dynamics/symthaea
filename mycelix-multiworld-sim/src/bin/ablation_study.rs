// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ablation study: quantify each mechanism's contribution to simulation outcomes.
//!
//! For each toggleable mechanism, runs N seeds with the mechanism ON (baseline)
//! vs OFF, then reports ΔCVS, Cohen's d, and 95% CI from bootstrap.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin ablation_study
//! cargo run --release --bin ablation_study -- --seeds 50 --ticks 1800
//! ```
//!
//! # Output
//!
//! Prints a ranked table of mechanism contributions to CVS, plus CSV to stdout.

use mycelix_multiworld_sim::config::{
    EpochConfig, PolicyConfig, SimulationConfig, WorldSeedConfig,
};
use mycelix_multiworld_sim::statistics::{bootstrap_ci, cohens_d, paired_t_test};
use mycelix_multiworld_sim::stochastic::StochasticEngine;
use mycelix_multiworld_sim::MultiWorldSimulator;

/// A mechanism that can be toggled for ablation.
struct Mechanism {
    name: &'static str,
    /// Apply the ablation (disable the mechanism) to a PolicyConfig.
    ablate: fn(&mut PolicyConfig),
}

const MECHANISMS: &[Mechanism] = &[
    Mechanism {
        name: "disasters",
        ablate: |p| p.disasters_enabled = false,
    },
    Mechanism {
        name: "turchin_cycles",
        ablate: |p| p.turchin_cycles_enabled = false,
    },
    Mechanism {
        name: "fep_immiseration",
        ablate: |p| p.fep_immiseration_enabled = false,
    },
    Mechanism {
        name: "trust_weighted_gov",
        ablate: |p| p.trust_weighted_governance = false,
    },
    Mechanism {
        name: "education",
        ablate: |p| p.education_enabled = false,
    },
    Mechanism {
        name: "factions",
        ablate: |p| p.faction_enabled = false,
    },
    Mechanism {
        name: "migration",
        ablate: |p| p.migration_enabled = false,
    },
    Mechanism {
        name: "amendments",
        ablate: |p| p.amendment_enabled = false,
    },
    Mechanism {
        name: "sacred_stillness",
        ablate: |p| p.sacred_stillness_enabled = false,
    },
];

fn make_config(seed: u64, ticks: u32, policy: PolicyConfig) -> SimulationConfig {
    SimulationConfig {
        total_ticks: ticks,
        seed,
        initial_worlds: vec![
            WorldSeedConfig {
                name: "Earth".into(),
                location: "Earth".into(),
                founding_tick: 0,
                initial_population: 500,
                initial_resources: 1.0,
            },
            WorldSeedConfig {
                name: "Artemis Base".into(),
                location: "Moon".into(),
                founding_tick: 0,
                initial_population: 30,
                initial_resources: 0.3,
            },
        ],
        epoch_configs: vec![EpochConfig {
            id: 0,
            name: "Ablation".into(),
            start_tick: 0,
            end_tick: ticks,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy,
    }
}

fn run_seed(seed: u64, ticks: u32, policy: &PolicyConfig) -> (f64, f64, usize) {
    let config = make_config(seed, ticks, policy.clone());
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    (
        report.final_cvs,
        sim.worlds.iter().map(|w| w.mean_phi()).sum::<f64>() / sim.worlds.len().max(1) as f64,
        report.final_population,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds: usize = args
        .iter()
        .position(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let ticks: u32 = args
        .iter()
        .position(|a| a == "--ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800); // 150 years

    let seeds: Vec<u64> = (0..n_seeds).map(|i| 42 + i as u64 * 17).collect();

    eprintln!("=== ABLATION STUDY ===");
    eprintln!(
        "Seeds: {n_seeds}, Ticks: {ticks} ({:.0} years)",
        ticks as f64 / 12.0
    );
    eprintln!();

    // Run baseline (all mechanisms ON)
    eprintln!("Running baseline ({n_seeds} seeds)...");
    let baseline: Vec<(f64, f64, usize)> = seeds
        .iter()
        .map(|&s| run_seed(s, ticks, &PolicyConfig::default()))
        .collect();
    let baseline_cvs: Vec<f64> = baseline.iter().map(|r| r.0).collect();
    let baseline_phi: Vec<f64> = baseline.iter().map(|r| r.1).collect();

    let mut rng = StochasticEngine::new(42);
    let baseline_ci = bootstrap_ci(&baseline_cvs, 0.95, 2000, &mut rng);

    eprintln!(
        "Baseline CVS: {:.4} [{:.4}, {:.4}]",
        baseline_ci.as_ref().map(|c| c.mean).unwrap_or(0.0),
        baseline_ci.as_ref().map(|c| c.lower).unwrap_or(0.0),
        baseline_ci.as_ref().map(|c| c.upper).unwrap_or(0.0)
    );
    eprintln!();

    // CSV header
    println!("mechanism,baseline_cvs,ablated_cvs,delta_cvs,delta_cvs_ci_lower,delta_cvs_ci_upper,cohens_d,p_value,baseline_phi,ablated_phi,delta_phi");

    // Run each ablation
    let mut results: Vec<(&str, f64, f64, f64)> = Vec::new(); // (name, delta_cvs, d, p)

    for mech in MECHANISMS {
        eprint!("Ablating {}...", mech.name);
        let mut ablated_policy = PolicyConfig::default();
        (mech.ablate)(&mut ablated_policy);

        let ablated: Vec<(f64, f64, usize)> = seeds
            .iter()
            .map(|&s| run_seed(s, ticks, &ablated_policy))
            .collect();
        let ablated_cvs: Vec<f64> = ablated.iter().map(|r| r.0).collect();
        let ablated_phi: Vec<f64> = ablated.iter().map(|r| r.1).collect();

        let mean_base = baseline_cvs.iter().sum::<f64>() / baseline_cvs.len() as f64;
        let mean_abl = ablated_cvs.iter().sum::<f64>() / ablated_cvs.len() as f64;
        let delta = mean_base - mean_abl;

        let d = cohens_d(&baseline_cvs, &ablated_cvs).unwrap_or(0.0);
        let p = paired_t_test(&baseline_cvs, &ablated_cvs)
            .map(|t| t.p_value)
            .unwrap_or(1.0);

        // Bootstrap CI on the deltas
        let deltas: Vec<f64> = baseline_cvs
            .iter()
            .zip(ablated_cvs.iter())
            .map(|(b, a)| b - a)
            .collect();
        let delta_ci = bootstrap_ci(&deltas, 0.95, 2000, &mut rng);

        let mean_base_phi = baseline_phi.iter().sum::<f64>() / baseline_phi.len() as f64;
        let mean_abl_phi = ablated_phi.iter().sum::<f64>() / ablated_phi.len() as f64;

        println!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{:.4},{:.4}",
            mech.name,
            mean_base,
            mean_abl,
            delta,
            delta_ci.as_ref().map(|c| c.lower).unwrap_or(0.0),
            delta_ci.as_ref().map(|c| c.upper).unwrap_or(0.0),
            d,
            p,
            mean_base_phi,
            mean_abl_phi,
            mean_base_phi - mean_abl_phi,
        );

        let sig = if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.1 {
            "*"
        } else {
            ""
        };
        eprintln!(" ΔCVS={:+.4} d={:.2} p={:.4} {}", delta, d, p, sig);

        results.push((mech.name, delta, d, p));
    }

    // Ranked summary
    results.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!();
    eprintln!("=== MECHANISM IMPORTANCE RANKING (by |ΔCVS|) ===");
    eprintln!(
        "{:<25} {:>10} {:>10} {:>10} {:>5}",
        "Mechanism", "ΔCVS", "Cohen's d", "p-value", "Sig"
    );
    eprintln!("{}", "-".repeat(65));
    for (name, delta, d, p) in &results {
        let sig = if *p < 0.01 {
            "***"
        } else if *p < 0.05 {
            "**"
        } else if *p < 0.1 {
            "*"
        } else {
            "n.s."
        };
        eprintln!(
            "{:<25} {:>+10.4} {:>10.3} {:>10.4} {:>5}",
            name, delta, d, p, sig
        );
    }
    eprintln!();
    eprintln!("Significance: *** p<0.01, ** p<0.05, * p<0.1, n.s. = not significant");
}
