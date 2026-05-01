// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sensitivity tornado: identify which parameters matter most.
//!
//! For each of N continuous parameters, runs seeds at default, +20%, and -20%.
//! Reports ΔCVS sorted by absolute effect size (tornado plot data).
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin sensitivity_tornado
//! cargo run --release --bin sensitivity_tornado -- --seeds 10 --ticks 1800
//! ```
//!
//! # Output
//!
//! CSV to stdout, ranked tornado table to stderr.

use mycelix_multiworld_sim::config::{
    EpochConfig, PolicyConfig, SimulationConfig, WorldSeedConfig,
};
use mycelix_multiworld_sim::constants::SimulationParams;
use mycelix_multiworld_sim::statistics::bootstrap_ci;
use mycelix_multiworld_sim::stochastic::StochasticEngine;
use mycelix_multiworld_sim::MultiWorldSimulator;

/// A parameter that can be varied for sensitivity analysis.
struct Parameter {
    name: &'static str,
    unit: &'static str,
    get: fn(&PolicyConfig, &SimulationParams) -> f64,
    set_policy: Option<fn(&mut PolicyConfig, f64)>,
    set_params: Option<fn(&mut SimulationParams, f64)>,
}

const PARAMETERS: &[Parameter] = &[
    Parameter {
        name: "pair_bond_rate",
        unit: "/tick",
        get: |p, _| p.pair_bond_rate,
        set_policy: Some(|p, v| p.pair_bond_rate = v),
        set_params: None,
    },
    Parameter {
        name: "care_effectiveness",
        unit: "mult",
        get: |p, _| p.care_effectiveness,
        set_policy: Some(|p, v| p.care_effectiveness = v),
        set_params: None,
    },
    Parameter {
        name: "trade_openness",
        unit: "[0,1]",
        get: |p, _| p.trade_openness,
        set_policy: Some(|p, v| p.trade_openness = v),
        set_params: None,
    },
    Parameter {
        name: "consciousness_boost",
        unit: "[0,1]",
        get: |p, _| p.consciousness_boost,
        set_policy: Some(|p, v| p.consciousness_boost = v),
        set_params: None,
    },
    Parameter {
        name: "spoilage_food",
        unit: "/tick",
        get: |_, s| s.spoilage_food,
        set_policy: None,
        set_params: Some(|s, v| s.spoilage_food = v),
    },
    Parameter {
        name: "radiation_mars",
        unit: "Sv/mo",
        get: |_, s| s.radiation_mars_sv_month,
        set_policy: None,
        set_params: Some(|s, v| s.radiation_mars_sv_month = v),
    },
    Parameter {
        name: "spoilage_energy",
        unit: "/tick",
        get: |_, s| s.spoilage_energy,
        set_policy: None,
        set_params: Some(|s, v| s.spoilage_energy = v),
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
            name: "Sensitivity".into(),
            start_tick: 0,
            end_tick: ticks,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy,
    }
}

fn run_seed(seed: u64, ticks: u32, policy: &PolicyConfig, params: &SimulationParams) -> f64 {
    let config = make_config(seed, ticks, policy.clone());
    let mut sim = MultiWorldSimulator::new(config).with_params(params.clone());
    let report = sim.run();
    report.final_cvs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds: usize = args
        .iter()
        .position(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let ticks: u32 = args
        .iter()
        .position(|a| a == "--ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800);
    let perturbation: f64 = args
        .iter()
        .position(|a| a == "--perturbation")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.20); // ±20%

    let seeds: Vec<u64> = (0..n_seeds).map(|i| 42 + i as u64 * 17).collect();
    let default_policy = PolicyConfig::default();
    let default_params = SimulationParams::default();

    eprintln!("=== SENSITIVITY TORNADO ===");
    eprintln!(
        "Seeds: {n_seeds}, Ticks: {ticks} ({:.0} years), Perturbation: ±{:.0}%",
        ticks as f64 / 12.0,
        perturbation * 100.0
    );
    eprintln!();

    // Run baseline
    eprintln!("Running baseline ({n_seeds} seeds)...");
    let baseline_cvs: Vec<f64> = seeds
        .iter()
        .map(|&s| run_seed(s, ticks, &default_policy, &default_params))
        .collect();
    let mean_baseline = baseline_cvs.iter().sum::<f64>() / baseline_cvs.len() as f64;
    eprintln!("Baseline CVS: {:.4}", mean_baseline);
    eprintln!();

    // CSV header
    println!("parameter,default,low_value,high_value,cvs_at_low,cvs_at_high,delta_low,delta_high,range,ci_low,ci_high");

    let mut results: Vec<(&str, f64, f64, f64)> = Vec::new(); // (name, delta_low, delta_high, range)

    for param in PARAMETERS {
        let default_val = (param.get)(&default_policy, &default_params);

        // Skip parameters with zero default (can't perturb by %)
        if default_val.abs() < 1e-10 {
            eprintln!("Skipping {} (default=0)", param.name);
            continue;
        }

        let low_val = default_val * (1.0 - perturbation);
        let high_val = default_val * (1.0 + perturbation);

        eprint!(
            "Testing {} ({:.4} → [{:.4}, {:.4}])...",
            param.name, default_val, low_val, high_val
        );

        // Run low
        let low_cvs: Vec<f64> = seeds
            .iter()
            .map(|&s| {
                let mut pol = default_policy.clone();
                let mut par = default_params.clone();
                if let Some(setter) = param.set_policy {
                    setter(&mut pol, low_val);
                }
                if let Some(setter) = param.set_params {
                    setter(&mut par, low_val);
                }
                run_seed(s, ticks, &pol, &par)
            })
            .collect();

        // Run high
        let high_cvs: Vec<f64> = seeds
            .iter()
            .map(|&s| {
                let mut pol = default_policy.clone();
                let mut par = default_params.clone();
                if let Some(setter) = param.set_policy {
                    setter(&mut pol, high_val);
                }
                if let Some(setter) = param.set_params {
                    setter(&mut par, high_val);
                }
                run_seed(s, ticks, &pol, &par)
            })
            .collect();

        let mean_low = low_cvs.iter().sum::<f64>() / low_cvs.len() as f64;
        let mean_high = high_cvs.iter().sum::<f64>() / high_cvs.len() as f64;
        let delta_low = mean_low - mean_baseline;
        let delta_high = mean_high - mean_baseline;
        let range = (delta_high - delta_low).abs();

        // Bootstrap CI on the range
        let range_samples: Vec<f64> = low_cvs
            .iter()
            .zip(high_cvs.iter())
            .map(|(l, h)| (h - l).abs())
            .collect();
        let mut rng = StochasticEngine::new(42);
        let ci = bootstrap_ci(&range_samples, 0.95, 1000, &mut rng);

        println!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:+.4},{:+.4},{:.4},{:.4},{:.4}",
            param.name,
            default_val,
            low_val,
            high_val,
            mean_low,
            mean_high,
            delta_low,
            delta_high,
            range,
            ci.as_ref().map(|c| c.lower).unwrap_or(0.0),
            ci.as_ref().map(|c| c.upper).unwrap_or(0.0),
        );

        eprintln!(
            " range={:.4} (low {:+.4}, high {:+.4})",
            range, delta_low, delta_high
        );
        results.push((param.name, delta_low, delta_high, range));
    }

    // Tornado ranking
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!();
    eprintln!("=== TORNADO RANKING (by sensitivity range) ===");
    eprintln!(
        "{:<25} {:>10} {:>10} {:>10}",
        "Parameter", "Δ at -20%", "Δ at +20%", "Range"
    );
    eprintln!("{}", "-".repeat(60));
    for (name, dl, dh, range) in &results {
        let bar_len = (range / results[0].3 * 30.0).round() as usize;
        let bar: String = "█".repeat(bar_len.max(1));
        eprintln!(
            "{:<25} {:>+10.4} {:>+10.4} {:>10.4} {}",
            name, dl, dh, range, bar
        );
    }
}
