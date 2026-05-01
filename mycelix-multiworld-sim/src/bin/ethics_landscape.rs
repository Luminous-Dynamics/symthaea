// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ethics Landscape Search: sweep the 4D ethical orientation space to find
//! the topology of civilization outcomes.
//!
//! Grid-searches (deontological, consequentialist, virtue_care, relational)
//! at N levels per dimension, running M seeds per grid point.
//!
//! Output: CSV of (deont, conseq, virtue, relat, mean_cvs, std_cvs, mean_phi, mean_pop)
//! suitable for 4D surface visualization or contour plots.
//!
//! # Usage
//!
//! ```bash
//! # Quick (3^4 = 81 points × 3 seeds = 243 runs, ~2h)
//! cargo run --release --bin ethics_landscape -- --levels 3 --seeds 3 --ticks 1200
//!
//! # Full (5^4 = 625 points × 5 seeds = 3125 runs, ~26h)
//! cargo run --release --bin ethics_landscape -- --levels 5 --seeds 5 --ticks 1800
//! ```

use mycelix_multiworld_sim::agent::EthicalOrientation;
use mycelix_multiworld_sim::config::{
    EpochConfig, PolicyConfig, SimulationConfig, WorldSeedConfig,
};
use mycelix_multiworld_sim::MultiWorldSimulator;

fn make_config(seed: u64, ticks: u32) -> SimulationConfig {
    let mut policy = PolicyConfig::default();
    policy.trust_weighted_governance = true; // always gated

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
            name: "Landscape".into(),
            start_tick: 0,
            end_tick: ticks,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy,
    }
}

fn run_point(seed: u64, ticks: u32, ethics: [f64; 4]) -> (f64, f64, usize) {
    let mut sim = MultiWorldSimulator::new(make_config(seed, ticks));
    sim.run_initialization();

    // Set all founding agents to the specified ethical orientation
    for world in &mut sim.worlds {
        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
            agent.ethics = EthicalOrientation {
                deontological: ethics[0],
                consequentialist: ethics[1],
                virtue_care: ethics[2],
                relational: ethics[3],
            };
        }
    }

    let report = sim.run();
    let phi = sim
        .worlds
        .iter()
        .map(|w| w.mean_phi() * w.population() as f64)
        .sum::<f64>()
        / report.final_population.max(1) as f64;

    (report.final_cvs, phi, report.final_population)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let levels: usize = args
        .iter()
        .position(|a| a == "--levels")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n_seeds: usize = args
        .iter()
        .position(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let ticks: u32 = args
        .iter()
        .position(|a| a == "--ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1200);

    let total_points = levels.pow(4);
    let total_runs = total_points * n_seeds;

    eprintln!("=== ETHICS LANDSCAPE SEARCH ===");
    eprintln!(
        "{}^4 = {} grid points x {} seeds = {} total runs",
        levels, total_points, n_seeds, total_runs
    );
    eprintln!("{} ticks ({:.0} years) per run", ticks, ticks as f64 / 12.0);
    eprintln!();

    // Generate grid points
    let values: Vec<f64> = (0..levels)
        .map(|i| 0.1 + (i as f64 / (levels - 1).max(1) as f64) * 0.8)
        .collect();

    // CSV header
    println!(
        "deont,conseq,virtue,relat,mean_cvs,std_cvs,mean_phi,std_phi,mean_pop,std_pop,n_seeds"
    );

    let seeds: Vec<u64> = (0..n_seeds).map(|i| 42 + i as u64 * 17).collect();
    let mut completed = 0;

    for &d in &values {
        for &c in &values {
            for &v in &values {
                for &r in &values {
                    let ethics = [d, c, v, r];
                    let mut cvs_vals = Vec::new();
                    let mut phi_vals = Vec::new();
                    let mut pop_vals = Vec::new();

                    for &seed in &seeds {
                        let (cvs, phi, pop) = run_point(seed, ticks, ethics);
                        cvs_vals.push(cvs);
                        phi_vals.push(phi);
                        pop_vals.push(pop as f64);
                    }

                    let mean_cvs = cvs_vals.iter().sum::<f64>() / n_seeds as f64;
                    let mean_phi = phi_vals.iter().sum::<f64>() / n_seeds as f64;
                    let mean_pop = pop_vals.iter().sum::<f64>() / n_seeds as f64;
                    let std_cvs = (cvs_vals.iter().map(|x| (x - mean_cvs).powi(2)).sum::<f64>()
                        / n_seeds as f64)
                        .sqrt();
                    let std_phi = (phi_vals.iter().map(|x| (x - mean_phi).powi(2)).sum::<f64>()
                        / n_seeds as f64)
                        .sqrt();
                    let std_pop = (pop_vals.iter().map(|x| (x - mean_pop).powi(2)).sum::<f64>()
                        / n_seeds as f64)
                        .sqrt();

                    println!(
                        "{:.2},{:.2},{:.2},{:.2},{:.6},{:.6},{:.6},{:.6},{:.0},{:.0},{}",
                        d,
                        c,
                        v,
                        r,
                        mean_cvs,
                        std_cvs,
                        mean_phi,
                        std_phi,
                        mean_pop,
                        std_pop,
                        n_seeds
                    );

                    completed += 1;
                    if completed % 10 == 0 {
                        eprintln!(
                            "  [{}/{}] ({:.1}%) last: [{:.2},{:.2},{:.2},{:.2}] CVS={:.4}",
                            completed,
                            total_points,
                            completed as f64 / total_points as f64 * 100.0,
                            d,
                            c,
                            v,
                            r,
                            mean_cvs
                        );
                    }
                }
            }
        }
    }

    // Find optimum
    eprintln!("\n=== LANDSCAPE COMPLETE ===");
}
