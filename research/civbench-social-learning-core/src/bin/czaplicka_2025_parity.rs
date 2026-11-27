// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Emit the pinned C model's time-series columns for differential testing.
//!
//! Arguments: N pER X L Cs Ci Cr r seed
//! The time horizon is fixed to the author's T=20*N.

use civbench_social_learning_core::czaplicka_2025::{simulate, Config};
use std::env;
use std::process;

fn usage() -> ! {
    eprintln!("usage: czaplicka-2025-parity N pER X L Cs Ci Cr r seed");
    process::exit(2);
}

fn parse<T: std::str::FromStr>(value: Option<&String>) -> T {
    value.and_then(|raw| raw.parse().ok()).unwrap_or_else(|| usage())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 9 {
        usage();
    }

    let population_size: usize = parse(args.get(0));
    let er_probability: f64 = parse(args.get(1));
    let declared_branch_count: usize = parse(args.get(2));
    let initial_effort_budget: i64 = parse(args.get(3));
    let social_learning_cost: i64 = parse(args.get(4));
    let innovation_cost: i64 = parse(args.get(5));
    let recommendation_cost: i64 = parse(args.get(6));
    let recommendation_probability: f64 = parse(args.get(7));
    let seed: u64 = parse(args.get(8));

    let config = Config {
        population_size,
        er_probability,
        declared_branch_count,
        initial_effort_budget,
        social_learning_cost,
        innovation_cost,
        recommendation_cost,
        recommendation_probability,
        steps_per_agent: 20,
        ..Config::default()
    };
    let run = simulate(config, seed).expect("parity config must validate");

    println!(
        "time\tavg_payoff\tavg_level\tavg_budget\tmax_payoff\tmax_level\tmax_ach_payoff\tn_items\tinnovations\tsocial_learning\trecommendations"
    );
    for step in &run.steps {
        println!(
            "{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{}\t{}\t{}\t{}",
            step.step,
            step.mean_payoff,
            step.mean_level,
            step.mean_remaining_budget,
            step.legacy_prefix_max_payoff_mean,
            step.legacy_prefix_max_level_mean,
            step.legacy_max_achievable_payoff_mean,
            step.cumulative_unique_innovations,
            step.cumulative_unique_innovations,
            step.cumulative_peer_copies,
            step.cumulative_recommendations,
        );
    }
}
