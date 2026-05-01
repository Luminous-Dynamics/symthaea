// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark: profile each phase of the simulation tick loop.
//!
//! Run: cargo run --release --bin bench_tick

use mycelix_multiworld_sim::config::SimulationConfig;
use mycelix_multiworld_sim::MultiWorldSimulator;
use std::time::Instant;

fn main() {
    println!("=== TICK LOOP BENCHMARK ===\n");

    // Run a 50-year sim and time the total
    let years = 50;
    let ticks = years * 12;

    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = ticks;
    config.seed = 42;

    let start = Instant::now();
    let mut sim = MultiWorldSimulator::new(config);
    let init_time = start.elapsed();

    let run_start = Instant::now();
    let report = sim.run();
    let run_time = run_start.elapsed();

    let total_time = start.elapsed();
    let per_tick = run_time / ticks;
    let ticks_per_sec = ticks as f64 / run_time.as_secs_f64();

    println!("Duration:       {} years ({} ticks)", years, ticks);
    println!("Final pop:      {}", report.final_population);
    println!("Final CVS:      {:.3}", report.final_cvs);
    println!("Civil wars:     {}", report.civil_wars);
    println!("Turchin trans:  {}", report.turchin_transitions);
    println!();
    println!("=== TIMING ===");
    println!("Init:           {:?}", init_time);
    println!("Run ({} ticks): {:?}", ticks, run_time);
    println!("Total:          {:?}", total_time);
    println!("Per tick:       {:?}", per_tick);
    println!("Ticks/sec:      {:.1}", ticks_per_sec);
    println!("Years/sec:      {:.1}", ticks_per_sec / 12.0);
    println!();

    // Estimate longer runs
    let est_300yr = run_time.as_secs_f64() * (300.0 / years as f64);
    let est_1000yr = run_time.as_secs_f64() * (1000.0 / years as f64);
    println!("=== ESTIMATES ===");
    println!(
        "300yr estimate:  {:.0}s ({:.1} min)",
        est_300yr,
        est_300yr / 60.0
    );
    println!(
        "1000yr estimate: {:.0}s ({:.1} min)",
        est_1000yr,
        est_1000yr / 60.0
    );
    println!();

    // Population scaling analysis
    let pop = report.final_population;
    println!("=== SCALING ===");
    println!("Population at end: {}", pop);
    println!("Agents iterated per tick: ~{}", pop);
    println!("If pop doubles: tick time ~doubles (O(N) agent loop)");
    println!();

    // Key insight: the simulation is O(N × T) where N=population, T=ticks
    // At 50yr: N~4000, T=600. Total iterations: ~2.4M
    // At 300yr: N~25000, T=3600. Total iterations: ~90M (37× more)
    // At 1000yr: N~60000, T=12000. Total iterations: ~720M (300× more)
    println!("=== OPTIMIZATION OPPORTUNITIES ===");
    println!("1. Agent loop: O(N) per tick. Use cohort aggregation for N > 10K");
    println!("2. Elite sort: O(N log N) per tick in cliodynamics. Cache top-20%");
    println!("3. Gini computation: O(N log N) per tick. Update incrementally");
    println!("4. Consciousness tick: O(N) per tick. Parallelize with rayon");
    println!("5. Death processing: O(N) scan for orphans. Use index");
}
