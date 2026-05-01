// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Interplanetary Consciousness Epidemiology Demo
//!
//! Simulates consciousness spread across Earth, Mars, and Europa colonies,
//! demonstrating the D→I bridge: within-world SIR-like contagion (Direction D)
//! coupled with cross-planetary synchronization (Direction I).
//!
//! Run: `cargo run --example interplanetary_consciousness_demo`

use mycelix_multiworld_sim::consciousness_epidemiology::Intervention;
use mycelix_multiworld_sim::interplanetary_consciousness::{
    InterplanetaryConsciousness, InterplanetaryEvent, PlanetaryBodyType, PlanetaryWorld,
};

const TOTAL_TICKS: u64 = 200;
const REPORT_INTERVAL: u64 = 10;

fn main() {
    print_banner();

    // --- Create 3 worlds ---
    let earth = PlanetaryWorld::new("Earth", PlanetaryBodyType::Earth, 10_000, 42);
    let mars = PlanetaryWorld::new("Mars Colony", PlanetaryBodyType::Mars, 2_000, 137);
    let europa = PlanetaryWorld::new("Europa Station", PlanetaryBodyType::Europa, 500, 271);

    let mut sim = InterplanetaryConsciousness::new(vec![earth, mars, europa], 3);

    println!("  Initial state:");
    print_table_header();
    print_world_rows(&sim, 0);
    print_separator();

    // --- Run simulation ---
    let mut blackout_count = 0u32;
    let mut divergence_alerts = 0u32;

    for tick in 1..=TOTAL_TICKS {
        // Apply interventions at scheduled ticks
        if tick == 50 {
            println!("\n  >>> Tick 50: Injecting Education intervention on Mars (boost=3.0)");
            sim.inject_intervention(1, Intervention::Education { boost: 3.0 });
        }
        if tick == 80 {
            println!("  >>> Tick 80: Injecting Mentorship intervention on Europa (pairs=50)");
            sim.inject_intervention(2, Intervention::Mentorship { pairs: 50 });
        }

        let events = sim.tick();

        // Track events
        for event in &events {
            match event {
                InterplanetaryEvent::BlackoutStarted { world_a, world_b } => {
                    blackout_count += 1;
                    if tick % REPORT_INTERVAL == 0 || tick <= 5 {
                        println!("  [!] Blackout: {world_a} <-> {world_b} (tick {tick})");
                    }
                }
                InterplanetaryEvent::BlackoutEnded {
                    world_a,
                    world_b,
                    divergence,
                } => {
                    if tick % REPORT_INTERVAL == 0 || *divergence > 0.05 {
                        println!(
                            "  [+] Blackout ended: {world_a} <-> {world_b} (divergence: {divergence:.4})"
                        );
                    }
                }
                InterplanetaryEvent::DivergenceAlert { worlds, gap } => {
                    divergence_alerts += 1;
                    if divergence_alerts <= 5 || tick % REPORT_INTERVAL == 0 {
                        println!(
                            "  [!!] Divergence alert: {} <-> {} (gap: {gap:.4})",
                            worlds.0, worlds.1
                        );
                    }
                }
                InterplanetaryEvent::ConsciousnessSync {
                    from,
                    to,
                    psi_delta,
                    ..
                } => {
                    if tick % (REPORT_INTERVAL * 5) == 0 {
                        println!("  [~] Sync: {from} -> {to} (psi delta: {psi_delta:.6})");
                    }
                }
                _ => {}
            }
        }

        // Print periodic summary
        if tick % REPORT_INTERVAL == 0 {
            println!();
            print_table_header();
            print_world_rows(&sim, tick);
            print_separator();
        }
    }

    // --- Final report ---
    let report = sim.report();

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    Final Cross-Planetary Report                 ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Total population:       {:>6}                                 ║",
        report.total_population
    );
    println!(
        "║  Global mean Phi:        {:.4}                                 ║",
        report.global_mean_phi
    );
    println!(
        "║  Max consciousness gap:  {:.4}                                 ║",
        report.max_consciousness_gap
    );
    println!(
        "║  Sync quality:           {:.4}                                 ║",
        report.sync_quality
    );
    println!(
        "║  Blackout events:        {:>4}                                   ║",
        blackout_count
    );
    println!(
        "║  Divergence alerts:      {:>4}                                   ║",
        divergence_alerts
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Per-World Results                                             ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    for (name, epi_report) in &report.per_world {
        println!("║  {:<20}                                          ║", name);
        println!(
            "║    Mean Phi:  {:.4}   R0: {:.3}   Gini: {:.4}              ║",
            epi_report.mean_phi, epi_report.r0_effective, epi_report.consciousness_gini
        );
        println!(
            "║    Peak tier: {:?} at tick {}   Herd immunity: {}    ║",
            epi_report.peak_tier,
            epi_report.peak_tick,
            if epi_report.herd_immunity_reached {
                "YES"
            } else {
                "NO "
            }
        );
        println!(
            "║    Attack rate: {:.2}%                                       ║",
            epi_report.final_attack_rate * 100.0
        );
    }

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Blackout Divergence Log                                       ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    if report.blackout_divergence.is_empty() {
        println!("║  (no active blackout divergences at end of simulation)         ║");
    } else {
        for (a, b, div) in &report.blackout_divergence {
            println!("║  {a} <-> {b}: divergence = {div:.4}");
        }
    }

    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // --- Consciousness gap over time summary ---
    println!(
        "  Consciousness gap trajectory (sampled every {} ticks):",
        REPORT_INTERVAL
    );
    println!("  Final gap: {:.4}", report.max_consciousness_gap);
    println!();
    println!(
        "  Demo complete. {} ticks simulated across 3 worlds.",
        TOTAL_TICKS
    );
}

fn print_banner() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   Interplanetary Consciousness Epidemiology Demo               ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║   Simulating consciousness spread across 3 worlds              ║");
    println!("║   Earth (10,000) + Mars (2,000) + Europa (500)                 ║");
    println!("║   Direction D (within-world) + Direction I (cross-planetary)   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_table_header() {
    println!(
        "  {:>4} | {:<16} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} | {:>6} | {:>6}",
        "Tick", "World", "Dorm", "Awak", "Aware", "Integ", "Trans", "Psi", "Gap"
    );
    println!(
        "  {:-<4}-+-{:-<16}-+-{:-<5}-+-{:-<5}-+-{:-<5}-+-{:-<5}-+-{:-<5}-+-{:-<6}-+-{:-<6}",
        "", "", "", "", "", "", "", "", ""
    );
}

fn print_world_rows(sim: &InterplanetaryConsciousness, tick: u64) {
    let gap = sim.consciousness_gap();
    for (i, w) in sim.worlds.iter().enumerate() {
        let counts = w.tier_counts();
        println!(
            "  {:>4} | {:<16} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} | {:.4} | {}",
            if i == 0 {
                format!("{tick}")
            } else {
                String::new()
            },
            w.name,
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            w.collective_psi,
            if i == 0 {
                format!("{gap:.4}")
            } else {
                String::new()
            },
        );
    }
}

fn print_separator() {
    println!(
        "  {:-<4}-+-{:-<16}-+-{:-<5}-+-{:-<5}-+-{:-<5}-+-{:-<5}-+-{:-<5}-+-{:-<6}-+-{:-<6}",
        "", "", "", "", "", "", "", "", ""
    );
}
