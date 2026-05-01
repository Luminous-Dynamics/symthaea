// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ethics Story: run a single seed and dump the moral narrative.
//!
//! Outputs:
//! 1. The civilization narrative (prose chronicle)
//! 2. Ethics trajectory (mean ethics per world every 60 ticks)
//! 3. Moral health diagnostics (guilt, harm, outrage saturation checks)
//!
//! Usage: cargo run --release --bin ethics_story -- [seed] [ticks]

use mycelix_multiworld_sim::agent::EthicalOrientation;
use mycelix_multiworld_sim::config::{
    EpochConfig, PolicyConfig, SimulationConfig, WorldSeedConfig,
};
use mycelix_multiworld_sim::MultiWorldSimulator;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let ticks: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1800);

    eprintln!(
        "=== ETHICS STORY — seed {} × {} ticks ({:.0} years) ===\n",
        seed,
        ticks,
        ticks as f64 / 12.0
    );

    let mut policy = PolicyConfig::default();
    policy.trust_weighted_governance = true;

    let config = SimulationConfig {
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
            name: "Story".into(),
            start_tick: 0,
            end_tick: ticks,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy,
    };

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    // === 1. Narrative Chronicle ===
    println!("╔══════════════════════════════════════════════════════╗");
    println!(
        "║  THE MORAL STORY OF SEED {}                         ║",
        seed
    );
    println!("╚══════════════════════════════════════════════════════╝\n");

    if !sim.narrative_engine.events.is_empty() {
        println!("{}", sim.narrative_engine.format_chronicle());
    } else {
        println!("(No narrative events recorded)");
    }

    // === 2. Ethics Trajectory ===
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  ETHICS TRAJECTORY                                   ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    println!("snapshot,pop,deont,conseq,virtue,relat,dominant,diversity");
    for snap in &report.epoch_snapshots {
        println!(
            "yr{:.0},{},{:.3},{:.3},{:.3},{:.3},{},{:.3}",
            snap.tick as f64 / 12.0,
            snap.total_population,
            snap.ethics_mean[0],
            snap.ethics_mean[1],
            snap.ethics_mean[2],
            snap.ethics_mean[3],
            ["deont", "conseq", "virtue", "relat"][snap
                .ethics_mean
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)],
            snap.ethics_diversity,
        );
    }

    // === 3. Moral Health Diagnostics ===
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  MORAL HEALTH DIAGNOSTICS                            ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    for world in &sim.worlds {
        let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
        let n = living.len().max(1) as f64;

        let mean_guilt: f64 = living.iter().map(|a| a.needs.affect.guilt).sum::<f64>() / n;
        let mean_harm: f64 = living.iter().map(|a| a.needs.affect.harm).sum::<f64>() / n;
        let mean_outrage: f64 = living.iter().map(|a| a.needs.affect.outrage).sum::<f64>() / n;
        let mean_load: f64 = living.iter().map(|a| a.needs.allostatic_load).sum::<f64>() / n;
        let mean_trauma: f64 = living.iter().map(|a| a.trauma_level).sum::<f64>() / n;

        let max_guilt = living
            .iter()
            .map(|a| a.needs.affect.guilt)
            .fold(0.0f64, f64::max);
        let max_harm = living
            .iter()
            .map(|a| a.needs.affect.harm)
            .fold(0.0f64, f64::max);

        let guilt_saturated = living.iter().filter(|a| a.needs.affect.guilt > 0.8).count();
        let harm_saturated = living.iter().filter(|a| a.needs.affect.harm > 0.8).count();
        let morally_injured = living.iter().filter(|a| a.needs.affect.harm > 0.4).count();

        let mean_ethics = EthicalOrientation::mean_of(&world.agents);
        let inst = &world.institutional_ethics;

        println!("World: {} (pop: {})", world.name, living.len());
        println!(
            "  Ethics mean:        [{:.3}, {:.3}, {:.3}, {:.3}] ({})",
            mean_ethics.deontological,
            mean_ethics.consequentialist,
            mean_ethics.virtue_care,
            mean_ethics.relational,
            mean_ethics.dominant()
        );
        println!(
            "  Institutional:      [{:.3}, {:.3}, {:.3}, {:.3}] ({})",
            inst.deontological,
            inst.consequentialist,
            inst.virtue_care,
            inst.relational,
            inst.dominant()
        );
        println!(
            "  Moral memories:     {} active",
            world.moral_memories.len()
        );
        for mem in &world.moral_memories {
            let dim_names = [
                "deontological",
                "consequentialist",
                "virtue_care",
                "relational",
            ];
            println!(
                "    yr {:.0}: lesson={}, strength={:.2}",
                mem.tick as f64 / 12.0,
                dim_names[mem.lesson_dimension],
                mem.strength(report.total_ticks)
            );
        }
        println!("  Affect diagnostics:");
        println!(
            "    mean guilt:       {:.4} (max: {:.4}, saturated: {})",
            mean_guilt, max_guilt, guilt_saturated
        );
        println!(
            "    mean harm:        {:.4} (max: {:.4}, saturated: {})",
            mean_harm, max_harm, harm_saturated
        );
        println!("    mean outrage:     {:.4}", mean_outrage);
        println!(
            "    morally injured:  {}/{} agents ({:.1}%)",
            morally_injured,
            living.len(),
            morally_injured as f64 / n * 100.0
        );
        println!("    mean load:        {:.4}", mean_load);
        println!("    mean trauma:      {:.4}", mean_trauma);
        println!();
    }

    // === Summary ===
    println!(
        "Final CVS: {:.4}, Phi: {:.4}, Pop: {}, Survived: {}",
        report.final_cvs, report.final_collective_phi, report.final_population, report.survived
    );
}
