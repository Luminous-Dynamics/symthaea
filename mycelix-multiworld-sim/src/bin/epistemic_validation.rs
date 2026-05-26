// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-World Epistemic Validation Simulation v2.0
//!
//! Compares the outcome of civilizations based on:
//! 1. Quorum Size (Researcher count)
//! 2. Topological Coherence (Algebraic Connectivity λ2)
//! 3. Knowledge Archival (Resontia Vaults)

use mycelix_multiworld_sim::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
use mycelix_multiworld_sim::knowledge::WorldKnowledge;
use mycelix_multiworld_sim::stochastic::StochasticEngine;
use mycelix_multiworld_sim::world::{ColonyParams, CulturalProfile, World, WorldResources};
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn create_world(name: &str, n_researchers: usize, lambda_2: f64) -> World {
    let params = ColonyParams {
        id: 0,
        name: name.to_string(),
        location: "Surface".to_string(),
        founded_tick: 0,
        parent_world_id: None,
        resources: WorldResources::earth_default(),
        culture: CulturalProfile::pioneer_default(),
        infrastructure_level: 0.8,
        max_population: 10_000,
        habitable_area_m2: 1_000_000.0,
    };

    let mut world = World::new_colony(params);

    let mut agents = Vec::new();
    for i in 0..100 {
        let mut skills = SkillVector::new();
        let sector = if i < n_researchers { 4 } else { 1 };
        skills.learn(sector, 0.8);

        agents.push(CivAgent {
            id: i as u64,
            birth_tick: 0,
            death_tick: None,
            sex: BiologicalSex::Male,
            world_id: 0,
            health: 1.0,
            skills,
            education_level: 0.8,
            consciousness: ConsciousnessState::nascent(),
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: Default::default(),
            tend_balance: 0.0,
            parent_ids: None,
            faction_id: None,
            generation: 0,
            trauma_level: 0.0,
            cumulative_dose_sv: 0.0,
            adversarial: None,
            coordination_understanding: 0.0,
            mycel_score: 0.1,
            sap_balance: 100.0,
            is_biological: true,
            wounds: Vec::new(),
            ethics: Default::default(),
            sovereign_profile: Default::default(),
            justice: Default::default(),
        });
    }

    world.agents = agents;
    world.knowledge.network_lambda_2 = lambda_2;
    world
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🌍 INITIATING TOPOLOGICAL TRUTH SIMULATION...");

    let mut rng = StochasticEngine::new(1337);

    // World A: The Coherent (High Quorum, High Coherence)
    let mut coherent = create_world("The Coherent", 40, 0.85);

    // World B: The Siloed (High Quorum, Low Coherence)
    let mut siloed = create_world("The Siloed", 40, 0.05);

    info!("⏳ Simulating 50 Years (600 Ticks) of Knowledge Evolution...");

    for tick in 1..=600 {
        coherent.knowledge.tick_knowledge_standalone(
            coherent.id,
            &coherent.name,
            &coherent.agents,
            tick,
            &mut rng,
        );
        siloed.knowledge.tick_knowledge_standalone(
            siloed.id,
            &siloed.name,
            &siloed.agents,
            tick,
            &mut rng,
        );

        if tick % 120 == 0 {
            let year = tick / 12;
            info!("📅 Year {}: ", year);
            info!(
                "   🔹 [Coherent] Membrane: {:.2} | Tech: {:.2} | Vaulted: {}",
                coherent.knowledge.epistemic_membrane,
                coherent.knowledge.mean_tech_level(),
                coherent.knowledge.resontia_anchored_claims
            );
            info!(
                "   🔸 [Siloed]   Membrane: {:.2} | Tech: {:.2} | Vaulted: {}",
                siloed.knowledge.epistemic_membrane,
                siloed.knowledge.mean_tech_level(),
                siloed.knowledge.resontia_anchored_claims
            );
        }
    }

    info!("✨ SIMULATION COMPLETE.");

    let coherent_tech = coherent.knowledge.mean_tech_level();
    let siloed_tech = siloed.knowledge.mean_tech_level();

    info!("📊 FINAL TOPOLOGICAL OUTCOME:");
    info!(
        "   ✅ Coherent Network achieved {:.2}x tech growth.",
        coherent_tech
    );
    info!(
        "   ⚠️  Siloed Network (Echo Chambers) achieved only {:.2}x tech growth.",
        siloed_tech
    );
    info!(
        "   💎 Knowledge permanently secured in vaults: {} (Coherent) vs {} (Siloed)",
        coherent.knowledge.resontia_anchored_claims, siloed.knowledge.resontia_anchored_claims
    );

    assert!(
        coherent_tech > siloed_tech,
        "Topological coherence must improve tech utility."
    );

    Ok(())
}
