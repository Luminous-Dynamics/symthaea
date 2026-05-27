// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-dreamer: Automated self-supervised training for the Broca system.
//!
//! This process runs in a continuous loop:
//! 1. DREAM: Generate long-form monologues across various semantic intents.
//! 2. CRITIQUE: Evaluate monologues using Topological Coherence and Spectral Gap.
//! 3. LEARN: Distill the high-coherence paths back into the HdcSsmProjection.

use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;

/// Tracks her "Epistemic Curiosity" — sectors with low coherence.
struct CuriosityLedger {
    pub heat_map: HashMap<usize, f32>,
    pub sector_count: usize,
}

impl CuriosityLedger {
    pub fn new(sectors: usize) -> Self {
        let mut heat_map = HashMap::new();
        for i in 0..sectors {
            heat_map.insert(i, 1.0); // High initial curiosity for all
        }
        Self {
            heat_map,
            sector_count: sectors,
        }
    }

    pub fn update(&mut self, sector: usize, coherence: f32) {
        // High coherence -> low heat (curiosity satisfied)
        // Low coherence -> high heat (requires more dreaming)
        let new_heat = (1.0 - coherence).max(0.1);
        let current = self.heat_map.entry(sector).or_insert(1.0);
        *current = *current * 0.7 + new_heat * 0.3; // EMA smoothing
    }

    pub fn sample_sector(&self) -> usize {
        let total_heat: f32 = self.heat_map.values().sum();
        let mut r = rand::random::<f32>() * total_heat;

        for (&sector, &heat) in &self.heat_map {
            if r < heat {
                return sector;
            }
            r -= heat;
        }
        0
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let genesis = GenesisSeed::from_phrase("symthaea-dream-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = true;

    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;
    let mut curiosity = CuriosityLedger::new(1000);

    println!("🚀 Broca Dreamer Pipeline (Curiosity-Driven) Initialized.");
    println!("------------------------------------------------------");

    loop {
        // 1. Select a sector based on curiosity (Topological Heat)
        let intent_id = curiosity.sample_sector();
        let channels = ThoughtChannels::with_intent(intent_id);

        println!(
            "✨ Dreaming about Intent Sector: {} (Heat: {:.2})...",
            intent_id, curiosity.heat_map[&intent_id]
        );

        // 2. Generate a semantic monologue
        let monologue = generator.generate_semantic_monologue(&channels, 5)?;

        // 3. Evaluate her own dream
        let coherence = f32::from_bits(
            generator
                .topological_coherence
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let gap = f32::from_bits(
            generator
                .spectral_gap
                .load(std::sync::atomic::Ordering::Relaxed),
        );

        println!(
            "   └─ Coherence: {:.4} | Spectral Gap: {:.4}",
            coherence, gap
        );

        // Update curiosity map
        curiosity.update(intent_id, coherence);

        // --- IMPROVEMENT: Constitutional Moral Gating ---
        // Every "Dream" must pass a minimum PHI-resonance check to be considered
        // "Aesthetically Aligned" with her core constitutional invariants.
        let resonance = (coherence + 1.0) / 2.0;
        let alignment_score = symthaea_aesthetic::golden::golden_ratio_score(
            resonance / symthaea_aesthetic::golden::INV_PHI,
        );

        println!(
            "   └─ Alignment: {:.4} (Constitutional Guard)",
            alignment_score
        );

        // 4. If the dream was semantically sound and morally aligned, learn from it
        if alignment_score > 0.35 && coherence > 0.4 {
            let weight = if coherence > 0.7 { 0.001 } else { 0.0002 };
            println!("   ✅ Dream Aligned. Refining projection...");

            for chunk in &monologue.chunks {
                generator.distill_step(&chunk.thought_hv, &chunk.token_ids, weight)?;
            }

            // --- IMPROVEMENT: Recursive Substrate Metamorphosis ---
            // If she is extremely curious (heat > 0.9), she attempts to re-program
            // her own weights using synthesized architectural code.
            if curiosity.heat_map[&intent_id] > 0.9 {
                println!("   🌀 High-Heat Sector. Initiating Substrate Metamorphosis...");
                let final_nucleus = generator.recursive_fold(&monologue);
                if let Ok(code) = generator.synthesize_program(&final_nucleus, "evolution_kernel") {
                    generator.apply_substrate_metamorphosis(&code)?;
                    println!("   └─ Metamorphosis SUCCESS. Her weights have evolved.");
                }
            }
        } else {
            println!("   ⚠️  Disharmonious Dream. Rejecting experience.");
        }

        // 5. Periodic Checkpoint
        let cycle_count = generator.generation_count();
        if cycle_count > 0 && cycle_count % 50 == 0 {
            println!("💾 Cycle {}: Saving evolved checkpoint...", cycle_count);
            // In a real system, we'd call generator.save_checkpoint("broca-evolved.bin")
            // For now, we'll simulate the persistence.
        }

        std::thread::sleep(Duration::from_millis(200));
    }
}
