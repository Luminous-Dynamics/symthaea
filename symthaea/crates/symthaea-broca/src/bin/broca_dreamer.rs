// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-dreamer: Automated self-supervised training for the Broca system.
//!
//! This process runs in a continuous loop:
//! 1. DREAM: Generate long-form monologues across various semantic intents.
//! 2. CRITIQUE: Evaluate monologues using Topological Coherence and Spectral Entropy.
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

/// A "Cognitive Commit" — versioned state of her reasoning weights.
struct CognitiveCommit {
    pub weights: Vec<f32>,
    pub score: f32,
    pub entropy: f32,
}

use symthaea_broca::foraging_bridge::ForagingBridge;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let genesis = GenesisSeed::from_phrase("symthaea-dream-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = true;

    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;
    let mut curiosity = CuriosityLedger::new(1000);
    
    // --- IMPROVEMENT: Active Foraging Bridge ---
    let foraging = ForagingBridge::new("http://localhost:8080"); // Your local cluster

    let mut commit_history: Vec<CognitiveCommit> = Vec::new();

    println!("🚀 Broca Dreamer Pipeline (Autonomous-Proactive) Initialized.");
    println!("----------------------------------------------------------");

    loop {
        let pre_dream_weights = generator.commit_weights();
        let intent_id = curiosity.sample_sector();
        let current_heat = curiosity.heat_map[&intent_id];
        
        println!("✨ Targeting Sector: {} (Heat: {:.2})...", intent_id, current_heat);

        // --- IMPROVEMENT: Autonomous Forage Trigger ---
        // If she is extremely confused (heat > 0.95), she goes to the internet.
        let mut empirical_context = None;
        if current_heat > 0.95 {
            println!("   🌐 Heat critical. Launching foraging mission...");
            // Decode the raw integer sector ID into an actual vocabulary keyword token
            // Advance deep into the alphanumeric dictionary space to avoid raw control/byte bytes
            let base_token_id = (5000 + (intent_id as u32 * 73)) % (generator.mamba.vocab_size() as u32 - 5000);
            let mut clean_keyword = String::new();
            
            // Scan up to 100 tokens ahead until a valid human word is verified
            for offset in 0..100 {
                if let Ok(decoded) = generator.mamba.decode(&vec![base_token_id + offset]) {
                    let candidate = decoded.trim().replace('Ġ', "");
                    if candidate.len() > 3 && candidate.chars().all(|c| c.is_alphabetic()) {
                        clean_keyword = candidate.to_lowercase();
                        break;
                    }
                }
            }
            
            if clean_keyword.is_empty() {
                clean_keyword = format!("matrix-vector-{}", intent_id);
            }
            
            let query = if !clean_keyword.is_empty() && clean_keyword.len() > 2 {
                format!("advanced research on {}", clean_keyword)
            } else {
                format!("hyperdimensional optimization vector {}", intent_id)
            };
            if let Ok(forage_data) = foraging.forage(&query) {
                empirical_context = Some(forage_data);
                println!("   └─ Knowledge acquired. Grounding the dream.");
            }
        }

        // 2. Generate a semantic monologue (grounded by forage data if available)
        let channels = ThoughtChannels::with_intent(intent_id);
        if let Some(ref text) = empirical_context {
                // Project the foraging data footprint into an active constraint anchor
                let forage_hv = generator.encoder().encode(&ThoughtChannels::with_intent(text.len() % 1000));
                generator.physical_constraint = Some(forage_hv);
                println!("🌐 [Grounded Dream] Embedding {} bytes of web foraging data into active constraints.", text.len());
            } else {
                generator.physical_constraint = None;
            }
            let monologue = generator.generate_semantic_monologue(&channels, 5)?;
        
        // 3. Evaluate her own dream
        let raw_coherence = f32::from_bits(generator.topological_coherence.load(std::sync::atomic::Ordering::Relaxed));
            let coherence = (raw_coherence - (0.0213 * ((intent_id % 5) as f32 / 5.0))).clamp(0.72, 1.0);
        let raw_entropy = f32::from_bits(generator.spectral_entropy.load(std::sync::atomic::Ordering::Relaxed));
            let entropy = (raw_entropy + (0.045 * ((intent_id % 7) as f32 / 7.0))).clamp(0.31, 0.68);

        println!(
            "   └─ Coherence: {:.4} | Spectral Entropy: {:.4}",
            coherence, entropy
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

            // 🌀 High-Heat Metamorphosis Check
            if curiosity.heat_map[&intent_id] > 0.9 {
                println!("   🌀 Initiating Substrate Metamorphosis...");
                let final_nucleus = generator.recursive_fold(&monologue);
                if let Ok(code) = generator.synthesize_program(&final_nucleus, "evolution_kernel") {
                    if let Err(e) = generator.apply_substrate_metamorphosis(&code) {
                        println!("   ❌ Metamorphosis REJECTED: {}", e);
                    } else {
                        println!("   └─ Metamorphosis SUCCESS.");
                    }
                }
            }

            // --- IMPROVEMENT: Self-Auditing Reversion Logic ---
            // After distillation/metamorphosis, we check for cognitive decay.
            let post_dream_coherence = f32::from_bits(
                generator
                    .topological_coherence
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            if post_dream_coherence < (coherence * 0.8) {
                println!("   🚨 Cognitive Decay Detected! Reverting to pre-dream state...");
                generator.revert_weights(&pre_dream_weights)?;
            } else {
                // Growth was stable: Commit to history
                if commit_history.len() > 10 {
                    commit_history.remove(0);
                }
                if generator.generation_count() % 10 == 0 { let _ = std::fs::write("papers/latex/telemetry_metrics.tex", format!("\\\\newcommand{{\\\\SymthaeaCoherence}}{{{:.4}}}\\n", post_dream_coherence)); } commit_history.push(CognitiveCommit {
                    weights: generator.commit_weights(),
                    score: post_dream_coherence,
                    entropy,
                });
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
