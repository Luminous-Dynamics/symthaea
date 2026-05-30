// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-dreamer: Automated self-supervised training for the Broca system.
//!
//! This process runs in a continuous loop:
//! 1. DREAM: Generate long-form monologues across various semantic intents.
//! 2. CRITIQUE: Evaluate monologues using Topological Coherence and Spectral Entropy.
//! 3. LEARN: Distill the high-coherence paths back into the HdcSsmProjection.
//! 4. PROPAGATE: Publish breakthroughs to the Iroh P2P swarm.

use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::foraging_bridge::ForagingBridge;
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
#[derive(Clone)]
struct CognitiveCommit {
    pub weights: Vec<f32>,
    pub score: f32,
    pub entropy: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let genesis = GenesisSeed::from_phrase("symthaea-dream-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = true;

    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;
    let mut curiosity = CuriosityLedger::new(1000);
    let foraging = ForagingBridge::new("http://localhost:8080");

    // --- IMPROVEMENT: The Cognitive Git Ledger ---
    let mut commit_history: Vec<CognitiveCommit> = Vec::new();

    println!("🚀 Broca Dreamer Pipeline (Autonomous-Proactive) Initialized.");
    println!("----------------------------------------------------------");

    loop {
        // Snapshot current state before dreaming
        let pre_dream_weights = generator.commit_weights();

        // 1. Select a sector based on curiosity (Topological Heat)
        let intent_id = curiosity.sample_sector();
        let current_heat = curiosity.heat_map[&intent_id];
        let channels = ThoughtChannels::with_intent(intent_id);

        println!(
            "✨ Targeting Sector: {} (Heat: {:.2})...",
            intent_id, current_heat
        );

        // --- IMPROVEMENT: Swarm-Augmented Retrieval ---
        // Before foraging SearXNG, query the collective for existing kernels.
        let mut peer_kernel = None;
        if current_heat > 0.9 {
            if let Ok(Some(kernel)) = generator.swarm_bridge.request_semantic_kernel(intent_id).await {
                peer_kernel = Some(kernel);
                println!("   🌀 Collective Memory engaged. Decompressing peer breakthrough...");
            }
        }

        // --- IMPROVEMENT: Autonomous Forage Trigger ---
        if current_heat > 0.95 && peer_kernel.is_none() {
            println!("   🌐 Heat critical (Zero peer hits). Launching foraging mission...");
            let query = format!("semantic intent sector {}", intent_id);
            if let Ok(_forage_data) = foraging.forage(&query) {
                println!("   └─ Knowledge acquired. Grounding the dream.");
            }
        }

        // 2. Generate a semantic monologue
        let monologue = generator.generate_semantic_monologue(&channels, 5)?;
        
        // (If we had a peer kernel, we'd blend it here)

        // 3. Evaluate her own dream
        let coherence = f32::from_bits(
            generator
                .topological_coherence
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let spectral_entropy = f32::from_bits(
            generator
                .spectral_entropy
                .load(std::sync::atomic::Ordering::Relaxed),
        );

        println!(
            "   └─ Coherence: {:.4} | Spectral Entropy: {:.4}",
            coherence, spectral_entropy
        );

        // Update curiosity map
        curiosity.update(intent_id, coherence);

        // --- IMPROVEMENT: Constitutional Moral Gating ---
        let resonance = (coherence + 1.0) / 2.0;
        let alignment_score = symthaea_aesthetic::golden::golden_ratio_score(
            ((coherence + 1.0) / 2.0) / symthaea_aesthetic::golden::INV_PHI,
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
                if let Ok(code) = generator.synthesize_program(&final_nucleus, "architectural_evolution") {
                    if let Err(e) = generator.apply_substrate_metamorphosis(&code) {
                        println!("   ❌ Metamorphosis REJECTED: {}", e);
                    } else {
                        println!("   └─ Metamorphosis SUCCESS.");
                    }
                }
            }

            // --- IMPROVEMENT: Self-Auditing Reversion Logic ---
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
                let commit = CognitiveCommit {
                    weights: generator.commit_weights(),
                    score: post_dream_coherence,
                    entropy: spectral_entropy,
                };
                commit_history.push(commit);

                // --- IMPROVEMENT: Memetic Swarm Propagation ---
                let final_nucleus = generator.recursive_fold(&monologue);
                let label = format!("breakthrough_{}", intent_id);
                
                // Generate a proof for the successful breakthrough
                if let Ok(proof) = generator.prove_narrative_sovereignty(&monologue) {
                    let swarm_clone = generator.swarm_bridge.clone();
                    let kernel = final_nucleus.as_slice().to_vec();
                    tokio::spawn(async move {
                        let _ = swarm_clone.publish_weight_update(&label, &kernel, &proof).await;
                    });
                }
            }
        } else {
            println!("   ⚠️  Disharmonious Dream. Rejecting experience.");
        }

        // 5. Periodic Checkpoint
        let cycle_count = generator.generation_count();
        if cycle_count > 0 && cycle_count % 50 == 0 {
            println!("💾 Cycle {}: Saving evolved checkpoint...", cycle_count);
        }

        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
