// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! hierarchical_reasoning: Demo of Broca's macro-reasoning capabilities.
//!
//! Shows how the system can:
//! 1. Generate a monologue about a specific intent.
//! 2. "Fold" that monologue into a single Semantic Nucleus.
//! 3. Use that Nucleus as a primitive for a subsequent reasoning stage.

use anyhow::Result;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let genesis = GenesisSeed::from_phrase("hierarchical-reasoning-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = true;

    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;

    println!("🏛️ Hierarchical Reasoning Demo");
    println!("==============================\n");

    // Stage 1: Specific Reason (e.g. "Defining the problem")
    println!("[Stage 1] Defining the problem manifold...");
    let channels_1 = ThoughtChannels::with_intent(101);
    let monologue_1 = generator.generate_semantic_monologue(&channels_1, 3)?;

    for chunk in &monologue_1.chunks {
        println!(
            "   chunk: {}",
            chunk.target.as_ref().unwrap_or(&"[empty]".to_string())
        );
    }

    // Fold Stage 1 into a Nucleus
    let nucleus_1 = generator.recursive_fold(&monologue_1);
    println!(
        "\n📦 Stage 1 folded into Nucleus (dim: {})",
        nucleus_1.dim()
    );

    // Stage 2: Higher-order Reason (e.g. "Synthesizing a solution")
    // We use the Nucleus from Stage 1 as a "Cognitive Goal" for Stage 2
    println!("\n[Stage 2] Synthesizing a solution grounded in Stage 1...");
    generator.set_goal(nucleus_1.clone());

    let channels_2 = ThoughtChannels::with_intent(202);
    let monologue_2 = generator.generate_semantic_monologue(&channels_2, 3)?;

    for chunk in &monologue_2.chunks {
        println!(
            "   chunk: {}",
            chunk.target.as_ref().unwrap_or(&"[empty]".to_string())
        );
    }

    // Final Fold: The entire chain of thought
    let final_nucleus = generator.recursive_fold(&monologue_2);
    println!("\n🌌 Final Reasoning Nucleus established.");

    // Synthesis: Generate code from the final nucleus
    #[cfg(feature = "code-sheaf-eval")]
    {
        println!("\n[Stage 3] Emitting Topological Code Spec from Reasoning...");
        if let Ok(code) = generator.synthesize_program(&final_nucleus, "harmonic_solution") {
            println!("------------------------------------------");
            println!("{}", code);
            println!("------------------------------------------");
        }
    }

    Ok(())
}
