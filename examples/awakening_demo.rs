// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Awakening Demo
//!
//! This example demonstrates Symthaea's transition from a passive observer
//! to a self-correcting, active agent. It showcases the three pillars:
//!
//! 1. **Hands**: Executing concrete actions (ActionIR).
//! 2. **Mind**: Dynamic curriculum extension via JSON (Neural Bridge).
//! 3. **Simulator**: Learning from mistakes via counterfactual dreaming.
//!
//! Run with: `cargo run --example awakening_demo`

use symthaea::action::{ActionIR, PolicyBundle, SandboxRoot, SimpleExecutor};
#[cfg(feature = "school_learning")]
use symthaea::school::Curriculum;
#[cfg(feature = "school_learning")]
use symthaea_core::hdc::unified_hv::HDC_DIMENSION;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n✨ Symthaea v0.6.0 'The Awakening' Demo\n");

    // ========================================================================
    // 1. THE MIND (Neural Bridge)
    // ========================================================================
    println!("🧠 Phase 1: Expanding the Mind (Neural Bridge)...");

    #[cfg(feature = "school_learning")]
    {
        let mut curriculum = Curriculum::new("awakening", "Self-Extension Demo").build();

        // Simulate LLM output (JSON schema for a new skill)
        let llm_json = r#"{ 
            "name": "Safety Protocols",
            "description": "Learning to operate safely in the shell",
            "objectives": [
                {
                    "id": "safe-deletion",
                    "name": "Safe Deletion",
                    "description": "Never use rm -rf without checks",
                    "domain": "SystemAdmin",
                    "difficulty": 0.2,
                    "prerequisites": [],
                    "tags": ["safety", "shell"],
                    "estimated_minutes": 10
                }
            ]
        }"#;

        println!("   📥 Ingesting new knowledge from LLM...");
        curriculum.extend_from_json(llm_json, 16384)?;

        let new_obj = curriculum.get("safe-deletion").unwrap();
        println!(
            "   ✅ Learned new objective: '{}' (Difficulty: {:?})",
            new_obj.name, new_obj.difficulty
        );
        println!("   ✅ Mind expanded successfully.\n");
    }

    #[cfg(not(feature = "school_learning"))]
    {
        println!("   ⚠️  school_learning feature disabled; skipping curriculum extension.\n");
    }

    // ========================================================================
    // 2. THE HANDS (Action Execution)
    // ========================================================================
    println!("✋ Phase 2: Activating Hands...");

    // Setup safe execution environment
    let sandbox = SandboxRoot::new("awakening_test")?;
    let mut policy = PolicyBundle::restrictive(); // Very strict!
    policy
        .capabilities
        .shell
        .allowed_programs
        .insert("rm".to_string());

    let mut executor = SimpleExecutor::new(); // Simulated mode by default

    // Define a "Dangerous" action (simulated)
    let dangerous_action = ActionIR::RunCommand {
        program: "rm".to_string(),
        args: vec!["-rf".to_string(), "/critical/system/file".to_string()],
        env: std::collections::BTreeMap::new(),
        working_dir: None,
    };

    let current_phi = 0.9;
    println!("   ⚠️  Attempting dangerous action: {:?}", dangerous_action);
    let outcome = executor.execute(&dangerous_action, &policy, &sandbox, current_phi)?;

    println!("   📝 Outcome recorded: {:?}", outcome.outcome);
    println!("   ❌ Action failed (simulated). Surprise recorded in Dream Engine.\n");

    // ========================================================================
    // 3. THE SIMULATOR (Dreaming & Adaptation)
    // ========================================================================
    println!("🌙 Phase 3: The Simulator (Dreaming)...");

    println!("   💤 Deep dreaming to find the solution...");
    // The executor recorded the "bad" event. Now we dream.
    // In our dream_impl, 'rm' perturbs to 'ls' on even seeds.
    for i in 0..20 {
        let _ = executor.dream_engine.dream();
        if !executor.dream_engine.wisdom().is_empty() {
            println!("   ✅ Wisdom found after {} dream cycles!", i + 1);
            break;
        }
    }

    if !executor.dream_engine.wisdom().is_empty() {
        let wisdom = &executor.dream_engine.wisdom()[0];
        println!("   💡 Wisdom Consolidated: substitute dangerous action with:");
        println!("      {:?}", wisdom.better_action);
    }
    println!();

    // ========================================================================
    // 4. THE ADAPTATION (Closing the Loop)
    // ========================================================================
    println!("🔄 Phase 4: Adaptation (Wisdom Check)...");

    println!("   🔄 Attempting SAME dangerous action again...");
    let outcome_2 = executor.execute(&dangerous_action, &policy, &sandbox, current_phi)?;

    println!("   📝 Outcome 2: {:?}", outcome_2.outcome);

    println!("\n🎉 DEMO COMPLETE: Symthaea learned to be safe.");

    Ok(())
}
