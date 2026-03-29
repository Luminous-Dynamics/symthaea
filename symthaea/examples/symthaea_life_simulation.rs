// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Life Simulation
//!
//! A proof-of-life demonstration showing the ContinuousMind subsystem:
//! - Awakening and consciousness monitoring
//! - Working memory injection via HDC perception
//! - Snapshot of cognitive state (Psi, cognitive load, emotion)

use symthaea::{ContinuousMind, MindConfig};
use symthaea_core::hdc::ContinuousHV;

fn main() {
    println!("Symthaea Life Simulation: Activating...");

    // 1. Initialize the Continuous Mind
    let config = MindConfig {
        dimension: 1024,
        tick_rate: 20.0,
        consciousness_monitoring: true,
        learning_enabled: true,
        ..Default::default()
    };

    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    println!("Mind Awakened.");

    // 2. Inject stimuli as HDC perceptions
    let stimuli = [
        "self-awareness",
        "attention",
        "prediction",
        "curiosity",
        "integration",
    ];

    println!(
        "\nInjecting {} stimuli as HDC perceptions...",
        stimuli.len()
    );
    for (i, label) in stimuli.iter().enumerate() {
        let hv = ContinuousHV::random(1024, (i as u64 + 1) * 42);
        mind.perceive_text(label, hv);
    }

    // 3. Observe state
    let snapshot = mind.snapshot();
    println!("\nMind State Snapshot:");
    println!("  Psi (consciousness):   {:.4}", snapshot.psi);
    println!("  Cognitive load:        {:.4}", snapshot.cognitive_load);
    println!("  Emotional valence:     {:.4}", snapshot.emotional_valence);
    println!("  Arousal:               {:.4}", snapshot.arousal);
    println!("  Is conscious:          {}", snapshot.is_conscious);
    println!("  Is active:             {}", snapshot.is_active);

    // 4. Show working memory state
    let wm = mind.working_memory();
    println!(
        "\nWorking Memory: {} items (capacity: {})",
        wm.len(),
        mind.config().working_memory_capacity
    );

    // 5. Extract structured thought
    let thought = mind.extract_structured_thought();
    println!("\nStructured Thought:");
    println!("  Intent:           {:?}", thought.semantic_intent);
    println!("  Response type:    {:?}", thought.response_type);
    println!("  Psi:              {:.4}", thought.psi);
    println!("  Meta-awareness:   {:.4}", thought.meta_awareness);
    println!("  Coherence:        {:.4}", thought.coherence);
    println!("  Epistemic status: {:?}", thought.epistemic_status);

    // 6. Graceful shutdown
    mind.request_shutdown();
    println!("\nSimulation Complete. The system is alive.");
}
