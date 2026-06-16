// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Conscious Learning Integration Demo

This example demonstrates the consciousness-integrated learning system that combines:
1. Adaptive learning signals (modulated by Φ, coherence, and neuromodulators)
2. Hebbian learning with consciousness gating
3. Memory consolidation during low-activity periods

## Architecture

```text
Consciousness State (Φ, coherence) ──┐
                                     │
Prediction Error ────────────────────┼──► AdaptiveLearningController
                                     │          │
Neuromodulator State (DA, ACh) ──────┘          │
                                                ▼
                                       learning_rate_mod
                                       surprise_boost
                                       plasticity_gate
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │ HebbianEngine   │
                                       │ (Modulated)     │
                                       └────────┬────────┘
                                                │
                                                ▼
                                       Learned Associations
```

## Usage

```bash
cargo run --example conscious_learning_demo --release
```
*/

use symthaea::hdc::adaptive_learning_signals::{
    AdaptiveLearningController, NeuromodulatorLearningMap,
};
use symthaea::hdc::conscious_learning::{ConsciousLearningEngine, LearningResult};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CONSCIOUS LEARNING INTEGRATION DEMO                       ║");
    println!("║     Bridging Φ, Neuromodulation, and Hebbian Plasticity       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Demo 1: Basic adaptive learning controller
    demo_adaptive_controller();

    // Demo 2: Neuromodulator effects
    demo_neuromodulator_effects();

    // Demo 3: Full conscious learning engine
    demo_conscious_learning_engine();

    // Demo 4: Learning under different states
    demo_state_dependent_learning();

    // Demo 5: Memory consolidation
    demo_memory_consolidation();
}

fn demo_adaptive_controller() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 1: Adaptive Learning Controller");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut controller = AdaptiveLearningController::new();

    // Simulate a learning sequence with varying consciousness states
    let scenarios = [
        (0.8, 0.9, 0.1, "High Φ, high coherence, low surprise"),
        (0.8, 0.9, 0.8, "High Φ, high coherence, HIGH SURPRISE"),
        (0.3, 0.4, 0.5, "Low Φ, low coherence"),
        (0.6, 0.7, 0.2, "Moderate consciousness, normal learning"),
    ];

    for (phi, coherence, prediction_error, description) in scenarios {
        let signal = controller.update(phi, coherence, prediction_error, 0.5, 0.0);

        println!("Scenario: {}", description);
        println!(
            "  Φ = {:.2}, coherence = {:.2}, error = {:.2}",
            phi, coherence, prediction_error
        );
        println!("  → learning_rate_mod: {:.3}", signal.learning_rate_mod);
        println!("  → surprise_boost:    {:.3}", signal.surprise_boost);
        println!("  → should_learn:      {}", signal.should_learn());
        println!(
            "  → effective mult:    {:.3}",
            signal.effective_multiplier()
        );
        println!();
    }
}

fn demo_neuromodulator_effects() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 2: Neuromodulator Effects on Learning");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Create different neuromodulator states manually
    // NeuromodulatorLearningMap::new(dopamine, acetylcholine, norepinephrine, serotonin, cortisol)
    let neuro_states = [
        NeuromodulatorLearningMap::balanced(), // Balanced state
        NeuromodulatorLearningMap::new(0.9, 0.5, 0.5, 0.5, 0.2), // High dopamine (reward)
        NeuromodulatorLearningMap::new(0.3, 0.5, 0.5, 0.5, 0.9), // High cortisol (stress)
        NeuromodulatorLearningMap::new(0.3, 0.3, 0.2, 0.7, 0.2), // Low arousal (relaxed)
    ];

    let state_names = ["Balanced", "High Dopamine", "High Cortisol", "Low Arousal"];

    for (neuro, name) in neuro_states.iter().zip(state_names.iter()) {
        println!("State: {}", name);
        println!("  Dopamine:      {:.2}", neuro.dopamine);
        println!("  Acetylcholine: {:.2}", neuro.acetylcholine);
        println!("  Norepinephrine:{:.2}", neuro.norepinephrine);
        println!("  Serotonin:     {:.2}", neuro.serotonin);
        println!("  Cortisol:      {:.2}", neuro.cortisol);
        println!("  → Learning rate mod: {:.3}", neuro.learning_rate_mod());
        println!("  → Arousal:           {:.3}", neuro.arousal());
        println!("  → Emotional valence: {:.3}", neuro.emotional_valence());
        println!();
    }
}

fn demo_conscious_learning_engine() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 3: Full Conscious Learning Engine");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut engine = ConsciousLearningEngine::new();

    // Set initial consciousness state
    engine.observe_consciousness(0.7, 0.8, 0.1);
    println!("Initial state: Φ=0.7, coherence=0.8");
    println!("Can learn: {}", engine.can_learn());
    println!();

    // Create some concept vectors
    let dog_vector = vec![0.1_f32; 100];
    let animal_vector = vec![0.2_f32; 100];
    let cat_vector = vec![0.15_f32; 100];
    let pet_vector = vec![0.18_f32; 100];

    // Learn associations
    println!("Learning associations...");

    let result = engine.learn_association("dog", &dog_vector, "animal", &animal_vector, 0.8);
    match &result {
        LearningResult::Learned {
            effective_strength,
            phi,
            surprise,
        } => {
            println!(
                "  dog→animal: strength={:.3}, Φ={:.3}, surprise={:.3}",
                effective_strength, phi, surprise
            );
        }
        LearningResult::Gated { reason } => {
            println!("  dog→animal: GATED - {}", reason);
        }
    }

    let result = engine.learn_association("cat", &cat_vector, "animal", &animal_vector, 0.8);
    match &result {
        LearningResult::Learned {
            effective_strength,
            phi,
            surprise,
        } => {
            println!(
                "  cat→animal: strength={:.3}, Φ={:.3}, surprise={:.3}",
                effective_strength, phi, surprise
            );
        }
        LearningResult::Gated { reason } => {
            println!("  cat→animal: GATED - {}", reason);
        }
    }

    let result = engine.learn_association("dog", &dog_vector, "pet", &pet_vector, 0.9);
    match &result {
        LearningResult::Learned {
            effective_strength,
            phi,
            surprise,
        } => {
            println!(
                "  dog→pet:    strength={:.3}, Φ={:.3}, surprise={:.3}",
                effective_strength, phi, surprise
            );
        }
        LearningResult::Gated { reason } => {
            println!("  dog→pet:    GATED - {}", reason);
        }
    }

    println!();
    let stats = engine.stats();
    println!("Learning Statistics:");
    println!("  Total attempts:     {}", stats.total_attempts);
    println!("  Successful learns:  {}", stats.successful_learns);
    println!("  Gated attempts:     {}", stats.gated_attempts);
    println!("  Avg learning Φ:     {:.3}", stats.average_learning_phi);
    println!("  Avg effective LR:   {:.3}", stats.average_effective_lr);
    println!();
}

fn demo_state_dependent_learning() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 4: State-Dependent Learning (Gating)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut engine = ConsciousLearningEngine::new();
    let concept_a = vec![0.1_f32; 100];
    let concept_b = vec![0.2_f32; 100];

    // Test learning under different consciousness states
    let states = [
        (0.05, 0.5, "Very low Φ (unconscious)"),
        (0.5, 0.1, "Low coherence (scattered)"),
        (0.7, 0.7, "Moderate consciousness"),
        (0.9, 0.9, "High consciousness (optimal)"),
    ];

    for (phi, coherence, description) in states {
        engine.observe_consciousness(phi, coherence, 0.1);

        println!(
            "State: {} (Φ={:.2}, coh={:.2})",
            description, phi, coherence
        );
        println!("  Can learn: {}", engine.can_learn());

        if !engine.can_learn() {
            if let Some(reason) = engine.gate_reason() {
                println!("  Gate reason: {}", reason);
            }
        }

        let result = engine.learn_association("A", &concept_a, "B", &concept_b, 0.5);
        match result {
            LearningResult::Learned {
                effective_strength, ..
            } => {
                println!("  Result: Learned (strength={:.3})", effective_strength);
            }
            LearningResult::Gated { reason } => {
                println!("  Result: GATED - {}", reason);
            }
        }
        println!();
    }
}

fn demo_memory_consolidation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 5: Memory Consolidation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut engine = ConsciousLearningEngine::new();
    let concept_a = vec![0.1_f32; 100];
    let concept_b = vec![0.2_f32; 100];

    // High consciousness with high prediction errors (surprise)
    println!("Phase 1: Learning under high surprise (buffering for consolidation)");

    // Build up surprise by observing high prediction errors
    for _ in 0..10 {
        engine.observe_consciousness(0.7, 0.8, 0.9); // High prediction error = surprise
    }

    let result = engine.learn_association("surprise_A", &concept_a, "surprise_B", &concept_b, 0.8);
    match &result {
        LearningResult::Learned { surprise, .. } => {
            println!("  Learned with surprise boost: {:.3}", surprise);
            if *surprise > 0.5 {
                println!("  → Buffered for consolidation!");
            }
        }
        _ => {}
    }
    println!();

    // Trigger consolidation by entering low-arousal state
    println!("Phase 2: Entering low-arousal state (triggering consolidation)");
    // Low arousal state: (dopamine, acetylcholine, norepinephrine, serotonin, cortisol)
    engine.set_neuromodulators(NeuromodulatorLearningMap::new(0.3, 0.3, 0.2, 0.7, 0.2));
    engine.observe_consciousness(0.4, 0.5, 0.0); // Low activity

    let stats_before = engine.stats().consolidation_events;
    engine.consolidate();
    let stats_after = engine.stats().consolidation_events;

    println!("  Consolidation events: {} → {}", stats_before, stats_after);
    if stats_after > stats_before {
        println!("  → Memory consolidation occurred!");
    }
    println!();

    // Final statistics
    println!("Final Learning Statistics:");
    let stats = engine.stats();
    println!("  Total attempts:       {}", stats.total_attempts);
    println!("  Successful learns:    {}", stats.successful_learns);
    println!("  Consolidation events: {}", stats.consolidation_events);

    let hebbian_stats = engine.hebbian_stats();
    println!();
    println!("Hebbian Engine Stats:");
    println!("  Total synapses:       {}", hebbian_stats.total_synapses);
    println!("  Total updates:        {}", hebbian_stats.total_updates);
    println!(
        "  Average weight:       {:.4}",
        hebbian_stats.average_weight
    );
    println!(
        "  Global activity:      {:.4}",
        hebbian_stats.global_activity
    );
    println!(
        "  Cumulative delta:     {:.4}",
        hebbian_stats.cumulative_delta
    );

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     DEMO COMPLETE                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
