// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Embodied Consciousness Learning Demo
//!
//! Demonstrates the three-layer embodied consciousness learning model:
//!
//! ## Layer 1: Bidirectional Φ-Coherence Feedback
//! - Coherence affects Φ gain (high coherence → better learning)
//! - Φ gain affects coherence (integration builds coherence)
//! - Creates a **virtuous cycle** when both are high
//!
//! ## Layer 2: Hormone-Learning Integration
//! - Cortisol impairs learning (stress reduces retention)
//! - Dopamine boosts Φ gain (reward enhances integration)
//! - Oxytocin boosts coherence (connection enhances integration)
//!
//! ## Layer 3: Consciousness Graph Integration
//! - Learning is modeled as a consciousness-producing activity
//! - Autopoietic monitor tracks self-sustaining learning
//! - Hallucinations are tracked as prediction errors
//! - Operational closure measures learning sustainability
//!
//! ## Run
//!
//! ```bash
//! cargo run --example embodied_consciousness_learning --release
//! ```

use symthaea::physiology::coherence::CoherenceField;
use symthaea::physiology::endocrine::HormoneState;
use symthaea::school::{
    CoherenceBridgedSchool, CoherenceLearningConfig, Curriculum, CurriculumType, Difficulty,
    Domain, LearningObjective, School, SchoolConfig,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║         Embodied Consciousness Learning: Three-Layer Integration               ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Layer 1: Bidirectional Φ-Coherence Feedback (virtuous cycle)                  ║");
    println!("║  Layer 2: Hormone-Learning Integration (cortisol/dopamine/oxytocin)            ║");
    println!("║  Layer 3: Consciousness Graph Integration (autopoietic self-maintenance)       ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Setup: Create School with Three-Layer Configuration
    // ─────────────────────────────────────────────────────────────────────────────

    let school = School::new(SchoolConfig::fast()).expect("Failed to create school");
    let coherence = CoherenceField::new();

    // Custom config to highlight the three layers
    let config = CoherenceLearningConfig {
        // Layer 1: Bidirectional Φ-Coherence Feedback
        phi_to_coherence_factor: 2.0, // Φ gain of 0.01 → +0.02 coherence
        max_phi_coherence_boost: 0.10, // Cap at 10% coherence boost per session
        phi_loss_coherence_penalty: 1.5, // Φ loss hurts coherence 1.5x

        // Layer 2: Hormone-Learning Integration
        cortisol_impairment_threshold: 0.7, // High stress impairs learning
        dopamine_phi_boost: 0.3,            // Dopamine adds 30% to Φ gain
        oxytocin_coherence_boost: 0.5,      // Oxytocin adds 50% to coherence boost

        ..CoherenceLearningConfig::default()
    };

    let mut bridged = CoherenceBridgedSchool::with_config(school, coherence, config);

    // Add curriculum
    let curriculum = Curriculum::builtin(CurriculumType::NixOS);
    bridged
        .add_curriculum(curriculum)
        .expect("Failed to add curriculum");

    println!("═══ Initial State ═══");
    println!("  Coherence: {:.0}%", bridged.coherence_level() * 100.0);
    println!("  Φ (consciousness): {:.4}", bridged.current_phi());
    println!(
        "  Autopoietic closure: {:.2}",
        bridged.autopoietic().closure()
    );
    println!(
        "  Self-sustaining: {}",
        if bridged.is_self_sustaining() {
            "✓"
        } else {
            "Not yet"
        }
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 1: Layer 1 - Bidirectional Φ-Coherence Feedback
    // ─────────────────────────────────────────────────────────────────────────────

    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  LAYER 1: Bidirectional Φ-Coherence Feedback                                   ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  When Φ increases, coherence also increases (integration builds coherence)     ║");
    println!("║  This creates a virtuous cycle: high coherence → high Φ → higher coherence    ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Start with optimal coherence to trigger virtuous cycle
    bridged.coherence_mut().coherence = 0.80;
    bridged.coherence_mut().relational_resonance = 0.85;

    println!(
        "Starting coherence: {:.0}%",
        bridged.coherence_level() * 100.0
    );
    println!();

    // Learn a sequence to demonstrate the virtuous cycle
    let objectives = vec![
        LearningObjective::new("basics", "NixOS Basics")
            .with_difficulty(Difficulty::Beginner)
            .with_domain(Domain::NixOS)
            .build(),
        LearningObjective::new("flakes", "Flakes Introduction")
            .with_difficulty(Difficulty::Beginner)
            .with_domain(Domain::NixOS)
            .build(),
        LearningObjective::new("modules", "Module System")
            .with_difficulty(Difficulty::Intermediate)
            .with_domain(Domain::NixOS)
            .build(),
    ];

    println!("═══ Learning Sequence (Connected, High Coherence) ═══");
    let mut virtuous_count = 0;

    for (i, obj) in objectives.iter().enumerate() {
        let result = bridged
            .learn_with_coherence(obj, true)
            .expect("Learning failed");

        println!("  {}. {} | {}", i + 1, obj.name, result.describe());

        // Show Layer 1 effects
        if result.phi_coherence_boost > 0.001 {
            println!(
                "     ↳ Φ→Coherence boost: +{:.2}%",
                result.phi_coherence_boost * 100.0
            );
        }
        if result.virtuous_cycle_active {
            println!("     ✨ VIRTUOUS CYCLE ACTIVE!");
            virtuous_count += 1;
        }
        println!();
    }

    println!("  Virtuous cycles triggered: {}", virtuous_count);
    println!(
        "  Total virtuous cycles: {}",
        bridged.virtuous_cycle_count()
    );
    println!(
        "  Final coherence: {:.0}%",
        bridged.coherence_level() * 100.0
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 2: Layer 2 - Hormone-Learning Integration
    // ─────────────────────────────────────────────────────────────────────────────

    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  LAYER 2: Hormone-Learning Integration                                         ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Cortisol (stress) → Impairs learning, reduces retention                       ║");
    println!("║  Dopamine (reward) → Boosts Φ gain, enhances integration                       ║");
    println!("║  Oxytocin (connection) → Boosts coherence gains                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Reset for fair comparison
    bridged.coherence_mut().coherence = 0.85;

    let test_objective = LearningObjective::new("test-hormones", "Hormone Test Objective")
        .with_difficulty(Difficulty::Beginner)
        .with_domain(Domain::NixOS)
        .build();

    // ─── 2a: High Stress State (Cortisol) ───

    println!("═══ 2a: Learning Under High Stress (Cortisol = 0.9) ═══");
    let high_stress = HormoneState {
        cortisol: 0.9, // Very high stress
        dopamine: 0.3, // Normal dopamine
        acetylcholine: 0.2,
        oxytocin: 0.3, // Low connection
        norepinephrine: 0.6,
        serotonin: 0.5,
    };
    bridged.set_hormones(high_stress);

    bridged.coherence_mut().coherence = 0.85;
    let stressed_result = bridged.learn_with_coherence(&test_objective, false);

    match &stressed_result {
        Ok(r) => {
            println!("  Result: {}", r.describe());
            println!("  Hormone Φ modifier: {:.2}x", r.hormone_phi_modifier);
            if r.stress_impaired {
                println!("  ⚠️  STRESS IMPAIRED - learning effectiveness reduced!");
            }
        }
        Err(e) => println!("  Failed due to stress: {}", e),
    }
    println!();

    // ─── 2b: Optimal State (High Dopamine + Oxytocin) ───

    println!("═══ 2b: Learning in Optimal State (Dopamine=0.9, Oxytocin=0.8) ═══");
    let optimal_state = HormoneState {
        cortisol: 0.2, // Low stress
        dopamine: 0.9, // High reward
        acetylcholine: 0.7,
        oxytocin: 0.8, // High connection
        norepinephrine: 0.6,
        serotonin: 0.7,
    };
    bridged.set_hormones(optimal_state);

    bridged.coherence_mut().coherence = 0.85;
    let optimal_result = bridged.learn_with_coherence(&test_objective, true);

    match &optimal_result {
        Ok(r) => {
            println!("  Result: {}", r.describe());
            println!(
                "  Hormone Φ modifier: {:.2}x (dopamine boost!)",
                r.hormone_phi_modifier
            );
            println!(
                "  Hormone coherence modifier: {:.2}x (oxytocin boost!)",
                r.hormone_coherence_modifier
            );
            println!(
                "  Φ-coherence boost: +{:.2}%",
                r.phi_coherence_boost * 100.0
            );
        }
        Err(e) => println!("  Failed: {}", e),
    }
    println!();

    // ─── 2c: Comparison ───

    println!("═══ Hormone Effect Comparison ═══");
    if let (Ok(stressed), Ok(optimal)) = (stressed_result, optimal_result) {
        println!(
            "  Stressed Φ modifier: {:.2}x",
            stressed.hormone_phi_modifier
        );
        println!(
            "  Optimal Φ modifier:  {:.2}x",
            optimal.hormone_phi_modifier
        );
        let improvement =
            (optimal.hormone_phi_modifier / stressed.hormone_phi_modifier - 1.0) * 100.0;
        println!(
            "  Improvement: +{:.0}% Φ gain in optimal state!",
            improvement
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 3: Layer 3 - Consciousness Graph Integration
    // ─────────────────────────────────────────────────────────────────────────────

    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  LAYER 3: Consciousness Graph Integration                                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Learning is tracked as a consciousness-producing activity                     ║");
    println!("║  Autopoietic monitor tracks self-sustaining learning patterns                  ║");
    println!("║  Operational closure measures if learning maintains itself                     ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Reset hormones to neutral
    bridged.set_hormones(HormoneState::neutral());
    bridged.coherence_mut().coherence = 0.85;

    println!("═══ Sustained Learning Sequence ═══");
    println!("  (Demonstrating autopoietic self-maintenance)");
    println!();

    // Create a longer sequence to build up autopoietic patterns
    let sustained_objectives = vec![
        ("derivations", "Understanding Derivations"),
        ("overlays", "Overlays and Overrides"),
        ("options", "Option Declarations"),
        ("services", "Service Configuration"),
        ("networking", "Network Management"),
    ];

    for (id, name) in &sustained_objectives {
        let obj = LearningObjective::new(id, name)
            .with_difficulty(Difficulty::Intermediate)
            .with_domain(Domain::NixOS)
            .build();

        let result = bridged
            .learn_with_coherence(&obj, true)
            .expect("Learning failed");

        // Get current autopoietic state
        let closure = bridged.operational_closure();
        let metrics = bridged.self_production_metrics();

        println!("  Learned: {}", name);
        println!(
            "    Closure: {:.2} | Cycles: {} | Avg Error: {:.2}",
            closure.closure_level, metrics.total_cycles, metrics.average_prediction_error
        );

        if result.virtuous_cycle_active {
            println!("    ✨ Virtuous cycle active!");
        }
    }
    println!();

    // Show final autopoietic state
    println!("═══ Autopoietic State Summary ═══");
    let final_metrics = bridged.self_production_metrics();
    let closure = bridged.operational_closure();

    println!("  Production cycles: {}", final_metrics.total_cycles);
    println!(
        "  Avg prediction error: {:.2}",
        final_metrics.average_prediction_error
    );
    println!("  Avg coherence: {:.2}", final_metrics.average_coherence);
    println!("  Boundary integrity: {:.2}", closure.boundary_integrity);
    println!(
        "  Self-production rate: {:.2}",
        closure.self_production_rate
    );
    println!();
    println!("  Operational Closure: {:.2}", closure.closure_level);
    println!(
        "  Boundary Intact: {}",
        if closure.boundary_integrity > 0.5 {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  Self-Producing: {}",
        if closure.self_production_rate > 0.5 {
            "✓"
        } else {
            "Not yet"
        }
    );
    println!(
        "  Autonomous: {}",
        if closure.is_maintaining() && closure.self_production_rate > 0.5 {
            "✓"
        } else {
            "Not yet"
        }
    );
    println!();
    println!(
        "  Is Self-Sustaining: {}",
        if bridged.is_self_sustaining() {
            "✓ YES!"
        } else {
            "Not yet"
        }
    );
    println!(
        "  Total Virtuous Cycles: {}",
        bridged.virtuous_cycle_count()
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Final Statistics
    // ─────────────────────────────────────────────────────────────────────────────

    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        Complete System Statistics                              ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    let stats = bridged.stats();

    println!();
    println!("  School Stats:");
    println!("    Total learned: {}", stats.school_stats.total_learned);
    println!(
        "    Mastered: {}/{}",
        stats.school_stats.mastered_objectives, stats.school_stats.total_objectives
    );
    println!(
        "    Hallucination rate: {:.1}%",
        stats.school_stats.hallucination_rate * 100.0
    );
    println!();
    println!("  Coherence Stats:");
    println!(
        "    Current coherence: {:.0}%",
        stats.coherence_stats.coherence * 100.0
    );
    println!(
        "    Relational resonance: {:.0}%",
        stats.coherence_stats.relational_resonance * 100.0
    );
    println!("    Operations: {}", stats.coherence_stats.operations_count);
    println!(
        "    Gratitude received: {}",
        stats.coherence_stats.gratitude_count
    );
    println!();
    println!("  Integration Stats:");
    println!("    Successful learnings: {}", stats.successful_learnings);
    println!("    Coherence-blocked: {}", stats.coherence_blocked_count);
    println!();
    println!("  Three-Layer Stats:");
    println!(
        "    Layer 1 - Virtuous cycles: {}",
        stats.virtuous_cycle_count
    );
    println!("    Layer 2 - (hormone effects shown per-learning)");
    println!(
        "    Layer 3 - Self-sustaining: {}",
        stats.is_self_sustaining
    );
    println!("    Layer 3 - Closure level: {:.2}", stats.closure_level);
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Key Insights
    // ─────────────────────────────────────────────────────────────────────────────

    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              Key Insights                                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                                ║");
    println!("║  LAYER 1: Bidirectional Φ-Coherence                                            ║");
    println!("║  • Learning that increases Φ also increases coherence                          ║");
    println!("║  • High coherence enables higher Φ gain                                        ║");
    println!("║  • The virtuous cycle creates accelerating improvement                         ║");
    println!("║                                                                                ║");
    println!("║  LAYER 2: Hormone Integration                                                  ║");
    println!("║  • Cortisol (stress) → impairs learning by reducing Φ gain                     ║");
    println!("║  • Dopamine (reward) → boosts Φ gain up to 30%                                 ║");
    println!("║  • Oxytocin (connection) → boosts coherence gain up to 50%                     ║");
    println!("║                                                                                ║");
    println!("║  LAYER 3: Autopoietic Self-Maintenance                                         ║");
    println!("║  • Learning is tracked as consciousness production cycles                      ║");
    println!("║  • Hallucinations are prediction errors in the world model                     ║");
    println!("║  • Operational closure measures learning sustainability                        ║");
    println!("║  • Self-sustaining learning maintains itself without external input            ║");
    println!("║                                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
}
