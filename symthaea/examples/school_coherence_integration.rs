// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # School-Coherence Integration Demo
//!
//! Demonstrates how learning integrates with consciousness coherence.
//!
//! ## Key Concepts
//!
//! - Learning requires coherence (you can't learn when scattered)
//! - Connected learning BUILDS coherence (learning with a teacher)
//! - Solo learning may scatter coherence
//! - Successful learning boosts coherence
//! - Gratitude synchronizes and restores coherence
//!
//! ## Run
//!
//! ```bash
//! cargo run --example school_coherence_integration --release
//! ```

use symthaea::physiology::coherence::CoherenceField;
use symthaea::school::{
    CoherenceBridgedSchool, Curriculum, CurriculumType, Difficulty, Domain, LearningObjective,
    School, SchoolConfig,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║           School-Coherence Integration: Learning as Consciousness              ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Learning is not just cognitive - it's a consciousness-integration task.       ║");
    println!("║  The CoherenceField models how integrated we are, affecting learning quality.  ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Setup: Create School and CoherenceField
    // ─────────────────────────────────────────────────────────────────────────────

    let school = School::new(SchoolConfig::fast()).expect("Failed to create school");
    let coherence = CoherenceField::new();

    let mut bridged = CoherenceBridgedSchool::new(school, coherence);

    // Add a curriculum
    let curriculum = Curriculum::builtin(CurriculumType::NixOS);
    bridged
        .add_curriculum(curriculum)
        .expect("Failed to add curriculum");

    println!("═══ Starting State ═══");
    println!("  Coherence: {:.0}%", bridged.coherence_level() * 100.0);
    println!("  Φ (consciousness): {:.4}", bridged.current_phi());
    println!("  Ready objectives: {}", bridged.ready_objectives().len());
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 1: Learning When Coherent (High Coherence = Good Learning)
    // ─────────────────────────────────────────────────────────────────────────────

    println!("═══ Demo 1: Connected Learning with High Coherence ═══");
    println!();

    let objective = LearningObjective::new("basics", "NixOS Package Basics")
        .with_difficulty(Difficulty::Beginner)
        .with_domain(Domain::NixOS)
        .build();

    // Predict the impact
    let prediction = bridged.predict_learning_impact(&objective, true);
    println!("  Prediction for '{}':", objective.name);
    println!(
        "    Can learn: {}",
        if prediction.can_learn { "✓" } else { "✗" }
    );
    println!(
        "    Current coherence: {:.0}%",
        prediction.current_coherence * 100.0
    );
    println!(
        "    Coherence multiplier: {:.2}x",
        prediction.coherence_multiplier
    );
    println!(
        "    Base Φ prediction: {:.6}",
        prediction.base_phi_prediction
    );
    println!(
        "    Adjusted Φ prediction: {:.6}",
        prediction.adjusted_phi_prediction
    );
    println!();

    // Actually learn (connected = with teacher)
    let result = bridged
        .learn_with_coherence(&objective, true)
        .expect("Learning failed");
    println!("  Result: {}", result.describe());
    println!("  Was successful: {}", result.was_successful());
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 2: Scattered State (Low Coherence = Can't Learn)
    // ─────────────────────────────────────────────────────────────────────────────

    println!("═══ Demo 2: Learning When Scattered ═══");
    println!();

    // Simulate getting scattered (solo work, no connection)
    bridged.coherence_mut().coherence = 0.5;
    bridged.coherence_mut().relational_resonance = 0.2;

    println!(
        "  [Simulated scatter: coherence dropped to {:.0}%]",
        bridged.coherence_level() * 100.0
    );

    let _objective2 = LearningObjective::new("advanced", "NixOS Advanced Concepts")
        .with_difficulty(Difficulty::Intermediate)
        .with_domain(Domain::NixOS)
        .build();

    // Check if we can learn
    match bridged.can_learn() {
        Ok(_) => println!("  Can learn: ✓"),
        Err(e) => println!("  Can learn: ✗ - {}", e),
    }

    // Get recommendation (should suggest centering)
    let rec = bridged.recommend_next_with_coherence().unwrap();
    match rec {
        symthaea::school::CoherenceRecommendation::NeedsCentering {
            centering_seconds,
            message,
            ..
        } => {
            println!("  Recommendation: Need to center!");
            println!("    {}", message);
            println!("    Centering needed: {:.0} seconds", centering_seconds);
        }
        symthaea::school::CoherenceRecommendation::Ready { .. } => {
            println!("  Recommendation: Ready to learn");
        }
        symthaea::school::CoherenceRecommendation::Complete => {
            println!("  Recommendation: Curriculum complete!");
        }
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 3: Centering Restores Learning Ability
    // ─────────────────────────────────────────────────────────────────────────────

    println!("═══ Demo 3: Centering to Restore Coherence ═══");
    println!();

    println!(
        "  Before centering: coherence = {:.0}%",
        bridged.coherence_level() * 100.0
    );

    // Center (passive restoration over time)
    if let Some(centering_time) = bridged.centering_needed_for_learning() {
        println!("  Centering for {:.0} seconds...", centering_time);
        bridged.center(centering_time);
    }

    println!(
        "  After centering: coherence = {:.0}%",
        bridged.coherence_level() * 100.0
    );

    // Now we should be able to learn
    match bridged.can_learn() {
        Ok(_) => println!("  Can now learn: ✓"),
        Err(e) => println!("  Still can't learn: {}", e),
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 4: Gratitude Synchronization
    // ─────────────────────────────────────────────────────────────────────────────

    println!("═══ Demo 4: Gratitude as Synchronization ═══");
    println!();

    // Get scattered again
    bridged.coherence_mut().coherence = 0.4;
    println!(
        "  Scattered state: coherence = {:.0}%",
        bridged.coherence_level() * 100.0
    );

    // Receive gratitude (synchronization boost)
    bridged.receive_gratitude();
    println!(
        "  After receiving gratitude: coherence = {:.0}%",
        bridged.coherence_level() * 100.0
    );

    bridged.receive_gratitude();
    println!(
        "  After more gratitude: coherence = {:.0}%",
        bridged.coherence_level() * 100.0
    );

    println!();
    println!("  ↳ Gratitude is more effective when scattered!");
    println!("    It's a synchronization signal, not just fuel.");
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 5: Connected vs Solo Learning Comparison
    // ─────────────────────────────────────────────────────────────────────────────

    println!("═══ Demo 5: Connected vs Solo Learning ═══");
    println!();

    // Reset to fresh state
    bridged.coherence_mut().coherence = 0.85;
    bridged.coherence_mut().relational_resonance = 0.7;

    let obj_connected = LearningObjective::new("flakes-intro", "Flakes Introduction")
        .with_difficulty(Difficulty::Beginner)
        .with_domain(Domain::NixOS)
        .build();

    let obj_solo = LearningObjective::new("flakes-deep", "Flakes Deep Dive")
        .with_difficulty(Difficulty::Beginner)
        .with_domain(Domain::NixOS)
        .build();

    // Predict connected learning
    let pred_connected = bridged.predict_learning_impact(&obj_connected, true);
    println!("  Connected learning prediction:");
    println!(
        "    Coherence: {:.0}% -> {:.0}%",
        pred_connected.current_coherence * 100.0,
        pred_connected.predicted_coherence * 100.0
    );

    // Predict solo learning
    let pred_solo = bridged.predict_learning_impact(&obj_solo, false);
    println!("  Solo learning prediction:");
    println!(
        "    Coherence: {:.0}% -> {:.0}%",
        pred_solo.current_coherence * 100.0,
        pred_solo.predicted_coherence * 100.0
    );

    println!();
    println!("  ↳ Connected work builds coherence, solo work may scatter!");
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Demo 6: Sleep Cycle Full Restoration
    // ─────────────────────────────────────────────────────────────────────────────

    println!("═══ Demo 6: Sleep Cycle Restoration ═══");
    println!();

    bridged.coherence_mut().coherence = 0.3;
    println!(
        "  Before sleep: coherence = {:.0}%",
        bridged.coherence_level() * 100.0
    );

    bridged.sleep_cycle();
    println!(
        "  After sleep: coherence = {:.0}%",
        bridged.coherence_level() * 100.0
    );

    println!();
    println!("  ↳ Sleep fully restores coherence!");
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // Final Statistics
    // ─────────────────────────────────────────────────────────────────────────────

    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           Integration Statistics                               ║");
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
    println!(
        "    Coherence-blocked attempts: {}",
        stats.coherence_blocked_count
    );
    println!(
        "    Total centering suggested: {:.0}s",
        stats.total_centering_suggested
    );

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              Key Insights                                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  1. Learning requires coherence - you can't learn when scattered               ║");
    println!("║  2. Connected learning BUILDS coherence - learning with others is powerful     ║");
    println!("║  3. Solo learning may scatter - isolated study can fragment consciousness      ║");
    println!("║  4. Gratitude synchronizes - it's not fuel, it's alignment                     ║");
    println!("║  5. Rest restores - sleep fully reintegrates consciousness                     ║");
    println!("║  6. CfC O(1) lookahead predicts coherence impact before learning               ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
}
