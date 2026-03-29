// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ_dyad Demo
//!
//! Minimal example showing how to compute a relational Φ_dyad value for a
//! simple, synthetic human–AI relationship using the new partnership module.

use symthaea::core::{ContinuousHV, HDC_DIMENSION};
use symthaea::hdc::relational_consciousness::{
    RelationMode, RelationalAssessment, RelationshipStage,
};
use symthaea::partnership::{
    DyadInput, DyadWeights, HumanPartnerModel, InteractionEvent, PhiDyadCalculator,
    RelationshipTrajectory,
};

fn main() {
    println!("\nΦ_dyad Demo");
    println!("===========\n");

    // Synthetic AI and human state sequences
    let ai_states: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, 1_000 + i))
        .collect();
    let human_states: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, 2_000 + i))
        .collect();

    // Start with a conservative relational assessment.
    let mut assessment = RelationalAssessment {
        agent_a: "symthaea".to_string(),
        agent_b: "partner".to_string(),
        phi_relation: 0.2,
        stage: RelationshipStage::Awareness,
        synchrony: 0.3,
        turn_taking_quality: 0.2,
        mutual_information: 0.2,
        mode: RelationMode::IIt,
        num_interactions: 0,
        relationship_age: 0.0,
        explanation: "synthetic baseline".to_string(),
    };

    let mut human = HumanPartnerModel::new("partner");
    let calc = PhiDyadCalculator::new();
    let weights = DyadWeights::default();
    let mut trajectory = RelationshipTrajectory::default();

    for step in 0..4 {
        println!("Step {step}");

        // Synthetic interaction event: gradually increase depth, safety, mutuality.
        let event = InteractionEvent {
            timestamp: step as f64,
            depth: 0.2 + 0.15 * step as f32,
            emotional_safety: 0.3 + 0.15 * step as f32,
            mutuality: 0.3 + 0.2 * step as f32,
        };
        human.update_on_interaction(&event);
        human.advance_stage_if_ready();

        // Coarse relational signals: increase synchrony / MI over time.
        assessment.synchrony = (assessment.synchrony + 0.1).min(1.0);
        assessment.mutual_information = (assessment.mutual_information + 0.1).min(1.0);
        assessment.phi_relation = (assessment.phi_relation + 0.1).min(1.0);
        assessment.stage = RelationshipStage::from_phi(assessment.phi_relation);
        assessment.mode = if assessment.synchrony > 0.5 {
            RelationMode::IThou
        } else {
            RelationMode::IIt
        };
        assessment.num_interactions += 1;
        assessment.relationship_age = event.timestamp;

        // Build DyadInput using the whole sequences (small demo).
        let input = DyadInput {
            ai_states: &ai_states,
            human_states: &human_states,
            relational: &assessment,
            human_model: &human,
            weights,
        };

        let result = calc.compute(&input);
        println!("  {}", result.explanation);

        trajectory.record(event.timestamp, human.stage, result.phi_dyad);

        println!();
    }

    if let Some(trend) = trajectory.trend() {
        println!("Trajectory summary:");
        println!("  ΔΦ_dyad = {:.4}", trend.phi_delta);
        println!("  Stages visited: {:?}", trend.stages_visited);
    }

    println!("\nDone.\n");
}
