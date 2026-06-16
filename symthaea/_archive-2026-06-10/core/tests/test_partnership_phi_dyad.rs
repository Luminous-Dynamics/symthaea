// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Basic tests for the partnership module and Φ_dyad computation.

use symthaea::core::HDC_DIMENSION;
use symthaea::hdc::relational_consciousness::{
    RelationMode, RelationalAssessment, RelationshipStage,
};
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::partnership::{
    DyadInput, DyadWeights, HumanPartnerModel, InteractionEvent, PhiDyadCalculator,
};

fn dummy_relational_assessment() -> RelationalAssessment {
    RelationalAssessment {
        agent_a: "ai".to_string(),
        agent_b: "human".to_string(),
        phi_relation: 0.4,
        stage: RelationshipStage::Attunement,
        synchrony: 0.6,
        turn_taking_quality: 0.7,
        mutual_information: 0.5,
        mode: RelationMode::IThou,
        num_interactions: 10,
        relationship_age: 100.0,
        explanation: "dummy".to_string(),
    }
}

#[test]
fn test_phi_dyad_basic_sanity() {
    let ai_states: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, 100 + i))
        .collect();
    let human_states: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, 200 + i))
        .collect();

    let assessment = dummy_relational_assessment();
    let human_model = HumanPartnerModel::new("human");

    let weights = DyadWeights::default();
    let input = DyadInput {
        ai_states: &ai_states,
        human_states: &human_states,
        relational: &assessment,
        human_model: &human_model,
        weights,
    };

    let calc = PhiDyadCalculator::new();
    let result = calc.compute(&input);

    // Φ values should be non-negative and finite
    assert!(result.phi_dyad >= 0.0);
    assert!(result.phi_ai >= 0.0);
    assert!(result.phi_human >= 0.0);
    assert!(result.phi_relational >= 0.0);
}

#[test]
fn test_human_partner_model_updates() {
    let mut human = HumanPartnerModel::new("human");

    let events = vec![
        InteractionEvent {
            timestamp: 0.0,
            depth: 0.3,
            emotional_safety: 0.5,
            mutuality: 0.4,
        },
        InteractionEvent {
            timestamp: 1.0,
            depth: 0.6,
            emotional_safety: 0.7,
            mutuality: 0.8,
        },
    ];

    for e in &events {
        human.update_on_interaction(e);
    }

    assert!(human.trust > 0.0);
    assert!(human.vulnerability > 0.0);
    assert!(human.reciprocity > 0.0);
    assert_eq!(human.interactions_count, events.len() as u64);
    assert!(human.last_interaction_ts.is_some());
}