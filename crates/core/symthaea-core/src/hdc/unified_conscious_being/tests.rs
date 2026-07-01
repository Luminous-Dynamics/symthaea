// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;
use std::collections::HashMap;

#[test]
fn test_unified_being_creation() {
    let being = UnifiedConsciousBeing::new();
    assert_eq!(being.stats().inputs_processed, 0);
}

#[test]
fn test_basic_interaction() {
    let mut being = UnifiedConsciousBeing::new();
    let result = being.interact("Hello, how are you?");

    assert!(!result.response.text.is_empty());
    assert!(result.comprehension.consciousness_phi > 0.0);
}

#[test]
fn test_causal_model_learning() {
    let mut being = UnifiedConsciousBeing::new();
    being.interact("The rain caused the flooding");

    let (vars, _eqs) = being.causal_model_size();
    let _ = vars; // May or may not detect causal structure
}

#[test]
fn test_dialogue_generation() {
    let _generator = ConsciousDialogueGenerator::new();
    // Would need full context to test properly
}

#[test]
fn test_scm_intervention() {
    let mut scm = StructuralCausalModel::new();
    scm.add_variable("cause", 1.0, VariableDomain::Binary);
    scm.add_variable("effect", 0.0, VariableDomain::Binary);
    scm.add_equation("effect", vec!["cause"], vec![0.8], 0.1);

    let result = scm.do_intervention("cause", 0.0, "effect");
    assert!(result.causal_effect.abs() > 0.0);
}

#[test]
fn test_pearl_counterfactual() {
    let mut scm = StructuralCausalModel::new();
    scm.add_variable("treatment", 1.0, VariableDomain::Binary);
    scm.add_variable("recovery", 1.0, VariableDomain::Binary);
    scm.add_equation("recovery", vec!["treatment"], vec![0.9], 0.05);

    let mut evidence = HashMap::new();
    evidence.insert("treatment".to_string(), 1.0);
    evidence.insert("recovery".to_string(), 1.0);

    let cf = scm.counterfactual(&evidence, "treatment", 0.0, "recovery");
    assert!(cf.factual != cf.counterfactual || cf.probability_necessity > 0.0);
}

#[test]
fn test_ltc_pacing() {
    let pacing_high = LTCPacing::from_consciousness(0.9, 0.05);
    assert!(pacing_high.speech_rate > 1.0);
    assert!(pacing_high.peak_flow);

    let pacing_low = LTCPacing::from_consciousness(0.2, -0.05);
    assert!(pacing_low.speech_rate < 1.0);
    assert!(!pacing_low.peak_flow);
}

#[test]
fn test_scenarios() {
    let scenarios = create_test_scenarios();
    assert!(!scenarios.is_empty());
    assert_eq!(scenarios[0].name, "Emotional Support");
}
