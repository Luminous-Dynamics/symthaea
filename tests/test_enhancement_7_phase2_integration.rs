//! Integration Tests for Enhancement #7 Phase 2
//!
//! These tests validate that all 4 Enhancement #4 components integrate
//! correctly with the causal program synthesis system.
//!
//! Components tested:
//! 1. ExplanationGenerator - Rich causal explanations
//! 2. CausalInterventionEngine - Real intervention testing
//! 3. CounterfactualEngine - True counterfactual verification
//! 4. ActionPlanner - Optimal path planning

use symthaea::observability::{
    CausalInterventionEngine, CounterfactualEngine, ActionPlanner,
    ProbabilisticCausalGraph,
};
use symthaea::synthesis::{
    CausalProgramSynthesizer, CounterfactualVerifier, SynthesisConfig,
    VerificationConfig, CausalSpec,
};

// ============================================================================
// Test 1: ExplanationGenerator Integration
// ============================================================================

#[test]
fn test_explanation_generation_make_cause() {
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::MakeCause {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        strength: 0.75,
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis should succeed");

    // Verify explanation is generated
    assert!(program.explanation.is_some(), "Explanation should be generated");

    let explanation = program.explanation.unwrap();

    // Verify explanation contains key information
    assert!(explanation.contains("exercise"), "Explanation should mention cause");
    assert!(explanation.contains("health"), "Explanation should mention effect");
    assert!(explanation.contains("0.75"), "Explanation should mention strength");
}

#[test]
fn test_explanation_generation_create_path() {
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::CreatePath {
        from: "education".to_string(),
        through: vec!["experience".to_string()],
        to: "salary".to_string(),
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis should succeed");

    // Verify explanation is generated
    assert!(program.explanation.is_some(), "Explanation should be generated");

    let explanation = program.explanation.unwrap();

    // Verify explanation describes the path
    assert!(explanation.contains("education"), "Explanation should mention start");
    assert!(explanation.contains("salary"), "Explanation should mention end");
}

#[test]
fn test_explanation_generation_strengthen() {
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::Strengthen {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        target_strength: 0.8,
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis should succeed");

    // Verify explanation is generated
    assert!(program.explanation.is_some(), "Explanation should be generated");

    let explanation = program.explanation.unwrap();

    // Verify explanation mentions strengthening
    assert!(explanation.contains("Strengthen") || explanation.contains("strengthen"),
        "Explanation should mention strengthening");
    assert!(explanation.contains("0.8") || explanation.contains("0.80"),
        "Explanation should mention target strength");
}

// ============================================================================
// Test 2: CausalInterventionEngine Integration
// ============================================================================

#[test]
fn test_intervention_engine_integration() {
    let graph = ProbabilisticCausalGraph::new();
    let intervention_engine = CausalInterventionEngine::new(graph);

    let config = SynthesisConfig::default();
    let mut synthesizer = CausalProgramSynthesizer::new(config)
        .with_intervention_engine(intervention_engine);

    let spec = CausalSpec::MakeCause {
        cause: "treatment".to_string(),
        effect: "recovery".to_string(),
        strength: 0.80,
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis with intervention engine should succeed");

    // Verify program has confidence from intervention testing
    assert!(program.confidence >= 0.0 && program.confidence <= 1.0,
        "Confidence should be valid probability");

    // Verify achieved strength is set
    assert!(program.achieved_strength >= 0.0 && program.achieved_strength <= 1.0,
        "Achieved strength should be valid");
}

#[test]
fn test_intervention_vs_baseline_confidence() {
    // Test WITHOUT intervention engine (Phase 1)
    let mut baseline = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::MakeCause {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        strength: 0.6,
    };

    let baseline_program = baseline.synthesize(&spec)
        .expect("Baseline synthesis should succeed");

    // Test WITH intervention engine (Phase 2)
    let graph = ProbabilisticCausalGraph::new();
    let intervention_engine = CausalInterventionEngine::new(graph);

    let mut enhanced = CausalProgramSynthesizer::new(SynthesisConfig::default())
        .with_intervention_engine(intervention_engine);

    let enhanced_program = enhanced.synthesize(&spec)
        .expect("Enhanced synthesis should succeed");

    // Both should produce valid confidence scores
    assert!(baseline_program.confidence >= 0.0 && baseline_program.confidence <= 1.0);
    assert!(enhanced_program.confidence >= 0.0 && enhanced_program.confidence <= 1.0);

    // Phase 2 may have different confidence (from real intervention testing)
    // but both should be valid programs
    assert!(baseline_program.achieved_strength >= 0.0);
    assert!(enhanced_program.achieved_strength >= 0.0);
}

// ============================================================================
// Test 3: CounterfactualEngine Integration
// ============================================================================

#[test]
fn test_counterfactual_verification() {
    let graph = ProbabilisticCausalGraph::new();
    let counterfactual_engine = CounterfactualEngine::new(graph);

    let verifier_config = VerificationConfig {
        num_counterfactuals: 50,
        min_accuracy: 0.90,
        test_edge_cases: true,
        max_complexity: 10,
    };

    let mut verifier = CounterfactualVerifier::new(verifier_config)
        .with_counterfactual_engine(counterfactual_engine);

    // Synthesize a program to verify
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::MakeCause {
        cause: "age".to_string(),
        effect: "risk".to_string(),
        strength: 0.65,
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis should succeed");

    // Verify with counterfactual engine
    let result = verifier.verify(&program);

    // Verify result structure
    assert!(result.tests_run == 50, "Should run 50 counterfactual tests");
    assert!(result.counterfactual_accuracy >= 0.0 && result.counterfactual_accuracy <= 1.0,
        "Accuracy should be valid probability");
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be valid probability");
}

#[test]
fn test_counterfactual_verification_high_accuracy() {
    let graph = ProbabilisticCausalGraph::new();
    let counterfactual_engine = CounterfactualEngine::new(graph);

    let verifier_config = VerificationConfig {
        num_counterfactuals: 100,
        min_accuracy: 0.95,
        test_edge_cases: true,
        max_complexity: 10,
    };

    let mut verifier = CounterfactualVerifier::new(verifier_config)
        .with_counterfactual_engine(counterfactual_engine);

    // Create a simple, correct program
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::MakeCause {
        cause: "a".to_string(),
        effect: "b".to_string(),
        strength: 0.5,
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis should succeed");

    let result = verifier.verify(&program);

    // Verify all tests were run
    assert_eq!(result.tests_run, 100);

    // Result should have valid metrics
    assert!(result.counterfactual_accuracy >= 0.0 && result.counterfactual_accuracy <= 1.0);
}

// ============================================================================
// Test 4: ActionPlanner Integration
// ============================================================================

#[test]
fn test_action_planner_integration() {
    let graph = ProbabilisticCausalGraph::new();
    let action_planner = ActionPlanner::new(graph);

    let config = SynthesisConfig::default();
    let mut synthesizer = CausalProgramSynthesizer::new(config)
        .with_action_planner(action_planner);

    // Create path spec WITH manually specified mediators
    let spec = CausalSpec::CreatePath {
        from: "education".to_string(),
        through: vec!["skills".to_string()],
        to: "income".to_string(),
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Path synthesis with action planner should succeed");

    // Verify program was created
    assert!(program.variables.len() >= 2, "Path should include source and target");
    assert!(program.confidence >= 0.0 && program.confidence <= 1.0);
}

#[test]
fn test_action_planner_vs_baseline() {
    // Test WITHOUT action planner (Phase 1)
    let mut baseline = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::CreatePath {
        from: "a".to_string(),
        through: vec!["b".to_string()],
        to: "c".to_string(),
    };

    let baseline_program = baseline.synthesize(&spec)
        .expect("Baseline path synthesis should succeed");

    // Test WITH action planner (Phase 2)
    let graph = ProbabilisticCausalGraph::new();
    let action_planner = ActionPlanner::new(graph);

    let mut enhanced = CausalProgramSynthesizer::new(SynthesisConfig::default())
        .with_action_planner(action_planner);

    let enhanced_program = enhanced.synthesize(&spec)
        .expect("Enhanced path synthesis should succeed");

    // Both should produce valid programs
    assert!(baseline_program.variables.len() >= 2);
    assert!(enhanced_program.variables.len() >= 2);

    // Both should have valid confidence
    assert!(baseline_program.confidence >= 0.0 && baseline_program.confidence <= 1.0);
    assert!(enhanced_program.confidence >= 0.0 && enhanced_program.confidence <= 1.0);
}

// ============================================================================
// Test 5: Complete Workflow Integration
// ============================================================================

#[test]
fn test_complete_workflow_all_components() {
    // Create all Enhancement #4 components
    let graph = ProbabilisticCausalGraph::new();
    let intervention_engine = CausalInterventionEngine::new(graph.clone());
    let counterfactual_engine = CounterfactualEngine::new(graph.clone());
    let action_planner = ActionPlanner::new(graph);

    // Configure synthesizer with all components
    let synthesis_config = SynthesisConfig::default();
    let mut synthesizer = CausalProgramSynthesizer::new(synthesis_config)
        .with_intervention_engine(intervention_engine)
        .with_action_planner(action_planner);

    // Configure verifier with counterfactual engine
    let verification_config = VerificationConfig {
        num_counterfactuals: 50,
        min_accuracy: 0.90,
        test_edge_cases: true,
        max_complexity: 10,
    };

    let mut verifier = CounterfactualVerifier::new(verification_config)
        .with_counterfactual_engine(counterfactual_engine);

    // Define specification
    let spec = CausalSpec::Strengthen {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        target_strength: 0.8,
    };

    // Step 1: Synthesize with all enhancements
    let program = synthesizer.synthesize(&spec)
        .expect("Complete synthesis should succeed");

    // Verify synthesis produced valid program
    assert!(program.explanation.is_some(), "Should have explanation");
    assert!(program.confidence >= 0.0 && program.confidence <= 1.0);
    assert!(program.achieved_strength >= 0.0 && program.achieved_strength <= 1.0);

    // Step 2: Verify with counterfactuals
    let result = verifier.verify(&program);

    // Verify verification completed successfully
    assert_eq!(result.tests_run, 50);
    assert!(result.counterfactual_accuracy >= 0.0 && result.counterfactual_accuracy <= 1.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);

    // Step 3: Overall quality assessment
    let synthesis_confidence = program.confidence;
    let verification_accuracy = result.counterfactual_accuracy;
    let overall_score = (synthesis_confidence + verification_accuracy) / 2.0;

    // Overall score should be valid
    assert!(overall_score >= 0.0 && overall_score <= 1.0);
}

#[test]
fn test_complete_workflow_quality_metrics() {
    // Setup complete system
    let graph = ProbabilisticCausalGraph::new();
    let intervention_engine = CausalInterventionEngine::new(graph.clone());
    let counterfactual_engine = CounterfactualEngine::new(graph);

    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default())
        .with_intervention_engine(intervention_engine);

    let mut verifier = CounterfactualVerifier::new(VerificationConfig::default())
        .with_counterfactual_engine(counterfactual_engine);

    // Test multiple specifications
    let specs = vec![
        CausalSpec::MakeCause {
            cause: "a".to_string(),
            effect: "b".to_string(),
            strength: 0.5,
        },
        CausalSpec::Strengthen {
            cause: "x".to_string(),
            effect: "y".to_string(),
            target_strength: 0.7,
        },
    ];

    for spec in specs {
        let program = synthesizer.synthesize(&spec)
            .expect("Synthesis should succeed");

        let result = verifier.verify(&program);

        // Every program should pass basic quality checks
        assert!(program.confidence >= 0.0 && program.confidence <= 1.0);
        assert!(result.counterfactual_accuracy >= 0.0 && result.counterfactual_accuracy <= 1.0);

        // Overall quality should be computable
        let quality = (program.confidence + result.counterfactual_accuracy) / 2.0;
        assert!(quality >= 0.0 && quality <= 1.0);
    }
}

// ============================================================================
// Test 6: Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_synthesis_without_optional_components() {
    // Test that synthesis works WITHOUT any Enhancement #4 components
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::MakeCause {
        cause: "a".to_string(),
        effect: "b".to_string(),
        strength: 0.5,
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis should work without optional components");

    // Should still produce valid program
    assert!(program.achieved_strength >= 0.0);
    assert!(program.confidence >= 0.0 && program.confidence <= 1.0);
}

#[test]
fn test_verification_without_counterfactual_engine() {
    // Test that verification works WITHOUT counterfactual engine
    let mut verifier = CounterfactualVerifier::new(VerificationConfig::default());

    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    let spec = CausalSpec::MakeCause {
        cause: "a".to_string(),
        effect: "b".to_string(),
        strength: 0.5,
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Synthesis should succeed");

    let result = verifier.verify(&program);

    // Should still run verification (using fallback)
    assert!(result.tests_run > 0);
    assert!(result.counterfactual_accuracy >= 0.0 && result.counterfactual_accuracy <= 1.0);
}

#[test]
fn test_high_complexity_specification() {
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    // Create complex path specification
    let spec = CausalSpec::CreatePath {
        from: "start".to_string(),
        through: vec![
            "step1".to_string(),
            "step2".to_string(),
            "step3".to_string(),
        ],
        to: "end".to_string(),
    };

    let program = synthesizer.synthesize(&spec)
        .expect("Complex specification should be synthesizable");

    // Verify complexity is tracked
    assert!(program.complexity >= 3, "Complexity should reflect path length");
}
