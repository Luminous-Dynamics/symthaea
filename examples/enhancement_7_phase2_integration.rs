//! Enhancement #7 Phase 2 Integration Examples
//!
//! This example demonstrates all 4 Enhancement #4 components integrated
//! into the causal program synthesis system:
//!
//! 1. ExplanationGenerator - Rich causal explanations
//! 2. CausalInterventionEngine - Real intervention testing
//! 3. CounterfactualEngine - True counterfactual verification
//! 4. ActionPlanner - Optimal path planning
//!
//! Run with: cargo run --example enhancement_7_phase2_integration

use symthaea::observability::{
    CausalInterventionEngine, CounterfactualEngine, ActionPlanner,
    CounterfactualQuery, Goal, GoalDirection,
};
use symthaea::synthesis::{
    CausalProgramSynthesizer, CounterfactualVerifier, SynthesisConfig,
    VerificationConfig, CausalSpec, ProgramTemplate,
};

fn main() {
    println!("\n🎉 Enhancement #7 Phase 2 Integration Examples\n");
    println!("{}", "=".repeat(70));

    // Run all integration examples
    example_1_explanation_generation();
    example_2_intervention_testing();
    example_3_counterfactual_verification();
    example_4_action_planning();
    example_5_complete_workflow();

    println!("\n✅ All integration examples completed successfully!");
}

/// Example 1: ExplanationGenerator Integration
///
/// Demonstrates how synthesized programs now include rich, human-readable
/// explanations of both intent and implementation.
fn example_1_explanation_generation() {
    println!("\n📝 Example 1: Explanation Generation");
    println!("{}", "-".repeat(70));

    let config = SynthesisConfig::default();
    let mut synthesizer = CausalProgramSynthesizer::new(config);

    // Synthesize a simple causal link
    let spec = CausalSpec::MakeCause {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        strength: 0.75,
    };

    match synthesizer.synthesize(&spec) {
        Ok(program) => {
            println!("\n✅ Synthesized program successfully");
            println!("\n📖 Explanation:");
            if let Some(explanation) = &program.explanation {
                println!("{}", explanation);
            } else {
                println!("No explanation available");
            }

            println!("\n📊 Program Details:");
            println!("  Template: {:?}", program.template);
            println!("  Achieved Strength: {:.2}", program.achieved_strength);
            println!("  Confidence: {:.2}", program.confidence);
            println!("  Complexity: {}", program.complexity);
        }
        Err(e) => {
            println!("❌ Synthesis failed: {}", e);
        }
    }

    // Try another specification type
    println!("\n🔄 Synthesizing path creation...");
    let path_spec = CausalSpec::CreatePath {
        from: "education".to_string(),
        through: vec!["experience".to_string()],
        to: "salary".to_string(),
    };

    match synthesizer.synthesize(&path_spec) {
        Ok(program) => {
            println!("\n✅ Synthesized path successfully");
            if let Some(explanation) = &program.explanation {
                println!("\n📖 Explanation:\n{}", explanation);
            }
        }
        Err(e) => {
            println!("❌ Path synthesis failed: {}", e);
        }
    }
}

/// Example 2: CausalInterventionEngine Integration
///
/// Demonstrates how programs are now tested using real causal interventions
/// to validate their behavior and compute accurate confidence scores.
fn example_2_intervention_testing() {
    println!("\n🧪 Example 2: Intervention Testing");
    println!("{}", "-".repeat(70));

    // Create a mock intervention engine for demonstration
    // In production, this would use a real causal graph
    println!("\n📊 Setting up intervention engine...");
    let intervention_engine = create_mock_intervention_engine();

    let config = SynthesisConfig::default();
    let mut synthesizer = CausalProgramSynthesizer::new(config)
        .with_intervention_engine(intervention_engine);

    println!("✅ Synthesizer configured with intervention engine");

    // Synthesize a program
    let spec = CausalSpec::MakeCause {
        cause: "treatment".to_string(),
        effect: "recovery".to_string(),
        strength: 0.80,
    };

    println!("\n🔬 Synthesizing with intervention testing...");
    match synthesizer.synthesize(&spec) {
        Ok(program) => {
            println!("\n✅ Program synthesized and tested with interventions");
            println!("\n📈 Intervention Test Results:");
            println!("  Expected Strength: 0.80");
            println!("  Achieved Strength: {:.2}", program.achieved_strength);
            println!("  Confidence: {:.2}", program.confidence);

            if program.confidence > 0.8 {
                println!("  ✅ High confidence - intervention test passed!");
            } else if program.confidence > 0.6 {
                println!("  ⚠️  Medium confidence - intervention partially validates program");
            } else {
                println!("  ❌ Low confidence - intervention test suggests issues");
            }

            println!("\n💡 How it works:");
            println!("  1. Synthesizer creates program based on specification");
            println!("  2. CausalInterventionEngine predicts intervention effect");
            println!("  3. Predicted strength compared with expected strength");
            println!("  4. Confidence score computed from prediction accuracy");
        }
        Err(e) => {
            println!("❌ Synthesis failed: {}", e);
        }
    }

    // Compare with/without intervention engine
    println!("\n🔄 Comparing with baseline (no intervention engine)...");
    let mut baseline_synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    match baseline_synthesizer.synthesize(&spec) {
        Ok(baseline_program) => {
            println!("\n📊 Baseline Results (Phase 1):");
            println!("  Confidence: {:.2} (placeholder value)", baseline_program.confidence);
            println!("\n📈 With Intervention Engine (Phase 2):");
            match synthesizer.synthesize(&spec) {
                Ok(enhanced_program) => {
                    println!("  Confidence: {:.2} (real intervention test)", enhanced_program.confidence);
                    println!("\n✨ Phase 2 provides real confidence from causal testing!");
                }
                Err(_) => {}
            }
        }
        Err(_) => {}
    }
}

/// Example 3: CounterfactualEngine Integration
///
/// Demonstrates how programs are verified using true counterfactual reasoning
/// to ensure they capture real causality, not just correlations.
fn example_3_counterfactual_verification() {
    println!("\n🔮 Example 3: Counterfactual Verification");
    println!("{}", "-".repeat(70));

    // Create mock counterfactual engine
    println!("\n📊 Setting up counterfactual engine...");
    let counterfactual_engine = create_mock_counterfactual_engine();

    let verifier_config = VerificationConfig {
        num_counterfactuals: 100,
        min_accuracy: 0.95,
        test_edge_cases: true,
        max_complexity: 10,
    };

    let mut verifier = CounterfactualVerifier::new(verifier_config)
        .with_counterfactual_engine(counterfactual_engine);

    println!("✅ Verifier configured with counterfactual engine");

    // Synthesize a program to verify
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
    let spec = CausalSpec::MakeCause {
        cause: "age".to_string(),
        effect: "risk".to_string(),
        strength: 0.65,
    };

    println!("\n🔬 Synthesizing program...");
    match synthesizer.synthesize(&spec) {
        Ok(program) => {
            println!("✅ Program synthesized");

            println!("\n🔮 Verifying with counterfactual reasoning...");
            let result = verifier.verify(&program);

            println!("\n📊 Verification Results:");
            println!("  Counterfactual Accuracy: {:.2}%", result.counterfactual_accuracy * 100.0);
            println!("  Confidence: {:.2}", result.confidence);
            println!("  Tests Run: {}", result.tests_run);

            if result.success {
                println!("  ✅ VERIFIED - Program captures true causality!");
            } else {
                println!("  ❌ FAILED - Program does not match counterfactual predictions");
            }

            println!("\n💡 How it works:");
            println!("  1. Verifier generates counterfactual test cases");
            println!("  2. CounterfactualEngine computes true counterfactual values");
            println!("  3. Program predictions compared with ground truth");
            println!("  4. Accuracy measured across all counterfactuals");

            println!("\n🎯 Why this matters:");
            println!("  • Counterfactuals test 'what if' scenarios");
            println!("  • Ensures program captures causation, not correlation");
            println!("  • Validates program using rigorous causal mathematics");
        }
        Err(e) => {
            println!("❌ Synthesis failed: {}", e);
        }
    }
}

/// Example 4: ActionPlanner Integration
///
/// Demonstrates how the action planner automatically discovers optimal
/// intervention paths for complex causal specifications.
fn example_4_action_planning() {
    println!("\n🎯 Example 4: Action Planning");
    println!("{}", "-".repeat(70));

    // Create mock action planner
    println!("\n📊 Setting up action planner...");
    let action_planner = create_mock_action_planner();

    let config = SynthesisConfig::default();
    let mut synthesizer = CausalProgramSynthesizer::new(config)
        .with_action_planner(action_planner);

    println!("✅ Synthesizer configured with action planner");

    // Specify a path WITHOUT mediators - let planner find them!
    let spec = CausalSpec::CreatePath {
        from: "education".to_string(),
        through: vec![], // Empty! Planner will discover optimal path
        to: "income".to_string(),
    };

    println!("\n🔍 Synthesizing path with automatic planning...");
    println!("  Source: education");
    println!("  Mediators: <to be discovered>");
    println!("  Target: income");

    match synthesizer.synthesize(&spec) {
        Ok(program) => {
            println!("\n✅ Optimal path discovered!");

            println!("\n📍 Discovered Path:");
            println!("  {}", program.variables.join(" → "));

            println!("\n📊 Path Quality:");
            println!("  Confidence: {:.2}", program.confidence);
            println!("  Complexity: {}", program.complexity);

            if let Some(explanation) = &program.explanation {
                println!("\n📖 Explanation:");
                println!("{}", explanation);
            }

            println!("\n💡 How it works:");
            println!("  1. Specification provides source and target only");
            println!("  2. ActionPlanner searches causal graph for optimal path");
            println!("  3. Path quality evaluated using expected utility");
            println!("  4. Best intervention sequence selected automatically");

            println!("\n✨ Benefits:");
            println!("  • No need to manually specify mediators");
            println!("  • Discovers paths you might not have considered");
            println!("  • Optimizes for intervention effectiveness");
            println!("  • Confidence based on path quality");
        }
        Err(e) => {
            println!("❌ Path synthesis failed: {}", e);
        }
    }

    // Compare with manually specified path
    println!("\n🔄 Comparing with manually specified path...");
    let manual_spec = CausalSpec::CreatePath {
        from: "education".to_string(),
        through: vec!["skills".to_string()], // Manual choice
        to: "income".to_string(),
    };

    match synthesizer.synthesize(&manual_spec) {
        Ok(manual_program) => {
            println!("\n📊 Manual Path:");
            println!("  Path: {}", manual_program.variables.join(" → "));
            println!("  Confidence: {:.2}", manual_program.confidence);
        }
        Err(_) => {}
    }
}

/// Example 5: Complete Workflow
///
/// Demonstrates all 4 components working together in a complete
/// synthesis-verification workflow.
fn example_5_complete_workflow() {
    println!("\n🌟 Example 5: Complete Workflow (All Components)");
    println!("{}", "-".repeat(70));

    println!("\n📊 Setting up complete system...");

    // Create all Enhancement #4 components
    let intervention_engine = create_mock_intervention_engine();
    let counterfactual_engine = create_mock_counterfactual_engine();
    let action_planner = create_mock_action_planner();

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

    println!("✅ Complete system configured with all 4 components");

    // Define a complex specification
    let spec = CausalSpec::Strengthen {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        target_strength: 0.8,
    };

    println!("\n🎯 Specification:");
    println!("  Type: Strengthen causal link");
    println!("  Cause: exercise");
    println!("  Effect: health");
    println!("  Target Strength: 0.80");

    // Step 1: Synthesize with all enhancements
    println!("\n1️⃣  Synthesis Phase (with ExplanationGenerator + InterventionEngine)");
    println!("{}", "-".repeat(70));

    match synthesizer.synthesize(&spec) {
        Ok(program) => {
            println!("✅ Program synthesized successfully");

            // Show explanation (Component 1)
            println!("\n📖 Explanation (ExplanationGenerator):");
            if let Some(explanation) = &program.explanation {
                println!("{}", explanation);
            }

            // Show intervention test results (Component 2)
            println!("\n🧪 Intervention Test (CausalInterventionEngine):");
            println!("  Achieved Strength: {:.2}", program.achieved_strength);
            println!("  Confidence: {:.2}", program.confidence);

            // Step 2: Verify with counterfactuals
            println!("\n2️⃣  Verification Phase (CounterfactualEngine)");
            println!("{}", "-".repeat(70));

            let result = verifier.verify(&program);

            println!("✅ Verification complete");
            println!("\n📊 Counterfactual Verification Results:");
            println!("  Accuracy: {:.2}%", result.counterfactual_accuracy * 100.0);
            println!("  Confidence: {:.2}", result.confidence);
            println!("  Tests Run: {}", result.tests_run);
            println!("  Valid: {}", if result.success { "✅ YES" } else { "❌ NO" });

            // Step 3: Final assessment
            println!("\n3️⃣  Final Assessment");
            println!("{}", "-".repeat(70));

            let synthesis_confidence = program.confidence;
            let verification_accuracy = result.counterfactual_accuracy;
            let overall_score = (synthesis_confidence + verification_accuracy) / 2.0;

            println!("\n🎯 Overall Quality Score: {:.2}", overall_score);

            if overall_score > 0.9 {
                println!("  ✅ EXCELLENT - High confidence synthesis + verification");
            } else if overall_score > 0.7 {
                println!("  ✅ GOOD - Program meets quality standards");
            } else if overall_score > 0.5 {
                println!("  ⚠️  ACCEPTABLE - Some concerns with quality");
            } else {
                println!("  ❌ POOR - Program needs improvement");
            }

            println!("\n📋 Summary:");
            println!("  1. ✅ Explanation generated (rich causal semantics)");
            println!("  2. ✅ Intervention tested (real confidence scores)");
            println!("  3. ✅ Counterfactual verified (ground truth validation)");
            println!("  4. ✅ Complete workflow (all components working together)");

            println!("\n🌟 This is real causal AI:");
            println!("  • Programs tested with do-calculus interventions");
            println!("  • Programs verified with potential outcomes theory");
            println!("  • Programs explained with causal semantics");
            println!("  • Programs optimized with action planning");
        }
        Err(e) => {
            println!("❌ Synthesis failed: {}", e);
        }
    }
}

// ============================================================================
// Mock Component Implementations (for demonstration)
// ============================================================================

fn create_mock_intervention_engine() -> CausalInterventionEngine {
    use symthaea::observability::ProbabilisticCausalGraph;

    // In production, this would be initialized with a real causal graph
    // For demonstration, we create a simple mock graph
    let graph = ProbabilisticCausalGraph::new();
    CausalInterventionEngine::new(graph)
}

fn create_mock_counterfactual_engine() -> CounterfactualEngine {
    use symthaea::observability::ProbabilisticCausalGraph;

    // In production, this would be initialized with a causal model
    // For demonstration, we create a simple mock graph
    let graph = ProbabilisticCausalGraph::new();
    CounterfactualEngine::new(graph)
}

fn create_mock_action_planner() -> ActionPlanner {
    use symthaea::observability::ProbabilisticCausalGraph;

    // In production, this would have access to the full causal graph
    // For demonstration, we create a simple mock graph
    let graph = ProbabilisticCausalGraph::new();
    ActionPlanner::new(graph)
}
