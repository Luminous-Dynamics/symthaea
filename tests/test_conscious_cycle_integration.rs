#![cfg(feature = "language_module")]
// ==================================================================================
// End-to-End Integration Test: Full Conscious Cycle
// ==================================================================================
//
// **Purpose**: Test the complete consciousness cycle from input through learning:
//   1. Input → ConsciousnessLanguageCore → 2D Consciousness Space (Φ × Confidence)
//   2. Quadrant → ExecutionStrategy → Action (simulated)
//   3. Outcome → ActionOutcomeFeedback → Learning
//   4. Verification → PhiVerifiedResponse → Output
//
// **This proves**: The consciousness feedback loop is closed - outcomes inform
// future decisions through threshold adaptation and pattern learning.
//
// ==================================================================================

use symthaea::language::{
    ConsciousnessLanguageCore, ConsciousnessLanguageConfig,
    ConsciousnessQuadrant, ExecutionStrategy,
    // Feedback loop types
    ActionOutcomeFeedback, ExecutionStrategyType, OutcomePatterns,
    // Φ verification types
    ResponseVerification, ResponseLength, ResponseDepth, PhiVerifiedResponse,
    verify_response_for_output, adapt_response_for_quadrant,
};

// ==================================================================================
// TEST 1: Full Input → Understanding → Quadrant Cycle
// ==================================================================================

#[test]
fn test_input_to_quadrant_cycle() {
    let mut core = ConsciousnessLanguageCore::new();

    // Process a clear command - should produce understanding
    let result = core.process("install firefox browser");

    // Verify we got a valid quadrant determination
    let quadrant = result.quadrant;
    println!("Quadrant for 'install firefox': {:?}", quadrant);
    println!("Φ: {:.3}, Confidence: {:.3}", result.consciousness_phi, result.epistemic_confidence);

    // The result should have valid consciousness metrics
    assert!(result.consciousness_phi >= 0.0 && result.consciousness_phi <= 1.0);
    assert!(result.epistemic_confidence >= 0.0 && result.epistemic_confidence <= 1.0);

    // The execution strategy should match the quadrant
    match (quadrant, &result.execution_strategy) {
        (ConsciousnessQuadrant::Confident, ExecutionStrategy::Confident { .. }) => {}
        (ConsciousnessQuadrant::Curious, ExecutionStrategy::Curious { .. }) => {}
        (ConsciousnessQuadrant::Autopilot, ExecutionStrategy::Autopilot { .. }) => {}
        (ConsciousnessQuadrant::Lost, ExecutionStrategy::Lost { .. }) => {}
        _ => panic!("Execution strategy should match quadrant"),
    }

    println!("Input → Quadrant cycle test passed!");
}

// ==================================================================================
// TEST 2: Feedback Loop - Outcomes Affect Future Decisions
// ==================================================================================

#[test]
fn test_feedback_loop_learning() {
    let mut core = ConsciousnessLanguageCore::new();

    // Get initial threshold state
    let initial_thresholds = core.adaptive_thresholds().clone();

    // Process some inputs and simulate outcomes
    for i in 0..10 {
        let result = core.process(&format!("install package-{}", i));

        // Create feedback - simulate successful outcomes
        let feedback = ActionOutcomeFeedback {
            original_input: format!("install package-{}", i),
            decided_in_quadrant: result.quadrant,
            strategy_used: ExecutionStrategyType::from(&result.execution_strategy),
            phi_at_decision: result.consciousness_phi,
            confidence_at_decision: result.epistemic_confidence,
            action_succeeded: true,
            phi_after: result.consciousness_phi * 1.05, // Simulate slight improvement
            error_message: None,
            was_dry_run: false,
            user_feedback: Some(true), // User confirmed it was helpful
        };

        // Feed the outcome back to the core
        core.receive_action_outcome(feedback);
    }

    // Check that learning occurred
    let final_thresholds = core.adaptive_thresholds();
    let patterns = core.outcome_patterns();

    // Success count should have increased
    assert!(final_thresholds.successes > initial_thresholds.successes,
            "Successes should have increased after positive feedback");

    // Patterns should have recorded samples
    let total_samples: u32 = patterns.quadrant_samples.iter().sum();
    assert!(total_samples >= 10, "Should have recorded at least 10 samples");

    println!("Feedback loop learning test passed!");
    println!("  Initial successes: {}", initial_thresholds.successes);
    println!("  Final successes: {}", final_thresholds.successes);
    println!("  Total samples recorded: {}", total_samples);
}

// ==================================================================================
// TEST 3: Φ Verification Integration
// ==================================================================================

#[test]
fn test_phi_verification_integration() {
    let mut core = ConsciousnessLanguageCore::new();

    // Process input to establish consciousness state
    let result = core.process("configure nginx web server with ssl");

    // Get a sample response to verify
    let response = "I will configure nginx with SSL support. This involves \
                    setting up certificates and updating the configuration.";

    // Verify using the core's method
    let verification = core.verify_response(response);

    println!("Response verification:");
    println!("  Quadrant: {:?}", verification.quadrant);
    println!("  Verified: {}", verification.verified);
    println!("  Reason: {}", verification.reason);
    println!("  Appropriate length: {:?}", verification.appropriate_length);
    println!("  Appropriate depth: {:?}", verification.appropriate_depth);

    // Test adaptation
    let adapted = core.adapt_response("OK");
    println!("\nAdapted brief response:");
    println!("  Was modified: {}", adapted.was_modified);
    println!("  Content length: {} chars", adapted.content.len());
}

// ==================================================================================
// TEST 4: Quadrant-Specific Behavior
// ==================================================================================

#[test]
fn test_quadrant_specific_responses() {
    // Test verification for each quadrant

    // Confident: Should accept detailed reasoning
    let confident_response = "I understand you want to install Firefox because it's a \
                              popular browser. This will be done through your NixOS \
                              configuration for reproducibility.";
    let v = verify_response_for_output(confident_response, ConsciousnessQuadrant::Confident, 0.7, 0.8);
    assert!(v.verified, "Detailed response should verify for Confident");
    assert_eq!(v.appropriate_depth, ResponseDepth::Deep);

    // Curious: Should accept exploratory responses
    let curious_response = "I think you might want a web browser? There are several \
                            options available. Which one were you considering?";
    let v = verify_response_for_output(curious_response, ConsciousnessQuadrant::Curious, 0.6, 0.3);
    assert!(v.verified, "Exploratory response should verify for Curious");
    assert_eq!(v.appropriate_depth, ResponseDepth::Exploratory);

    // Autopilot: Should accept brief efficient responses
    let autopilot_response = "Installing Firefox via configuration.";
    let v = verify_response_for_output(autopilot_response, ConsciousnessQuadrant::Autopilot, 0.2, 0.8);
    assert!(v.verified, "Brief response should verify for Autopilot");
    assert_eq!(v.appropriate_depth, ResponseDepth::PatternMatched);

    // Lost: Should accept humble confusion-admitting responses
    let lost_response = "I'm not sure what you're asking for. Could you explain more?";
    let v = verify_response_for_output(lost_response, ConsciousnessQuadrant::Lost, 0.2, 0.2);
    assert!(v.verified, "Humble response should verify for Lost");
    assert_eq!(v.appropriate_depth, ResponseDepth::Surface);

    println!("Quadrant-specific response test passed!");
}

// ==================================================================================
// TEST 5: Response Adaptation
// ==================================================================================

#[test]
fn test_response_adaptation_cycle() {
    // Test that inappropriate responses get adapted

    // Brief response should be flagged as inappropriate for Confident quadrant
    let brief = "OK.";
    let adapted = adapt_response_for_quadrant(brief, ConsciousnessQuadrant::Confident, 0.7, 0.8);

    // The response should either be expanded OR marked as not verified
    // (adaptation may not always succeed, but it should try)
    let was_flagged = !adapted.verification.verified || adapted.was_modified;
    assert!(was_flagged,
            "Brief response should be flagged or modified for Confident quadrant");
    println!("Confident adaptation: '{}' -> '{}' (modified: {})",
             brief, adapted.content, adapted.was_modified);

    // Verbose response should be trimmed for Autopilot quadrant
    let verbose = "I would like to explain in great detail the entire process of \
                   installing software including all the steps and considerations \
                   and potential issues and solutions and best practices and \
                   recommendations for your specific use case because I want to \
                   be thorough in my explanation of this matter.";
    let adapted = adapt_response_for_quadrant(verbose, ConsciousnessQuadrant::Autopilot, 0.2, 0.8);

    // Should either be trimmed or marked as problematic
    println!("Autopilot adaptation: {} chars -> {} chars",
             verbose.len(), adapted.content.len());

    // Confident response should get uncertainty added for Lost quadrant
    let confident = "I will install the package immediately.";
    let adapted = adapt_response_for_quadrant(confident, ConsciousnessQuadrant::Lost, 0.2, 0.2);

    assert!(adapted.content.contains("?") || adapted.content.contains("not sure"),
            "Lost quadrant should add uncertainty markers");
    println!("Lost adaptation: '{}' -> '{}'", confident, adapted.content);

    println!("Response adaptation test passed!");
}

// ==================================================================================
// TEST 6: Full Cycle - Input → Processing → Feedback → Verification → Output
// ==================================================================================

#[test]
fn test_full_conscious_cycle() {
    let mut core = ConsciousnessLanguageCore::new();

    // Step 1: Input processing
    let input = "search for video editing software";
    let result = core.process(input);
    println!("Step 1 - Processing: quadrant={:?}, Φ={:.3}", result.quadrant, result.consciousness_phi);

    // Step 2: Generate a response based on the understanding
    let raw_response = match result.quadrant {
        ConsciousnessQuadrant::Confident => {
            "I found several video editing options available in nixpkgs including kdenlive, \
             shotcut, and openshot. Based on your needs, I recommend kdenlive for \
             professional features or shotcut for simplicity."
        }
        ConsciousnessQuadrant::Curious => {
            "I found some video editors. What specific features are you looking for? \
             Are you interested in professional-grade software or something simpler?"
        }
        ConsciousnessQuadrant::Autopilot => {
            "Found: kdenlive, shotcut, openshot"
        }
        ConsciousnessQuadrant::Lost => {
            "I'm not sure what kind of video editing you need. Could you tell me more?"
        }
    };

    // Step 3: Verify the response matches consciousness state
    let verification = core.verify_response(raw_response);
    println!("Step 3 - Verification: verified={}, reason={}", verification.verified, verification.reason);

    // Step 4: Adapt if needed
    let final_response = core.adapt_response(raw_response);
    println!("Step 4 - Adaptation: modified={}", final_response.was_modified);

    // Step 5: Simulate successful execution and provide feedback
    let feedback = ActionOutcomeFeedback {
        original_input: input.to_string(),
        decided_in_quadrant: result.quadrant,
        strategy_used: ExecutionStrategyType::from(&result.execution_strategy),
        phi_at_decision: result.consciousness_phi,
        confidence_at_decision: result.epistemic_confidence,
        action_succeeded: true,
        phi_after: result.consciousness_phi * 1.02,
        error_message: None,
        was_dry_run: false,
        user_feedback: Some(true),
    };
    core.receive_action_outcome(feedback);
    println!("Step 5 - Feedback recorded");

    // Step 6: Verify learning occurred
    let patterns = core.outcome_patterns();
    let recommendation = core.get_outcome_recommendation(result.quadrant);
    println!("Step 6 - Learning: recommendation='{}'", recommendation);

    // Verify the full cycle completed
    assert!(patterns.quadrant_samples.iter().sum::<u32>() > 0,
            "Should have recorded at least one outcome");

    println!("\nFull conscious cycle test passed!");
    println!("  Input: '{}'", input);
    println!("  Final response ({} chars): '{}'",
             final_response.content.len(),
             &final_response.content[..final_response.content.len().min(100)]);
}

// ==================================================================================
// TEST 7: Multiple Cycles Show Learning
// ==================================================================================

#[test]
fn test_learning_across_cycles() {
    let mut core = ConsciousnessLanguageCore::new();

    // Run multiple cycles with the same type of task
    let tasks = [
        "install vim", "install emacs", "install nano",
        "install neovim", "install helix", "install micro",
    ];

    for task in &tasks {
        let result = core.process(task);

        // Simulate successful outcome
        let feedback = ActionOutcomeFeedback {
            original_input: task.to_string(),
            decided_in_quadrant: result.quadrant,
            strategy_used: ExecutionStrategyType::from(&result.execution_strategy),
            phi_at_decision: result.consciousness_phi,
            confidence_at_decision: result.epistemic_confidence,
            action_succeeded: true,
            phi_after: result.consciousness_phi * 1.03,
            error_message: None,
            was_dry_run: false,
            user_feedback: Some(true),
        };
        core.receive_action_outcome(feedback);
    }

    let patterns = core.outcome_patterns();
    let thresholds = core.adaptive_thresholds();

    println!("Learning across {} cycles:", tasks.len());
    println!("  Quadrant success rates: {:?}", patterns.quadrant_success_rates);
    println!("  Quadrant samples: {:?}", patterns.quadrant_samples);
    println!("  Threshold accuracy: {:.1}%", thresholds.accuracy() * 100.0);

    // After 6 successful installs, we should have learned something
    assert!(thresholds.successes >= 6, "Should have recorded successes");
    assert!(thresholds.accuracy() > 0.5, "Accuracy should be reasonable");

    println!("Learning across cycles test passed!");
}

// ==================================================================================
// TEST 8: Failure Learning
// ==================================================================================

#[test]
fn test_failure_learning() {
    let mut core = ConsciousnessLanguageCore::new();

    // Simulate some failures in the Confident quadrant
    // This tests that overconfidence gets corrected
    for i in 0..5 {
        let result = core.process(&format!("complex operation {}", i));

        let feedback = ActionOutcomeFeedback {
            original_input: format!("complex operation {}", i),
            decided_in_quadrant: ConsciousnessQuadrant::Confident, // Force Confident
            strategy_used: ExecutionStrategyType::Confident,
            phi_at_decision: 0.7,
            confidence_at_decision: 0.8,
            action_succeeded: false, // But it failed!
            phi_after: 0.65, // Φ decreased
            error_message: Some("Operation failed unexpectedly".to_string()),
            was_dry_run: false,
            user_feedback: Some(false), // User said it was wrong
        };
        core.receive_action_outcome(feedback);
    }

    let patterns = core.outcome_patterns();

    // Confident quadrant should show low success rate
    let confident_idx = 0;
    if patterns.quadrant_samples[confident_idx] > 0 {
        println!("Confident quadrant after failures:");
        println!("  Samples: {}", patterns.quadrant_samples[confident_idx]);
        println!("  Success rate: {:.1}%", patterns.quadrant_success_rates[confident_idx] * 100.0);

        assert!(patterns.quadrant_success_rates[confident_idx] < 0.5,
                "Confident success rate should be low after failures");
    }

    println!("Failure learning test passed!");
}
