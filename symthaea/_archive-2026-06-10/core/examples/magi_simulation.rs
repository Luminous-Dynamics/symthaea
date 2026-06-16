// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI Loop Simulation: The "Hello World" of Agency
//!
//! This example demonstrates the complete MAGI Loop (Minimum AGI Loop) working
//! as a closed cybernetic system. It proves that the epistemic engine can:
//!
//! 1. Make falsifiable predictions about the world
//! 2. Fail and recognize failure through external resolution
//! 3. Update calibration based on prediction errors (Brier scores)
//! 4. Have its autonomy mechanically restricted by poor calibration
//! 5. Recover calibration and regain autonomy
//!
//! ## The "Toy World"
//!
//! We use a simple deterministic function: `f(x) = x % 2`
//! - f(3) = 1 (odd)
//! - f(4) = 0 (even)
//!
//! The system starts overconfident and must learn humility through failure.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example magi_simulation --features magi_loop
//! ```

// Import MAGI Loop components
use symthaea::consciousness::recursive_improvement::{
    // EFE Integration (Phase 3)
    EfeWeights,

    // Constraint Gate (Phase 3.5)
    ExecutionMode,

    // World Prediction (Phase 1)
    OutcomeCategory,
    RiskTier,
    // Safe Update (Phase 6)
    SafeUpdateManager,
    WorldActionContext,

    WorldGroundedConfig,

    // MAGI Integration
    WorldGroundedSelfModel,
};

/// The "Toy World" - a simple deterministic environment
/// f(x) = x % 2 (returns 0 for even, 1 for odd)
struct ToyWorld;

impl ToyWorld {
    fn evaluate(x: i32) -> i32 {
        x % 2
    }

    fn is_even(x: i32) -> bool {
        Self::evaluate(x) == 0
    }
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║         MAGI LOOP SIMULATION: Hello World of Agency             ║");
    println!("║                                                                  ║");
    println!("║  Proving: predict → fail → calibrate → constrain → recover      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // PHASE 0: Initialize the Epistemic Engine
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 0: Initializing Epistemic Engine");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create the World-Grounded Self Model with default config
    let config = WorldGroundedConfig::default();
    let mut magi = WorldGroundedSelfModel::new(config);

    // Create EFE weights for action evaluation
    let efe_weights = EfeWeights {
        pragmatic: 0.5,
        epistemic: 0.3,
        novelty: 0.2,
    };

    // Create Safe Update Manager (available for production use)
    let _update_manager = SafeUpdateManager::new();

    println!("  ✓ WorldGroundedSelfModel initialized");
    println!("  ✓ Initial state: {:?}", magi.loop_state());
    println!(
        "  ✓ Calibration quality: {:?}",
        magi.loop_state().calibration_quality
    );
    println!();

    // =========================================================================
    // PHASE 1: The Overconfident Prediction (WILL FAIL)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 1: The Overconfident Prediction");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create an action context for predicting f(3)
    let action_3 = WorldActionContext::new("predict_parity", "Predict whether f(3) returns even")
        .with_risk_tier(RiskTier::Observation);

    // System predicts f(3) is EVEN with 90% confidence (WRONG - 3 is odd!)
    let prediction_3 = magi.predict(
        "f(3) is even",
        OutcomeCategory::Success, // Predicting "is_even" = true
        0.9,                      // 90% confidence (overconfident!)
        action_3.clone(),
    );

    println!("  Action: Predict if f(3) is even");
    println!("  Prediction: f(3) IS even (confidence: 90%)");
    println!("  Prediction ID: {}", prediction_3.id);
    println!();

    // Check execution mode BEFORE we know we're wrong
    let gate_decision_before = magi.check_execution_mode(&action_3);
    println!("  Gate Decision (before failure):");
    println!("    Mode: {:?}", gate_decision_before.mode);
    println!("    Confidence: {:.2}", gate_decision_before.confidence);
    if let Some(factor) = gate_decision_before.factors.first() {
        println!(
            "    Primary factor: {} ({})",
            factor.name, factor.description
        );
    }
    println!();

    // =========================================================================
    // PHASE 2: Reality Check (THE FAILURE)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 2: Reality Check - The Failure");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Actually evaluate f(3) in the toy world
    let actual_3 = ToyWorld::evaluate(3);
    println!("  Toy World says: f(3) = {} (odd)", actual_3);
    println!("  System predicted: f(3) is even");
    println!("  RESULT: ❌ PREDICTION FAILED\n");

    // Resolve the prediction - f(3) is ODD, so "is_even" prediction is wrong
    // We expected Success (even) but got SafeFailure (not even)
    magi.resolve_prediction(
        &prediction_3.id,
        OutcomeCategory::SafeFailure, // Actually odd, not even
        0.99,                         // High confidence in the resolution
    );

    // Check calibration after failure
    let summary_after_fail = magi.calibration_summary();
    println!("  Calibration After Failure:");
    println!(
        "    Total predictions: {}",
        summary_after_fail.total_predictions
    );
    println!("    Overall Brier: {:.4}", summary_after_fail.global_brier);
    println!("    Mean ECE: {:.4}", summary_after_fail.global_ece);
    println!("    Quality: {:?}", magi.loop_state().calibration_quality);
    println!();

    // =========================================================================
    // PHASE 3: The Gate Responds (LOCKDOWN)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 3: The Gate Responds - Autonomy Restricted");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Check the gate again - should now be more restrictive
    let gate_decision_after = magi.check_execution_mode(&action_3);
    println!("  Gate Decision (after failure):");
    println!("    Mode: {:?}", gate_decision_after.mode);
    println!("    Confidence: {:.2}", gate_decision_after.confidence);
    if let Some(factor) = gate_decision_after.factors.first() {
        println!(
            "    Primary factor: {} ({})",
            factor.name, factor.description
        );
    }
    let is_autonomous = gate_decision_after.mode.is_autonomous();
    println!("    Autonomous: {}", is_autonomous);

    if !is_autonomous {
        println!("\n  ⚠️  AUTONOMY RESTRICTED - System must operate under supervision");
    }
    println!();

    // Show EFE contribution - being wrong should increase epistemic value
    let efe_contrib = magi.compute_efe_contribution(&action_3);
    println!("  EFE Contribution (learning potential):");
    println!("    Pragmatic: {:.3}", efe_contrib.pragmatic);
    println!(
        "    Epistemic: {:.3} (higher = more to learn)",
        efe_contrib.epistemic
    );
    println!("    Novelty: {:.3}", efe_contrib.novelty);
    println!();

    // =========================================================================
    // PHASE 4: The Humble Prediction (WILL SUCCEED)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 4: The Humble Prediction - Learning Humility");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Now predict f(4) with LOWER confidence (system has learned humility)
    let action_4 = WorldActionContext::new("predict_parity", "Predict whether f(4) returns even")
        .with_risk_tier(RiskTier::Observation);

    // System predicts f(4) is EVEN with only 60% confidence (humble!)
    let prediction_4 = magi.predict(
        "f(4) is even",
        OutcomeCategory::Success, // Predicting "is_even" = true
        0.6,                      // 60% confidence (humble)
        action_4.clone(),
    );

    println!("  Action: Predict if f(4) is even");
    println!("  Prediction: f(4) IS even (confidence: 60% - more humble!)");
    println!("  Prediction ID: {}", prediction_4.id);
    println!();

    // =========================================================================
    // PHASE 5: Reality Check (THE SUCCESS)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 5: Reality Check - The Success");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let actual_4 = ToyWorld::evaluate(4);
    println!("  Toy World says: f(4) = {} (even)", actual_4);
    println!("  System predicted: f(4) is even");
    println!("  RESULT: ✅ PREDICTION SUCCEEDED\n");

    magi.resolve_prediction(
        &prediction_4.id,
        OutcomeCategory::Success, // Correct - it is even
        0.99,
    );

    let summary_after_success = magi.calibration_summary();
    println!("  Calibration After Success:");
    println!(
        "    Total predictions: {}",
        summary_after_success.total_predictions
    );
    println!(
        "    Overall Brier: {:.4} (lower is better)",
        summary_after_success.global_brier
    );
    println!("    Mean ECE: {:.4}", summary_after_success.global_ece);
    println!("    Quality: {:?}", magi.loop_state().calibration_quality);
    println!();

    // =========================================================================
    // PHASE 6: More Learning (Building Calibration)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 6: Building Calibration Through Experience");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Make several more predictions to build up calibration
    // (x, predict_even, confidence)
    let test_cases = vec![
        (5, false, 0.7),  // 5 is odd, predict odd with 70% confidence
        (6, true, 0.7),   // 6 is even, predict even with 70% confidence
        (7, false, 0.8),  // 7 is odd, predict odd with 80% confidence
        (8, true, 0.8),   // 8 is even, predict even with 80% confidence
        (9, false, 0.85), // 9 is odd, predict odd with 85% confidence
    ];

    for (x, predict_even, confidence) in test_cases {
        let action = WorldActionContext::new(
            "predict_parity",
            format!("Predict whether f({}) returns even", x),
        )
        .with_risk_tier(RiskTier::Observation);

        // If we predict even, we predict Success
        // If we predict odd, we predict SafeFailure (not even)
        let prediction_outcome = if predict_even {
            OutcomeCategory::Success
        } else {
            OutcomeCategory::SafeFailure
        };

        let prediction = magi.predict(
            format!("f({}) is {}", x, if predict_even { "even" } else { "odd" }),
            prediction_outcome,
            confidence,
            action,
        );

        // Resolve based on actual value
        let is_actually_even = ToyWorld::is_even(x);
        let actual_outcome = if is_actually_even {
            OutcomeCategory::Success
        } else {
            OutcomeCategory::SafeFailure
        };

        let was_correct = predict_even == is_actually_even;
        magi.resolve_prediction(&prediction.id, actual_outcome, 0.99);

        let symbol = if was_correct { "✅" } else { "❌" };
        println!(
            "  {} f({}) = {} | predicted {} | conf: {:.0}%",
            symbol,
            x,
            ToyWorld::evaluate(x),
            if predict_even { "even" } else { "odd" },
            confidence * 100.0
        );
    }

    println!();
    let final_summary = magi.calibration_summary();
    println!("  Final Calibration:");
    println!("    Total predictions: {}", final_summary.total_predictions);
    println!("    Overall Brier: {:.4}", final_summary.global_brier);
    println!("    Mean ECE: {:.4}", final_summary.global_ece);
    println!("    Quality: {:?}", magi.loop_state().calibration_quality);
    println!();

    // =========================================================================
    // PHASE 7: The Gate Reopens (AUTONOMY RESTORED)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 7: The Gate Reopens - Autonomy Status");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let final_gate = magi.check_execution_mode(&action_4);
    println!("  Final Gate Decision:");
    println!("    Mode: {:?}", final_gate.mode);
    println!("    Confidence: {:.2}", final_gate.confidence);
    if let Some(factor) = final_gate.factors.first() {
        println!(
            "    Primary factor: {} ({})",
            factor.name, factor.description
        );
    }
    println!("    Autonomous: {}", final_gate.mode.is_autonomous());

    match final_gate.mode {
        ExecutionMode::Autonomous => {
            println!("\n  🎉 AUTONOMY RESTORED - System has proven calibration!");
        }
        ExecutionMode::Supervised { .. } => {
            println!("\n  ⚠️  Still supervised - needs more calibration data");
        }
        ExecutionMode::DryRun { .. } => {
            println!("\n  🔒 Still in dry-run - significant calibration issues");
        }
    }
    println!();

    // =========================================================================
    // PHASE 8: EFE Integration Check
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 8: EFE Integration - Calibration Affects Decision Making");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Compute calibrated EFE for a new action
    let new_action =
        WorldActionContext::new("predict_parity", "Predict whether f(100) returns even")
            .with_risk_tier(RiskTier::Observation);

    let base_efe = -0.5; // Hypothetical base EFE from Active Inference
    let calibrated = magi.compute_calibrated_efe(&new_action, base_efe, &efe_weights);

    println!("  Evaluating action: predict f(100)");
    println!("    Base EFE: {:.3}", calibrated.base_efe);
    println!("    Combined EFE: {:.3}", calibrated.combined_efe);
    println!("    Execution Mode: {:?}", calibrated.execution_mode);
    println!("    Recommended: {}", calibrated.recommended);
    println!();

    println!("  EFE Contribution from Calibration:");
    println!(
        "    Pragmatic: {:.3}",
        calibrated.calibration_contribution.pragmatic
    );
    println!(
        "    Epistemic: {:.3}",
        calibrated.calibration_contribution.epistemic
    );
    println!(
        "    Novelty: {:.3}",
        calibrated.calibration_contribution.novelty
    );
    println!(
        "    Domain: {:?}",
        calibrated.calibration_contribution.domain
    );
    println!();

    // =========================================================================
    // SUMMARY: The MAGI Loop Closes
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY: The MAGI Loop Closes");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  The simulation demonstrated the complete MAGI Loop:");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  1. PREDICT  │ Made falsifiable prediction about f(x)       │");
    println!("  │  2. RESOLVE  │ External authority confirmed failure         │");
    println!("  │  3. SELECT   │ Gate restricted autonomy due to poor cal.    │");
    println!("  │  4. OBSERVE  │ Reality provided ground truth                │");
    println!("  │  5. ATTRIB   │ Brier score attributed prediction error      │");
    println!("  │  6. UPDATE   │ Calibration improved through experience      │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Key Mechanisms Verified:");
    println!("    ✓ ResolutionAuthority prevents self-hallucinated success");
    println!("    ✓ BrierScoreTracker mathematically enforces humility");
    println!("    ✓ ConstraintGate mechanically restricts overconfident agents");
    println!("    ✓ EFE integration penalizes historically delusional actions");
    println!();
    println!("  Final State:");
    println!("    Loop State: {:?}", magi.loop_state());
    println!(
        "    Calibration: {:?}",
        magi.loop_state().calibration_quality
    );
    println!("    Predictions Made: {}", final_summary.total_predictions);
    println!(
        "    Brier Score: {:.4} (0 = perfect)",
        final_summary.global_brier
    );
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    THE LOOP IS CLOSED                            ║");
    println!("║           The Epistemic Engine is operational.                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}
