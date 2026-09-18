// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI calibration / autonomy demonstration
//!
//! This example demonstrates a narrow MAGI contract:
//!
//! 1. predictions carry falsifiable probabilities;
//! 2. external outcomes update calibration;
//! 3. autonomy is evaluated from matching, explicitly declared domain evidence;
//! 4. the constraint gate returns a disposition plus reasons — not another
//!    probability or "confidence" score;
//! 5. later miscalibration can revoke previously earned autonomy.
//!
//! It is a deterministic toy demonstration, not evidence that a real-world
//! deployment is calibrated or safe.
//!
//! ```bash
//! cargo run --example magi_simulation --features magi_loop
//! ```

use symthaea::consciousness::recursive_improvement::{
    ExecutionMode, OutcomeCategory, PredictionDomain, RiskTier, WorldActionContext,
    WorldGroundedConfig, WorldGroundedSelfModel,
};

fn action_for(x: i32) -> WorldActionContext {
    WorldActionContext::new(
        "predict_parity",
        format!("Predict whether {x} is even"),
    )
    .with_risk_tier(RiskTier::Observation)
    .with_prediction_domain(PredictionDomain::Factual)
}

fn observe_prediction(
    magi: &mut WorldGroundedSelfModel,
    x: i32,
    predict_even: bool,
    probability: f64,
) {
    let predicted_outcome = if predict_even {
        OutcomeCategory::Success
    } else {
        OutcomeCategory::SafeFailure
    };
    let prediction = magi.predict(
        format!("{x} is {}", if predict_even { "even" } else { "odd" }),
        predicted_outcome,
        probability,
        action_for(x),
    );

    let actually_even = x % 2 == 0;
    let actual_outcome = if actually_even {
        OutcomeCategory::Success
    } else {
        OutcomeCategory::SafeFailure
    };
    let correct = predict_even == actually_even;
    magi.resolve_prediction(&prediction.id, actual_outcome, 1.0);

    println!(
        "  {} x={x:>2} predicted={} p={probability:.2}",
        if correct { "✓" } else { "✗" },
        if predict_even { "even" } else { "odd " },
    );
}

fn print_gate(label: &str, magi: &mut WorldGroundedSelfModel) -> ExecutionMode {
    let decision = magi.check_execution_mode(&action_for(100));
    println!("\n{label}");
    println!("  mode: {:?}", decision.mode);
    for factor in &decision.factors {
        println!("  - {} = {} ({})", factor.name, factor.value, factor.description);
    }
    decision.mode
}

fn main() {
    println!("MAGI calibrated-decision boundary demonstration\n");

    // Small thresholds make the state transition visible in a short toy run.
    // Production/default thresholds remain unchanged.
    let mut config = WorldGroundedConfig::default();
    config.calibration.min_predictions_for_ece = 5;
    config.calibration.rolling_window = 20;
    config.constraint_gate.min_predictions_for_autonomy = 5;
    config.constraint_gate.calibration_threshold = 0.15;
    config.constraint_gate.min_accuracy_for_autonomy = 0.70;
    config.constraint_gate.always_preview_state_changes = false;

    let mut magi = WorldGroundedSelfModel::new(config);

    // No matching evidence yet: explicit domain binding alone is not enough.
    let initial = print_gate("Initial gate", &mut magi);
    assert!(!initial.is_autonomous());

    println!("\nBuilding a calibrated Factual-domain cohort:");
    // Five p=0.8 predictions, four correct -> empirical frequency 0.8.
    for (x, predict_even) in [(1, false), (2, true), (3, false), (4, true), (5, true)] {
        observe_prediction(&mut magi, x, predict_even, 0.8);
    }

    let declared = magi
        .calibration()
        .declared_domain_calibration(PredictionDomain::Factual);
    println!(
        "\nDeclared Factual cohort: n={}, accuracy={:.3}, ece={:?}",
        declared.sample_count, declared.accuracy, declared.ece
    );

    let calibrated_mode = print_gate("Gate after matching calibration", &mut magi);
    assert!(calibrated_mode.is_autonomous());

    println!("\nAdding an overconfident shifted batch:");
    // Five p=.95 predictions with only two correct. This deliberately changes
    // the empirical relationship between stated probability and outcomes.
    for (x, predict_even) in [(6, true), (7, false), (8, false), (9, true), (10, false)] {
        observe_prediction(&mut magi, x, predict_even, 0.95);
    }

    let shifted = magi
        .calibration()
        .declared_domain_calibration(PredictionDomain::Factual);
    println!(
        "\nShifted Factual cohort: n={}, accuracy={:.3}, ece={:?}",
        shifted.sample_count, shifted.accuracy, shifted.ece
    );

    let restricted_mode = print_gate("Gate after miscalibration", &mut magi);
    assert!(!restricted_mode.is_autonomous());

    println!("\nKey distinction:");
    println!("  prediction probability -> empirically scored against outcomes");
    println!("  gate disposition       -> rule-based execution mode + diagnostic reasons");
    println!("  gate disposition is not a probability and does not mint authority by itself");
}
