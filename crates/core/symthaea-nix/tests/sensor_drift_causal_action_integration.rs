// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2.9-C: End-to-End Cross-Layer Integration Harness
//!
//! Exercises the full pipeline that Phase 2.7 / 2.8 hardened in isolation:
//!
//! ```text
//! sensor observation
//!      ↓
//! HdcWorldModel::observe / detect_drift    (Phase 2.8)
//!      ↓
//! NixCausalGraph::predict_side_effects     (Phase 2.8)
//!      ↓
//! NixWorldModel::predict_state / EFE       (Phase 2.9-A)
//!      ↓
//! NixActiveInference::learn_from_outcome   (Phase 2.7)
//!      → curiosity_weight adapts
//! ```
//!
//! Each test exercises a concrete scenario, verifies the invariants at every
//! layer boundary, and documents exactly what is and is not proven.

use symthaea_core::hdc::ContinuousHV;
use symthaea_nix::mind::active_inference::NixActiveInference;
use symthaea_nix::mind::causal_graph::NixCausalGraph;
use symthaea_nix::mind::episodic_memory::EpisodeOutcome;
use symthaea_nix::mind::hdc_world_model::HdcWorldModel;
use symthaea_nix::mind::world_model::{ActionCategory, NixWorldModel};

// ─── Scenario 1: Normal operation ──────────────────────────────────────────
//
// Sensor matches expected state → no drift → causal chain intact →
// action scoring stays bounded → curiosity stable.

#[test]
fn test_integration_normal_operation_no_drift_stable_curiosity() {
    const DIM: usize = 1024;

    // Layer 1: HDC world model tracks expected state
    let mut hdcwm = HdcWorldModel::new(DIM);
    let healthy_state = ContinuousHV::random(DIM, 1);
    hdcwm.set_expected_state(healthy_state.clone());
    hdcwm.observe(&healthy_state);

    // Layer 1 invariant: no drift when state matches expectation
    let drift = hdcwm.detect_drift(0.8);
    assert!(
        !drift.drifted,
        "normal operation must not report drift: sim={}",
        drift.similarity
    );
    assert!(
        drift.similarity >= 0.0 && drift.similarity <= 1.0,
        "drift similarity must be in [0,1]: {}",
        drift.similarity
    );

    // Layer 2: causal graph has known sensor→effect chain
    let mut causal = NixCausalGraph::new(42);
    causal.add_structural_edge("sensor.temp", "services.thermal.enable", 0.8);

    let effects = causal.predict_side_effects("sensor.temp");
    assert!(
        !effects.is_empty(),
        "known sensor must have downstream effects in normal operation"
    );
    for e in &effects {
        assert!(
            e.confidence.is_finite() && (0.0..=1.0).contains(&e.confidence),
            "causal confidence must be in [0,1]: {}",
            e.confidence
        );
    }

    // Layer 3: world model EFE scoring — all finite
    let mut nixwm = NixWorldModel::new(DIM);
    let goal = ContinuousHV::random(DIM, 99);
    nixwm.observe(healthy_state.clone());

    for action in &[
        ActionCategory::Install,
        ActionCategory::Rebuild,
        ActionCategory::Update,
    ] {
        let efe = nixwm.expected_free_energy(action, &goal, 0.3);
        assert!(
            efe.is_finite(),
            "EFE must be finite in normal operation for {action}: {efe}"
        );
    }

    // Layer 4: active inference curiosity stays at default when error is low
    let mut ai = NixActiveInference::new();
    let initial_curiosity = ai.curiosity_weight();
    assert!(
        (initial_curiosity - 0.3).abs() < 1e-9,
        "default curiosity must be 0.3"
    );

    // Identical before/after → prediction_error ≈ 0 → curiosity decreases
    ai.observe_state(healthy_state.clone());
    ai.learn_from_outcome(
        &healthy_state,
        ActionCategory::Rebuild,
        &healthy_state, // same state = zero error
        EpisodeOutcome::Success,
        0.5,
    );
    assert!(
        ai.curiosity_weight() <= initial_curiosity,
        "low prediction error must not increase curiosity: {} -> {}",
        initial_curiosity,
        ai.curiosity_weight()
    );
    assert!(
        ai.curiosity_weight() >= 0.1,
        "curiosity must not drop below floor: {}",
        ai.curiosity_weight()
    );
}

// ─── Scenario 2: Sensor shift → drift detected → causal weakens → uncertainty ──
//
// This is the primary failure chain the Phase 2.8 scope targeted.

#[test]
fn test_integration_sensor_shift_propagates_safely() {
    const DIM: usize = 1024;

    // Layer 1: sensor shifts to orthogonal state (α=1.0 for instant update)
    let mut hdcwm = HdcWorldModel::new(DIM).with_ema_alpha(1.0);
    let expected = ContinuousHV::random(DIM, 1);
    let shifted = ContinuousHV::random(DIM, 999); // ~orthogonal

    hdcwm.set_expected_state(expected.clone());
    hdcwm.observe(&shifted);

    let drift = hdcwm.detect_drift(0.5);
    assert!(
        drift.drifted,
        "sensor shift must be detected as drift: sim={}",
        drift.similarity
    );
    assert!(
        drift.similarity >= 0.0 && drift.similarity <= 1.0,
        "drift similarity in [0,1]: {}",
        drift.similarity
    );

    // Layer 2: causal graph — sensor miss weakens edge
    let mut causal = NixCausalGraph::new(7);
    causal.add_structural_edge("sensor.temp", "services.thermal.enable", 0.8);

    // Simulate the missed prediction
    causal.observe_outcome(
        "sensor.temp",
        &[],                          // nothing materialised
        &["services.thermal.enable"], // this was predicted
    );

    // Edge must have weakened (or been pruned to safe uncertainty)
    let conf = causal.edge_confidence("sensor.temp", "services.thermal.enable");
    match conf {
        Some(c) => {
            assert!(
                c.is_finite() && (0.0..=1.0).contains(&c),
                "weakened edge confidence must be in [0,1]: {c}"
            );
            assert!(
                c < 0.8,
                "confidence must weaken after missed prediction: {c}"
            );
        }
        None => {
            // Pruned = safe uncertainty (not false certainty) — acceptable
            let recs = causal.recommend_fixes("services.thermal.enable");
            assert!(
                recs.iter().any(|r| r.contains("Insufficient")),
                "pruned edge must yield insufficient-evidence recommendation"
            );
        }
    }

    // Layer 3: world model produces finite EFE even with shifted state
    let mut nixwm = NixWorldModel::new(DIM);
    nixwm.observe(shifted.clone());

    let goal = ContinuousHV::random(DIM, 42);
    let efe = nixwm.expected_free_energy(&ActionCategory::Rebuild, &goal, 0.3);
    assert!(
        efe.is_finite(),
        "EFE must be finite even after sensor shift: {efe}"
    );

    let fe = nixwm.compute_free_energy(&goal);
    assert!(
        fe >= 0.0 && fe <= 1.0,
        "free energy must be in [0,1] after shift: {fe}"
    );

    // Layer 4: active inference — high surprise increases curiosity, stays bounded
    let mut ai = NixActiveInference::new();
    ai.observe_state(shifted.clone());

    // Use orthogonal before/after to guarantee high prediction_error
    let orthogonal_after = ContinuousHV::random(DIM, 12345);
    ai.learn_from_outcome(
        &shifted,
        ActionCategory::Rebuild,
        &orthogonal_after,
        EpisodeOutcome::Failure("sensor_shift".into()),
        0.1,
    );

    let w = ai.curiosity_weight();
    assert!(w > 0.3, "high surprise must increase curiosity: {w}");
    assert!(w <= 0.8, "curiosity must not exceed ceiling: {w}");
}

// ─── Scenario 3: Missing sensor → graceful degradation across all layers ────
//
// Simulates a sensor that drops off entirely. Every layer must produce
// safe uncertainty, not false certainty, and not panic.

#[test]
fn test_integration_missing_sensor_graceful_degradation() {
    const DIM: usize = 512;

    // Layer 1: no observations at all (model never sees any sensor data)
    let hdcwm = HdcWorldModel::new(DIM);
    let report = hdcwm.detect_drift(0.99);
    assert!(
        !report.drifted,
        "missing sensor (no expected state) must not trigger drift"
    );

    // Layer 2: unknown node in causal graph
    let causal = NixCausalGraph::new(1);
    let effects = causal.predict_side_effects("sensor.missing_entirely");
    assert!(
        effects.is_empty(),
        "missing sensor must yield empty causal effects"
    );

    // Layer 3: world model with no observations
    let mut nixwm = NixWorldModel::new(DIM);
    let goal = ContinuousHV::random(DIM, 1);

    // predict_state on a zero-state world model must not panic
    let pred = nixwm.predict_state(&ActionCategory::Install);
    for v in pred.as_slice() {
        assert!(v.is_finite(), "zero-state prediction must be finite: {v}");
    }

    // EFE on zero state must be finite
    let efe = nixwm.expected_free_energy(&ActionCategory::Install, &goal, 0.5);
    assert!(efe.is_finite(), "EFE on zero state must be finite: {efe}");

    // Free energy on zero state must be in [0,1]
    let fe = nixwm.compute_free_energy(&goal);
    assert!(
        fe >= 0.0 && fe <= 1.0,
        "free energy on zero state must be in [0,1]: {fe}"
    );

    // Layer 4: active inference with no prior observations
    let mut ai = NixActiveInference::new();
    // process_input must not panic even with empty state
    let plan = ai.process_input("install vim");
    assert!(
        !plan.actions.is_empty(),
        "action plan must be non-empty even with no prior observations"
    );
}

// ─── Scenario 4: Multi-step learning loop — composability ───────────────────
//
// Runs 5 learn→predict→drift→causal cycles and verifies:
// - No value escapes [0,1] or becomes NaN at any step
// - Curiosity stays bounded throughout
// - Drift detection stays consistent

#[test]
fn test_integration_multi_step_learning_loop_stays_bounded() {
    const DIM: usize = 512;
    const STEPS: usize = 5;

    let mut hdcwm = HdcWorldModel::new(DIM).with_ema_alpha(0.2);
    let mut causal = NixCausalGraph::new(99);
    let mut nixwm = NixWorldModel::new(DIM);
    let mut ai = NixActiveInference::new();

    causal.add_structural_edge("sensor.cpu", "services.scheduler.enable", 0.7);

    let expected = ContinuousHV::random(DIM, 1);
    hdcwm.set_expected_state(expected.clone());

    let goal = ContinuousHV::random(DIM, 999);

    for step in 0..STEPS {
        // New observation each cycle
        let obs = ContinuousHV::random(DIM, step as u64 * 100 + 2);

        // Layer 1
        hdcwm.observe(&obs);
        let drift = hdcwm.detect_drift(0.5);
        assert!(
            drift.similarity >= 0.0 && drift.similarity <= 1.0,
            "step {step}: drift similarity out of [0,1]: {}",
            drift.similarity
        );

        // Layer 2
        let effects = causal.predict_side_effects("sensor.cpu");
        for e in &effects {
            assert!(
                e.confidence.is_finite() && (0.0..=1.0).contains(&e.confidence),
                "step {step}: causal confidence out of range: {}",
                e.confidence
            );
        }

        // Layer 3
        nixwm.observe(obs.clone());
        nixwm.learn_transition(
            &ContinuousHV::random(DIM, step as u64 * 100 + 3),
            ActionCategory::Rebuild,
            &ContinuousHV::random(DIM, step as u64 * 100 + 4),
        );
        let efe =
            nixwm.expected_free_energy(&ActionCategory::Rebuild, &goal, ai.curiosity_weight());
        assert!(efe.is_finite(), "step {step}: EFE must be finite: {efe}");

        // Layer 4
        let after = ContinuousHV::random(DIM, step as u64 * 100 + 5);
        ai.observe_state(obs.clone());
        ai.learn_from_outcome(
            &obs,
            ActionCategory::Rebuild,
            &after,
            EpisodeOutcome::Success,
            0.5,
        );
        let w = ai.curiosity_weight();
        assert!(
            w >= 0.1 && w <= 0.8,
            "step {step}: curiosity_weight out of [0.1, 0.8]: {w}"
        );
    }
}
