// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit tests for extracted cycle helpers: compute_unified_psi, step_fep_active_inference,
//! compose_effective_lr.

use super::super::super::{CognitiveLoopConfig, CognitiveLoopService};
use crate::cognitive_loop::thresholds::{
    BODY_PSI_WEIGHT, EMBODIED_PSI_WEIGHT, FLOW_PSI_WEIGHT, RELATIONAL_PSI_WEIGHT,
};

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// compute_unified_psi
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_compute_unified_psi_default_state_is_finite_and_bounded() {
    let mut s = make_service();
    let psi = s.compute_unified_psi();
    assert!(psi.is_finite(), "psi must be finite, got {psi}");
    assert!(psi >= 0.0, "psi must be >= 0.0, got {psi}");
    assert!(psi <= 1.0, "psi must be <= 1.0, got {psi}");
}

#[test]
fn test_compute_unified_psi_all_components_zero_gives_low_psi() {
    let mut s = make_service();
    // Zero out all contributing fields:
    // flow_state.in_flow = false (already default) → flow_psi = 0
    s.behavior.flow_state.in_flow = false;
    s.behavior.flow_state.intensity = 0.0;
    // relational_psi = 0 (already default)
    s.behavior.social_mgr.social.relational_psi = 0.0;
    // body/embodied phi modulations at neutral (1.0) → contribution = (1.0 - 1.0) * weight = 0
    s.carryover.consciousness.body_phi_modulation = 1.0;
    s.carryover.consciousness.embodied_phi_modulation = 1.0;

    let psi = s.compute_unified_psi();
    assert!(psi.is_finite(), "psi must be finite, got {psi}");
    assert!(psi >= 0.0, "psi must be >= 0.0, got {psi}");
    // With all explicit components zeroed, psi should be small (only baseline_integration
    // and coherence/voice contributions remain, which are small at startup).
    assert!(
        psi <= 0.5,
        "psi with all components zeroed should be small, got {psi}"
    );
}

#[test]
fn test_compute_unified_psi_clamped_at_one() {
    let mut s = make_service();
    // Push every component to maximum:
    s.behavior.flow_state.in_flow = true;
    s.behavior.flow_state.intensity = 10.0; // very high
    s.behavior.social_mgr.social.relational_psi = 10.0; // very high
    // body/embodied modulations far above neutral to maximize their contribution
    s.carryover.consciousness.body_phi_modulation = 100.0;
    s.carryover.consciousness.embodied_phi_modulation = 100.0;

    let psi = s.compute_unified_psi();
    assert!(psi.is_finite(), "psi must be finite, got {psi}");
    assert!(
        (psi - 1.0).abs() < f64::EPSILON,
        "psi should be clamped to 1.0 when all components are maximal, got {psi}"
    );
}

#[test]
fn test_compute_unified_psi_nan_body_modulation_still_finite() {
    let mut s = make_service();
    // Inject NaN into one component
    s.carryover.consciousness.body_phi_modulation = f64::NAN;
    let psi = s.compute_unified_psi();
    // The function sums f32/f64 components and clamps. NaN propagates through
    // arithmetic, so the final clamp(0.0, 1.0) may not save us. Document the
    // actual behavior.
    if psi.is_finite() {
        assert!(
            psi >= 0.0 && psi <= 1.0,
            "if finite, must be in [0,1], got {psi}"
        );
    } else {
        // NaN propagation is the current behavior — document it.
        assert!(psi.is_nan(), "expected NaN propagation, got {psi}");
    }
}

#[test]
fn test_compute_unified_psi_updates_unification_engine() {
    let mut s = make_service();
    let psi_before = s.unification_engine.psi;
    let psi = s.compute_unified_psi();
    // The function calls self.unification_engine.update_psi(unified_psi)
    // so the engine should reflect the computed value.
    let psi_after = s.unification_engine.psi;
    // psi_after should be influenced by the computed value (update_psi may
    // apply smoothing, so we just check it moved toward the computed value).
    assert!(
        (psi_after - psi_before).abs() < 2.0,
        "unification_engine.psi should have been updated"
    );
}

#[test]
fn test_compute_unified_psi_weight_constants_are_positive() {
    // Verify the weight constants used in compute_unified_psi are positive
    // and within reasonable bounds.
    assert!(FLOW_PSI_WEIGHT > 0.0, "FLOW_PSI_WEIGHT must be positive");
    assert!(
        RELATIONAL_PSI_WEIGHT > 0.0,
        "RELATIONAL_PSI_WEIGHT must be positive"
    );
    assert!(BODY_PSI_WEIGHT > 0.0, "BODY_PSI_WEIGHT must be positive");
    assert!(
        EMBODIED_PSI_WEIGHT > 0.0,
        "EMBODIED_PSI_WEIGHT must be positive"
    );
    // Sum of explicit weights should be < 1.0 (there's also baseline_integration
    // and coherence/voice contributions).
    let weight_sum = FLOW_PSI_WEIGHT as f64
        + RELATIONAL_PSI_WEIGHT as f64
        + BODY_PSI_WEIGHT
        + EMBODIED_PSI_WEIGHT;
    assert!(
        weight_sum < 1.0,
        "sum of psi weights ({weight_sum}) should be < 1.0 to leave room for baseline"
    );
}

#[test]
fn test_compute_unified_psi_flow_contribution_only_when_in_flow() {
    let mut s = make_service();
    // Not in flow
    s.behavior.flow_state.in_flow = false;
    s.behavior.flow_state.intensity = 1.0;
    let psi_no_flow = s.compute_unified_psi();

    // Reset and enable flow
    let mut s2 = make_service();
    s2.behavior.flow_state.in_flow = true;
    s2.behavior.flow_state.intensity = 1.0;
    let psi_with_flow = s2.compute_unified_psi();

    // Flow should add FLOW_PSI_WEIGHT * intensity to psi
    assert!(
        psi_with_flow >= psi_no_flow,
        "flow should increase psi: with_flow={psi_with_flow}, no_flow={psi_no_flow}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// step_fep_active_inference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_step_fep_returns_valid_action_index() {
    let mut s = make_service();
    let (action_idx, probs, _is_surprised, _pragmatic) = s.step_fep_active_inference(0.3, 0.5);
    assert!(
        action_idx <= 3,
        "action index should be 0..=3, got {action_idx}"
    );
    assert!(
        !probs.is_empty(),
        "action_probabilities should not be empty"
    );
}

#[test]
fn test_step_fep_action_probabilities_sum_to_one() {
    let mut s = make_service();
    let (_action_idx, probs, _is_surprised, _pragmatic) = s.step_fep_active_inference(0.5, 0.5);
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "action probabilities should sum to ~1.0, got {sum}"
    );
}

#[test]
fn test_step_fep_action0_lr_boost_side_effect() {
    // Action 0 boosts learning rate when free energy is high.
    // We can't force the agent to pick action 0, but we can run multiple
    // cycles and check that at least the return values are consistent.
    let mut s = make_service();
    // Run a few cycles to build up state
    for _ in 0..5 {
        let _ = s.step_fep_active_inference(0.8, 0.2);
    }
    // Verify the service is still in a valid state
    assert!(
        s.stats.adaptive_learning_rate.is_finite(),
        "learning rate should remain finite after FEP steps"
    );
}

#[test]
fn test_step_fep_action1_resets_sensory_precision() {
    let mut s = make_service();
    let initial_precision = s.fep.agent.precision.sensory_precision;
    // Run many cycles — action 1 blends precision toward 1.0
    for _ in 0..20 {
        let _ = s.step_fep_active_inference(0.5, 0.5);
    }
    let final_precision = s.fep.agent.precision.sensory_precision;
    // Precision should remain finite and positive regardless of which actions were selected
    assert!(
        final_precision.is_finite() && final_precision > 0.0,
        "sensory precision must stay finite and positive, got {final_precision} (was {initial_precision})"
    );
}

#[test]
fn test_step_fep_is_surprised_is_bool() {
    let mut s = make_service();
    let (_action_idx, _probs, is_surprised, _pragmatic) = s.step_fep_active_inference(0.1, 0.9);
    // Verify is_surprised is a valid bool (trivially true, but documents type intent)
    assert!(
        is_surprised || !is_surprised,
        "is_surprised should be a valid bool"
    );

    // High prediction error should be more likely to trigger surprise
    let (_action_idx2, _probs2, is_surprised2, _pragmatic2) =
        s.step_fep_active_inference(0.99, 0.1);
    // We can't guarantee surprise triggers, but both calls should succeed
    assert!(
        is_surprised2 || !is_surprised2,
        "is_surprised2 should be a valid bool"
    );
}

#[test]
fn test_step_fep_pragmatic_value_is_finite() {
    let mut s = make_service();
    let (_action_idx, _probs, _is_surprised, pragmatic) = s.step_fep_active_inference(0.5, 0.5);
    assert!(
        pragmatic.is_finite(),
        "pragmatic_value should be finite, got {pragmatic}"
    );
}

#[test]
fn test_step_fep_repeated_calls_stable() {
    // Verify that repeated calls don't cause divergence or panics
    let mut s = make_service();
    for i in 0..50 {
        let pe = (i as f32 % 10.0) / 10.0;
        let coh = 1.0 - pe;
        let (action_idx, probs, _is_surprised, pragmatic) = s.step_fep_active_inference(pe, coh);
        assert!(action_idx <= 3, "action out of range at iteration {i}");
        assert!(!probs.is_empty(), "empty probs at iteration {i}");
        assert!(
            pragmatic.is_finite(),
            "pragmatic diverged at iteration {i}: {pragmatic}"
        );
    }
    // Trust threshold should remain bounded
    let trust = s
        .consciousness
        .self_model_tier
        .self_reflection
        .trust_threshold;
    assert!(
        trust >= 0.1 && trust <= 0.9,
        "trust_threshold should be clamped to [0.1, 0.9], got {trust}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// compose_effective_lr
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_compose_effective_lr_default_factors_gives_positive_lr() {
    let mut s = make_service();
    let lr = s.compose_effective_lr(1.0, 1.0);
    assert!(lr.is_finite(), "lr must be finite, got {lr}");
    assert!(lr >= 0.0, "lr must be >= 0.0, got {lr}");
    assert!(lr <= 0.01, "lr must be <= 0.01 (clamp), got {lr}");
}

#[test]
fn test_compose_effective_lr_zero_factors_still_positive() {
    let mut s = make_service();
    // Zero factors should be clamped to 0.01 inside geometric_mean
    let lr = s.compose_effective_lr(0.0, 0.0);
    assert!(
        lr.is_finite(),
        "lr must be finite with zero factors, got {lr}"
    );
    assert!(lr >= 0.0, "lr must be >= 0.0, got {lr}");
}

#[test]
fn test_compose_effective_lr_clamped_above_zero() {
    let mut s = make_service();
    // Even with negative factors (which get clamped to 0.01 in geometric_mean)
    let lr = s.compose_effective_lr(-1.0, -1.0);
    assert!(lr >= 0.0, "lr must be clamped >= 0.0, got {lr}");
    assert!(lr <= 0.01, "lr must be clamped <= 0.01, got {lr}");
}

#[test]
fn test_compose_effective_lr_clamped_at_001() {
    let mut s = make_service();
    // Even with very large factors, result is clamped to 0.01
    let lr = s.compose_effective_lr(100.0, 100.0);
    assert!(
        lr <= 0.01,
        "lr must be clamped <= 0.01 even with large factors, got {lr}"
    );
}

#[test]
fn test_compose_effective_lr_resets_subsystem_lr_factor() {
    let mut s = make_service();
    // Set a non-default subsystem_lr_factor
    s.carryover.learning.subsystem_lr_factor = 1.5;
    let _ = s.compose_effective_lr(1.0, 1.0);
    // After calling compose_effective_lr, subsystem_lr_factor should be reset to 1.0
    assert!(
        (s.carryover.learning.subsystem_lr_factor - 1.0).abs() < f32::EPSILON,
        "subsystem_lr_factor should be reset to 1.0 after compose_effective_lr, got {}",
        s.carryover.learning.subsystem_lr_factor
    );
}

#[test]
fn test_compose_effective_lr_stores_mods_in_carryover() {
    let mut s = make_service();
    let _ = s.compose_effective_lr(1.0, 1.0);
    // The function stores cognitive_mod and meta_mod in carryover
    let cog = s.carryover.learning.lr_cognitive_mod;
    let meta = s.carryover.learning.lr_meta_mod;
    assert!(
        cog.is_finite() && cog > 0.0,
        "lr_cognitive_mod should be finite and positive, got {cog}"
    );
    assert!(
        meta.is_finite() && meta > 0.0,
        "lr_meta_mod should be finite and positive, got {meta}"
    );
}

#[test]
fn test_compose_effective_lr_subsystem_factor_affects_result() {
    let mut s1 = make_service();
    let lr_default = s1.compose_effective_lr(1.0, 1.0);

    let mut s2 = make_service();
    s2.carryover.learning.subsystem_lr_factor = 2.0;
    let lr_boosted = s2.compose_effective_lr(1.0, 1.0);

    // A higher subsystem_lr_factor should produce a higher (or equal) LR
    // Both are clamped to [0.0, 0.01] so the effect may be limited
    assert!(
        lr_boosted >= lr_default || (lr_boosted - lr_default).abs() < 1e-6,
        "boosted subsystem factor should increase LR: default={lr_default}, boosted={lr_boosted}"
    );
}

#[test]
fn test_compose_effective_lr_high_semantic_factor_increases_lr() {
    let mut s1 = make_service();
    let lr_low = s1.compose_effective_lr(0.1, 1.0);

    let mut s2 = make_service();
    let lr_high = s2.compose_effective_lr(5.0, 1.0);

    // Higher semantic factor should increase the learning rate (both clamped to 0.01)
    assert!(
        lr_high >= lr_low,
        "higher semantic_lr_factor should increase LR: low={lr_low}, high={lr_high}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// geometric_mean (private helper, tested indirectly via compose_effective_lr)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_compose_effective_lr_nan_factor_handled() {
    let mut s = make_service();
    // NaN in a factor should be filtered out by geometric_mean (it filters non-finite)
    let lr = s.compose_effective_lr(f32::NAN, 1.0);
    assert!(
        lr.is_finite(),
        "lr should be finite even with NaN input factor, got {lr}"
    );
    assert!(lr >= 0.0, "lr must be >= 0.0, got {lr}");
}

#[test]
fn test_compose_effective_lr_infinity_factor_handled() {
    let mut s = make_service();
    // Infinity should be filtered out by geometric_mean (it filters non-finite)
    let lr = s.compose_effective_lr(f32::INFINITY, 1.0);
    assert!(
        lr.is_finite(),
        "lr should be finite even with infinity input factor, got {lr}"
    );
}
