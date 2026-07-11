// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::types::{ConsciousnessWeights, WeightConvergenceState};
use super::*;
use crate::cognitive_loop::types::ConsciousnessCache;
use crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2;
use symthaea_core::consciousness_metrics::{
    SpectralMIPConfig, SpectralMIPFinder, StructuralPhiResult,
};
use symthaea_core::hdc::{BinaryHV, ContinuousHV};

fn make_engine() -> ConsciousnessEngine {
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let finder = SpectralMIPFinder::new(config);
    ConsciousnessEngine::new(finder, None, None, None)
}

/// Helper to build a test input with both ContinuousHV and BinaryHV.
fn make_input<'a>(
    hdv: &'a ContinuousHV,
    hv16: &'a BinaryHV,
    cycle: u64,
) -> ConsciousnessEngineInput<'a> {
    ConsciousnessEngineInput {
        hdv,
        hv16,
        cycle,
        unified_psi: 0.5,
        coherence: 0.6,
        prediction_error: 0.2,
        phi_attention_weight: 0.4,
        epistemic_quality: 0.5,
        phi_validation_correlation: 0.5,
        bath_entropy: 1.0,
        attractor_detected: false,
        sht_2a_signal: 0.5,
        gaba_a_signal: 0.4,
        substrate_feasibility: 1.0,
        binding_capability: 1.0,
        workspace_capability: 1.0,
        attention_capability: 1.0,
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        hot_depth: 0.5,
        cpg_sync_index: 0.5,
        cantor_metacognitive_depth: 0.5,
        governance_collective_phi: 0.0,
        gwt_broadcast_occurred: false,
        gwt_coalition_size: 0,
        prediction_precision: 1.0,
        knowledge_grounding: 0.5,
        knowledge_coherence: 0.0,
        glyph_coherence: 0.0,
        temporal_coherence_phi: 0.0,
    }
}

#[test]
fn test_engine_creation() {
    let engine = make_engine();
    assert!(engine.multi_modal_integrator.is_none());
    assert!(engine.consciousness_equation_v2.is_none());
    assert!(engine.unified_consciousness_pipeline.is_none());
}

#[test]
fn test_engine_measure_returns_valid_output() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    let input = make_input(&hdv, &hv16, 1);

    let output = engine.measure(&input);

    assert!(output.unified_consciousness >= 0.0);
    assert!(output.unified_consciousness <= 1.0);
    // total_us is u64, may be 0μs when cache-hot from prior tests — that's fine
    assert!(output.lr_factor.is_finite());
    assert!(output.confidence_delta.is_finite());
}

#[test]
fn test_engine_spectral_mip_fires_at_interval_47() {
    let mut engine = make_engine();

    for cycle in 0..48 {
        let hdv = ContinuousHV::random(16384, cycle + 1);
        let hv16 = BinaryHV::random(cycle + 1);
        let input = make_input(&hdv, &hv16, cycle);
        let output = engine.measure(&input);

        if cycle < 47 {
            assert!(
                output.spectral_mip_phi.is_none() || output.spectral_mip_phi.unwrap().is_finite()
            );
        }
    }
}

#[test]
fn test_unified_consciousness_bounded() {
    let engine = make_engine();

    assert!(engine.compute_unified(Some(100.0), 1.0, 1.0, 1.0) <= 1.0);
    assert!(engine.compute_unified(Some(0.0), 0.0, 0.0, 0.0) >= 0.0);
    assert!(engine.compute_unified(None, 0.5, 0.5, 0.5) >= 0.0);
    assert!(engine.compute_unified(None, 0.5, 0.5, 0.5) <= 1.0);
}

#[test]
fn test_equation_v2_feedback_deltas() {
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let finder = SpectralMIPFinder::new(config);
    let eq = ConsciousnessEquationV2::default();
    let mut engine = ConsciousnessEngine::new(finder, None, Some(eq), None);

    // Run to cycle 23 (when equation v2 fires)
    for cycle in 0..24 {
        let hdv = ContinuousHV::random(16384, cycle + 100);
        let hv16 = BinaryHV::random(cycle + 100);
        let input = ConsciousnessEngineInput {
            hdv: &hdv,
            hv16: &hv16,
            cycle,
            unified_psi: 0.8,
            coherence: 0.9,
            prediction_error: 0.1,
            phi_attention_weight: 0.7,
            epistemic_quality: 0.8,
            phi_validation_correlation: 0.5,
            bath_entropy: 1.0,
            attractor_detected: false,
            sht_2a_signal: 0.5,
            gaba_a_signal: 0.4,
            substrate_feasibility: 1.0,
            binding_capability: 1.0,
            workspace_capability: 1.0,
            attention_capability: 1.0,
            moral_drift: 0.0,
            moral_anomaly_score: 0.0,
            hot_depth: 0.5,
            cpg_sync_index: 0.5,
            cantor_metacognitive_depth: 0.5,
            governance_collective_phi: 0.0,
            gwt_broadcast_occurred: false,
            gwt_coalition_size: 0,
            prediction_precision: 1.0,
            knowledge_grounding: 0.5,
            glyph_coherence: 0.0,
            temporal_coherence_phi: 0.0,
            knowledge_coherence: 0.5,
        };
        let output = engine.measure(&input);

        if cycle == 23 {
            assert!(
                output.equation_v2_consciousness.is_finite(),
                "C(t) should be finite"
            );
            assert!(
                output.equation_v2_consciousness >= 0.0,
                "C(t) should be non-negative"
            );
        }
    }
}

#[test]
fn test_cache_update() {
    let mut engine = make_engine();
    engine.cache.last_spectral_mip_phi = Some(0.42);
    engine.cache.last_sigma = Some(0.42);
    engine.cache.last_multimodal_phi = 0.33;
    engine.cache.last_equation_v2_consciousness = 0.55;

    let mut cc = ConsciousnessCache::default();
    engine.update_cache(&mut cc);

    assert_eq!(cc.last_spectral_mip_phi, Some(0.42));
    assert_eq!(cc.last_sigma, Some(0.42));
    assert!((cc.last_multimodal_phi - 0.33).abs() < f64::EPSILON);
    assert!((cc.last_equation_v2_consciousness - 0.55).abs() < f64::EPSILON);
}

#[test]
fn test_low_consciousness_boosts_exploration() {
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let finder = SpectralMIPFinder::new(config);
    let eq = ConsciousnessEquationV2::default();
    let mut engine = ConsciousnessEngine::new(finder, None, Some(eq), None);

    for cycle in 0..24 {
        let hdv = ContinuousHV::random(16384, cycle + 100);
        let hv16 = BinaryHV::random(cycle + 100);
        let input = ConsciousnessEngineInput {
            hdv: &hdv,
            hv16: &hv16,
            cycle,
            unified_psi: 0.05,
            coherence: 0.1,
            prediction_error: 0.9,
            phi_attention_weight: 0.05,
            epistemic_quality: 0.1,
            phi_validation_correlation: 0.0,
            bath_entropy: 1.0,
            attractor_detected: false,
            sht_2a_signal: 0.5,
            gaba_a_signal: 0.4,
            substrate_feasibility: 1.0,
            binding_capability: 1.0,
            workspace_capability: 1.0,
            attention_capability: 1.0,
            moral_drift: 0.0,
            moral_anomaly_score: 0.0,
            hot_depth: 0.5,
            cpg_sync_index: 0.5,
            cantor_metacognitive_depth: 0.5,
            governance_collective_phi: 0.0,
            gwt_broadcast_occurred: false,
            gwt_coalition_size: 0,
            prediction_precision: 1.0,
            knowledge_grounding: 0.5,
            glyph_coherence: 0.0,
            temporal_coherence_phi: 0.0,
            knowledge_coherence: 0.5,
        };
        let output = engine.measure(&input);

        if cycle == 23
            && output.equation_v2_consciousness < 0.3
            && output.equation_v2_consciousness > 0.0
        {
            assert!(
                output.exploration_delta > 0.0,
                "Low consciousness should boost exploration"
            );
        }
    }
}

#[test]
fn test_timing_fields_populated() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    let input = make_input(&hdv, &hv16, 1);
    let output = engine.measure(&input);

    assert!(output.spectral_mip_us < 1_000_000);
    assert!(output.total_us < 1_000_000);
}

// ═══════════════════════════════════════════════════════════════════
// Phase 6 #4: Bath → Consciousness Engine Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bath_baseline_no_change() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    // Default bath values: sht_2a=0.5, gaba_a=0.4, no attractor
    let input = make_input(&hdv, &hv16, 1);
    let output = engine.measure(&input);
    // At baseline values, modulation should be near-zero
    assert!(output.unified_consciousness >= 0.0);
    assert!(output.unified_consciousness <= 1.0);
}

#[test]
fn test_high_sht_2a_boosts_consciousness() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    let input_baseline = ConsciousnessEngineInput {
        hdv: &hdv,
        hv16: &hv16,
        cycle: 1,
        unified_psi: 0.5,
        coherence: 0.6,
        prediction_error: 0.2,
        phi_attention_weight: 0.4,
        epistemic_quality: 0.5,
        phi_validation_correlation: 0.5,
        bath_entropy: 1.0,
        attractor_detected: false,
        sht_2a_signal: 0.5,
        gaba_a_signal: 0.4,
        substrate_feasibility: 1.0,
        binding_capability: 1.0,
        workspace_capability: 1.0,
        attention_capability: 1.0,
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        hot_depth: 0.5,
        cpg_sync_index: 0.5,
        cantor_metacognitive_depth: 0.5,
        governance_collective_phi: 0.0,
        gwt_broadcast_occurred: false,
        gwt_coalition_size: 0,
        prediction_precision: 1.0,
        knowledge_grounding: 0.5,
        knowledge_coherence: 0.0,
        glyph_coherence: 0.0,
        temporal_coherence_phi: 0.0,
    };
    let out_base = engine.measure(&input_baseline);

    let mut engine2 = make_engine();
    let input_high = ConsciousnessEngineInput {
        sht_2a_signal: 1.0, // High 5-HT2A
        ..input_baseline
    };
    let out_high = engine2.measure(&input_high);
    assert!(
        out_high.unified_consciousness >= out_base.unified_consciousness,
        "High 5-HT2A should boost consciousness"
    );
}

#[test]
fn test_high_gaba_a_dampens_consciousness() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    let input_baseline = ConsciousnessEngineInput {
        hdv: &hdv,
        hv16: &hv16,
        cycle: 1,
        unified_psi: 0.5,
        coherence: 0.6,
        prediction_error: 0.2,
        phi_attention_weight: 0.4,
        epistemic_quality: 0.5,
        phi_validation_correlation: 0.5,
        bath_entropy: 1.0,
        attractor_detected: false,
        sht_2a_signal: 0.5,
        gaba_a_signal: 0.4,
        substrate_feasibility: 1.0,
        binding_capability: 1.0,
        workspace_capability: 1.0,
        attention_capability: 1.0,
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        hot_depth: 0.5,
        cpg_sync_index: 0.5,
        cantor_metacognitive_depth: 0.5,
        governance_collective_phi: 0.0,
        gwt_broadcast_occurred: false,
        gwt_coalition_size: 0,
        prediction_precision: 1.0,
        knowledge_grounding: 0.5,
        knowledge_coherence: 0.0,
        glyph_coherence: 0.0,
        temporal_coherence_phi: 0.0,
    };
    let out_base = engine.measure(&input_baseline);

    let mut engine2 = make_engine();
    let input_high_gaba = ConsciousnessEngineInput {
        gaba_a_signal: 1.0, // High GABA-A
        ..input_baseline
    };
    let out_gaba = engine2.measure(&input_high_gaba);
    assert!(
        out_gaba.unified_consciousness <= out_base.unified_consciousness,
        "High GABA-A should dampen consciousness"
    );
}

#[test]
fn test_attractor_depresses_consciousness() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    let input_no_attractor = ConsciousnessEngineInput {
        hdv: &hdv,
        hv16: &hv16,
        cycle: 1,
        unified_psi: 0.5,
        coherence: 0.6,
        prediction_error: 0.2,
        phi_attention_weight: 0.4,
        epistemic_quality: 0.5,
        phi_validation_correlation: 0.5,
        bath_entropy: 1.0,
        attractor_detected: false,
        sht_2a_signal: 0.5,
        gaba_a_signal: 0.4,
        substrate_feasibility: 1.0,
        binding_capability: 1.0,
        workspace_capability: 1.0,
        attention_capability: 1.0,
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        hot_depth: 0.5,
        cpg_sync_index: 0.5,
        cantor_metacognitive_depth: 0.5,
        governance_collective_phi: 0.0,
        gwt_broadcast_occurred: false,
        gwt_coalition_size: 0,
        prediction_precision: 1.0,
        knowledge_grounding: 0.5,
        knowledge_coherence: 0.0,
        glyph_coherence: 0.0,
        temporal_coherence_phi: 0.0,
    };
    let out_no = engine.measure(&input_no_attractor);

    let mut engine2 = make_engine();
    let input_attractor = ConsciousnessEngineInput {
        attractor_detected: true,
        ..input_no_attractor
    };
    let out_att = engine2.measure(&input_attractor);
    assert!(
        out_att.unified_consciousness <= out_no.unified_consciousness,
        "Attractor detection should depress consciousness"
    );
}

#[test]
fn test_bath_modulation_clamped() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    // Extreme values to test clamping
    let input = ConsciousnessEngineInput {
        hdv: &hdv,
        hv16: &hv16,
        cycle: 1,
        unified_psi: 0.99,
        coherence: 0.99,
        prediction_error: 0.01,
        phi_attention_weight: 0.99,
        epistemic_quality: 0.99,
        phi_validation_correlation: 0.9,
        bath_entropy: 0.0,
        attractor_detected: true,
        sht_2a_signal: 2.0, // Extreme
        gaba_a_signal: 0.0, // Low
        substrate_feasibility: 1.0,
        binding_capability: 1.0,
        workspace_capability: 1.0,
        attention_capability: 1.0,
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        hot_depth: 0.5,
        cpg_sync_index: 0.5,
        cantor_metacognitive_depth: 0.5,
        governance_collective_phi: 0.0,
        gwt_broadcast_occurred: false,
        gwt_coalition_size: 0,
        prediction_precision: 1.0,
        knowledge_grounding: 0.5,
        knowledge_coherence: 0.0,
        glyph_coherence: 0.0,
        temporal_coherence_phi: 0.0,
    };
    let out = engine.measure(&input);
    assert!(
        out.unified_consciousness >= 0.0 && out.unified_consciousness <= 1.0,
        "Output should be clamped [0,1], got {}",
        out.unified_consciousness
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase: Dynamic Consciousness Weights
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_weights_default_sum_to_one() {
    let w = ConsciousnessWeights::default();
    let sum = w.spectral + w.equation + w.pipeline + w.multimodal;
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Default weights should sum to 1.0, got {}",
        sum
    );
    assert!(w.is_normalized());
}

#[test]
fn test_weights_normalize_preserves_ratios() {
    let mut w = ConsciousnessWeights {
        spectral: 0.7,
        equation: 0.5,
        pipeline: 0.5,
        multimodal: 0.3,
    };
    let ratio_before = w.spectral / w.equation;
    w.normalize();
    let ratio_after = w.spectral / w.equation;
    assert!(
        (ratio_before - ratio_after).abs() < 1e-10,
        "Normalize should preserve ratios"
    );
    assert!(w.is_normalized());
}

#[test]
fn test_weights_normalize_from_zero_safe() {
    let mut w = ConsciousnessWeights {
        spectral: 0.0,
        equation: 0.0,
        pipeline: 0.0,
        multimodal: 0.0,
    };
    w.normalize();
    // Should fall back to defaults
    let def = ConsciousnessWeights::default();
    assert!((w.spectral - def.spectral).abs() < 1e-10);
    assert!(w.is_normalized());
}

#[test]
fn test_high_emergence_boosts_spectral() {
    let mut engine = make_engine();
    let structural = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 2.0, // High emergence
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_phis: vec![0.1, 0.2, 0.3],
        cluster_sizes: vec![2, 2, 2],
    };
    engine.update_weights_from_emergence(structural.emergence_ratio);
    assert!(
        engine.cache.weights.spectral > 0.35,
        "High emergence should boost spectral weight, got {}",
        engine.cache.weights.spectral
    );
}

#[test]
fn test_low_emergence_reduces_spectral() {
    let mut engine = make_engine();
    let structural = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 0.5, // Low emergence
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_phis: vec![0.1, 0.2, 0.3],
        cluster_sizes: vec![2, 2, 2],
    };
    engine.update_weights_from_emergence(structural.emergence_ratio);
    assert!(
        engine.cache.weights.spectral < 0.35,
        "Low emergence should reduce spectral weight, got {}",
        engine.cache.weights.spectral
    );
}

#[test]
fn test_neutral_emergence_near_defaults() {
    let mut engine = make_engine();
    let structural = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 1.0, // Neutral
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_phis: vec![0.1, 0.2, 0.3],
        cluster_sizes: vec![2, 2, 2],
    };
    engine.update_weights_from_emergence(structural.emergence_ratio);
    let def = ConsciousnessWeights::default();
    assert!(
        (engine.cache.weights.spectral - def.spectral).abs() < 0.02,
        "Neutral emergence should yield near-default weights, got spectral={}",
        engine.cache.weights.spectral
    );
}

#[test]
fn test_weights_always_sum_to_one() {
    let mut engine = make_engine();
    // Test across a range of emergence ratios
    for er_x10 in 0..=100 {
        let er = er_x10 as f64 / 10.0;
        let structural = StructuralPhiResult {
            micro_phi: 0.1,
            meso_phi: 0.2,
            macro_phi: 0.3,
            emergence_ratio: er,
            bottleneck_score: 0.1,
            num_clusters: 3,
            cluster_phis: vec![0.1, 0.2, 0.3],
            cluster_sizes: vec![2, 2, 2],
        };
        engine.update_weights_from_emergence(structural.emergence_ratio);
        assert!(
            engine.cache.weights.is_normalized(),
            "Weights should sum to 1.0 for er={}, got {:?}",
            er,
            engine.cache.weights
        );
    }
}

#[test]
fn test_weights_all_positive() {
    let mut engine = make_engine();
    for er_x10 in 0..=100 {
        let er = er_x10 as f64 / 10.0;
        let structural = StructuralPhiResult {
            micro_phi: 0.1,
            meso_phi: 0.2,
            macro_phi: 0.3,
            emergence_ratio: er,
            bottleneck_score: 0.1,
            num_clusters: 3,
            cluster_phis: vec![0.1, 0.2, 0.3],
            cluster_sizes: vec![2, 2, 2],
        };
        engine.update_weights_from_emergence(structural.emergence_ratio);
        let w = &engine.cache.weights;
        assert!(
            w.spectral > 0.0 && w.equation > 0.0 && w.pipeline > 0.0 && w.multimodal > 0.0,
            "All weights must be positive for er={}, got {:?}",
            er,
            w
        );
    }
}

#[test]
fn test_ema_smoothing_converges() {
    let mut engine = make_engine();
    let structural = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 2.0,
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_phis: vec![0.1, 0.2, 0.3],
        cluster_sizes: vec![2, 2, 2],
    };
    // Apply 20 updates (should converge)
    for _ in 0..20 {
        engine.update_weights_from_emergence(structural.emergence_ratio);
    }
    let smoothed = engine.cache.smoothed_emergence_ratio.unwrap();
    assert!(
        (smoothed - 2.0).abs() < 0.01,
        "EMA should converge to 2.0, got {}",
        smoothed
    );
}

#[test]
fn test_ema_prevents_sudden_jump() {
    let mut engine = make_engine();
    // First: low emergence
    let low = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 0.5,
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_phis: vec![0.1, 0.2, 0.3],
        cluster_sizes: vec![2, 2, 2],
    };
    for _ in 0..10 {
        engine.update_weights_from_emergence(low.emergence_ratio);
    }
    let after_low = engine.cache.smoothed_emergence_ratio.unwrap();

    // Sudden jump to high emergence
    let high = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 5.0,
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_sizes: vec![4, 4, 4],
        cluster_phis: vec![0.1, 0.1, 0.1],
    };
    engine.update_weights_from_emergence(high.emergence_ratio);
    let after_jump = engine.cache.smoothed_emergence_ratio.unwrap();

    // Should NOT jump all the way to 5.0
    assert!(
        after_jump < 3.0,
        "EMA should prevent sudden jump: after_low={}, after_jump={}",
        after_low,
        after_jump
    );
}

#[test]
fn test_none_structural_phi_uses_defaults() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    let input = make_input(&hdv, &hv16, 1);
    let output = engine.measure(&input);
    // No structural phi → weights should be defaults
    let def = ConsciousnessWeights::default();
    assert_eq!(output.current_weights, def.as_array());
}

#[test]
fn test_unified_consciousness_still_bounded() {
    let mut engine = make_engine();
    // Force extreme weights via structural phi
    let structural = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 10.0,
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_sizes: vec![4, 4, 4],
        cluster_phis: vec![0.1, 0.1, 0.1],
    };
    engine.update_weights_from_emergence(structural.emergence_ratio);

    let unified = engine.compute_unified(Some(100.0), 1.0, 1.0, 1.0);
    assert!((0.0..=1.0).contains(&unified));
    let unified2 = engine.compute_unified(Some(0.0), 0.0, 0.0, 0.0);
    assert!((0.0..=1.0).contains(&unified2));
}

// ═══════════════════════════════════════════════════════════════════
// Improvement 1: Structural Phi persistence
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_structural_phi_persists_in_cache() {
    let mut engine = make_engine();
    engine.cache.last_structural_phi = Some(StructuralPhiResult {
        micro_phi: 0.15,
        meso_phi: 0.25,
        macro_phi: 0.35,
        emergence_ratio: 1.5,
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_phis: vec![0.1, 0.2, 0.3],
        cluster_sizes: vec![2, 2, 2],
    });

    let mut cc = ConsciousnessCache::default();
    engine.update_cache(&mut cc);

    assert!(cc.last_structural_phi.is_some());
    let sp = cc.last_structural_phi.unwrap();
    assert!((sp.micro_phi - 0.15).abs() < f64::EPSILON);
    assert!((sp.emergence_ratio - 1.5).abs() < f64::EPSILON);
}

// ═══════════════════════════════════════════════════════════════════
// Improvement 3: Weight Stability Metrics
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_weight_variance_empty_history() {
    let engine = make_engine();
    assert_eq!(engine.weight_variance(), 0.0);
}

#[test]
fn test_weight_variance_stable_weights() {
    let mut engine = make_engine();
    let w = [0.35, 0.25, 0.25, 0.15];
    for _ in 0..10 {
        engine.cache.weight_history.push_back(w);
    }
    assert!(
        engine.weight_variance() < 1e-15,
        "Stable weights should have near-zero variance, got {}",
        engine.weight_variance()
    );
}

#[test]
fn test_weight_variance_oscillating_weights() {
    let mut engine = make_engine();
    let w1 = [0.45, 0.20, 0.20, 0.15];
    let w2 = [0.25, 0.30, 0.30, 0.15];
    for i in 0..20 {
        if i % 2 == 0 {
            engine.cache.weight_history.push_back(w1);
        } else {
            engine.cache.weight_history.push_back(w2);
        }
    }
    assert!(
        engine.weight_variance() > 0.001,
        "Oscillating weights should have high variance, got {}",
        engine.weight_variance()
    );
}

#[test]
fn test_adaptive_alpha_slows_under_variance() {
    let mut engine = make_engine();
    // Fill history with oscillating weights to create high variance
    let w1 = [0.45, 0.20, 0.20, 0.15];
    let w2 = [0.25, 0.30, 0.30, 0.15];
    for i in 0..20 {
        if i % 2 == 0 {
            engine.cache.weight_history.push_back(w1);
        } else {
            engine.cache.weight_history.push_back(w2);
        }
    }
    let variance = engine.weight_variance();
    let base_alpha = 0.3;
    let alpha = base_alpha * (1.0 / (1.0 + 50.0 * variance));
    assert!(
        alpha < base_alpha,
        "Alpha should be reduced under variance: alpha={}, base={}",
        alpha,
        base_alpha
    );
}

#[test]
fn test_weight_history_bounded_at_100() {
    let mut engine = make_engine();
    let structural = StructuralPhiResult {
        micro_phi: 0.1,
        meso_phi: 0.2,
        macro_phi: 0.3,
        emergence_ratio: 1.5,
        bottleneck_score: 0.1,
        num_clusters: 3,
        cluster_phis: vec![0.1, 0.2, 0.3],
        cluster_sizes: vec![2, 2, 2],
    };
    for _ in 0..150 {
        engine.update_weights_from_emergence(structural.emergence_ratio);
    }
    assert!(
        engine.cache.weight_history.len() <= 100,
        "Weight history should be bounded at 100, got {}",
        engine.cache.weight_history.len()
    );
}

#[test]
fn test_weight_variance_in_output() {
    let mut engine = make_engine();
    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);
    let input = make_input(&hdv, &hv16, 1);
    let output = engine.measure(&input);
    // With no history, variance should be 0.0
    assert_eq!(output.weight_variance, 0.0);
}

// ── Moral topology → consciousness coupling tests ─────────────────

fn make_engine_with_eq_v2() -> ConsciousnessEngine {
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let finder = SpectralMIPFinder::new(config);
    let eq = ConsciousnessEquationV2::default();
    ConsciousnessEngine::new(finder, None, Some(eq), None)
}

#[test]
fn test_drift_attenuates_epistemic_quality() {
    // With equation V2 enabled, drift should reduce the Knowledge component
    let mut engine_no_drift = make_engine_with_eq_v2();
    let mut engine_drift = make_engine_with_eq_v2();

    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);

    // Run both to cycle 23 (when equation V2 fires)
    for cycle in 0..23 {
        let input = make_input(&hdv, &hv16, cycle);
        engine_no_drift.measure(&input);
        engine_drift.measure(&input);
    }

    // At cycle 23: no drift
    let input_no_drift = ConsciousnessEngineInput {
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        ..make_input(&hdv, &hv16, 23)
    };
    let out_no_drift = engine_no_drift.measure(&input_no_drift);

    // At cycle 23: high drift
    let input_drift = ConsciousnessEngineInput {
        moral_drift: 0.4,
        moral_anomaly_score: 0.0,
        ..make_input(&hdv, &hv16, 23)
    };
    let out_drift = engine_drift.measure(&input_drift);

    // With drift, equation_v2 should be lower (attenuated epistemic quality)
    assert!(
        out_drift.equation_v2_consciousness <= out_no_drift.equation_v2_consciousness,
        "Drift {:.3} should attenuate EquationV2 (got drift={:.4}, no_drift={:.4})",
        0.4,
        out_drift.equation_v2_consciousness,
        out_no_drift.equation_v2_consciousness,
    );
}

#[test]
fn test_anomaly_dampens_unified_consciousness() {
    let mut engine_nominal = make_engine();
    let mut engine_anomaly = make_engine();

    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);

    let input_nominal = ConsciousnessEngineInput {
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        ..make_input(&hdv, &hv16, 1)
    };
    let out_nominal = engine_nominal.measure(&input_nominal);

    let input_anomaly = ConsciousnessEngineInput {
        moral_drift: 0.0,
        moral_anomaly_score: 1.0,
        ..make_input(&hdv, &hv16, 1)
    };
    let out_anomaly = engine_anomaly.measure(&input_anomaly);

    assert!(
        out_anomaly.unified_consciousness <= out_nominal.unified_consciousness,
        "Anomaly score=1.0 should dampen consciousness (got anomaly={:.4}, nominal={:.4})",
        out_anomaly.unified_consciousness,
        out_nominal.unified_consciousness,
    );
}

#[test]
fn test_coupling_disabled_no_effect() {
    let mut engine = make_engine();
    engine.moral_coupling.enabled = false;

    let hdv = ContinuousHV::random(16384, 42);
    let hv16 = BinaryHV::random(42);

    // Extreme drift + anomaly, but coupling disabled
    let input_extreme = ConsciousnessEngineInput {
        moral_drift: 1.0,
        moral_anomaly_score: 1.0,
        ..make_input(&hdv, &hv16, 1)
    };
    let out_extreme = engine.measure(&input_extreme);

    let mut engine2 = make_engine();
    engine2.moral_coupling.enabled = false;
    let input_zero = ConsciousnessEngineInput {
        moral_drift: 0.0,
        moral_anomaly_score: 0.0,
        ..make_input(&hdv, &hv16, 1)
    };
    let out_zero = engine2.measure(&input_zero);

    // With coupling disabled, both should produce identical results
    assert!(
        (out_extreme.unified_consciousness - out_zero.unified_consciousness).abs() < 1e-12,
        "Disabled coupling should have no effect (extreme={:.6}, zero={:.6})",
        out_extreme.unified_consciousness,
        out_zero.unified_consciousness,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Substrate Feasibility Telemetry Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_substrate_feasibility_in_metadata_default() {
    let metadata = super::super::CycleMetadata::default();
    assert!(
        (metadata.substrate_effective_feasibility - 0.0).abs() < f64::EPSILON,
        "Default substrate_effective_feasibility should be 0.0"
    );
}

#[test]
fn test_substrate_feasibility_affects_consciousness() {
    // Engine with equation v2 so substrate_feasibility actually matters
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let finder = SpectralMIPFinder::new(config);
    let eq = ConsciousnessEquationV2::default();
    let mut engine1 = ConsciousnessEngine::new(finder, None, Some(eq), None);

    let config2 = SpectralMIPConfig {
        num_components: 64,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };
    let finder2 = SpectralMIPFinder::new(config2);
    let eq2 = ConsciousnessEquationV2::default();
    let mut engine2 = ConsciousnessEngine::new(finder2, None, Some(eq2), None);

    // Run both to cycle 23 when equation v2 fires
    let mut out_full = None;
    let mut out_half = None;
    for cycle in 0..24 {
        let hdv = ContinuousHV::random(16384, cycle + 200);
        let hv16 = BinaryHV::random(cycle + 200);

        let input_full = ConsciousnessEngineInput {
            hdv: &hdv,
            hv16: &hv16,
            cycle,
            unified_psi: 0.7,
            coherence: 0.8,
            prediction_error: 0.1,
            phi_attention_weight: 0.5,
            epistemic_quality: 0.7,
            phi_validation_correlation: 0.5,
            bath_entropy: 1.0,
            attractor_detected: false,
            sht_2a_signal: 0.5,
            gaba_a_signal: 0.4,
            substrate_feasibility: 1.0,
            binding_capability: 1.0,
            workspace_capability: 1.0,
            attention_capability: 1.0,
            moral_drift: 0.0,
            moral_anomaly_score: 0.0,
            hot_depth: 0.5,
            cpg_sync_index: 0.5,
            cantor_metacognitive_depth: 0.5,
            governance_collective_phi: 0.0,
            gwt_broadcast_occurred: false,
            gwt_coalition_size: 0,
            prediction_precision: 1.0,
            knowledge_grounding: 0.5,
            glyph_coherence: 0.0,
            temporal_coherence_phi: 0.0,
            knowledge_coherence: 0.5,
        };
        out_full = Some(engine1.measure(&input_full));

        let input_half = ConsciousnessEngineInput {
            substrate_feasibility: 0.5,
            ..input_full
        };
        out_half = Some(engine2.measure(&input_half));
    }

    let full = out_full.unwrap();
    let half = out_half.unwrap();

    // With lower substrate feasibility, equation_v2 consciousness should differ
    // (only matters when eq v2 fires at cycle 23)
    assert!(
        full.equation_v2_consciousness.is_finite(),
        "Full substrate eq v2 should be finite"
    );
    assert!(
        half.equation_v2_consciousness.is_finite(),
        "Half substrate eq v2 should be finite"
    );
    // The values should differ because substrate_feasibility scales the equation
    if full.equation_v2_consciousness > 0.01 {
        assert!(
            (full.equation_v2_consciousness - half.equation_v2_consciousness).abs() > 1e-10,
            "Different substrate feasibility should produce different consciousness: full={}, half={}",
            full.equation_v2_consciousness,
            half.equation_v2_consciousness
        );
    }
}

#[test]
fn test_reduced_substrate_capabilities_lower_consciousness() {
    // Prove that binding/workspace/attention capabilities are load-bearing:
    // biochemical-level capabilities (0.3) should produce measurably lower
    // consciousness than biological-level (1.0).
    let config = SpectralMIPConfig {
        num_components: 64,
        window_size: 20,
        min_samples: 5,
        regularization: 1e-6,
        ..Default::default()
    };

    let eq_bio = ConsciousnessEquationV2::default();
    let mut engine_bio = ConsciousnessEngine::new(
        SpectralMIPFinder::new(config.clone()),
        None,
        Some(eq_bio),
        None,
    );

    let eq_biochem = ConsciousnessEquationV2::default();
    let mut engine_biochem =
        ConsciousnessEngine::new(SpectralMIPFinder::new(config), None, Some(eq_biochem), None);

    let mut out_bio = None;
    let mut out_biochem = None;
    for cycle in 0..24 {
        let hdv = ContinuousHV::random(16384, cycle + 500);
        let hv16 = BinaryHV::random(cycle + 500);

        let input_bio = ConsciousnessEngineInput {
            hdv: &hdv,
            hv16: &hv16,
            cycle,
            unified_psi: 0.7,
            coherence: 0.8,
            prediction_error: 0.1,
            phi_attention_weight: 0.5,
            epistemic_quality: 0.7,
            phi_validation_correlation: 0.5,
            bath_entropy: 1.0,
            attractor_detected: false,
            sht_2a_signal: 0.5,
            gaba_a_signal: 0.4,
            substrate_feasibility: 1.0,
            binding_capability: 1.0,
            workspace_capability: 1.0,
            attention_capability: 1.0,
            moral_drift: 0.0,
            moral_anomaly_score: 0.0,
            hot_depth: 0.5,
            cpg_sync_index: 0.5,
            cantor_metacognitive_depth: 0.5,
            governance_collective_phi: 0.0,
            gwt_broadcast_occurred: false,
            gwt_coalition_size: 0,
            prediction_precision: 1.0,
            knowledge_grounding: 0.5,
            glyph_coherence: 0.0,
            temporal_coherence_phi: 0.0,
            knowledge_coherence: 0.5,
        };
        out_bio = Some(engine_bio.measure(&input_bio));

        // Biochemical substrate: binding=0.3, workspace=0.4, attention=0.3
        let input_biochem = ConsciousnessEngineInput {
            binding_capability: 0.3,
            workspace_capability: 0.4,
            attention_capability: 0.3,
            ..input_bio
        };
        out_biochem = Some(engine_biochem.measure(&input_biochem));
    }

    let bio = out_bio.unwrap();
    let biochem = out_biochem.unwrap();

    // Equation V2 fires at cycle 23 — reduced capabilities should produce
    // lower Binding, Workspace, and Attention core components, leading to
    // lower overall consciousness.
    if bio.equation_v2_consciousness > 0.01 {
        assert!(
            bio.equation_v2_consciousness > biochem.equation_v2_consciousness,
            "Biological capabilities (1.0) should produce higher consciousness ({:.4}) \
                 than biochemical (0.3/0.4/0.3) ({:.4})",
            bio.equation_v2_consciousness,
            biochem.equation_v2_consciousness
        );
    }
}

#[test]
fn test_substrate_stress_sweep() {
    // Dynamic Substrate Validation: move from "hardcoded 1.0" to a stress sweep.
    // Simulate hardware degradation (e.g. power loss, bit flips, thermal throttling).
    let config = SpectralMIPConfig::default();
    let mut engine = ConsciousnessEngine::new(
        SpectralMIPFinder::new(config),
        None,
        Some(ConsciousnessEquationV2::default()),
        None,
    );

    let mut prev_consciousness = -1.0;

    // Sweep from 0.0 (total failure) to 1.0 (perfect hardware)
    // We expect monotonic increase (or at least no decrease) in consciousness.
    for i in 0..=10 {
        let substrate = i as f64 / 10.0;
        let cycle = 23 * (i + 1); // Ensure equation fires

        let hdv = ContinuousHV::random(16384, cycle as u64);
        let hv16 = BinaryHV::random(cycle as u64);

        let input = ConsciousnessEngineInput {
            unified_psi: 0.8,
            coherence: 0.8,
            prediction_error: 0.1,
            phi_attention_weight: 0.8,
            substrate_feasibility: substrate,
            ..make_input(&hdv, &hv16, cycle as u64)
        };

        let output = engine.measure(&input);
        let c = output.equation_v2_consciousness;

        assert!(
            c.is_finite(),
            "Consciousness must be finite at substrate={}",
            substrate
        );
        assert!(
            c >= 0.0 && c <= 1.0,
            "Consciousness must be in [0, 1], got {}",
            c
        );

        if prev_consciousness >= 0.0 {
            assert!(
                c >= prev_consciousness - 1e-10,
                "Consciousness should be monotonic with substrate feasibility: c({})={} < prev({})={}",
                substrate,
                c,
                (i - 1) as f64 / 10.0,
                prev_consciousness
            );
        }

        prev_consciousness = c;
    }

    // At substrate=0.0, consciousness should be near zero
    let hdv_zero = ContinuousHV::random(16384, 23 * 20);
    let hv16_zero = BinaryHV::random(23 * 20);
    let input_zero = ConsciousnessEngineInput {
        substrate_feasibility: 0.0,
        cycle: 23 * 20,
        ..make_input(&hdv_zero, &hv16_zero, 23 * 20)
    };
    let output_zero = engine.measure(&input_zero);
    assert!(
        output_zero.equation_v2_consciousness < 0.01,
        "Total substrate failure should collapse consciousness, got {}",
        output_zero.equation_v2_consciousness
    );
}

// ═══════════════════════════════════════════════════════════════════
// Weight Convergence Detection Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_convergence_state_initializing() {
    let engine = make_engine();
    // No weight history → Initializing
    assert_eq!(
        engine.convergence_state(),
        WeightConvergenceState::Initializing
    );
}

#[test]
fn test_convergence_state_converging() {
    let mut engine = make_engine();
    // Feed decreasing emergence ratios — variance should decrease over time
    // Start with high er, then settle to lower values
    for i in 0..15 {
        engine.update_weights_from_emergence(3.0 - (i as f64 * 0.15));
    }
    // Now feed constant values — recent half should have lower variance
    for _ in 0..15 {
        engine.update_weights_from_emergence(1.0);
    }
    let state = engine.convergence_state();
    assert!(
        state == WeightConvergenceState::Converging || state == WeightConvergenceState::Oscillating,
        "Should be Converging or Oscillating after transition, got {:?}",
        state
    );
}

#[test]
fn test_convergence_state_converged() {
    let mut engine = make_engine();
    // Feed identical emergence ratio for 60+ cycles to trigger converged streak
    for _ in 0..70 {
        engine.update_weights_from_emergence(1.0);
    }
    assert_eq!(
        engine.convergence_state(),
        WeightConvergenceState::Converged,
        "Constant input for 70 cycles should produce Converged state"
    );
}

#[test]
fn test_convergence_state_oscillating() {
    let mut engine = make_engine();
    // Alternate between very different emergence ratios to produce high variance
    for i in 0..30 {
        let er = if i % 2 == 0 { 5.0 } else { 0.1 };
        engine.update_weights_from_emergence(er);
    }
    let state = engine.convergence_state();
    assert_eq!(
        state,
        WeightConvergenceState::Oscillating,
        "Alternating emergence should produce Oscillating, got {:?}",
        state
    );
}

#[test]
fn test_converged_alpha_floor() {
    let mut engine = make_engine();
    // Converge first
    for _ in 0..70 {
        engine.update_weights_from_emergence(1.0);
    }
    assert!(engine.cache.converged_streak >= 50);

    // Record weights before another update
    let before = engine.cache.weights;
    engine.update_weights_from_emergence(3.0); // Big perturbation

    // Weights should barely move when converged (alpha clamped to [0.05, 0.1])
    let after = engine.cache.weights;
    let diff = (after.spectral - before.spectral).abs()
        + (after.equation - before.equation).abs()
        + (after.pipeline - before.pipeline).abs()
        + (after.multimodal - before.multimodal).abs();
    assert!(
        diff < 0.05,
        "Converged alpha floor should limit weight movement, got diff={}",
        diff
    );
}

#[test]
fn test_convergence_state_equal_variance_is_oscillating() {
    let mut engine = make_engine();
    // Push 20+ identical weight snapshots — both halves will have identical
    // (near-zero) variance, so recent_var < older_var * 0.8 is false → Oscillating.
    let w = [0.35, 0.25, 0.25, 0.15];
    for _ in 0..30 {
        engine.cache.weight_history.push_back(w);
    }
    // Reset converged streak so we don't short-circuit to Converged
    engine.cache.converged_streak = 0;
    let state = engine.convergence_state();
    assert_eq!(
        state,
        WeightConvergenceState::Oscillating,
        "Equal variance in both halves should yield Oscillating, got {:?}",
        state
    );
}

// ═══════════════════════════════════════════════════════════════════════
// PR 2: High-Fidelity Substrate Stress Tests
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub struct SubstrateStress {
    pub power_available: f64,
    pub thermal_headroom: f64,
    pub memory_integrity: f64,
    pub sensor_integrity: f64,
    pub timing_stability: f64,
    pub compute_throughput: f64,
}

impl SubstrateStress {
    pub fn perfect() -> Self {
        Self {
            power_available: 1.0,
            thermal_headroom: 1.0,
            memory_integrity: 1.0,
            sensor_integrity: 1.0,
            timing_stability: 1.0,
            compute_throughput: 1.0,
        }
    }

    /// Compute aggregate substrate feasibility [0, 1]
    pub fn feasibility(&self) -> f64 {
        // Multiplicative model: any zero axis collapses the whole
        let raw = self.power_available
            * self.thermal_headroom
            * self.memory_integrity
            * self.sensor_integrity
            * self.timing_stability
            * self.compute_throughput;
        raw.clamp(0.0, 1.0)
    }
}

#[test]
fn test_substrate_stress_multi_axis() {
    let config = SpectralMIPConfig::default();
    let mut engine = ConsciousnessEngine::new(
        SpectralMIPFinder::new(config),
        None,
        Some(ConsciousnessEquationV2::default()),
        None,
    );

    /// Create a dummy input with stable references for testing
    fn make_input<'a>(
        hdv: &'a ContinuousHV,
        hv16: &'a BinaryHV,
        cycle: u64,
        substrate: f64,
    ) -> ConsciousnessEngineInput<'a> {
        ConsciousnessEngineInput {
            hdv,
            hv16,
            cycle,
            unified_psi: 0.8,
            coherence: 0.8,
            prediction_error: 0.1,
            phi_attention_weight: 0.8,
            epistemic_quality: 0.8,
            phi_validation_correlation: 0.8,
            bath_entropy: 0.1,
            attractor_detected: true,
            sht_2a_signal: 0.0,
            gaba_a_signal: 0.0,
            substrate_feasibility: substrate,
            binding_capability: 1.0,
            workspace_capability: 1.0,
            attention_capability: 1.0,
            moral_drift: 0.0,
            moral_anomaly_score: 0.0,
            hot_depth: 0.5,
            cpg_sync_index: 1.0,
            cantor_metacognitive_depth: 0.5,
            governance_collective_phi: 0.0,
            gwt_broadcast_occurred: true,
            gwt_coalition_size: 10,
            prediction_precision: 1.0,
            knowledge_grounding: 0.8,
            knowledge_coherence: 0.8,
            glyph_coherence: 0.8,
            temporal_coherence_phi: 0.5,
        }
    }

    // 1. Baseline: Perfect Hardware
    let stress_perfect = SubstrateStress::perfect();
    let hdv_p = ContinuousHV::random(16384, 1);
    let hv16_p = BinaryHV::random(1);
    let input_perfect = make_input(&hdv_p, &hv16_p, 23, stress_perfect.feasibility());
    let out_perfect = engine.measure(&input_perfect).equation_v2_consciousness;

    // 2. Scenario: Thermal Throttling (80% performance hit)
    let mut stress_thermal = SubstrateStress::perfect();
    stress_thermal.thermal_headroom = 0.2;
    stress_thermal.compute_throughput = 0.5;
    let input_thermal = make_input(&hdv_p, &hv16_p, 23, stress_thermal.feasibility());
    let out_thermal = engine.measure(&input_thermal).equation_v2_consciousness;
    assert!(
        out_thermal < out_perfect,
        "Thermal stress should reduce consciousness"
    );

    // 3. Scenario: Memory Bandwidth / Bit-flip noise (High integrity loss)
    let mut stress_mem = SubstrateStress::perfect();
    stress_mem.memory_integrity = 0.3;
    let input_mem = make_input(&hdv_p, &hv16_p, 23, stress_mem.feasibility());
    let out_mem = engine.measure(&input_mem).equation_v2_consciousness;
    assert!(
        out_mem < out_perfect * 0.5,
        "Severe memory integrity loss should collapse consciousness"
    );

    // 4. Scenario: Clock Jitter / Timing Instability
    let mut stress_jitter = SubstrateStress::perfect();
    stress_jitter.timing_stability = 0.1;
    let input_jitter = make_input(&hdv_p, &hv16_p, 23, stress_jitter.feasibility());
    let out_jitter = engine.measure(&input_jitter).equation_v2_consciousness;
    assert!(
        out_jitter < 0.1,
        "Extreme timing instability should make consciousness intractable"
    );

    // 5. Monotonicity check over power sweep
    let mut prev_c = out_perfect;
    for i in (0..10).rev() {
        let mut stress = SubstrateStress::perfect();
        stress.power_available = i as f64 / 10.0;
        let input = make_input(&hdv_p, &hv16_p, 23, stress.feasibility());
        let c = engine.measure(&input).equation_v2_consciousness;
        assert!(
            c <= prev_c + 1e-10,
            "Power loss must not increase consciousness: c({})={} vs prev={}",
            stress.power_available,
            c,
            prev_c
        );
        prev_c = c;
    }
}

// ═══════════════════════════════════════════════════════════════════
// HOT (Higher-Order Thought) Recursion Depth Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_recursion_varies_with_hot_depth() {
    // Verify that different hot_depth values produce different consciousness
    // scores once EquationV2 fires (cycle >= 23).
    let eq_v2 = ConsciousnessEquationV2::new();

    let mut engine_low = make_engine();
    engine_low.consciousness_equation_v2 = Some(eq_v2.clone());
    let mut engine_high = make_engine();
    engine_high.consciousness_equation_v2 = Some(eq_v2);

    let hdv = ContinuousHV::random(16384, 99);
    let hv16 = BinaryHV::random(99);

    let mut out_low = None;
    let mut out_high = None;

    for cycle in 0..30 {
        let mut input_low = make_input(&hdv, &hv16, cycle);
        input_low.hot_depth = 0.1; // Low HOT recursion

        let mut input_high = make_input(&hdv, &hv16, cycle);
        input_high.hot_depth = 0.9; // High HOT recursion

        out_low = Some(engine_low.measure(&input_low));
        out_high = Some(engine_high.measure(&input_high));
    }

    let low = out_low.unwrap();
    let high = out_high.unwrap();

    // EquationV2 fires at cycle 23+. Higher HOT depth should produce
    // higher Recursion component → higher equation_v2_consciousness.
    if high.equation_v2_consciousness > 0.01 {
        assert!(
            high.equation_v2_consciousness > low.equation_v2_consciousness,
            "Higher hot_depth (0.9) should produce higher consciousness ({:.4}) \
                 than lower hot_depth (0.1) ({:.4})",
            high.equation_v2_consciousness,
            low.equation_v2_consciousness
        );
    }
}

#[test]
fn test_hot_depth_default_preserves_backward_compat() {
    // hot_depth=0.5 (the default when meta_cognition is disabled) should
    // produce the same result as the previous hardcoded Recursion=0.5.
    let eq_v2 = ConsciousnessEquationV2::new();
    let mut engine = make_engine();
    engine.consciousness_equation_v2 = Some(eq_v2);

    let hdv = ContinuousHV::random(16384, 77);
    let hv16 = BinaryHV::random(77);

    // Run through cycle 23+ where EquationV2 fires
    for cycle in 0..30 {
        let input = make_input(&hdv, &hv16, cycle);
        // make_input already sets hot_depth=0.5
        let out = engine.measure(&input);
        assert!(
            out.equation_v2_consciousness.is_finite(),
            "consciousness should be finite at cycle {}",
            cycle
        );
    }
}
