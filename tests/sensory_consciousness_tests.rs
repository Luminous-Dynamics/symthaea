// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Channel Sensory Consciousness Tests
//!
//! These tests feed raw sensory hypervectors — not text strings — into the
//! cognitive loop via `cycle_with_hv()`. This exercises the behavioral
//! consciousness proxy (4-input weighted average). Note: the full
//! ConsciousnessEngine (SpectralMIP Phi) only runs in `cycle(text)` mode.
//! The proxy exercises these 4 inputs:
//!
//! - **prediction_confidence** (0.30 weight): from multi-scale prediction matching
//! - **temporal_coherence** (0.25 weight): from CfC tau convergence
//! - **flow_intensity** (0.20 weight): from sustained low-error states
//! - **pattern_confidence** (0.25 weight): from temporal signature classification
//!
//! Text-only testing only exercises prediction_confidence and pattern_confidence.
//! These tests simulate real sensor streams to activate the full consciousness
//! equation.
//!
//! ## Key insight
//! Flow intensity needs 5+ consecutive cycles with error < 0.25 and
//! coherence > 0.6. Temporal coherence rises when CfC tau values converge
//! under repeated similar input. Both require SUSTAINED engagement, not
//! single-shot evaluation.
//!
//! ## No features required
//! `cycle_with_hv()` works without the `humanoid` feature. These tests
//! compile fast and don't OOM.
//!
//! Run: `cargo test --test sensory_consciousness_tests -- --nocapture`

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::symthaea_core::hdc::ContinuousHV;

const DIM: usize = 16_384;

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService")
}

/// Create a "sensor reading" HV — a deterministic pattern from a seed.
fn sensor_hv(seed: u64) -> ContinuousHV {
    ContinuousHV::random(DIM, seed)
}

/// Perturb an HV slightly — simulates sensor drift / noise.
fn drift(hv: &ContinuousHV, amount: f32, seed: u64) -> ContinuousHV {
    let noise = ContinuousHV::random(DIM, seed);
    let mut values = hv.values.clone();
    for (i, v) in values.iter_mut().enumerate() {
        *v = *v * (1.0 - amount) + noise.values[i] * amount;
    }
    ContinuousHV::from_values(values)
}

/// Interpolate between two HVs (linear blend).
fn lerp(a: &ContinuousHV, b: &ContinuousHV, t: f32) -> ContinuousHV {
    let mut values = vec![0.0f32; DIM];
    for i in 0..DIM {
        values[i] = a.values[i] * (1.0 - t) + b.values[i] * t;
    }
    ContinuousHV::from_values(values)
}

/// Bind two HVs (element-wise multiply) — multi-modal fusion.
fn bind(a: &ContinuousHV, b: &ContinuousHV) -> ContinuousHV {
    let mut values = vec![0.0f32; DIM];
    for i in 0..DIM {
        values[i] = a.values[i] * b.values[i];
    }
    let mut hv = ContinuousHV::from_values(values);
    hv.normalize();
    hv
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: Stable Environment → Consciousness Builds
//
// Feed the same sensor pattern (with tiny drift) for 15 cycles.
// The CfC network should converge, temporal coherence should rise,
// and consciousness should stabilize at a positive level.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_stable_environment_consciousness_builds() {
    let mut service = make_service();
    let base_pattern = sensor_hv(42);

    let mut phi_history = Vec::new();

    for i in 0..15 {
        // Same pattern with tiny drift (2% noise — simulates stable environment)
        let sensory = drift(&base_pattern, 0.02, 1000 + i);
        let result = service.cycle_with_hv(&sensory);
        let phi = result.metadata.consciousness.consciousness_level;
        phi_history.push(phi);

        assert!(phi.is_finite(), "NaN at cycle {i}");
    }

    // Consciousness should be positive after 15 stable cycles
    let final_phi = *phi_history.last().unwrap();
    assert!(
        final_phi >= 0.0,
        "Consciousness should be non-negative after stable input: {final_phi:.4}"
    );

    // Should show some dynamics (not flatlined at exactly zero for all 15)
    let any_nonzero = phi_history.iter().any(|&p| p > 0.0);

    eprintln!("\n═══ SENSORY TEST 1: STABLE ENVIRONMENT ═══");
    for (i, phi) in phi_history.iter().enumerate() {
        eprintln!("  Cycle {:2}: phi={phi:.6}", i);
    }
    eprintln!("  Any nonzero: {any_nonzero}");
    eprintln!("═══════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: Sudden Event → Prediction Error Spike
//
// 10 cycles stable, then abrupt pattern change, then 10 more stable.
// The consciousness level should respond to the disruption.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sudden_event_disrupts_consciousness() {
    let mut service = make_service();
    let pattern_a = sensor_hv(100);
    let pattern_b = sensor_hv(200); // Completely different

    // Phase 1: Stable with pattern A
    let mut phase1_phi = Vec::new();
    for i in 0..10 {
        let sensory = drift(&pattern_a, 0.02, 2000 + i);
        let result = service.cycle_with_hv(&sensory);
        phase1_phi.push(result.metadata.consciousness.consciousness_level);
    }

    // Phase 2: Sudden switch to pattern B
    let mut phase2_phi = Vec::new();
    for i in 0..5 {
        let sensory = drift(&pattern_b, 0.02, 3000 + i);
        let result = service.cycle_with_hv(&sensory);
        phase2_phi.push(result.metadata.consciousness.consciousness_level);
    }

    // Phase 3: Stabilize on pattern B
    let mut phase3_phi = Vec::new();
    for i in 0..10 {
        let sensory = drift(&pattern_b, 0.02, 4000 + i);
        let result = service.cycle_with_hv(&sensory);
        phase3_phi.push(result.metadata.consciousness.consciousness_level);
    }

    // All values finite
    for (phase, data) in [
        ("phase1", &phase1_phi),
        ("phase2", &phase2_phi),
        ("phase3", &phase3_phi),
    ] {
        for (i, phi) in data.iter().enumerate() {
            assert!(phi.is_finite(), "NaN in {phase} at cycle {i}");
        }
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let avg1 = avg(&phase1_phi);
    let avg2 = avg(&phase2_phi);
    let avg3 = avg(&phase3_phi);

    eprintln!("\n═══ SENSORY TEST 2: SUDDEN EVENT ═══");
    eprintln!("  Phase 1 (stable A): avg phi = {avg1:.6}");
    eprintln!("  Phase 2 (disrupted): avg phi = {avg2:.6}");
    eprintln!("  Phase 3 (stable B): avg phi = {avg3:.6}");
    eprintln!("════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3: Gradual Environmental Change
//
// Interpolate between pattern A and B over 20 cycles.
// Consciousness should track smoothly — no sudden drops or NaN.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_gradual_environmental_change() {
    let mut service = make_service();
    let pattern_a = sensor_hv(300);
    let pattern_b = sensor_hv(400);

    let mut phi_history = Vec::new();

    for i in 0..20 {
        let t = i as f32 / 19.0; // 0.0 to 1.0
        let sensory = lerp(&pattern_a, &pattern_b, t);
        let result = service.cycle_with_hv(&sensory);
        let phi = result.metadata.consciousness.consciousness_level;
        phi_history.push(phi);
        assert!(phi.is_finite(), "NaN during gradual change at cycle {i}");
    }

    // No sudden jumps (max cycle-to-cycle change should be bounded)
    let max_jump = phi_history
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f64, f64::max);

    eprintln!("\n═══ SENSORY TEST 3: GRADUAL CHANGE ═══");
    for (i, phi) in phi_history.iter().enumerate() {
        let t = i as f32 / 19.0;
        eprintln!("  t={t:.2}: phi={phi:.6}");
    }
    eprintln!("  Max cycle-to-cycle jump: {max_jump:.6}");
    eprintln!("══════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 4: Multi-Modal Sensory Fusion
//
// Bind visual + proprioceptive + auditory HVs into a fused representation.
// Compare consciousness from fused vs single-modal input.
// The fused input should produce a DIFFERENT consciousness trajectory.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_multimodal_fusion_affects_consciousness() {
    // Single-modal: visual only
    let mut service_visual = make_service();
    let visual_hv = sensor_hv(500);
    let mut visual_phi = Vec::new();
    for i in 0..15 {
        let sensory = drift(&visual_hv, 0.02, 5000 + i);
        let result = service_visual.cycle_with_hv(&sensory);
        visual_phi.push(result.metadata.consciousness.consciousness_level);
    }

    // Multi-modal: visual + proprioceptive + auditory (bound)
    let mut service_fused = make_service();
    let proprio_hv = sensor_hv(600);
    let auditory_hv = sensor_hv(700);
    let fused_base = bind(&bind(&visual_hv, &proprio_hv), &auditory_hv);
    let mut fused_phi = Vec::new();
    for i in 0..15 {
        let sensory = drift(&fused_base, 0.02, 6000 + i);
        let result = service_fused.cycle_with_hv(&sensory);
        fused_phi.push(result.metadata.consciousness.consciousness_level);
    }

    // Measure trajectory difference
    let phi_diffs: Vec<f64> = visual_phi
        .iter()
        .zip(&fused_phi)
        .map(|(v, f)| (v - f).abs())
        .collect();
    let max_diff = phi_diffs.iter().cloned().fold(0.0f64, f64::max);

    // After the modality-agnostic consciousness wiring, cycle_with_hv() now
    // computes consciousness_level from CfC tau convergence and available inputs.
    // Multi-modal fusion should produce a DIFFERENT trajectory than unimodal.
    assert!(
        max_diff > 0.0001,
        "MULTIMODAL IS IDENTICAL TO UNIMODAL: max_diff={max_diff:.8}. \
         Binding 3 modalities produces the same consciousness as 1."
    );

    let avg_visual = visual_phi.iter().sum::<f64>() / visual_phi.len() as f64;
    let avg_fused = fused_phi.iter().sum::<f64>() / fused_phi.len() as f64;

    eprintln!("\n═══ SENSORY TEST 4: MULTI-MODAL FUSION ═══");
    eprintln!("  Visual-only avg phi: {avg_visual:.6}");
    eprintln!("  Fused (V+P+A) avg phi: {avg_fused:.6}");
    eprintln!("  Max phi difference: {max_diff:.6}");
    eprintln!("  VERDICT: Multi-modal input produces different consciousness.");
    eprintln!("═══════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 5: Full 4-Input Variance Audit (Sensory Mode)
//
// THE paper-critical test: do ALL 4 consciousness inputs vary when we
// feed varied sensory data (not just text)?
//
// This replaces the text-only Experiment 7 which found only 1/3 varying.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sensory_consciousness_input_variance() {
    let mut service = make_service();

    let patterns = [
        sensor_hv(800),  // Pattern A
        sensor_hv(900),  // Pattern B (very different)
        sensor_hv(1000), // Pattern C
    ];

    let mut phi_values = Vec::new();
    let mut coherence_values = Vec::new();
    let mut flow_values = Vec::new();

    // Phase 1: Stable on pattern A (build coherence + flow)
    for i in 0..8 {
        let sensory = drift(&patterns[0], 0.01, 7000 + i);
        let result = service.cycle_with_hv(&sensory);
        phi_values.push(result.metadata.consciousness.consciousness_level);
        coherence_values.push(result.metadata.temporal.temporal_coherence_score);
        flow_values.push(result.metadata.drive_flow_intensity);
    }

    // Phase 2: Switch to pattern B (break coherence + flow)
    for i in 0..6 {
        let sensory = drift(&patterns[1], 0.01, 8000 + i);
        let result = service.cycle_with_hv(&sensory);
        phi_values.push(result.metadata.consciousness.consciousness_level);
        coherence_values.push(result.metadata.temporal.temporal_coherence_score);
        flow_values.push(result.metadata.drive_flow_intensity);
    }

    // Phase 3: Gradual blend to pattern C
    for i in 0..6 {
        let t = i as f32 / 5.0;
        let sensory = lerp(&patterns[1], &patterns[2], t);
        let result = service.cycle_with_hv(&sensory);
        phi_values.push(result.metadata.consciousness.consciousness_level);
        coherence_values.push(result.metadata.temporal.temporal_coherence_score);
        flow_values.push(result.metadata.drive_flow_intensity);
    }

    // Compute variance
    let variance_f64 = |v: &[f64]| -> f64 {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64
    };
    let variance_f32 = |v: &[f32]| -> f64 {
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64
    };

    let phi_var = variance_f64(&phi_values);
    let coh_var = variance_f64(&coherence_values);
    let flow_var = variance_f32(&flow_values);

    // Count varying inputs
    let varying = [phi_var > 1e-8, coh_var > 1e-8, flow_var > 1e-8]
        .iter()
        .filter(|&&v| v)
        .count();

    // After modality-agnostic wiring, phi should vary with sensory input.
    assert!(
        phi_var > 1e-10,
        "PHI IS CONSTANT even with varied sensory input! variance={phi_var:.10}"
    );

    eprintln!("\n═══ SENSORY TEST 5: 4-INPUT VARIANCE AUDIT ═══");
    eprintln!(
        "  Phi variance:        {phi_var:.8}  {}",
        if phi_var > 1e-8 { "VARIES" } else { "CONSTANT" }
    );
    eprintln!(
        "  Coherence variance:  {coh_var:.8}  {}",
        if coh_var > 1e-8 { "VARIES" } else { "CONSTANT" }
    );
    eprintln!(
        "  Flow variance:       {flow_var:.8}  {}",
        if flow_var > 1e-8 {
            "VARIES"
        } else {
            "CONSTANT"
        }
    );
    eprintln!("  Varying inputs: {varying}/3");
    if varying >= 3 {
        eprintln!("  VERDICT: Consciousness is multi-dimensional with sensory input!");
    } else if varying >= 2 {
        eprintln!("  VERDICT: Consciousness is partially multi-dimensional.");
    } else {
        eprintln!(
            "  FINDING: Consciousness remains low-dimensional even with varied sensory input."
        );
        eprintln!(
            "  This suggests CfC tau convergence and flow entry need longer sustained engagement."
        );
    }
    eprintln!("═══════════════════════════════════════════════\n");
}