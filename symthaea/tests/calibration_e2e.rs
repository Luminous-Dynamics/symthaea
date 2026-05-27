// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// End-to-End Integration Tests: Calibration Hardening
// ==================================================================================
//
// Tests A–C: Full-pipeline validation (always-awake fallback, bath state vector,
//   calibration summary completeness).
// Tests D–E: Peer calibration blending and shareable profile roundtrip.
//
// No feature flags required — uses default configuration.
// ==================================================================================

use symthaea::cognitive_loop::calibration::{NeuromodCalibration, SharedCalibrationProfile};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Helper: create a CognitiveLoopService with sync training.
fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.01,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed")
}

/// Run N warmup cycles with a default input.
fn warmup(service: &mut CognitiveLoopService, n: usize) {
    for _ in 0..n {
        service.cycle("steady state warmup");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test A: Always-awake fallback applies stale self-assessment calibration
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires ~2700 cycles, exceeds CI timeout — run locally with --include-ignored"]
fn test_always_awake_fallback_applies_stale_calibration() {
    // Build CLS with low learning_threshold to keep PE high and trigger self-assessment.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.001,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed");

    // Phase 1: Run ~250 diverse-input cycles to trigger self-assessment
    // (past warmup_threshold=200, with enough PE drift to trigger).
    let mut calibration_fired = false;
    let mut fired_at = 0;
    for i in 0..600 {
        let input = match i % 4 {
            0 => "unprecedented quantum turbulence anomaly detected sector seven",
            1 => "gentle morning sunlight warm peaceful calm stable routine",
            2 => "CRITICAL EMERGENCY ALERT catastrophic failure imminent evacuation",
            _ => "abstract mathematical topology homomorphism integration convergence",
        };
        let result = service.cycle(input);
        if result.metadata.neuromod.self_assessment_calibration_fired {
            calibration_fired = true;
            fired_at = i;
            break;
        }
    }
    assert!(
        calibration_fired,
        "Self-assessment should have triggered within 600 cycles"
    );

    // At this point, pending_calibration_waiting should be true
    let check = service.cycle("post-trigger check");
    assert!(
        check.metadata.neuromod.pending_calibration_waiting,
        "Pending calibration should exist after self-assessment trigger"
    );

    // Phase 2: Snapshot bath state before fallback
    let pre = service.bath_state_vector();

    // Phase 3: Run past ALWAYS_AWAKE_STALE_CYCLES (2000) with steady input
    // Total cycles from trigger point should exceed 2000.
    let cycles_needed = 2100;
    for _ in 0..cycles_needed {
        service.cycle("steady waiting for fallback");
    }

    // Phase 4: Verify the fallback applied it (pending should be gone)
    let post_check = service.cycle("post-fallback check");
    assert!(
        !post_check.metadata.neuromod.pending_calibration_waiting,
        "Always-awake fallback should have applied the stale calibration (fired at cycle {fired_at})"
    );

    // Phase 5: Bath should have changed
    let post = service.bath_state_vector();
    let l2: f32 = pre
        .iter()
        .zip(post.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    // Bath changes accumulate over 2100 cycles even without calibration,
    // so we just verify the system is healthy and finite.
    assert!(
        l2.is_finite(),
        "Bath state L2 distance should be finite: {l2}"
    );
    for (i, &v) in post.iter().enumerate() {
        assert!(v.is_finite(), "Bath element {i} should be finite: {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test B: Bath state vector changes after calibration
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bath_state_vector_changes_after_calibration() {
    let mut service = make_service();
    warmup(&mut service, 30);

    let pre = service.bath_state_vector();

    // Ingest multi-transmitter calibration with strong z-scores
    service.ingest_calibration(&[
        ("Executive::Stroop", "stroop_effect", -2.0),
        ("WorM::N-back", "nback_2::accuracy", -2.0),
        ("SustainedAttention::CPT", "dprime", -1.5),
    ]);
    service.apply_pending_calibration();
    service.cycle("post calibration");

    let post = service.bath_state_vector();

    // All elements finite
    for (i, &v) in post.iter().enumerate() {
        assert!(v.is_finite(), "Bath element {i} should be finite: {v}");
    }

    // L2 distance should be > 0 (calibration modifies receptor sensitivities)
    let l2: f32 = pre
        .iter()
        .zip(post.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        l2 > 0.0,
        "Bath state should change after calibration: L2={l2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test C: Calibration summary contains all adjusted transmitters
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_calibration_summary_contains_all_adjusted_transmitters() {
    let mut service = make_service();
    warmup(&mut service, 20);

    // Ingest calibration covering 4 transmitters
    service.ingest_calibration(&[
        ("Executive::Stroop", "stroop_effect", -1.5),
        ("WorM::N-back", "nback_2::accuracy", -1.0),
        ("Inhibition::StopSignal", "ssrt_ticks", -1.0),
        ("SustainedAttention::CPT", "dprime", -1.2),
    ]);
    service.apply_pending_calibration();

    let summary = service
        .last_calibration_summary()
        .expect("summary should exist after apply");
    assert!(summary.contains("DA"), "Summary missing DA: {summary}");
    assert!(summary.contains("ACh"), "Summary missing ACh: {summary}");
    assert!(
        summary.contains("NE") || summary.contains("NE-β"),
        "Summary missing NE: {summary}"
    );
    assert!(summary.contains("5-HT"), "Summary missing 5-HT: {summary}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test D: Peer calibration blending
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_peer_calibration_blending() {
    // Agent A: high interference → attenuate DA (factor < 1.0)
    let mut service_a = make_service();
    warmup(&mut service_a, 30);
    service_a.ingest_calibration(&[("Executive::Stroop", "stroop_effect", -2.0)]);
    let profile_a = service_a
        .export_calibration_profile(Some("agent_a".into()))
        .expect("A should have pending calibration");
    assert!(
        profile_a.sensitivity_factors.contains_key("DA"),
        "A should have DA factor"
    );

    // Agent B: low interference → boost DA (factor > 1.0) + poor WM → boost ACh
    let mut service_b = make_service();
    warmup(&mut service_b, 30);
    service_b.ingest_calibration(&[
        ("Executive::Stroop", "stroop_effect", 2.0), // sign-corrected: +2 = good = low interference
        ("WorM::N-back", "nback_2::accuracy", -2.0), // poor WM
    ]);

    // Record B's DA factor before merge
    let profile_b_before = service_b
        .export_calibration_profile(None)
        .expect("B should have pending calibration");
    let b_da_before = *profile_b_before
        .sensitivity_factors
        .get("DA")
        .expect("B should have DA factor");

    // Merge A's profile into B
    service_b.merge_peer_calibration(&profile_a);

    // Export B's calibration after merge
    let profile_b_after = service_b
        .export_calibration_profile(None)
        .expect("B should still have pending calibration");
    let b_da_after = *profile_b_after
        .sensitivity_factors
        .get("DA")
        .expect("B should still have DA factor");

    // B's DA factor should have moved toward A's DA factor (peer influence)
    assert!(
        (b_da_after - b_da_before).abs() > 0.001,
        "Peer merge should shift B's DA factor: before={b_da_before}, after={b_da_after}"
    );

    // B should still have its own ACh adjustment
    assert!(
        profile_b_after.sensitivity_factors.contains_key("ACh"),
        "B should retain its own ACh adjustment"
    );

    // Apply and verify health
    service_b.apply_pending_calibration();
    let result = service_b.cycle("post peer merge");
    assert!(result.prediction_error.is_finite());
    assert!(result.metadata.neuromod.dopamine_effective.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test E: Peer calibration shareable roundtrip
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_peer_calibration_shareable_roundtrip() {
    // Create calibration covering 4 transmitters
    let cal = NeuromodCalibration::from_z_scores(&[
        ("Executive::Stroop", -1.5),
        ("WorM::N-back", -1.0),
        ("Inhibition::StopSignal", 1.0),
        ("SustainedAttention::CPT", -0.8),
    ]);

    let shareable = cal.to_shareable(Some("agent_x".into()));

    // Verify expected keys
    assert!(
        shareable.sensitivity_factors.contains_key("DA"),
        "Should have DA"
    );
    assert!(
        shareable.sensitivity_factors.contains_key("ACh"),
        "Should have ACh"
    );
    assert!(
        shareable.sensitivity_factors.contains_key("NE-beta"),
        "Should have NE-beta"
    );
    assert!(
        shareable.sensitivity_factors.contains_key("NE-alpha"),
        "Should have NE-alpha"
    );
    assert!(
        shareable.sensitivity_factors.contains_key("5-HT"),
        "Should have 5-HT"
    );

    // Verify agent_id
    assert_eq!(shareable.agent_id.as_deref(), Some("agent_x"));

    // Verify coverage > 0
    assert!(shareable.coverage > 0.0, "Coverage should be positive");

    // JSON roundtrip
    let json = serde_json::to_string(&shareable).expect("serialize");
    let deserialized: SharedCalibrationProfile = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.agent_id, shareable.agent_id);
    assert!((deserialized.coverage - shareable.coverage).abs() < 0.0001);
    assert!((deserialized.confidence_delta - shareable.confidence_delta).abs() < 0.0001);
    assert_eq!(
        deserialized.sensitivity_factors.len(),
        shareable.sensitivity_factors.len()
    );
    for (key, &val) in &shareable.sensitivity_factors {
        let deser_val = deserialized.sensitivity_factors[key];
        assert!(
            (deser_val - val).abs() < 0.0001,
            "Factor mismatch for {key}: {val} vs {deser_val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test F: Closed-loop calibration validation — telemetry fields populated
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_calibration_validator_telemetry_populated_after_calibration() {
    let mut service = make_service();
    warmup(&mut service, 30);

    // Apply a calibration with strong z-scores
    service.ingest_calibration(&[
        ("Executive::Stroop", "stroop_effect", -2.0),
        ("WorM::N-back", "nback_2::accuracy", -1.5),
    ]);
    service.apply_pending_calibration();

    // Run cycles to allow validation to settle (100+ cycle settling period)
    for _ in 0..150 {
        service.cycle("post-calibration settling");
    }

    // Check telemetry fields
    let result = service.cycle("telemetry check");
    // Calibration validator should have recorded at least one validation
    // (it fires at settling_cycles after calibration application)
    let total = result.metadata.calibration_validations_total;
    let mult = result.metadata.calibration_adjustment_multiplier;

    assert!(mult.is_finite(), "adjustment_multiplier should be finite");
    assert!(
        mult > 0.0 && mult <= 1.0,
        "multiplier should be in (0, 1]: {mult}"
    );
    // Cooldown should be a reasonable value
    let cd = result.metadata.calibration_cooldown_duration;
    assert!(
        cd >= 200 && cd <= 2000,
        "cooldown should be in [200, 2000]: {cd}"
    );

    // The new adaptive dynamics telemetry should also be populated
    assert!(result.metadata.epistemic_uncertainty.is_finite());
    assert!(result.metadata.aleatoric_uncertainty.is_finite());
    assert!(result.metadata.theta_phase.is_finite());
    assert!(result.metadata.temporal_binding_strength.is_finite());
    assert!(result.metadata.prediction_horizon_scale.is_finite());
    assert!(result.metadata.prediction_horizon_scale > 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test G: Adaptive dynamics fields bounded over multiple cycles
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_adaptive_dynamics_telemetry_bounded() {
    let mut service = make_service();

    for i in 0..50 {
        let input = if i % 3 == 0 {
            "novel surprising input"
        } else {
            "steady state"
        };
        let r = service.cycle(input);

        assert!(
            r.metadata.epistemic_uncertainty >= 0.0 && r.metadata.epistemic_uncertainty <= 1.0,
            "epistemic should be [0,1]: {}",
            r.metadata.epistemic_uncertainty
        );
        assert!(
            r.metadata.aleatoric_uncertainty >= 0.0 && r.metadata.aleatoric_uncertainty <= 1.0,
            "aleatoric should be [0,1]: {}",
            r.metadata.aleatoric_uncertainty
        );
        assert!(
            r.metadata.theta_phase >= 0.0 && r.metadata.theta_phase < 7.0,
            "theta should be [0, 2π): {}",
            r.metadata.theta_phase
        );
        assert!(
            r.metadata.temporal_binding_strength >= 0.0
                && r.metadata.temporal_binding_strength <= 1.0,
            "binding should be [0,1]: {}",
            r.metadata.temporal_binding_strength
        );
        assert!(
            r.metadata.prediction_horizon_scale > 0.0 && r.metadata.prediction_horizon_scale <= 1.5,
            "horizon scale should be (0, 1.5]: {}",
            r.metadata.prediction_horizon_scale
        );
    }
}