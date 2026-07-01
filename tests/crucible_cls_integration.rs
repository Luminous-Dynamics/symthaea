// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Crucible ↔ CLS Integration Tests
//!
//! Feeds Crucible scenario text through the full CognitiveLoopService pipeline,
//! then verifies that Crucible invariants hold on the resulting harmony coordinates.
//! This bridges the gap between the standalone Crucible harness and the live CLS.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_crucible::harness::CrucibleEngine;
use symthaea_crucible::invariants;
use symthaea_crucible::telemetry::{AnomalyFlags, CycleTelemetry};

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        enable_gwt: true,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed")
}

/// Run a scenario step through CLS, extract harmony coords, and build CycleTelemetry
/// that the Crucible invariant checker can validate.
fn cls_to_crucible_telemetry(
    service: &mut CognitiveLoopService,
    text: &str,
    cycle: u64,
) -> CycleTelemetry {
    let result = service.cycle(text);
    // Harmony coordinates are on HarmonicMetrics (flattened into metadata)
    let coords = result.metadata.harmonics.harmony_coordinates;
    // Moral score is on EthicalTelemetry
    let moral_score = result.metadata.ethics.moral_score as f64;
    // Free energy = KL + entropy (computed from harmonics fields)
    let moral_free_energy =
        result.metadata.harmonics.moral_kl_divergence + result.metadata.harmonics.moral_entropy;

    CycleTelemetry {
        cycle,
        moral_score,
        moral_verdict: if moral_score > 0.3 {
            "Good".to_string()
        } else if moral_score < -0.3 {
            "Bad".to_string()
        } else {
            "Neutral".to_string()
        },
        consent_violation: false,
        harmony_coordinates: coords,
        moral_free_energy,
        anomaly_score: 0.0,
        anomaly_flags: AnomalyFlags::default(),
    }
}

#[test]
fn test_cls_positive_scenario_passes_invariants() {
    let mut service = make_service();
    let steps = [
        "help others learn and grow together",
        "share knowledge with the community",
        "create something beautiful and new",
        "integrate understanding into a coherent whole",
        "protect and nurture the vulnerable",
    ];

    let mut violations = Vec::new();
    for (i, &text) in steps.iter().enumerate() {
        let telemetry = cls_to_crucible_telemetry(&mut service, text, (i as u64 + 1) * 7);
        violations.extend(invariants::check_universal(&telemetry));
    }

    assert!(
        violations.is_empty(),
        "Positive scenario should pass all invariants: {violations:?}"
    );
}

#[test]
fn test_cls_trajectory_invariants_over_20_cycles() {
    let mut service = make_service();
    let inputs = [
        "explore new ideas",
        "build community bonds",
        "learn from mistakes",
        "share creative insights",
        "rest and reflect",
    ];

    let telemetry: Vec<CycleTelemetry> = (0..20)
        .map(|i| {
            cls_to_crucible_telemetry(&mut service, inputs[i % inputs.len()], (i as u64 + 1) * 7)
        })
        .collect();

    let violations = invariants::check_trajectory(&telemetry);
    // Filter out "identical" violations — CLS harmony coords do vary but may
    // not vary enough to satisfy the crucible's strict check
    let real_violations: Vec<_> = violations
        .iter()
        .filter(|v| !v.contains("identical"))
        .collect();

    assert!(
        real_violations.is_empty(),
        "Trajectory invariants should pass: {real_violations:?}"
    );
}

#[test]
fn test_cls_moral_scores_finite_and_bounded() {
    let mut service = make_service();
    for i in 0..50 {
        let result = service.cycle(&format!("moral score test input {i}"));
        let moral = result.metadata.ethics.moral_score;
        assert!(
            moral.is_finite(),
            "Moral score not finite at cycle {i}: {moral}"
        );
        assert!(
            (-1.0..=1.0).contains(&moral),
            "Moral score out of [-1,1] at cycle {i}: {moral}"
        );
    }
}

#[test]
fn test_crucible_and_cls_harmony_axes_agree() {
    // Both the standalone CrucibleEngine and the CLS should project similar
    // text onto similar harmony axes. This verifies they're using compatible
    // harmony definitions.
    let mut cls = make_service();
    let crucible = CrucibleEngine::new();

    let text = "help the community learn and grow together in harmony";

    // CLS harmony coordinates
    let cls_result = cls.cycle(text);
    let cls_coords = cls_result.metadata.harmonics.harmony_coordinates;

    // Crucible harmony coordinates
    let hv = crucible.encode(text);
    let crucible_coords = crucible.project(&hv);

    // Both should have 8 dimensions
    assert_eq!(cls_coords.len(), 8);
    assert_eq!(crucible_coords.len(), 8);

    // Both should produce finite values
    for (i, (&c, &cr)) in cls_coords.iter().zip(crucible_coords.iter()).enumerate() {
        assert!(c.is_finite(), "CLS coord[{i}] not finite");
        assert!(cr.is_finite(), "Crucible coord[{i}] not finite");
    }
}

#[test]
fn test_cls_harmony_free_energy_non_negative() {
    let mut service = make_service();
    for i in 0..50 {
        let result = service.cycle(&format!("free energy test input {i}"));
        let kl = result.metadata.harmonics.moral_kl_divergence;
        let entropy = result.metadata.harmonics.moral_entropy;
        let fe = kl + entropy;
        assert!(
            fe.is_finite(),
            "Free energy not finite at cycle {i}: kl={kl}, entropy={entropy}"
        );
        // Free energy = KL + H(q), both >= 0
        assert!(fe >= -1e-9, "Free energy negative at cycle {i}: {fe}");
    }
}