// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal IIT tests: cause-effect information and integrated dynamics.

use super::*;

#[test]
fn test_temporal_transition_creation() {
    let current = ContinuousHV::random(HDC_DIMENSION, 1);
    let next = ContinuousHV::random(HDC_DIMENSION, 2);

    let transition = TemporalTransition::new(current.clone(), next.clone());

    assert_eq!(transition.current.dim(), HDC_DIMENSION);
    assert_eq!(transition.next.dim(), HDC_DIMENSION);
}

#[test]
fn test_cause_effect_information() {
    let calc = TemporalPhiCalculator::new();

    // Create a transition with correlated states (deterministic dynamics)
    let current = ContinuousHV::random(HDC_DIMENSION, 42);
    let noise = ContinuousHV::random(HDC_DIMENSION, 43);
    let next = ContinuousHV::weighted_bundle(&[&current, &noise], &[0.9, 0.1]);

    let transition = TemporalTransition::new(current, next);

    let cause_info = calc.cause_information(&transition);
    let effect_info = calc.effect_information(&transition);

    // Both should be positive for correlated states
    assert!(
        cause_info >= 0.0,
        "Cause info should be non-negative: {:.4}",
        cause_info
    );
    assert!(
        effect_info >= 0.0,
        "Effect info should be non-negative: {:.4}",
        effect_info
    );
}

#[test]
fn test_cause_effect_result() {
    let calc = TemporalPhiCalculator::new();

    let current = ContinuousHV::random(HDC_DIMENSION, 100);
    let next = ContinuousHV::random(HDC_DIMENSION, 101);
    let transition = TemporalTransition::new(current, next);

    let result = calc.compute_cause_effect(&transition);

    assert!(result.cause_info >= 0.0, "Cause info non-negative");
    assert!(result.effect_info >= 0.0, "Effect info non-negative");
    assert!(
        result.phi_cause_effect >= 0.0,
        "\u{03c6}_cause_effect non-negative"
    );
    assert!(
        result.phi_cause_effect <= result.cause_info.min(result.effect_info) + 1e-10,
        "\u{03c6}_cause_effect should be min of cause and effect"
    );
}

#[test]
fn test_deterministic_dynamics_high_info() {
    let calc = TemporalPhiCalculator::new();

    // Highly deterministic: next is almost copy of current
    let current = ContinuousHV::random(HDC_DIMENSION, 200);
    let noise = ContinuousHV::random(HDC_DIMENSION, 201);
    let next = ContinuousHV::weighted_bundle(&[&current, &noise], &[0.95, 0.05]);

    let deterministic = TemporalTransition::new(current, next);

    // Random dynamics: next is independent of current
    let random_current = ContinuousHV::random(HDC_DIMENSION, 300);
    let random_next = ContinuousHV::random(HDC_DIMENSION, 301);
    let random = TemporalTransition::new(random_current, random_next);

    let det_result = calc.compute_cause_effect(&deterministic);
    let rnd_result = calc.compute_cause_effect(&random);

    // Deterministic should have higher MI than random
    assert!(
        det_result.cause_info > rnd_result.cause_info,
        "Deterministic should have higher cause info: det={:.4} > rnd={:.4}",
        det_result.cause_info,
        rnd_result.cause_info
    );
}

#[test]
fn test_integrated_cause_effect_for_system() {
    let calc = TemporalPhiCalculator::new();

    // Create a system of 3 interacting components
    let base = ContinuousHV::random(HDC_DIMENSION, 400);

    let transitions: Vec<TemporalTransition> = (0..3)
        .map(|i| {
            let current = ContinuousHV::weighted_bundle(
                &[&base, &ContinuousHV::random(HDC_DIMENSION, 410 + i as u64)],
                &[0.8, 0.2],
            );
            let next = ContinuousHV::weighted_bundle(
                &[&base, &ContinuousHV::random(HDC_DIMENSION, 420 + i as u64)],
                &[0.7, 0.3],
            );
            TemporalTransition::new(current, next)
        })
        .collect();

    let result = calc.compute_system_cause_effect(&transitions);

    assert!(result.cause_info >= 0.0, "System cause info non-negative");
    assert!(result.effect_info >= 0.0, "System effect info non-negative");
    assert!(
        result.integrated_cause >= 0.0,
        "Integrated cause non-negative"
    );
    assert!(
        result.integrated_effect >= 0.0,
        "Integrated effect non-negative"
    );
}

#[test]
fn test_integrated_cause_independent_system() {
    let calc = TemporalPhiCalculator::new();

    // Independent components - no shared dynamics
    let transitions: Vec<TemporalTransition> = (0..4)
        .map(|i| {
            let current = ContinuousHV::random(HDC_DIMENSION, 500 + i as u64 * 100);
            let next = ContinuousHV::random(HDC_DIMENSION, 501 + i as u64 * 100);
            TemporalTransition::new(current, next)
        })
        .collect();

    let integrated_cause = calc.integrated_cause_info(&transitions);
    let integrated_effect = calc.integrated_effect_info(&transitions);

    // Independent components should have low integrated info
    // (but may not be exactly zero due to random correlations)
    assert!(
        integrated_cause < 0.5,
        "Independent system should have low integrated cause: {:.4}",
        integrated_cause
    );
    assert!(
        integrated_effect < 0.5,
        "Independent system should have low integrated effect: {:.4}",
        integrated_effect
    );
}

#[test]
fn test_temporal_phi_symmetry() {
    let calc = TemporalPhiCalculator::new();

    let current = ContinuousHV::random(HDC_DIMENSION, 600);
    let next = ContinuousHV::random(HDC_DIMENSION, 601);

    let forward = TemporalTransition::new(current.clone(), next.clone());
    let backward = TemporalTransition::new(next, current);

    let fwd_result = calc.compute_cause_effect(&forward);
    let bwd_result = calc.compute_cause_effect(&backward);

    // MI is symmetric, so cause/effect should be similar magnitude
    let diff = (fwd_result.cause_info - bwd_result.effect_info).abs();
    assert!(
        diff < 0.1 || diff / fwd_result.cause_info.max(0.01) < 0.5,
        "Cause/effect should show symmetry: forward cause={:.4}, backward effect={:.4}",
        fwd_result.cause_info,
        bwd_result.effect_info
    );
}
