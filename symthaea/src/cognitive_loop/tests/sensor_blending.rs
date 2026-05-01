// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the unified additive-sensor blend in phase_perception.
//!
//! Verifies that when a sensor modality (radio/SDR here — the STT path would
//! require real cpal hardware) fires during a cycle, the resulting CycleResult
//! differs from a baseline cycle with the same text input but no sensor
//! observations, AND all downstream values remain finite.
//!
//! This closes a gap in phase_coverage: those tests exercise the perception
//! phase in isolation but never verify that a blended sensor HV flows through
//! the whole pipeline (CfC → reasoning → moral → language) without corrupting
//! anything.

#![cfg(feature = "mesh")]

use super::super::managers::radio_dispatcher::SpectrumObservation;
use super::super::{CognitiveLoopConfig, CognitiveLoopService};

/// A spectrum observation that differs enough from "no observation" to produce
/// a nonzero HDC contribution in the perception bundle. The exact values are
/// not semantically meaningful for the test — we care that the HV is distinct.
fn test_observation() -> SpectrumObservation {
    SpectrumObservation {
        frequency_hz: 2_400_000_000, // 2.4 GHz WiFi band
        noise_floor_dbm: -70.0,
        snr_db: 3.0,
        jammed: true,
    }
}

fn all_finite(result: &super::super::CycleResult) -> bool {
    result.thought_vector.iter().all(|v| v.is_finite())
}

#[test]
fn radio_blend_does_not_panic() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service
        .spectrum_manager
        .inject_observation(test_observation());

    // Smoke: the cycle must complete without panicking when a sensor HV is
    // bundled into the perception encoding.
    let _ = service.cycle("the fox jumps");
}

#[test]
fn radio_blend_keeps_downstream_finite() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service
        .spectrum_manager
        .inject_observation(test_observation());

    let result = service.cycle("the fox jumps");

    assert!(
        all_finite(&result),
        "thought_vector must remain finite after sensor blend — NaN/Inf would indicate \
         that BinaryHV::bundle or downstream CfC chose a degenerate code path"
    );
    assert!(
        result.thought_vector.iter().any(|v| *v != 0.0),
        "thought_vector must be non-trivial — all zeros would mean the CfC \
         never processed the blended HV"
    );
}

#[test]
fn radio_blend_shifts_thought_vector_from_baseline() {
    // Two services with identical config + identical text input. One has a
    // radio observation pending, the other does not. Their thought_vectors
    // must differ — this is the whole reason we bundle the radio HV.
    let input = "the quick brown fox";

    let mut baseline = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let baseline_result = baseline.cycle(input);

    let mut with_radio = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    with_radio
        .spectrum_manager
        .inject_observation(test_observation());

    // Sanity-check: the injection actually queued, and perception_hv()
    // produces a distinct HV. If these fail, the delta assertion below can't
    // tell us anything useful.
    assert_eq!(
        with_radio.spectrum_manager.pending_observations().len(),
        1,
        "inject_observation must queue the observation"
    );
    assert!(
        with_radio.spectrum_manager.perception_hv().is_some(),
        "perception_hv must produce Some with one observation queued"
    );

    let blended_result = with_radio.cycle(input);

    // Post-cycle: did the radio HV actually survive into perception, or did
    // something drain it before phase_perception ran?
    // (This isn't observable from CycleResult alone — the failure mode we're
    // debugging is that the observation somehow evaporated.)

    assert_eq!(
        baseline_result.thought_vector.len(),
        blended_result.thought_vector.len(),
        "both cycles must produce same-dimensional thought vectors"
    );

    // Measure L1 distance. The blend perturbs perception.hv16_cached, which
    // flows through encoding → CfC → compressed_state → thought_vector. Even
    // a small perturbation should produce a non-zero delta; exact magnitude
    // depends on CfC dynamics.
    let delta: f32 = baseline_result
        .thought_vector
        .iter()
        .zip(blended_result.thought_vector.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        delta > 0.0,
        "radio blend should shift thought_vector from baseline, got L1 delta {delta}"
    );
}

#[test]
fn no_observation_behaves_like_baseline() {
    // The perception_hv() method must return None when pending_observations
    // is empty, so the blend path is a no-op. Two services with identical
    // config, identical text, and NO observations must produce identical
    // thought vectors — determinism precondition for the previous test.
    let input = "determinism check";

    let mut a = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result_a = a.cycle(input);

    let mut b = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result_b = b.cycle(input);

    assert_eq!(
        result_a.thought_vector, result_b.thought_vector,
        "two identical-config services with no sensor input must be deterministic — \
         if this fails, the previous test's delta assertion is untrustworthy"
    );
}
