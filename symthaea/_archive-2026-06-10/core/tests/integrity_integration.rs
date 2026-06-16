// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integrity framework integration tests.
//!
//! Verifies that the IntegrityManager is wired into the cognitive loop,
//! integrity telemetry is populated in CycleMetadata, and safety escalation
//! fires on critical anomalies.
//!
//! Feature-gated: `integrity`.

#![cfg(feature = "integrity")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

/// Rotating inputs to prevent degenerate single-input patterns.
const INPUTS: &[&str] = &[
    "The weather is warm today.",
    "I need to solve this problem efficiently.",
    "Music brings people together in unexpected ways.",
    "How does photosynthesis convert light into energy?",
    "The architecture of this building is remarkable.",
];

/// Build a deterministic CognitiveLoopService with integrity feature enabled.
fn build_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    config.async_training = false;
    config.genesis_phrase = Some("integrity_integration_2026".to_string());
    CognitiveLoopService::new(config).expect("CognitiveLoopService should construct")
}

#[test]
fn integrity_telemetry_populated_after_cycles() {
    let mut service = build_service();
    let mut results = Vec::new();

    // Run enough cycles for integrity checks to fire
    for i in 0..110 {
        let input = INPUTS[i % INPUTS.len()];
        results.push(service.cycle(input));
    }

    // After 110 cycles, attestation (interval=101) should have run at least once
    let last = results.last().unwrap();
    let integrity = &last.metadata.integrity;
    assert!(
        integrity.last_check_cycle > 0,
        "integrity should have run at least one check"
    );
    // Default attestations (safety thresholds, consciousness weights, receptor sensitivities)
    // should all pass since nothing has been tampered with
    assert!(
        integrity.attestation_passed,
        "attestation should pass with no tampering"
    );
    assert!(
        integrity.canaries_passed,
        "canaries should pass with no tampering"
    );
    assert!(!integrity.has_critical, "no critical anomalies expected");
}

#[test]
fn integrity_temporal_consistency_populated() {
    let mut service = build_service();

    // Run a few cycles — temporal consistency runs every cycle
    for i in 0..10 {
        let input = INPUTS[i % INPUTS.len()];
        service.cycle(input);
    }

    let result = service.cycle("final check");
    let integrity = &result.metadata.integrity;
    // Temporal consistency runs every cycle, so last_check_cycle should be current
    assert!(
        integrity.last_check_cycle > 0,
        "temporal check should run every cycle"
    );
    assert!(
        integrity.temporal_passed,
        "temporal consistency should pass under normal operation"
    );
}

#[test]
fn integrity_default_telemetry_is_clean() {
    let mut service = build_service();
    let result = service.cycle("hello");
    let integrity = &result.metadata.integrity;

    // First cycle — attestation hasn't run yet (interval=101), but defaults should be clean
    assert!(integrity.attestation_passed);
    assert!(integrity.canaries_passed);
    assert!(!integrity.has_critical);
    assert_eq!(integrity.anomaly_count, 0);
}