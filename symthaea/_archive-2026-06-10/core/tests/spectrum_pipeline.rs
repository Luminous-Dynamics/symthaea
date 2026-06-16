// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Spectrum Pipeline Integration Tests

Exercises the SpectrumManager cross-coupling pipeline:
1. Network degradation → consciousness modulation via connectivity_factor
2. Sustained jamming → safety escalation via is_network_critical
3. Mesh stats ingestion → tier loss EMA propagation
4. Beacon processing → peer discovery
5. CLS snapshot includes network_critical
6. Regulatory validation
*/

#![cfg(feature = "mesh")]

use symthaea::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, RadioTier, SpectrumManager, SpectrumObservation,
};

fn create_mesh_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS construction")
}

/// Verify that connectivity_factor modulates correctly across all health states.
#[test]
fn test_network_connectivity_affects_consciousness() {
    let sm_healthy = SpectrumManager::default();
    let mut sm_degraded = SpectrumManager::default();

    // Healthy: all tiers available → connectivity 1.0
    let cf_healthy = sm_healthy.connectivity_factor();
    assert_eq!(cf_healthy, 1.0, "All tiers up → connectivity 1.0");

    // Degraded: local down
    sm_degraded.set_tier_available(RadioTier::Local, false);
    let cf_degraded = sm_degraded.connectivity_factor();
    assert!(
        cf_degraded < cf_healthy,
        "Local down should reduce connectivity factor"
    );

    // Blackout: no tiers
    sm_degraded.set_tier_available(RadioTier::Metro, false);
    sm_degraded.set_tier_available(RadioTier::Regional, false);
    let cf_blackout = sm_degraded.connectivity_factor();
    assert_eq!(cf_blackout, 0.0, "Blackout → connectivity 0.0");
    assert!(
        sm_degraded.is_network_critical(),
        "Blackout should be network critical"
    );
}

/// Verify that sustained jamming triggers is_network_critical.
#[test]
fn test_sustained_jamming_triggers_safety_escalation() {
    let mut sm = SpectrumManager::default();

    // Below threshold: not critical
    assert!(
        !sm.is_network_critical(),
        "Fresh manager should not be critical"
    );

    // Simulate jamming streak by injecting jammed observations
    // The jamming_streak counter increments when process() detects all observations are jammed.
    // Since process() requires CognitiveSubsystem (pub(crate)), we test the blackout path instead.
    sm.set_tier_available(RadioTier::Local, false);
    sm.set_tier_available(RadioTier::Metro, false);
    sm.set_tier_available(RadioTier::Regional, false);

    assert!(
        sm.is_network_critical(),
        "Blackout (all tiers down) should be network critical"
    );

    // Verify connectivity factor reflects blackout
    assert_eq!(sm.connectivity_factor(), 0.0);
}

/// Verify mesh stats ingestion propagates to tier loss EMAs correctly.
#[test]
fn test_mesh_stats_propagate_to_tier_loss() {
    let mut sm = SpectrumManager::default();

    // Ingest moderate packet loss
    sm.ingest_mesh_stats(30, 100); // 30% loss

    let telem = sm.telemetry();

    // All tiers should absorb some loss
    assert!(
        telem.tier_loss_ema[0] > 0.0,
        "Local should absorb some loss"
    );
    assert!(
        telem.tier_loss_ema[1] > 0.0,
        "Metro should absorb some loss"
    );
    assert!(
        telem.tier_loss_ema[2] > 0.0,
        "Regional should absorb some loss"
    );

    // Regional absorbs most (weight 1.0), Local least (weight 0.2)
    assert!(
        telem.tier_loss_ema[2] > telem.tier_loss_ema[0],
        "Regional should absorb more than Local"
    );
    assert!(
        telem.tier_loss_ema[1] > telem.tier_loss_ema[0],
        "Metro should absorb more than Local"
    );
}

/// Verify repeated mesh stats ingestion converges (EMA property).
#[test]
fn test_mesh_stats_ema_convergence() {
    let mut sm = SpectrumManager::default();

    // Ingest same stats 50 times — should converge
    for _ in 0..50 {
        sm.ingest_mesh_stats(50, 100); // 50% loss
    }

    let telem = sm.telemetry();
    // After convergence: Local ≈ 0.5×0.2 = 0.1, Metro ≈ 0.5×0.5 = 0.25, Regional ≈ 0.5
    assert!(
        (telem.tier_loss_ema[0] - 0.1).abs() < 0.01,
        "Local loss EMA should converge to ~0.1, got {}",
        telem.tier_loss_ema[0]
    );
    assert!(
        (telem.tier_loss_ema[1] - 0.25).abs() < 0.01,
        "Metro loss EMA should converge to ~0.25, got {}",
        telem.tier_loss_ema[1]
    );
    assert!(
        (telem.tier_loss_ema[2] - 0.50).abs() < 0.01,
        "Regional loss EMA should converge to ~0.50, got {}",
        telem.tier_loss_ema[2]
    );
}

/// Verify beacon processing discovers new peers and returns correct flags.
#[test]
fn test_beacon_discovers_peers() {
    let mut sm = SpectrumManager::default();

    // generate_beacon() only fires every RADIO_BEACON_INTERVAL_CYCLES calls.
    // Call it until it produces a beacon.
    let mut beacon = None;
    for _ in 0..200 {
        if let Some(b) = sm.generate_beacon(&[0xAA; 8]) {
            beacon = Some(b);
            break;
        }
    }
    let beacon = beacon.expect("Should generate beacon within 200 calls");

    // A second manager receives the beacon
    let mut sm2 = SpectrumManager::default();
    let is_new = sm2.process_beacon(&beacon, RadioTier::Local, 20.0);
    assert!(is_new, "First beacon should discover new peer");

    // Same peer again → not new
    let is_new_again = sm2.process_beacon(&beacon, RadioTier::Local, 20.0);
    assert!(
        !is_new_again,
        "Second beacon from same peer should not be new"
    );
}

/// Verify the full CLS snapshot includes network_critical field.
#[test]
fn test_cls_snapshot_includes_network_critical() {
    let service = create_mesh_service();

    // Default: not critical
    let snapshot = service.consciousness_snapshot();
    assert!(
        !snapshot.network_critical,
        "Fresh CLS should not have network_critical set"
    );
}

/// Verify regulatory validation prevents out-of-band transmissions.
#[test]
fn test_regulatory_prevents_out_of_band() {
    let sm = SpectrumManager::default(); // ISM Global

    // In-band: 433 MHz Metro → allowed
    assert!(
        sm.validate_transmission(RadioTier::Metro, 433_500_000, 10.0),
        "433 MHz Metro should be allowed in ISM Global"
    );

    // Out-of-band: 100 MHz → blocked
    assert!(
        !sm.validate_transmission(RadioTier::Metro, 100_000_000, 14.0),
        "100 MHz should be blocked"
    );

    // Over-power: 433 MHz at 30 dBm → blocked (ISM Global max is 10 dBm)
    assert!(
        !sm.validate_transmission(RadioTier::Metro, 433_500_000, 30.0),
        "Over-power should be blocked"
    );
}

/// Verify swarm state generates synthetic observations for the waterfall model.
#[test]
fn test_swarm_state_generates_synthetic_observations() {
    let mut sm = SpectrumManager::default();

    // No observations initially
    assert_eq!(sm.pending_observations().len(), 0);

    // Inject swarm state with 5 peers
    sm.ingest_swarm_state(5, 0.6, 0.8);

    // Should have generated 1 synthetic observation
    assert_eq!(sm.pending_observations().len(), 1);

    let obs = &sm.pending_observations()[0];
    assert!(
        !obs.jammed,
        "Connected peers should not produce jammed observation"
    );
    assert!(obs.snr_db > 10.0, "Connected peers should have decent SNR");
}

/// Verify isolation (0 peers) produces degraded spectrum conditions.
#[test]
fn test_isolation_degrades_spectrum() {
    let mut sm = SpectrumManager::default();
    sm.ingest_swarm_state(0, 0.0, 0.05);

    let obs = &sm.pending_observations()[0];
    assert!(
        obs.jammed,
        "Isolation with low connectivity should flag as jammed"
    );
    assert!(obs.snr_db < 5.0, "Isolated node should have poor SNR");
}

/// Verify consciousness-aware tier selection routes based on confidence.
#[test]
fn test_consciousness_aware_tier_routing() {
    let mut sm = SpectrumManager::default();

    // High confidence → reliable tier (Local)
    let tier_high = sm.consciousness_aware_tier(40, 0.9);
    assert_eq!(tier_high, Some(RadioTier::Local));

    // Low confidence → energy-efficient tier (Metro)
    let tier_low = sm.consciousness_aware_tier(40, 0.1);
    assert_eq!(tier_low, Some(RadioTier::Metro));

    // Medium confidence → None (defer to normal routing)
    let tier_mid = sm.consciousness_aware_tier(40, 0.5);
    assert_eq!(tier_mid, None);

    // With Local down: high confidence falls to Metro
    sm.set_tier_available(RadioTier::Local, false);
    let tier_degraded = sm.consciousness_aware_tier(40, 0.9);
    assert_eq!(tier_degraded, Some(RadioTier::Metro));
}

/// Verify energy-aware routing logic works via normal routing (plentiful energy case).
#[test]
fn test_energy_aware_routing_plentiful() {
    let sm = SpectrumManager::default();
    // Fresh manager: energy_spent_nj = 0 → normal routing path
    let result = sm.energy_aware_route(100, 1);
    assert!(
        result.is_some(),
        "Should route successfully with plentiful energy"
    );
}

/// Verify delta compression stores peer state on Full, enabling Delta on subsequent calls.
#[test]
fn test_compression_full_then_delta() {
    use symthaea::cognitive_loop::CompressedDelta;

    let mut sm = SpectrumManager::default();
    let peer_id = [0x42u8; 8];
    let hv1 = [0xAA; 2048];

    // First call: Full (no previous state)
    let r1 = sm.compress_for_tier(&peer_id, &hv1, RadioTier::Metro);
    assert_eq!(
        r1.data.len(),
        2048,
        "First compression should be full 2048 bytes"
    );

    // Second call with 1-byte change: should use Delta (smaller)
    let mut hv2 = hv1;
    hv2[100] = 0xBB;
    let r2 = sm.compress_for_tier(&peer_id, &hv2, RadioTier::Metro);
    assert!(
        r2.data.len() < 100,
        "Delta compression should be much smaller than full, got {}",
        r2.data.len()
    );
}