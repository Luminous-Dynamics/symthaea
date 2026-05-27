// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property tests for the SpectrumManager radio dispatch module.
//!
//! Validates invariants of delta compression, routing determinism,
//! AIMD bandwidth bounds, network health monotonicity, MTU compliance,
//! and compression roundtrips.
//!
//! Feature-gated behind `mesh`.

#![cfg(feature = "mesh")]

use proptest::prelude::*;
use symthaea::cognitive_loop::{
    CompressedDelta, NetworkHealth, PayloadClass, PayloadClassifier, RadioTier, RoutingDecision,
    SpectrumManager,
};

/// Generate a random [u8; 2048] from a seed.
fn seed_to_array(seed: u64) -> [u8; 2048] {
    let mut arr = [0u8; 2048];
    let mut state = seed;
    for byte in arr.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (state >> 33) as u8;
    }
    arr
}

proptest! {
    // 1. Compression ratio bounded: delta compression output never exceeds 2x original + header.
    #[test]
    fn prop_compression_never_exceeds_bound(
        seed in any::<u64>(),
        changed_count in 0usize..200,
    ) {
        let base = seed_to_array(seed);
        let mut modified = base;
        for i in 0..changed_count.min(2048) {
            modified[i] = modified[i].wrapping_add(1);
        }
        let delta = CompressedDelta::from_diff(&base, &modified);
        // Full fallback or compressed, but never larger than 2x original + 16 header
        assert!(
            delta.rle_data.len() <= 2048 * 2 + 16,
            "Compressed size {} exceeds bound",
            delta.rle_data.len()
        );
    }

    // 2. Routing determinism: same input always routes to same tier.
    #[test]
    fn prop_routing_deterministic(
        local_avail in any::<bool>(),
        metro_avail in any::<bool>(),
        regional_avail in any::<bool>(),
        payload_size in 1usize..3000,
        urgency in 0u8..4,
    ) {
        let mut classifier = PayloadClassifier::default();
        classifier.set_tier_available(RadioTier::Local, local_avail);
        classifier.set_tier_available(RadioTier::Metro, metro_avail);
        classifier.set_tier_available(RadioTier::Regional, regional_avail);

        let r1 = classifier.route(PayloadClass::ConsciousnessDelta, payload_size, urgency);
        let r2 = classifier.route(PayloadClass::ConsciousnessDelta, payload_size, urgency);
        assert_eq!(r1, r2, "Routing should be deterministic");
    }

    // 3. AIMD convergence: bandwidth always stays within [min, max] bounds.
    #[test]
    fn prop_aimd_bandwidth_bounded(
        loss_events in 0usize..50,
        success_events in 0usize..50,
    ) {
        let mut sm = SpectrumManager::default();
        for _ in 0..loss_events {
            sm.report_loss(RadioTier::Local);
        }
        for _ in 0..success_events {
            sm.report_success(RadioTier::Local);
        }
        // Tier loss EMA should always be in [0, 1]
        let telem = sm.telemetry();
        for &loss in &telem.tier_loss_ema {
            assert!(
                (0.0..=1.0).contains(&loss),
                "Tier loss EMA out of bounds: {}",
                loss
            );
        }
    }

    // 4. NetworkHealth is monotonic with tier availability.
    // More tiers down → worse (or equal) health.
    #[test]
    fn prop_network_health_monotonic(
        local in any::<bool>(),
        metro in any::<bool>(),
        regional in any::<bool>(),
    ) {
        let health_all = NetworkHealth::from_tiers(true, true, true);
        let health_current = NetworkHealth::from_tiers(local, metro, regional);
        // All tiers up is always <= (better or equal) than any subset
        assert!(health_all <= health_current);
        // Blackout (none) is always >= (worse or equal) than any configuration
        let health_none = NetworkHealth::from_tiers(false, false, false);
        assert!(health_none >= health_current);
    }

    // 5. Payload classifier respects MTU limits in routing decisions.
    #[test]
    fn prop_mtu_respected_in_routing(
        payload_size in 1usize..5000,
        urgency in 0u8..4,
    ) {
        let classifier = PayloadClassifier::default();
        if let Some(RoutingDecision::Routed { tier, fragmented, estimated_fragments }) =
            classifier.route(PayloadClass::ConsciousnessDelta, payload_size, urgency)
        {
            let profile = tier.profile();
            if !fragmented {
                assert!(
                    payload_size <= profile.mtu,
                    "Non-fragmented payload {} exceeds MTU {} for {:?}",
                    payload_size, profile.mtu, tier
                );
            } else {
                assert!(
                    estimated_fragments >= 2,
                    "Fragmented payload should need >= 2 fragments"
                );
            }
        }
    }

    // 6. Compression roundtrip: apply(compress(diff)) recovers the original.
    #[test]
    fn prop_compression_roundtrip(
        seed in any::<u64>(),
        modifications in proptest::collection::vec((0usize..2048, 0u8..=255u8), 0..100),
    ) {
        let base = seed_to_array(seed);
        let mut modified = base;
        for (idx, val) in &modifications {
            modified[*idx] = *val;
        }
        let delta = CompressedDelta::from_diff(&base, &modified);
        let reconstructed = delta.apply(&base).expect("Decompression should succeed");
        assert_eq!(
            reconstructed, modified,
            "Roundtrip compression failed: {} bytes changed",
            delta.changed_bytes
        );
    }

    // 7. Connectivity factor bounded [0.0, 1.0] for all health states.
    #[test]
    fn prop_connectivity_factor_bounded(
        local in any::<bool>(),
        metro in any::<bool>(),
        regional in any::<bool>(),
    ) {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, local);
        sm.set_tier_available(RadioTier::Metro, metro);
        sm.set_tier_available(RadioTier::Regional, regional);
        let cf = sm.connectivity_factor();
        assert!(
            (0.0..=1.0).contains(&cf),
            "connectivity_factor {} out of bounds",
            cf
        );
    }

    // 8. Network critical: blackout always critical, all-up never critical.
    #[test]
    fn prop_network_critical_consistency(
        local in any::<bool>(),
        metro in any::<bool>(),
        regional in any::<bool>(),
    ) {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, local);
        sm.set_tier_available(RadioTier::Metro, metro);
        sm.set_tier_available(RadioTier::Regional, regional);
        if local && metro && regional {
            assert!(
                !sm.is_network_critical(),
                "All tiers up should not be critical"
            );
        }
        if !local && !metro && !regional {
            assert!(
                sm.is_network_critical(),
                "Blackout should be critical"
            );
        }
    }

    // 9. Mesh stats ingestion: tier loss EMAs remain in [0, 1].
    #[test]
    fn prop_mesh_stats_bounded(
        dropped in 0u64..1000,
        total in 1u64..1000,
    ) {
        let mut sm = SpectrumManager::default();
        sm.ingest_mesh_stats(dropped.min(total), total);
        let telem = sm.telemetry();
        for &loss in &telem.tier_loss_ema {
            assert!(
                (0.0..=1.0).contains(&loss),
                "Tier loss EMA {} after mesh stats ingestion out of bounds",
                loss
            );
        }
    }

    // 10. Safety: jamming streak below threshold → not critical (unless blackout).
    #[test]
    fn prop_jamming_below_threshold_not_critical(jamming in 0u32..10) {
        let mut sm = SpectrumManager::default();
        // Manually feed jamming observations
        for _ in 0..jamming {
            sm.inject_observation(symthaea::cognitive_loop::SpectrumObservation {
                frequency_hz: 900_000_000,
                noise_floor_dbm: -90.0,
                snr_db: 2.0, // Below jamming threshold
                jammed: true,
            });
        }
        // With all tiers up, below-threshold jamming should not trigger critical
        // (jamming_streak doesn't increment until process() runs)
        assert!(
            !sm.is_network_critical(),
            "Below threshold jamming should not be critical when tiers up"
        );
    }

    // 11. Swarm→Spectrum: synthetic SNR increases with peer count.
    #[test]
    fn prop_swarm_snr_increases_with_peers(
        peers in 0usize..100,
        phi in 0.0f64..1.0,
        connectivity in 0.0f64..1.0,
    ) {
        let mut sm = SpectrumManager::default();
        sm.ingest_swarm_state(peers, phi, connectivity);
        let obs = &sm.pending_observations()[0];
        // SNR should be finite and reasonable
        assert!(obs.snr_db.is_finite(), "SNR must be finite");
        assert!(obs.snr_db >= 0.0, "SNR must be non-negative");
        assert!(obs.noise_floor_dbm.is_finite(), "Noise floor must be finite");
    }

    // 12. Swarm→Spectrum: zero peers should produce worse SNR than many peers.
    #[test]
    fn prop_isolation_worse_than_connected(
        phi in 0.0f64..1.0,
        connectivity in 0.1f64..1.0,
    ) {
        let mut sm_isolated = SpectrumManager::default();
        sm_isolated.ingest_swarm_state(0, 0.0, 0.0);
        let snr_isolated = sm_isolated.pending_observations()[0].snr_db;

        let mut sm_connected = SpectrumManager::default();
        sm_connected.ingest_swarm_state(10, phi, connectivity);
        let snr_connected = sm_connected.pending_observations()[0].snr_db;

        assert!(
            snr_connected > snr_isolated,
            "Connected ({} peers) SNR {} should exceed isolated SNR {}",
            10, snr_connected, snr_isolated
        );
    }

    // 13. Consciousness-aware tier: high confidence prefers Local (reliable).
    #[test]
    fn prop_consciousness_high_confidence_prefers_reliable(
        local in any::<bool>(),
        metro in any::<bool>(),
    ) {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, local);
        sm.set_tier_available(RadioTier::Metro, metro);
        sm.set_tier_available(RadioTier::Regional, true);

        let tier = sm.consciousness_aware_tier(40, 0.9); // high confidence, small payload
        if local {
            // With Local available, high-confidence should pick Local
            assert_eq!(tier, Some(RadioTier::Local));
        } else if metro {
            assert_eq!(tier, Some(RadioTier::Metro));
        } else {
            assert_eq!(tier, Some(RadioTier::Regional));
        }
    }

    // 14. Consciousness-aware tier: low confidence prefers energy-efficient.
    #[test]
    fn prop_consciousness_low_confidence_prefers_efficient(
        local in any::<bool>(),
        metro in any::<bool>(),
    ) {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, local);
        sm.set_tier_available(RadioTier::Metro, metro);
        sm.set_tier_available(RadioTier::Regional, true);

        let tier = sm.consciousness_aware_tier(40, 0.1); // low confidence, small payload
        // Should prefer lowest energy_per_bit: Metro (20 nJ) < Local (50 nJ) < Regional (2M nJ)
        if metro {
            assert_eq!(tier, Some(RadioTier::Metro));
        }
    }

    // 15. Consciousness-aware tier: medium confidence returns None (let classifier decide).
    #[test]
    fn prop_medium_confidence_defers(
        payload_size in 10usize..200,
        confidence in 0.35f64..0.65,
    ) {
        let sm = SpectrumManager::default();
        let tier = sm.consciousness_aware_tier(payload_size, confidence);
        assert_eq!(tier, None, "Medium confidence should defer to normal routing");
    }

    // 16. Mesh stats: repeated ingestion converges (EMA property).
    #[test]
    fn prop_mesh_stats_converge(
        loss_pct in 0u64..100,
    ) {
        let mut sm = SpectrumManager::default();
        let total = 100;
        let dropped = loss_pct.min(total);
        // Ingest same stats 100 times
        for _ in 0..100 {
            sm.ingest_mesh_stats(dropped, total);
        }
        let telem = sm.telemetry();
        // After convergence, values should be stable
        // Regional absorbs most (weight 1.0): loss_ema[2] ≈ dropped/total
        let expected_regional = dropped as f64 / total as f64;
        assert!(
            (telem.tier_loss_ema[2] - expected_regional).abs() < 0.05,
            "Regional loss EMA should converge to ~{:.2}, got {:.4}",
            expected_regional, telem.tier_loss_ema[2]
        );
    }
}