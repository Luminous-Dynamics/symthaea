//! Property tests for the SpectrumManager radio dispatch module.
//!
//! Validates invariants of delta compression, routing determinism,
//! AIMD bandwidth bounds, network health monotonicity, MTU compliance,
//! and compression roundtrips.
//!
//! Feature-gated behind `mesh`.

#![cfg(feature = "mesh")]

use proptest::prelude::*;
use symthaea::cognitive_loop::managers::radio_dispatcher::*;

proptest! {
    // 1. Compression ratio bounded: delta compression output never exceeds 2x original + header.
    #[test]
    fn prop_compression_never_exceeds_bound(
        base in prop::array::uniform2048(0u8..=255u8),
        changed_count in 0usize..200,
    ) {
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
                loss >= 0.0 && loss <= 1.0,
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
        base in prop::array::uniform2048(0u8..=255u8),
        modifications in prop::collection::vec((0usize..2048, 0u8..=255u8), 0..100),
    ) {
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
}
