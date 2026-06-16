// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Inoculation — Property Tests
//!
//! Property-based tests for invariants across the five sovereign functions.

#[cfg(feature = "mesh")]
mod proptest_sovereign {
    use proptest::prelude::*;

    // ── Clock Properties ─────────────────────────────────────────────────

    proptest! {
        /// Mesh time offset must be bounded by the range of peer offsets.
        #[test]
        fn prop_consensus_bounded(
            offsets in proptest::collection::vec(-1_000_000i64..1_000_000, 3..20),
            phis in proptest::collection::vec(0.0f32..1.0, 3..20),
        ) {
            use symthaea::swarm::mesh::mesh_time::{MeshTimeConsensus, TimeSample};

            let n = offsets.len().min(phis.len());
            if n < 3 { return Ok(()); }

            let mut engine = MeshTimeConsensus::new();
            let ts = 1_000_000_000u64;
            for i in 0..n {
                engine.add_sample(TimeSample {
                    peer_id: [i as u8, 0, 0, 0, 0, 0, 0, 0],
                    offset_us: offsets[i],
                    rtt_us: 100,
                    stratum: 1,
                    phi: phis[i],
                    local_timestamp_us: ts,
                });
            }
            engine.recompute_consensus(ts);

            let min = *offsets[..n].iter().min().unwrap();
            let max = *offsets[..n].iter().max().unwrap();
            let offset = engine.offset_us();
            // Consensus should be within the range of inputs (with some tolerance)
            prop_assert!(offset >= min - 1000 && offset <= max + 1000,
                "Consensus {} outside range [{}, {}]", offset, min, max);
        }

        /// Mesh time must be monotonically non-decreasing within one engine.
        #[test]
        fn prop_mesh_now_monotonic(
            _dummy in 0u32..1,
        ) {
            use symthaea::swarm::mesh::mesh_time::MeshTimeConsensus;
            let engine = MeshTimeConsensus::new();
            let t1 = engine.mesh_now();
            let t2 = engine.mesh_now();
            prop_assert!(t2 >= t1, "mesh_now should be monotonic: {} < {}", t2, t1);
        }

        /// Beacon counter must always increase.
        #[test]
        fn prop_beacon_counter_monotonic(
            n in 1usize..100,
        ) {
            use symthaea::swarm::mesh::mesh_time::MeshTimeConsensus;
            let mut engine = MeshTimeConsensus::new();
            let mut prev = 0u64;
            for _ in 0..n {
                let counter = engine.next_beacon_counter();
                prop_assert!(counter > prev, "Counter not monotonic: {} <= {}", counter, prev);
                prev = counter;
            }
        }
    }

    // ── Trust Properties ─────────────────────────────────────────────────

    proptest! {
        /// Trust scores must remain in [0.0, 1.0].
        #[test]
        fn prop_trust_bounded(
            direct in -2.0f64..3.0,
            transitive in -2.0f64..3.0,
            reputation in -2.0f64..3.0,
            tier in 0u8..5,
        ) {
            use symthaea::swarm::web_of_trust::UnifiedTrust;
            let ut = UnifiedTrust::compute(direct, transitive, reputation, tier);
            prop_assert!(ut.composite >= 0.0 && ut.composite <= 1.0,
                "Composite trust out of bounds: {}", ut.composite);
        }

        /// Transitive trust must decay with hops.
        #[test]
        fn prop_transitive_decays(
            trust in 0.1f64..1.0,
            hops in 2usize..5,
        ) {
            use symthaea::swarm::web_of_trust::TrustGraph;
            let mut g = TrustGraph::new();
            // Create chain: a→b→c→...
            let nodes: Vec<String> = (0..=hops).map(|i| format!("n{}", i)).collect();
            for i in 0..hops {
                g.set_trust(&nodes[i], &nodes[i + 1], trust, false);
            }
            let direct = g.direct_trust(&nodes[0], &nodes[1]);
            let transitive = g.transitive_trust(&nodes[0], &nodes[hops], hops);
            // Transitive trust should be less than direct trust
            prop_assert!(transitive <= direct + 0.01,
                "Transitive {} should be <= direct {}", transitive, direct);
        }
    }

    // ── Name Properties ──────────────────────────────────────────────────

    proptest! {
        /// Valid names must roundtrip through parse → canonical → parse.
        #[test]
        fn prop_name_roundtrip(
            segments in proptest::collection::vec("[a-z0-9]{1,10}", 1..5),
        ) {
            use symthaea::swarm::name_resolver::MeshName;
            // Filter out segments starting/ending with '-'
            let segments: Vec<String> = segments.into_iter()
                .filter(|s| !s.starts_with('-') && !s.ends_with('-') && !s.is_empty())
                .collect();
            if segments.is_empty() { return Ok(()); }

            let joined = segments.join("/");
            let name = MeshName::parse(&joined).unwrap();
            let canonical = name.to_canonical();
            let reparsed = MeshName::parse(&canonical).unwrap();
            prop_assert_eq!(name.segments, reparsed.segments);
        }

        /// Name hash must be deterministic.
        #[test]
        fn prop_name_hash_deterministic(
            seg in "[a-z]{1,10}",
        ) {
            use symthaea::swarm::name_resolver::MeshName;
            let name = MeshName::parse(&seg).unwrap();
            let h1 = name.hash();
            let h2 = name.hash();
            prop_assert_eq!(h1, h2);
        }
    }

    // ── Social Fabric Properties ──────────────────────────────────────────

    proptest! {
        /// Resonance scores must be in [0.0, 1.0+bonus].
        #[test]
        fn prop_resonance_bounded(
            _dummy in 0u32..1,
        ) {
            use symthaea::swarm::resonance_graph::ResonanceGraph;
            use symthaea_core::hdc::BinaryHV;

            let mut graph = ResonanceGraph::new();
            graph.set_our_state(BinaryHV::random(42));

            let items: Vec<_> = (0..10).map(|i| symthaea::swarm::resonance_graph::ContentRef {
                source_peer: format!("p{}", i),
                content_hash: [i as u8; 32],
                hdv_embedding: BinaryHV::random(i + 100),
                domain: "test".to_string(),
                created_at: 0,
            }).collect();

            let ranked = graph.rank_content(&items, 10);
            for r in &ranked {
                prop_assert!(r.resonance >= 0.0 && r.resonance <= 1.0,
                    "Resonance out of bounds: {}", r.resonance);
                prop_assert!(r.score >= 0.0,
                    "Score should be non-negative: {}", r.score);
            }
        }
    }

    // ── Survival Properties ──────────────────────────────────────────────

    proptest! {
        /// Resource type classification must always produce a valid variant.
        #[test]
        fn prop_resource_classification_total(
            key in ".*",
        ) {
            use symthaea::swarm::mesh::sensor_iot::ResourceType;
            let _ = ResourceType::classify(&key);
            // Should never panic — classification is total
        }

        /// Forecast confidence must be in [0.0, 1.0].
        #[test]
        fn prop_forecast_confidence_bounded(
            rate in 0.0f64..1000.0,
            n in 1usize..50,
        ) {
            use symthaea::swarm::mesh::sensor_forecast::DemandForecaster;
            let mut fc = DemandForecaster::new();
            for i in 0..n {
                fc.record_consumption("test", rate, i % 24);
            }
            let f = fc.forecast("test", 24, 12, 0);
            prop_assert!(f.confidence >= 0.0 && f.confidence <= 1.0,
                "Confidence out of bounds: {}", f.confidence);
        }
    }
}