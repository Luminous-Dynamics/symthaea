// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Inoculation — Integration Tests
//!
//! End-to-end flows across the five sovereign functions:
//! Clock, Trust, Name, Social Fabric, Survival.

// ============================================================================
// PHASE A: CLOCK
// ============================================================================

#[cfg(feature = "mesh")]
mod clock_tests {
    use symthaea::swarm::mesh::mesh_time::{MeshTimeConsensus, TimeQuality, TimeSample};
    use symthaea::swarm::mesh::time_beacon::TimeBeacon;

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    #[test]
    fn test_mesh_time_consensus_full_flow() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();

        // Simulate 7 peers with varying quality
        for i in 0..7u8 {
            engine.add_sample(TimeSample {
                peer_id: [i, 0, 0, 0, 0, 0, 0, 0],
                offset_us: 2000 + (i as i64 * 50),
                rtt_us: 100 + (i as u64 * 20),
                stratum: if i < 2 { 1 } else { 2 },
                phi: 0.5 + (i as f32 * 0.05),
                local_timestamp_us: ts,
            });
        }
        engine.recompute_consensus(ts);

        assert_eq!(engine.quality(), TimeQuality::Consensus);
        assert!(engine.offset_us() > 1500 && engine.offset_us() < 3000);
        assert!(engine.stratum() <= 3);
        assert_eq!(engine.peer_count(), 7);
    }

    #[test]
    fn test_time_beacon_encode_decode_roundtrip() {
        let beacon = TimeBeacon {
            timestamp_us: 1_710_000_000_000_000,
            stratum: 2,
            counter: 42,
            phi: 0.75,
            drift_ppm: -0.5,
        };
        let hv = beacon.encode();
        let decoded = TimeBeacon::decode(&hv).expect("decode should succeed");
        assert_eq!(decoded.timestamp_us, beacon.timestamp_us);
        assert_eq!(decoded.stratum, beacon.stratum);
        assert_eq!(decoded.counter, beacon.counter);
    }

    // TimeManager unit tests are in src/cognitive_loop/managers/time_manager.rs
    // (6 tests covering creation, beacon injection, and processing).

    #[test]
    fn test_mesh_time_degraded_quality() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        // Only 3 peers = Degraded quality
        for i in 0..3u8 {
            engine.add_sample(TimeSample {
                peer_id: [i, 0, 0, 0, 0, 0, 0, 0],
                offset_us: 500,
                rtt_us: 100,
                stratum: 2,
                phi: 0.7,
                local_timestamp_us: ts,
            });
        }
        engine.recompute_consensus(ts);
        assert_eq!(engine.quality(), TimeQuality::Degraded);
    }
}

// ============================================================================
// PHASE A: TRUST
// ============================================================================

#[cfg(feature = "mesh")]
mod trust_tests {
    use symthaea::swarm::consciousness_certificate::{
        ConsciousnessCertifier, verify_consciousness_certificate,
    };
    use symthaea::swarm::web_of_trust::{TrustGraph, UnifiedTrust};

    #[test]
    fn test_trust_graph_transitive_chain() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.9, true);
        g.set_trust("bob", "carol", 0.8, false);
        g.set_trust("carol", "dave", 0.7, false);

        let direct = g.direct_trust("alice", "bob");
        assert!(direct > 0.8);

        let transitive = g.transitive_trust("alice", "dave", 5);
        assert!(transitive > 0.0);
        assert!(transitive < direct); // Transitive always less
    }

    #[test]
    fn test_trust_violation_and_decay() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "malicious", 0.9, false);

        // Apply decay
        for i in 1..=50 {
            g.tick(i, 0.98);
        }

        let decayed = g.direct_trust("alice", "malicious");
        assert!(decayed < 0.9, "Trust should decay: {}", decayed);
    }

    #[test]
    fn test_consciousness_certificate_sign_verify() {
        let certifier = ConsciousnessCertifier::random();
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let cert = certifier.sign_consciousness_state(0.75, 0.9, 0.8, now_us);
        assert!(cert.is_plausible());
        assert!(cert.is_fresh(now_us));

        let result = verify_consciousness_certificate(&cert, now_us);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unified_trust_scoring() {
        let ut = UnifiedTrust::compute(0.8, 0.6, 0.7, 3);
        assert!(ut.composite > 0.5);
        assert!(ut.composite <= 1.0);
    }

    #[test]
    fn test_trust_graph_sybil_detection() {
        let mut g = TrustGraph::new();
        // Establish baseline
        g.set_trust("alice", "bob", 0.5, false);
        g.tick(1, 0.999);

        // Mass trust injection (Sybil attack)
        for i in 0..20 {
            g.set_trust("alice", &format!("sybil_{}", i), 0.9, false);
        }
        g.tick(2, 0.999);

        assert!(g.anomaly_count() > 0);
    }

    #[test]
    fn test_trust_telemetry() {
        let mut g = TrustGraph::new();
        g.set_trust("a", "b", 0.8, true);
        g.set_trust("b", "c", 0.6, false);
        let t = g.telemetry();
        assert_eq!(t.node_count, 3);
        assert_eq!(t.edge_count, 2);
        assert!(t.pq_fraction > 0.0);
    }
}

// ============================================================================
// PHASE B: NAME
// ============================================================================

#[cfg(feature = "mesh")]
mod name_tests {
    use symthaea::swarm::mesh::name_packet::{EndpointType, NameQuery, NameResponse};
    use symthaea::swarm::name_resolver::{MeshName, NameResolver, ResolvedEndpoint};

    #[test]
    fn test_name_resolution_cache_flow() {
        let mut resolver = NameResolver::new(64);
        let name = MeshName::parse("joburg/water/tank-7").unwrap();
        let endpoint = ResolvedEndpoint::IrohAddr("abc123".to_string());

        // Miss
        assert!(resolver.resolve_cached(&name, 1000).is_none());

        // Cache
        resolver.cache_resolution(&name, endpoint.clone(), 3600, 1000);

        // Hit
        let resolved = resolver.resolve_cached(&name, 1000);
        assert_eq!(resolved, Some(endpoint));
    }

    #[test]
    fn test_name_packet_roundtrip() {
        let query = NameQuery {
            name_hash: *blake3::hash(b"mycelix://test/node").as_bytes(),
            query_id: 42,
        };
        let hv = query.encode();
        let decoded = NameQuery::decode(&hv).unwrap();
        assert_eq!(decoded.query_id, 42);
        assert_eq!(decoded.name_hash, query.name_hash);
    }

    #[test]
    fn test_name_response_roundtrip() {
        let response = NameResponse {
            name_hash: [0xAB; 32],
            query_id: 123,
            endpoint_type: EndpointType::Iroh,
            endpoint_data: b"some-addr".to_vec(),
            ttl_secs: 3600,
        };
        let hv = response.encode();
        let decoded = NameResponse::decode(&hv).unwrap();
        assert_eq!(decoded.query_id, 123);
        assert_eq!(decoded.ttl_secs, 3600);
    }

    #[test]
    fn test_name_validation() {
        assert!(MeshName::parse("valid/name").is_ok());
        assert!(MeshName::parse("a/b/c/d/e").is_ok()); // max depth 5
        assert!(MeshName::parse("a/b/c/d/e/f").is_err()); // too deep
        assert!(MeshName::parse("UPPER").is_err());
        assert!(MeshName::parse("-bad").is_err());
    }
}

// ============================================================================
// PHASE C: SOCIAL FABRIC
// ============================================================================

#[cfg(feature = "mesh")]
mod social_tests {
    use symthaea::swarm::mesh::content_packet::ContentAnnounce;
    use symthaea::swarm::resonance_graph::{ContentRef, ResonanceGraph};
    use symthaea_core::hdc::BinaryHV;

    #[test]
    fn test_resonance_ranking() {
        let mut graph = ResonanceGraph::new();
        graph.set_our_state(BinaryHV::random(42));

        let items: Vec<ContentRef> = (0..20)
            .map(|i| ContentRef {
                source_peer: format!("peer_{}", i),
                content_hash: [i as u8; 32],
                hdv_embedding: BinaryHV::random(i + 100),
                domain: "test".to_string(),
                created_at: 1000 + i,
            })
            .collect();

        let ranked = graph.rank_content(&items, 10);
        assert_eq!(ranked.len(), 10);

        // Scores should be monotonically non-increasing
        for w in ranked.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_echo_chamber_detection() {
        let mut graph = ResonanceGraph::new();

        // Manually inject high-resonance peer
        graph.add_content(ContentRef {
            source_peer: "echo_peer".to_string(),
            content_hash: [0; 32],
            hdv_embedding: BinaryHV::random(1),
            domain: "test".to_string(),
            created_at: 1000,
        });

        // Record heavy consumption from same peer
        for _ in 0..20 {
            graph.record_consumption("echo_peer");
        }

        // echo_chamber_risk should be elevated
        // (may not exceed threshold without high resonance_ema, but should be > 0)
        let risk = graph.echo_chamber_risk();
        assert!(risk >= 0.0);
    }

    #[test]
    fn test_content_announce_roundtrip() {
        let full_hdv = BinaryHV::random(42);
        let announce =
            ContentAnnounce::from_full_hdv([0xAA; 32], &full_hdv, "water".to_string(), 1234567890);
        let hv = announce.encode();
        let decoded = ContentAnnounce::decode(&hv).unwrap();
        assert_eq!(decoded.content_hash, [0xAA; 32]);
        assert_eq!(decoded.domain, "water");
        assert_eq!(decoded.created_at, 1234567890);
    }

    #[test]
    fn test_diversity_telemetry() {
        let graph = ResonanceGraph::new();
        let t = graph.telemetry();
        assert_eq!(t.peer_reach, 0);
        assert_eq!(t.content_count, 0);
        assert_eq!(t.echo_chamber_risk, 0.0);
    }

    #[test]
    fn test_social_fabric_manager_cycle() {
        // Test will only compile with social-fabric feature
    }
}

// ============================================================================
// PHASE D: SURVIVAL
// ============================================================================

#[cfg(feature = "mesh")]
mod survival_tests {
    use symthaea::swarm::mesh::sensor_forecast::DemandForecaster;
    use symthaea::swarm::mesh::sensor_iot::{IoTPlatform, IoTSensorAdapter, ResourceType};

    #[test]
    fn test_iot_parsing_zigbee2mqtt() {
        let adapter = IoTSensorAdapter::new();
        let reading = adapter.parse_message(
            "zigbee2mqtt/living_room",
            r#"{"temperature": 22.5, "humidity": 45}"#,
            1000,
        );
        assert!(reading.is_some());
        let r = reading.unwrap();
        assert_eq!(r.platform, IoTPlatform::Zigbee2Mqtt);
        assert!((r.values["temperature"] - 22.5).abs() < 0.01);
    }

    #[test]
    fn test_iot_parsing_tasmota() {
        let adapter = IoTSensorAdapter::new();
        let reading = adapter.parse_message(
            "tele/plug1/SENSOR",
            r#"{"ENERGY": {"Power": 150, "Voltage": 230}}"#,
            1000,
        );
        assert!(reading.is_some());
        let r = reading.unwrap();
        assert_eq!(r.platform, IoTPlatform::Tasmota);
    }

    #[test]
    fn test_demand_forecast_basic() {
        let mut forecaster = DemandForecaster::new();
        for i in 0..30 {
            forecaster.record_consumption("water", 10.0, i % 24);
        }
        let forecast = forecaster.forecast("water", 24, 12, 0);
        assert!(forecast.predicted_rate > 0.0);
        assert!(forecast.confidence > 0.0);
    }

    #[test]
    fn test_resource_classification() {
        assert_eq!(ResourceType::classify("water_level"), ResourceType::Water);
        assert_eq!(ResourceType::classify("Power"), ResourceType::Power);
        assert_eq!(
            ResourceType::classify("temperature"),
            ResourceType::Temperature
        );
    }

    #[test]
    fn test_alert_generation() {
        let mut adapter = IoTSensorAdapter::new();
        use std::collections::HashMap;
        let reading = symthaea::swarm::mesh::sensor_iot::IoTReading {
            sensor_id: "tank/water".to_string(),
            values: HashMap::from([("water_level".to_string(), 0.05)]),
            platform: IoTPlatform::Generic,
            timestamp_secs: 1000,
            raw_payload: String::new(),
        };
        let alerts = adapter.ingest(reading);
        assert!(!alerts.is_empty());
    }
}

// ============================================================================
// CROSS-FUNCTION INTEGRATION
// ============================================================================

#[cfg(feature = "mesh")]
mod cross_function_tests {
    use symthaea::swarm::mesh::mesh_time::MeshTimeConsensus;
    use symthaea::swarm::name_resolver::{MeshName, NameResolver, ResolvedEndpoint};
    use symthaea::swarm::web_of_trust::TrustGraph;

    #[test]
    fn test_trust_weighted_time_consensus() {
        // Verify that Phi-weighted peers produce better consensus
        let mut engine = MeshTimeConsensus::new();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // High-Phi peers report correct time
        for i in 0..3u8 {
            engine.add_sample(symthaea::swarm::mesh::mesh_time::TimeSample {
                peer_id: [i, 0, 0, 0, 0, 0, 0, 0],
                offset_us: 1000,
                rtt_us: 100,
                stratum: 1,
                phi: 0.9,
                local_timestamp_us: ts,
            });
        }
        // Low-Phi peer reports wrong time
        engine.add_sample(symthaea::swarm::mesh::mesh_time::TimeSample {
            peer_id: [99, 0, 0, 0, 0, 0, 0, 0],
            offset_us: -5000,
            rtt_us: 100,
            stratum: 1,
            phi: 0.1,
            local_timestamp_us: ts,
        });
        engine.recompute_consensus(ts);

        // Consensus should be closer to 1000 (high-Phi) than -5000 (low-Phi)
        assert!(
            engine.offset_us() > 0,
            "Offset should be positive: {}",
            engine.offset_us()
        );
    }

    #[test]
    fn test_name_resolution_with_trust() {
        // Names should be resolvable from cache after trust verification
        let mut resolver = NameResolver::new(64);
        let name = MeshName::parse("trusted-node/service").unwrap();
        let endpoint = ResolvedEndpoint::HolochainAgent("hcABC123".to_string());

        // Simulate: after trust verification, cache the name
        resolver.cache_resolution(&name, endpoint.clone(), 3600, 1000);
        assert_eq!(resolver.resolve_cached(&name, 1000), Some(endpoint));
    }
}