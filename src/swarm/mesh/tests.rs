// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;

fn test_hv(seed: u8) -> BinaryHV {
    let mut bytes = [0u8; 2048];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = seed.wrapping_mul(i as u8).wrapping_add((i >> 3) as u8);
    }
    BinaryHV(bytes)
}

#[test]
fn wisdom_packet_roundtrip() {
    let packet = WisdomPacket {
        source_id: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        sequence: 42,
        phi: 0.73,
        urgency: MeshUrgency::Cruise,
        timestamp_s: 1_700_000_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0xFF),
    };

    let bytes = packet.to_bytes();
    assert_eq!(bytes.len(), WISDOM_PACKET_SIZE);
    assert_eq!(bytes[0], WISDOM_PACKET_VERSION);

    let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.source_id, packet.source_id);
    assert_eq!(decoded.sequence, 42);
    assert!((decoded.phi - 0.73).abs() < 1e-6);
    assert_eq!(decoded.urgency, MeshUrgency::Cruise);
    assert_eq!(decoded.timestamp_s, 1_700_000_000);
    assert_eq!(decoded.payload_type, PayloadType::WisdomVector);
    assert_eq!(decoded.wisdom.0, packet.wisdom.0);
}

#[test]
fn wisdom_packet_fragment_count() {
    let packet = WisdomPacket {
        source_id: [0; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0xAA),
    };

    let frags = packet.fragment();
    // 2104 / 214 = 9.84 → 10 data + 1 FEC = 11
    assert_eq!(frags.len(), 11);
}

#[test]
fn wisdom_packet_full_radio_roundtrip() {
    let original = WisdomPacket {
        source_id: [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
        sequence: 1337,
        phi: 0.91,
        urgency: MeshUrgency::Critical,
        timestamp_s: 1_708_000_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x42),
    };

    // Fragment
    let frags = original.fragment();

    // Simulate radio: serialize, drop fragment 6, deserialize
    let mut assembler = WisdomPacket::assembler(original.thought_id(), 11);
    let mut buf = [0u8; LORA_MTU];

    for (i, frag) in frags.iter().enumerate() {
        if i == 6 {
            continue; // lost in transit
        }
        let len = frag.to_bytes(&mut buf);
        let decoded = LoRaFragment::from_bytes(&buf[..len]).unwrap();
        assembler.feed(&decoded);
    }

    assert!(assembler.is_complete());
    let recovered = WisdomPacket::from_assembler(&assembler).unwrap();

    assert_eq!(recovered.source_id, original.source_id);
    assert_eq!(recovered.sequence, 1337);
    assert!((recovered.phi - 0.91).abs() < 1e-6);
    assert_eq!(recovered.urgency, MeshUrgency::Critical);
    assert_eq!(recovered.wisdom.0, original.wisdom.0);
}

#[test]
fn mesh_urgency_byte_roundtrip() {
    assert_eq!(MeshUrgency::from_byte(0), MeshUrgency::Cruise);
    assert_eq!(MeshUrgency::from_byte(1), MeshUrgency::Normal);
    assert_eq!(MeshUrgency::from_byte(2), MeshUrgency::Critical);
    assert_eq!(MeshUrgency::from_byte(255), MeshUrgency::Critical);
}

#[test]
fn payload_type_byte_roundtrip() {
    assert_eq!(PayloadType::from_byte(0), PayloadType::WisdomVector);
    assert_eq!(PayloadType::from_byte(1), PayloadType::Affective);
    assert_eq!(PayloadType::from_byte(2), PayloadType::Heartbeat);
    assert_eq!(PayloadType::from_byte(3), PayloadType::Gradient);
}

#[test]
fn wisdom_packet_too_short_rejected() {
    assert!(WisdomPacket::from_bytes(&[0; 100]).is_none());
    assert!(WisdomPacket::from_bytes(&[0; WISDOM_PACKET_SIZE - 1]).is_none());
    assert!(WisdomPacket::from_bytes(&[0; WISDOM_PACKET_SIZE + 1]).is_none());
}

#[test]
fn wisdom_packet_rejects_legacy_or_unknown_wire_version() {
    let packet = WisdomPacket {
        source_id: [0; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(1),
    };
    let mut bytes = packet.to_bytes();
    bytes[0] = 1;
    assert!(WisdomPacket::from_bytes(&bytes).is_none());
}

#[test]
fn cycle_urgency_to_mesh_urgency() {
    use crate::cognitive_loop::types::CycleUrgency;

    assert_eq!(
        MeshUrgency::from(CycleUrgency::Critical),
        MeshUrgency::Critical
    );
    assert_eq!(MeshUrgency::from(CycleUrgency::Normal), MeshUrgency::Normal);
    assert_eq!(MeshUrgency::from(CycleUrgency::Cruise), MeshUrgency::Cruise);
}

// ====================================================================
// extract_affective Tests
// ====================================================================

/// Build an Affective WisdomPacket with specific VAD floats in the wisdom field.
fn affective_packet(
    valence: f32,
    arousal: f32,
    dominance: f32,
    intensity: f32,
    confidence: f32,
) -> WisdomPacket {
    let mut wisdom_bytes = [0u8; 2048];
    wisdom_bytes[0..4].copy_from_slice(&valence.to_le_bytes());
    wisdom_bytes[4..8].copy_from_slice(&arousal.to_le_bytes());
    wisdom_bytes[8..12].copy_from_slice(&dominance.to_le_bytes());
    wisdom_bytes[12..16].copy_from_slice(&intensity.to_le_bytes());
    wisdom_bytes[16..20].copy_from_slice(&confidence.to_le_bytes());
    WisdomPacket {
        source_id: [0xAF; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::Affective,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV(wisdom_bytes),
    }
}

#[test]
fn test_extract_affective_valid() {
    let pkt = affective_packet(0.5, 0.7, -0.3, 0.8, 0.95);
    let affect = pkt.extract_affective().unwrap();
    assert!((affect.valence - 0.5).abs() < 1e-6);
    assert!((affect.arousal - 0.7).abs() < 1e-6);
    assert!((affect.dominance - (-0.3)).abs() < 1e-6);
    assert!((affect.intensity - 0.8).abs() < 1e-6);
    assert!((affect.confidence - 0.95).abs() < 1e-6);
    assert_eq!(affect.timestamp_ms, 1_700_000_000);
    assert_eq!(affect.sequence, 1);
}

#[test]
fn test_extract_affective_wrong_type() {
    let mut pkt = affective_packet(0.5, 0.7, -0.3, 0.8, 0.95);
    pkt.payload_type = PayloadType::WisdomVector;
    assert!(pkt.extract_affective().is_none());
}

#[test]
fn test_extract_affective_nan_rejected() {
    let pkt = affective_packet(f32::NAN, 0.7, -0.3, 0.8, 0.95);
    assert!(pkt.extract_affective().is_none());
}

#[test]
fn test_extract_affective_clamped() {
    // Values outside range should be clamped
    let pkt = affective_packet(2.0, -1.0, 5.0, 3.0, -0.5);
    let affect = pkt.extract_affective().unwrap();
    assert!((affect.valence - 1.0).abs() < 1e-6);
    assert!((affect.arousal - 0.0).abs() < 1e-6);
    assert!((affect.dominance - 1.0).abs() < 1e-6);
    assert!((affect.intensity - 1.0).abs() < 1e-6);
    assert!((affect.confidence - 0.0).abs() < 1e-6);
}

// ====================================================================
// extract_gradient / from_gradient Tests
// ====================================================================

#[test]
fn test_extract_gradient_valid() {
    let msg = crate::swarm::GradientMessage {
        source_id: [0u8; 32],
        gradient_data: vec![0.1, -0.2, 0.3, 0.0],
        trust_score: 0.85,
        noise_scale: 0.0,
        timestamp: 1_700_000_000_000,
        sample_count: 100,
        model_version: 3,
    };
    let pkt = WisdomPacket::from_gradient([0xAB; 8], 42, &msg).unwrap();
    let extracted = pkt.extract_gradient().unwrap();

    assert_eq!(extracted.gradient_data.len(), 4);
    assert!((extracted.gradient_data[0] - 0.1).abs() < 1e-6);
    assert!((extracted.gradient_data[1] - (-0.2)).abs() < 1e-6);
    assert!((extracted.gradient_data[2] - 0.3).abs() < 1e-6);
    assert!((extracted.gradient_data[3] - 0.0).abs() < 1e-6);
    assert!((extracted.trust_score - 0.85).abs() < 1e-6);
    assert_eq!(extracted.timestamp, 1_700_000_000_000);
    assert_eq!(extracted.sample_count, 100);
    assert_eq!(extracted.model_version, 3);
    // source_id should be zero-padded from 8 → 32 bytes
    assert_eq!(&extracted.source_id[..8], &[0xAB; 8]);
    assert_eq!(&extracted.source_id[8..], &[0u8; 24]);
}

#[test]
fn test_extract_gradient_wrong_type() {
    let pkt = WisdomPacket {
        source_id: [0; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV([0; 2048]),
    };
    assert!(pkt.extract_gradient().is_none());
}

#[test]
fn test_extract_gradient_nan_rejected() {
    // NaN in gradient data — manually build the packet to test extraction rejects it
    let mut bytes = [0u8; 2048];
    bytes[0..4].copy_from_slice(&3u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&0.5f32.to_le_bytes());
    bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
    bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
    bytes[24..32].copy_from_slice(&1u64.to_le_bytes());
    bytes[32..36].copy_from_slice(&0.1f32.to_le_bytes());
    bytes[36..40].copy_from_slice(&f32::NAN.to_le_bytes());
    bytes[40..44].copy_from_slice(&0.3f32.to_le_bytes());

    let pkt = WisdomPacket {
        source_id: [0; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::Gradient,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV(bytes),
    };
    assert!(pkt.extract_gradient().is_none());
}

#[test]
fn test_extract_gradient_nan_trust_rejected() {
    let mut bytes = [0u8; 2048];
    bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&f32::NAN.to_le_bytes()); // NaN trust
    bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
    bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
    bytes[24..32].copy_from_slice(&1u64.to_le_bytes());
    bytes[32..36].copy_from_slice(&0.1f32.to_le_bytes());

    let pkt = WisdomPacket {
        source_id: [0; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::Gradient,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV(bytes),
    };
    assert!(pkt.extract_gradient().is_none());
}

#[test]
fn test_extract_gradient_overflow_rejected() {
    // gradient_count claims 999 floats, but not enough bytes
    let mut bytes = [0u8; 2048];
    bytes[0..4].copy_from_slice(&999u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&0.5f32.to_le_bytes());
    bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
    bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
    bytes[24..32].copy_from_slice(&1u64.to_le_bytes());

    let pkt = WisdomPacket {
        source_id: [0; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::Gradient,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV(bytes),
    };
    assert!(pkt.extract_gradient().is_none());
}

#[test]
fn test_gradient_roundtrip() {
    let original = crate::swarm::GradientMessage {
        source_id: [0u8; 32],
        gradient_data: vec![1.0, -2.5, 0.001, 42.0, -0.0],
        trust_score: 0.73,
        noise_scale: 0.01, // not preserved (set to 0 on extraction)
        timestamp: 1_708_000_000_000,
        sample_count: 500,
        model_version: 7,
    };

    let pkt = WisdomPacket::from_gradient([0x42; 8], 99, &original).unwrap();
    assert_eq!(pkt.payload_type, PayloadType::Gradient);
    assert_eq!(pkt.sequence, 99);

    let extracted = pkt.extract_gradient().unwrap();
    assert_eq!(extracted.gradient_data.len(), original.gradient_data.len());
    for (a, b) in extracted.gradient_data.iter().zip(&original.gradient_data) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }
    assert!((extracted.trust_score - original.trust_score).abs() < 1e-6);
    assert_eq!(extracted.timestamp, original.timestamp);
    assert_eq!(extracted.sample_count, original.sample_count);
    assert_eq!(extracted.model_version, original.model_version);
    assert_eq!(extracted.noise_scale, 0.0); // not preserved
}

#[test]
fn test_from_gradient_too_large() {
    let msg = crate::swarm::GradientMessage {
        source_id: [0u8; 32],
        gradient_data: vec![0.0; 505], // 32 + 505*4 = 2052 > 2048
        trust_score: 0.5,
        noise_scale: 0.0,
        timestamp: 0,
        sample_count: 1,
        model_version: 1,
    };
    assert!(WisdomPacket::from_gradient([0; 8], 1, &msg).is_none());
}

// ====================================================================
// MeshPeerRegistry Tests
// ====================================================================

#[test]
fn test_peer_registry_tracks_packets() {
    let mut registry = MeshPeerRegistry::new();

    let pkt_a1 = WisdomPacket {
        source_id: [0xAA; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x01),
    };
    let pkt_a2 = WisdomPacket {
        source_id: [0xAA; 8],
        sequence: 2,
        phi: 0.6,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x02),
    };
    let pkt_b = WisdomPacket {
        source_id: [0xBB; 8],
        sequence: 1,
        phi: 0.9,
        urgency: MeshUrgency::Critical,
        timestamp_s: 0,
        payload_type: PayloadType::Heartbeat,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x03),
    };

    registry.update(&pkt_a1);
    registry.update(&pkt_a2);
    registry.update(&pkt_b);

    assert_eq!(registry.peer_count(), 2);
    let peers = registry.active_peers();
    let peer_a = peers.iter().find(|p| p.source_id == [0xAA; 8]).unwrap();
    assert_eq!(peer_a.packets_received, 2);
    assert!((peer_a.last_phi - 0.6).abs() < 1e-6);

    let peer_b = peers.iter().find(|p| p.source_id == [0xBB; 8]).unwrap();
    assert_eq!(peer_b.packets_received, 1);
}

#[test]
fn test_peer_registry_expire_stale() {
    let mut registry = MeshPeerRegistry::with_timeout(std::time::Duration::from_millis(10));

    let pkt = WisdomPacket {
        source_id: [0xCC; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Cruise,
        timestamp_s: 0,
        payload_type: PayloadType::Heartbeat,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x04),
    };
    registry.update(&pkt);
    assert_eq!(registry.peer_count(), 1);

    // Wait for entry to become stale (use generous margin for CI runners)
    std::thread::sleep(std::time::Duration::from_millis(50));
    let expired = registry.expire_stale();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0], [0xCC; 8]);
    assert_eq!(registry.peer_count(), 0);
}

#[test]
fn test_peer_registry_average_phi() {
    let mut registry = MeshPeerRegistry::new();

    // No peers → 0.0
    assert_eq!(registry.average_phi(), 0.0);

    let pkt1 = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 1,
        phi: 0.4,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x10),
    };
    let pkt2 = WisdomPacket {
        source_id: [0x02; 8],
        sequence: 1,
        phi: 0.8,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x20),
    };

    registry.update(&pkt1);
    registry.update(&pkt2);

    let avg = registry.average_phi();
    assert!((avg - 0.6).abs() < 1e-6, "Expected ~0.6, got {avg}");
}

// ====================================================================
// from_affective Roundtrip Test
// ====================================================================

// ====================================================================
// Health Score Tests
// ====================================================================

#[test]
fn test_health_score_no_activity() {
    let stats = MeshStats::default();
    assert_eq!(stats.health_score(0), 0.0);
    assert_eq!(stats.health_score(5), 0.0);
}

#[test]
fn test_health_score_healthy_mesh() {
    let stats = MeshStats {
        wisdom_sent: 50,
        wisdom_received: 48,
        heartbeats_sent: 20,
        heartbeats_received: 18,
        peers_expired: 1,
        ..Default::default()
    };
    let score = stats.health_score(5);
    // Connectivity: 5/5 = 1.0 → 0.40
    // Bidirectionality: min(70,66)/max(70,66) = 66/70 ≈ 0.943 → 0.377
    // Stability: 1 - 1/(1+66+1) ≈ 0.985 → 0.197
    // Total ≈ 0.97+
    assert!(score > 0.9, "Healthy mesh score should be high: {score}");
}

#[test]
fn test_health_score_send_only() {
    let stats = MeshStats {
        wisdom_sent: 100,
        heartbeats_sent: 50,
        ..Default::default()
    };
    let score = stats.health_score(3);
    // Connectivity: 3/5 = 0.6 → 0.24
    // Bidirectionality: 0 (no receives) → 0.0
    // Stability: 1 - 0/(0+0+1) = 1.0 → 0.20
    // Total = 0.44
    assert!(score < 0.5, "Send-only mesh should have low score: {score}");
    // But not zero — we do have connectivity
    assert!(score > 0.0, "Score should be > 0 with peers: {score}");
}

// ====================================================================
// from_affective Roundtrip Test
// ====================================================================

#[test]
fn test_from_affective_roundtrip() {
    let affect = crate::swarm::AffectiveState {
        valence: 0.6,
        arousal: 0.8,
        dominance: -0.3,
        intensity: 0.7,
        thermodynamic_load: 0.0,
        confidence: 0.95,
        timestamp_ms: 1_700_000_000,
        sequence: 42,
    };

    let pkt = WisdomPacket::from_affective([0xAF; 8], 42, &affect);
    assert_eq!(pkt.payload_type, PayloadType::Affective);
    assert_eq!(pkt.sequence, 42);

    let extracted = pkt.extract_affective().unwrap();
    assert!((extracted.valence - 0.6).abs() < 1e-6);
    assert!((extracted.arousal - 0.8).abs() < 1e-6);
    assert!((extracted.dominance - (-0.3)).abs() < 1e-6);
    assert!((extracted.intensity - 0.7).abs() < 1e-6);
    assert!((extracted.confidence - 0.95).abs() < 1e-6);
    assert_eq!(extracted.timestamp_ms, 1_700_000_000);
    assert_eq!(extracted.sequence, 42);
}

// ====================================================================
// Per-Peer Rate Limiting Tests
// ====================================================================

#[test]
fn test_rate_limit_allows_under_limit() {
    let mut registry = MeshPeerRegistry::new();
    let peer_id = [0xAA; 8];

    // Register peer first
    let pkt = WisdomPacket {
        source_id: peer_id,
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x01),
    };
    registry.update(&pkt);

    // 99 calls should all be under limit (window_count 1..=99)
    for _ in 0..99 {
        assert!(!registry.is_rate_limited(&peer_id), "Should be under limit");
    }
}

#[test]
fn test_rate_limit_blocks_over_limit() {
    let mut registry = MeshPeerRegistry::new();
    let peer_id = [0xBB; 8];

    let pkt = WisdomPacket {
        source_id: peer_id,
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x02),
    };
    registry.update(&pkt);

    // First 100 should pass (window_count 1..=100)
    for _ in 0..100 {
        registry.is_rate_limited(&peer_id);
    }
    // 101st should be blocked (window_count == 101 > 100)
    assert!(
        registry.is_rate_limited(&peer_id),
        "101st call should be rate limited"
    );
}

#[test]
fn test_rate_limit_window_resets() {
    let mut registry = MeshPeerRegistry::new();
    let peer_id = [0xCC; 8];

    let pkt = WisdomPacket {
        source_id: peer_id,
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x03),
    };
    registry.update(&pkt);

    // Exhaust the limit
    for _ in 0..101 {
        registry.is_rate_limited(&peer_id);
    }
    assert!(registry.is_rate_limited(&peer_id), "Should be rate limited");

    // Manually reset window_start to simulate time passage
    if let Some(entry) = registry.peers.get_mut(&peer_id) {
        entry.window_start = std::time::Instant::now() - std::time::Duration::from_secs(11);
    }

    // Should be allowed again after window reset
    assert!(
        !registry.is_rate_limited(&peer_id),
        "Should be allowed after window reset"
    );
}

// ====================================================================
// TTL Wire Format Tests (Item 4)
// ====================================================================

#[test]
fn test_ttl_wire_roundtrip() {
    let packet = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 5,
        wisdom: test_hv(0x01),
    };
    let bytes = packet.to_bytes();
    let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.ttl, 5);
}

#[test]
fn test_ttl_zero_backward_compat() {
    // Legacy packets with zeroed reserved bytes should parse with ttl=0
    let packet = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x01),
    };
    let bytes = packet.to_bytes();
    assert_eq!(bytes[WISDOM_PACKET_TTL_OFFSET], 0);
    let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.ttl, 0);
}

// ====================================================================
// Packet Authentication Tests (Item 1)
// ====================================================================

#[test]
fn test_packet_mac_roundtrip() {
    let key = [0x42u8; 32];
    let mut packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    let mut bytes = packet.to_bytes();
    let mac = compute_packet_mac(&bytes, &key);
    // Independently generated with Python's stdlib HMAC/SHA-256 over the
    // version-2 wire image (the 32-byte tag field is all zeroes).
    assert_eq!(
        mac,
        [
            0x09, 0x1a, 0x84, 0xf8, 0xf0, 0xba, 0x7b, 0x85, 0xa0, 0x49, 0xb9, 0x5e, 0x3c, 0xf2,
            0x1c, 0x10, 0x0a, 0x41, 0x3c, 0xbc, 0x26, 0x29, 0xf2, 0x51, 0xa2, 0x89, 0xc5, 0x5b,
            0x17, 0x4f, 0xbc, 0x70,
        ]
    );
    packet.auth_mac = mac;
    bytes = packet.to_bytes();
    assert!(verify_packet_mac(&bytes, &key));
}

#[test]
fn test_packet_mac_rejects_tampered() {
    let key = [0x42u8; 32];
    let mut packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    let mut bytes = packet.to_bytes();
    let mac = compute_packet_mac(&bytes, &key);
    packet.auth_mac = mac;
    bytes = packet.to_bytes();
    // Tamper with a wisdom byte
    bytes[100] ^= 0xFF;
    assert!(!verify_packet_mac(&bytes, &key));
}

#[test]
fn test_packet_mac_rejects_wrong_key() {
    let key_a = [0x42u8; 32];
    let key_b = [0x99u8; 32];
    let mut packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    let mut bytes = packet.to_bytes();
    let mac = compute_packet_mac(&bytes, &key_a);
    packet.auth_mac = mac;
    bytes = packet.to_bytes();
    assert!(!verify_packet_mac(&bytes, &key_b));
}

/// Regression for the historical 8-bit tag: knowing or guessing the first
/// byte is no longer sufficient because all 256 tag bits are verified.
#[test]
fn test_packet_mac_rejects_first_byte_only_tag() {
    let key = [0x42u8; 32];
    let mut packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    let full_tag = compute_packet_mac(&packet.to_bytes(), &key);
    packet.auth_mac[0] = full_tag[0];

    assert!(!verify_packet_mac(&packet.to_bytes(), &key));
}

#[test]
fn test_packet_mac_rejects_wrong_length_or_wire_version() {
    let key = [0x42u8; 32];
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    let mut bytes = packet.to_bytes();
    let tag = compute_packet_mac(&bytes, &key);
    bytes[WISDOM_PACKET_AUTH_TAG_START..WISDOM_PACKET_AUTH_TAG_END].copy_from_slice(&tag);

    let mut with_trailer = bytes.to_vec();
    with_trailer.push(0);
    assert!(!verify_packet_mac(&with_trailer, &key));

    bytes[0] = 1;
    assert!(!verify_packet_mac(&bytes, &key));
}

#[test]
fn test_packet_mac_zero_without_key() {
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0xAB),
    };
    assert_eq!(packet.auth_mac, [0; 32]);
}

#[test]
fn test_packet_mac_preserved_through_serde() {
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0xAB; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    let bytes = packet.to_bytes();
    let decoded = WisdomPacket::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.auth_mac, [0xAB; 32]);
    assert_eq!(decoded.ttl, 3);
}

#[test]
fn test_packet_mac_fragment_roundtrip() {
    let key = [0x77u8; 32];
    let mut packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    // Sign the packet
    let bytes = packet.to_bytes();
    let mac = compute_packet_mac(&bytes, &key);
    packet.auth_mac = mac;

    // Fragment and reassemble (drop fragment 3)
    let frags = packet.fragment();
    let mut assembler = WisdomPacket::assembler(packet.thought_id(), 11);
    let mut buf = [0u8; LORA_MTU];
    for (i, frag) in frags.iter().enumerate() {
        if i == 3 {
            continue;
        }
        let len = frag.to_bytes(&mut buf);
        let decoded_frag = LoRaFragment::from_bytes(&buf[..len]).unwrap();
        assembler.feed(&decoded_frag);
    }
    assert!(assembler.is_complete());
    let recovered = WisdomPacket::from_assembler(&assembler).unwrap();
    let recovered_bytes = recovered.to_bytes();
    assert!(verify_packet_mac(&recovered_bytes, &key));
}

// ====================================================================
// Quarantined legacy HDC tag compatibility tests
// ====================================================================

#[cfg(feature = "insecure-experimental-crypto")]
#[test]
fn test_wisdom_packet_hdc_mac_roundtrip() {
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: BinaryHV::random(42),
    };
    let key = BinaryHV::random(99);
    let mac = packet.compute_hdc_mac(&key);
    assert!(packet.verify_hdc_mac(&key, &mac));
}

#[cfg(feature = "insecure-experimental-crypto")]
#[test]
fn test_wisdom_packet_hdc_mac_wrong_key_fails() {
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: BinaryHV::random(42),
    };
    let key_a = BinaryHV::random(99);
    let key_b = BinaryHV::random(100);
    let mac = packet.compute_hdc_mac(&key_a);
    assert!(!packet.verify_hdc_mac(&key_b, &mac));
}

#[cfg(feature = "insecure-experimental-crypto")]
#[test]
fn test_wisdom_packet_hdc_mac_tampered_wisdom_fails() {
    let key = BinaryHV::random(99);
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: BinaryHV::random(42),
    };
    let mac = packet.compute_hdc_mac(&key);

    // Tamper: change wisdom
    let mut tampered = packet;
    tampered.wisdom = BinaryHV::random(43);
    assert!(!tampered.verify_hdc_mac(&key, &mac));
}

#[cfg(feature = "insecure-experimental-crypto")]
#[test]
fn test_wisdom_packet_hdc_mac_noisy_verify() {
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: BinaryHV::random(42),
    };
    let key = BinaryHV::random(99);
    let mac = packet.compute_hdc_mac(&key);
    assert!(packet.verify_hdc_mac_noisy(&key, &mac, 0.95));

    let wrong_key = BinaryHV::random(100);
    assert!(!packet.verify_hdc_mac_noisy(&wrong_key, &mac, 0.95));
}

// ====================================================================
// Priority Ordering Test (Item 2)
// ====================================================================

#[test]
fn test_payload_type_priority_ordering() {
    assert!(PayloadType::Heartbeat.priority() > PayloadType::WisdomVector.priority());
    assert!(PayloadType::WisdomVector.priority() > PayloadType::Affective.priority());
    assert!(PayloadType::Affective.priority() > PayloadType::Gradient.priority());
    assert_eq!(PayloadType::Heartbeat.priority(), 3);
    assert_eq!(PayloadType::Gradient.priority(), 0);
}

// ====================================================================
// MeshPeerRegistry has_peer Test (Item 5)
// ====================================================================

#[test]
fn test_peer_registry_has_peer() {
    let mut registry = MeshPeerRegistry::new();
    let peer_id = [0xDD; 8];
    assert!(!registry.has_peer(&peer_id));

    let pkt = WisdomPacket {
        source_id: peer_id,
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x01),
    };
    registry.update(&pkt);
    assert!(registry.has_peer(&peer_id));
}

// ====================================================================
// Compression Tests (Item 3)
// ====================================================================

#[test]
fn test_compress_decompress_roundtrip() {
    let packet = WisdomPacket {
        source_id: [0xDE; 8],
        sequence: 42,
        phi: 0.7,
        urgency: MeshUrgency::Normal,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xAB),
    };
    let raw = packet.to_bytes();
    let compressed = compress_packet(&raw);
    let decompressed = decompress_packet(&compressed).unwrap();
    assert_eq!(decompressed.len(), WISDOM_PACKET_SIZE);
    assert_eq!(&decompressed[..], &raw[..]);
}

#[test]
fn test_compress_heartbeat_uses_envelope() {
    // Heartbeat is mostly zeros — should compress (with lz4 feature)
    // or at least produce a valid envelope
    let packet = WisdomPacket {
        source_id: [0; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Cruise,
        timestamp_s: 0,
        payload_type: PayloadType::Heartbeat,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV([0u8; 2048]),
    };
    let raw = packet.to_bytes();
    let compressed = compress_packet(&raw);
    // First byte must be a valid compression header
    assert!(compressed[0] == COMPRESS_NONE || compressed[0] == COMPRESS_LZ4);
    // Roundtrip must succeed
    let decompressed = decompress_packet(&compressed).unwrap();
    assert_eq!(&decompressed[..], &raw[..]);
}

#[test]
fn test_decompress_invalid_header() {
    let data = vec![0xFF, 0x01, 0x02];
    assert!(decompress_packet(&data).is_none());
}

#[test]
fn test_decompress_tolerates_trailing_bytes() {
    // Simulate FEC adding trailing garbage after COMPRESS_NONE envelope
    let packet = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: test_hv(0x42),
    };
    let raw = packet.to_bytes();
    let mut envelope = compress_packet(&raw);
    // Append trailing garbage (simulating FEC padding)
    envelope.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    let decompressed = decompress_packet(&envelope).unwrap();
    assert_eq!(&decompressed[..], &raw[..]);
}

// ====================================================================
// Item 2: Compression ratio telemetry tests
// ====================================================================

#[test]
fn test_compression_ratio_no_data() {
    let stats = MeshStats::default();
    assert_eq!(stats.compression_ratio(), 1.0);
}

#[test]
fn test_compression_ratio_with_data() {
    let mut stats = MeshStats::default();
    stats.bytes_before_compression = 1000;
    stats.bytes_after_compression = 500;
    let ratio = stats.compression_ratio();
    assert!((ratio - 0.5).abs() < 1e-10, "Expected ~0.5, got {ratio}");
}

// ====================================================================
// Item 4: Constructor field verification tests
// ====================================================================

#[test]
fn test_from_affective_sets_ttl_and_urgency() {
    let affect = crate::swarm::AffectiveState {
        valence: 0.5,
        arousal: 0.3,
        dominance: 0.1,
        intensity: 0.3,
        thermodynamic_load: 0.0,
        confidence: 0.9,
        timestamp_ms: 1_700_000_000_000,
        sequence: 1,
    };
    let pkt = WisdomPacket::from_affective([0xAF; 8], 42, &affect);
    assert_eq!(pkt.ttl, MESH_DEFAULT_TTL);
    assert!(matches!(pkt.urgency, MeshUrgency::Cruise));
    assert!(matches!(pkt.payload_type, PayloadType::Affective));
    assert_eq!(pkt.auth_mac, [0; 32]);
}

#[test]
fn test_from_gradient_sets_ttl_and_urgency() {
    let msg = crate::swarm::GradientMessage {
        source_id: [0; 32],
        gradient_data: vec![0.1, 0.2, 0.3],
        trust_score: 0.95,
        noise_scale: 0.0,
        timestamp: 1_700_000_000_000,
        sample_count: 100,
        model_version: 1,
    };
    let pkt = WisdomPacket::from_gradient([0xBB; 8], 10, &msg).unwrap();
    assert_eq!(pkt.ttl, MESH_DEFAULT_TTL);
    assert!(matches!(pkt.urgency, MeshUrgency::Normal));
    assert!(matches!(pkt.payload_type, PayloadType::Gradient));
    assert_eq!(pkt.auth_mac, [0; 32]);
}

#[test]
fn test_heartbeat_packet_fields() {
    let pkt = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 0,
        phi: 0.5,
        urgency: MeshUrgency::Cruise,
        timestamp_s: 1_700_000_000,
        payload_type: PayloadType::Heartbeat,
        auth_mac: [0; 32],
        ttl: MESH_DEFAULT_TTL,
        wisdom: BinaryHV::zero(),
    };
    assert_eq!(pkt.ttl, MESH_DEFAULT_TTL);
    assert!(matches!(pkt.urgency, MeshUrgency::Cruise));
    assert!(matches!(pkt.payload_type, PayloadType::Heartbeat));
    assert_eq!(pkt.auth_mac, [0; 32]);
}

// ====================================================================
// Item 6: Compression edge case tests
// ====================================================================

#[test]
fn test_compress_none_envelope_size() {
    // Random-ish data that won't compress (no LZ4 benefit)
    let raw = test_hv(0x77);
    let packet = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: raw,
    };
    let raw_bytes = packet.to_bytes();
    let envelope = compress_packet(&raw_bytes);
    // Without lz4_compression feature (or when LZ4 doesn't help),
    // envelope should be exactly 1 + WISDOM_PACKET_SIZE
    assert!(
        envelope.len() <= 1 + WISDOM_PACKET_SIZE,
        "Envelope {} should be <= {} (1 + WISDOM_PACKET_SIZE)",
        envelope.len(),
        1 + WISDOM_PACKET_SIZE
    );
    // First byte must be a valid compression header
    assert!(
        envelope[0] == COMPRESS_NONE || envelope[0] == COMPRESS_LZ4,
        "First byte must be a valid compression header"
    );
}

#[test]
fn test_decompress_malformed_envelope_fallback() {
    // A buffer starting with 0xFF (unknown header) should return None
    let mut data = vec![0xFF];
    data.extend_from_slice(&[0u8; WISDOM_PACKET_SIZE]);
    assert!(
        decompress_packet(&data).is_none(),
        "Unknown header 0xFF should return None"
    );
}

// ====================================================================
// Round 7, Item 6: AIMD bandwidth observability counters
// ====================================================================

#[test]
fn test_mesh_stats_default_aimd_counters() {
    let stats = MeshStats::default();
    assert_eq!(stats.bandwidth_increases, 0);
    assert_eq!(stats.bandwidth_decreases, 0);
}

// ====================================================================
// Round 7, Item 1: End-to-End Compression Validation Tests
// ====================================================================

#[test]
fn test_compress_fragment_reassemble_roundtrip() {
    use super::lora_fragment::{FragmentAssembler, LoRaFragment, fragment};

    let packet = WisdomPacket {
        source_id: [0xAA; 8],
        sequence: 99,
        phi: 0.42,
        urgency: MeshUrgency::Critical,
        timestamp_s: 1_700_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 3,
        wisdom: test_hv(0xBB),
    };
    let original_bytes = packet.to_bytes();

    // Compress
    let envelope = compress_packet(&original_bytes);

    // Fragment the compressed envelope (low-level API)
    let thought_id = packet.thought_id();
    let frags = fragment(thought_id, &envelope);
    assert!(!frags.is_empty());

    // Reassemble
    let mut assembler = FragmentAssembler::new(thought_id, frags.len() as u8, envelope.len());
    let mut buf = [0u8; LORA_MTU];
    for frag in &frags {
        let len = frag.to_bytes(&mut buf);
        let decoded = LoRaFragment::from_bytes(&buf[..len]).unwrap();
        assembler.feed(&decoded);
    }
    assert!(assembler.is_complete());
    let reassembled_envelope = assembler.assemble().expect("assembly should succeed");

    // Decompress
    let decompressed =
        decompress_packet(&reassembled_envelope).expect("decompression should succeed");

    assert_eq!(
        decompressed, original_bytes,
        "Full pipeline roundtrip: compress → fragment → reassemble → decompress should recover original"
    );
}

#[test]
fn test_compress_fragment_reassemble_with_fec_loss() {
    use super::lora_fragment::{FragmentAssembler, LoRaFragment, fragment};

    let packet = WisdomPacket {
        source_id: [0xCC; 8],
        sequence: 200,
        phi: 0.95,
        urgency: MeshUrgency::Normal,
        timestamp_s: 2_000_000,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 2,
        wisdom: test_hv(0xDD),
    };
    let original_bytes = packet.to_bytes();
    let envelope = compress_packet(&original_bytes);
    let thought_id = packet.thought_id();
    let frags = fragment(thought_id, &envelope);
    let total = frags.len();
    assert!(total >= 2, "Need at least 2 fragments for FEC test");

    // Drop the 2nd data fragment (index 1) — FEC should recover
    let mut assembler = FragmentAssembler::new(thought_id, total as u8, envelope.len());
    let mut buf = [0u8; LORA_MTU];
    for (i, frag) in frags.iter().enumerate() {
        if i == 1 {
            continue; // simulate single loss
        }
        let len = frag.to_bytes(&mut buf);
        let decoded = LoRaFragment::from_bytes(&buf[..len]).unwrap();
        assembler.feed(&decoded);
    }

    assert!(
        assembler.is_complete(),
        "FEC should recover 1 lost fragment"
    );
    assert!(
        assembler.used_fec_recovery(),
        "Should have used FEC recovery"
    );
    let reassembled_envelope = assembler.assemble().expect("assembly should succeed");
    let decompressed =
        decompress_packet(&reassembled_envelope).expect("decompression should succeed");
    assert_eq!(
        decompressed, original_bytes,
        "FEC-recovered roundtrip should match original"
    );
}

#[test]
fn test_compress_fragment_reassemble_heartbeat() {
    use super::lora_fragment::{FragmentAssembler, LoRaFragment, fragment};

    let packet = WisdomPacket {
        source_id: [0xEE; 8],
        sequence: 500,
        phi: 0.3,
        urgency: MeshUrgency::Cruise,
        timestamp_s: 3_000_000,
        payload_type: PayloadType::Heartbeat,
        auth_mac: [0; 32],
        ttl: 1,
        wisdom: BinaryHV::zero(), // heartbeat has zero BinaryHV
    };
    let original_bytes = packet.to_bytes();
    let envelope = compress_packet(&original_bytes);
    let thought_id = packet.thought_id();
    let frags = fragment(thought_id, &envelope);

    let mut assembler = FragmentAssembler::new(thought_id, frags.len() as u8, envelope.len());
    let mut buf = [0u8; LORA_MTU];
    for frag in &frags {
        let len = frag.to_bytes(&mut buf);
        let decoded = LoRaFragment::from_bytes(&buf[..len]).unwrap();
        assembler.feed(&decoded);
    }
    assert!(assembler.is_complete());
    let reassembled_envelope = assembler
        .assemble()
        .expect("heartbeat assembly should succeed");
    let decompressed =
        decompress_packet(&reassembled_envelope).expect("heartbeat decompression should succeed");
    assert_eq!(
        decompressed, original_bytes,
        "Heartbeat roundtrip should match"
    );
}

// ====================================================================
// Round 7, Item 2: Partition Detection Tests
// ====================================================================

#[test]
fn test_is_partitioned_true() {
    let registry = MeshPeerRegistry::new();
    let stats = MeshStats {
        peers_expired: 2,
        wisdom_received: 5,
        ..Default::default()
    };
    assert!(registry.is_partitioned(&stats));
}

#[test]
fn test_is_partitioned_false_with_peers() {
    let mut registry = MeshPeerRegistry::new();
    let packet = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV::zero(),
    };
    registry.update(&packet);
    let stats = MeshStats {
        peers_expired: 1,
        wisdom_received: 5,
        ..Default::default()
    };
    assert!(!registry.is_partitioned(&stats));
}

#[test]
fn test_is_partitioned_false_never_had_peers() {
    let registry = MeshPeerRegistry::new();
    let stats = MeshStats::default();
    assert!(!registry.is_partitioned(&stats));
}

#[test]
fn test_stale_peer_count() {
    let mut registry = MeshPeerRegistry::new();
    let packet = WisdomPacket {
        source_id: [0x01; 8],
        sequence: 1,
        phi: 0.5,
        urgency: MeshUrgency::Normal,
        timestamp_s: 0,
        payload_type: PayloadType::WisdomVector,
        auth_mac: [0; 32],
        ttl: 0,
        wisdom: BinaryHV::zero(),
    };
    registry.update(&packet);
    // With a very short timeout, the peer we just added should be fresh
    assert_eq!(
        registry.stale_peer_count(std::time::Duration::from_secs(60)),
        0
    );
    // With zero timeout, everything is stale
    assert_eq!(registry.stale_peer_count(std::time::Duration::ZERO), 1);
}

// ====================================================================
// Round 7, Item 5: Encryption Tests (feature-gated)
// ====================================================================

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_encrypt_decrypt_roundtrip() {
    let key = [0x42u8; 32];
    let source_id = [0x01u8; 8];
    let sequence = 12345u32;
    let plaintext = b"Hello mesh encryption!";

    let ciphertext = encrypt_packet(plaintext, &key, &source_id, 0xAB, sequence);
    assert_ne!(&ciphertext[AEAD_NONCE_SIZE..], plaintext);

    let decrypted = decrypt_packet(&ciphertext, &key).expect("decryption should succeed");
    assert_eq!(decrypted, plaintext);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_decrypt_wrong_key_fails() {
    let key_a = [0x42u8; 32];
    let key_b = [0x99u8; 32];
    let source_id = [0x01u8; 8];
    let plaintext = b"secret data";

    let ciphertext = encrypt_packet(plaintext, &key_a, &source_id, 0xAB, 1);
    assert!(decrypt_packet(&ciphertext, &key_b).is_none());
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_decrypt_tampered_ciphertext_fails() {
    let key = [0x42u8; 32];
    let source_id = [0x01u8; 8];
    let plaintext = b"tamper test";

    let mut ciphertext = encrypt_packet(plaintext, &key, &source_id, 0xAB, 1);
    // Flip a byte in the ciphertext portion (after the nonce)
    if ciphertext.len() > AEAD_NONCE_SIZE + 1 {
        ciphertext[AEAD_NONCE_SIZE + 1] ^= 0xFF;
    }
    assert!(decrypt_packet(&ciphertext, &key).is_none());
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_decrypt_truncated_fails() {
    let key = [0x42u8; 32];
    // Too short — less than nonce + tag
    let short = vec![0u8; AEAD_NONCE_SIZE + AEAD_TAG_SIZE - 1];
    assert!(decrypt_packet(&short, &key).is_none());
}

// -- Nonce construction tests --

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_build_nonce_layout() {
    let source = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    let nonce = build_nonce(&source, 2, 0xAB, 0x12345678);
    // source_id[0..6] | type | epoch | sequence LE
    assert_eq!(&nonce[..6], &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
    assert_eq!(nonce[6], 2); // payload_type
    assert_eq!(nonce[7], 0xAB); // epoch
    assert_eq!(&nonce[8..12], &0x12345678u32.to_le_bytes());
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_typed_nonce_prevents_cross_type_collision() {
    let key = [0x42u8; 32];
    let source_id = [0x01u8; 8];
    let plaintext = b"same sequence, different type";

    // Same key, same sequence — but different payload_type
    let ct_wisdom = encrypt_packet_typed(plaintext, &key, &source_id, 0, 0, 1);
    let ct_heartbeat = encrypt_packet_typed(plaintext, &key, &source_id, 2, 0, 1);

    // Ciphertexts must differ (different nonces)
    assert_ne!(ct_wisdom, ct_heartbeat);

    // Both decrypt successfully
    assert_eq!(
        decrypt_packet(&ct_wisdom, &key).unwrap(),
        plaintext.to_vec()
    );
    assert_eq!(
        decrypt_packet(&ct_heartbeat, &key).unwrap(),
        plaintext.to_vec()
    );
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_typed_nonce_epoch_prevents_restart_collision() {
    let key = [0x42u8; 32];
    let source_id = [0x01u8; 8];
    let plaintext = b"restart safety";

    // Same key, same type, same sequence — but different epoch
    let ct_epoch_a = encrypt_packet_typed(plaintext, &key, &source_id, 0, 0x11, 1);
    let ct_epoch_b = encrypt_packet_typed(plaintext, &key, &source_id, 0, 0x22, 1);

    // Ciphertexts must differ (different nonces)
    assert_ne!(ct_epoch_a, ct_epoch_b);

    // Both decrypt successfully
    assert!(decrypt_packet(&ct_epoch_a, &key).is_some());
    assert!(decrypt_packet(&ct_epoch_b, &key).is_some());
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_encrypt_packet_typed_roundtrip() {
    let key = [0x42u8; 32];
    let source_id = [0x01u8; 8];
    let plaintext = b"typed encryption test";

    let ciphertext = encrypt_packet_typed(plaintext, &key, &source_id, 3, 0xFF, 999);
    let decrypted = decrypt_packet(&ciphertext, &key).expect("decryption should succeed");
    assert_eq!(decrypted, plaintext);
}

// -- XChaCha20-Poly1305 tests --

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_xchacha_roundtrip() {
    let key = [0x42u8; 32];
    let plaintext = b"XChaCha20-Poly1305 nonce-misuse resistant";

    let ciphertext = encrypt_packet_xchacha(plaintext, &key);
    // First 24 bytes are the random nonce
    assert!(ciphertext.len() > XCHACHA_NONCE_SIZE + AEAD_TAG_SIZE);

    let decrypted = decrypt_packet_xchacha(&ciphertext, &key).expect("should decrypt");
    assert_eq!(decrypted, plaintext);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_xchacha_wrong_key_fails() {
    let key_a = [0x42u8; 32];
    let key_b = [0x99u8; 32];
    let plaintext = b"key mismatch";

    let ciphertext = encrypt_packet_xchacha(plaintext, &key_a);
    assert!(decrypt_packet_xchacha(&ciphertext, &key_b).is_none());
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_xchacha_unique_nonces() {
    let key = [0x42u8; 32];
    let plaintext = b"nonce uniqueness";

    let ct1 = encrypt_packet_xchacha(plaintext, &key);
    let ct2 = encrypt_packet_xchacha(plaintext, &key);

    // Random nonces should differ
    assert_ne!(&ct1[..XCHACHA_NONCE_SIZE], &ct2[..XCHACHA_NONCE_SIZE]);
    // Both should decrypt to the same plaintext
    assert_eq!(
        decrypt_packet_xchacha(&ct1, &key).unwrap(),
        decrypt_packet_xchacha(&ct2, &key).unwrap()
    );
}

// -- Key Rotation tests --

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_rotating_key_pair_basic() {
    let key = [0x11u8; 32];
    let pair = RotatingKeyPair::new(key);
    assert_eq!(pair.current_key(), &key);
    assert!(!pair.is_rotating());
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_rotating_key_pair_epoch_initialized() {
    // Two RotatingKeyPairs with the same key should have different epochs
    // (random initialization) — verifying nonce uniqueness across restarts.
    let key = [0x11u8; 32];
    let pair_a = RotatingKeyPair::new(key);
    let pair_b = RotatingKeyPair::new(key);
    // With 256 possible epochs, collision probability is 1/256.
    // We don't assert inequality (could collide ~1/256), but verify epoch is accessible
    // and both pairs are independently initialized.
    let _epoch_a = pair_a.epoch();
    let _epoch_b = pair_b.epoch();
    // Both key pairs should report the same current key
    assert_eq!(
        pair_a.current_key(),
        &key,
        "pair_a should hold the provided key"
    );
    assert_eq!(
        pair_b.current_key(),
        &key,
        "pair_b should hold the provided key"
    );
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_rotating_key_pair_epoch_in_ciphertext() {
    // Encrypt the same data with the same key but different epochs
    // (via two RotatingKeyPair instances) — ciphertexts should differ
    // because the epoch byte enters the nonce.
    let key = [0x42u8; 32];
    let source = [0xAA; 8];
    let plaintext = b"epoch nonce safety test";

    // Force different epochs by using encrypt_packet directly
    let ct_a = encrypt_packet(plaintext, &key, &source, 0x11, 1);
    let ct_b = encrypt_packet(plaintext, &key, &source, 0x22, 1);

    assert_ne!(
        ct_a, ct_b,
        "Different epochs must produce different ciphertexts"
    );

    // Both must decrypt successfully
    assert_eq!(decrypt_packet(&ct_a, &key).unwrap(), plaintext);
    assert_eq!(decrypt_packet(&ct_b, &key).unwrap(), plaintext);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_context_overlay_changes_ciphertext() {
    let base_key = [0x42u8; 32];
    let source = [0xAA; 8];
    let plaintext = b"sensor-bound encryption test";

    // Encrypt without overlay
    let mut pair = RotatingKeyPair::new(base_key);
    let ct_plain = pair.encrypt(plaintext, &source, 1);

    // Encrypt with overlay
    let overlay = [0xFFu8; 32];
    pair.set_context_overlay(Some(overlay));
    let ct_overlay = pair.encrypt(plaintext, &source, 1);

    // Ciphertexts must differ (different effective keys)
    assert_ne!(
        ct_plain, ct_overlay,
        "Context overlay must change the ciphertext"
    );

    // Both must decrypt with their respective effective keys
    assert_eq!(decrypt_packet(&ct_plain, &base_key).unwrap(), plaintext);

    // Overlay key: base XOR overlay
    let mut effective = base_key;
    for (k, o) in effective.iter_mut().zip(overlay.iter()) {
        *k ^= o;
    }
    assert_eq!(decrypt_packet(&ct_overlay, &effective).unwrap(), plaintext);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_context_overlay_symmetric() {
    // Two peers with same base key + same overlay derive same effective key
    let base_key = [0x42u8; 32];
    let overlay = [0xABu8; 32];
    let source = [0xAA; 8];
    let plaintext = b"peer-to-peer context binding";

    let mut sender = RotatingKeyPair::new(base_key);
    sender.set_context_overlay(Some(overlay));

    let mut receiver = RotatingKeyPair::new(base_key);
    receiver.set_context_overlay(Some(overlay));

    // Sender encrypts
    let ct = sender.encrypt(plaintext, &source, 1);

    // Receiver decrypts with same effective key
    let effective = receiver.effective_key();
    let decrypted = decrypt_packet(&ct, &effective).expect("Same overlay should decrypt");
    assert_eq!(decrypted, plaintext);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_rotating_key_pair_rotation() {
    let old_key = [0x11u8; 32];
    let new_key = [0x22u8; 32];
    let source = [0xAA; 8];
    let plaintext = b"hello mesh";

    let mut pair = RotatingKeyPair::new(old_key);

    // Encrypt with old key
    let ct_old = pair.encrypt(plaintext, &source, 1);

    // Rotate to new key with 100 tick grace
    pair.rotate(new_key, 50, 100);
    assert!(pair.is_rotating());
    assert_eq!(pair.current_key(), &new_key);

    // New encryptions use new key
    let ct_new = pair.encrypt(plaintext, &source, 2);

    // During grace: both old and new ciphertext decrypt
    assert!(
        pair.decrypt(&ct_old).is_some(),
        "Old ciphertext should decrypt during grace"
    );
    assert!(
        pair.decrypt(&ct_new).is_some(),
        "New ciphertext should decrypt during grace"
    );

    // After grace expires: old key is discarded
    pair.tick(150); // past grace_expires_at=150
    assert!(!pair.is_rotating());
    assert!(
        pair.decrypt(&ct_old).is_none(),
        "Old ciphertext should fail after grace"
    );
    assert!(
        pair.decrypt(&ct_new).is_some(),
        "New ciphertext should still work"
    );
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_rotating_key_pair_grace_not_expired() {
    let old_key = [0x33u8; 32];
    let new_key = [0x44u8; 32];
    let mut pair = RotatingKeyPair::new(old_key);
    pair.rotate(new_key, 100, 50); // grace expires at tick 150

    // Tick within grace period
    pair.tick(149);
    assert!(pair.is_rotating(), "Should still be rotating before expiry");

    // Tick at exact expiry
    pair.tick(150);
    assert!(!pair.is_rotating(), "Should expire at grace_expires_at");
}

// -- X25519 Key Agreement tests --

#[cfg(feature = "mesh-key-exchange")]
#[test]
fn test_peer_key_store_dh_agreement() {
    // Two peers independently derive the same symmetric key
    let mut store_a = PeerKeyStore::new([0xAA; 32]);
    let mut store_b = PeerKeyStore::new([0xBB; 32]);

    let pub_a = store_a.public_key();
    let pub_b = store_b.public_key();
    assert_ne!(pub_a, pub_b, "Different secrets → different public keys");

    let source_a: [u8; 8] = [0x0A; 8];
    let source_b: [u8; 8] = [0x0B; 8];

    let key_ab = store_a.agree(source_b, &pub_b);
    let key_ba = store_b.agree(source_a, &pub_a);

    assert_eq!(key_ab, key_ba, "DH shared secret must be symmetric");
    assert_eq!(store_a.peer_count(), 1);
    assert_eq!(store_b.peer_count(), 1);
}

#[cfg(feature = "mesh-key-exchange")]
#[test]
fn test_peer_key_store_encrypt_decrypt_roundtrip() {
    let mut store_a = PeerKeyStore::new([0xCC; 32]);
    let mut store_b = PeerKeyStore::new([0xDD; 32]);

    let pub_a = store_a.public_key();
    let pub_b = store_b.public_key();

    let source_a: [u8; 8] = [0x0A; 8];
    let source_b: [u8; 8] = [0x0B; 8];

    store_a.agree(source_b, &pub_b);
    store_b.agree(source_a, &pub_a);

    let key_a = store_a.peer_key(&source_b).unwrap();
    let plaintext = b"wisdom vector payload";
    let ct = encrypt_packet(plaintext, key_a, &source_a, 0xAB, 1);

    let key_b = store_b.peer_key(&source_a).unwrap();
    let pt = decrypt_packet(&ct, key_b).expect("peer key should decrypt");
    assert_eq!(&pt, plaintext);
}

#[cfg(feature = "mesh-key-exchange")]
#[test]
fn test_peer_key_store_wrong_peer_key_fails() {
    let mut store_a = PeerKeyStore::new([0x11; 32]);
    let mut store_b = PeerKeyStore::new([0x22; 32]);
    let store_c = PeerKeyStore::new([0x33; 32]);

    let pub_b = store_b.public_key();
    let pub_c = store_c.public_key();

    let source_b: [u8; 8] = [0x0B; 8];
    let source_c: [u8; 8] = [0x0C; 8];

    // A agrees with B
    store_a.agree(source_b, &pub_b);
    // B agrees with A
    let pub_a = store_a.public_key();
    store_b.agree([0x0A; 8], &pub_a);

    let key_a_for_b = store_a.peer_key(&source_b).unwrap();
    let ct = encrypt_packet(b"secret", key_a_for_b, &[0x0A; 8], 0xAB, 1);

    // C's public key is different — derive a different shared secret
    let mut store_a2 = PeerKeyStore::new([0x11; 32]);
    store_a2.agree(source_c, &pub_c);
    let wrong_key = store_a2.peer_key(&source_c).unwrap();

    assert!(
        decrypt_packet(&ct, wrong_key).is_none(),
        "Wrong peer key must fail"
    );
}

#[cfg(feature = "mesh-key-exchange")]
#[test]
fn test_peer_key_store_remove_peer() {
    let mut store = PeerKeyStore::new([0x55; 32]);
    let peer_pub = [0x66; 32];
    let source: [u8; 8] = [0x0A; 8];
    store.agree(source, &peer_pub);
    assert_eq!(store.peer_count(), 1);
    store.remove_peer(&source);
    assert_eq!(store.peer_count(), 0);
    assert!(store.peer_key(&source).is_none());
}

// -- Fragment-level AEAD tests --

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_fragment_encrypt_decrypt_roundtrip() {
    let key = [0x99u8; 32];
    let source = [0xAA; 8];
    let payload = b"fragment payload data here";

    let ct = encrypt_fragment(payload, &key, &source, 42, 3);
    assert!(
        ct.len() > payload.len(),
        "Ciphertext should include overhead"
    );

    let pt = decrypt_fragment(&ct, &key).expect("should decrypt");
    assert_eq!(&pt, payload);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_fragment_different_indices_different_ciphertext() {
    let key = [0xBBu8; 32];
    let source = [0xCC; 8];
    let payload = b"same payload";

    let ct0 = encrypt_fragment(payload, &key, &source, 1, 0);
    let ct1 = encrypt_fragment(payload, &key, &source, 1, 1);

    // Different nonces (different fragment_index) → different ciphertext
    assert_ne!(
        ct0, ct1,
        "Different fragment indices must produce different ciphertext"
    );

    // Both decrypt correctly
    assert_eq!(decrypt_fragment(&ct0, &key).unwrap(), payload);
    assert_eq!(decrypt_fragment(&ct1, &key).unwrap(), payload);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_fragment_tampered_rejected() {
    let key = [0xDDu8; 32];
    let source = [0xEE; 8];
    let payload = b"tamper target";

    let mut ct = encrypt_fragment(payload, &key, &source, 5, 0);
    // Tamper one byte in the ciphertext (after the nonce)
    if ct.len() > AEAD_NONCE_SIZE + 1 {
        ct[AEAD_NONCE_SIZE + 1] ^= 0xFF;
    }
    assert!(
        decrypt_fragment(&ct, &key).is_none(),
        "Tampered fragment must be rejected"
    );
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn test_fragment_wrong_key_rejected() {
    let key_a = [0x11u8; 32];
    let key_b = [0x22u8; 32];
    let source = [0x33; 8];
    let ct = encrypt_fragment(b"secret", &key_a, &source, 1, 0);
    assert!(
        decrypt_fragment(&ct, &key_b).is_none(),
        "Wrong key must fail"
    );
}
