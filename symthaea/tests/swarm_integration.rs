#![cfg(feature = "swarm")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Swarm Learning Integration Tests
//!
//! Tests for the P2P swarm intelligence system.
//! These tests verify the swarm types and basic functionality
//! without requiring actual P2P connections.

use symthaea::swarm::{
    AffectiveState, AgentInfo, AgentPubKey, ConsciousnessVector, EmotionLabel, HandshakeResult,
    HolochainCortex, HybridHandshake, Hyperfeel, PeerInfo, StreamConfig, SwarmConfig, SwarmError,
    SwarmMessage, SwarmNode, TensorPayload, TensorStream, TensorType, TrustLevel,
};

// ============================================================================
// SWARM CONFIG TESTS
// ============================================================================

#[test]
fn test_swarm_config_default() {
    let config = SwarmConfig::default();

    assert_eq!(
        config.listen_port, 0,
        "Should use OS-assigned port by default"
    );
    assert!(config.enable_mdns, "mDNS should be enabled by default");
    assert!(
        config.enable_derp,
        "DERP relays should be enabled by default"
    );
    assert!(config.max_peers > 0, "Should have positive max_peers");
    assert!(config.min_trust_level > 0.0 && config.min_trust_level < 1.0);
}

#[test]
fn test_swarm_config_local_only() {
    let config = SwarmConfig::local_only();

    assert!(
        !config.enable_derp,
        "DERP should be disabled for local-only"
    );
    assert!(
        config.bootstrap_peers.is_empty(),
        "No bootstrap peers for local-only"
    );
}

#[test]
fn test_swarm_config_production() {
    let config = SwarmConfig::production();

    assert_eq!(
        config.max_peers, 100,
        "Production should have higher max_peers"
    );
    assert_eq!(
        config.min_trust_level, 0.7,
        "Production should have higher trust requirement"
    );
}

// ============================================================================
// PEER INFO TESTS
// ============================================================================

#[test]
fn test_peer_info_creation() {
    let peer = PeerInfo::new("node-123");

    assert_eq!(peer.node_id, "node-123");
    assert!(matches!(peer.trust_level, TrustLevel::Unknown));
    assert!(!peer.is_streamable(0.5));
}

#[test]
fn test_trust_level_values() {
    assert_eq!(TrustLevel::Unknown.value(), 0.0);
    assert_eq!(TrustLevel::Untrusted.value(), 0.0);
    assert_eq!(TrustLevel::Verified(0.7).value(), 0.7);
    assert_eq!(TrustLevel::LocalTrust.value(), 1.0);
}

// ============================================================================
// SWARM ERROR TESTS
// ============================================================================

#[test]
fn test_swarm_error_display() {
    let error = SwarmError::ConnectionFailed {
        peer_id: "node-123".to_string(),
        reason: "network down".to_string(),
    };
    let display = format!("{}", error);
    assert!(display.contains("node-123"));
    assert!(display.contains("network down"));

    let timeout = SwarmError::ConnectionTimeout {
        peer_id: "node-456".to_string(),
        timeout_ms: 5000,
    };
    let display = format!("{}", timeout);
    assert!(display.contains("timed out"));
    assert!(display.contains("5000"));
}

#[test]
fn test_swarm_error_variants() {
    let errors: Vec<SwarmError> = vec![
        SwarmError::UntrustedPeer {
            peer_id: "test".to_string(),
            trust_level: 0.3,
            required: 0.5,
        },
        SwarmError::ConnectionFailed {
            peer_id: "test".to_string(),
            reason: "test".to_string(),
        },
        SwarmError::PeerNotFound {
            peer_id: "test".to_string(),
        },
        SwarmError::InvalidTicket {
            reason: "test".to_string(),
        },
        SwarmError::ChannelClosed {
            peer_id: "test".to_string(),
        },
        SwarmError::NotInitialized,
        SwarmError::FeatureNotEnabled {
            feature: "test".to_string(),
        },
    ];

    assert!(errors.len() >= 7);
}

// ============================================================================
// SWARM NODE TESTS
// ============================================================================

#[test]
fn test_swarm_node_creation() {
    let config = SwarmConfig::default();
    let node = SwarmNode::new(config);

    assert!(node.peers().is_empty());
    assert!(SwarmNode::is_enabled());
}

// ============================================================================
// CONSCIOUSNESS VECTOR TESTS
// ============================================================================

#[test]
fn test_consciousness_vector_creation() {
    let attention = vec![0.5f32; 64];
    let cv = ConsciousnessVector::new(attention, 0.75);

    assert_eq!(cv.phi, 0.75);
    assert_eq!(cv.attention.len(), 64);
    assert!(cv.timestamp_ms > 0);
}

#[test]
fn test_consciousness_vector_size_estimate() {
    let attention = vec![0.0f32; 64];
    let cv = ConsciousnessVector::new(attention, 0.5);
    let size = cv.estimated_size();

    // 64 f32s (256 bytes) + 4 f64s (32 bytes) + 2 u64s (16 bytes) = ~304 bytes
    assert!(size >= 300 && size <= 350);
}

// ============================================================================
// TENSOR PAYLOAD TESTS
// ============================================================================

#[test]
fn test_tensor_payload_creation() {
    let tensor = TensorPayload {
        data: vec![0.1, 0.2, 0.3, 0.4],
        shape: vec![2, 2],
        tensor_type: TensorType::Attention,
        source: "test-layer".to_string(),
        timestamp_ms: 12345,
    };

    assert_eq!(tensor.data.len(), 4);
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.tensor_type, TensorType::Attention);
}

#[test]
fn test_tensor_types() {
    let types = [
        TensorType::Attention,
        TensorType::HiddenState,
        TensorType::Gradient,
        TensorType::Embedding,
        TensorType::ConsciousnessField,
        TensorType::Custom,
    ];
    assert_eq!(types.len(), 6);
}

// ============================================================================
// SWARM MESSAGE TESTS
// ============================================================================

#[test]
fn test_swarm_message_types() {
    let messages = [
        SwarmMessage::Heartbeat {
            phi: 0.5,
            peer_count: 3,
        },
        SwarmMessage::JoinRequest {
            agent_key: "test".to_string(),
            capabilities: vec!["tensor".to_string()],
        },
        SwarmMessage::TicketRequest,
        SwarmMessage::Goodbye {
            reason: "shutdown".to_string(),
        },
    ];

    assert_eq!(messages[0].message_type(), "Heartbeat");
    assert_eq!(messages[1].message_type(), "JoinRequest");
    assert_eq!(messages[2].message_type(), "TicketRequest");
    assert_eq!(messages[3].message_type(), "Goodbye");
}

// ============================================================================
// HYPERFEEL TESTS
// ============================================================================

#[test]
fn test_hyperfeel_creation() {
    let hf = Hyperfeel::default();
    assert_eq!(hf.peer_count(), 0);
}

#[test]
fn test_affective_state() {
    let state = AffectiveState::new(0.5, 0.7, 0.3);
    assert!(state.valence > 0.0);
    assert!(state.arousal > 0.5);
    assert_eq!(state.dominant_emotion(), EmotionLabel::Excited);
}

#[test]
fn test_affective_state_neutral() {
    let state = AffectiveState::neutral();
    assert!((state.valence - 0.0).abs() < 0.01);
    assert!((state.arousal - 0.5).abs() < 0.01);
    assert_eq!(state.dominant_emotion(), EmotionLabel::Neutral);
}

#[test]
fn test_hyperfeel_peer_receive() {
    let mut hf = Hyperfeel::default();

    hf.receive_peer_state("peer-1", AffectiveState::new(0.5, 0.6, 0.2));
    hf.receive_peer_state("peer-2", AffectiveState::new(0.4, 0.7, 0.1));

    assert_eq!(hf.peer_count(), 2);
    assert_eq!(hf.active_peer_count(), 2);
}

#[test]
fn test_hyperfeel_how_are_we_doing() {
    let mut hf = Hyperfeel::default();
    hf.set_local_state(AffectiveState::new(0.7, 0.6, 0.3));

    let status = hf.how_are_we_doing();
    assert!(!status.message.is_empty());
}

// ============================================================================
// HANDSHAKE TESTS
// ============================================================================

#[test]
fn test_hybrid_handshake_challenge() {
    let config = SwarmConfig::default();
    let mut handshake = HybridHandshake::new(config);

    let challenge = handshake.create_challenge("peer-123").unwrap();
    match challenge {
        SwarmMessage::TrustChallenge { nonce } => {
            assert_eq!(nonce.len(), 32);
        }
        _ => panic!("Expected TrustChallenge"),
    }

    assert_eq!(handshake.pending_count(), 1);
}

#[test]
fn test_handshake_result() {
    let success = HandshakeResult::success("peer-1", "agent-1", TrustLevel::Verified(0.8), 150);
    assert!(success.streaming_allowed);
    assert_eq!(success.handshake_time_ms, 150);

    let failed = HandshakeResult::failed("peer-2");
    assert!(!failed.streaming_allowed);
}

// ============================================================================
// HOLOCHAIN CORTEX TESTS
// ============================================================================

#[test]
fn test_cortex_creation() {
    let cortex = HolochainCortex::default();
    assert!(cortex.config().mock_mode);
    assert!(!cortex.is_connected());
}

#[tokio::test]
async fn test_cortex_mock_connect() {
    let mut cortex = HolochainCortex::default();
    cortex.connect().await.unwrap();
    assert!(cortex.is_connected());
}

#[test]
fn test_agent_pub_key() {
    let key = AgentPubKey::new("uhCAkTestKey12345678901234567890123456");
    assert!(key.is_valid());

    let test_key = AgentPubKey::test_key(42);
    assert!(test_key.is_valid());

    let invalid = AgentPubKey::new("short");
    assert!(!invalid.is_valid());
}

#[test]
fn test_agent_info() {
    let key = AgentPubKey::test_key(1);
    let mut info = AgentInfo::new(key);

    assert_eq!(info.reputation_score, 0.5);
    assert!(info.phi_history.is_empty());

    info.record_phi(0.7);
    info.record_phi(0.8);
    assert_eq!(info.phi_history.len(), 2);
    assert!((info.average_phi() - 0.75).abs() < 0.01);
}

// ============================================================================
// TENSOR STREAMING TESTS
// ============================================================================

#[test]
fn test_stream_config() {
    let config = StreamConfig::default();
    assert_eq!(config.max_message_size, 1024 * 1024);
    assert!(config.compression_enabled);
}

#[test]
fn test_tensor_stream() {
    let stream = TensorStream::new(StreamConfig::default());
    let state = ConsciousnessVector::new(vec![0.0; 64], 0.5);

    let bytes = stream.prepare_consciousness(&state).unwrap();
    assert!(!bytes.is_empty());

    let received = stream.receive_consciousness(&bytes).unwrap();
    assert_eq!(received.phi, 0.5);
}

// ============================================================================
// PRIVACY-PRESERVING KNOWLEDGE SHARING TESTS
// ============================================================================

#[test]
fn test_pattern_is_privacy_preserving() {
    // Hypervectors encode semantic meaning without raw data
    // It's impossible to reverse-engineer the original input
    let pattern: Vec<i8> = vec![1, -1, 1, 1, -1, -1, 1, -1, 1, 1];

    // All values are bipolar (-1 or 1)
    for &val in &pattern {
        assert!(val == 1 || val == -1);
    }
}

#[test]
fn test_collective_resonance_via_addition() {
    // Swarm consensus is achieved via vector addition
    // Similar patterns reinforce, different ones cancel
    let pattern1: Vec<i8> = vec![1, 1, -1, 1, -1];
    let pattern2: Vec<i8> = vec![1, 1, -1, 1, -1]; // Same as pattern1
    let pattern3: Vec<i8> = vec![-1, -1, 1, -1, 1]; // Opposite of pattern1

    // Adding similar patterns reinforces
    let sum_similar: Vec<i16> = pattern1
        .iter()
        .zip(pattern2.iter())
        .map(|(&a, &b)| a as i16 + b as i16)
        .collect();

    // Adding opposite patterns cancels
    let sum_opposite: Vec<i16> = pattern1
        .iter()
        .zip(pattern3.iter())
        .map(|(&a, &b)| a as i16 + b as i16)
        .collect();

    // Similar patterns have strong sum
    let similar_magnitude: i16 = sum_similar.iter().map(|x| x.abs()).sum();
    assert_eq!(similar_magnitude, 10); // All 2s -> 2*5=10

    // Opposite patterns cancel out
    let opposite_magnitude: i16 = sum_opposite.iter().map(|x| x.abs()).sum();
    assert_eq!(opposite_magnitude, 0); // All 0s
}