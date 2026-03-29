// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::tcp_backend::TcpBackend;
use super::*;
use crate::swarm::federated_cfc::{DifferentialPrivacyConfig, GradientMessage};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

fn should_skip_tcp(err: &NetworkError) -> bool {
    match err {
        NetworkError::Internal { reason } => {
            reason.contains("Operation not permitted") || reason.contains("Permission denied")
        }
        _ => false,
    }
}

async fn tcp_backend_or_skip() -> Option<TcpBackend> {
    match TcpBackend::new("127.0.0.1:0".parse().unwrap()).await {
        Ok(backend) => Some(backend),
        Err(err) => {
            if should_skip_tcp(&err) {
                eprintln!("Skipping TCP backend tests: {err}");
                None
            } else {
                panic!("TCP backend init failed: {err}");
            }
        }
    }
}

// =========================================================================
// Configuration Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = FederatedNetworkConfig::default();
    assert_eq!(config.num_nodes, 3);
    assert_eq!(config.sync_interval_ms, 5000);
    assert_eq!(config.timeout_ms, 10000);
    assert!(!config.enable_dp);
    assert!(!config.enable_byzantine_tolerance);
}

#[test]
fn test_config_with_num_nodes() {
    let config = FederatedNetworkConfig::default().with_num_nodes(5);
    assert_eq!(config.num_nodes, 5);
    assert_eq!(config.min_nodes_for_aggregation, 2);
}

#[test]
fn test_config_with_dp() {
    let config = FederatedNetworkConfig::default()
        .with_differential_privacy(DifferentialPrivacyConfig::moderate_privacy());
    assert!(config.enable_dp);
    assert!(config.dp_config.is_some());
}

#[test]
fn test_config_for_testing() {
    let config = FederatedNetworkConfig::for_testing();
    assert_eq!(config.sync_interval_ms, 100);
    assert_eq!(config.timeout_ms, 1000);
}

// =========================================================================
// Message Tests
// =========================================================================

#[test]
fn test_message_types() {
    let gradient = GradientMessage::new([0u8; 32], vec![0.1; 10], 0.8);
    let msg = FederatedMessage::GradientShare(gradient);
    assert_eq!(msg.message_type(), "GradientShare");

    let msg = FederatedMessage::Heartbeat {
        node_id: [0u8; 32],
        round: 1,
        trust_score: 0.9,
        timestamp: 0,
    };
    assert_eq!(msg.message_type(), "Heartbeat");
}

#[test]
fn test_message_source_node_id() {
    let gradient = GradientMessage::new([1u8; 32], vec![0.1; 10], 0.8);
    let msg = FederatedMessage::GradientShare(gradient);
    assert_eq!(msg.source_node_id(), Some([1u8; 32]));

    let msg = FederatedMessage::SyncResponse {
        weights: vec![],
        round: 0,
    };
    assert_eq!(msg.source_node_id(), None);
}

// =========================================================================
// Node Tests
// =========================================================================

#[test]
fn test_node_creation() {
    let node = FederatedNode::new([0u8; 32], NodeAddress::Unassigned);
    assert_eq!(node.node_id, [0u8; 32]);
    assert_eq!(node.trust_score, 0.5);
    assert!(node.is_active);
}

#[test]
fn test_node_with_random_id() {
    let node = FederatedNode::with_random_id(NodeAddress::Unassigned);
    assert!(node.node_id != [0u8; 32]);
}

#[test]
fn test_node_short_id() {
    let mut id = [0u8; 32];
    id[0] = 0xAB;
    id[1] = 0xCD;
    let node = FederatedNode::new(id, NodeAddress::Unassigned);
    assert_eq!(node.short_id().len(), 16);
    assert!(node.short_id().starts_with("abcd"));
}

#[test]
fn test_node_staleness() {
    let mut node = FederatedNode::new([0u8; 32], NodeAddress::Unassigned);
    assert!(!node.is_stale(60000)); // Not stale within 1 minute

    // Set heartbeat to past
    node.last_heartbeat = 0;
    assert!(node.is_stale(60000)); // Should be stale now
}

// =========================================================================
// Node Address Tests
// =========================================================================

#[test]
fn test_node_address_display() {
    let addr = NodeAddress::Channel("test-123".to_string());
    assert_eq!(format!("{}", addr), "channel://test-123");

    let socket_addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let addr = NodeAddress::Socket(socket_addr);
    assert_eq!(format!("{}", addr), "tcp://127.0.0.1:8080");

    let addr = NodeAddress::Unassigned;
    assert_eq!(format!("{}", addr), "unassigned");
}

// =========================================================================
// Local Channel Backend Tests
// =========================================================================

#[tokio::test]
async fn test_local_backend_creation() {
    let backend = LocalChannelBackend::new();
    assert!(backend.is_ready());
    assert!(matches!(backend.local_address(), NodeAddress::Channel(_)));
}

#[tokio::test]
async fn test_local_backend_with_id() {
    let backend = LocalChannelBackend::with_id("my-node".to_string());
    assert_eq!(backend.local_id(), "my-node");
}

#[tokio::test]
async fn test_local_backend_send_to_unknown() {
    let backend = LocalChannelBackend::new();
    let result = backend
        .send(
            &NodeAddress::Channel("unknown".to_string()),
            FederatedMessage::Heartbeat {
                node_id: [0u8; 32],
                round: 0,
                trust_score: 0.5,
                timestamp: 0,
            },
        )
        .await;
    assert!(matches!(result, Err(NetworkError::NodeNotFound { .. })));
}

#[tokio::test]
async fn test_local_backend_communication() {
    let backend1 = LocalChannelBackend::with_id("node-1".to_string());
    let backend2 = LocalChannelBackend::with_id("node-2".to_string());

    // Cross-register
    backend1.register_peer_sender("node-2", backend2.get_sender());
    backend2.register_peer_sender("node-1", backend1.get_sender());

    // Send from 1 to 2
    let msg = FederatedMessage::Heartbeat {
        node_id: [1u8; 32],
        round: 5,
        trust_score: 0.8,
        timestamp: 12345,
    };

    backend1
        .send(&NodeAddress::Channel("node-2".to_string()), msg.clone())
        .await
        .unwrap();

    // Receive on 2
    let (source, received) = backend2.receive(Duration::from_secs(1)).await.unwrap();
    assert_eq!(source, NodeAddress::Channel("node-1".to_string()));
    assert_eq!(received.message_type(), "Heartbeat");
}

#[tokio::test]
async fn test_local_backend_broadcast() {
    let backend1 = LocalChannelBackend::with_id("node-1".to_string());
    let backend2 = LocalChannelBackend::with_id("node-2".to_string());
    let backend3 = LocalChannelBackend::with_id("node-3".to_string());

    // Register all peers with node-1
    backend1.register_peer_sender("node-2", backend2.get_sender());
    backend1.register_peer_sender("node-3", backend3.get_sender());

    // Broadcast from node-1
    let msg = FederatedMessage::Heartbeat {
        node_id: [1u8; 32],
        round: 1,
        trust_score: 0.5,
        timestamp: 0,
    };

    let sent_count = backend1.broadcast(msg).await.unwrap();
    assert_eq!(sent_count, 2);

    // Both should receive
    let _ = backend2.receive(Duration::from_secs(1)).await.unwrap();
    let _ = backend3.receive(Duration::from_secs(1)).await.unwrap();
}

#[tokio::test]
async fn test_local_backend_receive_timeout() {
    let backend = LocalChannelBackend::new();
    let result = backend.receive(Duration::from_millis(10)).await;
    assert!(matches!(result, Err(NetworkError::Timeout { .. })));
}

// =========================================================================
// Coordinator Tests
// =========================================================================

#[tokio::test]
async fn test_coordinator_creation() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

    assert_eq!(coordinator.current_round(), 0);
    assert_eq!(coordinator.peer_count(), 0);
    assert_eq!(coordinator.get_weights().len(), 10);
}

#[tokio::test]
async fn test_coordinator_update_weights() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

    coordinator.update_weights(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let weights = coordinator.get_weights();

    assert_eq!(weights.len(), 5);
    assert!((weights[0] - 1.0).abs() < 0.001);
    assert!((weights[4] - 5.0).abs() < 0.001);
}

#[tokio::test]
async fn test_coordinator_register_peer() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

    let peer = FederatedNode::with_random_id(NodeAddress::Channel("peer-1".to_string()));
    let peer_id = peer.node_id;

    coordinator.register_peer(peer).await;
    assert_eq!(coordinator.peer_count(), 1);

    coordinator.unregister_peer(&peer_id, "test").await;
    assert_eq!(coordinator.peer_count(), 0);
}

#[tokio::test]
async fn test_coordinator_stats() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

    let stats = coordinator.stats();
    assert_eq!(stats.active_nodes, 0);
    assert_eq!(stats.current_round, 0);
    assert_eq!(stats.gradients_received, 0);
}

#[tokio::test]
async fn test_coordinator_event_subscription() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

    let mut rx = coordinator.subscribe_events();

    let peer = FederatedNode::with_random_id(NodeAddress::Channel("peer-1".to_string()));
    coordinator.register_peer(peer).await;

    let event = rx.try_recv();
    assert!(event.is_ok());
    assert!(matches!(
        event.unwrap(),
        CoordinatorEvent::NodeJoined { .. }
    ));
}

// =========================================================================
// Integration Tests
// =========================================================================

#[tokio::test]
async fn test_create_test_network() {
    let coordinators = create_test_network(3, 10).await;

    assert_eq!(coordinators.len(), 3);

    for coordinator in &coordinators {
        // Each coordinator should see 2 peers
        assert_eq!(coordinator.peer_count(), 2);
        assert_eq!(coordinator.get_weights().len(), 10);
    }
}

#[tokio::test]
async fn test_gradient_sharing() {
    let coordinators = create_test_network(3, 10).await;

    // Update weights on coordinator 0
    coordinators[0].update_weights(vec![1.0; 10]);

    // Share gradient from coordinator 0
    let sent = coordinators[0].share_gradient(0.0).await.unwrap();
    assert_eq!(sent, 2); // Should send to 2 peers

    // Give a bit of time for messages to arrive
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Coordinator 1 should be able to receive the gradient
    let result = coordinators[1]
        .backend()
        .receive(Duration::from_millis(100))
        .await;
    assert!(result.is_ok());

    let (_, msg) = result.unwrap();
    assert_eq!(msg.message_type(), "GradientShare");
}

#[tokio::test]
async fn test_heartbeat_broadcast() {
    let coordinators = create_test_network(2, 5).await;

    // Send heartbeat from coordinator 0
    let sent = coordinators[0].send_heartbeat().await.unwrap();
    assert_eq!(sent, 1); // Should send to 1 peer

    // Coordinator 1 should receive it
    let (_, msg) = coordinators[1]
        .backend()
        .receive(Duration::from_millis(100))
        .await
        .unwrap();
    assert_eq!(msg.message_type(), "Heartbeat");
}

#[tokio::test]
async fn test_full_sync_round() {
    let coordinators = create_test_network(3, 4).await;

    // Set different initial weights for each coordinator
    coordinators[0].update_weights(vec![1.0, 0.0, 0.0, 0.0]);
    coordinators[1].update_weights(vec![0.0, 1.0, 0.0, 0.0]);
    coordinators[2].update_weights(vec![0.0, 0.0, 1.0, 0.0]);

    // Run sync round on coordinator 0 (acting as coordinator)
    // First, the other coordinators need to share their gradients
    coordinators[1].share_gradient(0.0).await.unwrap();
    coordinators[2].share_gradient(0.0).await.unwrap();

    // Run sync round on coordinator 0
    let result = coordinators[0].run_sync_round().await;
    assert!(result.is_ok());

    // The result may or may not have aggregated weights depending on timing
    // In a real scenario, we'd need more sophisticated synchronization
}

#[tokio::test]
async fn test_coordinator_shutdown() {
    let config = FederatedNetworkConfig::for_testing();
    let mut coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

    // Should not panic
    coordinator.shutdown().await;
}

// =========================================================================
// Error Handling Tests
// =========================================================================

#[test]
fn test_network_error_display() {
    let err = NetworkError::Timeout {
        operation: "receive".to_string(),
        timeout_ms: 1000,
    };
    assert!(format!("{}", err).contains("1000"));
    assert!(format!("{}", err).contains("receive"));

    let err = NetworkError::NodeNotFound {
        node_id: "abc123".to_string(),
    };
    assert!(format!("{}", err).contains("abc123"));
}

#[test]
fn test_network_error_timeout_construction_and_format() {
    let err = NetworkError::Timeout {
        operation: "gradient_sync".to_string(),
        timeout_ms: 5000,
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("5000"));
    assert!(display.contains("gradient_sync"));
    assert!(display.contains("Timeout"));
    assert!(!debug.is_empty());
    assert!(debug.contains("Timeout"));
}

#[test]
fn test_network_error_connection_failed_construction_and_format() {
    let err = NetworkError::ConnectionFailed {
        target: "192.168.1.100:8080".to_string(),
        reason: "connection refused".to_string(),
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("192.168.1.100:8080"));
    assert!(display.contains("connection refused"));
    assert!(!debug.is_empty());
    assert!(debug.contains("ConnectionFailed"));
}

#[test]
fn test_network_error_channel_closed_construction_and_format() {
    let err = NetworkError::ChannelClosed {
        reason: "receiver dropped".to_string(),
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("receiver dropped"));
    assert!(display.contains("Channel closed"));
    assert!(!debug.is_empty());
    assert!(debug.contains("ChannelClosed"));
}

#[test]
fn test_network_error_send_failed_construction_and_format() {
    let err = NetworkError::SendFailed {
        reason: "broken pipe".to_string(),
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("broken pipe"));
    assert!(display.contains("Send failed"));
    assert!(!debug.is_empty());
    assert!(debug.contains("SendFailed"));
}

#[test]
fn test_network_error_receive_failed_construction_and_format() {
    let err = NetworkError::ReceiveFailed {
        reason: "unexpected EOF".to_string(),
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("unexpected EOF"));
    assert!(display.contains("Receive failed"));
    assert!(!debug.is_empty());
    assert!(debug.contains("ReceiveFailed"));
}

#[test]
fn test_network_error_node_not_found_construction_and_format() {
    let err = NetworkError::NodeNotFound {
        node_id: "abcdef0123456789".to_string(),
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("abcdef0123456789"));
    assert!(display.contains("Node not found"));
    assert!(!debug.is_empty());
    assert!(debug.contains("NodeNotFound"));
}

#[test]
fn test_network_error_serialization_construction_and_format() {
    let err = NetworkError::Serialization {
        reason: "invalid bincode payload".to_string(),
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("invalid bincode payload"));
    assert!(display.contains("Serialization error"));
    assert!(!debug.is_empty());
    assert!(debug.contains("Serialization"));
}

#[test]
fn test_network_error_internal_construction_and_format() {
    let err = NetworkError::Internal {
        reason: "failed to bind TCP listener".to_string(),
    };
    let display = format!("{}", err);
    let debug = format!("{:?}", err);

    assert!(!display.is_empty());
    assert!(display.contains("failed to bind TCP listener"));
    assert!(display.contains("Internal error"));
    assert!(!debug.is_empty());
    assert!(debug.contains("Internal"));
}

#[test]
fn test_network_error_implements_std_error() {
    let err = NetworkError::Timeout {
        operation: "test".to_string(),
        timeout_ms: 100,
    };
    // Verify the error can be used as a dyn std::error::Error
    let err_ref: &dyn std::error::Error = &err;
    assert!(!err_ref.to_string().is_empty());
    // NetworkError has no source() impl, so source should be None
    assert!(err_ref.source().is_none());
}

#[test]
fn test_all_network_error_variants_have_nonempty_display() {
    let errors: Vec<NetworkError> = vec![
        NetworkError::Timeout {
            operation: "op".to_string(),
            timeout_ms: 1,
        },
        NetworkError::ConnectionFailed {
            target: "t".to_string(),
            reason: "r".to_string(),
        },
        NetworkError::ChannelClosed {
            reason: "r".to_string(),
        },
        NetworkError::SendFailed {
            reason: "r".to_string(),
        },
        NetworkError::ReceiveFailed {
            reason: "r".to_string(),
        },
        NetworkError::NodeNotFound {
            node_id: "n".to_string(),
        },
        NetworkError::Serialization {
            reason: "r".to_string(),
        },
        NetworkError::Internal {
            reason: "r".to_string(),
        },
    ];

    assert_eq!(errors.len(), 8, "Expected 8 NetworkError variants");

    for (i, err) in errors.iter().enumerate() {
        let display = format!("{}", err);
        let debug = format!("{:?}", err);
        assert!(
            !display.is_empty(),
            "Variant {} has empty Display output",
            i
        );
        assert!(!debug.is_empty(), "Variant {} has empty Debug output", i);
    }
}

#[test]
fn test_network_error_display_messages_are_distinct() {
    let errors: Vec<NetworkError> = vec![
        NetworkError::Timeout {
            operation: "op".to_string(),
            timeout_ms: 1,
        },
        NetworkError::ConnectionFailed {
            target: "t".to_string(),
            reason: "r".to_string(),
        },
        NetworkError::ChannelClosed {
            reason: "r".to_string(),
        },
        NetworkError::SendFailed {
            reason: "r".to_string(),
        },
        NetworkError::ReceiveFailed {
            reason: "r".to_string(),
        },
        NetworkError::NodeNotFound {
            node_id: "n".to_string(),
        },
        NetworkError::Serialization {
            reason: "r".to_string(),
        },
        NetworkError::Internal {
            reason: "r".to_string(),
        },
    ];

    let displays: Vec<String> = errors.iter().map(|e| format!("{}", e)).collect();

    // Each variant should produce a unique prefix (the part before the user-supplied content)
    for i in 0..displays.len() {
        for j in (i + 1)..displays.len() {
            assert_ne!(
                displays[i], displays[j],
                "Variants {} and {} produce identical Display output: '{}'",
                i, j, displays[i]
            );
        }
    }
}

// =========================================================================
// TCP Backend Tests
// =========================================================================

#[tokio::test]
async fn test_tcp_backend_binds_and_ready() {
    let backend = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    // The accept loop should have set the ready flag
    // Give it a brief moment to start
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(backend.is_ready());

    // local_address should be a Socket with a real port (not 0)
    match backend.local_address() {
        NodeAddress::Socket(addr) => {
            assert_ne!(addr.port(), 0);
            assert_eq!(addr.ip(), std::net::Ipv4Addr::LOCALHOST);
        }
        other => panic!("Expected Socket address, got {:?}", other),
    }
}

#[tokio::test]
async fn test_tcp_two_node_send_receive() {
    let backend1 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };
    let backend2 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    // Register each with the other
    let node1 = FederatedNode::new([1u8; 32], backend1.local_address());
    let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
    backend1.register_node(&node2).await.unwrap();
    backend2.register_node(&node1).await.unwrap();

    // Send a Heartbeat from backend1 to backend2
    let msg = FederatedMessage::Heartbeat {
        node_id: [1u8; 32],
        round: 1,
        trust_score: 0.5,
        timestamp: 42,
    };
    backend1.send(&backend2.local_address(), msg).await.unwrap();

    // Receive on backend2
    let (_addr, received) = backend2.receive(Duration::from_secs(5)).await.unwrap();
    match received {
        FederatedMessage::Heartbeat {
            node_id,
            round,
            trust_score,
            timestamp,
        } => {
            assert_eq!(node_id, [1u8; 32]);
            assert_eq!(round, 1);
            assert!((trust_score - 0.5).abs() < f32::EPSILON);
            assert_eq!(timestamp, 42);
        }
        other => panic!("Expected Heartbeat, got {:?}", other.message_type()),
    }
}

#[tokio::test]
async fn test_tcp_broadcast() {
    let backend1 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };
    let backend2 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };
    let backend3 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    // Register nodes 2 and 3 with node 1
    let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
    let node3 = FederatedNode::new([3u8; 32], backend3.local_address());
    backend1.register_node(&node2).await.unwrap();
    backend1.register_node(&node3).await.unwrap();

    // Broadcast from node 1
    let msg = FederatedMessage::Heartbeat {
        node_id: [1u8; 32],
        round: 10,
        trust_score: 0.9,
        timestamp: 100,
    };
    let sent_count = backend1.broadcast(msg).await.unwrap();
    assert_eq!(sent_count, 2);

    // Both node 2 and node 3 should receive
    let (_addr2, received2) = backend2.receive(Duration::from_secs(5)).await.unwrap();
    assert!(matches!(
        received2,
        FederatedMessage::Heartbeat { round: 10, .. }
    ));

    let (_addr3, received3) = backend3.receive(Duration::from_secs(5)).await.unwrap();
    assert!(matches!(
        received3,
        FederatedMessage::Heartbeat { round: 10, .. }
    ));
}

#[tokio::test]
async fn test_tcp_large_message() {
    let backend1 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };
    let backend2 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
    backend1.register_node(&node2).await.unwrap();

    // Send a ModelUpdate with 10,000 weights
    let weights: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.001).collect();
    let msg = FederatedMessage::ModelUpdate {
        weights: weights.clone(),
        round: 42,
        source_id: [1u8; 32],
        timestamp: 999,
    };

    backend1.send(&backend2.local_address(), msg).await.unwrap();

    let (_addr, received) = backend2.receive(Duration::from_secs(5)).await.unwrap();
    match received {
        FederatedMessage::ModelUpdate {
            weights: recv_weights,
            round,
            source_id,
            ..
        } => {
            assert_eq!(round, 42);
            assert_eq!(source_id, [1u8; 32]);
            assert_eq!(recv_weights.len(), 10_000);
            // Verify first and last values survived the round-trip
            assert!((recv_weights[0] - 0.0).abs() < f32::EPSILON);
            assert!((recv_weights[9_999] - 9.999).abs() < 0.001);
        }
        other => panic!("Expected ModelUpdate, got {:?}", other.message_type()),
    }
}

#[tokio::test]
async fn test_tcp_concurrent_sends() {
    let backend1 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };
    let backend2 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
    backend1.register_node(&node2).await.unwrap();

    let target_addr = backend2.local_address();
    let backend1 = Arc::new(backend1);

    // Spawn 10 concurrent send tasks
    let num_tasks = 10usize;
    let mut handles = Vec::new();

    for i in 0..num_tasks {
        let b1 = Arc::clone(&backend1);
        let target = target_addr.clone();
        handles.push(tokio::spawn(async move {
            let msg = FederatedMessage::Heartbeat {
                node_id: [1u8; 32],
                round: i as u64,
                trust_score: 0.5,
                timestamp: i as u64,
            };
            b1.send(&target, msg).await
        }));
    }

    // Wait for all sends to complete
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    // Receive all 10 messages on backend2
    let mut received_rounds = std::collections::HashSet::new();
    for _ in 0..num_tasks {
        let (_addr, msg) = backend2.receive(Duration::from_secs(5)).await.unwrap();
        if let FederatedMessage::Heartbeat { round, .. } = msg {
            received_rounds.insert(round);
        }
    }

    assert_eq!(received_rounds.len(), num_tasks);
    for i in 0..num_tasks {
        assert!(received_rounds.contains(&(i as u64)), "Missing round {}", i);
    }
}

#[tokio::test]
async fn test_tcp_receive_timeout() {
    let backend = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    let result = backend.receive(Duration::from_millis(50)).await;
    assert!(matches!(result, Err(NetworkError::Timeout { .. })));
}

#[tokio::test]
async fn test_tcp_send_to_non_socket_address() {
    let backend = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    let result = backend
        .send(
            &NodeAddress::Channel("some-channel".to_string()),
            FederatedMessage::Heartbeat {
                node_id: [0u8; 32],
                round: 0,
                trust_score: 0.5,
                timestamp: 0,
            },
        )
        .await;

    assert!(matches!(result, Err(NetworkError::SendFailed { .. })));
}

#[tokio::test]
async fn test_tcp_register_unregister_node() {
    let backend = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    let node_id = [42u8; 32];
    let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
    let node = FederatedNode::new(node_id, NodeAddress::Socket(addr));

    // Register
    backend.register_node(&node).await.unwrap();
    assert!(backend.registered_nodes.read().contains_key(&node_id));

    // Unregister
    backend.unregister_node(&node_id).await.unwrap();
    assert!(!backend.registered_nodes.read().contains_key(&node_id));
}

#[tokio::test]
async fn test_tcp_bidirectional_communication() {
    let backend1 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };
    let backend2 = match tcp_backend_or_skip().await {
        Some(backend) => backend,
        None => return,
    };

    // Register in both directions
    let node1 = FederatedNode::new([1u8; 32], backend1.local_address());
    let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
    backend1.register_node(&node2).await.unwrap();
    backend2.register_node(&node1).await.unwrap();

    // Send from 1 -> 2
    let msg1 = FederatedMessage::Heartbeat {
        node_id: [1u8; 32],
        round: 1,
        trust_score: 0.5,
        timestamp: 0,
    };
    backend1
        .send(&backend2.local_address(), msg1)
        .await
        .unwrap();

    let (_addr, received1) = backend2.receive(Duration::from_secs(5)).await.unwrap();
    assert!(matches!(
        received1,
        FederatedMessage::Heartbeat { round: 1, .. }
    ));

    // Send from 2 -> 1
    let msg2 = FederatedMessage::Heartbeat {
        node_id: [2u8; 32],
        round: 2,
        trust_score: 0.7,
        timestamp: 0,
    };
    backend2
        .send(&backend1.local_address(), msg2)
        .await
        .unwrap();

    let (_addr, received2) = backend1.receive(Duration::from_secs(5)).await.unwrap();
    assert!(matches!(
        received2,
        FederatedMessage::Heartbeat { round: 2, .. }
    ));
}
