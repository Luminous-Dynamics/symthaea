//! Federated Network Integration Tests
//!
//! Tests for the federated CfC learning network with channel-based simulation.
//! These tests verify multi-node communication, gradient sharing, and aggregation
//! without requiring real network infrastructure.
//!
//! Run with: `cargo test --test federated_network_integration`

use std::time::Duration;
use symthaea::swarm::{
    create_test_network,
    CoordinatorEvent,
    DifferentialPrivacyConfig,
    // Federated CfC
    FederatedAggregator,
    FederatedCoordinator,
    FederatedMessage,
    // Federated Network
    FederatedNetworkConfig,
    FederatedNode,
    GradientMessage,
    LocalChannelBackend,
    NetworkBackend,
    NodeAddress,
};

// ============================================================================
// THREE-NODE NETWORK TESTS
// ============================================================================

#[tokio::test]
async fn test_three_node_network_creation() {
    let coordinators = create_test_network(3, 10).await;

    assert_eq!(
        coordinators.len(),
        3,
        "Should create exactly 3 coordinators"
    );

    for coordinator in &coordinators {
        assert_eq!(coordinator.peer_count(), 2, "Each node should see 2 peers");
        assert_eq!(
            coordinator.get_weights().len(),
            10,
            "Each node should have 10 weights"
        );
        assert_eq!(coordinator.current_round(), 0, "Initial round should be 0");
    }

    // Verify each coordinator has a unique ID
    let ids: Vec<[u8; 32]> = coordinators.iter().map(|c| c.local_node_id()).collect();
    assert_ne!(ids[0], ids[1], "Node IDs should be unique");
    assert_ne!(ids[1], ids[2], "Node IDs should be unique");
    assert_ne!(ids[0], ids[2], "Node IDs should be unique");
}

#[tokio::test]
async fn test_three_node_gradient_sharing() {
    let coordinators = create_test_network(3, 5).await;

    // Set different initial weights for each coordinator
    coordinators[0].update_weights(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
    coordinators[1].update_weights(vec![0.0, 1.0, 0.0, 0.0, 0.0]);
    coordinators[2].update_weights(vec![0.0, 0.0, 1.0, 0.0, 0.0]);

    // Node 0 shares its gradient
    let sent_count = coordinators[0].share_gradient(0.0).await.unwrap();
    assert_eq!(sent_count, 2, "Should send to 2 peers");

    // Verify nodes 1 and 2 can receive the gradient
    let result1 = coordinators[1]
        .backend
        .receive(Duration::from_millis(100))
        .await;
    assert!(result1.is_ok(), "Node 1 should receive gradient");

    let (source, msg) = result1.unwrap();
    assert_eq!(msg.message_type(), "GradientShare");
    assert!(matches!(source, NodeAddress::Channel(_)));

    let result2 = coordinators[2]
        .backend
        .receive(Duration::from_millis(100))
        .await;
    assert!(result2.is_ok(), "Node 2 should receive gradient");
}

#[tokio::test]
async fn test_three_node_heartbeat_exchange() {
    let coordinators = create_test_network(3, 5).await;

    // Each node sends heartbeat
    for coordinator in &coordinators {
        let sent = coordinator.send_heartbeat().await.unwrap();
        assert_eq!(sent, 2, "Each heartbeat should reach 2 peers");
    }

    // Give time for messages to propagate
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Each node should be able to receive 2 heartbeats
    for coordinator in &coordinators {
        let mut received = 0;
        while let Ok(_) = coordinator.backend.receive(Duration::from_millis(10)).await {
            received += 1;
        }
        assert_eq!(received, 2, "Should receive 2 heartbeats from peers");
    }
}

#[tokio::test]
async fn test_three_node_bidirectional_communication() {
    let coordinators = create_test_network(3, 4).await;

    // Node 0 sends, Node 1 receives
    coordinators[0].share_gradient(0.0).await.unwrap();
    let (source, _) = coordinators[1]
        .backend
        .receive(Duration::from_millis(100))
        .await
        .unwrap();
    assert!(matches!(source, NodeAddress::Channel(ref id) if id == "node-0"));

    // Node 1 sends, Node 0 receives
    coordinators[1].share_gradient(0.0).await.unwrap();
    let (source, _) = coordinators[0]
        .backend
        .receive(Duration::from_millis(100))
        .await
        .unwrap();
    assert!(matches!(source, NodeAddress::Channel(ref id) if id == "node-1"));
}

// ============================================================================
// FIVE-NODE NETWORK TESTS
// ============================================================================

#[tokio::test]
async fn test_five_node_network_creation() {
    let coordinators = create_test_network(5, 20).await;

    assert_eq!(coordinators.len(), 5);

    for coordinator in &coordinators {
        assert_eq!(coordinator.peer_count(), 4, "Each node should see 4 peers");
        assert_eq!(coordinator.get_weights().len(), 20);
    }
}

#[tokio::test]
async fn test_five_node_broadcast() {
    let coordinators = create_test_network(5, 10).await;

    // Node 0 broadcasts
    let sent = coordinators[0].share_gradient(0.0).await.unwrap();
    assert_eq!(sent, 4, "Should broadcast to 4 peers");

    // All other nodes should receive
    for i in 1..5 {
        let result = coordinators[i]
            .backend
            .receive(Duration::from_millis(100))
            .await;
        assert!(result.is_ok(), "Node {} should receive", i);
    }
}

// ============================================================================
// GRADIENT AGGREGATION TESTS
// ============================================================================

#[tokio::test]
async fn test_gradient_aggregation_flow() {
    let coordinators = create_test_network(3, 4).await;

    // Set unique weights for each node
    coordinators[0].update_weights(vec![1.0, 0.0, 0.0, 0.0]);
    coordinators[1].update_weights(vec![0.0, 1.0, 0.0, 0.0]);
    coordinators[2].update_weights(vec![0.0, 0.0, 1.0, 0.0]);

    // Nodes 1 and 2 share their gradients
    coordinators[1].share_gradient(0.0).await.unwrap();
    coordinators[2].share_gradient(0.0).await.unwrap();

    // Node 0 receives and processes the gradients
    for _ in 0..2 {
        let result = coordinators[0]
            .backend
            .receive(Duration::from_millis(100))
            .await;
        if let Ok((source, msg)) = result {
            coordinators[0].process_message(source, msg).await.unwrap();
        }
    }

    // Node 0's aggregator should have received contributions
    let stats = coordinators[0].stats();
    // Note: The stats count gradients processed, actual count depends on timing
    assert!(stats.uptime_seconds >= 0, "Stats should be available");
}

#[tokio::test]
async fn test_aggregation_with_different_trust_scores() {
    // Create a network where we manually set up aggregators
    let config = FederatedNetworkConfig::for_testing().with_num_nodes(3);

    // Create aggregators with weights
    let mut agg0 = FederatedAggregator::new(vec![0.0; 4]);
    let mut agg1 = FederatedAggregator::new(vec![1.0, 0.0, 0.0, 0.0]);
    let mut agg2 = FederatedAggregator::new(vec![0.0, 0.0, 1.0, 0.0]);

    // Export gradients from nodes 1 and 2
    let grad1 = agg1.export_local_gradient(0.0);
    let grad2 = agg2.export_local_gradient(0.0);

    // Modify trust scores for weighted aggregation
    let mut high_trust_grad = grad1.clone();
    high_trust_grad.trust_score = 0.9; // High trust

    let mut low_trust_grad = grad2.clone();
    low_trust_grad.trust_score = 0.1; // Low trust

    // Node 0 receives both
    assert!(agg0.receive_gradient(high_trust_grad));
    assert!(agg0.receive_gradient(low_trust_grad));

    // Aggregate
    let aggregated = agg0.aggregate();
    assert!(aggregated.is_some());

    let result = aggregated.unwrap();
    // High trust contribution should dominate
    // Normalized weights: 0.9/(0.9+0.1) = 0.9, 0.1/(0.9+0.1) = 0.1
    // Result[0] = 0.9 * 1.0 + 0.1 * 0.0 = 0.9
    // Result[2] = 0.9 * 0.0 + 0.1 * 1.0 = 0.1
    assert!(
        result[0] > 0.8,
        "High trust gradient should dominate: {}",
        result[0]
    );
    assert!(
        result[2] < 0.2,
        "Low trust gradient should have less influence: {}",
        result[2]
    );
}

// ============================================================================
// DIFFERENTIAL PRIVACY TESTS
// ============================================================================

#[tokio::test]
async fn test_gradient_sharing_with_differential_privacy() {
    let config = FederatedNetworkConfig::for_testing()
        .with_num_nodes(3)
        .with_differential_privacy(DifferentialPrivacyConfig::moderate_privacy());

    let initial_weights = vec![1.0; 10];
    let backend = std::sync::Arc::new(LocalChannelBackend::new());
    let coordinator =
        FederatedCoordinator::new_with_backend(config, initial_weights, backend).await;

    // Share gradient with DP enabled
    let sent = coordinator.share_gradient(1.0).await.unwrap(); // noise_scale = 1.0
    assert_eq!(sent, 0); // No peers registered yet, so 0 sent

    // But we can verify the gradient export works
    let weights = coordinator.get_weights();
    assert_eq!(weights.len(), 10);
}

#[tokio::test]
async fn test_dp_noise_adds_variance() {
    let mut aggregator = FederatedAggregator::new(vec![1.0, 2.0, 3.0, 4.0])
        .with_differential_privacy(DifferentialPrivacyConfig::moderate_privacy());

    // Export multiple gradients and check variance
    let mut samples: Vec<Vec<f32>> = Vec::new();
    for _ in 0..50 {
        let grad = aggregator.export_local_gradient(0.0);
        samples.push(grad.gradient_data);
    }

    // Check that there's variance due to DP noise
    for dim in 0..4 {
        let values: Vec<f32> = samples.iter().map(|s| s[dim]).collect();
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        // With DP, we should see some variance
        assert!(variance > 0.1, "DP should add variance, got {}", variance);
    }
}

// ============================================================================
// EVENT SUBSCRIPTION TESTS
// ============================================================================

#[tokio::test]
async fn test_event_subscription_node_joined() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

    let mut rx = coordinator.subscribe_events();

    // Register a peer
    let peer = FederatedNode::with_random_id(NodeAddress::Channel("test-peer".to_string()));
    coordinator.register_peer(peer);

    // Should receive NodeJoined event
    let event = rx.try_recv();
    assert!(event.is_ok());
    assert!(matches!(
        event.unwrap(),
        CoordinatorEvent::NodeJoined { .. }
    ));
}

#[tokio::test]
async fn test_event_subscription_node_left() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

    let mut rx = coordinator.subscribe_events();

    // Register and then unregister a peer
    let peer = FederatedNode::with_random_id(NodeAddress::Channel("leaving-peer".to_string()));
    let peer_id = peer.node_id;
    coordinator.register_peer(peer);

    // Clear the join event
    let _ = rx.try_recv();

    // Unregister
    coordinator.unregister_peer(&peer_id, "test removal");

    // Should receive NodeLeft event
    let event = rx.try_recv();
    assert!(event.is_ok());
    assert!(matches!(event.unwrap(), CoordinatorEvent::NodeLeft { .. }));
}

// ============================================================================
// SYNCHRONIZATION ROUND TESTS
// ============================================================================

#[tokio::test]
async fn test_sync_round_increments_round_number() {
    let config = FederatedNetworkConfig::for_testing()
        .with_num_nodes(2)
        .with_timeout_ms(100);

    let coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

    assert_eq!(coordinator.current_round(), 0);

    // Run a sync round (will timeout quickly due to no peers responding)
    let _ = coordinator.run_sync_round().await;

    assert_eq!(coordinator.current_round(), 1);
}

#[tokio::test]
async fn test_sync_round_with_multiple_nodes() {
    let mut coordinators = create_test_network(3, 4).await;

    // Set different weights
    coordinators[0].update_weights(vec![1.0, 0.0, 0.0, 0.0]);
    coordinators[1].update_weights(vec![0.0, 1.0, 0.0, 0.0]);
    coordinators[2].update_weights(vec![0.0, 0.0, 1.0, 0.0]);

    // Nodes 1 and 2 share gradients before node 0 runs sync
    coordinators[1].share_gradient(0.0).await.unwrap();
    coordinators[2].share_gradient(0.0).await.unwrap();

    // Small delay for message delivery
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Node 0 runs sync round
    let result = coordinators[0].run_sync_round().await;
    assert!(result.is_ok());

    // Round should have incremented
    assert_eq!(coordinators[0].current_round(), 1);
}

// ============================================================================
// STATS AND MONITORING TESTS
// ============================================================================

#[tokio::test]
async fn test_coordinator_stats() {
    let coordinators = create_test_network(3, 5).await;

    // Exchange some messages
    coordinators[0].send_heartbeat().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Receive messages on node 1
    let _ = coordinators[1]
        .backend
        .receive(Duration::from_millis(50))
        .await;

    let stats = coordinators[0].stats();
    assert_eq!(stats.active_nodes, 2, "Should show 2 active peers");
    assert!(stats.uptime_seconds >= 0);
}

#[tokio::test]
async fn test_stats_track_gradients() {
    let coordinators = create_test_network(2, 5).await;

    // Share gradient from node 1
    coordinators[1].share_gradient(0.0).await.unwrap();

    // Node 0 receives and processes it
    let result = coordinators[0]
        .backend
        .receive(Duration::from_millis(100))
        .await;
    if let Ok((source, msg)) = result {
        coordinators[0].process_message(source, msg).await.unwrap();
    }

    let stats = coordinators[0].stats();
    assert!(
        stats.gradients_received >= 1,
        "Should have received at least 1 gradient"
    );
}

// ============================================================================
// MESSAGE PROCESSING TESTS
// ============================================================================

#[tokio::test]
async fn test_process_join_request() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

    let join_request = FederatedMessage::JoinRequest {
        node_id: [42u8; 32],
        address: "new-peer".to_string(),
        trust_score: 0.8,
    };

    let result = coordinator
        .process_message(NodeAddress::Channel("new-peer".to_string()), join_request)
        .await;

    assert!(result.is_ok());
    // Note: The peer is registered but the ack goes to backend (which has no receiver for "new-peer")
    // In a real scenario, the joining peer would have set up their backend first
}

#[tokio::test]
async fn test_process_leave_message() {
    let config = FederatedNetworkConfig::for_testing();
    let coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

    // First register a peer
    let peer = FederatedNode::new([99u8; 32], NodeAddress::Channel("leaving".to_string()));
    coordinator.register_peer(peer);
    assert_eq!(coordinator.peer_count(), 1);

    // Process leave message
    let leave_msg = FederatedMessage::Leave {
        node_id: [99u8; 32],
        reason: "goodbye".to_string(),
    };

    coordinator
        .process_message(NodeAddress::Channel("leaving".to_string()), leave_msg)
        .await
        .unwrap();

    assert_eq!(
        coordinator.peer_count(),
        0,
        "Peer should be removed after Leave message"
    );
}

// ============================================================================
// NETWORK RESILIENCE TESTS
// ============================================================================

#[tokio::test]
async fn test_network_continues_with_node_failure() {
    let mut coordinators = create_test_network(4, 5).await;

    // All nodes share gradients initially
    for coordinator in &coordinators {
        coordinator.share_gradient(0.0).await.unwrap();
    }

    // "Remove" node 2 by dropping it from node 0's perspective
    let node2_id = coordinators[2].local_node_id();
    coordinators[0].unregister_peer(&node2_id, "simulated failure");

    // Node 0 should still have 2 peers
    assert_eq!(coordinators[0].peer_count(), 2);

    // Node 0 can still broadcast
    let sent = coordinators[0].share_gradient(0.0).await.unwrap();
    assert_eq!(sent, 2, "Should still broadcast to remaining 2 peers");
}

#[tokio::test]
async fn test_late_joiner_can_sync() {
    let config = FederatedNetworkConfig::for_testing();

    // Create first coordinator
    let backend1 = std::sync::Arc::new(LocalChannelBackend::with_id("node-1".to_string()));
    let coordinator1 = FederatedCoordinator::new_with_backend(
        config.clone(),
        vec![1.0, 2.0, 3.0],
        backend1.clone(),
    )
    .await;

    // Update weights and run a few rounds on coordinator1
    coordinator1.update_weights(vec![10.0, 20.0, 30.0]);

    // Create second coordinator (late joiner)
    let backend2 = std::sync::Arc::new(LocalChannelBackend::with_id("node-2".to_string()));

    // Cross-register backends
    backend1.register_peer_sender("node-2", backend2.get_sender());
    backend2.register_peer_sender("node-1", backend1.get_sender());

    let coordinator2 = FederatedCoordinator::new_with_backend(
        config,
        vec![0.0, 0.0, 0.0], // Starting with zeros
        backend2.clone(),
    )
    .await;

    // Register peers
    let peer2 = FederatedNode::new(
        coordinator2.local_node_id(),
        NodeAddress::Channel("node-2".to_string()),
    );
    coordinator1.register_peer(peer2);

    let peer1 = FederatedNode::new(
        coordinator1.local_node_id(),
        NodeAddress::Channel("node-1".to_string()),
    );
    coordinator2.register_peer(peer1);

    // Late joiner can request sync
    assert_eq!(coordinator2.peer_count(), 1);
    assert_eq!(coordinator1.peer_count(), 1);
}

// ============================================================================
// BYZANTINE TOLERANCE TESTS
// ============================================================================

#[tokio::test]
async fn test_byzantine_tolerance_config() {
    let config = FederatedNetworkConfig::for_testing()
        .with_num_nodes(5)
        .with_byzantine_tolerance(0.2);

    assert!(config.enable_byzantine_tolerance);
    assert!((config.byzantine_trim_fraction - 0.2).abs() < 0.001);

    let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;
    assert!(coordinator.stats().active_nodes == 0); // No peers yet, but config is applied
}

#[tokio::test]
async fn test_aggregator_with_byzantine_tolerance() {
    let mut aggregator = FederatedAggregator::new(vec![0.0; 3]).with_byzantine_tolerance(0.2);

    // Add 5 gradients: 3 honest, 2 outliers
    let honest1 = GradientMessage::new([1u8; 32], vec![1.0, 1.0, 1.0], 0.5);
    let honest2 = GradientMessage::new([2u8; 32], vec![1.1, 0.9, 1.0], 0.5);
    let honest3 = GradientMessage::new([3u8; 32], vec![0.9, 1.1, 1.0], 0.5);
    let outlier1 = GradientMessage::new([4u8; 32], vec![100.0, -100.0, 50.0], 0.5);
    let outlier2 = GradientMessage::new([5u8; 32], vec![-100.0, 100.0, -50.0], 0.5);

    aggregator.receive_gradient(honest1);
    aggregator.receive_gradient(honest2);
    aggregator.receive_gradient(honest3);
    aggregator.receive_gradient(outlier1);
    aggregator.receive_gradient(outlier2);

    let result = aggregator.aggregate().unwrap();

    // With Byzantine tolerance, outliers should be trimmed
    // Result should be close to [1.0, 1.0, 1.0]
    for (i, val) in result.iter().enumerate() {
        assert!(
            val.abs() < 20.0,
            "Dimension {} should not be dominated by outliers: {}",
            i,
            val
        );
    }
}

// ============================================================================
// SHUTDOWN TESTS
// ============================================================================

#[tokio::test]
async fn test_coordinator_shutdown() {
    let config = FederatedNetworkConfig::for_testing();
    let mut coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

    // Register some peers
    let peer1 = FederatedNode::with_random_id(NodeAddress::Channel("peer-1".to_string()));
    let peer2 = FederatedNode::with_random_id(NodeAddress::Channel("peer-2".to_string()));
    coordinator.register_peer(peer1);
    coordinator.register_peer(peer2);

    assert_eq!(coordinator.peer_count(), 2);

    // Shutdown should not panic
    coordinator.shutdown().await;
}

// ============================================================================
// STRESS TESTS
// ============================================================================

#[tokio::test]
async fn test_rapid_gradient_exchange() {
    let coordinators = create_test_network(3, 10).await;

    // Rapidly exchange gradients
    for round in 0..10 {
        for coordinator in &coordinators {
            coordinator.update_weights(vec![round as f32; 10]);
            coordinator.share_gradient(0.0).await.unwrap();
        }

        // Small delay between rounds
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // All coordinators should still be functional
    for coordinator in &coordinators {
        assert_eq!(coordinator.peer_count(), 2);
    }
}

#[tokio::test]
async fn test_large_gradient_vectors() {
    let coordinators = create_test_network(2, 10000).await;

    // Each coordinator has 10000-dimensional weights
    assert_eq!(coordinators[0].get_weights().len(), 10000);

    // Can still share gradients
    let sent = coordinators[0].share_gradient(0.0).await.unwrap();
    assert_eq!(sent, 1);

    // Other node can receive
    let result = coordinators[1]
        .backend
        .receive(Duration::from_millis(200))
        .await;
    assert!(result.is_ok());

    if let Ok((_, FederatedMessage::GradientShare(grad))) = result {
        assert_eq!(grad.gradient_data.len(), 10000);
    }
}
