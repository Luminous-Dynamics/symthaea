//! Integration tests for RoundCoordinator
//!
//! Tests the full round lifecycle including:
//! - Multi-round coordination
//! - Event broadcasting
//! - Byzantine detection integration
//! - Concurrent submissions

use std::sync::Arc;
use std::time::Duration;

use fl_aggregator::{
    aggregator::{AggregatorConfig, AsyncAggregator},
    byzantine::Defense,
    coordinator::{CoordinatorConfig, CoordinatorEvent, RoundCoordinator, RoundState},
    Gradient,
};
use ndarray::Array1;
use tokio::time::timeout;

// =============================================================================
// Test Helpers
// =============================================================================

fn create_coordinator(min_participants: usize, expected_nodes: usize) -> RoundCoordinator {
    let config = CoordinatorConfig::default()
        .with_min_participants(min_participants)
        .with_round_timeout(Duration::from_secs(30))
        .without_holochain();

    let agg_config = AggregatorConfig::default()
        .with_expected_nodes(expected_nodes)
        .with_defense(Defense::FedAvg);

    let aggregator = AsyncAggregator::new(agg_config);
    RoundCoordinator::new(config, aggregator)
}

fn create_gradient(dim: usize, base_value: f32) -> Gradient {
    Array1::from_vec((0..dim).map(|i| base_value + i as f32 * 0.1).collect())
}

// =============================================================================
// Basic Round Lifecycle Tests
// =============================================================================

#[tokio::test]
async fn test_single_round_lifecycle() {
    let coordinator = create_coordinator(3, 5);

    // Register nodes
    coordinator.register_node("node_a").await.unwrap();
    coordinator.register_node("node_b").await.unwrap();
    coordinator.register_node("node_c").await.unwrap();

    // Start round
    coordinator.start_round().await.unwrap();
    let info = coordinator.round_info().await;
    assert_eq!(info.round, 1);
    assert_eq!(info.state, RoundState::Collecting);

    // Submit gradients
    coordinator
        .submit_gradient("node_a", create_gradient(100, 1.0))
        .await
        .unwrap();
    coordinator
        .submit_gradient("node_b", create_gradient(100, 2.0))
        .await
        .unwrap();
    coordinator
        .submit_gradient("node_c", create_gradient(100, 3.0))
        .await
        .unwrap();

    // Verify participants
    let info = coordinator.round_info().await;
    assert_eq!(info.participants.len(), 3);

    // Finalize
    coordinator.finalize_round().await.unwrap();

    let info = coordinator.round_info().await;
    assert_eq!(info.state, RoundState::Complete);
    assert!(info.result_hash.is_some());
}

#[tokio::test]
async fn test_multi_round_sequence() {
    let coordinator = create_coordinator(2, 4);

    // Register nodes
    for i in 0..4 {
        coordinator
            .register_node(format!("node_{}", i))
            .await
            .unwrap();
    }

    // Run 3 rounds
    for round_num in 1..=3 {
        coordinator.start_round().await.unwrap();

        // Submit from first 2 nodes
        coordinator
            .submit_gradient("node_0", create_gradient(50, round_num as f32))
            .await
            .unwrap();
        coordinator
            .submit_gradient("node_1", create_gradient(50, round_num as f32 + 0.5))
            .await
            .unwrap();

        coordinator.finalize_round().await.unwrap();

        let info = coordinator.round_info().await;
        assert_eq!(info.round, round_num);
        assert_eq!(info.state, RoundState::Complete);
    }

    assert_eq!(coordinator.current_round().await, 3);
}

// =============================================================================
// Event Broadcasting Tests
// =============================================================================

#[tokio::test]
async fn test_event_broadcasting() {
    let coordinator = create_coordinator(2, 3);
    let mut rx = coordinator.subscribe();

    coordinator.register_node("n1").await.unwrap();
    coordinator.register_node("n2").await.unwrap();

    // Start round - should emit RoundStarted
    coordinator.start_round().await.unwrap();

    let event = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("Timeout waiting for event")
        .expect("Channel closed");

    match event {
        CoordinatorEvent::RoundStarted {
            round,
            expected_participants,
        } => {
            assert_eq!(round, 1);
            assert_eq!(expected_participants, 2);
        }
        _ => panic!("Expected RoundStarted event, got {:?}", event),
    }

    // Submit gradient - should emit GradientReceived
    coordinator
        .submit_gradient("n1", create_gradient(10, 1.0))
        .await
        .unwrap();

    let event = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("Timeout")
        .expect("Channel closed");

    match event {
        CoordinatorEvent::GradientReceived {
            round,
            node_id,
            total_received,
        } => {
            assert_eq!(round, 1);
            assert_eq!(node_id, "n1");
            assert_eq!(total_received, 1);
        }
        _ => panic!("Expected GradientReceived event"),
    }

    // Complete round
    coordinator
        .submit_gradient("n2", create_gradient(10, 2.0))
        .await
        .unwrap();

    // Skip the GradientReceived for n2
    let _ = rx.recv().await;

    coordinator.finalize_round().await.unwrap();

    // Collect remaining events and find RoundCompleted
    let mut found_completed = false;
    for _ in 0..5 {
        if let Ok(Ok(event)) = timeout(Duration::from_millis(100), rx.recv()).await {
            if let CoordinatorEvent::RoundCompleted { round, participants, .. } = event {
                assert_eq!(round, 1);
                assert_eq!(participants.len(), 2);
                found_completed = true;
                break;
            }
        }
    }
    assert!(found_completed, "Did not receive RoundCompleted event");
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_duplicate_submission_rejected() {
    let coordinator = create_coordinator(2, 3);

    coordinator.register_node("node_x").await.unwrap();
    coordinator.start_round().await.unwrap();

    // First submission succeeds
    coordinator
        .submit_gradient("node_x", create_gradient(10, 1.0))
        .await
        .unwrap();

    // Second submission should fail
    let result = coordinator
        .submit_gradient("node_x", create_gradient(10, 2.0))
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_submission_before_round_start_rejected() {
    let coordinator = create_coordinator(2, 3);

    coordinator.register_node("early_bird").await.unwrap();

    // Try to submit before round starts
    let result = coordinator
        .submit_gradient("early_bird", create_gradient(10, 1.0))
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_unregistered_node_rejected() {
    let coordinator = create_coordinator(2, 3);

    coordinator.register_node("registered").await.unwrap();
    coordinator.start_round().await.unwrap();

    // Try to submit from unregistered node
    let result = coordinator
        .submit_gradient("unregistered", create_gradient(10, 1.0))
        .await;

    assert!(result.is_err());
}

// =============================================================================
// Concurrent Submission Tests
// =============================================================================

#[tokio::test]
async fn test_concurrent_submissions() {
    let coordinator = Arc::new(create_coordinator(5, 10));

    // Register nodes
    for i in 0..10 {
        coordinator
            .register_node(format!("concurrent_{}", i))
            .await
            .unwrap();
    }

    coordinator.start_round().await.unwrap();

    // Spawn concurrent submissions
    let mut handles = vec![];
    for i in 0..5 {
        let coord = Arc::clone(&coordinator);
        let handle = tokio::spawn(async move {
            let node_id = format!("concurrent_{}", i);
            let gradient = create_gradient(100, i as f32);
            coord.submit_gradient(&node_id, gradient).await
        });
        handles.push(handle);
    }

    // Wait for all submissions
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let info = coordinator.round_info().await;
    assert_eq!(info.participants.len(), 5);

    coordinator.finalize_round().await.unwrap();
    assert_eq!(coordinator.round_info().await.state, RoundState::Complete);
}

// =============================================================================
// Defense Integration Tests
// =============================================================================

#[tokio::test]
async fn test_krum_defense_integration() {
    let config = CoordinatorConfig::default()
        .with_min_participants(5) // Krum needs n >= 2f + 3 = 5 for f=1
        .with_round_timeout(Duration::from_secs(30))
        .without_holochain();

    let agg_config = AggregatorConfig::default()
        .with_expected_nodes(6)
        .with_defense(Defense::Krum { f: 1 });

    let aggregator = AsyncAggregator::new(agg_config);
    let coordinator = RoundCoordinator::new(config, aggregator);

    // Register nodes
    for i in 0..6 {
        coordinator
            .register_node(format!("krum_node_{}", i))
            .await
            .unwrap();
    }

    coordinator.start_round().await.unwrap();

    // Submit 4 honest gradients (similar values)
    for i in 0..4 {
        coordinator
            .submit_gradient(
                &format!("krum_node_{}", i),
                Array1::from_vec(vec![1.0, 1.0, 1.0]),
            )
            .await
            .unwrap();
    }

    // Submit 1 Byzantine gradient (extreme outlier)
    coordinator
        .submit_gradient("krum_node_4", Array1::from_vec(vec![1000.0, 1000.0, 1000.0]))
        .await
        .unwrap();

    coordinator.finalize_round().await.unwrap();

    let info = coordinator.round_info().await;
    assert_eq!(info.state, RoundState::Complete);
    // Krum should have selected one of the honest gradients
}

// =============================================================================
// State Machine Tests
// =============================================================================

#[tokio::test]
async fn test_state_transitions() {
    let coordinator = create_coordinator(2, 3);

    coordinator.register_node("s1").await.unwrap();
    coordinator.register_node("s2").await.unwrap();

    // Initial state
    assert_eq!(coordinator.round_info().await.state, RoundState::Waiting);

    // Start round
    coordinator.start_round().await.unwrap();
    assert_eq!(coordinator.round_info().await.state, RoundState::Collecting);

    // Submit
    coordinator
        .submit_gradient("s1", create_gradient(5, 1.0))
        .await
        .unwrap();
    coordinator
        .submit_gradient("s2", create_gradient(5, 2.0))
        .await
        .unwrap();

    // Still collecting until finalize
    assert_eq!(coordinator.round_info().await.state, RoundState::Collecting);

    // Finalize
    coordinator.finalize_round().await.unwrap();
    assert_eq!(coordinator.round_info().await.state, RoundState::Complete);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[tokio::test]
async fn test_minimum_participants_boundary() {
    let coordinator = create_coordinator(2, 5);

    coordinator.register_node("min1").await.unwrap();
    coordinator.register_node("min2").await.unwrap();

    coordinator.start_round().await.unwrap();

    // Submit exactly minimum
    coordinator
        .submit_gradient("min1", create_gradient(10, 1.0))
        .await
        .unwrap();
    coordinator
        .submit_gradient("min2", create_gradient(10, 2.0))
        .await
        .unwrap();

    // Should be able to finalize with exactly min_participants
    coordinator.finalize_round().await.unwrap();
    assert_eq!(coordinator.round_info().await.state, RoundState::Complete);
}

#[tokio::test]
async fn test_large_gradient_handling() {
    let coordinator = create_coordinator(2, 3);

    coordinator.register_node("large1").await.unwrap();
    coordinator.register_node("large2").await.unwrap();

    coordinator.start_round().await.unwrap();

    // Large gradients (1M elements)
    let large_dim = 1_000_000;
    coordinator
        .submit_gradient("large1", create_gradient(large_dim, 1.0))
        .await
        .unwrap();
    coordinator
        .submit_gradient("large2", create_gradient(large_dim, 2.0))
        .await
        .unwrap();

    coordinator.finalize_round().await.unwrap();

    let info = coordinator.round_info().await;
    assert_eq!(info.state, RoundState::Complete);
    assert!(info.result_hash.is_some());
}

// =============================================================================
// Round Info Tests
// =============================================================================

#[tokio::test]
async fn test_round_info_completeness() {
    let coordinator = create_coordinator(2, 3);

    coordinator.register_node("info1").await.unwrap();
    coordinator.register_node("info2").await.unwrap();

    coordinator.start_round().await.unwrap();

    let info = coordinator.round_info().await;
    assert_eq!(info.round, 1);
    assert_eq!(info.state, RoundState::Collecting);
    assert!(info.started_at > 0);
    assert_eq!(info.ended_at, 0);
    assert!(info.participants.is_empty());
    assert!(info.result_hash.is_none());

    coordinator
        .submit_gradient("info1", create_gradient(10, 1.0))
        .await
        .unwrap();
    coordinator
        .submit_gradient("info2", create_gradient(10, 2.0))
        .await
        .unwrap();

    coordinator.finalize_round().await.unwrap();

    let info = coordinator.round_info().await;
    assert_eq!(info.state, RoundState::Complete);
    // ended_at should be >= started_at (may be equal if test runs fast)
    assert!(info.ended_at >= info.started_at);
    assert_eq!(info.participants.len(), 2);
    assert!(info.result_hash.is_some());
}
