//! Unit tests for the core Aggregator module.
//!
//! Tests cover:
//! - Basic submission and aggregation
//! - Memory limits
//! - Round lifecycle
//! - Error handling
//! - Concurrent async operations

use fl_aggregator::aggregator::{Aggregator, AggregatorConfig, AsyncAggregator};
use fl_aggregator::byzantine::Defense;
use fl_aggregator::Gradient;
use ndarray::Array1;

// ============================================================================
// Test Utilities
// ============================================================================

fn create_gradient(dim: usize, value: f32) -> Gradient {
    Array1::from_elem(dim, value)
}

fn create_random_gradient(dim: usize, seed: u64) -> Gradient {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)))
}

// ============================================================================
// Basic Aggregator Tests
// ============================================================================

#[test]
fn test_aggregator_creation() {
    let config = AggregatorConfig::default();
    let aggregator = Aggregator::new(config);

    assert_eq!(aggregator.current_round(), 0);
    assert_eq!(aggregator.submission_count(), 0);
}

#[test]
fn test_submit_gradient() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    let grad = create_gradient(100, 1.0);
    let result = aggregator.submit("node1", grad);

    assert!(result.is_ok());
    assert_eq!(aggregator.submission_count(), 1);
}

#[test]
fn test_duplicate_submission_rejected() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    let grad = create_gradient(100, 1.0);
    aggregator.submit("node1", grad.clone()).unwrap();

    // Second submission from same node should fail
    let result = aggregator.submit("node1", grad);
    assert!(result.is_err());
    assert_eq!(aggregator.submission_count(), 1);
}

#[test]
fn test_full_round_aggregation() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(3)
        .with_defense(Defense::FedAvg);
    let mut aggregator = Aggregator::new(config);

    // Submit from 3 nodes
    aggregator.submit("node1", create_gradient(100, 1.0)).unwrap();
    aggregator.submit("node2", create_gradient(100, 2.0)).unwrap();
    aggregator.submit("node3", create_gradient(100, 3.0)).unwrap();

    assert!(aggregator.is_round_complete());

    // Finalize round
    let result = aggregator.finalize_round().unwrap();

    // Average of 1, 2, 3 = 2
    assert!((result[0] - 2.0).abs() < 0.001);
    assert_eq!(aggregator.current_round(), 1);
    assert_eq!(aggregator.submission_count(), 0);
}

#[test]
fn test_partial_round_aggregation() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(5)
        .with_defense(Defense::FedAvg);
    let mut aggregator = Aggregator::new(config);

    // Submit from only 3 nodes (not complete)
    aggregator.submit("node1", create_gradient(100, 1.0)).unwrap();
    aggregator.submit("node2", create_gradient(100, 2.0)).unwrap();
    aggregator.submit("node3", create_gradient(100, 3.0)).unwrap();

    assert!(!aggregator.is_round_complete());

    // Should still be able to finalize with partial data
    let result = aggregator.finalize_round();
    assert!(result.is_ok());
}

// ============================================================================
// Memory Limit Tests
// ============================================================================

#[test]
fn test_memory_limit_enforcement() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(5)
        .with_max_memory(1000); // Very small limit
    let mut aggregator = Aggregator::new(config);

    // Create large gradient that exceeds memory limit
    let large_grad = create_gradient(1_000_000, 1.0); // ~4MB

    let result = aggregator.submit("node1", large_grad);

    // Should fail due to memory limit
    assert!(result.is_err());
}

#[test]
fn test_memory_tracking() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(3)
        .with_max_memory(100_000_000); // 100MB
    let mut aggregator = Aggregator::new(config);

    let initial_memory = aggregator.memory_usage();
    assert_eq!(initial_memory, 0);

    // Submit gradient
    let grad = create_gradient(10000, 1.0); // ~40KB
    aggregator.submit("node1", grad).unwrap();

    let after_memory = aggregator.memory_usage();
    assert!(after_memory > 0);
    assert!(after_memory >= 10000 * 4); // At least gradient size
}

// ============================================================================
// Gradient Validation Tests
// ============================================================================

#[test]
fn test_nan_gradient_rejected() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    let mut grad = create_gradient(100, 1.0);
    grad[50] = f32::NAN; // Insert NaN

    let result = aggregator.submit("node1", grad);
    assert!(result.is_err());
}

#[test]
fn test_infinity_gradient_rejected() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    let mut grad = create_gradient(100, 1.0);
    grad[50] = f32::INFINITY;

    let result = aggregator.submit("node1", grad);
    assert!(result.is_err());
}

#[test]
fn test_negative_infinity_rejected() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    let mut grad = create_gradient(100, 1.0);
    grad[50] = f32::NEG_INFINITY;

    let result = aggregator.submit("node1", grad);
    assert!(result.is_err());
}

#[test]
fn test_dimension_limit() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    // Config default max is 100M, create one larger
    let huge_grad = create_gradient(101_000_000, 1.0);

    let result = aggregator.submit("node1", huge_grad);
    assert!(result.is_err());
}

// ============================================================================
// Round Lifecycle Tests
// ============================================================================

#[test]
fn test_multiple_rounds() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(2)
        .with_defense(Defense::FedAvg);
    let mut aggregator = Aggregator::new(config);

    // Round 0
    aggregator.submit("node1", create_gradient(10, 1.0)).unwrap();
    aggregator.submit("node2", create_gradient(10, 3.0)).unwrap();
    let r0 = aggregator.finalize_round().unwrap();
    assert!((r0[0] - 2.0).abs() < 0.001);

    // Round 1
    aggregator.submit("node1", create_gradient(10, 5.0)).unwrap();
    aggregator.submit("node2", create_gradient(10, 7.0)).unwrap();
    let r1 = aggregator.finalize_round().unwrap();
    assert!((r1[0] - 6.0).abs() < 0.001);

    assert_eq!(aggregator.current_round(), 2);
}

#[test]
fn test_reset() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    aggregator.submit("node1", create_gradient(10, 1.0)).unwrap();
    aggregator.submit("node2", create_gradient(10, 2.0)).unwrap();

    assert_eq!(aggregator.submission_count(), 2);

    // Reset clears everything including round
    aggregator.reset();

    assert_eq!(aggregator.submission_count(), 0);
    assert_eq!(aggregator.current_round(), 0);
}

// ============================================================================
// Defense Algorithm Integration Tests
// ============================================================================

#[test]
fn test_krum_defense() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(5)
        .with_defense(Defense::Krum { f: 1 });
    let mut aggregator = Aggregator::new(config);

    // Submit honest gradients
    for i in 0..4 {
        aggregator.submit(
            format!("honest{}", i),
            create_gradient(100, 1.0 + i as f32 * 0.1)
        ).unwrap();
    }

    // Submit Byzantine gradient (outlier)
    aggregator.submit("byzantine", create_gradient(100, 100.0)).unwrap();

    let result = aggregator.finalize_round().unwrap();

    // Krum should select an honest gradient, result should be close to 1.x
    assert!(result[0] < 10.0, "Krum should select honest gradient");
}

#[test]
fn test_median_defense() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(5)
        .with_defense(Defense::Median);
    let mut aggregator = Aggregator::new(config);

    aggregator.submit("node1", create_gradient(10, 1.0)).unwrap();
    aggregator.submit("node2", create_gradient(10, 2.0)).unwrap();
    aggregator.submit("node3", create_gradient(10, 3.0)).unwrap();
    aggregator.submit("node4", create_gradient(10, 4.0)).unwrap();
    aggregator.submit("node5", create_gradient(10, 1000.0)).unwrap(); // Outlier

    let result = aggregator.finalize_round().unwrap();

    // Median of [1, 2, 3, 4, 1000] = 3
    assert!((result[0] - 3.0).abs() < 0.001);
}

#[test]
fn test_trimmed_mean_defense() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(5)
        .with_defense(Defense::TrimmedMean { beta: 0.2 });
    let mut aggregator = Aggregator::new(config);

    aggregator.submit("node1", create_gradient(10, 0.0)).unwrap();   // Trimmed (low)
    aggregator.submit("node2", create_gradient(10, 2.0)).unwrap();
    aggregator.submit("node3", create_gradient(10, 3.0)).unwrap();
    aggregator.submit("node4", create_gradient(10, 4.0)).unwrap();
    aggregator.submit("node5", create_gradient(10, 100.0)).unwrap(); // Trimmed (high)

    let result = aggregator.finalize_round().unwrap();

    // After trimming: [2, 3, 4], mean = 3
    assert!((result[0] - 3.0).abs() < 0.001);
}

// ============================================================================
// Async Aggregator Tests
// ============================================================================

#[tokio::test]
async fn test_async_aggregator_creation() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let aggregator = AsyncAggregator::new(config);

    let status = aggregator.status().await;
    assert_eq!(status.round, 0);
    assert_eq!(status.submitted_nodes, 0);
}

#[tokio::test]
async fn test_async_register_and_submit() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let aggregator = AsyncAggregator::new(config);

    // Must register before submit
    aggregator.register_node("node1").await.unwrap();

    let grad = create_gradient(100, 1.0);
    let result = aggregator.submit("node1", grad).await;

    assert!(result.is_ok());

    let status = aggregator.status().await;
    assert_eq!(status.submitted_nodes, 1);
}

#[tokio::test]
async fn test_async_unregistered_node_rejected() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let aggregator = AsyncAggregator::new(config);

    // Try to submit without registering
    let grad = create_gradient(100, 1.0);
    let result = aggregator.submit("unregistered", grad).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_async_concurrent_submissions() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(10)
        .with_defense(Defense::FedAvg);
    let aggregator = std::sync::Arc::new(AsyncAggregator::new(config));

    // Register all nodes first
    for i in 0..10 {
        aggregator.register_node(format!("node{}", i)).await.unwrap();
    }

    // Spawn concurrent submissions
    let mut handles = vec![];
    for i in 0..10 {
        let agg = aggregator.clone();
        let handle = tokio::spawn(async move {
            let grad = create_gradient(100, i as f32);
            agg.submit(format!("node{}", i), grad).await
        });
        handles.push(handle);
    }

    // Wait for all
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let status = aggregator.status().await;
    assert_eq!(status.submitted_nodes, 10);
    assert!(status.is_complete);
}

#[tokio::test]
async fn test_async_full_round() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(3)
        .with_defense(Defense::FedAvg);
    let aggregator = AsyncAggregator::new(config);

    // Register and submit
    for i in 1..=3 {
        aggregator.register_node(format!("node{}", i)).await.unwrap();
        aggregator.submit(format!("node{}", i), create_gradient(10, i as f32)).await.unwrap();
    }

    let result = aggregator.force_finalize().await.unwrap();

    assert!((result[0] - 2.0).abs() < 0.001);

    let status = aggregator.status().await;
    assert_eq!(status.round, 1);
}

#[tokio::test]
async fn test_async_get_aggregated_gradient() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(2)
        .with_defense(Defense::FedAvg);
    let aggregator = AsyncAggregator::new(config);

    aggregator.register_node("node1").await.unwrap();
    aggregator.register_node("node2").await.unwrap();

    // Not complete yet - should return None
    aggregator.submit("node1", create_gradient(10, 1.0)).await.unwrap();
    let result = aggregator.get_aggregated_gradient().await.unwrap();
    assert!(result.is_none());

    // Now complete - should return aggregated gradient
    aggregator.submit("node2", create_gradient(10, 3.0)).await.unwrap();
    let result = aggregator.get_aggregated_gradient().await.unwrap();
    assert!(result.is_some());
    assert!((result.unwrap()[0] - 2.0).abs() < 0.001);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_finalize_empty_round() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    // Try to finalize with no submissions
    let result = aggregator.finalize_round();
    assert!(result.is_err());
}

#[test]
fn test_dimension_mismatch() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    aggregator.submit("node1", create_gradient(100, 1.0)).unwrap();

    // Different dimension should fail
    let result = aggregator.submit("node2", create_gradient(200, 1.0));
    assert!(result.is_err());
}

// ============================================================================
// Node ID Tests
// ============================================================================

#[test]
fn test_various_node_id_formats() {
    let config = AggregatorConfig::default().with_expected_nodes(5);
    let mut aggregator = Aggregator::new(config);

    // Test various node ID formats
    aggregator.submit("simple", create_gradient(10, 1.0)).unwrap();
    aggregator.submit("with-dash", create_gradient(10, 1.0)).unwrap();
    aggregator.submit("with_underscore", create_gradient(10, 1.0)).unwrap();
    aggregator.submit("CamelCase", create_gradient(10, 1.0)).unwrap();
    aggregator.submit("123numeric", create_gradient(10, 1.0)).unwrap();

    assert_eq!(aggregator.submission_count(), 5);
}

#[test]
fn test_string_and_str_node_ids() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    // Using &str
    aggregator.submit("str_id", create_gradient(10, 1.0)).unwrap();

    // Using String
    aggregator.submit(String::from("string_id"), create_gradient(10, 1.0)).unwrap();

    assert_eq!(aggregator.submission_count(), 2);
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_many_nodes() {
    let num_nodes = 100;
    let config = AggregatorConfig::default()
        .with_expected_nodes(num_nodes)
        .with_defense(Defense::FedAvg);
    let mut aggregator = Aggregator::new(config);

    for i in 0..num_nodes {
        aggregator.submit(
            format!("node{}", i),
            create_random_gradient(1000, i as u64)
        ).unwrap();
    }

    assert!(aggregator.is_round_complete());

    let result = aggregator.finalize_round();
    assert!(result.is_ok());
}

#[test]
fn test_large_gradients() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(3)
        .with_defense(Defense::FedAvg);
    let mut aggregator = Aggregator::new(config);

    let dim = 1_000_000; // 1M parameters

    aggregator.submit("node1", create_gradient(dim, 1.0)).unwrap();
    aggregator.submit("node2", create_gradient(dim, 2.0)).unwrap();
    aggregator.submit("node3", create_gradient(dim, 3.0)).unwrap();

    let result = aggregator.finalize_round().unwrap();
    assert_eq!(result.len(), dim);
    assert!((result[0] - 2.0).abs() < 0.001);
}

// ============================================================================
// Configuration Builder Tests
// ============================================================================

#[test]
fn test_config_builder() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(50)
        .with_defense(Defense::Median)
        .with_max_memory(1_000_000_000);

    assert_eq!(config.expected_nodes, 50);
    assert_eq!(config.defense, Defense::Median);
    assert_eq!(config.max_memory_bytes, 1_000_000_000);
}
