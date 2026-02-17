//! Comprehensive integration tests for the unified FL aggregator.
//!
//! These tests verify:
//! - Dense gradient flow with Byzantine defense
//! - Hypervector bundling operations
//! - Binary hypervector majority voting
//! - Payload type tracking and conversions
//! - Byzantine resilience across defense algorithms
//! - Round lifecycle management

mod fixtures;

use fl_aggregator::{
    Aggregator, AggregatorConfig, AsyncAggregator, Defense, DefenseConfig,
    byzantine::ByzantineAggregator,
    payload::{
        Payload, PayloadType, UnifiedPayload,
        DenseGradient, Hypervector, BinaryHypervector,
        SparseGradient, QuantizedGradient,
        DenseAggregatable, HdcAggregatable,
    },
};

use fixtures::{
    generate_clustered_dense_gradients,
    generate_byzantine_gradient,
    generate_byzantine_gradient_with_magnitude,
    generate_clustered_hypervectors,
    generate_binary_hypervectors_with_known_majority,
    compute_mean,
    compute_median,
    euclidean_distance,
    l2_norm,
    assert_gradients_close,
};

use ndarray::Array1;

// =============================================================================
// Test: Dense Gradient Flow with Krum Defense
// =============================================================================

/// Test that Krum defense correctly filters Byzantine gradients.
///
/// Scenario:
/// - 4 honest nodes submit gradients clustered around a center
/// - 1 Byzantine node submits an outlier gradient
/// - Krum should select a gradient from the honest cluster
#[test]
fn test_dense_gradient_flow() {
    const DIM: usize = 100;
    const HONEST_NODES: usize = 4;

    // Generate honest gradients
    let (honest_gradients, center) = generate_clustered_dense_gradients(HONEST_NODES, DIM);

    // Generate Byzantine gradient
    let byzantine_gradient = generate_byzantine_gradient(DIM);

    // Create aggregator with Krum defense (f=1 means tolerate 1 Byzantine)
    let config = AggregatorConfig::default()
        .with_expected_nodes(HONEST_NODES + 1)
        .with_defense(Defense::Krum { f: 1 });

    let mut aggregator = Aggregator::new(config);

    // Submit honest gradients
    for (i, gradient) in honest_gradients.iter().enumerate() {
        aggregator
            .submit(format!("honest_{}", i), gradient.clone())
            .expect("Honest submission should succeed");
    }

    // Submit Byzantine gradient
    aggregator
        .submit("byzantine_0", byzantine_gradient.clone())
        .expect("Byzantine submission should succeed");

    // Verify round is complete
    assert!(aggregator.is_round_complete());

    // Finalize and get result
    let result = aggregator.finalize_round().expect("Finalization should succeed");

    // Verify result is close to honest cluster, not Byzantine
    let dist_to_center = euclidean_distance(&result, &center);
    let dist_to_byzantine = euclidean_distance(&result, &byzantine_gradient);

    // Result should be much closer to honest cluster than to Byzantine
    assert!(
        dist_to_center < dist_to_byzantine / 10.0,
        "Result should be close to honest cluster. \
         Distance to center: {}, Distance to Byzantine: {}",
        dist_to_center, dist_to_byzantine
    );

    // Result should be one of the honest gradients (Krum selects, doesn't average)
    let is_honest = honest_gradients
        .iter()
        .any(|g| euclidean_distance(&result, g) < 0.01);
    assert!(is_honest, "Krum should select an honest gradient");

    println!(
        "Dense gradient flow test passed. \
         Distance to center: {:.4}, Distance to Byzantine: {:.4}",
        dist_to_center, dist_to_byzantine
    );
}

// =============================================================================
// Test: Hypervector Bundling
// =============================================================================

/// Test that hypervector bundling produces component-wise average.
#[test]
fn test_hypervector_bundling() {
    // Create hypervectors with known values
    let hv1 = Hypervector::new(vec![10, 20, 30, 40, 50]);
    let hv2 = Hypervector::new(vec![20, 30, 40, 50, 60]);
    let hv3 = Hypervector::new(vec![30, 40, 50, 60, 70]);

    // Bundle them
    let refs = [&hv1, &hv2, &hv3];
    let bundled = Hypervector::bundle(&refs).expect("Bundling should succeed");

    // Expected: component-wise average
    // (10+20+30)/3=20, (20+30+40)/3=30, (30+40+50)/3=40, etc.
    let expected = vec![20i8, 30, 40, 50, 60];

    assert_eq!(bundled.components, expected);
    assert_eq!(bundled.dimension(), 5);
    assert!(bundled.is_valid());
}

/// Test bundling with larger hypervectors.
#[test]
fn test_hypervector_bundling_large() {
    let dim = 1024;
    let n = 10;

    let (hypervectors, _expected_base) = generate_clustered_hypervectors(n, dim);

    // Bundle
    let refs: Vec<_> = hypervectors.iter().collect();
    let bundled = Hypervector::bundle(&refs).expect("Bundling should succeed");

    // Result should be close to base (perturbations average out)
    assert_eq!(bundled.components.len(), dim);

    // Check similarity - bundled should be similar to any of the input vectors
    let sim = bundled.cosine_similarity(&hypervectors[0]);
    assert!(
        sim > 0.9,
        "Bundled vector should be similar to inputs: similarity = {}",
        sim
    );
}

/// Test bundling empty vector fails gracefully.
#[test]
fn test_hypervector_bundle_empty() {
    let refs: Vec<&Hypervector> = vec![];
    let result = Hypervector::bundle(&refs);
    assert!(result.is_none(), "Bundling empty vector should return None");
}

/// Test bundling mismatched dimensions fails.
#[test]
fn test_hypervector_bundle_dimension_mismatch() {
    let hv1 = Hypervector::new(vec![10, 20, 30]);
    let hv2 = Hypervector::new(vec![10, 20]); // Different dimension

    let refs = [&hv1, &hv2];
    let result = Hypervector::bundle(&refs);
    assert!(result.is_none(), "Bundling mismatched dimensions should return None");
}

// =============================================================================
// Test: Binary Hypervector Majority Voting
// =============================================================================

/// Test that binary hypervector bundling uses majority voting correctly.
#[test]
fn test_binary_majority_voting() {
    // Create BinaryHVs with known bit patterns
    // Pattern: majority should win at each position
    let bhv1 = BinaryHypervector::from_bits(&[true, true, false, false, true]);
    let bhv2 = BinaryHypervector::from_bits(&[true, false, true, false, true]);
    let bhv3 = BinaryHypervector::from_bits(&[true, true, true, false, false]);

    // Bundle with majority voting
    let refs = [&bhv1, &bhv2, &bhv3];
    let bundled = BinaryHypervector::bundle(&refs).expect("Bundle should succeed");

    // Expected majority:
    // Bit 0: true (3/3)
    // Bit 1: true (2/3)
    // Bit 2: true (2/3)
    // Bit 3: false (0/3)
    // Bit 4: true (2/3)
    assert_eq!(bundled.get_bit(0), Some(true));
    assert_eq!(bundled.get_bit(1), Some(true));
    assert_eq!(bundled.get_bit(2), Some(true));
    assert_eq!(bundled.get_bit(3), Some(false));
    assert_eq!(bundled.get_bit(4), Some(true));
}

/// Test binary majority voting with known result.
#[test]
fn test_binary_majority_voting_known_result() {
    let (hvs, expected) = generate_binary_hypervectors_with_known_majority(7, 16);

    let refs: Vec<_> = hvs.iter().collect();
    let bundled = BinaryHypervector::bundle(&refs).expect("Bundle should succeed");

    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(
            bundled.get_bit(i), Some(exp),
            "Bit {} mismatch: expected {}",
            i, exp
        );
    }
}

/// Test binary HV operations - XOR binding.
#[test]
fn test_binary_hypervector_xor() {
    let bhv1 = BinaryHypervector::from_bits(&[true, false, true, false]);
    let bhv2 = BinaryHypervector::from_bits(&[true, true, false, false]);

    let xored = bhv1.xor(&bhv2).expect("XOR should succeed");

    // XOR: true^true=false, false^true=true, true^false=true, false^false=false
    assert_eq!(xored.get_bit(0), Some(false));
    assert_eq!(xored.get_bit(1), Some(true));
    assert_eq!(xored.get_bit(2), Some(true));
    assert_eq!(xored.get_bit(3), Some(false));
}

/// Test Hamming distance calculation.
#[test]
fn test_binary_hypervector_hamming_distance() {
    let bhv1 = BinaryHypervector::from_bits(&[true, true, true, true, false, false, false, false]);
    let bhv2 = BinaryHypervector::from_bits(&[true, true, false, false, true, true, false, false]);

    let distance = bhv1.hamming_distance(&bhv2);
    // Positions 2,3,4,5 differ = 4 bits
    assert_eq!(distance, 4);

    let similarity = bhv1.hamming_similarity(&bhv2);
    // 4/8 = 0.5 difference, so similarity = 0.5
    assert!((similarity - 0.5).abs() < 0.01);
}

// =============================================================================
// Test: Payload Type Validation
// =============================================================================

/// Test that each payload type correctly reports its type.
#[test]
fn test_payload_type_tracking() {
    // DenseGradient
    let dense = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(dense.payload_type(), PayloadType::DenseGradient);
    assert!(dense.is_valid());
    assert_eq!(dense.dimension(), 3);
    assert_eq!(dense.size_bytes(), 12); // 3 * 4 bytes

    // Hypervector
    let hyper = Hypervector::new(vec![10, 20, 30, 40]);
    assert_eq!(hyper.payload_type(), PayloadType::HyperEncoded);
    assert!(hyper.is_valid());
    assert_eq!(hyper.dimension(), 4);
    assert_eq!(hyper.size_bytes(), 4); // 4 * 1 byte

    // BinaryHypervector
    let binary = BinaryHypervector::from_bits(&[true, false, true, false, true, false, true, false]);
    assert_eq!(binary.payload_type(), PayloadType::BinaryHypervector);
    assert!(binary.is_valid());
    assert_eq!(binary.dimension(), 8);
    assert_eq!(binary.size_bytes(), 1); // 8 bits = 1 byte

    // SparseGradient
    let sparse = SparseGradient::new(
        vec![0, 5, 10],
        vec![1.0, 2.0, 3.0],
        100,
    );
    assert_eq!(sparse.payload_type(), PayloadType::SparseGradient);
    assert!(sparse.is_valid());
    assert_eq!(sparse.dimension(), 100);

    // QuantizedGradient
    let quant_dense = DenseGradient::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    let quantized = QuantizedGradient::from_dense(&quant_dense, 8);
    assert_eq!(quantized.payload_type(), PayloadType::QuantizedGradient);
    assert!(quantized.is_valid());
    assert_eq!(quantized.dimension, 5);
}

/// Test UnifiedPayload type tracking.
#[test]
fn test_unified_payload_type_tracking() {
    let dense = UnifiedPayload::Dense(DenseGradient::from_vec(vec![1.0, 2.0]));
    assert_eq!(dense.payload_type(), PayloadType::DenseGradient);

    let hyper = UnifiedPayload::Hyper(Hypervector::new(vec![10, 20]));
    assert_eq!(hyper.payload_type(), PayloadType::HyperEncoded);

    let binary = UnifiedPayload::Binary(BinaryHypervector::from_bits(&[true, false]));
    assert_eq!(binary.payload_type(), PayloadType::BinaryHypervector);

    let sparse = UnifiedPayload::Sparse(SparseGradient::new(vec![0], vec![1.0], 10));
    assert_eq!(sparse.payload_type(), PayloadType::SparseGradient);

    let quantized = UnifiedPayload::Quantized(
        QuantizedGradient::from_dense(&DenseGradient::from_vec(vec![1.0]), 8)
    );
    assert_eq!(quantized.payload_type(), PayloadType::QuantizedGradient);
}

/// Test to_dense() conversion for convertible types.
#[test]
fn test_payload_to_dense_conversion() {
    // Dense -> Dense (identity)
    let dense_payload = UnifiedPayload::Dense(DenseGradient::from_vec(vec![1.0, 2.0, 3.0]));
    let dense_result = dense_payload.to_dense();
    assert!(dense_result.is_some());
    assert_eq!(dense_result.unwrap().to_vec(), vec![1.0, 2.0, 3.0]);

    // Sparse -> Dense
    let sparse = SparseGradient::new(vec![0, 2], vec![1.0, 3.0], 4);
    let sparse_payload = UnifiedPayload::Sparse(sparse);
    let sparse_result = sparse_payload.to_dense();
    assert!(sparse_result.is_some());
    let sparse_dense = sparse_result.unwrap();
    assert_eq!(sparse_dense.to_vec(), vec![1.0, 0.0, 3.0, 0.0]);

    // Quantized -> Dense
    let quant_source = DenseGradient::from_vec(vec![0.0, 1.0, 2.0]);
    let quantized = QuantizedGradient::from_dense(&quant_source, 8);
    let quant_payload = UnifiedPayload::Quantized(quantized);
    let quant_result = quant_payload.to_dense();
    assert!(quant_result.is_some());
    let quant_dense = quant_result.unwrap();
    // Values should be approximately equal (within quantization error)
    for (a, b) in quant_source.to_vec().iter().zip(quant_dense.to_vec().iter()) {
        assert!((a - b).abs() < 0.1, "Quantization error too large: {} vs {}", a, b);
    }

    // Hyper -> Dense (not convertible)
    let hyper_payload = UnifiedPayload::Hyper(Hypervector::new(vec![10, 20]));
    assert!(hyper_payload.to_dense().is_none());

    // Binary -> Dense (not convertible)
    let binary_payload = UnifiedPayload::Binary(BinaryHypervector::from_bits(&[true]));
    assert!(binary_payload.to_dense().is_none());
}

/// Test DenseAggregatable trait implementations.
#[test]
fn test_dense_aggregatable_trait() {
    // DenseGradient round-trip
    let dense = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
    let converted = dense.to_dense_gradient();
    let back = DenseGradient::from_dense_gradient(converted);
    assert_eq!(dense.to_vec(), back.to_vec());

    // SparseGradient round-trip
    let sparse = SparseGradient::new(vec![0, 2], vec![1.0, 3.0], 4);
    let sparse_dense = sparse.to_dense_gradient();
    assert_eq!(sparse_dense.to_vec(), vec![1.0, 0.0, 3.0, 0.0]);

    // QuantizedGradient round-trip
    let quant_source = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
    let quantized = QuantizedGradient::from_dense(&quant_source, 8);
    let quant_dense = quantized.to_dense_gradient();
    assert_eq!(quant_dense.dimension(), 3);
}

/// Test HdcAggregatable trait implementations.
#[test]
fn test_hdc_aggregatable_trait() {
    // Hypervector bundling via trait
    let hv1 = Hypervector::new(vec![10, 20, 30]);
    let hv2 = Hypervector::new(vec![20, 30, 40]);
    let bundled = Hypervector::bundle_hdc(&[&hv1, &hv2]).expect("Bundle should succeed");
    assert_eq!(bundled.components, vec![15, 25, 35]);

    // Similarity via trait
    let sim = hv1.similarity(&hv2);
    assert!(sim > 0.9, "Similar vectors should have high similarity");

    // BinaryHypervector bundling via trait
    let bhv1 = BinaryHypervector::from_bits(&[true, false]);
    let bhv2 = BinaryHypervector::from_bits(&[true, true]);
    let bhv3 = BinaryHypervector::from_bits(&[true, false]);
    let bhv_bundled = BinaryHypervector::bundle_hdc(&[&bhv1, &bhv2, &bhv3])
        .expect("Bundle should succeed");
    assert_eq!(bhv_bundled.get_bit(0), Some(true)); // 3/3 = majority true
    assert_eq!(bhv_bundled.get_bit(1), Some(false)); // 1/3 = majority false

    // Hamming similarity
    let bhv_sim = bhv1.similarity(&bhv2);
    assert!((bhv_sim - 0.5).abs() < 0.01, "1/2 bits same = 0.5 similarity");
}

// =============================================================================
// Test: Byzantine Resilience - Krum
// =============================================================================

/// Test Krum defense filters outliers correctly.
#[test]
fn test_byzantine_resilience_krum() {
    const DIM: usize = 50;
    const HONEST: usize = 6;
    const BYZANTINE: usize = 2;

    let (honest, center) = generate_clustered_dense_gradients(HONEST, DIM);
    let byzantine1 = generate_byzantine_gradient(DIM);
    let byzantine2 = generate_byzantine_gradient_with_magnitude(DIM, 200.0);

    // Krum with f=2 needs n >= 2*2+3 = 7 gradients. We have 8.
    let config = DefenseConfig::with_defense(Defense::Krum { f: BYZANTINE });
    let aggregator = ByzantineAggregator::new(config);

    let mut all_gradients = honest.clone();
    all_gradients.push(byzantine1.clone());
    all_gradients.push(byzantine2.clone());

    let result = aggregator.aggregate(&all_gradients).expect("Aggregation should succeed");

    // Result should be close to honest cluster
    let dist_to_center = euclidean_distance(&result, &center);
    let dist_to_byz1 = euclidean_distance(&result, &byzantine1);
    let dist_to_byz2 = euclidean_distance(&result, &byzantine2);

    assert!(
        dist_to_center < dist_to_byz1 / 5.0,
        "Krum result should be far from Byzantine 1"
    );
    assert!(
        dist_to_center < dist_to_byz2 / 5.0,
        "Krum result should be far from Byzantine 2"
    );

    println!("Krum resilience: dist_to_center={:.4}, dist_to_byz1={:.4}, dist_to_byz2={:.4}",
             dist_to_center, dist_to_byz1, dist_to_byz2);
}

// =============================================================================
// Test: Byzantine Resilience - Median
// =============================================================================

/// Test Median defense handles outliers correctly.
#[test]
fn test_byzantine_resilience_median() {
    const DIM: usize = 50;
    const HONEST: usize = 5;

    let (honest, center) = generate_clustered_dense_gradients(HONEST, DIM);
    let byzantine = generate_byzantine_gradient_with_magnitude(DIM, 1000.0);

    let config = DefenseConfig::with_defense(Defense::Median);
    let aggregator = ByzantineAggregator::new(config);

    let mut all_gradients = honest.clone();
    all_gradients.push(byzantine.clone());

    let result = aggregator.aggregate(&all_gradients).expect("Aggregation should succeed");

    // Median should be robust to the outlier
    let dist_to_center = euclidean_distance(&result, &center);
    let dist_to_byzantine = euclidean_distance(&result, &byzantine);

    // Compute what the median of honest gradients would be
    let honest_median = compute_median(&honest);
    let dist_to_honest_median = euclidean_distance(&result, &honest_median);

    assert!(
        dist_to_honest_median < dist_to_byzantine / 10.0,
        "Median result should be close to honest median, not Byzantine"
    );

    println!("Median resilience: dist_to_center={:.4}, dist_to_byz={:.4}",
             dist_to_center, dist_to_byzantine);
}

// =============================================================================
// Test: Byzantine Resilience - TrimmedMean
// =============================================================================

/// Test TrimmedMean defense correctly trims outliers.
#[test]
fn test_byzantine_resilience_trimmed_mean() {
    const DIM: usize = 50;
    const HONEST: usize = 8;

    let (honest, center) = generate_clustered_dense_gradients(HONEST, DIM);

    // Add outliers on both tails
    let byzantine_high = generate_byzantine_gradient_with_magnitude(DIM, 500.0);
    let byzantine_low = generate_byzantine_gradient_with_magnitude(DIM, -500.0);

    // Beta = 0.1 with 10 gradients trims 1 from each end
    let config = DefenseConfig::with_defense(Defense::TrimmedMean { beta: 0.1 });
    let aggregator = ByzantineAggregator::new(config);

    let mut all_gradients = honest.clone();
    all_gradients.push(byzantine_high.clone());
    all_gradients.push(byzantine_low.clone());

    let result = aggregator.aggregate(&all_gradients).expect("Aggregation should succeed");

    // Result should be close to honest center
    let dist_to_center = euclidean_distance(&result, &center);
    let honest_mean = compute_mean(&honest);
    let dist_to_honest_mean = euclidean_distance(&result, &honest_mean);

    // The result should be quite close to the honest mean
    assert!(
        dist_to_honest_mean < 2.0,
        "TrimmedMean result should be close to honest mean: dist={}",
        dist_to_honest_mean
    );

    println!("TrimmedMean resilience: dist_to_center={:.4}, dist_to_honest_mean={:.4}",
             dist_to_center, dist_to_honest_mean);
}

// =============================================================================
// Test: Byzantine Resilience - MultiKrum
// =============================================================================

/// Test MultiKrum averages multiple good gradients.
#[test]
fn test_byzantine_resilience_multi_krum() {
    const DIM: usize = 50;
    const HONEST: usize = 6;

    let (honest, center) = generate_clustered_dense_gradients(HONEST, DIM);
    let byzantine = generate_byzantine_gradient(DIM);

    // MultiKrum with f=1, k=4 (select and average top 4)
    let config = DefenseConfig::with_defense(Defense::MultiKrum { f: 1, k: 4 });
    let aggregator = ByzantineAggregator::new(config);

    let mut all_gradients = honest.clone();
    all_gradients.push(byzantine.clone());

    let result = aggregator.aggregate(&all_gradients).expect("Aggregation should succeed");

    // Result should be close to honest cluster
    let dist_to_center = euclidean_distance(&result, &center);
    let dist_to_byzantine = euclidean_distance(&result, &byzantine);

    assert!(
        dist_to_center < dist_to_byzantine / 5.0,
        "MultiKrum result should be close to honest cluster"
    );

    println!("MultiKrum resilience: dist_to_center={:.4}, dist_to_byzantine={:.4}",
             dist_to_center, dist_to_byzantine);
}

// =============================================================================
// Test: Byzantine Resilience - FedAvg (No Defense)
// =============================================================================

/// Test FedAvg is vulnerable to Byzantine attacks (for comparison).
#[test]
fn test_byzantine_vulnerability_fedavg() {
    const DIM: usize = 50;
    const HONEST: usize = 4;

    let (honest, center) = generate_clustered_dense_gradients(HONEST, DIM);
    let byzantine = generate_byzantine_gradient_with_magnitude(DIM, 100.0);

    let config = DefenseConfig::with_defense(Defense::FedAvg);
    let aggregator = ByzantineAggregator::new(config);

    let mut all_gradients = honest.clone();
    all_gradients.push(byzantine.clone());

    let result = aggregator.aggregate(&all_gradients).expect("Aggregation should succeed");

    // FedAvg should be affected by the Byzantine gradient
    let _dist_to_center = euclidean_distance(&result, &center);
    let honest_mean = compute_mean(&honest);
    let dist_to_honest_mean = euclidean_distance(&result, &honest_mean);

    // FedAvg should be pulled away from honest mean
    assert!(
        dist_to_honest_mean > 5.0,
        "FedAvg should be affected by Byzantine: dist={}",
        dist_to_honest_mean
    );

    println!("FedAvg vulnerability: dist_to_honest_mean={:.4} (expected to be high)",
             dist_to_honest_mean);
}

// =============================================================================
// Test: Round Lifecycle
// =============================================================================

/// Test complete round lifecycle: register -> submit -> aggregate -> next round.
#[test]
fn test_round_lifecycle() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(3)
        .with_defense(Defense::FedAvg);

    let mut aggregator = Aggregator::new(config);

    // Initial state
    assert_eq!(aggregator.current_round(), 0);
    assert_eq!(aggregator.submission_count(), 0);
    assert!(!aggregator.is_round_complete());

    // Submit gradients
    aggregator.submit("node_0", Array1::from_vec(vec![1.0, 2.0, 3.0]))
        .expect("Submit 1");
    assert_eq!(aggregator.submission_count(), 1);
    assert!(!aggregator.is_round_complete());

    aggregator.submit("node_1", Array1::from_vec(vec![4.0, 5.0, 6.0]))
        .expect("Submit 2");
    assert_eq!(aggregator.submission_count(), 2);
    assert!(!aggregator.is_round_complete());

    aggregator.submit("node_2", Array1::from_vec(vec![7.0, 8.0, 9.0]))
        .expect("Submit 3");
    assert_eq!(aggregator.submission_count(), 3);
    assert!(aggregator.is_round_complete());

    // Finalize round
    let result = aggregator.finalize_round().expect("Finalize");

    // Verify aggregation result (FedAvg = mean)
    assert_gradients_close(&result, &Array1::from_vec(vec![4.0, 5.0, 6.0]), 0.01);

    // Verify state transition to next round
    assert_eq!(aggregator.current_round(), 1);
    assert_eq!(aggregator.submission_count(), 0);
    assert!(!aggregator.is_round_complete());

    // Can submit to new round
    aggregator.submit("node_0", Array1::from_vec(vec![10.0, 11.0, 12.0]))
        .expect("Submit to round 1");
    assert_eq!(aggregator.submission_count(), 1);
}

/// Test that duplicate submissions are rejected.
#[test]
fn test_round_duplicate_submission_rejected() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    aggregator.submit("node_0", Array1::from_vec(vec![1.0, 2.0]))
        .expect("First submit");

    let result = aggregator.submit("node_0", Array1::from_vec(vec![3.0, 4.0]));
    assert!(result.is_err(), "Duplicate submission should fail");
}

/// Test dimension mismatch detection.
#[test]
fn test_round_dimension_mismatch() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    aggregator.submit("node_0", Array1::from_vec(vec![1.0, 2.0, 3.0]))
        .expect("First submit");

    let result = aggregator.submit("node_1", Array1::from_vec(vec![1.0, 2.0]));
    assert!(result.is_err(), "Dimension mismatch should fail");
}

/// Test async aggregator round lifecycle.
#[tokio::test]
async fn test_async_round_lifecycle() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(2)
        .with_defense(Defense::FedAvg);

    let aggregator = AsyncAggregator::new(config);

    // Register nodes
    aggregator.register_node("async_node_0").await.expect("Register 0");
    aggregator.register_node("async_node_1").await.expect("Register 1");

    // Verify registration
    assert!(aggregator.is_registered("async_node_0").await);
    assert!(aggregator.is_registered("async_node_1").await);
    assert!(!aggregator.is_registered("unknown").await);

    // Submit gradients
    aggregator.submit("async_node_0", Array1::from_vec(vec![1.0, 2.0]))
        .await.expect("Submit 0");
    assert!(!aggregator.is_round_complete().await);

    aggregator.submit("async_node_1", Array1::from_vec(vec![3.0, 4.0]))
        .await.expect("Submit 1");
    assert!(aggregator.is_round_complete().await);

    // Get aggregated result
    let result = aggregator.get_aggregated_gradient().await
        .expect("Get result")
        .expect("Result should be Some");

    assert_gradients_close(&result, &Array1::from_vec(vec![2.0, 3.0]), 0.01);

    // Check status
    let status = aggregator.status().await;
    assert_eq!(status.round, 1); // Advanced to next round
    assert!(status.registered_nodes >= 2);
}

/// Test unregistered node is rejected in async aggregator.
#[tokio::test]
async fn test_async_unregistered_node_rejected() {
    let config = AggregatorConfig::default().with_expected_nodes(2);
    let aggregator = AsyncAggregator::new(config);

    // Submit without registering
    let result = aggregator.submit("unregistered", Array1::from_vec(vec![1.0]))
        .await;

    assert!(result.is_err(), "Unregistered node should be rejected");
}

// =============================================================================
// Test: Memory and Resource Management
// =============================================================================

/// Test memory usage tracking.
#[test]
fn test_memory_usage_tracking() {
    let config = AggregatorConfig::default().with_expected_nodes(3);
    let mut aggregator = Aggregator::new(config);

    assert_eq!(aggregator.memory_usage(), 0);

    // Submit gradient with 100 f32 values = 400 bytes
    aggregator.submit("node_0", Array1::zeros(100)).expect("Submit");
    assert_eq!(aggregator.memory_usage(), 400);

    // Submit another
    aggregator.submit("node_1", Array1::zeros(100)).expect("Submit");
    assert_eq!(aggregator.memory_usage(), 800);
}

/// Test memory limit enforcement.
#[test]
fn test_memory_limit_enforcement() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(3)
        .with_max_memory(500); // 500 bytes limit

    let mut aggregator = Aggregator::new(config);

    // First gradient: 100 f32s = 400 bytes (OK)
    aggregator.submit("node_0", Array1::zeros(100)).expect("Submit 1");

    // Second gradient: would exceed 500 bytes
    let result = aggregator.submit("node_1", Array1::zeros(100));
    assert!(result.is_err(), "Should fail due to memory limit");
}

// =============================================================================
// Test: Binary <-> Hypervector Conversion
// =============================================================================

/// Test conversion between binary and regular hypervectors.
#[test]
fn test_binary_hypervector_conversion() {
    let original = BinaryHypervector::from_bits(&[true, false, true, false, true, true, false, false]);

    // Convert to Hypervector
    let hyper = original.to_hypervector();
    assert_eq!(hyper.dimension(), 8);
    assert_eq!(hyper.components[0], 127);  // true -> 127
    assert_eq!(hyper.components[1], -128); // false -> -128
    assert_eq!(hyper.components[2], 127);
    assert_eq!(hyper.components[3], -128);

    // Convert back
    let back = BinaryHypervector::from_hypervector(&hyper);
    assert_eq!(back.bit_dimension, 8);
    assert_eq!(back.get_bit(0), Some(true));
    assert_eq!(back.get_bit(1), Some(false));
    assert_eq!(back.get_bit(2), Some(true));
    assert_eq!(back.get_bit(3), Some(false));
}

// =============================================================================
// Test: Gradient Validation
// =============================================================================

/// Test NaN values are detected in gradients.
#[test]
fn test_nan_gradient_detection() {
    let config = AggregatorConfig::default().with_expected_nodes(2);
    let mut aggregator = Aggregator::new(config);

    let mut invalid = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    invalid[1] = f32::NAN;

    let result = aggregator.submit("node_0", invalid);
    assert!(result.is_err(), "NaN gradient should be rejected");
}

/// Test Infinity values are detected in gradients.
#[test]
fn test_infinity_gradient_detection() {
    let config = AggregatorConfig::default().with_expected_nodes(2);
    let mut aggregator = Aggregator::new(config);

    let invalid = Array1::from_vec(vec![1.0, f32::INFINITY, 3.0]);

    let result = aggregator.submit("node_0", invalid);
    assert!(result.is_err(), "Infinity gradient should be rejected");
}

/// Test DenseGradient validation.
#[test]
fn test_dense_gradient_validation() {
    let valid = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(valid.is_valid());
    assert!(valid.validate_finite().is_ok());

    let with_nan = DenseGradient::from_vec(vec![1.0, f32::NAN, 3.0]);
    assert!(!with_nan.is_valid());
    assert!(with_nan.validate_finite().is_err());

    let with_inf = DenseGradient::from_vec(vec![f32::INFINITY, 2.0, 3.0]);
    assert!(!with_inf.is_valid());
    assert!(with_inf.validate_finite().is_err());
}

// =============================================================================
// Test: Gradient Clipping and Normalization
// =============================================================================

/// Test gradient clipping in Byzantine aggregator.
#[test]
fn test_gradient_clipping() {
    let config = DefenseConfig::with_defense(Defense::FedAvg)
        .with_clip_norm(1.0);

    let aggregator = ByzantineAggregator::new(config);

    // Large gradient that should be clipped
    let large = Array1::from_vec(vec![10.0, 0.0, 0.0]);
    let normal = Array1::from_vec(vec![0.5, 0.5, 0.0]);

    let result = aggregator.aggregate(&[large, normal]).expect("Aggregate");

    // Result should have bounded norm
    let result_norm = l2_norm(&result);
    assert!(result_norm < 2.0, "Result norm should be bounded: {}", result_norm);
}

/// Test gradient normalization.
#[test]
fn test_gradient_normalization() {
    let config = DefenseConfig::with_defense(Defense::FedAvg)
        .with_normalization();

    let aggregator = ByzantineAggregator::new(config);

    let g1 = Array1::from_vec(vec![3.0, 4.0, 0.0]); // norm = 5
    let g2 = Array1::from_vec(vec![0.0, 6.0, 8.0]); // norm = 10

    let result = aggregator.aggregate(&[g1, g2]).expect("Aggregate");

    // After normalization, both inputs have norm 1, so result should have norm ~1
    let result_norm = l2_norm(&result);
    assert!(
        (result_norm - 1.0).abs() < 0.5,
        "Normalized result should have norm near 1: {}",
        result_norm
    );
}

// =============================================================================
// Test: Edge Cases
// =============================================================================

/// Test aggregating single gradient.
#[test]
fn test_single_gradient_aggregation() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(1)
        .with_defense(Defense::FedAvg);

    let mut aggregator = Aggregator::new(config);

    let gradient = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    aggregator.submit("only_node", gradient.clone()).expect("Submit");

    let result = aggregator.finalize_round().expect("Finalize");
    assert_gradients_close(&result, &gradient, 0.001);
}

/// Test reset functionality.
#[test]
fn test_aggregator_reset() {
    let config = AggregatorConfig::default()
        .with_expected_nodes(2)
        .with_defense(Defense::FedAvg); // Use FedAvg to avoid Krum minimum node requirements

    let mut aggregator = Aggregator::new(config);

    aggregator.submit("node_0", Array1::from_vec(vec![1.0, 2.0])).expect("Submit");
    aggregator.submit("node_1", Array1::from_vec(vec![3.0, 4.0])).expect("Submit");
    aggregator.finalize_round().expect("Finalize");

    assert_eq!(aggregator.current_round(), 1);

    aggregator.reset();

    assert_eq!(aggregator.current_round(), 0);
    assert_eq!(aggregator.submission_count(), 0);
}

/// Test empty round error.
#[test]
fn test_empty_round_error() {
    let config = AggregatorConfig::default().with_expected_nodes(2);
    let mut aggregator = Aggregator::new(config);

    let result = aggregator.finalize_round();
    assert!(result.is_err(), "Finalizing empty round should fail");
}

// =============================================================================
// Test: Sparse and Quantized Gradient Integration
// =============================================================================

/// Test sparse gradient conversion and aggregation readiness.
#[test]
fn test_sparse_gradient_integration() {
    // Use a larger gradient with more zeros for better compression
    let values = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0];
    let dense = DenseGradient::from_vec(values.clone());
    let sparse = SparseGradient::from_dense(&dense, 0.5);

    // Verify sparse representation (only non-zero values kept)
    assert_eq!(sparse.indices, vec![0, 2, 4, 8]);
    assert_eq!(sparse.values, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(sparse.full_dimension, 10);

    // Convert back and verify
    let recovered = sparse.to_dense();
    assert_eq!(recovered.to_vec(), values);

    // Check sparsity ratio (4 non-zero out of 10)
    let sparsity = sparse.sparsity_ratio();
    assert!((sparsity - 0.4).abs() < 0.01, "Sparsity should be 0.4: {}", sparsity);

    // Verify is_valid
    assert!(sparse.is_valid());
}

/// Test quantized gradient at different bit depths.
#[test]
fn test_quantized_gradient_bit_depths() {
    let source = DenseGradient::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);

    for bits in [1u8, 2, 4, 8] {
        let quantized = QuantizedGradient::from_dense(&source, bits);
        assert!(quantized.is_valid());
        assert_eq!(quantized.bits_per_value, bits);

        let recovered = quantized.to_dense();
        assert_eq!(recovered.dimension(), 5);

        // Lower bits = more quantization error
        let max_error = 1.0 / (1u32 << bits) as f32 + 0.1;
        for (a, b) in source.to_vec().iter().zip(recovered.to_vec().iter()) {
            assert!(
                (a - b).abs() <= max_error,
                "Quantization error too large at {} bits: {} vs {}",
                bits, a, b
            );
        }
    }
}
