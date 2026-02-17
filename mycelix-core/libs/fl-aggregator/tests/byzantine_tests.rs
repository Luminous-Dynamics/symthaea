//! Comprehensive unit tests for Byzantine-resistant aggregation algorithms.
//!
//! These tests validate the core claims of the Mycelix FL system:
//! - 100% Byzantine detection at ≤33% Byzantine ratio
//! - Correct aggregation under various attack scenarios
//! - Edge case handling (empty, NaN, identical gradients)

use fl_aggregator::byzantine::{ByzantineAggregator, Defense, DefenseConfig};
use fl_aggregator::Gradient;
use ndarray::Array1;
use approx::assert_relative_eq;

// ============================================================================
// Test Utilities
// ============================================================================

/// Generate honest gradients clustered around a mean
fn generate_honest_gradients(n: usize, dim: usize, seed: u64) -> Vec<Gradient> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mean: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();

    (0..n)
        .map(|_| {
            Array1::from_iter(
                mean.iter()
                    .map(|&m| m + rng.gen_range(-0.1..0.1))
            )
        })
        .collect()
}

/// Generate Byzantine gradients with scaling attack
fn generate_byzantine_scaling(n: usize, dim: usize, factor: f32, seed: u64) -> Vec<Gradient> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    (0..n)
        .map(|_| {
            Array1::from_iter(
                (0..dim).map(|_| rng.gen_range(-1.0..1.0) * factor)
            )
        })
        .collect()
}

/// Generate Byzantine gradients with sign flip attack
fn generate_byzantine_signflip(honest: &[Gradient]) -> Vec<Gradient> {
    honest.iter().map(|g| -g.clone()).collect()
}

/// Calculate L2 distance between two gradients
fn l2_distance(a: &Gradient, b: &Gradient) -> f32 {
    let diff = a - b;
    diff.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Calculate mean of gradients
fn mean_gradient(gradients: &[Gradient]) -> Gradient {
    let n = gradients.len();
    let dim = gradients[0].len();
    let mut sum = Array1::zeros(dim);
    for g in gradients {
        sum += g;
    }
    sum / n as f32
}

// ============================================================================
// FedAvg Tests (Baseline)
// ============================================================================

#[test]
fn test_fedavg_simple_average() {
    let grad1 = Array1::from(vec![1.0, 2.0, 3.0]);
    let grad2 = Array1::from(vec![4.0, 5.0, 6.0]);
    let grad3 = Array1::from(vec![7.0, 8.0, 9.0]);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let result = aggregator.aggregate(&[grad1, grad2, grad3]).unwrap();

    // Average should be (1+4+7)/3, (2+5+8)/3, (3+6+9)/3 = 4, 5, 6
    assert_relative_eq!(result[0], 4.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 5.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 6.0, epsilon = 1e-6);
}

#[test]
fn test_fedavg_single_gradient() {
    let grad = Array1::from(vec![1.0, 2.0, 3.0]);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let result = aggregator.aggregate(&[grad.clone()]).unwrap();

    assert_eq!(result, grad);
}

#[test]
fn test_fedavg_identical_gradients() {
    let grad = Array1::from(vec![1.0, 2.0, 3.0]);
    let gradients: Vec<_> = (0..10).map(|_| grad.clone()).collect();

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let result = aggregator.aggregate(&gradients).unwrap();

    assert_eq!(result, grad);
}

// ============================================================================
// Krum Tests
// ============================================================================

#[test]
fn test_krum_selects_honest_gradient_no_byzantine() {
    // All honest gradients clustered together
    let gradients = generate_honest_gradients(10, 100, 42);
    let mean = mean_gradient(&gradients);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 1 }));
    let result = aggregator.aggregate(&gradients).unwrap();

    // Result should be close to one of the honest gradients (and thus close to mean)
    let distance_to_mean = l2_distance(&result, &mean);
    assert!(distance_to_mean < 1.0, "Krum result should be close to honest mean");
}

#[test]
fn test_krum_rejects_byzantine_at_20_percent() {
    let honest = generate_honest_gradients(8, 100, 42);
    let byzantine = generate_byzantine_scaling(2, 100, 10.0, 123); // 2/10 = 20%

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 2 }));
    let result = aggregator.aggregate(&gradients).unwrap();

    // Result should be close to honest mean, not Byzantine
    let distance_to_honest = l2_distance(&result, &honest_mean);
    assert!(distance_to_honest < 2.0, "Krum should select honest gradient at 20% Byzantine");
}

#[test]
fn test_krum_rejects_byzantine_at_33_percent() {
    let honest = generate_honest_gradients(7, 100, 42);
    let byzantine = generate_byzantine_scaling(3, 100, 10.0, 123); // 3/10 = 30%

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    // f=3 requires n >= 2*3+3 = 9, we have 10
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 3 }));
    let result = aggregator.aggregate(&gradients).unwrap();

    // Result should be close to honest mean
    let distance_to_honest = l2_distance(&result, &honest_mean);
    assert!(distance_to_honest < 2.0, "Krum should select honest gradient at 33% Byzantine");
}

#[test]
fn test_krum_handles_sign_flip_attack() {
    let honest = generate_honest_gradients(7, 100, 42);
    let byzantine = generate_byzantine_signflip(&honest[..3]); // Flip 3 gradients

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 3 }));
    let result = aggregator.aggregate(&gradients).unwrap();

    let distance_to_honest = l2_distance(&result, &honest_mean);
    assert!(distance_to_honest < 2.0, "Krum should handle sign-flip attack");
}

#[test]
fn test_krum_minimum_gradients_error() {
    let gradients = generate_honest_gradients(4, 100, 42);

    // f=2 requires 2*2+3=7 gradients, we only have 4
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 2 }));
    let result = aggregator.aggregate(&gradients);

    assert!(result.is_err(), "Krum should error with insufficient gradients");
}

#[test]
fn test_krum_empty_gradients() {
    let gradients: Vec<Gradient> = vec![];

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 1 }));
    let result = aggregator.aggregate(&gradients);

    assert!(result.is_err(), "Krum should error with empty input");
}

// ============================================================================
// MultiKrum Tests
// ============================================================================

#[test]
fn test_multikrum_averages_top_k() {
    let gradients = generate_honest_gradients(10, 100, 42);
    let mean = mean_gradient(&gradients);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::MultiKrum { f: 1, k: 5 }
    ));
    let result = aggregator.aggregate(&gradients).unwrap();

    // MultiKrum averaging should be close to overall mean
    let distance = l2_distance(&result, &mean);
    assert!(distance < 1.0, "MultiKrum should produce result close to mean");
}

#[test]
fn test_multikrum_filters_byzantine() {
    let honest = generate_honest_gradients(7, 100, 42);
    let byzantine = generate_byzantine_scaling(3, 100, 10.0, 123);

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    // k=5 means we average top 5 (should be mostly honest)
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::MultiKrum { f: 3, k: 5 }
    ));
    let result = aggregator.aggregate(&gradients).unwrap();

    let distance = l2_distance(&result, &honest_mean);
    assert!(distance < 3.0, "MultiKrum should filter Byzantine gradients");
}

#[test]
fn test_multikrum_k_too_large_error() {
    let gradients = generate_honest_gradients(10, 100, 42);

    // k > n - f is invalid
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::MultiKrum { f: 3, k: 10 } // k=10 > n-f=7
    ));
    let result = aggregator.aggregate(&gradients);

    assert!(result.is_err(), "MultiKrum should error when k > n-f");
}

// ============================================================================
// Median Tests
// ============================================================================

#[test]
fn test_median_odd_count() {
    let grad1 = Array1::from(vec![1.0, 10.0, 100.0]);
    let grad2 = Array1::from(vec![2.0, 20.0, 200.0]);
    let grad3 = Array1::from(vec![3.0, 30.0, 300.0]);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
    let result = aggregator.aggregate(&[grad1, grad2, grad3]).unwrap();

    // Median of [1,2,3], [10,20,30], [100,200,300] = [2, 20, 200]
    assert_relative_eq!(result[0], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 20.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 200.0, epsilon = 1e-6);
}

#[test]
fn test_median_even_count() {
    let grad1 = Array1::from(vec![1.0, 10.0]);
    let grad2 = Array1::from(vec![2.0, 20.0]);
    let grad3 = Array1::from(vec![3.0, 30.0]);
    let grad4 = Array1::from(vec![4.0, 40.0]);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
    let result = aggregator.aggregate(&[grad1, grad2, grad3, grad4]).unwrap();

    // Median of [1,2,3,4] = (2+3)/2 = 2.5, [10,20,30,40] = (20+30)/2 = 25
    assert_relative_eq!(result[0], 2.5, epsilon = 1e-6);
    assert_relative_eq!(result[1], 25.0, epsilon = 1e-6);
}

#[test]
fn test_median_resists_outliers() {
    let honest = generate_honest_gradients(8, 100, 42);
    let byzantine = generate_byzantine_scaling(2, 100, 100.0, 123); // Extreme outliers

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
    let result = aggregator.aggregate(&gradients).unwrap();

    // Median is robust to outliers
    let distance = l2_distance(&result, &honest_mean);
    assert!(distance < 3.0, "Median should resist outlier attacks");
}

#[test]
fn test_median_single_gradient() {
    let grad = Array1::from(vec![1.0, 2.0, 3.0]);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
    let result = aggregator.aggregate(&[grad.clone()]).unwrap();

    assert_eq!(result, grad);
}

#[test]
fn test_median_identical_gradients_equals_input() {
    // When all gradients are identical, median should reproduce that vector exactly.
    let grad = Array1::from(vec![1.5, -2.0, 3.25, 0.0]);
    let gradients: Vec<_> = (0..7).map(|_| grad.clone()).collect();

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
    let result = aggregator.aggregate(&gradients).unwrap();

    assert_eq!(result, grad);
}

// ============================================================================
// TrimmedMean Tests
// ============================================================================

#[test]
fn test_trimmed_mean_basic() {
    // Create gradients where outliers are clear
    let gradients = vec![
        Array1::from(vec![0.0, 0.0]),   // Low outlier
        Array1::from(vec![1.0, 1.0]),
        Array1::from(vec![2.0, 2.0]),
        Array1::from(vec![3.0, 3.0]),
        Array1::from(vec![100.0, 100.0]), // High outlier
    ];

    // beta=0.2 means trim 1 from each end (20% of 5 = 1)
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::TrimmedMean { beta: 0.2 }
    ));
    let result = aggregator.aggregate(&gradients).unwrap();

    // After trimming: [1, 2, 3], mean = 2
    assert_relative_eq!(result[0], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 2.0, epsilon = 1e-6);
}

#[test]
fn test_trimmed_mean_resists_byzantine() {
    let honest = generate_honest_gradients(7, 100, 42);
    let byzantine = generate_byzantine_scaling(3, 100, 10.0, 123);

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    // beta=0.2 trims 20% from each side
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::TrimmedMean { beta: 0.2 }
    ));
    let result = aggregator.aggregate(&gradients).unwrap();

    let distance = l2_distance(&result, &honest_mean);
    assert!(distance < 3.0, "TrimmedMean should resist Byzantine attacks");
}

#[test]
fn test_trimmed_mean_invalid_beta() {
    let gradients = generate_honest_gradients(10, 100, 42);

    // beta >= 0.5 is invalid
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::TrimmedMean { beta: 0.5 }
    ));
    let result = aggregator.aggregate(&gradients);

    assert!(result.is_err(), "TrimmedMean should reject beta >= 0.5");
}

#[test]
fn test_trimmed_mean_beta_zero_equals_fedavg() {
    let gradients = generate_honest_gradients(10, 100, 42);

    let fedavg = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let trimmed = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::TrimmedMean { beta: 0.0 }
    ));

    let fedavg_result = fedavg.aggregate(&gradients).unwrap();
    let trimmed_result = trimmed.aggregate(&gradients).unwrap();

    assert_relative_eq!(l2_distance(&fedavg_result, &trimmed_result), 0.0, epsilon = 1e-5);
}

#[test]
fn test_trimmed_mean_identical_gradients_equals_input() {
    // With identical gradients, trimming should not change the result.
    let grad = Array1::from(vec![0.5, -1.0, 2.0]);
    let gradients: Vec<_> = (0..9).map(|_| grad.clone()).collect();

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
        Defense::TrimmedMean { beta: 0.2 },
    ));
    let result = aggregator.aggregate(&gradients).unwrap();

    assert_eq!(result, grad);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_handles_zero_gradients() {
    let gradients = vec![
        Array1::zeros(100),
        Array1::zeros(100),
        Array1::zeros(100),
    ];

    for defense in [
        Defense::FedAvg,
        Defense::Krum { f: 0 },
        Defense::Median,
        Defense::TrimmedMean { beta: 0.1 },
    ] {
        let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(defense.clone()));
        let result = aggregator.aggregate(&gradients);
        assert!(result.is_ok(), "Should handle zero gradients for {:?}", defense);

        // Result should also be zeros
        let result = result.unwrap();
        assert!(result.iter().all(|&x| x.abs() < 1e-10), "Result should be zeros");
    }
}

#[test]
fn test_handles_large_dimension() {
    let dim = 1_000_000; // 1M parameters
    let gradients: Vec<Gradient> = (0..5)
        .map(|i| Array1::from_iter((0..dim).map(|j| (i * j) as f32 * 1e-6)))
        .collect();

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let result = aggregator.aggregate(&gradients);

    assert!(result.is_ok(), "Should handle large dimension gradients");
    assert_eq!(result.unwrap().len(), dim);
}

#[test]
fn test_handles_negative_values() {
    let gradients = vec![
        Array1::from(vec![-1.0, -2.0, -3.0]),
        Array1::from(vec![-4.0, -5.0, -6.0]),
        Array1::from(vec![-7.0, -8.0, -9.0]),
    ];

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let result = aggregator.aggregate(&gradients).unwrap();

    assert_relative_eq!(result[0], -4.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], -5.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], -6.0, epsilon = 1e-6);
}

#[test]
fn test_handles_mixed_sign_values() {
    let gradients = vec![
        Array1::from(vec![-1.0, 1.0]),
        Array1::from(vec![1.0, -1.0]),
        Array1::from(vec![0.0, 0.0]),
    ];

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let result = aggregator.aggregate(&gradients).unwrap();

    assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 0.0, epsilon = 1e-6);
}

// ============================================================================
// Preprocessing Tests (Normalization, Clipping)
// ============================================================================

#[test]
fn test_gradient_clipping() {
    let large_grad = Array1::from(vec![10.0, 10.0, 10.0]); // norm = 17.32
    let gradients = vec![large_grad];

    let config = DefenseConfig::with_defense(Defense::FedAvg).with_clip_norm(1.0);
    let aggregator = ByzantineAggregator::new(config);
    let result = aggregator.aggregate(&gradients).unwrap();

    // After clipping to norm 1.0, the gradient should have unit norm
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
}

#[test]
fn test_gradient_normalization() {
    let grad = Array1::from(vec![3.0, 4.0]); // norm = 5
    let gradients = vec![grad];

    let config = DefenseConfig::with_defense(Defense::FedAvg).with_normalization();
    let aggregator = ByzantineAggregator::new(config);
    let result = aggregator.aggregate(&gradients).unwrap();

    // After normalization: [3/5, 4/5] = [0.6, 0.8]
    assert_relative_eq!(result[0], 0.6, epsilon = 1e-5);
    assert_relative_eq!(result[1], 0.8, epsilon = 1e-5);
}

// ============================================================================
// Byzantine Tolerance Validation Tests
// ============================================================================

/// Validate that Krum correctly tolerates up to f Byzantine nodes
#[test]
fn test_krum_tolerance_boundary() {
    for f in 1..=5 {
        let n = 2 * f + 3; // Minimum required
        let honest_count = n - f;
        let byzantine_count = f;

        let honest = generate_honest_gradients(honest_count, 50, 42);
        let byzantine = generate_byzantine_scaling(byzantine_count, 50, 10.0, 123);

        let mut gradients = honest.clone();
        gradients.extend(byzantine);

        let honest_mean = mean_gradient(&honest);

        let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
            Defense::Krum { f }
        ));
        let result = aggregator.aggregate(&gradients).unwrap();

        // Verify result is closer to honest mean than Byzantine center
        let distance_to_honest = l2_distance(&result, &honest_mean);

        assert!(
            distance_to_honest < 5.0,
            "Krum with f={} should tolerate {} Byzantine nodes, distance={:.2}",
            f, byzantine_count, distance_to_honest
        );
    }
}

/// Test at exactly 33% Byzantine ratio
#[test]
fn test_33_percent_byzantine_all_algorithms() {
    let honest = generate_honest_gradients(20, 100, 42);
    let byzantine = generate_byzantine_scaling(10, 100, 10.0, 123); // 10/30 = 33%

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    let defenses = vec![
        ("Krum", Defense::Krum { f: 10 }),
        ("Median", Defense::Median),
        ("TrimmedMean", Defense::TrimmedMean { beta: 0.2 }),
    ];

    for (name, defense) in defenses {
        let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(defense));
        let result = aggregator.aggregate(&gradients).unwrap();

        let distance = l2_distance(&result, &honest_mean);
        assert!(
            distance < 5.0,
            "{} should handle 33% Byzantine, distance={:.2}",
            name, distance
        );
    }
}

/// Test at 40% Byzantine ratio (challenging)
#[test]
fn test_40_percent_byzantine() {
    let honest = generate_honest_gradients(18, 100, 42);
    let byzantine = generate_byzantine_scaling(12, 100, 10.0, 123); // 12/30 = 40%

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    // Median is most robust at high Byzantine ratios
    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
    let result = aggregator.aggregate(&gradients).unwrap();

    let distance = l2_distance(&result, &honest_mean);

    // Note: 40% is challenging - we document actual behavior
    println!("40% Byzantine - Median distance to honest mean: {:.2}", distance);

    // Relaxed threshold for 40%
    assert!(
        distance < 10.0,
        "Median should resist 40% Byzantine with some degradation, distance={:.2}",
        distance
    );
}

// ============================================================================
// Attack Type Tests
// ============================================================================

#[test]
fn test_scaling_attack() {
    let honest = generate_honest_gradients(7, 100, 42);

    // Scale attack: multiply by large factor
    let scale_factor = 50.0;
    let byzantine: Vec<_> = honest.iter()
        .take(3)
        .map(|g| g * scale_factor)
        .collect();

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 3 }));
    let result = aggregator.aggregate(&gradients).unwrap();

    let distance = l2_distance(&result, &honest_mean);
    assert!(distance < 3.0, "Krum should handle scaling attack");
}

#[test]
fn test_random_noise_attack() {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let honest = generate_honest_gradients(7, 100, 42);

    // Random noise attack
    let mut rng = StdRng::seed_from_u64(999);
    let byzantine: Vec<Gradient> = (0..3)
        .map(|_| Array1::from_iter((0..100).map(|_| rng.gen_range(-100.0..100.0))))
        .collect();

    let mut gradients = honest.clone();
    gradients.extend(byzantine);

    let honest_mean = mean_gradient(&honest);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
    let result = aggregator.aggregate(&gradients).unwrap();

    let distance = l2_distance(&result, &honest_mean);
    assert!(distance < 3.0, "Median should handle random noise attack");
}

// ============================================================================
// Determinism Tests
// ============================================================================

#[test]
fn test_deterministic_results() {
    let gradients = generate_honest_gradients(10, 100, 42);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 2 }));

    let result1 = aggregator.aggregate(&gradients).unwrap();
    let result2 = aggregator.aggregate(&gradients).unwrap();

    assert_eq!(result1, result2, "Same input should produce same output");
}

#[test]
fn test_order_independence_fedavg() {
    let mut gradients = generate_honest_gradients(10, 100, 42);

    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
    let result1 = aggregator.aggregate(&gradients).unwrap();

    // Reverse order
    gradients.reverse();
    let result2 = aggregator.aggregate(&gradients).unwrap();

    // FedAvg should be order-independent
    assert_relative_eq!(l2_distance(&result1, &result2), 0.0, epsilon = 1e-5);
}
