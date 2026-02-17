//! Test fixtures for unified FL aggregator integration tests.
//!
//! Provides helper functions for generating test data including:
//! - Clustered dense gradients (honest nodes)
//! - Byzantine (malicious) gradients
//! - Clustered hypervectors
//! - Mathematical utilities for verification

#![allow(dead_code)] // Some utilities are provided for potential future use

use fl_aggregator::{
    Gradient,
    payload::{Hypervector, BinaryHypervector, DenseGradient},
};
use ndarray::Array1;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

/// Generate `n` clustered dense gradients around a center point.
///
/// Creates gradients that cluster tightly around a center value,
/// simulating honest nodes with similar model updates.
///
/// # Arguments
/// * `n` - Number of gradients to generate
/// * `dim` - Dimension of each gradient
///
/// # Returns
/// A tuple of (gradients, center) where center is the cluster centroid.
pub fn generate_clustered_dense_gradients(n: usize, dim: usize) -> (Vec<Gradient>, Gradient) {
    let mut rng = StdRng::seed_from_u64(42); // Deterministic for reproducibility

    // Center point around which honest gradients cluster
    let center_dist = Uniform::new(-1.0f32, 1.0);
    let center: Gradient = Array1::from_vec(
        (0..dim).map(|_| center_dist.sample(&mut rng)).collect()
    );

    // Generate clustered gradients with small perturbations
    let noise_dist = Uniform::new(-0.1f32, 0.1);
    let gradients: Vec<Gradient> = (0..n)
        .map(|_| {
            let noise: Array1<f32> = Array1::from_vec(
                (0..dim).map(|_| noise_dist.sample(&mut rng)).collect()
            );
            &center + &noise
        })
        .collect();

    (gradients, center)
}

/// Generate `n` clustered dense gradients with controlled variance.
///
/// # Arguments
/// * `n` - Number of gradients to generate
/// * `dim` - Dimension of each gradient
/// * `variance` - Maximum noise magnitude for each component
///
/// # Returns
/// A tuple of (gradients, center)
pub fn generate_clustered_dense_gradients_with_variance(
    n: usize,
    dim: usize,
    variance: f32,
) -> (Vec<Gradient>, Gradient) {
    let mut rng = StdRng::seed_from_u64(42);

    let center_dist = Uniform::new(-1.0f32, 1.0);
    let center: Gradient = Array1::from_vec(
        (0..dim).map(|_| center_dist.sample(&mut rng)).collect()
    );

    let noise_dist = Uniform::new(-variance, variance);
    let gradients: Vec<Gradient> = (0..n)
        .map(|_| {
            let noise: Array1<f32> = Array1::from_vec(
                (0..dim).map(|_| noise_dist.sample(&mut rng)).collect()
            );
            &center + &noise
        })
        .collect();

    (gradients, center)
}

/// Generate a Byzantine (malicious) gradient far from the cluster.
///
/// Creates an outlier gradient that Byzantine defense algorithms should detect.
///
/// # Arguments
/// * `dim` - Dimension of the gradient
///
/// # Returns
/// A gradient with values far from typical honest gradients.
pub fn generate_byzantine_gradient(dim: usize) -> Gradient {
    // Byzantine gradient with large values (10-100x normal magnitude)
    let mut rng = StdRng::seed_from_u64(999);
    let dist = Uniform::new(50.0f32, 100.0);

    Array1::from_vec(
        (0..dim).map(|_| dist.sample(&mut rng)).collect()
    )
}

/// Generate a Byzantine gradient with specified magnitude.
///
/// # Arguments
/// * `dim` - Dimension of the gradient
/// * `magnitude` - Approximate magnitude of each component (can be negative)
pub fn generate_byzantine_gradient_with_magnitude(dim: usize, magnitude: f32) -> Gradient {
    let mut rng = StdRng::seed_from_u64(999);

    // Handle negative magnitudes by using absolute value for the range
    let abs_mag = magnitude.abs();
    let (low, high) = if magnitude >= 0.0 {
        (abs_mag * 0.8, abs_mag * 1.2)
    } else {
        (-abs_mag * 1.2, -abs_mag * 0.8)
    };

    let dist = Uniform::new(low, high);

    Array1::from_vec(
        (0..dim).map(|_| dist.sample(&mut rng)).collect()
    )
}

/// Generate `n` clustered hypervectors with known values.
///
/// # Arguments
/// * `n` - Number of hypervectors to generate
/// * `dim` - Dimension of each hypervector
///
/// # Returns
/// A tuple of (hypervectors, expected_bundle) where expected_bundle is the
/// component-wise average.
pub fn generate_clustered_hypervectors(n: usize, dim: usize) -> (Vec<Hypervector>, Vec<i8>) {
    let mut rng = StdRng::seed_from_u64(123);

    // Generate base values that we know the exact bundled result for
    let base: Vec<i8> = (0..dim)
        .map(|i| ((i % 256) as i16 - 128) as i8)
        .collect();

    // Create n hypervectors with small perturbations
    let noise_dist = Uniform::new(-5i8, 5);
    let hypervectors: Vec<Hypervector> = (0..n)
        .map(|_| {
            let components: Vec<i8> = base
                .iter()
                .map(|&b| {
                    let noise = noise_dist.sample(&mut rng);
                    (b as i16 + noise as i16).clamp(-128, 127) as i8
                })
                .collect();
            Hypervector::new(components)
        })
        .collect();

    // The expected bundle is approximately the base (since perturbations average out)
    (hypervectors, base)
}

/// Generate hypervectors with known exact values for precise testing.
///
/// # Arguments
/// * `values` - Slice of component vectors, each becoming a hypervector
///
/// # Returns
/// Vector of Hypervectors with the exact specified values.
pub fn generate_exact_hypervectors(values: &[Vec<i8>]) -> Vec<Hypervector> {
    values.iter().map(|v| Hypervector::new(v.clone())).collect()
}

/// Generate binary hypervectors with known bit patterns.
///
/// # Arguments
/// * `patterns` - Slice of bit pattern vectors
///
/// # Returns
/// Vector of BinaryHypervectors.
pub fn generate_binary_hypervectors(patterns: &[Vec<bool>]) -> Vec<BinaryHypervector> {
    patterns.iter().map(|p| BinaryHypervector::from_bits(p)).collect()
}

/// Generate binary hypervectors for majority voting tests.
///
/// Creates patterns where the majority can be easily computed.
///
/// # Arguments
/// * `n` - Number of hypervectors
/// * `dim` - Bit dimension
///
/// # Returns
/// A tuple of (hypervectors, expected_result) where expected_result is the
/// majority-voted result.
pub fn generate_binary_hypervectors_with_known_majority(
    n: usize,
    dim: usize,
) -> (Vec<BinaryHypervector>, Vec<bool>) {
    // For deterministic majority, create n patterns where each bit position
    // has a known majority
    let mut patterns: Vec<Vec<bool>> = vec![vec![false; dim]; n];
    let mut expected: Vec<bool> = vec![false; dim];

    // For each bit position, set more than half of the vectors to match expected
    let majority_count = (n / 2) + 1;

    for bit_pos in 0..dim {
        // Alternate expected values for variety
        let target = bit_pos % 2 == 0;
        expected[bit_pos] = target;

        // Set majority of vectors to this value
        for vec_idx in 0..majority_count {
            patterns[vec_idx][bit_pos] = target;
        }
        // Remaining vectors get opposite value
        for vec_idx in majority_count..n {
            patterns[vec_idx][bit_pos] = !target;
        }
    }

    let hvs: Vec<BinaryHypervector> = patterns
        .iter()
        .map(|p| BinaryHypervector::from_bits(p))
        .collect();

    (hvs, expected)
}

/// Compute the mean (average) of a collection of gradients.
///
/// # Arguments
/// * `gradients` - Slice of gradients to average
///
/// # Returns
/// The component-wise mean gradient.
pub fn compute_mean(gradients: &[Gradient]) -> Gradient {
    if gradients.is_empty() {
        return Array1::zeros(0);
    }

    let dim = gradients[0].len();
    let n = gradients.len() as f32;

    let mut sum = Array1::zeros(dim);
    for g in gradients {
        sum = sum + g;
    }

    sum / n
}

/// Compute the component-wise median of gradients.
///
/// # Arguments
/// * `gradients` - Slice of gradients
///
/// # Returns
/// The component-wise median gradient.
pub fn compute_median(gradients: &[Gradient]) -> Gradient {
    if gradients.is_empty() {
        return Array1::zeros(0);
    }

    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);

    for d in 0..dim {
        let mut values: Vec<f32> = gradients.iter().map(|g| g[d]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = values.len() / 2;
        result[d] = if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        };
    }

    result
}

/// Calculate Euclidean (L2) distance between two gradients.
///
/// # Arguments
/// * `a` - First gradient
/// * `b` - Second gradient
///
/// # Returns
/// The Euclidean distance.
pub fn euclidean_distance(a: &Gradient, b: &Gradient) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate L2 norm of a gradient.
pub fn l2_norm(g: &Gradient) -> f32 {
    g.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
}

/// Calculate cosine similarity between two gradients.
pub fn cosine_similarity(a: &Gradient, b: &Gradient) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Create a DenseGradient from values.
pub fn dense_gradient_from(values: Vec<f32>) -> DenseGradient {
    DenseGradient::from_vec(values)
}

/// Assert two gradients are approximately equal within tolerance.
pub fn assert_gradients_close(a: &Gradient, b: &Gradient, tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Gradient dimensions must match");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tolerance,
            "Gradient mismatch at index {}: {} vs {} (diff: {}, tolerance: {})",
            i, x, y, (x - y).abs(), tolerance
        );
    }
}

/// Assert a gradient is within a certain distance of a target.
pub fn assert_gradient_near(actual: &Gradient, target: &Gradient, max_distance: f32) {
    let distance = euclidean_distance(actual, target);
    assert!(
        distance <= max_distance,
        "Gradient distance {} exceeds max allowed {}",
        distance, max_distance
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_clustered_gradients() {
        let (gradients, center) = generate_clustered_dense_gradients(5, 10);

        assert_eq!(gradients.len(), 5);
        assert_eq!(center.len(), 10);

        // All gradients should be close to center
        for g in &gradients {
            let dist = euclidean_distance(g, &center);
            assert!(dist < 1.0, "Gradient too far from center: {}", dist);
        }
    }

    #[test]
    fn test_generate_byzantine_gradient() {
        let (honest, center) = generate_clustered_dense_gradients(5, 10);
        let byzantine = generate_byzantine_gradient(10);

        // Byzantine should be far from center
        let byz_dist = euclidean_distance(&byzantine, &center);
        let max_honest_dist = honest.iter()
            .map(|g| euclidean_distance(g, &center))
            .fold(0.0f32, |a, b| a.max(b));

        assert!(
            byz_dist > max_honest_dist * 10.0,
            "Byzantine gradient should be much farther: {} vs {}",
            byz_dist, max_honest_dist
        );
    }

    #[test]
    fn test_compute_mean() {
        let g1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let g2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let g3 = Array1::from_vec(vec![7.0, 8.0, 9.0]);

        let mean = compute_mean(&[g1, g2, g3]);

        assert_eq!(mean.len(), 3);
        assert!((mean[0] - 4.0).abs() < 1e-6);
        assert!((mean[1] - 5.0).abs() < 1e-6);
        assert!((mean[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0, 0.0]);

        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_hypervector_majority() {
        let (hvs, expected) = generate_binary_hypervectors_with_known_majority(5, 8);

        // Bundle them
        let refs: Vec<_> = hvs.iter().collect();
        let bundled = BinaryHypervector::bundle(&refs).unwrap();

        // Check each bit matches expected
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                bundled.get_bit(i), Some(exp),
                "Bit {} mismatch: expected {}", i, exp
            );
        }
    }
}
