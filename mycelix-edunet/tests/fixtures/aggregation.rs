// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Aggregation test fixtures
//
// Generators for gradient vectors and FL aggregation test data

use super::{seeded_rng, DEFAULT_SEED};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

/// Generate random gradient vectors with Gaussian distribution
///
/// # Arguments
/// * `participant_count` - Number of participants (gradient vectors)
/// * `dims` - Dimensionality of each gradient vector
/// * `mean` - Mean of the Gaussian distribution
/// * `stddev` - Standard deviation of the Gaussian distribution
/// * `seed` - Random seed for reproducibility
pub fn gaussian(
    participant_count: usize,
    dims: usize,
    mean: f32,
    stddev: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(mean, stddev).unwrap();

    (0..participant_count)
        .map(|_| (0..dims).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

/// Generate random gradient vectors with Gaussian distribution (default parameters)
///
/// Uses mean=0.0, stddev=1.0
pub fn gaussian_default(participant_count: usize, dims: usize) -> Vec<Vec<f32>> {
    gaussian(participant_count, dims, 0.0, 1.0, DEFAULT_SEED)
}

/// Generate random gradient vectors with uniform distribution
///
/// # Arguments
/// * `participant_count` - Number of participants
/// * `dims` - Dimensionality of each gradient vector
/// * `min` - Minimum value
/// * `max` - Maximum value
/// * `seed` - Random seed for reproducibility
pub fn uniform(
    participant_count: usize,
    dims: usize,
    min: f32,
    max: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = seeded_rng(seed);
    let uniform = Uniform::new(min, max);

    (0..participant_count)
        .map(|_| (0..dims).map(|_| uniform.sample(&mut rng)).collect())
        .collect()
}

/// Generate random gradient vectors with uniform distribution (default parameters)
///
/// Uses range [-1.0, 1.0]
pub fn uniform_default(participant_count: usize, dims: usize) -> Vec<Vec<f32>> {
    uniform(participant_count, dims, -1.0, 1.0, DEFAULT_SEED)
}

/// Generate gradient vectors with injected outliers for robustness testing
///
/// Generates mostly normal gradients with a few extreme outliers
pub fn with_outliers(
    participant_count: usize,
    dims: usize,
    outlier_count: usize,
    outlier_magnitude: f32,
) -> Vec<Vec<f32>> {
    let mut gradients = gaussian_default(participant_count - outlier_count, dims);

    // Add outliers
    let mut rng = seeded_rng(DEFAULT_SEED + 1000);
    for _ in 0..outlier_count {
        let outlier: Vec<f32> = (0..dims)
            .map(|_| {
                if rng.gen_bool(0.5) {
                    outlier_magnitude
                } else {
                    -outlier_magnitude
                }
            })
            .collect();
        gradients.push(outlier);
    }

    gradients
}

/// Generate sparse gradient vectors (many zeros)
///
/// # Arguments
/// * `participant_count` - Number of participants
/// * `dims` - Dimensionality
/// * `sparsity` - Fraction of values that are zero (0.0 to 1.0)
pub fn sparse(participant_count: usize, dims: usize, sparsity: f32) -> Vec<Vec<f32>> {
    let mut rng = seeded_rng(DEFAULT_SEED);
    let normal = Normal::new(0.0, 1.0).unwrap();

    (0..participant_count)
        .map(|_| {
            (0..dims)
                .map(|_| {
                    if rng.gen::<f32>() < sparsity {
                        0.0
                    } else {
                        normal.sample(&mut rng)
                    }
                })
                .collect()
        })
        .collect()
}

/// Generate gradient vectors where all participants have identical gradients
///
/// Useful for testing aggregation correctness
pub fn identical(participant_count: usize, dims: usize, value: f32) -> Vec<Vec<f32>> {
    vec![vec![value; dims]; participant_count]
}

/// Generate gradient vectors with known mean for testing
///
/// Each gradient is the target mean plus small Gaussian noise
pub fn with_known_mean(
    participant_count: usize,
    dims: usize,
    target_mean: Vec<f32>,
    noise_stddev: f32,
) -> Vec<Vec<f32>> {
    assert_eq!(target_mean.len(), dims, "Target mean must match dimensions");

    let mut rng = seeded_rng(DEFAULT_SEED);
    let noise = Normal::new(0.0, noise_stddev).unwrap();

    (0..participant_count)
        .map(|_| {
            target_mean
                .iter()
                .map(|&mean_val| mean_val + noise.sample(&mut rng))
                .collect()
        })
        .collect()
}

/// Generate gradient vectors for Byzantine attack simulation
///
/// Creates honest participants and Byzantine (malicious) participants
pub fn byzantine_scenario(
    honest_count: usize,
    byzantine_count: usize,
    dims: usize,
    attack_magnitude: f32,
) -> Vec<Vec<f32>> {
    let mut gradients = gaussian_default(honest_count, dims);

    // Add Byzantine gradients (all same direction, large magnitude)
    let byzantine_gradient = vec![attack_magnitude; dims];
    for _ in 0..byzantine_count {
        gradients.push(byzantine_gradient.clone());
    }

    gradients
}

/// Calculate the L2 norm of a gradient vector
pub fn l2_norm(gradient: &[f32]) -> f32 {
    gradient.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Calculate the mean gradient across all participants
pub fn mean_gradient(gradients: &[Vec<f32>]) -> Vec<f32> {
    let dims = gradients[0].len();
    let n = gradients.len() as f32;

    (0..dims)
        .map(|i| gradients.iter().map(|g| g[i]).sum::<f32>() / n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_generation() {
        let gradients = gaussian_default(10, 100);

        assert_eq!(gradients.len(), 10);
        assert_eq!(gradients[0].len(), 100);

        // Check values are reasonable (within 5 standard deviations)
        for grad in &gradients {
            for &val in grad {
                assert!(val.abs() < 5.0);
            }
        }
    }

    #[test]
    fn test_uniform_generation() {
        let gradients = uniform_default(5, 50);

        assert_eq!(gradients.len(), 5);
        assert_eq!(gradients[0].len(), 50);

        // Check values are in range
        for grad in &gradients {
            for &val in grad {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_with_outliers() {
        let gradients = with_outliers(10, 50, 2, 100.0);

        assert_eq!(gradients.len(), 10);

        // Check that at least some values are outliers
        let max_val = gradients
            .iter()
            .flat_map(|g| g.iter())
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);

        assert!(max_val > 50.0); // Should have extreme values
    }

    #[test]
    fn test_sparse_gradients() {
        let gradients = sparse(5, 100, 0.9); // 90% sparse

        assert_eq!(gradients.len(), 5);

        // Count zeros
        let zero_count: usize = gradients
            .iter()
            .flat_map(|g| g.iter())
            .filter(|&&x| x == 0.0)
            .count();

        let total_count = 5 * 100;
        let zero_ratio = zero_count as f32 / total_count as f32;

        // Should be approximately 90% zeros (with some tolerance)
        assert!(zero_ratio > 0.8 && zero_ratio < 1.0);
    }

    #[test]
    fn test_identical_gradients() {
        let gradients = identical(7, 10, 3.14);

        assert_eq!(gradients.len(), 7);
        for grad in &gradients {
            assert_eq!(grad.len(), 10);
            for &val in grad {
                assert!((val - 3.14).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_with_known_mean() {
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradients = with_known_mean(100, 5, target.clone(), 0.1);

        let calculated_mean = mean_gradient(&gradients);

        // Mean should be close to target
        for (i, &val) in calculated_mean.iter().enumerate() {
            assert!((val - target[i]).abs() < 0.5); // Within 0.5 of target
        }
    }

    #[test]
    fn test_byzantine_scenario() {
        let gradients = byzantine_scenario(20, 5, 10, 100.0);

        assert_eq!(gradients.len(), 25);

        // Last 5 should be Byzantine
        let byzantine_grad = &gradients[24];
        for &val in byzantine_grad {
            assert!((val - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_l2_norm() {
        let grad = vec![3.0, 4.0]; // 3-4-5 triangle
        let norm = l2_norm(&grad);

        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_gradient() {
        let gradients = vec![vec![1.0, 2.0, 3.0], vec![5.0, 6.0, 7.0]];

        let mean = mean_gradient(&gradients);

        assert_eq!(mean.len(), 3);
        assert!((mean[0] - 3.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);
        assert!((mean[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_deterministic_generation() {
        let grad1 = gaussian(5, 10, 0.0, 1.0, 42);
        let grad2 = gaussian(5, 10, 0.0, 1.0, 42);

        assert_eq!(grad1, grad2);
    }
}
