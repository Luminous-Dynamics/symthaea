// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Aggregation methods for federated learning.
//!
//! Provides Byzantine-robust aggregation primitives for federated learning.

use crate::{AggregationConfig, AggregationError, Result};

/// Compute trimmed mean of vectors.
///
/// Trims the top and bottom `trim_percent` of values for each dimension,
/// then computes the mean. This provides robustness against outliers and
/// poisoning attacks.
///
/// # Arguments
///
/// * `updates` - Model updates from participants (each vector has same dimension)
/// * `config` - Aggregation configuration controlling trim percentage and minimums
///
/// # Returns
///
/// Aggregated model update with same dimension as inputs.
///
/// # Errors
///
/// - `EmptyUpdates`: No updates provided
/// - `InsufficientUpdates`: Fewer than `min_updates` provided
/// - `DimensionMismatch`: Updates have different dimensions
/// - `InvalidTrimPercent`: Trim percent outside [0.0, 0.5]
///
/// # Examples
///
/// ```
/// use edunet_agg::{trimmed_mean, AggregationConfig};
///
/// let updates = vec![
///     vec![1.0, 2.0],
///     vec![2.0, 3.0],
///     vec![3.0, 4.0],
///     vec![100.0, 200.0], // Outlier from malicious participant
/// ];
///
/// let config = AggregationConfig {
///     trim_percent: 0.25,
///     min_updates: 3,
/// };
///
/// let result = trimmed_mean(&updates, &config).unwrap();
/// // Outlier is trimmed, result is mean of [1,2,3] = 2.0 and [2,3,4] = 3.0
/// assert!((result[0] - 2.5).abs() < 0.1);
/// assert!((result[1] - 3.5).abs() < 0.1);
/// ```
pub fn trimmed_mean(
    updates: &[Vec<f32>],
    config: &AggregationConfig,
) -> Result<Vec<f32>> {
    if updates.is_empty() {
        return Err(AggregationError::EmptyUpdates);
    }

    if updates.len() < config.min_updates {
        return Err(AggregationError::InsufficientUpdates {
            got: updates.len(),
            min: config.min_updates,
        });
    }

    if config.trim_percent < 0.0 || config.trim_percent > 0.5 {
        return Err(AggregationError::InvalidTrimPercent(config.trim_percent));
    }

    let dim = updates[0].len();

    // Verify all updates have same dimension
    for update in updates {
        if update.len() != dim {
            return Err(AggregationError::DimensionMismatch {
                expected: dim,
                got: update.len(),
            });
        }
    }

    let mut result = vec![0.0; dim];
    // Determine how many elements to trim on each side.
    // Use ceil to avoid zero when a small trim_percent is requested with few updates.
    let trim_count = (updates.len() as f64 * config.trim_percent).ceil() as usize;
    let remaining = updates.len().saturating_sub(2 * trim_count);

    // Guard against trimming everything (or more than everything)
    if remaining == 0 {
        return Err(AggregationError::TrimmedAwayAllValues);
    }

    // For each dimension
    for d in 0..dim {
        // Collect values for this dimension
        let mut values: Vec<f32> = updates.iter().map(|u| u[d]).collect();

        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Trim and compute mean
        let trimmed = &values[trim_count..(values.len() - trim_count)];
        let sum: f32 = trimmed.iter().sum();
        result[d] = sum / trimmed.len() as f32;
    }

    Ok(result)
}

/// Compute median of vectors.
///
/// For each dimension, computes the median value across all updates.
/// Provides maximum robustness against outliers (50% breakdown point).
///
/// # Arguments
///
/// * `updates` - Model updates from participants
///
/// # Returns
///
/// Aggregated model update with median values per dimension.
///
/// # Errors
///
/// - `EmptyUpdates`: No updates provided
/// - `DimensionMismatch`: Updates have different dimensions
///
/// # Examples
///
/// ```
/// use edunet_agg::median;
///
/// let updates = vec![
///     vec![1.0, 2.0],
///     vec![2.0, 3.0],
///     vec![3.0, 4.0],
///     vec![100.0, 200.0], // Extreme outlier - median is still robust!
/// ];
///
/// let result = median(&updates).unwrap();
/// // Median = (2nd + 3rd values) / 2
/// assert_eq!(result[0], 2.5);
/// assert_eq!(result[1], 3.5);
/// ```
///
/// # Note
///
/// Median provides the strongest robustness but is less statistically efficient
/// than trimmed mean when most participants are honest.
pub fn median(updates: &[Vec<f32>]) -> Result<Vec<f32>> {
    if updates.is_empty() {
        return Err(AggregationError::EmptyUpdates);
    }

    let dim = updates[0].len();

    // Verify all updates have same dimension
    for update in updates {
        if update.len() != dim {
            return Err(AggregationError::DimensionMismatch {
                expected: dim,
                got: update.len(),
            });
        }
    }

    let mut result = vec![0.0; dim];

    // For each dimension
    for d in 0..dim {
        // Collect values for this dimension
        let mut values: Vec<f32> = updates.iter().map(|u| u[d]).collect();

        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute median
        let mid = values.len() / 2;
        result[d] = if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        };
    }

    Ok(result)
}

/// Compute weighted mean of vectors.
///
/// Combines updates with participant-specific weights, useful for incorporating
/// heterogeneity in data sizes or participant reputation.
///
/// # Arguments
///
/// * `updates` - Model updates from participants
/// * `weights` - Positive weights for each update (e.g., sample counts, reputation scores)
///
/// # Returns
///
/// Weighted average of updates, normalized by total weight.
///
/// # Errors
///
/// - `EmptyUpdates`: No updates provided
/// - `DimensionMismatch`: Updates and weights have different lengths, or updates have different dimensions
/// - `InvalidWeight`: Non-positive weight encountered
///
/// # Examples
///
/// ```
/// use edunet_agg::weighted_mean;
///
/// let updates = vec![
///     vec![1.0, 2.0], // Participant with 100 samples
///     vec![3.0, 4.0], // Participant with 300 samples
/// ];
///
/// let weights = vec![100.0, 300.0]; // Sample counts
///
/// let result = weighted_mean(&updates, &weights).unwrap();
/// // Result weighted toward second participant: (1*100 + 3*300)/400 = 2.5
/// assert_eq!(result[0], 2.5);
/// assert_eq!(result[1], 3.5);
/// ```
///
/// # Use Cases
///
/// - **Data heterogeneity**: Weight by number of training samples
/// - **Reputation systems**: Weight by participant stake or track record
/// - **Validation-based**: Weight by inverse validation loss
pub fn weighted_mean(
    updates: &[Vec<f32>],
    weights: &[f64],
) -> Result<Vec<f32>> {
    if updates.is_empty() {
        return Err(AggregationError::EmptyUpdates);
    }

    if updates.len() != weights.len() {
        return Err(AggregationError::DimensionMismatch {
            expected: updates.len(),
            got: weights.len(),
        });
    }

    // Verify all weights are positive
    for &w in weights {
        if w <= 0.0 {
            return Err(AggregationError::InvalidWeight(w));
        }
    }

    let dim = updates[0].len();

    // Verify all updates have same dimension
    for update in updates {
        if update.len() != dim {
            return Err(AggregationError::DimensionMismatch {
                expected: dim,
                got: update.len(),
            });
        }
    }

    let total_weight: f64 = weights.iter().sum();
    let mut result = vec![0.0; dim];

    for (update, &weight) in updates.iter().zip(weights.iter()) {
        for (d, &value) in update.iter().enumerate() {
            result[d] += value * (weight / total_weight) as f32;
        }
    }

    Ok(result)
}

/// Clip L2 norm of a vector to a maximum value.
///
/// If ||v||₂ > max_norm, scales v to have norm exactly max_norm.
/// This is a key privacy protection mechanism used in differential privacy.
///
/// # Arguments
///
/// * `vector` - Model update vector to clip (modified in-place)
/// * `max_norm` - Maximum allowed L2 norm
///
/// # Examples
///
/// ```
/// use edunet_agg::clip_l2_norm;
///
/// let mut update = vec![3.0, 4.0]; // L2 norm = sqrt(9 + 16) = 5.0
/// clip_l2_norm(&mut update, 1.0);
///
/// // Vector is scaled to norm 1.0: [3/5, 4/5] = [0.6, 0.8]
/// assert!((update[0] - 0.6).abs() < 0.01);
/// assert!((update[1] - 0.8).abs() < 0.01);
///
/// // Verify L2 norm
/// let norm: f32 = update.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 0.001);
/// ```
///
/// # Privacy Properties
///
/// Gradient clipping bounds the sensitivity of the aggregation function,
/// which is necessary for differential privacy guarantees. The privacy
/// parameter ε scales with `max_norm`.
///
/// # Performance
///
/// - Time: O(d) where d is vector dimension
/// - Space: O(1) (in-place modification)
pub fn clip_l2_norm(vector: &mut [f32], max_norm: f32) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for x in vector.iter_mut() {
            *x *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trimmed_mean() {
        let updates = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![100.0, 200.0], // Outlier
        ];

        let config = AggregationConfig {
            trim_percent: 0.25, // Trim top and bottom 25%
            min_updates: 3,
        };

        let result = trimmed_mean(&updates, &config).unwrap();

        // Should ignore the outlier and average the middle values
        assert!((result[0] - 2.5).abs() < 0.1);
        assert!((result[1] - 3.5).abs() < 0.1);
    }

    #[test]
    fn test_median() {
        let updates = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![100.0, 200.0], // Outlier - median is robust
        ];

        let result = median(&updates).unwrap();

        // Median should be between 2nd and 3rd values
        assert_eq!(result[0], 2.5);
        assert_eq!(result[1], 3.5);
    }

    #[test]
    fn test_weighted_mean() {
        let updates = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        let weights = vec![1.0, 3.0]; // Second update weighted 3x

        let result = weighted_mean(&updates, &weights).unwrap();

        // Result should be closer to second update
        assert_eq!(result[0], 2.5); // (1*1 + 3*3) / 4
        assert_eq!(result[1], 3.5); // (2*1 + 4*3) / 4
    }

    #[test]
    fn test_clip_l2_norm() {
        let mut vector = vec![3.0, 4.0]; // L2 norm = 5.0
        clip_l2_norm(&mut vector, 1.0);

        // Should be scaled to norm 1.0
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_insufficient_updates() {
        let updates = vec![vec![1.0, 2.0]];

        let config = AggregationConfig::default();

        let result = trimmed_mean(&updates, &config);
        assert!(matches!(
            result,
            Err(AggregationError::InsufficientUpdates { .. })
        ));
    }

    // Edge case tests
    #[test]
    fn test_trimmed_mean_boundary_trim_percent() {
        let updates = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];

        // Trim 0% - should be same as mean
        let config_zero = AggregationConfig {
            trim_percent: 0.0,
            min_updates: 1,
        };
        let result = trimmed_mean(&updates, &config_zero).unwrap();
        assert_eq!(result[0], 3.0); // (1+2+3+4+5)/5

        // Trim 40% - should trim 2 from each end, leaving middle value
        let config_high = AggregationConfig {
            trim_percent: 0.4,
            min_updates: 1,
        };
        let result = trimmed_mean(&updates, &config_high).unwrap();
        // Trims 2 from each end (40% of 5 = 2), leaving just the middle value: 3.0
        assert_eq!(result[0], 3.0);
    }

    #[test]
    fn test_median_odd_length() {
        let updates = vec![vec![1.0], vec![2.0], vec![3.0]];

        let result = median(&updates).unwrap();
        assert_eq!(result[0], 2.0); // Middle value
    }

    #[test]
    fn test_median_even_length() {
        let updates = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];

        let result = median(&updates).unwrap();
        assert_eq!(result[0], 2.5); // Average of two middle values
    }

    #[test]
    fn test_median_single_update() {
        let updates = vec![vec![42.0, 100.0]];

        let result = median(&updates).unwrap();
        assert_eq!(result[0], 42.0);
        assert_eq!(result[1], 100.0);
    }

    #[test]
    fn test_weighted_mean_equal_weights() {
        let updates = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![1.0, 1.0];

        let result = weighted_mean(&updates, &weights).unwrap();

        // Equal weights = simple average
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
    }

    #[test]
    fn test_weighted_mean_single_dominant_weight() {
        let updates = vec![vec![1.0, 2.0], vec![100.0, 200.0]];
        let weights = vec![0.001, 999.999]; // Second weight dominates

        let result = weighted_mean(&updates, &weights).unwrap();

        // Should be very close to second update
        assert!((result[0] - 100.0).abs() < 0.1);
        assert!((result[1] - 200.0).abs() < 0.1);
    }

    #[test]
    fn test_clip_l2_norm_no_clipping() {
        let mut vector = vec![0.3, 0.4]; // L2 norm = 0.5
        let original = vector.clone();
        clip_l2_norm(&mut vector, 1.0);

        // Should not be modified if norm < max_norm
        assert_eq!(vector, original);
    }

    #[test]
    fn test_clip_l2_norm_zero_vector() {
        let mut vector = vec![0.0, 0.0, 0.0];
        clip_l2_norm(&mut vector, 1.0);

        // Should remain zero
        assert_eq!(vector, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clip_l2_norm_large_vector() {
        let mut vector = vec![10.0; 100]; // Large dimension
        clip_l2_norm(&mut vector, 1.0);

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_updates_error() {
        let updates: Vec<Vec<f32>> = vec![];
        let config = AggregationConfig::default();

        assert!(matches!(
            trimmed_mean(&updates, &config),
            Err(AggregationError::EmptyUpdates)
        ));

        assert!(matches!(median(&updates), Err(AggregationError::EmptyUpdates)));

        let weights = vec![];
        assert!(matches!(
            weighted_mean(&updates, &weights),
            Err(AggregationError::EmptyUpdates)
        ));
    }

    #[test]
    fn test_dimension_mismatch() {
        let updates = vec![
            vec![1.0, 2.0],      // 2D
            vec![3.0, 4.0],      // 2D
            vec![5.0, 6.0],      // 2D
            vec![7.0, 8.0, 9.0], // 3D - mismatch!
        ];

        // Need enough updates to pass min_updates check
        let config = AggregationConfig {
            trim_percent: 0.1,
            min_updates: 3,
        };

        assert!(matches!(
            trimmed_mean(&updates, &config),
            Err(AggregationError::DimensionMismatch { .. })
        ));

        assert!(matches!(
            median(&updates),
            Err(AggregationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_invalid_weight() {
        let updates = vec![vec![1.0], vec![2.0]];
        let bad_weights = vec![1.0, -1.0]; // Negative weight!

        assert!(matches!(
            weighted_mean(&updates, &bad_weights),
            Err(AggregationError::InvalidWeight(_))
        ));

        let zero_weights = vec![1.0, 0.0]; // Zero weight!
        assert!(matches!(
            weighted_mean(&updates, &zero_weights),
            Err(AggregationError::InvalidWeight(_))
        ));
    }

    #[test]
    fn test_invalid_trim_percent() {
        let updates = vec![vec![1.0], vec![2.0], vec![3.0]];

        let bad_config = AggregationConfig {
            trim_percent: -0.1, // Negative!
            min_updates: 1,
        };
        assert!(matches!(
            trimmed_mean(&updates, &bad_config),
            Err(AggregationError::InvalidTrimPercent(_))
        ));

        let bad_config2 = AggregationConfig {
            trim_percent: 0.6, // > 0.5!
            min_updates: 1,
        };
        assert!(matches!(
            trimmed_mean(&updates, &bad_config2),
            Err(AggregationError::InvalidTrimPercent(_))
        ));
    }

    #[test]
    fn test_trimmed_mean_all_values_trimmed_error() {
        let updates = vec![vec![1.0], vec![2.0]];
        let config = AggregationConfig {
            trim_percent: 0.5, // Would trim both values
            min_updates: 2,
        };

        let result = trimmed_mean(&updates, &config);
        assert!(matches!(
            result,
            Err(AggregationError::TrimmedAwayAllValues)
        ));
    }
}
