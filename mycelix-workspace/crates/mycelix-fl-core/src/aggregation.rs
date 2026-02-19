//! FL Aggregation Algorithms
//!
//! Five canonical aggregation methods for Byzantine-resistant gradient aggregation.

use std::collections::HashMap;
use thiserror::Error;

use crate::types::{AggregatedGradient, AggregationMethod, GradientUpdate};

/// Aggregation errors
#[derive(Debug, Error)]
pub enum AggregationError {
    #[error("No gradient updates provided")]
    NoUpdates,
    #[error("Gradient array is empty for participant {0}")]
    EmptyGradients(String),
    #[error("Gradient size mismatch: expected {expected}, got {actual} for {participant_id}")]
    GradientSizeMismatch {
        participant_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("Invalid trim percentage: {0} (must be 0.0-0.5)")]
    InvalidTrimPercentage(f32),
    #[error("Not enough participants for Krum: need at least 3, got {0}")]
    NotEnoughForKrum(usize),
    #[error("Invalid numSelect for Krum: {0}")]
    InvalidKrumSelect(usize),
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(u32),
    #[error("Loss is not finite for participant {0}")]
    InvalidLoss(String),
    #[error("No participants met trust threshold")]
    NoTrustedParticipants,
    #[error("Too many Byzantine participants detected")]
    TooManyByzantine,
}

/// Validate gradient updates have consistent dimensions
pub fn validate_gradient_consistency(updates: &[GradientUpdate]) -> Result<(), AggregationError> {
    if updates.is_empty() {
        return Err(AggregationError::NoUpdates);
    }

    let expected_size = updates[0].gradients.len();
    if expected_size == 0 {
        return Err(AggregationError::EmptyGradients(
            updates[0].participant_id.clone(),
        ));
    }

    for update in updates.iter().skip(1) {
        if update.gradients.len() != expected_size {
            return Err(AggregationError::GradientSizeMismatch {
                participant_id: update.participant_id.clone(),
                expected: expected_size,
                actual: update.gradients.len(),
            });
        }
    }

    Ok(())
}

/// Validate gradient update metadata
fn validate_update_metadata(update: &GradientUpdate) -> Result<(), AggregationError> {
    if update.metadata.batch_size == 0 {
        return Err(AggregationError::InvalidBatchSize(0));
    }
    if !update.metadata.loss.is_finite() {
        return Err(AggregationError::InvalidLoss(update.participant_id.clone()));
    }
    Ok(())
}

/// Federated Averaging (FedAvg)
///
/// Standard weighted average based on batch sizes.
pub fn fedavg(updates: &[GradientUpdate]) -> Result<Vec<f32>, AggregationError> {
    validate_gradient_consistency(updates)?;
    for update in updates {
        validate_update_metadata(update)?;
    }

    let gradient_size = updates[0].gradients.len();
    let mut result = vec![0.0f32; gradient_size];
    let total_samples: u64 = updates.iter().map(|u| u.metadata.batch_size as u64).sum();

    if total_samples == 0 {
        return Err(AggregationError::InvalidBatchSize(0));
    }

    for update in updates {
        let weight = update.metadata.batch_size as f32 / total_samples as f32;
        for (i, grad) in update.gradients.iter().enumerate() {
            result[i] += grad * weight;
        }
    }

    Ok(result)
}

/// Trimmed Mean Aggregation
///
/// Removes top and bottom percentile per dimension before averaging.
/// Robust to up to `trim_percentage` Byzantine participants at each extreme.
pub fn trimmed_mean(
    updates: &[GradientUpdate],
    trim_percentage: f32,
) -> Result<Vec<f32>, AggregationError> {
    validate_gradient_consistency(updates)?;

    if !(0.0..0.5).contains(&trim_percentage) {
        return Err(AggregationError::InvalidTrimPercentage(trim_percentage));
    }

    let gradient_size = updates[0].gradients.len();
    let mut result = vec![0.0f32; gradient_size];
    let trim_count = (updates.len() as f32 * trim_percentage).floor() as usize;

    let mut values: Vec<f32> = Vec::with_capacity(updates.len());

    for (i, res) in result.iter_mut().enumerate() {
        values.clear();
        values.extend(updates.iter().map(|u| u.gradients[i]));
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let trimmed = &values[trim_count..values.len().saturating_sub(trim_count)];
        if trimmed.is_empty() {
            *res = 0.0;
        } else {
            *res = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
        }
    }

    Ok(result)
}

/// Coordinate-wise Median
///
/// Robust to up to 50% Byzantine participants.
pub fn coordinate_median(updates: &[GradientUpdate]) -> Result<Vec<f32>, AggregationError> {
    validate_gradient_consistency(updates)?;

    let gradient_size = updates[0].gradients.len();
    let mut result = vec![0.0f32; gradient_size];
    let mut values: Vec<f32> = Vec::with_capacity(updates.len());
    let mid = updates.len() / 2;
    let is_even = updates.len().is_multiple_of(2);

    for (i, res) in result.iter_mut().enumerate() {
        values.clear();
        values.extend(updates.iter().map(|u| u.gradients[i]));
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if is_even {
            *res = (values[mid - 1] + values[mid]) / 2.0;
        } else {
            *res = values[mid];
        }
    }

    Ok(result)
}

/// Krum Aggregation
///
/// Selects the gradient closest to its neighbors.
/// Tolerates up to (n-2)/2 Byzantine participants.
pub fn krum(updates: &[GradientUpdate], num_select: usize) -> Result<Vec<f32>, AggregationError> {
    validate_gradient_consistency(updates)?;

    let n = updates.len();
    if n < 3 {
        return Err(AggregationError::NotEnoughForKrum(n));
    }
    if num_select < 1 || num_select > n {
        return Err(AggregationError::InvalidKrumSelect(num_select));
    }

    let num_neighbors = n - 2;

    // Pairwise distances
    let mut distances_flat: Vec<f32> = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&updates[i].gradients, &updates[j].gradients);
            distances_flat[i * n + j] = dist;
            distances_flat[j * n + i] = dist;
        }
    }

    // Krum scores
    let mut sorted_distances: Vec<f32> = Vec::with_capacity(n);
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);

    for i in 0..n {
        sorted_distances.clear();
        sorted_distances.extend(
            (0..n)
                .filter(|&j| j != i)
                .map(|j| distances_flat[i * n + j]),
        );
        sorted_distances
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let score: f32 = sorted_distances[..num_neighbors].iter().sum();
        scores.push((i, score));
    }

    scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Average selected updates
    let gradient_size = updates[0].gradients.len();
    let mut result = vec![0.0f32; gradient_size];
    let total_samples: u64 = scores[..num_select]
        .iter()
        .map(|(idx, _)| updates[*idx].metadata.batch_size as u64)
        .sum();

    if total_samples == 0 {
        return Err(AggregationError::InvalidBatchSize(0));
    }

    for (idx, _) in &scores[..num_select] {
        let update = &updates[*idx];
        let weight = update.metadata.batch_size as f32 / total_samples as f32;
        for (i, grad) in update.gradients.iter().enumerate() {
            result[i] += grad * weight;
        }
    }

    Ok(result)
}

/// Trust-Weighted Aggregation
///
/// Weights contributions by participant reputation scores.
pub fn trust_weighted(
    updates: &[GradientUpdate],
    reputations: &HashMap<String, f32>,
    trust_threshold: f32,
) -> Result<AggregatedGradient, AggregationError> {
    validate_gradient_consistency(updates)?;
    for update in updates {
        validate_update_metadata(update)?;
    }

    let gradient_size = updates[0].gradients.len();
    let mut result = vec![0.0f32; gradient_size];
    let mut total_weight: f32 = 0.0;
    let mut excluded_count = 0;
    let mut weights: HashMap<String, f32> = HashMap::new();

    for update in updates {
        let rep = match reputations.get(&update.participant_id) {
            Some(&r) => r,
            None => {
                excluded_count += 1;
                continue;
            }
        };

        if rep < trust_threshold {
            excluded_count += 1;
            continue;
        }

        let weight = rep * update.metadata.batch_size as f32;
        weights.insert(update.participant_id.clone(), weight);
        total_weight += weight;
    }

    if weights.is_empty() {
        return Err(AggregationError::NoTrustedParticipants);
    }

    for update in updates {
        if let Some(&weight) = weights.get(&update.participant_id) {
            let normalized = weight / total_weight;
            for (i, grad) in update.gradients.iter().enumerate() {
                result[i] += grad * normalized;
            }
        }
    }

    Ok(AggregatedGradient::new(
        result,
        updates[0].model_version,
        updates.len() - excluded_count,
        excluded_count,
        AggregationMethod::TrustWeighted,
    ))
}

/// Multi-Krum Aggregation
///
/// Selects the top-k gradients by Krum score (sum of distances to nearest
/// n-f-2 neighbors) and averages them. Combines Krum's Byzantine resilience
/// with the stability of averaging multiple selections.
///
/// Requires n >= 2*f + 3 participants.
///
/// # Arguments
/// * `updates` - Gradient updates from participants
/// * `f` - Maximum number of Byzantine participants to tolerate
/// * `k` - Number of top gradients to select and average
// Ported from fl-aggregator/src/byzantine.rs (MultiKrum variant)
pub fn multi_krum(
    updates: &[GradientUpdate],
    f: usize,
    k: usize,
) -> Result<Vec<f32>, AggregationError> {
    validate_gradient_consistency(updates)?;

    let n = updates.len();
    let min_required = 2 * f + 3;
    if n < min_required {
        return Err(AggregationError::NotEnoughForKrum(n));
    }
    if k == 0 || k > n.saturating_sub(f) {
        return Err(AggregationError::InvalidKrumSelect(k));
    }

    let take_count = n.saturating_sub(f).saturating_sub(2);

    // Pairwise distances
    let mut distances_flat: Vec<f32> = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&updates[i].gradients, &updates[j].gradients);
            distances_flat[i * n + j] = dist;
            distances_flat[j * n + i] = dist;
        }
    }

    // Krum scores: sum of distances to nearest (n-f-2) neighbors
    let mut sorted_distances: Vec<f32> = Vec::with_capacity(n);
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);

    for i in 0..n {
        sorted_distances.clear();
        sorted_distances.extend(
            (0..n)
                .filter(|&j| j != i)
                .map(|j| distances_flat[i * n + j]),
        );
        sorted_distances
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let score: f32 = sorted_distances.iter().take(take_count).sum();
        scores.push((i, score));
    }

    // Sort by score (lowest = most central)
    scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Average top-k selected gradients (equal weight)
    let gradient_size = updates[0].gradients.len();
    let mut result = vec![0.0f32; gradient_size];
    let selected_count = k.min(scores.len());

    for (idx, _) in &scores[..selected_count] {
        for (i, grad) in updates[*idx].gradients.iter().enumerate() {
            result[i] += grad;
        }
    }

    for val in result.iter_mut() {
        *val /= selected_count as f32;
    }

    Ok(result)
}

/// Geometric Median Aggregation via Weiszfeld algorithm
///
/// Iteratively approximates the geometric median — the point minimizing
/// the sum of Euclidean distances to all input gradients. More robust
/// than coordinate-wise median as it considers cross-dimensional structure.
///
/// # Arguments
/// * `updates` - Gradient updates from participants
/// * `max_iterations` - Maximum Weiszfeld iterations (typically 100)
/// * `tolerance` - Convergence tolerance (typically 1e-6)
// Ported from fl-aggregator/src/byzantine.rs (GeometricMedian variant)
pub fn geometric_median(
    updates: &[GradientUpdate],
    max_iterations: usize,
    tolerance: f32,
) -> Result<Vec<f32>, AggregationError> {
    validate_gradient_consistency(updates)?;

    let n = updates.len();
    let dim = updates[0].gradients.len();

    // Initialize with arithmetic mean
    let mut median = vec![0.0f32; dim];
    for update in updates {
        for (i, g) in update.gradients.iter().enumerate() {
            median[i] += g;
        }
    }
    for val in median.iter_mut() {
        *val /= n as f32;
    }

    // Weiszfeld iteration
    for _iter in 0..max_iterations {
        let mut numerator = vec![0.0f32; dim];
        let mut denominator = 0.0f32;

        for update in updates {
            let dist = euclidean_distance(&median, &update.gradients);
            if dist > 1e-10 {
                let weight = 1.0 / dist;
                for (i, g) in update.gradients.iter().enumerate() {
                    numerator[i] += g * weight;
                }
                denominator += weight;
            }
        }

        if denominator < 1e-10 {
            break;
        }

        let mut new_median = Vec::with_capacity(dim);
        for val in &numerator {
            new_median.push(val / denominator);
        }

        let change = euclidean_distance(&new_median, &median);
        median = new_median;

        if change < tolerance {
            break;
        }
    }

    Ok(median)
}

/// Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// L2 norm of a vector
pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_updates() -> Vec<GradientUpdate> {
        vec![
            GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![0.2, 0.3, 0.4], 100, 0.4),
            GradientUpdate::new("p3".into(), 1, vec![0.15, 0.25, 0.35], 100, 0.45),
        ]
    }

    #[test]
    fn test_fedavg() {
        let updates = test_updates();
        let result = fedavg(&updates).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.15).abs() < 0.001);
        assert!((result[1] - 0.25).abs() < 0.001);
        assert!((result[2] - 0.35).abs() < 0.001);
    }

    #[test]
    fn test_fedavg_weighted() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.0, 0.0], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![1.0, 1.0], 300, 0.4),
        ];
        let result = fedavg(&updates).unwrap();
        assert!((result[0] - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_trimmed_mean() {
        let mut updates = test_updates();
        // Add outlier
        updates.push(GradientUpdate::new(
            "byz".into(),
            1,
            vec![100.0, -50.0, 200.0],
            100,
            0.5,
        ));
        let result = trimmed_mean(&updates, 0.25).unwrap();
        // Outlier should be trimmed; result close to honest consensus
        assert!((result[0] - 0.175).abs() < 0.1);
    }

    #[test]
    fn test_coordinate_median() {
        let updates = test_updates();
        let result = coordinate_median(&updates).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.15).abs() < 0.001);
        assert!((result[1] - 0.25).abs() < 0.001);
        assert!((result[2] - 0.35).abs() < 0.001);
    }

    #[test]
    fn test_krum() {
        let updates = test_updates();
        let result = krum(&updates, 1).unwrap();
        assert_eq!(result.len(), 3);
        // Krum selects one gradient closest to neighbors
        for val in &result {
            assert!(*val > 0.0);
        }
    }

    #[test]
    fn test_krum_insufficient() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.1], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![0.2], 100, 0.4),
        ];
        assert!(krum(&updates, 1).is_err());
    }

    #[test]
    fn test_trust_weighted() {
        let updates = test_updates();
        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9_f32);
        reps.insert("p2".to_string(), 0.3_f32);
        reps.insert("p3".to_string(), 0.8_f32);

        let result = trust_weighted(&updates, &reps, 0.5).unwrap();
        assert_eq!(result.participant_count, 2); // p2 excluded (below 0.5)
        assert_eq!(result.excluded_count, 1);
    }

    #[test]
    fn test_empty_updates() {
        assert!(fedavg(&[]).is_err());
        assert!(trimmed_mean(&[], 0.1).is_err());
        assert!(coordinate_median(&[]).is_err());
    }

    #[test]
    fn test_size_mismatch() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![0.1], 100, 0.5),
        ];
        assert!(fedavg(&updates).is_err());
    }

    #[test]
    fn test_invalid_trim() {
        let updates = test_updates();
        assert!(trimmed_mean(&updates, 0.6).is_err());
        assert!(trimmed_mean(&updates, -0.1).is_err());
    }

    #[test]
    fn test_multi_krum_rejects_byzantine() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![1.0, 2.0, 3.0], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![1.1, 2.1, 3.1], 100, 0.4),
            GradientUpdate::new("p3".into(), 1, vec![1.2, 2.2, 3.2], 100, 0.45),
            GradientUpdate::new("p4".into(), 1, vec![0.9, 1.9, 2.9], 100, 0.5),
            GradientUpdate::new("byz".into(), 1, vec![100.0, 200.0, 300.0], 100, 0.3),
        ];
        let result = multi_krum(&updates, 1, 3).unwrap();
        // Average of 3 best should be near [1, 2, 3], not pulled by Byzantine
        assert!(result[0] < 5.0);
        assert!(result[1] < 5.0);
    }

    #[test]
    fn test_multi_krum_all_honest() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![1.0, 2.0], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![1.1, 2.1], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![1.2, 2.2], 100, 0.5),
            GradientUpdate::new("p4".into(), 1, vec![0.9, 1.9], 100, 0.5),
            GradientUpdate::new("p5".into(), 1, vec![1.05, 2.05], 100, 0.5),
        ];
        let result = multi_krum(&updates, 1, 3).unwrap();
        // Should be a reasonable average near [1.0, 2.0]
        assert!((result[0] - 1.0).abs() < 0.5);
        assert!((result[1] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_multi_krum_insufficient() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![1.0], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![2.0], 100, 0.5),
        ];
        // f=1 needs n >= 5
        assert!(multi_krum(&updates, 1, 1).is_err());
    }

    #[test]
    fn test_multi_krum_invalid_k() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![1.0], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![2.0], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![3.0], 100, 0.5),
            GradientUpdate::new("p4".into(), 1, vec![4.0], 100, 0.5),
            GradientUpdate::new("p5".into(), 1, vec![5.0], 100, 0.5),
        ];
        // k=0 is invalid
        assert!(multi_krum(&updates, 1, 0).is_err());
    }

    #[test]
    fn test_geometric_median_unit_square() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.0, 0.0], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![1.0, 0.0], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![0.0, 1.0], 100, 0.5),
            GradientUpdate::new("p4".into(), 1, vec![1.0, 1.0], 100, 0.5),
        ];
        let result = geometric_median(&updates, 100, 1e-6).unwrap();
        // Geometric median of unit square is approximately (0.5, 0.5)
        assert!((result[0] - 0.5).abs() < 0.1);
        assert!((result[1] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_geometric_median_robust_to_outlier() {
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![1.0, 2.0], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![1.1, 2.1], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![0.9, 1.9], 100, 0.5),
            GradientUpdate::new("byz".into(), 1, vec![100.0, 200.0], 100, 0.3),
        ];
        let result = geometric_median(&updates, 100, 1e-6).unwrap();
        // Should be closer to [1, 2] than to the outlier
        assert!((result[0] - 1.0).abs() < 1.0);
        assert!((result[1] - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_l2_norm() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
        assert!((l2_norm(&[]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_zero() {
        assert!((l2_norm(&[]) - 0.0).abs() < 1e-6);
    }
}
