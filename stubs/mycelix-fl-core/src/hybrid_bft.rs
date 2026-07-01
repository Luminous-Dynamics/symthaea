//! Hybrid BFT aggregation (stub)

use crate::types::GradientUpdate;

#[derive(Debug, Clone)]
pub struct HybridBftConfig {
    pub min_reputation: f32,
    pub trim_fraction: f32,
    pub sample_dims: usize,
    pub reputation_exponent: f32,
    pub reputation_outlier_weight: f32,
}

impl Default for HybridBftConfig {
    fn default() -> Self {
        Self {
            min_reputation: 0.3,
            trim_fraction: 0.1,
            sample_dims: 0,
            reputation_exponent: 2.0,
            reputation_outlier_weight: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReputationGradient {
    pub update: GradientUpdate,
    pub reputation: f32,
}

#[derive(Debug)]
pub struct HybridAggregationResult {
    pub aggregated: Vec<f32>,
    pub gated_count: usize,
    pub surviving_count: usize,
    pub total_weight: f32,
    pub trimmed_indices: Vec<usize>,
}

pub fn hybrid_trimmed_mean(
    contributions: &[ReputationGradient],
    config: &HybridBftConfig,
) -> Option<HybridAggregationResult> {
    if contributions.is_empty() {
        return None;
    }

    // Phase 1: Reputation Gate
    let gated: Vec<(usize, &ReputationGradient)> = contributions
        .iter()
        .enumerate()
        .filter(|(_, c)| c.reputation >= config.min_reputation)
        .collect();

    let gated_count = gated.len();
    if gated_count < 2 {
        return None;
    }

    let dim = gated[0].1.update.gradients.len();
    if dim == 0 {
        return None;
    }

    // Phase 2: Reputation-Weighted Outlier Detection
    let trim_count = ((gated_count as f32 * config.trim_fraction) as usize).min(gated_count / 2);

    let (surviving, trimmed_indices) = if trim_count > 0 && gated_count >= 4 {
        let sample_dims = if config.sample_dims > 0 && config.sample_dims < dim {
            config.sample_dims
        } else {
            (dim / 10).clamp(1, 100)
        };
        let step = (dim / sample_dims).max(1);
        let sampled: Vec<usize> = (0..dim).step_by(step).collect();
        let num_sampled = sampled.len();

        let mut outlier_scores: Vec<f64> = vec![0.0; gated_count];

        for &d in &sampled {
            let mut values: Vec<(usize, f32)> = gated
                .iter()
                .enumerate()
                .map(|(local_idx, (_, c))| (local_idx, c.update.gradients[d]))
                .collect();
            values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..trim_count.min(values.len()) {
                let low_idx = values[i].0;
                let high_idx = values[values.len() - 1 - i].0;

                let low_rep = gated[low_idx].1.reputation;
                let high_rep = gated[high_idx].1.reputation;

                let rep_weight = config.reputation_outlier_weight;
                let low_penalty =
                    1.0 + rep_weight * (1.0 - low_rep.powf(config.reputation_exponent));
                let high_penalty =
                    1.0 + rep_weight * (1.0 - high_rep.powf(config.reputation_exponent));

                outlier_scores[low_idx] += low_penalty as f64;
                outlier_scores[high_idx] += high_penalty as f64;
            }
        }

        let max_possible = num_sampled as f64 * (1.0 + config.reputation_outlier_weight as f64);
        for score in outlier_scores.iter_mut() {
            *score /= max_possible.max(1.0);
        }

        let mut indexed_scores: Vec<(usize, f64)> = outlier_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let trimmed_local_indices: Vec<usize> = indexed_scores
            .iter()
            .take(trim_count)
            .map(|(i, _)| *i)
            .collect();

        let trimmed_original_indices: Vec<usize> = trimmed_local_indices
            .iter()
            .map(|&local_idx| gated[local_idx].0)
            .collect();

        let surviving: Vec<&ReputationGradient> = gated
            .iter()
            .enumerate()
            .filter(|(i, _)| !trimmed_local_indices.contains(i))
            .map(|(_, (_, c))| *c)
            .collect();

        (surviving, trimmed_original_indices)
    } else {
        let surviving: Vec<&ReputationGradient> = gated.iter().map(|(_, c)| *c).collect();
        (surviving, vec![])
    };

    let surviving_count = surviving.len();
    if surviving_count == 0 {
        return None;
    }

    // Phase 3: Reputation^exp Weighted Aggregation
    let weights: Vec<f32> = surviving
        .iter()
        .map(|c| c.reputation.powf(config.reputation_exponent))
        .collect();

    let total_weight: f32 = weights.iter().sum();
    if total_weight <= 0.0 {
        return None;
    }

    let mut aggregated = vec![0.0f32; dim];
    for (contrib, &weight) in surviving.iter().zip(weights.iter()) {
        let normalized = weight / total_weight;
        for (i, &g) in contrib.update.gradients.iter().enumerate() {
            aggregated[i] += g * normalized;
        }
    }

    Some(HybridAggregationResult {
        aggregated,
        gated_count,
        surviving_count,
        total_weight,
        trimmed_indices,
    })
}

pub fn effective_byzantine_fraction(
    total_nodes: usize,
    byzantine_nodes: usize,
    avg_byzantine_reputation: f32,
    avg_honest_reputation: f32,
    reputation_exponent: f32,
) -> f32 {
    let honest_nodes = total_nodes - byzantine_nodes;
    let byz_power = byzantine_nodes as f32 * avg_byzantine_reputation.powf(reputation_exponent);
    let honest_power = honest_nodes as f32 * avg_honest_reputation.powf(reputation_exponent);
    let total_power = byz_power + honest_power;
    if total_power <= 0.0 {
        return 0.0;
    }
    byz_power / total_power
}
