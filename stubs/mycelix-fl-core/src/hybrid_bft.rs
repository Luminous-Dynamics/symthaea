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
    // Step 1: reputation gating
    let gated_with_idx: Vec<(usize, &ReputationGradient)> = contributions
        .iter()
        .enumerate()
        .filter(|(_, c)| c.reputation >= config.min_reputation)
        .collect();
    let gated_count = gated_with_idx.len();
    if gated_count < 2 {
        return None;
    }
    let dim = gated_with_idx[0].1.update.gradients.len();
    if dim == 0 {
        return None;
    }

    // Step 2: coordinate-wise trimming
    let trim_count = ((gated_count as f32) * config.trim_fraction).ceil() as usize;
    let mut trimmed_set = std::collections::HashSet::new();

    for d in 0..dim {
        let mut vals: Vec<(usize, f32, f32)> = gated_with_idx
            .iter()
            .map(|&(orig_idx, c)| {
                let v = if d < c.update.gradients.len() {
                    c.update.gradients[d]
                } else {
                    0.0
                };
                let w = c.reputation.powf(config.reputation_exponent)
                    * (1.0 - config.reputation_outlier_weight * (1.0 - c.reputation));
                (orig_idx, v, w)
            })
            .collect();
        vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Trim from both ends
        for i in 0..trim_count.min(vals.len() / 2) {
            trimmed_set.insert(vals[i].0);
        }
        for i in 0..trim_count.min(vals.len() / 2) {
            trimmed_set.insert(vals[vals.len() - 1 - i].0);
        }
    }

    // Step 3: weighted aggregation of surviving contributions
    let surviving: Vec<(usize, &ReputationGradient)> = gated_with_idx
        .iter()
        .filter(|(idx, _)| !trimmed_set.contains(idx))
        .copied()
        .collect();
    let surviving_count = surviving.len();

    let weights: Vec<f32> = surviving
        .iter()
        .map(|(_, c)| c.reputation.powf(config.reputation_exponent))
        .collect();
    let total_weight: f32 = weights.iter().sum();
    if total_weight <= 0.0 || surviving.is_empty() {
        // Fall back to gated set if trimming removed everything
        let weights: Vec<f32> = gated_with_idx
            .iter()
            .map(|(_, c)| c.reputation.powf(config.reputation_exponent))
            .collect();
        let total_weight: f32 = weights.iter().sum();
        let mut aggregated = vec![0.0f32; dim];
        for ((_, contrib), &weight) in gated_with_idx.iter().zip(weights.iter()) {
            let normalized = weight / total_weight;
            for (i, &g) in contrib.update.gradients.iter().enumerate() {
                aggregated[i] += g * normalized;
            }
        }
        return Some(HybridAggregationResult {
            aggregated,
            gated_count,
            surviving_count: gated_count,
            total_weight,
            trimmed_indices: vec![],
        });
    }

    let mut aggregated = vec![0.0f32; dim];
    for ((_, contrib), &weight) in surviving.iter().zip(weights.iter()) {
        let normalized = weight / total_weight;
        for (i, &g) in contrib.update.gradients.iter().enumerate() {
            aggregated[i] += g * normalized;
        }
    }

    let mut trimmed_indices: Vec<usize> = trimmed_set.into_iter().collect();
    trimmed_indices.sort();

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
