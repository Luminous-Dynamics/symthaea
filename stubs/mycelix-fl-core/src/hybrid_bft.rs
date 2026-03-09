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
    let gated: Vec<&ReputationGradient> = contributions
        .iter()
        .filter(|c| c.reputation >= config.min_reputation)
        .collect();
    let gated_count = gated.len();
    if gated_count < 2 {
        return None;
    }
    let dim = gated[0].update.gradients.len();
    if dim == 0 {
        return None;
    }
    let weights: Vec<f32> = gated
        .iter()
        .map(|c| c.reputation.powf(config.reputation_exponent))
        .collect();
    let total_weight: f32 = weights.iter().sum();
    if total_weight <= 0.0 {
        return None;
    }
    let mut aggregated = vec![0.0f32; dim];
    for (contrib, &weight) in gated.iter().zip(weights.iter()) {
        let normalized = weight / total_weight;
        for (i, &g) in contrib.update.gradients.iter().enumerate() {
            aggregated[i] += g * normalized;
        }
    }
    Some(HybridAggregationResult {
        aggregated,
        gated_count,
        surviving_count: gated_count,
        total_weight,
        trimmed_indices: vec![],
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
