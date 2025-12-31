//! # Causal Emergence Detection
//!
//! Identifies when macro-level patterns have more causal power than micro-level components.
//!
//! ## The Emergence Problem
//!
//! Consciousness may be an emergent property that cannot be reduced to its parts.
//! This module quantifies emergence by comparing:
//!
//! - **Macro Φ**: Integrated information of the whole system
//! - **Micro Φ sum**: Sum of Φ values for individual partitions
//!
//! If Φ_macro > Σ Φ_micro, we have **causal emergence**: the whole has more
//! causal power than the sum of its parts.
//!
//! ## Scientific Basis
//!
//! - Hoel et al. (2013): Quantifying causal emergence
//! - Tononi & Koch (2015): IIT and causal emergence
//! - Albantakis et al. (2023): IIT 4.0 emergence measures
//!
//! ## Example Usage
//!
//! ```rust
//! use symthaea::hdc::causal_emergence::{CausalEmergenceDetector, EmergenceConfig};
//! use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
//!
//! let detector = CausalEmergenceDetector::new(EmergenceConfig::default());
//!
//! // Create a topology
//! let topology = ConsciousnessTopology::ring(8, 16384, 42);
//!
//! // Detect emergence
//! let result = detector.detect(&topology);
//! println!("Emergence: {:.4}", result.emergence);
//! println!("Is emergent: {}", result.is_emergent);
//! ```

use crate::hdc::real_hv::RealHV;
use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
use crate::hdc::phi_real::RealPhiCalculator;
use serde::{Deserialize, Serialize};

/// Configuration for emergence detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceConfig {
    /// Minimum partition size to analyze
    pub min_partition_size: usize,

    /// Maximum number of partitions to try
    pub max_partitions: usize,

    /// Threshold for significance (emergence > threshold)
    pub emergence_threshold: f64,

    /// Whether to use natural partitioning based on topology
    pub use_natural_partitions: bool,
}

impl Default for EmergenceConfig {
    fn default() -> Self {
        Self {
            min_partition_size: 2,
            max_partitions: 8,
            emergence_threshold: 0.0,
            use_natural_partitions: true,
        }
    }
}

/// Result of emergence detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceResult {
    /// Φ of the whole system
    pub phi_macro: f64,

    /// Sum of Φ values for partitions
    pub phi_micro_sum: f64,

    /// Emergence = phi_macro - phi_micro_sum
    pub emergence: f64,

    /// Whether system is emergent (emergence > 0)
    pub is_emergent: bool,

    /// Emergence ratio = phi_macro / phi_micro_sum
    pub emergence_ratio: f64,

    /// Φ values for each partition
    pub partition_phis: Vec<f64>,

    /// Partition sizes
    pub partition_sizes: Vec<usize>,

    /// Number of partitions analyzed
    pub num_partitions: usize,

    /// Normalized emergence (per node)
    pub normalized_emergence: f64,
}

/// Detailed partition information
#[derive(Debug, Clone)]
pub struct Partition {
    /// Indices of nodes in this partition
    pub node_indices: Vec<usize>,

    /// Node representations
    pub representations: Vec<RealHV>,

    /// Φ value for this partition
    pub phi: f64,
}

/// Main causal emergence detector
#[derive(Debug)]
pub struct CausalEmergenceDetector {
    config: EmergenceConfig,
    phi_calculator: RealPhiCalculator,
}

impl CausalEmergenceDetector {
    /// Create a new detector with given configuration
    pub fn new(config: EmergenceConfig) -> Self {
        Self {
            config,
            phi_calculator: RealPhiCalculator::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(EmergenceConfig::default())
    }

    /// Detect causal emergence in a consciousness topology
    pub fn detect(&self, topology: &ConsciousnessTopology) -> EmergenceResult {
        let n = topology.node_representations.len();

        if n < self.config.min_partition_size * 2 {
            // Too small to partition meaningfully
            return EmergenceResult {
                phi_macro: 0.0,
                phi_micro_sum: 0.0,
                emergence: 0.0,
                is_emergent: false,
                emergence_ratio: 1.0,
                partition_phis: Vec::new(),
                partition_sizes: Vec::new(),
                num_partitions: 0,
                normalized_emergence: 0.0,
            };
        }

        // Step 1: Compute macro Φ (whole system)
        let phi_macro = self.phi_calculator.compute(&topology.node_representations);

        // Step 2: Generate partitions
        let partitions = if self.config.use_natural_partitions {
            self.natural_partitions(topology)
        } else {
            self.uniform_partitions(topology)
        };

        // Step 3: Compute Φ for each partition
        let partition_phis: Vec<f64> = partitions.iter()
            .map(|p| {
                if p.representations.len() >= 2 {
                    self.phi_calculator.compute(&p.representations)
                } else {
                    0.0
                }
            })
            .collect();

        let partition_sizes: Vec<usize> = partitions.iter()
            .map(|p| p.representations.len())
            .collect();

        // Step 4: Compute emergence metrics
        let phi_micro_sum: f64 = partition_phis.iter().sum();
        let emergence = phi_macro - phi_micro_sum;
        let is_emergent = emergence > self.config.emergence_threshold;
        let emergence_ratio = phi_macro / phi_micro_sum.max(0.001);
        let normalized_emergence = emergence / n as f64;

        EmergenceResult {
            phi_macro,
            phi_micro_sum,
            emergence,
            is_emergent,
            emergence_ratio,
            partition_phis,
            partition_sizes,
            num_partitions: partitions.len(),
            normalized_emergence,
        }
    }

    /// Generate partitions based on topology connectivity
    fn natural_partitions(&self, topology: &ConsciousnessTopology) -> Vec<Partition> {
        let n = topology.node_representations.len();

        if n <= self.config.min_partition_size * 2 {
            return self.uniform_partitions(topology);
        }

        // Use spectral clustering based on similarity matrix
        let mut partitions = Vec::new();

        // Build similarity matrix
        let mut similarities = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    similarities[i][j] = topology.node_representations[i]
                        .similarity(&topology.node_representations[j]) as f64;
                }
            }
        }

        // Find communities using greedy modularity
        let communities = self.find_communities(&similarities, n);

        for community in communities {
            if community.len() >= self.config.min_partition_size {
                let representations: Vec<RealHV> = community.iter()
                    .map(|&i| topology.node_representations[i].clone())
                    .collect();

                partitions.push(Partition {
                    node_indices: community,
                    representations,
                    phi: 0.0, // Will be computed later
                });
            }
        }

        // If no good communities found, fall back to uniform
        if partitions.is_empty() {
            return self.uniform_partitions(topology);
        }

        partitions
    }

    /// Find communities using simple greedy algorithm
    fn find_communities(&self, similarities: &[Vec<f64>], n: usize) -> Vec<Vec<usize>> {
        let num_partitions = (n / self.config.min_partition_size)
            .min(self.config.max_partitions)
            .max(2);

        // Initialize each node in its own community
        let mut community_of: Vec<usize> = (0..n).collect();
        let mut communities: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        // Merge communities greedily until we reach target number
        while communities.len() > num_partitions {
            let (merge_a, merge_b) = self.find_best_merge(similarities, &communities, &community_of);

            if merge_a == merge_b {
                break;
            }

            // Merge community b into community a
            let to_merge: Vec<usize> = communities[merge_b].clone();
            for &node in &to_merge {
                community_of[node] = merge_a;
            }
            communities[merge_a].extend(to_merge);
            communities.remove(merge_b);

            // Update community indices
            for node in 0..n {
                if community_of[node] > merge_b {
                    community_of[node] -= 1;
                }
            }
        }

        communities
    }

    /// Find best pair of communities to merge
    fn find_best_merge(
        &self,
        similarities: &[Vec<f64>],
        communities: &[Vec<usize>],
        _community_of: &[usize],
    ) -> (usize, usize) {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_pair = (0, 0);

        for i in 0..communities.len() {
            for j in (i + 1)..communities.len() {
                // Compute average similarity between communities
                let mut sum = 0.0;
                let mut count = 0;

                for &a in &communities[i] {
                    for &b in &communities[j] {
                        sum += similarities[a][b];
                        count += 1;
                    }
                }

                let avg_sim = if count > 0 { sum / count as f64 } else { 0.0 };

                if avg_sim > best_score {
                    best_score = avg_sim;
                    best_pair = (i, j);
                }
            }
        }

        best_pair
    }

    /// Generate uniform (equal-sized) partitions
    fn uniform_partitions(&self, topology: &ConsciousnessTopology) -> Vec<Partition> {
        let n = topology.node_representations.len();
        let num_partitions = (n / self.config.min_partition_size)
            .min(self.config.max_partitions)
            .max(1);
        let partition_size = n / num_partitions;

        let mut partitions = Vec::new();
        let mut start = 0;

        for i in 0..num_partitions {
            let end = if i == num_partitions - 1 {
                n
            } else {
                start + partition_size
            };

            let node_indices: Vec<usize> = (start..end).collect();
            let representations: Vec<RealHV> = node_indices.iter()
                .map(|&i| topology.node_representations[i].clone())
                .collect();

            partitions.push(Partition {
                node_indices,
                representations,
                phi: 0.0,
            });

            start = end;
        }

        partitions
    }

    /// Compute emergence with different partition schemes and return best
    pub fn detect_optimal(&self, topology: &ConsciousnessTopology) -> (EmergenceResult, String) {
        // Try both natural and uniform partitioning
        let natural_config = EmergenceConfig {
            use_natural_partitions: true,
            ..self.config.clone()
        };
        let uniform_config = EmergenceConfig {
            use_natural_partitions: false,
            ..self.config.clone()
        };

        let detector_natural = CausalEmergenceDetector::new(natural_config);
        let detector_uniform = CausalEmergenceDetector::new(uniform_config);

        let result_natural = detector_natural.detect(topology);
        let result_uniform = detector_uniform.detect(topology);

        // Return the one with higher emergence
        if result_natural.emergence > result_uniform.emergence {
            (result_natural, "natural".to_string())
        } else {
            (result_uniform, "uniform".to_string())
        }
    }

    /// Analyze emergence across different partition counts
    pub fn sweep_partitions(&self, topology: &ConsciousnessTopology) -> Vec<(usize, EmergenceResult)> {
        let n = topology.node_representations.len();
        let max_partitions = n / 2;

        let mut results = Vec::new();

        for num_parts in 2..=max_partitions.min(8) {
            let config = EmergenceConfig {
                max_partitions: num_parts,
                min_partition_size: 2,
                use_natural_partitions: false,
                ..self.config.clone()
            };
            let detector = CausalEmergenceDetector::new(config);
            let result = detector.detect(topology);
            results.push((num_parts, result));
        }

        results
    }
}

/// Measures the strength of emergence at different scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleEmergence {
    /// Emergence values at different partition counts
    pub scale_emergence: Vec<(usize, f64)>,

    /// Scale with maximum emergence
    pub optimal_scale: usize,

    /// Maximum emergence value
    pub max_emergence: f64,

    /// Whether any scale shows emergence
    pub has_emergence: bool,
}

impl CausalEmergenceDetector {
    /// Compute emergence at multiple scales
    pub fn multi_scale_emergence(&self, topology: &ConsciousnessTopology) -> MultiScaleEmergence {
        let results = self.sweep_partitions(topology);

        let scale_emergence: Vec<(usize, f64)> = results.iter()
            .map(|(scale, r)| (*scale, r.emergence))
            .collect();

        let (optimal_scale, max_emergence) = scale_emergence.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(s, e)| (s, e))
            .unwrap_or((2, 0.0));

        let has_emergence = scale_emergence.iter().any(|(_, e)| *e > 0.0);

        MultiScaleEmergence {
            scale_emergence,
            optimal_scale,
            max_emergence,
            has_emergence,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;

    fn create_test_topology(n: usize) -> ConsciousnessTopology {
        // Use the canonical ring topology generator with correct RealHV types
        ConsciousnessTopology::ring(n, HDC_DIMENSION, 42)
    }

    #[test]
    fn test_emergence_detection() {
        let detector = CausalEmergenceDetector::default_config();
        let topology = create_test_topology(8);

        let result = detector.detect(&topology);

        println!("Φ macro: {:.4}", result.phi_macro);
        println!("Φ micro sum: {:.4}", result.phi_micro_sum);
        println!("Emergence: {:.4}", result.emergence);
        println!("Is emergent: {}", result.is_emergent);
        println!("Partitions: {}", result.num_partitions);

        // The system should have measurable Φ
        assert!(result.phi_macro > 0.0, "System should have positive Φ");
    }

    #[test]
    fn test_uniform_partitions() {
        let config = EmergenceConfig {
            use_natural_partitions: false,
            max_partitions: 4,
            ..Default::default()
        };
        let detector = CausalEmergenceDetector::new(config);
        let topology = create_test_topology(8);

        let result = detector.detect(&topology);

        println!("Uniform partitions: {:?}", result.partition_sizes);
        assert!(result.partition_sizes.len() >= 2);
    }

    #[test]
    fn test_multi_scale() {
        let detector = CausalEmergenceDetector::default_config();
        let topology = create_test_topology(8);

        let multi = detector.multi_scale_emergence(&topology);

        println!("Scale emergence:");
        for (scale, emergence) in &multi.scale_emergence {
            println!("  {} partitions: {:.4}", scale, emergence);
        }
        println!("Optimal scale: {}", multi.optimal_scale);
        println!("Max emergence: {:.4}", multi.max_emergence);
    }

    #[test]
    fn test_emergence_ratio() {
        let detector = CausalEmergenceDetector::default_config();
        let topology = create_test_topology(8);

        let result = detector.detect(&topology);

        println!("Emergence ratio: {:.4}", result.emergence_ratio);
        println!("Normalized emergence: {:.6}", result.normalized_emergence);

        // Ratio should be defined
        assert!(result.emergence_ratio.is_finite());
    }

    #[test]
    fn test_small_topology() {
        let detector = CausalEmergenceDetector::default_config();
        let topology = create_test_topology(4);

        let result = detector.detect(&topology);

        // Small topology should still work
        println!("Small topology Φ: {:.4}", result.phi_macro);
    }
}
