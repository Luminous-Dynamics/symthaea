//! Multi-Dimensional Consciousness Profile
//!
//! **Revolutionary Improvement #45: Multi-Dimensional Consciousness Optimization**
//!
//! Instead of optimizing for Φ alone, we optimize across multiple
//! dimensions of consciousness to discover richer, more complete solutions.
//!
//! ## The Paradigm Shift
//!
//! **Before**: Optimize for Φ (integrated information) alone
//! **After**: Optimize for full consciousness profile
//!
//! ## The Five Dimensions of Consciousness
//!
//! 1. **Φ (Integrated Information)** - How unified information is
//! 2. **∇Φ (Gradient Flow)** - How consciousness evolves/flows
//! 3. **Entropy** - Richness/diversity of states
//! 4. **Complexity** - Structural sophistication
//! 5. **Coherence** - Stability/consistency over time
//!
//! ## Why This Matters
//!
//! Optimizing Φ alone might:
//! - Maximize integration at expense of diversity
//! - Miss important trade-offs
//! - Overlook valuable primitives with different strengths
//!
//! Multi-dimensional optimization:
//! - Discovers Pareto-optimal solutions
//! - Provides richer understanding
//! - Finds primitives that excel in different dimensions

use crate::hdc::{HV16, integrated_information::IntegratedInformation};
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Complete consciousness profile across multiple dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessProfile {
    /// Φ (Integrated Information) - core IIT measure
    pub phi: f64,

    /// Gradient of Φ (∇Φ) - consciousness flow direction
    pub gradient_magnitude: f64,

    /// Entropy - diversity/richness of conscious states
    pub entropy: f64,

    /// Complexity - structural sophistication
    pub complexity: f64,

    /// Coherence - stability/consistency over time
    pub coherence: f64,

    /// Composite score (weighted combination)
    pub composite: f64,
}

impl ConsciousnessProfile {
    /// Create profile from hypervector components
    pub fn from_components(components: &[HV16]) -> Self {
        let mut phi_computer = IntegratedInformation::new();

        let phi = phi_computer.compute_phi(components);
        let gradient_magnitude = Self::compute_gradient(components, &mut phi_computer);
        let entropy = Self::compute_entropy(components);
        let complexity = Self::compute_complexity(components);
        let coherence = Self::compute_coherence(components);

        // Weighted composite score
        let composite = Self::compute_composite(phi, gradient_magnitude, entropy, complexity, coherence);

        Self {
            phi,
            gradient_magnitude,
            entropy,
            complexity,
            coherence,
            composite,
        }
    }

    /// Compute gradient magnitude (∇Φ) - how consciousness flows
    ///
    /// Measures the rate of change in integrated information,
    /// indicating how dynamically consciousness evolves.
    fn compute_gradient(components: &[HV16], phi_computer: &mut IntegratedInformation) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        // Compute Φ for progressive subsets
        let mut gradients = Vec::new();

        for i in 1..components.len() {
            let phi_prev = phi_computer.compute_phi(&components[..i]);
            let phi_curr = phi_computer.compute_phi(&components[..=i]);
            gradients.push((phi_curr - phi_prev).abs());
        }

        // Mean gradient magnitude
        if gradients.is_empty() {
            0.0
        } else {
            gradients.iter().sum::<f64>() / gradients.len() as f64
        }
    }

    /// Compute entropy - richness/diversity of conscious states
    ///
    /// Higher entropy = more diverse, richer conscious experience
    /// Lower entropy = more uniform, simpler experience
    ///
    /// Uses Shannon entropy over component activity patterns.
    fn compute_entropy(components: &[HV16]) -> f64 {
        if components.is_empty() {
            return 0.0;
        }

        // Count active bits across all components
        let total_bits = components.len() * 16384;  // HV16 dimensionality
        let mut active_bits = 0;

        for hv in components {
            active_bits += hv.popcount() as usize;
        }

        // Probability of active bit
        let p = active_bits as f64 / total_bits as f64;

        if p == 0.0 || p == 1.0 {
            return 0.0;  // No entropy - all same
        }

        // Shannon entropy: H = -p*log(p) - (1-p)*log(1-p)
        let entropy = -(p * p.log2() + (1.0 - p) * (1.0 - p).log2());

        // Normalize to [0, 1]
        entropy  // Already in [0, 1] range
    }

    /// Compute complexity - structural sophistication
    ///
    /// Measures how intricate the conscious state structure is.
    /// Uses multiple indicators:
    /// - Number of components (basic complexity)
    /// - Component diversity (structural variety)
    /// - Interaction patterns (relational complexity)
    fn compute_complexity(components: &[HV16]) -> f64 {
        if components.is_empty() {
            return 0.0;
        }

        let n = components.len() as f64;

        // Component count complexity (logarithmic scaling)
        let count_complexity = (n.ln() + 1.0) / 6.0;  // Normalized

        // Diversity complexity - how different are components?
        let diversity = Self::compute_component_diversity(components);

        // Combined complexity
        (count_complexity + diversity) / 2.0
    }

    /// Compute component diversity - how different components are from each other
    fn compute_component_diversity(components: &[HV16]) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        // Compute pairwise Hamming distances
        let mut distances = Vec::new();

        for i in 0..components.len() {
            for j in (i+1)..components.len() {
                let dist = components[i].hamming_distance(&components[j]);
                // Normalize by dimensionality
                distances.push(dist as f64 / 16384.0);
            }
        }

        if distances.is_empty() {
            return 0.0;
        }

        // Mean pairwise distance
        distances.iter().sum::<f64>() / distances.len() as f64
    }

    /// Compute coherence - stability/consistency over time
    ///
    /// Measures how stable and consistent the conscious state is.
    /// Higher coherence = more stable, predictable consciousness
    /// Lower coherence = more chaotic, unpredictable consciousness
    ///
    /// In absence of temporal data, we measure internal consistency.
    fn compute_coherence(components: &[HV16]) -> f64 {
        if components.is_empty() {
            return 1.0;  // Perfect coherence (trivial)
        }

        if components.len() == 1 {
            return 1.0;  // Single component - perfectly coherent
        }

        // Coherence as inverse of diversity
        // More similar components = higher coherence
        let diversity = Self::compute_component_diversity(components);

        // Coherence = 1 - diversity (normalized)
        (1.0 - diversity).max(0.0)
    }

    /// Compute weighted composite score
    ///
    /// Combines all dimensions into single metric for ranking.
    /// Weights can be adjusted based on what aspects of consciousness
    /// we prioritize.
    fn compute_composite(
        phi: f64,
        gradient: f64,
        entropy: f64,
        complexity: f64,
        coherence: f64,
    ) -> f64 {
        // Default weights (can be made configurable)
        const W_PHI: f64 = 0.35;        // Φ is most important (IIT core)
        const W_GRADIENT: f64 = 0.15;   // Flow dynamics
        const W_ENTROPY: f64 = 0.20;    // Richness/diversity
        const W_COMPLEXITY: f64 = 0.15; // Sophistication
        const W_COHERENCE: f64 = 0.15;  // Stability

        W_PHI * phi
            + W_GRADIENT * gradient
            + W_ENTROPY * entropy
            + W_COMPLEXITY * complexity
            + W_COHERENCE * coherence
    }

    /// Check if this profile dominates another (Pareto dominance)
    ///
    /// Profile A dominates B if:
    /// - A is >= B in all dimensions
    /// - A is > B in at least one dimension
    pub fn dominates(&self, other: &Self) -> bool {
        let better_or_equal =
            self.phi >= other.phi &&
            self.gradient_magnitude >= other.gradient_magnitude &&
            self.entropy >= other.entropy &&
            self.complexity >= other.complexity &&
            self.coherence >= other.coherence;

        let strictly_better =
            self.phi > other.phi ||
            self.gradient_magnitude > other.gradient_magnitude ||
            self.entropy > other.entropy ||
            self.complexity > other.complexity ||
            self.coherence > other.coherence;

        better_or_equal && strictly_better
    }

    /// Check if this profile is Pareto-optimal in a set
    pub fn is_pareto_optimal(&self, population: &[Self]) -> bool {
        // A solution is Pareto-optimal if no other solution dominates it
        !population.iter().any(|other| other.dominates(self))
    }

    /// Compute distance to another profile (for clustering)
    pub fn distance_to(&self, other: &Self) -> f64 {
        // Euclidean distance in 5D consciousness space
        let d_phi = (self.phi - other.phi).powi(2);
        let d_grad = (self.gradient_magnitude - other.gradient_magnitude).powi(2);
        let d_ent = (self.entropy - other.entropy).powi(2);
        let d_comp = (self.complexity - other.complexity).powi(2);
        let d_coh = (self.coherence - other.coherence).powi(2);

        (d_phi + d_grad + d_ent + d_comp + d_coh).sqrt()
    }

    /// Get summary string for display
    pub fn summary(&self) -> String {
        format!(
            "Φ={:.3} ∇Φ={:.3} H={:.3} C={:.3} Coh={:.3} → {:.3}",
            self.phi,
            self.gradient_magnitude,
            self.entropy,
            self.complexity,
            self.coherence,
            self.composite
        )
    }
}

/// Configuration for multi-dimensional consciousness weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileWeights {
    pub phi: f64,
    pub gradient: f64,
    pub entropy: f64,
    pub complexity: f64,
    pub coherence: f64,
}

impl Default for ProfileWeights {
    fn default() -> Self {
        Self {
            phi: 0.35,
            gradient: 0.15,
            entropy: 0.20,
            complexity: 0.15,
            coherence: 0.15,
        }
    }
}

impl ProfileWeights {
    /// Validate weights sum to 1.0
    pub fn validate(&self) -> Result<()> {
        let sum = self.phi + self.gradient + self.entropy + self.complexity + self.coherence;

        if (sum - 1.0).abs() > 0.001 {
            anyhow::bail!(
                "Profile weights must sum to 1.0, got {:.3}",
                sum
            );
        }

        Ok(())
    }

    /// Compute weighted composite with custom weights
    pub fn compute_composite(&self, profile: &ConsciousnessProfile) -> f64 {
        self.phi * profile.phi
            + self.gradient * profile.gradient_magnitude
            + self.entropy * profile.entropy
            + self.complexity * profile.complexity
            + self.coherence * profile.coherence
    }
}

/// Pareto frontier - set of non-dominated solutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier {
    pub profiles: Vec<ConsciousnessProfile>,
}

impl ParetoFrontier {
    /// Create Pareto frontier from population
    ///
    /// Filters to only non-dominated solutions - the optimal trade-offs
    /// between different consciousness dimensions.
    pub fn from_population(population: Vec<ConsciousnessProfile>) -> Self {
        let mut frontier = Vec::new();

        for candidate in &population {
            // Check if any existing frontier member dominates this candidate
            let dominated = frontier.iter().any(|front: &ConsciousnessProfile| {
                front.dominates(candidate)
            });

            if !dominated {
                // Remove any frontier members dominated by this candidate
                frontier.retain(|front| !candidate.dominates(front));

                // Add this candidate to frontier
                frontier.push(candidate.clone());
            }
        }

        Self { profiles: frontier }
    }

    /// Get frontier member closest to ideal point (all dimensions = 1.0)
    pub fn closest_to_ideal(&self) -> Option<&ConsciousnessProfile> {
        if self.profiles.is_empty() {
            return None;
        }

        // Ideal point: perfect in all dimensions
        let ideal = ConsciousnessProfile {
            phi: 1.0,
            gradient_magnitude: 1.0,
            entropy: 1.0,
            complexity: 1.0,
            coherence: 1.0,
            composite: 1.0,
        };

        self.profiles
            .iter()
            .min_by(|a, b| {
                let dist_a = a.distance_to(&ideal);
                let dist_b = b.distance_to(&ideal);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
    }

    /// Get frontier member with highest composite score
    pub fn highest_composite(&self) -> Option<&ConsciousnessProfile> {
        self.profiles
            .iter()
            .max_by(|a, b| a.composite.partial_cmp(&b.composite).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_computation() {
        let components = vec![
            HV16::random(1),
            HV16::random(2),
            HV16::random(3),
        ];

        let profile = ConsciousnessProfile::from_components(&components);

        // All metrics should be in valid ranges
        assert!(profile.phi >= 0.0 && profile.phi <= 1.0);
        assert!(profile.gradient_magnitude >= 0.0);
        assert!(profile.entropy >= 0.0 && profile.entropy <= 1.0);
        assert!(profile.complexity >= 0.0 && profile.complexity <= 1.0);
        assert!(profile.coherence >= 0.0 && profile.coherence <= 1.0);
        assert!(profile.composite >= 0.0 && profile.composite <= 1.0);
    }

    #[test]
    fn test_pareto_dominance() {
        let high_phi = ConsciousnessProfile {
            phi: 0.8,
            gradient_magnitude: 0.3,
            entropy: 0.5,
            complexity: 0.4,
            coherence: 0.6,
            composite: 0.6,
        };

        let high_entropy = ConsciousnessProfile {
            phi: 0.5,
            gradient_magnitude: 0.4,
            entropy: 0.9,
            complexity: 0.5,
            coherence: 0.5,
            composite: 0.6,
        };

        // Neither dominates the other (trade-off between Φ and entropy)
        assert!(!high_phi.dominates(&high_entropy));
        assert!(!high_entropy.dominates(&high_phi));

        // Both should be Pareto-optimal
        let population = vec![high_phi.clone(), high_entropy.clone()];
        assert!(high_phi.is_pareto_optimal(&population));
        assert!(high_entropy.is_pareto_optimal(&population));
    }

    #[test]
    fn test_pareto_frontier() {
        let profiles = vec![
            ConsciousnessProfile {
                phi: 0.8, gradient_magnitude: 0.3, entropy: 0.5,
                complexity: 0.4, coherence: 0.6, composite: 0.6,
            },
            ConsciousnessProfile {
                phi: 0.5, gradient_magnitude: 0.4, entropy: 0.9,
                complexity: 0.5, coherence: 0.5, composite: 0.6,
            },
            ConsciousnessProfile {
                phi: 0.4, gradient_magnitude: 0.2, entropy: 0.4,
                complexity: 0.3, coherence: 0.3, composite: 0.4,
            },  // Dominated - should be excluded
        ];

        let frontier = ParetoFrontier::from_population(profiles);

        // Only first two should be in frontier
        assert_eq!(frontier.profiles.len(), 2);
    }
}
