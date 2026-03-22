// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi gradient computation for consciousness-guided architecture optimization.

use super::genome::ArchitectureGenome;
use super::phenotype::DecodedArchitecture;

// ═══════════════════════════════════════════════════════════════════════════════
// PHI GRADIENT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Phi gradient with respect to architecture parameters
#[derive(Debug, Clone)]
pub struct PhiGradient {
    /// Gradient of Phi w.r.t. connection density
    pub d_density: f64,
    /// Gradient of Phi w.r.t. modularity
    pub d_modularity: f64,
    /// Gradient of Phi w.r.t. bridge ratio
    pub d_bridge_ratio: f64,
    /// Gradient of Phi w.r.t. tau ratio
    pub d_tau_ratio: f64,
    /// Gradient of Phi w.r.t. binding strength
    pub d_binding_strength: f64,
    /// Gradient of Phi w.r.t. recurrence
    pub d_recurrence: f64,
    /// Magnitude of the gradient
    pub magnitude: f64,
}

impl PhiGradient {
    /// Compute gradient via finite differences
    pub fn compute(genome: &ArchitectureGenome, epsilon: f32) -> Self {
        let base_arch = DecodedArchitecture::from_genome(genome);
        let base_phi = base_arch.compute_phi();

        // Perturb each parameter and compute gradient
        let d_density = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.connection_density = (g.connection_density + e).clamp(0.05, 0.95);
        });

        let d_modularity = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.modularity = (g.modularity + e).clamp(0.0, 1.0);
        });

        let d_bridge_ratio = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.bridge_ratio = (g.bridge_ratio + e).clamp(0.0, 0.8);
        });

        let d_tau_ratio = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.tau_ratio = (g.tau_ratio + e).clamp(0.1, 0.9);
        });

        let d_binding_strength = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.binding_strength = (g.binding_strength + e).clamp(0.1, 1.0);
        });

        let d_recurrence = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.recurrence = (g.recurrence + e).clamp(0.0, 1.0);
        });

        let magnitude = (d_density.powi(2)
            + d_modularity.powi(2)
            + d_bridge_ratio.powi(2)
            + d_tau_ratio.powi(2)
            + d_binding_strength.powi(2)
            + d_recurrence.powi(2))
        .sqrt();

        Self {
            d_density,
            d_modularity,
            d_bridge_ratio,
            d_tau_ratio,
            d_binding_strength,
            d_recurrence,
            magnitude,
        }
    }

    fn grad_component<F>(
        genome: &ArchitectureGenome,
        _base_phi: f64,
        epsilon: f32,
        mutate: F,
    ) -> f64
    where
        F: Fn(&mut ArchitectureGenome, f32),
    {
        // Forward perturbation
        let mut g_plus = genome.clone();
        mutate(&mut g_plus, epsilon);
        let arch_plus = DecodedArchitecture::from_genome(&g_plus);
        let phi_plus = arch_plus.compute_phi();

        // Backward perturbation
        let mut g_minus = genome.clone();
        mutate(&mut g_minus, -epsilon);
        let arch_minus = DecodedArchitecture::from_genome(&g_minus);
        let phi_minus = arch_minus.compute_phi();

        // Central difference
        (phi_plus - phi_minus) / (2.0 * epsilon as f64)
    }

    /// Apply gradient to genome (gradient ascent)
    pub fn apply(&self, genome: &mut ArchitectureGenome, learning_rate: f32) {
        genome.connection_density += (learning_rate as f64 * self.d_density) as f32;
        genome.connection_density = genome.connection_density.clamp(0.05, 0.95);

        genome.modularity += (learning_rate as f64 * self.d_modularity) as f32;
        genome.modularity = genome.modularity.clamp(0.0, 1.0);

        genome.bridge_ratio += (learning_rate as f64 * self.d_bridge_ratio) as f32;
        genome.bridge_ratio = genome.bridge_ratio.clamp(0.0, 0.8);

        genome.tau_ratio += (learning_rate as f64 * self.d_tau_ratio) as f32;
        genome.tau_ratio = genome.tau_ratio.clamp(0.1, 0.9);

        genome.binding_strength += (learning_rate as f64 * self.d_binding_strength) as f32;
        genome.binding_strength = genome.binding_strength.clamp(0.1, 1.0);

        genome.recurrence += (learning_rate as f64 * self.d_recurrence) as f32;
        genome.recurrence = genome.recurrence.clamp(0.0, 1.0);
    }

    /// Get direction as unit vector
    pub fn direction(&self) -> [f64; 6] {
        let mag = self.magnitude.max(1e-10);
        [
            self.d_density / mag,
            self.d_modularity / mag,
            self.d_bridge_ratio / mag,
            self.d_tau_ratio / mag,
            self.d_binding_strength / mag,
            self.d_recurrence / mag,
        ]
    }

    /// Cosine similarity with another gradient (for convergence detection)
    pub fn cosine_similarity(&self, other: &PhiGradient) -> f64 {
        let dot = self.d_density * other.d_density
            + self.d_modularity * other.d_modularity
            + self.d_bridge_ratio * other.d_bridge_ratio
            + self.d_tau_ratio * other.d_tau_ratio
            + self.d_binding_strength * other.d_binding_strength
            + self.d_recurrence * other.d_recurrence;

        dot / (self.magnitude * other.magnitude).max(1e-10)
    }
}

/// Velocity state for momentum-based gradient optimization
#[derive(Debug, Clone, Default)]
pub struct GradientVelocity {
    /// Velocity for connection density
    pub v_density: f32,
    /// Velocity for modularity
    pub v_modularity: f32,
    /// Velocity for bridge ratio
    pub v_bridge_ratio: f32,
    /// Velocity for tau ratio
    pub v_tau_ratio: f32,
    /// Velocity for binding strength
    pub v_binding_strength: f32,
    /// Velocity for recurrence
    pub v_recurrence: f32,
}

impl GradientVelocity {
    /// Create new zero velocity
    pub fn new() -> Self {
        Self::default()
    }

    /// Get velocity magnitude
    pub fn magnitude(&self) -> f32 {
        (self.v_density.powi(2)
            + self.v_modularity.powi(2)
            + self.v_bridge_ratio.powi(2)
            + self.v_tau_ratio.powi(2)
            + self.v_binding_strength.powi(2)
            + self.v_recurrence.powi(2))
        .sqrt()
    }

    /// Decay velocity (for simulated annealing)
    pub fn decay(&mut self, factor: f32) {
        self.v_density *= factor;
        self.v_modularity *= factor;
        self.v_bridge_ratio *= factor;
        self.v_tau_ratio *= factor;
        self.v_binding_strength *= factor;
        self.v_recurrence *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small genome for fast gradient tests.
    fn small_genome() -> ArchitectureGenome {
        ArchitectureGenome {
            num_nodes: 6,
            hdc_dim: 256,
            connection_density: 0.4,
            modularity: 0.5,
            bridge_ratio: 0.3,
            tau_ratio: 0.5,
            binding_strength: 0.5,
            recurrence: 0.3,
            ..Default::default()
        }
    }

    /// Helper: build a known gradient.
    fn unit_gradient() -> PhiGradient {
        PhiGradient {
            d_density: 1.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 1.0,
        }
    }

    #[test]
    fn test_compute_produces_finite_gradient() {
        let genome = small_genome();
        let grad = PhiGradient::compute(&genome, 0.01);

        assert!(grad.d_density.is_finite());
        assert!(grad.d_modularity.is_finite());
        assert!(grad.d_bridge_ratio.is_finite());
        assert!(grad.d_tau_ratio.is_finite());
        assert!(grad.d_binding_strength.is_finite());
        assert!(grad.d_recurrence.is_finite());
        assert!(grad.magnitude.is_finite());
        assert!(grad.magnitude >= 0.0);
    }

    #[test]
    fn test_magnitude_consistent_with_components() {
        let grad = PhiGradient {
            d_density: 3.0,
            d_modularity: 4.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 5.0, // sqrt(9+16) = 5
        };
        let recomputed = (grad.d_density.powi(2)
            + grad.d_modularity.powi(2)
            + grad.d_bridge_ratio.powi(2)
            + grad.d_tau_ratio.powi(2)
            + grad.d_binding_strength.powi(2)
            + grad.d_recurrence.powi(2))
        .sqrt();
        assert!((recomputed - grad.magnitude).abs() < 1e-10);
    }

    #[test]
    fn test_direction_is_unit_vector() {
        let grad = PhiGradient {
            d_density: 3.0,
            d_modularity: 4.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 5.0,
        };
        let dir = grad.direction();
        let norm: f64 = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "direction should be unit vector, got norm={}",
            norm
        );
        assert!((dir[0] - 0.6).abs() < 1e-9); // 3/5
        assert!((dir[1] - 0.8).abs() < 1e-9); // 4/5
    }

    #[test]
    fn test_direction_zero_gradient_safe() {
        let grad = PhiGradient {
            d_density: 0.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 0.0,
        };
        let dir = grad.direction();
        // Should not panic; magnitude clamped to 1e-10
        for d in &dir {
            assert!(d.is_finite());
        }
    }

    #[test]
    fn test_cosine_similarity_identical_gradients() {
        let g = PhiGradient {
            d_density: 1.0,
            d_modularity: 2.0,
            d_bridge_ratio: 3.0,
            d_tau_ratio: 4.0,
            d_binding_strength: 5.0,
            d_recurrence: 6.0,
            magnitude: (1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0f64).sqrt(),
        };
        let sim = g.cosine_similarity(&g);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_opposite_gradients() {
        let g1 = PhiGradient {
            d_density: 1.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 1.0,
        };
        let g2 = PhiGradient {
            d_density: -1.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 1.0,
        };
        let sim = g1.cosine_similarity(&g2);
        assert!(
            (sim + 1.0).abs() < 1e-9,
            "opposite gradients should have sim=-1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_gradients() {
        let g1 = PhiGradient {
            d_density: 1.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 1.0,
        };
        let g2 = PhiGradient {
            d_density: 0.0,
            d_modularity: 1.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 1.0,
        };
        let sim = g1.cosine_similarity(&g2);
        assert!(
            sim.abs() < 1e-9,
            "orthogonal gradients should have sim=0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_apply_moves_genome_in_gradient_direction() {
        let mut genome = small_genome();
        let original_density = genome.connection_density;

        let grad = unit_gradient(); // positive d_density only
        grad.apply(&mut genome, 0.1);

        assert!(
            genome.connection_density > original_density,
            "density should increase: {} -> {}",
            original_density,
            genome.connection_density
        );
    }

    #[test]
    fn test_apply_clamps_within_bounds() {
        let mut genome = ArchitectureGenome {
            connection_density: 0.94,
            modularity: 0.01,
            bridge_ratio: 0.79,
            tau_ratio: 0.11,
            binding_strength: 0.99,
            recurrence: 0.01,
            ..Default::default()
        };

        let grad = PhiGradient {
            d_density: 100.0,
            d_modularity: -100.0,
            d_bridge_ratio: 100.0,
            d_tau_ratio: -100.0,
            d_binding_strength: 100.0,
            d_recurrence: -100.0,
            magnitude: 244.9,
        };

        grad.apply(&mut genome, 1.0);

        assert!(genome.connection_density >= 0.05 && genome.connection_density <= 0.95);
        assert!(genome.modularity >= 0.0 && genome.modularity <= 1.0);
        assert!(genome.bridge_ratio >= 0.0 && genome.bridge_ratio <= 0.8);
        assert!(genome.tau_ratio >= 0.1 && genome.tau_ratio <= 0.9);
        assert!(genome.binding_strength >= 0.1 && genome.binding_strength <= 1.0);
        assert!(genome.recurrence >= 0.0 && genome.recurrence <= 1.0);
    }

    #[test]
    fn test_apply_zero_learning_rate_no_change() {
        let mut genome = small_genome();
        let orig = genome.clone();
        let grad = PhiGradient {
            d_density: 5.0,
            d_modularity: 5.0,
            d_bridge_ratio: 5.0,
            d_tau_ratio: 5.0,
            d_binding_strength: 5.0,
            d_recurrence: 5.0,
            magnitude: 12.2,
        };

        grad.apply(&mut genome, 0.0);

        assert!((genome.connection_density - orig.connection_density).abs() < f32::EPSILON);
        assert!((genome.modularity - orig.modularity).abs() < f32::EPSILON);
        assert!((genome.bridge_ratio - orig.bridge_ratio).abs() < f32::EPSILON);
        assert!((genome.tau_ratio - orig.tau_ratio).abs() < f32::EPSILON);
        assert!((genome.binding_strength - orig.binding_strength).abs() < f32::EPSILON);
        assert!((genome.recurrence - orig.recurrence).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_velocity_new_is_zero() {
        let v = GradientVelocity::new();
        assert!(v.magnitude() < 1e-9);
        assert!((v.v_density).abs() < f32::EPSILON);
        assert!((v.v_modularity).abs() < f32::EPSILON);
        assert!((v.v_bridge_ratio).abs() < f32::EPSILON);
        assert!((v.v_tau_ratio).abs() < f32::EPSILON);
        assert!((v.v_binding_strength).abs() < f32::EPSILON);
        assert!((v.v_recurrence).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_velocity_decay_halves() {
        let mut v = GradientVelocity {
            v_density: 1.0,
            v_modularity: 2.0,
            v_bridge_ratio: 3.0,
            v_tau_ratio: 4.0,
            v_binding_strength: 5.0,
            v_recurrence: 6.0,
        };
        let mag_before = v.magnitude();
        v.decay(0.5);
        let mag_after = v.magnitude();

        // Decay by 0.5 should halve magnitude
        assert!(
            (mag_after - mag_before * 0.5).abs() < 1e-5,
            "decay(0.5) should halve magnitude: {} vs {}",
            mag_after,
            mag_before * 0.5
        );
        assert!((v.v_density - 0.5).abs() < f32::EPSILON);
        assert!((v.v_modularity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_velocity_magnitude_formula() {
        let v = GradientVelocity {
            v_density: 1.0,
            v_modularity: 2.0,
            v_bridge_ratio: 2.0,
            v_tau_ratio: 0.0,
            v_binding_strength: 0.0,
            v_recurrence: 0.0,
        };
        // sqrt(1 + 4 + 4) = 3
        assert!((v.magnitude() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_velocity_default_equals_new() {
        let v1 = GradientVelocity::new();
        let v2 = GradientVelocity::default();
        assert!((v1.magnitude() - v2.magnitude()).abs() < f32::EPSILON);
    }
}
