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

        let magnitude = (d_density.powi(2) + d_modularity.powi(2) + d_bridge_ratio.powi(2)
            + d_tau_ratio.powi(2) + d_binding_strength.powi(2) + d_recurrence.powi(2)).sqrt();

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

    fn grad_component<F>(genome: &ArchitectureGenome, _base_phi: f64, epsilon: f32, mutate: F) -> f64
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
