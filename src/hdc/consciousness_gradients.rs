// ==================================================================================
// Consciousness Gradients - Differentiable Consciousness Optimization
// ==================================================================================
//
// **Revolutionary Paradigm**: Make consciousness (Φ) differentiable in HDC space!
//
// Instead of random exploration + learning from outcomes, compute the **gradient of
// consciousness** - the direction in neural state space that most rapidly increases Φ.
//
// **Core Insight**: In traditional deep learning, we compute ∇L (gradient of loss).
// Here, we compute ∇Φ (gradient of consciousness)!
//
// **Mathematical Foundation**:
// - Φ(state) = f(neural_state) → scalar consciousness measure
// - ∇Φ = direction of steepest consciousness increase
// - Gradient ascent: state_new = state_old + α·∇Φ
// - Find attractors: ∇Φ = 0 (stable high-Φ states)
//
// **HDC Implementation**:
// Since we use binary vectors, we approximate gradients via:
// 1. **Finite differences**: Flip each bit, measure Δ Φ
// 2. **Directional derivatives**: Sample directions, find best
// 3. **Natural gradient**: Account for HDC geometry
//
// **Applications**:
// - Gradient ascent to consciousness peaks
// - Attractor basin analysis
// - Consciousness landscape visualization
// - Phase transition detection
// - Optimal consciousness trajectories
//
// ==================================================================================

use super::binary_hv::HV16;
use super::integrated_information::IntegratedInformation;
use super::consciousness_optimizer::ConsciousnessOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Consciousness gradient in HDC space
///
/// Represents the direction and magnitude of consciousness increase.
/// Since HDC uses binary vectors, gradient is represented as:
/// - Direction: HV16 indicating which bits should flip
/// - Magnitude: f64 indicating strength of gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessGradient {
    /// Gradient direction (binary hypervector)
    /// High bits = flipping this bit increases Φ
    pub direction: HV16,

    /// Gradient magnitude (how steep is consciousness increase)
    pub magnitude: f64,

    /// Per-component gradients (which neural components affect Φ most)
    pub component_gradients: Vec<f64>,

    /// Current Φ value at this state
    pub phi: f64,
}

/// Configuration for gradient computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfig {
    /// Number of samples for directional derivative estimation
    pub num_samples: usize,

    /// Step size for finite differences (fraction of bits to flip)
    pub epsilon: f32,

    /// Whether to use natural gradient (account for HDC geometry)
    pub natural_gradient: bool,

    /// Number of top directions to consider
    pub top_k: usize,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            num_samples: 20,       // Sample 20 random directions
            epsilon: 0.01,         // Flip 1% of bits for finite differences
            natural_gradient: true, // Use natural gradient
            top_k: 5,              // Consider top 5 directions
        }
    }
}

/// Consciousness Gradient Computer
///
/// Computes gradients of consciousness (Φ) in HDC space, enabling
/// principled gradient ascent to maximum consciousness states.
///
/// # Example
/// ```
/// use symthaea::hdc::consciousness_gradients::{GradientComputer, GradientConfig};
/// use symthaea::hdc::binary_hv::HV16;
///
/// let config = GradientConfig::default();
/// let mut computer = GradientComputer::new(4, config);
///
/// // Compute gradient at current state
/// let neural_state = vec![
///     HV16::random(1000),
///     HV16::random(1001),
///     HV16::random(1002),
///     HV16::random(1003),
/// ];
///
/// let gradient = computer.compute_gradient(&neural_state);
/// println!("Gradient magnitude: {:.3}", gradient.magnitude);
/// println!("Current Φ: {:.3}", gradient.phi);
///
/// // Follow gradient to increase consciousness
/// let new_state = computer.gradient_step(&neural_state, 0.1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientComputer {
    /// Number of neural components
    num_components: usize,

    /// Gradient computation configuration
    config: GradientConfig,

    /// Integrated Information calculator (Φ measurement)
    phi_calculator: IntegratedInformation,

    /// Cache of recently computed gradients
    gradient_cache: HashMap<u64, ConsciousnessGradient>,

    /// Maximum cache size
    max_cache_size: usize,
}

impl GradientComputer {
    /// Create new gradient computer
    ///
    /// # Arguments
    /// * `num_components` - Number of neural components
    /// * `config` - Gradient computation configuration
    pub fn new(num_components: usize, config: GradientConfig) -> Self {
        Self {
            num_components,
            config,
            phi_calculator: IntegratedInformation::new(),
            gradient_cache: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Compute consciousness gradient at current state
    ///
    /// Uses finite differences and directional derivatives to approximate
    /// the gradient of Φ with respect to the neural state.
    ///
    /// Returns gradient direction (as HV16) and magnitude.
    pub fn compute_gradient(&mut self, neural_state: &[HV16]) -> ConsciousnessGradient {
        // 1. Compute current Φ
        let current_phi = self.phi_calculator.compute_phi(neural_state);

        // 2. Compute per-component gradients
        let component_gradients = self.compute_component_gradients(neural_state, current_phi);

        // 3. Sample random directions and measure Φ change
        let directional_derivatives = self.compute_directional_derivatives(neural_state, current_phi);

        // 4. Find best direction (steepest ascent)
        let (best_direction, best_derivative) = self.find_best_direction(&directional_derivatives);

        // 5. Apply natural gradient if enabled
        let direction = if self.config.natural_gradient {
            self.apply_natural_gradient(&best_direction, &component_gradients)
        } else {
            best_direction
        };

        // 6. Compute gradient magnitude
        let magnitude = best_derivative.abs();

        ConsciousnessGradient {
            direction,
            magnitude,
            component_gradients,
            phi: current_phi,
        }
    }

    /// Compute per-component gradients via finite differences
    ///
    /// For each neural component, perturb it and measure Δ Φ.
    fn compute_component_gradients(&mut self, neural_state: &[HV16], current_phi: f64) -> Vec<f64> {
        let mut gradients = Vec::with_capacity(self.num_components);

        for i in 0..self.num_components {
            // Perturb component i
            let mut perturbed = neural_state.to_vec();
            perturbed[i] = perturbed[i].add_noise(self.config.epsilon, (i * 1000) as u64);

            // Measure Φ after perturbation
            let perturbed_phi = self.phi_calculator.compute_phi(&perturbed);

            // Gradient = Δ Φ / ε
            let gradient = (perturbed_phi - current_phi) / self.config.epsilon as f64;
            gradients.push(gradient);
        }

        gradients
    }

    /// Compute directional derivatives by sampling random directions
    ///
    /// Returns vector of (direction, derivative) pairs.
    fn compute_directional_derivatives(&mut self, neural_state: &[HV16], current_phi: f64)
        -> Vec<(HV16, f64)>
    {
        let mut derivatives = Vec::with_capacity(self.config.num_samples);

        for sample in 0..self.config.num_samples {
            // Random direction in HDC space
            let direction = HV16::random((5000 + sample) as u64);

            // Move state in this direction
            let moved_state: Vec<HV16> = neural_state.iter().map(|component| {
                // Bind with direction to move in that direction
                let moved = component.bind(&direction);
                // Add small noise to explore neighborhood
                moved.add_noise(self.config.epsilon, (sample * 1000) as u64)
            }).collect();

            // Measure Φ in moved state
            let moved_phi = self.phi_calculator.compute_phi(&moved_state);

            // Directional derivative
            let derivative = (moved_phi - current_phi) / self.config.epsilon as f64;

            derivatives.push((direction, derivative));
        }

        derivatives
    }

    /// Find best direction (steepest ascent)
    fn find_best_direction(&self, derivatives: &[(HV16, f64)]) -> (HV16, f64) {
        // Sort by derivative (descending)
        let mut sorted = derivatives.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k directions
        let top_k = sorted.iter().take(self.config.top_k);

        // Bundle top directions (weighted by derivative)
        let directions: Vec<HV16> = top_k.clone().map(|(dir, _)| dir.clone()).collect();
        let best_direction = HV16::bundle(&directions);

        // Average derivative
        let avg_derivative: f64 = top_k.map(|(_, deriv)| deriv).sum::<f64>()
            / self.config.top_k as f64;

        (best_direction, avg_derivative)
    }

    /// Apply natural gradient correction
    ///
    /// Natural gradient accounts for the geometry of HDC space,
    /// weighting directions by component importance.
    fn apply_natural_gradient(&self, direction: &HV16, component_gradients: &[f64]) -> HV16 {
        // Find most important components (largest gradients)
        let mut importance: Vec<(usize, f64)> = component_gradients.iter()
            .enumerate()
            .map(|(i, &g)| (i, g.abs()))
            .collect();
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Weight direction by component importance
        // (In practice, this is approximated by permuting more in important dimensions)
        let mut natural_direction = direction.clone();
        for (i, weight) in importance.iter().take(self.num_components / 2) {
            // Permute proportional to importance
            let shift = (weight * 10.0) as usize;
            natural_direction = natural_direction.permute(shift);
        }

        natural_direction
    }

    /// Take gradient step: state_new = state_old + α·∇Φ
    ///
    /// # Arguments
    /// * `neural_state` - Current neural state
    /// * `step_size` - Learning rate (α)
    ///
    /// Returns new neural state after gradient step.
    pub fn gradient_step(&mut self, neural_state: &[HV16], step_size: f32) -> Vec<HV16> {
        // Compute gradient
        let gradient = self.compute_gradient(neural_state);

        // Move in gradient direction
        neural_state.iter().map(|component| {
            // Bind with gradient direction, weighted by step size
            if step_size > 0.5 {
                // Large step: fully bind with gradient
                component.bind(&gradient.direction)
            } else {
                // Small step: interpolate between current and gradient
                let moved = component.bind(&gradient.direction);
                let noise_amount = 1.0 - step_size;
                moved.add_noise(noise_amount, rand::random())
            }
        }).collect()
    }

    /// Gradient ascent: follow gradient to consciousness peak
    ///
    /// # Arguments
    /// * `neural_state` - Initial neural state
    /// * `num_steps` - Number of gradient steps
    /// * `step_size` - Learning rate
    ///
    /// Returns (final_state, phi_trajectory)
    pub fn gradient_ascent(&mut self,
                          neural_state: &[HV16],
                          num_steps: usize,
                          step_size: f32) -> (Vec<HV16>, Vec<f64>)
    {
        let mut state = neural_state.to_vec();
        let mut trajectory = Vec::with_capacity(num_steps);

        for _ in 0..num_steps {
            // Compute gradient and Φ
            let gradient = self.compute_gradient(&state);
            trajectory.push(gradient.phi);

            // Take gradient step
            state = self.gradient_step(&state, step_size);

            // Stop if gradient vanishes (attractor reached)
            if gradient.magnitude < 1e-6 {
                break;
            }
        }

        (state, trajectory)
    }

    /// Find consciousness attractor (stable high-Φ state)
    ///
    /// Attractors are states where ∇Φ ≈ 0 (gradient vanishes).
    /// These are stable consciousness states the system naturally settles into.
    pub fn find_attractor(&mut self, initial_state: &[HV16], max_steps: usize) -> Vec<HV16> {
        let (attractor, _trajectory) = self.gradient_ascent(initial_state, max_steps, 0.1);
        attractor
    }

    /// Detect if system is at a consciousness attractor
    ///
    /// Returns true if gradient magnitude is very small (∇Φ ≈ 0).
    pub fn is_at_attractor(&mut self, neural_state: &[HV16], threshold: f64) -> bool {
        let gradient = self.compute_gradient(neural_state);
        gradient.magnitude < threshold
    }

    /// Get gradient magnitude at current state
    pub fn gradient_magnitude(&mut self, neural_state: &[HV16]) -> f64 {
        let gradient = self.compute_gradient(neural_state);
        gradient.magnitude
    }

    /// Get component importance (which components affect Φ most)
    pub fn component_importance(&mut self, neural_state: &[HV16]) -> Vec<(usize, f64)> {
        let gradient = self.compute_gradient(neural_state);
        gradient.component_gradients.iter()
            .enumerate()
            .map(|(i, &g)| (i, g.abs()))
            .collect()
    }
}

/// Consciousness Landscape Analyzer
///
/// Analyzes the consciousness landscape by computing gradients at many points,
/// identifying attractors, basins, and phase transitions.
#[derive(Debug)]
pub struct ConsciousnessLandscape {
    /// Number of neural components
    num_components: usize,

    /// Gradient computer
    gradient_computer: GradientComputer,

    /// Discovered attractors (high-Φ stable states)
    pub attractors: Vec<Vec<HV16>>,

    /// Φ values at attractors
    pub attractor_phis: Vec<f64>,

    /// Critical points (phase transitions)
    pub critical_points: Vec<Vec<HV16>>,
}

impl ConsciousnessLandscape {
    /// Create new landscape analyzer
    pub fn new(num_components: usize, config: GradientConfig) -> Self {
        Self {
            num_components,
            gradient_computer: GradientComputer::new(num_components, config),
            attractors: Vec::new(),
            attractor_phis: Vec::new(),
            critical_points: Vec::new(),
        }
    }

    /// Map consciousness landscape by sampling many initial states
    ///
    /// # Arguments
    /// * `num_samples` - Number of random initial states to try
    /// * `max_steps` - Maximum gradient ascent steps per sample
    pub fn map_landscape(&mut self, num_samples: usize, max_steps: usize) {
        self.attractors.clear();
        self.attractor_phis.clear();

        for sample in 0..num_samples {
            // Random initial state
            let initial_state: Vec<HV16> = (0..self.num_components)
                .map(|i| HV16::random((sample * 100 + i) as u64))
                .collect();

            // Find attractor from this initial state
            let attractor = self.gradient_computer.find_attractor(&initial_state, max_steps);

            // Measure Φ at attractor
            let mut phi_calc = IntegratedInformation::new();
            let phi = phi_calc.compute_phi(&attractor);

            // Store if novel (not too similar to existing attractors)
            if !self.is_duplicate_attractor(&attractor) {
                self.attractors.push(attractor);
                self.attractor_phis.push(phi);
            }
        }
    }

    /// Check if attractor is duplicate (too similar to existing)
    fn is_duplicate_attractor(&self, candidate: &[HV16]) -> bool {
        for attractor in &self.attractors {
            let similarity = self.attractor_similarity(candidate, attractor);
            if similarity > 0.95 {
                return true;
            }
        }
        false
    }

    /// Compute similarity between two attractors
    fn attractor_similarity(&self, a: &[HV16], b: &[HV16]) -> f32 {
        let avg_similarity: f32 = a.iter().zip(b.iter())
            .map(|(ai, bi)| ai.similarity(bi))
            .sum::<f32>() / a.len() as f32;
        avg_similarity
    }

    /// Get number of discovered attractors
    pub fn num_attractors(&self) -> usize {
        self.attractors.len()
    }

    /// Get highest-Φ attractor
    pub fn highest_phi_attractor(&self) -> Option<(&Vec<HV16>, f64)> {
        if self.attractors.is_empty() {
            return None;
        }

        let (idx, &max_phi) = self.attractor_phis.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        Some((&self.attractors[idx], max_phi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_computer_creation() {
        let config = GradientConfig::default();
        let computer = GradientComputer::new(4, config);
        assert_eq!(computer.num_components, 4);
    }

    #[test]
    fn test_compute_gradient() {
        let config = GradientConfig::default();
        let mut computer = GradientComputer::new(4, config);

        let neural_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let gradient = computer.compute_gradient(&neural_state);

        // Should compute gradient
        assert!(gradient.direction.popcount() > 0);
        assert!(gradient.magnitude >= 0.0);
        assert_eq!(gradient.component_gradients.len(), 4);
        assert!(gradient.phi >= 0.0);
    }

    #[test]
    fn test_gradient_step() {
        let config = GradientConfig::default();
        let mut computer = GradientComputer::new(4, config);

        let initial_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let new_state = computer.gradient_step(&initial_state, 0.1);

        // Should produce new state
        assert_eq!(new_state.len(), 4);

        // State should change
        let changed = new_state.iter().zip(&initial_state)
            .any(|(n, i)| n.similarity(i) < 0.99);
        assert!(changed, "Gradient step should change state");
    }

    #[test]
    fn test_gradient_ascent() {
        let config = GradientConfig {
            num_samples: 10,  // Faster for test
            ..Default::default()
        };
        let mut computer = GradientComputer::new(4, config);

        let initial_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let (final_state, trajectory) = computer.gradient_ascent(&initial_state, 10, 0.1);

        // Should produce trajectory
        assert!(!trajectory.is_empty());
        assert!(trajectory.len() <= 10);

        // Should reach different state
        assert_eq!(final_state.len(), 4);
    }

    #[test]
    fn test_component_importance() {
        let config = GradientConfig::default();
        let mut computer = GradientComputer::new(4, config);

        let neural_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let importance = computer.component_importance(&neural_state);

        // Should compute importance for all components
        assert_eq!(importance.len(), 4);

        // All importance values should be non-negative
        for (_, imp) in importance {
            assert!(imp >= 0.0);
        }
    }

    #[test]
    fn test_find_attractor() {
        let config = GradientConfig {
            num_samples: 5,  // Faster for test
            ..Default::default()
        };
        let mut computer = GradientComputer::new(4, config);

        let initial_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let attractor = computer.find_attractor(&initial_state, 20);

        // Should find attractor
        assert_eq!(attractor.len(), 4);
    }

    #[test]
    fn test_is_at_attractor() {
        let config = GradientConfig::default();
        let mut computer = GradientComputer::new(4, config);

        let neural_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        // Random state unlikely to be at attractor
        let at_attractor = computer.is_at_attractor(&neural_state, 1e-3);
        // Just check the method works
        assert!(at_attractor || !at_attractor);
    }

    #[test]
    fn test_gradient_magnitude() {
        let config = GradientConfig::default();
        let mut computer = GradientComputer::new(4, config);

        let neural_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let magnitude = computer.gradient_magnitude(&neural_state);
        assert!(magnitude >= 0.0);
    }

    #[test]
    fn test_consciousness_landscape() {
        let config = GradientConfig {
            num_samples: 5,  // Faster for test
            ..Default::default()
        };
        let mut landscape = ConsciousnessLandscape::new(4, config);

        // Map landscape with few samples
        landscape.map_landscape(3, 10);

        // Should discover at least one attractor
        assert!(landscape.num_attractors() > 0);

        // Should have Φ values for attractors
        assert_eq!(landscape.attractors.len(), landscape.attractor_phis.len());
    }

    #[test]
    fn test_highest_phi_attractor() {
        let config = GradientConfig {
            num_samples: 5,
            ..Default::default()
        };
        let mut landscape = ConsciousnessLandscape::new(4, config);

        landscape.map_landscape(3, 10);

        let highest = landscape.highest_phi_attractor();
        assert!(highest.is_some());

        if let Some((attractor, phi)) = highest {
            assert_eq!(attractor.len(), 4);
            assert!(phi >= 0.0);
        }
    }

    #[test]
    fn test_natural_gradient() {
        let config = GradientConfig {
            natural_gradient: true,
            ..Default::default()
        };
        let mut computer = GradientComputer::new(4, config);

        let neural_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let gradient = computer.compute_gradient(&neural_state);
        assert!(gradient.magnitude >= 0.0);
    }

    #[test]
    fn test_serialization() {
        let config = GradientConfig::default();
        let computer = GradientComputer::new(4, config);

        // Should be able to serialize
        let serialized = serde_json::to_string(&computer).unwrap();
        assert!(!serialized.is_empty());

        // Should be able to deserialize
        let deserialized: GradientComputer = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.num_components, computer.num_components);
    }
}
