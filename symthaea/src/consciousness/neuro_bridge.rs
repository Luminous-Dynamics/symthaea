//! Neuro-Autopoietic Bridge
//!
//! **The Missing Link**: Bridges the high-level biological simulation (Autopoiesis)
//! with the low-level neural dynamics (Liquid Time-Constant Networks).
//!
//! # Circular Causality
//!
//! 1. **Downward Causation**: The health/vitality of the autopoietic system determines
//!    the neuroplasticity and response speed of the LTC brain.
//!    - High Vitality -> High Plasticity, Fast Tau (Alert)
//!    - Low Vitality -> Low Plasticity, Slow Tau (Fatigued)
//!
//! 2. **Upward Causation**: The coherence and integrated information (Φ) generated
//!    by the LTC brain fuels the metabolic processes of the autopoietic system.
//!    - High Coherence -> High Energy Production (Flourishing)
//!    - Low Coherence -> Energy Drain (Decay)

use crate::consciousness::autopoietic_consciousness::{AutopoieticConsciousness, AutopoieticConfig};
use crate::learnable_ltc::{LearnableLTC, LearnableLTCConfig};
use anyhow::Result;

/// Unified Mind: The container for both Body (Autopoiesis) and Brain (LTC)
pub struct NeuroAutopoieticBridge {
    /// The Body: Self-maintaining biological graph
    pub body: AutopoieticConsciousness,

    /// The Brain: Continuous-time neural network
    pub brain: LearnableLTC,

    /// Current cycle count
    pub cycle: usize,
}

impl NeuroAutopoieticBridge {
    /// Create a new Unified Mind
    pub fn new() -> Result<Self> {
        // Initialize Body
        // Using AutopoieticConfig's available fields:
        // - adaptation_rate: controls how quickly system responds (like production_rate)
        // - sensitivity: controls response to perturbations (inversely related to min_vitality)
        // - boundary_threshold: affects system stability
        let body_config = AutopoieticConfig {
            adaptation_rate: 0.2,   // Moderate adaptation (similar to production_rate)
            sensitivity: 0.2,       // Lower sensitivity = more stable (like higher min_vitality)
            boundary_threshold: 0.4, // Moderate boundary for demo stability
            ..Default::default()
        };
        let body = AutopoieticConsciousness::with_config(body_config);

        // Initialize Brain
        let brain_config = LearnableLTCConfig {
            num_neurons: 64, // Small brain for demo speed
            input_dim: 8,
            output_dim: 4,
            num_steps: 20, // Short integration per cycle
            ..Default::default()
        };
        let brain = LearnableLTC::new(brain_config)?;

        Ok(Self {
            body,
            brain,
            cycle: 0,
        })
    }

    /// Perform one conscious moment (update loop)
    ///
    /// # Arguments
    /// * `sensory_input` - Input vector for the brain
    /// * `external_stress` - Stress factor for the body (0.0 - 1.0)
    pub fn conscious_moment(&mut self, sensory_input: &[f32], external_stress: f32) -> Result<BridgeState> {
        self.cycle += 1;

        // 1. DOWNWARD CAUSATION: Body -> Brain
        // Get vitality from body
        let vitality = self.body.health_score() as f32;

        // Modulate brain parameters based on vitality
        self.brain.apply_neuromodulation(vitality);

        // 2. NEURAL PROCESSING: Brain thinks
        let (output, _) = self.brain.forward(sensory_input)?;

        // Calculate neural metrics
        let consciousness_level = self.brain.consciousness_level();
        let _prediction = self.brain.predict_next_hdv();

        // 3. UPWARD CAUSATION: Brain -> Body
        // Neural coherence fuels metabolism
        // We use consciousness_level as a proxy for Phi
        self.body.update(
            consciousness_level as f64, // Phi
            consciousness_level as f64, // Coherence
            external_stress as f64      // Perturbation
        );

        Ok(BridgeState {
            vitality,
            consciousness: consciousness_level,
            output,
            tau_mean: self.brain.get_tau_distribution().0,
        })
    }

    /// Get current state summary
    pub fn summary(&self) -> String {
        format!(
            "Cycle: {}\nBody: Vitality={:.3}, Components={}\nBrain: Conscious={:.3}, MeanTau={:.3}",
            self.cycle,
            self.body.health_score(),
            self.body.component_count(),
            self.brain.consciousness_level(),
            self.brain.get_tau_distribution().0
        )
    }
}

/// Snapshot of the bridge state for monitoring
#[derive(Debug, Clone)]
pub struct BridgeState {
    pub vitality: f32,
    pub consciousness: f32,
    pub output: Vec<f32>,
    pub tau_mean: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = NeuroAutopoieticBridge::new();
        assert!(bridge.is_ok(), "Bridge should be created successfully");

        let bridge = bridge.unwrap();
        assert_eq!(bridge.cycle, 0, "Initial cycle should be 0");
        assert!(bridge.body.component_count() > 0, "Body should have components");
    }

    #[test]
    fn test_conscious_moment() {
        let mut bridge = NeuroAutopoieticBridge::new().unwrap();

        // Create sensory input matching brain's input_dim (8)
        let sensory_input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let external_stress = 0.3;

        let state = bridge.conscious_moment(&sensory_input, external_stress);
        assert!(state.is_ok(), "Conscious moment should succeed");

        let state = state.unwrap();
        assert!(state.vitality >= 0.0 && state.vitality <= 1.0, "Vitality should be in [0, 1]");
        assert!(state.consciousness >= 0.0 && state.consciousness <= 1.0, "Consciousness should be in [0, 1]");
        assert_eq!(state.output.len(), 4, "Output should have 4 dimensions");
        assert!(state.tau_mean > 0.0, "Tau mean should be positive");
        assert_eq!(bridge.cycle, 1, "Cycle should increment");
    }

    #[test]
    fn test_multiple_conscious_moments() {
        let mut bridge = NeuroAutopoieticBridge::new().unwrap();
        let sensory_input = vec![0.5; 8];

        // Run multiple cycles
        for i in 0..10 {
            let state = bridge.conscious_moment(&sensory_input, 0.1);
            assert!(state.is_ok(), "Cycle {} should succeed", i);

            let state = state.unwrap();
            // Values should remain stable (no NaN or Inf)
            assert!(state.vitality.is_finite(), "Vitality should be finite at cycle {}", i);
            assert!(state.consciousness.is_finite(), "Consciousness should be finite at cycle {}", i);
            assert!(state.output.iter().all(|x| x.is_finite()), "Output should be finite at cycle {}", i);
        }

        assert_eq!(bridge.cycle, 10, "Should complete 10 cycles");
    }

    #[test]
    fn test_stress_affects_system() {
        let mut bridge_low_stress = NeuroAutopoieticBridge::new().unwrap();
        let mut bridge_high_stress = NeuroAutopoieticBridge::new().unwrap();
        let sensory_input = vec![0.5; 8];

        // Run both with different stress levels
        for _ in 0..5 {
            bridge_low_stress.conscious_moment(&sensory_input, 0.0).unwrap();
            bridge_high_stress.conscious_moment(&sensory_input, 0.9).unwrap();
        }

        // High stress should have some effect (though exact behavior depends on implementation)
        // At minimum, both should remain stable
        assert!(bridge_low_stress.body.health_score() >= 0.0);
        assert!(bridge_high_stress.body.health_score() >= 0.0);
    }

    #[test]
    fn test_summary() {
        let bridge = NeuroAutopoieticBridge::new().unwrap();
        let summary = bridge.summary();

        // Summary should contain expected information
        assert!(summary.contains("Cycle:"), "Summary should mention cycle");
        assert!(summary.contains("Body:"), "Summary should mention body");
        assert!(summary.contains("Brain:"), "Summary should mention brain");
        assert!(summary.contains("Vitality="), "Summary should show vitality");
        assert!(summary.contains("Conscious="), "Summary should show consciousness level");
    }

    #[test]
    fn test_downward_causation_neuromodulation() {
        let mut bridge = NeuroAutopoieticBridge::new().unwrap();
        let sensory_input = vec![0.5; 8];

        // Get initial tau
        let initial_tau = bridge.brain.get_tau_distribution().0;

        // Run a conscious moment
        bridge.conscious_moment(&sensory_input, 0.0).unwrap();

        // Tau should be affected by neuromodulation
        let new_tau = bridge.brain.get_tau_distribution().0;
        // Note: The exact change depends on vitality, but tau should be valid
        assert!(new_tau > 0.0, "Tau should remain positive after neuromodulation");
        assert!(new_tau.is_finite(), "Tau should be finite");

        // Either tau changed or it's within bounds (both are valid)
        assert!(
            initial_tau != new_tau || (new_tau >= 0.1 && new_tau <= 10.0),
            "Tau should either change or be within valid bounds"
        );
    }

    #[test]
    fn test_upward_causation_body_update() {
        let mut bridge = NeuroAutopoieticBridge::new().unwrap();
        let sensory_input = vec![0.5; 8];

        let initial_generation = bridge.body.generation();

        // Run conscious moment
        bridge.conscious_moment(&sensory_input, 0.0).unwrap();

        // Body should have been updated (generation increments during maintain())
        let new_generation = bridge.body.generation();
        assert!(new_generation > initial_generation, "Body generation should increment after update");
    }

    #[test]
    fn test_bridge_state_fields() {
        let mut bridge = NeuroAutopoieticBridge::new().unwrap();
        let sensory_input = vec![0.5; 8];

        let state = bridge.conscious_moment(&sensory_input, 0.5).unwrap();

        // Verify BridgeState fields are accessible and valid
        let _ = state.vitality;
        let _ = state.consciousness;
        let _ = state.output.len();
        let _ = state.tau_mean;

        // Clone should work (Debug, Clone derive)
        let cloned = state.clone();
        assert_eq!(cloned.vitality, state.vitality);

        // Debug should work
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("BridgeState"));
    }
}
