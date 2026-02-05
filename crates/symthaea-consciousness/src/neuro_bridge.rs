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
        let body_config = AutopoieticConfig {
            min_vitality: 0.1,
            decay_rate: 0.005, // Slower decay for demo stability
            production_rate: 0.2,
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
        let prediction = self.brain.predict_next_hdv();
        
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
