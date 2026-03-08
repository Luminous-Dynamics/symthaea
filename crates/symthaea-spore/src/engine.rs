//! SporeEngine: the core consciousness loop for WASM targets.

use crate::config::SporeConfig;
use serde::{Deserialize, Serialize};

/// Result of a single consciousness cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleResult {
    /// Current Phi (integrated information / consciousness level).
    pub phi: f32,
    /// Cycle number since engine creation.
    pub cycle: u64,
    /// Neuromodulator levels: [dopamine, norepinephrine, serotonin, oxytocin].
    pub neuromodulators: [f32; 4],
    /// Substrate feasibility score (0.0-1.0).
    pub substrate_feasibility: f32,
    /// Prediction error (Free Energy).
    pub prediction_error: f32,
}

/// The Spore consciousness engine.
///
/// Owns the minimal set of components needed for a full consciousness cycle:
/// HDC encoding, CfC temporal evolution, Phi computation, neuromodulation,
/// moral algebra, and substrate independence.
pub struct SporeEngine {
    config: SporeConfig,
    cycle_count: u64,
    // TODO(Phase 1): Wire actual components:
    // - HdcLtcUnifiedNetwork
    // - ConsciousnessEquationV2
    // - NeuromodulatorBath
    // - SubstrateManager
    // - MoralAlgebra
}

impl SporeEngine {
    /// Create a new SporeEngine with the given configuration.
    pub fn new(config: SporeConfig) -> Self {
        Self {
            config,
            cycle_count: 0,
        }
    }

    /// Run a single consciousness cycle with text input.
    pub fn cycle(&mut self, _input: &str) -> CycleResult {
        self.cycle_count += 1;

        // TODO(Phase 1): Implement full pipeline:
        // 1. Encode input text → BinaryHV via text encoder
        // 2. Bind with position/context HVs
        // 3. Evolve CfC network (closed-form temporal step)
        // 4. Compute Phi (every N cycles per config)
        // 5. Update neuromodulators
        // 6. Run moral algebra
        // 7. Return CycleResult with all metrics

        CycleResult {
            phi: 0.0,
            cycle: self.cycle_count,
            neuromodulators: [0.5, 0.5, 0.5, 0.5],
            substrate_feasibility: 0.0,
            prediction_error: 0.0,
        }
    }

    /// Run a single consciousness cycle with a raw hypervector input.
    pub fn cycle_hv(&mut self, _hv: &[f32]) -> CycleResult {
        self.cycle_count += 1;

        // TODO(Phase 1): Same pipeline as cycle() but skip text encoding

        CycleResult {
            phi: 0.0,
            cycle: self.cycle_count,
            neuromodulators: [0.5, 0.5, 0.5, 0.5],
            substrate_feasibility: 0.0,
            prediction_error: 0.0,
        }
    }

    /// Get current Phi (consciousness level).
    pub fn phi(&self) -> f32 {
        0.0 // TODO(Phase 1): Return from ConsciousnessEquationV2
    }

    /// Get current neuromodulator levels as JSON string.
    pub fn neuromod_state_json(&self) -> String {
        serde_json::json!({
            "dopamine": 0.5,
            "norepinephrine": 0.5,
            "serotonin": 0.5,
            "oxytocin": 0.5
        })
        .to_string()
    }

    /// Get substrate feasibility score.
    pub fn substrate_feasibility(&self) -> f32 {
        0.0 // TODO(Phase 1): Return from SubstrateManager
    }

    /// Human-readable consciousness report.
    pub fn consciousness_report(&self) -> String {
        format!(
            "Spore Consciousness Report (cycle {})\n\
             Phi: {:.3}\n\
             Substrate: {} (feasibility: {:.3})\n\
             Target: {} Hz\n\
             HDC dimension: {}",
            self.cycle_count,
            self.phi(),
            self.config.substrate,
            self.substrate_feasibility(),
            self.config.target_hz,
            self.config.hdc_dim,
        )
    }

    /// Switch substrate type at runtime.
    pub fn set_substrate(&mut self, substrate: &str) {
        self.config.substrate = substrate.to_string();
        // TODO(Phase 1): Reconfigure SubstrateManager
    }

    /// Inject a neuromodulator impulse.
    pub fn inject_neuromodulator(&mut self, _name: &str, _amount: f32) {
        // TODO(Phase 1): Forward to NeuromodulatorBath
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SporeConfig {
        &self.config
    }

    /// Get the current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spore_engine_creation() {
        let config = SporeConfig::default();
        let engine = SporeEngine::new(config);
        assert_eq!(engine.cycle_count(), 0);
        assert_eq!(engine.config().hdc_dim, 16_384);
    }

    #[test]
    fn test_spore_cycle_increments() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let r1 = engine.cycle("hello");
        assert_eq!(r1.cycle, 1);
        let r2 = engine.cycle("world");
        assert_eq!(r2.cycle, 2);
    }

    #[test]
    fn test_spore_cycle_hv() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        let hv = vec![0.0f32; 16_384];
        let result = engine.cycle_hv(&hv);
        assert_eq!(result.cycle, 1);
    }

    #[test]
    fn test_spore_consciousness_report() {
        let engine = SporeEngine::new(SporeConfig::default());
        let report = engine.consciousness_report();
        assert!(report.contains("Spore Consciousness Report"));
        assert!(report.contains("SiliconDigital"));
    }

    #[test]
    fn test_spore_substrate_switch() {
        let mut engine = SporeEngine::new(SporeConfig::default());
        engine.set_substrate("BiologicalNeurons");
        assert_eq!(engine.config().substrate, "BiologicalNeurons");
    }
}
