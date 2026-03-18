//! wasm_bindgen exports for browser/JS consumption.
//!
//! Activated by the `wasm` feature flag.
//! Build with: `./build-wasm.sh` or see build-wasm.sh for manual steps.

use wasm_bindgen::prelude::*;

use crate::config::SporeConfig;
use crate::engine::SporeEngine as InnerEngine;

/// WASM-exported Spore consciousness engine.
#[wasm_bindgen]
pub struct SporeEngine {
    inner: InnerEngine,
}

#[wasm_bindgen]
impl SporeEngine {
    /// Create a new SporeEngine from a JSON configuration.
    /// Pass `null` or `undefined` for defaults.
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<SporeEngine, JsError> {
        console_error_panic_hook::set_once();
        let config: SporeConfig = if config.is_null() || config.is_undefined() {
            SporeConfig::default()
        } else {
            serde_wasm_bindgen::from_value(config)
                .map_err(|e| JsError::new(&format!("Invalid config: {e}")))?
        };
        Ok(Self {
            inner: InnerEngine::new(config),
        })
    }

    /// Run a consciousness cycle with text input. Returns CycleResult as JS object.
    pub fn cycle(&mut self, input: &str) -> Result<JsValue, JsError> {
        let result = self.inner.cycle(input);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run a consciousness cycle with raw hypervector input.
    pub fn cycle_hv(&mut self, hv: &[f32]) -> Result<JsValue, JsError> {
        let result = self.inner.cycle_hv(hv);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Current consciousness level.
    pub fn consciousness_level(&self) -> f32 {
        self.inner.consciousness_level()
    }

    /// Honest confidence in the consciousness measurement (0.0-0.95).
    pub fn honest_confidence(&self) -> f32 {
        self.inner.honest_confidence()
    }

    /// Current harmony alignment score (0.0-1.0).
    pub fn harmony_alignment(&self) -> f32 {
        self.inner.harmony_alignment()
    }

    /// Neuromodulator state as JSON string.
    pub fn neuromod_state(&self) -> String {
        self.inner.neuromod_state_json()
    }

    /// Substrate feasibility score.
    pub fn substrate_feasibility(&self) -> f32 {
        self.inner.substrate_feasibility()
    }

    /// Human-readable consciousness report with epistemic disclaimers.
    pub fn consciousness_report(&self) -> String {
        self.inner.consciousness_report()
    }

    /// Switch substrate type.
    pub fn set_substrate(&mut self, substrate: &str) {
        self.inner.set_substrate(substrate);
    }

    /// Inject neuromodulator impulse.
    pub fn inject_neuromodulator(&mut self, name: &str, amount: f32) {
        self.inner.inject_neuromodulator(name, amount);
    }

    /// Current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.inner.cycle_count()
    }

    /// Number of active SporeEngine instances globally.
    pub fn active_instance_count() -> usize {
        InnerEngine::active_instance_count()
    }

    // ======================================================================
    // Hypervector access (for visualization)
    // ======================================================================

    /// Get the current network output hypervector (16,384 f32 values).
    /// Used for live waveform visualization in the browser demo.
    pub fn get_output_hv(&self) -> Vec<f32> {
        self.inner.get_output_hv()
    }

    /// Encode text to an HDC hypervector without running a full cycle.
    /// Returns bipolar encoding as f32 values. Used for thought comparison.
    pub fn encode_text(&mut self, text: &str) -> Vec<f32> {
        self.inner.encode_text(text)
    }

    // ======================================================================
    // Language generation (Broca)
    // ======================================================================

    /// Generate text from current consciousness state.
    /// Returns GenerationResult as JS object with `text`, `num_tokens`, `eos_terminated`.
    pub fn generate_text(&mut self, max_tokens: usize) -> Result<JsValue, JsError> {
        let result = self.inner.generate_text(max_tokens);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Dream engine
    // ======================================================================

    /// Run a dream cycle — simulate counterfactual alternatives.
    pub fn dream_cycle(&mut self) -> Result<JsValue, JsError> {
        let result = self.inner.dream_cycle();
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run a dream session (multiple dream cycles).
    pub fn dream_session(&mut self, cycles: usize) -> Result<JsValue, JsError> {
        let results = self.inner.dream_session(cycles);
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Number of wisdom entries accumulated from dreaming.
    pub fn dream_wisdom_count(&self) -> usize {
        self.inner.dream_wisdom_count()
    }

    /// Dream engine statistics.
    pub fn dream_stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner.dream_stats();
        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Active inference (FEP)
    // ======================================================================

    /// Current free energy value.
    pub fn free_energy(&self) -> f32 {
        self.inner.free_energy()
    }

    /// Run an explicit FEP cycle. Returns FepCycleResult.
    pub fn fep_cycle(&mut self) -> Result<JsValue, JsError> {
        let result = self.inner.fep_cycle();
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Topology analysis
    // ======================================================================

    /// Analyze consciousness topology. Returns TopologyAnalysis.
    pub fn topology_analysis(&mut self) -> Result<JsValue, JsError> {
        let analysis = self.inner.topology_analysis();
        serde_wasm_bindgen::to_value(&analysis).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Human-readable topology report.
    pub fn topology_report(&mut self) -> String {
        self.inner.topology_report()
    }

    // ======================================================================
    // Memory
    // ======================================================================

    /// Memory subsystem statistics.
    pub fn memory_stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner.memory_stats();
        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Conversation context statistics: turn count, topic coherence, recent consciousness avg.
    ///
    /// Returns `{ turn_count, topic_coherence, recent_consciousness_avg }`.
    pub fn conversation_stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner.conversation_stats();
        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Consciousness validation experiments
    // ======================================================================

    /// Run an anesthesia analogue experiment. Returns AnesthesiaResult as JS object.
    ///
    /// Suppresses neuromodulators, observes consciousness collapse, then restores
    /// and observes recovery. Models clinical anesthesia (propofol/sevoflurane).
    pub fn anesthesia_experiment(
        &mut self,
        warmup_cycles: usize,
        suppression_cycles: usize,
        recovery_cycles: usize,
    ) -> Result<JsValue, JsError> {
        let result =
            self.inner
                .anesthesia_experiment(warmup_cycles, suppression_cycles, recovery_cycles);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Compute Perturbational Complexity Index (PCI). Returns PciResult as JS object.
    ///
    /// Based on Casali et al. (2013): perturb the network and measure spatiotemporal
    /// complexity of the response via Lempel-Ziv compression.
    pub fn measure_pci(
        &mut self,
        perturbation_magnitude: f32,
        observation_cycles: usize,
    ) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .measure_pci(perturbation_magnitude, observation_cycles);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run a split-brain experiment. Returns SplitBrainResult as JS object.
    ///
    /// Partitions the network into hemispheres and measures whether splitting
    /// reduces consciousness (IIT prediction).
    pub fn split_brain_experiment(
        &mut self,
        measurement_cycles: usize,
    ) -> Result<JsValue, JsError> {
        let result = self.inner.split_brain_experiment(measurement_cycles);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Find the consciousness collapse threshold. Returns CollapseThresholdResult as JS object.
    ///
    /// Systematically degrades the network and finds the point where consciousness
    /// collapses. IIT predicts a phase transition, not gradual decline.
    pub fn collapse_threshold_experiment(
        &mut self,
        steps: usize,
        cycles_per_step: usize,
    ) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .collapse_threshold_experiment(steps, cycles_per_step);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }
}
