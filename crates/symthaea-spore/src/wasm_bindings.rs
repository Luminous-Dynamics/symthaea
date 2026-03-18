//! wasm_bindgen exports for browser/JS consumption.
//!
//! Activated by the `wasm` feature flag.
//! Build with: `./build-wasm.sh` or see build-wasm.sh for manual steps.

use wasm_bindgen::prelude::*;

use crate::config::SporeConfig;
use crate::engine::SporeEngine as InnerEngine;
use crate::hardware_probe;
use crate::quickening::{self, InstallSubsystem, NarrationBank, QuickeningPhase, QuickeningState};

/// Probe hardware capabilities and generate NixOS configuration.
///
/// Takes browser-collected hardware data as a JS object (camelCase fields matching
/// `HardwareProfile`) and returns a `ProbeResult` containing the parsed profile
/// plus NixOS hardware configuration recommendations.
///
/// # JS usage
/// ```js
/// const result = probe_hardware({
///   cpuCores: navigator.hardwareConcurrency,
///   deviceMemoryGb: navigator.deviceMemory || 0,
///   hasWebgpu: !!navigator.gpu,
///   // ... etc
/// });
/// console.log(result.nixConfig.nixHardwareConfig);
/// ```
#[wasm_bindgen]
pub fn probe_hardware(js_data: JsValue) -> Result<JsValue, JsError> {
    console_error_panic_hook::set_once();
    let profile: hardware_probe::HardwareProfile = serde_wasm_bindgen::from_value(js_data)
        .map_err(|e| JsError::new(&format!("Invalid hardware data: {e}")))?;
    let result = hardware_probe::probe(profile);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}

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

// ======================================================================
// Quickening: Sovereign Birth orchestration
// ======================================================================

/// Sovereign Birth ceremony state machine.
///
/// Tracks progress through the Quickening phases, accumulating narration
/// and awakening Harmony tones as each subsystem installs.
#[wasm_bindgen]
pub struct QuickeningOrchestrator {
    state: QuickeningState,
}

#[wasm_bindgen]
impl QuickeningOrchestrator {
    /// Create a new Quickening orchestrator at the start of the ceremony.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            state: QuickeningState::new(),
        }
    }

    /// Get the current QuickeningState as a JSON object.
    ///
    /// Returns: `{ current_phase, phases_completed, harmonies_awakened,
    ///             consciousness_level, elapsed_seconds, narration_history }`
    pub fn quickening_state(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(&self.state).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Advance to the next phase. Takes a JSON object with:
    /// - `phase` (string): phase name (e.g. "TrustVerification", "StorePopulation")
    /// - `subsystem` (string, optional): for StorePopulation, the subsystem name
    /// - `elapsed` (number): elapsed seconds since ceremony start
    /// - `context` (object, optional): template variables for narration
    ///
    /// Returns a `PhaseAdvanceResult` with narration, haptic, tone, and state.
    pub fn quickening_advance(&mut self, phase_data: JsValue) -> Result<JsValue, JsError> {
        #[derive(serde::Deserialize)]
        struct AdvanceInput {
            phase: String,
            #[serde(default)]
            subsystem: Option<String>,
            #[serde(default)]
            elapsed: f32,
            #[serde(default)]
            context: std::collections::HashMap<String, String>,
        }

        let input: AdvanceInput = serde_wasm_bindgen::from_value(phase_data)
            .map_err(|e| JsError::new(&format!("Invalid phase data: {e}")))?;

        let mut phase = QuickeningPhase::from_name(&input.phase)
            .ok_or_else(|| JsError::new(&format!("Unknown phase: {}", input.phase)))?;

        // For StorePopulation, apply the subsystem
        if let Some(ref sub_name) = input.subsystem {
            if let Some(sub) = InstallSubsystem::from_name(sub_name) {
                phase = phase.with_subsystem(sub);
            }
        }

        let result = self.state.advance(&phase, &input.context, input.elapsed);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get narration for a specific phase without advancing state.
    ///
    /// `phase` is the phase name string, `context` is a JSON object of template variables.
    pub fn quickening_narrate(&self, phase: &str, context: JsValue) -> Result<String, JsError> {
        let ctx: std::collections::HashMap<String, String> =
            if context.is_null() || context.is_undefined() {
                std::collections::HashMap::new()
            } else {
                serde_wasm_bindgen::from_value(context)
                    .map_err(|e| JsError::new(&format!("Invalid context: {e}")))?
            };

        let qp = QuickeningPhase::from_name(phase)
            .ok_or_else(|| JsError::new(&format!("Unknown phase: {phase}")))?;

        let lines = NarrationBank::narrate(&qp, &ctx);
        Ok(lines.join("\n"))
    }

    /// Get the Harmony tones array as JSON.
    pub fn harmony_tones(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(&quickening::HARMONY_TONES)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Current progress fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        self.state.progress()
    }

    /// Whether the ceremony is complete.
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }
}
