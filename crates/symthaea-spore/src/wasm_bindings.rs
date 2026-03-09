//! wasm_bindgen exports for browser/JS consumption.
//!
//! Activated by the `wasm` feature flag.
//! Build with: `wasm-pack build --target web --features wasm`

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
}
