//! Demo runner wrapping CognitiveLoopService for the WebSocket demo.
//!
//! Manages cycle execution and input state for the live demo.

use crate::api::ws::DemoCycleData;
use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Demo runner that wraps a CognitiveLoopService.
pub struct DemoRunner {
    service: CognitiveLoopService,
    current_input: String,
    cycle_count: usize,
}

impl DemoRunner {
    /// Create a new demo runner with default configuration.
    pub fn new() -> anyhow::Result<Self> {
        let config = CognitiveLoopConfig::default();
        let service = CognitiveLoopService::new(config)?;

        Ok(Self {
            service,
            current_input: "consciousness emerges from integrated information".to_string(),
            cycle_count: 0,
        })
    }

    /// Set the text input for the next cycle.
    pub fn set_input(&mut self, text: &str) {
        self.current_input = text.to_string();
    }

    /// Update thermodynamics state.
    pub fn update_thermodynamics(&mut self, load: f32) {
        self.service.thermodynamic_load = load;
        self.service.mood_temperature = 0.5 + (load * 1.5);
    }

    /// Run one cognitive cycle and return compact telemetry.
    pub fn run_cycle(&mut self) -> DemoCycleData {
        self.cycle_count += 1;

        let result = self.service.cycle(&self.current_input);
        let m = &result.metadata;

        DemoCycleData {
            cycle: self.cycle_count,
            prediction_error: result.prediction_error,
            consciousness_level: m.consciousness_level,
            narrative_self_psi: m.narrative_self_psi,
            valence: m.affective_valence,
            arousal: m.affective_arousal,
            mood_temperature: m.mood_temperature,
            thermodynamic_load: m.thermodynamic_load,
            moral_score: m.value_evaluator_score,
            coherence: m.harmonic_field_coherence,
            flow_state: m.gwt_broadcast,
            cycle_time_us: result.cycle_time_us,
            surprise_triggered: m.surprise_triggered,
            gwt_broadcast: m.gwt_broadcast,
            dream_insights: m.dream_insights,
            reasoning_confidence: m.reasoning_confidence,
            resonance_frequency: m.resonance_frequency,
            input_text: self.current_input.clone(),
            thought_vector: result.thought_vector,
        }
    }

    /// Reset the service to initial state.
    pub fn reset(&mut self) {
        if let Ok(service) = CognitiveLoopService::new(CognitiveLoopConfig::default()) {
            self.service = service;
            self.cycle_count = 0;
            self.current_input = "consciousness emerges from integrated information".to_string();
        }
    }
}
