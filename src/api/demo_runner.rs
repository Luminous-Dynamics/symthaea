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
            flow_state: m.attention.gwt_broadcast,
            cycle_time_us: result.cycle_time_us,
            surprise_triggered: m.surprise_triggered,
            gwt_broadcast: m.attention.gwt_broadcast,
            dream_insights: m.memory.dream_insights,
            reasoning_confidence: m.reasoning_confidence,
            resonance_frequency: m.resonance_frequency,
            input_text: self.current_input.clone(),
            thought_vector: result.thought_vector,
            // Phase 6: neuromodulator bath telemetry
            neuromod_state_vector: vec![
                m.neuromod.dopamine_effective,
                m.neuromod.noradrenaline_effective,
                m.neuromod.serotonin_effective,
                m.neuromod.acetylcholine_effective,
                m.neuromod.neuromod_gaba_effective,
                m.neuromod.neuromod_oxytocin_effective,
                m.neuromod.neuromod_glutamate_effective,
                m.neuromod.neuromod_adenosine_effective,
                m.neuromod.neuromod_endocannabinoid_effective,
            ],
            bath_entropy: m.neuromod.neuromod_bath_entropy,
            allostatic_load: m.neuromod.neuromod_allostatic_load,
            ei_ratio: m.neuromod.neuromod_ei_ratio,
            sleep_pressure: m.neuromod.neuromod_sleep_pressure,
            active_injection_count: m.neuromod.active_injection_count,
            attractor_detected: m.neuromod.neuromod_attractor_detected,
            // Mesh telemetry
            mesh_health_score: m.mesh_health_score,
            mesh_peer_count: m.mesh_peer_count,
            mesh_bytes_sent: m.mesh_bytes_sent,
            mesh_bytes_received: m.mesh_bytes_received,
            mesh_compression_ratio: m.mesh_compression_ratio,
            mesh_bandwidth_budget: m.mesh_bandwidth_budget,
            mesh_packets_throttled: m.mesh_packets_throttled,
            // Post-Phase 6: phase tracker visualization
            bath_centroid: self.service.bath_phase_tracker().centroid().to_vec(),
            bath_variance: self.service.bath_phase_tracker().variance().to_vec(),
            bath_trajectory: self
                .service
                .bath_phase_tracker()
                .trajectory(20)
                .into_iter()
                .map(|s| s.to_vec())
                .collect(),
            bath_projection_2d: {
                let c = self.service.bath_phase_tracker().centroid();
                // [DA+NE mean, 5-HT+GABA mean]
                vec![(c[0] + c[1]) / 2.0, (c[2] + c[4]) / 2.0]
            },
            bath_phase_label: self.service.bath_phase_label().to_string(),
            // Moral topology: conscience radar
            harmony_coordinates: m.harmony_coordinates,
            harmony_labels: vec![
                "Resonant Coherence".into(),
                "Pan-Sentient Flourishing".into(),
                "Integral Wisdom".into(),
                "Infinite Play".into(),
                "Universal Interconnectedness".into(),
                "Sacred Reciprocity".into(),
                "Evolutionary Progression".into(),
            ],
            moral_free_energy: m.moral_topo_free_energy,
            moral_kl_divergence: m.moral_kl_divergence,
            moral_entropy: m.moral_entropy,
            moral_surprise: m.moral_surprise,
            moral_scenario_distribution: m.moral_scenario_distribution,
            moral_prior_distribution: m.moral_prior_distribution,
            moral_betti: [m.moral_topo_beta_0, m.moral_topo_beta_1, m.moral_topo_beta_2],
            moral_unity: m.moral_topo_unity,
            moral_completeness: m.moral_topo_completeness,
            moral_circularity: m.moral_topo_circularity,
            moral_dominant_harmony: m.moral_topo_dominant_harmony,
            moral_persistent_features: self
                .service
                .ethics_engine()
                .moral_topology()
                .last_persistent_features()
                .to_vec(),
            moral_persistence_diagram: self
                .service
                .ethics_engine()
                .moral_topology()
                .persistence_diagram(),
            moral_trajectory: self
                .service
                .ethics_engine()
                .moral_topology()
                .trajectory(20)
                .into_iter()
                .map(|p| p.coordinates)
                .collect(),
            moral_drift: self
                .service
                .ethics_engine()
                .moral_topology()
                .moral_drift(20),
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
