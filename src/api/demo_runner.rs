//! Demo runner wrapping CognitiveLoopService for the WebSocket demo.
//!
//! Manages cycle execution and input state for the live demo.
//! When the `vision-manifold` feature is enabled, also provides an optional
//! vision pipeline that feeds camera frames through the cognitive loop.

use crate::api::ws::DemoCycleData;
use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[cfg(feature = "vision-manifold")]
use symthaea_vision_manifold::{CameraManifold, VisionConfig};

/// Demo runner that wraps a CognitiveLoopService.
pub struct DemoRunner {
    service: CognitiveLoopService,
    current_input: String,
    cycle_count: usize,
    /// When true, sensitive vector fields are zeroed before sending over WebSocket.
    /// Scalar aggregates (consciousness_level, mesh_health_score, etc.) are kept.
    pub redact_telemetry: bool,
    /// Optional vision manifold for visual input processing.
    #[cfg(feature = "vision-manifold")]
    vision: Option<CameraManifold>,
    /// Whether to run vision manifold each cycle.
    #[cfg(feature = "vision-manifold")]
    pub vision_enabled: bool,
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
            redact_telemetry: false,
            #[cfg(feature = "vision-manifold")]
            vision: None,
            #[cfg(feature = "vision-manifold")]
            vision_enabled: false,
        })
    }

    /// Enable the vision manifold with a mock camera source.
    #[cfg(feature = "vision-manifold")]
    pub fn enable_vision(&mut self, width: u32, height: u32) {
        let cfg = VisionConfig::default();
        self.vision = Some(CameraManifold::with_mock(cfg, width, height));
        self.vision_enabled = true;
    }

    /// Disable the vision manifold.
    #[cfg(feature = "vision-manifold")]
    pub fn disable_vision(&mut self) {
        self.vision_enabled = false;
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
    ///
    /// When vision is enabled, also ticks the camera manifold and feeds
    /// its state HV through `cycle_with_hv()` alongside the text cycle.
    pub fn run_cycle(&mut self) -> DemoCycleData {
        self.cycle_count += 1;

        // Optionally run vision manifold and feed through cognitive loop
        #[cfg(feature = "vision-manifold")]
        let vision_tel = if self.vision_enabled {
            if let Some(ref mut cam) = self.vision {
                if let Ok(tel) = cam.tick() {
                    // Feed the vision state into the cognitive loop's fast path
                    let state = cam.manifold().state().clone();
                    let _ = self.service.cycle_with_hv(&state);
                    let horizons = cam.manifold().evaluate_horizons();
                    Some((tel, horizons))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let result = self.service.cycle(&self.current_input);
        let m = &result.metadata;

        let mut data = DemoCycleData {
            cycle: self.cycle_count,
            prediction_error: result.prediction_error,
            consciousness_level: m.consciousness_level,
            narrative_self_psi: m.narrative_self_psi,
            valence: m.affective_valence,
            arousal: m.affective_arousal,
            mood_temperature: m.mood_temperature,
            thermodynamic_load: m.thermodynamic_load,
            moral_score: m.ethics.value_evaluator_score,
            coherence: m.harmonics.harmonic_field_coherence,
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
            mesh_health_score: m.mesh.mesh_health_score,
            mesh_peer_count: m.mesh.mesh_peer_count,
            mesh_bytes_sent: m.mesh.mesh_bytes_sent,
            mesh_bytes_received: m.mesh.mesh_bytes_received,
            mesh_compression_ratio: m.mesh.mesh_compression_ratio,
            mesh_bandwidth_budget: m.mesh.mesh_bandwidth_budget,
            mesh_packets_throttled: m.mesh.mesh_packets_throttled,
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
            harmony_coordinates: m.harmonics.harmony_coordinates,
            harmony_labels: vec![
                "Resonant Coherence".into(),
                "Pan-Sentient Flourishing".into(),
                "Integral Wisdom".into(),
                "Infinite Play".into(),
                "Universal Interconnectedness".into(),
                "Sacred Reciprocity".into(),
                "Evolutionary Progression".into(),
            ],
            moral_free_energy: m.ethics.moral_topo_free_energy,
            moral_kl_divergence: m.harmonics.moral_kl_divergence,
            moral_entropy: m.harmonics.moral_entropy,
            moral_surprise: m.harmonics.moral_surprise,
            moral_scenario_distribution: m.harmonics.moral_scenario_distribution,
            moral_prior_distribution: m.harmonics.moral_prior_distribution,
            moral_betti: [
                m.ethics.moral_topo_beta_0,
                m.ethics.moral_topo_beta_1,
                m.ethics.moral_topo_beta_2,
            ],
            moral_unity: m.ethics.moral_topo_unity,
            moral_completeness: m.ethics.moral_topo_completeness,
            moral_circularity: m.ethics.moral_topo_circularity,
            moral_dominant_harmony: m.ethics.moral_topo_dominant_harmony,
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
            // Moral anomaly detection
            moral_anomaly_score: m.ethics.moral_anomaly_score,
            moral_value_inversion: m.ethics.moral_value_inversion,
            moral_free_energy_spike: m.ethics.moral_free_energy_spike,
            // Vision manifold telemetry (defaults, overwritten below if active)
            vision_active: false,
            vision_prediction_error: 0.0,
            vision_coherence: 0.0,
            vision_attention_entropy: 0.0,
            vision_salient_patches: 0,
            vision_frame_sequence: 0,
            vision_horizon_errors: vec![],
            vision_encode_us: 0,
            vision_evolve_us: 0,
            vision_training_triggered: false,
        };

        // Populate vision telemetry if active
        #[cfg(feature = "vision-manifold")]
        if let Some((ref tel, ref horizons)) = vision_tel {
            data.vision_active = true;
            data.vision_prediction_error = tel.prediction_error;
            data.vision_coherence = tel.manifold_coherence;
            data.vision_attention_entropy = tel.attention_entropy;
            data.vision_salient_patches = tel.num_salient_patches;
            data.vision_frame_sequence = tel.frame_sequence;
            data.vision_horizon_errors = horizons.errors.clone();
            data.vision_encode_us = tel.encode_time_us;
            data.vision_evolve_us = tel.evolve_time_us;
            data.vision_training_triggered = tel.training_triggered;
        }

        // Redact sensitive vector fields if requested (Item 3: telemetry protection).
        // Keeps scalar aggregates (consciousness_level, mesh_health_score, etc.) intact.
        if self.redact_telemetry {
            data.thought_vector = vec![];
            data.neuromod_state_vector = vec![];
            data.bath_trajectory = vec![];
            data.moral_trajectory = vec![];
            data.bath_centroid = vec![];
        }

        data
    }

    /// Reset the service to initial state.
    pub fn reset(&mut self) {
        if let Ok(service) = CognitiveLoopService::new(CognitiveLoopConfig::default()) {
            self.service = service;
            self.cycle_count = 0;
            self.current_input = "consciousness emerges from integrated information".to_string();
        }
        #[cfg(feature = "vision-manifold")]
        if let Some(ref mut cam) = self.vision {
            cam.reset();
        }
    }
}
