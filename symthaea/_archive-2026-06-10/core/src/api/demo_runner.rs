// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    /// Optional mesh daemon orchestrator (spawned edge consciousness kernel).
    #[cfg(feature = "mesh")]
    pub mesh_daemon: Option<crate::swarm::mesh::MeshDaemonOrchestrator>,
}

impl DemoRunner {
    /// Create a new demo runner with default configuration.
    pub fn new() -> anyhow::Result<Self> {
        let mut config = CognitiveLoopConfig::default();
        #[cfg(feature = "vision-manifold")]
        {
            config.enable_vision_manifold = true;
        }
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
            #[cfg(feature = "mesh")]
            mesh_daemon: None,
        })
    }

    /// Spawn the mesh daemon subprocess (spore-mesh-daemon).
    ///
    /// Gracefully degrades: if the binary is not found, logs a warning and continues.
    #[cfg(feature = "mesh")]
    pub async fn enable_mesh_daemon(&mut self) {
        use crate::swarm::mesh::{DaemonConfig, MeshDaemonOrchestrator};

        let config = DaemonConfig::default();
        match MeshDaemonOrchestrator::spawn(config).await {
            Ok(orch) => {
                tracing::info!("Mesh daemon spawned (spore-mesh-daemon, 15Hz edge kernel)");
                self.mesh_daemon = Some(orch);
            }
            Err(e) => {
                tracing::warn!(error = %e, "Mesh daemon not available — running without local mesh kernel");
            }
        }
    }

    /// Drain consciousness outputs from the mesh daemon and return as SwarmEvents.
    ///
    /// Called each tick from the WebSocket handler. Returns empty vec if no daemon.
    #[cfg(feature = "mesh")]
    pub fn drain_mesh_daemon(&mut self) -> Vec<crate::swarm::mesh::ConsciousnessOutput> {
        if let Some(ref mut daemon) = self.mesh_daemon {
            daemon.drain_outputs()
        } else {
            Vec::new()
        }
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

    /// Enable Iroh P2P swarm and spawn inbound accept loop.
    ///
    /// Call this once after construction, before the demo router is built.
    /// Silently degrades to local-only if attestation fails or the feature is disabled.
    #[cfg(all(feature = "identity", feature = "swarm"))]
    pub async fn enable_p2p(&mut self) {
        match self.service.enable_network_attestation().await {
            Ok(()) => {
                tracing::info!("Demo: Iroh P2P swarm active");
                if let Some(svc) = self.service.network_service().cloned() {
                    let node_id = svc.node_id();
                    if !node_id.is_empty() {
                        tracing::info!("Demo: Iroh node ID: {}", node_id);
                    }
                    match svc.create_ticket() {
                        Ok(ticket) => {
                            tracing::info!("Demo: bootstrap ticket (share with peers): {}", ticket)
                        }
                        Err(e) => tracing::debug!(error = %e, "Demo: ticket not available yet"),
                    }
                    tokio::spawn(svc.accept_connections());
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Demo: P2P disabled — running local-only");
            }
        }
    }

    /// Return the Iroh node's JSON-serialised EndpointAddr ticket (for peer bootstrap).
    ///
    /// Returns `None` when the `swarm` + `identity` features are not enabled or P2P
    /// has not been initialised. Returns `Some(Err(...))` if the node is initialised
    /// but ticket creation fails (e.g. endpoint not yet bound).
    #[cfg(all(feature = "identity", feature = "swarm"))]
    pub fn iroh_ticket(&self) -> Option<Result<String, String>> {
        self.service
            .network_service()
            .map(|svc| svc.create_ticket().map_err(|e| e.to_string()))
    }

    /// Get a cloneable reference to the NetworkService (for async broadcast from WS handler).
    pub fn network_service_arc(&self) -> Option<&std::sync::Arc<crate::swarm::NetworkService>> {
        self.service.network_service()
    }

    /// Get the swarm event sender for injecting mesh daemon consciousness updates.
    pub fn swarm_event_sender(
        &self,
    ) -> std::sync::mpsc::Sender<crate::cognitive_loop::managers::swarm_manager::SwarmEvent> {
        self.service.swarm_event_sender()
    }

    /// Node ID (hex-encoded EndpointId public key) — available without the full ticket.
    pub fn iroh_node_id(&self) -> String {
        self.service
            .network_service()
            .map(|svc| svc.node_id())
            .unwrap_or_default()
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

        // Inject camera frame into cognitive loop's internal VisionBridge.
        // The frame is processed during cycle() through the perception phase.
        #[cfg(feature = "vision-manifold")]
        if self.vision_enabled {
            if let Some(ref mut cam) = self.vision {
                let _ = cam.tick(); // advance mock camera for frame sequencing
            }
            let w = self.service.config().vision_frame_width;
            let h = self.service.config().vision_frame_height;
            let mock_frame = vec![128u8; (w * h) as usize];
            self.service.inject_vision_frame(mock_frame);
        }

        let result = self.service.cycle(&self.current_input);
        let m = &result.metadata;

        let mut data = DemoCycleData {
            cycle: self.cycle_count,
            prediction_error: result.prediction_error,
            consciousness_level: m.consciousness.consciousness_level,
            narrative_self_psi: m.narrative_self_psi,
            valence: m.embodied.affective_valence,
            arousal: m.embodied.affective_arousal,
            mood_temperature: m.embodied.mood_temperature,
            thermodynamic_load: m.temporal.thermodynamic_load,
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
            // Swarm P2P telemetry (Iroh)
            swarm_peers: self.service.swarm_connected_peers() as u32,
            network_mean_phi: self
                .service
                .network_service()
                .map(|svc| svc.network_mean_phi())
                .unwrap_or(0.0),
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
            moral_drift_alert: m.ethics.moral_drift_alert,
            moral_fragmentation_increase: m.ethics.moral_fragmentation_increase,
            moral_anomaly_response_applied: m.ethics.moral_anomaly_response_applied,
            moral_trajectory_convergence: m.ethics.moral_trajectory_convergence,
            moral_convergence_severity: m.ethics.moral_convergence_severity,
            moral_matched_hazard: m.ethics.moral_matched_hazard.clone(),
            moral_convergence_explanation: {
                let report = self.service.convergence_status();
                if report.severity > 0.0 || report.convergence_detected {
                    Some(self.service.convergence_explanation())
                } else {
                    None
                }
            },
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
            // Consciousness engine telemetry
            consciousness_weights: m.consciousness.consciousness_weights,
            consciousness_weight_variance: m.consciousness.consciousness_weight_variance,
            weight_convergence_state: m.consciousness.weight_convergence_state.clone(),
            // Structural Phi decomposition
            structural_micro_phi: m.structural.structural_micro_phi,
            structural_meso_phi: m.structural.structural_meso_phi,
            structural_macro_phi: m.structural.structural_macro_phi,
            structural_emergence_ratio: m.structural.structural_emergence_ratio,
            // Substrate
            substrate_feasibility: m.substrate_effective_feasibility,
            ..Default::default()
        };

        // Populate vision telemetry from CycleMetadata (internal VisionBridge path)
        #[cfg(feature = "vision-manifold")]
        if let Some(ref vt) = m.vision {
            data.vision_active = true;
            data.vision_prediction_error = vt.prediction_error;
            data.vision_coherence = vt.manifold_coherence;
            data.vision_attention_entropy = vt.attention_entropy;
            data.vision_salient_patches = vt.num_salient_patches;
            data.vision_frame_sequence = vt.frame_sequence;
            data.vision_training_triggered = vt.training_triggered;
            data.vision_encode_us = vt.encode_time_us;
            data.vision_evolve_us = vt.evolve_time_us;
        }
        #[cfg(feature = "vision-manifold")]
        if self.vision_enabled {
            if let Some(horizons) = self.service.vision_evaluate_horizons() {
                data.vision_horizon_errors = horizons.errors;
            }
        }

        // Populate therapeutic telemetry from CycleMetadata
        #[cfg(feature = "therapeutic")]
        {
            data.therapeutic_distress = m.therapeutic.therapeutic_client_distress;
            data.therapeutic_alliance = m.therapeutic.therapeutic_alliance;
            data.therapeutic_crisis_active = m.therapeutic.therapeutic_crisis_active;
            data.therapeutic_strategy = m.therapeutic.therapeutic_strategy.clone();
            data.therapeutic_narrative_coherence = m.therapeutic.therapeutic_narrative_coherence;
            data.therapeutic_clinical_severity = m.therapeutic.therapeutic_clinical_severity;
            data.therapeutic_serotonin_debt = m.therapeutic.therapeutic_serotonin_debt;
            data.therapeutic_dopamine_debt = m.therapeutic.therapeutic_dopamine_debt;
            data.therapeutic_dream_accuracy = m.therapeutic.therapeutic_dream_accuracy;
            data.therapeutic_resilience_ratio = m.therapeutic.therapeutic_resilience_ratio;
            data.therapeutic_rupture_count = m.therapeutic.therapeutic_rupture_count;
            data.therapeutic_last_rupture_type =
                m.therapeutic.therapeutic_last_rupture_type.clone();
            data.therapeutic_repair_rate = m.therapeutic.therapeutic_repair_rate;
            data.therapeutic_withdrawal_count = m.therapeutic.therapeutic_withdrawal_count;
            data.therapeutic_confrontation_count = m.therapeutic.therapeutic_confrontation_count;
            data.therapeutic_rdoc_profile = m.therapeutic.therapeutic_rdoc_profile;
            data.therapeutic_perpetuating_factors =
                m.therapeutic.therapeutic_perpetuating_factors.clone();
            data.therapeutic_protective_factors =
                m.therapeutic.therapeutic_protective_factors.clone();
            data.therapeutic_strategy_effectiveness =
                m.therapeutic.therapeutic_strategy_effectiveness.clone();
            data.therapeutic_temporal_coherence = m.therapeutic.therapeutic_temporal_coherence;

            // Shadow work telemetry (observability mode)
            data.shadow_total_pressure = m.therapeutic.shadow_total_pressure;
            data.shadow_fragment_count = m.therapeutic.shadow_fragment_count;
            data.shadow_peak_pressure = m.therapeutic.shadow_peak_pressure;
            data.shadow_surfacing_indicated = m.therapeutic.shadow_surfacing_indicated;
            data.shadow_pressure_trend = m.therapeutic.shadow_pressure_trend;
            data.shadow_to_narrative_ratio = m.therapeutic.shadow_to_narrative_ratio;
            data.shadow_dream_queue_depth = m.therapeutic.shadow_dream_queue_depth;
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

        // Clamp any NaN/Infinity to 0.0 before JSON serialization — serde_json
        // has no representation for non-finite floats and would error or produce null.
        data.sanitize_finite();

        data
    }

    /// Reset the service to initial state.
    pub fn reset(&mut self) {
        let mut config = CognitiveLoopConfig::default();
        #[cfg(feature = "vision-manifold")]
        {
            config.enable_vision_manifold = true;
        }
        if let Ok(service) = CognitiveLoopService::new(config) {
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
