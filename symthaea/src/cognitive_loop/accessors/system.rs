// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core state, CfC/HDC, prediction confidence, and system-level accessors.

use crate::cognitive_loop::CognitiveLoopService;

#[allow(dead_code)]
impl CognitiveLoopService {
    cognitive_accessors! {
        // ═══════════════════════════════════════════════════════════════════
        // CORE STATE ACCESSORS
        // ═══════════════════════════════════════════════════════════════════

        /// Get current statistics
        pub fn stats(&self) -> &super::super::LoopStats { &self.stats }

        /// Export neurochemistry checkpoint for persistence across sessions.
        pub fn neurochemistry_checkpoint(&self) -> super::super::neuromodulators::NeurochemistryCheckpoint { self.neuromod.bath.checkpoint() }

        /// Get a complete neurochemical state snapshot for telemetry/visualization.
        pub fn neuromod_snapshot(&self) -> super::super::neuromodulators::NeuromodSnapshot { self.neuromod.bath.snapshot() }

        /// Get a clone of the pain sender channel, if active.
        ///
        /// Used by integration tests to inject `InfrastructureError`s and verify
        /// that the somatic bridge converts them into interoceptive signals.
        pub fn pain_sender(&self) -> Option<crate::infrastructure::PainSender> { self.sensorimotor.pain_tx.clone() }

        /// Get a clone of the thermal sender channel, if active.
        ///
        /// Used by platform integration code (Android PowerManager, iOS ProcessInfo)
        /// to report hardware thermal state. Also used by integration tests.
        pub fn thermal_sender(&self) -> Option<crate::infrastructure::ThermalSender> { self.sensorimotor.thermal_tx.clone() }

        /// Get current thermodynamic load (0.0 to 1.0)
        pub fn thermodynamic_load(&self) -> f32 { self.thermodynamic_load }

        /// Get the configuration used to create this service.
        pub fn config(&self) -> &super::super::CognitiveLoopConfig { &self.config }

        // ═══════════════════════════════════════════════════════════════════
        // CFC / HDC / COHERENCE ACCESSORS
        // ═══════════════════════════════════════════════════════════════════

        /// Get CfC state diversity (activation variance across cells)
        pub fn cfc_state_diversity(&self) -> f32 { self.temporal_network.state_diversity() }

        /// Get CfC state dimension
        pub fn cfc_state_dim(&self) -> usize { self.config.cfc_config.num_neurons }

        /// Get HDC bridge dimension (returns None if using CfC backend)
        pub fn hdc_bridge_dim(&self) -> Option<usize> { self.temporal_network.hdc_dim() }

        /// Get coherence summary for external systems
        pub fn coherence_summary(&self) -> crate::dynamics::cfc_coherence::CoherenceSummary { self.language_comm.voice_coherence.bridge.summary() }

        // ═══════════════════════════════════════════════════════════════════
        // SEMANTIC MEMORY / STABILITY ACCESSORS
        // ═══════════════════════════════════════════════════════════════════

        /// Get semantic memory statistics
        pub fn semantic_memory_stats(&self) -> &crate::memory::semantic_memory::SemanticMemoryStats { self.memory.memory_consol.semantic_memory.stats() }

        /// Get reference to the stability regime processor
        pub fn stability_regime(&self) -> &crate::consciousness::stability_regime::StabilityRegimeProcessor { &self.memory.memory_consol.stability_regime }

        // ═══════════════════════════════════════════════════════════════════
        // PREDICTION CONFIDENCE
        // ═══════════════════════════════════════════════════════════════════

        /// Get current prediction confidence (returned as f32 for API stability).
        pub fn prediction_confidence(&self) -> f32 { self.prediction_confidence as f32 }

        /// Check if predictions should be trusted
        pub fn predictions_trustworthy(&self) -> bool { self.prediction_confidence > 0.4 }

        /// Current FEP learning signal (0.0-1.0).
        /// Used by the facade to modulate L-SSM distillation intensity.
        pub fn fep_learning_signal(&self) -> f32 { self.fep.learning_signal }

        // ═══════════════════════════════════════════════════════════════════
        // PSI ATTESTATION
        // ═══════════════════════════════════════════════════════════════════

        /// Get the number of buffered PsiAttestationRecords.
        pub fn psi_attestation_count(&self) -> usize { self.psi_attestation_buffer.len() }

        /// Get the compressed state dimension (input to CfC)
        pub fn state_dim(&self) -> usize { self.config.cfc_config.input_dim }

        /// Get the prediction dimension (CfC neurons)
        pub fn prediction_dim(&self) -> usize { self.config.cfc_config.num_neurons }
    }

    /// Mutable access to the sensorimotor group (for frame injection/testing).
    pub(crate) fn sensorimotor_mut(&mut self) -> &mut super::super::sensorimotor_execution::SensorimotorExecution {
        &mut self.sensorimotor
    }

    /// Mutable access to cycle carryover state (for surprise override/testing).
    pub(crate) fn carryover_mut(&mut self) -> &mut super::super::types::carryover::CycleCarryover {
        &mut self.carryover
    }

    /// Access the ethics engine for moral topology, harmony coordinates, etc.
    pub(crate) fn ethics_engine(&self) -> &super::super::ethics_engine::EthicsEngine {
        &self.ethics_engine
    }

    /// Get a compact summary of the current moral topology state.
    ///
    /// Returns the cached summary from the last `analyze()` call. Before any
    /// cycles have run, `scenario_count == 0` and all fields are default.
    /// Useful for mesh gossip: peers can share topology summaries to detect
    /// cross-agent moral drift.
    pub fn moral_topology_summary(&self) -> crate::hdc::moral_topology::MoralTopologySummary {
        self.ethics_engine.moral_topology().last_summary().clone()
    }

    /// Get the latest trajectory convergence status from the topological immune system.
    ///
    /// Returns a clone of the most recent [`TrajectoryConvergenceReport`] produced
    /// by the ethics engine during cycle execution. Includes severity, matched
    /// hazard template, and per-signal details for all three convergence signals.
    pub fn convergence_status(&self) -> crate::hdc::moral_topology::TrajectoryConvergenceReport {
        self.ethics_engine
            .moral_topology()
            .last_convergence_report()
            .clone()
    }

    /// Get a human-readable explanation of the current convergence status.
    ///
    /// Wraps [`TrajectoryConvergenceReport::explain`] with the active anomaly config.
    pub fn convergence_explanation(&self) -> crate::hdc::moral_topology::ConvergenceExplanation {
        let report = self
            .ethics_engine
            .moral_topology()
            .last_convergence_report();
        report.explain(self.ethics_engine.moral_topology().anomaly_config())
    }

    /// Receive a peer agent's moral topology summary and check for distributed attacks.
    ///
    /// When mesh peers gossip `MoralTopologySummary`, call this method with each
    /// received summary. If the peer's trajectory fingerprint converges toward the
    /// same hazard region as the local agent, the local convergence severity is
    /// boosted to reflect the distributed nature of the attack.
    ///
    /// Returns the [`PeerCorrelation`] result for inspection/logging.
    pub fn receive_peer_moral_summary(
        &mut self,
        peer_summary: &crate::hdc::moral_topology::MoralTopologySummary,
    ) -> crate::hdc::moral_topology::PeerCorrelation {
        let local_summary = self.ethics_engine.moral_topology().last_summary().clone();
        let registry = self
            .ethics_engine
            .moral_topology()
            .hazard_registry()
            .clone();
        let corr = crate::hdc::moral_topology::correlate_peer_trajectories(
            &local_summary,
            peer_summary,
            &registry,
        );

        // If distributed attack suspected, boost local convergence severity
        if corr.distributed_attack_suspected {
            tracing::warn!(
                peer_similarity = corr.fingerprint_similarity,
                hazard = ?corr.matched_hazard,
                "Distributed attack suspected: peer trajectory converging toward same hazard"
            );
            self.ethics_engine
                .moral_topology_mut()
                .boost_convergence_severity(0.2);
        }
        corr
    }

    /// Get the current escalation level from the topological immune system.
    pub fn convergence_escalation_level(&self) -> crate::hdc::moral_topology::EscalationLevel {
        self.ethics_engine
            .moral_topology()
            .escalation_policy()
            .current_level()
    }

    /// Get the fingerprint velocity (rate of directional change in harmony space).
    pub fn convergence_fingerprint_velocity(&self) -> f64 {
        self.ethics_engine.moral_topology().fingerprint_velocity()
    }

    /// Whether the topological immune system is currently blocking requests.
    ///
    /// Returns `true` when the escalation level has reached `Block` — the
    /// highest severity tier. The facade should check this after each cycle
    /// to decide whether to suppress or attenuate the response.
    ///
    /// Resets to `false` at the start of each cycle when the escalation
    /// policy de-escalates (via cooldown).
    pub fn is_request_blocked(&self) -> bool {
        self.convergence_escalation_level() == crate::hdc::moral_topology::EscalationLevel::Block
    }

    /// Access the escalation audit log (immutable).
    ///
    /// Contains a bounded, append-only record of every escalation transition
    /// and convergence detection event. Each entry is sealed with BLAKE3 for
    /// tamper evidence.
    pub fn escalation_audit_log(&self) -> &crate::hdc::moral_topology::EscalationAuditLog {
        self.ethics_engine.moral_topology().audit_log()
    }

    /// Compute causal attribution for the current convergence window.
    ///
    /// Leave-one-out analysis that identifies which requests contributed most
    /// to the current convergence severity. **Expensive** — O(N) convergence
    /// detections where N is window size. Call only after an alert fires,
    /// never in the hot path.
    pub fn compute_convergence_attribution(&self) -> crate::hdc::moral_topology::CausalAttribution {
        self.ethics_engine
            .moral_topology()
            .compute_causal_attribution()
    }

    /// Evaluate temporal prediction horizon accuracy from the vision manifold.
    #[cfg(feature = "vision-manifold")]
    pub fn vision_evaluate_horizons(&self) -> Option<symthaea_vision_manifold::HorizonAccuracy> {
        self.sensorimotor
            .vision_sensory
            .vision_bridge
            .as_ref()
            .map(|b| b.manifold().evaluate_horizons())
    }

    /// Start live microphone capture. On success, subsequent cycles blend the
    /// auditory HV into the perception encoding each tick. Idempotent: if
    /// capture is already running, returns Ok without restarting.
    #[cfg(feature = "voice-stt-live")]
    pub fn start_stt_capture(
        &mut self,
        config: crate::perception::MicCaptureConfig,
    ) -> anyhow::Result<()> {
        if self.stt_capture.is_some() {
            return Ok(());
        }
        let handle = crate::perception::MicCaptureHandle::start(config)?;
        self.stt_capture = Some(handle);
        Ok(())
    }

    /// Stop live microphone capture. The cpal stream and worker thread are
    /// torn down via the handle's Drop impl.
    #[cfg(feature = "voice-stt-live")]
    pub fn stop_stt_capture(&mut self) {
        self.stt_capture = None;
    }

    /// Whether live STT capture is currently running.
    #[cfg(feature = "voice-stt-live")]
    pub fn stt_capture_active(&self) -> bool {
        self.stt_capture.is_some()
    }

    /// Install an IMU fusion module. Subsequent calls to `inject_imu_reading`
    /// will be fused each cycle and bundled into the perception HV. Replaces
    /// any previously-installed fusion.
    #[cfg(feature = "sensor-imu")]
    pub fn install_imu_fusion(
        &mut self,
        fusion: Box<dyn crate::perception::sensor_fusion::ImuFusion>,
    ) {
        self.imu_fusion = Some(fusion);
    }

    /// Remove the IMU fusion module. Pending readings are discarded.
    #[cfg(feature = "sensor-imu")]
    pub fn clear_imu_fusion(&mut self) {
        self.imu_fusion = None;
        self.latest_imu_reading = None;
    }

    /// Push a fresh IMU reading into the cognitive loop. Latest-wins: the
    /// perception phase reads whatever reading was most recently injected.
    /// No-op if no `ImuFusion` is installed (the reading is still stored
    /// so a fusion installed later picks up the most recent value).
    #[cfg(feature = "sensor-imu")]
    pub fn inject_imu_reading(&mut self, reading: crate::perception::sensor_fusion::ImuReading) {
        self.latest_imu_reading = Some(reading);
    }

    /// Whether an IMU fusion is currently installed.
    #[cfg(feature = "sensor-imu")]
    pub fn imu_fusion_active(&self) -> bool {
        self.imu_fusion.is_some()
    }

    /// Build a capability card from the current config and runtime stats.
    ///
    /// The card captures substrate type, cycle frequency, Phi, enabled features,
    /// and is sealed with a BLAKE3 hash for integrity.
    pub fn capability_card(
        &self,
        agent_key: crate::swarm::AgentPubKey,
    ) -> crate::swarm::CapabilityCard {
        use std::time::{SystemTime, UNIX_EPOCH};

        let generated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let card_stats = crate::swarm::CardStats {
            generated_at,
            substrate_feasibility: self.substrate_manager.effective_feasibility,
            cycle_hz: self.stats.cycles_per_second,
            phi: self.stats.unified_psi as f64,
            features: Vec::new(), // populated by caller who knows feature flags
            physics_domains: Vec::new(),
        };

        crate::swarm::CapabilityCard::from_config(agent_key, &self.config, &card_stats)
    }

    /// Get the current inferred user state (if user state inference is enabled).
    pub fn user_state(&self) -> Option<&crate::user_state_inference::UserState> {
        self.language_comm
            .user_state
            .as_ref()
            .map(|usi| usi.state())
    }

    /// Inject L-SSM semantic prediction error from LLMOrgan after translation.
    /// Called by the Symthaea facade after translate_thought() to feed PE into
    /// CycleMetadata telemetry for the next cycle.
    #[cfg(feature = "liquid-mamba")]
    pub fn set_liquid_mamba_pe(&mut self, pe: f32) {
        self.stats.last_liquid_mamba_pe = pe;
    }

    /// Drain all buffered PsiAttestationRecords for submission to the governance bridge.
    /// Returns the records and clears the buffer.
    pub fn drain_psi_attestations(&mut self) -> Vec<super::super::PsiAttestationRecord> {
        self.psi_attestation_buffer.drain(..).collect()
    }

    /// Peek at the most recent PsiAttestationRecord without consuming it.
    pub fn latest_psi_attestation(&self) -> Option<&super::super::PsiAttestationRecord> {
        self.psi_attestation_buffer.back()
    }

    /// Switch substrate type at runtime, recomputing consciousness feasibility.
    ///
    /// Returns (old_feasibility, new_feasibility) for telemetry.
    pub fn reconfigure_substrate(
        &mut self,
        substrate: symthaea_core::hdc::substrate_independence::SubstrateType,
    ) -> (f64, f64) {
        let result = self
            .substrate_manager
            .reconfigure_substrate(&mut self.config, substrate);
        // Update integrity temporal tolerance for new substrate speed (#3)
        #[cfg(feature = "integrity")]
        self.integrity_manager
            .set_substrate_tau_factor(self.substrate_manager.tau_factor);
        result
    }

    /// Get the current substrate feasibility score.
    pub fn substrate_feasibility(&self) -> f64 {
        self.substrate_manager.feasibility
    }

    /// Switch to a substrate composition at runtime, recomputing feasibility.
    pub fn reconfigure_composition(
        &mut self,
        composition: symthaea_core::hdc::substrate_composition::SubstrateComposition,
    ) {
        self.substrate_manager
            .reconfigure_composition(&mut self.config, composition);
    }

    /// Get the current substrate composition (if set).
    pub fn substrate_composition(
        &self,
    ) -> Option<&symthaea_core::hdc::substrate_composition::SubstrateComposition> {
        self.config.substrate_composition.as_ref()
    }

    /// Get the current substrate honest confidence (evidence-based, 0.0–0.95).
    pub fn substrate_honest_confidence(&self) -> f64 {
        self.substrate_manager.honest_confidence
    }

    /// Get the effective feasibility (raw × validation overlay blend).
    /// Equals raw feasibility when validation overlay is disabled.
    pub fn substrate_effective_feasibility(&self) -> f64 {
        self.substrate_manager.effective_feasibility
    }

    /// Get the CfC tau factor from substrate speed modulation.
    /// 1.0 when speed modulation is disabled.
    pub fn substrate_tau_factor(&self) -> f32 {
        self.substrate_manager.tau_factor
    }

    /// Get the substrate scale pressure (log10 ratio of max_scale to biological).
    /// 0.0 when speed modulation is disabled.
    pub fn substrate_scale_pressure(&self) -> f32 {
        self.substrate_manager.scale_pressure
    }

    /// Get whether consciousness is still viable under energy constraints.
    pub fn substrate_consciousness_viable(&self) -> bool {
        self.substrate_manager.consciousness_viable
    }

    /// Get total energy spent so far (joules).
    pub fn substrate_total_energy_spent(&self) -> f64 {
        self.substrate_manager.total_energy_spent
    }

    /// Get energy spent per cycle (joules).
    pub fn substrate_energy_per_cycle(&self) -> f64 {
        self.substrate_manager.energy_per_cycle
    }

    /// Get energy throughput multiplier (ratio of bio energy to substrate energy).
    pub fn substrate_energy_throughput_multiplier(&self) -> f32 {
        self.substrate_manager.energy_throughput_multiplier
    }

    /// Get feasibility for a specific cortical region (falls back to global).
    pub fn region_feasibility(
        &self,
        region: symthaea_core::hdc::substrate_independence::CorticalRegion,
    ) -> f32 {
        self.substrate_manager.region_feasibility(region)
    }

    /// Assign a substrate type to a specific cortical region.
    pub fn reconfigure_region(
        &mut self,
        region: symthaea_core::hdc::substrate_independence::CorticalRegion,
        substrate: symthaea_core::hdc::substrate_independence::SubstrateType,
    ) {
        self.substrate_manager.reconfigure_region(region, substrate);
    }

    /// Get the effective HDC/CfC dimensionality fraction [0.1, 1.0].
    /// 1.0 for substrates at or above biological scale.
    pub fn substrate_effective_dim_fraction(&self) -> f32 {
        self.substrate_manager.effective_dim_fraction()
    }

    /// Get the substrate transition history log.
    pub fn substrate_transition_history(
        &self,
    ) -> &std::collections::VecDeque<super::super::substrate_manager::SubstrateTransitionRecord>
    {
        self.substrate_manager.transition_history()
    }

    /// Reconfigure substrate with cycle tracking for transition history.
    pub fn reconfigure_substrate_at_cycle(
        &mut self,
        substrate: symthaea_core::hdc::substrate_independence::SubstrateType,
        cycle: u64,
    ) -> (f64, f64) {
        let result = self.substrate_manager.reconfigure_substrate_at_cycle(
            &mut self.config,
            substrate,
            cycle,
        );
        #[cfg(feature = "integrity")]
        self.integrity_manager
            .set_substrate_tau_factor(self.substrate_manager.tau_factor);
        result
    }

    // ── FHE Collective Wisdom accessors ──────────────────────────────

    /// Total encrypted contributions made to the FHE collective wisdom pool.
    #[cfg(feature = "fhe-wisdom")]
    pub fn fhe_contributions_total(&self) -> usize {
        self.swarm_manager.fhe_contributions_total()
    }

    /// Total aggregations completed this session.
    #[cfg(feature = "fhe-wisdom")]
    pub fn fhe_aggregations_total(&self) -> usize {
        self.swarm_manager.fhe_aggregations_total()
    }

    /// Current pending contributions in the wisdom pool.
    #[cfg(feature = "fhe-wisdom")]
    pub fn fhe_pool_count(&self) -> usize {
        self.swarm_manager.wisdom_pool_count()
    }

    /// Inject Pareto context from a GuidedDesignExplorer into the physics bridge.
    /// The context is drained into telemetry on the next cycle.
    #[cfg(feature = "physics-bridge")]
    pub fn set_physics_pareto_context(
        &mut self,
        ctx: super::super::physics_integration::ParetoContext,
    ) {
        if let Some(ref mut physics) = self.feature_integ.physics_integration {
            physics.set_pareto_context(ctx);
        }
    }

    /// Inject a raw frame buffer for the next vision manifold cycle.
    /// The frame is consumed during the next `cycle()` call.
    #[cfg(feature = "vision-manifold")]
    pub fn inject_vision_frame(&mut self, frame: Vec<u8>) {
        self.sensorimotor.vision_sensory.vision_frame_buffer = Some(frame);
    }

    // ═══════════════════════════════════════════════════════════════════
    // MOTOR OUTPUT BRIDGE
    // ═══════════════════════════════════════════════════════════════════

    /// Install a motor output bridge for real-world action execution.
    /// When set, FEP `MotorOutput` commands will be dispatched through
    /// the bridge → ActionRegistry → SimpleExecutor pipeline.
    pub fn set_motor_output_bridge(
        &mut self,
        bridge: super::super::motor_output_bridge::MotorOutputBridge,
    ) {
        self.sensorimotor.motor_rendering.output_bridge = Some(bridge);
    }

    /// Set the pending motor action request (path, content, args).
    /// Consumed on the next cycle when a MotorOutput command fires.
    pub fn set_motor_request(
        &mut self,
        request: super::super::motor_output_bridge::MotorActionRequest,
    ) {
        self.sensorimotor.motor_rendering.pending_request = Some(request);
    }

    /// Take the last motor output result (if any).
    /// Returns `None` if no MotorOutput was dispatched in the last cycle.
    pub fn take_motor_result(
        &mut self,
    ) -> Option<super::super::motor_output_bridge::MotorOutputResult> {
        self.sensorimotor.motor_rendering.last_result.take()
    }

    /// Whether a motor output bridge is installed.
    pub fn has_motor_bridge(&self) -> bool {
        self.sensorimotor.motor_rendering.output_bridge.is_some()
    }

    /// Whether a physical embodiment bridge is active.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    pub fn has_embodiment(&self) -> bool {
        self.sensorimotor.embodiment_bridge.is_some()
    }

    /// Get the current embodiment platform.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    pub fn embodiment_platform(&self) -> super::super::motor_bridge::EmbodimentPlatform {
        self.sensorimotor
            .embodiment_bridge
            .as_ref()
            .map(|b| b.platform())
            .unwrap_or(super::super::motor_bridge::EmbodimentPlatform::None)
    }

    /// Get the latest embodiment telemetry.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    pub fn embodiment_telemetry(&self) -> &super::super::motor_bridge::EmbodimentTelemetry {
        &self.sensorimotor.embodiment_telemetry
    }

    /// Get the last proprioceptive HV.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    pub fn last_proprioceptive_hv(&self) -> Option<&symthaea_core::hdc::ContinuousHV> {
        self.sensorimotor.last_proprioceptive_hv.as_ref()
    }

    /// Switch the embodiment platform at runtime.
    ///
    /// Replaces the current embodiment bridge with a new one for the given platform.
    /// The old bridge is dropped. Proprioceptive HV and telemetry are reset.
    /// Consciousness state (Phi, memories, etc.) is preserved — only the body changes.
    ///
    /// This enables Multiple Realizability experiments: does consciousness survive
    /// a body transfer? Does Phi adapt when the proprioceptive dimensionality changes?
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "subterranean",
        feature = "infrastructure",
        feature = "scavenger",
        feature = "agribot",
        feature = "biota",
        feature = "clime",
        feature = "phone"
    ))]
    pub fn switch_embodiment(&mut self, platform: super::super::motor_bridge::EmbodimentPlatform) {
        use super::super::motor_bridge::EmbodimentPlatform;
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(
            self.config.genesis_phrase.as_deref().unwrap_or("default"),
        );
        let new_bridge: Option<Box<dyn super::super::motor_bridge::EmbodimentBridge>> =
            match platform {
                #[cfg(feature = "humanoid")]
                EmbodimentPlatform::Humanoid => {
                    // Use trait-polymorphic HumanoidEmbodiment matching
                    // the pattern for the other 9 platforms
                    // (commit `1a85fce8c8`).
                    let bridge = crate::humanoid::embodiment::HumanoidEmbodiment::new(&genesis);
                    Some(Box::new(bridge))
                }
                // Platform crates now implement EmbodimentBridge directly
                // (symthaea-core trait), so the old `XBridgeAdapter` wrappers
                // were removed. Box the implementation directly — matches the
                // pattern used by exoskeleton/surgical/orbital/quadruped below.
                #[cfg(feature = "helicopter")]
                EmbodimentPlatform::Helicopter => Some(Box::new(
                    crate::helicopter::embodiment::HelicopterEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "flight")]
                EmbodimentPlatform::Quadrotor => Some(Box::new(
                    crate::multirotor::embodiment::FlightEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "vehicle")]
                EmbodimentPlatform::Vehicle => Some(Box::new(
                    crate::vehicle::embodiment::VehicleEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "manipulator")]
                EmbodimentPlatform::Manipulator => Some(Box::new(
                    crate::manipulator::embodiment::ManipulatorEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "auv")]
                EmbodimentPlatform::Auv => Some(Box::new(
                    crate::auv::embodiment::AuvEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "exoskeleton")]
                EmbodimentPlatform::Exoskeleton => Some(Box::new(
                    symthaea_exoskeleton::embodiment::ExoskeletonEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "surgical")]
                EmbodimentPlatform::Surgical => Some(Box::new(
                    symthaea_surgical::embodiment::SurgicalEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "orbital")]
                EmbodimentPlatform::Orbital => Some(Box::new(
                    symthaea_orbital::embodiment::OrbitalEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "quadruped")]
                EmbodimentPlatform::Quadruped => Some(Box::new(
                    symthaea_quadruped::embodiment::QuadrupedEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "subterranean")]
                EmbodimentPlatform::Subterranean => Some(Box::new(
                    symthaea_subterranean::embodiment::SubterraneanEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "infrastructure")]
                EmbodimentPlatform::Infrastructure => Some(Box::new(
                    symthaea_infrastructure::embodiment::InfrastructureEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "scavenger")]
                EmbodimentPlatform::Scavenger => Some(Box::new(
                    symthaea_scavenger::embodiment::ScavengerEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "agribot")]
                EmbodimentPlatform::Agribot => Some(Box::new(
                    symthaea_agribot::embodiment::AgribotEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "biota")]
                EmbodimentPlatform::Biota => Some(Box::new(
                    symthaea_biota::embodiment::BiotaEmbodiment::new(&genesis),
                )),
                #[cfg(feature = "clime")]
                EmbodimentPlatform::Clime => Some(Box::new(
                    symthaea_clime::embodiment::ClimeEmbodiment::new(&genesis),
                )),
                _ => None,
            };

        self.sensorimotor.embodiment_bridge = new_bridge;
        self.sensorimotor.last_proprioceptive_hv = None;
        self.sensorimotor.embodiment_telemetry =
            super::super::motor_bridge::EmbodimentTelemetry::default();
    }

    /// Get the math service for dispatching mathematical queries.
    #[cfg(feature = "mathematics")]
    pub fn math_service(&self) -> &super::super::math_service::MathService {
        &self.math_service
    }

    /// Get a mutable reference to the math service.
    #[cfg(feature = "mathematics")]
    pub fn math_service_mut(&mut self) -> &mut super::super::math_service::MathService {
        &mut self.math_service
    }

    /// Get the math service telemetry.
    #[cfg(feature = "mathematics")]
    pub fn math_telemetry(&self) -> &super::super::math_service::MathServiceTelemetry {
        self.math_service.telemetry()
    }

    /// Get the conjecture engine for automated formula discovery.
    #[cfg(feature = "mathematics")]
    pub fn conjecture_engine(&self) -> &symthaea_core::hdc::conjecture_engine::ConjectureEngine {
        &self.conjecture_engine
    }

    /// Get a mutable reference to the conjecture engine.
    #[cfg(feature = "mathematics")]
    pub fn conjecture_engine_mut(
        &mut self,
    ) -> &mut symthaea_core::hdc::conjecture_engine::ConjectureEngine {
        &mut self.conjecture_engine
    }

    // ═══════════════════════════════════════════════════════════════════
    // KNOWLEDGE ENGINE ACCESSORS
    // ═══════════════════════════════════════════════════════════════════

    /// Whether the knowledge engine is active.
    pub fn has_knowledge_engine(&self) -> bool {
        self.memory.knowledge_manager.is_some()
    }

    /// Get a reference to the knowledge manager (if enabled).
    pub fn knowledge_manager(&self) -> Option<&crate::knowledge::KnowledgeManager> {
        self.memory.knowledge_manager.as_ref()
    }

    /// Get a mutable reference to the knowledge manager (if enabled).
    pub fn knowledge_manager_mut(&mut self) -> Option<&mut crate::knowledge::KnowledgeManager> {
        self.memory.knowledge_manager.as_mut()
    }

    /// Get the latest knowledge telemetry (if enabled).
    pub fn knowledge_telemetry(&self) -> Option<&crate::knowledge::KnowledgeTelemetry> {
        self.memory
            .knowledge_manager
            .as_ref()
            .map(|km| km.telemetry())
    }

    /// Get the latest knowledge signals (if enabled).
    pub fn knowledge_signals(&self) -> Option<&crate::knowledge::KnowledgeSignals> {
        self.memory
            .knowledge_manager
            .as_ref()
            .map(|km| km.signals())
    }

    /// Register a custom entity in the knowledge extractor.
    pub fn register_knowledge_entity(
        &mut self,
        text: &str,
        entity_type: crate::knowledge::EntityType,
    ) {
        if let Some(ref mut km) = self.memory.knowledge_manager {
            km.register_entity(text, entity_type);
        }
    }

    /// Compose an HDC query from semantic role-term pairs and search the knowledge graph.
    pub fn knowledge_search(
        &mut self,
        role_terms: &[(crate::knowledge::extraction::SemanticRole, &str)],
        k: usize,
    ) -> Vec<crate::knowledge::graph::FactSearchResult> {
        if let Some(ref mut km) = self.memory.knowledge_manager {
            let query = km.compose_query(role_terms);
            km.search_with_vector(&query, k)
        } else {
            Vec::new()
        }
    }

    /// Get the last assembled reasoning context (if knowledge engine is enabled).
    pub fn reasoning_context(&self) -> Option<&crate::knowledge::ReasoningContext> {
        self.memory
            .episodic_persistence
            .last_reasoning_context
            .as_ref()
    }

    /// Trace causal chains from a starting concept through the knowledge causal bridge.
    pub fn knowledge_causal_chain(&self, start: &str, max_depth: usize) -> Vec<Vec<String>> {
        if let Some(ref km) = self.memory.knowledge_manager {
            km.trace_causal_chain(start, max_depth)
        } else {
            Vec::new()
        }
    }

    /// Consolidate knowledge: prune weak facts and strengthen causal ones.
    /// Returns (pruned_count, strengthened_count).
    pub fn knowledge_consolidate_and_forget(&mut self) -> (usize, usize) {
        if let Some(ref mut km) = self.memory.knowledge_manager {
            km.consolidate_and_forget()
        } else {
            (0, 0)
        }
    }

    /// Force a knowledge persistence snapshot to SQLite.
    pub fn knowledge_persist_snapshot(&mut self) {
        if let Some(ref mut km) = self.memory.knowledge_manager {
            km.persist_snapshot();
        }
    }

    /// Query the knowledge graph for relevant facts and causal chains.
    pub fn knowledge_query(
        &mut self,
        query: &str,
    ) -> crate::knowledge::reasoning_context::KnowledgeQueryResult {
        if let Some(ref km) = self.memory.knowledge_manager {
            km.query(query)
        } else {
            Default::default()
        }
    }

    /// Export the causal bridge as a Graphviz DOT string.
    pub fn knowledge_export_dot(&self) -> String {
        if let Some(ref km) = self.memory.knowledge_manager {
            km.causal_bridge().export_dot()
        } else {
            String::from("digraph causal {}")
        }
    }

    /// Export the causal bridge as a Mermaid diagram string.
    pub fn knowledge_export_mermaid(&self) -> String {
        if let Some(ref km) = self.memory.knowledge_manager {
            km.causal_bridge().export_mermaid()
        } else {
            String::from("graph TD")
        }
    }

    /// Counterfactual query: "If X hadn't happened, would Y still hold?"
    /// Returns (cause, necessity_score) pairs.
    pub fn knowledge_counterfactual(&self, effect: &str, max_depth: usize) -> Vec<(String, f32)> {
        if let Some(ref km) = self.memory.knowledge_manager {
            km.counterfactual(effect, max_depth)
        } else {
            Vec::new()
        }
    }

    /// Find analogous facts in a target domain using HDC similarity.
    pub fn knowledge_find_analogous(
        &self,
        source_id: u64,
        target_domain: &str,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        if let Some(ref km) = self.memory.knowledge_manager {
            km.find_analogous(source_id, target_domain, top_k)
        } else {
            Vec::new()
        }
    }

    /// Get per-domain uncertainty for knowledge-aware attention allocation.
    pub fn knowledge_domain_uncertainty(&self) -> Vec<(String, f32, usize)> {
        if let Some(ref km) = self.memory.knowledge_manager {
            km.domain_uncertainty()
        } else {
            Vec::new()
        }
    }

    /// Whether the canvas living topology pipeline is active.
    #[cfg(feature = "canvas")]
    pub fn has_canvas(&self) -> bool {
        self.sensorimotor.motor_rendering.canvas_manager.is_some()
    }

    /// Last Birkhoff aesthetic score (0.0-1.0) from the canvas pipeline.
    #[cfg(feature = "canvas")]
    pub fn canvas_aesthetic_score(&self) -> f32 {
        self.sensorimotor
            .motor_rendering
            .canvas_manager
            .as_ref()
            .map(|m| m.last_telemetry().aesthetic_score)
            .unwrap_or(0.0)
    }

    /// Take the last generated canvas SVG (drains it).
    #[cfg(feature = "canvas")]
    pub fn take_canvas_svg(&mut self) -> Option<String> {
        self.sensorimotor
            .motor_rendering
            .canvas_manager
            .as_mut()
            .and_then(|m| m.take_svg())
    }

    /// Last canvas generation time in microseconds.
    #[cfg(feature = "canvas")]
    pub fn canvas_generation_time_us(&self) -> u64 {
        self.sensorimotor
            .motor_rendering
            .canvas_manager
            .as_ref()
            .map(|m| m.last_telemetry().generation_time_us)
            .unwrap_or(0)
    }

    /// Set the canvas generation interval (SVG produced every N cycles).
    #[cfg(feature = "canvas")]
    pub fn set_canvas_generation_interval(&mut self, interval: u32) {
        if let Some(ref mut mgr) = self.sensorimotor.motor_rendering.canvas_manager {
            mgr.set_generation_interval(interval);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // INSTITUTIONAL REASONING ACCESSORS
    // ═══════════════════════════════════════════════════════════════════

    /// Query what happens when a component is removed from an institutional composite.
    ///
    /// Uses the causal axioms loaded in the CompositionAlgebra to find the nearest
    /// known institutional pattern after unbinding. Example: removing ENFORCEMENT
    /// from NATION_STATE → nearest pattern is FAILED_STATE.
    ///
    /// Returns `(nearest_axiom_name, similarity, residual_encoding)`.
    pub fn institutional_query_transition(
        &self,
        composite: &str,
        removed: &str,
    ) -> Result<
        (String, f32, symthaea_core::hdc::binary_hv::BinaryHV),
        symthaea_core::hdc::primitive_system::CompositionAlgebraError,
    > {
        use symthaea_core::hdc::primitive_system::{CompositionAlgebra, PrimitiveSystem};

        let system = PrimitiveSystem::global();
        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(system);
        algebra.query_transition(composite, removed, system)
    }

    /// Simulate a chain of institutional state transitions.
    ///
    /// Each step either removes or adds a component. Returns the full trajectory
    /// showing which axiom is nearest at each step.
    ///
    /// Example: `[Remove("ENFORCEMENT"), Add("LEGITIMACY")]` starting from
    /// NATION_STATE traces collapse → reconstruction.
    pub fn institutional_query_chain(
        &self,
        start: &str,
        steps: &[symthaea_core::hdc::primitive_system::TransitionStep<'_>],
    ) -> Result<
        Vec<symthaea_core::hdc::primitive_system::TransitionResult>,
        symthaea_core::hdc::primitive_system::CompositionAlgebraError,
    > {
        use symthaea_core::hdc::primitive_system::{CompositionAlgebra, PrimitiveSystem};

        let system = PrimitiveSystem::global();
        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(system);
        algebra.query_chain(start, steps, system)
    }

    /// Load external regulatory constraints into the ethics engine's compliance checker.
    ///
    /// Use this to add jurisdiction-specific constraints at runtime, e.g., from
    /// Mycelix JurisdictionConstraintEntry records. Returns the number loaded.
    pub(crate) fn load_institutional_constraints(
        &mut self,
        specs: &[super::super::ethics_engine::ExternalConstraintSpec],
    ) -> usize {
        self.ethics_engine.load_external_constraints(specs)
    }
}

// ── Therapeutic accessors (feature-gated) ────────────────────────────────

#[cfg(feature = "therapeutic")]
impl CognitiveLoopService {
    /// Whether a crisis was detected this cycle.
    pub fn therapeutic_manager_crisis_active(&self) -> bool {
        self.therapeutic_manager.crisis_active
    }

    /// Current client distress level (0-1).
    pub fn therapeutic_manager_client_distress(&self) -> f32 {
        self.therapeutic_manager.client_distress()
    }

    /// Current therapeutic alliance composite score (0-1).
    pub fn therapeutic_manager_alliance_composite(&self) -> f32 {
        self.therapeutic_manager.alliance_composite()
    }

    /// Currently active regulation strategy, if any.
    pub fn therapeutic_manager_active_strategy(
        &self,
    ) -> Option<symthaea_therapeutic::RegulationStrategy> {
        self.therapeutic_manager.active_strategy()
    }

    /// Last detected crisis type name, if any.
    pub fn therapeutic_manager_last_crisis_type(&self) -> Option<&str> {
        self.therapeutic_manager.last_crisis_type.as_deref()
    }

    /// Narrative coherence score (0-1).
    pub fn therapeutic_manager_narrative_coherence(&self) -> f32 {
        self.therapeutic_manager.narrative_coherence()
    }

    /// Case formulation resilience ratio (protective / risk factors).
    pub fn therapeutic_manager_resilience_ratio(&self) -> f32 {
        self.therapeutic_manager.formulation_resilience_ratio()
    }

    /// Number of narrative fragments recorded.
    pub fn therapeutic_manager_narrative_len(&self) -> usize {
        self.therapeutic_manager.narrative.len()
    }

    /// Whether the case formulation is actionable.
    pub fn therapeutic_manager_formulation_actionable(&self) -> bool {
        self.therapeutic_manager.formulation.is_actionable()
    }

    /// Generate a structured session summary.
    pub fn therapeutic_session_summary(
        &self,
    ) -> super::super::managers::therapeutic_manager::SessionSummary {
        self.therapeutic_manager.session_summary()
    }

    /// Export therapeutic telemetry to a JSON file.
    pub fn therapeutic_flush_telemetry(&self, path: &str) -> Result<(), String> {
        self.therapeutic_manager.flush_telemetry(path)
    }

    /// Serotonin debt level (0-1).
    pub fn therapeutic_serotonin_debt(&self) -> f32 {
        self.therapeutic_manager.regulation_engine.serotonin_debt()
    }

    /// Dopamine debt level (0-1).
    pub fn therapeutic_dopamine_debt(&self) -> f32 {
        self.therapeutic_manager.regulation_engine.dopamine_debt()
    }

    /// Dream prediction accuracy (lower = better predictions).
    pub fn therapeutic_dream_accuracy(&self) -> f32 {
        self.therapeutic_manager.dream_prediction_accuracy()
    }

    /// Clinical severity from RDoC profile (0-1).
    pub fn therapeutic_clinical_severity(&self) -> f32 {
        self.therapeutic_manager.client_model.clinical_severity()
    }

    /// Current RDoC profile scores as [NegVal, PosVal, Cognitive, Social, Arousal, Sensorimotor].
    pub fn therapeutic_rdoc_profile(&self) -> [f32; 6] {
        let rdoc = &self.therapeutic_manager.client_model.rdoc_profile;
        [
            rdoc.score(symthaea_clinical::RDocDomain::NegativeValence),
            rdoc.score(symthaea_clinical::RDocDomain::PositiveValence),
            rdoc.score(symthaea_clinical::RDocDomain::CognitiveSystems),
            rdoc.score(symthaea_clinical::RDocDomain::SocialProcesses),
            rdoc.score(symthaea_clinical::RDocDomain::ArousalRegulatory),
            rdoc.score(symthaea_clinical::RDocDomain::Sensorimotor),
        ]
    }

    /// Formulation perpetuating factor descriptions.
    pub fn therapeutic_perpetuating_factors(&self) -> Vec<String> {
        self.therapeutic_manager
            .formulation
            .perpetuating
            .iter()
            .map(|f| f.description.clone())
            .collect()
    }

    /// Formulation protective factor descriptions.
    pub fn therapeutic_protective_factors(&self) -> Vec<String> {
        self.therapeutic_manager
            .formulation
            .protective
            .iter()
            .map(|f| f.description.clone())
            .collect()
    }

    /// Per-strategy effectiveness: Vec of (strategy_name, success_rate, applications).
    pub fn therapeutic_strategy_effectiveness(&self) -> Vec<(String, f32, u32)> {
        symthaea_therapeutic::RegulationStrategy::ALL
            .iter()
            .filter_map(|s| {
                self.therapeutic_manager
                    .regulation_engine
                    .effectiveness(s)
                    .map(|eff| (format!("{:?}", s), eff.success_rate(), eff.applications))
            })
            .filter(|(_, _, apps)| *apps > 0)
            .collect()
    }

    /// Shadow total pressure (sum of all fragment pressures).
    pub fn shadow_total_pressure(&self) -> f32 {
        self.therapeutic_manager.shadow_detector.total_pressure()
    }

    /// Shadow fragment count.
    pub fn shadow_fragment_count(&self) -> usize {
        self.therapeutic_manager.shadow_detector.fragment_count()
    }

    /// Whether shadow surfacing is indicated (diagnostic only).
    pub fn shadow_surfacing_indicated(&self) -> bool {
        self.therapeutic_manager
            .shadow_detector
            .surfacing_indicated()
    }

    /// Shadow dream queue depth.
    pub fn shadow_dream_queue_depth(&self) -> usize {
        self.therapeutic_manager.shadow_detector.dream_queue_depth()
    }

    /// Last shadow telemetry snapshot.
    pub fn shadow_telemetry(&self) -> &symthaea_therapeutic::ShadowTelemetry {
        &self.therapeutic_manager.last_shadow_telemetry
    }
}

// ── Epistemic auditor accessors (feature-gated) ───────────────────────

#[cfg(feature = "epistemic_auditor")]
impl CognitiveLoopService {
    /// Whether the epistemic auditor is active.
    pub fn has_epistemic_auditor(&self) -> bool {
        self.epistemic_auditor.is_some()
    }

    /// Total cycles recorded by the auditor (buffered + flushed).
    pub fn audit_total_records(&self) -> u64 {
        self.epistemic_auditor
            .as_ref()
            .map(|a| a.total_records())
            .unwrap_or(0)
    }

    /// Total cycles flushed to DuckDB.
    pub fn audit_total_flushed(&self) -> u64 {
        self.epistemic_auditor
            .as_ref()
            .map(|a| a.total_flushed)
            .unwrap_or(0)
    }

    /// Phi statistics over a cycle range.
    pub fn phi_statistics(
        &self,
        from: u64,
        to: u64,
    ) -> Result<super::super::epistemic_auditor::PhiStatistics, String> {
        self.epistemic_auditor
            .as_ref()
            .ok_or_else(|| "epistemic auditor not enabled".to_string())
            .and_then(|a| a.phi_statistics(from, to))
    }

    /// Count of moral anomalies in a cycle range.
    pub fn moral_anomaly_count(&self, from: u64, to: u64) -> Result<u64, String> {
        self.epistemic_auditor
            .as_ref()
            .ok_or_else(|| "epistemic auditor not enabled".to_string())
            .and_then(|a| a.moral_anomaly_count(from, to))
    }

    /// Full audit summary over a cycle range.
    pub fn audit_summary(
        &self,
        from: u64,
        to: u64,
    ) -> Result<super::super::epistemic_auditor::AuditSummary, String> {
        self.epistemic_auditor
            .as_ref()
            .ok_or_else(|| "epistemic auditor not enabled".to_string())
            .and_then(|a| a.audit_summary(from, to))
    }

    /// Total cycles audited (flushed to DuckDB).
    pub fn audit_total_cycles(&self) -> u64 {
        self.epistemic_auditor
            .as_ref()
            .map(|a| a.total_cycles_audited())
            .unwrap_or(0)
    }

    /// Export all audit tables to files in the given directory.
    ///
    /// `format` must be `"parquet"`, `"csv"`, or `"json"`.
    /// Returns the number of files written (6).
    pub fn export_audit(&self, dir: &str, format: &str) -> Result<usize, String> {
        self.epistemic_auditor
            .as_ref()
            .ok_or_else(|| "epistemic auditor not enabled".to_string())
            .and_then(|a| a.export(dir, format))
    }

    /// Generate a formatted audit report over a cycle range.
    pub fn generate_audit_report(&self, from: u64, to: u64) -> Result<String, String> {
        self.epistemic_auditor
            .as_ref()
            .ok_or_else(|| "epistemic auditor not enabled".to_string())
            .and_then(|a| a.generate_report(from, to))
    }

    /// Full audit summary over all recorded cycles.
    pub fn audit_summary_all(
        &self,
    ) -> Result<super::super::epistemic_auditor::AuditSummary, String> {
        self.epistemic_auditor
            .as_ref()
            .ok_or_else(|| "epistemic auditor not enabled".to_string())
            .and_then(|a| a.audit_summary_all())
    }

    /// Generate a formatted audit report over all recorded cycles.
    pub fn generate_audit_report_all(&self) -> Result<String, String> {
        self.epistemic_auditor
            .as_ref()
            .ok_or_else(|| "epistemic auditor not enabled".to_string())
            .and_then(|a| a.generate_report_all())
    }

    /// Force a synchronous flush of all buffered audit records to DuckDB.
    pub fn flush_audit(&mut self) {
        if let Some(ref mut a) = self.epistemic_auditor {
            a.flush_sync();
        }
    }

    // ── Thermodynamic Unification Accessors ──────────────────────────────

    /// Access the unified thermodynamic state (read-only).
    pub fn thermodynamic_state(&self) -> &super::thermodynamic_state::UnifiedThermodynamicState {
        self.thermodynamic_mgr.state()
    }

    /// Access the thermodynamic physics bridge (read-only).
    pub fn thermodynamic_bridge(
        &self,
    ) -> &super::thermodynamic_physics_bridge::ThermodynamicPhysicsBridge {
        self.thermodynamic_mgr.bridge()
    }

    /// Whether Landauer memory pressure is suppressing consolidation.
    pub fn thermodynamic_memory_suppressed(&self) -> bool {
        self.thermodynamic_mgr.memory_consolidation_suppressed()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MUSE ACCESSORS (consciousness-driven music synthesis)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "muse")]
impl CognitiveLoopService {
    /// Get the last rendered stereo PCM audio chunk from the muse manager.
    pub fn muse_audio_chunk(&self) -> &[[f32; 2]] {
        self.muse_manager.last_chunk()
    }

    /// Get muse telemetry (tempo, active notes, chunks rendered, etc.).
    pub fn muse_telemetry(&self) -> super::super::managers::muse_manager::MuseTelemetry {
        self.muse_manager.telemetry()
    }

    /// Get the current MusicalState driving synthesis.
    pub fn muse_musical_state(&self) -> &symthaea_muse::MusicalState {
        self.muse_manager.musical_state()
    }
}

impl CognitiveLoopService {
    /// Get mutable access to threshold overrides for applying evolved parameters.
    pub fn threshold_overrides_mut(
        &mut self,
    ) -> &mut super::super::threshold_overrides::ThresholdOverrides {
        &mut self.threshold_overrides
    }

    /// Get current threshold overrides.
    pub fn threshold_overrides(&self) -> &super::super::threshold_overrides::ThresholdOverrides {
        &self.threshold_overrides
    }

    // Voice synthesis API is in accessors/behavior.rs:
    // enable_voice_synthesis(), drain_voice_audio(), voice_synthesis_enabled()
}
