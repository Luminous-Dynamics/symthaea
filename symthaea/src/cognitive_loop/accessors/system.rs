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
        pub fn pain_sender(&self) -> Option<crate::infrastructure::PainSender> { self.pain_tx.clone() }

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
        pub fn coherence_summary(&self) -> crate::dynamics::cfc_coherence::CoherenceSummary { self.coherence_bridge.summary() }

        // ═══════════════════════════════════════════════════════════════════
        // SEMANTIC MEMORY / STABILITY ACCESSORS
        // ═══════════════════════════════════════════════════════════════════

        /// Get semantic memory statistics
        pub fn semantic_memory_stats(&self) -> &crate::memory::semantic_memory::SemanticMemoryStats { self.semantic_memory.stats() }

        /// Get reference to the stability regime processor
        pub fn stability_regime(&self) -> &crate::consciousness::stability_regime::StabilityRegimeProcessor { &self.stability_regime }

        // ═══════════════════════════════════════════════════════════════════
        // PREDICTION CONFIDENCE
        // ═══════════════════════════════════════════════════════════════════

        /// Get current prediction confidence (returned as f32 for API stability).
        pub fn prediction_confidence(&self) -> f32 { self.prediction_confidence as f32 }

        /// Check if predictions should be trusted
        pub fn predictions_trustworthy(&self) -> bool { self.prediction_confidence > 0.4 }

        /// Current FEP learning signal (0.0-1.0).
        /// Used by the facade to modulate L-SSM distillation intensity.
        pub fn fep_learning_signal(&self) -> f32 { self.fep_learning_signal }

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

    /// Evaluate temporal prediction horizon accuracy from the vision manifold.
    #[cfg(feature = "vision-manifold")]
    pub fn vision_evaluate_horizons(
        &self,
    ) -> Option<symthaea_vision_manifold::HorizonAccuracy> {
        self.vision_bridge
            .as_ref()
            .map(|b| b.manifold().evaluate_horizons())
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
            substrate_feasibility: self.substrate_manager.effective_feasibility as f64,
            cycle_hz: self.stats.cycles_per_second,
            phi: self.stats.unified_psi as f64,
            features: Vec::new(), // populated by caller who knows feature flags
            physics_domains: Vec::new(),
        };

        crate::swarm::CapabilityCard::from_config(agent_key, &self.config, &card_stats)
    }

    /// Get the current inferred user state (if user state inference is enabled).
    pub fn user_state(&self) -> Option<&crate::user_state_inference::UserState> {
        self.user_state.as_ref().map(|usi| usi.state())
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
        self.substrate_manager
            .reconfigure_substrate(&mut self.config, substrate)
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

    /// Inject a raw frame buffer for the next vision manifold cycle.
    /// The frame is consumed during the next `cycle()` call.
    #[cfg(feature = "vision-manifold")]
    pub fn inject_vision_frame(&mut self, frame: Vec<u8>) {
        self.vision_frame_buffer = Some(frame);
    }

}
