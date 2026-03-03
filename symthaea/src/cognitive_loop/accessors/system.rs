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
        let old = self.substrate_feasibility;
        let old_type = self.config.substrate_type;
        let canonical = substrate.canonical();
        self.substrate_feasibility =
            Self::requirements_for(&canonical).consciousness_feasibility();
        self.config.substrate_type = canonical;
        // Clear any stale composition — single-substrate mode now.
        self.config.substrate_composition = None;
        self.pending_substrate_transition = Some(format!(
            "{:?} -> {:?} ({:.3} -> {:.3})",
            old_type, canonical, old, self.substrate_feasibility
        ));
        self.recompute_effective_feasibility();
        self.recompute_substrate_dynamics();
        (old, self.substrate_feasibility)
    }

    /// Get the current substrate feasibility score.
    pub fn substrate_feasibility(&self) -> f64 {
        self.substrate_feasibility
    }

    /// Switch to a substrate composition at runtime, recomputing feasibility.
    pub fn reconfigure_composition(
        &mut self,
        composition: symthaea_core::hdc::substrate_composition::SubstrateComposition,
    ) {
        let old_feas = self.substrate_feasibility;
        self.substrate_feasibility = composition.feasibility();
        self.pending_substrate_transition = Some(format!(
            "-> {} ({:.3} -> {:.3})",
            composition.name, old_feas, self.substrate_feasibility
        ));
        self.config.substrate_composition = Some(composition);
        self.recompute_effective_feasibility();
        self.recompute_substrate_dynamics();
    }

    /// Get the current substrate composition (if set).
    pub fn substrate_composition(
        &self,
    ) -> Option<&symthaea_core::hdc::substrate_composition::SubstrateComposition> {
        self.config.substrate_composition.as_ref()
    }

    /// Get the current substrate honest confidence (evidence-based, 0.0–0.95).
    pub fn substrate_honest_confidence(&self) -> f64 {
        self.substrate_honest_confidence
    }

    /// Get the effective feasibility (raw × validation overlay blend).
    /// Equals raw feasibility when validation overlay is disabled.
    pub fn substrate_effective_feasibility(&self) -> f64 {
        self.substrate_effective_feasibility
    }

    /// Get the CfC tau factor from substrate speed modulation.
    /// 1.0 when speed modulation is disabled.
    pub fn substrate_tau_factor(&self) -> f32 {
        self.substrate_tau_factor
    }

    /// Get the substrate scale pressure (log10 ratio of max_scale to biological).
    /// 0.0 when speed modulation is disabled.
    pub fn substrate_scale_pressure(&self) -> f32 {
        self.substrate_scale_pressure
    }

    /// Recompute effective feasibility from raw feasibility × validation overlay.
    /// Called after any substrate/composition change and at startup.
    fn recompute_effective_feasibility(&mut self) {
        // Compute honest confidence from SubstrateValidationFramework.
        // When a composition is set, weight-blend confidence from all components.
        let framework =
            symthaea_core::hdc::substrate_validation::SubstrateValidationFramework::new();
        self.substrate_honest_confidence =
            if let Some(ref comp) = self.config.substrate_composition {
                let mut blended = 0.0f64;
                for (sub, &weight) in &comp.weights {
                    let conf = match Self::substrate_validation_key(sub) {
                        Some(k) => framework.honest_feasibility(k),
                        None => Self::THEORETICAL_CONFIDENCE,
                    };
                    blended += conf * weight as f64;
                }
                blended
            } else {
                match Self::substrate_validation_key(&self.config.substrate_type) {
                    Some(k) => framework.honest_feasibility(k),
                    None => Self::THEORETICAL_CONFIDENCE,
                }
            };

        if self.config.enable_validation_overlay {
            let floor = self.config.validation_skepticism_floor;
            let confidence = self.substrate_honest_confidence;
            self.substrate_effective_feasibility =
                self.substrate_feasibility * (floor + (1.0 - floor) * confidence);
        } else {
            self.substrate_effective_feasibility = self.substrate_feasibility;
        }
    }

    /// Recompute substrate speed/scale modulation factors.
    /// Called after any substrate change and at startup.
    /// When a composition is set, weight-blends speed/scale from all components.
    fn recompute_substrate_dynamics(&mut self) {
        use symthaea_core::hdc::substrate_independence::SubstrateType;

        if !self.config.enable_substrate_speed_modulation {
            self.substrate_tau_factor = 1.0;
            self.substrate_scale_pressure = 0.0;
            return;
        }

        let bio_speed = SubstrateType::BiologicalNeurons.operation_speed();
        let bio_scale = SubstrateType::BiologicalNeurons.max_scale();

        let (sub_speed, sub_scale) =
            if let Some(ref comp) = self.config.substrate_composition {
                let mut speed = 0.0f64;
                let mut scale = 0.0f64;
                for (sub, &weight) in &comp.weights {
                    speed += sub.operation_speed() * weight as f64;
                    scale += sub.max_scale() * weight as f64;
                }
                (speed, scale)
            } else {
                (
                    self.config.substrate_type.operation_speed(),
                    self.config.substrate_type.max_scale(),
                )
            };

        // log_ratio > 0 when substrate is faster than biological
        let log_ratio = (bio_speed / sub_speed).log10();
        // Compress 12 orders of magnitude to [0.5, 2.0] tau factor
        self.substrate_tau_factor = (1.0 + 0.5 * log_ratio / 9.0).clamp(0.5, 2.0) as f32;

        self.substrate_scale_pressure = (sub_scale / bio_scale).log10() as f32;
    }

    /// Map a canonical SubstrateType to its pre-built SubstrateRequirements profile.
    /// Unknown/future variants fall back to silicon_digital().
    pub(crate) fn requirements_for(
        substrate: &symthaea_core::hdc::substrate_independence::SubstrateType,
    ) -> symthaea_core::hdc::substrate_independence::SubstrateRequirements {
        use symthaea_core::hdc::substrate_independence::{SubstrateRequirements, SubstrateType};
        match substrate.canonical() {
            SubstrateType::BiologicalNeurons => SubstrateRequirements::biological_neurons(),
            SubstrateType::SiliconDigital => SubstrateRequirements::silicon_digital(),
            SubstrateType::QuantumComputer => SubstrateRequirements::quantum_computer(),
            SubstrateType::PhotonicProcessor => SubstrateRequirements::photonic_processor(),
            SubstrateType::NeuromorphicChip => SubstrateRequirements::neuromorphic_chip(),
            SubstrateType::BiochemicalComputer => SubstrateRequirements::biochemical_computer(),
            SubstrateType::HybridSystem => SubstrateRequirements::hybrid_system(),
            SubstrateType::ExoticSubstrate => SubstrateRequirements::exotic_substrate(),
            _ => SubstrateRequirements::silicon_digital(),
        }
    }

    /// Map SubstrateType to validation framework key string.
    ///
    /// Substrates not in the framework (photonic, neuromorphic, biochemical, exotic)
    /// return None — callers should fall back to Theoretical confidence (0.10).
    pub(crate) fn substrate_validation_key(
        substrate: &symthaea_core::hdc::substrate_independence::SubstrateType,
    ) -> Option<&'static str> {
        use symthaea_core::hdc::substrate_independence::SubstrateType;
        match substrate.canonical() {
            SubstrateType::BiologicalNeurons => Some("biological"),
            SubstrateType::SiliconDigital => Some("silicon"),
            SubstrateType::QuantumComputer => Some("quantum"),
            SubstrateType::HybridSystem => Some("hybrid"),
            // Photonic, Neuromorphic, Biochemical, Exotic: no framework entry.
            // All are Theoretical at best — caller uses THEORETICAL_CONFIDENCE fallback.
            _ => None,
        }
    }

    /// Default honest confidence for substrates not in the validation framework.
    /// Matches EvidenceLevel::Theoretical.confidence() = 0.10.
    pub(crate) const THEORETICAL_CONFIDENCE: f64 = 0.10;
}
