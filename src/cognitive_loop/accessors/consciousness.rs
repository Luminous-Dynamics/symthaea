// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness snapshot, pattern classification, primitive tier borrowers,
//! experience bus, and attention visualization accessors.

use crate::cognitive_loop::CognitiveLoopService;
use crate::dynamics::temporal_signatures::{ConsciousnessPattern, TemporalStateSummary};

use super::super::snapshot::ConsciousnessSnapshot;

#[allow(dead_code)]
impl CognitiveLoopService {
    cognitive_accessors! {
        // ═══════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS PATTERN
        // ═══════════════════════════════════════════════════════════════════

        /// Get current consciousness pattern classification
        pub fn consciousness_pattern(&self) -> (ConsciousnessPattern, f32) { self.language_comm.voice_coherence.temporal.classify_state() }

        /// Get full temporal state summary
        pub fn temporal_state_summary(&self) -> TemporalStateSummary { self.language_comm.voice_coherence.temporal.summary() }
    }

    // ========== Consciousness Snapshot ==========

    /// Get a complete snapshot of current consciousness state
    pub fn consciousness_snapshot(&self) -> ConsciousnessSnapshot {
        let (pattern, pattern_confidence) =
            self.language_comm.voice_coherence.temporal.classify_state();
        let temporal_summary = self.language_comm.voice_coherence.temporal.summary();
        let reflection_summary = self.consciousness.self_model_tier.self_reflection.summary();
        let thresholds = self
            .consciousness
            .self_model_tier
            .self_reflection
            .get_thresholds();
        let (emotion_nudge, _) = self.behavior.emotion_contagion.pattern_nudge();

        let consciousness_level = ConsciousnessSnapshot::compute_consciousness_level(
            self.prediction_confidence as f32,
            self.language_comm
                .voice_coherence
                .bridge
                .smoothed_coherence(),
            self.behavior.flow_state.intensity,
            pattern_confidence,
        );

        ConsciousnessSnapshot {
            cycle: self.stats.total_cycles,
            consciousness_level,
            pattern,
            pattern_confidence,
            prediction_error: self.stats.avg_prediction_error,
            prediction_confidence: self.prediction_confidence as f32,
            predictions_trustworthy: self.predictions_trustworthy(),
            effective_learning_rate: self.stats.adaptive_learning_rate,
            learning_effectiveness: self
                .consciousness
                .self_model_tier
                .self_reflection
                .learning_effectiveness(),
            in_flow: self.behavior.flow_state.in_flow,
            flow_intensity: self.behavior.flow_state.intensity,
            flow_streak: self.behavior.flow_state.streak,
            flow_learning_boost: self.behavior.flow_state.learning_boost,
            boredom: self.behavior.curiosity_drive.boredom,
            curiosity: self.behavior.curiosity_drive.curiosity,
            exploration_urge: self.behavior.curiosity_drive.exploration_urge as f32,
            exploring: self.behavior.curiosity_drive.should_explore(),
            novelty_bonus: self.behavior.curiosity_drive.novelty_bonus,
            emotional_valence: self.unification_engine.emotional.state().valence as f32,
            emotional_arousal: self.unification_engine.emotional.state().arousal as f32,
            has_emotional_content: self.has_emotional_content(),
            emotion_nudge,
            self_assessment: self
                .consciousness
                .self_model_tier
                .self_reflection
                .self_assessment,
            reflection_count: reflection_summary.reflection_count,
            adjustments_made: reflection_summary.adjustments_made,
            next_reflection_in: reflection_summary.next_reflection_in,
            action_hint: self.behavior.adaptive_behavior.action_hint,
            speech_rate_multiplier: self.behavior.adaptive_behavior.speech_rate_multiplier,
            pause_multiplier: self.behavior.adaptive_behavior.pause_multiplier,
            learning_paused: self.behavior.adaptive_behavior.pause_learning,
            flow_threshold: thresholds.flow_error,
            boredom_threshold: thresholds.boredom,
            trust_threshold: thresholds.trust,
            temporal_coherence: self
                .language_comm
                .voice_coherence
                .bridge
                .smoothed_coherence(),
            tau_mean: temporal_summary.features.mean,
            tau_trend: temporal_summary.features.trend,
            cognitive_depth: self.cognitive_depth,
            unified_psi: self.unification_engine.psi as f32,
            unified_valence: self.unification_engine.emotional.state().valence as f32,
            unified_arousal: self.unification_engine.emotional.state().arousal as f32,
            unified_dominance: self.unification_engine.emotional.state().dominance as f32,
            unified_discrete_emotion: self.unification_engine.emotional.state().discrete_emotion,
            emotional_pattern: self.unification_engine.emotional.detect_pattern(),
            emotional_description: self.unification_engine.emotional.state().describe(),
            snapshot_timestamp_nanos: self.start_time.elapsed().as_nanos() as u64,
            current_flow_duration_secs: self.behavior.flow_state.current_flow_duration_secs(),
            total_flow_time_secs: self.behavior.flow_state.total_flow_time_with_current(),
            flow_periods: self.behavior.flow_state.flow_periods,
            avg_flow_duration_secs: self.behavior.flow_state.avg_flow_duration_secs,
            fep_free_energy: self
                .fep
                .agent
                .last_fe_components
                .as_ref()
                .map(|fe| fe.total)
                .unwrap_or(0.0),
            fep_precision: self.fep.agent.precision.perceptual_precision(),
            spectral_mip_phi: self.carryover.consciousness.last_spectral_mip_phi,
            spectral_mip_adapted: self.carryover.consciousness.last_spectral_mip_adapted,
            spectral_mip_active_dim_count: self
                .carryover
                .consciousness
                .last_spectral_mip_active_dim_count,
            harmonies_alignment: self
                .ethics_engine
                .harmonies_integrator()
                .map(|h| h.stats().avg_alignment)
                .unwrap_or(0.0),
            empathic_compassion: self
                .primitive_tier
                .empathic_unification
                .as_ref()
                .map(|_| 0.0) // Compassion is per-cycle; snapshot shows lifetime average
                .unwrap_or(0.0),
            sigma: self.carryover.consciousness.last_sigma,
            integrity_critical: {
                #[cfg(feature = "integrity")]
                {
                    self.integrity_manager.has_critical_anomaly()
                }
                #[cfg(not(feature = "integrity"))]
                {
                    false
                }
            },
            network_critical: {
                #[cfg(feature = "mesh")]
                {
                    self.spectrum_manager.is_network_critical()
                }
                #[cfg(not(feature = "mesh"))]
                {
                    false
                }
            },
            avg_cycle_time_us: self.stats.avg_cycle_time_us,
            cycles_per_second: self.stats.cycles_per_second,
        }
    }

    /// Get a concise status line for logging/display
    pub fn status_line(&self) -> String {
        self.consciousness_snapshot().status()
    }

    /// Check if system needs attention (via snapshot)
    pub fn snapshot_needs_attention(&self) -> bool {
        self.consciousness_snapshot().needs_attention()
    }

    /// Get current consciousness level (0.0 to 1.0)
    pub fn consciousness_level(&self) -> f32 {
        let (_, pattern_confidence) = self.language_comm.voice_coherence.temporal.classify_state();
        ConsciousnessSnapshot::compute_consciousness_level(
            self.prediction_confidence as f32,
            self.language_comm
                .voice_coherence
                .bridge
                .smoothed_coherence(),
            self.behavior.flow_state.intensity,
            pattern_confidence,
        )
    }

    /// Check if current state matches a specific consciousness pattern
    pub fn is_consciousness_state(&self, pattern: ConsciousnessPattern) -> bool {
        self.language_comm
            .voice_coherence
            .temporal
            .is_state(pattern)
    }

    /// Get similarity to a specific consciousness pattern
    pub fn consciousness_pattern_similarity(&self, pattern: ConsciousnessPattern) -> f32 {
        self.language_comm
            .voice_coherence
            .temporal
            .similarity_to(pattern)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRIMITIVE TIER BORROWERS
    // ═══════════════════════════════════════════════════════════════════════

    /// Borrow the temporal primitives analyzer (if enabled).
    pub fn temporal_analyzer(
        &self,
    ) -> Option<&crate::consciousness::temporal_primitives::ConsciousnessTemporalAnalyzer> {
        self.primitive_tier.temporal_analyzer.as_ref()
    }

    /// Borrow the primitive lattice (if enabled).
    pub fn primitive_lattice(
        &self,
    ) -> Option<&crate::consciousness::primitive_lattice::PrimitiveLattice> {
        self.primitive_tier.primitive_lattice.as_ref()
    }

    /// Borrow the compositionality engine (if enabled).
    pub fn compositionality_engine(
        &self,
    ) -> Option<&crate::consciousness::compositionality::CompositionalityEngine> {
        self.primitive_tier.compositionality_engine.as_ref()
    }

    /// Borrow the unified value evaluator (owned by EthicsEngine).
    pub fn value_evaluator(
        &self,
    ) -> Option<&crate::consciousness::unified_value_evaluator::UnifiedValueEvaluator> {
        self.ethics_engine.value_evaluator()
    }

    /// Borrow the harmonic field (if enabled).
    pub fn harmonic_field(&self) -> Option<&crate::consciousness::harmonics::HarmonicField> {
        self.primitive_tier.harmonic_field.as_ref()
    }

    /// Borrow the primitive reasoner (if enabled).
    pub fn primitive_reasoner(
        &self,
    ) -> Option<&crate::consciousness::primitive_reasoning::PrimitiveReasoner> {
        self.primitive_tier.primitive_reasoner.as_ref()
    }

    /// Borrow the adaptive reasoner (if enabled).
    pub fn adaptive_reasoner(
        &self,
    ) -> Option<&crate::consciousness::adaptive_reasoning::AdaptiveReasoner> {
        self.primitive_tier.adaptive_reasoner.as_ref()
    }

    /// Borrow the causal self-explainer (if enabled).
    pub fn causal_explainer(
        &self,
    ) -> Option<&crate::consciousness::causal_explanation::CausalExplainer> {
        self.primitive_tier.causal_explainer.as_ref()
    }

    /// Borrow the context-aware optimizer (if enabled).
    pub fn context_optimizer(
        &self,
    ) -> Option<&crate::consciousness::context_aware_evolution::ContextAwareOptimizer> {
        self.primitive_tier.context_optimizer.as_ref()
    }

    /// Borrow the evolution coordinator (if enabled).
    pub fn evolution_coordinator(
        &self,
    ) -> Option<&crate::consciousness::evolution_bridge::EvolutionCoordinator> {
        self.primitive_tier.evolution_coordinator.as_ref()
    }

    /// Borrow the harmonies integrator (owned by EthicsEngine).
    pub fn harmonies_integrator(
        &self,
    ) -> Option<&crate::consciousness::harmonies_integration::HarmoniesIntegrator> {
        self.ethics_engine.harmonies_integrator()
    }

    /// Borrow the composition rule engine (if enabled).
    pub fn composition_rule_engine(
        &self,
    ) -> Option<&crate::consciousness::primitive_composition_rules::CompositionRuleEngine> {
        self.primitive_tier.composition_rule_engine.as_ref()
    }

    /// Borrow the semantic value embedder (if enabled).
    pub fn semantic_value_embedder(
        &self,
    ) -> Option<&crate::consciousness::semantic_value_embedder::SemanticValueEmbedder> {
        self.primitive_tier.semantic_value_embedder.as_ref()
    }

    /// Borrow the dissipative consciousness model (if enabled).
    pub(crate) fn dissipative_consciousness(
        &self,
    ) -> Option<&crate::consciousness::dissipative_consciousness::DissipativeConsciousness> {
        self.primitive_tier.dissipative_consciousness.as_ref()
    }

    /// Borrow the epistemic conflict detector (if enabled).
    pub fn epistemic_conflict_detector(
        &self,
    ) -> Option<&crate::consciousness::epistemic_conflict::ConflictDetector> {
        self.primitive_tier.epistemic_conflict_detector.as_ref()
    }

    /// Borrow the consciousness equation v2 (owned by ConsciousnessEngine).
    pub fn consciousness_equation_v2(
        &self,
    ) -> Option<&crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2> {
        self.consciousness
            .consciousness_engine
            .consciousness_equation_v2()
    }

    /// Borrow the hierarchical LTC (if enabled).
    pub fn hierarchical_ltc(
        &self,
    ) -> Option<&crate::consciousness::hierarchical_ltc::HierarchicalLTC> {
        self.primitive_tier.hierarchical_ltc.as_ref()
    }

    /// Borrow the theory calibrator (if enabled).
    pub fn theory_calibrator(
        &self,
    ) -> Option<&crate::consciousness::epistemic_conflict::TheoryCalibrator> {
        self.primitive_tier.theory_calibrator.as_ref()
    }

    /// Borrow the holographic consciousness analyzer (if enabled).
    pub(crate) fn holographic_analyzer(
        &self,
    ) -> Option<&crate::consciousness::consciousness_holography::HolographicConsciousnessAnalyzer>
    {
        self.primitive_tier.holographic_analyzer.as_ref()
    }

    /// Borrow the differentiable consciousness model (if enabled).
    pub fn differentiable_consciousness(
        &self,
    ) -> Option<&crate::consciousness::differentiable::DifferentiableConsciousness> {
        self.primitive_tier.differentiable_consciousness.as_ref()
    }

    /// Borrow the affective consciousness analyzer (if enabled).
    pub fn affective_consciousness(
        &self,
    ) -> Option<&crate::consciousness::affective_consciousness::AffectiveConsciousnessAnalyzer>
    {
        self.primitive_tier.affective_consciousness.as_ref()
    }

    /// Borrow the unified consciousness pipeline (owned by ConsciousnessEngine).
    pub fn unified_consciousness_pipeline(
        &self,
    ) -> Option<&crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline>
    {
        self.consciousness
            .consciousness_engine
            .unified_consciousness_pipeline()
    }

    /// Borrow the multi-modal integrator (owned by ConsciousnessEngine).
    pub fn multi_modal_integrator(
        &self,
    ) -> Option<&crate::consciousness::multi_modal_integration::MultiModalIntegrator> {
        self.consciousness
            .consciousness_engine
            .multi_modal_integrator()
    }

    /// Borrow the synthetic states NSM grounding (if enabled).
    pub(crate) fn synthetic_grounding(
        &self,
    ) -> Option<&crate::consciousness::synthetic_states::SyntheticStatesNSMGrounding> {
        self.primitive_tier.synthetic_grounding.as_ref()
    }

    /// Borrow the epistemic decision gate (if enabled).
    pub fn epistemic_gate(
        &self,
    ) -> Option<&crate::consciousness::gis_integration::EpistemicDecisionGate> {
        self.primitive_tier.epistemic_gate.as_ref()
    }

    /// Borrow the meta-cognitive reasoner (if enabled).
    pub fn meta_cognitive_reasoner(
        &self,
    ) -> Option<&crate::consciousness::meta_reasoning::MetaCognitiveReasoner> {
        self.primitive_tier.meta_cognitive_reasoner.as_ref()
    }

    /// Borrow the code primitive router (if enabled).
    pub fn code_primitive_router(
        &self,
    ) -> Option<&crate::consciousness::code_primitives::CodePrimitiveRouter> {
        self.primitive_tier.code_primitive_router.as_ref()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXPERIENCE BUS: Principled signals + Eight Harmonies
    // ═══════════════════════════════════════════════════════════════════════

    /// Current principled signals from experience bus.
    pub fn experience_signals(&self) -> Option<&crate::experience::PrincipledSignals> {
        self.experience_bus.as_ref().map(|bus| bus.signals())
    }

    /// KosmicSong state (Eight Harmonies + GIS + moral uncertainty).
    pub fn kosmic_state(&self) -> Option<&crate::experience::KosmicSong> {
        self.experience_bus.as_ref().map(|bus| bus.kosmic())
    }

    /// Current guiding question from wisdom system.
    pub fn guiding_question(&self) -> Option<&'static str> {
        self.experience_bus
            .as_ref()
            .map(|bus| bus.current_guiding_question())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ATTENTION VISUALIZATION ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get attention visualization summary (snapshot count, avg entropy, top attended).
    pub fn attention_summary(&self) -> Option<crate::visualization::AttentionSummary> {
        self.attention_visualizer.as_ref().map(|viz| viz.summary())
    }

    /// Export attention history as JSON for external analysis tools.
    pub fn attention_history_json(&self) -> Option<String> {
        self.attention_visualizer.as_ref().and_then(|viz| {
            viz.export_json()
                .map_err(|e| {
                    tracing::debug!(error = %e, "Attention visualizer JSON export failed");
                })
                .ok()
        })
    }

    /// Render attention heatmap as ASCII art (inputs x time).
    pub fn attention_heatmap(&self) -> Option<String> {
        self.attention_visualizer
            .as_ref()
            .map(|viz| viz.render_heatmap())
    }

    /// MCE calibration telemetry: (bottleneck dimension name, softmin factor,
    /// weighted sum) from the most recent cycle. Added 2026-07-18 for the
    /// post-scale-fix CL/safety-tier recalibration (consciousness_level
    /// saturates ~0.95 now that all MCE inputs float high — see keystone
    /// Phase-5 results; this exposes WHICH dimension binds per regime).
    pub fn mce_summary(&self) -> (String, f64, f64) {
        (
            self.carryover.consciousness.mce_bottleneck_name.clone(),
            self.carryover.consciousness.mce_softmin,
            self.carryover.consciousness.mce_weighted_sum,
        )
    }
}
