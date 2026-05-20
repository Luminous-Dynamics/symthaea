// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness metrics computation extracted from cycle.rs.
//!
//! Contains: primitive consciousness, temporal primitives, lattice, compositionality,
//! value evaluator, consciousness profile, context-aware evolution, semantic value embedder,
//! harmonies integration, composition rules, fiduciary harmonics, primitive reasoning,
//! causal self-explanation, adaptive reasoning, epistemic tiers, phi validation,
//! dissipative consciousness, epistemic conflict, consciousness equation v2.

use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::join as rayon_join;

use super::CognitiveLoopService;
use super::thresholds::{
    CAUSAL_BINDING_THRESHOLD, HARMONIC_FIELD_BOOST_FACTOR, HARMONIC_FIELD_BOOST_THRESHOLD,
    PHI_VALIDATION_HIGH_THRESHOLD, PHI_VALIDATION_LOW_THRESHOLD, REASONING_CONFIDENCE_BOOST_FACTOR,
    REASONING_CONFIDENCE_BOOST_THRESHOLD, SPECTRAL_WEIGHT_BASE, SPECTRAL_WEIGHT_SCALE,
    TEMPORAL_CHAIN_BOOST_FACTOR, TEMPORAL_CONTINUITY_BOOST_FACTOR,
    TEMPORAL_CONTINUITY_BOOST_THRESHOLD, TEMPORAL_REPLAY_TRIGGER,
};

/// Values computed by the consciousness metrics phase.
/// Passed to later phases that need these results.
pub(crate) struct ConsciousnessMetrics {
    pub primitive_psi: f64,
    pub active_primitive_names: Vec<String>,
    pub temporal_causal_chains: usize,
    pub temporal_continuity: f64,
    pub temporal_max_chain_length: usize,
    pub chain_cycle_numbers: Vec<u64>,
    pub causal_codebook_entries: Vec<(String, Vec<f32>)>,
    pub continuity_replay_needed: bool,
    pub lattice_height: usize,
    pub lattice_width: usize,
    pub lattice_join_concept: Option<String>,
    pub compositionality_total: usize,
    pub value_evaluator_score: f64,
    pub value_evaluator_decision: String,
    pub value_gate_factor: f32,
    pub consciousness_profile_composite: f64,
    pub synergy_enhanced_composite: f64,
    pub emergent_properties_count: usize,
    pub reasoning_context: String,
    pub context_phi_weight: f64,
    pub value_embeddings_created: u64,
    pub value_cache_hit_rate: f32,
    pub harmonies_alignment: f32,
    pub harmonies_approved: bool,
    pub composition_rule_applied: String,
    pub harmonic_field_coherence: f64,
    pub harmonic_love_resonance: f64,
    pub harmonic_interferences: usize,
    pub reasoning_chain_confidence: f32,
    pub reasoning_chain_depth: usize,
    pub causal_relations_count: usize,
    pub causal_avg_confidence: f64,
    pub adaptive_reasoning_phi: f64,
    pub epistemic_quality: f64,
    pub phi_validation_correlation: f64,
    pub dissipative_health: f64,
    pub dissipative_regime: String,
    pub dissipative_entropy_rate: f64,
    pub epistemic_phi_eff: f64,
    pub epistemic_conflict_count: usize,
    pub equation_v2_consciousness: f64,
}

impl CognitiveLoopService {
    /// Compute consciousness metrics from the current cycle state.
    ///
    /// This is Phase "consciousness metrics" — extracted from cycle.rs lines ~1634-2370.
    /// All logic and behavior is preserved exactly.
    pub(crate) fn compute_consciousness_metrics(
        &mut self,
        state: &super::CycleState<'_>,
        module_timings: &mut super::ModuleTimings,
    ) -> ConsciousnessMetrics {
        let hv16_cached = *state.hv16_cached;
        let unified_psi = state.unified_psi;
        let coherence = state.coherence;
        let prediction_error = state.prediction_error;
        let phi_attention_weight = state.phi_attention_weight;
        let compressed_state = state.compressed_state;
        let input = state.input;
        let urgency = state.urgency;
        // ═══════════════════════════════════════════════════════════════════════
        // PRIMITIVE CONSCIOUSNESS: Decompose consciousness state into primitives
        // Provides explainable consciousness by mapping HDC encodings to the
        // 9-tier primitive system with activation tracking and binding.
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // Science: Tononi & Koch (2015) — primitives of consciousness experience
        // ═══════════════════════════════════════════════════════════════════════
        let (primitive_psi, active_primitive_names) =
            if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut processor) = self.primitive_tier.primitive_processor {
                    let timestamp = self.stats.total_cycles as f64 * 0.02; // 50Hz → seconds
                    let state = processor.process_input(&hv16_cached, timestamp);
                    let names: Vec<String> = state
                        .all_active()
                        .iter()
                        .take(4)
                        .map(|ap| ap.primitive.name.clone())
                        .collect();
                    (state.phi, names)
                } else {
                    (0.0, Vec::new())
                }
            } else {
                (0.0, Vec::new())
            };

        // ═══════════════════════════════════════════════════════════════════════
        // TEMPORAL PRIMITIVES: Allen's Interval Algebra on conscious states
        // [extracted to compute_temporal_primitives_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
        ) = self.compute_temporal_primitives_phase(
            hv16_cached,
            unified_psi,
            coherence,
            module_timings,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // PRIMITIVE LATTICE: Structural metrics from tier system
        // [extracted to compute_lattice_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (lattice_height, lattice_width, lattice_join_concept) =
            self.compute_lattice_phase(&active_primitive_names, module_timings);

        // ═══════════════════════════════════════════════════════════════════════
        // COMPOSITIONALITY ENGINE: Algebraic composition of primitives
        // Tracks composition stats; actual compositions are demand-driven.
        // Science: Category Theory (Mac Lane 1998), HDC algebraic operators.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let compositionality_total =
            if let Some(ref compositionality) = self.primitive_tier.compositionality_engine {
                compositionality.get_stats().total_compositions
            } else {
                0
            };
        module_timings.compositionality = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED VALUE EVALUATOR: Eight Harmonies alignment scoring
        // [extracted to compute_value_evaluator_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (value_evaluator_score, value_evaluator_decision, value_gate_factor) =
            self.compute_value_evaluator_phase(unified_psi, module_timings);

        // ═══════════════════════════════════════════════════════════════════════
        // PARALLEL CONSCIOUSNESS METRICS: phi_validation + dissipative +
        // consciousness_profile (Branch A) ‖ fiduciary_harmonics +
        // primitive_reasoning + epistemic_conflict (Branch B).
        //
        // These 6 phases access disjoint fields of PrimitiveTierManager and
        // produce only numeric results + deferred feedback. When the `parallel`
        // feature is enabled, rayon::join runs them concurrently.
        // ═══════════════════════════════════════════════════════════════════════
        let has_primitive_processor = self.primitive_tier.primitive_processor.is_some();

        // Snapshot carryover values needed by parallel branches (read-only copies).
        let snap_phi_validation_correlation = self.carryover.quality.phi_validation_correlation;
        let snap_phi_spectral_weight = self.carryover.quality.phi_spectral_weight;
        let snap_dissipative_health = self.carryover.quality.last_dissipative_health;
        let snap_profile_composite = self.carryover.history.last_profile_composite;
        let snap_synergy_composite = self.carryover.history.last_synergy_composite;
        let snap_emergent_count = self.carryover.history.last_emergent_count;
        let snap_harmonic_coherence = self.carryover.consciousness.last_harmonic_coherence;
        let snap_phi_eff = self.carryover.quality.last_phi_eff;
        let snap_body_phi_modulation = self.carryover.consciousness.body_phi_modulation;
        let snap_prediction_confidence = self.prediction_confidence;
        let total_cycles = self.stats.total_cycles;

        // Split disjoint borrows from primitive_tier and carryover for parallel branches.
        // Rust's borrow checker verifies these are non-overlapping struct fields.
        let branch_a_phi_validation = &mut self.primitive_tier.phi_validation;
        let branch_a_dissipative = &mut self.primitive_tier.dissipative_consciousness;
        let branch_a_recent_hvs = &mut self.carryover.history.recent_hvs;

        let branch_b_harmonic_field = &mut self.primitive_tier.harmonic_field;
        let branch_b_harmonic_resolver = &self.primitive_tier.harmonic_resolver;
        let branch_b_primitive_reasoner = &mut self.primitive_tier.primitive_reasoner;
        let branch_b_epistemic_conflict = &mut self.primitive_tier.epistemic_conflict_detector;
        let branch_b_theory_calibrator = &self.primitive_tier.theory_calibrator;

        // `mut` required for non-parallel FnMut call; unused_mut when `parallel`
        // moves the closure into rayon_join.
        #[allow(unused_mut)]
        let mut branch_a_fn = || {
            super::helpers::parallel_consciousness_branch_a(
                branch_a_phi_validation,
                branch_a_dissipative,
                has_primitive_processor,
                hv16_cached,
                branch_a_recent_hvs,
                prediction_error,
                coherence,
                unified_psi,
                total_cycles,
                snap_phi_validation_correlation,
                snap_phi_spectral_weight,
                snap_dissipative_health,
                snap_profile_composite,
                snap_synergy_composite,
                snap_emergent_count,
            )
        };
        #[allow(unused_mut)]
        let mut branch_b_fn = || {
            super::helpers::parallel_consciousness_branch_b(
                branch_b_harmonic_field,
                branch_b_harmonic_resolver,
                branch_b_primitive_reasoner,
                branch_b_epistemic_conflict,
                branch_b_theory_calibrator,
                coherence,
                prediction_error,
                unified_psi,
                snap_prediction_confidence,
                total_cycles,
                snap_harmonic_coherence,
                snap_phi_eff,
                snap_body_phi_modulation,
            )
        };

        #[cfg(feature = "parallel")]
        let (branch_a, branch_b) = {
            use std::panic::AssertUnwindSafe;
            rayon_join(
                || {
                    std::panic::catch_unwind(AssertUnwindSafe(branch_a_fn)).unwrap_or_else(|_| {
                        tracing::error!(
                            "Consciousness Branch A (phi_validation/dissipative/profile) panicked"
                        );
                        super::helpers::ConsciousnessMetricsBranchA {
                            phi_validation_correlation: snap_phi_validation_correlation,
                            phi_validation_timing: 0,
                            dissipative_health: 0.0,
                            dissipative_regime: String::new(),
                            dissipative_entropy_rate: 0.0,
                            dissipative_timing: 0,
                            consciousness_profile_composite: snap_profile_composite,
                            synergy_enhanced_composite: snap_synergy_composite,
                            emergent_properties_count: snap_emergent_count,
                            consciousness_profile_timing: 0,
                            new_phi_validation_correlation: None,
                            new_phi_spectral_weight: None,
                            new_dissipative_health: None,
                            new_profile_composite: None,
                            new_synergy_composite: None,
                            new_emergent_count: None,
                            deferred: Vec::new(),
                        }
                    })
                },
                || {
                    std::panic::catch_unwind(AssertUnwindSafe(branch_b_fn)).unwrap_or_else(|_| {
                        tracing::error!(
                            "Consciousness Branch B (harmonics/reasoning/epistemic) panicked"
                        );
                        super::helpers::ConsciousnessMetricsBranchB {
                            harmonic_field_coherence: snap_harmonic_coherence,
                            harmonic_love_resonance: 0.0,
                            harmonic_interferences: 0,
                            harmonics_timing: 0,
                            reasoning_chain_confidence: 0.0,
                            reasoning_chain_depth: 0,
                            reasoning_timing: 0,
                            epistemic_phi_eff: snap_phi_eff,
                            epistemic_conflict_count: 0,
                            epistemic_conflict_timing: 0,
                            new_harmonic_coherence: None,
                            new_phi_eff: None,
                            new_epistemic_conflict_count: None,
                            epistemic_reasoning_override: false,
                            deferred: Vec::new(),
                        }
                    })
                },
            )
        };
        #[cfg(not(feature = "parallel"))]
        let (branch_a, branch_b) = (branch_a_fn(), branch_b_fn());

        // ── Apply deferred feedback from parallel branches ──────────────
        for fb in branch_a.deferred.iter().chain(branch_b.deferred.iter()) {
            match fb {
                super::helpers::DeferredFeedback::AdjustConfidence(tag, delta) => {
                    self.adjust_confidence(tag, *delta);
                }
                super::helpers::DeferredFeedback::ScaleLr(tag, factor) => {
                    self.scale_lr(tag, *factor);
                }
                super::helpers::DeferredFeedback::AdjustExploration(tag, delta) => {
                    self.adjust_exploration(tag, *delta);
                }
                super::helpers::DeferredFeedback::ScaleExploration(tag, factor) => {
                    self.scale_exploration(tag, *factor);
                }
                super::helpers::DeferredFeedback::ScaleSubsystemLr(factor) => {
                    self.carryover.learning.subsystem_lr_factor *= factor;
                }
            }
        }

        // ── Write back cached carryover values from Branch A ────────────
        if let Some(v) = branch_a.new_phi_validation_correlation {
            self.carryover.quality.phi_validation_correlation = v;
        }
        if let Some(v) = branch_a.new_phi_spectral_weight {
            self.carryover.quality.phi_spectral_weight = v;
        }
        if let Some(v) = branch_a.new_dissipative_health {
            self.carryover.quality.last_dissipative_health = v;
        }
        if let Some(v) = branch_a.new_profile_composite {
            self.carryover.history.last_profile_composite = v;
        }
        if let Some(v) = branch_a.new_synergy_composite {
            self.carryover.history.last_synergy_composite = v;
        }
        if let Some(v) = branch_a.new_emergent_count {
            self.carryover.history.last_emergent_count = v;
        }

        // ── Write back cached carryover values from Branch B ────────────
        if let Some(v) = branch_b.new_harmonic_coherence {
            self.carryover.consciousness.last_harmonic_coherence = v;
        }
        if let Some(v) = branch_b.new_phi_eff {
            self.carryover.quality.last_phi_eff = v;
        }
        if let Some(v) = branch_b.new_epistemic_conflict_count {
            self.carryover.quality.last_epistemic_conflict_count = v;
        }
        if branch_b.epistemic_reasoning_override {
            self.carryover.quality.epistemic_reasoning_override = true;
        }

        // ── Unpack parallel branch results ──────────────────────────────
        let phi_validation_correlation = branch_a.phi_validation_correlation;
        let dissipative_health = branch_a.dissipative_health;
        let dissipative_regime = branch_a.dissipative_regime;
        let dissipative_entropy_rate = branch_a.dissipative_entropy_rate;
        let consciousness_profile_composite = branch_a.consciousness_profile_composite;
        let synergy_enhanced_composite = branch_a.synergy_enhanced_composite;
        let emergent_properties_count = branch_a.emergent_properties_count;
        let harmonic_field_coherence = branch_b.harmonic_field_coherence;
        let harmonic_love_resonance = branch_b.harmonic_love_resonance;
        let harmonic_interferences = branch_b.harmonic_interferences;
        let reasoning_chain_confidence = branch_b.reasoning_chain_confidence;
        let reasoning_chain_depth = branch_b.reasoning_chain_depth;
        let epistemic_phi_eff = branch_b.epistemic_phi_eff;
        let epistemic_conflict_count = branch_b.epistemic_conflict_count;

        // ── Write module timings from parallel branches ──────────────────
        module_timings.phi_validation = branch_a.phi_validation_timing;
        module_timings.dissipative_consciousness = branch_a.dissipative_timing;
        module_timings.consciousness_profile = branch_a.consciousness_profile_timing;
        module_timings.harmonics = branch_b.harmonics_timing;
        module_timings.primitive_reasoning = branch_b.reasoning_timing;
        module_timings.epistemic_conflict = branch_b.epistemic_conflict_timing;

        // ═══════════════════════════════════════════════════════════════════════
        // CONTEXT-AWARE EVOLUTION: Dynamic Φ/Harmonic/Epistemic weighting
        // Detects reasoning context from input text and adjusts objective weights.
        // Runs every cycle (keyword match is O(1) in input length).
        // Science: Gigerenzer (2007) — ecological rationality, context-adaptive reasoning.
        // ═══════════════════════════════════════════════════════════════════════
        let (reasoning_context, context_phi_weight) =
            if let Some(ref optimizer) = self.primitive_tier.context_optimizer {
                let ctx = optimizer.detect_context(input, None);
                let weights = optimizer.get_weights_for_context(&ctx);
                (ctx.description().to_string(), weights.phi_weight)
            } else {
                (String::new(), 0.0)
            };

        // ═══════════════════════════════════════════════════════════════════════
        // SEMANTIC VALUE EMBEDDER: Value-aligned embeddings grounded in primitives
        // Projects compressed state into value-aware space using primitive-tier
        // harmony bases. Cached — repeated inputs hit O(1) lookup.
        // Science: Schwartz (2012) — value theory, Kanerva (2009) — HDC semantics.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (value_embeddings_created, value_cache_hit_rate) =
            if let Some(ref mut embedder) = self.primitive_tier.semantic_value_embedder {
                if self.stats.total_cycles % 11 == 0 {
                    let continuous = symthaea_core::hdc::ContinuousHV::from_slice(compressed_state);
                    let _concept =
                        embedder.embed(format!("cycle_{}", self.stats.total_cycles), continuous);
                    (
                        embedder.stats().embeddings_created,
                        embedder.cache_hit_rate(),
                    )
                } else {
                    (
                        embedder.stats().embeddings_created,
                        embedder.cache_hit_rate(),
                    )
                }
            } else {
                (0, 0.0)
            };
        module_timings.semantic_value_embedder = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // HARMONIES INTEGRATOR: Now handled by EthicsEngine (called in cycle.rs).
        // We read cached values; feedback already applied by engine output handler.
        // Science: Schwartz (2012) — basic human values, Deci & Ryan (2000).
        // ═══════════════════════════════════════════════════════════════════════
        let harmonies_alignment = self.ethics_engine.last_harmonies_alignment();
        let harmonies_approved = self.ethics_engine.last_harmonies_approved();

        // ═══════════════════════════════════════════════════════════════════════
        // COMPOSITION RULES: Domain-specific HDC binding operator selection
        // Selects the best composition rule (temporal-physical, mathematical,
        // consciousness, cross-tier) for the top-2 active primitives.
        // Stateless — O(1) rule lookup per cycle.
        // Science: Plate (2003) — holographic reduced representations.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let composition_rule_applied =
            if let Some(ref rule_engine) = self.primitive_tier.composition_rule_engine {
                if active_primitive_names.len() >= 2 {
                    let system = symthaea_core::hdc::primitive_system::PrimitiveSystem::global();
                    let tier1 = system
                        .get(&active_primitive_names[0])
                        .map(|p| p.tier)
                        .unwrap_or(symthaea_core::hdc::primitive_system::PrimitiveTier::NSM);
                    let tier2 = system
                        .get(&active_primitive_names[1])
                        .map(|p| p.tier)
                        .unwrap_or(symthaea_core::hdc::primitive_system::PrimitiveTier::NSM);
                    rule_engine.matching_rule_name(tier1, tier2).to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
        module_timings.composition_rules = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CAUSAL SELF-EXPLANATION: Pearl causal model of primitive→Phi effects
        // [extracted to compute_causal_self_explanation_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (causal_relations_count, causal_avg_confidence) = self
            .compute_causal_self_explanation_phase(
                hv16_cached,
                &active_primitive_names,
                primitive_psi,
                module_timings,
            );

        // ═══════════════════════════════════════════════════════════════════════
        // ADAPTIVE REASONING: Q-learning-guided primitive selection
        // Builds reasoning chains with RL-optimized primitive selection.
        // Amortized: every 47 cycles (Q-learning step + chain construction, co-prime).
        // Science: Sutton & Barto (2018) — reinforcement learning + HDC.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let epistemic_override = self.carryover.quality.epistemic_reasoning_override;
        let adaptive_reasoning_phi =
            if let Some(ref mut reasoner) = self.primitive_tier.adaptive_reasoner {
                if (self.stats.total_cycles % 97 == 0 && self.stats.total_cycles > 0)
                    || epistemic_override
                {
                    if epistemic_override {
                        self.carryover.quality.epistemic_reasoning_override = false;
                        self.stats.epistemic_reasoning_accelerations += 1;
                    }
                    match reasoner.reason_adaptive(hv16_cached, 5) {
                        Ok(chain) => chain.total_phi,
                        Err(_) => 0.0,
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
        module_timings.adaptive_reasoning = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // EPISTEMIC TIERS: 3-axis epistemic classification of Phi measurements
        // Classifies the current Phi measurement's empirical, normative, and
        // materiality status. Lightweight — computed each cycle.
        // Science: Mycelix Epistemic Charter v2.0.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let epistemic_quality = if self.primitive_tier.primitive_processor.is_some() {
            if self.stats.total_cycles % 97 == 0 {
                use crate::consciousness::epistemic_tiers::*;
                // Classify based on cycle count (more cycles = higher empirical tier)
                let empirical = if self.stats.total_cycles > 1000 {
                    EmpiricalTier::E3CryptographicallyProven
                } else if self.stats.total_cycles > 100 {
                    EmpiricalTier::E2PrivatelyVerifiable
                } else if self.stats.total_cycles > 10 {
                    EmpiricalTier::E1Testimonial
                } else {
                    EmpiricalTier::E0Null
                };
                let coord = EpistemicCoordinate::new(
                    empirical,
                    NormativeTier::N0Personal,
                    MaterialityTier::M1Temporal,
                );
                let q = coord.quality_score();
                self.carryover.quality.last_epistemic_quality = q;
                q
            } else {
                self.carryover.quality.last_epistemic_quality
            }
        } else {
            0.0
        };
        module_timings.epistemic_tiers = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS EQUATION V2: Unified 7-theory formula
        // [extracted to compute_equation_v2_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let equation_v2_consciousness = self.compute_equation_v2_phase(
            unified_psi,
            coherence,
            prediction_error,
            phi_attention_weight,
            module_timings,
        );

        ConsciousnessMetrics {
            primitive_psi,
            active_primitive_names,
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
            lattice_height,
            lattice_width,
            lattice_join_concept,
            compositionality_total,
            value_evaluator_score,
            value_evaluator_decision,
            value_gate_factor,
            consciousness_profile_composite,
            synergy_enhanced_composite,
            emergent_properties_count,
            reasoning_context,
            context_phi_weight,
            value_embeddings_created,
            value_cache_hit_rate,
            harmonies_alignment,
            harmonies_approved,
            composition_rule_applied,
            harmonic_field_coherence,
            harmonic_love_resonance,
            harmonic_interferences,
            reasoning_chain_confidence,
            reasoning_chain_depth,
            causal_relations_count,
            causal_avg_confidence,
            adaptive_reasoning_phi,
            epistemic_quality,
            phi_validation_correlation,
            dissipative_health,
            dissipative_regime,
            dissipative_entropy_rate,
            epistemic_phi_eff,
            epistemic_conflict_count,
            equation_v2_consciousness,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Extracted subsystem methods — compute + apply feedback + return values
    // ═══════════════════════════════════════════════════════════════════════════

    /// Temporal Primitives: Allen's Interval Algebra on conscious states.
    ///
    /// Records conscious intervals each cycle; amortized causal chain detection
    /// (every 47 cycles) and continuity analysis (every 97 cycles). Applies feedback:
    /// temporal continuity and causal chains boost prediction confidence; causal chains
    /// boost episodic consolidation; continuity gaps trigger demand replay.
    ///
    /// Science: Allen (1983), Varela (1999).
    #[allow(clippy::type_complexity)]
    pub(in crate::cognitive_loop) fn compute_temporal_primitives_phase(
        &mut self,
        hv16_cached: symthaea_core::hdc::BinaryHV,
        unified_psi: f64,
        coherence: f32,
        module_timings: &mut super::ModuleTimings,
    ) -> (usize, f64, usize, Vec<u64>, Vec<(String, Vec<f32>)>, bool) {
        let _t = Instant::now();
        let (
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
        ) = if let Some(ref mut analyzer) = self.primitive_tier.temporal_analyzer {
            // Record this cycle as a conscious interval
            let timestamp = self.stats.total_cycles as f64 * 0.02; // 50Hz → seconds
            use crate::consciousness::temporal_primitives::{
                ConsciousInterval, PhiTrend, TemporalInterval,
            };
            let mut ti = TemporalInterval::new(
                format!("c{}", self.stats.total_cycles),
                timestamp,
                timestamp + 0.02,
            )
            .unwrap_or_else(|_| {
                // SAFETY: 0.0 < 0.02 is always valid — this cannot fail.
                // Using expect() here is acceptable because the constants are
                // compile-time verifiable (end > start).
                TemporalInterval::new("c_fallback", 0.0, 0.02)
                    .expect("hardcoded valid interval 0.0 < 0.02")
            });
            ti.phi = Some(unified_psi);
            let mut interval = ConsciousInterval::new(
                ti,
                unified_psi,
                coherence as f64,
                if self.stats.total_cycles > 0 {
                    0.5
                } else {
                    0.0
                },
            );
            interval.phi_trend = if unified_psi > self.carryover.history.consciousness_level + 0.01
            {
                PhiTrend::Rising
            } else if unified_psi < self.carryover.history.consciousness_level - 0.01 {
                PhiTrend::Falling
            } else {
                PhiTrend::Stable
            };
            interval.content = Some(hv16_cached);
            analyzer.add_interval(interval);

            // Amortized analysis: causal chains every 47 cycles (co-prime)
            let (chains, ccn, cce) = if self.stats.total_cycles % 47 == 0
                && self.stats.total_cycles > 0
            {
                let detected = analyzer.detect_causal_chains(3);
                let count = detected.len();
                let max_len = detected
                    .iter()
                    .map(|c| c.intervals.len())
                    .max()
                    .unwrap_or(0);
                self.carryover.quality.causal_chain_count = count;

                // Track A: Extract cycle numbers from genuine chains for episodic consolidation
                let cycle_nums: Vec<u64> = analyzer.genuine_chain_cycle_numbers();

                // Track A: Build causal codebook entries by binding content BinaryHVs
                let codebook_entries: Vec<(String, Vec<f32>)> = detected
                    .iter()
                    .filter(|c| {
                        c.genuine_causation && c.causal_strength > CAUSAL_BINDING_THRESHOLD as f64
                    })
                    .filter_map(|c| {
                        let contents: Vec<&crate::hdc::binary_hv::BinaryHV> = c
                            .intervals
                            .iter()
                            .filter_map(|id| analyzer.interval_content(id))
                            .collect();
                        if contents.len() >= 2 {
                            let bound = crate::hdc::BinaryHV::bind_chain(&contents);
                            // Compress to resonator dim via same pipeline as compressed_state
                            let continuous = bound.to_continuous();
                            let compressed = self
                                .encoder
                                .compress_for_ltc(&continuous, self.config.cfc_config.input_dim);
                            Some((
                                format!("causal_{}_{}", self.stats.total_cycles, c.intervals.len()),
                                compressed,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();

                ((count, max_len), cycle_nums, codebook_entries)
            } else {
                (
                    (self.carryover.quality.causal_chain_count, 0),
                    Vec::new(),
                    Vec::new(),
                )
            };

            // Amortized analysis: continuity every 97 cycles (co-prime)
            let (continuity, cont_replay) =
                if self.stats.total_cycles % 97 == 0 && self.stats.total_cycles > 0 {
                    let analysis = analyzer.analyze_continuity();
                    self.carryover.quality.temporal_continuity = analysis.continuity_score;
                    // Track A: Detect continuity gaps that warrant demand replay
                    let replay_needed = analysis.continuity_score < TEMPORAL_REPLAY_TRIGGER as f64
                        || analysis.gap_count > 5;
                    (analysis.continuity_score, replay_needed)
                } else {
                    (self.carryover.quality.temporal_continuity, false)
                };

            (chains.0, continuity, chains.1, ccn, cce, cont_replay)
        } else {
            (0, 0.0, 0, Vec::new(), Vec::new(), false)
        };
        module_timings.temporal_analyzer = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Temporal continuity → prediction confidence (stable time-axis = reliable predictions)
        if temporal_continuity > TEMPORAL_CONTINUITY_BOOST_THRESHOLD {
            let boost = ((temporal_continuity - TEMPORAL_CONTINUITY_BOOST_THRESHOLD)
                * TEMPORAL_CONTINUITY_BOOST_FACTOR) as f32; // up to +1.5%
            self.adjust_confidence("temporal_continuity", boost);
        }

        // FEEDBACK: Causal chain detection → confidence boost (the system found real structure)
        if temporal_causal_chains > 2 {
            let chain_boost =
                (temporal_causal_chains.min(10) as f32 - 2.0) * TEMPORAL_CHAIN_BOOST_FACTOR; // +0.5% per chain, up to +4%
            self.adjust_confidence("causal_chain_detect", chain_boost);
        }

        // Track A-1: Causal chain → episodic memory consolidation boost
        if !chain_cycle_numbers.is_empty() {
            if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                replay.boost_causal_consolidation(&chain_cycle_numbers, 0.15);
            }
        }

        // Track A-3: Continuity gaps → demand replay
        if continuity_replay_needed {
            if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                replay.trigger_demand_replay();
            }
        }

        (
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
        )
    }

    /// Dissipative Consciousness: Prigogine thermodynamic self-organization.
    ///
    /// Tracks entropy production, order parameter, and edge-of-chaos criticality.
    /// Applies feedback: thermodynamic regime drives cognitive adjustments via
    /// `recommend_action()` — exploration, coherence, differentiation, integration.
    ///
    /// Science: Prigogine (1977), Kauffman (1993), England (2013).
    pub(in crate::cognitive_loop) fn compute_dissipative_phase(
        &mut self,
        prediction_error: f32,
        coherence: f32,
        unified_psi: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> (f64, String, f64) {
        let _t = Instant::now();
        let (dissipative_health, dissipative_regime, dissipative_entropy_rate) =
            if let Some(ref mut dc) = self.primitive_tier.dissipative_consciousness {
                let energy = prediction_error as f64;
                let info = coherence as f64 * unified_psi;
                dc.update(unified_psi, energy, info, coherence as f64);
                let health = dc.health_score();
                self.carryover.quality.last_dissipative_health = health;
                (
                    health,
                    dc.current_regime().as_str().into(),
                    dc.entropy_production_rate,
                )
            } else {
                (0.0, String::new(), 0.0)
            };
        module_timings.dissipative_consciousness = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Dissipative regime → exploration + learning rate modulation
        // Science: Prigogine (1977) — dissipative structures self-organize at edge of chaos.
        // Use recommend_action() to translate thermodynamic state into cognitive adjustments.
        if let Some(ref dc) = self.primitive_tier.dissipative_consciousness {
            use crate::consciousness::dissipative_consciousness::DissipativeAction;
            match dc.recommend_action() {
                DissipativeAction::Maintain { .. } => {
                    // Optimal regime: slight confidence boost (system is well-organized)
                    self.adjust_confidence("dissipative_maintain", 0.005);
                }
                DissipativeAction::IncreaseActivity {
                    suggested_increase, ..
                } => {
                    // Near equilibrium: boost exploration proportional to suggested increase
                    let explore_boost = (suggested_increase * 0.15).min(0.05) as f32;
                    self.adjust_exploration("dissipative_activity", explore_boost);
                    self.adjust_confidence("dissipative_equilibrium", -0.01);
                }
                DissipativeAction::IncreaseCoherence { .. } => {
                    // Chaotic: suppress exploration, boost learning to restore coherence
                    self.scale_exploration("dissipative_coherence", 0.9);
                    self.scale_lr("dissipative_coherence", 1.05);
                }
                DissipativeAction::IncreaseDifferentiation { .. } => {
                    // Too ordered: nudge exploration up, learning down slightly
                    self.adjust_exploration("dissipative_differentiation", 0.02);
                    self.adjust_confidence("dissipative_ordered", -0.005);
                }
                DissipativeAction::IncreaseIntegration { .. } => {
                    // Too differentiated: boost learning rate for better integration
                    self.scale_lr("dissipative_integration", 1.03);
                }
            }
        }

        (
            dissipative_health,
            dissipative_regime,
            dissipative_entropy_rate,
        )
    }

    /// Consciousness Equation V2: Unified 7-theory formula C(t) = σ × Σ × S × ρ.
    ///
    /// Combines Integration, Binding, Workspace, Attention, Recursion, Efficacy,
    /// Knowledge into a single consciousness score with PAC modulation.
    /// Applies feedback: high consciousness boosts confidence + episodic consolidation;
    /// low consciousness boosts exploration.
    ///
    /// Science: Tononi (2004), Baars (1988), Friston (2010), Graziano (2013).
    /// Now delegates to ConsciousnessEngine (called in cycle.rs).
    /// Reads the cached equation_v2 value — all feedback (confidence, exploration,
    /// episodic consolidation) is already applied by the engine output handler.
    pub(in crate::cognitive_loop) fn compute_equation_v2_phase(
        &mut self,
        _unified_psi: f64,
        _coherence: f32,
        _prediction_error: f32,
        _phi_attention_weight: f32,
        _module_timings: &mut super::ModuleTimings,
    ) -> f64 {
        // Engine owns ConsciousnessEquationV2 and fires every 23 cycles.
        // Feedback (confidence, exploration, episodic replay) applied in cycle.rs.
        self.carryover.consciousness.last_equation_v2_consciousness
    }

    /// Primitive Lattice: Structural metrics from the consciousness tier system.
    ///
    /// Reads height/width from the precomputed lattice (cached after first cycle).
    /// Computes lattice join for active primitives every 7 cycles.
    /// Applies feedback: deep lattice (height > 5) reduces LR on first cycle.
    ///
    /// Science: Davey & Priestley (2002) — lattice theory for knowledge systems.
    pub(in crate::cognitive_loop) fn compute_lattice_phase(
        &mut self,
        active_primitive_names: &[String],
        module_timings: &mut super::ModuleTimings,
    ) -> (usize, usize, Option<String>) {
        let _t = Instant::now();
        let (lattice_height, lattice_width, lattice_join_concept) =
            if let Some(ref lattice) = self.primitive_tier.primitive_lattice {
                // Properties (height/width/modularity) are O(n²–n³) on the lattice graph.
                // The lattice is immutable after construction → compute once on first cycle,
                // cache in stats, and reuse. This eliminates ~31ms/cycle overhead.
                let (height, width) = if self.stats.lattice_height_cached == 0 {
                    let props = lattice.properties();
                    self.stats.lattice_height_cached = props.height;
                    self.stats.lattice_width_cached = props.width;
                    (props.height, props.width)
                } else {
                    (
                        self.stats.lattice_height_cached,
                        self.stats.lattice_width_cached,
                    )
                };

                // FEEDBACK: Lattice height (integration depth) → LR modulation (once)
                if height > 5 && self.stats.total_cycles == 0 {
                    let depth_factor = 1.0 - (height.min(9) as f32 - 5.0) * 0.01;
                    self.carryover.learning.subsystem_lr_factor *= depth_factor;
                }

                // Track B: Lattice join for concept composition (every 7 cycles — join is O(1) via precomputed table)
                let join_concept =
                    if active_primitive_names.len() >= 2 && self.stats.total_cycles % 7 == 0 {
                        let mut best_join: Option<usize> = None;
                        for i in 0..active_primitive_names.len() {
                            for j in (i + 1)..active_primitive_names.len() {
                                if let (Some(a), Some(b)) = (
                                    lattice.element_index_by_name(&active_primitive_names[i]),
                                    lattice.element_index_by_name(&active_primitive_names[j]),
                                ) {
                                    if let Some(idx) = lattice.join(a, b) {
                                        match best_join {
                                            Some(prev)
                                                if lattice.elements[idx].tier
                                                    < lattice.elements[prev].tier =>
                                            {
                                                best_join = Some(idx)
                                            }
                                            None => best_join = Some(idx),
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        best_join.map(|idx| lattice.elements[idx].name.clone())
                    } else {
                        None
                    };

                (height, width, join_concept)
            } else {
                (0, 0, None)
            };
        module_timings.primitive_lattice = _t.elapsed().as_micros() as u64;

        (lattice_height, lattice_width, lattice_join_concept)
    }

    /// Unified Value Evaluator: Eight Harmonies alignment scoring.
    ///
    /// Evaluates cognitive action against fiduciary harmonics every 19 cycles.
    /// Applies feedback: Veto decision drastically reduces learning rate.
    ///
    /// Science: Panksepp (1998) affective neuroscience + value alignment.
    /// Now delegates to EthicsEngine (called in cycle.rs).
    /// Reads cached value evaluator score — all feedback (LR gating)
    /// is applied by the engine output handler.
    pub(in crate::cognitive_loop) fn compute_value_evaluator_phase(
        &mut self,
        _unified_psi: f64,
        _module_timings: &mut super::ModuleTimings,
    ) -> (f64, String, f32) {
        // Engine owns UnifiedValueEvaluator and fires every 19 cycles.
        // Feedback (LR gating, value_gate_applied_count) applied in cycle.rs.
        let score = self.ethics_engine.last_value_score();
        (score, String::new(), 1.0)
    }

    /// Fiduciary Harmonics: Eight Harmonies field coherence + interference detection.
    ///
    /// Drives harmonic levels from consciousness metrics every 11 cycles, detects
    /// and resolves value tensions. Applies feedback: high coherence boosts LR;
    /// interferences reduce prediction confidence.
    ///
    /// Science: Whitehead (1929), Deci & Ryan (2000) — value coherence theory.
    pub(in crate::cognitive_loop) fn compute_fiduciary_harmonics_phase(
        &mut self,
        coherence: f32,
        prediction_error: f32,
        unified_psi: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> (f64, f64, usize) {
        let _t = Instant::now();
        let (harmonic_field_coherence, harmonic_love_resonance, harmonic_interferences) =
            if let Some(ref mut field) = self.primitive_tier.harmonic_field {
                if self.stats.total_cycles % 11 == 0 {
                    // Drive harmonic levels from consciousness metrics
                    use crate::consciousness::harmonics::FiduciaryHarmonic;
                    field.set_level(FiduciaryHarmonic::ResonantCoherence, coherence as f64);
                    field.set_level(
                        FiduciaryHarmonic::EvolutionaryProgression,
                        (prediction_error as f64 * 2.0).clamp(0.0, 1.0),
                    ); // high error = high evolution pressure
                    field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.prediction_confidence,
                    );
                    field.set_level(
                        FiduciaryHarmonic::PanSentientFlourishing,
                        unified_psi.clamp(0.0, 1.0),
                    );
                    field.detect_interferences();
                    // Resolve interferences if any were detected
                    if !field.interferences.is_empty() {
                        if let Some(ref resolver) = self.primitive_tier.harmonic_resolver {
                            let _resolution = resolver.resolve(field);
                        }
                    }
                    self.carryover.consciousness.last_harmonic_coherence = field.field_coherence;
                    (
                        field.field_coherence,
                        field.infinite_love_resonance,
                        field.interferences.len(),
                    )
                } else {
                    (self.carryover.consciousness.last_harmonic_coherence, 0.0, 0)
                }
            } else {
                (0.0, 0.0, 0)
            };
        module_timings.harmonics = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Harmonic coherence → LR stability (coherent values = stable learning)
        if harmonic_field_coherence > HARMONIC_FIELD_BOOST_THRESHOLD as f64 {
            let harmony_boost = 1.0
                + ((harmonic_field_coherence - HARMONIC_FIELD_BOOST_THRESHOLD as f64)
                    * HARMONIC_FIELD_BOOST_FACTOR as f64) as f32; // up to +2%
            self.carryover.learning.subsystem_lr_factor *= harmony_boost;
        }
        // FEEDBACK: Harmonic interferences → reduce confidence (value tensions = uncertainty)
        if harmonic_interferences > 0 {
            let interference_penalty = (harmonic_interferences.min(3) as f32) * 0.01; // -1% per interference
            self.adjust_confidence("harmonic_interference", -interference_penalty);
        }

        (
            harmonic_field_coherence,
            harmonic_love_resonance,
            harmonic_interferences,
        )
    }

    /// Causal Self-Explanation: Pearl causal model of primitive→Phi effects.
    ///
    /// Learns which primitives cause which Phi changes every 23 cycles by constructing
    /// reasoning chains from active primitives and feeding them to the causal explainer.
    ///
    /// Science: Pearl (2009) — causal inference, Woodward (2003) — interventionism.
    pub(in crate::cognitive_loop) fn compute_causal_self_explanation_phase(
        &mut self,
        hv16_cached: symthaea_core::hdc::BinaryHV,
        active_primitive_names: &[String],
        primitive_psi: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> (usize, f64) {
        let _t = Instant::now();
        let (causal_relations_count, causal_avg_confidence) = if let Some(ref mut explainer) =
            self.primitive_tier.causal_explainer
        {
            if self.stats.total_cycles % 23 == 0
                && self.stats.total_cycles > 0
                && !active_primitive_names.is_empty()
            {
                // Construct PrimitiveExecution entries from active primitives
                if let Some(ref mut processor) = self.primitive_tier.primitive_processor {
                    let timestamp = self.stats.total_cycles as f64 * 0.02;
                    let state = processor.process_input(&hv16_cached, timestamp);
                    let chain = {
                        let mut c = crate::consciousness::primitive_reasoning::ReasoningChain::new(
                            hv16_cached,
                        );
                        for ap in state.all_active().iter().take(4) {
                            let exec = crate::consciousness::primitive_reasoning::PrimitiveExecution {
                                    primitive: ap.primitive.clone(),
                                    input: hv16_cached,
                                    output: hv16_cached.bind(&crate::hdc::BinaryHV::random(
                                        self.stats.total_cycles as u64,
                                    )),
                                    transformation: crate::consciousness::primitive_reasoning::TransformationType::Bind,
                                    phi_contribution: primitive_psi * ap.activation,
                                    timestamp,
                                };
                            c.executions.push(exec);
                        }
                        c
                    };
                    explainer.learn_from_chain(&chain, "cognitive_cycle");
                }
                // Only call summarize_understanding on learning cycles (expensive)
                let summary = explainer.summarize_understanding();
                self.carryover.history.last_causal_relations = summary.total_causal_relations;
                self.carryover.history.last_causal_confidence = summary.average_confidence;
                (summary.total_causal_relations, summary.average_confidence)
            } else {
                // Non-learning cycle: return cached summary (avoids ~880µs/cycle)
                (
                    self.carryover.history.last_causal_relations,
                    self.carryover.history.last_causal_confidence,
                )
            }
        } else {
            (0, 0.0)
        };
        module_timings.causal_explanation = _t.elapsed().as_micros() as u64;

        (causal_relations_count, causal_avg_confidence)
    }

    /// Consciousness Profile: multi-dimensional assessment with synergy detection.
    ///
    /// Maintains ring buffer of last 4 BinaryHVs, computes 5-axis profile (Phi,
    /// gradient, entropy, complexity, coherence) every 47 cycles, then detects
    /// non-linear dimension synergies.
    ///
    /// Science: Tononi (2004), Koch (2012).
    pub(in crate::cognitive_loop) fn compute_consciousness_profile_phase(
        &mut self,
        hv16_cached: symthaea_core::hdc::BinaryHV,
        module_timings: &mut super::ModuleTimings,
    ) -> (f64, f64, usize) {
        let _t = Instant::now();
        // Maintain ring buffer of last 4 BinaryHVs for multi-component profile
        // Capacity bound: 4 elements — evict before push to prevent transient over-capacity
        if self.carryover.history.recent_hvs.len() >= 4 {
            self.carryover.history.recent_hvs.pop_front();
        }
        self.carryover.history.recent_hvs.push_back(hv16_cached);
        let result = if self.stats.total_cycles % 47 == 0
            && self.primitive_tier.primitive_processor.is_some()
        {
            let profile =
                crate::consciousness::consciousness_profile::ConsciousnessProfile::from_components(
                    self.carryover.history.recent_hvs.make_contiguous(),
                );
            let composite = profile.composite;
            let synergy =
                crate::consciousness::dimension_synergies::SynergyProfile::from_base(profile);
            self.carryover.history.last_profile_composite = composite;
            self.carryover.history.last_synergy_composite = synergy.enhanced_composite;
            self.carryover.history.last_emergent_count = synergy.emergent_properties.len();
            (
                composite,
                synergy.enhanced_composite,
                synergy.emergent_properties.len(),
            )
        } else {
            (
                self.carryover.history.last_profile_composite,
                self.carryover.history.last_synergy_composite,
                self.carryover.history.last_emergent_count,
            )
        };
        module_timings.consciousness_profile = _t.elapsed().as_micros() as u64;
        result
    }

    /// Phi Validation: empirical validation of Phi against synthetic states.
    ///
    /// Runs a validation study every 499 cycles (co-prime, expensive). Adjusts
    /// spectral weight based on correlation quality.
    ///
    /// Science: Casali et al. (2013).
    pub(in crate::cognitive_loop) fn compute_phi_validation_phase(
        &mut self,
        module_timings: &mut super::ModuleTimings,
    ) -> f64 {
        let _t = Instant::now();
        let correlation = if let Some(ref mut validator) = self.primitive_tier.phi_validation {
            if self.stats.total_cycles % 997 == 0 && self.stats.total_cycles >= 997 {
                let results = validator.run_validation_study(10);
                let r = results.pearson_r;
                self.carryover.quality.phi_validation_correlation = r;
                // Adjust spectral weight based on validation quality
                if r > PHI_VALIDATION_HIGH_THRESHOLD {
                    self.carryover.quality.phi_spectral_weight = (SPECTRAL_WEIGHT_BASE
                        + (r - PHI_VALIDATION_HIGH_THRESHOLD) as f32 * SPECTRAL_WEIGHT_SCALE)
                        .clamp(0.4, 0.8);
                } else if r < PHI_VALIDATION_LOW_THRESHOLD && r > 0.0 {
                    self.carryover.quality.phi_spectral_weight = (SPECTRAL_WEIGHT_BASE
                        - (PHI_VALIDATION_LOW_THRESHOLD - r) as f32 * SPECTRAL_WEIGHT_SCALE)
                        .clamp(0.4, 0.8);
                }
                r
            } else {
                self.carryover.quality.phi_validation_correlation
            }
        } else {
            0.0
        };
        module_timings.phi_validation = _t.elapsed().as_micros() as u64;
        correlation
    }

    /// Epistemic Conflict: multi-theory conflict detection + Φ_eff reliability weighting.
    ///
    /// Compares IIT, GWT, AST, PP, RPT, 4E scores every 97 cycles. Computes
    /// Φ_eff = Φ × R^γ. High conflict count triggers epistemic reasoning override.
    ///
    /// Science: IIT (Tononi 2015), GWT (Baars 1988), AST (Graziano 2013).
    pub(in crate::cognitive_loop) fn compute_epistemic_conflict_phase(
        &mut self,
        unified_psi: f64,
        coherence: f32,
        prediction_error: f32,
        module_timings: &mut super::ModuleTimings,
    ) -> (f64, usize) {
        let _t = Instant::now();
        let result = if let (Some(detector), Some(calibrator)) = (
            &mut self.primitive_tier.epistemic_conflict_detector,
            &self.primitive_tier.theory_calibrator,
        ) {
            if self.stats.total_cycles % 97 == 0 && self.stats.total_cycles > 0 {
                use crate::consciousness::epistemic_conflict::{
                    ConflictMatrix, MultiTheoryMetrics, compute_phi_eff,
                };
                let metrics = MultiTheoryMetrics {
                    phi: unified_psi,
                    gwt: coherence as f64 * 0.8,
                    ast: coherence as f64,
                    pp: 1.0 - prediction_error as f64,
                    rpt: coherence as f64 * 0.9,
                    embodiment: self.carryover.consciousness.body_phi_modulation,
                    unified: unified_psi,
                };
                let matrix: ConflictMatrix = detector.detect(&metrics);
                let phi_eff_result = compute_phi_eff(&metrics, calibrator);
                self.carryover.quality.last_phi_eff = phi_eff_result.phi_eff;
                self.carryover.quality.last_epistemic_conflict_count = matrix.conflicts.len();
                if matrix.conflicts.len() > 5 {
                    self.carryover.quality.epistemic_reasoning_override = true;
                }
                (phi_eff_result.phi_eff, matrix.conflicts.len())
            } else {
                (self.carryover.quality.last_phi_eff, 0)
            }
        } else {
            (0.0, 0)
        };
        module_timings.epistemic_conflict = _t.elapsed().as_micros() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleUrgency};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    fn make_cycle_state<'a>(
        compressed_state: &'a [f32],
        output: &'a [f32],
        hv: &'a symthaea_core::hdc::BinaryHV,
        input: &'a str,
    ) -> super::super::CycleState<'a> {
        super::super::CycleState {
            compressed_state,
            output,
            prediction_error: 0.2,
            coherence: 0.5,
            unified_psi: 0.3,
            phi_attention_weight: 0.5,
            hv16_cached: hv,
            input,
            urgency: CycleUrgency::Normal,
            attention_budget_exceeded: false,
            predictive_budget_gated: false,
            #[cfg(feature = "vision-manifold")]
            scene_recognized: false,
            #[cfg(feature = "semantic-encoder")]
            semantic_embedding: None,
        }
    }

    // ── compute_consciousness_metrics ─────────────────────────────────

    #[test]
    fn consciousness_metrics_default_does_not_panic() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.5f32; 64];
        let output = vec![0.0f32; 64];
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let state = make_cycle_state(&compressed, &output, &hv, "test input");
        let mut timings = super::super::ModuleTimings::default();
        let metrics = s.compute_consciousness_metrics(&state, &mut timings);
        assert!(metrics.primitive_psi.is_finite());
        assert!(metrics.value_evaluator_score.is_finite());
    }

    #[test]
    fn consciousness_metrics_all_fields_finite() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.1f32; 64];
        let output = vec![0.0f32; 64];
        let hv = symthaea_core::hdc::BinaryHV::random(99);
        let state = make_cycle_state(&compressed, &output, &hv, "fields check");
        let mut timings = super::super::ModuleTimings::default();
        let m = s.compute_consciousness_metrics(&state, &mut timings);
        assert!(m.temporal_continuity.is_finite());
        assert!(m.consciousness_profile_composite.is_finite());
        assert!(m.synergy_enhanced_composite.is_finite());
        assert!(m.harmonic_field_coherence.is_finite());
        assert!(m.harmonic_love_resonance.is_finite());
        assert!(m.dissipative_health.is_finite());
        assert!(m.dissipative_entropy_rate.is_finite());
        assert!(m.epistemic_phi_eff.is_finite());
        assert!(m.equation_v2_consciousness.is_finite());
        assert!(m.phi_validation_correlation.is_finite());
        assert!(m.adaptive_reasoning_phi.is_finite());
        assert!(m.epistemic_quality.is_finite());
    }

    #[test]
    fn consciousness_metrics_empty_input() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.0f32; 64];
        let output = vec![0.0f32; 64];
        let hv = symthaea_core::hdc::BinaryHV::zero();
        let state = make_cycle_state(&compressed, &output, &hv, "");
        let mut timings = super::super::ModuleTimings::default();
        let m = s.compute_consciousness_metrics(&state, &mut timings);
        assert!(m.context_phi_weight.is_finite());
        assert!(m.reasoning_chain_confidence.is_finite());
    }

    #[test]
    fn consciousness_metrics_zero_psi_and_coherence() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![0.0f32; 64];
        let output = vec![0.0f32; 64];
        let hv = symthaea_core::hdc::BinaryHV::zero();
        let mut state = make_cycle_state(&compressed, &output, &hv, "zero psi");
        state.unified_psi = 0.0;
        state.coherence = 0.0;
        state.prediction_error = 0.0;
        let mut timings = super::super::ModuleTimings::default();
        let m = s.compute_consciousness_metrics(&state, &mut timings);
        assert!(m.primitive_psi.is_finite());
        assert!(m.value_gate_factor.is_finite());
    }

    #[test]
    fn consciousness_metrics_high_prediction_error() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let compressed = vec![1.0f32; 64];
        let output = vec![0.5f32; 64];
        let hv = symthaea_core::hdc::BinaryHV::random(7);
        let mut state = make_cycle_state(&compressed, &output, &hv, "high error");
        state.prediction_error = 1.0;
        let mut timings = super::super::ModuleTimings::default();
        let m = s.compute_consciousness_metrics(&state, &mut timings);
        assert!(m.dissipative_health.is_finite());
        assert!(m.harmonic_field_coherence.is_finite());
    }

    // ── compute_temporal_primitives_phase ──────────────────────────────

    #[test]
    fn temporal_primitives_default_returns_zeros() {
        let mut s = make_service();
        s.stats.total_cycles = 1;
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let mut timings = super::super::ModuleTimings::default();
        let (chains, continuity, max_len, cycle_nums, codebook, replay) =
            s.compute_temporal_primitives_phase(hv, 0.3, 0.5, &mut timings);
        assert_eq!(chains, 0);
        assert!((continuity - 0.0).abs() < f64::EPSILON);
        assert_eq!(max_len, 0);
        assert!(cycle_nums.is_empty());
        assert!(codebook.is_empty());
        assert!(!replay);
    }

    // ── compute_lattice_phase ─────────────────────────────────────────

    #[test]
    fn lattice_phase_no_lattice_returns_zeros() {
        let mut config = super::super::CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = false;
        let mut s = super::super::CognitiveLoopService::new(config).unwrap();
        let primitives: Vec<String> = vec![];
        let mut timings = super::super::ModuleTimings::default();
        let (height, width, join) = s.compute_lattice_phase(&primitives, &mut timings);
        assert_eq!(height, 0);
        assert_eq!(width, 0);
        assert!(join.is_none());
    }

    // ── compute_value_evaluator_phase ─────────────────────────────────

    #[test]
    fn value_evaluator_returns_defaults() {
        let mut s = make_service();
        let mut timings = super::super::ModuleTimings::default();
        let (score, decision, gate) = s.compute_value_evaluator_phase(0.5, &mut timings);
        assert!(score.is_finite());
        assert!(gate.is_finite());
        // Decision is empty string when delegated to engine
        let _ = decision; // just verify it compiles
    }

    // ── compute_fiduciary_harmonics_phase ──────────────────────────────

    #[test]
    fn fiduciary_harmonics_no_field_returns_zeros() {
        let mut config = super::super::CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = false;
        let mut s = super::super::CognitiveLoopService::new(config).unwrap();
        let mut timings = super::super::ModuleTimings::default();
        let (coherence, love, interferences) =
            s.compute_fiduciary_harmonics_phase(0.5, 0.2, 0.3, &mut timings);
        assert!((coherence - 0.0).abs() < f64::EPSILON);
        assert!((love - 0.0).abs() < f64::EPSILON);
        assert_eq!(interferences, 0);
    }

    // ── compute_dissipative_phase ─────────────────────────────────────

    #[test]
    fn dissipative_phase_no_module_returns_zeros() {
        let mut config = super::super::CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = false;
        let mut s = super::super::CognitiveLoopService::new(config).unwrap();
        let mut timings = super::super::ModuleTimings::default();
        let (health, regime, entropy_rate) =
            s.compute_dissipative_phase(0.2, 0.5, 0.3, &mut timings);
        assert!((health - 0.0).abs() < f64::EPSILON);
        assert!(regime.is_empty());
        assert!((entropy_rate - 0.0).abs() < f64::EPSILON);
    }

    // ── compute_equation_v2_phase ─────────────────────────────────────

    #[test]
    fn equation_v2_reads_carryover() {
        let mut s = make_service();
        s.carryover.consciousness.last_equation_v2_consciousness = 0.42;
        let mut timings = super::super::ModuleTimings::default();
        let result = s.compute_equation_v2_phase(0.5, 0.5, 0.2, 0.4, &mut timings);
        assert!((result - 0.42).abs() < f64::EPSILON);
    }

    // ── compute_causal_self_explanation_phase ──────────────────────────

    #[test]
    fn causal_self_explanation_no_explainer_returns_zeros() {
        let mut s = make_service();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let primitives = vec!["test".to_string()];
        let mut timings = super::super::ModuleTimings::default();
        let (count, confidence) =
            s.compute_causal_self_explanation_phase(hv, &primitives, 0.3, &mut timings);
        assert_eq!(count, 0);
        assert!((confidence - 0.0).abs() < f64::EPSILON);
    }

    // ── Multiple cycles stability ─────────────────────────────────────

    #[test]
    fn consciousness_metrics_multiple_cycles_stable() {
        let mut s = make_service();
        let compressed = vec![0.3f32; 64];
        let output = vec![0.1f32; 64];
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        for cycle in 1..=5 {
            s.stats.total_cycles = cycle;
            let state = make_cycle_state(&compressed, &output, &hv, "stable input");
            let mut timings = super::super::ModuleTimings::default();
            let m = s.compute_consciousness_metrics(&state, &mut timings);
            assert!(
                m.primitive_psi.is_finite(),
                "cycle {cycle}: primitive_psi not finite"
            );
            assert!(
                m.consciousness_profile_composite.is_finite(),
                "cycle {cycle}: profile not finite"
            );
        }
    }
}
