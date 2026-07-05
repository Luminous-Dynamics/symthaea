// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Free functions for parallel post-processing branches (Phase 5).
//!
//! These are NOT methods on `&mut self` — they take explicit disjoint borrows
//! so the borrow checker is satisfied for `rayon::join`'s Send requirements.

use std::fmt::Write;
use std::time::Instant;

use crate::causal::CausalLoopEnhancer;
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_consciousness::PrimitiveConsciousnessState;
use crate::consciousness::stability_regime::{RegimeTransition, StabilityRegimeProcessor};
use crate::memory::semantic_memory::SemanticMemory;
use symthaea_core::hdc::binary_hv::BinaryHV;

use super::super::{
    ClosedLearningLoop, CognitiveLoopService, CycleLearningResult, CycleUrgency,
    EpisodicMemoryBridge,
};

/// Context for the episodic/learning parallel branch (read-only values).
pub(in crate::cognitive_loop) struct EpisodicLearningContext<'a> {
    pub prediction_error: f32,
    pub in_flow: bool,
    pub input: &'a str,
    pub compressed_state: &'a [f32],
    pub emotional_valence: f32,
    pub phi: f32,
    pub total_cycles: usize,
    pub smoothed_coh: f64,
    pub detected_primitives: &'a [String],
    pub memory_context_boost: f32,
    pub wm_importance_boost: f32,
    /// Full HDC vector for PolarQuant compression into episodic memory.
    #[cfg(feature = "turbo-quant")]
    pub full_hdv: Option<&'a symthaea_core::hdc::unified_hv::ContinuousHV>,
}

/// Run stability regime processing (pre-parallel).
///
/// Processes input through the stability regime and seeds neighbor exploration
/// for any crystallized primitives. Returns timing in microseconds.
pub(in crate::cognitive_loop) fn run_stability_regime(
    regime: &mut StabilityRegimeProcessor,
    discovery: &mut crate::consciousness::primitive_discovery::PrimitiveDiscoveryService,
    hv16: &BinaryHV,
    delta_t: f32,
    total_cycles: usize,
    urgency: CycleUrgency,
) -> u64 {
    let t = Instant::now();
    if urgency.should_run(total_cycles, 3, 5, 20) {
        let timestamp = total_cycles as f64 * delta_t as f64;
        let (_regime_state, transitions) = regime.process_input(hv16, delta_t, timestamp);

        for transition in &transitions {
            if let RegimeTransition::Crystallized {
                primitive_name,
                encoding,
            } = transition
            {
                discovery.seed_neighbor_exploration(primitive_name, encoding);
            }
        }
    }
    t.elapsed().as_micros() as u64
}

/// Parallel Branch A: Semantic memory storage + causal enhancement.
pub(in crate::cognitive_loop) fn parallel_semantic_causal(
    semantic_memory: &mut SemanticMemory,
    causal_enhancer: &mut Option<CausalLoopEnhancer>,
    semantic_hdc: Vec<f32>,
    compressed_state: &[f32],
    output: &[f32],
    prediction_error: f32,
    total_cycles: usize,
) -> Option<crate::memory::semantic_memory::SemanticEntry> {
    // Semantic memory: store HDC vector + prediction error for future similarity lookup
    let evicted = semantic_memory.store_with_timestamp(
        semantic_hdc,
        prediction_error,
        None,
        total_cycles as u64,
    );

    // Causal enhancement: track (input, output) pairs and discover structure
    if let Some(enhancer) = causal_enhancer {
        enhancer.record_cycle_from_f32(compressed_state, output);

        if enhancer.should_discover() {
            let causal_graph = enhancer.run_discovery();

            if !causal_graph.is_empty() {
                tracing::info!(
                    edges = causal_graph.edges.len(),
                    cycle = total_cycles,
                    "Causal structure discovered in cognitive loop"
                );
                enhancer.log_discoveries();
            }
        }
    }

    evicted
}

/// Parallel Branch B: Episodic memory + resonator storage + primitive-belief bridge + closed learning.
///
/// Returns the memory context boost that should be applied to prediction_confidence
/// after the rayon::join completes (via `adjust_confidence`).
#[allow(clippy::too_many_arguments)]
pub(in crate::cognitive_loop) fn parallel_episodic_learning(
    episodic_memory: &mut EpisodicMemoryBridge,
    resonator_memory: &mut Option<crate::dynamics::resonator::ResonatorMemory>,
    primitive_belief_bridge: &mut PrimitiveBeliefBridge,
    prev_primitive_state: &mut Option<PrimitiveConsciousnessState>,
    fep_learning_signal: &mut f32,
    closed_learning_loop: &mut ClosedLearningLoop,
    ctx: &EpisodicLearningContext<'_>,
    cycle_learning_result: CycleLearningResult,
) -> f32 {
    // Episodic memory: encode significant experiences.
    // IIT gate: consciousness (Phi) is necessary for memory formation
    // (Tononi & Koch 2015). Require minimum Phi for consolidation.
    let phi_sufficient = ctx.phi > super::super::thresholds::PHI_EPISODIC_CONSOLIDATION_MIN;
    if phi_sufficient
        && (ctx.prediction_error > super::super::thresholds::PREDICTION_ERROR_EPISODIC_MIN
            || ctx.in_flow)
    {
        let hdv_sample: Vec<f32> =
            ctx.compressed_state[..64.min(ctx.compressed_state.len())].to_vec();
        let mem_id = episodic_memory.encode(
            ctx.input,
            hdv_sample,
            ctx.emotional_valence,
            ctx.phi,
            ctx.total_cycles,
        );

        // Attach PolarQuant-compressed full HDC vector for high-fidelity recall.
        // Reduces 64KB (16,384 x f32) to ~8KB at 4-bit quantization.
        #[cfg(feature = "turbo-quant")]
        if let Some(full_hv) = ctx.full_hdv {
            if let Ok(compressor) = crate::hdc::hv_compression::HvCompressor::new(4) {
                let compressed = compressor.compress(full_hv);
                episodic_memory.attach_compressed_hv(mem_id, compressed);
            }
        }
    }

    // Resonator memory: store with bound attributes for factorized recall
    if let Some(res_mem) = resonator_memory {
        let res_dim_ok = ctx.compressed_state.len() == res_mem.resonator.config.dim;
        if res_dim_ok
            && (ctx.prediction_error > super::super::thresholds::PREDICTION_ERROR_EPISODIC_MIN
                || ctx.in_flow)
        {
            // Quantize valence -> nearest band
            let val_label = if ctx.emotional_valence
                > super::super::thresholds::EMOTIONAL_VALENCE_POSITIVE_BIN
            {
                "positive"
            } else if ctx.emotional_valence
                < super::super::thresholds::EMOTIONAL_VALENCE_NEGATIVE_BIN
            {
                "negative"
            } else {
                "neutral"
            };

            // Quantize phi -> nearest band
            let phi_label = if ctx.phi > super::super::thresholds::PHI_QUANTIZATION_HIGH {
                "high"
            } else if ctx.phi > super::super::thresholds::PHI_QUANTIZATION_MEDIUM {
                "medium"
            } else {
                "low"
            };

            // Clone codebook HVs in a narrow scope to end the immutable borrow
            // before the mutable `store()` call. The clone is required because
            // `store(&mut self)` conflicts with the immutable codebook borrow.
            let val_hv = res_mem
                .resonator
                .codebooks
                .get(1)
                .and_then(|cb| cb.symbols.iter().find(|(l, _)| l == val_label))
                .map(|(_, hv)| hv.clone());
            let phi_hv = res_mem
                .resonator
                .codebooks
                .get(2)
                .and_then(|cb| cb.symbols.iter().find(|(l, _)| l == phi_label))
                .map(|(_, hv)| hv.clone());

            if let (Some(v_hv), Some(p_hv)) = (val_hv, phi_hv) {
                // Pre-sized String avoids reallocation (ep_ + up to 20 digits)
                let mut ep_key = String::with_capacity(23);
                let _ = write!(ep_key, "ep_{}", ctx.total_cycles);
                res_mem.store(
                    ep_key.as_str(),
                    &[
                        ("content", "input", ctx.compressed_state),
                        ("valence", val_label, &v_hv),
                        ("phi_level", phi_label, &p_hv),
                    ],
                    ctx.phi + ctx.wm_importance_boost,
                );
            }
        }
    }

    // Primitive-Belief Bridge: map primitives to beliefs, compute TD signals
    let prim_state = CognitiveLoopService::build_primitive_state(
        ctx.detected_primitives,
        ctx.smoothed_coh,
        ctx.total_cycles as f64,
    );

    if let &mut Some(ref prev_state) = prev_primitive_state {
        let pred_error = primitive_belief_bridge.compute_prediction_error(prev_state, &prim_state);
        let td_signal = primitive_belief_bridge.td_error_signal(&pred_error);
        *fep_learning_signal +=
            td_signal as f32 * super::super::thresholds::TD_ERROR_FEP_COUPLING_SCALE;
        *fep_learning_signal = fep_learning_signal.clamp(-1.0, 1.0);
    }

    *prev_primitive_state = Some(prim_state);

    // Closed learning loop: update Q-values, modulated by FEP learning signal.
    // Friston (2010): plasticity scales with expected information gain.
    let plasticity =
        1.0 + *fep_learning_signal * super::super::thresholds::FEP_PLASTICITY_SCALING_FACTOR; // [-1,1] → [0.5, 1.5]
    closed_learning_loop.update_with_plasticity(cycle_learning_result, plasticity);

    // Return the memory context boost for deferred application via adjust_confidence
    ctx.memory_context_boost
}

// ═══════════════════════════════════════════════════════════════════════════
// Consciousness metrics parallel branches (Phase "consciousness metrics").
//
// These free functions extract disjoint borrows from PrimitiveTierManager
// so that `rayon::join` can run them concurrently. Feedback effects
// (adjust_confidence, scale_lr, adjust_exploration) are deferred and
// returned as `DeferredFeedback` for application after the join completes.
// ═══════════════════════════════════════════════════════════════════════════

/// Deferred feedback action to be applied to CognitiveLoopService after parallel join.
pub(in crate::cognitive_loop) enum DeferredFeedback {
    AdjustConfidence(&'static str, f32),
    ScaleLr(&'static str, f32),
    AdjustExploration(&'static str, f32),
    ScaleExploration(&'static str, f32),
    ScaleSubsystemLr(f32),
}

/// Results from consciousness metrics Branch A:
/// phi_validation + dissipative_consciousness + consciousness_profile.
pub(in crate::cognitive_loop) struct ConsciousnessMetricsBranchA {
    // Phi validation
    pub phi_validation_correlation: f64,
    pub phi_validation_timing: u64,
    // Dissipative consciousness
    pub dissipative_health: f64,
    pub dissipative_regime: String,
    pub dissipative_entropy_rate: f64,
    pub dissipative_timing: u64,
    // Consciousness profile
    pub consciousness_profile_composite: f64,
    pub synergy_enhanced_composite: f64,
    pub emergent_properties_count: usize,
    pub consciousness_profile_timing: u64,
    // Cached values to write back to carryover
    pub new_phi_validation_correlation: Option<f64>,
    pub new_phi_spectral_weight: Option<f32>,
    pub new_dissipative_health: Option<f64>,
    pub new_profile_composite: Option<f64>,
    pub new_synergy_composite: Option<f64>,
    pub new_emergent_count: Option<usize>,
    // Deferred feedback
    pub deferred: Vec<DeferredFeedback>,
}

/// Results from consciousness metrics Branch B:
/// fiduciary_harmonics + primitive_reasoning + epistemic_conflict.
pub(in crate::cognitive_loop) struct ConsciousnessMetricsBranchB {
    // Fiduciary harmonics
    pub harmonic_field_coherence: f64,
    pub harmonic_love_resonance: f64,
    pub harmonic_interferences: usize,
    pub harmonics_timing: u64,
    // Primitive reasoning
    pub reasoning_chain_confidence: f32,
    pub reasoning_chain_depth: usize,
    pub reasoning_timing: u64,
    // Epistemic conflict
    pub epistemic_phi_eff: f64,
    pub epistemic_conflict_count: usize,
    pub epistemic_conflict_timing: u64,
    // Cached values to write back to carryover
    pub new_harmonic_coherence: Option<f64>,
    pub new_phi_eff: Option<f64>,
    pub new_epistemic_conflict_count: Option<usize>,
    pub epistemic_reasoning_override: bool,
    // Deferred feedback
    pub deferred: Vec<DeferredFeedback>,
}

/// Branch A: phi_validation + dissipative + consciousness_profile.
///
/// Accesses disjoint fields: `phi_validation`, `dissipative_consciousness`,
/// `primitive_processor` (read-only), plus carryover snapshots.
#[allow(clippy::too_many_arguments)]
pub(in crate::cognitive_loop) fn parallel_consciousness_branch_a(
    phi_validation: &mut Option<crate::consciousness::phi_validation::PhiValidationFramework>,
    dissipative_consciousness: &mut Option<
        crate::consciousness::dissipative_consciousness::DissipativeConsciousness,
    >,
    has_primitive_processor: bool,
    hv16_cached: BinaryHV,
    recent_hvs: &mut std::collections::VecDeque<BinaryHV>,
    prediction_error: f32,
    coherence: f32,
    unified_psi: f64,
    total_cycles: usize,
    // Snapshot of carryover values (read)
    last_phi_validation_correlation: f64,
    last_phi_spectral_weight: f32,
    last_dissipative_health: f64,
    last_profile_composite: f64,
    last_synergy_composite: f64,
    last_emergent_count: usize,
) -> ConsciousnessMetricsBranchA {
    use super::super::thresholds::{
        PHI_VALIDATION_HIGH_THRESHOLD, PHI_VALIDATION_LOW_THRESHOLD, SPECTRAL_WEIGHT_BASE,
        SPECTRAL_WEIGHT_SCALE,
    };
    let mut deferred = Vec::new();

    // ── Phi validation ──────────────────────────────────────────────────
    let t = Instant::now();
    let (phi_validation_correlation, new_phi_validation, new_spectral) =
        if let Some(validator) = phi_validation {
            if total_cycles % 997 == 0 && total_cycles >= 997 {
                let results = validator.run_validation_study(10);
                let r = results.pearson_r;
                let mut new_spectral = None;
                if r > PHI_VALIDATION_HIGH_THRESHOLD {
                    new_spectral = Some(
                        (SPECTRAL_WEIGHT_BASE
                            + (r - PHI_VALIDATION_HIGH_THRESHOLD) as f32 * SPECTRAL_WEIGHT_SCALE)
                            .clamp(0.4, 0.8),
                    );
                } else if r < PHI_VALIDATION_LOW_THRESHOLD && r > 0.0 {
                    new_spectral = Some(
                        (SPECTRAL_WEIGHT_BASE
                            - (PHI_VALIDATION_LOW_THRESHOLD - r) as f32 * SPECTRAL_WEIGHT_SCALE)
                            .clamp(0.4, 0.8),
                    );
                }
                (r, Some(r), new_spectral)
            } else {
                (last_phi_validation_correlation, None, None)
            }
        } else {
            (0.0, None, None)
        };
    let phi_validation_timing = t.elapsed().as_micros() as u64;

    // ── Dissipative consciousness ───────────────────────────────────────
    let t = Instant::now();
    let (dissipative_health, dissipative_regime, dissipative_entropy_rate, new_dissipative) =
        if let Some(dc) = dissipative_consciousness {
            let energy = prediction_error as f64;
            let info = coherence as f64 * unified_psi;
            dc.update(unified_psi, energy, info, coherence as f64);
            let health = dc.health_score();

            // Deferred feedback from recommend_action()
            use crate::consciousness::dissipative_consciousness::DissipativeAction;
            match dc.recommend_action() {
                DissipativeAction::Maintain { .. } => {
                    deferred.push(DeferredFeedback::AdjustConfidence(
                        "dissipative_maintain",
                        0.005,
                    ));
                }
                DissipativeAction::IncreaseActivity {
                    suggested_increase, ..
                } => {
                    let explore_boost = (suggested_increase * 0.15).min(0.05) as f32;
                    deferred.push(DeferredFeedback::AdjustExploration(
                        "dissipative_activity",
                        explore_boost,
                    ));
                    deferred.push(DeferredFeedback::AdjustConfidence(
                        "dissipative_equilibrium",
                        -0.01,
                    ));
                }
                DissipativeAction::IncreaseCoherence { .. } => {
                    deferred.push(DeferredFeedback::ScaleExploration(
                        "dissipative_coherence",
                        0.9,
                    ));
                    deferred.push(DeferredFeedback::ScaleLr("dissipative_coherence", 1.05));
                }
                DissipativeAction::IncreaseDifferentiation { .. } => {
                    deferred.push(DeferredFeedback::AdjustExploration(
                        "dissipative_differentiation",
                        0.02,
                    ));
                    deferred.push(DeferredFeedback::AdjustConfidence(
                        "dissipative_ordered",
                        -0.005,
                    ));
                }
                DissipativeAction::IncreaseIntegration { .. } => {
                    deferred.push(DeferredFeedback::ScaleLr("dissipative_integration", 1.03));
                }
            }

            (
                health,
                dc.current_regime().as_str().into(),
                dc.entropy_production_rate,
                Some(health),
            )
        } else {
            (0.0, String::new(), 0.0, None)
        };
    let dissipative_timing = t.elapsed().as_micros() as u64;

    // ── Consciousness profile ───────────────────────────────────────────
    let t = Instant::now();
    if recent_hvs.len() >= 4 {
        recent_hvs.pop_front();
    }
    recent_hvs.push_back(hv16_cached);
    let (
        consciousness_profile_composite,
        synergy_enhanced_composite,
        emergent_properties_count,
        new_profile,
        new_synergy,
        new_emergent,
    ) = if total_cycles % 47 == 0 && has_primitive_processor {
        let profile =
            crate::consciousness::consciousness_profile::ConsciousnessProfile::from_components(
                recent_hvs.make_contiguous(),
            );
        let composite = profile.composite;
        let synergy = crate::consciousness::dimension_synergies::SynergyProfile::from_base(profile);
        (
            composite,
            synergy.enhanced_composite,
            synergy.emergent_properties.len(),
            Some(composite),
            Some(synergy.enhanced_composite),
            Some(synergy.emergent_properties.len()),
        )
    } else {
        (
            last_profile_composite,
            last_synergy_composite,
            last_emergent_count,
            None,
            None,
            None,
        )
    };
    let consciousness_profile_timing = t.elapsed().as_micros() as u64;

    ConsciousnessMetricsBranchA {
        phi_validation_correlation,
        phi_validation_timing,
        dissipative_health,
        dissipative_regime,
        dissipative_entropy_rate,
        dissipative_timing,
        consciousness_profile_composite,
        synergy_enhanced_composite,
        emergent_properties_count,
        consciousness_profile_timing,
        new_phi_validation_correlation: new_phi_validation,
        new_phi_spectral_weight: new_spectral,
        new_dissipative_health: new_dissipative,
        new_profile_composite: new_profile,
        new_synergy_composite: new_synergy,
        new_emergent_count: new_emergent,
        deferred,
    }
}

/// Branch B: fiduciary_harmonics + primitive_reasoning + epistemic_conflict.
///
/// Accesses disjoint fields: `harmonic_field`, `harmonic_resolver`,
/// `primitive_reasoner`, `epistemic_conflict_detector`, `theory_calibrator`.
#[allow(clippy::too_many_arguments)]
pub(in crate::cognitive_loop) fn parallel_consciousness_branch_b(
    harmonic_field: &mut Option<crate::consciousness::harmonics::HarmonicField>,
    harmonic_resolver: &Option<crate::consciousness::harmonics::HarmonicResolver>,
    primitive_reasoner: &mut Option<crate::consciousness::primitive_reasoning::PrimitiveReasoner>,
    epistemic_conflict_detector: &mut Option<
        crate::consciousness::epistemic_conflict::ConflictDetector,
    >,
    theory_calibrator: &Option<crate::consciousness::epistemic_conflict::TheoryCalibrator>,
    coherence: f32,
    prediction_error: f32,
    unified_psi: f64,
    prediction_confidence: f64,
    total_cycles: usize,
    // Snapshot of carryover (read)
    last_harmonic_coherence: f64,
    last_phi_eff: f64,
    body_phi_modulation: f64,
) -> ConsciousnessMetricsBranchB {
    use super::super::thresholds::{
        HARMONIC_FIELD_BOOST_FACTOR, HARMONIC_FIELD_BOOST_THRESHOLD,
        REASONING_CONFIDENCE_BOOST_FACTOR, REASONING_CONFIDENCE_BOOST_THRESHOLD,
    };
    let mut deferred = Vec::new();

    // ── Fiduciary harmonics ──────────────────────────────────────────────
    let t = Instant::now();
    let (harmonic_field_coherence, harmonic_love_resonance, harmonic_interferences, new_harmonic) =
        if let Some(field) = harmonic_field {
            if total_cycles % 11 == 0 {
                use crate::consciousness::harmonics::FiduciaryHarmonic;
                field.set_level(FiduciaryHarmonic::ResonantCoherence, coherence as f64);
                field.set_level(
                    FiduciaryHarmonic::EvolutionaryProgression,
                    (prediction_error as f64 * 2.0).clamp(0.0, 1.0),
                );
                field.set_level(FiduciaryHarmonic::IntegralWisdom, prediction_confidence);
                field.set_level(
                    FiduciaryHarmonic::PanSentientFlourishing,
                    unified_psi.clamp(0.0, 1.0),
                );
                field.detect_interferences();
                if !field.interferences.is_empty() {
                    if let Some(resolver) = harmonic_resolver {
                        let _resolution = resolver.resolve(field);
                    }
                }
                (
                    field.field_coherence,
                    field.infinite_love_resonance,
                    field.interferences.len(),
                    Some(field.field_coherence),
                )
            } else {
                (last_harmonic_coherence, 0.0, 0, None)
            }
        } else {
            (0.0, 0.0, 0, None)
        };
    let harmonics_timing = t.elapsed().as_micros() as u64;

    // Deferred harmonics feedback
    if harmonic_field_coherence > HARMONIC_FIELD_BOOST_THRESHOLD as f64 {
        let harmony_boost = 1.0
            + ((harmonic_field_coherence - HARMONIC_FIELD_BOOST_THRESHOLD as f64)
                * HARMONIC_FIELD_BOOST_FACTOR as f64) as f32;
        deferred.push(DeferredFeedback::ScaleSubsystemLr(harmony_boost));
    }
    if harmonic_interferences > 0 {
        let interference_penalty = (harmonic_interferences.min(3) as f32) * 0.01;
        deferred.push(DeferredFeedback::AdjustConfidence(
            "harmonic_interference",
            -interference_penalty,
        ));
    }

    // ── Primitive reasoning ─────────────────────────────────────────────
    let t = Instant::now();
    let (reasoning_chain_confidence, reasoning_chain_depth) =
        if let Some(reasoner) = primitive_reasoner {
            if total_cycles % 47 == 0 && total_cycles > 0 {
                let result = reasoner.reason("cognitive_state", &[]);
                (result.confidence, result.reasoning_chain.len())
            } else {
                (0.0, 0)
            }
        } else {
            (0.0, 0)
        };
    let reasoning_timing = t.elapsed().as_micros() as u64;

    // Deferred reasoning feedback
    if reasoning_chain_confidence > REASONING_CONFIDENCE_BOOST_THRESHOLD {
        let reason_boost = (reasoning_chain_confidence - REASONING_CONFIDENCE_BOOST_THRESHOLD)
            * REASONING_CONFIDENCE_BOOST_FACTOR;
        deferred.push(DeferredFeedback::AdjustConfidence(
            "reasoning_chain",
            reason_boost,
        ));
    }

    // ── Epistemic conflict ──────────────────────────────────────────────
    let t = Instant::now();
    let (
        epistemic_phi_eff,
        epistemic_conflict_count,
        new_phi_eff,
        new_conflict_count,
        epi_override,
    ) = if let (Some(detector), Some(calibrator)) = (epistemic_conflict_detector, theory_calibrator)
    {
        if total_cycles % 97 == 0 && total_cycles > 0 {
            use crate::consciousness::epistemic_conflict::{
                ConflictMatrix, MultiTheoryMetrics, compute_phi_eff,
            };
            let metrics = MultiTheoryMetrics {
                phi: unified_psi,
                gwt: coherence as f64 * 0.8,
                ast: coherence as f64,
                pp: 1.0 - prediction_error as f64,
                rpt: coherence as f64 * 0.9,
                embodiment: body_phi_modulation,
                unified: unified_psi,
            };
            let matrix: ConflictMatrix = detector.detect(&metrics);
            let phi_eff_result = compute_phi_eff(&metrics, calibrator);
            let override_flag = matrix.conflicts.len() > 5;
            (
                phi_eff_result.phi_eff,
                matrix.conflicts.len(),
                Some(phi_eff_result.phi_eff),
                Some(matrix.conflicts.len()),
                override_flag,
            )
        } else {
            (last_phi_eff, 0, None, None, false)
        }
    } else {
        (0.0, 0, None, None, false)
    };
    let epistemic_conflict_timing = t.elapsed().as_micros() as u64;

    ConsciousnessMetricsBranchB {
        harmonic_field_coherence,
        harmonic_love_resonance,
        harmonic_interferences,
        harmonics_timing,
        reasoning_chain_confidence,
        reasoning_chain_depth,
        reasoning_timing,
        epistemic_phi_eff,
        epistemic_conflict_count,
        epistemic_conflict_timing,
        new_harmonic_coherence: new_harmonic,
        new_phi_eff,
        new_epistemic_conflict_count: new_conflict_count,
        epistemic_reasoning_override: epi_override,
        deferred,
    }
}
