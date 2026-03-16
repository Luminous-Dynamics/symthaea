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
    if let Some(ref mut enhancer) = causal_enhancer {
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
    let phi_sufficient = ctx.phi > 0.2;
    if phi_sufficient && (ctx.prediction_error > 0.1 || ctx.in_flow) {
        let hdv_sample: Vec<f32> =
            ctx.compressed_state[..64.min(ctx.compressed_state.len())].to_vec();
        episodic_memory.encode(
            ctx.input,
            hdv_sample,
            ctx.emotional_valence,
            ctx.phi,
            ctx.total_cycles,
        );
    }

    // Resonator memory: store with bound attributes for factorized recall
    if let Some(ref mut res_mem) = resonator_memory {
        let res_dim_ok = ctx.compressed_state.len() == res_mem.resonator.config.dim;
        if res_dim_ok && (ctx.prediction_error > 0.1 || ctx.in_flow) {
            // Quantize valence -> nearest band
            let val_label = if ctx.emotional_valence > 0.3 {
                "positive"
            } else if ctx.emotional_valence < -0.3 {
                "negative"
            } else {
                "neutral"
            };
            let val_hv = res_mem
                .resonator
                .codebooks
                .get(1)
                .and_then(|cb| cb.symbols.iter().find(|(l, _)| l == val_label))
                .map(|(_, hv)| hv.clone());

            // Quantize phi -> nearest band
            let phi_label = if ctx.phi > 0.7 {
                "high"
            } else if ctx.phi > 0.3 {
                "medium"
            } else {
                "low"
            };
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

    if let Some(ref prev_state) = prev_primitive_state {
        let pred_error = primitive_belief_bridge.compute_prediction_error(prev_state, &prim_state);
        let td_signal = primitive_belief_bridge.td_error_signal(&pred_error);
        *fep_learning_signal += td_signal as f32 * 0.2;
        *fep_learning_signal = fep_learning_signal.clamp(-1.0, 1.0);
    }

    *prev_primitive_state = Some(prim_state);

    // Closed learning loop: update Q-values, modulated by FEP learning signal.
    // Friston (2010): plasticity scales with expected information gain.
    let plasticity = 1.0 + *fep_learning_signal * 0.5; // [-1,1] → [0.5, 1.5]
    closed_learning_loop.update_with_plasticity(cycle_learning_result, plasticity);

    // Return the memory context boost for deferred application via adjust_confidence
    ctx.memory_context_boost
}
