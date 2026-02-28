//! Perception phase of the cognitive cycle.
//!
//! Extracts Phases 0–1.2 from the original `cycle()` method:
//! safety pre-check, thalamic routing, negation detection, moral evaluation,
//! strategy selection, HDC encoding, surprise exploration, input memoization,
//! ethics engine, and urgency computation.
//!
//! Strategy selection and encoding are delegated to `cycle_strategy.rs`.

use std::time::Instant;

use super::cycle::PerceptionPhaseResult;
use super::{CognitiveLoopService, CycleResult, ModuleTimings};

impl CognitiveLoopService {
    /// Perception phase: safety checks, thalamic routing, moral evaluation,
    /// strategy selection, HDC encoding, surprise exploration, input memoization,
    /// ethics engine, urgency computation.
    ///
    /// Returns `Ok(PerceptionPhaseResult)` on success, or `Err(CycleResult)` if the
    /// safety gateway blocks the input (early return).
    pub(super) fn phase_perception(
        &mut self,
        input: &str,
        cycle_start: Instant,
        module_timings: &mut ModuleTimings,
    ) -> Result<PerceptionPhaseResult, CycleResult> {
        // ── Cycle init: startup suppression, biorhythm, nociception, neuromod bath ──
        let init = self.run_cycle_init(module_timings);
        let exploration_urge_start = init.exploration_urge_start;
        let startup_suppressed = init.startup_suppressed;
        let startup_warmup_progress = init.startup_warmup_progress;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.1: Safety Pre-check (fast amygdala veto)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(blocked) = self.safety_precheck(input, cycle_start) {
            return Err(blocked);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0: Thalamic Routing (Cognitive Depth Selection)
        // ═══════════════════════════════════════════════════════════════════════
        self.update_cognitive_depth();

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.3: Negation Detection (guards moral evaluation)
        // ═══════════════════════════════════════════════════════════════════════
        let input_negation_polarity = self.detect_negation_polarity(input);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.4: Moral Evaluation (throttled: every Nth cycle or on new input)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (moral_score, moral_concern_detected, moral_judgment) =
            self.run_moral_phase(input, input_negation_polarity);
        module_timings.moral_algebra = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.5: Strategy Selection (extracted to cycle_strategy.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let strategy_result = self.run_strategy_selection(moral_concern_detected);
        let selected_strategy = strategy_result.selected_strategy;
        let agency_strategy_override = strategy_result.agency_strategy_override;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASES 1–1.2: Encoding + Preprocessing (extracted to cycle_strategy.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let encoding = self.run_encoding_and_preprocessing(input, module_timings);

        // Stash negation polarity for metadata
        let negation_detected = input_negation_polarity;

        Ok(PerceptionPhaseResult {
            encoding_result: encoding.encoding_result,
            encoding_us: module_timings.core_hdc_encode,
            hv16_cached: encoding.hv16_cached,
            compressed_state: encoding.compressed_state,
            phi_attention_weight: encoding.phi_attention_weight,
            exploration_urge_start,
            startup_suppressed,
            startup_warmup_progress,
            input_memoized: encoding.input_memoized,
            input_similarity: encoding.input_similarity,
            moral_concern_detected,
            moral_score,
            moral_judgment,
            selected_strategy,
            agency_strategy_override,
            soul_alignment: encoding.soul_alignment,
            negation_detected,
            surprise_triggered: encoding.surprise_triggered,
            exploration_action: encoding.exploration_action,
            urgency: encoding.urgency,
            error_pattern: encoding.error_pattern,
            predicted_urgency: encoding.predicted_urgency,
            prediction_coherence_urgency_bias: encoding.prediction_coherence_urgency_bias,
            prediction_error: encoding.prediction_error,
            effective_threshold: encoding.effective_threshold,
            memo_threshold: encoding.memo_threshold,
        })
    }
}
