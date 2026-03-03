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
    /// Returns `Ok(PerceptionPhaseResult)` on success, or `Err(Box<CycleResult>)` if the
    /// safety gateway blocks the input (early return).
    ///
    /// The Err variant is boxed because `CycleResult` is ~4KB (contains `BinaryHV`),
    /// which would bloat the `Result` on the stack even though the error path is rare.
    pub(super) fn phase_perception(
        &mut self,
        input: &str,
        cycle_start: Instant,
        module_timings: &mut ModuleTimings,
    ) -> Result<PerceptionPhaseResult, Box<CycleResult>> {
        // ── Cycle init: startup suppression, biorhythm, nociception, neuromod bath ──
        let init = self.run_cycle_init(module_timings);
        let exploration_urge_start = init.exploration_urge_start;
        let startup_suppressed = init.startup_suppressed;
        let startup_warmup_progress = init.startup_warmup_progress;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.1: Safety Pre-check (fast amygdala veto)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(blocked) = self.safety_precheck(input, cycle_start) {
            return Err(Box::new(blocked));
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

        // Push encoded HV into ring buffers for Phi-Dyad computation.
        // The encoding hdv represents the AI's cognitive state for this input.
        // We also use it as a human-proxy state (input → perceived partner state).
        if self.phi_dyad.is_some() {
            let ai_hv = encoding.encoding_result.hdv.clone();
            let input_hv = ai_hv.clone(); // Same encoding as partner proxy
            if self.recent_ai_hvs.len() >= 4 {
                self.recent_ai_hvs.remove(0);
            }
            self.recent_ai_hvs.push(ai_hv);
            if self.recent_input_hvs.len() >= 4 {
                self.recent_input_hvs.remove(0);
            }
            self.recent_input_hvs.push(input_hv);
        }

        Ok(PerceptionPhaseResult {
            encoding_result: encoding.encoding_result,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::CognitiveLoopConfig;

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn perception_returns_ok_for_normal_input() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc.phase_perception("hello world", Instant::now(), &mut timings);
        assert!(result.is_ok(), "phase_perception should return Ok for normal input");
    }

    #[test]
    fn perception_result_has_finite_fields() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc
            .phase_perception("test input", Instant::now(), &mut timings)
            .unwrap();
        assert!(result.moral_score.is_finite());
        assert!(result.phi_attention_weight.is_finite());
        assert!(result.prediction_error.is_finite());
        assert!(result.soul_alignment.is_finite());
        assert!(result.negation_detected.is_finite());
        assert!(result.exploration_urge_start.is_finite());
        assert!(result.effective_threshold.is_finite());
    }

    #[test]
    fn perception_encoding_result_populated() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc
            .phase_perception("encoding check", Instant::now(), &mut timings)
            .unwrap();
        assert!(!result.encoding_result.hdv.values.is_empty());
        assert!(result.encoding_result.peak_attention.is_finite());
    }

    #[test]
    fn perception_phi_dyad_ring_buffer_caps_at_4() {
        let mut cfg = CognitiveLoopConfig::default();
        cfg.enable_primitive_consciousness = true;
        let mut svc = CognitiveLoopService::new(cfg).unwrap();
        let mut timings = ModuleTimings::default();
        for i in 0..6 {
            let _ = svc.phase_perception(&format!("input {i}"), Instant::now(), &mut timings);
        }
        assert!(svc.recent_ai_hvs.len() <= 4);
        assert!(svc.recent_input_hvs.len() <= 4);
    }

    #[test]
    fn perception_moral_timing_recorded() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let _ = svc.phase_perception("moral timing", Instant::now(), &mut timings);
        assert!(timings.moral_algebra < 10_000_000);
    }

    #[test]
    fn perception_startup_warmup_on_first_cycle() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc
            .phase_perception("first cycle", Instant::now(), &mut timings)
            .unwrap();
        assert!(result.startup_warmup_progress >= 0.0 && result.startup_warmup_progress <= 1.0);
    }
}
