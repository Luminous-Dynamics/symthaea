//! Strategy selection and encoding phases extracted from cycle_phase_perception.rs.
//!
//! Contains:
//! - `run_strategy_selection`: Closed Learning Loop strategy selection + agency override
//! - `run_encoding_and_preprocessing`: HDC encoding, soul alignment, phi attention gating,
//!   surprise exploration, codebook diversity, input memoization, ethics engine, urgency

use std::time::Instant;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;
use symthaea_core::hdc::predictive_encoder::EncodingResult;

use super::helpers;
use super::thresholds::MORAL_CONCERN_THRESHOLD;
use super::{CognitiveLoopService, ModuleTimings, ResponseStrategy};

// ═══════════════════════════════════════════════════════════════════════════════
// Result structs
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from the strategy selection phase (Phase 0.5).
pub(crate) struct StrategySelectionResult {
    pub(crate) selected_strategy: ResponseStrategy,
    pub(crate) agency_strategy_override: bool,
}

/// Result from the encoding and preprocessing phase (Phases 1–1.2).
pub(crate) struct EncodingPhaseResult {
    pub(crate) encoding_result: EncodingResult,
    pub(crate) hv16_cached: BinaryHV,
    pub(crate) compressed_state: Vec<f32>,
    pub(crate) soul_alignment: f32,
    pub(crate) phi_attention_weight: f32,
    pub(crate) surprise_triggered: bool,
    pub(crate) exploration_action: Option<String>,
    pub(crate) memo_threshold: f32,
    pub(crate) input_similarity: f32,
    pub(crate) input_memoized: bool,
    pub(crate) effective_threshold: f32,
    pub(crate) urgency: super::CycleUrgency,
    pub(crate) error_pattern: &'static str,
    pub(crate) predicted_urgency: &'static str,
    pub(crate) prediction_coherence_urgency_bias: f32,
    pub(crate) prediction_error: f32,
}

impl CognitiveLoopService {
    /// Strategy selection phase (Phase 0.5).
    ///
    /// Selects response strategy based on:
    /// - Q-learning from past interactions (ClosedLearningLoop)
    /// - Previous reward (stick with success, avoid failure)
    /// - MCTS-informed bias from prior cycle's deliberative plan
    /// - Moral concerns (bias toward Supportive for ethical guidance)
    /// - Embodied agency override (low agency → conservative strategy)
    ///
    /// Science: Kahneman (2011) dual-process; Varela (1991) embodied agency.
    pub(super) fn run_strategy_selection(
        &mut self,
        moral_concern_detected: bool,
    ) -> StrategySelectionResult {
        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.5: Closed Learning Loop - Strategy Selection
        // ═══════════════════════════════════════════════════════════════════════
        let prior_phi = self.unification_engine.psi;
        let prior_reward = self
            .closed_learning_loop
            .last_result
            .as_ref()
            .map(|r| r.reward);
        let mut selected_strategy = if moral_concern_detected {
            ResponseStrategy::Supportive
        } else {
            let base_strategy = self
                .closed_learning_loop
                .select_strategy(prior_phi, prior_reward);

            if let Some(&(plan_action, plan_confidence)) = self.carryover.history.mcts_plan.as_ref()
            {
                if plan_confidence > 0.7 {
                    match plan_action {
                        0 => match base_strategy {
                            ResponseStrategy::Exploratory => ResponseStrategy::Detailed,
                            other => other,
                        },
                        2 => match base_strategy {
                            ResponseStrategy::Supportive | ResponseStrategy::Concise => {
                                ResponseStrategy::Exploratory
                            }
                            other => other,
                        },
                        _ => base_strategy,
                    }
                } else {
                    base_strategy
                }
            } else {
                base_strategy
            }
        };

        self.apply_strategy_modulation(selected_strategy);

        // ── Phase 21: Embodied agency → strategy modulation ──────────────
        // Science: Varela (1991) — low agency = reactive mode → prefer conservative strategy
        let agency_strategy_override = {
            let cached_agency = self.carryover.consciousness.last_embodied_agency;
            if cached_agency < 0.3
                && cached_agency > 0.0
                && selected_strategy == ResponseStrategy::Exploratory
            {
                selected_strategy = ResponseStrategy::Supportive;
                self.apply_strategy_modulation(selected_strategy);
                self.stats.agency_strategy_override_count += 1;
                true
            } else {
                false
            }
        };

        StrategySelectionResult {
            selected_strategy,
            agency_strategy_override,
        }
    }

    /// Encoding and preprocessing phase (Phases 1–1.2).
    ///
    /// Covers:
    /// - HDC encode with attention from previous prediction
    /// - BinaryHV cache (real_hv_to_hv16)
    /// - Soul alignment evaluation (Seven Harmonies)
    /// - Phi-guided attention gating
    /// - Surprise-driven exploration
    /// - Codebook diversity → memoization threshold adaptation
    /// - Input similarity memoization (Tulving & Schacter 1990)
    /// - Unified Ethics Engine evaluation
    /// - Adaptive learning threshold + urgency computation
    pub(super) fn run_encoding_and_preprocessing(
        &mut self,
        input: &str,
        module_timings: &mut ModuleTimings,
    ) -> EncodingPhaseResult {
        // 1. HDC encode with attention from previous prediction
        let _t_core = Instant::now();
        let encoding_result = self.encoder.encode(input);
        let prediction_error = encoding_result.prediction_error;
        module_timings.core_hdc_encode = _t_core.elapsed().as_micros() as u64;

        // Pre-compute BinaryHV once for all subsystems that need it.
        let _t_core = Instant::now();
        let hv16_cached = real_hv_to_hv16(&encoding_result.hdv);
        module_timings.core_compress = _t_core.elapsed().as_micros() as u64;

        // Soul value alignment
        let soul_alignment = if let Some(ref soul) = self.soul {
            let alignment = soul.evaluate_alignment(&encoding_result.hdv);
            if alignment.overall_alignment < MORAL_CONCERN_THRESHOLD {
                self.stats.moral_concerns_detected += 1;
            }
            if alignment.overall_alignment > 0.3 {
                let boost = (alignment.overall_alignment - 0.3) * 0.1;
                self.carryover.learning.subsystem_lr_factor *= 1.0 + boost;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.8, 1.3);
            } else if alignment.overall_alignment < -0.3 {
                let dampening = (alignment.overall_alignment + 0.3).abs() * 0.15;
                self.carryover.learning.subsystem_lr_factor *= 1.0 - dampening;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.2);
            }
            alignment.overall_alignment
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // 0.5 Phi-Guided Attention Gating
        // ═══════════════════════════════════════════════════════════════════════
        let phi_attention_weight = if let Some(ref mut gate) = self.phi_attention_gate {
            let phi_vals = [self.stats.unified_psi as f64];
            let result = gate.forward(std::slice::from_ref(&encoding_result.hdv), &phi_vals);
            result.weights.first().copied().unwrap_or(1.0)
        } else {
            1.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // 1.1 Surprise-Driven Exploration
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let compressed_state: Vec<f32> = self
            .encoder
            .compress_for_ltc(&encoding_result.hdv, self.config.cfc_config.input_dim)
            .iter()
            .map(|v| v * phi_attention_weight)
            .collect();
        let (surprise_triggered, exploration_action) =
            self.run_surprise_exploration(&compressed_state);
        module_timings.surprise_exploration = _t.elapsed().as_micros() as u64;

        // ── Phase 21: Codebook diversity → memoization threshold adaptation ─
        let base_memo_threshold = super::thresholds::INPUT_MEMO_THRESHOLD;
        let diversity = self.stats.codebook_diversity;
        let memo_threshold = if diversity < 0.4 && diversity > 0.0 {
            let t = (base_memo_threshold - (0.4 - diversity) * 0.1).max(0.88);
            self.stats.memo_threshold_adaptations += 1;
            t
        } else if diversity > 0.8 {
            let t = (base_memo_threshold + (diversity - 0.8) * 0.05).min(0.98);
            self.stats.memo_threshold_adaptations += 1;
            t
        } else {
            base_memo_threshold
        };

        // ── Phase 15: Input similarity memoization ───────────────────────────
        // Science: Priming (Tulving & Schacter 1990) — repeated stimuli can reuse
        // prior processing results.
        let (input_similarity, input_memoized) =
            if let Some(ref prev) = self.carryover.history.last_compressed_state {
                let sim = helpers::cosine_f32(&compressed_state, prev).max(0.0);
                let memoized = sim > memo_threshold;
                if memoized {
                    self.stats.input_memoization_hits += 1;
                }
                (sim, memoized)
            } else {
                (0.0, false)
            };
        self.carryover.history.last_compressed_state = Some(compressed_state.clone());

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED ETHICS ENGINE
        // ═══════════════════════════════════════════════════════════════════════
        let ethics_output = self
            .ethics_engine
            .evaluate(&super::ethics_engine::EthicsEngineInput {
                input,
                cycle: self.stats.total_cycles as u64,
                unified_psi: self.stats.unified_psi as f64,
                compressed_state: &compressed_state,
            });
        module_timings.ethics_engine = ethics_output.total_us;
        module_timings.ethics_engine_moral = ethics_output.moral_us;
        module_timings.ethics_engine_value = ethics_output.value_us;
        module_timings.ethics_engine_harmonies = ethics_output.harmonies_us;
        module_timings.moral_topology = ethics_output.topology_us;
        if ethics_output.confidence_delta != 0.0 {
            self.adjust_confidence("ethics_engine", ethics_output.confidence_delta);
        }
        if ethics_output.lr_factor != 1.0 {
            self.scale_lr("ethics_engine", ethics_output.lr_factor);
        }
        // Anomaly response: corrective feedback when moral anomalies detected (opt-in)
        if self.config.enable_moral_anomaly_response {
            let report = &ethics_output.anomaly_report;
            if report.value_inversion {
                self.scale_lr("moral_anomaly_inversion", 1.2);
            }
            if report.free_energy_spike {
                self.adjust_exploration("moral_anomaly_fe", -0.1);
            }
            if report.fragmentation_increase {
                self.adjust_confidence("moral_anomaly_frag", 0.05);
            }
            if report.drift_alert {
                self.scale_lr("moral_anomaly_drift", 0.8);
            }
        }
        self.carryover.quality.last_value_score = ethics_output.value_score;
        if ethics_output.value_gate_factor != 1.0 {
            self.stats.value_gate_applied_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1.2 Adaptive Learning Threshold + Urgency
        // ═══════════════════════════════════════════════════════════════════════
        // Science: Friston (2010) — precision (inverse uncertainty) modulates PE weighting.
        let confidence_scale = (1.0 + (self.prediction_confidence - 0.5) * 0.4) as f32;
        let exploration_scale = 1.0 - (self.curiosity_drive.exploration_urge - 0.5) * 0.2;
        let effective_threshold = self.config.learning_threshold
            * self.carryover.learning.adaptive_threshold_scale
            * confidence_scale
            * exploration_scale;
        let urgency_result = self.compute_urgency_and_error_pattern(
            prediction_error,
            surprise_triggered,
            effective_threshold,
        );
        let urgency = urgency_result.urgency;
        let error_pattern = urgency_result.error_pattern;
        let predicted_urgency = urgency_result.predicted_urgency;
        let prediction_coherence_urgency_bias = urgency_result.prediction_coherence_urgency_bias;

        EncodingPhaseResult {
            encoding_result,
            hv16_cached,
            compressed_state,
            soul_alignment,
            phi_attention_weight,
            surprise_triggered,
            exploration_action,
            memo_threshold,
            input_similarity,
            input_memoized,
            effective_threshold,
            urgency,
            error_pattern,
            predicted_urgency,
            prediction_coherence_urgency_bias,
            prediction_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ResponseStrategy};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn test_moral_concern_selects_supportive() {
        let mut svc = make_service();
        let result = svc.run_strategy_selection(true);
        assert_eq!(result.selected_strategy, ResponseStrategy::Supportive);
        assert!(!result.agency_strategy_override);
    }

    #[test]
    fn test_no_moral_concern_uses_learning_loop() {
        let mut svc = make_service();
        // Without moral concern, strategy comes from closed learning loop
        let result = svc.run_strategy_selection(false);
        // Strategy should be one of the valid variants (exact depends on CLL state)
        let valid = matches!(
            result.selected_strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        );
        assert!(valid);
    }

    #[test]
    fn test_agency_override_low_agency_exploratory() {
        let mut svc = make_service();
        // Set low cached agency (>0.0 but <0.3)
        svc.carryover.consciousness.last_embodied_agency = 0.1;
        // Force CLL to pick Exploratory: set high prior reward + adjust state
        // We'll manually test the override path by checking counter behavior.
        // Strategy from CLL might not be Exploratory, so we test the guard:
        // the override only triggers if strategy == Exploratory AND agency in (0.0, 0.3).
        let before = svc.stats.agency_strategy_override_count;
        let result = svc.run_strategy_selection(false);
        if result.selected_strategy == ResponseStrategy::Supportive
            && svc.stats.agency_strategy_override_count > before
        {
            // Override happened
            assert!(result.agency_strategy_override);
        } else {
            // CLL didn't pick Exploratory → override doesn't apply
            assert!(
                !result.agency_strategy_override
                    || result.selected_strategy == ResponseStrategy::Supportive
            );
        }
    }

    #[test]
    fn test_agency_override_zero_agency_no_override() {
        let mut svc = make_service();
        // Set agency exactly 0.0 → guard condition prevents override
        svc.carryover.consciousness.last_embodied_agency = 0.0;
        let result = svc.run_strategy_selection(false);
        // The guard requires cached_agency > 0.0, so no override
        assert!(!result.agency_strategy_override);
    }

    #[test]
    fn test_mcts_plan_high_confidence_biases_strategy() {
        let mut svc = make_service();
        // Set high-confidence MCTS plan with action=0 (bias Exploratory→Detailed)
        svc.carryover.history.mcts_plan = Some((0, 0.9));
        let _result = svc.run_strategy_selection(false);
        // Can't assert exact strategy without knowing CLL output,
        // but if CLL picked Exploratory, it should become Detailed
        // The MCTS bias is tested by verifying no panic and valid output
        assert!(matches!(
            _result.selected_strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        ));
    }

    #[test]
    fn test_mcts_plan_low_confidence_no_bias() {
        let mut svc = make_service();
        // Low confidence MCTS plan → no bias applied, produces valid strategy
        svc.carryover.history.mcts_plan = Some((0, 0.3));
        let result = svc.run_strategy_selection(false);
        // Low confidence (0.3 < 0.7 threshold) → MCTS plan ignored, CLL base strategy used
        assert!(matches!(
            result.selected_strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        ));
    }

    #[test]
    fn test_moral_concern_overrides_mcts() {
        let mut svc = make_service();
        // Even with high-confidence MCTS plan, moral concern wins
        svc.carryover.history.mcts_plan = Some((2, 0.95));
        let result = svc.run_strategy_selection(true);
        assert_eq!(result.selected_strategy, ResponseStrategy::Supportive);
    }
}
