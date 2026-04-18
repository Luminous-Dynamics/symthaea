// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perception phase of the cognitive cycle.
//!
//! Extracts Phases 0–1.2 from the original `cycle()` method:
//! safety pre-check, thalamic routing, negation detection, moral evaluation,
//! strategy selection, HDC encoding, surprise exploration, input memoization,
//! ethics engine, and urgency computation.
//!
//! Strategy selection and encoding are delegated to `cycle_strategy.rs`.

use std::time::Instant;

use super::phase_results::{
    PercEncoding, PercExploration, PercMath, PercMoral, PercStrategy, PercUrgency,
    PerceptionPhaseResult,
};
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
        let (
            moral_score,
            moral_concern_detected,
            moral_judgment,
            spinozist_affect_coords,
            spinozist_fluctuatio,
            spinozist_ambiguous,
            spinozist_confidence,
        ) = self.run_moral_phase(input, input_negation_polarity);
        module_timings.moral_algebra = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.4b: Math Intent Detection (classify input for solver routing)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        #[cfg(feature = "mathematics")]
        let math_result = {
            let problem_type = super::math_service::MathService::classify_problem(input);
            if problem_type != super::math_service::MathProblemType::Unknown {
                PercMath {
                    math_detected: true,
                    problem_type: Some(problem_type),
                    phi: 0.0,        // Populated during dynamics phase if solved
                    confidence: 0.0, // Populated during dynamics phase if solved
                }
            } else {
                PercMath::default()
            }
        };
        #[cfg(not(feature = "mathematics"))]
        let math_result = PercMath::default();
        module_timings.math_service = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.5: Strategy Selection (extracted to cycle_strategy.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let strategy_result = self.run_strategy_selection(moral_concern_detected);
        let selected_strategy = strategy_result.selected_strategy;
        let agency_strategy_override = strategy_result.agency_strategy_override;
        let social_strategy_bias = strategy_result.social_strategy_bias;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASES 1–1.2: Encoding + Preprocessing (extracted to cycle_strategy.rs)
        // ═══════════════════════════════════════════════════════════════════════
        #[allow(unused_mut)]
        let mut encoding = self.run_encoding_and_preprocessing(input, module_timings);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 1.2b: Additive sensory modality blend
        // ═══════════════════════════════════════════════════════════════════════
        // Live STT (voice-stt-live) and radio/SDR spectrum (mesh) are both
        // ambient sensory inputs, not semantic modifiers. Collect any modality
        // HVs that fired this cycle and bundle them into the perception HV
        // with one call so all sources share the same thresholding pass
        // instead of being biased by sequential bundling order.
        #[allow(unused_mut)]
        let mut aux_sensor_hvs: Vec<crate::hdc::BinaryHV> = Vec::new();

        #[cfg(feature = "voice-stt-live")]
        if let Some(ref stt) = self.stt_capture {
            if let Some(stt_chv) = stt.drain_latest() {
                aux_sensor_hvs.push(crate::hdc::BinaryHV::from_bipolar(&stt_chv.values));
            }
        }

        #[cfg(feature = "mesh")]
        if let Some(radio_hv) = self.spectrum_manager.perception_hv() {
            aux_sensor_hvs.push(radio_hv);
        }

        // IMU (sensor-imu): fuse the most-recent injected reading into an
        // HV and treat it as another additive sensory modality. Skipped if
        // either the fusion module or the reading is absent.
        #[cfg(feature = "sensor-imu")]
        if let (Some(ref fusion), Some(ref reading)) = (&self.imu_fusion, &self.latest_imu_reading)
        {
            let imu_chv = fusion.fuse(reading);
            aux_sensor_hvs.push(crate::hdc::BinaryHV::from_bipolar(&imu_chv.values));
        }

        if !aux_sensor_hvs.is_empty() {
            aux_sensor_hvs.insert(0, encoding.hv16_cached);
            encoding.hv16_cached = crate::hdc::BinaryHV::bundle(&aux_sensor_hvs);
            // Keep the continuous view in sync — the CfC input path at
            // cycle_strategy.rs:464 reads encoding_result.hdv (not hv16_cached),
            // so without this line the blended sensors would never reach the
            // thought-vector generator.
            encoding.encoding_result.hdv = encoding.hv16_cached.to_continuous();
        }

        // ACh-modulated scene memory thresholds
        #[cfg(feature = "vision-manifold")]
        if let Some(ref mut bridge) = self.sensorimotor.vision_sensory.vision_bridge {
            let ach = self.neuromod.bath.acetylcholine.effective();
            let ach_factor = ach.max(super::thresholds::VISION_ACH_FLOOR);
            let base_coh = self.config.scene_memory_coherence_threshold;
            let base_err = self.config.scene_memory_error_threshold;
            let base_damp = self.config.scene_memory_dampen_factor;
            bridge.manifold_mut().set_scene_store_thresholds(
                (base_coh / ach_factor).clamp(
                    super::thresholds::VISION_COHERENCE_CLAMP_MIN,
                    super::thresholds::VISION_COHERENCE_CLAMP_MAX,
                ),
                (base_err * ach_factor.min(super::thresholds::VISION_ACH_SCALE_CAP)).clamp(
                    super::thresholds::VISION_ERROR_CLAMP_MIN,
                    super::thresholds::VISION_ERROR_CLAMP_MAX,
                ),
            );
            bridge.manifold_mut().set_scene_dampen_factor(
                (base_damp * ach_factor.min(super::thresholds::VISION_ACH_SCALE_CAP)).clamp(
                    super::thresholds::VISION_DAMPEN_CLAMP_MIN,
                    super::thresholds::VISION_DAMPEN_CLAMP_MAX,
                ),
            );
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 1.3: Vision Manifold (frame → attention-boosted HDC encoding)
        // ═══════════════════════════════════════════════════════════════════════
        #[cfg(feature = "vision-manifold")]
        let (visual_hv, vision_telemetry) = if let Some(ref mut bridge) =
            self.sensorimotor.vision_sensory.vision_bridge
        {
            let frame = self
                .sensorimotor
                .vision_sensory
                .vision_frame_buffer
                .take()
                .unwrap_or_else(|| {
                    vec![
                        128u8;
                        (self.config.vision_frame_width * self.config.vision_frame_height) as usize
                    ]
                });
            let w = self.config.vision_frame_width;
            let h = self.config.vision_frame_height;
            let dt = self.config.cfc_config.delta_t;
            let (hv, tel) = bridge.process_frame_with_telemetry(&frame, w, h, 1, dt);
            (Some(hv), Some(tel))
        } else {
            (None, None)
        };

        // Cross-manifold predictor: predict cognitive state from vision state
        #[cfg(feature = "vision-manifold")]
        let cross_manifold_prediction_error =
            if let Some(ref mut pred) = self.sensorimotor.vision_sensory.cross_manifold_predictor {
                if let Some(ref vis_hv) = visual_hv {
                    let vis_chv = symthaea_core::hdc::ContinuousHV::from_slice(vis_hv.as_slice());
                    let _predicted = pred.predict_cognitive(&vis_chv);
                    pred.prediction_error()
                } else {
                    0.0
                }
            } else {
                0.0
            };

        // ── Vision mean surprise + horizon errors + scene recognition ──
        #[cfg(feature = "vision-manifold")]
        let vision_mean_surprise = self
            .sensorimotor
            .vision_sensory
            .vision_bridge
            .as_ref()
            .map(|b| b.manifold().surprise_map().mean_surprise())
            .unwrap_or(0.0);

        #[cfg(feature = "vision-manifold")]
        let vision_horizon_errors = self
            .sensorimotor
            .vision_sensory
            .vision_bridge
            .as_ref()
            .map(|b| b.manifold().evaluate_horizons().errors)
            .unwrap_or_default();

        #[cfg(feature = "vision-manifold")]
        let scene_recognized = vision_telemetry
            .as_ref()
            .map(|t| t.scene_recognition_similarity > 0.0)
            .unwrap_or(false);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 1.4: Foveation Dispatch (dorsal surprise → ventral recognition)
        // ═══════════════════════════════════════════════════════════════════════
        #[cfg(feature = "foveation")]
        let fov_results: Vec<symthaea_foveation::FoveationResult> = if !self.config.enable_foveation
        {
            Vec::new()
        } else {
            let mut collected = Vec::new();
            if let Some(ref fov_mutex) = self.sensorimotor.vision_sensory.foveation_manager {
                if let Ok(mut fov) = fov_mutex.lock() {
                    // Neuromod modulation: NE → surprise threshold, DA → concurrent capacity
                    let ne = self.neuromod.bath.noradrenaline.effective();
                    let da = self.neuromod.bath.dopamine.effective();
                    fov.modulate(ne, da);

                    // Feed current frame to foveation
                    if let Some(ref bridge) = self.sensorimotor.vision_sensory.vision_bridge {
                        let w = self.config.vision_frame_width;
                        let h = self.config.vision_frame_height;
                        let frame_count = bridge.frame_count();
                        let fb = symthaea_foveation::FrameBuffer {
                            pixels: vec![128u8; (w * h) as usize],
                            width: w,
                            height: h,
                            channels: 1,
                            frame_id: frame_count,
                            timestamp_us: frame_count * 20_000, // ~50fps
                        };
                        fov.on_frame(fb);

                        // Feed salient patches with real motion vectors from manifold
                        let regions = bridge.salient_regions();
                        let motion = bridge.manifold().motion_vectors();
                        let grid_cols = bridge.manifold().surprise_map().grid().cols;
                        let patches: Vec<_> = regions
                            .iter()
                            .map(|r| {
                                let mv_idx = r.grid_row * grid_cols + r.grid_col;
                                let vel = motion.get(mv_idx).copied().unwrap_or([0.0, 0.0]);
                                (r.grid_row, r.grid_col, r.surprise, vel)
                            })
                            .collect();
                        fov.on_saliency(&patches);
                    }

                    // Tick and drain
                    let now_us = self.start_time.elapsed().as_micros() as u64;
                    fov.tick(now_us);
                    collected = fov.drain_results();
                    // Attention budget: cap dispatches per cycle
                    // Substrate speed scaling: faster substrates afford more dispatches
                    let tau_scale = self.substrate_manager.tau_factor.max(0.5);
                    let max = ((self.config.foveation_max_dispatches as f32) * tau_scale)
                        .round()
                        .max(1.0) as usize;
                    if collected.len() > max {
                        // Keep the highest-confidence results
                        collected.sort_by(|a, b| {
                            b.confidence
                                .partial_cmp(&a.confidence)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        collected.truncate(max);
                    }
                    for result in &collected {
                        tracing::debug!(
                            target: "cognitive_loop::foveation",
                            grid = ?(result.grid_row, result.grid_col),
                            confidence = result.confidence,
                            "Foveation recognition"
                        );
                    }

                    // Win 2: Ventral→Dorsal feedback — recognized patches dampen
                    // dorsal surprise so the system stops re-foveating known regions.
                    // P3-B: Hebbian goal-template learning — recognized patches nudge
                    // the task HV toward what was actually seen (Hebb 1949).
                    if let Some(ref mut bridge) = self.sensorimotor.vision_sensory.vision_bridge {
                        for result in &collected {
                            bridge.dampen_patch_surprise(
                                result.grid_row,
                                result.grid_col,
                                result.confidence,
                            );
                            bridge.learn_from_recognized_patch(
                                &result.semantic_hv,
                                result.confidence,
                            );
                        }
                    }
                }
            }
            collected
        };

        // ── GWT injection: feed foveation results into Global Workspace ──
        #[cfg(feature = "foveation")]
        if let Some(ref mut gwt) = self.consciousness.gwt_mgr.gwt {
            for result in &fov_results {
                let binary_hv = result.semantic_hv.to_binary(0.0);
                let activation = result.confidence as f64;
                let module = match &result.content {
                    symthaea_foveation::RecognizedContent::Text(_) => "foveation_ocr",
                    symthaea_foveation::RecognizedContent::Object { .. } => "foveation_embed",
                    symthaea_foveation::RecognizedContent::Caption(_) => "foveation_caption",
                    symthaea_foveation::RecognizedContent::Unknown => "foveation_unknown",
                };
                gwt.submit_strategy(
                    "foveation_recognition",
                    activation,
                    vec![binary_hv],
                    vec![module.to_string()],
                );
            }
        }

        // ── Surprise dampening: reduce surprise at recognized locations ──
        #[cfg(feature = "foveation")]
        if let Some(ref mut bridge) = self.sensorimotor.vision_sensory.vision_bridge {
            for result in &fov_results {
                if result.confidence > 0.7 {
                    // Predict current position using velocity + processing latency
                    let dt_frames = result.processing_time_us as f32 / 20_000.0; // ~50fps
                    let pred_row = (result.grid_row as f32 + result.velocity[1] * dt_frames)
                        .round()
                        .max(0.0) as usize;
                    let pred_col = (result.grid_col as f32 + result.velocity[0] * dt_frames)
                        .round()
                        .max(0.0) as usize;
                    bridge.manifold_mut().surprise_map_mut().dampen(
                        pred_row, pred_col, 0.5, // halve surprise at this location
                    );
                }
            }
        }

        // ── Multimodal HV binding: foveation → cognitive state ──
        // Treisman (1980): feature integration theory — visual features bind into
        // unified percept alongside linguistic encoding.
        #[cfg(feature = "foveation")]
        if !fov_results.is_empty() {
            use super::thresholds::FOVEATION_HV_BINDING_WEIGHT;
            let main_hv = &encoding.encoding_result.hdv;
            let mut hvs: Vec<&symthaea_core::hdc::ContinuousHV> = vec![main_hv];
            let mut weights: Vec<f32> = vec![1.0];
            for result in &fov_results {
                hvs.push(&result.semantic_hv);
                weights.push(result.confidence * FOVEATION_HV_BINDING_WEIGHT);
            }
            encoding.encoding_result.hdv =
                symthaea_core::hdc::ContinuousHV::weighted_bundle(&hvs, &weights);
        }

        #[cfg(feature = "foveation")]
        let foveation_recognition_count = fov_results.len();
        #[cfg(feature = "foveation")]
        let foveation_top_confidence = fov_results
            .iter()
            .map(|r| r.confidence)
            .fold(0.0f32, f32::max);

        // Stash negation polarity for metadata
        let negation_detected = input_negation_polarity;

        // Push encoded HV into ring buffers for Phi-Dyad computation.
        // The encoding hdv represents the AI's cognitive state for this input.
        // We also use it as a human-proxy state (input → perceived partner state).
        if self.behavior.social_mgr.phi_dyad.is_some() {
            let ai_hv = encoding.encoding_result.hdv.clone();
            let input_hv = ai_hv.clone(); // Same encoding as partner proxy
            if self.behavior.social_mgr.recent_ai_hvs.len() >= 4 {
                self.behavior.social_mgr.recent_ai_hvs.remove(0);
            }
            self.behavior.social_mgr.recent_ai_hvs.push(ai_hv);
            if self.behavior.social_mgr.recent_input_hvs.len() >= 4 {
                self.behavior.social_mgr.recent_input_hvs.remove(0);
            }
            self.behavior.social_mgr.recent_input_hvs.push(input_hv);
        }

        Ok(PerceptionPhaseResult {
            encoding: PercEncoding {
                encoding_result: encoding.encoding_result,
                hv16_cached: encoding.hv16_cached,
                compressed_state: encoding.compressed_state,
                phi_attention_weight: encoding.phi_attention_weight,
                input_memoized: encoding.input_memoized,
                input_similarity: encoding.input_similarity,
                memo_threshold: encoding.memo_threshold,
                effective_threshold: encoding.effective_threshold,
                temporal_binding_strength: encoding.temporal_binding_strength,
            },
            moral: PercMoral {
                moral_concern_detected,
                moral_score,
                moral_judgment,
                soul_alignment: encoding.soul_alignment,
                moral_affect_coords: spinozist_affect_coords,
                moral_fluctuatio_tension: spinozist_fluctuatio,
                moral_is_ambiguous: spinozist_ambiguous,
                moral_epistemic_confidence: spinozist_confidence,
            },
            strategy: PercStrategy {
                selected_strategy,
                agency_strategy_override,
                social_strategy_bias,
            },
            exploration: PercExploration {
                exploration_urge_start,
                surprise_triggered: encoding.surprise_triggered,
                exploration_action: encoding.exploration_action,
            },
            urgency: PercUrgency {
                urgency: encoding.urgency,
                error_pattern: encoding.error_pattern,
                predicted_urgency: encoding.predicted_urgency,
                prediction_coherence_urgency_bias: encoding.prediction_coherence_urgency_bias,
                prediction_error: encoding.prediction_error,
                error_slope: encoding.error_slope,
                oscillation_ratio: encoding.oscillation_ratio,
            },
            math: math_result,
            startup_suppressed,
            startup_warmup_progress,
            negation_detected,
            #[cfg(feature = "vision-manifold")]
            vision_mean_surprise,
            #[cfg(feature = "vision-manifold")]
            cross_manifold_prediction_error,
            #[cfg(feature = "vision-manifold")]
            vision_horizon_errors,
            #[cfg(feature = "vision-manifold")]
            scene_recognized,
            #[cfg(feature = "vision-manifold")]
            vision_telemetry,
            #[cfg(feature = "foveation")]
            foveation_recognition_count,
            #[cfg(feature = "foveation")]
            foveation_top_confidence,
        })
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
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
        assert!(
            result.is_ok(),
            "phase_perception should return Ok for normal input"
        );
    }

    #[test]
    fn perception_result_has_finite_fields() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc
            .phase_perception("test input", Instant::now(), &mut timings)
            .unwrap();
        assert!(result.moral.moral_score.is_finite());
        assert!(result.encoding.phi_attention_weight.is_finite());
        assert!(result.urgency.prediction_error.is_finite());
        assert!(result.moral.soul_alignment.is_finite());
        assert!(result.negation_detected.is_finite());
        assert!(result.exploration.exploration_urge_start.is_finite());
        assert!(result.encoding.effective_threshold.is_finite());
    }

    #[test]
    fn perception_encoding_result_populated() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc
            .phase_perception("encoding check", Instant::now(), &mut timings)
            .unwrap();
        assert!(!result.encoding.encoding_result.hdv.values.is_empty());
        assert!(result.encoding.encoding_result.peak_attention.is_finite());
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
        assert!(svc.behavior.social_mgr.recent_ai_hvs.len() <= 4);
        assert!(svc.behavior.social_mgr.recent_input_hvs.len() <= 4);
    }

    #[test]
    fn perception_moral_timing_recorded() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let _ = svc.phase_perception("moral timing", Instant::now(), &mut timings);
        assert!(timings.moral_algebra < 10_000_000);
    }

    #[test]
    #[cfg(feature = "mathematics")]
    fn perception_detects_math_intent() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc
            .phase_perception("solve the linear system Ax=b", Instant::now(), &mut timings)
            .unwrap();
        assert!(result.math.math_detected, "should detect math intent");
        assert_eq!(
            result.math.problem_type,
            Some(crate::cognitive_loop::math_service::MathProblemType::LinearSystem)
        );
        assert!(
            timings.math_service < 10_000_000,
            "math service timing should be recorded"
        );
    }

    #[test]
    #[cfg(feature = "mathematics")]
    fn perception_no_math_for_normal_input() {
        let mut svc = make_service();
        let mut timings = ModuleTimings::default();
        let result = svc
            .phase_perception("hello world", Instant::now(), &mut timings)
            .unwrap();
        assert!(!result.math.math_detected);
        assert!(result.math.problem_type.is_none());
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
