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
        let mut encoding_res = self.run_encoding_and_preprocessing(input, module_timings);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 1.2b: Additive sensory modality blend
        // ═══════════════════════════════════════════════════════════════════════
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

        #[cfg(feature = "sensor-imu")]
        if let (Some(ref fusion), Some(ref reading)) = (&self.imu_fusion, &self.latest_imu_reading)
        {
            let imu_chv = fusion.fuse(reading);
            aux_sensor_hvs.push(crate::hdc::BinaryHV::from_bipolar(&imu_chv.values));
        }

        // Feel Metabolic Stress
        if let Some(ref mut conductor) = self.metabolic_conductor {
            let _homeostasis = conductor.tick();
            let stress_hv = conductor.encode_metabolic_stress();
            if stress_hv.as_slice().iter().any(|&v| v != 0.0) {
                aux_sensor_hvs.push(stress_hv.to_binary(0.0));
            }
        }

        if !aux_sensor_hvs.is_empty() {
            aux_sensor_hvs.insert(0, encoding_res.hv16_cached);
            encoding_res.hv16_cached = crate::hdc::BinaryHV::bundle(&aux_sensor_hvs);
            encoding_res.encoding_result.hdv = encoding_res.hv16_cached.to_continuous();
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
        let mut vision_telemetry = None;
        #[cfg(feature = "vision-manifold")]
        let mut vision_mean_surprise = 0.0;
        #[cfg(feature = "vision-manifold")]
        let mut vision_horizon_errors = Vec::new();
        #[cfg(feature = "vision-manifold")]
        let mut visual_hv = None;
        #[cfg(feature = "vision-manifold")]
        let mut final_dim = 16384;
        #[cfg(feature = "vision-manifold")]
        let mut scene_recognized = false;

        #[cfg(feature = "vision-manifold")]
        if let Some(ref mut bridge) = self.sensorimotor.vision_sensory.vision_bridge {
            let pixels = self
                .sensorimotor
                .vision_sensory
                .vision_frame_buffer
                .as_ref()
                .map(|f| f.as_slice())
                .unwrap_or(&[]);
            let w = self.config.vision_frame_width;
            let h = self.config.vision_frame_height;
            let channels = if bridge.manifold().encoder().config().enable_color {
                3
            } else {
                1
            };
            let dt = self.config.cfc_config.delta_t;

            // Store initial dimension to detect dilation/constriction during processing
            let dim_before = bridge.manifold().hdc_dim();

            let (hv, telemetry) = bridge.process_frame_with_telemetry(pixels, w, h, channels, dt);
            visual_hv = Some(hv);

            // ── Holographic Dilation Trigger (Hardcoded override) ──
            let manifold = bridge.manifold_mut();
            let dim_after_fep = manifold.hdc_dim();
            let current_cycle = manifold.frame_count();
            let last_change = manifold.last_dilation_cycle();

            vision_mean_surprise = manifold.surprise_map().mean_surprise();
            vision_horizon_errors = manifold.evaluate_horizons().errors;
            scene_recognized = telemetry.scene_recognition_similarity > 0.0;

            if current_cycle >= last_change + super::thresholds::VISION_DILATION_COOLDOWN {
                let target = if vision_mean_surprise
                    > super::thresholds::VISION_SURPRISE_DILATION_THRESHOLD
                {
                    Some(symthaea_core::hdc::HdcDimensionality::Ultra)
                } else if dim_after_fep > 16384 {
                    Some(symthaea_core::hdc::HdcDimensionality::Standard)
                } else {
                    None
                };

                if let Some(t) = target {
                    if dim_after_fep != t.dimension() {
                        bridge.dilate(t);
                    }
                }
            }

            // Sync cost and apply thermodynamic load if we ended up in Ultra mode
            final_dim = bridge.manifold().hdc_dim();
            if final_dim > dim_before && final_dim > 16384 {
                // Resolution increased to Ultra: apply one-time dilation cost
                self.thermodynamic_load = (self.thermodynamic_load + 0.08).min(1.0);
            }

            // Apply accumulated vision compute costs (geodesics, dreaming, etc.)
            let cost = bridge.manifold().telemetry().last_geodesic_cost;
            if cost > 0.0 {
                self.thermodynamic_load = (self.thermodynamic_load + cost).min(1.0);
            }

            // Sync metrics for snapshot visibility
            let manifold = bridge.manifold();
            let fep = manifold.last_fep();
            self.carryover.quality.last_vision_hdc_dim = final_dim as u32;
            self.carryover.quality.last_vision_free_energy = fep.free_energy;
            self.carryover.quality.last_vision_complexity = fep.complexity;
            self.carryover.quality.last_vision_accuracy = fep.accuracy;

            vision_telemetry = Some(telemetry);
        }

        // Cross-manifold predictor
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

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 1.1: Surprise and Exploration
        // ═══════════════════════════════════════════════════════════════════════
        let prediction_error = encoding_res.prediction_error;
        let (surprise_triggered, exploration_action) =
            self.run_surprise_exploration(&encoding_res.compressed_state);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 1.2: Urgency Phase
        // ═══════════════════════════════════════════════════════════════════════
        let urgency_result = self.compute_urgency_and_error_pattern(
            prediction_error,
            surprise_triggered,
            encoding_res.effective_threshold,
        );

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
                    let ne = self.neuromod.bath.noradrenaline.effective();
                    let da = self.neuromod.bath.dopamine.effective();
                    fov.modulate(ne, da);

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
                            timestamp_us: frame_count * 20_000,
                        };
                        fov.on_frame(fb);

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

                    let now_us = self.start_time.elapsed().as_micros() as u64;
                    fov.tick(now_us);
                    collected = fov.drain_results();
                    let tau_scale = self.substrate_manager.tau_factor.max(0.5);
                    let max = ((self.config.foveation_max_dispatches as f32) * tau_scale)
                        .round()
                        .max(1.0) as usize;
                    if collected.len() > max {
                        collected.sort_by(|a, b| {
                            b.confidence
                                .partial_cmp(&a.confidence)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        collected.truncate(max);
                    }
                    if let Some(ref mut bridge) = self.sensorimotor.vision_sensory.vision_bridge {
                        for result in &collected {
                            bridge.dampen_patch_surprise(result.grid_row, result.grid_col, 0.5);
                        }
                    }
                }
            }
            collected
        };

        #[cfg(feature = "foveation")]
        let foveation_recognition_count = fov_results.len();
        #[cfg(feature = "foveation")]
        let foveation_top_confidence = fov_results.iter().map(|r| r.confidence).fold(0.0, f32::max);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 1.5: Vision & Foveation Blending (Join into Cognitive Stream)
        // ═══════════════════════════════════════════════════════════════════════
        #[cfg(feature = "vision-manifold")]
        {
            let mut vision_hvs = Vec::new();
            if let Some(ref vis_hv) = visual_hv {
                vision_hvs.push(symthaea_core::hdc::BinaryHV::from_bipolar(
                    vis_hv.as_slice(),
                ));
            }
            #[cfg(feature = "foveation")]
            for fov_res in &fov_results {
                vision_hvs.push(symthaea_core::hdc::BinaryHV::from_bipolar(
                    fov_res.semantic_hv.as_slice(),
                ));
            }
            if !vision_hvs.is_empty() {
                vision_hvs.insert(0, encoding_res.hv16_cached);
                encoding_res.hv16_cached = symthaea_core::hdc::BinaryHV::bundle(&vision_hvs);
                encoding_res.encoding_result.hdv = encoding_res.hv16_cached.to_continuous();
            }
        }

        Ok(PerceptionPhaseResult {
            encoding: PercEncoding {
                encoding_result: encoding_res.encoding_result,
                hv16_cached: encoding_res.hv16_cached,
                compressed_state: encoding_res.compressed_state,
                phi_attention_weight: encoding_res.phi_attention_weight,
                input_memoized: encoding_res.input_memoized,
                input_similarity: encoding_res.input_similarity,
                memo_threshold: encoding_res.memo_threshold,
                effective_threshold: encoding_res.effective_threshold,
                temporal_binding_strength: encoding_res.temporal_binding_strength,
            },
            moral: PercMoral {
                moral_score,
                moral_concern_detected,
                moral_judgment,
                soul_alignment: 0.0,
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
                surprise_triggered,
                exploration_action,
            },
            urgency: PercUrgency {
                urgency: urgency_result.urgency,
                error_pattern: urgency_result.error_pattern,
                predicted_urgency: urgency_result.predicted_urgency,
                prediction_coherence_urgency_bias: urgency_result.prediction_coherence_urgency_bias,
                prediction_error,
                error_slope: urgency_result.error_slope,
                oscillation_ratio: urgency_result.oscillation_ratio,
            },
            math: math_result,
            startup_suppressed,
            startup_warmup_progress,
            negation_detected: input_negation_polarity,
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
}
