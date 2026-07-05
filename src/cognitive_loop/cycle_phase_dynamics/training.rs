// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use ndarray::Array1;
#[cfg(feature = "parallel")]
use rayon::join as rayon_join;
use std::time::Instant;

use super::super::feedback_state::Priority;
use super::super::helpers;
use super::super::phase_results::{DynMath, PerceptionPhaseResult};
use super::super::thresholds::*;
use super::super::training::TrainingSample;
use super::super::{CognitiveLoopService, CycleLearningResult, TrainingMethod};
use crate::consciousness::fep_active_inference::Observation;

impl CognitiveLoopService {
    /// Training dispatch, stats update, Broca generation, and parallel post-processing.
    ///
    /// Performs CfC weight update (sync or async), glutamate feedback, goal progress,
    /// statistics update, school learning, causal attention, Broca SSM generation,
    /// and rayon-parallel episodic/semantic post-processing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn phase_dynamics_training(
        &mut self,
        input: &str,
        perception: &PerceptionPhaseResult,
        prediction_error: f32,
        effective_lr: f32,
        delta_t: f32,
        previous_state: Option<&[f32]>,
        output: &[f32],
        coherence: f32,
        semantic_hdc: Vec<f32>,
        urgency: super::super::CycleUrgency,
        selected_strategy: super::super::flow::ResponseStrategy,
        surprise_triggered: bool,
        memory_context_boost: f32,
        cycle_start: Instant,
        math_result: &DynMath,
        semantic_lr_factor: f32,
        module_timings: &mut super::super::ModuleTimings,
        fep_surprise: f32,
        fep_pragmatic_value: f32,
    ) -> super::TrainingPostResult {
        let neuromod_threshold =
            perception.encoding.effective_threshold * self.neuromod.bath.threshold_gate();

        // 11. Learn if error is significant
        let _t_core = Instant::now();
        let consciousness_awake =
            self.carryover.history.consciousness_level > 0.0 || self.stats.total_cycles < 20;
        let (learning_occurred, training_loss) = if prediction_error > neuromod_threshold
            && !self.behavior.adaptive_behavior.pause_learning
            && !self.carryover.quality.narrative_veto_active
            && consciousness_awake
        {
            self.stats.learning_cycles += 1;

            let (train_input, train_target, lr) = if let Some(prev) = previous_state {
                (
                    prev.iter().copied().collect::<Array1<f32>>(),
                    perception
                        .encoding
                        .compressed_state
                        .iter()
                        .copied()
                        .collect(),
                    effective_lr,
                )
            } else {
                let train_input: Array1<f32> = perception
                    .encoding
                    .compressed_state
                    .iter()
                    .copied()
                    .collect();
                let train_target: Array1<f32> = perception
                    .encoding
                    .compressed_state
                    .iter()
                    .copied()
                    .collect();
                (train_input, train_target, effective_lr * 0.1)
            };

            // Compute vision-surprise importance weight for training
            #[cfg(feature = "vision-manifold")]
            let importance = (TRAINING_BASE_IMPORTANCE
                + perception.cross_manifold_prediction_error * VISION_TRAINING_IMPORTANCE_SCALE
                + perception.vision_mean_surprise * VISION_SURPRISE_TRAINING_IMPORTANCE_SCALE)
                .min(TRAINING_MAX_IMPORTANCE);
            #[cfg(not(feature = "vision-manifold"))]
            let importance = TRAINING_BASE_IMPORTANCE;

            if let Some(ref mut trainer) = self.async_trainer {
                trainer.send(TrainingSample {
                    input: train_input,
                    target: train_target,
                    dt: delta_t,
                    learning_rate: lr,
                    method: self.config.training_method,
                    avg_loss: self.stats.avg_training_loss,
                    importance,
                });
                (true, None)
            } else {
                let result = match self.config.training_method {
                    TrainingMethod::Spsa => {
                        self.stats.spsa_fallback_steps += 1;
                        self.temporal_network.train_step_spsa(
                            &train_input,
                            &train_target,
                            delta_t,
                            lr,
                        )
                    }
                    TrainingMethod::Bptt => {
                        self.stats.bptt_steps += 1;
                        self.temporal_network.train_step_bptt(
                            &train_input,
                            &train_target,
                            delta_t,
                            lr,
                        )
                    }
                    TrainingMethod::BpttWithSpsaFallback => {
                        let old_loss = self.stats.avg_training_loss;
                        let bptt_result = self.temporal_network.train_step_bptt(
                            &train_input,
                            &train_target,
                            delta_t,
                            lr,
                        );
                        match bptt_result {
                            Ok(loss)
                                if loss.is_finite()
                                    && (old_loss <= 0.0 || loss < old_loss * 2.0) =>
                            {
                                self.stats.bptt_steps += 1;
                                Ok(loss)
                            }
                            _ => {
                                self.stats.spsa_fallback_steps += 1;
                                self.temporal_network.train_step_spsa(
                                    &train_input,
                                    &train_target,
                                    delta_t,
                                    lr,
                                )
                            }
                        }
                    }
                };

                match result {
                    Ok(loss) => {
                        self.update_loss_stats(loss);
                        (true, Some(loss))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "CfC core training step failed");
                        (false, None)
                    }
                }
            }
        } else {
            (false, None)
        };
        module_timings.core_training = _t_core.elapsed().as_micros() as u64;

        // #13: Report learning activity to glutamate channel
        {
            let is_night =
                self.biorhythm_mgr.rhythm.phase == crate::chronobiology::CircadianPhase::Night;
            self.neuromod
                .bath
                .report_learning(effective_lr, prediction_error, is_night);
            let fatigue = self.neuromod.bath.learning_fatigue_factor();
            if fatigue < 1.0 {
                self.scale_lr_pri("glutamate_fatigue", fatigue, Priority::Homeostatic);
            }
        }

        // Goal←Cognition feedback
        if !learning_occurred && self.carryover.urgency.consecutive_low_error > 5 {
            if let Some(top) = self.fep.goal_system.top_goal() {
                let top_id = top.id.clone();
                let delta = (GOAL_DELTA_BASE_STEP as f64
                    * (1.0 + self.prediction_confidence * GOAL_DELTA_CONFIDENCE_SCALE as f64))
                    as f32;
                self.fep.goal_system.update_progress(&top_id, delta);
            }
        }

        // 12. Update statistics
        self.update_stats(prediction_error, cycle_start.elapsed());

        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        let consciousness_resize_factor = 1.0
            + (self.carryover.history.consciousness_level as f32 - CONSCIOUSNESS_RESIZE_CENTER)
                * CONSCIOUSNESS_RESIZE_SCALE;
        self.temporal_network
            .maybe_resize(prediction_error * consciousness_resize_factor);

        self.stats.temporal_coherence = coherence;
        self.stats.effective_learning_rate = effective_lr;
        self.stats.coherence_phi_contribution =
            self.language_comm.voice_coherence.bridge.phi_contribution();

        #[cfg(feature = "school_learning")]
        let school_predicted_phi_gain =
            if self.stats.total_cycles % super::super::thresholds::SCHOOL_LEARNING_INTERVAL == 0 {
                if let Some(ref school) = self.feature_integ.school_bridge {
                    match school.recommend_next() {
                        Ok(r) if r.predicted_phi_gain > 0.001 => r.predicted_phi_gain,
                        Ok(_) => 0.0,
                        Err(e) => {
                            tracing::debug!(error = %e, "School bridge recommend_next failed");
                            0.0
                        }
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
        #[cfg(not(feature = "school_learning"))]
        let school_predicted_phi_gain = 0.0f32;

        let causal_attention_boost =
            if self.stats.total_cycles % super::super::thresholds::CAUSAL_STRUCTURE_INTERVAL == 0 {
                if let Some(ref mut cc) = self.feature_integ.causal_consciousness {
                    let vars: Vec<Vec<f64>> = perception
                        .encoding
                        .compressed_state
                        .chunks(8)
                        .map(|chunk: &[f32]| chunk.iter().map(|&v| v as f64).collect())
                        .collect();
                    if vars.len() >= 2 {
                        let attention = cc.attention.compute_attention(&vars);
                        let top_strength = attention
                            .iter()
                            .enumerate()
                            .flat_map(|(i, row)| {
                                row.iter()
                                    .enumerate()
                                    .filter(move |&(j, _)| i != j)
                                    .map(|(_, &v)| v)
                            })
                            .fold(0.0f64, f64::max);
                        if top_strength > CAUSAL_ATTENTION_STRENGTH_THRESHOLD as f64 {
                            top_strength as f32
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
        if causal_attention_boost > 0.0 {
            // Amplified from ×0.05 to be behaviorally meaningful.
            // Science: Pearl (2000) — strong causal structure justifies confidence.
            self.adjust_confidence(
                "causal_attention",
                causal_attention_boost * CAUSAL_ATTENTION_CONFIDENCE_SCALE,
            );
        }

        // ── Broca SSM language generation + feedback ─────────────────────────
        #[cfg(feature = "ssm_language")]
        self.run_broca_generation(
            prediction_error,
            surprise_triggered,
            coherence,
            effective_lr,
            math_result,
            &perception.encoding.encoding_result.detected_primitives,
            fep_surprise,
            fep_pragmatic_value,
        );

        // ── BrocaLite fallback: lightweight always-on language generation ────
        // When full Broca didn't fire this cycle (wrong cadence, low consciousness,
        // or ssm_language not enabled), BrocaLite provides basic consciousness-coupled
        // text generation from the current cycle state.
        if self.language_comm.last_broca_text.is_none() {
            let emo = self.unification_engine.emotional.state();
            let signals = super::super::broca_bridge::BrocaConsciousnessSignals {
                epistemic_confidence: coherence.clamp(0.0, 1.0),
                emotional_valence: emo.valence as f32,
                emotional_arousal: emo.arousal as f32,
                emotional_warmth: 0.5, // neutral warmth default
                consciousness_level: self.carryover.history.consciousness_level as f32,
                meta_awareness: coherence, // approximate from coherence
                coherence,
                knowledge_grounding: 0.5,
                detected_primitives: perception
                    .encoding
                    .encoding_result
                    .detected_primitives
                    .clone(),
                fep_surprise,
                fep_pragmatic_value,
                ..Default::default()
            };
            if let Some(result) = self
                .language_comm
                .broca_lite
                .generate_from_signals_with_input(&signals, Some(input))
            {
                self.language_comm.last_broca_text = Some(result.text.clone());
                self.language_comm.last_language_source = Some("broca_lite".into());

                // Also send to LLM channel for higher-quality async translation.
                // LLM response will be available in a future cycle via drain.
                if let Some(ref llm) = self.llm_language {
                    let _ = llm.send(super::super::llm_language_channel::LlmLanguageRequest {
                        input_text: input.to_string(),
                        signals: signals.clone(),
                        broca_lite_text: Some(result.text),
                        cycle_num: self.stats.total_cycles as u64,
                    });
                }
            }
        }

        // ── Drain LLM language responses: upgrade BrocaLite output with LLM text ──
        if let Some(ref llm) = self.llm_language {
            for response in llm.drain_responses() {
                if response.from_llm && !response.text.is_empty() {
                    // LLM response ready — upgrade language output
                    self.language_comm.last_broca_text = Some(response.text);
                    self.language_comm.last_language_source = Some("llm".into());
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PARALLEL POST-PROCESSING
        // ═══════════════════════════════════════════════════════════════════════
        let _t_core = Instant::now();

        let pp_total_cycles = self.stats.total_cycles;
        let pp_in_flow = self.behavior.flow_state.in_flow;
        let pp_emotional_valence = self.unification_engine.emotional.state().valence as f32;
        let pp_phi = self.unification_engine.psi as f32;
        let pp_smoothed_coh = coherence as f64;
        let pp_wm_importance_boost =
            self.fep.world_model.avg_error.clamp(0.0, 1.0) * WORLD_MODEL_ERROR_IMPORTANCE_SCALE;
        let pp_thalamic_salience = match self.cognitive_depth {
            super::super::CognitiveDepth::DeepThought => THALAMIC_DEEP_SALIENCE,
            super::super::CognitiveDepth::Cortical => 0.0,
            super::super::CognitiveDepth::Reflex => THALAMIC_REFLEX_SALIENCE,
        };
        let pp_learning_threshold = self.config.learning_threshold;

        let cycle_reward = self.compute_reward_signal(prediction_error, pp_learning_threshold);

        let cycle_learning_result = CycleLearningResult {
            reward: cycle_reward,
            strategy_used: selected_strategy,
            successful: prediction_error < pp_learning_threshold && pp_in_flow,
            prediction_error,
            coherence,
        };

        let (evicted_semantic, memory_confidence_boost) = {
            let stability_regime = &mut self.memory.memory_consol.stability_regime;
            let discovery_service = &mut self.memory.memory_consol.discovery_service;
            let semantic_memory = &mut self.memory.memory_consol.semantic_memory;
            let causal_enhancer = &mut self.memory.causal_enhancer;
            let episodic_memory = &mut self.fep.episodic_memory;
            let primitive_belief_bridge = &mut self.primitive_belief_bridge;
            let closed_learning_loop = &mut self.fep.closed_learning_loop;
            let fep_learning_signal = &mut self.fep.learning_signal;
            let prev_primitive_state = &mut self.prev_primitive_state;
            let resonator_memory = &mut self.memory.memory_consol.resonator_memory;

            module_timings.stability_regime = helpers::run_stability_regime(
                stability_regime,
                discovery_service,
                &perception.encoding.hv16_cached,
                delta_t,
                pp_total_cycles,
                urgency,
            );

            let episodic_ctx = helpers::EpisodicLearningContext {
                prediction_error,
                in_flow: pp_in_flow,
                input,
                compressed_state: &perception.encoding.compressed_state,
                emotional_valence: pp_emotional_valence,
                phi: pp_phi,
                total_cycles: pp_total_cycles,
                smoothed_coh: pp_smoothed_coh,
                detected_primitives: &perception.encoding.encoding_result.detected_primitives,
                memory_context_boost,
                wm_importance_boost: pp_wm_importance_boost + pp_thalamic_salience,
                #[cfg(feature = "turbo-quant")]
                full_hdv: Some(&perception.encoding.encoding_result.hdv),
            };

            {
                let semantic_fn = || {
                    helpers::parallel_semantic_causal(
                        semantic_memory,
                        causal_enhancer,
                        semantic_hdc,
                        &perception.encoding.compressed_state,
                        output,
                        prediction_error,
                        pp_total_cycles,
                    )
                };
                let episodic_fn = || {
                    helpers::parallel_episodic_learning(
                        episodic_memory,
                        resonator_memory,
                        primitive_belief_bridge,
                        prev_primitive_state,
                        fep_learning_signal,
                        closed_learning_loop,
                        &episodic_ctx,
                        cycle_learning_result,
                    )
                };
                #[cfg(feature = "parallel")]
                {
                    use std::panic::AssertUnwindSafe;
                    let (sem, epi) = rayon_join(
                        || {
                            std::panic::catch_unwind(AssertUnwindSafe(semantic_fn))
                                .unwrap_or_else(|_| {
                                    tracing::error!("Parallel Branch A (semantic/causal) panicked — returning None");
                                    None
                                })
                        },
                        || {
                            std::panic::catch_unwind(AssertUnwindSafe(episodic_fn))
                                .unwrap_or_else(|_| {
                                    tracing::error!("Parallel Branch B (episodic/learning) panicked — returning 0.0");
                                    0.0
                                })
                        },
                    );
                    (sem, epi)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    (semantic_fn(), episodic_fn())
                }
            }
        };
        // Phase 2: Route evicted semantic entries to graduation pipeline.
        // Evicted entries survived a full ring buffer rotation, so they're worth
        // considering for long-term storage. The MemoryCoordinator applies quality
        // filtering (min WM steps, psi threshold) before actual graduation.
        let had_semantic_eviction = evicted_semantic.is_some();
        if let Some(evicted) = evicted_semantic {
            let steps_survived = pp_total_cycles.saturating_sub(evicted.timestamp as usize) as u64;
            self.memory
                .memory_consol
                .memory_coordinator
                .queue_graduation(crate::memory::memory_coordinator::GraduationEvent {
                    content: symthaea_core::hdc::ContinuousHV::from_vec(evicted.hdc_vector),
                    label: evicted.category.unwrap_or_default(),
                    steps_survived,
                    final_activation: (1.0 - evicted.prediction_error).max(0.0) as f64,
                    psi_at_graduation: pp_phi as f64,
                    coherence_at_graduation: coherence as f64,
                    source: crate::memory::memory_coordinator::MemorySource::SemanticEviction,
                    is_verified: false,
                });
        }

        // Apply memory context boost to confidence after rayon::join (deferred from parallel branch)
        if memory_confidence_boost.abs() > f32::EPSILON {
            self.adjust_confidence("memory_context_boost", memory_confidence_boost);
        }

        module_timings.core_parallel_postprocess = _t_core.elapsed().as_micros() as u64;

        self.stats.semantic_hits = self
            .memory
            .memory_consol
            .semantic_memory
            .stats()
            .semantic_hits;
        self.stats.semantic_misses = self
            .memory
            .memory_consol
            .semantic_memory
            .stats()
            .semantic_misses;
        self.stats.semantic_lr_factor = semantic_lr_factor;
        self.stats.semantic_avg_retrieved_error = self
            .memory
            .memory_consol
            .semantic_memory
            .stats()
            .avg_retrieved_error;
        self.stats.semantic_entries_stored = self
            .memory
            .memory_consol
            .semantic_memory
            .stats()
            .total_stored;

        super::TrainingPostResult {
            learning_occurred,
            training_loss,
            effective_lr,
            cycle_reward,
            had_semantic_eviction,
            school_predicted_phi_gain,
        }
    }

    /// Broca SSM language generation + feedback.
    ///
    /// Demand-driven: generate only when consciousness is sufficient AND there's
    /// novel content worth articulating. Minimum cadence 7 to prevent spam.
    /// Biologically: Broca's area activates for speech production when there's
    /// something meaningful to express (Hickok & Poeppel 2007).
    #[cfg(feature = "ssm_language")]
    fn run_broca_generation(
        &mut self,
        prediction_error: f32,
        surprise_triggered: bool,
        coherence: f32,
        effective_lr: f32,
        math_result: &DynMath,
        detected_primitives: &[String],
        fep_surprise: f32,
        fep_pragmatic_value: f32,
    ) {
        let broca_psi = self.unification_engine.psi as f32;
        let broca_novelty = prediction_error > self.config.learning_threshold || surprise_triggered;
        // Attention fatigue → Broca cadence gating.
        // High fatigue → widen spacing (don't generate when attention depleted).
        // Science: Mackworth (1948) — vigilance decrement degrades production quality.
        let fatigue_spacing_boost = if self
            .consciousness
            .self_model_tier
            .attention_schema
            .as_ref()
            .map(|s| s.fatigue_level())
            .unwrap_or(0.0)
            > 0.6
        {
            3
        } else {
            0
        };
        // Governance urgency → Broca cadence modulation.
        #[cfg(feature = "mycelix")]
        let governance_spacing_boost = {
            let pending = self.governance_mgr.pending_event_count();
            let phi = self.governance_mgr.last_collective_phi();
            let urgency_boost: usize = if pending > 3 { 2 } else { 0 };
            let phi_boost: usize = if phi > 0.01 && phi < 0.3 { 2 } else { 0 };
            urgency_boost + phi_boost
        };
        #[cfg(not(feature = "mycelix"))]
        let governance_spacing_boost: usize = 0;
        // Cantor fractal depth → Broca cadence: deep recursion = deliberate speech.
        // Science: Goldman-Rakic (1996) — prefrontal recursion depth → utterance complexity.
        let cantor_spacing_boost = {
            use crate::cognitive_loop::thresholds::{
                CANTOR_DEPTH_BROCA_SPACING_BOOST, CANTOR_SURPRISE_BROCA_SPACING_BOOST,
                CANTOR_SURPRISE_BROCA_THRESHOLD,
            };
            let depth_boost = if self
                .cantor_dream
                .broadcast_buffer
                .last()
                .map(|crhv| crhv.depth > 5)
                .unwrap_or(false)
            {
                CANTOR_DEPTH_BROCA_SPACING_BOOST
            } else {
                0
            };
            let surprise_boost =
                if self.cantor_dream.dream_surprise > CANTOR_SURPRISE_BROCA_THRESHOLD {
                    CANTOR_SURPRISE_BROCA_SPACING_BOOST
                } else {
                    0
                };
            depth_boost + surprise_boost
        };
        // Glyph modality → Broca cadence spacing: Threshold/Metaharmonic = +2 cycles.
        // Science: Schooler (2002) — metacognitive shifts require processing pauses.
        #[cfg(feature = "glyph_codex")]
        let glyph_spacing_boost: usize = match self.glyph_manager.dominant_modality() {
            crate::hdc::glyph_basis::FieldModality::Threshold
            | crate::hdc::glyph_basis::FieldModality::Metaharmonic => 2,
            _ => 0,
        };
        #[cfg(not(feature = "glyph_codex"))]
        let glyph_spacing_boost: usize = 0;
        // Quality EMA → cadence: widen spacing when generation quality is poor.
        // Science: Levelt (1989) — speech production monitoring adjusts output rate.
        #[cfg(feature = "ssm_language")]
        let quality_spacing_boost: usize = {
            let qe = self.language_manager.quality_ema();
            if qe < super::super::thresholds::BROCA_QUALITY_CADENCE_THRESHOLD {
                2
            } else {
                0
            }
        };
        #[cfg(not(feature = "ssm_language"))]
        let quality_spacing_boost: usize = 0;
        let broca_min_spacing = if self.stats.tom_prediction_mismatch_ema > 0.5 {
            5 + fatigue_spacing_boost
                + governance_spacing_boost
                + cantor_spacing_boost
                + glyph_spacing_boost
                + quality_spacing_boost
        } else {
            7 + fatigue_spacing_boost
                + governance_spacing_boost
                + cantor_spacing_boost
                + glyph_spacing_boost
                + quality_spacing_boost
        };
        let broca_should_generate =
            broca_psi > 0.4 && broca_novelty && self.stats.total_cycles % broca_min_spacing != 0;
        if !broca_should_generate {
            return;
        }

        // Community mode → Broca tone modulation
        #[cfg(feature = "mycelix")]
        let (mode_valence_nudge, mode_arousal_nudge, mode_warmth) = {
            match self.governance_mgr.community_mode() {
                Some(crate::mycelix::collective_identity::CommunityMode::Exploratory) => {
                    (0.0, 0.05, 0.4)
                }
                Some(crate::mycelix::collective_identity::CommunityMode::Protective) => {
                    (0.0, -0.05, 0.7)
                }
                Some(crate::mycelix::collective_identity::CommunityMode::Creative) => {
                    (0.05, 0.03, 0.5)
                }
                Some(crate::mycelix::collective_identity::CommunityMode::Reflective) => {
                    (0.0, -0.03, 0.5)
                }
                None => (0.0, 0.0, 0.5),
            }
        };
        #[cfg(not(feature = "mycelix"))]
        let (mode_valence_nudge, mode_arousal_nudge, mode_warmth) = (0.0f32, 0.0f32, 0.5f32);

        // Phase 2: Compose NSM semantic HV directly from detected primitive names.
        // Uses UniversalSemantics to look up each prime by name and bundle them,
        // avoiding the circular round-trip through GroundedUnderstanding.understand().
        let (nsm_semantic_hv, nsm_semantic_confidence) = if detected_primitives.is_empty() {
            (None, 0.0_f32)
        } else {
            use symthaea_core::hdc::universal_semantics::{SemanticPrime, UniversalSemantics};
            let semantics = UniversalSemantics::new();
            // Map detected primitive names (e.g., "FEEL", "BAD") to SemanticPrime enums.
            // detected_primitives may contain non-NSM names (e.g., "CAUSE", "ACTION")
            // which won't match — that's fine, we just skip them.
            let matched_primes: Vec<SemanticPrime> = detected_primitives
                .iter()
                .filter_map(|name| {
                    symthaea_core::hdc::universal_semantics::SemanticPrime::from_name(name)
                })
                .collect();
            if matched_primes.is_empty() {
                (None, 0.0)
            } else {
                let prime_hvs: Vec<symthaea_core::hdc::binary_hv::BinaryHV> = matched_primes
                    .iter()
                    .map(|p| *semantics.get_prime(*p))
                    .collect();
                let bundled = symthaea_core::hdc::binary_hv::BinaryHV::bundle(&prime_hvs);
                let confidence = (matched_primes.len() as f32 / detected_primitives.len() as f32)
                    .clamp(0.0, 1.0);
                if confidence > super::super::thresholds::NSM_MIN_CONFIDENCE {
                    (Some(bundled.to_continuous()), confidence)
                } else {
                    (None, confidence)
                }
            }
        };

        // Pre-extract knowledge signals before mutable borrow of broca_manager.
        let km_grounding = self
            .knowledge_manager()
            .map(|km| {
                let s = km.signals();
                ((s.relevance * 0.6 + (1.0 - s.uncertainty) * 0.4) as f32).clamp(0.0, 1.0)
            })
            .unwrap_or(0.5);
        let km_context = self
            .knowledge_manager()
            .map(|km| km.top_grounded_facts(5))
            .unwrap_or_default();

        // Generate in a scoped borrow, then apply feedback outside
        let broca_feedback = if let Some(ref mut broca) = self.language_comm.broca_manager {
            let math_epistemic_penalty = if math_result.epistemic_caveat.is_some() {
                0.3
            } else if math_result.solved && math_result.multipath_verified {
                -0.1
            } else {
                0.0
            };
            // Modulate epistemic confidence by language quality EMA:
            // poor generation quality → lower confidence → more hedging.
            let lang_quality_factor = 0.7 + 0.3 * self.language_manager.quality_ema();
            let signals = super::super::broca_bridge::BrocaConsciousnessSignals {
                epistemic_confidence: ((self.carryover.quality.last_epistemic_confidence
                    - math_epistemic_penalty)
                    * lang_quality_factor)
                    .clamp(0.0, 1.0),
                emotional_valence: self.unification_engine.emotional.state().valence as f32
                    + mode_valence_nudge,
                emotional_arousal: self.unification_engine.emotional.state().arousal as f32
                    + mode_arousal_nudge,
                emotional_warmth: mode_warmth,
                consciousness_level: broca_psi,
                meta_awareness: self.carryover.learning.self_model_accuracy as f32,
                coherence,
                knowledge_grounding: km_grounding,
                knowledge_context: km_context,
                #[cfg(feature = "therapeutic")]
                therapeutic_intent: if self.therapeutic_manager.crisis_active {
                    7.0
                } else {
                    self.therapeutic_manager
                        .active_strategy()
                        .map(|s| s.intent_code())
                        .unwrap_or(0.0)
                },
                #[cfg(feature = "therapeutic")]
                alliance_quality: self.therapeutic_manager.alliance_composite(),
                #[cfg(feature = "therapeutic")]
                client_distress_level: self.therapeutic_manager.client_distress(),
                #[cfg(feature = "therapeutic")]
                intervention_depth: self
                    .therapeutic_manager
                    .active_strategy()
                    .map(|s| s.min_alliance())
                    .unwrap_or(0.0),
                ethics_blocked: self
                    .ethics_verdict_override
                    .unwrap_or(self.last_ethics_verdict)
                    == super::super::ethics_engine::EthicalVerdict::Blocked,
                // Merge discourse memory: recurring primes from recent generations
                // get added to active_primes for topic continuity.
                // Science: Pickering & Garrod (2004) — alignment via shared priming.
                detected_primitives: {
                    let mut primes = detected_primitives.to_vec();
                    let discourse_primes = broca.recurring_discourse_primes(0.3);
                    for dp in discourse_primes {
                        if !primes.contains(&dp) {
                            primes.push(dp);
                        }
                    }
                    primes
                },
                primitive_grounding: if detected_primitives.is_empty() {
                    0.0
                } else {
                    // Estimate: each primitive maps roughly to one input concept.
                    // Cap at 1.0 (perfect decomposition).
                    (detected_primitives.len() as f32 / 10.0).clamp(0.0, 1.0)
                },
                // Phase 2: Compose NSM semantic content vector from detected primitives.
                // Use GroundedUnderstanding to build a BinaryHV → ContinuousHV.
                semantic_hv: nsm_semantic_hv.clone(),
                semantic_confidence: nsm_semantic_confidence,

                // Epistemic Cube: populated every cycle in cycle_subsystems.rs
                // from epistemic confidence, social context, knowledge grounding, and phi.
                cube_e_tier: self.carryover.quality.last_cube_e_tier,
                cube_n_tier: self.carryover.quality.last_cube_n_tier,
                cube_m_tier: self.carryover.quality.last_cube_m_tier,
                cube_h_value: self.carryover.quality.last_cube_h_value,
                cube_quality: self.carryover.quality.last_cube_quality,
                code_channels: self.language_comm.broca_code_channels.take(),
                fep_surprise,
                fep_pragmatic_value,
            };

            // ── Epistemic: Sacred Stillness modulates confidence ──
            // If generating about a domain with known unknowns, reduce confidence
            // to encourage hedged/careful language (epistemic humility).
            #[cfg(feature = "epistemic")]
            let signals = {
                let mut s = signals;
                if let Some(ref ku) = self.known_unknowns {
                    // Use knowledge context to detect domain
                    for fact in &s.knowledge_context {
                        let domain =
                            crate::knowledge::claim_priority::ClaimPrioritizer::infer_domain(fact);
                        let modifier = ku.modulate_confidence(&domain) as f32;
                        if modifier < 1.0 {
                            s.epistemic_confidence *= modifier;
                        }
                    }
                }
                s
            };

            if let Some(mut result) = broca.generate(signals) {
                if !result.text.is_empty() {
                    #[cfg(feature = "therapeutic")]
                    let text = if self.config.enable_therapeutic {
                        self.therapeutic_manager
                            .scope_guard
                            .apply_disclaimers(&result.text)
                    } else {
                        result.text.clone()
                    };
                    #[cfg(not(feature = "therapeutic"))]
                    let text = std::mem::take(&mut result.text);
                    self.language_comm.last_broca_text = Some(text);
                    self.language_comm.last_language_source = Some("broca".into());
                }

                // ── Factcheck bridge: extract claims from Broca output ──
                #[cfg(all(feature = "mycelix", feature = "ssm_language"))]
                if let Some(ref broca_text) = self.language_comm.last_broca_text {
                    let cycle_num = self.stats.total_cycles as u64;
                    // Epistemic: prioritize claims before submitting for validation.
                    // High-stakes domains (health, safety) checked first; low-priority skipped.
                    #[cfg(feature = "epistemic")]
                    let broca_text_for_factcheck = {
                        use crate::knowledge::claim_priority::ClaimPrioritizer;
                        let claims =
                            super::super::broca_factcheck::BrocaFactcheckBridge::extract_claims(
                                broca_text,
                            );
                        if !claims.is_empty() {
                            let prioritizer = ClaimPrioritizer::new(Default::default());
                            let prioritized = prioritizer.prioritize(&claims);
                            let top_claims: Vec<String> =
                                prioritized.into_iter().map(|p| p.claim).collect();
                            self.factcheck_bridge.submit_for_verification(&top_claims);
                        }
                        // Process pending verdicts from previous cycle
                        let modulation = self.factcheck_bridge.process_verdicts(cycle_num);
                        modulation
                    };
                    #[cfg(not(feature = "epistemic"))]
                    let broca_text_for_factcheck = self
                        .factcheck_bridge
                        .on_broca_generation(broca_text, cycle_num);

                    let _modulation = broca_text_for_factcheck;
                    // If factcheck says suppress, clear the output
                    if _modulation.suppress {
                        self.language_comm.last_broca_text = None;
                        tracing::info!(
                            target: "cognitive_loop::factcheck",
                            cycle = cycle_num,
                            "Broca output suppressed by factcheck bridge (high-confidence False verdict)"
                        );
                    }
                }

                #[cfg(feature = "liquid-mamba")]
                let semantic_pe = result.semantic_pe;
                #[cfg(not(feature = "liquid-mamba"))]
                let semantic_pe = 0.0_f32;
                let broca_quality = result.final_coherence * BROCA_QUALITY_COHERENCE_WEIGHT
                    + (1.0 - semantic_pe.min(1.0)) * BROCA_QUALITY_PE_WEIGHT
                    + result.long_coherence * BROCA_QUALITY_LONG_COHERENCE_WEIGHT;
                let broca_quality = broca_quality.clamp(0.0, 1.0);

                // EMA computed by LanguageManager; bridge to stats for backward compat.
                self.stats.broca_quality_ema = self.language_manager.quality_ema();
                self.stats.broca_generation_count += 1;

                if broca_quality < BROCA_LOW_QUALITY_THRESHOLD {
                    self.stats.broca_low_quality_streak =
                        self.stats.broca_low_quality_streak.saturating_add(1);
                } else {
                    self.stats.broca_low_quality_streak = 0;
                }

                if self.stats.broca_low_quality_streak >= 3 {
                    broca.consciousness_threshold = (broca.consciousness_threshold
                        + BROCA_CONSCIOUSNESS_THRESHOLD_INCREASE)
                        .min(BROCA_CONSCIOUSNESS_THRESHOLD_MAX);
                } else if self.language_manager.quality_ema() > BROCA_QUALITY_HIGH_THRESHOLD
                    && broca.consciousness_threshold > BROCA_CONSCIOUSNESS_THRESHOLD_MIN
                {
                    broca.consciousness_threshold = (broca.consciousness_threshold
                        - BROCA_CONSCIOUSNESS_THRESHOLD_DECREASE)
                        .max(BROCA_CONSCIOUSNESS_THRESHOLD_MIN);
                }

                broca.last_telemetry.quality = broca_quality;
                broca.last_telemetry.long_coherence = result.long_coherence;
                broca.last_telemetry.semantic_pe = semantic_pe;

                #[cfg(feature = "liquid-mamba")]
                let deferred_semantic_pe = result.semantic_pe;
                #[cfg(not(feature = "liquid-mamba"))]
                let deferred_semantic_pe = 0.0_f32;

                Some((
                    result.final_coherence,
                    broca_quality,
                    result.veto_triggered,
                    deferred_semantic_pe,
                ))
            } else {
                None
            }
        } else {
            None
        };

        // Apply deferred feedback outside the broca_manager borrow
        if let Some((final_coherence, _broca_quality, veto_triggered, deferred_sem_pe)) =
            broca_feedback
        {
            // ── Coherence → confidence: delegated to LanguageManager (confidence_delta) ──
            // ── Quality → LR boost: delegated to LanguageManager (lr_modulation) ──

            if veto_triggered {
                self.scale_exploration_pri(
                    "broca_veto",
                    super::super::thresholds::BROCA_VETO_EXPLORATION_SCALE,
                    Priority::Aesthetic,
                );
            }

            // Phase 4: NSM expressive coverage → consciousness feedback.
            // Use primitive grounding as a proxy for expressive coverage
            // (full NsmCoherenceTracker integration will replace this).
            // Science: Levelt (1989) — self-monitoring; Rosenthal (2005) — HOT theory.
            {
                let nsm_coverage = self
                    .language_comm
                    .broca_manager
                    .as_ref()
                    .map(|b| b.last_telemetry.nsm_grounding)
                    .unwrap_or(0.0);
                if nsm_coverage > 0.0 {
                    // Confidence modulation: coverage > 0.5 boosts, < 0.5 dampens.
                    let coverage_delta = (nsm_coverage - 0.5)
                        * super::super::thresholds::NSM_COVERAGE_CONFIDENCE_SCALE;
                    self.adjust_confidence_pri(
                        "nsm_expressive_coverage",
                        coverage_delta,
                        Priority::Aesthetic,
                    );
                    // Exploration modulation: high coverage → consolidate.
                    if nsm_coverage > 0.5 {
                        self.scale_exploration_pri(
                            "nsm_coverage_consolidate",
                            super::super::thresholds::NSM_COVERAGE_EXPLORATION_SCALE
                                * (nsm_coverage - 0.5),
                            Priority::Aesthetic,
                        );
                    }
                }
            }

            // Broca → Phi bidirectional feedback.
            // Articulating a thought is itself information integration: high-quality
            // generation (high coherence AND high NSM prime coverage) demonstrates
            // that the system successfully unified semantic content into coherent
            // output. This should reinforce consciousness level.
            // Science: Dehaene (2014) — global workspace broadcasting of linguistic
            // content is a signature of conscious access.
            {
                let nsm_cov = self
                    .language_comm
                    .broca_manager
                    .as_ref()
                    .map(|b| b.last_telemetry.nsm_prime_coverage)
                    .unwrap_or(0.0);
                // Composite quality: coherence × (0.5 + 0.5 × coverage)
                // High coherence alone gets partial credit; both together get full credit.
                let articulation_quality =
                    final_coherence * (0.5 + 0.5 * nsm_cov.max(0.0).min(1.0));
                if articulation_quality > 0.3 {
                    // Scale is small (±2%) to avoid runaway feedback loops.
                    let phi_boost = (articulation_quality - 0.3)
                        * super::super::thresholds::NSM_BROCA_PHI_SCALE;
                    self.unification_engine.psi =
                        (self.unification_engine.psi + phi_boost as f64).clamp(0.0, 1.0);
                }
            }

            let _ = deferred_sem_pe;
            #[cfg(feature = "liquid-mamba")]
            if deferred_sem_pe > 0.1 {
                let sem_obs = Observation::from_consciousness_state(
                    deferred_sem_pe as f64,
                    0.1,
                    coherence as f64,
                    effective_lr as f64,
                );
                self.fep.agent.perceive(&sem_obs);
            }
        }

        // Broca quality → attention budget (Levelt 1989 — monitoring loop)
        let broca_qe = self.language_manager.quality_ema();
        if broca_qe > 0.7 {
            let contraction = 1.0 - (broca_qe - 0.7) * 0.15;
            self.scale_confidence_pri("broca_attention_contract", contraction, Priority::Aesthetic);
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleResult};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default())
            .expect("default config must initialize")
    }

    fn run_cycles(svc: &mut CognitiveLoopService, n: usize, input: &str) -> Vec<CycleResult> {
        (0..n).map(|_| svc.cycle(input)).collect()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXISTING TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn dynamics_produces_finite_cfc_output() {
        let mut svc = make_service();
        let result = svc.cycle("dynamics finite check");
        for (i, &v) in result.output.iter().enumerate() {
            assert!(v.is_finite(), "CfC output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn dynamics_prediction_error_finite_non_nan() {
        let mut svc = make_service();
        let result = svc.cycle("prediction error check");
        assert!(result.prediction_error.is_finite());
        assert!(!result.prediction_error.is_nan());
    }

    #[test]
    fn dynamics_reasoning_gate_not_blocked_when_engine_absent() {
        let mut svc = make_service();
        let result = svc.cycle("reasoning gate test");
        // reasoning_engine is disabled by default, so the gate never fires
        assert!(!result.metadata.reasoning_gate_blocked);
    }

    #[test]
    fn dynamics_fep_fields_populated() {
        let mut svc = make_service();
        let result = svc.cycle("fep check");
        assert!(result.metadata.fep.fep_accuracy.is_finite());
        assert!(result.metadata.fep.fep_complexity.is_finite());
        assert!(result.metadata.fep.fep_surprise.is_finite());
    }

    #[test]
    fn dynamics_learning_occurred_flag_consistent() {
        let mut svc = make_service();
        let result = svc.cycle("learning flag");
        if result.learning_occurred {
            // training_loss is None when async_training=true (default)
            if let Some(loss) = result.training_loss {
                assert!(loss.is_finite());
            }
        }
    }

    #[test]
    fn dynamics_actual_effective_lr_zero_when_no_learning() {
        let mut cfg = CognitiveLoopConfig::default();
        // Max valid threshold (1.0) — PE rarely exceeds this on first cycle
        cfg.learning_threshold = 1.0;
        let mut svc = CognitiveLoopService::new(cfg).expect("test config must initialize");
        let result = svc.cycle("no learning");
        if !result.learning_occurred {
            assert_eq!(result.metadata.actual_effective_lr, 0.0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DELTA_T CHAIN STABILITY: Verify tau products stay finite & bounded
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn dynamics_delta_t_finite_across_many_cycles() {
        // The delta_t chain multiplies 9 factors. After 50 cycles with varied input,
        // the resulting CfC outputs must remain finite — no NaN/Inf propagation.
        let mut svc = make_service();
        let inputs = [
            "novel surprising stimulus",
            "familiar repeated pattern",
            "emotional high-arousal event",
            "calm consolidation phase",
            "ambiguous uncertain signal",
        ];
        for i in 0..50 {
            let result = svc.cycle(inputs[i % inputs.len()]);
            for (j, &v) in result.output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "CfC output[{j}] not finite at cycle {i}: {v}"
                );
            }
            assert!(
                result.prediction_error.is_finite(),
                "PE not finite at cycle {i}"
            );
        }
    }

    #[test]
    fn dynamics_cfc_output_bounded_magnitude() {
        // CfC outputs should not explode to extreme magnitudes. Verify they stay
        // within a reasonable range across 30 cycles (the exact range depends on
        // the network, but ±100 is conservative for normalized HDC inputs).
        let mut svc = make_service();
        let results = run_cycles(&mut svc, 30, "magnitude check");
        for (i, r) in results.iter().enumerate() {
            for (j, &v) in r.output.iter().enumerate() {
                assert!(
                    v.abs() < 100.0,
                    "CfC output[{j}] at cycle {i} has extreme magnitude: {v}"
                );
            }
        }
    }

    #[test]
    fn dynamics_prediction_error_bounded_after_warmup() {
        // After warmup (15 cycles), prediction error should stabilize. It can be
        // noisy early but should converge. Verify it stays in [0, 5] range.
        let mut svc = make_service();
        let results = run_cycles(&mut svc, 40, "pe stability");
        for (i, r) in results.iter().enumerate().skip(15) {
            assert!(
                r.prediction_error >= 0.0 && r.prediction_error < 5.0,
                "PE out of expected range at cycle {i}: {}",
                r.prediction_error
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // NUMERICAL STABILITY: FEP, EMA, and cascading computations
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn dynamics_fep_fields_stay_finite_over_many_cycles() {
        // FEP accuracy, complexity, surprise are EMA-updated each cycle.
        // Verify no NaN accumulation over 50 cycles.
        let mut svc = make_service();
        let results = run_cycles(&mut svc, 50, "fep stability");
        for (i, r) in results.iter().enumerate() {
            let fep = &r.metadata.fep;
            assert!(
                fep.fep_accuracy.is_finite(),
                "NaN fep_accuracy at cycle {i}"
            );
            assert!(
                fep.fep_complexity.is_finite(),
                "NaN fep_complexity at cycle {i}"
            );
            assert!(
                fep.fep_surprise.is_finite(),
                "NaN fep_surprise at cycle {i}"
            );
            assert!(
                fep.fep_td_error.is_finite(),
                "NaN fep_td_error at cycle {i}"
            );
        }
    }

    #[test]
    fn dynamics_homeostasis_efficiency_stays_bounded() {
        // Homeostasis efficiency is EMA-clamped to [0.5, 1.5]. Verify this holds
        // across many cycles with varied input that drives different valence dynamics.
        let mut svc = make_service();
        let inputs = [
            "positive valence stimulus",
            "negative valence stimulus",
            "neutral observation",
        ];
        for i in 0..60 {
            let result = svc.cycle(inputs[i % inputs.len()]);
            let eff = result.metadata.homeostasis_efficiency;
            assert!(
                eff.is_finite() && eff >= 0.5 && eff <= 1.5,
                "Homeostasis efficiency out of [0.5, 1.5] at cycle {i}: {eff}"
            );
        }
    }

    #[test]
    fn dynamics_prediction_coherence_finite() {
        // Prediction coherence is computed every 11 cycles.
        // Verify the EMA'd value stays finite.
        let mut svc = make_service();
        let results = run_cycles(&mut svc, 33, "coherence check");
        for (i, r) in results.iter().enumerate() {
            let coh = r.metadata.prediction_coherence;
            assert!(
                coh.is_finite(),
                "prediction_coherence NaN at cycle {i}: {coh}"
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MULTI-CYCLE CASCADE: EMA drift and velocity fields
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn dynamics_velocity_fields_finite_after_warmup() {
        // Coherence velocity, confidence velocity, and quality EMA fields are
        // computed as cycle-to-cycle deltas. Verify they don't drift to NaN/Inf.
        let mut svc = make_service();
        let results = run_cycles(&mut svc, 50, "velocity check");
        for (i, r) in results.iter().enumerate() {
            let q = &r.metadata.quality;
            assert!(
                q.coherence_velocity.is_finite(),
                "coherence_velocity NaN at cycle {i}"
            );
        }
    }

    #[test]
    fn dynamics_consciousness_level_finite_and_bounded() {
        // consciousness_level is the integrated consciousness score.
        // Verify it stays in [0, 1] and finite across cycles.
        let mut svc = make_service();
        let results = run_cycles(&mut svc, 50, "consciousness bounded");
        for (i, r) in results.iter().enumerate() {
            let cl = r.metadata.consciousness.consciousness_level;
            assert!(
                cl.is_finite() && cl >= 0.0 && cl <= 1.0,
                "consciousness_level out of [0,1] at cycle {i}: {cl}"
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LEARNING RATE INTERACTION: Dynamics → feedback LR cascade
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn dynamics_effective_lr_finite_under_varied_input() {
        // The effective LR is computed from PE, plasticity, and modulation factors.
        // Verify it stays finite and non-negative across varied inputs.
        let mut svc = make_service();
        let inputs = [
            "completely novel input alpha",
            "partially familiar pattern beta",
            "well-known repeated gamma",
        ];
        for i in 0..60 {
            let result = svc.cycle(inputs[i % inputs.len()]);
            let lr = result.metadata.actual_effective_lr;
            assert!(
                lr.is_finite() && lr >= 0.0,
                "effective_lr not valid at cycle {i}: {lr}"
            );
        }
    }

    #[test]
    fn dynamics_no_nan_in_key_metadata_across_100_cycles() {
        // Stress test: run 100 cycles and verify critical metadata fields stay finite.
        // This exercises long-running EMA accumulation and cross-cycle carryover.
        let mut svc = make_service();
        for i in 0..100 {
            let result = svc.cycle("stress test nan check");
            let m = &result.metadata;
            assert!(m.actual_effective_lr.is_finite(), "NaN LR at cycle {i}");
            assert!(
                m.consciousness.consciousness_level.is_finite(),
                "NaN consciousness at cycle {i}"
            );
            assert!(
                m.prediction_coherence.is_finite(),
                "NaN pred_coherence at cycle {i}"
            );
            assert!(
                m.temporal.holographic_unity.is_finite(),
                "NaN holographic_unity at cycle {i}"
            );
            assert!(
                m.temporal.holographic_binding.is_finite(),
                "NaN holographic_binding at cycle {i}"
            );
            assert!(
                m.homeostasis_efficiency.is_finite(),
                "NaN homeostasis at cycle {i}"
            );
            assert!(result.prediction_error.is_finite(), "NaN PE at cycle {i}");
            for (j, &v) in result.output.iter().enumerate() {
                assert!(v.is_finite(), "NaN output[{j}] at cycle {i}");
            }
        }
    }
}
