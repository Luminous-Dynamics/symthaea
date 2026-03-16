use std::time::Instant;

use super::{LateConsciousnessContext, LateConsciousnessResult};
use crate::cognitive_loop::CognitiveLoopService;

impl CognitiveLoopService {
    /// Run late consciousness monitors: prefrontal cortex, meta-cognition, virtual body,
    /// affective bridge, user state, narrative self, predictive processing, hierarchical
    /// free energy, predictive self, attention schema, and phi attention.
    ///
    /// Extracted from cycle.rs — all logic and behavior preserved exactly.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::cognitive_loop) fn run_late_consciousness_monitors(
        &mut self,
        ctx: &LateConsciousnessContext,
        module_timings: &mut crate::cognitive_loop::ModuleTimings,
    ) -> LateConsciousnessResult {
        // ═══════════════════════════════════════════════════════════════════════
        // PREFRONTAL CORTEX: Executive control and working memory gating
        // Amortized: Critical=every cycle, Normal=every 2nd, Cruise=every 5th
        // PFC inhibition is a stable control policy (Miller & Cohen 2001);
        // caching the veto for 5 cycles (~100ms at 50Hz) is within the
        // temporal binding window for executive control.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let prefrontal_veto = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 5) {
            if let Some(ref mut pfc) = self.prefrontal {
                // Add current input as a working memory item
                let wm_item = crate::brain::prefrontal::WorkingMemoryItem::new(
                    format!("cycle_{}", self.stats.total_cycles),
                    symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(
                        ctx.compressed_state.to_vec(),
                    ),
                );
                pfc.add_to_memory(wm_item);

                // Advance time (decay activations, evict expired items)
                pfc.tick();

                // Check memory utilization — high utilization triggers inhibition
                let utilization = pfc.memory_contents().len() as f32 / 7.0; // default capacity
                let veto = utilization > self.config.learning_threshold.max(0.8);

                if veto {
                    tracing::debug!(
                        utilization,
                        cycle = self.stats.total_cycles,
                        "Prefrontal veto: working memory overloaded"
                    );
                }

                // Graduate evicted items to episodic memory
                // Track 3h: WM eviction → resonator routing pipeline
                // Science: Frankland & Bontempi (2005) — systems consolidation routes
                // memories through hippocampal replay for categorical integration
                let graduates = pfc.drain_graduates();
                if !graduates.is_empty() {
                    for grad in &graduates {
                        // Route graduate through resonator for importance scoring
                        let grad_importance =
                            if let Some(ref mut res_mem) = self.memory_consol.resonator_memory {
                                let res_dim = res_mem.resonator.config.dim;
                                let grad_vals = &grad.embedding.values;
                                if grad_vals.len() >= res_dim && !res_mem.episodes.is_empty() {
                                    // Project to resonator dim and find best episode match
                                    let projected: Vec<f32> =
                                        grad_vals.iter().take(res_dim).copied().collect();
                                    let best_sim = res_mem
                                        .episodes
                                        .iter()
                                        .map(|ep| {
                                            let dot: f32 = ep
                                                .hv
                                                .iter()
                                                .zip(projected.iter())
                                                .map(|(a, b)| a * b)
                                                .sum();
                                            let denom =
                                                (ep.hv.iter().map(|x| x * x).sum::<f32>().sqrt()
                                                    * projected
                                                        .iter()
                                                        .map(|x| x * x)
                                                        .sum::<f32>()
                                                        .sqrt())
                                                .max(1e-8);
                                            let sim = dot / denom;
                                            if sim.is_finite() {
                                                sim.clamp(-1.0, 1.0)
                                            } else {
                                                0.0
                                            }
                                        })
                                        .fold(0.0f32, f32::max);
                                    // High resonator match → boost importance (consolidation-worthy)
                                    // Low match → novel content, still store but with base importance
                                    ctx.pp_phi + best_sim * 0.2
                                } else {
                                    ctx.pp_phi
                                }
                            } else {
                                ctx.pp_phi
                            };
                        // Route through MemoryCoordinator for quality filtering instead of
                        // bypassing directly to fep.episodic_memory.encode().
                        self.memory_consol.memory_coordinator.queue_graduation(
                            crate::memory::memory_coordinator::GraduationEvent {
                                content: grad.embedding.clone(),
                                label: grad.id.clone(),
                                steps_survived: self.stats.total_cycles as u64 - grad.added_at,
                                final_activation: grad_importance as f64,
                                psi_at_graduation: ctx.pp_phi as f64,
                                coherence_at_graduation: 0.0,
                                source: Default::default(),
                                is_verified: false,
                            },
                        );
                    }
                    tracing::debug!(
                        count = graduates.len(),
                        "Prefrontal graduates queued for memory coordinator"
                    );
                }

                self.carryover.quality.cached_prefrontal_veto = veto;
                veto
            } else {
                false
            }
        } else {
            // Reuse cached veto from last computed cycle
            self.carryover.quality.cached_prefrontal_veto
        };

        // FEEDBACK: Prefrontal veto suppresses exploration (executive control)
        // Science: Miller & Cohen (2001) — PFC inhibits impulsive exploration when WM overloaded
        if prefrontal_veto {
            self.set_exploration("prefrontal_veto", 0.0);

            // FEEDBACK: WM overload triggers emergency consolidation (Baddeley 2000)
            // Science: Working memory overflow should push items to long-term storage,
            // not just block exploration. Force episodic consolidation to free WM slots.
            self.fep.episodic_memory.consolidate_recent();
        }

        // FEEDBACK: Dual-veto freeze detection and recovery (Fuchs 2008 multistability)
        // Science: When reasoning gate AND prefrontal veto both fire, system is paralyzed:
        // exploration=0, learning=0. Soften both to allow partial recovery.
        if ctx.reasoning_gate_blocked && prefrontal_veto {
            self.set_exploration("dual_veto_freeze", 0.3);
            self.set_lr("dual_veto_freeze", self.fep.lr_boost.max(1.0) as f32);
            tracing::debug!(
                cycle = self.stats.total_cycles,
                "Dual-veto freeze detected: softening both gates for recovery"
            );
        }

        module_timings.prefrontal = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // META-COGNITION: Recursive self-modeling and learning rate modulation
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (meta_cognitive_accuracy, meta_cognitive_depth) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut meta) = self.self_model_tier.meta_cognition {
                    meta.update_self_model(ctx.prediction_error);
                    meta.deepen_recursion();
                    let accuracy = meta.accuracy();
                    let depth = meta.depth();
                    if accuracy > 0.7 {
                        let boost = 1.0 + (accuracy - 0.7) * 0.5; // up to 1.15x
                        self.carryover.learning.subsystem_lr_factor *= boost;
                    }
                    (accuracy, depth)
                } else {
                    (0.0, 0)
                }
            } else {
                // Read cached accuracy/depth without updating (avoid 0.0 in telemetry on skip)
                self.self_model_tier
                    .meta_cognition
                    .as_ref()
                    .map(|m| (m.accuracy(), m.depth()))
                    .unwrap_or((0.0, 0))
            };

        // ── Moral circularity → deepen meta-cognitive recursion ──
        // When β₁ > 0 (circular reasoning patterns detected in the moral
        // topology), trigger additional meta-cognitive introspection.
        if let Some(ref mut meta) = self.self_model_tier.meta_cognition {
            let topo = self.ethics_engine.moral_topology().last_summary();
            if topo.beta_1 > 0 {
                meta.deepen_recursion();
                tracing::debug!(
                    target: "cognitive_loop::moral_topology",
                    beta_1 = topo.beta_1,
                    cycle = self.stats.total_cycles,
                    "Moral circularity detected — deepening meta-cognitive recursion"
                );
            }
        }

        module_timings.meta_cognition = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // VIRTUAL BODY: Map cognitive signals to interoceptive states
        // Urgency-gated: Critical=always, Normal=always, Cruise=every 2nd
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (body_psi_modulation, body_valence, body_arousal) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 1, 2) {
                if let Some(ref mut body) = self.vision_sensory.virtual_body {
                    let signals = crate::cognitive_loop::virtual_body::CognitiveSignals {
                        prediction_error: ctx.prediction_error,
                        coherence: ctx.coherence,
                        prediction_confidence: self.prediction_confidence as f32,
                        unified_psi: ctx.unified_psi,
                        flow_intensity: self.flow_state.intensity,
                        in_flow: self.flow_state.in_flow,
                        curiosity_boredom: self.curiosity_drive.boredom,
                        fep_learning_signal: self.fep.learning_signal,
                        error_trend: self.stats.error_trend,
                        cycles_per_second: self.stats.cycles_per_second,
                        target_frequency: self.config.target_frequency,
                    };
                    let state = body.update(&signals);
                    self.carryover.consciousness.body_phi_modulation = state.phi_modulation;
                    self.carryover.history.body_arousal = state.arousal;
                    (state.phi_modulation, state.valence, state.arousal)
                } else {
                    (1.0, 0.0, 0.0)
                }
            } else {
                // Urgency-skipped: use carryover for phi_modulation and arousal; valence has no
                // carryover so use neutral 0.0 (lightweight — doesn't trigger somatic marker feedback).
                (
                    self.carryover.consciousness.body_phi_modulation,
                    0.0,
                    self.carryover.history.body_arousal,
                )
            };

        module_timings.virtual_body = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Body valence modulates prediction confidence (Damasio somatic markers)
        // Science: Damasio (1999) — positive somatic state boosts cognitive coherence;
        // negative somatic state signals danger → dampen confidence
        if body_valence > 0.3 {
            self.adjust_confidence("body_valence_pos", body_valence * 0.02);
        } else if body_valence < -0.3 {
            self.adjust_confidence("body_valence_neg", body_valence * 0.03);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // AFFECTIVE BRIDGE: Evaluate somatic markers from cognitive signals
        // Runs every cycle (lightweight: ~5 arithmetic ops + blend)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (affective_valence, affective_arousal) =
            if let Some(ref mut bridge) = self.consciousness_state.affective_bridge {
                let moral_score = self
                    .ethics_values
                    .last_moral_judgment
                    .as_ref()
                    .map(|j| j.moral_score)
                    .unwrap_or(0.0);
                // Social modulation: feed ToM signals into affect (Decety & Chaminade 2003)
                // Social trust/cooperation injected by Mind module via set_social_signals()
                let affect = bridge.evaluate_from_signals_with_social(
                    ctx.prediction_error,
                    ctx.surprise_triggered,
                    ctx.unified_psi,
                    moral_score,
                    self.social_mgr.social.social_trust,
                    self.social_mgr.social.social_cooperation_rate,
                    0.0, // peer_valence: future — aggregate from social inbox
                );
                (affect.valence, affect.arousal)
            } else {
                (0.0, 0.5)
            };

        // FEEDBACK: Positive affect broadens exploration (Fredrickson 2001 broaden-and-build)
        if affective_valence > 0.2 && self.consciousness_state.affective_bridge.is_some() {
            self.curiosity_drive.boredom *= 1.05;
        }
        // FEEDBACK: Arousal gates learning consolidation (Russell 1980 VAD model)
        // Science: Steriade (1996) — high arousal suppresses consolidation;
        // low arousal enhances it. Arousal trap detection + recovery in helper.
        self.manage_arousal_trap(affective_arousal);
        module_timings.affective_bridge = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // USER STATE INFERENCE: Infer cognitive load, frustration, engagement
        // Runs every cycle (lightweight: keyword detection + rolling averages)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut usi) = self.language_comm.user_state {
            let had_error = ctx.prediction_error > 0.8;
            usi.process(ctx.input, had_error);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // NARRATIVE SELF: Process experience and track self-Φ
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let narrative_self_psi = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut narrative) = self.self_model_tier.narrative_self {
                let significance = if ctx.moral_concern_detected {
                    0.8
                } else {
                    (ctx.prediction_error as f64).clamp(0.0, 1.0)
                };
                narrative.process_experience(
                    &ctx.hv16_cached,
                    ctx.input,
                    ctx.prediction_error < self.config.learning_threshold,
                    ctx.coherence as f64,
                    significance,
                );
                narrative.self_phi()
            } else {
                0.0
            }
        } else {
            // Read cached self_phi without processing (avoid 0.0 triggering weak-identity feedback)
            self.self_model_tier
                .narrative_self
                .as_ref()
                .map(|n| n.self_phi())
                .unwrap_or(0.0)
        };

        // FEEDBACK: Narrative self-Phi modulates prediction confidence (identity coherence)
        // Science: Gallagher (2000) — strong narrative identity stabilizes learning
        if narrative_self_psi > 0.5 {
            self.scale_confidence("narrative_self_strong", 1.02);
        } else if narrative_self_psi > 0.0 && narrative_self_psi < 0.2 {
            self.scale_confidence("narrative_self_weak", 0.95);
        }

        // FEEDBACK: Narrative self-Phi modulates moral sensitivity (Gallagher & Hutto 2007)
        // Science: Strong narrative identity constrains moral reasoning (values are stable);
        // weak/incoherent identity amplifies moral sensitivity (recalibration needed)
        if narrative_self_psi > 0.7 {
            // High self-coherence → stabilize moral score (dampen fluctuations)
            // Multiply moral learning signal toward 1.0 (neutral)
            self.fep.learning_signal *= 1.0 + (narrative_self_psi as f32 - 0.7) * 0.1;
        } else if narrative_self_psi > 0.0 && narrative_self_psi < 0.2 {
            // Low self-coherence → amplify moral concern sensitivity
            self.adaptive_behavior.attention_sensitivity *=
                1.0 + (0.2 - narrative_self_psi as f32) * 0.15;
        }

        module_timings.narrative_self = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PREDICTIVE PROCESSING: Hierarchical predictive coding + precision
        // Runs every cycle (lightweight: BinaryHV → prediction → free energy)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (predictive_free_energy, predictive_psi_modulation) = if let Some(ref mut mind) =
            self.consciousness_state.predictive_mind
        {
            if self.consciousness_state.affective_bridge.is_some() {
                mind.precision
                    .apply_affective_modulation(affective_arousal as f64, affective_valence as f64);
            }
            let state = mind.process(&ctx.hv16_cached);
            // Smooth predictive phi modulation with previous-cycle value.
            // Friston (2010): precision estimates should evolve gradually to
            // maintain hierarchical model stability.
            let prev_mod = self.carryover.consciousness.predictive_phi_modulation;
            let alpha = crate::cognitive_loop::thresholds::PREDICTIVE_PHI_MODULATION_SMOOTHING;
            let smoothed_mod = state.phi_modulation * (1.0 - alpha) + prev_mod * alpha;
            self.carryover.consciousness.predictive_phi_modulation = smoothed_mod;
            (state.free_energy, smoothed_mod)
        } else {
            (0.0, 1.0)
        };

        // FEEDBACK: Predictive phi modulation gates plasticity (Friston 2010)
        // Clamp and scale to avoid destabilizing the base learner in single-module ablations.
        let modulation = (predictive_psi_modulation - 1.0).clamp(-0.15, 0.15) as f32;
        let coherence_scale = 0.5 + 0.5 * ctx.coherence.clamp(0.0, 1.0);
        let delta = modulation * 0.10 * coherence_scale; // ±1.5% max, coherence-weighted
        self.carryover.learning.subsystem_lr_factor *= 1.0 + delta;
        module_timings.predictive_processing = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // HIERARCHICAL FREE ENERGY: Multi-level variational decomposition
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // Science: Friston (2008) — hierarchical predictive processing
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let hierarchical_total_free_energy =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut hfe) = self.consciousness_monitors.hierarchical_free_energy {
                    // FEEDBACK: Phi→precision coupling — higher integrated information
                    // sharpens lower-level precision (Feldman & Friston 2010, §7.4).
                    // This creates a causal mechanism: consciousness improves perceptual accuracy.
                    let psi_boost = (ctx.unified_psi * 0.5).clamp(0.0, 0.5);
                    let base_decay = hfe.config.precision_decay;
                    for level in &mut hfe.levels {
                        let base_precision = base_decay.powi(level.level as i32);
                        level.precision = base_precision * (1.0 + psi_boost);
                    }

                    // Build observation from compressed state (clamped to state_dim)
                    let obs: Vec<f64> = ctx
                        .compressed_state
                        .iter()
                        .take(hfe.config.state_dim)
                        .map(|&x| x as f64)
                        .collect();
                    hfe.update_beliefs(&obs);
                    hfe.total_free_energy()
                } else {
                    0.0
                }
            } else {
                0.0
            };

        // FEEDBACK: High hierarchical free energy suppresses exploration AND boosts learning
        // Science: Friston (2008) — poor model → focus on learning, not exploring
        if hierarchical_total_free_energy > 1.0 {
            let fe_factor = (1.0 / (1.0 + hierarchical_total_free_energy * 0.05)) as f32;
            self.curiosity_drive.boredom *= fe_factor; // suppress exploration urge (gentler)
                                                       // Boost LR proportional to free energy (poor model → learn harder)
                                                       // Capped at +10% to avoid overshooting in short ablation windows
            let hfe_lr_boost = (1.0 + (hierarchical_total_free_energy * 0.02).min(0.1)) as f32;
            self.scale_lr("hierarchical_free_energy", hfe_lr_boost);
        }

        module_timings.hierarchical_free_energy = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PREDICTIVE SELF: Evaluate action safety via self-state prediction
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let predictive_self_safety = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut pred_self) = self.self_model_tier.predictive_self {
                if let Some(ref narrative) = self.self_model_tier.narrative_self {
                    pred_self.observe(narrative);
                }
                pred_self.learn_from_outcome_raw(ctx.unified_psi, ctx.coherence as f64);
                pred_self.confidence() as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        // FEEDBACK: Low self-model confidence reduces learning rate (precision-weighting)
        // Science: Clark (2013) — low precision on self-model predictions should reduce LR
        if predictive_self_safety > 0.0 && predictive_self_safety < 0.4 {
            let safety_factor = 0.85 + predictive_self_safety * 0.375; // 0.85-1.0
            self.carryover.learning.subsystem_lr_factor *= safety_factor;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // ATTENTION SCHEMA: Track attention state and generate control signals
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let attention_schema_focus = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut schema) = self.self_model_tier.attention_schema {
                let salience = ctx.prediction_error.max(0.1);
                let update = schema.update(ctx.hv16_cached, salience);
                let gain = if update.control_signal > 0.3 {
                    ((update.control_signal - 0.3) * 0.6).min(0.3)
                } else if update.control_signal < 0.2 {
                    -0.1
                } else {
                    0.0
                };
                self.adaptive_behavior.attention_sensitivity *= 1.0 + gain;
                update.new_intensity
            } else {
                0.0
            }
        } else {
            0.0
        };

        module_timings.attention_schema = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Attention focus gates novelty-seeking vs focus-locking (Baars 1988 GWT)
        // Science: Low focus → attention is scattered, force novelty-seeking to re-engage;
        // high focus → deep attention, suppress context-switching to maintain flow
        if attention_schema_focus > 0.0 {
            if attention_schema_focus < 0.3 {
                // Attenuated 50%: ACh attention_factor already scales attention via the bath
                let novelty_push = (0.3 - attention_schema_focus) * 0.06;
                self.adjust_exploration("attention_deficit", novelty_push);
            } else if attention_schema_focus > 0.8 {
                let focus_lock = (attention_schema_focus - 0.8) * 0.15;
                self.adaptive_behavior.exploration_factor *= (1.0 - focus_lock).max(0.7);
            }
        }

        // FEEDBACK: Vigilance fatigue drives attention shift pressure (Mackworth 1948, AST-1)
        // Science: Sustained attention on a single target degrades performance after ~30 cycles.
        // The schema's fatigue_level() reflects accumulated focus duration. When fatigue is
        // high, increase exploration to encourage attention re-allocation.
        if let Some(ref schema) = self.self_model_tier.attention_schema {
            let fatigue = schema.fatigue_level();
            if fatigue > 0.5 {
                // Graduated exploration push: 0.0 at fatigue=0.5, up to 0.04 at fatigue=1.0
                let fatigue_push = (fatigue - 0.5) * 0.08;
                self.adjust_exploration("vigilance_fatigue", fatigue_push);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHI ATTENTION: Adaptive Φ-weighted attention routing
        // Observes current Phi and gates expensive actions by consciousness level.
        // Science: Dehaene (2014) — conscious access enables flexible routing
        // ═══════════════════════════════════════════════════════════════════════
        let (psi_attention_avg, phi_suppress) =
            if let Some(ref mut phi_attn) = self.ethics_values.phi_attention {
                phi_attn.observe(ctx.unified_psi as f32);
                let suppress = !phi_attn.allows_action(
                    crate::consciousness::phi_attention::ActionType::StateModifying,
                    ctx.unified_psi as f32,
                );
                (phi_attn.phi_average().unwrap_or(0.0), suppress)
            } else {
                (0.0, false)
            };
        if phi_suppress {
            // Attenuated: 5-HT confidence_delta implicitly reduces exploration
            self.scale_exploration("phi_gate_suppress", 0.85);
        }

        // Attention visualization: record snapshot for debugging/introspection
        if let Some(ref mut viz) = self.attention_visualizer {
            let snapshot = crate::visualization::AttentionSnapshot::new(
                vec![
                    "psi".into(),
                    "coherence".into(),
                    "body".into(),
                    "attention".into(),
                ],
                vec![
                    ctx.unified_psi,
                    ctx.coherence as f64,
                    body_psi_modulation,
                    psi_attention_avg as f64,
                ],
                vec![
                    ctx.unified_psi as f32,
                    ctx.coherence,
                    body_psi_modulation as f32,
                    psi_attention_avg,
                ],
                1.0,
            );
            viz.record(snapshot);
        }

        LateConsciousnessResult {
            prefrontal_veto,
            meta_cognitive_accuracy,
            meta_cognitive_depth,
            body_psi_modulation,
            body_valence,
            body_arousal,
            affective_valence,
            affective_arousal,
            narrative_self_psi,
            predictive_free_energy,
            predictive_psi_modulation,
            hierarchical_total_free_energy,
            predictive_self_safety,
            attention_schema_focus,
            attention_fatigue: self
                .self_model_tier
                .attention_schema
                .as_ref()
                .map(|s| s.fatigue_level())
                .unwrap_or(0.0),
            attention_prediction_accuracy: self
                .self_model_tier
                .attention_schema
                .as_ref()
                .map(|s| s.prediction_accuracy() as f32)
                .unwrap_or(0.0),
            psi_attention_avg,
        }
    }
}
