//! Late consciousness monitors and integration extracted from cycle.rs.
//!
//! Contains: prefrontal cortex, meta-cognition, virtual body, affective bridge,
//! user state inference, narrative self, predictive processing, hierarchical free
//! energy, predictive self, attention schema, phi attention, attention visualization,
//! GWT integration, cross-modal binding, consciousness monitors (resonance, quantum
//! coherence), phenomenal binding, temporal consciousness, consciousness thermodynamics,
//! embodied cognition, narrative-GWT integration, unified living mind, and master
//! consciousness equation.

use std::time::Instant;

use super::CognitiveLoopService;

/// Input context for the late consciousness monitors phase.
/// Contains read-only values computed by earlier cycle phases.
pub(super) struct LateConsciousnessContext<'a> {
    pub prediction_error: f32,
    pub coherence: f32,
    pub unified_psi: f64,
    pub hv16_cached: symthaea_core::hdc::BinaryHV,
    pub compressed_state: &'a [f32],
    pub input: &'a str,
    pub urgency: super::CycleUrgency,
    pub moral_concern_detected: bool,
    pub surprise_triggered: bool,
    pub reasoning_gate_blocked: bool,
    pub pp_phi: f32,
    pub peak_attention: f32,
}

/// Results from the late consciousness monitors phase (prefrontal through attention schema).
pub(super) struct LateConsciousnessResult {
    pub prefrontal_veto: bool,
    pub meta_cognitive_accuracy: f32,
    pub meta_cognitive_depth: u8,
    pub body_psi_modulation: f64,
    pub body_valence: f32,
    pub body_arousal: f32,
    pub affective_valence: f32,
    pub affective_arousal: f32,
    pub narrative_self_psi: f64,
    pub predictive_free_energy: f64,
    pub predictive_psi_modulation: f64,
    pub hierarchical_total_free_energy: f64,
    pub predictive_self_safety: f32,
    pub attention_schema_focus: f32,
    pub psi_attention_avg: f32,
}

/// Results from the consciousness integration phase (GWT through master consciousness equation).
pub(super) struct ConsciousnessIntegrationResult {
    pub gwt_broadcast: bool,
    pub cross_modal_binding_strength: f32,
    pub cross_modal_psi: f64,
    pub resonance_frequency: f64,
    pub quantum_coherence_level: f64,
    pub phenomenal_binding_strength: f64,
    pub phenomenal_fragmented: bool,
    pub temporal_coherence_score: f64,
    pub temporal_discontinuity: bool,
    pub thermodynamic_entropy: f64,
    pub thermodynamic_free_energy: f64,
    pub embodied_psi_modulation: f64,
    pub embodied_agency: f64,
    pub narrative_gwt_veto: bool,
    pub narrative_gwt_self_psi: f64,
    pub living_mind_vitality: f64,
    pub living_mind_coherence: f64,
    pub consciousness_level: f64,
}

impl CognitiveLoopService {
    /// Run late consciousness monitors: prefrontal cortex, meta-cognition, virtual body,
    /// affective bridge, user state, narrative self, predictive processing, hierarchical
    /// free energy, predictive self, attention schema, and phi attention.
    ///
    /// Extracted from cycle.rs — all logic and behavior preserved exactly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_late_consciousness_monitors(
        &mut self,
        ctx: &LateConsciousnessContext,
        module_timings: &mut super::ModuleTimings,
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
                        let grad_importance = if let Some(ref mut res_mem) = self.resonator_memory {
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
                                        ep.hv
                                            .iter()
                                            .zip(projected.iter())
                                            .map(|(a, b)| a * b)
                                            .sum::<f32>()
                                            / (ep.hv.iter().map(|x| x * x).sum::<f32>().sqrt()
                                                * projected
                                                    .iter()
                                                    .map(|x| x * x)
                                                    .sum::<f32>()
                                                    .sqrt())
                                            .max(1e-8)
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
                        self.episodic_memory.encode(
                            &grad.id,
                            grad.embedding
                                .values
                                .iter()
                                .take(64)
                                .copied()
                                .collect::<Vec<_>>(),
                            0.0,
                            grad_importance,
                            self.stats.total_cycles,
                        );
                    }
                    tracing::debug!(
                        count = graduates.len(),
                        "Prefrontal graduated items to episodic memory (resonator-routed)"
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
            self.curiosity_drive.exploration_urge = 0.0;

            // FEEDBACK: WM overload triggers emergency consolidation (Baddeley 2000)
            // Science: Working memory overflow should push items to long-term storage,
            // not just block exploration. Force episodic consolidation to free WM slots.
            self.episodic_memory.consolidate_recent();
        }

        // FEEDBACK: Dual-veto freeze detection and recovery (Fuchs 2008 multistability)
        // Science: When reasoning gate AND prefrontal veto both fire, system is paralyzed:
        // exploration=0, learning=0. Soften both to allow partial recovery.
        if ctx.reasoning_gate_blocked && prefrontal_veto {
            self.curiosity_drive.exploration_urge = 0.3;
            self.fep_lr_boost = self.fep_lr_boost.max(1.0); // enforce fep_lr_boost >= 1.0 invariant
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
                if let Some(ref mut meta) = self.meta_cognition {
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
                self.meta_cognition
                    .as_ref()
                    .map(|m| (m.accuracy(), m.depth()))
                    .unwrap_or((0.0, 0))
            };

        module_timings.meta_cognition = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // VIRTUAL BODY: Map cognitive signals to interoceptive states
        // Urgency-gated: Critical=always, Normal=always, Cruise=every 2nd
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (body_psi_modulation, body_valence, body_arousal) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 1, 2) {
                if let Some(ref mut body) = self.virtual_body {
                    let signals = super::virtual_body::CognitiveSignals {
                        prediction_error: ctx.prediction_error,
                        coherence: ctx.coherence,
                        prediction_confidence: self.prediction_confidence,
                        unified_psi: ctx.unified_psi,
                        flow_intensity: self.flow_state.intensity,
                        in_flow: self.flow_state.in_flow,
                        curiosity_boredom: self.curiosity_drive.boredom,
                        fep_learning_signal: self.fep_learning_signal,
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
            self.prediction_confidence =
                (self.prediction_confidence + body_valence * 0.02).clamp(0.0, 1.0);
        } else if body_valence < -0.3 {
            self.prediction_confidence =
                (self.prediction_confidence + body_valence * 0.03).clamp(0.0, 1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // AFFECTIVE BRIDGE: Evaluate somatic markers from cognitive signals
        // Runs every cycle (lightweight: ~5 arithmetic ops + blend)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (affective_valence, affective_arousal) =
            if let Some(ref mut bridge) = self.affective_bridge {
                let moral_score = self
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
                    self.social_trust,
                    self.social_cooperation_rate,
                    0.0, // peer_valence: future — aggregate from social inbox
                );
                (affect.valence, affect.arousal)
            } else {
                (0.0, 0.5)
            };

        // FEEDBACK: Positive affect broadens exploration (Fredrickson 2001 broaden-and-build)
        if affective_valence > 0.2 && self.affective_bridge.is_some() {
            self.curiosity_drive.boredom *= 1.05;
        }
        // FEEDBACK: Arousal gates learning consolidation (Russell 1980 VAD model)
        // Science: Steriade (1996) — high arousal (fight-or-flight) suppresses consolidation;
        // low arousal (rest) enhances memory consolidation (REM/slow-wave effect)
        if affective_arousal > 0.7 {
            let arousal_suppress = ((affective_arousal - 0.7) * 0.5).min(0.15);
            self.fep_lr_boost = (self.fep_lr_boost * (1.0 - arousal_suppress)).max(1.0);

            // Arousal trap detection (Yerkes-Dodson 1908 — inverted-U performance curve)
            // Science: Prolonged high arousal suppresses LR → error stays high → arousal stays
            // high → positive feedback trap. After 10 stuck cycles, force exploration escape.
            if affective_arousal > 0.8 {
                self.carryover.urgency.arousal_trap_counter = self
                    .carryover
                    .urgency
                    .arousal_trap_counter
                    .saturating_add(1);
            }
            // ── Phase 15: Active arousal recovery mode ────────────────────
            // Science: Porges (2011) — polyvagal theory: recovery from high arousal
            // requires active parasympathetic engagement, not just waiting. After 5+
            // consecutive high-arousal cycles, slow CfC processing (increase tau) to
            // give the system time to stabilize before the hard reset at counter > 10.
            if self.carryover.urgency.arousal_trap_counter > 5
                && self.carryover.urgency.arousal_trap_counter <= 10
            {
                let recovery_intensity =
                    (self.carryover.urgency.arousal_trap_counter - 5) as f32 / 5.0;
                // Gradual LR dampening: up to 20% reduced learning during recovery
                self.fep_lr_boost = (self.fep_lr_boost * (1.0 - recovery_intensity * 0.2)).max(1.0);
                // Slight exploration boost to help escape
                self.curiosity_drive.exploration_urge = (self.curiosity_drive.exploration_urge
                    + recovery_intensity * 0.05)
                    .clamp(0.0, 1.0);
                self.stats.arousal_recovery_cycles += 1;
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    counter = self.carryover.urgency.arousal_trap_counter,
                    recovery_intensity,
                    "Arousal recovery mode: dampening LR, boosting exploration"
                );
            }
            if self.carryover.urgency.arousal_trap_counter > 10 {
                self.curiosity_drive.exploration_urge = 1.0; // forced escape attempt
                self.prediction_confidence *= 0.9; // reset confidence to allow re-learning
                self.carryover.urgency.arousal_trap_counter = 0;
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Arousal trap escape: forced exploration after 10 high-arousal cycles"
                );
            }
        } else {
            // Reset trap counter when arousal drops below threshold
            self.carryover.urgency.arousal_trap_counter = 0;

            if affective_arousal < 0.3 {
                let consolidation_boost = ((0.3 - affective_arousal) * 0.3).min(0.1);
                self.fep_lr_boost =
                    (self.fep_lr_boost * (1.0 + consolidation_boost)).clamp(1.0, 2.0);
            }
        }
        module_timings.affective_bridge = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // USER STATE INFERENCE: Infer cognitive load, frustration, engagement
        // Runs every cycle (lightweight: keyword detection + rolling averages)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut usi) = self.user_state {
            let had_error = ctx.prediction_error > 0.8;
            usi.process(ctx.input, had_error);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // NARRATIVE SELF: Process experience and track self-Φ
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let narrative_self_psi = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut narrative) = self.narrative_self {
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
            self.narrative_self
                .as_ref()
                .map(|n| n.self_phi())
                .unwrap_or(0.0)
        };

        // FEEDBACK: Narrative self-Phi modulates prediction confidence (identity coherence)
        // Science: Gallagher (2000) — strong narrative identity stabilizes learning
        if narrative_self_psi > 0.5 {
            self.prediction_confidence = (self.prediction_confidence * 1.02).clamp(0.0, 1.0);
        } else if narrative_self_psi > 0.0 && narrative_self_psi < 0.2 {
            self.prediction_confidence = (self.prediction_confidence * 0.95).clamp(0.0, 1.0);
        }

        // FEEDBACK: Narrative self-Phi modulates moral sensitivity (Gallagher & Hutto 2007)
        // Science: Strong narrative identity constrains moral reasoning (values are stable);
        // weak/incoherent identity amplifies moral sensitivity (recalibration needed)
        if narrative_self_psi > 0.7 {
            // High self-coherence → stabilize moral score (dampen fluctuations)
            // Multiply moral learning signal toward 1.0 (neutral)
            self.fep_learning_signal *= 1.0 + (narrative_self_psi as f32 - 0.7) * 0.1;
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
            self.predictive_mind
        {
            if self.affective_bridge.is_some() {
                mind.precision
                    .apply_affective_modulation(affective_arousal as f64, affective_valence as f64);
            }
            let state = mind.process(&ctx.hv16_cached);
            self.carryover.consciousness.predictive_phi_modulation = state.phi_modulation;
            (state.free_energy, state.phi_modulation)
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
                if let Some(ref mut hfe) = self.hierarchical_free_energy {
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
            self.fep_lr_boost = (self.fep_lr_boost * hfe_lr_boost).clamp(1.0, 1.3);
        }

        module_timings.hierarchical_free_energy = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PREDICTIVE SELF: Evaluate action safety via self-state prediction
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let predictive_self_safety = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut pred_self) = self.predictive_self {
                if let Some(ref narrative) = self.narrative_self {
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
            if let Some(ref mut schema) = self.attention_schema {
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
                let novelty_push = (0.3 - attention_schema_focus) * 0.12;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + novelty_push).clamp(0.0, 1.0);
            } else if attention_schema_focus > 0.8 {
                let focus_lock = (attention_schema_focus - 0.8) * 0.15;
                self.adaptive_behavior.exploration_factor *= (1.0 - focus_lock).max(0.7);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHI ATTENTION: Adaptive Φ-weighted attention routing
        // Observes current Phi and gates expensive actions by consciousness level.
        // Science: Dehaene (2014) — conscious access enables flexible routing
        // ═══════════════════════════════════════════════════════════════════════
        let psi_attention_avg = if let Some(ref mut phi_attn) = self.phi_attention {
            phi_attn.observe(ctx.unified_psi as f32);
            // Gate: only allow state-modifying actions when Phi is sufficient
            if !phi_attn.allows_action(
                crate::consciousness::phi_attention::ActionType::StateModifying,
                ctx.unified_psi as f32,
            ) {
                // Low consciousness → reduce exploration (don't take risky actions unconsciously)
                self.curiosity_drive.exploration_urge *= 0.7;
            }
            phi_attn.phi_average().unwrap_or(0.0)
        } else {
            0.0
        };

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
            psi_attention_avg,
        }
    }

    /// Run consciousness integration: GWT, cross-modal binding, consciousness monitors
    /// (resonance, quantum coherence), phenomenal binding, temporal consciousness,
    /// consciousness thermodynamics, embodied cognition, narrative-GWT integration,
    /// unified living mind, and master consciousness equation.
    ///
    /// Extracted from cycle.rs — all logic and behavior preserved exactly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_consciousness_integration(
        &mut self,
        ctx: &LateConsciousnessContext,
        late: &LateConsciousnessResult,
        module_timings: &mut super::ModuleTimings,
    ) -> ConsciousnessIntegrationResult {
        // ═══════════════════════════════════════════════════════════════════════
        // GWT INTEGRATION: Submit encoding to global workspace for broadcast
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let gwt_broadcast = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut gwt) = self.gwt {
                let activation = (1.0 - ctx.prediction_error as f64).clamp(0.0, 1.0);
                gwt.submit_strategy(
                    "cognitive_loop",
                    activation,
                    vec![ctx.hv16_cached],
                    vec!["encoder".to_string()],
                );
                let result = gwt.process();
                result.broadcast_occurred
            } else {
                false
            }
        } else {
            false
        };

        // FEEDBACK: GWT broadcast boosts confidence (conscious access moment)
        // Science: Baars (1988) — broadcast = conscious access, should amplify integration
        if gwt_broadcast {
            self.prediction_confidence = (self.prediction_confidence
                + super::cycle::GWT_BROADCAST_CONFIDENCE_BOOST)
                .clamp(0.0, 1.0);
        }

        module_timings.gwt = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CROSS-MODAL BINDING: Bind HDC encodings across modalities
        // Runs every cycle (lightweight: 2 HV ops + similarity)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (cross_modal_binding_strength, cross_modal_psi) = self.update_cross_modal_binding(
            &ctx.hv16_cached,
            late.affective_valence,
            late.predictive_free_energy,
        );

        module_timings.cross_modal_binding = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS MONITORS: Resonance + Quantum coherence
        // Urgency-gated: skip in Cruise mode
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Pre-compute to avoid borrow conflict with mutable subsystem references below
        let wm_utilization = self.prefrontal_utilization();
        let resonance_frequency = if ctx.urgency.run_consciousness_monitors() {
            if let Some(ref mut resonance) = self.consciousness_resonance {
                let dims = [
                    ctx.unified_psi,
                    ctx.coherence as f64,
                    wm_utilization,
                    self.adaptive_behavior.attention_sensitivity as f64,
                    (self.stats.total_cycles.min(100) as f64 / 100.0),
                    late.body_psi_modulation,
                    self.prediction_confidence as f64,
                ];
                let state = resonance.analyze(dims);
                state.dominant_frequency
            } else {
                0.0
            }
        } else {
            self.carryover.history.resonance_frequency // use previous cycle's value instead of 0.0
        };

        // FEEDBACK: Resonance frequency modulates attention sensitivity (Engel 2001)
        // Stable resonance near 0.5 → sharp attention; deviant frequency → diffuse
        if resonance_frequency > 0.0 {
            let resonance_quality = 1.0 - (resonance_frequency - 0.5).abs() * 2.0; // peaks at 0.5
            let attention_mod = 1.0 + (resonance_quality as f32 - 0.5) * 0.1; // ±5%
            self.adaptive_behavior.attention_sensitivity *= attention_mod;
        }

        let quantum_coherence_level = if ctx.urgency.run_consciousness_monitors() {
            if let Some(ref mut qc) = self.quantum_coherence {
                qc.observe(&ctx.hv16_cached, ctx.unified_psi);
                qc.coherence()
            } else {
                0.0
            }
        } else {
            self.carryover.consciousness.quantum_coherence // use previous cycle's value instead of 0.0
        };

        // FEEDBACK: Quantum coherence modulates prediction confidence (Penrose & Hameroff 2014)
        // High coherence → quantum-enhanced processing → slightly boost confidence
        // Decoherence → noisy processing → reduce confidence
        if quantum_coherence_level > 0.6 {
            let qc_boost = (quantum_coherence_level - 0.6) as f32 * 0.05; // up to +2%
            self.prediction_confidence = (self.prediction_confidence + qc_boost).clamp(0.0, 1.0);
        } else if quantum_coherence_level > 0.0 && quantum_coherence_level < 0.2 {
            self.prediction_confidence *= 0.98; // slight reduction during decoherence
        }

        module_timings.consciousness_resonance = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHENOMENAL BINDING: Temporal synchronization across consciousness dims
        // Urgency-gated: same as consciousness monitors (skip in Cruise)
        // Science: Singer & Gray (1989) — temporal binding hypothesis
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (phenomenal_binding_strength, phenomenal_fragmented) =
            if ctx.urgency.run_consciousness_monitors() {
                if let Some(ref mut binding) = self.phenomenal_binding {
                    let dims = [
                        ctx.unified_psi,
                        ctx.coherence as f64,
                        wm_utilization,
                        self.adaptive_behavior.attention_sensitivity as f64,
                        (self.stats.total_cycles.min(100) as f64 / 100.0),
                        late.body_psi_modulation,
                        self.prediction_confidence as f64,
                    ];
                    binding.observe_all(&dims);
                    let strength = binding.phenomenal_binding_strength();
                    let fragmented = binding.detect_fragmentation().is_some();
                    (strength, fragmented)
                } else {
                    (0.0, false)
                }
            } else {
                (0.0, false)
            };

        // FEEDBACK: Fragmentation suppresses exploration (Singer 1989)
        // When consciousness is fragmented, focus on integration not exploration
        if phenomenal_fragmented {
            self.curiosity_drive.boredom *= 0.8;
            self.adaptive_behavior.exploration_factor *= 0.7;
        }

        // FEEDBACK: High binding strength (flow) boosts learning rate (Csikszentmihalyi 1990)
        if phenomenal_binding_strength > 0.8 {
            let binding_boost = ((phenomenal_binding_strength - 0.8) * 0.2) as f32; // up to +4%
            self.carryover.learning.subsystem_lr_factor *= 1.0 + binding_boost;
        }

        // FEEDBACK: Binding strength gates WM access via attention sensitivity (Tononi 2015 IIT)
        // Science: High integrated information → more can be held in working memory;
        // low binding → restrict input (WM fragmented, accept less)
        if phenomenal_binding_strength > 0.7 {
            let wm_boost = ((phenomenal_binding_strength - 0.7) * 0.1) as f32; // up to +3%
            self.adaptive_behavior.attention_sensitivity *= 1.0 + wm_boost;
        } else if phenomenal_binding_strength > 0.0 && phenomenal_binding_strength < 0.4 {
            let wm_restrict = ((0.4 - phenomenal_binding_strength) * 0.08) as f32;
            self.adaptive_behavior.attention_sensitivity *= (1.0 - wm_restrict).max(0.8);
        }

        module_timings.phenomenal_binding = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // TEMPORAL CONSCIOUSNESS: Track Phi trajectory, continuity, identity
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (temporal_coherence_score, temporal_discontinuity) =
            if ctx.urgency.run_consciousness_monitors() {
                if let Some(ref mut temporal) = self.temporal_consciousness {
                    temporal.observe(
                        &ctx.hv16_cached,
                        ctx.unified_psi,
                        self.narrative_self.as_ref(),
                        self.predictive_self.as_ref(),
                    );
                    let coherence = temporal.overall_temporal_coherence();
                    let healthy = temporal.is_temporally_healthy();
                    (coherence, !healthy)
                } else {
                    (0.0, false)
                }
            } else {
                (0.0, false)
            };

        // FEEDBACK: Temporal discontinuity resets adaptation (context shift re-calibration)
        // Science: Varela (1999) — temporal discontinuities require re-orientation
        if temporal_discontinuity {
            self.fep_lr_boost = 1.0;
            self.prediction_confidence *= 0.8;
            // Lower learning threshold to learn more aggressively after discontinuity
            self.carryover.learning.adaptive_threshold_scale =
                (self.carryover.learning.adaptive_threshold_scale * 0.8).clamp(0.6, 1.5);
        } else if temporal_coherence_score > 0.8 {
            // High temporal coherence → model is reliable, raise threshold (learn less often)
            self.carryover.learning.adaptive_threshold_scale =
                (self.carryover.learning.adaptive_threshold_scale * 1.01).clamp(0.6, 1.5);
        } else {
            // Slowly return toward baseline
            self.carryover.learning.adaptive_threshold_scale +=
                (1.0 - self.carryover.learning.adaptive_threshold_scale) * 0.02;
        }

        // ── Phase 21: Temporal discontinuity recovery cascade ────────────
        // Science: Context shift detection requires graduated recovery, not just instant reset
        if temporal_discontinuity {
            self.carryover.urgency.discontinuity_streak += 1;
        } else {
            self.carryover.urgency.discontinuity_streak =
                self.carryover.urgency.discontinuity_streak.saturating_sub(1);
        }
        let streak = self.carryover.urgency.discontinuity_streak;
        if streak >= 3 {
            // Persistent discontinuity: aggressive recovery
            self.last_prediction = None; // invalidate stale predictions
            self.fep_lr_boost = (self.fep_lr_boost * 1.5).min(3.0);
            self.curiosity_drive.exploration_urge *= 0.7;
            self.stats.discontinuity_cascade_count += 1;
        }

        // FEEDBACK: High temporal coherence strengthens narrative self engagement
        // Science: Damasio (2010) — temporal continuity is the substrate of selfhood
        if temporal_coherence_score > 0.6 {
            if let Some(ref mut narrative) = self.narrative_self {
                let continuity_boost = (temporal_coherence_score - 0.6) * 0.1; // up to +4%
                narrative.boost_coherence(continuity_boost);
            }
        }

        // FEEDBACK: Temporal coherence ↔ attention mutual coupling (Engel et al. 2001)
        // Science: Temporal binding via phase synchrony — attention must gate synchronization.
        // Low temporal coherence → attention is fragmenting the time-axis → penalize sensitivity
        // to prevent amplification of incoherent states. High coherence → attention is stable.
        if temporal_coherence_score > 0.0 && temporal_coherence_score < 0.4 {
            let coherence_penalty = ((0.4 - temporal_coherence_score) * 0.1) as f32;
            self.adaptive_behavior.attention_sensitivity *= (1.0 - coherence_penalty).max(0.85);
        }

        module_timings.temporal_consciousness = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS THERMODYNAMICS: Phase transitions & free energy
        // Urgency-gated: same as consciousness monitors (skip in Cruise)
        // Science: Friston (2010) — free energy, Kelso — critical fluctuations
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (thermodynamic_entropy, thermodynamic_free_energy) =
            if ctx.urgency.run_consciousness_monitors() {
                if let Some(ref mut thermo) = self.consciousness_thermodynamics {
                    let dims = [
                        ctx.unified_psi,
                        ctx.coherence as f64,
                        wm_utilization,
                        self.adaptive_behavior.attention_sensitivity as f64,
                        (self.stats.total_cycles.min(100) as f64 / 100.0),
                        late.body_psi_modulation,
                        self.prediction_confidence as f64,
                    ];
                    let state = thermo.analyze(dims);
                    // FEEDBACK: Phase-dependent exploration modulation (Kelso 1995)
                    use crate::consciousness::consciousness_thermodynamics::ConsciousnessPhase;
                    match state.phase {
                        ConsciousnessPhase::Critical => {
                            // Edge of chaos — maximum creativity, boost exploration
                            self.curiosity_drive.boredom *= 1.1;
                            self.adaptive_behavior.exploration_factor *= 1.15;
                        }
                        ConsciousnessPhase::Flow => {
                            // Superfluid state — boost learning rate
                            self.carryover.learning.subsystem_lr_factor *= 1.05;
                        }
                        ConsciousnessPhase::Chaotic => {
                            // Fragmented — suppress exploration, seek stability
                            self.curiosity_drive.boredom *= 0.7;
                            self.adaptive_behavior.exploration_factor *= 0.5;
                        }
                        ConsciousnessPhase::Frozen => {
                            // Rigid — nudge toward exploration to unfreeze
                            self.curiosity_drive.boredom *= 1.05;
                        }
                        _ => {} // Normal, Unified — no modulation needed
                    }
                    (state.entropy, state.free_energy)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

        module_timings.consciousness_thermodynamics = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Thermodynamic entropy magnitude modulates exploration intensity
        // Science: Ulanowicz (2009) — entropy quantifies degrees of freedom in the system.
        // High entropy → system has many accessible states, exploration is cheap → boost;
        // Low entropy → system is ordered, consolidation is productive → dampen exploration.
        // This complements the phase-based modulation above with continuous magnitude scaling.
        if thermodynamic_entropy > 0.0 {
            if thermodynamic_entropy > 0.7 {
                let entropy_boost = ((thermodynamic_entropy - 0.7) * 0.1).min(0.1) as f32;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + entropy_boost).clamp(0.0, 1.0);
            } else if thermodynamic_entropy < 0.3 {
                let consolidation_bias = ((0.3 - thermodynamic_entropy) * 0.08).min(0.08) as f32;
                self.fep_lr_boost =
                    (self.fep_lr_boost * (1.0 + consolidation_bias)).clamp(1.0, 2.0);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // EMBODIED COGNITION: Bridge virtual body to full body schema
        // Urgency-gated: Critical=always, Normal=always, Cruise=every 2nd
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (embodied_psi_modulation, embodied_agency) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 1, 2) {
                if let Some(ref mut embodied) = self.embodied_cognition {
                    if let Some(ref body) = self.virtual_body {
                        embodied.update_interoception(body.interoceptive_state().clone());
                    }
                    let response = embodied.process();
                    self.carryover.consciousness.embodied_phi_modulation = response.phi_modulation;

                    // Wire embodied signals into cognitive loop:
                    // 1. Homeostatic deviation increases urgency (survival takes priority)
                    // Science: Damasio (1999) — somatic markers guide decision-making
                    if response.homeostatic_deviation > 0.5 {
                        self.carryover.urgency.consecutive_low_error = 0; // prevent Cruise when body is stressed
                    }
                    // 2. Sensorimotor surprise blends into exploration urge
                    // Science: Friston (2010) — interoceptive surprise drives active inference
                    if response.sensorimotor_surprise > 0.3 {
                        let body_nudge = (response.sensorimotor_surprise * 0.1).min(0.15) as f32;
                        self.curiosity_drive.exploration_urge =
                            (self.curiosity_drive.exploration_urge + body_nudge).clamp(0.0, 1.0);
                    }
                    // 3. High allostatic load suppresses learning (conserve resources)
                    // Science: McEwen (2004) — allostatic overload impairs plasticity
                    if response.allostatic_load > 0.7 {
                        self.fep_lr_boost = (self.fep_lr_boost
                            * (1.0 - (response.allostatic_load - 0.7) as f32 * 0.5))
                            .max(1.0);
                    }

                    (response.phi_modulation, response.sense_of_agency)
                } else {
                    (1.0, 0.0)
                }
            } else {
                // Urgency-skipped: use carryover for phi_modulation; agency has no carryover.
                (self.carryover.consciousness.embodied_phi_modulation, 0.0)
            };

        module_timings.embodied_cognition = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Embodied agency modulates exploration risk tolerance
        // Science: Friston, Stephan et al. (2015) — sense of agency enables bold action
        // High agency → allow riskier exploration; low agency → cautious retreat
        if embodied_agency > 0.7 {
            let agency_boost = ((embodied_agency - 0.7) * 0.15) as f32; // up to +4.5%
            self.adaptive_behavior.exploration_factor *= 1.0 + agency_boost;
        } else if embodied_agency > 0.0 && embodied_agency < 0.3 {
            let caution = ((0.3 - embodied_agency) * 0.1) as f32; // up to -3%
            self.curiosity_drive.exploration_urge *= (1.0 - caution).max(0.7);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // NARRATIVE-GWT INTEGRATION: Consciousness governance capstone
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (narrative_gwt_veto, narrative_gwt_self_psi) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut ngwt) = self.narrative_gwt {
                    let activation = (1.0 - ctx.prediction_error as f64).clamp(0.0, 1.0);
                    let veto = ngwt.submit_content(
                        "cognitive_loop",
                        vec![ctx.hv16_cached],
                        ctx.input,
                        vec!["encoder".to_string(), "temporal".to_string()],
                        activation,
                    );
                    let vetoed = veto.map(|v| v.vetoed).unwrap_or(false);
                    let _result = ngwt.process();
                    (vetoed, ngwt.self_phi())
                } else {
                    (false, 0.0)
                }
            } else {
                (false, 0.0)
            };

        // Store narrative-GWT veto for next cycle's learning gate
        self.carryover.quality.narrative_veto_active = narrative_gwt_veto;
        module_timings.narrative_gwt = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED LIVING MIND: life-mind continuity (full_consciousness only)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Integrates autopoietic self-maintenance, enactive sense-making, and
        // predictive processing into a unified vitality/coherence measure.
        #[cfg(feature = "full_consciousness")]
        let (living_mind_vitality, living_mind_coherence) = {
            // Update autopoietic self-maintenance with current consciousness signals
            self.autopoietic.update(
                ctx.unified_psi,
                ctx.coherence as f64,
                ctx.prediction_error as f64,
            );

            // Map cognitive loop action to enactive ActionType based on adaptive behavior
            let enactive_action = match self.adaptive_behavior.action_hint {
                super::ActionHint::Explore => {
                    crate::consciousness::enactive_cognition::ActionType::Explore
                }
                super::ActionHint::SeekInput => {
                    crate::consciousness::enactive_cognition::ActionType::Observe
                }
                super::ActionHint::SlowDown => {
                    crate::consciousness::enactive_cognition::ActionType::Reflect
                }
                super::ActionHint::SpeedUp => {
                    crate::consciousness::enactive_cognition::ActionType::Execute
                }
                _ => crate::consciousness::enactive_cognition::ActionType::Observe,
            };

            // Build perception summary from current cycle signals
            let perception = crate::consciousness::enactive_cognition::PerceptionSummary {
                features: {
                    let mut f = std::collections::HashMap::new();
                    f.insert("prediction_error".into(), ctx.prediction_error as f64);
                    f.insert("coherence".into(), ctx.coherence as f64);
                    f.insert("phi".into(), ctx.unified_psi);
                    f
                },
                surprise: ctx.prediction_error as f64,
                affordances: Vec::new(), // detected_primitives not available in this context
            };

            // Run enactive sense-making cycle
            let enacted_meaning = self.enactive.cycle(enactive_action, perception, ctx.input);

            // Wire enactive meaning into cognitive loop:
            // 1. High relevance boosts attention sensitivity (salient = attend more)
            // Science: Thompson (2007) — enacted meaning modulates attention
            if enacted_meaning.meaning.relevance > 0.6 {
                let relevance_gain = (enacted_meaning.meaning.relevance * 0.1).min(0.15) as f32;
                self.adaptive_behavior.attention_sensitivity *= 1.0 + relevance_gain;
            }
            // 2. Negative valence strengthens narrative veto tendency (caution)
            // Science: Colombetti (2014) — affect and enaction are inseparable
            if enacted_meaning.meaning.valence < -0.5 {
                self.prediction_confidence *= (1.0 + enacted_meaning.meaning.valence * 0.1) as f32;
                self.prediction_confidence = self.prediction_confidence.clamp(0.0, 1.0);
            }

            // Integrate all subsystems into unified living state
            let free_energy = self.fep_agent.current_free_energy();
            let unified_state = self.unified_living_mind.integrate(
                &self.autopoietic,
                &self.enactive,
                ctx.unified_psi,
                free_energy,
            );

            (unified_state.vitality, unified_state.coherence)
        };

        #[cfg(not(feature = "full_consciousness"))]
        let (living_mind_vitality, living_mind_coherence) = (0.0, 0.0);
        module_timings.living_mind = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // MASTER CONSCIOUSNESS EQUATION: comprehensive consciousness metric
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Run every 10th cycle to amortize cost. Maps cognitive loop signals to
        // the 8-factor ConsciousnessInputs: Phi, Broadcast, WorkingMemory,
        // Attention, Recurrence, Embodiment, Knowledge, Synchrony.
        // Urgency-adaptive: Critical=every 5th, Normal=every 10th, Cruise=every 20th
        let consciousness_level = if ctx.urgency.should_run(self.stats.total_cycles, 5, 10, 20) {
            let inputs = crate::consciousness::master_consciousness_equation::ConsciousnessInputs {
                phi: ctx.unified_psi,
                broadcast: ctx.coherence as f64, // coherence ~ global workspace broadcast
                working_memory: self.prefrontal_utilization(),
                attention: ctx.peak_attention as f64,
                recurrence: (self.stats.total_cycles.min(100) as f64 / 100.0), // ramp up over 100 cycles
                embodiment: late.body_psi_modulation, // virtual body provides embodiment
                knowledge: self.prediction_confidence as f64,
                synchrony: (0.3 + self.flow_state.intensity as f64 * 0.7).clamp(0.1, 1.0),
            };
            let level = self.master_equation.compute(&inputs).consciousness_level;

            // Track consciousness level for learning gating (Task C)
            self.carryover.history.consciousness_level = level;

            // FEEDBACK: MCE consciousness level boosts learning rate (decaying)
            // Science: Dehaene (2014) — conscious access improves encoding
            if level > 0.0 {
                self.carryover.learning.mce_lr_boost =
                    (level * super::cycle::MCE_LR_BOOST_SCALE as f64) as f32;
            } else {
                self.carryover.learning.mce_lr_boost *= super::cycle::MCE_BOOST_DECAY;
            }

            // FEEDBACK: Consciousness gates consolidation intensity (Dehaene 2014)
            // Science: Only conscious moments produce durable memories. Scale episodic
            // consolidation by consciousness level — low consciousness → skip storage,
            // high consciousness → prioritize memory encoding.
            if level > 0.5 {
                // Trigger demand-driven consolidation at high consciousness
                self.episodic_memory.consolidate_recent();
            }
            // Scale learning signal by consciousness quality (gradual, not on/off)
            // This complements the binary consciousness_awake gate with continuous modulation
            self.fep_learning_signal *= (0.5 + level as f32 * 0.5).clamp(0.5, 1.0);

            level
        } else {
            // Decay MCE LR boost between MCE firings
            self.carryover.learning.mce_lr_boost *= super::cycle::MCE_BOOST_DECAY;
            0.0
        };

        // Store resonance frequency and quantum coherence for next cycle's feedback
        self.carryover.history.resonance_frequency = resonance_frequency;
        self.carryover.consciousness.quantum_coherence = quantum_coherence_level;
        module_timings.master_consciousness_equation = _t.elapsed().as_micros() as u64;

        ConsciousnessIntegrationResult {
            gwt_broadcast,
            cross_modal_binding_strength,
            cross_modal_psi,
            resonance_frequency,
            quantum_coherence_level,
            phenomenal_binding_strength,
            phenomenal_fragmented,
            temporal_coherence_score,
            temporal_discontinuity,
            thermodynamic_entropy,
            thermodynamic_free_energy,
            embodied_psi_modulation,
            embodied_agency,
            narrative_gwt_veto,
            narrative_gwt_self_psi,
            living_mind_vitality,
            living_mind_coherence,
            consciousness_level,
        }
    }
}
