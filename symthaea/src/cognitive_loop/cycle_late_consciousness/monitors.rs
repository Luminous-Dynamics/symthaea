// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
                let utilization = pfc.memory_contents().len() as f32
                    / super::super::thresholds::PFC_WORKING_MEMORY_CAPACITY; // default capacity
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
                        let grad_importance = if let Some(ref mut res_mem) =
                            self.memory.memory_consol.resonator_memory
                        {
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
                                        let denom = (ep
                                            .hv
                                            .iter()
                                            .map(|x| x * x)
                                            .sum::<f32>()
                                            .sqrt()
                                            * projected.iter().map(|x| x * x).sum::<f32>().sqrt())
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
                                ctx.pp_phi
                                    + best_sim * super::super::thresholds::RESONATOR_MATCH_BOOST
                            } else {
                                ctx.pp_phi
                            }
                        } else {
                            ctx.pp_phi
                        };
                        // Route through MemoryCoordinator for quality filtering instead of
                        // bypassing directly to fep.episodic_memory.encode().
                        self.memory
                            .memory_consol
                            .memory_coordinator
                            .queue_graduation(crate::memory::memory_coordinator::GraduationEvent {
                                content: grad.embedding.clone(),
                                label: grad.id.clone(),
                                steps_survived: (self.stats.total_cycles as u64)
                                    .saturating_sub(grad.added_at),
                                final_activation: grad_importance as f64,
                                psi_at_graduation: ctx.pp_phi as f64,
                                coherence_at_graduation: ctx.coherence as f64,
                                source: Default::default(),
                                is_verified: false,
                            });
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
                if let Some(ref mut meta) = self.consciousness.self_model_tier.meta_cognition {
                    meta.update_self_model(ctx.prediction_error);
                    meta.deepen_recursion();
                    let accuracy = meta.accuracy();
                    let depth = meta.depth();
                    if accuracy > super::super::thresholds::META_COGNITIVE_ACCURACY_LR_THRESHOLD {
                        let boost = 1.0
                            + (accuracy
                                - super::super::thresholds::META_COGNITIVE_ACCURACY_LR_THRESHOLD)
                                * super::super::thresholds::META_COGNITIVE_LR_BOOST_SCALE; // up to 1.15x
                        self.carryover.learning.subsystem_lr_factor *= boost;
                    }
                    (accuracy, depth)
                } else {
                    (0.0, 0)
                }
            } else {
                // Read cached accuracy/depth without updating (avoid 0.0 in telemetry on skip)
                self.consciousness
                    .self_model_tier
                    .meta_cognition
                    .as_ref()
                    .map(|m| (m.accuracy(), m.depth()))
                    .unwrap_or((0.0, 0))
            };

        // ── Moral circularity → deepen meta-cognitive recursion ──
        // When β₁ > 0 (circular reasoning patterns detected in the moral
        // topology), trigger additional meta-cognitive introspection.
        if let Some(ref mut meta) = self.consciousness.self_model_tier.meta_cognition {
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
                if let Some(ref mut body) = self.sensorimotor.vision_sensory.virtual_body {
                    let signals = crate::cognitive_loop::virtual_body::CognitiveSignals {
                        prediction_error: ctx.prediction_error,
                        coherence: ctx.coherence,
                        prediction_confidence: self.prediction_confidence as f32,
                        unified_psi: ctx.unified_psi,
                        flow_intensity: self.behavior.flow_state.intensity,
                        in_flow: self.behavior.flow_state.in_flow,
                        curiosity_boredom: self.behavior.curiosity_drive.boredom,
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
        let body_valence_conf_delta = if body_valence
            > super::super::thresholds::BODY_VALENCE_POSITIVE_THRESHOLD
        {
            let delta = body_valence * super::super::thresholds::BODY_VALENCE_CONFIDENCE_POS_SCALE;
            self.adjust_confidence("body_valence_pos", delta);
            delta
        } else if body_valence < super::super::thresholds::BODY_VALENCE_NEGATIVE_THRESHOLD {
            let delta = body_valence * super::super::thresholds::BODY_VALENCE_CONFIDENCE_NEG_SCALE;
            self.adjust_confidence("body_valence_neg", delta);
            delta
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // AFFECTIVE BRIDGE: Evaluate somatic markers from cognitive signals
        // Runs every cycle (lightweight: ~5 arithmetic ops + blend)
        //
        // Phase 1 Affect Consolidation: Feed somatic signals into the unified
        // EmotionalBridge (UnifiedEmotionalState) so all consumers read from
        // a single canonical source. Legacy AffectiveBridge still runs as
        // fallback when enabled, but consumers read from unified state.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let moral_score = self
            .ethics_values
            .last_moral_judgment
            .as_ref()
            .map(|j| j.moral_score)
            .unwrap_or(0.0);

        // Primary path: update unified EmotionalBridge with somatic signals
        self.unification_engine
            .emotional
            .update_from_somatic_signals(
                ctx.prediction_error,
                ctx.surprise_triggered,
                ctx.unified_psi,
                moral_score,
                self.behavior.social_mgr.social.social_trust,
                self.behavior.social_mgr.social.social_cooperation_rate,
                0.0, // peer_valence: future — aggregate from social inbox
            );

        // Legacy fallback: keep AffectiveBridge in sync (deprecated — will be removed)
        if let Some(ref mut bridge) = self.consciousness_state.affective_bridge {
            bridge.evaluate_from_signals_with_social(
                ctx.prediction_error,
                ctx.surprise_triggered,
                ctx.unified_psi,
                moral_score,
                self.behavior.social_mgr.social.social_trust,
                self.behavior.social_mgr.social.social_cooperation_rate,
                0.0,
            );
        }

        // Read from unified state for downstream consumers
        let unified_emo = self.unification_engine.emotional.state();
        let affective_valence = unified_emo.valence as f32;
        let affective_arousal = unified_emo.arousal as f32;

        // FEEDBACK: Positive affect broadens exploration (Fredrickson 2001 broaden-and-build)
        // Now reads from unified state — no longer gated on AffectiveBridge being enabled
        if affective_valence > super::super::thresholds::AFFECTIVE_VALENCE_BROADEN_THRESHOLD {
            self.behavior.curiosity_drive.boredom *=
                super::super::thresholds::AFFECTIVE_VALENCE_CURIOSITY_FACTOR;
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
            let had_error =
                ctx.prediction_error > super::super::thresholds::USER_STATE_ERROR_THRESHOLD;
            usi.process(ctx.input, had_error);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // NARRATIVE SELF: Process experience and track self-Φ
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let narrative_self_psi = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut narrative) = self.consciousness.self_model_tier.narrative_self {
                let significance = if ctx.moral_concern_detected {
                    super::super::thresholds::NARRATIVE_MORAL_SIGNIFICANCE
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
            self.consciousness
                .self_model_tier
                .narrative_self
                .as_ref()
                .map(|n| n.self_phi())
                .unwrap_or(0.0)
        };

        // FEEDBACK: Narrative self-Phi modulates prediction confidence (identity coherence)
        // Science: Gallagher (2000) — strong narrative identity stabilizes learning
        let narrative_self_conf_factor =
            if narrative_self_psi > super::super::thresholds::NARRATIVE_SELF_STRONG_THRESHOLD {
                self.scale_confidence(
                    "narrative_self_strong",
                    super::super::thresholds::NARRATIVE_SELF_CONFIDENCE_BOOST,
                );
                super::super::thresholds::NARRATIVE_SELF_CONFIDENCE_BOOST
            } else if narrative_self_psi > 0.0
                && narrative_self_psi < super::super::thresholds::NARRATIVE_SELF_WEAK_THRESHOLD
            {
                self.scale_confidence(
                    "narrative_self_weak",
                    super::super::thresholds::NARRATIVE_SELF_CONFIDENCE_DAMPEN,
                );
                super::super::thresholds::NARRATIVE_SELF_CONFIDENCE_DAMPEN
            } else {
                1.0
            };

        // FEEDBACK: Narrative self-Phi modulates moral sensitivity (Gallagher & Hutto 2007)
        // Science: Strong narrative identity constrains moral reasoning (values are stable);
        // weak/incoherent identity amplifies moral sensitivity (recalibration needed)
        if narrative_self_psi > super::super::thresholds::NARRATIVE_SELF_HIGH_THRESHOLD {
            // High self-coherence → stabilize moral score (dampen fluctuations)
            // Multiply moral learning signal toward 1.0 (neutral)
            self.fep.learning_signal *= 1.0
                + (narrative_self_psi as f32
                    - super::super::thresholds::NARRATIVE_SELF_HIGH_THRESHOLD as f32)
                    * super::super::thresholds::NARRATIVE_SELF_MORAL_STABILIZE_SCALE;
        } else if narrative_self_psi > 0.0
            && narrative_self_psi < super::super::thresholds::NARRATIVE_SELF_WEAK_THRESHOLD
        {
            // Low self-coherence → amplify moral concern sensitivity
            self.behavior.adaptive_behavior.attention_sensitivity *= 1.0
                + (super::super::thresholds::NARRATIVE_SELF_WEAK_THRESHOLD as f32
                    - narrative_self_psi as f32)
                    * super::super::thresholds::NARRATIVE_SELF_MORAL_SENSITIVITY_SCALE;
        }

        // CROSS-COUPLING: Strong narrative identity stabilizes social trust (Baumeister & Leary 1995)
        // Science: Coherent self-identity supports consistent social relationships
        if narrative_self_psi > super::super::thresholds::NARRATIVE_SELF_HIGH_THRESHOLD {
            self.behavior.social_mgr.social.social_trust *=
                super::super::thresholds::NARRATIVE_SELF_SOCIAL_TRUST_BOOST;
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
        let modulation = (predictive_psi_modulation - 1.0).clamp(
            -super::super::thresholds::PREDICTIVE_PHI_MAX_MODULATION as f64,
            super::super::thresholds::PREDICTIVE_PHI_MAX_MODULATION as f64,
        ) as f32;
        let coherence_scale = super::super::thresholds::PREDICTIVE_PHI_COHERENCE_BASELINE
            + (1.0 - super::super::thresholds::PREDICTIVE_PHI_COHERENCE_BASELINE)
                * ctx.coherence.clamp(0.0, 1.0);
        let predictive_phi_lr_delta_val =
            modulation * super::super::thresholds::PREDICTIVE_PHI_LR_SCALE * coherence_scale; // ±1.5% max, coherence-weighted
        self.carryover.learning.subsystem_lr_factor *= 1.0 + predictive_phi_lr_delta_val;
        module_timings.predictive_processing = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // HIERARCHICAL FREE ENERGY: Multi-level variational decomposition
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // Science: Friston (2008) — hierarchical predictive processing
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let hierarchical_total_free_energy =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut hfe) = self
                    .consciousness
                    .consciousness_monitors
                    .hierarchical_free_energy
                {
                    // FEEDBACK: Phi→precision coupling — higher integrated information
                    // sharpens lower-level precision (Feldman & Friston 2010, §7.4).
                    // This creates a causal mechanism: consciousness improves perceptual accuracy.
                    #[cfg(feature = "unified_precision")]
                    {
                        // Unified precision: Phi × interoceptive × blanket modulation
                        // Science: Parr & Friston (2019) — precision IS attention
                        let phi_factor = ctx.unified_psi;
                        // Interoceptive factor from affective state (positive valence → higher precision)
                        let intero_factor = 1.0 + 0.2 * (affective_valence as f64).clamp(-1.0, 1.0);
                        // Blanket factor: use prediction confidence as proxy for blanket openness
                        let blanket_factor = (self.prediction_confidence as f64).clamp(0.0, 1.0);
                        hfe.update_precisions(phi_factor, intero_factor, blanket_factor);
                    }
                    #[cfg(not(feature = "unified_precision"))]
                    {
                        let psi_boost = (ctx.unified_psi * 0.5).clamp(0.0, 0.5);
                        let base_decay = hfe.config.precision_decay;
                        for level in &mut hfe.levels {
                            let base_precision = base_decay.powi(level.level as i32);
                            level.precision = base_precision * (1.0 + psi_boost);
                        }
                    }

                    // Build observation from compressed state (clamped to state_dim)
                    let obs: Vec<f64> = ctx
                        .compressed_state
                        .iter()
                        .take(hfe.config.state_dim)
                        .map(|&x| x as f64)
                        .collect();
                    hfe.update_beliefs(&obs);
                    let decomp = hfe.compute_free_energy();
                    // Feed HFE data to ThermodynamicManager
                    self.thermodynamic_mgr.set_hfe(
                        decomp.total,
                        decomp.complexity,
                        decomp.accuracy,
                    );
                    decomp.total
                } else {
                    0.0
                }
            } else {
                0.0
            };

        // FEEDBACK: High hierarchical free energy suppresses exploration AND boosts learning
        // Science: Friston (2008) — poor model → focus on learning, not exploring
        let hfe_lr_boost_applied = if hierarchical_total_free_energy
            > super::super::thresholds::HFE_EXPLORATION_THRESHOLD
        {
            let fe_factor = (1.0
                / (1.0
                    + hierarchical_total_free_energy
                        * super::super::thresholds::HFE_EXPLORATION_DAMPING))
                as f32;
            self.behavior.curiosity_drive.boredom *= fe_factor; // suppress exploration urge (gentler)
            // Boost LR proportional to free energy (poor model → learn harder)
            // Capped at +10% to avoid overshooting in short ablation windows
            let hfe_lr_boost = (1.0
                + (hierarchical_total_free_energy * super::super::thresholds::HFE_LR_BOOST_SCALE)
                    .min(super::super::thresholds::HFE_LR_BOOST_MAX))
                as f32;
            self.scale_lr("hierarchical_free_energy", hfe_lr_boost);
            hfe_lr_boost
        } else {
            1.0
        };

        // NOTE: Jarzynski→HFE and Onsager→coherence feedback now consolidated in
        // ThermodynamicIntegration::run_cycle() (thermodynamic_integration.rs).

        module_timings.hierarchical_free_energy = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PREDICTIVE SELF: Evaluate action safety via self-state prediction
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let predictive_self_safety = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut pred_self) = self.consciousness.self_model_tier.predictive_self {
                if let Some(ref narrative) = self.consciousness.self_model_tier.narrative_self {
                    pred_self.observe(narrative);
                }

                // Wire behavioral fields from cycle context into the predictive self-model.
                // moral_score: from the most recent ethical judgment
                let moral_score = self
                    .ethics_values
                    .last_moral_judgment
                    .as_ref()
                    .map(|j| j.moral_score as f64)
                    .unwrap_or(0.0);
                // exploration_urge: noradrenaline-driven exploration drive
                let exploration_urge = self.neuromod.bath.noradrenaline.effective() as f64;
                // behavioral_coherence: consistency of moral_score across recent history.
                // Derived from the predictor's own history — low variance = high coherence.
                let behavioral_coherence = {
                    let history = &pred_self.predictor.history;
                    if history.len() >= 2 {
                        let scores: Vec<f64> = history.iter().map(|s| s.moral_score).collect();
                        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                            / scores.len() as f64;
                        // Map variance → coherence: low variance = high coherence
                        (1.0_f64 - variance.sqrt().min(1.0)).clamp(0.0, 1.0)
                    } else {
                        0.5 // neutral default when insufficient history
                    }
                };

                pred_self.observe_behavioral(moral_score, exploration_urge, behavioral_coherence);

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
        if predictive_self_safety > 0.0
            && predictive_self_safety < super::super::thresholds::PREDICTIVE_SELF_SAFETY_THRESHOLD
        {
            let safety_factor = super::super::thresholds::PREDICTIVE_SELF_SAFETY_MIN
                + predictive_self_safety * super::super::thresholds::PREDICTIVE_SELF_SAFETY_SCALE; // 0.85-1.0
            self.carryover.learning.subsystem_lr_factor *= safety_factor;

            // CROSS-COUPLING: Low self-prediction confidence → boost exploration
            // Science: Clark (2013) — uncertain self-models need evidence gathering
            let uncertainty_push = (super::super::thresholds::PREDICTIVE_SELF_SAFETY_THRESHOLD
                - predictive_self_safety)
                * super::super::thresholds::PREDICTIVE_SELF_EXPLORATION_SCALE;
            self.adjust_exploration("low_self_prediction", uncertainty_push);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // ATTENTION SCHEMA: Track attention state and generate control signals
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let attention_schema_focus = if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut schema) = self.consciousness.self_model_tier.attention_schema {
                let salience = ctx.prediction_error.max(0.1);
                let update = schema.update(ctx.hv16_cached, salience);
                let gain = if update.control_signal
                    > super::super::thresholds::ATTENTION_FOCUS_GAIN_THRESHOLD
                {
                    ((update.control_signal
                        - super::super::thresholds::ATTENTION_FOCUS_GAIN_THRESHOLD)
                        * super::super::thresholds::ATTENTION_GAIN_SCALE)
                        .min(super::super::thresholds::ATTENTION_MAX_GAIN)
                } else if update.control_signal
                    < super::super::thresholds::ATTENTION_DEFOCUS_THRESHOLD
                {
                    super::super::thresholds::ATTENTION_NEGATIVE_GAIN
                } else {
                    0.0
                };
                self.behavior.adaptive_behavior.attention_sensitivity *= 1.0 + gain;
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
            if attention_schema_focus < super::super::thresholds::ATTENTION_FOCUS_GAIN_THRESHOLD {
                // Attenuated 50%: ACh attention_factor already scales attention via the bath
                let novelty_push = (super::super::thresholds::ATTENTION_FOCUS_GAIN_THRESHOLD
                    - attention_schema_focus)
                    * super::super::thresholds::ATTENTION_DEFICIT_NOVELTY_SCALE;
                self.adjust_exploration("attention_deficit", novelty_push);
            } else if attention_schema_focus
                > super::super::thresholds::ATTENTION_DEEP_FOCUS_THRESHOLD
            {
                let focus_lock = (attention_schema_focus
                    - super::super::thresholds::ATTENTION_DEEP_FOCUS_THRESHOLD)
                    * super::super::thresholds::ATTENTION_FOCUS_LOCK_SCALE;
                self.behavior.adaptive_behavior.exploration_factor *= (1.0 - focus_lock)
                    .max(super::super::thresholds::ATTENTION_MIN_EXPLORATION_IN_FOCUS);
            }
        }

        // FEEDBACK: Vigilance fatigue drives attention shift pressure (Mackworth 1948, AST-1)
        // Science: Sustained attention on a single target degrades performance after ~30 cycles.
        // The schema's fatigue_level() reflects accumulated focus duration. When fatigue is
        // high, increase exploration to encourage attention re-allocation.
        if let Some(ref schema) = self.consciousness.self_model_tier.attention_schema {
            let fatigue = schema.fatigue_level();
            if fatigue > super::super::thresholds::VIGILANCE_FATIGUE_THRESHOLD {
                // Graduated exploration push: 0.0 at fatigue=0.5, up to 0.04 at fatigue=1.0
                let fatigue_push = (fatigue
                    - super::super::thresholds::VIGILANCE_FATIGUE_THRESHOLD)
                    * super::super::thresholds::VIGILANCE_FATIGUE_EXPLORATION_SCALE;
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
            self.scale_exploration(
                "phi_gate_suppress",
                super::super::thresholds::PHI_GATE_SUPPRESS_EXPLORATION,
            );
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
            predictive_behavioral_error: self
                .consciousness
                .self_model_tier
                .predictive_self
                .as_ref()
                .map(|ps| ps.stats.behavioral_prediction_error as f32)
                .unwrap_or(0.0),
            attention_schema_focus,
            attention_fatigue: self
                .consciousness
                .self_model_tier
                .attention_schema
                .as_ref()
                .map(|s| s.fatigue_level())
                .unwrap_or(0.0),
            attention_prediction_accuracy: self
                .consciousness
                .self_model_tier
                .attention_schema
                .as_ref()
                .map(|s| s.prediction_accuracy() as f32)
                .unwrap_or(0.0),
            psi_attention_avg,
            hierarchical_free_energy_lr_boost: hfe_lr_boost_applied,
            predictive_phi_lr_delta: predictive_phi_lr_delta_val,
            body_valence_confidence_delta: body_valence_conf_delta,
            narrative_self_confidence_factor: narrative_self_conf_factor,
        }
    }
}
