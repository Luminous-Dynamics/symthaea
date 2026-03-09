use std::time::Instant;

use super::{ConsciousnessIntegrationResult, LateConsciousnessContext, LateConsciousnessResult};
use crate::cognitive_loop::CognitiveLoopService;

impl CognitiveLoopService {
    /// Run consciousness integration: GWT, cross-modal binding, consciousness monitors
    /// (resonance, quantum coherence), phenomenal binding, temporal consciousness,
    /// consciousness thermodynamics, embodied cognition, narrative-GWT integration,
    /// unified living mind, and master consciousness equation.
    ///
    /// Extracted from cycle.rs — all logic and behavior preserved exactly.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::cognitive_loop) fn run_consciousness_integration(
        &mut self,
        ctx: &LateConsciousnessContext,
        late: &LateConsciousnessResult,
        module_timings: &mut crate::cognitive_loop::ModuleTimings,
    ) -> ConsciousnessIntegrationResult {
        // ═══════════════════════════════════════════════════════════════════════
        // GWT INTEGRATION: Submit encoding to global workspace for broadcast
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (gwt_broadcast, gwt_coalition_size) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut gwt) = self.gwt_mgr.gwt {
                    let activation = (1.0 - ctx.prediction_error as f64).clamp(0.0, 1.0);
                    // Submit current encoding with activation-weighted salience
                    gwt.submit_strategy(
                        "cognitive_loop",
                        activation,
                        vec![ctx.hv16_cached],
                        vec!["encoder".to_string()],
                    );
                    // If previous cycle's subsystems requested broadcast, boost salience
                    if self.carryover.gwt_broadcast_occurred {
                        gwt.submit_strategy(
                            "cross_domain_priming",
                            activation * 1.2, // Slightly higher salience for primed content
                            vec![ctx.hv16_cached],
                            vec!["priming".to_string()],
                        );
                    }
                    let result = gwt.process();
                    let coalition_size = result
                        .winning_coalition
                        .as_ref()
                        .map(|c| c.members.len() as u32)
                        .unwrap_or(0);
                    (result.broadcast_occurred, coalition_size)
                } else {
                    (false, 0)
                }
            } else {
                (false, 0)
            };

        // Store GWT state in carryover for cross-domain coupling (next cycle)
        self.carryover.gwt_broadcast_occurred = gwt_broadcast;
        self.carryover.gwt_coalition_size = gwt_coalition_size;

        // FEEDBACK: GWT broadcast boosts confidence (conscious access moment)
        // Science: Baars (1988) — broadcast = conscious access, should amplify integration
        if gwt_broadcast {
            self.adjust_confidence(
                "gwt_broadcast",
                crate::cognitive_loop::thresholds::GWT_BROADCAST_CONFIDENCE_BOOST,
            );
        }

        module_timings.gwt = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CROSS-MODAL BINDING: Bind HDC encodings across modalities
        // Runs every cycle (lightweight: 2 HV ops + similarity)
        // Substrate-gated: skip when substrate is degraded
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (cross_modal_binding_strength, cross_modal_psi) =
            if self.substrate_manager.should_degrade_consciousness() {
                (0.0, 0.0)
            } else {
                self.update_cross_modal_binding(
                    &ctx.hv16_cached,
                    late.affective_valence,
                    late.predictive_free_energy,
                )
            };

        module_timings.cross_modal_binding = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS MONITORS: Resonance + Quantum coherence
        // Urgency-gated: skip in Cruise mode
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Pre-compute to avoid borrow conflict with mutable subsystem references below
        let wm_utilization = self.prefrontal_utilization();
        let resonance_frequency = if ctx.urgency.run_consciousness_monitors() {
            if let Some(ref mut resonance) = self.consciousness_monitors.resonance {
                let dims = [
                    ctx.unified_psi,
                    ctx.coherence as f64,
                    wm_utilization,
                    self.adaptive_behavior.attention_sensitivity as f64,
                    (self.stats.total_cycles.min(100) as f64 / 100.0),
                    late.body_psi_modulation,
                    self.prediction_confidence,
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
            if let Some(ref mut qc) = self.consciousness_monitors.quantum_coherence {
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
            self.adjust_confidence("quantum_coherence_high", qc_boost);
        } else if quantum_coherence_level > 0.0 && quantum_coherence_level < 0.2 {
            self.scale_confidence("quantum_decoherence", 0.98);
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
                if let Some(ref mut binding) = self.consciousness_monitors.phenomenal_binding {
                    let dims = [
                        ctx.unified_psi,
                        ctx.coherence as f64,
                        wm_utilization,
                        self.adaptive_behavior.attention_sensitivity as f64,
                        (self.stats.total_cycles.min(100) as f64 / 100.0),
                        late.body_psi_modulation,
                        self.prediction_confidence,
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
                if let Some(ref mut temporal) = self.consciousness_monitors.temporal {
                    temporal.observe(
                        &ctx.hv16_cached,
                        ctx.unified_psi,
                        self.self_model_tier.narrative_self.as_ref(),
                        self.self_model_tier.predictive_self.as_ref(),
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
            self.set_lr("temporal_discontinuity", 1.0);
            self.scale_confidence("temporal_discontinuity", 0.8);
            // Lower learning threshold to learn more aggressively after discontinuity
            self.scale_threshold("temporal_discontinuity", 0.8);
        } else if temporal_coherence_score > 0.8 {
            // High temporal coherence → model is reliable, raise threshold (learn less often)
            self.scale_threshold("temporal_high_coherence", 1.01);
        } else {
            // Slowly return toward baseline (homeostasis drift toward 1.0)
            let drift = (1.0 - self.carryover.learning.adaptive_threshold_scale) * 0.02;
            self.adjust_threshold("homeostasis_drift", drift as f32);
        }

        // ── Phase 21: Temporal discontinuity recovery cascade ────────────
        // Science: Context shift detection requires graduated recovery, not just instant reset
        if temporal_discontinuity {
            self.carryover.urgency.discontinuity_streak += 1;
        } else {
            self.carryover.urgency.discontinuity_streak = self
                .carryover
                .urgency
                .discontinuity_streak
                .saturating_sub(1);
        }
        let streak = self.carryover.urgency.discontinuity_streak;
        if streak >= 3 {
            // Persistent discontinuity: aggressive recovery
            self.last_prediction = None; // invalidate stale predictions
            self.scale_lr("persistent_discontinuity", 1.5);
            self.scale_exploration("discontinuity_recovery", 0.7);
            self.stats.discontinuity_cascade_count += 1;
        }

        // FEEDBACK: High temporal coherence strengthens narrative self engagement
        // Science: Damasio (2010) — temporal continuity is the substrate of selfhood
        if temporal_coherence_score > 0.6 {
            if let Some(ref mut narrative) = self.self_model_tier.narrative_self {
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
                if let Some(ref mut thermo) = self.consciousness_monitors.thermodynamics {
                    let dims = [
                        ctx.unified_psi,
                        ctx.coherence as f64,
                        wm_utilization,
                        self.adaptive_behavior.attention_sensitivity as f64,
                        (self.stats.total_cycles.min(100) as f64 / 100.0),
                        late.body_psi_modulation,
                        self.prediction_confidence,
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
                self.adjust_exploration("thermo_entropy", entropy_boost);
            } else if thermodynamic_entropy < 0.3 {
                let consolidation_bias = ((0.3 - thermodynamic_entropy) * 0.08).min(0.08) as f32;
                self.scale_lr("low_entropy_consolidate", 1.0 + consolidation_bias);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // EMBODIED COGNITION: Bridge virtual body to full body schema
        // Urgency-gated: Critical=always, Normal=always, Cruise=every 2nd
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (embodied_psi_modulation, embodied_agency) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 1, 2) {
                if let Some(ref mut embodied) = self.consciousness_monitors.embodied {
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
                        self.adjust_exploration("sensorimotor_surprise", body_nudge);
                    }
                    // 3. High allostatic load suppresses learning (conserve resources)
                    // Science: McEwen (2004) — allostatic overload impairs plasticity
                    if response.allostatic_load > 0.7 {
                        self.scale_lr(
                            "allostatic_overload",
                            1.0 - (response.allostatic_load - 0.7) as f32 * 0.5,
                        );
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
            self.scale_exploration("embodied_caution", (1.0 - caution).max(0.7));
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
                crate::cognitive_loop::ActionHint::Explore => {
                    crate::consciousness::enactive_cognition::ActionType::Explore
                }
                crate::cognitive_loop::ActionHint::SeekInput => {
                    crate::consciousness::enactive_cognition::ActionType::Observe
                }
                crate::cognitive_loop::ActionHint::SlowDown => {
                    crate::consciousness::enactive_cognition::ActionType::Reflect
                }
                crate::cognitive_loop::ActionHint::SpeedUp => {
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
                self.scale_confidence(
                    "enacted_meaning_neg",
                    (1.0 + enacted_meaning.meaning.valence * 0.1) as f32,
                );
            }

            // Integrate all subsystems into unified living state
            let free_energy = self.fep.agent.current_free_energy();
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
            // Wire embodiment factor from cognitive loop signals.
            // Science: Friston (2010) — low PE = good embodied prediction (sensorimotor accuracy)
            // Science: Barrett (2017) — interoceptive coherence from allostatic regulation
            self.master_equation.embodiment_factor.record_prediction(
                1.0 - ctx.prediction_error as f64,
                1.0 - ctx.prediction_error as f64,
            );
            // Use allostatic load as direct interoceptive coherence signal.
            // Low allostatic load = high body coherence (expected ≈ actual).
            {
                let allostatic = self.neuromod.bath.allostatic_load;
                let coherence = 1.0 - allostatic as f64;
                self.master_equation.embodiment_factor.update_interoceptive(
                    coherence, coherence,
                );
            }

            // Wire narrative coherence with lightweight episodes (every 5 cycles)
            // Doubled from every-10 for faster autobiographical integration.
            // Science: Damasio (2010) — self emerges from autobiographical narrative continuity.
            // Conway (2005) — narrative identity forms from dense episodic sampling.
            if self.stats.total_cycles % 5 == 0 {
                let valence = (1.0 - ctx.prediction_error as f64).clamp(-1.0, 1.0);
                self.master_equation
                    .narrative_coherence
                    .add_episode(format!("cycle_{}", self.stats.total_cycles), valence);
            }

            // Wire future scenarios from prediction confidence (every 25 cycles).
            // Doubled from every-50 for faster future_simulation_depth growth.
            // Science: Schacter et al. (2012) — prospection uses same networks as episodic memory
            if self.stats.total_cycles % 25 == 0 {
                let horizon = ((1.0 - ctx.prediction_error as f64) * 10.0).max(1.0) as usize;
                self.master_equation
                    .narrative_coherence
                    .add_future_scenario(
                        format!("prediction_horizon_{}", self.stats.total_cycles),
                        horizon,
                        self.prediction_confidence.clamp(0.0, 1.0),
                        (1.0 - ctx.prediction_error as f64).clamp(-1.0, 1.0),
                    );
            }

            // Wire SocialEmbedding from existing cognitive signals.
            // The system already HAS a self-model (meta-cognition, predictive self,
            // attention schema) AND processes user input (implicitly modeling another
            // agent), but SocialEmbedding never received either, leaving Soc at 0.35.
            // Science: Gallese (2001) — self-other distinction requires self-model;
            // Frith & Frith (2006) — Theory of Mind from predicting others' behavior.
            if self.stats.total_cycles % 10 == 0 {
                // Self-model: what the system knows about itself
                let self_goals = vec![
                    "reduce_prediction_error".to_string(),
                    "maintain_coherence".to_string(),
                    "integrate_information".to_string(),
                ];
                let self_beliefs = vec![
                    format!("confidence_{:.1}", self.prediction_confidence),
                    format!("meta_accuracy_{:.1}", late.meta_cognitive_accuracy),
                    format!("coherence_{:.1}", ctx.coherence),
                    format!("safety_{:.1}", late.predictive_self_safety),
                ];
                self.master_equation.social_embedding.update_self_model(
                    self_goals,
                    self_beliefs,
                    late.affective_valence as f64,
                );

                // User agent model: the system IS modeling the user (their input
                // drives prediction, their patterns are tracked by social_coherence).
                // Prediction accuracy serves as ToM accuracy proxy.
                let user_goals = vec!["communicate".to_string(), "seek_understanding".to_string()];
                let user_beliefs = vec![
                    format!("trust_{:.1}", self.social_mgr.social.social_trust),
                    format!(
                        "cooperation_{:.1}",
                        self.social_mgr.social.social_cooperation_rate
                    ),
                ];
                self.master_equation.social_embedding.update_agent_model(
                    "user",
                    user_beliefs,
                    user_goals,
                    0.0, // neutral valence (we don't know user's emotions)
                    self.social_mgr.social.social_prediction_accuracy as f64,
                );

                // Feed prediction accuracy as ToM feedback — when the system
                // correctly predicts user input patterns, its "other_modeling_accuracy"
                // should reflect that.
                let accuracy = (1.0 - ctx.prediction_error as f64).clamp(0.0, 1.0);
                self.master_equation
                    .social_embedding
                    .record_tom_prediction("user", accuracy);
                self.master_equation
                    .social_embedding
                    .provide_tom_feedback("user", accuracy);
            }

            // Blend SpectralMIP Phi into the MCE's Φ input when cached.
            // unified_psi is a lightweight proxy (~0.22) that underestimates actual
            // information integration. SpectralMIP (expensive, every ~97 cycles) is
            // the gold-standard measure. Use the higher of the two, discounting
            // spectral by 0.6 for conservatism (it uses Gaussian MI, not TPM-based IIT).
            // Science: Tononi (2004) — Φ should reflect the system's best available
            // estimate of information integration, not just the cheapest proxy.
            let spectral_boost = self
                .carryover
                .consciousness
                .last_spectral_mip_phi
                .map(|phi| (phi * 0.6).clamp(0.0, 1.0))
                .unwrap_or(0.0);
            let phi_input = ctx.unified_psi.max(spectral_boost).clamp(0.0, 1.0);

            // Enrich broadcast: raw CfC coherence (~0.51) is too narrow a proxy
            // for "global workspace broadcast." GWT broadcast events ARE broadcast
            // by definition (Baars 1988). Blend coherence with GWT success and
            // attention focus for a richer broadcast measure.
            // Science: Dehaene et al. (2006) — conscious access = ignition + broadcast.
            let gwt_boost = if gwt_broadcast {
                0.2 + 0.1 * (gwt_coalition_size.min(5) as f64 / 5.0)
            } else {
                0.0
            };
            let attention_boost = ctx.peak_attention as f64 * 0.15;
            let broadcast_input =
                (ctx.coherence as f64 + gwt_boost + attention_boost).clamp(0.0, 1.0);

            let inputs = crate::consciousness::master_consciousness_equation::ConsciousnessInputs {
                phi: phi_input,
                broadcast: broadcast_input,
                working_memory: self.prefrontal_utilization(),
                attention: ctx.peak_attention as f64,
                // CfC is recurrent from cycle 1 — starting at 0 is dishonest and
                // crushes the softmin (τ=0.1). Floor at 0.3, ramp to 1.0 over 100 cycles.
                // Science: Elman (1990) — recurrent networks have temporal integration from t=0.
                recurrence: (0.3 + 0.7 * self.stats.total_cycles.min(100) as f64 / 100.0),
                embodiment: late.body_psi_modulation, // virtual body provides embodiment
                // Floor at 0.2: even with no predictions, the system has SOME knowledge
                // of its own state. Prevents softmin death spiral where knowledge→0
                // crushes consciousness, preventing the learning that would raise it.
                knowledge: self.prediction_confidence.max(0.2),
                // Synchrony: baseline + coherence + attention + flow.
                // Attention IS synchrony — selective amplification requires
                // phase-locked neural coordination (Fries 2015). Raised baseline
                // from 0.3→0.35: idle neural networks exhibit background
                // synchrony (Buzsáki 2006), but not so high that moral/attentional
                // perturbations are masked. Attention adds a second source.
                synchrony: (0.35
                    + ctx.coherence as f64 * 0.25
                    + ctx.peak_attention as f64 * 0.15
                    + self.flow_state.intensity as f64 * 0.25)
                    .clamp(0.1, 1.0),
            };
            let mce_result = self.master_equation.compute(&inputs);
            let level = mce_result.consciousness_level;

            // Cache MCE factor telemetry for output phase
            self.carryover.consciousness.mce_bottleneck_name = mce_result.bottleneck_name.clone();
            self.carryover.consciousness.mce_softmin = mce_result.bottleneck_factor;
            self.carryover.consciousness.mce_weighted_sum = mce_result.weighted_sum;
            self.carryover.consciousness.mce_narrative = mce_result.narrative_coherence;
            self.carryover.consciousness.mce_social = mce_result.social_embedding;

            // Track consciousness level for learning gating (Task C)
            self.carryover.history.consciousness_level = level;

            // FEEDBACK: MCE consciousness level boosts learning rate (decaying)
            // Science: Dehaene (2014) — conscious access improves encoding
            if level > 0.0 {
                self.carryover.learning.mce_lr_boost =
                    (level * crate::cognitive_loop::thresholds::MCE_LR_BOOST_SCALE as f64) as f32;
            } else {
                self.carryover.learning.mce_lr_boost *=
                    crate::cognitive_loop::thresholds::MCE_BOOST_DECAY;
            }

            // FEEDBACK: Consciousness gates consolidation intensity (Dehaene 2014)
            // Science: Only conscious moments produce durable memories. Scale episodic
            // consolidation by consciousness level — low consciousness → skip storage,
            // high consciousness → prioritize memory encoding.
            // Adaptive threshold: consolidate when consciousness exceeds its own rolling
            // average minus a margin. This ensures consolidation fires during relatively
            // "aware" moments regardless of absolute consciousness range.
            let ema = &mut self.carryover.history.consciousness_ema;
            *ema = *ema * 0.95 + level * 0.05; // EMA α=0.05, ~20-cycle half-life
            let consolidation_threshold = (*ema - 0.1).max(0.2); // floor at 0.2
            if level > consolidation_threshold {
                // Trigger demand-driven consolidation at above-average consciousness
                self.fep.episodic_memory.consolidate_recent();
            }
            // Scale learning signal by consciousness quality (gradual, not on/off)
            // This complements the binary consciousness_awake gate with continuous modulation
            self.fep.learning_signal *= (0.5_f32 + level as f32 * 0.5_f32).clamp(0.5_f32, 1.0_f32);

            level
        } else {
            // Decay MCE LR boost between MCE firings
            self.carryover.learning.mce_lr_boost *=
                crate::cognitive_loop::thresholds::MCE_BOOST_DECAY;
            // Carry forward last computed consciousness level rather than dropping to 0.
            // The MCE is expensive so we gate its frequency, but consciousness doesn't
            // vanish between measurements — it persists with gradual decay.
            // Science: Tononi (2004) — Phi is a continuous property, not episodic.
            self.carryover.history.consciousness_level * 0.98 // gentle decay between measurements
        };

        // Social cognition modulation: accurate ToM predictions boost consciousness.
        // Science: Frith & Frith (2006) — social cognition recruits higher-order
        // mentalizing networks that correlate with conscious processing capacity.
        // Scale: 0.95 at accuracy=0.0, 1.05 at accuracy=1.0 (mild +/-5% modulation).
        let social_accuracy = self.social_mgr.social.social_prediction_accuracy;
        let social_mod = 0.95 + 0.1 * social_accuracy as f64;
        let consciousness_level = (consciousness_level * social_mod).clamp(0.0, 1.0);

        // Store resonance frequency and quantum coherence for next cycle's feedback
        self.carryover.history.resonance_frequency = resonance_frequency;
        self.carryover.consciousness.quantum_coherence = quantum_coherence_level;
        module_timings.master_consciousness_equation = _t.elapsed().as_micros() as u64;

        ConsciousnessIntegrationResult {
            gwt_broadcast,
            gwt_coalition_size,
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
