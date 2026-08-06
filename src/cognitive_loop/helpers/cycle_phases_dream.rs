// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream engine phase: recording, simulation, wisdom application.
//!
//! Contains `run_dream_phase`.

use std::time::Instant;

use super::super::CognitiveLoopService;
use super::cycle_phases::DreamPhaseResult;

impl CognitiveLoopService {
    /// Dream engine phase: record surprise events, run dream simulations during Cruise
    /// urgency, apply accumulated wisdom to exploration/confidence, and manage the
    /// dream feedback bridge for context-aware priors.
    ///
    /// Extracted from cycle() -- all logic and behavior preserved exactly.
    pub(in crate::cognitive_loop) fn run_dream_phase(
        &mut self,
        state: &super::super::CycleState<'_>,
        prediction: &[f32],
        module_timings: &mut super::super::ModuleTimings,
    ) -> DreamPhaseResult {
        let compressed_state = state.compressed_state;
        let output = state.output;
        let prediction_error = state.prediction_error;
        let unified_psi = state.unified_psi;
        let hv16_cached = state.hv16_cached;
        let urgency = state.urgency;
        let _t = Instant::now();
        // 1. Every cycle: record high-surprise events for later dreaming.
        // 2. During Cruise urgency: run a dream cycle to discover better actions.
        // 3. Apply accumulated wisdom to bias exploration toward Phi-optimal choices.
        let mut dream_insights: usize = 0;
        let mut dream_phi_improvement: f32 = 0.0;
        let mut dream_wisdom_count: usize = 0;
        if let Some(ref mut dream) = self.dream_engine {
            // Record: use compressed state as "state", output as "action",
            // and prediction as "outcome" — these align with the dream API dimensions
            let dream_state: Vec<f32> = compressed_state.iter().take(64).copied().collect();
            let dream_action: Vec<f32> = output.iter().take(32).copied().collect();
            let dream_outcome: Vec<f32> = prediction.iter().take(64).copied().collect();
            // Weight surprise by consciousness level and narrative self-coherence:
            // Science: Tononi (2015) — consciousness = integrated information = memory salience
            // Narrative→Dream coupling (Conway 2005): self-relevant memories encode preferentially.
            let narrative_salience = self
                .consciousness
                .self_model_tier
                .narrative_self
                .as_ref()
                .map(
                    |n: &crate::consciousness::narrative_self::NarrativeSelfModel| {
                        1.0 + n.self_phi() as f32 * 0.5
                    },
                ) // 1.0 to 1.5x boost
                .unwrap_or(1.0);
            let mut phi_weighted_surprise =
                prediction_error * (1.0 + unified_psi as f32).clamp(1.0, 2.0) * narrative_salience;

            // Knowledge-aware dream consolidation: contradictions and deep causal
            // chains boost consolidation priority for offline integration.
            // Science: Festinger (1957) — cognitive dissonance drives consolidation;
            //          Hobson & Friston (2012) — active inference in dreams consolidates causal models.
            if let Some(ref km) = self.memory.knowledge_manager {
                let sigs = km.signals();
                // Active contradictions boost consolidation by 20%
                if sigs.contradiction_signal.is_finite() && sigs.contradiction_signal > 0.0 {
                    phi_weighted_surprise *=
                        1.0 + super::super::thresholds::DREAM_KNOWLEDGE_CONTRADICTION_BOOST as f32;
                }
                // Deep causal chains boost consolidation by 10%
                if sigs.causal_depth.is_finite()
                    && sigs.causal_depth
                        > super::super::thresholds::DREAM_KNOWLEDGE_CAUSAL_DEPTH_THRESHOLD
                {
                    phi_weighted_surprise *=
                        1.0 + super::super::thresholds::DREAM_KNOWLEDGE_CAUSAL_DEPTH_BOOST as f32;
                }
            }
            // Reasoning reliability → dream consolidation boost: surprising events
            // that contradicted high-confidence reasoning get more dream time.
            // Science: Hobson & Friston (2012) — predictive processing theory of
            // dreaming; high-confidence prediction errors are preferentially consolidated.
            #[cfg(feature = "reasoning_engine")]
            if let Some(ref reasoning_engine) = self.reasoning_engine {
                if let Some(last_event) = reasoning_engine.last_event() {
                    let r = last_event.reliability;
                    if r > super::super::thresholds::REASONING_RELIABILITY_THRESHOLD {
                        // High reliability + surprise → boost consolidation weight
                        let boost =
                            (r - 0.5) * super::super::thresholds::DREAM_REASONING_RELIABILITY_SCALE;
                        phi_weighted_surprise *= 1.0 + boost as f32;
                    }
                }
            }
            // Scene recognition boost: recognized visual contexts encode preferentially.
            // Conway (2005) — self-relevant and context-rich memories encode preferentially.
            #[cfg(feature = "vision-manifold")]
            let phi_weighted_surprise = if state.scene_recognized {
                phi_weighted_surprise * super::super::thresholds::SCENE_RECOGNITION_DREAM_BOOST
            } else {
                phi_weighted_surprise
            };
            dream.record(
                &dream_state,
                dream_action,
                &dream_outcome,
                phi_weighted_surprise,
            );

            // Therapeutic dream recording: encode current therapeutic state as a
            // dream action so the dream engine can generate counterfactual therapeutic
            // scenarios. Only records when therapeutic manager has an active strategy
            // and surprise is meaningful.
            // Science: Barrett (2001) — dreams process emotional concerns;
            // Cartwright (2010) — dream content reflects waking emotional regulation.
            #[cfg(feature = "therapeutic")]
            if self.therapeutic_manager.active_strategy().is_some() && phi_weighted_surprise > 0.3 {
                let therapeutic_action = self.therapeutic_manager.dream_action_vector();
                // Use client RDoC profile as dream state, current compressed state as outcome
                let therapeutic_dream_state: Vec<f32> = therapeutic_action[0..6].to_vec();
                dream.record(
                    &therapeutic_dream_state,
                    therapeutic_action,
                    &dream_outcome,
                    phi_weighted_surprise * 0.8, // Slightly lower priority than primary events
                );
            }

            // Shadow→Dream bridge: record high-pressure shadow fragments as dream events.
            // The dream engine generates counterfactuals: "what if this shadow content
            // were integrated rather than repressed?"
            // Science: Jung (1951) — shadow integration through active imagination;
            // Hartmann (1998) — dreams as "safe space" for processing threatening content.
            #[cfg(feature = "therapeutic")]
            if let Some(shadow_input) = self
                .therapeutic_manager
                .shadow_detector
                .next_dream_fragment()
            {
                dream.record(
                    &shadow_input.state_vector,
                    shadow_input.action_vector,
                    &dream_outcome,
                    phi_weighted_surprise * 0.6, // Lower priority than primary/therapeutic
                );
            }

            // Dream during Cruise urgency (low-error steady state) or periodically.
            // Consolidation pressure modulates frequency: when GWT broadcasts pile up
            // faster than they're processed, dream more often to integrate them.
            // Base: every 20 cycles. At pressure 0.7+: every 5 cycles (4× faster).
            let pressure = self.memory_manager.consolidation_pressure();
            let memory_load = (dream.memory_size() as f32 / 100.0).min(1.0);
            let combined_pressure = (pressure + memory_load * 0.3).min(1.0);
            let dynamic_normal_interval = if combined_pressure > 0.7 {
                5 // urgent: dream every 5 cycles
            } else if combined_pressure > 0.4 {
                // interpolate 20→5 over [0.4, 0.7]
                let t = ((combined_pressure - 0.4) / 0.3).min(1.0);
                20 - (t * 15.0) as usize
            } else {
                20 // base rate
            };
            // Adaptive LR-driven interval: high learning rate → consolidate more often.
            // When the system is actively encoding new experience, dream consolidation
            // should keep pace. Take the minimum with the pressure-based interval so
            // either signal (memory pressure OR high LR) can accelerate dreaming.
            // Science: Diekelmann & Born (2010) — sleep consolidation scales with learning load.
            let lr_driven_interval =
                self.cantor_dream.adaptive_interval(self.fep.lr_boost) as usize;
            let dynamic_normal_interval = dynamic_normal_interval.min(lr_driven_interval);
            // Sacred Stillness modulates dream depth: high SS activation = deeper consolidation
            // Science: Walker & Stickgold (2006) — rest quality enhances memory consolidation
            let stillness_depth_factor = if self.stats.circadian_stillness_boost > 0.1 {
                0.7 // dream more frequently during rest (reduce interval by 30%)
            } else {
                1.0
            };
            let dynamic_normal_interval =
                (dynamic_normal_interval as f32 * stillness_depth_factor) as usize;
            // Moral fragmentation → dream urgency: high L₀ harmonic = isolated clusters
            let fragmentation_factor = {
                let summary = self.ethics_engine.moral_topology().last_summary();
                match summary.hodge_fractions {
                    Some(ref f) if f.harmonic > 0.8 => 0.5,
                    Some(ref f) if f.harmonic > 0.5 => 0.8,
                    _ => 1.0,
                }
            };
            let dynamic_normal_interval =
                (dynamic_normal_interval as f64 * fragmentation_factor) as usize;
            let dynamic_normal_interval = dynamic_normal_interval.max(3);
            if matches!(urgency, super::super::CycleUrgency::Cruise)
                || urgency.should_run(self.stats.total_cycles, 10, dynamic_normal_interval, 5)
            {
                match dream.dream() {
                    Ok(result) => {
                        dream_insights = result.insights;
                        dream_phi_improvement = result.best_phi_improvement;

                        if result.insights > 0 {
                            tracing::debug!(
                                insights = result.insights,
                                phi_improvement = result.best_phi_improvement,
                                simulations = result.simulations_run,
                                cycle = self.stats.total_cycles,
                                "Dream replay generated insights"
                            );

                            // Dream→Narrative coupling: dream insights feed narrative self-model.
                            // Science: Revonsuo (2000) — dreaming enhances threat simulation
                            // and narrative integration of novel experiences.
                            if let Some(ref mut narrative) =
                                self.consciousness.self_model_tier.narrative_self
                            {
                                narrative.process_experience(
                                    hv16_cached,
                                    &format!("dream_insight_{}", result.insights),
                                    true, // counterfactual-validated
                                    unified_psi,
                                    result.best_phi_improvement as f64,
                                );
                            }

                            // Autobiography narration: once enough new episodes
                            // have accumulated since the last narration, hand the
                            // current life_story off to the background LLM
                            // narrator (see managers::autobiography_bridge docs
                            // for why this can't just run inline — it's async
                            // and LLM-backed).
                            if let Some(episode_count) = self
                                .consciousness
                                .self_model_tier
                                .narrative_self
                                .as_ref()
                                .map(|n| n.autobio.life_story.len())
                            {
                                let cycle_num = self.stats.total_cycles as u64;
                                if let Some(ref mut narrator) =
                                    self.consciousness.self_model_tier.autobiography_narrator
                                {
                                    if narrator.should_narrate(episode_count) {
                                        if let Some(ref narrative) =
                                            self.consciousness.self_model_tier.narrative_self
                                        {
                                            narrator.submit(
                                                crate::cognitive_loop::managers::AutobiographyRequest {
                                                    episodes: narrative.autobio.life_story.clone(),
                                                    self_name: "Symthaea".to_string(),
                                                    cycle_num,
                                                },
                                            );
                                        }
                                    }
                                }
                            }

                            // Dream→MCE Narrative coupling: dream insights are
                            // high-integration episodes (counterfactual-validated).
                            // Science: Walker (2009) — sleep-dependent memory consolidation
                            // strengthens autobiographical narrative structure.
                            // Valence: phi improvement is intrinsically positive.
                            self.consciousness
                                .master_equation
                                .narrative_coherence
                                .add_episode(
                                    format!(
                                        "dream_insight_{}_phi{:.3}",
                                        result.insights, result.best_phi_improvement
                                    ),
                                    (result.best_phi_improvement as f64 * 0.5).clamp(-0.5, 0.5),
                                );
                        }
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, cycle = self.stats.total_cycles, "Dream replay failed");
                    }
                }

                // Dream→Knowledge feedback: after a dream cycle, strengthen knowledge
                // graph edges whose topics were recently relevant to the cognitive loop.
                // This closes the loop: knowledge feeds dream salience (above), and
                // dream consolidation quality flows back to reinforce the knowledge graph.
                // Science: Rasch & Born (2013) — targeted memory reactivation during
                // NREM sleep selectively strengthens task-relevant memory traces.
                if dream_phi_improvement > 0.0 {
                    if let Some(ref mut km) = self.memory.knowledge_manager {
                        let topics = km.top_grounded_facts(
                            super::super::thresholds::KNOWLEDGE_EPISODIC_MAX_PER_DREAM,
                        );
                        km.apply_dream_consolidation(&topics, dream_phi_improvement as f64);
                    }
                }

                // Shadow→Dream feedback: record dream Phi improvement back to shadow fragments.
                // Positive improvement = dream found integration path worth pursuing.
                // Science: Hartmann (1998) — dreams create new connections (integration);
                // Barrett (2001) — dream problem-solving correlates with waking insight.
                #[cfg(feature = "therapeutic")]
                if dream_phi_improvement > 0.0 {
                    // Feed improvement to the highest-pressure fragment
                    let frag_count = self.therapeutic_manager.shadow_detector.fragment_count();
                    if frag_count > 0 {
                        self.therapeutic_manager
                            .shadow_detector
                            .record_dream_result(0, dream_phi_improvement);
                    }
                }

                // Threat memory consolidation: decay old patterns, boost recent ones.
                // Threat patterns get +30% salience boost during dream consolidation.
                // Science: Walker (2017) — threat-related memories preferentially
                // consolidated during sleep for survival-relevant pattern retention.
                #[cfg(feature = "sentinel")]
                {
                    self.threat_memory
                        .decay_old_patterns(self.stats.total_cycles as usize);
                }
            }

            dream_wisdom_count = dream.wisdom().len();

            // Feed dream wisdom into DreamFeedbackBridge for context-aware priors.
            // Bridge converts Wisdom → action priors + confidence adjustments keyed
            // by context hash, enabling future cycles to leverage dream discoveries.
            #[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
            for wisdom in dream.wisdom().iter() {
                let context_hash = crate::consciousness::recursive_improvement::hash_context(
                    &wisdom.context_state,
                );
                let insight = crate::consciousness::recursive_improvement::DreamInsight::new(
                    context_hash,
                    wisdom.context_state.clone(), // original action = context state
                    wisdom.better_action.clone(), // alternative action
                    wisdom.phi_improvement as f64,
                );
                self.dream_feedback_bridge.process_insight(insight);
            }

            // Feed therapeutic dream wisdom back into regulation engine.
            // When the dream engine discovers a counterfactual therapeutic action
            // that would have improved Phi, bias future strategy selection accordingly.
            // Science: Barrett (2001) — dreams process emotional concerns, informing
            // waking regulation; Cartwright (2010) — dream rehearsal improves coping.
            #[cfg(feature = "therapeutic")]
            for wisdom in dream.wisdom().iter() {
                // Therapeutic actions encode strategy ordinal at index 9
                if wisdom.better_action.len() > 9 && wisdom.phi_improvement > 0.01 {
                    let strategy_ordinal = wisdom.better_action[9];
                    if strategy_ordinal >= 0.0 && strategy_ordinal <= 6.0 {
                        self.therapeutic_manager
                            .regulation_engine
                            .incorporate_dream_wisdom(
                                strategy_ordinal as u8,
                                wisdom.phi_improvement,
                            );
                    }
                }
            }

            // Apply wisdom: if we have accumulated wisdom, modulate exploration
            // toward states where dream counterfactuals found Phi improvements
            if !dream.wisdom().is_empty() {
                let avg_phi_improvement: f32 = dream
                    .wisdom()
                    .iter()
                    .map(|w| w.phi_improvement)
                    .sum::<f32>()
                    / dream.wisdom().len() as f32;
                // Dream wisdom boosts exploration when Phi improvements are found
                let wisdom_exploration_boost = (avg_phi_improvement * 0.5).clamp(0.0, 0.2);
                self.adjust_exploration("dream_wisdom", wisdom_exploration_boost);

                // FEEDBACK: Dream Phi insights feed forward into waking prediction confidence
                // Science: Prospective consciousness — offline simulation prepares waking cognition.
                // Dream-discovered Phi improvements signal that exploration can yield better states,
                // boosting confidence that the system can navigate toward them.
                if avg_phi_improvement > 0.01 {
                    let dream_confidence_boost = (avg_phi_improvement * 0.1).min(0.05);
                    self.adjust_confidence("dream_phi_insight", dream_confidence_boost);
                }
            }
        }

        // FEEDBACK: Current-cycle dream insights boost learning signal
        // (reinforces pathways that produced the insight)
        if dream_phi_improvement > 0.05 {
            self.fep.learning_signal *= 1.0 + (dream_phi_improvement * 0.2).min(0.15);
        }

        // Dream feedback bridge: adjust prediction confidence based on accumulated
        // dream priors. Context hash from compressed state enables context-specific
        // calibration — contexts where dreams found better alternatives get a boost.
        #[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
        if self.dream_feedback_bridge.num_priors() > 0 {
            let context_hash = crate::consciousness::recursive_improvement::hash_context(
                &compressed_state[..64.min(compressed_state.len())],
            );
            let (adjusted, was_informed) = self
                .dream_feedback_bridge
                .adjust_confidence(self.prediction_confidence, context_hash);
            if was_informed {
                self.set_confidence("dream_feedback", (adjusted as f32).clamp(0.0, 1.0));
            }
            // Decay priors every 199 cycles to forget stale wisdom (co-prime)
            if self.stats.total_cycles % 199 == 0 {
                self.dream_feedback_bridge.decay_priors(0.95);
            }
        }

        // ── Knowledge → Episodic memory bridge: REMOVED (Reconciliation Plan Phase 3,
        // 2026-07-28) ──────────────────────────────────────────────────────────────
        // This used to promote high-confidence facts into `experience::memory::
        // ExperienceRecord` (formerly name-colliding `EpisodicMemory`) during dreams, but that
        // store had zero read consumers anywhere in the live cognitive loop — confirmed via
        // `ExperienceBus::generate_with_experience`'s doc comment, the only reader, having
        // zero callers. Removed rather than left computing-and-discarding real work (BLAKE3-
        // adjacent fact promotion) on every dream cycle where `signals.relevance > 0.3`. The
        // real episodic→semantic consolidation path is "Item 7" immediately below, which reads
        // from the canonical `MemoryCoordinator`/`episodic_replay::EpisodicMemory` system, not
        // this removed block — the two were unrelated despite similar naming.

        // ── Item 7: Episodic → semantic promotion ────────────────────────────
        // When episodic memories survive 3+ dream replays (tracked via coordinator
        // retrieval counts), distill them into semantic knowledge graph facts.
        // Science: Stickgold & Walker (2013) — sleep-dependent memory consolidation;
        //          McClelland et al. (1995) — complementary learning systems theory.
        if let Some(ref mut km) = self.memory.knowledge_manager {
            let top = self
                .memory
                .memory_consol
                .memory_coordinator
                .most_replayed(5);
            for (content_hash, replay_count) in &top {
                if *replay_count >= 3 {
                    let label = format!("episode_0x{:016x}", content_hash);
                    km.process(
                        &format!("Consolidated: {}", label),
                        self.stats.total_cycles as u64,
                    );
                    tracing::trace!(
                        label = %label,
                        replays = replay_count,
                        "Episodic→semantic promotion: consolidated episode into knowledge graph"
                    );
                }
            }
        }

        module_timings.dream_replay = _t.elapsed().as_micros() as u64;

        DreamPhaseResult {
            dream_insights,
            dream_phi_improvement,
            dream_wisdom_count,
        }
    }
}
