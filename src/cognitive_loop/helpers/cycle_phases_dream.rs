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
                .self_model_tier
                .narrative_self
                .as_ref()
                .map(
                    |n: &crate::consciousness::narrative_self::NarrativeSelfModel| {
                        1.0 + n.self_phi() as f32 * 0.5
                    },
                ) // 1.0 to 1.5x boost
                .unwrap_or(1.0);
            let phi_weighted_surprise =
                prediction_error * (1.0 + unified_psi as f32).clamp(1.0, 2.0) * narrative_salience;
            // Scene recognition boost: recognized visual contexts encode preferentially.
            // Conway (2005) — self-relevant and context-rich memories encode preferentially.
            #[cfg(feature = "vision-manifold")]
            let phi_weighted_surprise = if state.scene_recognized {
                phi_weighted_surprise
                    * super::super::thresholds::SCENE_RECOGNITION_DREAM_BOOST
            } else {
                phi_weighted_surprise
            };
            dream.record(
                &dream_state,
                dream_action,
                &dream_outcome,
                phi_weighted_surprise,
            );

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
                let t = (combined_pressure - 0.4) / 0.3;
                20 - (t * 15.0) as usize
            } else {
                20 // base rate
            };
            if matches!(urgency, super::super::CycleUrgency::Cruise)
                || urgency.should_run(
                    self.stats.total_cycles,
                    10,
                    dynamic_normal_interval,
                    5,
                )
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
                            if let Some(ref mut narrative) = self.self_model_tier.narrative_self {
                                narrative.process_experience(
                                    hv16_cached,
                                    &format!("dream_insight_{}", result.insights),
                                    true, // counterfactual-validated
                                    unified_psi,
                                    result.best_phi_improvement as f64,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, cycle = self.stats.total_cycles, "Dream replay failed");
                    }
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

        module_timings.dream_replay = _t.elapsed().as_micros() as u64;

        DreamPhaseResult {
            dream_insights,
            dream_phi_improvement,
            dream_wisdom_count,
        }
    }
}
