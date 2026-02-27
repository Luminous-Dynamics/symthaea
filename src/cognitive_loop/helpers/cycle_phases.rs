//! Extracted cycle phases: resonator codebook, episodic replay, and dream engine.
//!
//! Each method is a self-contained phase of the main cognitive loop, taking only
//! the inputs it needs and returning results via dedicated result structs. All
//! logic and side effects are preserved exactly from the original cycle.rs.

use std::time::Instant;

use super::super::temporal_network::TemporalNetwork;
use super::super::CognitiveLoopService;

// ═══════════════════════════════════════════════════════════════════════════════
// Result structs for extracted cycle phases
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from the resonator codebook growth + high-phi promotion + diversity phase.
pub(in crate::cognitive_loop) struct ResonatorCodebookResult {
    pub resonator_promotions: usize,
    pub codebook_evictions: usize,
    pub codebook_diversity: f32,
    pub codebook_utilization_rate: f32,
}

/// Result from the dream engine phase (recording, dreaming, wisdom application).
pub(in crate::cognitive_loop) struct DreamPhaseResult {
    pub dream_insights: usize,
    pub dream_phi_improvement: f32,
    pub dream_wisdom_count: usize,
}

/// Result from the episodic replay and memory coordinator phase.
pub(in crate::cognitive_loop) struct EpisodicReplayResult {
    pub surprise_replay_batch_size: usize,
    /// Phasic DA burst replay boost (number of extra episodes, 0 if DA < threshold).
    pub phasic_da_replay_boost: usize,
}

/// Result from the hyper-parameter optimization phase.
pub(in crate::cognitive_loop) struct ParameterOptimizationResult {
    pub best_tau_scale: f32,
    pub phi_gain: f64,
    pub swap_occurred: bool,
}

impl CognitiveLoopService {
    /// Autonomous Hyper-Parameter Optimization (The Meta-Forge).
    ///
    /// Periodically explores variations of the brain's internal dynamics
    /// (e.g. time constants) using historical high-Phi episodes.
    pub(in crate::cognitive_loop) fn run_parameter_optimization_phase(
        &mut self,
    ) -> ParameterOptimizationResult {
        let mut result = ParameterOptimizationResult {
            best_tau_scale: 1.0,
            phi_gain: 0.0,
            swap_occurred: false,
        };

        // Only run every 500 cycles to amortize cost
        if self.stats.total_cycles % 500 != 0 {
            return result;
        }

        if let Some(ref replay) = self.phi_episodic_replay {
            let episodes = replay.get_top_episodes(16);
            if episodes.is_empty() {
                return result;
            }

            let baseline_phi = self.simulate_episodes(&episodes, 1.0);
            let mut best_phi = baseline_phi;
            let mut best_scale = 1.0;

            // Simple swarm of candidate tau scales [0.8, 0.9, 1.1, 1.2]
            for scale in [0.8, 0.9, 1.1, 1.2] {
                let candidate_phi = self.simulate_episodes(&episodes, scale);
                if candidate_phi > best_phi {
                    best_phi = candidate_phi;
                    best_scale = scale;
                }
            }

            result.best_tau_scale = best_scale;
            result.phi_gain = best_phi - baseline_phi;

            // If we found an improvement > 5%, hot-swap the live network
            if result.phi_gain > 0.05 {
                self.temporal_network.scale_tau_all(best_scale);
                result.swap_occurred = true;
                tracing::info!(
                    target: "symthaea::forge::optimization",
                    gain = result.phi_gain,
                    new_scale = best_scale,
                    "Brain hot-swapped: Hyper-parameter optimization successful!"
                );
            }
        }

        result
    }

    /// Resonator codebook growth, high-Phi episode promotion, diversity computation,
    /// utilization tracking, and diversity-driven exploration governor.
    ///
    /// Extracted from cycle() -- all logic and behavior preserved exactly.
    pub(in crate::cognitive_loop) fn run_resonator_codebook_phase(
        &mut self,
        epistemic_gate_approved: bool,
        compressed_state: &[f32],
        active_primitive_names: &[String],
        causal_codebook_entries: &[(String, Vec<f32>)],
        reflection_thresholds: &super::super::drives::ReflectionThresholds,
        module_timings: &mut super::super::ModuleTimings,
    ) -> ResonatorCodebookResult {
        // ═══════════════════════════════════════════════════════════════════════
        // RESONATOR CODEBOOK GROWTH: add novel patterns to semantic codebook
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Gate codebook growth on epistemic approval — don't learn from rejected inputs
        if epistemic_gate_approved {
            if let Some(ref mut res_mem) = self.resonator_memory {
                let res_dim_ok = compressed_state.len() == res_mem.resonator.config.dim;
                if res_dim_ok
                    && self.stats.total_cycles % self.config.resonator_growth_interval == 0
                {
                    if let Some(ref mut semantic_cb) = res_mem.resonator.codebooks.get_mut(0) {
                        // Check novelty: max similarity to existing symbols
                        let max_sim = semantic_cb
                            .symbols
                            .iter()
                            .map(|(_, hv)| super::cosine_f32(compressed_state, hv))
                            .fold(0.0f32, f32::max);

                        if max_sim < self.config.resonator_novelty_threshold
                            && semantic_cb.len() < self.config.resonator_max_symbols
                        {
                            semantic_cb.add(
                                &format!("learned_{}", self.stats.total_cycles),
                                compressed_state.to_vec(),
                            );

                            // Track B: Lattice meet for semantic grounding of learned symbol
                            if let Some(ref lattice) = self.primitive_tier.primitive_lattice {
                                if active_primitive_names.len() >= 2 {
                                    if let (Some(a), Some(b)) = (
                                        lattice.element_index_by_name(&active_primitive_names[0]),
                                        lattice.element_index_by_name(&active_primitive_names[1]),
                                    ) {
                                        if let Some(meet_idx) = lattice.meet(a, b) {
                                            let last = semantic_cb.symbols.len() - 1;
                                            semantic_cb.symbols[last].0 = format!(
                                                "learned_{}_{}",
                                                self.stats.total_cycles,
                                                lattice.elements[meet_idx].name
                                            );
                                        }
                                    }
                                }
                            }

                            tracing::trace!(
                                symbols = semantic_cb.len(),
                                max_sim,
                                cycle = self.stats.total_cycles,
                                "Resonator: novel pattern added to semantic codebook"
                            );
                        }
                    }
                }
            }

            // Track A-2: Causal chain content → resonator codebook symbols
            if !causal_codebook_entries.is_empty() {
                if let Some(ref mut res_mem) = self.resonator_memory {
                    for (label, hv) in causal_codebook_entries {
                        if let Some(ref mut semantic_cb) = res_mem.resonator.codebooks.get_mut(0) {
                            if semantic_cb.len() < self.config.resonator_max_symbols
                                && hv.len() == res_mem.resonator.config.dim
                            {
                                semantic_cb.add(label, hv.clone());
                            }
                        }
                    }
                }
            }
        } // end epistemic_gate_approved guard for codebook growth

        module_timings.resonator_codebook = _t.elapsed().as_micros() as u64;

        // Track 3c: High-Phi episodes → resonator codebook promotion
        // Science: Dehaene (2014) — conscious access creates durable representations
        // Co-prime cadence (97 cycles) avoids interference with other periodic tasks
        let _t = Instant::now();
        let mut resonator_promotions: usize = 0;
        let mut codebook_evictions: usize = 0;
        if self.stats.total_cycles % 97 == 0 && self.stats.total_cycles > 0 {
            let top_eps = self
                .phi_episodic_replay
                .as_ref()
                .map(|replay| replay.get_top_episodes(3))
                .unwrap_or_default();

            if !top_eps.is_empty() {
                if let Some(ref mut res_mem) = self.resonator_memory {
                    let dim = res_mem.resonator.config.dim;
                    if let Some(ref mut semantic_cb) = res_mem.resonator.codebooks.get_mut(0) {
                        for ep in &top_eps {
                            if ep.psi > 0.5 {
                                let ep_vec = &ep.input.values;
                                if ep_vec.len() != dim {
                                    continue;
                                }

                                // Track 3c-evict: Prune most redundant entry when at capacity
                                // Science: competitive learning — maintain codebook diversity
                                if semantic_cb.len() >= self.config.resonator_max_symbols
                                    && semantic_cb.len() > 1
                                {
                                    let n = semantic_cb.symbols.len();
                                    let mut max_redundancy = f32::MIN;
                                    let mut evict_idx = 0;
                                    for i in 0..n {
                                        let avg_sim: f32 = (0..n)
                                            .filter(|&j| j != i)
                                            .map(|j| {
                                                super::cosine_f32(
                                                    &semantic_cb.symbols[i].1,
                                                    &semantic_cb.symbols[j].1,
                                                )
                                            })
                                            .sum::<f32>()
                                            / (n - 1) as f32;
                                        if avg_sim > max_redundancy {
                                            max_redundancy = avg_sim;
                                            evict_idx = i;
                                        }
                                    }
                                    semantic_cb.symbols.remove(evict_idx);
                                    codebook_evictions += 1;
                                }

                                if semantic_cb.len() < self.config.resonator_max_symbols {
                                    semantic_cb.add(
                                        &format!("phi_{:.0}_{}", ep.psi * 100.0, ep.timestamp),
                                        ep_vec.clone(),
                                    );
                                    resonator_promotions += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        module_timings.high_phi_promotion = _t.elapsed().as_micros() as u64;

        // Track 3e: Codebook diversity metric
        // Science: competitive learning — low diversity = redundant representations
        // Compute average pairwise cosine distance (every 50 cycles to amortize cost)
        let codebook_diversity: f32 = if self.stats.total_cycles % 50 == 0 {
            if let Some(ref res_mem) = self.resonator_memory {
                if let Some(semantic_cb) = res_mem.resonator.codebooks.first() {
                    let n = semantic_cb.symbols.len();
                    if n >= 2 {
                        let mut total_dist = 0.0f32;
                        let mut pairs = 0u32;
                        for i in 0..n {
                            for j in (i + 1)..n {
                                let sim = super::cosine_f32(
                                    &semantic_cb.symbols[i].1,
                                    &semantic_cb.symbols[j].1,
                                );
                                total_dist += 1.0 - sim; // distance = 1 - similarity
                                pairs += 1;
                            }
                        }
                        if pairs > 0 {
                            total_dist / pairs as f32
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
            }
        } else {
            self.stats.codebook_diversity // carry forward cached value
        };

        // ── Track 5d: Codebook utilization rate ─────────────────────────────
        // Science: Kohonen (1982) — self-organizing maps need active symbol usage
        // Compute fraction of codebook symbols that match recent input (similarity > 0.2).
        // Low utilization → too many dead symbols → slow codebook growth.
        let codebook_utilization_rate: f32 = if self.stats.total_cycles % 50 == 0 {
            if let Some(ref res_mem) = self.resonator_memory {
                if let Some(semantic_cb) = res_mem.resonator.codebooks.first() {
                    let n = semantic_cb.symbols.len();
                    if n > 0 && compressed_state.len() == res_mem.resonator.config.dim {
                        let utilized = semantic_cb
                            .symbols
                            .iter()
                            .filter(|(_, hv)| super::cosine_f32(compressed_state, hv) > 0.2)
                            .count();
                        let rate = utilized as f32 / n as f32;
                        // EMA update
                        self.stats.codebook_utilization_rate =
                            self.stats.codebook_utilization_rate * 0.8 + rate * 0.2;
                        // Low utilization → increase novelty threshold (harder to add)
                        if rate < 0.2 && self.config.resonator_novelty_threshold < 0.9 {
                            self.config.resonator_novelty_threshold =
                                (self.config.resonator_novelty_threshold + 0.02).min(0.9);
                        } else if rate > 0.6 && self.config.resonator_novelty_threshold > 0.3 {
                            // High utilization → lower novelty threshold (easier to add)
                            self.config.resonator_novelty_threshold =
                                (self.config.resonator_novelty_threshold - 0.01).max(0.3);
                        }
                        rate
                    } else {
                        self.stats.codebook_utilization_rate
                    }
                } else {
                    self.stats.codebook_utilization_rate
                }
            } else {
                0.0
            }
        } else {
            self.stats.codebook_utilization_rate
        };

        // Track 3f: Codebook diversity → exploration governor
        // Science: competitive learning — low diversity signals representational collapse
        // Low diversity → boost exploration urge (seek novel inputs)
        // High diversity → allow exploitation (good codebook coverage)
        let div_low = reflection_thresholds.diversity_low;
        let div_high = reflection_thresholds.diversity_high;
        if codebook_diversity > 0.0 {
            if codebook_diversity < div_low {
                // Representational collapse risk — boost exploration
                let diversity_boost = (div_low - codebook_diversity) * 0.2;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + diversity_boost).clamp(0.0, 1.0);
            } else if codebook_diversity > div_high {
                // Good coverage — allow exploitation, dampen exploration slightly
                let exploit_dampen = (codebook_diversity - div_high) * 0.1;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge - exploit_dampen).clamp(0.0, 1.0);
            }
        }

        ResonatorCodebookResult {
            resonator_promotions,
            codebook_evictions,
            codebook_diversity,
            codebook_utilization_rate,
        }
    }

    /// Episodic replay session: demand-driven consolidation triggers, replay with
    /// surprise-boosted batch sizes, resonator factorization, adaptive scheduling,
    /// and memory coordinator graduation.
    ///
    /// Extracted from cycle() -- all logic and behavior preserved exactly.
    pub(in crate::cognitive_loop) fn run_episodic_replay_and_memory_phase(
        &mut self,
        state: &super::super::CycleState<'_>,
        memory_context_boost: f32,
        fep_surprise: f64,
        surprise_thresh: f64,
        module_timings: &mut super::super::ModuleTimings,
    ) -> EpisodicReplayResult {
        let prediction_error = state.prediction_error;
        let coherence = state.coherence;
        let compressed_state = state.compressed_state;
        let output = state.output;
        let mut surprise_replay_batch_size: usize = 0;
        let mut phasic_da_replay_boost: usize = 0;

        // ═══════════════════════════════════════════════════════════════════════
        // DEMAND-DRIVEN CONSOLIDATION TRIGGERS
        // ═══════════════════════════════════════════════════════════════════════
        // Trigger early episodic replay when:
        //   (a) prediction error spikes >2x the moving average, or
        //   (b) semantic memory returned zero hits (retrieval miss)
        // The periodic 100-cycle floor is still enforced by should_replay().
        let _t = Instant::now();
        if let Some(ref mut replay) = self.phi_episodic_replay {
            let avg_err = self.stats.avg_prediction_error;
            let error_spike = avg_err > 0.01 && prediction_error > avg_err * 2.0;
            let semantic_miss = self.semantic_memory.stats().semantic_misses > 0
                && memory_context_boost == 0.0 // no episodic memories recalled this cycle
                && self.stats.total_cycles > 10;

            if error_spike || semantic_miss {
                replay.trigger_demand_replay();
                tracing::trace!(
                    error_spike,
                    semantic_miss,
                    cycle = self.stats.total_cycles,
                    "Demand-driven consolidation triggered"
                );
            }
        }

        module_timings.demand_consolidation = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // SEQUENTIAL: Episodic replay + Memory coordinator
        // ═══════════════════════════════════════════════════════════════════════
        // These remain sequential because:
        // - Episodic replay needs &mut temporal_network for CfC retraining
        // - Memory coordinator needs &mut phi_episodic_replay after replay completes
        let _t = Instant::now();
        if let Some(ref mut replay) = self.phi_episodic_replay {
            let coherence_summary = self.coherence_bridge.summary();
            let current_phi = coherence_summary.smoothed_coherence as f64;

            let input_hv =
                symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(compressed_state.to_vec());
            let output_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(output.to_vec());

            let episode = crate::memory::episodic_replay::Episode::with_metadata(
                input_hv,
                output_hv,
                current_phi,
                self.stats.total_cycles as u64,
                prediction_error,
                self.emotion_contagion.smoothed_valence(),
                coherence_summary.coherence,
            )
            .with_dopamine(self.neuromodulator_bath.dopamine.effective());

            let stored = replay.store_if_significant(episode);
            if stored {
                tracing::trace!(
                    phi = current_phi,
                    cycle = self.stats.total_cycles,
                    "High-Phi episode stored for replay"
                );
            }

            if replay.should_replay() {
                // ── Track 5f: FEP surprise → replay batch size modulation ────────
                // Science: Mnih et al. (2015) — prioritized experience replay:
                // high surprise = high learning potential → replay more episodes
                let base_batch = replay.config.batch_size;
                let surprise_batch_boost = if fep_surprise > surprise_thresh {
                    // High surprise → up to 2x batch size
                    let boost_factor =
                        ((fep_surprise - surprise_thresh) / surprise_thresh).min(1.0) as f32;
                    (base_batch as f32 * boost_factor).round() as usize
                } else {
                    0
                };
                // DA-tagged sleep consolidation: Night phase → bigger replay batches
                // Science: Walker & Stickgold (2006) — DA-tagged memories consolidate during sleep
                let sleep_boost = if self.biorhythm.phase == crate::chronobiology::CircadianPhase::Night {
                    let factor = self.neuromodulator_bath.sleep_consolidation_boost();
                    (base_batch as f32 * (factor - 1.0)).round() as usize
                } else {
                    0
                };
                // #2: Phasic DA burst → replay amplification (Lisman & Grace 2005)
                let phasic_da_boost = {
                    let da_ph = self.neuromodulator_bath.da_phasic();
                    if da_ph > 0.3 {
                        ((da_ph - 0.3) * base_batch as f32 * 1.5).round() as usize
                    } else {
                        0
                    }
                };
                phasic_da_replay_boost = phasic_da_boost;
                let boosted_batch = base_batch + surprise_batch_boost + sleep_boost + phasic_da_boost;
                // Temporarily set boosted batch size for this replay session
                let original_batch = replay.config.batch_size;
                replay.config.batch_size = boosted_batch;
                surprise_replay_batch_size = boosted_batch;

                if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                    let learning_rate = self.config.cfc_config.learning_rate;
                    let result = replay.replay_session(cfc, learning_rate);

                    if !result.skipped {
                        tracing::debug!(
                            episodes = result.episodes_replayed,
                            avg_loss = result.average_loss,
                            avg_psi = result.average_psi,
                            "Episodic replay session completed"
                        );

                        // Track 3g: Dream consolidation — resonator factorization of replayed episodes
                        // Science: Stickgold (2005) — sleep replay extracts gist representations
                        // After episodic replay, factorize top episodes through the resonator to
                        // extract clean semantic components and strengthen codebook representations.
                        if let Some(ref mut res_mem) = self.resonator_memory {
                            if !res_mem.resonator.codebooks.is_empty() {
                                let res_dim = res_mem.resonator.config.dim;
                                let top_eps = replay.get_top_episodes(3);
                                for ep in &top_eps {
                                    // Project episode input down to resonator dim
                                    let ep_vals = &ep.input.values;
                                    if ep_vals.len() >= res_dim {
                                        let projected: Vec<f32> =
                                            ep_vals.iter().take(res_dim).copied().collect();
                                        if let Ok(factors) = res_mem.resonator.factorize(&projected)
                                        {
                                            // Each factor strengthens its codebook entry via re-exposure
                                            // This is the "gist extraction" — dreaming distills episodes
                                            // into their categorical components
                                            for (label, _factor_hv) in &factors {
                                                tracing::trace!(
                                                    label,
                                                    psi = ep.psi,
                                                    "Dream factorized episode component"
                                                );
                                            }
                                            let _ = factors.len(); // factorization itself updates resonator state
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Restore original batch size after replay session
                replay.config.batch_size = original_batch;
                if surprise_batch_boost > 0 {
                    self.stats.surprise_boosted_replays += 1;
                }
            }
        }

        // Track 4d: Adaptive replay scheduling — modulate interval based on error volatility
        // Science: McClelland et al. (1995) — fast-changing environments need more replay
        if self.stats.total_cycles % 50 == 0 && self.stats.total_cycles > 50 {
            if let Some(ref mut replay) = self.phi_episodic_replay {
                // Variance = E[X²] - E[X]² (from EMA-tracked moments)
                let error_variance = (self.stats.avg_prediction_error_sq
                    - self.stats.avg_prediction_error * self.stats.avg_prediction_error)
                    .max(0.0);
                replay.adapt_replay_interval(error_variance);
            }
        }

        // Memory coordinator: broadcast signals and process graduations
        {
            let coord_phi = self.coherence_bridge.smoothed_coherence() as f64;
            let coord_coherence = coherence as f64;
            self.memory_coordinator.update_signals_with_sigma(
                coord_phi,
                coord_coherence,
                self.carryover.consciousness.last_sigma,
            );

            if let Some(ref mut replay) = self.phi_episodic_replay {
                let graduated = self.memory_coordinator.process_graduations(replay);
                if graduated > 0 {
                    tracing::debug!(
                        graduated,
                        "Memory coordinator graduated items to episodic storage"
                    );
                }
            }
        }

        module_timings.episodic_replay = _t.elapsed().as_micros() as u64;

        EpisodicReplayResult {
            surprise_replay_batch_size,
            phasic_da_replay_boost,
        }
    }

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
                .narrative_self
                .as_ref()
                .map(|n| 1.0 + n.self_phi() as f32 * 0.5) // 1.0 to 1.5x boost
                .unwrap_or(1.0);
            let phi_weighted_surprise =
                prediction_error * (1.0 + unified_psi as f32).clamp(1.0, 2.0) * narrative_salience;
            dream.record(
                &dream_state,
                dream_action,
                &dream_outcome,
                phi_weighted_surprise,
            );

            // Dream during Cruise urgency (low-error steady state) or every 20th cycle
            if matches!(urgency, super::super::CycleUrgency::Cruise)
                || urgency.should_run(self.stats.total_cycles, 10, 20, 5)
            {
                if let Ok(result) = dream.dream() {
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
                        if let Some(ref mut narrative) = self.narrative_self {
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
                self.curiosity_drive.exploration_urge = (self.curiosity_drive.exploration_urge
                    + wisdom_exploration_boost)
                    .clamp(0.0, 1.0);

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
            self.fep_learning_signal *= 1.0 + (dream_phi_improvement * 0.2).min(0.15);
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
                .adjust_confidence(self.prediction_confidence as f64, context_hash);
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
