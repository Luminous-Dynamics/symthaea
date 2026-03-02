//! Extracted cycle phases: resonator codebook, episodic replay, dream engine,
//! urgency computation, init/preprocessing, and end-of-cycle processing.
//!
//! Each method is a self-contained phase of the main cognitive loop, taking only
//! the inputs it needs and returning results via dedicated result structs. All
//! logic and side effects are preserved exactly from the original cycle.rs.

use std::time::Instant;

use super::super::neuromodulators::NeuromodulatorBathExt;
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

/// Result from the urgency computation and error pattern analysis phase.
pub(in crate::cognitive_loop) struct UrgencyResult {
    pub urgency: super::super::CycleUrgency,
    pub error_pattern: &'static str,
    pub predicted_urgency: &'static str,
    pub prediction_coherence_urgency_bias: f32,
}

/// Result from the cycle init and preprocessing phase.
pub(in crate::cognitive_loop) struct CycleInitResult {
    pub exploration_urge_start: f32,
    pub startup_suppressed: bool,
    pub startup_warmup_progress: f32,
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
                self.adjust_exploration("codebook_collapse", diversity_boost);
            } else if codebook_diversity > div_high {
                // Good coverage — allow exploitation, dampen exploration slightly
                let exploit_dampen = (codebook_diversity - div_high) * 0.1;
                self.adjust_exploration("codebook_stable", -exploit_dampen);
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
            .with_dopamine(self.neuromod.bath.dopamine.effective())
            .with_bath_state(self.neuromod.bath.state_vector());

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
                let base_batch = replay.batch_size();
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
                let sleep_boost =
                    if self.biorhythm.phase == crate::chronobiology::CircadianPhase::Night {
                        let factor = self.neuromod.bath.sleep_consolidation_boost();
                        (base_batch as f32 * (factor - 1.0)).round() as usize
                    } else {
                        0
                    };
                // #2: Phasic DA burst → replay amplification (Lisman & Grace 2005)
                let phasic_da_boost = {
                    let da_ph = self.neuromod.bath.da_phasic();
                    if da_ph > 0.3 {
                        ((da_ph - 0.3) * base_batch as f32 * 1.5).round() as usize
                    } else {
                        0
                    }
                };
                phasic_da_replay_boost = phasic_da_boost;
                let boosted_batch =
                    base_batch + surprise_batch_boost + sleep_boost + phasic_da_boost;
                // Temporarily set boosted batch size for this replay session
                let original_batch = replay.batch_size();
                replay.set_batch_size(boosted_batch);
                surprise_replay_batch_size = boosted_batch;

                if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                    let learning_rate = self.config.cfc_config.learning_rate;
                    // State-dependent replay: prioritize episodes encoded in similar bath state
                    // Science: Godden & Baddeley (1975) — state-dependent memory
                    let current_bath = Some(self.neuromod.bath.state_vector());
                    let result =
                        replay.replay_session_conditioned(cfc, learning_rate, current_bath);

                    if !result.skipped {
                        tracing::debug!(
                            episodes = result.episodes_replayed,
                            avg_loss = result.average_loss,
                            avg_psi = result.average_psi,
                            "Episodic replay session completed"
                        );

                        // Record retrievals for consolidation tracking
                        // Science: Nader et al. (2000) — retrieval triggers reconsolidation
                        {
                            let top_eps_for_tracking =
                                replay.get_top_episodes(result.episodes_replayed.min(10));
                            for ep in &top_eps_for_tracking {
                                let hash =
                                    crate::memory::memory_coordinator::content_hash(&ep.input);
                                self.memory_coordinator.record_retrieval(hash);
                            }
                        }

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
                replay.set_batch_size(original_batch);
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

    // ═════════════════════════════════════════════════════════════════════════
    // Urgency computation + error pattern analysis
    // Extracted from cycle.rs lines 472-684 (zero behavioral change).
    // ═════════════════════════════════════════════════════════════════════════

    /// Compute urgency mode via adaptive threshold, hysteresis, error pattern
    /// analysis, and mode transition smoothing.
    ///
    /// Mutates: `self.carryover.urgency`, `self.carryover.history.error_history`,
    /// `self.carryover.learning.adaptive_threshold_scale`, `self.stats`,
    /// `self.neuromod.bath`.
    pub(in crate::cognitive_loop) fn compute_urgency_and_error_pattern(
        &mut self,
        prediction_error: f32,
        surprise_triggered: bool,
        effective_threshold: f32,
    ) -> UrgencyResult {
        // Track consecutive low-error cycles for Cruise eligibility
        if prediction_error < effective_threshold {
            self.carryover.urgency.consecutive_low_error = self
                .carryover
                .urgency
                .consecutive_low_error
                .saturating_add(1);
        } else {
            self.carryover.urgency.consecutive_low_error = 0;
        }

        // Use smoothed error for urgency to prevent jitter from single-cycle noise spikes.
        // Science: Dynamical systems — threshold-based switching needs hysteresis to prevent
        // oscillation. EMA smoothing damps transient spikes; prev_urgency adds hysteresis.
        let smoothed_urgency_error = if self.stats.total_cycles > 5 {
            // Blend instantaneous (70%) with running average (30%) for responsiveness + smoothing
            prediction_error * 0.7 + self.stats.avg_prediction_error * 0.3
        } else {
            prediction_error // Use raw error during bootstrap
        };

        // Hysteresis: require stronger signal to LEAVE current urgency level
        // #1: D2-mediated behavioral flexibility gates mode transitions (Frank 2005).
        // High D2 → easier transitions (lower hysteresis), low D2 → perseveration.
        let flexibility = self.neuromod.bath.behavioral_flexibility();
        let flex_mod = 1.0 / flexibility; // 0.67–1.43 (inverted: high flex = lower threshold)
        let base_hysteresis = match self.carryover.urgency.urgency {
            super::super::CycleUrgency::Cruise => effective_threshold * 1.2 * flex_mod,
            super::super::CycleUrgency::Critical => effective_threshold * 0.8 * flex_mod,
            _ => effective_threshold * flex_mod,
        };

        // ── Phase 17: Predictive interval tuning via error pattern ──────
        // Science: Clark (2013) — predictive brain anticipates state changes.
        // Rising error pattern → lower threshold (prepare to escalate).
        // Falling error pattern → raise threshold (allow settling).
        let error_history_len = self.carryover.history.error_history.len();
        let pattern_mod = if error_history_len >= 4 {
            // Direct index: newest = len-1, 4th-newest = len-4 (avoids Vec alloc)
            let newest = self.carryover.history.error_history[error_history_len - 1];
            let oldest_4 = self.carryover.history.error_history[error_history_len - 4];
            let slope = (newest - oldest_4) / 3.0;
            if slope > 0.02 {
                0.9f32
            }
            // Rising → easier to escalate
            else if slope < -0.02 {
                1.1
            }
            // Falling → easier to de-escalate
            else {
                1.0
            }
        } else {
            1.0
        };

        // ── Phase 18: Prediction coherence → urgency bias ─────────────────
        // Science: Bar (2009) — temporal prediction consistency signals model quality.
        // Uses previous cycle's avg coherence (current not yet computed at urgency time).
        // Low coherence (<0.3) → model confused across horizons → bias toward Critical.
        // High coherence (>0.7) → model confident → permit Cruise (raise threshold).
        let prev_coherence = self.stats.avg_prediction_coherence;
        let coherence_mod = if prev_coherence < 0.3 && prev_coherence > 0.0 {
            0.85f32 // Lower threshold → easier to escalate (model confused)
        } else if prev_coherence > 0.7 {
            1.1 // Raise threshold → permit Cruise (model confident)
        } else {
            1.0
        };
        let prediction_coherence_urgency_bias = coherence_mod - 1.0;

        let hysteresis_threshold = base_hysteresis * pattern_mod * coherence_mod;
        let error_urgency = super::super::CycleUrgency::from_state(
            smoothed_urgency_error,
            hysteresis_threshold,
            surprise_triggered,
            self.carryover.urgency.consecutive_low_error,
        );

        // Compose CognitiveDepth with error-based urgency:
        // Reflex → cap at Cruise (skip heavy subsystems for familiar inputs)
        // DeepThought → floor at Normal (force full processing for novel/high-stakes)
        // Cortical → use error-based urgency as-is
        let raw_urgency = match self.cognitive_depth {
            super::super::CognitiveDepth::Reflex => match error_urgency {
                super::super::CycleUrgency::Critical => super::super::CycleUrgency::Normal,
                _ => super::super::CycleUrgency::Cruise,
            },
            super::super::CognitiveDepth::DeepThought => match error_urgency {
                super::super::CycleUrgency::Cruise => super::super::CycleUrgency::Normal,
                _ => error_urgency,
            },
            super::super::CognitiveDepth::Cortical => error_urgency,
        };

        // ── Phase 17: Cross-temporal error pattern learning ──────────────
        // Science: Rao & Ballard (1999) — hierarchical predictive coding tracks error
        // trajectories across time, not just instantaneous snapshots.
        // Maintain rolling window of prediction errors, classify pattern.
        let error_history = &mut self.carryover.history.error_history;
        while error_history.len() >= 16 {
            error_history.pop_front();
        }
        error_history.push_back(prediction_error);

        let (error_pattern, predicted_urgency) = if error_history.len() >= 4 {
            let len = error_history.len();
            // Direct index: newest = len-1, 4th-newest = len-4 (avoids Vec alloc)
            let newest = error_history[len - 1];
            let oldest_of_4 = error_history[len - 4];
            // Compute linear trend (simple slope)
            let slope = (newest - oldest_of_4) / 3.0; // newest - oldest, normalized
                                                      // Count sign changes for oscillation detection (index pairs avoid collect→Vec)
            let mut sign_changes = 0u32;
            let ref_val = oldest_of_4;
            for i in 0..len.saturating_sub(1) {
                let diff_cur = error_history[i + 1] - error_history[i];
                let diff_ref = error_history[i] - ref_val;
                if diff_cur.signum() != diff_ref.signum() {
                    sign_changes += 1;
                }
            }
            let oscillation_ratio = if len > 2 {
                sign_changes as f32 / (len - 1) as f32
            } else {
                0.0
            };
            // Spike detection: current error > 2× running mean
            let mean_err = error_history.iter().sum::<f32>() / len as f32;
            let is_spike = prediction_error > mean_err * 2.0 && prediction_error > 0.1;

            let pattern = if is_spike {
                "Spike"
            } else if oscillation_ratio > 0.6 {
                "Oscillating"
            } else if slope > 0.02 {
                "Rising"
            } else if slope < -0.02 {
                "Falling"
            } else {
                "Stable"
            };
            // Predict urgency 5 cycles ahead from pattern
            let predicted = match pattern {
                "Rising" | "Spike" => "Critical",
                "Oscillating" => "Normal",
                "Falling" | "Stable" => {
                    if self.carryover.urgency.consecutive_low_error > 15 {
                        "Cruise"
                    } else {
                        "Normal"
                    }
                }
                _ => "Normal",
            };
            (pattern, predicted)
        } else {
            ("Warmup", "Normal")
        };

        // #9: Error trend → DA baseline modulation (Schultz 2016)
        self.neuromod.bath.modulate_from_error_trend(error_pattern);

        // ── Phase 17: Mode transition smoothing ──────────────────────────
        // Science: Kelso (1995) — metastable coordination dynamics: transitions between
        // attractor states should be smooth, not abrupt. Ramp mode_confidence over 5 cycles.
        let urgency;
        if raw_urgency != self.carryover.urgency.prev_urgency {
            // Mode changed — start transition
            self.stats.mode_transitions += 1;
            self.carryover.urgency.mode_confidence = 0.0;
            self.carryover.urgency.mode_stability_counter = 0;
            // During transition, stay in the HIGHER urgency (more cautious)
            let raw_level = match raw_urgency {
                super::super::CycleUrgency::Critical => 2,
                super::super::CycleUrgency::Normal => 1,
                super::super::CycleUrgency::Cruise => 0,
            };
            let prev_level = match self.carryover.urgency.prev_urgency {
                super::super::CycleUrgency::Critical => 2,
                super::super::CycleUrgency::Normal => 1,
                super::super::CycleUrgency::Cruise => 0,
            };
            urgency = if raw_level > prev_level {
                raw_urgency // escalating → use new immediately
            } else {
                // de-escalating → hold old urgency for 1 cycle
                self.carryover.urgency.prev_urgency
            };
            self.carryover.urgency.prev_urgency = raw_urgency;
        } else {
            // Same mode — ramp confidence
            self.carryover.urgency.mode_stability_counter = self
                .carryover
                .urgency
                .mode_stability_counter
                .saturating_add(1);
            self.carryover.urgency.mode_confidence =
                (self.carryover.urgency.mode_stability_counter as f32 / 5.0).min(1.0);
            urgency = raw_urgency;
        }
        self.stats.avg_mode_stability = self.stats.avg_mode_stability * 0.9
            + self.carryover.urgency.mode_stability_counter as f32 * 0.1;

        UrgencyResult {
            urgency,
            error_pattern,
            predicted_urgency,
            prediction_coherence_urgency_bias,
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Cycle init and preprocessing
    // Extracted from cycle.rs lines 100-216 (zero behavioral change).
    // ═════════════════════════════════════════════════════════════════════════

    /// Startup transient suppression, biorhythm refresh, nociception, and
    /// neuromodulator bath update. Run at the very start of each cycle.
    ///
    /// Mutates: `self.stats`, `self.curiosity_drive`, `self.carryover`,
    /// `self.feedback_state`, `self.subsystem_collector`, `self.biorhythm`,
    /// `self.neuromod.bath`, `self.somatic_bridge`, `self.emotion_contagion`,
    /// `self.thermodynamic_load`, `self.neuromod.phase_tracker`,
    /// `self.neuromod.drift_tracker`.
    pub(in crate::cognitive_loop) fn run_cycle_init(
        &mut self,
        module_timings: &mut super::super::ModuleTimings,
    ) -> CycleInitResult {
        // ── Phase 17: Startup transient suppression ─────────────────────────
        // Science: Hopfield (1982) — recurrent networks require settling time before
        // producing reliable dynamics. During warmup (cycles 0–50), suppress learning
        // rate and curiosity to prevent cementing transient noise as learned patterns.
        let startup_warmup_cycles = super::super::thresholds::STARTUP_WARMUP_CYCLES;
        let startup_suppressed = self.stats.total_cycles <= startup_warmup_cycles;
        let startup_warmup_progress = if startup_suppressed {
            self.stats.total_cycles as f32 / startup_warmup_cycles as f32
        } else {
            1.0
        };
        if startup_suppressed {
            self.stats.startup_suppressed_cycles += 1;
            // Ramp learning rate from 20% → 100% over warmup period
            let lr_scale = 0.2 + 0.8 * startup_warmup_progress;
            self.stats.adaptive_learning_rate *= lr_scale;
            // Suppress curiosity during transient (let CfC settle)
            self.scale_exploration("startup_warmup", startup_warmup_progress);
        }

        // Snapshot exploration_urge for end-of-cycle budget clamping (Task B)
        let exploration_urge_start = self.curiosity_drive.exploration_urge as f32;

        // Snapshot confidence for end-of-cycle drift clamping (Task G)
        self.carryover.learning.prediction_confidence = self.prediction_confidence;

        // ── Phase 2.2: Begin feedback proposal collection for this cycle ────
        self.feedback_state.begin_cycle();

        // Apply consensus overrides from the previous cycle as Set proposals,
        // syncing actual fields so both direct-mutation and proposal paths
        // start from the same base value.
        {
            let (consensus_conf, consensus_lr, consensus_explore, consensus_threshold) =
                self.feedback_state.apply_pending_consensus();
            if let Some(conf) = consensus_conf {
                self.set_confidence("consensus_writeback", conf as f32);
            }
            if let Some(lr) = consensus_lr {
                self.set_lr("consensus_writeback", lr as f32);
            }
            if let Some(explore) = consensus_explore {
                self.set_exploration("consensus_writeback", explore as f32);
            }
            if let Some(thresh) = consensus_threshold {
                self.set_threshold("consensus_writeback", thresh as f32);
            }
        }

        self.feedback_state.snapshot_cycle_start(
            self.prediction_confidence,
            self.fep_lr_boost,
            self.curiosity_drive.exploration_urge,
            self.carryover.learning.adaptive_threshold_scale,
        );
        // ── Phase 2.3: Clear subsystem output collector ────
        self.subsystem_collector.clear();

        // Chronobiology: refresh biorhythm every 97 cycles (co-prime amortization)
        self.biorhythm_refresh_counter += 1;
        if self.biorhythm_refresh_counter >= super::super::thresholds::BIORHYTHM_INTERVAL {
            self.biorhythm = crate::chronobiology::Biorhythm::current();
            // #14: Use effective_hour (with phase offset) for circadian modulation
            let effective_hour = self.biorhythm.effective_hour();
            self.neuromod
                .bath
                .modulate_circadian_continuous(effective_hour);
            // #14: Entrain phase offset toward zero each refresh
            self.biorhythm.entrain();
            // Record personality profile for drift detection
            let profile = self.neuromod.bath.personality_profile();
            self.neuromod.drift_tracker.record(&profile);
            // #4: Personality drift → anomaly recovery (Turrigiano 2008)
            if self.neuromod.drift_tracker.is_anomalous()
                && self.carryover.urgency.anomaly_drift_recovery == 0
            {
                self.neuromod.bath.engage_anomaly_recovery();
                self.carryover.urgency.anomaly_drift_recovery = 50;
            }
            self.biorhythm_refresh_counter = 0;
        }
        // #4: Countdown and disengage drift recovery
        if self.carryover.urgency.anomaly_drift_recovery > 0 {
            self.carryover.urgency.anomaly_drift_recovery -= 1;
            if self.carryover.urgency.anomaly_drift_recovery == 0 {
                self.neuromod.bath.disengage_anomaly_recovery();
            }
        }
        // ── Sleep→Wake transition: apply sleep recovery (Xie et al. 2013) ──
        {
            let is_sleep_now = self.biorhythm.phase == crate::chronobiology::CircadianPhase::Night;
            if self.neuromod.was_sleeping && !is_sleep_now {
                let quality =
                    (self.neuromod.bath.allostatic_recovery_cycles as f32 / 100.0).clamp(0.0, 1.0);
                self.neuromod.bath.apply_sleep_recovery(quality);

                // ── Psych-bench calibration: receptor sensitivity tuning ──
                // Apply any pending calibration during sleep→wake, mirroring
                // synaptic homeostasis (Tononi & Cirelli 2006): receptor
                // sensitivities adjust during sleep to correct performance drift.
                //
                // Gate: require minimum sleep duration for calibration to take effect.
                // Too-short sleep doesn't provide enough homeostatic consolidation.
                // Science: Tononi & Cirelli (2006) — synaptic homeostasis hypothesis.
                const MIN_SLEEP_FOR_CALIBRATION: u32 = 50; // ~1s at 50Hz
                let recovery_cycles = self.neuromod.bath.allostatic_recovery_cycles;
                if self.neuromod.pending_calibration.is_some() {
                    if recovery_cycles >= MIN_SLEEP_FOR_CALIBRATION {
                        self.apply_pending_calibration();
                        // Reset self-assessment cooldown: external calibration supersedes
                        self.neuromod.self_assessment.reset_after_calibration();
                    } else {
                        tracing::warn!(
                            recovery_cycles,
                            min = MIN_SLEEP_FOR_CALIBRATION,
                            "Sleep too short — deferring calibration to next sleep→wake"
                        );
                        // pending_calibration kept for next sleep→wake
                    }
                }
            }

            // ── Wake→Sleep transition: optional calibration battery spawn ──
            // Spawn calibration battery subprocess at sleep onset so results
            // are ready by the next sleep→wake transition.
            if !self.neuromod.was_sleeping && is_sleep_now {
                if self.neuromod.pending_calibration.is_none() {
                    self.spawn_calibration_battery(self.stats.total_cycles as u64);
                }
            }

            self.neuromod.was_sleeping = is_sleep_now;
        }

        // Apply circadian plasticity to learning rate (Night=high plasticity, Day=low)
        // Halved: bath circadian baselines (Phase 2) provide the other 50%
        let plasticity_half = 1.0 + (self.biorhythm.plasticity_mod as f32 - 1.0) * 0.5;
        let circadian_lr = self.stats.adaptive_learning_rate * plasticity_half;
        self.stats.adaptive_learning_rate = circadian_lr.clamp(0.0001, 0.1);

        // ═══════════════════════════════════════════════════════════════════════
        // NOCICEPTION: Drain infrastructure errors and convert to felt signals
        // ═══════════════════════════════════════════════════════════════════════
        self.somatic_bridge.update();
        let somatic_signals = self.somatic_bridge.to_interoceptive_signals();
        // Apply somatic stress to thermodynamic load (additive)
        self.thermodynamic_load =
            (self.thermodynamic_load + somatic_signals.thermodynamic_load_delta).min(1.0);
        // Apply arousal spike from severe infrastructure damage
        if somatic_signals.arousal_spike > 0.0 {
            self.emotion_contagion.arousal =
                (self.emotion_contagion.arousal + somatic_signals.arousal_spike).min(1.0);
        }
        // #5: Forward somatic stress to neuromodulator bath (McEwen 2007)
        let somatic_stress_level = self.somatic_bridge.systemic_stress() as f32;
        self.neuromod.bath.apply_stress(somatic_stress_level);

        // ═══════════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH: Produce from previous cycle's signals (Phase A)
        // Science: Doya (2002) — DA/NE/5-HT/ACh unify metalearning modulation.
        // Uses carryover values (previous cycle) to avoid ordering dependencies.
        // ═══════════════════════════════════════════════════════════════════════
        {
            let neuromod_inputs = super::super::neuromodulators::NeuromodulatorInputs {
                prediction_error: self.stats.avg_prediction_error,
                surprise: self.stats.avg_prediction_error > self.config.learning_threshold * 3.0,
                reward_signal: self.carryover.quality.last_value_score as f32,
                coherence: self.carryover.history.cached_coherence.unwrap_or(0.5),
                arousal: self.emotion_contagion.arousal,
                binding_strength: self.carryover.quality.last_phenomenal_binding as f32,
                epistemic_confidence: self.carryover.quality.last_epistemic_confidence,
                flow_active: self.flow_state.in_flow,
                // Consciousness → neuromod baseline modulation (Dehaene et al. 2006)
                consciousness_level: self.carryover.consciousness.last_sigma.map(|s| s as f32),
                // Moral judgment → oxytocin/DA (Zak 2012)
                moral_signal: Some(self.carryover.quality.last_moral_score),
            };
            self.neuromod.bath.update(&neuromod_inputs);
        }

        // ── Phase 5: Post-update bath wiring ────────────────────────────────
        // Record bath state for phase space analysis
        self.neuromod
            .phase_tracker
            .record(self.neuromod.bath.state_vector());
        // Allostatic load accumulation (McEwen 1998)
        {
            let cortisol = self.neuromod.bath.to_hormone_state().cortisol as f32;
            let is_sleep = self.biorhythm.phase == crate::chronobiology::CircadianPhase::Night;
            self.neuromod
                .bath
                .accumulate_allostatic_load(cortisol, is_sleep);
            // Adenosine clearance during sleep (Xie et al. 2013 — glymphatic)
            if is_sleep {
                self.neuromod.bath.clear_adenosine_sleep();
            }
        }

        // ── Phase transition detection (hysteresis-based, Kelso 1995) ──
        {
            let label = self.neuromod.bath.phase_label();
            self.neuromod.phase_detector.update(label);
        }

        // ── Bath metrics export (Prometheus gauges) ──
        #[cfg(feature = "api_module")]
        {
            let sv = self.neuromod.bath.state_vector();
            crate::api::metrics::update_bath_metrics(
                crate::api::metrics::global(),
                &sv,
                self.neuromod.bath.allostatic_load,
                self.neuromod.bath.ei_ratio(),
                self.neuromod.bath.sleep_pressure(),
                self.neuromod.bath.active_injections.len(),
            );
        }

        // ═══════════════════════════════════════════════════════════════════════
        // SELF-ASSESSMENT: Metacognitive performance monitoring
        // Tracks EMA of prediction error, coherence, confidence calibration,
        // attention utilization. Triggers self-calibration when drift > 1σ.
        // Science: Schmidhuber (2010) — formal theory of intrinsic motivation.
        // ═══════════════════════════════════════════════════════════════════════
        {
            let drift_anomalous = self.neuromod.drift_tracker.is_anomalous();
            // Use bath 5-HT effective directly as sustained attention proxy.
            // Previous approach (attention_sensitivity) was contaminated by ACh
            // multiplier accumulation across cycles. 5-HT effective is the clean
            // signal: low 5-HT → poor sustained attention → high "utilization".
            let sht_eff = self.neuromod.bath.serotonin.effective();
            // Inhibition error: fraction of gating signals (prefrontal veto) that fired.
            // Binary for now; extend to multi-signal average when more gates are tracked.
            let inhibition_error_rate = if self.carryover.quality.cached_prefrontal_veto {
                1.0
            } else {
                0.0
            };
            let sa_input = super::super::calibration::SelfAssessmentInput {
                prediction_error: self.stats.avg_prediction_error,
                coherence: self.carryover.history.cached_coherence.unwrap_or(0.5),
                confidence_calibration_error: (self.prediction_confidence as f32
                    - (1.0 - self.stats.avg_prediction_error.min(1.0)))
                .abs(),
                // Invert 5-HT: low serotonin → high utilization (sustained attn deficit)
                attention_utilization: (1.0 - sht_eff).clamp(0.0, 1.0),
                inhibition_error_rate,
                drift_anomalous,
                // Phase 1F: 5 new proxy signals for expanded 9-transmitter calibration
                social_coherence: self.neuromod.bath.oxytocin.effective() * 0.5 + 0.5,
                ei_ratio: self.neuromod.bath.ei_ratio(),
                excitotoxicity_risk: self.neuromod.bath.excitotoxicity_risk(),
                sleep_pressure: self.neuromod.bath.adenosine.effective(),
                allostatic_load: self.neuromod.bath.allostatic_load,
            };
            self.neuromod.self_assessment.update(&sa_input);

            // Poll async calibration battery (non-blocking).
            self.poll_calibration_battery();

            // Check if self-assessment triggers calibration.
            // Guard: don't overwrite pending external (psych-bench) calibration —
            // external calibrations are higher quality than internal proxy z-scores.
            if self.neuromod.pending_calibration.is_none() {
                if let Some(cal) = self.neuromod.self_assessment.check_trigger(drift_anomalous) {
                    tracing::info!(
                        adjustments = cal.adjustments.len(),
                        confidence_delta = cal.confidence_delta,
                        drift_anomalous,
                        "Self-assessment triggered auto-calibration"
                    );
                    self.neuromod.pending_calibration = Some(cal);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE -1: Ingest background-trained weights (non-blocking)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut trainer) = self.async_trainer {
            if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                trainer.apply_latest_weights(cfc);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // MORAL FREE ENERGY → EXPLORATION BOOST (FEP-principled)
        // High moral free energy = novel moral territory = boost exploration
        // to encounter scenarios in underrepresented harmony dimensions.
        //
        // Replaces raw topology completeness with continuous FEP signal:
        //   F = D_KL(q || p) + H(q)
        // where q = current harmony distribution, p = prior/expected.
        // High F → large KL divergence from moral prior → explore more.
        //
        // Science: Friston (2010) — active inference drives exploration to
        // minimize expected free energy; here applied to the moral manifold.
        // ═══════════════════════════════════════════════════════════════════════
        {
            // Copy values to avoid borrow overlap with adjust_exploration(&mut self)
            let free_energy = self.ethics_engine.last_moral_free_energy().free_energy;
            let gain = self.ethics_engine.moral_exploration_gain();
            let (scenario_count, completeness) = {
                let topo = self.ethics_engine.moral_topology().last_summary();
                (topo.scenario_count, topo.completeness)
            };

            // Bidirectional feedback: last cycle's PE modulates FE→exploration gain.
            // Uses avg_prediction_error as the outcome signal and the EMA-smoothed FE
            // as baseline — when exploration reduced PE below the smoothed FE expectation,
            // the coupling strengthens; when PE rose, it decays.
            if self.stats.total_cycles > 5 {
                let pe = self.stats.avg_prediction_error;
                let fe_ema = self.ethics_engine.moral_fe_ema() as f32;
                // Normalize: baseline PE ~0.3 for typical FE-driven exploration cycles
                let baseline_pe = 0.3_f32 + fe_ema * 0.1;
                self.ethics_engine
                    .feedback_exploration_outcome(pe, baseline_pe);
            }

            // FEP-driven: continuous moral free energy signal with adaptive gain.
            // F > 0.5 → novel moral territory → explore (scaled by adaptive gain)
            // F < 0.5 → familiar moral ground → no exploration boost
            // Gain adapts via feedback_exploration_outcome() [0.05, 0.25].
            if free_energy > 0.5 {
                let fe_boost = ((free_energy - 0.5) * gain as f64).min(0.2) as f32;
                self.adjust_exploration("moral_free_energy", fe_boost);
            }

            // Topology completeness still provides structural signal:
            // When fewer than 3 of 7 harmonies explored, boost regardless of F.
            // This catches cold-start (prior is zero, F is undefined/zero).
            if scenario_count >= 3 && completeness < 0.3 {
                let structural_boost = (0.3 - completeness) * 0.3; // up to +0.09
                self.adjust_exploration("moral_topology_gap", structural_boost as f32);
            }
        }

        let _ = module_timings; // consumed by caller for timing
        CycleInitResult {
            exploration_urge_start,
            startup_suppressed,
            startup_warmup_progress,
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // End-of-cycle stats and telemetry
    // Extracted from cycle.rs post-metadata section (zero behavioral change).
    // ═════════════════════════════════════════════════════════════════════════

    /// Update cumulative stats, neuromod EMA, and populate remaining metadata fields.
    ///
    /// Called after the metadata struct literal is assembled.
    pub(in crate::cognitive_loop) fn run_end_of_cycle_stats(
        &mut self,
        metadata: &mut super::super::CycleMetadata,
        resonator_wm_primed: bool,
        resonator_promotions: usize,
        codebook_evictions: usize,
        codebook_diversity: f32,
        fep_surprise: f64,
        surprise_thresh: f64,
        neuromod_attention_alloc: f32,
        phasic_da_replay_boost: usize,
        ne_reorienting_boost: f32,
        ne_arousal_feedback: f32,
        confidence_velocity: f32,
        sht_crash_dip: f32,
        exploration_sht_drain: f32,
    ) {
        // Apply neuromodulator telemetry (replaces flat fields with nested struct)
        metadata.neuromod = self.collect_neuromod_telemetry(neuromod_attention_alloc);

        // Phase 4: local-variable telemetry fields (not bath-derived)
        metadata.neuromod_phasic_replay_boost = phasic_da_replay_boost;
        metadata.neuromod_ne_reorienting_boost = ne_reorienting_boost;
        metadata.neuromod_drift_recovery_remaining = self.carryover.urgency.anomaly_drift_recovery;

        // Populate inhibition error count from metadata flags (prefrontal veto,
        // reasoning gate block, safety block). Feeds back into self-assessment
        // NE proxy via SelfAssessmentInput::inhibition_error_rate next cycle.
        metadata.neuromod.inhibition_errors_this_cycle = metadata.prefrontal_veto as u8
            + metadata.reasoning_gate_blocked as u8
            + metadata.safety_blocked as u8;
        metadata.ne_arousal_feedback = ne_arousal_feedback;
        metadata.confidence_velocity = confidence_velocity;
        metadata.sht_crash_dip = sht_crash_dip > 0.0;
        metadata.exploration_sht_drain = exploration_sht_drain;

        // Update cumulative stats for resonator-memory loop diagnostics
        if resonator_wm_primed {
            self.stats.resonator_wm_primed_count += 1;
        }
        self.stats.resonator_promotions_total += resonator_promotions as u64;
        self.stats.codebook_evictions_total += codebook_evictions as u64;
        if codebook_diversity > 0.0 {
            self.stats.codebook_diversity = codebook_diversity;
        }
        if fep_surprise > surprise_thresh {
            self.stats.fep_surprise_replay_boosts += 1;
        }

        // Exocortex trigger counter
        if self.neuromod.bath.should_query_exocortex() {
            self.stats.exocortex_triggers += 1;
        }

        // Neuromodulator EMA stats (alpha=0.05)
        {
            let alpha = 0.05_f32;
            let da = self.neuromod.bath.dopamine.effective();
            let ne = self.neuromod.bath.noradrenaline.effective();
            let sht = self.neuromod.bath.serotonin.effective();
            let ach = self.neuromod.bath.acetylcholine.effective();
            self.stats.avg_dopamine += alpha * (da - self.stats.avg_dopamine);
            self.stats.avg_noradrenaline += alpha * (ne - self.stats.avg_noradrenaline);
            self.stats.avg_serotonin += alpha * (sht - self.stats.avg_serotonin);
            self.stats.avg_acetylcholine += alpha * (ach - self.stats.avg_acetylcholine);
        }

        // Populate v0.8.0 Resonance Metadata
        metadata.thermodynamic_load = self.thermodynamic_load;
        metadata.somatic_stress = self.somatic_bridge.systemic_stress();
        metadata.mood_temperature = self.mood_temperature;
        // Phase 2.2: feedback proposal attribution telemetry
        metadata.feedback.feedback_confidence_proposals =
            self.feedback_state.confidence.len() as u32;
        metadata.feedback.feedback_lr_proposals =
            self.feedback_state.learning_rate.len() as u32;
        metadata.feedback.feedback_exploration_proposals =
            self.feedback_state.exploration.len() as u32;
        metadata.feedback.feedback_threshold_proposals =
            self.feedback_state.threshold.len() as u32;
        // Consensus outcomes from last end_cycle() integration
        if let Some(ref consensus) = self.feedback_state.last_consensus {
            metadata.feedback.consensus_confidence = consensus.consensus_confidence;
            metadata.feedback.consensus_lr = consensus.consensus_lr;
            metadata.feedback.consensus_exploration = consensus.consensus_exploration;
            metadata.feedback.consensus_threshold = consensus.consensus_threshold;
        }
        if self.config.trace_feedback {
            metadata.feedback.feedback_trace_confidence = self
                .feedback_state
                .confidence
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
            metadata.feedback.feedback_trace_lr = self
                .feedback_state
                .learning_rate
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
            metadata.feedback.feedback_trace_exploration = self
                .feedback_state
                .exploration
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
            metadata.feedback.feedback_trace_threshold = self
                .feedback_state
                .threshold
                .dump_proposals()
                .into_iter()
                .map(|(s, d)| (s.to_string(), d))
                .collect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═══════════════════════════════════════════════════════════════════════════
    // Result struct construction and field verification
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resonator_codebook_result_fields() {
        let result = ResonatorCodebookResult {
            resonator_promotions: 3,
            codebook_evictions: 1,
            codebook_diversity: 0.75,
            codebook_utilization_rate: 0.5,
        };
        assert_eq!(result.resonator_promotions, 3);
        assert_eq!(result.codebook_evictions, 1);
        assert!((result.codebook_diversity - 0.75).abs() < f32::EPSILON);
        assert!((result.codebook_utilization_rate - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dream_phase_result_fields() {
        let result = DreamPhaseResult {
            dream_insights: 5,
            dream_phi_improvement: 0.12,
            dream_wisdom_count: 2,
        };
        assert_eq!(result.dream_insights, 5);
        assert!((result.dream_phi_improvement - 0.12).abs() < f32::EPSILON);
        assert_eq!(result.dream_wisdom_count, 2);
    }

    #[test]
    fn test_episodic_replay_result_fields() {
        let result = EpisodicReplayResult {
            surprise_replay_batch_size: 16,
            phasic_da_replay_boost: 4,
        };
        assert_eq!(result.surprise_replay_batch_size, 16);
        assert_eq!(result.phasic_da_replay_boost, 4);
    }

    #[test]
    fn test_parameter_optimization_result_defaults() {
        let result = ParameterOptimizationResult {
            best_tau_scale: 1.0,
            phi_gain: 0.0,
            swap_occurred: false,
        };
        assert!((result.best_tau_scale - 1.0).abs() < f32::EPSILON);
        assert!((result.phi_gain - 0.0).abs() < f64::EPSILON);
        assert!(!result.swap_occurred);
    }

    #[test]
    fn test_urgency_result_fields() {
        let result = UrgencyResult {
            urgency: super::super::super::CycleUrgency::Normal,
            error_pattern: "Stable",
            predicted_urgency: "Normal",
            prediction_coherence_urgency_bias: 0.0,
        };
        assert!(matches!(
            result.urgency,
            super::super::super::CycleUrgency::Normal
        ));
        assert_eq!(result.error_pattern, "Stable");
        assert_eq!(result.predicted_urgency, "Normal");
        assert!((result.prediction_coherence_urgency_bias).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cycle_init_result_fields() {
        let result = CycleInitResult {
            exploration_urge_start: 0.3,
            startup_suppressed: true,
            startup_warmup_progress: 0.5,
        };
        assert!((result.exploration_urge_start - 0.3).abs() < f32::EPSILON);
        assert!(result.startup_suppressed);
        assert!((result.startup_warmup_progress - 0.5).abs() < f32::EPSILON);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // run_cycle_init tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_run_cycle_init_startup_suppressed_at_cycle_zero() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 0;
        let mut timings = super::super::super::ModuleTimings::default();
        let result = service.run_cycle_init(&mut timings);
        assert!(
            result.startup_suppressed,
            "Cycle 0 should be startup suppressed"
        );
        assert!(
            (result.startup_warmup_progress).abs() < f32::EPSILON,
            "Warmup progress at cycle 0 should be 0.0, got {}",
            result.startup_warmup_progress
        );
    }

    #[test]
    fn test_run_cycle_init_startup_suppressed_midway() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 25; // half of 50
        let mut timings = super::super::super::ModuleTimings::default();
        let result = service.run_cycle_init(&mut timings);
        assert!(
            result.startup_suppressed,
            "Cycle 25 should be startup suppressed"
        );
        assert!(
            (result.startup_warmup_progress - 0.5).abs() < 0.01,
            "Warmup progress at cycle 25 should be ~0.5, got {}",
            result.startup_warmup_progress
        );
    }

    #[test]
    fn test_run_cycle_init_not_suppressed_after_warmup() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100; // well past 50
        let mut timings = super::super::super::ModuleTimings::default();
        let result = service.run_cycle_init(&mut timings);
        assert!(
            !result.startup_suppressed,
            "Cycle 100 should NOT be startup suppressed"
        );
        assert!(
            (result.startup_warmup_progress - 1.0).abs() < f32::EPSILON,
            "Warmup progress past warmup should be 1.0, got {}",
            result.startup_warmup_progress
        );
    }

    #[test]
    fn test_run_cycle_init_lr_suppressed_during_warmup() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        let base_lr = service.stats.adaptive_learning_rate;
        service.stats.total_cycles = 10; // early warmup (10/50 = 20%)
        let mut timings = super::super::super::ModuleTimings::default();
        let _result = service.run_cycle_init(&mut timings);
        // During warmup, LR is scaled by 0.2 + 0.8 * progress, then clamped
        // and multiplied by circadian plasticity. It should be less than or
        // equal to the base learning rate.
        assert!(
            service.stats.adaptive_learning_rate <= base_lr + 0.001,
            "LR during warmup ({}) should not exceed base ({})",
            service.stats.adaptive_learning_rate,
            base_lr
        );
        assert!(
            service.stats.adaptive_learning_rate >= 0.0001,
            "LR should not go below minimum clamp"
        );
    }

    #[test]
    fn test_run_cycle_init_exploration_urge_suppressed_during_warmup() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.curiosity_drive.exploration_urge = 0.8;
        service.stats.total_cycles = 10; // 10/50 = 0.2 progress
        let mut timings = super::super::super::ModuleTimings::default();
        let result = service.run_cycle_init(&mut timings);
        // exploration_urge_start should capture the value AFTER suppression
        // Original 0.8 * 0.2 = 0.16
        assert!(
            result.exploration_urge_start < 0.8,
            "Exploration urge should be suppressed during warmup, got {}",
            result.exploration_urge_start
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // compute_urgency_and_error_pattern tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_urgency_warmup_pattern_for_short_history() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100;
        let threshold = service.config.learning_threshold;
        // With no error history, pattern should be "Warmup"
        let result = service.compute_urgency_and_error_pattern(0.01, false, threshold);
        assert_eq!(
            result.error_pattern, "Warmup",
            "Short error history should yield Warmup pattern"
        );
        assert_eq!(
            result.predicted_urgency, "Normal",
            "Warmup pattern should predict Normal urgency"
        );
    }

    #[test]
    fn test_urgency_consecutive_low_error_tracking() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100;
        let threshold = service.config.learning_threshold;
        // Reset consecutive counter
        service.carryover.urgency.consecutive_low_error = 0;
        // Low error below threshold should increment consecutive counter
        let low_error = threshold * 0.5;
        let _result = service.compute_urgency_and_error_pattern(low_error, false, threshold);
        assert!(
            service.carryover.urgency.consecutive_low_error > 0,
            "Low error should increment consecutive_low_error"
        );
    }

    #[test]
    fn test_urgency_consecutive_low_error_resets_on_high_error() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100;
        let threshold = service.config.learning_threshold;
        service.carryover.urgency.consecutive_low_error = 20;
        // High error above threshold should reset consecutive counter
        let high_error = threshold * 2.0;
        let _result = service.compute_urgency_and_error_pattern(high_error, false, threshold);
        assert_eq!(
            service.carryover.urgency.consecutive_low_error, 0,
            "High error should reset consecutive_low_error"
        );
    }

    #[test]
    fn test_urgency_mode_transition_increments_stat() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100;
        let threshold = service.config.learning_threshold;
        service.carryover.urgency.prev_urgency = super::super::super::CycleUrgency::Cruise;
        service.carryover.urgency.consecutive_low_error = 0;
        let transitions_before = service.stats.mode_transitions;
        // Trigger a Normal urgency (default cognitive depth = Cortical, high enough error)
        let _result = service.compute_urgency_and_error_pattern(threshold * 1.5, false, threshold);
        // Since prev_urgency was Cruise and the new one is likely Normal/Critical,
        // a mode transition should have been counted
        assert!(
            service.stats.mode_transitions > transitions_before,
            "Mode transition count should increment when urgency changes"
        );
    }

    #[test]
    fn test_urgency_stable_pattern_from_constant_errors() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100;
        let threshold = service.config.learning_threshold;
        // Push 5 identical low errors to create a stable pattern
        for _ in 0..5 {
            service.carryover.history.error_history.push_back(0.05);
        }
        let result = service.compute_urgency_and_error_pattern(0.05, false, threshold);
        // With constant errors, slope should be near-zero: pattern = "Stable"
        assert_eq!(
            result.error_pattern, "Stable",
            "Constant errors should yield Stable pattern, got {}",
            result.error_pattern
        );
    }

    #[test]
    fn test_urgency_prediction_coherence_bias_low_coherence() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100;
        let threshold = service.config.learning_threshold;
        // Set low prediction coherence to trigger the bias
        service.stats.avg_prediction_coherence = 0.2; // < 0.3 and > 0.0
        let result = service.compute_urgency_and_error_pattern(0.05, false, threshold);
        // coherence_mod = 0.85, bias = 0.85 - 1.0 = -0.15
        assert!(
            (result.prediction_coherence_urgency_bias - (-0.15)).abs() < 0.01,
            "Low coherence should produce negative bias, got {}",
            result.prediction_coherence_urgency_bias
        );
    }

    #[test]
    fn test_urgency_prediction_coherence_bias_high_coherence() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 100;
        let threshold = service.config.learning_threshold;
        // Set high prediction coherence
        service.stats.avg_prediction_coherence = 0.8; // > 0.7
        let result = service.compute_urgency_and_error_pattern(0.05, false, threshold);
        // coherence_mod = 1.1, bias = 1.1 - 1.0 = 0.1
        assert!(
            (result.prediction_coherence_urgency_bias - 0.1).abs() < 0.01,
            "High coherence should produce positive bias, got {}",
            result.prediction_coherence_urgency_bias
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // run_parameter_optimization_phase tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_parameter_optimization_skips_non_500_cycles() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 42; // not divisible by 500
        let result = service.run_parameter_optimization_phase();
        assert!(
            (result.best_tau_scale - 1.0).abs() < f32::EPSILON,
            "Should return default tau scale on non-500 cycles"
        );
        assert!(
            (result.phi_gain).abs() < f64::EPSILON,
            "Should return zero phi gain on non-500 cycles"
        );
        assert!(
            !result.swap_occurred,
            "No swap should occur on non-500 cycles"
        );
    }

    #[test]
    fn test_parameter_optimization_runs_on_500_cycles_no_episodes() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.total_cycles = 500; // divisible by 500
        let result = service.run_parameter_optimization_phase();
        // With no phi_episodic_replay or empty episodes, should return defaults
        assert!(
            (result.best_tau_scale - 1.0).abs() < f32::EPSILON,
            "Should return default tau scale with no episodes"
        );
        assert!(!result.swap_occurred, "No swap with no episodes");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // run_end_of_cycle_stats tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_end_of_cycle_stats_accumulates_promotions() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        let initial_promotions = service.stats.resonator_promotions_total;
        let initial_evictions = service.stats.codebook_evictions_total;
        let mut metadata = super::super::super::CycleMetadata::default();
        service.run_end_of_cycle_stats(
            &mut metadata,
            false, // resonator_wm_primed
            5,     // resonator_promotions
            2,     // codebook_evictions
            0.6,   // codebook_diversity
            0.5,   // fep_surprise
            0.3,   // surprise_thresh (surprise > thresh -> boost counted)
            0.0,   // neuromod_attention_alloc
            0,     // phasic_da_replay_boost
            0.0,   // ne_reorienting_boost
            0.0,   // ne_arousal_feedback
            0.0,   // confidence_velocity
            0.0,   // sht_crash_dip
            0.0,   // exploration_sht_drain
        );
        assert_eq!(
            service.stats.resonator_promotions_total,
            initial_promotions + 5,
            "Promotions should accumulate"
        );
        assert_eq!(
            service.stats.codebook_evictions_total,
            initial_evictions + 2,
            "Evictions should accumulate"
        );
    }

    #[test]
    fn test_end_of_cycle_stats_codebook_diversity_updated() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        let mut metadata = super::super::super::CycleMetadata::default();
        service.run_end_of_cycle_stats(
            &mut metadata,
            false,
            0,
            0,
            0.85, // codebook_diversity > 0.0 -> should update
            0.0,
            0.3,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert!(
            (service.stats.codebook_diversity - 0.85).abs() < f32::EPSILON,
            "Codebook diversity should be updated to 0.85, got {}",
            service.stats.codebook_diversity
        );
    }

    #[test]
    fn test_end_of_cycle_stats_zero_diversity_not_stored() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        service.stats.codebook_diversity = 0.5; // pre-existing value
        let mut metadata = super::super::super::CycleMetadata::default();
        service.run_end_of_cycle_stats(
            &mut metadata,
            false,
            0,
            0,
            0.0, // codebook_diversity == 0.0 -> should NOT update
            0.0,
            0.3,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert!(
            (service.stats.codebook_diversity - 0.5).abs() < f32::EPSILON,
            "Zero codebook diversity should not overwrite existing value"
        );
    }

    #[test]
    fn test_end_of_cycle_stats_surprise_replay_boost_counted() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        let initial_boosts = service.stats.fep_surprise_replay_boosts;
        let mut metadata = super::super::super::CycleMetadata::default();
        service.run_end_of_cycle_stats(
            &mut metadata,
            false,
            0,
            0,
            0.0,
            0.8, // fep_surprise
            0.3, // surprise_thresh -- surprise > thresh -> count incremented
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(
            service.stats.fep_surprise_replay_boosts,
            initial_boosts + 1,
            "FEP surprise replay boost should be counted when surprise > thresh"
        );
    }

    #[test]
    fn test_end_of_cycle_stats_wm_primed_counted() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        let initial = service.stats.resonator_wm_primed_count;
        let mut metadata = super::super::super::CycleMetadata::default();
        service.run_end_of_cycle_stats(
            &mut metadata,
            true, // resonator_wm_primed
            0,
            0,
            0.0,
            0.0,
            0.3,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(
            service.stats.resonator_wm_primed_count,
            initial + 1,
            "WM primed count should increment"
        );
    }

    #[test]
    fn test_end_of_cycle_stats_neuromod_ema_updates() {
        let mut service =
            CognitiveLoopService::new(super::super::super::CognitiveLoopConfig::default()).unwrap();
        let mut metadata = super::super::super::CycleMetadata::default();
        service.run_end_of_cycle_stats(
            &mut metadata,
            false,
            0,
            0,
            0.0,
            0.0,
            0.3,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        // EMA with alpha=0.05 should produce finite values
        assert!(
            service.stats.avg_dopamine.is_finite(),
            "avg_dopamine should be finite after EMA update"
        );
        assert!(
            service.stats.avg_noradrenaline.is_finite(),
            "avg_noradrenaline should be finite after EMA update"
        );
        assert!(
            service.stats.avg_serotonin.is_finite(),
            "avg_serotonin should be finite after EMA update"
        );
        assert!(
            service.stats.avg_acetylcholine.is_finite(),
            "avg_acetylcholine should be finite after EMA update"
        );
    }
}
