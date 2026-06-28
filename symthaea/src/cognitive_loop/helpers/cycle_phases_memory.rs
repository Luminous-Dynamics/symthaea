// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resonator codebook growth, episodic replay, and memory coordinator phases.
//!
//! Contains `run_resonator_codebook_phase` and `run_episodic_replay_and_memory_phase`.

use std::time::Instant;

use super::super::CognitiveLoopService;
use super::super::temporal_network::TemporalNetwork;
use super::cycle_phases::{EpisodicReplayResult, ResonatorCodebookResult};

impl CognitiveLoopService {
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
            if let Some(ref mut res_mem) = self.memory.memory_consol.resonator_memory {
                let res_dim_ok = compressed_state.len() == res_mem.resonator.config.dim;
                if res_dim_ok
                    && self.config.resonator_growth_interval > 0
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
                if let Some(ref mut res_mem) = self.memory.memory_consol.resonator_memory {
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
        if self.stats.total_cycles
            % crate::cognitive_loop::thresholds::MEMORY_HIGH_PHI_PROMOTION_CADENCE
            == 0
            && self.stats.total_cycles > 0
        {
            let top_eps = self
                .memory
                .episodic_persistence
                .replay
                .as_ref()
                .map(|replay| replay.get_top_episodes(3))
                .unwrap_or_default();

            if !top_eps.is_empty() {
                if let Some(ref mut res_mem) = self.memory.memory_consol.resonator_memory {
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
        let codebook_diversity: f32 = if self.stats.total_cycles
            % crate::cognitive_loop::thresholds::MEMORY_CODEBOOK_DIVERSITY_INTERVAL
            == 0
        {
            if let Some(ref res_mem) = self.memory.memory_consol.resonator_memory {
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
        let codebook_utilization_rate: f32 = if self.stats.total_cycles
            % crate::cognitive_loop::thresholds::MEMORY_CODEBOOK_UTILIZATION_INTERVAL
            == 0
        {
            if let Some(ref res_mem) = self.memory.memory_consol.resonator_memory {
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
                        self.stats.codebook_utilization_rate = self.stats.codebook_utilization_rate
                            * crate::cognitive_loop::thresholds::MEMORY_CODEBOOK_UTIL_EMA_DECAY
                            + rate
                                * crate::cognitive_loop::thresholds::MEMORY_CODEBOOK_UTIL_EMA_NEW;
                        // Low utilization → increase novelty threshold (harder to add)
                        if rate < crate::cognitive_loop::thresholds::MEMORY_CODEBOOK_LOW_UTILIZATION
                            && self.config.resonator_novelty_threshold < 0.9
                        {
                            self.config.resonator_novelty_threshold =
                                (self.config.resonator_novelty_threshold + crate::cognitive_loop::thresholds::MEMORY_NOVELTY_THRESHOLD_INCREASE).min(0.9);
                        } else if rate
                            > crate::cognitive_loop::thresholds::MEMORY_CODEBOOK_HIGH_UTILIZATION
                            && self.config.resonator_novelty_threshold > 0.3
                        {
                            // High utilization → lower novelty threshold (easier to add)
                            self.config.resonator_novelty_threshold =
                                (self.config.resonator_novelty_threshold - crate::cognitive_loop::thresholds::MEMORY_NOVELTY_THRESHOLD_DECREASE).max(0.3);
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
        if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
            let avg_err = self.stats.avg_prediction_error;
            let error_spike = avg_err > 0.01 && prediction_error > avg_err * 2.0;
            let semantic_miss = self.memory.memory_consol.semantic_memory.stats().semantic_misses > 0
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
        let mut memory_db_flushed = false;
        if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
            let coherence_summary = self.language_comm.voice_coherence.bridge.summary();
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
                self.unification_engine.emotional.state().valence as f32,
                coherence_summary.coherence,
            )
            .with_dopamine(self.neuromod.bath.dopamine.effective())
            .with_bath_state(self.neuromod.bath.state_vector());

            // Attach semantic embedding for content-based retrieval
            #[cfg(feature = "semantic-encoder")]
            let episode = if let Some(ref emb) = state.semantic_embedding {
                episode.with_semantic_embedding(emb.clone())
            } else {
                episode
            };

            let stored = replay.store_if_significant(episode);
            if stored {
                tracing::trace!(
                    phi = current_phi,
                    cycle = self.stats.total_cycles,
                    "High-Phi episode stored for replay"
                );
            }

            // Contextual retrieval: boost prediction confidence from similar past episodes
            #[cfg(feature = "semantic-encoder")]
            if let Some(ref emb) = state.semantic_embedding {
                let similar = replay.retrieve_by_embedding_similarity(emb, 3);
                if !similar.is_empty() {
                    let avg_pe: f32 = similar
                        .iter()
                        .filter_map(|(ep, _)| ep.prediction_error)
                        .sum::<f32>()
                        / similar.len().max(1) as f32;
                    if avg_pe < self.stats.avg_prediction_error * 0.5 {
                        self.prediction_confidence = (self.prediction_confidence + 0.02).min(1.0);
                    }
                }
            }

            if self.config.episodic_replay_training && replay.should_replay() {
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
                let sleep_boost = if self.biorhythm_mgr.rhythm.phase
                    == crate::chronobiology::CircadianPhase::Night
                {
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
                    let base_learning_rate = self.config.cfc_config.learning_rate;

                    // NREM/REM phase modulation: advance phase and adapt replay strategy.
                    // NREM: focused replay (state-dependent, high LR, recent + old interleaved)
                    // REM: broad replay (no state bias, lower LR, cross-episode abstraction)
                    // Science: WSCL (2024) — alternating NREM/REM prevents catastrophic forgetting
                    let phase = self.cantor_dream.advance_phase();
                    let (learning_rate, current_bath) = match phase {
                        super::super::cantor_dream_manager::DreamPhase::Nrem => {
                            // NREM: state-dependent, full LR — strengthen specific traces
                            (base_learning_rate, Some(self.neuromod.bath.state_vector()))
                        }
                        super::super::cantor_dream_manager::DreamPhase::Rem => {
                            // REM: context-free, 0.5x LR — abstract cross-episode patterns
                            // No bath conditioning forces sampling across diverse states
                            (base_learning_rate * 0.5, None)
                        }
                    };

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
                                self.memory
                                    .memory_consol
                                    .memory_coordinator
                                    .record_retrieval(hash);
                            }
                        }

                        // Track 3g: Dream consolidation — resonator factorization of replayed episodes
                        // Science: Stickgold (2005) — sleep replay extracts gist representations
                        // After episodic replay, factorize top episodes through the resonator to
                        // extract clean semantic components and strengthen codebook representations.
                        if let Some(ref mut res_mem) = self.memory.memory_consol.resonator_memory {
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

        // Track 3h: Cantor dream consolidation — cleanup buffered CRHVs through persistent engine
        // Science: Stickgold (2005) — sleep replay extracts gist; Hobson (2009) — dreaming
        // consolidates multi-level representations. CantorCleanupEngine unbinds each Cantor
        // layer, cleans against the codebook, and rebuilds — preserving faint peripheral
        // layers (shift/27, shift/81) that standard flat cleanup would strip.
        // This prevents "metacognitive amnesia" — loss of subtle associative structure.
        //
        // PERSISTENT ENGINE: Unlike the previous ephemeral approach, the CantorCleanupEngine
        // lives on CLS and accumulates codebook entries across dream cycles. Each dream
        // refreshes the codebook from resonator memory AND retains prior consolidated entries.
        // Science: Walker (2009) — memory consolidation requires stable, long-lived stores;
        //          Diekelmann & Born (2010) — repeated replay progressively strengthens traces.
        if !self.cantor_dream.broadcast_buffer.is_empty() {
            // Refresh persistent engine's codebook from resonator memory (additive — new entries
            // supplement existing consolidated knowledge rather than replacing it)
            if let Some(ref res_mem) = self.memory.memory_consol.resonator_memory {
                for cb in &res_mem.resonator.codebooks {
                    for (label, continuous_vec) in &cb.symbols {
                        let mut bytes = [0u8; 2048];
                        for (i, &val) in continuous_vec.iter().enumerate() {
                            if i / 8 < 2048 && val > 0.0 {
                                bytes[i / 8] |= 1 << (i % 8);
                            }
                        }
                        let bhv = symthaea_core::hdc::BinaryHV(bytes);
                        self.cantor_dream.cleanup_engine.codebook.add(label, bhv);
                    }
                }
            }

            // Drain the buffer — each CRHV gets cleaned and rebuilt
            let crhvs: Vec<_> = self.cantor_dream.broadcast_buffer.drain(..).collect();
            let mut dream_surprise_sum = 0.0f32;
            let mut dream_count = 0u32;
            let mut high_quality_count = 0u32;

            // DEPTH-STRATIFIED EVICTION: Before adding new entries, if the codebook
            // is near capacity, evict from the most crowded depth stratum.
            // This prevents shallow broadcasts from crowding out rare deep fractals.
            // Science: He et al. (2016) — residual learning preserves information
            // at different abstraction levels; each depth is a distinct abstraction.
            let codebook_cap = crate::cognitive_loop::thresholds::CANTOR_CODEBOOK_MAX_ENTRIES;
            if self.cantor_dream.cleanup_engine.codebook.len() > codebook_cap * 3 / 4 {
                // Count entries per depth stratum (encoded in label as "d{depth}_...")
                let mut depth_counts = [0usize; 8]; // depths 0-7
                for d in 0..8 {
                    depth_counts[d] = self
                        .cantor_dream
                        .cleanup_engine
                        .codebook
                        .count_by_prefix(&format!("d{d}_"));
                }
                // Find most crowded stratum
                let max_stratum = depth_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                // Evict oldest entry from that stratum (FIFO — first match removed)
                if depth_counts[max_stratum] > 2 {
                    let prefix = format!("d{max_stratum}_");
                    self.cantor_dream
                        .cleanup_engine
                        .codebook
                        .evict_by_prefix(&prefix);
                }
            }

            for crhv in &crhvs {
                let pre_ss = crhv.self_similarity();
                let result = self.cantor_dream.cleanup_engine.cleanup(crhv);
                let post_ss = result.cleaned.self_similarity();
                dream_surprise_sum += (pre_ss - post_ss).abs();
                dream_count += 1;
                // Closed-loop: high-quality cleanups grow the codebook permanently,
                // but only if the new entry is sufficiently different from existing ones.
                // Science: Born & Wilhelm (2012) — sleep consolidation strengthens
                // representations that survive replay without degradation.
                if result.quality
                    > crate::cognitive_loop::thresholds::CANTOR_DREAM_QUALITY_THRESHOLD
                {
                    high_quality_count += 1;
                    // Depth-stratified label: "d{depth}_dream_consolidated_{N}"
                    let label = format!(
                        "d{}_dream_consolidated_{}",
                        crhv.depth, self.cantor_dream.cleanup_engine.cleanups_performed
                    );
                    self.cantor_dream.cleanup_engine.codebook.add_if_diverse(
                        &label,
                        result.cleaned.base,
                        crate::cognitive_loop::thresholds::CANTOR_CODEBOOK_DIVERSITY_THRESHOLD,
                    );
                }
            }
            // Update dream surprise EMA — Friston (2010): surprise drives plasticity.
            if dream_count > 0 {
                let batch_surprise = dream_surprise_sum / dream_count as f32;
                let decay = crate::cognitive_loop::thresholds::CANTOR_SURPRISE_EMA_DECAY;
                self.cantor_dream.dream_surprise =
                    decay * self.cantor_dream.dream_surprise + (1.0 - decay) * batch_surprise;
            }
            // Feed surprise into FEP learning signal — novel fractal structure
            // indicates model inadequacy requiring plasticity boost.
            if self.cantor_dream.dream_surprise > 0.01 {
                self.fep.learning_signal += self.cantor_dream.dream_surprise * 0.2;
                self.fep.learning_signal = self.fep.learning_signal.clamp(-1.0, 1.0);
            }
            // Dream→Learning reliability coupling: fraction of high-quality cleanups
            // (those exceeding CANTOR_DREAM_QUALITY_THRESHOLD) measures consolidation
            // reliability. High reliability boosts waking plasticity for enhanced encoding.
            // Science: Diekelmann & Born (2010) — effective consolidation enhances
            // subsequent encoding; Walker (2017) — post-sleep learning enhancement.
            if dream_count > 0 {
                let consolidation_reliability = high_quality_count as f32 / dream_count as f32;
                self.learning_manager
                    .apply_dream_consolidation_boost(consolidation_reliability);
            }
            tracing::debug!(
                cleanups = self.cantor_dream.cleanup_engine.cleanups_performed,
                layers_cleaned = self.cantor_dream.cleanup_engine.layers_cleaned,
                layers_failed = self.cantor_dream.cleanup_engine.layers_failed,
                codebook_size = self.cantor_dream.cleanup_engine.codebook.len(),
                dream_surprise = self.cantor_dream.dream_surprise,
                "Cantor dream consolidation complete (persistent engine)"
            );
        }

        // Track 4d: Adaptive replay scheduling — modulate interval based on error volatility
        // Science: McClelland et al. (1995) — fast-changing environments need more replay
        if self.config.episodic_replay_training
            && self.stats.total_cycles % 50 == 0
            && self.stats.total_cycles > 50
        {
            if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                // Variance = E[X²] - E[X]² (from EMA-tracked moments)
                let error_variance = (self.stats.avg_prediction_error_sq
                    - self.stats.avg_prediction_error * self.stats.avg_prediction_error)
                    .max(0.0);
                replay.adapt_replay_interval(error_variance);
            }
        }

        // Memory coordinator: broadcast signals and process graduations
        {
            let coord_phi = self
                .language_comm
                .voice_coherence
                .bridge
                .smoothed_coherence() as f64;
            let coord_coherence = coherence as f64;
            self.memory
                .memory_consol
                .memory_coordinator
                .update_signals_with_sigma(
                    coord_phi,
                    coord_coherence,
                    self.carryover.consciousness.last_sigma,
                );

            if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                let graduated = self
                    .memory
                    .memory_consol
                    .memory_coordinator
                    .process_graduations(replay);
                if graduated > 0 {
                    tracing::debug!(
                        graduated,
                        "Memory coordinator graduated items to episodic storage"
                    );
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PERIODIC MEMORY FLUSH: Persist top episodes to SQLite
        // Co-prime cadence (199 cycles ≈ 0.85s at 234Hz) avoids interference
        // with other periodic tasks. Uses background thread to stay off rayon pool.
        // ═══════════════════════════════════════════════════════════════════════
        if self.stats.total_cycles % 199 == 0
            && self.stats.total_cycles > 0
            && (self.memory.episodic_persistence.storage_runtime.is_some()
                || self.memory.episodic_persistence.db.is_some())
        {
            use std::sync::atomic::Ordering;

            if !self
                .memory
                .episodic_persistence
                .flush_in_progress
                .load(Ordering::Relaxed)
            {
                if let Some(ref replay) = self.memory.episodic_persistence.replay {
                    let top_episodes = replay.get_top_episodes(16);
                    if !top_episodes.is_empty() {
                        let storage_runtime =
                            self.memory.episodic_persistence.storage_runtime.clone();
                        let db = self.memory.episodic_persistence.db.clone();
                        let flush_guard =
                            self.memory.episodic_persistence.flush_in_progress.clone();
                        flush_guard.store(true, Ordering::Relaxed);

                        let records: Vec<crate::databases::MemoryRecord> = top_episodes
                            .iter()
                            .enumerate()
                            .map(|(i, ep)| {
                                // Threshold continuous HV to binary: positive → 1
                                let mut bytes = [0u8; 2048];
                                for (j, &val) in ep.input.values.iter().enumerate() {
                                    if j / 8 < 2048 && val > 0.0 {
                                        bytes[j / 8] |= 1 << (j % 8);
                                    }
                                }
                                let encoding = symthaea_core::hdc::binary_hv::BinaryHV(bytes);
                                crate::databases::MemoryRecord {
                                    id: format!("ep_{}_{}", ep.timestamp, i),
                                    memory_type: crate::databases::MemoryType::Episodic,
                                    encoding,
                                    content: String::new(),
                                    timestamp_ms: ep.timestamp as u64 * 20, // ~20ms per cycle at 50Hz
                                    valence: ep.valence.unwrap_or(0.0),
                                    arousal: 0.5,
                                    psi: ep.psi,
                                    topics: Vec::new(),
                                    metadata: "{}".to_string(),
                                    consolidation_strength: ep.psi,
                                    retrieval_count: 0,
                                }
                            })
                            .collect();

                        memory_db_flushed = true;
                        if let Some(runtime) = storage_runtime {
                            match runtime.try_store_memory_batch(records) {
                                Ok(()) => {
                                    tracing::debug!(
                                        "Memory flush: episodes queued to storage runtime"
                                    );
                                }
                                Err(e) => {
                                    memory_db_flushed = false;
                                    tracing::warn!(error = %e, "Memory flush queue failed");
                                }
                            }
                            flush_guard.store(false, Ordering::Relaxed);
                        } else if let Some(db) = db {
                            std::thread::spawn(move || {
                                match db.store_batch_sync(&records) {
                                    Ok(n) => {
                                        tracing::debug!(
                                            stored = n,
                                            "Memory flush: episodes persisted to SQLite"
                                        );
                                    }
                                    Err(e) => {
                                        tracing::warn!(error = %e, "Memory flush failed");
                                    }
                                }
                                flush_guard.store(false, Ordering::Relaxed);
                            });
                        } else {
                            memory_db_flushed = false;
                            flush_guard.store(false, Ordering::Relaxed);
                        }
                    }
                }
            }
        }

        module_timings.episodic_replay = _t.elapsed().as_micros() as u64;

        EpisodicReplayResult {
            surprise_replay_batch_size,
            phasic_da_replay_boost,
            memory_db_flushed,
        }
    }
}
