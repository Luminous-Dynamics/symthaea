// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
        let (gwt_broadcast, gwt_coalition_size, gwt_activation) =
            if ctx.urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut gwt) = self.consciousness.gwt_mgr.gwt {
                    // GWT activation via multi-source salience (not just 1-PE).
                    // Science: Dehaene & Changeux (2011) — workspace ignition requires
                    // convergent bottom-up salience from multiple sources.
                    // Pure (1-PE) fails for text cognition where PE ≈ 0.9 inherently.
                    //
                    // Salience = max(prediction_match, coherence, novelty, confidence)
                    // max() ensures any strong signal can drive workspace entry.
                    let prediction_match = (1.0 - ctx.prediction_error as f64).clamp(0.0, 1.0);
                    let coherence_salience = ctx.coherence as f64;
                    // Novelty: high PE + high curiosity = salient novel content
                    let novelty_salience = (ctx.prediction_error as f64
                        * self.behavior.curiosity_drive.curiosity as f64)
                        .clamp(0.0, 1.0);
                    let confidence_salience = self.prediction_confidence;
                    let activation = prediction_match
                        .max(coherence_salience)
                        .max(novelty_salience)
                        .max(confidence_salience)
                        .clamp(0.0, 1.0);
                    // Submit current encoding with activation-weighted salience.
                    // List all active contributing modules as supporters — larger
                    // coalitions get synergy boost and higher Phi estimate, which is
                    // critical for crossing the workspace entry threshold.
                    // Science: Dehaene (2011) — ignition requires convergent evidence
                    // from multiple specialized processors.
                    let mut supporters = vec!["encoder".to_string()];
                    if self.consciousness.self_model_tier.narrative_self.is_some() {
                        supporters.push("narrative_self".to_string());
                    }
                    if self.consciousness.self_model_tier.meta_cognition.is_some() {
                        supporters.push("meta_cognition".to_string());
                    }
                    if self
                        .consciousness
                        .consciousness_monitors
                        .phenomenal_binding
                        .is_some()
                    {
                        supporters.push("phenomenal_binding".to_string());
                    }
                    if self.config.enable_knowledge_engine {
                        supporters.push("knowledge".to_string());
                    }
                    gwt.submit_strategy(
                        "cognitive_loop",
                        activation,
                        vec![ctx.hv16_cached],
                        supporters,
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
                    // CTC: Wire binding coherence and PAC modulation index into GWT
                    // Science: Fries (2015) — phase-locked binding gates workspace entry
                    #[cfg(feature = "ctc_wiring")]
                    {
                        // Use equation V2's cached PAC modulation and multimodal binding
                        let binding_c = self
                            .consciousness
                            .consciousness_engine
                            .last_multimodal_binding_coherence();
                        let pac = self
                            .consciousness
                            .consciousness_engine
                            .last_pac_modulation();
                        gwt.set_ctc_signals(binding_c, pac);
                    }

                    let result = gwt.process();
                    let coalition_size = result
                        .winning_coalition
                        .as_ref()
                        .map(|c| c.members.len() as u32)
                        .unwrap_or(0);
                    (result.broadcast_occurred, coalition_size, activation as f32)
                } else {
                    (false, 0, 0.0)
                }
            } else {
                (false, 0, self.cantor_dream.last_activation)
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

            // CANTOR PROMOTION: Wrap broadcast BinaryHV as Cantor Recursive Hypervector.
            // When a thought becomes "conscious" (survives workspace competition and is
            // broadcast), promote it to CRHV to preserve multi-scale fractal structure.
            // These accumulate in the buffer and are cleaned during dream consolidation
            // via CantorCleanupEngine, preventing metacognitive amnesia.
            // Science: Dehaene (2014) — ignition creates multi-scale cortical resonance;
            //          Hobson (2009) — dreaming consolidates multi-level representations.
            //
            // ADAPTIVE DEPTH: Stronger GWT activations produce deeper fractals.
            // activation ∈ [0,1] maps to depth ∈ [2,7]:
            //   - Weak broadcast (act~0.2): depth 2 (base + shift/3)
            //   - Medium broadcast (act~0.5): depth 4 (base + shift/3 + shift/9 + shift/27)
            //   - Strong broadcast (act~1.0): depth 7 (full recursive self-similarity)
            // Science: Dehaene et al. (2006) — ignition strength ∝ recruited cortical layers.
            use symthaea_core::hdc::cantor_recursive_hv::CantorRecursiveHV;
            self.cantor_dream.last_activation = gwt_activation;
            use crate::cognitive_loop::thresholds::{CANTOR_DEPTH_MAX, CANTOR_DEPTH_MIN};
            let depth_range = CANTOR_DEPTH_MAX - CANTOR_DEPTH_MIN;
            let adaptive_depth =
                CANTOR_DEPTH_MIN + (gwt_activation * depth_range as f32).round() as usize;

            // RADIAL CANTOR PROMOTION: When cross-modal binding is strong (previous cycle),
            // wrap the broadcast in RadialCantor geometry instead of flat CRHV.
            // RadialCantor applies concentric band structure that preserves perceptual
            // hierarchy (core = abstract, periphery = contextual).
            // Science: Treisman & Gelade (1980) — feature integration requires binding;
            //          strong binding justifies geometric structure in the representation.
            let binding_strength = self.carryover.quality.last_phenomenal_binding as f32;
            let crhv = if binding_strength
                > crate::cognitive_loop::thresholds::CANTOR_RADIAL_BINDING_THRESHOLD
            {
                use symthaea_core::hdc::multidimensional_cantor::RadialCantorHV;
                let radial = RadialCantorHV::new(
                    ctx.hv16_cached,
                    crate::cognitive_loop::thresholds::CANTOR_RADIAL_BANDS,
                );
                // Wrap the radial vector as a CRHV (preserves pipeline compatibility)
                CantorRecursiveHV::from_base_with_depth(radial.vector, adaptive_depth)
            } else {
                CantorRecursiveHV::from_base_with_depth(ctx.hv16_cached, adaptive_depth)
            };
            // Cap at 32 to bound memory — oldest broadcasts evicted first (ring semantics)
            if self.cantor_dream.broadcast_buffer.len() >= 32 {
                self.cantor_dream.broadcast_buffer.remove(0);
            }
            self.cantor_dream.broadcast_buffer.push(crhv);

            // RESONANCE COUPLING: Detect coherent CRHV pairs in the buffer.
            // When multiple fractal representations share high base-vector similarity,
            // they form a "resonant coalition" that amplifies workspace integration.
            // Science: Edelman & Tononi (2000) — reentrant signaling creates dynamic
            //          neural coalitions; Singer (1999) — binding by synchrony.
            let buf = &self.cantor_dream.broadcast_buffer;
            if buf.len() >= 2 {
                let threshold =
                    crate::cognitive_loop::thresholds::CANTOR_RESONANCE_SIMILARITY_THRESHOLD;
                let mut resonant_pairs = 0u32;
                let n = buf.len();
                // Compare last 8 entries (recent window) to avoid O(n²) over full buffer
                let window_start = n.saturating_sub(8);
                for i in window_start..n {
                    for j in (i + 1)..n {
                        if buf[i].base.similarity(&buf[j].base) > threshold {
                            resonant_pairs += 1;
                        }
                    }
                }
                // Normalize: max pairs in window of 8 = C(8,2) = 28
                let max_pairs = {
                    let w = (n - window_start) as u32;
                    w * (w - 1) / 2
                };
                self.cantor_dream.resonance_boost = if max_pairs > 0 {
                    (resonant_pairs as f32 / max_pairs as f32).min(1.0)
                } else {
                    0.0
                };
                if self.cantor_dream.resonance_boost > 0.1 {
                    self.adjust_confidence(
                        "cantor_resonance",
                        crate::cognitive_loop::thresholds::CANTOR_RESONANCE_CONFIDENCE_BOOST
                            * self.cantor_dream.resonance_boost,
                    );
                }
                // CANTOR → EXPLORATION: Surprise drives exploration, resonance drives exploitation.
                // Science: Sutton & Barto (2018) — surprise-driven explore/exploit tradeoff.
                if self.cantor_dream.dream_surprise
                    > crate::cognitive_loop::thresholds::CANTOR_SURPRISE_EXPLORATION_THRESHOLD
                {
                    self.adjust_exploration(
                        "cantor_dream_surprise",
                        crate::cognitive_loop::thresholds::CANTOR_SURPRISE_EXPLORATION_BOOST
                            * self.cantor_dream.dream_surprise,
                    );
                }
                if self.cantor_dream.resonance_boost
                    > crate::cognitive_loop::thresholds::CANTOR_RESONANCE_EXPLORATION_DAMPEN_THRESHOLD
                {
                    self.adjust_exploration(
                        "cantor_resonance_exploit",
                        crate::cognitive_loop::thresholds::CANTOR_RESONANCE_EXPLORATION_DAMPEN
                            * self.cantor_dream.resonance_boost,
                    );
                }
            }
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
            if let Some(ref mut resonance) = self.consciousness.consciousness_monitors.resonance {
                let dims = [
                    ctx.unified_psi,
                    ctx.coherence as f64,
                    wm_utilization,
                    self.behavior.adaptive_behavior.attention_sensitivity as f64,
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
            let attention_mod = 1.0
                + (resonance_quality as f32 - 0.5)
                    * crate::cognitive_loop::thresholds::RESONANCE_ATTENTION_SCALE; // ±5%
            self.behavior.adaptive_behavior.attention_sensitivity *= attention_mod;
        }

        let quantum_coherence_level = if ctx.urgency.run_consciousness_monitors() {
            if let Some(ref mut qc) = self.consciousness.consciousness_monitors.quantum_coherence {
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
        if quantum_coherence_level
            > crate::cognitive_loop::thresholds::QUANTUM_COHERENCE_HIGH_THRESHOLD
        {
            let qc_boost = (quantum_coherence_level
                - crate::cognitive_loop::thresholds::QUANTUM_COHERENCE_HIGH_THRESHOLD)
                as f32
                * crate::cognitive_loop::thresholds::QUANTUM_COHERENCE_CONFIDENCE_BOOST_SCALE
                    as f32; // up to +2%
            self.adjust_confidence("quantum_coherence_high", qc_boost);
        } else if quantum_coherence_level > 0.0
            && quantum_coherence_level
                < crate::cognitive_loop::thresholds::QUANTUM_COHERENCE_LOW_THRESHOLD
        {
            self.scale_confidence(
                "quantum_decoherence",
                crate::cognitive_loop::thresholds::QUANTUM_DECOHERENCE_CONFIDENCE_SCALE,
            );
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
                if let Some(ref mut binding) =
                    self.consciousness.consciousness_monitors.phenomenal_binding
                {
                    let dims = [
                        ctx.unified_psi,
                        ctx.coherence as f64,
                        wm_utilization,
                        self.behavior.adaptive_behavior.attention_sensitivity as f64,
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

        // Therapeutic narrative coherence modulates phenomenal binding.
        // Fragmented narratives (traumatic memory, incoherent self-model)
        // weaken binding; coherent narratives strengthen it.
        // Science: Gallagher (2000) — narrative self-model supports unified experience;
        // Damasio (1999) — autobiographical self grounds phenomenal consciousness.
        #[cfg(feature = "therapeutic")]
        let phenomenal_binding_strength = {
            let narrative_coh = self.therapeutic_manager.narrative_coherence() as f64;
            // Modulate binding: 0.85 (low coherence) to 1.05 (high coherence)
            let narrative_factor = 0.85 + 0.20 * narrative_coh;
            (phenomenal_binding_strength * narrative_factor).clamp(0.0, 1.0)
        };

        // FEEDBACK: Fragmentation suppresses exploration (Singer 1989)
        // When consciousness is fragmented, focus on integration not exploration
        if phenomenal_fragmented {
            self.behavior.curiosity_drive.boredom *=
                crate::cognitive_loop::thresholds::PHENOMENAL_FRAGMENTATION_BOREDOM_SCALE;
            self.behavior.adaptive_behavior.exploration_factor *=
                crate::cognitive_loop::thresholds::PHENOMENAL_FRAGMENTATION_EXPLORATION_SCALE;
        }

        // FEEDBACK: High binding strength (flow) boosts learning rate (Csikszentmihalyi 1990)
        if phenomenal_binding_strength
            > crate::cognitive_loop::thresholds::PHENOMENAL_BINDING_HIGH_THRESHOLD
        {
            let binding_boost = ((phenomenal_binding_strength
                - crate::cognitive_loop::thresholds::PHENOMENAL_BINDING_HIGH_THRESHOLD)
                * crate::cognitive_loop::thresholds::PHENOMENAL_BINDING_LR_BOOST_SCALE as f64)
                as f32; // up to +4%
            self.carryover.learning.subsystem_lr_factor *= 1.0 + binding_boost;
        }

        // FEEDBACK: Binding strength gates WM access via attention sensitivity (Tononi 2015 IIT)
        // Science: High integrated information → more can be held in working memory;
        // low binding → restrict input (WM fragmented, accept less)
        if phenomenal_binding_strength
            > crate::cognitive_loop::thresholds::WM_BINDING_HIGH_THRESHOLD
        {
            let wm_boost = ((phenomenal_binding_strength
                - crate::cognitive_loop::thresholds::WM_BINDING_HIGH_THRESHOLD)
                * crate::cognitive_loop::thresholds::WM_BINDING_BOOST_SCALE)
                as f32;
            self.behavior.adaptive_behavior.attention_sensitivity *= 1.0 + wm_boost;
        } else if phenomenal_binding_strength > 0.0
            && phenomenal_binding_strength
                < crate::cognitive_loop::thresholds::WM_BINDING_LOW_THRESHOLD
        {
            let wm_restrict = ((crate::cognitive_loop::thresholds::WM_BINDING_LOW_THRESHOLD
                - phenomenal_binding_strength)
                * crate::cognitive_loop::thresholds::WM_BINDING_RESTRICT_SCALE)
                as f32;
            self.behavior.adaptive_behavior.attention_sensitivity *= (1.0 - wm_restrict)
                .max(crate::cognitive_loop::thresholds::WM_BINDING_MIN_SENSITIVITY);
        }

        module_timings.phenomenal_binding = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // TEMPORAL CONSCIOUSNESS: Track Phi trajectory, continuity, identity
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (temporal_coherence_score, temporal_discontinuity) =
            if ctx.urgency.run_consciousness_monitors() {
                if let Some(ref mut temporal) = self.consciousness.consciousness_monitors.temporal {
                    temporal.observe(
                        &ctx.hv16_cached,
                        ctx.unified_psi,
                        self.consciousness.self_model_tier.narrative_self.as_ref(),
                        self.consciousness.self_model_tier.predictive_self.as_ref(),
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
            self.scale_confidence(
                "temporal_discontinuity",
                crate::cognitive_loop::thresholds::TEMPORAL_DISCONTINUITY_CONFIDENCE_SCALE,
            );
            // Lower learning threshold to learn more aggressively after discontinuity
            self.scale_threshold(
                "temporal_discontinuity",
                crate::cognitive_loop::thresholds::TEMPORAL_DISCONTINUITY_CONFIDENCE_SCALE,
            );
        } else if temporal_coherence_score
            > crate::cognitive_loop::thresholds::TEMPORAL_COHERENCE_HIGH_THRESHOLD
        {
            // High temporal coherence → model is reliable, raise threshold (learn less often)
            self.scale_threshold(
                "temporal_high_coherence",
                crate::cognitive_loop::thresholds::TEMPORAL_COHERENCE_THRESHOLD_BOOST,
            );
        } else {
            // Slowly return toward baseline (homeostasis drift toward 1.0)
            let drift = (1.0 - self.carryover.learning.adaptive_threshold_scale)
                * crate::cognitive_loop::thresholds::HOMEOSTASIS_DRIFT_RATE;
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
        if streak >= crate::cognitive_loop::thresholds::TEMPORAL_DISCONTINUITY_PERSISTENT_THRESHOLD
        {
            // Persistent discontinuity: aggressive recovery
            self.last_prediction = None; // invalidate stale predictions
            self.scale_lr(
                "persistent_discontinuity",
                crate::cognitive_loop::thresholds::TEMPORAL_DISCONTINUITY_PERSISTENT_LR_BOOST,
            );
            self.scale_exploration(
                "discontinuity_recovery",
                crate::cognitive_loop::thresholds::PHENOMENAL_FRAGMENTATION_EXPLORATION_SCALE,
            );
            self.stats.discontinuity_cascade_count += 1;
        }

        // FEEDBACK: High temporal coherence strengthens narrative self engagement
        // Science: Damasio (2010) — temporal continuity is the substrate of selfhood
        if temporal_coherence_score
            > crate::cognitive_loop::thresholds::TEMPORAL_NARRATIVE_THRESHOLD
        {
            if let Some(ref mut narrative) = self.consciousness.self_model_tier.narrative_self {
                let continuity_boost = (temporal_coherence_score
                    - crate::cognitive_loop::thresholds::TEMPORAL_NARRATIVE_THRESHOLD)
                    * crate::cognitive_loop::thresholds::TEMPORAL_NARRATIVE_BOOST_SCALE;
                narrative.boost_coherence(continuity_boost);
            }
        }

        // FEEDBACK: Temporal coherence ↔ attention mutual coupling (Engel et al. 2001)
        // Science: Temporal binding via phase synchrony — attention must gate synchronization.
        // Low temporal coherence → attention is fragmenting the time-axis → penalize sensitivity
        // to prevent amplification of incoherent states. High coherence → attention is stable.
        if temporal_coherence_score > 0.0
            && temporal_coherence_score
                < crate::cognitive_loop::thresholds::TEMPORAL_ATTENTION_PENALTY_THRESHOLD
        {
            let coherence_penalty =
                ((crate::cognitive_loop::thresholds::TEMPORAL_ATTENTION_PENALTY_THRESHOLD
                    - temporal_coherence_score)
                    * crate::cognitive_loop::thresholds::TEMPORAL_ATTENTION_PENALTY_SCALE)
                    as f32;
            self.behavior.adaptive_behavior.attention_sensitivity *= (1.0 - coherence_penalty)
                .max(crate::cognitive_loop::thresholds::TEMPORAL_ATTENTION_MIN);
        }

        module_timings.temporal_consciousness = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS THERMODYNAMICS: Phase transitions & free energy
        // Urgency-gated: same as consciousness monitors (skip in Cruise)
        // Science: Friston (2010) — free energy, Kelso — critical fluctuations
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (thermodynamic_entropy, thermodynamic_free_energy) = if ctx
            .urgency
            .run_consciousness_monitors()
        {
            if let Some(ref mut thermo) = self.consciousness.consciousness_monitors.thermodynamics {
                let dims = [
                    ctx.unified_psi,
                    ctx.coherence as f64,
                    wm_utilization,
                    self.behavior.adaptive_behavior.attention_sensitivity as f64,
                    (self.stats.total_cycles.min(100) as f64 / 100.0),
                    late.body_psi_modulation,
                    self.prediction_confidence,
                ];
                let state = thermo.analyze(dims);
                // Feed analyzer data to ThermodynamicManager
                self.thermodynamic_mgr.set_analyzer(
                    state.entropy,
                    state.free_energy,
                    state.temperature,
                    state.phase,
                );
                // FEEDBACK: Phase-dependent exploration modulation (Kelso 1995)
                use crate::consciousness::consciousness_thermodynamics::ConsciousnessPhase;
                match state.phase {
                    ConsciousnessPhase::Critical => {
                        // Edge of chaos — maximum creativity, boost exploration
                        self.behavior.curiosity_drive.boredom *=
                            crate::cognitive_loop::thresholds::THERMO_CRITICAL_CURIOSITY_BOOST;
                        self.behavior.adaptive_behavior.exploration_factor *= crate::cognitive_loop::thresholds::THERMODYNAMIC_STRESS_EXPLORATION_BOOST;
                    }
                    ConsciousnessPhase::Flow => {
                        // Superfluid state — boost learning rate
                        self.carryover.learning.subsystem_lr_factor *=
                            crate::cognitive_loop::thresholds::THERMO_FLOW_LR_BOOST;
                    }
                    ConsciousnessPhase::Chaotic => {
                        // Fragmented — suppress exploration, seek stability
                        self.behavior.curiosity_drive.boredom *= crate::cognitive_loop::thresholds::PHENOMENAL_FRAGMENTATION_EXPLORATION_SCALE;
                        self.behavior.adaptive_behavior.exploration_factor *= crate::cognitive_loop::thresholds::THERMODYNAMIC_RECOVERY_EXPLORATION_SCALE;
                    }
                    ConsciousnessPhase::Frozen => {
                        // Rigid — nudge toward exploration to unfreeze
                        self.behavior.curiosity_drive.boredom *=
                            crate::cognitive_loop::thresholds::THERMO_FROZEN_CURIOSITY_BOOST;
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
            if thermodynamic_entropy
                > crate::cognitive_loop::thresholds::THERMODYNAMIC_ENTROPY_HIGH_THRESHOLD
            {
                let entropy_boost = ((thermodynamic_entropy
                    - crate::cognitive_loop::thresholds::THERMODYNAMIC_ENTROPY_HIGH_THRESHOLD)
                    * 0.1)
                    .min(0.1) as f32;
                self.adjust_exploration("thermo_entropy", entropy_boost);
            } else if thermodynamic_entropy
                < crate::cognitive_loop::thresholds::THERMODYNAMIC_ENTROPY_LOW_THRESHOLD
            {
                let consolidation_bias =
                    ((crate::cognitive_loop::thresholds::THERMODYNAMIC_ENTROPY_LOW_THRESHOLD
                        - thermodynamic_entropy)
                        * 0.08)
                        .min(0.08) as f32;
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
                if let Some(ref mut embodied) = self.consciousness.consciousness_monitors.embodied {
                    if let Some(ref body) = self.sensorimotor.vision_sensory.virtual_body {
                        embodied.update_interoception(body.interoceptive_state().clone());
                    }
                    let response = embodied.process();
                    self.carryover.consciousness.embodied_phi_modulation = response.phi_modulation;

                    // Wire embodied signals into cognitive loop:
                    // 1. Homeostatic deviation increases urgency (survival takes priority)
                    // Science: Damasio (1999) — somatic markers guide decision-making
                    {
                        use crate::cognitive_loop::thresholds::{
                            ALLOSTATIC_LOAD_DANGER_THRESHOLD, ALLOSTATIC_LOAD_LR_DAMPEN,
                            HOMEOSTATIC_DEVIATION_THRESHOLD, SENSORIMOTOR_SURPRISE_EXPLORE_SCALE,
                            SENSORIMOTOR_SURPRISE_THRESHOLD,
                        };
                        if response.homeostatic_deviation > HOMEOSTATIC_DEVIATION_THRESHOLD {
                            self.carryover.urgency.consecutive_low_error = 0; // prevent Cruise when body is stressed
                        }
                        // 2. Sensorimotor surprise blends into exploration urge
                        // Science: Friston (2010) — interoceptive surprise drives active inference
                        if response.sensorimotor_surprise > SENSORIMOTOR_SURPRISE_THRESHOLD {
                            let body_nudge = (response.sensorimotor_surprise
                                * SENSORIMOTOR_SURPRISE_EXPLORE_SCALE)
                                .min(0.15) as f32;
                            self.adjust_exploration("sensorimotor_surprise", body_nudge);
                        }
                        // 3. High allostatic load suppresses learning (conserve resources)
                        // Science: McEwen (2004) — allostatic overload impairs plasticity
                        if response.allostatic_load > ALLOSTATIC_LOAD_DANGER_THRESHOLD {
                            self.scale_lr(
                                "allostatic_overload",
                                1.0 - (response.allostatic_load - ALLOSTATIC_LOAD_DANGER_THRESHOLD)
                                    as f32
                                    * ALLOSTATIC_LOAD_LR_DAMPEN as f32,
                            );
                        }
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
        {
            use crate::cognitive_loop::thresholds::{
                EMBODIED_AGENCY_BOOST_SCALE, EMBODIED_AGENCY_CAUTION_FLOOR,
                EMBODIED_AGENCY_CAUTION_SCALE, EMBODIED_AGENCY_HIGH_THRESHOLD,
                EMBODIED_AGENCY_LOW_THRESHOLD,
            };
            if embodied_agency > EMBODIED_AGENCY_HIGH_THRESHOLD {
                let agency_boost = ((embodied_agency - EMBODIED_AGENCY_HIGH_THRESHOLD)
                    * EMBODIED_AGENCY_BOOST_SCALE) as f32;
                self.behavior.adaptive_behavior.exploration_factor *= 1.0 + agency_boost;
            } else if embodied_agency > 0.0 && embodied_agency < EMBODIED_AGENCY_LOW_THRESHOLD {
                let caution = ((EMBODIED_AGENCY_LOW_THRESHOLD - embodied_agency)
                    * EMBODIED_AGENCY_CAUTION_SCALE) as f32;
                self.scale_exploration(
                    "embodied_caution",
                    (1.0 - caution).max(EMBODIED_AGENCY_CAUTION_FLOOR),
                );
            }
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
            let enactive_action = match self.behavior.adaptive_behavior.action_hint {
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
                self.behavior.adaptive_behavior.attention_sensitivity *= 1.0 + relevance_gain;
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

        // PRE-MCE NARRATIVE: surprise-triggered episodes bypass MCE gating.
        // Without this, episodes only accumulate inside the MCE block (every 5-20
        // cycles), creating a circular dependency: low C → infrequent MCE → few
        // episodes → low N → low C. Surprise events are encoded pre-consciously.
        // Science: Ranganath & Rainer (2003) — novel stimuli are encoded into
        // memory even without full conscious access (pre-attentive encoding).
        if ctx.surprise_triggered {
            let valence = -(ctx.prediction_error as f64 * 0.3); // surprise is mildly negative
            self.consciousness
                .master_equation
                .narrative_coherence
                .add_episode(
                    format!("surprise_pre_pe{:.2}", ctx.prediction_error),
                    valence,
                );
        }

        // Run every 10th cycle to amortize cost. Maps cognitive loop signals to
        // the 8-factor ConsciousnessInputs: Phi, Broadcast, WorkingMemory,
        // Attention, Recurrence, Embodiment, Knowledge, Synchrony.
        // Urgency-adaptive: Critical=every 5th, Normal=every 10th, Cruise=every 20th
        let consciousness_level = if ctx.urgency.should_run(self.stats.total_cycles, 5, 10, 20) {
            // Wire embodiment factor from cognitive loop signals.
            // Science: Friston (2010) — low PE = good embodied prediction (sensorimotor accuracy)
            // Science: Barrett (2017) — interoceptive coherence from allostatic regulation
            self.consciousness
                .master_equation
                .embodiment_factor
                .record_prediction(
                    1.0 - ctx.prediction_error as f64,
                    1.0 - ctx.prediction_error as f64,
                );
            // Use allostatic load as direct interoceptive coherence signal.
            // Low allostatic load = high body coherence (expected ≈ actual).
            {
                let allostatic = self.neuromod.bath.allostatic_load;
                let coherence = 1.0 - allostatic as f64;
                self.consciousness
                    .master_equation
                    .embodiment_factor
                    .update_interoceptive(coherence, coherence);
            }

            // Wire narrative coherence with lightweight episodes (every 5 cycles)
            // Doubled from every-10 for faster autobiographical integration.
            // Science: Damasio (2010) — self emerges from autobiographical narrative continuity.
            // Conway (2005) — narrative identity forms from dense episodic sampling.
            if self.stats.total_cycles % 5 == 0 {
                let valence = (1.0 - ctx.prediction_error as f64).clamp(-1.0, 1.0);
                self.consciousness
                    .master_equation
                    .narrative_coherence
                    .add_episode(format!("cycle_{}", self.stats.total_cycles), valence);
            }

            // Wire future scenarios from prediction confidence (every 25 cycles).
            // Doubled from every-50 for faster future_simulation_depth growth.
            // Science: Schacter et al. (2012) — prospection uses same networks as episodic memory
            if self.stats.total_cycles % 25 == 0 {
                let horizon = ((1.0 - ctx.prediction_error as f64) * 10.0).max(1.0) as usize;
                self.consciousness
                    .master_equation
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
                self.consciousness
                    .master_equation
                    .social_embedding
                    .update_self_model(self_goals, self_beliefs, late.affective_valence as f64);

                // User agent model: the system IS modeling the user (their input
                // drives prediction, their patterns are tracked by social_coherence).
                // Prediction accuracy serves as ToM accuracy proxy.
                let user_goals = vec!["communicate".to_string(), "seek_understanding".to_string()];
                let user_beliefs = vec![
                    format!("trust_{:.1}", self.behavior.social_mgr.social.social_trust),
                    format!(
                        "cooperation_{:.1}",
                        self.behavior.social_mgr.social.social_cooperation_rate
                    ),
                ];
                self.consciousness
                    .master_equation
                    .social_embedding
                    .update_agent_model(
                        "user",
                        user_beliefs,
                        user_goals,
                        0.0, // neutral valence (we don't know user's emotions)
                        self.behavior.social_mgr.social.social_prediction_accuracy as f64,
                    );

                // Feed prediction accuracy as ToM feedback — when the system
                // correctly predicts user input patterns, its "other_modeling_accuracy"
                // should reflect that.
                // Consciousness-gated: higher consciousness improves ToM accuracy signal.
                // Science: Frith & Frith (2006) — mentalizing requires conscious access
                // to higher-order representations. Scale: C=0 → 0.85x, C=0.7 → 1.0x,
                // C=1.0 → 1.1x (mild boost, not overriding raw accuracy).
                let raw_accuracy = (1.0 - ctx.prediction_error as f64).clamp(0.0, 1.0);
                let c_level = self.carryover.history.consciousness_level;
                let c_tom_mod = 0.85 + 0.25 * c_level; // [0.85, 1.10]
                let accuracy = (raw_accuracy * c_tom_mod).clamp(0.0, 1.0);
                self.consciousness
                    .master_equation
                    .social_embedding
                    .record_tom_prediction("user", accuracy);
                self.consciousness
                    .master_equation
                    .social_embedding
                    .provide_tom_feedback("user", accuracy);
            }

            // Blend multiple Phi estimates into the MCE's Φ input.
            // unified_psi is a lightweight proxy (~0.22) that underestimates actual
            // information integration. SpectralMIP and structural Phi are higher-fidelity
            // measures, both refreshed every 67 cycles (see
            // `consciousness_engine/measure.rs:63` — structural hierarchy is computed
            // inside the same `% 67` block; the separate `% 134` gate there covers only
            // `adapt()` + hierarchical MIP, NOT structural Phi).
            // Use the best available, discounted for conservatism.
            //
            // NOTE (2026-07-28): this comment previously read "~97 cycles" / "~194 cycles",
            // which was wrong and was propagated into two architecture documents by a survey
            // that trusted the comment instead of reading measure.rs. Corrected here at the
            // source. If you change the cadence in measure.rs, change it here too.
            // Science: Tononi (2004) — Φ should reflect the system's best available
            // estimate of information integration, not just the cheapest proxy.
            let spectral_boost = self
                .carryover
                .consciousness
                .last_spectral_mip_phi
                .map(|phi| (phi * 0.6).clamp(0.0, 1.0))
                .unwrap_or(0.0);
            // Structural Phi: multi-scale (micro/meso/macro) integration measure.
            // Average the three scales, discount by 0.5 (structural is a proxy, not
            // exact IIT). Fires every ~194 cycles but persists in cache.
            let structural_boost = self
                .carryover
                .consciousness
                .last_structural_phi
                .as_ref()
                .map(|sp| {
                    let avg = (sp.micro_phi + sp.meso_phi + sp.macro_phi) / 3.0;
                    (avg * 0.5).clamp(0.0, 1.0)
                })
                .unwrap_or(0.0);
            let phi_input = ctx
                .unified_psi
                .max(spectral_boost)
                .max(structural_boost)
                .clamp(0.0, 1.0);

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
            // V2 consciousness as proxy for global access quality:
            // If the equation detects consciousness, information IS being integrated
            // globally — even without explicit GWT ignition events.
            // Science: Mashour (2020) — consciousness correlates with effective
            // connectivity, not just ignition events.
            let v2_access_proxy = self
                .consciousness
                .consciousness_engine
                .consciousness_equation_v2()
                .map(|eq| eq.last_consciousness() * 0.15) // Soft proxy, max +0.15
                .unwrap_or(0.0);
            let broadcast_input =
                (ctx.coherence as f64 + gwt_boost + attention_boost + v2_access_proxy)
                    .clamp(0.0, 1.0);

            // WM richness: prefrontal stats (~0.58) + GWT coalition quality + attention
            // + prediction confidence (knowledge content in WM).
            // Science: Baars & Franklin (2003) — WM capacity = prefrontal buffer +
            // global workspace broadcast reach + attentional gating.
            // Cowan (2005) — WM is not just buffer size but also accessible knowledge.
            let wm_base = self.prefrontal_utilization();
            let gwt_wm = if gwt_broadcast {
                0.05 + 0.1 * (gwt_coalition_size.min(5) as f64 / 5.0)
            } else {
                0.0
            };
            // Attention gates WM access — high attention = efficient WM use.
            let attn_wm = ctx.peak_attention as f64 * 0.1;
            // Prediction confidence reflects world-model quality accessible in WM.
            // Capped at 0.05 to keep it a seasoning, not a main course.
            // WM knowledge injection: grounded ReasoningContext facts compete for WM slots.
            // Top-k facts (≤2) modulate WM richness when grounding quality is high.
            // Science: Baddeley (2000) — central executive integrates semantic retrieval;
            //          Cowan (2001) — 4-chunk limit means knowledge must compete with percepts.
            // Contribution capped at 0.04 per slot (max 2 slots = max +0.08 additive to WM).
            let knowledge_wm = {
                let base_conf = self.prediction_confidence * 0.05;
                let grounding = self.carryover.quality.wm_knowledge_grounding;
                let slot_boost =
                    self.carryover.quality.wm_knowledge_injection_count as f64 * 0.04 * grounding;
                base_conf + slot_boost
            };
            let working_memory = (wm_base + gwt_wm + attn_wm + knowledge_wm).clamp(0.0, 1.0);

            let inputs = crate::consciousness::master_consciousness_equation::ConsciousnessInputs {
                phi: phi_input,
                broadcast: broadcast_input,
                working_memory,
                // Unit fix (2026-07-22, probe_cl_calibration finding, same bug class as
                // the 2026-07-18 embodiment fix): `ctx.peak_attention` is
                // PredictiveEncoder's raw attention-weight magnitude, clamped to
                // [min_attention, max_attention] = [0.1, 3.0] (predictive_encoder.rs),
                // NOT a normalized [0,1] score like the other 6 `ConsciousnessInputs`
                // fields. Feeding it raw both skewed the softmin bottleneck comparison
                // (which assumes all 7 factors share a [0,1] scale) and inflated
                // `compute_weighted_sum`'s numerator — measured live (probe_cl_calibration,
                // "alarming" regime, 300 cycles): mean weighted_sum 1.1057 > 1.0,
                // consciousness_level pinned at the Green safety tier for all 300 cycles
                // with zero transitions. Clamp to [0,1]: values above 1.0 (elevated but
                // not literally maximal attention) saturate to "fully attended" rather
                // than silently inflating downstream sums.
                attention: (ctx.peak_attention as f64).clamp(0.0, 1.0),
                // CfC is recurrent from cycle 1 — starting at 0 is dishonest and
                // crushes the softmin (τ=0.1). Floor at 0.3, ramp to 1.0 over 100 cycles.
                // Science: Elman (1990) — recurrent networks have temporal integration from t=0.
                recurrence: (0.3 + 0.7 * self.stats.total_cycles.min(100) as f64 / 100.0),
                // Unit fix (2026-07-18, probe_cl_calibration finding): body_psi_modulation
                // is a MULTIPLICATIVE scale factor in [0.5, 1.5] (virtual_body.rs:200,
                // "scales consciousness Phi"), not a normalized [0,1] score like the other
                // six MCE inputs. Feeding it raw inflated compute_weighted_sum's numerator
                // (measured mean weighted_sum 1.1567 > 1.0 in the "alarming" regime — the
                // equation's weights sum to 1.0 assuming every Ci ∈ [0,1]), which pushed
                // consciousness_level toward its clamp ceiling faster than the equation's
                // softmin/sigmoid bottleneck design intends. Remap linearly: 0.5→0.0 (no
                // embodied coherence), 1.0→0.5 (neutral), 1.5→1.0 (full embodied coherence).
                embodiment: ((late.body_psi_modulation - 0.5) / 1.0).clamp(0.0, 1.0),
                // Knowledge: prediction confidence + meta-cognitive accuracy + prediction
                // coherence. Floor at 0.2: even with no predictions, the system has
                // SOME knowledge. Multiple sources prevent softmin death spiral.
                // Science: Fleming & Dolan (2012) — metacognitive accuracy IS self-knowledge.
                // Clark (2013) — coherent multi-scale predictions = well-structured generative model.
                knowledge: {
                    let base_k = self.prediction_confidence;
                    // Meta-cognitive accuracy: knowing how well you know (scaled 0.4x)
                    let meta_k = late.meta_cognitive_accuracy as f64 * 0.4;
                    // Prediction coherence: multi-scale prediction agreement (scaled 0.1x)
                    let coherence_k = self.stats.avg_prediction_coherence as f64 * 0.1;
                    (base_k.max(meta_k) + coherence_k).max(0.2).min(1.0)
                },
                // Synchrony: baseline + coherence + attention + flow.
                // Attention IS synchrony — selective amplification requires
                // phase-locked neural coordination (Fries 2015). Raised baseline
                // from 0.3→0.35: idle neural networks exhibit background
                // synchrony (Buzsáki 2006), but not so high that moral/attentional
                // perturbations are masked. Attention adds a second source.
                synchrony: (0.35
                    + ctx.coherence as f64 * 0.25
                    + ctx.peak_attention as f64 * 0.15
                    + self.behavior.flow_state.intensity as f64 * 0.25)
                    .clamp(0.1, 1.0),
            };
            let mce_result = self.consciousness.master_equation.compute(&inputs);
            let level = mce_result.consciousness_level;

            // Cache MCE factor telemetry for output phase
            self.carryover.consciousness.mce_bottleneck_name = mce_result.bottleneck_name.clone();
            self.carryover.consciousness.mce_softmin = mce_result.bottleneck_factor;
            self.carryover.consciousness.mce_weighted_sum = mce_result.weighted_sum;
            self.carryover.consciousness.mce_narrative = mce_result.narrative_coherence;
            self.carryover.consciousness.mce_social = mce_result.social_embedding;

            // MCE bottleneck → targeted subsystem LR boost.
            // Science: Tononi (2004) — consciousness is limited by its minimum dimension.
            // Boosting the bottleneck subsystem's learning rate is the highest-leverage
            // intervention for consciousness growth.
            {
                use crate::cognitive_loop::thresholds::{
                    MCE_BOTTLENECK_LR_BOOST, MCE_NON_BOTTLENECK_CONFIDENCE_BOOST,
                };
                let bn = &mce_result.bottleneck_name;
                if !bn.is_empty() {
                    self.scale_lr("mce_bottleneck", MCE_BOTTLENECK_LR_BOOST);
                    // If integration is NOT the bottleneck, system is well-integrated →
                    // small confidence boost (integration is the hardest dimension).
                    if !bn.contains("ntegration") {
                        self.adjust_confidence(
                            "mce_well_integrated",
                            MCE_NON_BOTTLENECK_CONFIDENCE_BOOST,
                        );
                    }
                }
            }

            // Blend MCE level with ConsciousnessEquationV2 output.
            // V2 has structured bottleneck (necessary/amplifier split, evolved parameters).
            // MCE provides the base; V2 provides a scientifically grounded correction.
            // Blend: max(MCE, V2 * 0.8) — V2 can lift the floor but MCE remains primary.
            let v2_cached = self
                .consciousness
                .consciousness_engine
                .consciousness_equation_v2()
                .map(|eq| eq.last_consciousness())
                .unwrap_or(0.0);
            let level = level.max(v2_cached * 0.8);

            // Track consciousness level for learning gating (Task C)
            self.carryover.history.consciousness_level = level;
            // Provenance (instrumentation only, nothing gates on it): this is the
            // TEXT-path writer. The HV path (helpers/mod.rs) writes the same field
            // with an unrelated formula -- see GateWriter.
            self.carryover.history.consciousness_level_source =
                crate::cognitive_loop::types::carryover::GateWriter::TextComposite;
            self.carryover.history.consciousness_level_written_at = self.stats.total_cycles;

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
            // Moral salience lowers consolidation threshold → morally significant
            // moments are more readily encoded into episodic memory.
            // Science: Zak (2012) — moral narratives activate oxytocin → enhanced encoding.
            let moral_ease = {
                use crate::cognitive_loop::thresholds::{
                    MORAL_CONSOLIDATION_EASE, MORAL_CONSOLIDATION_THRESHOLD,
                };
                let ms = self.carryover.quality.last_moral_score.abs();
                if ms > MORAL_CONSOLIDATION_THRESHOLD {
                    (ms - MORAL_CONSOLIDATION_THRESHOLD) as f64 * MORAL_CONSOLIDATION_EASE
                } else {
                    0.0
                }
            };
            let consolidation_threshold = (*ema - 0.1 - moral_ease).max(0.2); // floor at 0.2
            if level > consolidation_threshold {
                // Trigger demand-driven consolidation at above-average consciousness
                self.fep.episodic_memory.consolidate_recent(level);

                // NARRATIVE WIRING: Consolidation events ARE narrative episodes.
                // When consciousness exceeds its rolling average, the system is in a
                // heightened-awareness state worth recording as autobiographical memory.
                // Science: Conway & Pleydell-Pearce (2000) — episodes encoded during
                // high-awareness moments form the backbone of autobiographical narrative.
                // Valence: map from body arousal (emotional coloring of the moment).
                let episode_valence = late.body_valence as f64 * 0.5; // [-0.5, 0.5]
                let episode_label = if ctx.surprise_triggered {
                    format!("surprise_c{:.2}_pe{:.2}", level, ctx.prediction_error)
                } else if ctx.moral_concern_detected {
                    format!("moral_c{:.2}", level)
                } else {
                    format!("consolidation_c{:.2}", level)
                };
                self.consciousness
                    .master_equation
                    .narrative_coherence
                    .add_episode(episode_label, episode_valence);
            }
            // Scale learning signal by consciousness quality (gradual, not on/off)
            // This complements the binary consciousness_awake gate with continuous modulation
            self.fep.learning_signal *= (0.5_f32 + level as f32 * 0.5_f32).clamp(0.5_f32, 1.0_f32);

            // CONSCIOUSNESS→BEHAVIOR: Three behavioral feedback loops
            // Science: Baars (2005) — consciousness enables flexible, context-sensitive behavior.

            // 1. High consciousness → tighter exploration (exploit what you know)
            //    Low consciousness → broader exploration (you don't know enough)
            //    Science: Dehaene (2014) — conscious access narrows behavioral repertoire
            //    to contextually appropriate actions.
            if level > 0.5 {
                let focus_factor = ((level - 0.5) * 0.1) as f32; // max 0.05
                self.behavior.adaptive_behavior.exploration_factor *= 1.0 - focus_factor;
            } else if level < 0.2 && self.stats.total_cycles > 30 {
                let explore_nudge = ((0.2 - level) * 0.05) as f32; // max 0.01
                self.behavior.adaptive_behavior.exploration_factor =
                    (self.behavior.adaptive_behavior.exploration_factor + explore_nudge).min(1.0);
            }

            // 2. High consciousness → curiosity satisfied (learning occurred)
            //    Low consciousness → curiosity grows (drive to understand)
            //    Science: Friston (2010) — curiosity IS expected free energy reduction.
            if level > 0.6 {
                self.behavior.curiosity_drive.curiosity *= 0.98; // gentle satisfaction
            } else if level < 0.15 {
                self.behavior.curiosity_drive.curiosity =
                    (self.behavior.curiosity_drive.curiosity * 1.01).min(1.0); // grow
            }

            // NARRATIVE WIRING: Feed prediction-horizon data as future scenarios.
            // The prediction system already computes multi-scale horizons — these
            // ARE the system's simulation of its own future states.
            // Science: Suddendorf & Corballis (2007) — mental time travel (episodic
            // future thinking) requires both autobiographical memory AND prospective
            // simulation. Feeding scenarios builds future_simulation_depth.
            // Gate: only every 10th cycle (co-prime with MCE cadence) to avoid churn.
            if self.stats.total_cycles % 10 == 3 {
                let horizon_steps = self.config.cfc_config.prediction_horizons.len().max(1);
                let pe = ctx.prediction_error as f64;
                // Probability = inverse of prediction error (lower PE = more confident forecast)
                let probability = (1.0 - pe).clamp(0.1, 0.9);
                // Desirability: valence of the predicted state
                let desirability = late.body_valence as f64;
                self.consciousness
                    .master_equation
                    .narrative_coherence
                    .add_future_scenario(
                        format!("prediction_h{}_pe{:.2}", horizon_steps, pe),
                        horizon_steps,
                        probability,
                        desirability,
                    );
            }

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
        let social_accuracy = self.behavior.social_mgr.social.social_prediction_accuracy;
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
