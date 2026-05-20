// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::borrow::Cow;
use std::time::Instant;

use super::super::CognitiveLoopService;
use super::super::feedback_state::Priority;
use super::super::helpers;
use super::super::phase_results::PerceptionPhaseResult;
#[cfg(feature = "cpg")]
use super::super::thresholds::CPG_SYNC_TAU_FLOOR;
use super::super::thresholds::*;

impl CognitiveLoopService {
    /// Semantic memory lookup, CfC temporal step, multi-scale prediction,
    /// uncertainty decomposition, and hierarchical world model update.
    ///
    /// Computes tau-modulated CfC step, extracts predictions, decomposes
    /// epistemic/aleatoric uncertainty, and updates world model stiffness.
    pub(super) fn phase_dynamics_cfc_planning(
        &mut self,
        perception: &PerceptionPhaseResult,
        pre_update_coherence: f32,
        resonator_best_sim: f32,
        module_timings: &mut super::super::ModuleTimings,
    ) -> super::CfcPlanningResult {
        // ═══════════════════════════════════════════════════════════════════════
        // 2a. SEMANTIC MEMORY
        // ═══════════════════════════════════════════════════════════════════════
        let _t_core = Instant::now();
        let semantic_hdc: Cow<'_, [f32]> = self
            .temporal_network
            .project_to_hdc_vec(&perception.encoding.compressed_state)
            .map(Cow::Owned)
            .unwrap_or(Cow::Borrowed(&perception.encoding.compressed_state));
        let current_phi_for_lr = pre_update_coherence as f64;
        let mut semantic_lr_factor = self
            .memory
            .memory_consol
            .semantic_memory
            .compute_lr_factor_phi_weighted(
                &semantic_hdc,
                3,
                current_phi_for_lr,
                self.stats.total_cycles as u64,
            );
        module_timings.core_semantic_lookup = _t_core.elapsed().as_micros() as u64;

        // ── Phase 20: Epistemic gate → semantic memory LR bidirectionality ───
        let prev_epistemic = self.carryover.quality.last_epistemic_confidence;
        let epistemic_semantic_lr_mod: f32 =
            if prev_epistemic < EPISTEMIC_SEMANTIC_CAUTION_THRESHOLD && prev_epistemic > 0.0 {
                let caution = EPISTEMIC_SEMANTIC_CAUTION_BASE
                    + prev_epistemic * EPISTEMIC_SEMANTIC_CAUTION_SCALE;
                semantic_lr_factor *= caution;
                self.stats.epistemic_semantic_mod_count += 1;
                caution - 1.0
            } else if prev_epistemic > EPISTEMIC_SEMANTIC_BOOST_THRESHOLD {
                let boost = 1.0_f32
                    + (prev_epistemic - EPISTEMIC_SEMANTIC_BOOST_THRESHOLD)
                        * EPISTEMIC_SEMANTIC_BOOST_SCALE;
                semantic_lr_factor *= boost;
                self.stats.epistemic_semantic_mod_count += 1;
                boost - 1.0
            } else {
                0.0
            };

        // 2b. Physics bridge: blend physics-informed HDC into compressed state.
        // Only clone when physics-bridge is active AND integration exists;
        // otherwise borrow directly to skip a ~1KB Vec allocation per cycle.
        #[cfg(feature = "physics-bridge")]
        let _compressed_owned;
        #[cfg(feature = "physics-bridge")]
        let compressed_for_cfc: &[f32] =
            if let Some(ref mut physics) = self.feature_integ.physics_integration {
                let mut buf = perception.encoding.compressed_state.clone();
                physics.query_cycle(
                    self.stats.total_cycles,
                    self.config.physics_bridge_query_interval,
                    self.config.physics_bridge_blend_weight,
                    self.substrate_manager.tau_factor,
                    self.substrate_manager.scale_pressure,
                    &perception.encoding.hv16_cached,
                    &mut buf,
                );
                _compressed_owned = buf;
                &_compressed_owned
            } else {
                &perception.encoding.compressed_state
            };
        #[cfg(not(feature = "physics-bridge"))]
        let compressed_for_cfc: &[f32] = &perception.encoding.compressed_state;

        // 3. Copy into pre-allocated ndarray buffer for CfC (avoids per-cycle heap alloc).
        // We take() the buffer, fill it, and put it back after use to satisfy the
        // borrow checker (get_multi_scale_prediction takes &mut self).
        let mut input_array =
            std::mem::replace(&mut self.cfc_input_buffer, ndarray::Array1::zeros(0));
        if let Some(buf) = input_array.as_slice_mut() {
            let len = compressed_for_cfc.len().min(buf.len());
            buf[..len].copy_from_slice(&compressed_for_cfc[..len]);
            // Zero any trailing elements if buffer is larger
            for v in &mut buf[len..] {
                *v = 0.0;
            }
        }

        // 4. Step CfC forward with current input
        let resonance_tau_factor = if self.carryover.history.resonance_frequency > 0.0 {
            let deviation = (self.carryover.history.resonance_frequency as f32
                - RESONANCE_TAU_CENTER as f32)
                .clamp(-0.5, 0.5);
            1.0 - (deviation * RESONANCE_TAU_SCALE)
        } else {
            1.0
        };
        let arousal_tau_factor =
            if (self.carryover.history.body_arousal - 0.5).abs() > AROUSAL_TAU_DEADZONE {
                1.0 + (self.carryover.history.body_arousal - 0.5) * AROUSAL_TAU_SENSITIVITY
            } else {
                1.0
            };
        let codebook_tau_factor = if resonator_best_sim > CODEBOOK_FAMILIAR_THRESHOLD {
            1.0 - (resonator_best_sim - CODEBOOK_FAMILIAR_THRESHOLD) * CODEBOOK_FAMILIAR_TAU_SCALE
        } else if resonator_best_sim > 0.0 && resonator_best_sim < CODEBOOK_NOVEL_THRESHOLD {
            1.0 + (CODEBOOK_NOVEL_THRESHOLD - resonator_best_sim) * CODEBOOK_NOVEL_TAU_SCALE
        } else {
            1.0
        };
        let arousal_recovery_tau_factor;
        let arousal_recovery_active;
        if self.carryover.urgency.arousal_trap_counter > AROUSAL_TRAP_RECOVERY_MIN_CYCLES {
            // Recovery intensity ramps from 0→1 over the ramp window, then stays at 1.0.
            // BUG FIX: Previously capped at counter=10, leaving extended traps unassisted.
            let recovery_intensity = ((self.carryover.urgency.arousal_trap_counter
                - AROUSAL_TRAP_RECOVERY_MIN_CYCLES) as f32
                / AROUSAL_TRAP_RECOVERY_RAMP_CYCLES)
                .min(1.0);
            arousal_recovery_tau_factor = 1.0 + recovery_intensity * AROUSAL_RECOVERY_TAU_SCALE;
            arousal_recovery_active = true;
        } else {
            arousal_recovery_tau_factor = 1.0;
            arousal_recovery_active = false;
        }

        // FEP surprise → CfC time constant modulation.
        // Friston (2010): high surprise (free energy) accelerates inference dynamics;
        // low surprise allows consolidation via slower dynamics.
        // Factor: [0.8, 1.2] — moderate modulation to prevent instability.
        let fep_tau_factor = if let Some(ref fe) = self.fep.agent.last_fe_components {
            let surprise_norm = (fe.surprise as f32).clamp(0.0, 2.0) / 2.0; // [0, 1]
            1.0 - surprise_norm * FEP_SURPRISE_TAU_SCALE // high surprise → 0.8 (faster), low → 1.0
        } else {
            1.0
        };

        // ODE trajectory planning: simulate forward trajectories via Dormand-Prince
        // to compute expected free energy over future horizons.
        // Friston (2010): genuine active inference requires planning through simulation.
        // The trajectory surprise augments the FEP tau factor for more informed dynamics.
        let fep_tau_factor = if let Some(_best_action) =
            self.fep.plan_trajectories(self.stats.total_cycles as u64)
        {
            let traj_surprise = self.fep.trajectory_telemetry.best_trajectory_surprise as f32;
            let traj_surprise_norm = traj_surprise.clamp(0.0, 2.0) / 2.0;
            // Blend trajectory surprise into tau: augments single-step FEP surprise
            fep_tau_factor * (1.0 - traj_surprise_norm * 0.1) // ±10% modulation
        } else {
            fep_tau_factor
        };

        // Session 10 Item 3: Coherence velocity tau factor.
        // Session 11 Item 3: Gate behind cycle > 5 to avoid spurious velocity from default init.
        let coherence_velocity_tau_factor = {
            let cv = self.carryover.quality.coherence_velocity;
            if self.stats.total_cycles > RESONATOR_STARTUP_CYCLES
                && cv > COHERENCE_VELOCITY_TAU_THRESHOLD
            {
                COHERENCE_VELOCITY_TAU_BOOST
            } else if self.stats.total_cycles > RESONATOR_STARTUP_CYCLES
                && cv < -COHERENCE_VELOCITY_TAU_THRESHOLD
            {
                COHERENCE_VELOCITY_TAU_DAMPEN
            } else {
                1.0
            }
        };

        // Prediction horizon → CfC temporal integration depth.
        // Clark (2013): high PE → contract horizons (focus near-term);
        // low PE → expand horizons (exploit stability for planning).
        // This complements FEP surprise tau (Friston 2010) — they work in synergy:
        // FEP surprise drives fast dynamics, horizon scale drives planning depth.
        let prediction_horizon_tau = {
            let pe = self.stats.avg_prediction_error.clamp(0.0, 1.0);
            let pe_scale = if pe > HORIZON_PE_CONTRACT_THRESHOLD {
                1.0 - (pe - HORIZON_PE_CONTRACT_THRESHOLD) * HORIZON_PE_CONTRACT_RATE
            } else if pe < HORIZON_PE_EXPAND_THRESHOLD {
                1.0 + (HORIZON_PE_EXPAND_THRESHOLD - pe) * HORIZON_PE_EXPAND_RATE
            } else {
                1.0
            };
            let slope = perception.urgency.error_slope;
            let slope_scale = if slope > HORIZON_SLOPE_THRESHOLD {
                1.0 - (slope - HORIZON_SLOPE_THRESHOLD).min(HORIZON_SLOPE_CONTRACT_CAP)
                    * HORIZON_SLOPE_CONTRACT_RATE
            } else if slope < -HORIZON_SLOPE_THRESHOLD {
                1.0 + (-slope - HORIZON_SLOPE_THRESHOLD).min(HORIZON_SLOPE_EXPAND_CAP)
                    * HORIZON_SLOPE_EXPAND_RATE
            } else {
                1.0
            };
            (pe_scale * slope_scale)
                .clamp(PREDICTION_HORIZON_MIN_SCALE, PREDICTION_HORIZON_MAX_SCALE)
        };

        // 10th factor: Thermal bridge — platform heat → CfC slowdown.
        // Science: Angilletta (2009) thermal performance curves.
        let thermal_tau_factor = self.sensorimotor.thermal_bridge.signals().tau_factor as f32;

        // 11th factor: Neuroevolution champion τ — evolved tau_base ratio.
        // When neuroevolution discovers a better tau_base, blend it toward the
        // live CfC dynamics. Ratio >1 = evolved organism prefers slower dynamics.
        // Science: Hasani et al. (2021) — τ is the primary CfC evolvable.
        #[cfg(feature = "neuroevolution")]
        let neuroevo_tau_factor = {
            let champ = self.neuroevolution_manager.champion_suggestion();
            if champ.active {
                // Blend: 90% default + 10% evolved ratio (conservative)
                let evolved_ratio = champ.tau_base / NEUROEVO_DEFAULT_TAU_BASE;
                let blended =
                    NEUROEVO_BLEND_DEFAULT_WEIGHT + NEUROEVO_BLEND_EVOLVED_WEIGHT * evolved_ratio;
                blended.clamp(NEUROEVO_TAU_CLAMP_MIN, NEUROEVO_TAU_CLAMP_MAX)
            } else {
                1.0
            }
        };
        #[cfg(not(feature = "neuroevolution"))]
        let neuroevo_tau_factor: f32 = 1.0;

        // 12th factor: CPG oscillation gating — desynchronized oscillators slow dynamics.
        // sync_index=1.0 → tau=1.0 (no change), sync_index=0.0 → tau=CPG_SYNC_TAU_FLOOR.
        // Gated behind warmup to avoid spurious boost from initial phase presets.
        // Science: Buzsáki (2006) — neural oscillation synchrony gates integration rate.
        #[cfg(feature = "cpg")]
        let tau_cpg = {
            if self.stats.total_cycles > DYNAMICS_STARTUP_WARMUP_CYCLES {
                let sync = self.cpg_manager.sync_index() as f32;
                let sync_clamped = sync.clamp(0.0, 1.0);
                (CPG_SYNC_TAU_FLOOR + (1.0 - CPG_SYNC_TAU_FLOOR) * sync_clamped)
                    .clamp(CPG_TAU_CLAMP_MIN, CPG_TAU_CLAMP_MAX)
            } else {
                1.0
            }
        };
        #[cfg(not(feature = "cpg"))]
        let tau_cpg: f32 = 1.0;

        // 13th factor: Phi → tau feedback — SpectralMIP Phi modulates CfC speed.
        // Higher integrated information → faster temporal dynamics, closing the
        // causal loop between consciousness measurement and the dynamics that produce it.
        // Uses previous cycle's Phi (from carryover) to avoid circular dependency.
        // Science: Tononi (2004) — Phi as causally efficacious, not epiphenomenal.
        let phi_tau_factor: f32 = if self.config.enable_phi_tau_feedback {
            if let Some(phi) = self.carryover.consciousness.last_spectral_mip_phi {
                // Sigmoid normalization: map Phi ∈ [0, ∞) → (0, 1) centered at reference
                let normalized = 1.0
                    / (1.0
                        + (-(phi - super::super::thresholds::PHI_TAU_REFERENCE)
                            * super::super::thresholds::PHI_TAU_SIGMOID_STEEPNESS)
                            .exp());
                // Linear map sigmoid output [0, 1] → [floor, ceiling]
                let floor = super::super::thresholds::PHI_TAU_FLOOR;
                let ceiling = super::super::thresholds::PHI_TAU_CEILING;
                floor + (ceiling - floor) * normalized as f32
            } else {
                // No Phi yet (warmup) — neutral
                1.0
            }
        } else {
            1.0
        };

        let delta_t = self.config.cfc_config.delta_t
            * resonance_tau_factor
            * arousal_tau_factor
            * codebook_tau_factor
            * arousal_recovery_tau_factor
            * fep_tau_factor
            * coherence_velocity_tau_factor
            * prediction_horizon_tau
            * self
                .sensorimotor
                .somatic_bridge
                .to_interoceptive_signals()
                .tau_slowdown_factor as f32
            * self.substrate_manager.tau_factor
            * thermal_tau_factor
            * neuroevo_tau_factor
            * tau_cpg
            * phi_tau_factor;
        let _t_core = Instant::now();
        if let Err(e) = self.temporal_network.step(&input_array, delta_t) {
            tracing::warn!(error = %e, "CfC temporal step failed — continuing with stale state");
        }

        // Phase 3: Scale-limited CfC hidden state masking.
        // When substrate has fewer computational units than biological (negative
        // scale_pressure), mask out a fraction of hidden state dimensions.
        // Science: Berry & Srivastava (2018) — HDC capacity ~ D^(5/3).
        if self.config.enable_substrate_encoding_noise {
            let frac = self.substrate_manager.effective_dim_fraction();
            if frac < 1.0 {
                match self.temporal_network.read_state() {
                    Ok(mut state) => {
                        let mask_start = (frac * state.len() as f32) as usize;
                        for h in state.as_slice_mut().unwrap_or(&mut [])[mask_start..].iter_mut() {
                            *h = 0.0;
                        }
                        if let Err(e) = self.temporal_network.inject(&state) {
                            tracing::warn!(err = %e, "substrate mask inject failed");
                        }
                    }
                    _ => {
                        tracing::warn!(
                            "CfC read_state failed during substrate mask — skipping mask"
                        );
                    }
                }
            }
        }

        // ── Spectral entropy → CfC hidden state masking (Phase B) ───────────────
        // High spectral entropy means the CfC dynamics are too broadband — mask
        // out a fraction of dimensions to force focused processing.
        // Science: Buzsáki (2006) — broadband entropy constrains processing depth.
        #[cfg(feature = "spectral_state")]
        if self.config.enable_substrate_encoding_noise {
            let spectral_entropy = self.spectral_manager.telemetry().spectral_entropy;
            if spectral_entropy > super::super::thresholds::SPECTRAL_ENTROPY_THRESHOLD {
                let overflow = (spectral_entropy
                    - super::super::thresholds::SPECTRAL_ENTROPY_THRESHOLD)
                    / super::super::thresholds::SPECTRAL_ENTROPY_THRESHOLD;
                // spectral_frac: 1.0 at threshold, MASK_FLOOR at 2× threshold
                let spectral_frac = (1.0 - overflow as f32)
                    .max(super::super::thresholds::SPECTRAL_ENTROPY_MASK_FLOOR);
                // Don't over-mask: use the maximum of substrate and spectral fractions
                let substrate_frac = self.substrate_manager.effective_dim_fraction();
                let frac = substrate_frac.max(spectral_frac);
                if frac < 1.0 {
                    match self.temporal_network.read_state() {
                        Ok(mut state) => {
                            let mask_start = (frac * state.len() as f32) as usize;
                            for h in
                                state.as_slice_mut().unwrap_or(&mut [])[mask_start..].iter_mut()
                            {
                                *h = 0.0;
                            }
                            if let Err(e) = self.temporal_network.inject(&state) {
                                tracing::warn!(err = %e, "spectral entropy mask inject failed");
                            }
                        }
                        _ => {
                            tracing::warn!(
                                "CfC read_state failed during spectral entropy mask — skipping mask"
                            );
                        }
                    }
                }
            }
        }

        module_timings.core_cfc_step = _t_core.elapsed().as_micros() as u64;

        // 5. Get multi-scale predictions
        let _t_core = Instant::now();
        let (prediction, raw_predictions) = self.get_multi_scale_prediction(&input_array);

        // 5a. JEPA: parallel latent-space prediction alongside CfC.
        // Uses the CfC input as "current state" and the multi-scale prediction
        // as "next state" approximation. The JEPA predictor learns to anticipate
        // the target encoder's representation of the next state — cheaper than
        // full observation-space prediction (128D latent vs full CfC dimension).
        #[cfg(feature = "jepa")]
        if let Some(ref mut jepa) = self.jepa_engine {
            let jepa_dim = jepa.config().input_dim;
            // Normalize both vectors to JEPA's input_dim (pad with zeros or truncate)
            let mut current_vec = input_array.to_vec();
            current_vec.resize(jepa_dim, 0.0);
            let current_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(current_vec);
            let mut next_vec = prediction.clone();
            next_vec.resize(jepa_dim, 0.0);
            let next_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(next_vec);

            // Use last cycle's FEP action (stored on fep module after each FEP step).
            // CfC planning runs before this cycle's FEP, so we use the previous action.
            // This is correct: JEPA predicts "given what I did last, what state am I in now?"
            let action = self.fep.last_action_idx;
            let lr = self.config.cfc_config.learning_rate;

            // Train step: forward + backward + EMA update (inline — latent ops are cheap)
            let _jepa_loss = jepa.train_step(&current_hv, &next_hv, action, lr);

            // Track energy cost in substrate manager
            self.substrate_manager.jepa_energy += jepa.config().energy_cost_per_forward;
        }

        // Return the buffer to CLS for reuse next cycle (zero-alloc swap)
        self.cfc_input_buffer = input_array;

        let prediction_coherence = if self.stats.total_cycles
            % super::super::thresholds::PREDICTION_COHERENCE_INTERVAL
            == 0
        {
            let coh = Self::compute_prediction_coherence_from_cache(&raw_predictions);
            self.stats.avg_prediction_coherence = self.stats.avg_prediction_coherence
                * COHERENCE_PREDICTION_EMA
                + coh * (1.0 - COHERENCE_PREDICTION_EMA);
            if coh < COHERENCE_LOW_THRESHOLD {
                let coh_dampen = (COHERENCE_LOW_THRESHOLD - coh) * COHERENCE_LOW_DAMPEN_SCALE;
                self.scale_confidence("pred_coherence_low", 1.0 - coh_dampen);
            }
            if coh > COHERENCE_HIGH_THRESHOLD {
                let coh_boost = (coh - COHERENCE_HIGH_THRESHOLD) * COHERENCE_CONFIDENCE_BOOST;
                self.adjust_confidence("pred_coherence_high", coh_boost);
            }
            coh
        } else {
            self.stats.avg_prediction_coherence
        };

        // 5b. Epistemic vs aleatoric uncertainty decomposition.
        // Epistemic (model uncertainty): prediction disagreement across horizons — reducible
        // by exploration. Aleatoric (data noise): mean per-horizon prediction variance — not
        // reducible. Only epistemic uncertainty should drive exploration.
        // Depeweg et al. (2018): decomposing uncertainty for active learning.
        let (epistemic_uncertainty, aleatoric_uncertainty) = if raw_predictions.len() >= 2 {
            // Epistemic ≈ 1 - cross-horizon coherence (disagreement = model uncertainty)
            let epistemic = (1.0 - prediction_coherence).clamp(0.0, 1.0);

            // Aleatoric ≈ mean within-dimension variance across predictions
            // Use min length across all prediction vectors — HierarchicalCfC can produce
            // jagged vectors, and indexing by [0].len() would panic on shorter ones.
            let dim = raw_predictions
                .iter()
                .map(|p| p.len())
                .min()
                .unwrap_or(0)
                .max(1);
            let n = raw_predictions.len() as f32;
            let mut mean_var = 0.0f32;
            for d in 0..dim {
                let mean: f32 = raw_predictions.iter().map(|p| p[d]).sum::<f32>() / n;
                let var: f32 = raw_predictions
                    .iter()
                    .map(|p| (p[d] - mean).powi(2))
                    .sum::<f32>()
                    / n;
                mean_var += var;
            }
            let aleatoric_raw = mean_var / dim as f32;
            let aleatoric = if aleatoric_raw.is_finite() {
                aleatoric_raw.sqrt().clamp(0.0, 1.0)
            } else {
                0.0
            };
            (epistemic, aleatoric)
        } else {
            (EPISTEMIC_UNCERTAINTY_DEFAULT, ALEATORIC_UNCERTAINTY_DEFAULT) // defaults when insufficient data
        };

        // Only epistemic uncertainty drives exploration (aleatoric is irreducible noise).
        // Use smoothed epistemic for stability; raw for responsiveness on first cycle.
        // Depeweg et al. (2018): decomposing uncertainty for active learning.
        let smoothed_eu = self.carryover.quality.smoothed_epistemic_uncertainty;
        let eu_for_exploration = if smoothed_eu > 0.0 {
            smoothed_eu
        } else {
            epistemic_uncertainty
        };
        if eu_for_exploration > EPISTEMIC_EXPLORE_THRESHOLD
            && self.stats.total_cycles % super::super::thresholds::EPISTEMIC_MODULATION_INTERVAL
                == 0
        {
            let mut epistemic_explore =
                (eu_for_exploration - EPISTEMIC_EXPLORE_THRESHOLD) * EPISTEMIC_EXPLORE_SCALE;
            // Oscillation + high uncertainty = confused AND unstable → stronger exploration.
            // Doya (2002) + Schmidhuber (2010): compound uncertainty warrants aggressive search.
            if perception.urgency.oscillation_ratio > EPISTEMIC_OSCILLATION_THRESHOLD {
                epistemic_explore *= EPISTEMIC_OSCILLATION_MULTIPLIER;
            }
            self.adjust_exploration("epistemic_uncertainty", epistemic_explore);
        } else if eu_for_exploration < EPISTEMIC_LOW_THRESHOLD
            && self.stats.total_cycles % super::super::thresholds::EPISTEMIC_MODULATION_INTERVAL
                == 0
        {
            // Low epistemic uncertainty → dampen exploration (model is confident).
            self.adjust_exploration("epistemic_low", -EPISTEMIC_LOW_DAMPEN);
        }

        // 6. Get current CfC state as output
        let output = self
            .temporal_network
            .read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|e| {
                tracing::warn!(err = %e, "CfC read_state failed in output read — using zero state");
                vec![0.0; self.config.cfc_config.num_neurons]
            });
        module_timings.core_predict = _t_core.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // 6b. World Model
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        self.fep
            .world_model
            .update_sensory(&perception.encoding.compressed_state);

        // Incorporate causal structure into world model (every 41 cycles, co-prime).
        // Pearl (2009): causal knowledge provides structural priors beyond correlation.
        if self.stats.total_cycles % super::super::thresholds::CAUSAL_STRUCTURE_INTERVAL == 0 {
            if let Some(ref enhancer) = self.memory.causal_enhancer {
                let graph = enhancer.current_graph();
                if !graph.is_empty() {
                    let edges: Vec<(usize, usize, f32)> = graph
                        .edges
                        .iter()
                        .map(|e| (e.from, e.to, e.strength as f32))
                        .collect();
                    self.fep.world_model.incorporate_causal_structure(&edges);
                }
            }
        }

        let wm_stiffness = self.fep.world_model.avg_error.clamp(0.0, 1.0);
        if self.stats.total_cycles > DYNAMICS_POST_BOOT_CYCLES {
            if wm_stiffness > WORLD_MODEL_STIFFNESS_THRESHOLD {
                let stiffness_nudge = (wm_stiffness - WORLD_MODEL_STIFFNESS_THRESHOLD)
                    * WORLD_MODEL_STIFFNESS_LR_SCALE;
                self.adjust_lr_pri("wm_stiff", stiffness_nudge, Priority::Homeostatic);
            } else if wm_stiffness < WORLD_MODEL_SPONGINESS_THRESHOLD {
                let spongy_dampen =
                    (WORLD_MODEL_SPONGINESS_THRESHOLD - wm_stiffness) * WORLD_MODEL_SPONGY_LR_SCALE;
                self.scale_lr_pri("wm_spongy", 1.0 - spongy_dampen, Priority::Homeostatic);
            }
        }

        let level_errors = self.fep.world_model.level_errors();
        let mut wm_sensory_mismatch = false;
        if level_errors.len() >= 2 && self.stats.total_cycles > DYNAMICS_STARTUP_WARMUP_CYCLES {
            let sensory_error = level_errors[0];
            let abstract_error = level_errors[level_errors.len() - 1];
            if abstract_error
                > sensory_error * super::super::thresholds::WORLD_MODEL_CONFUSION_RATIO
                && abstract_error > super::super::thresholds::WORLD_MODEL_ERROR_FLOOR
            {
                self.adjust_exploration_pri(
                    "conceptual_confusion",
                    super::super::thresholds::CONCEPTUAL_CONFUSION_EXPLORATION,
                    Priority::Homeostatic,
                );
            }
            wm_sensory_mismatch = sensory_error
                > abstract_error * super::super::thresholds::WORLD_MODEL_MISMATCH_RATIO
                && sensory_error > super::super::thresholds::WORLD_MODEL_ERROR_FLOOR;
        }
        module_timings.world_model = _t.elapsed().as_micros() as u64;

        // Convert semantic_hdc to owned Vec for the caller
        let semantic_hdc_owned = semantic_hdc.into_owned();

        super::CfcPlanningResult {
            semantic_hdc: semantic_hdc_owned,
            semantic_lr_factor,
            epistemic_semantic_lr_mod,
            delta_t,
            output,
            prediction,
            prediction_coherence,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            wm_sensory_mismatch,
            fep_tau_factor,
            phi_tau_factor,
            prediction_horizon_tau,
            arousal_recovery_active,
            arousal_recovery_tau_factor,
        }
    }
}
