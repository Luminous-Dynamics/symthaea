// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Urgency computation, error pattern analysis, and parameter optimization.

use super::super::CognitiveLoopService;
use super::cycle_phases::{ParameterOptimizationResult, UrgencyResult};

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

        if let Some(ref replay) = self.memory.episodic_persistence.replay {
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
        // Session 11 Item 2: Wire hysteresis_factor into urgency computation.
        // Sustained stability relaxes hysteresis (factor decays from 1.0 toward 0.5),
        // permitting smoother mode transitions after long stable runs.
        // Science: Kelso (1995) — sustained stability permits relaxed mode boundaries.
        let hysteresis_relax = self.carryover.quality.hysteresis_factor;
        let base_hysteresis = match self.carryover.urgency.urgency {
            super::super::CycleUrgency::Cruise => {
                effective_threshold * (1.0 + 0.2 * hysteresis_relax) * flex_mod
            }
            super::super::CycleUrgency::Critical => {
                effective_threshold * (1.0 - 0.2 * hysteresis_relax) * flex_mod
            }
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
            // Proportional bias: slope magnitude drives threshold adjustment.
            // Clark (2013): predictive brain scales anticipatory response to trend strength.
            if slope > 0.02 {
                // Rising errors → lower threshold proportionally (easier to escalate)
                (1.0 - (slope - 0.02).min(0.15) * 0.67).max(0.85)
            } else if slope < -0.02 {
                // Falling errors → raise threshold proportionally (easier to de-escalate)
                (1.0 + (-slope - 0.02).min(0.15) * 0.67).min(1.15)
            } else {
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

        // Track sustained low coherence for forced escalation.
        // Science: Bar (2009) — persistent incoherence signals model failure needing reallocation.
        if prev_coherence > 0.0 && prev_coherence < 0.2 {
            self.carryover.urgency.consecutive_low_coherence += 1;
        } else {
            self.carryover.urgency.consecutive_low_coherence = 0;
        }

        // High transition cost → widen hysteresis band to resist unnecessary switching.
        // Kelso (1995): costly transitions justify stronger hysteresis.
        let transition_cost_mod = if self.stats.avg_transition_cost > 0.1 {
            1.0 + (self.stats.avg_transition_cost - 0.1).min(0.2) * 0.5 // up to +10%
        } else {
            1.0
        };
        // Session 9 Item 6: Consolidation → urgency de-escalation bias.
        // During memory consolidation, bias toward lower urgency for reduced interference.
        // Born & Wilhelm (2012): memory consolidation during rest requires reduced interference.
        let consolidation_mod = if self.is_consolidating {
            1.15 // Raise threshold by 15% → harder to escalate
        } else {
            1.0
        };
        let hysteresis_threshold =
            base_hysteresis * pattern_mod * coherence_mod * transition_cost_mod * consolidation_mod;
        let error_urgency = super::super::CycleUrgency::from_state(
            smoothed_urgency_error,
            hysteresis_threshold,
            surprise_triggered,
            self.carryover.urgency.consecutive_low_error,
        );

        // Session 15 Item 2: Early warning at 5+ consecutive low-coherence cycles.
        // Mild exploration boost + confidence dampening before the hard Critical at 10.
        // Science: Bar (2009) — moderate incoherence suggests model drift, not failure.
        let low_coherence_early_warning = self.carryover.urgency.consecutive_low_coherence >= 5
            && self.carryover.urgency.consecutive_low_coherence
                <= super::super::thresholds::LOW_COHERENCE_EXPLORATION_THRESHOLD;
        if low_coherence_early_warning {
            self.adjust_exploration("low_coherence_early", 0.02);
            self.scale_confidence("low_coherence_early", 0.98);
        }

        // Sustained low coherence (>10 cycles) forces Critical — model needs full reprocessing.
        // Also boost exploration: if the model is persistently confused, seek novel inputs.
        // Schmidhuber (2010): curiosity-driven learning from persistent prediction failure.
        //
        // GATE: If consciousness is high (Phi > 0.5), the system is genuinely integrated
        // despite low prediction coherence. Don't escalate to Critical — the low coherence
        // is a measurement artifact (e.g., CfC predictions aren't aligned with HDC compressed
        // space), not a genuine model failure. This prevents the "high consciousness but
        // stuck in Critical" trap observed in watch telemetry.
        let consciousness_level = self.carryover.history.consciousness_level;
        let error_urgency = if self.carryover.urgency.consecutive_low_coherence
            > super::super::thresholds::LOW_COHERENCE_EXPLORATION_THRESHOLD
            && consciousness_level < 0.5
        {
            self.adjust_exploration(
                "sustained_low_coherence",
                super::super::thresholds::LOW_COHERENCE_EXPLORATION_BOOST,
            );
            super::super::CycleUrgency::Critical
        } else {
            error_urgency
        };

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

        let (error_pattern, predicted_urgency, error_slope, oscillation_ratio) =
            if error_history.len() >= 4 {
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
                    "Falling" | "Stable" if self.carryover.urgency.consecutive_low_error > 15 => {
                        "Cruise"
                    }
                    "Falling" | "Stable" => "Normal",
                    _ => "Normal",
                };
                (pattern, predicted, slope, oscillation_ratio)
            } else {
                ("Warmup", "Normal", 0.0, 0.0)
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
                // De-escalating → hold old urgency for 1 cycle.
                // High transition cost → require stronger signal (hysteresis tightening).
                // Kelso (1995): costly transitions justify stronger hysteresis.
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

        // Track mode transition cost: PE spike in the cycle immediately after a transition.
        // Kelso (1995): attractor transitions incur transient processing costs.
        if self.carryover.urgency.mode_stability_counter == 1 {
            // Just settled into new mode — measure transition cost as current PE
            let cost = prediction_error.max(0.0);
            self.stats.avg_transition_cost = self.stats.avg_transition_cost * 0.8 + cost * 0.2;
        }

        // Sustained Critical urgency → LR boost.
        // If stuck in Critical for >5 cycles, boost learning rate to accelerate adaptation.
        // Science: Yerkes-Dodson (1908) — moderate arousal optimizes performance,
        // but sustained high arousal should increase plasticity.
        if matches!(urgency, super::super::CycleUrgency::Critical)
            && self.carryover.urgency.mode_stability_counter > 5
        {
            let sustained_boost =
                ((self.carryover.urgency.mode_stability_counter - 5) as f32 * 0.01).min(0.1);
            self.scale_lr("sustained_critical", 1.0 + sustained_boost);
        }

        // Oscillating error pattern → LR dampening to prevent chaotic weight updates.
        // Doya (2002): oscillating error signals indicate meta-uncertainty about the
        // learning target; reducing LR prevents gradient interference.
        if oscillation_ratio > super::super::thresholds::ERROR_OSCILLATION_BIFURCATION {
            // Session 16 Item 8: Very high oscillation → bifurcation point.
            // System is at a phase transition — freeze LR and boost exploration to find
            // new attractor basin. Kelso (1995): high oscillation near bifurcation.
            self.scale_lr(
                "error_bifurcation",
                super::super::thresholds::ERROR_OSCILLATION_BIFURCATION_LR,
            );
            self.scale_confidence(
                "error_bifurcation",
                super::super::thresholds::ERROR_OSCILLATION_BIFURCATION_LR,
            );
            self.adjust_exploration(
                "error_bifurcation",
                super::super::thresholds::ERROR_OSCILLATION_BIFURCATION_EXPLORE,
            );
        } else if oscillation_ratio > 0.5 {
            let osc_dampen = 1.0 - (oscillation_ratio - 0.5) * 0.2; // 0.9–1.0
            self.scale_lr("error_oscillation", osc_dampen.max(0.9));
            // Session 11 Item 6: Oscillation → confidence dampening.
            // Oscillating errors = meta-uncertainty about target → confidence should drop too.
            // Science: Doya (2002) — meta-uncertainty degrades both learning and confidence.
            self.scale_confidence("error_oscillation", osc_dampen.max(0.9));
        }

        // Session 10 Item 5: Error pattern → predictive learning rate.
        // Rising errors → increase LR (faster adaptation needed);
        // Falling → slow down (converging); Oscillating → dampen (meta-uncertainty).
        // Science: Schultz (2016) — error trend direction predicts optimal plasticity level.
        {
            use super::super::thresholds::{
                ERROR_PATTERN_FALLING_LR, ERROR_PATTERN_OSCILLATING_LR, ERROR_PATTERN_RISING_LR,
            };
            match error_pattern {
                "Rising" | "Spike" => {
                    self.scale_lr("error_pattern_rising", ERROR_PATTERN_RISING_LR);
                }
                "Falling" => {
                    self.scale_lr("error_pattern_falling", ERROR_PATTERN_FALLING_LR);
                }
                "Oscillating" => {
                    self.scale_lr("error_pattern_oscillating", ERROR_PATTERN_OSCILLATING_LR);
                }
                _ => {} // Stable / Warmup — no change
            }
        }

        // Session 10 Item 7: Mode stability → adaptive hysteresis relaxation.
        // After sustained stability (>20 cycles in same mode), gradually relax hysteresis
        // toward baseline, permitting smoother future transitions.
        // Science: Kelso (1995) — sustained stability permits relaxed mode boundaries.
        {
            use super::super::thresholds::{
                HYSTERESIS_RELAXATION_FLOOR, HYSTERESIS_RELAXATION_RATE,
                HYSTERESIS_RELAXATION_THRESHOLD,
            };
            if self.carryover.urgency.mode_stability_counter > HYSTERESIS_RELAXATION_THRESHOLD {
                self.carryover.quality.hysteresis_factor =
                    (self.carryover.quality.hysteresis_factor * HYSTERESIS_RELAXATION_RATE)
                        .max(HYSTERESIS_RELAXATION_FLOOR);
            } else if self.carryover.urgency.mode_stability_counter == 0 {
                // Mode just changed — reset hysteresis to full
                self.carryover.quality.hysteresis_factor = 1.0;
            }
        }

        // Session 15 Item 3: Sustained mode stability → dampen exploration.
        // 50+ cycles in same mode → system has settled → reduce exploration pressure.
        // Science: Friston (2010) — low free energy states require less active inference.
        if self.carryover.urgency.mode_stability_counter > 50 {
            self.scale_exploration("mode_stable", 0.98);
        }

        UrgencyResult {
            urgency,
            error_pattern,
            predicted_urgency,
            prediction_coherence_urgency_bias,
            error_slope,
            oscillation_ratio,
        }
    }
}
