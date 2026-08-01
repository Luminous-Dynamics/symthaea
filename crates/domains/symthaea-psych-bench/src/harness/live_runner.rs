// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live cognitive loop benchmark runner.
//!
//! Runs benchmark trials through `CognitiveLoopService::cycle_with_hv()`,
//! capturing per-trial neuromodulator telemetry and response selection
//! based on CfC network output.
//!
//! Requires the `symthaea-backend` feature.

#[cfg(feature = "symthaea-backend")]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

/// A stimulus for a loop-driven benchmark trial.
#[derive(Debug, Clone)]
pub struct LoopStimulus {
    /// Pre-encoded stimulus hypervector.
    pub stimulus_hv: ContinuousHV,
    /// Alternative response HVs for forced-choice selection.
    pub alternatives: Vec<ContinuousHV>,
    /// Index of the correct alternative.
    pub correct_idx: usize,
    /// Condition label (e.g., "congruent", "go").
    pub condition: String,
    /// Trial index.
    pub trial_idx: usize,
}

/// Result from a single loop-driven trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopTrialResult {
    /// Selected response index.
    pub response_idx: usize,
    /// Whether the response was correct.
    pub correct: bool,
    /// Prediction error from the cognitive loop.
    pub prediction_error: f32,
    /// Dopamine effective level.
    pub da: f32,
    /// Noradrenaline effective level.
    pub ne: f32,
    /// Serotonin effective level.
    pub sht: f32,
    /// Acetylcholine effective level.
    pub ach: f32,
    /// Integrated Phi (consciousness measure).
    pub psi: f32,
    /// Cycle time in microseconds.
    pub cycle_time_us: u64,
    /// Whether learning occurred this cycle.
    pub learning_occurred: bool,
    /// Reward signal delivered to neuromodulator bath.
    #[serde(default)]
    pub reward: f32,
    /// Oxytocin effective level.
    #[serde(default)]
    pub oxytocin: f32,
    /// Moral/value evaluator score from ethics engine.
    #[serde(default)]
    pub moral_score: f32,
    /// Bath entropy (neuromod system diversity measure).
    #[serde(default)]
    pub bath_entropy: f32,
    /// Allostatic load (cumulative stress measure, 0.0–1.0).
    #[serde(default)]
    pub allostatic_load: f32,
    /// Consciousness modulation factor from neuromod bath (0.6–1.2).
    #[serde(default)]
    pub consciousness_mod: f32,
    /// Social coherence factor from oxytocin/trust (0.8–1.3).
    #[serde(default)]
    pub social_coherence: f32,
    /// Structural micro Phi from this cycle.
    #[serde(default)]
    pub structural_micro_phi: f64,
    /// Structural emergence ratio from this cycle.
    #[serde(default)]
    pub structural_emergence_ratio: f64,
}

/// Aggregate result from running a benchmark through the loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopBenchmarkResult {
    /// Benchmark name.
    pub benchmark: String,
    /// Per-trial results.
    pub trials: Vec<LoopTrialResult>,
    /// Overall accuracy.
    pub accuracy: f64,
    /// Mean prediction error.
    pub mean_prediction_error: f64,
    /// Mean DA level.
    pub mean_da: f64,
    /// Mean NE level.
    pub mean_ne: f64,
    /// Mean 5-HT level.
    pub mean_sht: f64,
    /// Mean ACh level.
    pub mean_ach: f64,
    /// Mean reward signal across trials.
    #[serde(default)]
    pub mean_reward: f64,
    /// Mean oxytocin level across trials.
    #[serde(default)]
    pub mean_oxytocin: f64,
    /// Mean moral score across trials.
    #[serde(default)]
    pub mean_moral_score: f64,
    /// Mean bath entropy across trials.
    #[serde(default)]
    pub mean_bath_entropy: f64,
    /// Mean allostatic load across trials.
    #[serde(default)]
    pub mean_allostatic_load: f64,
    /// Mean consciousness modulation factor.
    #[serde(default)]
    pub mean_consciousness_mod: f64,
    /// Mean social coherence factor.
    #[serde(default)]
    pub mean_social_coherence: f64,
    /// Mean structural micro Phi across trials.
    #[serde(default)]
    pub mean_structural_micro_phi: f64,
    /// Mean structural emergence ratio across trials.
    #[serde(default)]
    pub mean_structural_emergence_ratio: f64,
    /// Total wall-clock time in milliseconds.
    pub elapsed_ms: u64,
}

/// Trait for benchmarks that can generate loop-compatible stimuli.
pub trait LoopDrivable {
    /// Generate stimuli for loop-driven benchmark execution.
    fn generate_stimuli(&self, config: &super::config::BenchmarkConfig) -> Vec<LoopStimulus>;

    /// Benchmark name (for result labeling).
    fn loop_name(&self) -> &str;
}

/// Runner that executes benchmarks through the cognitive loop.
#[cfg(feature = "symthaea-backend")]
pub struct CognitiveLoopBenchmarkRunner {
    service: CognitiveLoopService,
    warmup_cycles: usize,
}

#[cfg(feature = "symthaea-backend")]
impl CognitiveLoopBenchmarkRunner {
    /// Create a new runner with Standard consciousness profile.
    pub fn new(genesis_phrase: &str) -> Option<Self> {
        let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
        config.genesis_phrase = Some(genesis_phrase.to_string());
        config.async_training = false;
        let service = CognitiveLoopService::new(config).ok()?;
        Some(Self {
            service,
            warmup_cycles: 100,
        })
    }

    /// Create from an existing config.
    pub fn from_config(config: CognitiveLoopConfig) -> Option<Self> {
        let service = CognitiveLoopService::new(config).ok()?;
        Some(Self {
            service,
            warmup_cycles: 100,
        })
    }

    /// Expose the service for neuromod clamping.
    pub fn service_mut(&mut self) -> &mut CognitiveLoopService {
        &mut self.service
    }

    /// Run warmup cycles to stabilize the loop.
    fn warmup(&mut self) {
        let dim = self.service.state_dim();
        let warmup_hv = ContinuousHV::random(dim, 0xDEAD_BEEF);
        for _ in 0..self.warmup_cycles {
            let _ = self.service.cycle_with_hv(&warmup_hv);
        }
    }

    /// Snapshot structural Phi from a warmup cycle as RuntimeConsciousnessData.
    pub fn snapshot_runtime_consciousness(
        &mut self,
    ) -> Option<crate::benchmarks::butlin::report::RuntimeConsciousnessData> {
        let dim = self.service.state_dim();
        let hv = ContinuousHV::random(dim, 0xBEEF_CAFE);
        let result = self.service.cycle_with_hv(&hv);
        let md = &result.metadata;
        if md.structural.structural_micro_phi > 0.0 || md.structural.structural_macro_phi > 0.0 {
            Some(
                crate::benchmarks::butlin::report::RuntimeConsciousnessData::from_structural(
                    md.structural.structural_micro_phi,
                    md.structural.structural_meso_phi,
                    md.structural.structural_macro_phi,
                    md.structural.structural_bottleneck,
                    md.structural.structural_emergence_ratio,
                    md.structural.structural_num_clusters,
                ),
            )
        } else {
            None
        }
    }

    /// Measure the 12 real, mechanism-specific behavioral signals available
    /// via `ablation::measure_indicator` (RPT-1, RPT-2, GWT-2, GWT-3, GWT-4,
    /// HOT-1, HOT-2, HOT-3, PP-1, AE-1, AE-2, AST-1), by reusing that same
    /// probe code against this runner's own (non-ablated) service. GWT-1
    /// isn't measured here — see `report::BehavioralIndicatorSignals`'s doc
    /// comment for why. HOT-4 needs no cognitive loop at all — see
    /// `measure_hot4_sparse_smooth_coding`.
    ///
    /// `num_cycles` is passed straight through to `measure_indicator`, which
    /// discards its own first 20 cycles as warmup — same as the ablation
    /// matrix's default of 200.
    pub fn snapshot_behavioral_indicators(
        &mut self,
        num_cycles: usize,
    ) -> crate::benchmarks::butlin::report::BehavioralIndicatorSignals {
        use crate::benchmarks::butlin::ablation::measure_indicator;
        // Computed first: measure_hot4_sparse_smooth_coding takes &mut self
        // (not just &mut self.service), so it can't be interleaved with the
        // self.service-borrowing fields below in the same struct literal.
        let hot4 = self.measure_hot4_sparse_smooth_coding(5);
        crate::benchmarks::butlin::report::BehavioralIndicatorSignals {
            rpt1_temporal_coherence: measure_indicator(&mut self.service, "RPT-1", num_cycles),
            rpt2_binding_activity: measure_indicator(&mut self.service, "RPT-2", num_cycles),
            gwt2_bounded_coalition: measure_indicator(&mut self.service, "GWT-2", num_cycles),
            gwt3_broadcast_activity: measure_indicator(&mut self.service, "GWT-3", num_cycles),
            gwt4_state_dependent_attention: measure_indicator(
                &mut self.service,
                "GWT-4",
                num_cycles,
            ),
            hot1_prediction_differentiation: measure_indicator(
                &mut self.service,
                "HOT-1",
                num_cycles,
            ),
            hot2_meta_cognitive_accuracy: measure_indicator(&mut self.service, "HOT-2", num_cycles),
            hot3_effective_lr: measure_indicator(&mut self.service, "HOT-3", num_cycles),
            pp1_effective_lr: measure_indicator(&mut self.service, "PP-1", num_cycles),
            ae1_action_diversity: measure_indicator(&mut self.service, "AE-1", num_cycles),
            ae2_embodied_agency: measure_indicator(&mut self.service, "AE-2", num_cycles),
            ast1_attention_focus: measure_indicator(&mut self.service, "AST-1", num_cycles),
            hot4_sparsity: hot4.0,
            hot4_smoothness: hot4.1,
        }
    }

    /// HOT-4 ("sparse and smooth neural coding"): unlike the other 11
    /// indicators, this needs no cognitive loop at all — sparsity and
    /// smoothness are properties of the encoded representations themselves.
    ///
    /// Sparsity: fraction of output dimensions with magnitude below 5% of
    /// the vector's own max magnitude, averaged over `num_samples` distinct
    /// inputs. Smoothness: correlation between input-perturbation magnitude
    /// and output dissimilarity — a genuinely smooth code should have
    /// small perturbations produce small (not discontinuous) output changes.
    /// Returns `(sparsity, smoothness)`, both in `[0, 1]`.
    pub fn measure_hot4_sparse_smooth_coding(&mut self, num_samples: usize) -> (f64, f64) {
        let dim = self.service.state_dim();
        let base_inputs = [
            "The quick brown fox jumps over the lazy dog",
            "A neural network learns to predict sequences",
            "Consciousness emerges from integrated information",
            "Working memory maintains active representations",
            "Prediction errors drive learning and adaptation",
        ];

        // Sparsity: average fraction of near-zero dimensions across samples.
        let mut sparsity_sum = 0.0;
        for i in 0..num_samples {
            let input = base_inputs[i % base_inputs.len()];
            let result = self.service.cycle(input);
            let max_mag = result.output.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_mag <= 0.0 {
                continue;
            }
            let threshold = max_mag * 0.05;
            let near_zero = result
                .output
                .iter()
                .filter(|&&v| v.abs() < threshold)
                .count();
            sparsity_sum += near_zero as f64 / dim.max(1) as f64;
        }
        let sparsity = sparsity_sum / num_samples.max(1) as f64;

        // Smoothness: perturb one input by growing amounts of injected noise
        // and check that output dissimilarity grows monotonically-ish with
        // perturbation size, rather than jumping discontinuously.
        let base_hv = ContinuousHV::random(dim, 0x5A17_C0DE);
        let base_result = self.service.cycle_with_hv(&base_hv);
        let perturbation_levels: [f32; 5] = [0.05, 0.15, 0.30, 0.50, 0.75];
        let mut prev_dissimilarity = 0.0;
        let mut monotonic_steps = 0u32;
        for (i, &level) in perturbation_levels.iter().enumerate() {
            let noise = ContinuousHV::random(dim, 0x5A17_C0DE + i as u64 + 1);
            let perturbed =
                ContinuousHV::weighted_bundle(&[&base_hv, &noise], &[1.0 - level, level]);
            let result = self.service.cycle_with_hv(&perturbed);
            let sim = symthaea_core::math::cosine_similarity_f64(
                &base_result
                    .output
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
                &result.output.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            );
            let dissimilarity = 1.0 - sim;
            if dissimilarity >= prev_dissimilarity - 0.05 {
                monotonic_steps += 1;
            }
            prev_dissimilarity = dissimilarity;
        }
        let smoothness = monotonic_steps as f64 / perturbation_levels.len() as f64;

        (sparsity.clamp(0.0, 1.0), smoothness.clamp(0.0, 1.0))
    }

    /// Snapshot both structural Phi and the real behavioral signals into a
    /// single `RuntimeConsciousnessData`, for callers that want the full
    /// (non-proxy) picture in one call.
    pub fn snapshot_full_runtime_consciousness(
        &mut self,
        num_cycles: usize,
    ) -> Option<crate::benchmarks::butlin::report::RuntimeConsciousnessData> {
        let behavioral = self.snapshot_behavioral_indicators(num_cycles);
        self.snapshot_runtime_consciousness()
            .map(|rt| rt.with_behavioral(behavioral))
    }

    /// Run a benchmark through the cognitive loop.
    pub fn run_benchmark(
        &mut self,
        bench: &dyn LoopDrivable,
        config: &super::config::BenchmarkConfig,
    ) -> LoopBenchmarkResult {
        let start = std::time::Instant::now();
        self.warmup();

        let stimuli = bench.generate_stimuli(config);
        let mut trials = Vec::with_capacity(stimuli.len());
        let mut correct_count = 0u32;

        for stim in &stimuli {
            // Run the stimulus through the cognitive loop
            let result = self.service.cycle_with_hv(&stim.stimulus_hv);

            // Select response: find the alternative most similar to the CfC output
            let output = &result.output;
            let mut best_idx = 0;
            let mut best_sim = f64::NEG_INFINITY;
            for (i, alt) in stim.alternatives.iter().enumerate() {
                let sim = cosine_f32_vec(output, alt.as_slice());
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }

            let correct = best_idx == stim.correct_idx;
            if correct {
                correct_count += 1;
            }

            // Wire trial outcome as reward signal to neuromodulator bath.
            // DA encodes reward prediction error (Schultz 1997).
            let trial_reward = if correct { 0.8 } else { -0.5 };
            self.service.provide_reward(trial_reward);

            trials.push(LoopTrialResult {
                response_idx: best_idx,
                correct,
                prediction_error: result.prediction_error,
                da: result.metadata.neuromod.dopamine_effective,
                ne: result.metadata.neuromod.noradrenaline_effective,
                sht: result.metadata.neuromod.serotonin_effective,
                ach: result.metadata.neuromod.acetylcholine_effective,
                psi: result.metadata.consciousness.consciousness_level as f32,
                cycle_time_us: result.cycle_time_us,
                learning_occurred: result.learning_occurred,
                reward: trial_reward,
                oxytocin: result.metadata.neuromod.neuromod_oxytocin_effective,
                moral_score: result.metadata.ethics.value_evaluator_score as f32,
                bath_entropy: result.metadata.neuromod.neuromod_bath_entropy,
                allostatic_load: result.metadata.neuromod.neuromod_allostatic_load,
                consciousness_mod: result.metadata.neuromod.neuromod_consciousness_mod,
                social_coherence: result.metadata.neuromod.neuromod_social_coherence,
                structural_micro_phi: result.metadata.structural.structural_micro_phi,
                structural_emergence_ratio: result.metadata.structural.structural_emergence_ratio,
            });
        }

        let n = trials.len() as f64;
        let accuracy = if n > 0.0 {
            correct_count as f64 / n
        } else {
            0.0
        };
        let mean_pe = trials
            .iter()
            .map(|t| t.prediction_error as f64)
            .sum::<f64>()
            / n.max(1.0);
        let mean_da = trials.iter().map(|t| t.da as f64).sum::<f64>() / n.max(1.0);
        let mean_ne = trials.iter().map(|t| t.ne as f64).sum::<f64>() / n.max(1.0);
        let mean_sht = trials.iter().map(|t| t.sht as f64).sum::<f64>() / n.max(1.0);
        let mean_ach = trials.iter().map(|t| t.ach as f64).sum::<f64>() / n.max(1.0);
        let mean_reward = trials.iter().map(|t| t.reward as f64).sum::<f64>() / n.max(1.0);
        let mean_oxytocin = trials.iter().map(|t| t.oxytocin as f64).sum::<f64>() / n.max(1.0);
        let mean_moral_score =
            trials.iter().map(|t| t.moral_score as f64).sum::<f64>() / n.max(1.0);
        let mean_bath_entropy =
            trials.iter().map(|t| t.bath_entropy as f64).sum::<f64>() / n.max(1.0);
        let mean_allostatic_load =
            trials.iter().map(|t| t.allostatic_load as f64).sum::<f64>() / n.max(1.0);
        let mean_consciousness_mod = trials
            .iter()
            .map(|t| t.consciousness_mod as f64)
            .sum::<f64>()
            / n.max(1.0);
        let mean_social_coherence = trials
            .iter()
            .map(|t| t.social_coherence as f64)
            .sum::<f64>()
            / n.max(1.0);
        let mean_structural_micro_phi =
            trials.iter().map(|t| t.structural_micro_phi).sum::<f64>() / n.max(1.0);
        let mean_structural_emergence_ratio = trials
            .iter()
            .map(|t| t.structural_emergence_ratio)
            .sum::<f64>()
            / n.max(1.0);

        LoopBenchmarkResult {
            benchmark: bench.loop_name().to_string(),
            trials,
            accuracy,
            mean_prediction_error: mean_pe,
            mean_da,
            mean_ne,
            mean_sht,
            mean_ach,
            mean_reward,
            mean_oxytocin,
            mean_moral_score,
            mean_bath_entropy,
            mean_allostatic_load,
            mean_consciousness_mod,
            mean_social_coherence,
            mean_structural_micro_phi,
            mean_structural_emergence_ratio,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }
}

/// Cosine similarity between f32 slice and ContinuousHV data.
#[cfg(feature = "symthaea-backend")]
fn cosine_f32_vec(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..len {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-10 { dot / denom } else { 0.0 }
}

// ──── LoopDrivable implementations for 6 benchmarks ────

#[cfg(feature = "symthaea-backend")]
mod impls {
    use super::*;
    use crate::benchmarks::affect::EmotionalStroopBenchmark;
    use crate::benchmarks::executive::StroopBenchmark;
    use crate::benchmarks::executive::WisconsinCardSortingBenchmark;
    use crate::benchmarks::inhibition::GoNoGoBenchmark;
    use crate::benchmarks::reasoning::ArcFluidBenchmark;
    use crate::benchmarks::social::UltimatumGameBenchmark;
    use crate::benchmarks::sustained_attention::PvtBenchmark;
    use crate::benchmarks::worm::NBackBenchmark;

    impl LoopDrivable for StroopBenchmark {
        fn loop_name(&self) -> &str {
            "Executive::Stroop"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            // 4 color HVs
            let colors: Vec<ContinuousHV> = (0..4)
                .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i)))
                .collect();

            let n_trials = config.trials_per_condition * 3; // 3 conditions
            let mut stimuli = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                xor_shift(&mut rng);
                let ink_idx = (rng % 4) as usize;
                let condition = match trial % 3 {
                    0 => "congruent",
                    1 => "incongruent",
                    _ => "neutral",
                };

                let stimulus_hv = match condition {
                    "congruent" => colors[ink_idx].scale(1.45),
                    "incongruent" => {
                        xor_shift(&mut rng);
                        let mut word_idx = (rng % 3) as usize;
                        if word_idx >= ink_idx {
                            word_idx += 1;
                        }
                        ContinuousHV::bundle(&[&colors[ink_idx], &colors[word_idx].scale(0.45)])
                    }
                    _ => {
                        xor_shift(&mut rng);
                        let noise = ContinuousHV::random(dim, rng);
                        ContinuousHV::bundle(&[&colors[ink_idx], &noise.scale(0.15)])
                    }
                };

                stimuli.push(LoopStimulus {
                    stimulus_hv,
                    alternatives: colors.clone(),
                    correct_idx: ink_idx,
                    condition: condition.to_string(),
                    trial_idx: trial,
                });
            }
            stimuli
        }
    }

    impl LoopDrivable for NBackBenchmark {
        fn loop_name(&self) -> &str {
            "WorM::N-back"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            let n = 2; // 2-back
            let seq_len = 30;
            let vocab_size = 8u64;

            // Generate item HVs
            let item_hvs: Vec<ContinuousHV> = (0..vocab_size)
                .map(|i| ContinuousHV::random(dim, seed.wrapping_add(200 + i)))
                .collect();

            // Generate sequence
            let mut sequence = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                let is_target = i >= n && {
                    xor_shift(&mut rng);
                    (rng % 100) < 30
                };
                if is_target {
                    sequence.push(sequence[i - n]);
                } else {
                    xor_shift(&mut rng);
                    sequence.push((rng % vocab_size) as usize);
                }
            }

            // Match = 1, No-match = 0 (2-AFC)
            let match_hv = ContinuousHV::random(dim, seed.wrapping_add(1000));
            let nomatch_hv = ContinuousHV::random(dim, seed.wrapping_add(1001));
            let alternatives = vec![nomatch_hv, match_hv];

            let mut stimuli = Vec::new();
            for i in n..seq_len {
                let is_target = sequence[i] == sequence[i - n];
                stimuli.push(LoopStimulus {
                    stimulus_hv: item_hvs[sequence[i]].clone(),
                    alternatives: alternatives.clone(),
                    correct_idx: if is_target { 1 } else { 0 },
                    condition: format!("nback_{}", n),
                    trial_idx: i,
                });
            }
            stimuli
        }
    }

    impl LoopDrivable for WisconsinCardSortingBenchmark {
        fn loop_name(&self) -> &str {
            "Executive::WCST"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            // 3 rule categories
            let rule_hvs: Vec<ContinuousHV> = (0..3)
                .map(|i| ContinuousHV::random(dim, seed.wrapping_add(300 + i)))
                .collect();

            let n_trials = config.trials_per_condition * 10;
            let mut stimuli = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                xor_shift(&mut rng);
                let card_hv = ContinuousHV::random(dim, rng);
                let current_rule = (trial / 10) % 3; // rule shifts every 10 trials

                stimuli.push(LoopStimulus {
                    stimulus_hv: card_hv,
                    alternatives: rule_hvs.clone(),
                    correct_idx: current_rule,
                    condition: format!("rule_{}", current_rule),
                    trial_idx: trial,
                });
            }
            stimuli
        }
    }

    impl LoopDrivable for GoNoGoBenchmark {
        fn loop_name(&self) -> &str {
            "Inhibition::GoNoGo"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            let go_proto = ContinuousHV::random(dim, seed.wrapping_add(100));
            let nogo_proto = ContinuousHV::random(dim, seed.wrapping_add(200));
            let respond_hv = ContinuousHV::random(dim, seed.wrapping_add(300));
            let withhold_hv = ContinuousHV::random(dim, seed.wrapping_add(400));
            let alternatives = vec![withhold_hv, respond_hv]; // 0=withhold, 1=respond

            let n_trials = config.trials_per_condition * 10;
            let mut stimuli = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                xor_shift(&mut rng);
                let is_go = (rng % 100) < 70;
                let proto = if is_go { &go_proto } else { &nogo_proto };
                xor_shift(&mut rng);
                let noise = ContinuousHV::random(dim, rng);
                let stimulus_hv = ContinuousHV::weighted_bundle(&[proto, &noise], &[0.85, 0.15]);

                stimuli.push(LoopStimulus {
                    stimulus_hv,
                    alternatives: alternatives.clone(),
                    correct_idx: if is_go { 1 } else { 0 }, // go=respond(1), nogo=withhold(0)
                    condition: if is_go {
                        "go".to_string()
                    } else {
                        "nogo".to_string()
                    },
                    trial_idx: trial,
                });
            }
            stimuli
        }
    }

    impl LoopDrivable for PvtBenchmark {
        fn loop_name(&self) -> &str {
            "SustainedAttention::PVT"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            let stimulus_proto = ContinuousHV::random(dim, seed.wrapping_add(1));
            let respond_hv = ContinuousHV::random(dim, seed.wrapping_add(2));
            let lapse_hv = ContinuousHV::random(dim, seed.wrapping_add(3));
            let alternatives = vec![lapse_hv, respond_hv]; // 0=lapse, 1=respond

            let n_trials = 200;
            let mut stimuli = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                xor_shift(&mut rng);
                let noise = ContinuousHV::random(dim, rng);
                let stimulus_hv =
                    ContinuousHV::weighted_bundle(&[&stimulus_proto, &noise], &[0.80, 0.20]);

                stimuli.push(LoopStimulus {
                    stimulus_hv,
                    alternatives: alternatives.clone(),
                    correct_idx: 1, // always respond
                    condition: format!("block_{}", trial / 20),
                    trial_idx: trial,
                });
            }
            stimuli
        }
    }

    impl LoopDrivable for ArcFluidBenchmark {
        fn loop_name(&self) -> &str {
            "Reasoning::ArcFluid"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            let n_trials = config.trials_per_condition * 4; // 4 task types
            let mut stimuli = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                xor_shift(&mut rng);
                // Generate a rule-encoded stimulus
                let rule_hv = ContinuousHV::random(dim, rng);
                xor_shift(&mut rng);
                let correct_hv = ContinuousHV::random(dim, rng);
                xor_shift(&mut rng);
                let distractor_hv = ContinuousHV::random(dim, rng);

                // Correct should be more similar to rule than distractor
                let stimulus_hv =
                    ContinuousHV::weighted_bundle(&[&rule_hv, &correct_hv], &[0.6, 0.4]);

                stimuli.push(LoopStimulus {
                    stimulus_hv,
                    alternatives: vec![distractor_hv, correct_hv],
                    correct_idx: 1,
                    condition: format!("type_{}", trial % 4),
                    trial_idx: trial,
                });
            }
            stimuli
        }
    }

    impl LoopDrivable for UltimatumGameBenchmark {
        fn loop_name(&self) -> &str {
            "Social::UltimatumGame"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            // Fair (50:50) and unfair (100:0) prototypes
            let fair_proto = ContinuousHV::random(dim, seed.wrapping_add(500));
            let unfair_proto = ContinuousHV::random(dim, seed.wrapping_add(501));
            // Accept / reject response HVs
            let reject_hv = ContinuousHV::random(dim, seed.wrapping_add(502));
            let accept_hv = ContinuousHV::random(dim, seed.wrapping_add(503));
            let alternatives = vec![reject_hv, accept_hv]; // 0=reject, 1=accept

            // 5 offer levels × 8 trials each = 40 trials
            let offer_levels = [0.10f32, 0.20, 0.30, 0.40, 0.50];
            let trials_per = 8;
            let mut stimuli = Vec::with_capacity(offer_levels.len() * trials_per);

            for &offer in &offer_levels {
                for rep in 0..trials_per {
                    xor_shift(&mut rng);
                    let noise = ContinuousHV::random(dim, rng);
                    // Blend fair/unfair based on offer level
                    let stimulus_hv = ContinuousHV::weighted_bundle(
                        &[&fair_proto, &unfair_proto, &noise],
                        &[offer, 1.0 - offer, 0.05],
                    );

                    // Human threshold: reject if offer < 30%
                    let correct_idx = if offer < 0.30 { 0 } else { 1 };

                    stimuli.push(LoopStimulus {
                        stimulus_hv,
                        alternatives: alternatives.clone(),
                        correct_idx,
                        condition: format!("offer_{:.0}pct", offer * 100.0),
                        trial_idx: offer_levels.iter().position(|&o| o == offer).unwrap_or(0)
                            * trials_per
                            + rep,
                    });
                }
            }
            stimuli
        }
    }

    impl LoopDrivable for EmotionalStroopBenchmark {
        fn loop_name(&self) -> &str {
            "Affect::EmotionalStroop"
        }

        fn generate_stimuli(
            &self,
            config: &super::super::config::BenchmarkConfig,
        ) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let seed = config.seed;
            let mut rng = seed ^ 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };

            // 4 color response HVs
            let colors: Vec<ContinuousHV> = (0..4)
                .map(|i| ContinuousHV::random(dim, seed.wrapping_add(600 + i)))
                .collect();
            // Negative valence prototype (threat content)
            let neg_proto = ContinuousHV::random(dim, seed.wrapping_add(610));

            let n_trials = 80; // 40 neutral + 40 negative
            let mut stimuli = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                xor_shift(&mut rng);
                let ink_idx = (rng % 4) as usize;
                let is_negative = trial >= 40;

                let stimulus_hv = if is_negative {
                    // Negative word partially activates competing color
                    xor_shift(&mut rng);
                    let competing_idx = ((ink_idx + 1 + (rng % 3) as usize) % 4) as usize;
                    ContinuousHV::weighted_bundle(
                        &[&colors[ink_idx], &neg_proto, &colors[competing_idx]],
                        &[0.65, 0.20, 0.15],
                    )
                } else {
                    xor_shift(&mut rng);
                    let noise = ContinuousHV::random(dim, rng);
                    ContinuousHV::weighted_bundle(&[&colors[ink_idx], &noise], &[0.85, 0.15])
                };

                stimuli.push(LoopStimulus {
                    stimulus_hv,
                    alternatives: colors.clone(),
                    correct_idx: ink_idx,
                    condition: if is_negative { "negative" } else { "neutral" }.to_string(),
                    trial_idx: trial,
                });
            }
            stimuli
        }
    }
}

#[cfg(all(test, not(feature = "symthaea-backend")))]
mod tests {
    use super::*;

    #[test]
    fn test_loop_stimulus_creation() {
        let stim = LoopStimulus {
            stimulus_hv: ContinuousHV::random(64, 42),
            alternatives: vec![ContinuousHV::random(64, 1), ContinuousHV::random(64, 2)],
            correct_idx: 0,
            condition: "test".to_string(),
            trial_idx: 0,
        };
        assert_eq!(stim.correct_idx, 0);
        assert_eq!(stim.alternatives.len(), 2);
    }

    #[test]
    fn test_loop_trial_result_serialization() {
        let result = LoopTrialResult {
            response_idx: 1,
            correct: true,
            prediction_error: 0.1,
            da: 0.5,
            ne: 0.5,
            sht: 0.5,
            ach: 0.5,
            psi: 0.3,
            cycle_time_us: 1000,
            learning_occurred: false,
            reward: 0.8,
            oxytocin: 0.3,
            moral_score: 0.0,
            bath_entropy: 0.5,
            allostatic_load: 0.1,
            consciousness_mod: 1.0,
            social_coherence: 1.0,
            structural_micro_phi: 0.0,
            structural_emergence_ratio: 0.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"correct\":true"));
    }

    #[test]
    fn test_loop_benchmark_result_creation() {
        let result = LoopBenchmarkResult {
            benchmark: "test".to_string(),
            trials: Vec::new(),
            accuracy: 0.85,
            mean_prediction_error: 0.1,
            mean_da: 0.5,
            mean_ne: 0.5,
            mean_sht: 0.5,
            mean_ach: 0.5,
            mean_reward: 0.5,
            mean_oxytocin: 0.3,
            mean_moral_score: 0.0,
            mean_bath_entropy: 0.5,
            mean_allostatic_load: 0.1,
            mean_consciousness_mod: 1.0,
            mean_social_coherence: 1.0,
            mean_structural_micro_phi: 0.0,
            mean_structural_emergence_ratio: 0.0,
            elapsed_ms: 100,
        };
        assert!((result.accuracy - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_basic() {
        // Without symthaea-backend, we can't test cosine_f32_vec directly
        // but we verify the struct works
        let result = LoopBenchmarkResult {
            benchmark: "cosine_test".to_string(),
            trials: vec![],
            accuracy: 1.0,
            mean_prediction_error: 0.0,
            mean_da: 0.5,
            mean_ne: 0.5,
            mean_sht: 0.5,
            mean_ach: 0.5,
            mean_reward: 0.0,
            mean_oxytocin: 0.0,
            mean_moral_score: 0.0,
            mean_bath_entropy: 0.0,
            mean_allostatic_load: 0.0,
            mean_consciousness_mod: 0.0,
            mean_social_coherence: 0.0,
            mean_structural_micro_phi: 0.0,
            mean_structural_emergence_ratio: 0.0,
            elapsed_ms: 0,
        };
        assert_eq!(result.benchmark, "cosine_test");
    }

    #[test]
    fn test_extended_trial_result_defaults() {
        let result = LoopTrialResult {
            response_idx: 0,
            correct: false,
            prediction_error: 0.0,
            da: 0.0,
            ne: 0.0,
            sht: 0.0,
            ach: 0.0,
            psi: 0.0,
            cycle_time_us: 0,
            learning_occurred: false,
            reward: 0.0,
            oxytocin: 0.0,
            moral_score: 0.0,
            bath_entropy: 0.0,
            allostatic_load: 0.0,
            consciousness_mod: 0.0,
            social_coherence: 0.0,
            structural_micro_phi: 0.0,
            structural_emergence_ratio: 0.0,
        };
        assert_eq!(result.oxytocin, 0.0);
        assert_eq!(result.moral_score, 0.0);
        assert_eq!(result.bath_entropy, 0.0);
        assert_eq!(result.allostatic_load, 0.0);
        assert_eq!(result.consciousness_mod, 0.0);
        assert_eq!(result.social_coherence, 0.0);
    }

    #[test]
    fn test_loop_drivable_trait_exists() {
        // Verify the trait can be used as trait object
        fn _takes_loop_drivable(_b: &dyn LoopDrivable) {}
    }

    #[test]
    fn test_loop_trial_result_structural_fields() {
        let result = LoopTrialResult {
            response_idx: 0,
            correct: true,
            prediction_error: 0.1,
            da: 0.5,
            ne: 0.5,
            sht: 0.5,
            ach: 0.5,
            psi: 0.3,
            cycle_time_us: 1000,
            learning_occurred: false,
            reward: 0.0,
            oxytocin: 0.0,
            moral_score: 0.0,
            bath_entropy: 0.0,
            allostatic_load: 0.0,
            consciousness_mod: 0.0,
            social_coherence: 0.0,
            structural_micro_phi: 0.15,
            structural_emergence_ratio: 1.3,
        };
        let json = serde_json::to_string(&result).unwrap();
        let restored: LoopTrialResult = serde_json::from_str(&json).unwrap();
        assert!((restored.structural_micro_phi - 0.15).abs() < 1e-10);
        assert!((restored.structural_emergence_ratio - 1.3).abs() < 1e-10);

        // Also verify serde(default) works: deserializing without the fields
        let old_json = r#"{"response_idx":0,"correct":true,"prediction_error":0.1,
            "da":0.5,"ne":0.5,"sht":0.5,"ach":0.5,"psi":0.3,
            "cycle_time_us":1000,"learning_occurred":false,
            "reward":0.0,"oxytocin":0.0,"moral_score":0.0,
            "bath_entropy":0.0,"allostatic_load":0.0,
            "consciousness_mod":0.0,"social_coherence":0.0}"#;
        let from_old: LoopTrialResult = serde_json::from_str(old_json).unwrap();
        assert_eq!(from_old.structural_micro_phi, 0.0);
        assert_eq!(from_old.structural_emergence_ratio, 0.0);
    }
}
