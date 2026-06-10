// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark and ablation configuration.

use crate::benchmarks::butlin::report::RuntimeConsciousnessData;
use serde::{Deserialize, Serialize};

/// Configuration for running a benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// HDC dimension (default: 512).
    pub dimension: usize,
    /// Number of trials per condition (default: 20).
    pub trials_per_condition: usize,
    /// Working memory capacity override (default: 7).
    pub working_memory_capacity: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Enable social coherence (Theory of Mind).
    pub enable_social: bool,
    /// Enable FEP active inference.
    pub enable_fep: bool,
    /// FEP planning horizon (default: 3).
    pub planning_horizon: usize,
    /// FEP action temperature (default: 1.0).
    pub action_temperature: f64,
    /// Optional label for this configuration (used in ablation reports).
    pub label: Option<String>,
    /// Time pressure level for speed-accuracy tradeoff (0.0 = none, 1.0 = maximum).
    ///
    /// At higher time pressure, benchmarks that support it will use fewer
    /// deliberation ticks, noisier encoding, and faster response thresholds,
    /// trading accuracy for speed (a speed-accuracy tradeoff curve).
    pub time_pressure: f64,
    /// Enable adaptive trial counts (default: false).
    ///
    /// When enabled, benchmarks increase trial counts until the key metric's
    /// CI half-width is within `precision_target × |mean|`, up to `max_trials`.
    #[serde(default)]
    pub adaptive_trials: bool,
    /// Minimum trials for adaptive mode (default: 10).
    #[serde(default = "default_min_trials")]
    pub min_trials: usize,
    /// Maximum trials for adaptive mode (default: 200).
    #[serde(default = "default_max_trials")]
    pub max_trials: usize,
    /// Precision target: CI half-width as fraction of |mean| (default: 0.05).
    #[serde(default = "default_precision_target")]
    pub precision_target: f64,
    /// Use SSM temporal backend for benchmarks that support it (default: false).
    ///
    /// When enabled, replaces ad-hoc decay models with principled state-space
    /// recurrence from `symthaea-ssm`. Affects: ProspectiveMemory, AttentionalBlink,
    /// SerialRecall, DigitSpan.
    #[serde(default)]
    pub ssm_backend: bool,
    /// Difficulty level (0.0 = current/easy, 1.0 = hard).
    ///
    /// Modulates benchmark parameters (temperature, SNR, interference) to push
    /// accuracy into human-like ranges (70-95%). Default 0.0 = unchanged behavior.
    #[serde(default)]
    pub difficulty: f64,
    /// Enable per-trial trace collection (default: false).
    ///
    /// When enabled, benchmarks that support it will populate `BenchmarkResult.trial_trace`
    /// with per-trial outcomes for fine-grained analysis.
    #[serde(default)]
    pub trial_trace: bool,
    /// Encoding noise level (0.0 = clean, 1.0 = maximum degradation).
    ///
    /// Models the degradation of HDC representations when consciousness subsystems
    /// are ablated. Top-down predictions from FEP/social/WM systems normally refine
    /// bottom-up encodings; removing them increases noise in the encoding space.
    /// Used by ablation presets and propagated to benchmark similarity computations.
    #[serde(default)]
    pub encoding_noise: f64,

    /// Optional runtime consciousness data from the structural Phi engine.
    /// When present, Butlin indicators blend static architectural scores with
    /// live measurements for theory-aligned accuracy.
    #[serde(default)]
    pub runtime_consciousness: Option<RuntimeConsciousnessData>,

    /// D2 NoGo pathway inhibition strength [0.0, 1.0] (default 0.3).
    /// Models dopamine D2-mediated response suppression (Frank 2005).
    #[serde(default = "default_neuromod_d2_inhibition")]
    pub neuromod_d2_inhibition: f64,

    /// NE phasic brake boost [0.0, 1.0] (default 0.25).
    /// Models norepinephrine phasic emergency stop signal (Aron 2007).
    #[serde(default = "default_neuromod_ne_phasic")]
    pub neuromod_ne_phasic: f64,

    /// Language reparse cost scale (default 0.8).
    /// Controls the effort of syntactic reanalysis in garden-path sentences.
    #[serde(default = "default_language_reparse_cost_scale")]
    pub language_reparse_cost_scale: f64,

    /// Language priming decay per SOA step (default 0.85).
    /// Exponential activation retention for semantic priming.
    #[serde(default = "default_language_priming_decay")]
    pub language_priming_decay: f64,

    /// Language coherence EMA alpha (default 0.15).
    /// Smoothing factor for topic drift monitoring.
    #[serde(default = "default_language_coherence_alpha")]
    pub language_coherence_alpha: f64,

    /// Language frequency boost for high-frequency words (default 0.15).
    /// Advantage for recognizing common words in lexical decision.
    #[serde(default = "default_language_frequency_boost")]
    pub language_frequency_boost: f64,

    /// Attention lapse rate [0.0, 1.0] (default 0.0).
    ///
    /// Models the probability of an attention lapse on any given trial
    /// (Wichmann & Hill, 2001). On lapse trials, the response is random
    /// regardless of stimulus quality. This is the primary source of
    /// individual differences in psychometric test-retest reliability:
    /// subjects with higher lapse rates produce lower, noisier scores.
    ///
    /// Used by `ReliabilityBattery` to create between-subject variance.
    #[serde(default)]
    pub lapse_rate: f64,
}

fn default_min_trials() -> usize {
    10
}
fn default_max_trials() -> usize {
    200
}
fn default_precision_target() -> f64 {
    0.05
}
fn default_neuromod_d2_inhibition() -> f64 {
    0.3
}
fn default_neuromod_ne_phasic() -> f64 {
    0.25
}
fn default_language_reparse_cost_scale() -> f64 {
    0.8
}
fn default_language_priming_decay() -> f64 {
    0.85
}
fn default_language_coherence_alpha() -> f64 {
    0.15
}
fn default_language_frequency_boost() -> f64 {
    0.15
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            trials_per_condition: 20,
            working_memory_capacity: 7,
            seed: 42,
            enable_social: true,
            enable_fep: true,
            planning_horizon: 3,
            action_temperature: 1.0,
            label: None,
            time_pressure: 0.0,
            adaptive_trials: false,
            min_trials: default_min_trials(),
            max_trials: default_max_trials(),
            precision_target: default_precision_target(),
            ssm_backend: false,
            difficulty: 0.0,
            trial_trace: false,
            encoding_noise: 0.0,
            runtime_consciousness: None,
            neuromod_d2_inhibition: default_neuromod_d2_inhibition(),
            neuromod_ne_phasic: default_neuromod_ne_phasic(),
            language_reparse_cost_scale: default_language_reparse_cost_scale(),
            language_priming_decay: default_language_priming_decay(),
            language_coherence_alpha: default_language_coherence_alpha(),
            language_frequency_boost: default_language_frequency_boost(),
            lapse_rate: 0.0,
        }
    }
}

impl BenchmarkConfig {
    /// Create a deterministic seed for a specific trial.
    pub fn trial_seed(&self, benchmark: &str, condition: &str, trial: usize) -> u64 {
        let mut h: u64 = self.seed;
        for b in benchmark.bytes().chain(condition.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^ (trial as u64).wrapping_mul(0x9E3779B97F4A7C15)
    }

    /// Attach runtime consciousness data for Butlin indicator blending.
    pub fn with_runtime_consciousness(mut self, data: RuntimeConsciousnessData) -> Self {
        self.runtime_consciousness = Some(data);
        self
    }

    /// Compute effective encoding noise combining `encoding_noise` and `time_pressure`.
    ///
    /// Both ablation (encoding_noise) and speed-accuracy tradeoff (time_pressure)
    /// degrade HDC representations. Returns a combined noise weight in [0.0, 1.0]
    /// for use in benchmark similarity computations.
    pub fn effective_noise(&self) -> f64 {
        (self.encoding_noise * 1.75 + self.time_pressure * 0.20).clamp(0.0, 1.0)
    }

    /// Check whether an attention lapse occurs on this trial.
    ///
    /// Uses the trial seed to make the lapse deterministic per
    /// (seed, domain, trial_idx) triple. Returns `true` if the subject
    /// lapses on this trial (response should be randomized).
    ///
    /// Reference: Wichmann & Hill (2001), "The psychometric function"
    pub fn should_lapse(&self, domain: &str, trial_idx: usize) -> bool {
        if self.lapse_rate <= 0.0 {
            return false;
        }
        let h = self.trial_seed(domain, "lapse", trial_idx);
        (h % 10000) as f64 / 10000.0 < self.lapse_rate
    }

    /// Apply the lapse model to a binary (2-AFC) decision.
    ///
    /// If the subject lapses on this trial, returns a random response
    /// (50% chance correct). Otherwise returns the veridical `correct` value.
    pub fn check_correct(&self, correct: bool, domain: &str, trial_idx: usize) -> bool {
        if self.should_lapse(domain, trial_idx) {
            let h = self.trial_seed(domain, "lapse_response", trial_idx);
            h % 2 == 0
        } else {
            correct
        }
    }

    /// Apply the lapse model to an N-AFC decision.
    ///
    /// If the subject lapses on this trial, returns a random choice index
    /// in `[0, n_choices)`. Otherwise returns the veridical `chosen_idx`.
    pub fn check_choice(
        &self,
        chosen_idx: usize,
        n_choices: usize,
        domain: &str,
        trial_idx: usize,
    ) -> usize {
        if self.should_lapse(domain, trial_idx) {
            let h = self.trial_seed(domain, "lapse_choice", trial_idx);
            h as usize % n_choices
        } else {
            chosen_idx
        }
    }
}

/// Named ablation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    /// Human-readable name (e.g., "Full Consciousness", "HDC Only").
    pub name: String,
    /// Base configuration with modified parameters.
    pub base: BenchmarkConfig,
}

/// Pre-built ablation presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AblationPreset {
    /// Full system with all subsystems enabled.
    FullConsciousness,
    /// CfC temporal network only, no FEP or social.
    CfcOnly,
    /// No FEP active inference.
    NoFep,
    /// No social coherence / Theory of Mind.
    NoSocial,
    /// Reduced working memory (capacity = 3).
    ReducedWm,
    /// Pure HDC encoding, minimal processing.
    HdcOnly,
}

impl AblationPreset {
    /// All presets in order.
    pub fn all() -> &'static [AblationPreset] {
        &[
            AblationPreset::FullConsciousness,
            AblationPreset::CfcOnly,
            AblationPreset::NoFep,
            AblationPreset::NoSocial,
            AblationPreset::ReducedWm,
            AblationPreset::HdcOnly,
        ]
    }

    /// Convert to an `AblationConfig` with the given base seed.
    ///
    /// Each preset sets `encoding_noise` proportional to the subsystems removed.
    /// Rationale: consciousness subsystems (FEP, social, WM) provide top-down
    /// predictive refinement of HDC encodings. Removing them degrades the
    /// signal-to-noise ratio of perceptual representations.
    pub fn to_config(self, seed: u64) -> AblationConfig {
        let mut base = BenchmarkConfig {
            seed,
            ..Default::default()
        };
        let name = match self {
            AblationPreset::FullConsciousness => {
                // All systems intact: clean encodings
                base.encoding_noise = 0.0;
                "Full Consciousness"
            }
            AblationPreset::CfcOnly => {
                base.enable_fep = false;
                base.enable_social = false;
                base.encoding_noise = 0.35;
                "CfC Only"
            }
            AblationPreset::NoFep => {
                base.enable_fep = false;
                base.encoding_noise = 0.25;
                "No FEP"
            }
            AblationPreset::NoSocial => {
                base.enable_social = false;
                base.encoding_noise = 0.15;
                "No Social"
            }
            AblationPreset::ReducedWm => {
                base.working_memory_capacity = 3;
                base.encoding_noise = 0.30;
                "Reduced WM (K=3)"
            }
            AblationPreset::HdcOnly => {
                base.enable_fep = false;
                base.enable_social = false;
                base.working_memory_capacity = 3;
                base.encoding_noise = 0.50;
                "HDC Only"
            }
        };
        AblationConfig {
            name: name.to_string(),
            base,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_seed_deterministic() {
        let config = BenchmarkConfig::default();
        let a = config.trial_seed("worm", "nback_2", 0);
        let b = config.trial_seed("worm", "nback_2", 0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_trial_seed_differs_by_trial() {
        let config = BenchmarkConfig::default();
        let a = config.trial_seed("worm", "nback_2", 0);
        let b = config.trial_seed("worm", "nback_2", 1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_ablation_presets_count() {
        assert_eq!(AblationPreset::all().len(), 6);
    }

    #[test]
    fn test_benchmark_config_with_runtime() {
        let data = RuntimeConsciousnessData::from_structural(0.1, 0.2, 0.3, 0.05, 1.5, 4);
        let config = BenchmarkConfig::default().with_runtime_consciousness(data.clone());
        assert!(config.runtime_consciousness.is_some());
        let rc = config.runtime_consciousness.unwrap();
        assert!((rc.micro_phi - 0.1).abs() < f64::EPSILON);
        assert_eq!(rc.num_clusters, 4);
    }
}
