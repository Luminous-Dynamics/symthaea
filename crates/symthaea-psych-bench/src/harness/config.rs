//! Benchmark and ablation configuration.

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
    pub fn to_config(self, seed: u64) -> AblationConfig {
        let mut base = BenchmarkConfig {
            seed,
            ..Default::default()
        };
        let name = match self {
            AblationPreset::FullConsciousness => "Full Consciousness",
            AblationPreset::CfcOnly => {
                base.enable_fep = false;
                base.enable_social = false;
                "CfC Only"
            }
            AblationPreset::NoFep => {
                base.enable_fep = false;
                "No FEP"
            }
            AblationPreset::NoSocial => {
                base.enable_social = false;
                "No Social"
            }
            AblationPreset::ReducedWm => {
                base.working_memory_capacity = 3;
                "Reduced WM (K=3)"
            }
            AblationPreset::HdcOnly => {
                base.enable_fep = false;
                base.enable_social = false;
                base.working_memory_capacity = 3;
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
}
