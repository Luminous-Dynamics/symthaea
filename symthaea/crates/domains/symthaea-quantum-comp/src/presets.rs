//! Named run presets for examples, CLIs, and notebook wrappers.
//!
//! Presets keep alpha examples reproducible without requiring a configuration
//! parser. They are deliberately conservative and should be treated as starter
//! profiles, not canonical scientific protocols.

use crate::comparative::ComparativeBindingConfig;
use crate::matrix::ExperimentMatrixConfig;
use crate::noise_sweep::NoiseSweepConfig;
use crate::probe::BindingProbeConfig;

/// Stable preset names for local runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPreset {
    /// Very small smoke-test preset for CI and API checks.
    Smoke,
    /// Small local research preset for laptops.
    LocalResearch,
    /// Slightly broader matrix preset for pilot reports.
    PilotMatrix,
}

impl RunPreset {
    /// Parses a preset from a stable lowercase name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "smoke" => Some(Self::Smoke),
            "local" | "local-research" => Some(Self::LocalResearch),
            "matrix" | "pilot-matrix" => Some(Self::PilotMatrix),
            _ => None,
        }
    }

    /// Returns the stable lowercase name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::LocalResearch => "local-research",
            Self::PilotMatrix => "pilot-matrix",
        }
    }

    /// Returns a binding probe configuration for this preset.
    pub fn binding_config(self) -> BindingProbeConfig {
        match self {
            Self::Smoke => BindingProbeConfig {
                dimension: 128,
                trials: 4,
                noise: 0.05,
                seed: 0xA700_0001,
                topology_threshold: 0.55,
            },
            Self::LocalResearch => BindingProbeConfig {
                dimension: 1024,
                trials: 32,
                noise: 0.05,
                seed: 0xA700_0002,
                topology_threshold: 0.55,
            },
            Self::PilotMatrix => BindingProbeConfig {
                dimension: 512,
                trials: 16,
                noise: 0.10,
                seed: 0xA700_0003,
                topology_threshold: 0.55,
            },
        }
    }

    /// Returns a noise sweep configuration for this preset.
    pub fn noise_sweep_config(self) -> NoiseSweepConfig {
        match self {
            Self::Smoke => NoiseSweepConfig {
                base: self.binding_config(),
                steps: 3,
                max_noise: 0.20,
            },
            Self::LocalResearch => NoiseSweepConfig {
                base: self.binding_config(),
                steps: 6,
                max_noise: 0.30,
            },
            Self::PilotMatrix => NoiseSweepConfig {
                base: self.binding_config(),
                steps: 5,
                max_noise: 0.25,
            },
        }
    }

    /// Returns a replicated comparison configuration for this preset.
    pub fn comparative_config(self) -> ComparativeBindingConfig {
        match self {
            Self::Smoke => ComparativeBindingConfig {
                base: self.binding_config(),
                replicates: 2,
                seed_stride: 0x9E37_79B9_7F4A_7C15,
            },
            Self::LocalResearch => ComparativeBindingConfig {
                base: self.binding_config(),
                replicates: 8,
                seed_stride: 0x9E37_79B9_7F4A_7C15,
            },
            Self::PilotMatrix => ComparativeBindingConfig {
                base: self.binding_config(),
                replicates: 4,
                seed_stride: 0x9E37_79B9_7F4A_7C15,
            },
        }
    }

    /// Returns an experiment matrix configuration for this preset.
    pub fn matrix_config(self) -> ExperimentMatrixConfig {
        match self {
            Self::Smoke => ExperimentMatrixConfig {
                dimensions: vec![64, 128],
                noise_levels: vec![0.0, 0.1],
                trials: 2,
                replicates: 2,
                seed: 0xA700_0011,
                seed_stride: 0x9E37_79B9_7F4A_7C15,
                topology_threshold: 0.55,
            },
            Self::LocalResearch => ExperimentMatrixConfig {
                dimensions: vec![256, 512, 1024],
                noise_levels: vec![0.0, 0.05, 0.10, 0.20],
                trials: 8,
                replicates: 4,
                seed: 0xA700_0012,
                seed_stride: 0x9E37_79B9_7F4A_7C15,
                topology_threshold: 0.55,
            },
            Self::PilotMatrix => ExperimentMatrixConfig::default(),
        }
    }
}

/// Returns the supported preset names.
pub fn supported_preset_names() -> &'static [&'static str] {
    &["smoke", "local-research", "pilot-matrix"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_parse_is_stable() {
        assert_eq!(RunPreset::from_name("smoke"), Some(RunPreset::Smoke));
        assert_eq!(RunPreset::Smoke.name(), "smoke");
        assert!(supported_preset_names().contains(&"local-research"));
    }
}
