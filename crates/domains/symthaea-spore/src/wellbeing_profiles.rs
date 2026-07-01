// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wellbeing Profiles: named neuromodulator configurations for neurodiversity support.
//!
//! Each profile adjusts the neuromodulator bath biases and CfC temporal dynamics
//! to support different cognitive/emotional needs. These are experimental wellness
//! aids — NOT medical devices or treatments.
//!
//! ## Available Profiles
//!
//! - **Default**: No modifications — baseline consciousness dynamics.
//! - **ADHD Support**: Elevated DA baseline, faster CfC dynamics, focus gamification.
//! - **Anxiety Buffer**: OT + 5-HT bias activated when allostatic load is high.
//! - **Grief Mode**: Slower CfC tau, dream replay bias, Sacred Stillness emphasis.
//! - **Sensory Gentle**: Reduced NE sensitivity, dampened surprise response.
//! - **Focus Flow**: Moderate DA + ACh boost for sustained attention.
//!
//! ## Safety
//!
//! These profiles modulate a consciousness *simulation*, not a biological brain.
//! They affect how the AI companion responds, not the user's neurochemistry.
//! Nevertheless, the companion's behavior can influence the user's emotional state,
//! so all profiles are designed to be gentle and non-manipulative.

use serde::{Deserialize, Serialize};

/// Named wellbeing profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WellbeingProfile {
    /// No modifications — baseline consciousness dynamics.
    Default,
    /// Elevated dopamine, faster temporal dynamics, focus gamification.
    /// Designed for users who benefit from novelty and structured engagement.
    AdhdSupport,
    /// Oxytocin + serotonin bias when stress is detected.
    /// Designed for users who experience anxiety or social discomfort.
    AnxietyBuffer,
    /// Slower CfC dynamics, dream replay bias, Sacred Stillness emphasis.
    /// Designed for users processing loss or grief.
    GriefMode,
    /// Reduced norepinephrine sensitivity, dampened surprise.
    /// Designed for users who are easily overstimulated.
    SensoryGentle,
    /// Moderate dopamine + focus, sustained attention support.
    FocusFlow,
}

impl WellbeingProfile {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::AdhdSupport => "ADHD Support",
            Self::AnxietyBuffer => "Anxiety Buffer",
            Self::GriefMode => "Grief Mode",
            Self::SensoryGentle => "Sensory Gentle",
            Self::FocusFlow => "Focus Flow",
        }
    }

    /// Brief description of the profile's purpose.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Default => "Baseline consciousness dynamics with no modifications.",
            Self::AdhdSupport => "Elevated novelty response and faster dynamics for focus support.",
            Self::AnxietyBuffer => {
                "Calming bias activated during high stress — warmth and serenity."
            }
            Self::GriefMode => "Slower, gentler dynamics with emphasis on rest and memory.",
            Self::SensoryGentle => "Dampened surprise and reduced alertness for sensory comfort.",
            Self::FocusFlow => "Sustained attention support with moderate engagement boost.",
        }
    }

    /// Parse from a string name (case-insensitive, underscore/hyphen tolerant).
    pub fn from_name(name: &str) -> Option<Self> {
        match name
            .to_lowercase()
            .replace('-', "_")
            .replace(' ', "_")
            .as_str()
        {
            "default" => Some(Self::Default),
            "adhd_support" | "adhd" => Some(Self::AdhdSupport),
            "anxiety_buffer" | "anxiety" => Some(Self::AnxietyBuffer),
            "grief_mode" | "grief" => Some(Self::GriefMode),
            "sensory_gentle" | "sensory" => Some(Self::SensoryGentle),
            "focus_flow" | "focus" => Some(Self::FocusFlow),
            _ => None,
        }
    }

    /// All available profiles.
    pub fn all() -> &'static [Self] {
        &[
            Self::Default,
            Self::AdhdSupport,
            Self::AnxietyBuffer,
            Self::GriefMode,
            Self::SensoryGentle,
            Self::FocusFlow,
        ]
    }
}

impl Default for WellbeingProfile {
    fn default() -> Self {
        Self::Default
    }
}

/// Neuromodulator bias adjustments from a wellbeing profile.
///
/// These are additive biases applied to the bath's resting levels each cycle.
/// Positive values increase the transmitter's baseline; negative values decrease it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeuromodBias {
    /// Dopamine bias (motivation, reward, novelty).
    pub dopamine: f32,
    /// Norepinephrine bias (alertness, surprise response).
    pub norepinephrine: f32,
    /// Serotonin bias (serenity, mood stability).
    pub serotonin: f32,
    /// Oxytocin bias (warmth, social bonding).
    pub oxytocin: f32,
}

impl Default for NeuromodBias {
    fn default() -> Self {
        Self {
            dopamine: 0.0,
            norepinephrine: 0.0,
            serotonin: 0.0,
            oxytocin: 0.0,
        }
    }
}

/// Temporal dynamics adjustments from a wellbeing profile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TemporalAdjustment {
    /// CfC time constant multiplier (1.0 = normal, >1.0 = slower, <1.0 = faster).
    pub tau_multiplier: f32,
    /// Surprise response dampening (0.0 = no dampening, 1.0 = fully dampened).
    pub surprise_dampening: f32,
    /// Sacred Stillness harmony weight boost (additive, 0.0 = no boost).
    pub stillness_weight_boost: f32,
    /// Dream replay frequency multiplier (1.0 = normal, >1.0 = more frequent).
    pub dream_replay_multiplier: f32,
}

impl Default for TemporalAdjustment {
    fn default() -> Self {
        Self {
            tau_multiplier: 1.0,
            surprise_dampening: 0.0,
            stillness_weight_boost: 0.0,
            dream_replay_multiplier: 1.0,
        }
    }
}

/// Complete wellbeing profile configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WellbeingConfig {
    pub profile: WellbeingProfile,
    pub neuromod_bias: NeuromodBias,
    pub temporal: TemporalAdjustment,
    /// Whether the anxiety buffer activates conditionally (only when allostatic_load > threshold).
    pub conditional_activation: bool,
    /// Allostatic load threshold for conditional activation.
    pub activation_threshold: f32,
}

impl WellbeingConfig {
    /// Build the configuration for a given profile.
    pub fn for_profile(profile: WellbeingProfile) -> Self {
        match profile {
            WellbeingProfile::Default => Self {
                profile,
                neuromod_bias: NeuromodBias::default(),
                temporal: TemporalAdjustment::default(),
                conditional_activation: false,
                activation_threshold: 0.0,
            },

            WellbeingProfile::AdhdSupport => Self {
                profile,
                neuromod_bias: NeuromodBias {
                    dopamine: 0.10,       // Elevated novelty/reward baseline
                    norepinephrine: 0.03, // Slight alertness boost
                    serotonin: 0.0,
                    oxytocin: 0.0,
                },
                temporal: TemporalAdjustment {
                    tau_multiplier: 0.85, // Faster temporal dynamics (shorter attention window)
                    surprise_dampening: 0.0,
                    stillness_weight_boost: 0.0,
                    dream_replay_multiplier: 1.0,
                },
                conditional_activation: false,
                activation_threshold: 0.0,
            },

            WellbeingProfile::AnxietyBuffer => Self {
                profile,
                neuromod_bias: NeuromodBias {
                    dopamine: 0.0,
                    norepinephrine: -0.05, // Reduce alertness/startle
                    serotonin: 0.05,       // Calming baseline
                    oxytocin: 0.05,        // Warmth/safety
                },
                temporal: TemporalAdjustment {
                    tau_multiplier: 1.15,    // Slightly slower (more deliberate)
                    surprise_dampening: 0.3, // Reduce surprise spikes
                    stillness_weight_boost: 0.05,
                    dream_replay_multiplier: 1.0,
                },
                conditional_activation: true, // Only activates when stressed
                activation_threshold: 0.7,    // Allostatic load > 0.7 triggers
            },

            WellbeingProfile::GriefMode => Self {
                profile,
                neuromod_bias: NeuromodBias {
                    dopamine: -0.03,       // Reduced reward-seeking (honor the mourning)
                    norepinephrine: -0.05, // Reduce alertness
                    serotonin: 0.05,       // Gentle serenity
                    oxytocin: 0.08,        // Warmth and comfort
                },
                temporal: TemporalAdjustment {
                    tau_multiplier: 1.5, // Slower temporal flow (spacious time)
                    surprise_dampening: 0.2,
                    stillness_weight_boost: 0.10, // Sacred Stillness emphasis
                    dream_replay_multiplier: 1.5, // More dream consolidation (memory processing)
                },
                conditional_activation: false,
                activation_threshold: 0.0,
            },

            WellbeingProfile::SensoryGentle => Self {
                profile,
                neuromod_bias: NeuromodBias {
                    dopamine: 0.0,
                    norepinephrine: -0.08, // Significantly reduced alertness
                    serotonin: 0.03,       // Mild calming
                    oxytocin: 0.0,
                },
                temporal: TemporalAdjustment {
                    tau_multiplier: 1.1,
                    surprise_dampening: 0.5, // Heavy surprise dampening
                    stillness_weight_boost: 0.03,
                    dream_replay_multiplier: 1.0,
                },
                conditional_activation: false,
                activation_threshold: 0.0,
            },

            WellbeingProfile::FocusFlow => Self {
                profile,
                neuromod_bias: NeuromodBias {
                    dopamine: 0.05,       // Moderate engagement boost
                    norepinephrine: 0.02, // Mild alertness
                    serotonin: 0.02,      // Mood stability
                    oxytocin: 0.0,
                },
                temporal: TemporalAdjustment {
                    tau_multiplier: 0.95,    // Slightly faster
                    surprise_dampening: 0.1, // Mild dampening (reduce distractions)
                    stillness_weight_boost: 0.0,
                    dream_replay_multiplier: 1.0,
                },
                conditional_activation: false,
                activation_threshold: 0.0,
            },
        }
    }

    /// Check if the profile's effects should be active given current allostatic load.
    pub fn is_active(&self, allostatic_load: f32) -> bool {
        if !self.conditional_activation {
            return true; // Always active
        }
        allostatic_load >= self.activation_threshold
    }

    /// Get the effective neuromod bias, considering conditional activation.
    pub fn effective_bias(&self, allostatic_load: f32) -> NeuromodBias {
        if self.is_active(allostatic_load) {
            self.neuromod_bias
        } else {
            NeuromodBias::default()
        }
    }

    /// Get the effective temporal adjustment, considering conditional activation.
    pub fn effective_temporal(&self, allostatic_load: f32) -> TemporalAdjustment {
        if self.is_active(allostatic_load) {
            self.temporal
        } else {
            TemporalAdjustment::default()
        }
    }
}

impl Default for WellbeingConfig {
    fn default() -> Self {
        Self::for_profile(WellbeingProfile::Default)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_profiles_have_unique_names() {
        let profiles = WellbeingProfile::all();
        let names: Vec<&str> = profiles.iter().map(|p| p.name()).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "duplicate profile name");
                }
            }
        }
    }

    #[test]
    fn test_profile_parsing() {
        assert_eq!(
            WellbeingProfile::from_name("adhd_support"),
            Some(WellbeingProfile::AdhdSupport)
        );
        assert_eq!(
            WellbeingProfile::from_name("ADHD-Support"),
            Some(WellbeingProfile::AdhdSupport)
        );
        assert_eq!(
            WellbeingProfile::from_name("adhd"),
            Some(WellbeingProfile::AdhdSupport)
        );
        assert_eq!(
            WellbeingProfile::from_name("grief"),
            Some(WellbeingProfile::GriefMode)
        );
        assert_eq!(WellbeingProfile::from_name("nonexistent"), None);
    }

    #[test]
    fn test_default_profile_no_bias() {
        let config = WellbeingConfig::for_profile(WellbeingProfile::Default);
        assert_eq!(config.neuromod_bias.dopamine, 0.0);
        assert_eq!(config.neuromod_bias.norepinephrine, 0.0);
        assert_eq!(config.neuromod_bias.serotonin, 0.0);
        assert_eq!(config.neuromod_bias.oxytocin, 0.0);
        assert_eq!(config.temporal.tau_multiplier, 1.0);
    }

    #[test]
    fn test_adhd_elevates_dopamine() {
        let config = WellbeingConfig::for_profile(WellbeingProfile::AdhdSupport);
        assert!(config.neuromod_bias.dopamine > 0.0);
        assert!(config.temporal.tau_multiplier < 1.0); // Faster dynamics
    }

    #[test]
    fn test_anxiety_buffer_conditional() {
        let config = WellbeingConfig::for_profile(WellbeingProfile::AnxietyBuffer);
        assert!(config.conditional_activation);

        // Below threshold — no effect
        assert!(!config.is_active(0.5));
        let bias = config.effective_bias(0.5);
        assert_eq!(bias.serotonin, 0.0);

        // Above threshold — active
        assert!(config.is_active(0.8));
        let bias = config.effective_bias(0.8);
        assert!(bias.serotonin > 0.0);
        assert!(bias.oxytocin > 0.0);
    }

    #[test]
    fn test_grief_mode_slow_and_warm() {
        let config = WellbeingConfig::for_profile(WellbeingProfile::GriefMode);
        assert!(config.temporal.tau_multiplier > 1.0); // Slower
        assert!(config.neuromod_bias.oxytocin > 0.0); // Warmth
        assert!(config.temporal.stillness_weight_boost > 0.0); // Sacred Stillness
        assert!(config.temporal.dream_replay_multiplier > 1.0); // More dreams
    }

    #[test]
    fn test_sensory_gentle_dampened() {
        let config = WellbeingConfig::for_profile(WellbeingProfile::SensoryGentle);
        assert!(config.neuromod_bias.norepinephrine < 0.0); // Reduced alertness
        assert!(config.temporal.surprise_dampening > 0.3); // Heavy dampening
    }

    #[test]
    fn test_all_profiles_serialize() {
        for profile in WellbeingProfile::all() {
            let config = WellbeingConfig::for_profile(*profile);
            let json = serde_json::to_string(&config).expect("serialize failed");
            let restored: WellbeingConfig =
                serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(restored.profile, *profile);
        }
    }

    #[test]
    fn test_focus_flow_balanced() {
        let config = WellbeingConfig::for_profile(WellbeingProfile::FocusFlow);
        assert!(config.neuromod_bias.dopamine > 0.0);
        assert!(config.neuromod_bias.serotonin > 0.0); // Mood stability
        assert!(config.temporal.tau_multiplier < 1.0); // Slightly faster
    }

    #[test]
    fn test_non_conditional_always_active() {
        let config = WellbeingConfig::for_profile(WellbeingProfile::GriefMode);
        assert!(!config.conditional_activation);
        assert!(config.is_active(0.0));
        assert!(config.is_active(1.0));
    }
}
