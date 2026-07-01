// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Difficulty scaling models for calibrated benchmark performance.
//!
//! Provides a `difficulty` parameter (0.0 = current/easy, 1.0 = hard) that modulates
//! benchmark parameters to push accuracy into human-like ranges (70-95%).
//!
//! Three model types:
//! - **Default**: temperature-only scaling (increases decision stochasticity)
//! - **Interference**: amplifies competing activations (Stroop, Flanker, Bimanual)
//! - **SNR**: degrades signal-to-noise ratio (RME, FOK, ToM, SemanticPriming)

use serde::{Deserialize, Serialize};

/// Difficulty model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyModelType {
    /// Temperature scaling only — increases response stochasticity.
    Default,
    /// Amplifies competing/interfering activations.
    Interference,
    /// Degrades signal-to-noise ratio (weaker encoding).
    Snr,
}

/// Difficulty model parameters for a benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyModel {
    /// Model type.
    pub model_type: DifficultyModelType,
    /// Temperature scale factor: temperature *= 1.0 + temp_scale * difficulty.
    pub temp_scale: f64,
    /// SNR reduction: signal_strength *= 1.0 - snr_reduction * difficulty.
    pub snr_reduction: f64,
    /// Interference amplification: interference *= 1.0 + interference_scale * difficulty.
    pub interference_scale: f64,
}

impl DifficultyModel {
    /// Apply temperature scaling given difficulty level.
    /// Returns a multiplier for the benchmark's temperature parameter.
    pub fn temperature_multiplier(&self, difficulty: f64) -> f64 {
        1.0 + self.temp_scale * difficulty.clamp(0.0, 1.0)
    }

    /// Apply SNR scaling given difficulty level.
    /// Returns a multiplier for signal strength (< 1.0 means weaker signal).
    pub fn signal_multiplier(&self, difficulty: f64) -> f64 {
        (1.0 - self.snr_reduction * difficulty.clamp(0.0, 1.0)).max(0.1)
    }

    /// Apply interference scaling given difficulty level.
    /// Returns a multiplier for interfering activation strength.
    pub fn interference_multiplier(&self, difficulty: f64) -> f64 {
        1.0 + self.interference_scale * difficulty.clamp(0.0, 1.0)
    }
}

/// Registry: returns the appropriate difficulty model for a benchmark name.
pub fn difficulty_model_for(name: &str) -> DifficultyModel {
    match name {
        // Interference model: benchmarks where competing activations drive errors
        "Executive::Stroop" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.6,
            snr_reduction: 0.0,
            interference_scale: 0.8, // reading_automaticity amplification
        },
        "Executive::Flanker" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.5,
            snr_reduction: 0.0,
            interference_scale: 0.7,
        },
        "Motor::Bimanual" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.4,
            snr_reduction: 0.0,
            interference_scale: 0.6,
        },
        "Motor::ProprioceptiveDrift" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.35,
            snr_reduction: 0.40, // multisensory signal degradation
            interference_scale: 0.0,
        },
        "Affect::EmotionalStroop" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.5,
            snr_reduction: 0.0,
            interference_scale: 0.7,
        },

        // Interference model: working memory benchmarks with competing representations
        "WorM::N-back" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.3,
            snr_reduction: 0.0,
            interference_scale: 0.50, // lure interference from recent non-target items
        },
        "WorM::DigitSpan" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.25,
            snr_reduction: 0.0,
            interference_scale: 0.40, // proactive interference from prior lists
        },
        "WorM::SerialRecall" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.3,
            snr_reduction: 0.0,
            interference_scale: 0.45, // output interference from earlier recalls
        },
        "WorM::Binding" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.25,
            snr_reduction: 0.0,
            interference_scale: 0.35, // feature binding crosstalk between items
        },
        "WorM::ChangeDetection" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.2,
            snr_reduction: 0.0,
            interference_scale: 0.30, // change blindness from similar distractors
        },
        "WorM::SpatialUpdating" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.3,
            snr_reduction: 0.0,
            interference_scale: 0.40, // spatial interference from prior positions
        },

        // Interference model: memory agent tasks with temporal competition
        "MemoryAgent::LongRange" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.35,
            snr_reduction: 0.0,
            interference_scale: 0.50, // temporal interference across long delays
        },
        "MemoryAgent::ConflictResolution" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.4,
            snr_reduction: 0.0,
            interference_scale: 0.55, // source confusion between conflicting memories
        },

        // SNR model: benchmarks where signal degradation drives errors

        // Executive / inhibition tasks — noise in control signals
        "Executive::WCST" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.40, // set-shifting noise: rule feedback degrades
            interference_scale: 0.0,
        },
        "Inhibition::StopSignal" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.35,
            snr_reduction: 0.45, // stop signal degradation: weaker inhibitory cue
            interference_scale: 0.0,
        },

        // Attention tasks — target signal masked or degraded
        "Attention::AttentionalBlink" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.4,
            snr_reduction: 0.50, // T2 masking: second target deeply masked
            interference_scale: 0.0,
        },
        "Attention::VisualSearch" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.35, // distractor noise: target-distractor similarity
            interference_scale: 0.0,
        },

        // Sustained attention tasks — vigilance decrement
        "SustainedAttention::CPT" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.25,
            snr_reduction: 0.30, // target degradation: perceptual load on go trials
            interference_scale: 0.0,
        },
        "SustainedAttention::SART" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.35, // inhibition noise: no-go target less salient
            interference_scale: 0.0,
        },
        "SustainedAttention::PVT" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.2,
            snr_reduction: 0.25, // vigilance signal: stimulus intensity decrement
            interference_scale: 0.0,
        },

        // Social cognition — cue subtlety
        "ToMBench::Hinting" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.35, // social cue noise: indirect hints harder to detect
            interference_scale: 0.0,
        },

        "Social::RME" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.4,
            interference_scale: 0.0,
        },
        "Metacognition::FeelingOfKnowing" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.35,
            interference_scale: 0.0,
        },
        "ToMBench::FalseBelief" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.4,
            snr_reduction: 0.3,
            interference_scale: 0.0,
        },
        "Language::SemanticPriming" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.35,
            interference_scale: 0.0,
        },
        "Language::LexicalDecision" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.3,
            interference_scale: 0.0,
        },
        "Social::UltimatumGame" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.5,
            snr_reduction: 0.2,
            interference_scale: 0.0,
        },
        "Social::SocialNorm" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.4,
            snr_reduction: 0.3,
            interference_scale: 0.0,
        },
        "Social::PrisonersDilemma" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.5,
            snr_reduction: 0.25,
            interference_scale: 0.0,
        },
        "Social::PublicGoods" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.45,
            snr_reduction: 0.30,
            interference_scale: 0.0,
        },
        "Social::DictatorGame" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.4,
            snr_reduction: 0.20,
            interference_scale: 0.0,
        },
        "Social::Machiavelli" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.5,
            snr_reduction: 0.25,
            interference_scale: 0.0,
        },
        "ToMBench::FauxPas" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.4,
            snr_reduction: 0.55,
            interference_scale: 0.0,
        },
        "ToMBench::Persuasion" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.4,
            snr_reduction: 0.50,
            interference_scale: 0.0,
        },
        "ToMBench::StrangeStory" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.35,
            snr_reduction: 0.45,
            interference_scale: 0.0,
        },

        "Substrate::Transfer" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.35, // substrate noise amplification
            interference_scale: 0.0,
        },

        // Binding — temporal discrimination is SNR-limited
        "Binding::TemporalOrder" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.40, // temporal position signal degrades
            interference_scale: 0.0,
        },

        // Speech — categorical perception is SNR-limited
        "Speech::PhonemeDiscrimination" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.35,
            snr_reduction: 0.35, // phoneme encoding signal degrades
            interference_scale: 0.0,
        },

        // Consciousness — blindsight is SNR-limited (threshold detection)
        "Consciousness::Blindsight" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.40, // stimulus signal degrades with difficulty
            interference_scale: 0.0,
        },

        // Attention — mismatch negativity is SNR-limited (deviant detection)
        "Attention::MismatchNegativity" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.35, // prediction error signal degrades
            interference_scale: 0.0,
        },

        // Metacognition — change blindness is SNR-limited (scene comparison)
        "Metacognition::ChangeBlindness" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.40, // change signal diluted by scene complexity
            interference_scale: 0.0,
        },

        // Default model: temperature-only scaling for everything else
        _ => DifficultyModel {
            model_type: DifficultyModelType::Default,
            temp_scale: 0.8,
            snr_reduction: 0.0,
            interference_scale: 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difficulty_zero_is_identity() {
        let model = difficulty_model_for("Executive::Stroop");
        assert!((model.temperature_multiplier(0.0) - 1.0).abs() < 1e-10);
        assert!((model.signal_multiplier(0.0) - 1.0).abs() < 1e-10);
        assert!((model.interference_multiplier(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_difficulty_increases_temperature() {
        let model = difficulty_model_for("Executive::Stroop");
        let t0 = model.temperature_multiplier(0.0);
        let t1 = model.temperature_multiplier(1.0);
        assert!(
            t1 > t0,
            "difficulty should increase temperature: {} > {}",
            t1,
            t0
        );
    }

    #[test]
    fn test_snr_model_degrades_signal() {
        let model = difficulty_model_for("Social::RME");
        assert_eq!(model.model_type, DifficultyModelType::Snr);
        let s0 = model.signal_multiplier(0.0);
        let s1 = model.signal_multiplier(1.0);
        assert!(s1 < s0, "difficulty should degrade signal: {} < {}", s1, s0);
        assert!(s1 > 0.0, "signal should never reach zero");
    }

    #[test]
    fn test_interference_model_amplifies() {
        let model = difficulty_model_for("Executive::Stroop");
        assert_eq!(model.model_type, DifficultyModelType::Interference);
        let i0 = model.interference_multiplier(0.0);
        let i1 = model.interference_multiplier(1.0);
        assert!(
            i1 > i0,
            "difficulty should amplify interference: {} > {}",
            i1,
            i0
        );
    }

    #[test]
    fn test_default_model_for_unknown() {
        let model = difficulty_model_for("SomeUnknown::Benchmark");
        assert_eq!(model.model_type, DifficultyModelType::Default);
    }

    #[test]
    fn test_tombench_faux_pas_snr() {
        let model = difficulty_model_for("ToMBench::FauxPas");
        assert_eq!(model.model_type, DifficultyModelType::Snr);
        assert!((model.snr_reduction - 0.55).abs() < 1e-10);
        let s = model.signal_multiplier(1.0);
        assert!(s < 1.0 && s > 0.0);
    }

    #[test]
    fn test_tombench_persuasion_snr() {
        let model = difficulty_model_for("ToMBench::Persuasion");
        assert_eq!(model.model_type, DifficultyModelType::Snr);
        assert!((model.snr_reduction - 0.50).abs() < 1e-10);
    }

    #[test]
    fn test_tombench_strange_story_snr() {
        let model = difficulty_model_for("ToMBench::StrangeStory");
        assert_eq!(model.model_type, DifficultyModelType::Snr);
        assert!((model.snr_reduction - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_difficulty_clamped() {
        let model = difficulty_model_for("Executive::Stroop");
        // Values outside [0, 1] should be clamped
        let t_neg = model.temperature_multiplier(-1.0);
        let t_over = model.temperature_multiplier(2.0);
        assert!((t_neg - model.temperature_multiplier(0.0)).abs() < 1e-10);
        assert!((t_over - model.temperature_multiplier(1.0)).abs() < 1e-10);
    }

    /// All registered benchmark difficulty models have valid, consistent parameters.
    #[test]
    fn test_difficulty_model_coverage() {
        let registered = [
            // Interference models
            "Executive::Stroop",
            "Executive::Flanker",
            "Motor::Bimanual",
            "Affect::EmotionalStroop",
            "WorM::N-back",
            "WorM::DigitSpan",
            "WorM::SerialRecall",
            "WorM::Binding",
            "WorM::ChangeDetection",
            "WorM::SpatialUpdating",
            "MemoryAgent::LongRange",
            "MemoryAgent::ConflictResolution",
            // SNR models
            "Executive::WCST",
            "Inhibition::StopSignal",
            "Attention::AttentionalBlink",
            "Attention::VisualSearch",
            "SustainedAttention::CPT",
            "SustainedAttention::SART",
            "SustainedAttention::PVT",
            "ToMBench::Hinting",
            "Social::RME",
            "Metacognition::FeelingOfKnowing",
            "ToMBench::FalseBelief",
            "Language::SemanticPriming",
            "Language::LexicalDecision",
            "Social::UltimatumGame",
            "Social::SocialNorm",
            "Social::PrisonersDilemma",
            "Social::PublicGoods",
            "Social::DictatorGame",
            "Social::Machiavelli",
            "ToMBench::FauxPas",
            "ToMBench::Persuasion",
            "ToMBench::StrangeStory",
            "Motor::ProprioceptiveDrift",
            "Substrate::Transfer",
            "Binding::TemporalOrder",
            "Speech::PhonemeDiscrimination",
            "Consciousness::Blindsight",
            "Attention::MismatchNegativity",
            "Metacognition::ChangeBlindness",
        ];

        for name in &registered {
            let model = difficulty_model_for(name);

            // Must NOT be Default — every registered name should have a specific model
            assert_ne!(
                model.model_type,
                DifficultyModelType::Default,
                "{name} fell through to Default model"
            );

            // temp_scale must be positive
            assert!(
                model.temp_scale > 0.0,
                "{name}: temp_scale must be positive, got {}",
                model.temp_scale
            );

            // SNR-specific checks
            if model.model_type == DifficultyModelType::Snr {
                assert!(
                    model.snr_reduction > 0.0 && model.snr_reduction <= 1.0,
                    "{name}: snr_reduction must be in (0, 1], got {}",
                    model.snr_reduction
                );
                assert!(
                    model.interference_scale == 0.0,
                    "{name}: SNR model should have zero interference_scale, got {}",
                    model.interference_scale
                );
                // Signal at max difficulty must remain positive
                let sig = model.signal_multiplier(1.0);
                assert!(
                    sig > 0.0,
                    "{name}: signal_multiplier at difficulty=1.0 must be > 0, got {}",
                    sig
                );
            }

            // Interference-specific checks
            if model.model_type == DifficultyModelType::Interference {
                assert!(
                    model.interference_scale > 0.0 && model.interference_scale <= 1.0,
                    "{name}: interference_scale must be in (0, 1], got {}",
                    model.interference_scale
                );
                assert!(
                    model.snr_reduction == 0.0,
                    "{name}: Interference model should have zero snr_reduction, got {}",
                    model.snr_reduction
                );
            }

            // Monotonicity: difficulty=1 must be harder than difficulty=0
            let t0 = model.temperature_multiplier(0.0);
            let t1 = model.temperature_multiplier(1.0);
            assert!(
                t1 > t0,
                "{name}: temperature must increase with difficulty ({t0} -> {t1})"
            );
        }
    }
}
