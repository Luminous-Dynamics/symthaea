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
        "Affect::EmotionalStroop" => DifficultyModel {
            model_type: DifficultyModelType::Interference,
            temp_scale: 0.5,
            snr_reduction: 0.0,
            interference_scale: 0.7,
        },

        // SNR model: benchmarks where signal degradation drives errors
        "Social::RME" => DifficultyModel {
            model_type: DifficultyModelType::Snr,
            temp_scale: 0.3,
            snr_reduction: 0.4,
            interference_scale: 0.0,
        },
        "Metacognition::FOK" => DifficultyModel {
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
        assert!(t1 > t0, "difficulty should increase temperature: {} > {}", t1, t0);
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
        assert!(i1 > i0, "difficulty should amplify interference: {} > {}", i1, i0);
    }

    #[test]
    fn test_default_model_for_unknown() {
        let model = difficulty_model_for("SomeUnknown::Benchmark");
        assert_eq!(model.model_type, DifficultyModelType::Default);
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
}
