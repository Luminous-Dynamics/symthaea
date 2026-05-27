// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio Entropy and Surprise Analyzer for Symthaea Muse.
//!
//! Uses mel-spectrogram features and CfC temporal dynamics to track musical
//! expectation and detect topological "surprise" events.

use crate::mel_extractor::{MelConfig, MelExtractor};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use symthaea_fep::ActiveInferenceAgent;

/// Analysis results for a musical segment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioSurpriseReport {
    pub surprise_curve: Vec<f32>,
    pub entropy_curve: Vec<f32>,
    pub max_surprise: f32,
    pub mean_entropy: f32,
    pub surprise_peaks: Vec<usize>, // Frame indices
}

/// The Audio Surprise Meter: tracks musical structure as a topology of expectation.
pub struct AudioSurpriseMeter {
    extractor: MelExtractor,
    agent: ActiveInferenceAgent,
    config: MelConfig,
}

impl AudioSurpriseMeter {
    pub fn new(config: MelConfig) -> Self {
        let n_mels = config.n_mels;
        Self {
            extractor: MelExtractor::new(config.clone()),
            // FEP agent to track expectation in mel-space
            agent: ActiveInferenceAgent::new(symthaea_fep::ActiveInferenceAgentConfig {
                state_dim: 16,
                obs_dim: n_mels,
                num_actions: 1, // Passive observer
                ..Default::default()
            }),
            config,
        }
    }

    /// Analyze audio samples and produce a surprise/entropy report.
    pub fn analyze(&mut self, samples: &[f32]) -> AudioSurpriseReport {
        let frames = self.extractor.extract(samples);
        let mut report = AudioSurpriseReport::default();

        if frames.is_empty() {
            return report;
        }

        let mut total_entropy = 0.0;

        for (i, mel_frame) in frames.iter().enumerate() {
            // 1. Map mel-spectrogram directly to observation
            let obs_values: Vec<f64> = mel_frame.iter().map(|&x| x as f64).collect();
            let observation = symthaea_fep::Observation::new(obs_values, 1.0, "audio_mel");

            // 2. Perform Active Inference step (Observer mode)
            let result = self.agent.perceive(&observation);

            let surprise = result.free_energy.total as f32;
            let entropy = result.updated_belief.entropy() as f32;

            report.surprise_curve.push(surprise);
            report.entropy_curve.push(entropy);

            if surprise > report.max_surprise {
                report.max_surprise = surprise;
            }

            // Peak detection: simple threshold for Muse v0.1
            if surprise > 0.6 {
                report.surprise_peaks.push(i);
            }

            total_entropy += entropy;
        }

        report.mean_entropy = total_entropy / frames.len() as f32;
        report
    }

    /// Convert a mel frame to an HDC AudioVector for manifold integration.
    pub fn encode_to_hdc(&self, mel_frame: &[f32]) -> ContinuousHV {
        // Linear projection of mel bins to 16K HDC space
        ContinuousHV::from_vec(mel_frame.to_vec()).dilate(16384)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surprise_on_sine_sweep() {
        let config = MelConfig::default();
        let mut meter = AudioSurpriseMeter::new(config.clone());

        // Generate 2 seconds of audio: 1s static sine, 1s sweep (high surprise)
        let mut samples = Vec::new();
        let sr = config.sample_rate as f32;

        // Static 440Hz
        for i in 0..sr as usize {
            samples.push((i as f32 * 440.0 * std::f32::consts::TAU / sr).sin());
        }
        // Rapid Sweep to 880Hz
        for i in 0..sr as usize {
            let f = 440.0 + (i as f32 / sr) * 440.0;
            samples.push((i as f32 * f * std::f32::consts::TAU / sr).sin());
        }

        let report = meter.analyze(&samples);
        assert!(!report.surprise_curve.is_empty());

        // Surprise should be higher in the second half (the sweep)
        let first_half_avg = report.surprise_curve[..report.surprise_curve.len() / 2]
            .iter()
            .sum::<f32>();
        let second_half_avg = report.surprise_curve[report.surprise_curve.len() / 2..]
            .iter()
            .sum::<f32>();

        // active inference needs a few cycles to settle, but the sweep is entropic
        assert!(second_half_avg > first_half_avg * 1.1);
    }
}
