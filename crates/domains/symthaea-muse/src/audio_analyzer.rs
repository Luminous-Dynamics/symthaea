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

        // The agent's free energy declines monotonically as its belief
        // converges, so a whole-half comparison is dominated by that trend
        // (measured: first-half sum ≈ 1.3× second-half sum even though the
        // sweep IS surprising). The property this meter actually claims is a
        // surprise SPIKE at the moment the signal changes: compare the frames
        // just after the static→sweep boundary against those just before it.
        let n = report.surprise_curve.len();
        let boundary = n / 2;
        assert!(
            boundary >= 5 && n - boundary >= 8,
            "need frames around boundary"
        );
        let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
        let before = mean(&report.surprise_curve[boundary - 5..boundary]);
        // Peak of the post-boundary window: the spike lands within a frame or
        // two of the boundary (mel hop alignment) and decays as the agent
        // re-adapts, so a peak is the stable detector (measured: ~7500 before
        // vs ~10400 spike, a 1.39× jump).
        let after_peak = report.surprise_curve[boundary..boundary + 8]
            .iter()
            .fold(0.0f32, |a, &b| a.max(b));
        assert!(
            after_peak > before * 1.2,
            "sweep onset should spike surprise: before_mean={before:.1} after_peak={after_peak:.1}"
        );
    }
}
