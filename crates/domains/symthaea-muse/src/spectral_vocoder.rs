// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CfC-driven spectral synthesizer: consciousness dynamics generate spectral
//! envelopes, rendered by a sinusoidal oscillator bank.
//!
//! The CfC hidden state evolves at ~1kHz control rate, generating 100-bin mel
//! spectral envelopes. One phase-accumulating sine oscillator per mel bin
//! renders these at audio rate (additive resynthesis). This is NOT a
//! phase-reconstruction vocoder — there is no FFT/ISTFT and no Griffin-Lim
//! iteration (an earlier header claimed "Griffin-Lim-like overlap-add").
//! The key insight stands: the **temporal dynamics of the sound itself**
//! emerge from the continuous-time neural ODE, not from note-level decisions.
//!
//! # Architecture
//!
//! ```text
//! MusicalState → HDC encoder → 16,384D HV
//!   → HdcLtcUnifiedNetwork (3 layers, evolve at dt=1ms)
//!   → output HV → MelDecoder → 100-bin mel frame
//!   → per-bin sine oscillator bank → PCM @ 44.1kHz
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::MusicalState;

/// Number of mel frequency bins in the spectral envelope.
pub const MEL_BINS: usize = 64;
/// Control rate: how many CfC steps per second.
pub const CONTROL_RATE_HZ: f32 = 1000.0;

/// Mel decoder: project 16,384D HV back to mel frequency bins.
pub struct MelDecoder {
    /// One projection vector per mel bin (genesis-seeded).
    projections: Vec<ContinuousHV>,
    mel_dim: usize,
}

impl MelDecoder {
    /// Create decoder with genesis-seeded projection vectors.
    pub fn new(mel_dim: usize, genesis: &GenesisSeed) -> Self {
        let projections = (0..mel_dim)
            .map(|i| genesis.hv(&format!("spectral_vocoder_mel_{i}"), HDC_DIMENSION))
            .collect();
        Self {
            projections,
            mel_dim,
        }
    }

    /// Decode a 16,384D HV to mel frame via dot-product projection.
    pub fn decode(&self, hv: &ContinuousHV) -> Vec<f32> {
        self.projections
            .iter()
            .map(|proj| {
                let sim = hv.similarity(proj); // [-1, 1]
                (sim + 1.0) * 0.5 // normalize to [0, 1]
            })
            .collect()
    }
}

/// CfC-driven spectral vocoder.
pub struct SpectralVocoder {
    network: HdcLtcUnifiedNetwork,
    decoder: MelDecoder,
    /// Current phase accumulators for oscillator bank.
    phases: Vec<f32>,
    sample_rate: u32,
    /// Samples per mel frame at control rate.
    hop_size: usize,
}

impl SpectralVocoder {
    /// Create a new spectral vocoder.
    pub fn new(genesis: &GenesisSeed, sample_rate: u32) -> Self {
        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![4, 8, 4],
            neuron_config: UnifiedConfig {
                tau_base: 0.02,
                backbone_tau: 0.3,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);
        let decoder = MelDecoder::new(MEL_BINS, genesis);
        let hop_size = (sample_rate as f32 / CONTROL_RATE_HZ) as usize;

        Self {
            network,
            decoder,
            phases: vec![0.0; MEL_BINS],
            sample_rate,
            hop_size,
        }
    }

    /// Encode consciousness state as input HV for the CfC network.
    fn encode_state(state: &MusicalState, genesis: &GenesisSeed) -> ContinuousHV {
        let channels: Vec<(f32, &str)> = vec![
            (state.consciousness_level, "sv_consciousness"),
            (state.valence, "sv_valence"),
            (state.arousal, "sv_arousal"),
            (state.dopamine, "sv_dopamine"),
            (state.serotonin, "sv_serotonin"),
            (state.noradrenaline, "sv_noradrenaline"),
            (state.harmony_activations[0], "sv_h0"),
            (state.harmony_activations[3], "sv_h3"),
            (state.harmony_activations[7], "sv_h7"),
            (state.prediction_error, "sv_pred_err"),
        ];

        let mut result = ContinuousHV::zero(HDC_DIMENSION);
        for (value, label) in &channels {
            let basis = genesis.hv(label, HDC_DIMENSION);
            result = result.add(&basis.scale(*value));
        }
        result.normalize()
    }

    /// Generate a chunk of PCM audio from consciousness state.
    ///
    /// Returns stereo samples. The CfC network evolves once per mel frame
    /// (~1ms), producing spectral envelopes that the oscillator bank renders.
    pub fn render_chunk(&mut self, state: &MusicalState, chunk_samples: usize) -> Vec<[f32; 2]> {
        let genesis = GenesisSeed::from_phrase("spectral-vocoder-v1");
        let input_hv = Self::encode_state(state, &genesis);
        let sr = self.sample_rate as f32;

        let mut output = vec![[0.0f32; 2]; chunk_samples];
        let dt = 1.0 / CONTROL_RATE_HZ;

        let mut sample_idx = 0;
        while sample_idx < chunk_samples {
            // Evolve CfC network (one control-rate step)
            self.network.evolve_closed_form(dt, &input_hv);
            let output_hv = self.network.output();

            // Decode to mel frame
            let mel = self.decoder.decode(&output_hv);

            // Render mel frame to audio via additive oscillator bank
            let frame_samples = self.hop_size.min(chunk_samples - sample_idx);
            for i in 0..frame_samples {
                let mut sample_l = 0.0f32;
                let mut sample_r = 0.0f32;

                for (bin, &amplitude) in mel.iter().enumerate() {
                    // Mel bin → frequency (simplified linear mapping)
                    let freq = mel_bin_to_hz(bin, MEL_BINS, sr);
                    if freq >= sr * 0.5 {
                        break;
                    }

                    // Phase accumulation
                    self.phases[bin] += freq / sr * std::f32::consts::TAU;
                    if self.phases[bin] > std::f32::consts::TAU {
                        self.phases[bin] -= std::f32::consts::TAU;
                    }

                    let osc = self.phases[bin].sin() * amplitude * 0.05;
                    // Slight stereo spread: odd bins left, even bins right
                    let pan = if bin % 2 == 0 { 0.6 } else { 0.4 };
                    sample_l += osc * (1.0 - pan);
                    sample_r += osc * pan;
                }

                let idx = sample_idx + i;
                if idx < output.len() {
                    output[idx] = [sample_l, sample_r];
                }
            }

            sample_idx += frame_samples;
        }

        output
    }

    /// Reset vocoder state.
    pub fn reset(&mut self) {
        self.phases.fill(0.0);
    }
}

/// Convert mel bin index to frequency in Hz.
/// Uses a simplified log-linear mapping covering 80Hz to Nyquist.
fn mel_bin_to_hz(bin: usize, total_bins: usize, sample_rate: f32) -> f32 {
    let min_hz = 80.0f32;
    let max_hz = sample_rate * 0.45; // slightly below Nyquist
    let t = bin as f32 / total_bins.max(1) as f32;
    // Mel-spaced: linear in mel scale = log in Hz
    min_hz * (max_hz / min_hz).powf(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-spectral-vocoder")
    }

    #[test]
    fn mel_decoder_produces_correct_length() {
        let genesis = test_genesis();
        let decoder = MelDecoder::new(MEL_BINS, &genesis);
        let hv = genesis.hv("test", HDC_DIMENSION);
        let mel = decoder.decode(&hv);
        assert_eq!(mel.len(), MEL_BINS);
    }

    #[test]
    fn mel_values_normalized() {
        let genesis = test_genesis();
        let decoder = MelDecoder::new(MEL_BINS, &genesis);
        let hv = genesis.hv("test", HDC_DIMENSION);
        let mel = decoder.decode(&hv);
        for &v in &mel {
            assert!(v >= 0.0 && v <= 1.0, "mel value out of range: {v}");
        }
    }

    #[test]
    fn vocoder_produces_audio() {
        let genesis = test_genesis();
        let mut vocoder = SpectralVocoder::new(&genesis, 44100);
        let state = MusicalState::default();
        let chunk = vocoder.render_chunk(&state, 1024);
        assert_eq!(chunk.len(), 1024);
        // Should have non-zero audio
        let has_signal = chunk
            .iter()
            .any(|s| s[0].abs() > 0.0001 || s[1].abs() > 0.0001);
        assert!(has_signal, "vocoder should produce audio signal");
    }

    #[test]
    fn different_states_different_output() {
        let genesis = test_genesis();
        let n = 4096; // enough samples for CfC to diverge

        let mut v1 = SpectralVocoder::new(&genesis, 44100);
        let calm = MusicalState {
            consciousness_level: 0.9,
            arousal: 0.1,
            ..Default::default()
        };
        let chunk1 = v1.render_chunk(&calm, n);

        let mut v2 = SpectralVocoder::new(&genesis, 44100);
        let intense = MusicalState {
            consciousness_level: 0.2,
            arousal: 0.9,
            noradrenaline: 0.8,
            ..Default::default()
        };
        let chunk2 = v2.render_chunk(&intense, n);

        // Waveforms should differ (compare sample-by-sample)
        let diff_count = chunk1
            .iter()
            .zip(chunk2.iter())
            .filter(|(a, b)| (a[0] - b[0]).abs() > 1e-6)
            .count();
        assert!(
            diff_count > n / 4,
            "different states should produce different waveforms: {diff_count}/{n} samples differ"
        );
    }

    #[test]
    fn mel_bin_frequency_mapping() {
        let low = mel_bin_to_hz(0, 64, 44100.0);
        let high = mel_bin_to_hz(63, 64, 44100.0);
        assert!(
            low >= 80.0 && low < 200.0,
            "lowest bin should be ~80Hz: {low}"
        );
        assert!(high > 10000.0, "highest bin should be >10kHz: {high}");
        assert!(
            high < 22050.0,
            "highest bin should be below Nyquist: {high}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let genesis = test_genesis();
        let mut vocoder = SpectralVocoder::new(&genesis, 44100);
        let state = MusicalState::default();
        vocoder.render_chunk(&state, 1024);
        assert!(vocoder.phases.iter().any(|&p| p != 0.0));
        vocoder.reset();
        assert!(vocoder.phases.iter().all(|&p| p == 0.0));
    }
}
