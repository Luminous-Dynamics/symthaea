// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Speech-to-Thought Encoder using HDC-LTC unified neuron.
//!
//! Maps mel spectrogram frames to ThoughtChannels through a learned
//! HdcLtcUnifiedNeuron that accumulates prosodic context over an utterance.
//!
//! Architecture:
//!   mel(40D) → weighted basis bundling → ContinuousHV(16384D)
//!              → LTC neuron evolve(dt, mel_hv)
//!              → state(16384D) → 43 channel readout via learned probes
//!
//! The same HdcLtcUnifiedNeuron architecture powers Broca's LanguageController
//! and VocalTract — using it here creates architectural symmetry between
//! speech perception and speech production.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::encoder::{NUM_CHANNELS, ThoughtChannels};

/// Number of mel spectrogram bins (standard for 16kHz audio).
const MEL_BINS: usize = 40;

/// Speech-to-Thought encoder using a single HDC-LTC unified neuron.
///
/// Processes sequential mel spectrogram frames through an LTC neuron
/// whose state accumulates prosodic context. The final state is projected
/// to 43 ThoughtChannels via learned probe vectors.
pub struct SpeechThoughtEncoder {
    /// LTC neuron that evolves with each mel frame.
    neuron: HdcLtcUnifiedNeuron,
    /// Mel-bin basis vectors (genesis-seeded, quasi-orthogonal).
    mel_basis: Vec<ContinuousHV>,
    /// Channel readout probes: similarity(probe_i, state) → channel_i.
    channel_probes: Vec<ContinuousHV>,
    /// Channel biases (added before sigmoid).
    channel_biases: Vec<f32>,
    /// Running count of frames processed (for diagnostics).
    frames_processed: usize,
}

impl SpeechThoughtEncoder {
    /// Create a new encoder with genesis-seeded initialization.
    pub fn new(genesis: &GenesisSeed) -> Self {
        // Configure LTC neuron for speech timescales
        let config = UnifiedConfig {
            tau_base: 0.05,    // 50ms base time constant (vowel duration)
            backbone_tau: 0.3, // State-dependent modulation
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 1e-5,
            ..UnifiedConfig::default()
        };

        // Derive a deterministic u64 seed from genesis phrase
        let seed_bytes = blake3::hash(
            format!("{}::speech_thought_encoder::neuron", genesis.phrase()).as_bytes(),
        );
        let seed_u64 = u64::from_le_bytes(seed_bytes.as_bytes()[..8].try_into().unwrap());

        let neuron = HdcLtcUnifiedNeuron::new(config, seed_u64);

        // Genesis-seeded mel basis vectors (40 quasi-orthogonal HVs)
        let mel_basis: Vec<ContinuousHV> = (0..MEL_BINS)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("speech_encoder::mel_bin_{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        // Genesis-seeded channel probes (43 readout vectors)
        let channel_probes: Vec<ContinuousHV> = (0..NUM_CHANNELS)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("speech_encoder::probe_{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let channel_biases = vec![0.0f32; NUM_CHANNELS];

        Self {
            neuron,
            mel_basis,
            channel_probes,
            channel_biases,
            frames_processed: 0,
        }
    }

    /// Encode a single mel spectrogram frame as a ContinuousHV.
    ///
    /// Uses weighted basis bundling: HV = normalize(Σ mel[i] × basis[i]).
    fn mel_to_hv(&self, mel: &[f32]) -> ContinuousHV {
        let mut values = vec![0.0f32; HDC_DIMENSION];
        for (i, &val) in mel.iter().take(MEL_BINS).enumerate() {
            let weight = val.abs();
            if weight < 1e-8 {
                continue;
            }
            let sign = val.signum();
            for (j, &basis_val) in self.mel_basis[i].as_slice().iter().enumerate() {
                values[j] += sign * weight * basis_val;
            }
        }
        // Normalize to unit length
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut values {
                *v /= norm;
            }
        }
        ContinuousHV::from_values(values)
    }

    /// Process one mel frame through the LTC neuron.
    ///
    /// The neuron state evolves via closed-form LTC dynamics:
    /// x(t+dt) = σ × x_∞ + (1-σ) × x(t)
    pub fn encode_frame(&mut self, mel: &[f32], dt: f32) {
        let mel_hv = self.mel_to_hv(mel);
        self.neuron.evolve(dt, &mel_hv);
        self.frames_processed += 1;
    }

    /// Read current ThoughtChannels from neuron state.
    ///
    /// Each channel is: sigmoid(similarity(probe_i, state) + bias_i).
    pub fn read_channels(&self) -> ThoughtChannels {
        let state = self.neuron.state();
        let mut channels = ThoughtChannels::default();

        for i in 0..NUM_CHANNELS {
            let sim = state.similarity(&self.channel_probes[i]);
            // sigmoid(sim * scale + bias) → [0, 1]
            let logit = sim * 4.0 + self.channel_biases[i]; // scale=4 for steeper sigmoid
            let value = 1.0 / (1.0 + (-logit).exp());
            channels.channels[i] = value;
        }

        channels
    }

    /// Process an entire utterance and return final ThoughtChannels.
    ///
    /// Resets neuron state, processes all mel frames sequentially,
    /// then reads channels from the accumulated state.
    pub fn encode_utterance(&mut self, mel_frames: &[Vec<f32>], dt: f32) -> ThoughtChannels {
        self.reset();
        for frame in mel_frames {
            self.encode_frame(frame, dt);
        }
        self.read_channels()
    }

    /// Reset neuron state between utterances.
    pub fn reset(&mut self) {
        self.neuron.reset();
        self.frames_processed = 0;
    }

    /// Train the channel probes on a single (utterance, target) pair.
    ///
    /// Uses gradient descent on MSE loss between predicted and target channels.
    /// Only updates the probe vectors and biases — the neuron weights are fixed
    /// (they evolve through Hebbian learning during encode_frame).
    pub fn train_step(&mut self, target: &ThoughtChannels, lr: f32) {
        let state = self.neuron.state().clone();
        let state_slice = state.as_slice();

        for i in 0..NUM_CHANNELS {
            let sim = state.similarity(&self.channel_probes[i]);
            let logit = sim * 4.0 + self.channel_biases[i];
            let predicted = 1.0 / (1.0 + (-logit).exp());
            let target_val = target.channels[i];

            let error = predicted - target_val;
            if error.abs() < 1e-6 {
                continue;
            }

            // d_loss/d_logit = 2 × error × sigmoid'(logit) = 2 × error × pred × (1 - pred)
            let sigmoid_deriv = predicted * (1.0 - predicted);
            let d_logit = 2.0 * error * sigmoid_deriv;

            // d_logit/d_probe = 4.0 × state (since logit = 4 × sim(probe, state) + bias)
            // d_logit/d_bias = 1.0
            let probe_lr = lr * d_logit * 4.0;
            let bias_lr = lr * d_logit;

            // Update probe: probe_i -= lr × gradient
            let probe_values = self.channel_probes[i].as_slice().to_vec();
            let mut new_probe = probe_values;
            for (j, v) in new_probe.iter_mut().enumerate() {
                *v -= probe_lr * state_slice[j];
            }
            self.channel_probes[i] = ContinuousHV::from_values(new_probe);

            // Update bias
            self.channel_biases[i] -= bias_lr;
        }
    }

    /// Compute MSE loss between predicted and target channels.
    pub fn loss(&self, target: &ThoughtChannels) -> f32 {
        let predicted = self.read_channels();
        let mut mse = 0.0f32;
        for i in 0..NUM_CHANNELS {
            let diff = predicted.channels[i] - target.channels[i];
            mse += diff * diff;
        }
        mse / NUM_CHANNELS as f32
    }

    /// Number of frames processed since last reset.
    pub fn frames_processed(&self) -> usize {
        self.frames_processed
    }

    /// Save encoder weights (probes + biases) to a binary file.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Header: magic + version + num_channels
        file.write_all(b"SPTE")?; // Speech Thought Encoder
        file.write_all(&1u32.to_le_bytes())?; // version
        file.write_all(&(NUM_CHANNELS as u32).to_le_bytes())?;

        // Probes: NUM_CHANNELS × HDC_DIMENSION floats
        for probe in &self.channel_probes {
            for &v in probe.as_slice() {
                file.write_all(&v.to_le_bytes())?;
            }
        }

        // Biases: NUM_CHANNELS floats
        for &b in &self.channel_biases {
            file.write_all(&b.to_le_bytes())?;
        }

        Ok(())
    }

    /// Load encoder weights from a binary file.
    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        genesis: &GenesisSeed,
    ) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"SPTE" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid speech encoder file",
            ));
        }

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);

        file.read_exact(&mut buf4)?;
        let n_channels = u32::from_le_bytes(buf4) as usize;

        let mut encoder = Self::new(genesis);

        // Load probes
        for i in 0..n_channels.min(NUM_CHANNELS) {
            let mut values = vec![0.0f32; HDC_DIMENSION];
            for v in &mut values {
                file.read_exact(&mut buf4)?;
                *v = f32::from_le_bytes(buf4);
            }
            encoder.channel_probes[i] = ContinuousHV::from_values(values);
        }

        // Load biases
        for i in 0..n_channels.min(NUM_CHANNELS) {
            file.read_exact(&mut buf4)?;
            encoder.channel_biases[i] = f32::from_le_bytes(buf4);
        }

        Ok(encoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creates_and_resets() {
        let genesis = GenesisSeed::from_phrase("test-speech-encoder");
        let mut enc = SpeechThoughtEncoder::new(&genesis);

        // Process a synthetic mel frame
        let mel = vec![0.5f32; MEL_BINS];
        enc.encode_frame(&mel, 0.01);
        assert_eq!(enc.frames_processed(), 1);

        enc.reset();
        assert_eq!(enc.frames_processed(), 0);
    }

    #[test]
    fn test_read_channels_returns_valid_range() {
        let genesis = GenesisSeed::from_phrase("test-speech-encoder");
        let mut enc = SpeechThoughtEncoder::new(&genesis);

        // Process some frames
        for _ in 0..10 {
            let mel = vec![0.3f32; MEL_BINS];
            enc.encode_frame(&mel, 0.01);
        }

        let channels = enc.read_channels();
        for i in 0..NUM_CHANNELS {
            assert!(
                channels.channels[i] >= 0.0 && channels.channels[i] <= 1.0,
                "Channel {i} out of range: {}",
                channels.channels[i]
            );
        }
    }

    #[test]
    fn test_different_inputs_produce_different_channels() {
        let genesis = GenesisSeed::from_phrase("test-speech-encoder");

        // Encode a "bright" utterance (high-frequency energy)
        let mut enc = SpeechThoughtEncoder::new(&genesis);
        for _ in 0..20 {
            let mut mel = vec![0.0f32; MEL_BINS];
            for i in 20..40 {
                mel[i] = 0.8; // High frequency
            }
            enc.encode_frame(&mel, 0.01);
        }
        let bright = enc.read_channels();

        // Encode a "dark" utterance (low-frequency energy)
        enc.reset();
        for _ in 0..20 {
            let mut mel = vec![0.0f32; MEL_BINS];
            for i in 0..20 {
                mel[i] = 0.8; // Low frequency
            }
            enc.encode_frame(&mel, 0.01);
        }
        let dark = enc.read_channels();

        // They should differ
        let diff: f32 = bright
            .channels
            .iter()
            .zip(dark.channels.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "Bright and dark should differ: {diff}");
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let genesis = GenesisSeed::from_phrase("test-speech-encoder-train");
        let mut enc = SpeechThoughtEncoder::new(&genesis);

        // Create a target
        let mut target = ThoughtChannels::default();
        target.channels[9] = 0.8; // High valence
        target.channels[10] = 0.6; // Moderate arousal

        // Encode some frames
        for _ in 0..10 {
            let mel = vec![0.5f32; MEL_BINS];
            enc.encode_frame(&mel, 0.01);
        }

        let loss_before = enc.loss(&target);

        // Train for several steps
        for _ in 0..20 {
            enc.train_step(&target, 0.01);
        }

        let loss_after = enc.loss(&target);
        assert!(
            loss_after < loss_before,
            "Loss should decrease: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let genesis = GenesisSeed::from_phrase("test-speech-encoder-save");
        let enc = SpeechThoughtEncoder::new(&genesis);

        let path = std::env::temp_dir().join("test_speech_encoder.bin");
        enc.save(&path).unwrap();

        let loaded = SpeechThoughtEncoder::load(&path, &genesis).unwrap();

        // Verify probes match
        for i in 0..NUM_CHANNELS {
            let sim = enc.channel_probes[i].similarity(&loaded.channel_probes[i]);
            assert!(sim > 0.999, "Probe {i} mismatch after save/load: sim={sim}");
        }

        let _ = std::fs::remove_file(&path);
    }
}
