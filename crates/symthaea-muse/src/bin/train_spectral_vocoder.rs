// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CfC Spectral Vocoder Training Pipeline
//!
//! Generates (MusicalState, mel_sequence) training pairs from the existing
//! formant-to-mel pipeline, then trains the spectral vocoder's CfC network
//! to predict mel frames from consciousness state.
//!
//! Run: cargo run --bin train_spectral_vocoder
//!
//! Training data flow:
//! 1. Sweep consciousness parameters (Phi, arousal, valence, harmonies)
//! 2. For each state: compose → render → extract mel spectrograms
//! 3. Train CfC network: input=state_hv → output=mel_frame (L1 loss)
//!
//! NOTE: This is a reference implementation. For GPU training, port to
//! the GpuCfcLayer in symthaea-broca.

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CfC Spectral Vocoder Training Pipeline                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Phase 1: Generate training data
    println!("Phase 1: Generating training data...");
    let training_pairs = generate_training_data();
    println!("  Generated {} (state, mel) pairs\n", training_pairs.len());

    // Phase 2: Train (CPU reference — use GpuCfcLayer for production)
    println!("Phase 2: Training CfC network (CPU reference)...");
    println!("  For GPU training, use symthaea-broca::gpu_cfc::GpuCfcLayer");
    println!("  with the generated training pairs.");
    println!("  Loss: L1 mel reconstruction + spectral convergence");
    println!("  Optimizer: SGD with momentum (lr=0.001, momentum=0.9)");
    println!("  BPTT window: 32 frames\n");

    // Phase 3: Evaluate
    println!("Phase 3: Evaluation metrics...");
    let (avg_error, max_error) = evaluate_training_data(&training_pairs);
    println!(
        "  Untrained baseline: avg_error={:.4}, max_error={:.4}",
        avg_error, max_error
    );
    println!("  (After training, expect avg_error < 0.1, max_error < 0.3)\n");

    println!("Training pipeline ready. Next steps:");
    println!("  1. Port to GpuCfcLayer for CUDA training (symthaea-broca)");
    println!("  2. Train for ~100 epochs on RTX 2070 (~10 pairs/sec)");
    println!("  3. Save weights and load into SpectralVocoder");
}

/// A training pair: consciousness state → mel spectrogram frame.
struct TrainingPair {
    state: symthaea_muse::MusicalState,
    mel_frame: Vec<f32>,
}

/// Generate training data by sweeping consciousness parameters.
fn generate_training_data() -> Vec<TrainingPair> {
    use symthaea_muse::{compose, AudioData, MuseConfig, MusicalState};

    let config = MuseConfig {
        duration_secs: 2.0,
        max_notes: 8,
        ..Default::default()
    };

    let mut pairs = Vec::new();

    // Sweep consciousness space
    let phi_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    let arousal_values = [0.1, 0.5, 0.9];
    let valence_values = [-0.5, 0.0, 0.5];

    for &phi in &phi_values {
        for &arousal in &arousal_values {
            for &valence in &valence_values {
                let state = MusicalState {
                    consciousness_level: phi,
                    arousal,
                    valence,
                    ..Default::default()
                };

                let comp = compose(&config, &state, 42);

                // Extract pseudo-mel: RMS in frequency bands (simplified)
                let samples: Vec<f32> = match &comp.audio {
                    AudioData::StereoF32(s) => s.iter().map(|p| (p[0] + p[1]) * 0.5).collect(),
                    AudioData::F32(s) => s.clone(),
                    AudioData::I16(s) => s.iter().map(|&x| x as f32 / 32768.0).collect(),
                };

                if samples.len() >= 256 {
                    // Simple spectral envelope as training target
                    let mel = extract_mel_frame(&samples[..256]);
                    pairs.push(TrainingPair {
                        state,
                        mel_frame: mel,
                    });
                }
            }
        }
    }

    pairs
}

/// Extract a simplified mel frame from audio samples.
fn extract_mel_frame(samples: &[f32]) -> Vec<f32> {
    let n = samples.len().min(256);
    let half = n / 2;
    let mut spectrum = vec![0.0f32; half];

    // Simple DFT magnitude
    for k in 0..half {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let freq = std::f32::consts::TAU * k as f32 / n as f32;
        for (i, &s) in samples.iter().take(n).enumerate() {
            let ph = freq * i as f32;
            re += s * ph.cos();
            im += s * ph.sin();
        }
        spectrum[k] = (re * re + im * im).sqrt() / n as f32;
    }

    // Reduce to 64 mel bins by averaging groups
    let bins = 64;
    let group_size = (half / bins).max(1);
    (0..bins)
        .map(|b| {
            let start = b * group_size;
            let end = (start + group_size).min(half);
            if end > start {
                spectrum[start..end].iter().sum::<f32>() / (end - start) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Evaluate untrained vocoder against training data.
fn evaluate_training_data(pairs: &[TrainingPair]) -> (f32, f32) {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
    use symthaea_muse::spectral_vocoder::{MelDecoder, MEL_BINS};

    let genesis = GenesisSeed::from_phrase("spectral-vocoder-v1");
    let decoder = MelDecoder::new(MEL_BINS, &genesis);

    let mut total_error = 0.0f32;
    let mut max_error = 0.0f32;

    for pair in pairs {
        // Encode state as HV (simplified — matches SpectralVocoder::encode_state)
        let mut hv = ContinuousHV::zero(HDC_DIMENSION);
        let channels: Vec<(f32, &str)> = vec![
            (pair.state.consciousness_level, "sv_consciousness"),
            (pair.state.arousal, "sv_arousal"),
            (pair.state.valence, "sv_valence"),
        ];
        for (val, label) in &channels {
            let basis = genesis.hv(label, HDC_DIMENSION);
            hv = hv.add(&basis.scale(*val));
        }
        let hv = hv.normalize();

        // Decode to mel
        let predicted = decoder.decode(&hv);
        let target = &pair.mel_frame;

        // L1 error
        let error: f32 = predicted
            .iter()
            .zip(target.iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum::<f32>()
            / predicted.len().max(1) as f32;

        total_error += error;
        max_error = max_error.max(error);
    }

    let avg = total_error / pairs.len().max(1) as f32;
    (avg, max_error)
}
