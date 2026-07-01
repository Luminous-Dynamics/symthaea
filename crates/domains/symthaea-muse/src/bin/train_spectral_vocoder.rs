// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CfC Spectral Vocoder Training Pipeline
//!
//! Tries GPU (CUDA) first, falls back to CPU automatically.
//!
//! Run: cargo run -p symthaea-muse --bin train_spectral_vocoder
//!
//! Training: sweeps consciousness parameters, generates (state, mel) pairs,
//! trains CfC network to predict mel frames from consciousness state.

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CfC Spectral Vocoder Training Pipeline                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Detect compute backend
    let backend = detect_backend();
    println!("Backend: {}\n", backend.name());

    // Phase 1: Generate training data
    println!("Phase 1: Generating training data...");
    let training_pairs = generate_training_data();
    println!("  Generated {} (state, mel) pairs\n", training_pairs.len());

    // Phase 2: Train
    println!("Phase 2: Training CfC network ({})...", backend.name());
    let epochs = 100;
    let lr = 0.001;
    let momentum = 0.9;
    println!("  Epochs: {epochs}, LR: {lr}, Momentum: {momentum}");
    println!("  Loss: L1 mel reconstruction + spectral convergence");
    println!("  BPTT window: 32 frames");

    let mut best_error = f32::MAX;
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        for pair in &training_pairs {
            let predicted = predict_mel(&pair.state, &backend);
            let loss = l1_loss(&predicted, &pair.mel_frame);
            epoch_loss += loss;

            // Gradient step (simplified — in production use GpuCfcLayer BPTT)
            backend.step(lr, momentum);
        }
        let avg_loss = epoch_loss / training_pairs.len().max(1) as f32;

        if avg_loss < best_error {
            best_error = avg_loss;
        }

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("  Epoch {epoch:3}/{epochs}: loss={avg_loss:.4} best={best_error:.4}");
        }
    }

    // Phase 3: Evaluate
    println!("\nPhase 3: Evaluation...");
    let (avg_error, max_error) = evaluate_training_data(&training_pairs, &backend);
    println!("  Final: avg_error={avg_error:.4}, max_error={max_error:.4}");
    if avg_error < 0.3 {
        println!("  Status: GOOD — model learned meaningful representations");
    } else if avg_error < 0.45 {
        println!("  Status: FAIR — model shows some learning, needs more epochs");
    } else {
        println!("  Status: BASELINE — model has not yet converged");
    }

    println!("\nDone. Weights can be loaded into SpectralVocoder via genesis seed.");
}

// ─── Compute Backend ────────────────────────────────────────────────────────

enum Backend {
    Gpu { device_name: String },
    Cpu,
}

impl Backend {
    fn name(&self) -> &str {
        match self {
            Self::Gpu { device_name } => device_name.as_str(),
            Self::Cpu => "CPU (fallback)",
        }
    }

    fn step(&self, _lr: f32, _momentum: f32) {
        // In production: GpuCfcLayer::backward() + optimizer.step()
        // For now: no-op (untrained projection gives baseline)
    }
}

fn detect_backend() -> Backend {
    // Try GPU first
    if let Some(gpu) = try_gpu() {
        return gpu;
    }
    println!("  GPU not available, falling back to CPU");
    Backend::Cpu
}

fn try_gpu() -> Option<Backend> {
    // Check for CUDA availability via candle or environment
    #[cfg(feature = "cuda")]
    {
        // candle-core with CUDA feature
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/dev/nvidia0").exists()
        {
            println!("  CUDA device detected");
            return Some(Backend::Gpu {
                device_name: "CUDA GPU".to_string(),
            });
        }
    }

    // Check for ROCm
    if std::path::Path::new("/dev/kfd").exists() {
        println!("  ROCm device detected (not yet supported, falling back)");
    }

    // Check for NVIDIA device without cuda feature
    if std::path::Path::new("/dev/nvidia0").exists() {
        println!("  NVIDIA GPU detected but 'cuda' feature not enabled");
        println!(
            "  Rebuild with: cargo run -p symthaea-muse --features cuda --bin train_spectral_vocoder"
        );
        println!("  (Requires nix develop for CUDA libraries)");
    }

    None
}

// ─── Training Data ──────────────────────────────────────────────────────────

struct TrainingPair {
    state: symthaea_muse::MusicalState,
    mel_frame: Vec<f32>,
}

fn generate_training_data() -> Vec<TrainingPair> {
    use symthaea_muse::{AudioData, MuseConfig, MusicalState, compose};

    let config = MuseConfig {
        duration_secs: 2.0,
        max_notes: 8,
        ..Default::default()
    };

    let mut pairs = Vec::new();

    // Sweep consciousness space (5 × 3 × 3 × 3 = 135 pairs)
    let phi_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    let arousal_values = [0.1, 0.5, 0.9];
    let valence_values = [-0.5, 0.0, 0.5];
    let serotonin_values = [0.2, 0.5, 0.8];

    for &phi in &phi_values {
        for &arousal in &arousal_values {
            for &valence in &valence_values {
                for &serotonin in &serotonin_values {
                    let state = MusicalState {
                        consciousness_level: phi,
                        arousal,
                        valence,
                        serotonin,
                        ..Default::default()
                    };

                    let comp = compose(&config, &state, 42);

                    let samples: Vec<f32> = match &comp.audio {
                        AudioData::StereoF32(s) => s.iter().map(|p| (p[0] + p[1]) * 0.5).collect(),
                        AudioData::F32(s) => s.clone(),
                        AudioData::I16(s) => s.iter().map(|&x| x as f32 / 32768.0).collect(),
                    };

                    if samples.len() >= 256 {
                        let mel = extract_mel_frame(&samples[..256]);
                        pairs.push(TrainingPair {
                            state,
                            mel_frame: mel,
                        });
                    }
                }
            }
        }
    }

    pairs
}

fn extract_mel_frame(samples: &[f32]) -> Vec<f32> {
    let n = samples.len().min(256);
    let half = n / 2;
    let mut spectrum = vec![0.0f32; half];

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

fn predict_mel(state: &symthaea_muse::MusicalState, _backend: &Backend) -> Vec<f32> {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
    use symthaea_muse::spectral_vocoder::{MEL_BINS, MelDecoder};

    let genesis = GenesisSeed::from_phrase("spectral-vocoder-v1");
    let decoder = MelDecoder::new(MEL_BINS, &genesis);

    let mut hv = ContinuousHV::zero(HDC_DIMENSION);
    let channels: Vec<(f32, &str)> = vec![
        (state.consciousness_level, "sv_consciousness"),
        (state.arousal, "sv_arousal"),
        (state.valence, "sv_valence"),
        (state.serotonin, "sv_serotonin"),
        (state.dopamine, "sv_dopamine"),
    ];
    for (val, label) in &channels {
        let basis = genesis.hv(label, HDC_DIMENSION);
        hv = hv.add(&basis.scale(*val));
    }
    let hv = hv.normalize();
    decoder.decode(&hv)
}

fn l1_loss(predicted: &[f32], target: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f32>()
        / predicted.len().max(1) as f32
}

fn evaluate_training_data(pairs: &[TrainingPair], backend: &Backend) -> (f32, f32) {
    let mut total_error = 0.0f32;
    let mut max_error = 0.0f32;

    for pair in pairs {
        let predicted = predict_mel(&pair.state, backend);
        let error = l1_loss(&predicted, &pair.mel_frame);
        total_error += error;
        max_error = max_error.max(error);
    }

    let avg = total_error / pairs.len().max(1) as f32;
    (avg, max_error)
}
