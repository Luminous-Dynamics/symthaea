// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train Direct Linear Classifier with Ground Truth Alignments
//!
//! Architecture: Mel+Deltas -> Context Stacking -> LTC Reservoir -> Ridge Regression -> Phoneme Logits
//!
//! No HDC binarization - direct continuous-space classification.
//! Still uses Ridge Regression (closed-form, no backprop).

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::path::PathBuf;

use symthaea_stt::{
    AudioConfig, AudioFrontend, CrystalReservoir, DirectAccumulator, DirectClassifier,
    DirectClassifierConfig, LtcCell, LtcConfig, RFActivation, RandomProjection, id_to_audio_path,
    load_alignments,
};

#[derive(Parser)]
#[command(
    name = "train-direct",
    about = "Train direct linear classifier with ground truth alignments",
    version,
    author
)]
struct Cli {
    /// Parquet file with alignments
    #[arg(short, long)]
    alignments: PathBuf,

    /// LibriSpeech audio directory
    #[arg(short = 'd', long)]
    audio_dir: PathBuf,

    /// Output classifier file
    #[arg(short, long, default_value = "data/models/direct_classifier.bin")]
    output: PathBuf,

    /// Maximum utterances to process (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Reservoir hidden size
    #[arg(long, default_value = "1024")]
    reservoir_size: usize,

    /// Ridge regression lambda
    #[arg(long, default_value = "0.01")]
    ridge_lambda: f32,

    /// Frame hop duration (seconds)
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Number of context frames to stack
    #[arg(long, default_value = "7")]
    context_frames: usize,

    /// Use delta features
    #[arg(long, default_value = "true")]
    use_deltas: bool,

    /// Spectral radius for reservoir
    #[arg(long, default_value = "0.9")]
    spectral_radius: f32,

    /// Prior correction scale (0.0 = no correction, 1.0 = full Bayesian debiasing)
    #[arg(long, default_value = "0.0")]
    prior_scale: f32,

    /// Skip reservoir and classify on raw stacked features
    #[arg(long)]
    skip_reservoir: bool,

    /// Random nonlinear feature expansion dimension (0 = disabled)
    #[arg(long, default_value = "0")]
    random_features: usize,

    /// Activation function for random features (tanh or relu)
    #[arg(long, default_value = "tanh")]
    activation: String,

    /// Per-utterance cepstral mean normalization
    #[arg(long)]
    cmn: bool,

    /// Per-utterance cepstral mean + variance normalization (overrides --cmn)
    #[arg(long)]
    cmvn: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Use Gabor STRF features instead of random projection
    #[arg(long)]
    use_gabor: bool,
}

/// Stack multiple frames together for temporal context
fn stack_frames(features: &[Vec<f32>], center: usize, context_frames: usize) -> Vec<f32> {
    let n_dims = if features.is_empty() {
        0
    } else {
        features[0].len()
    };
    let half = context_frames / 2;
    let mut stacked = Vec::with_capacity(n_dims * context_frames);

    for offset in 0..context_frames {
        let idx = (center + offset).saturating_sub(half);
        let idx = idx.min(features.len().saturating_sub(1));
        stacked.extend_from_slice(&features[idx]);
    }

    stacked
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("    DIRECT CLASSIFIER TRAINING                             ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("    Reservoir Computing + Ridge Regression (No HDC)        ").cyan()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!();
    println!("  Architecture: Mel+Δ+ΔΔ -> Context Stack -> LTC Reservoir -> Linear Classifier");
    println!("  Training: Ridge Regression (closed-form, no backprop)");
    println!("  Output: Direct phoneme logits (no binarization)");
    println!();

    // Check paths
    if !cli.alignments.exists() {
        eprintln!(
            "{} Alignments file not found: {:?}",
            style("ERROR:").red().bold(),
            cli.alignments
        );
        std::process::exit(1);
    }
    if !cli.audio_dir.exists() {
        eprintln!(
            "{} Audio directory not found: {:?}",
            style("ERROR:").red().bold(),
            cli.audio_dir
        );
        std::process::exit(1);
    }

    // Load alignments
    println!("Loading alignments from {:?}...", cli.alignments);
    let alignments = match load_alignments(&cli.alignments) {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "{} Failed to load alignments: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };
    println!("  ✓ Loaded {} utterance alignments", alignments.len());

    // Collect all unique phonemes (sorted for deterministic ordering)
    let mut phoneme_set: HashSet<String> = HashSet::new();
    for alignment in alignments.values() {
        for segment in &alignment.phonemes {
            phoneme_set.insert(segment.phoneme.clone());
        }
    }
    let mut phonemes: Vec<String> = phoneme_set.into_iter().collect();
    phonemes.sort();
    println!("  ✓ Found {} unique phonemes", phonemes.len());

    // Feature dimensions
    let n_mels = 40;
    let feature_dim = if cli.use_deltas { n_mels * 3 } else { n_mels };
    let input_size = feature_dim * cli.context_frames;

    // Parse activation function
    let rf_activation = match cli.activation.as_str() {
        "relu" | "ReLU" => RFActivation::ReLU,
        _ => RFActivation::Tanh,
    };

    // Create optional Gabor STRF reservoir
    let gabor_reservoir: Option<CrystalReservoir> = if cli.use_gabor {
        Some(CrystalReservoir::new(
            cli.random_features.max(4096), // Use random_features as n_filters, default 4096
            cli.context_frames,
            feature_dim,
            n_mels,
        ))
    } else {
        None
    };

    // Determine classifier input dim and create optional random projection
    let (classifier_dim, random_proj) = if cli.use_gabor {
        let n_filters = cli.random_features.max(4096);
        (n_filters, None)
    } else if cli.random_features > 0 {
        let base_dim = if cli.skip_reservoir {
            input_size
        } else {
            cli.reservoir_size
        };
        let proj = RandomProjection::with_activation(
            base_dim,
            cli.random_features,
            42,
            rf_activation.clone(),
        );
        (cli.random_features, Some(proj))
    } else if cli.skip_reservoir {
        (input_size, None)
    } else {
        (cli.reservoir_size, None)
    };

    // Create classifier
    let config = DirectClassifierConfig {
        reservoir_size: classifier_dim,
        ridge_lambda: cli.ridge_lambda,
        frame_duration: cli.frame_hop,
        context_frames: cli.context_frames,
    };

    let mut classifier = DirectClassifier::new(&phonemes, config.clone());

    println!("\n  Configuration:");
    if cli.use_gabor {
        let n_filters = cli.random_features.max(4096);
        println!("    Mode: GABOR STRF (Crystal Reservoir)");
        println!(
            "    Input: {} dims → Gabor filters → {} dims",
            input_size, n_filters
        );
    } else if cli.random_features > 0 {
        let base = if cli.skip_reservoir {
            input_size
        } else {
            cli.reservoir_size
        };
        let act_name = match rf_activation {
            RFActivation::ReLU => "ReLU",
            _ => "tanh",
        };
        println!("    Mode: RANDOM FEATURES (ELM)");
        println!(
            "    Base input: {} dims → random {} → {} dims",
            base, act_name, cli.random_features
        );
    } else if cli.skip_reservoir {
        println!("    Mode: DIRECT (no reservoir)");
        println!(
            "    Classifier input: {} dims (raw stacked features)",
            input_size
        );
    } else {
        println!("    Mode: RESERVOIR");
        println!("    Reservoir size: {}", cli.reservoir_size);
        println!("    Spectral radius: {}", cli.spectral_radius);
    }
    println!(
        "    Feature dim: {} (mel={}, deltas={})",
        feature_dim, n_mels, cli.use_deltas
    );
    println!(
        "    Context frames: {} -> {} input dims",
        cli.context_frames, input_size
    );
    println!("    Ridge lambda: {}", cli.ridge_lambda);
    println!("    CMN: {}", cli.cmn);
    println!("    Phoneme classes: {}", phonemes.len());

    // Create audio frontend
    let audio_config = AudioConfig::default();
    let mut frontend = AudioFrontend::new(audio_config);

    // Create LTC reservoir (only if not skipping)
    let mut reservoir = if !cli.skip_reservoir {
        let mut ltc_config = LtcConfig::default();
        ltc_config.hidden_size = cli.reservoir_size;
        Some(LtcCell::new_reservoir(
            input_size,
            ltc_config,
            cli.spectral_radius,
            1.0,
        ))
    } else {
        None
    };

    // Create accumulator with appropriate input dim
    let mut accumulator = DirectAccumulator::new(classifier_dim, phonemes.len());

    // Batch buffers for cache-tiled dgemm accumulation
    const BATCH_SIZE: usize = 256;
    let mut batch_f64: Vec<f64> = Vec::with_capacity(BATCH_SIZE * classifier_dim);
    let mut batch_targets: Vec<usize> = Vec::with_capacity(BATCH_SIZE);

    // Determine how many utterances to process
    let total = if cli.max_utterances > 0 {
        alignments.len().min(cli.max_utterances)
    } else {
        alignments.len()
    };

    println!("\n  Processing {} utterances...", total);

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut processed = 0;
    let mut skipped = 0;
    let mut total_samples = 0usize;

    for (i, (id, alignment)) in alignments.iter().take(total).enumerate() {
        pb.set_position(i as u64);

        // Find audio file
        let audio_path = id_to_audio_path(id, &cli.audio_dir);
        let audio_path = match audio_path {
            Some(p) if p.exists() => p,
            _ => {
                skipped += 1;
                continue;
            }
        };

        // Load audio
        let (samples, _sample_rate) = match AudioFrontend::load_audio(&audio_path) {
            Ok(a) => a,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        // Extract features (with or without deltas)
        let mut features = if cli.use_deltas {
            frontend.extract_features_with_deltas(&samples)
        } else {
            frontend.extract_features(&samples)
        };

        if features.is_empty() {
            skipped += 1;
            continue;
        }

        // Per-utterance cepstral mean (and optionally variance) normalization
        if (cli.cmn || cli.cmvn) && !features.is_empty() {
            let n_dims = features[0].len();
            let n_frames = features.len() as f32;
            let mut mean = vec![0.0f32; n_dims];
            for frame in &features {
                for (i, &v) in frame.iter().enumerate() {
                    mean[i] += v;
                }
            }
            for m in &mut mean {
                *m /= n_frames;
            }
            for frame in &mut features {
                for (i, v) in frame.iter_mut().enumerate() {
                    *v -= mean[i];
                }
            }
            // Variance normalization
            if cli.cmvn {
                let mut var = vec![0.0f32; n_dims];
                for frame in &features {
                    for (i, &v) in frame.iter().enumerate() {
                        var[i] += v * v;
                    }
                }
                for v in &mut var {
                    *v = (*v / n_frames).sqrt().max(1e-6);
                }
                for frame in &mut features {
                    for (i, v) in frame.iter_mut().enumerate() {
                        *v /= var[i];
                    }
                }
            }
        }

        // Get frame representations (either reservoir states or raw stacked features)
        let frame_representations: Vec<Vec<f32>> = if let Some(ref mut res) = reservoir {
            // Run through LTC reservoir
            res.reset();
            let mut states = Vec::with_capacity(features.len());
            for frame_idx in 0..features.len() {
                let input = if cli.context_frames > 1 {
                    stack_frames(&features, frame_idx, cli.context_frames)
                } else {
                    features[frame_idx].clone()
                };
                res.forward(&input, cli.frame_hop);
                states.push(res.state().to_vec());
            }
            states
        } else {
            // Use raw stacked features directly
            (0..features.len())
                .map(|frame_idx| {
                    if cli.context_frames > 1 {
                        stack_frames(&features, frame_idx, cli.context_frames)
                    } else {
                        features[frame_idx].clone()
                    }
                })
                .collect()
        };

        // For each phoneme segment, add center frames to accumulator
        for segment in &alignment.phonemes {
            let target_idx = match classifier.phoneme_index(&segment.phoneme) {
                Some(idx) => idx,
                None => continue,
            };

            // Convert time to frame index
            let start_frame = (segment.start / cli.frame_hop as f64) as usize;
            let end_frame = (segment.end / cli.frame_hop as f64) as usize;

            // Use center portion of segment
            let segment_len = end_frame.saturating_sub(start_frame);
            let margin = segment_len / 4;
            let center_start =
                (start_frame + margin).min(frame_representations.len().saturating_sub(1));
            let center_end = end_frame
                .saturating_sub(margin)
                .min(frame_representations.len())
                .max(center_start + 1);

            for frame_idx in center_start..center_end {
                if frame_idx < frame_representations.len() {
                    let feature = if let Some(ref gabor) = gabor_reservoir {
                        // Gabor STRF projection
                        gabor.forward(&frame_representations[frame_idx])
                    } else if let Some(ref proj) = random_proj {
                        // Random projection
                        proj.forward(&frame_representations[frame_idx])
                    } else {
                        // No projection
                        frame_representations[frame_idx].clone()
                    };
                    for &v in &feature {
                        batch_f64.push(v as f64);
                    }
                    batch_targets.push(target_idx);
                    total_samples += 1;
                    if batch_targets.len() >= BATCH_SIZE {
                        accumulator.add_batch(&batch_f64, &batch_targets, batch_targets.len());
                        batch_f64.clear();
                        batch_targets.clear();
                    }
                }
            }
        }

        processed += 1;

        if i % 100 == 0 {
            pb.set_message(format!("{} samples", total_samples));
        }
    }

    // Flush remaining batch
    if !batch_targets.is_empty() {
        accumulator.add_batch(&batch_f64, &batch_targets, batch_targets.len());
    }

    pb.finish_with_message("done");

    println!("\n  Statistics:");
    println!("    Processed: {} utterances", processed);
    println!("    Skipped: {} utterances", skipped);
    println!("    Total training samples: {}", total_samples);

    // Solve Ridge Regression
    println!("\n  Solving Ridge Regression...");
    println!("    Computing W = (X^T X + λI)^{{-1}} X^T Y");
    println!(
        "    Matrix: {}x{} -> {}x{}",
        classifier_dim,
        classifier_dim,
        phonemes.len(),
        classifier_dim
    );

    let weights = accumulator.solve(cli.ridge_lambda);
    classifier.set_weights(weights);

    // Store random projection if used
    classifier.random_projection = random_proj;

    // Set log priors for Bayesian debiasing at inference time
    classifier.set_log_priors(accumulator.class_counts(), cli.prior_scale);
    println!(
        "  ✓ Weight matrix computed (prior_scale={})",
        cli.prior_scale
    );

    // Show class distribution
    if cli.verbose {
        let counts = accumulator.class_counts();
        let mut indexed: Vec<(usize, usize)> = counts.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\n  Top 10 phoneme frequencies:");
        for &(idx, count) in indexed.iter().take(10) {
            let pct = count as f32 / total_samples as f32 * 100.0;
            println!("    {}: {} ({:.1}%)", phonemes[idx], count, pct);
        }
    }

    // Diagnostic: Test on a few samples
    if cli.verbose {
        println!("\n  Diagnostic: Testing classifier quality...");

        let mut correct = 0;
        let mut total_test = 0;

        for (id, alignment) in alignments.iter().take(10) {
            let audio_path = match id_to_audio_path(id, &cli.audio_dir) {
                Some(p) if p.exists() => p,
                _ => continue,
            };

            let (samples, _) = match AudioFrontend::load_audio(&audio_path) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let mut features = if cli.use_deltas {
                frontend.extract_features_with_deltas(&samples)
            } else {
                frontend.extract_features(&samples)
            };

            // Apply CMN/CMVN if enabled
            if (cli.cmn || cli.cmvn) && !features.is_empty() {
                let n_dims = features[0].len();
                let n_frames = features.len() as f32;
                let mut mean = vec![0.0f32; n_dims];
                for frame in &features {
                    for (i, &v) in frame.iter().enumerate() {
                        mean[i] += v;
                    }
                }
                for m in &mut mean {
                    *m /= n_frames;
                }
                for frame in &mut features {
                    for (i, v) in frame.iter_mut().enumerate() {
                        *v -= mean[i];
                    }
                }
                if cli.cmvn {
                    let mut var = vec![0.0f32; n_dims];
                    for frame in &features {
                        for (i, &v) in frame.iter().enumerate() {
                            var[i] += v * v;
                        }
                    }
                    for v in &mut var {
                        *v = (*v / n_frames).sqrt().max(1e-6);
                    }
                    for frame in &mut features {
                        for (i, v) in frame.iter_mut().enumerate() {
                            *v /= var[i];
                        }
                    }
                }
            }

            let states: Vec<Vec<f32>> = if let Some(ref mut res) = reservoir {
                res.reset();
                let mut s = Vec::new();
                for frame_idx in 0..features.len() {
                    let input = if cli.context_frames > 1 {
                        stack_frames(&features, frame_idx, cli.context_frames)
                    } else {
                        features[frame_idx].clone()
                    };
                    res.forward(&input, cli.frame_hop);
                    s.push(res.state().to_vec());
                }
                s
            } else {
                (0..features.len())
                    .map(|frame_idx| {
                        if cli.context_frames > 1 {
                            stack_frames(&features, frame_idx, cli.context_frames)
                        } else {
                            features[frame_idx].clone()
                        }
                    })
                    .collect()
            };

            for segment in alignment.phonemes.iter().take(10) {
                let start_frame = (segment.start / cli.frame_hop as f64) as usize;
                let end_frame = (segment.end / cli.frame_hop as f64) as usize;
                let center = (start_frame + end_frame) / 2;

                if center < states.len() {
                    // Apply projection if needed (Gabor requires explicit projection)
                    let projected = if let Some(ref gabor) = gabor_reservoir {
                        gabor.forward(&states[center])
                    } else {
                        states[center].clone()
                    };
                    let (decoded, _conf) = classifier.decode(&projected);
                    total_test += 1;
                    if decoded == segment.phoneme {
                        correct += 1;
                    }
                }
            }
        }

        if total_test > 0 {
            let accuracy = correct as f32 / total_test as f32 * 100.0;
            println!(
                "    Frame accuracy: {:.1}% ({}/{})",
                accuracy, correct, total_test
            );
        }
    }

    // Save classifier
    println!("\n  Saving classifier to {:?}...", cli.output);

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    if let Err(e) = classifier.save(&cli.output) {
        eprintln!("{} Failed to save: {}", style("ERROR:").red().bold(), e);
        std::process::exit(1);
    }

    println!("  ✓ Saved Direct Classifier");

    println!(
        "\n{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("         DIRECT CLASSIFIER TRAINING COMPLETE               ")
            .bold()
            .green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!("\n  Next: eval-direct --classifier {:?}", cli.output);
}
