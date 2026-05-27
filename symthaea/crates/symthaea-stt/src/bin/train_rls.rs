// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train RLS (Recursive Least Squares) Classifier
//!
//! Architecture: Mel+Deltas -> Context Stack -> Feature Projection -> Online RLS
//!
//! Uses Sherman-Morrison-Woodbury formula for O(D²) per-sample updates.
//! Mathematically equivalent to batch Ridge Regression, computed incrementally.

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::path::PathBuf;

use symthaea_stt::{
    AudioConfig, AudioFrontend, CrystalReservoir, DirectClassifier, DirectClassifierConfig,
    FastRlsClassifier, RFActivation, RandomProjection, id_to_audio_path, load_alignments,
};

#[derive(Parser)]
#[command(
    name = "train-rls",
    about = "Train RLS classifier with online discriminative learning",
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
    #[arg(short, long, default_value = "data/models/rls_classifier.bin")]
    output: PathBuf,

    /// Maximum utterances to process (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Number of projected features
    #[arg(long, default_value = "4096")]
    n_features: usize,

    /// Frame hop duration (seconds)
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Number of context frames to stack
    #[arg(long, default_value = "7")]
    context_frames: usize,

    /// Use delta features
    #[arg(long, default_value = "true")]
    use_deltas: bool,

    /// Per-utterance cepstral mean normalization
    #[arg(long)]
    cmn: bool,

    /// Per-utterance CMVN
    #[arg(long)]
    cmvn: bool,

    /// Use Gabor STRF features instead of random projection
    #[arg(long)]
    use_gabor: bool,

    /// RLS regularization parameter (delta)
    #[arg(long, default_value = "0.01")]
    delta: f32,

    /// RLS forgetting factor (1.0 = no forgetting)
    #[arg(long, default_value = "1.0")]
    lambda: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
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
        style("═══════════════════════════════════════════════════════════").yellow()
    );
    println!(
        "{}",
        style("    RLS CLASSIFIER TRAINING                                ")
            .bold()
            .yellow()
    );
    println!(
        "{}",
        style("    Online Discriminative Learning (Sherman-Morrison)      ").yellow()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").yellow()
    );
    println!();
    println!("  Architecture: Mel+Δ+ΔΔ -> Context Stack -> Projection -> RLS");
    println!("  Training: Online Ridge Regression (O(D²) per sample)");
    println!("  Output: Discriminative weight matrix");
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

    // Collect all unique phonemes
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

    // Create feature projector (Random or Gabor)
    let random_proj: Option<RandomProjection>;
    let gabor_reservoir: Option<CrystalReservoir>;

    if cli.use_gabor {
        println!("\n  Creating Gabor STRF Reservoir...");
        gabor_reservoir = Some(CrystalReservoir::new(
            cli.n_features,
            cli.context_frames,
            feature_dim,
            n_mels,
        ));
        random_proj = None;
        println!("    Gabor filters: {}", cli.n_features);
    } else {
        println!("\n  Creating Random Projection...");
        random_proj = Some(RandomProjection::with_activation(
            input_size,
            cli.n_features,
            42,
            RFActivation::ReLU,
        ));
        gabor_reservoir = None;
        println!("    Random features: {} (ReLU)", cli.n_features);
    }

    // Create RLS classifier
    println!("\n  Creating RLS Classifier...");
    let mut classifier =
        FastRlsClassifier::new(phonemes.clone(), cli.n_features, cli.delta, cli.lambda);
    println!("    Feature dim: {}", cli.n_features);
    println!("    Delta (regularization): {}", cli.delta);
    println!("    Lambda (forgetting): {}", cli.lambda);

    println!("\n  Configuration:");
    if cli.use_gabor {
        println!("    Mode: GABOR + RLS");
    } else {
        println!("    Mode: RANDOM FEATURES + RLS");
    }
    println!(
        "    Input: {} dims -> {} projected dims",
        input_size, cli.n_features
    );
    println!("    CMVN: {}", cli.cmvn);
    println!("    Phoneme classes: {}", phonemes.len());

    // Create audio frontend
    let audio_config = AudioConfig::default();
    let mut frontend = AudioFrontend::new(audio_config);

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
            .template("{spinner:.yellow} [{bar:40.yellow/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("█▓-"),
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

        // Extract features
        let mut features = if cli.use_deltas {
            frontend.extract_features_with_deltas(&samples)
        } else {
            frontend.extract_features(&samples)
        };

        if features.is_empty() {
            skipped += 1;
            continue;
        }

        // Per-utterance CMVN
        if (cli.cmn || cli.cmvn) && !features.is_empty() {
            let n_dims = features[0].len();
            let n_frames = features.len() as f32;
            let mut mean = vec![0.0f32; n_dims];
            for frame in &features {
                for (j, &v) in frame.iter().enumerate() {
                    mean[j] += v;
                }
            }
            for m in &mut mean {
                *m /= n_frames;
            }
            for frame in &mut features {
                for (j, v) in frame.iter_mut().enumerate() {
                    *v -= mean[j];
                }
            }
            if cli.cmvn {
                let mut var = vec![0.0f32; n_dims];
                for frame in &features {
                    for (j, &v) in frame.iter().enumerate() {
                        var[j] += v * v;
                    }
                }
                for v in &mut var {
                    *v = (*v / n_frames).sqrt().max(1e-6);
                }
                for frame in &mut features {
                    for (j, v) in frame.iter_mut().enumerate() {
                        *v /= var[j];
                    }
                }
            }
        }

        // For each phoneme segment, project and update RLS
        for segment in &alignment.phonemes {
            let target_idx = phonemes.iter().position(|p| p == &segment.phoneme);
            let target_idx = match target_idx {
                Some(idx) => idx,
                None => continue,
            };

            // Convert time to frame index
            let start_frame = (segment.start / cli.frame_hop as f64) as usize;
            let end_frame = (segment.end / cli.frame_hop as f64) as usize;

            // Use center portion of segment
            let segment_len = end_frame.saturating_sub(start_frame);
            let margin = segment_len / 4;
            let center_start = (start_frame + margin).min(features.len().saturating_sub(1));
            let center_end = end_frame
                .saturating_sub(margin)
                .min(features.len())
                .max(center_start + 1);

            for frame_idx in center_start..center_end {
                if frame_idx < features.len() {
                    // Stack context frames
                    let stacked = stack_frames(&features, frame_idx, cli.context_frames);

                    // Project through feature extractor
                    let projected = if let Some(ref gabor) = gabor_reservoir {
                        gabor.forward(&stacked)
                    } else if let Some(ref proj) = random_proj {
                        proj.forward(&stacked)
                    } else {
                        stacked
                    };

                    // Update RLS classifier
                    classifier.update(&projected, target_idx);
                    total_samples += 1;
                }
            }
        }

        processed += 1;

        if i % 50 == 0 {
            pb.set_message(format!("{} samples", total_samples));
        }
    }

    pb.finish_with_message("done");

    println!("\n  Statistics:");
    println!("    Processed: {} utterances", processed);
    println!("    Skipped: {} utterances", skipped);
    println!("    Total training samples: {}", total_samples);

    // Show class distribution
    if cli.verbose {
        let mut indexed: Vec<(usize, usize)> =
            classifier.counts.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\n  Top 10 phoneme frequencies:");
        for &(idx, count) in indexed.iter().take(10) {
            let pct = count as f32 / total_samples as f32 * 100.0;
            println!("    {}: {} ({:.1}%)", phonemes[idx], count, pct);
        }
    }

    // Diagnostic: Test classifier quality
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

            // Apply CMVN if enabled
            if (cli.cmn || cli.cmvn) && !features.is_empty() {
                let n_dims = features[0].len();
                let n_frames = features.len() as f32;
                let mut mean = vec![0.0f32; n_dims];
                for frame in &features {
                    for (j, &v) in frame.iter().enumerate() {
                        mean[j] += v;
                    }
                }
                for m in &mut mean {
                    *m /= n_frames;
                }
                for frame in &mut features {
                    for (j, v) in frame.iter_mut().enumerate() {
                        *v -= mean[j];
                    }
                }
                if cli.cmvn {
                    let mut var = vec![0.0f32; n_dims];
                    for frame in &features {
                        for (j, &v) in frame.iter().enumerate() {
                            var[j] += v * v;
                        }
                    }
                    for v in &mut var {
                        *v = (*v / n_frames).sqrt().max(1e-6);
                    }
                    for frame in &mut features {
                        for (j, v) in frame.iter_mut().enumerate() {
                            *v /= var[j];
                        }
                    }
                }
            }

            for segment in alignment.phonemes.iter().take(10) {
                let start_frame = (segment.start / cli.frame_hop as f64) as usize;
                let end_frame = (segment.end / cli.frame_hop as f64) as usize;
                let center = (start_frame + end_frame) / 2;

                if center < features.len() {
                    let stacked = stack_frames(&features, center, cli.context_frames);
                    let projected = if let Some(ref gabor) = gabor_reservoir {
                        gabor.forward(&stacked)
                    } else if let Some(ref proj) = random_proj {
                        proj.forward(&stacked)
                    } else {
                        stacked
                    };

                    let (class_idx, _) = classifier.classify(&projected);
                    total_test += 1;
                    if phonemes.get(class_idx) == Some(&segment.phoneme) {
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

    // Save classifier in DirectClassifier format (compatible with eval-direct)
    println!("\n  Saving RLS classifier to {:?}...", cli.output);

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Convert RLS weights to DirectClassifier format
    // RLS stores weights as flat [n_classes * feature_dim]
    // DirectClassifier expects [n_classes][feature_dim]
    let n_classes = phonemes.len();
    let feature_dim = cli.n_features;
    let mut weights_2d: Vec<Vec<f32>> = Vec::with_capacity(n_classes);
    for class_idx in 0..n_classes {
        let start = class_idx * feature_dim;
        let end = start + feature_dim;
        weights_2d.push(classifier.weights[start..end].to_vec());
    }

    // Create DirectClassifier config
    let config = DirectClassifierConfig {
        reservoir_size: feature_dim,
        ridge_lambda: 0.01, // Not used for inference
        frame_duration: 0.01,
        context_frames: cli.context_frames,
    };

    // Create DirectClassifier with RLS weights
    let mut direct_classifier = DirectClassifier::new(&phonemes, config);
    direct_classifier.set_weights(weights_2d);
    direct_classifier.random_projection = random_proj;
    // Note: RLS has no bias term, DirectClassifier bias is already zeroed

    // Save as JSON (same format as train-direct)
    if let Err(e) = direct_classifier.save(&cli.output) {
        eprintln!("{} Failed to save: {}", style("ERROR:").red().bold(), e);
        std::process::exit(1);
    }

    println!("  ✓ Saved RLS Classifier (DirectClassifier format)");

    println!(
        "\n{}",
        style("═══════════════════════════════════════════════════════════").yellow()
    );
    println!(
        "{}",
        style("         RLS CLASSIFIER TRAINING COMPLETE                  ")
            .bold()
            .green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").yellow()
    );
    println!(
        "\n  Next: eval-direct --classifier {:?} --skip-reservoir --use-deltas --viterbi-penalty 0.95",
        cli.output
    );
}
