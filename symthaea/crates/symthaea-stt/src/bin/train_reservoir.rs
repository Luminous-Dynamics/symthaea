// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train Liquid Projection Encoder with Ground Truth Alignments
//!
//! Uses Reservoir Computing + Ridge Regression to learn a readout
//! that maps LTC reservoir states to phoneme hypervectors.
//!
//! This is the "Sensory Cortex Upgrade" - non-linear transformation before HDC.
//!
//! Usage:
//! ```bash
//! cargo run --release --bin train-reservoir -- \
//!     --alignments data/alignments/dev_clean.parquet \
//!     --audio-dir data/librispeech/LibriSpeech/dev-clean \
//!     --output data/models/liquid_projection.bin
//! ```

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::path::PathBuf;

use symthaea_stt::{
    AudioConfig, AudioFrontend, HDC_DIM, LiquidProjection, LiquidProjectionConfig, LtcCell,
    LtcConfig, RidgeAccumulator, id_to_audio_path, load_alignments,
};

#[derive(Parser)]
#[command(
    name = "train-reservoir",
    about = "Train Liquid Projection encoder with ground truth alignments",
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

    /// Output encoder file
    #[arg(short, long, default_value = "data/models/liquid_projection.bin")]
    output: PathBuf,

    /// Maximum utterances to process (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Reservoir hidden size
    #[arg(long, default_value = "256")]
    reservoir_size: usize,

    /// Ridge regression lambda
    #[arg(long, default_value = "0.01")]
    ridge_lambda: f32,

    /// Frame hop duration (seconds)
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Number of context frames to stack (1 = no stacking)
    #[arg(long, default_value = "1")]
    context_frames: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Stack multiple frames together for temporal context
fn stack_frames(features: &[Vec<f32>], center: usize, context_frames: usize) -> Vec<f32> {
    let n_mels = if features.is_empty() {
        40
    } else {
        features[0].len()
    };
    let half = context_frames / 2;
    let mut stacked = Vec::with_capacity(n_mels * context_frames);

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
        style("    LIQUID PROJECTION TRAINING                             ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("    Reservoir Computing + Ridge Regression                 ").cyan()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!();
    println!("  Architecture: Mel -> LTC Reservoir -> Learned Readout -> HDC");
    println!("  Training: Ridge Regression (no backprop needed)");
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
    println!("📖 Loading alignments from {:?}...", cli.alignments);
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
    let phonemes: Vec<String> = phoneme_set.into_iter().collect();
    println!("  ✓ Found {} unique phonemes", phonemes.len());

    // Create encoder configuration
    let config = LiquidProjectionConfig {
        reservoir_size: cli.reservoir_size,
        ridge_lambda: cli.ridge_lambda,
        frame_duration: cli.frame_hop,
    };

    // Create the encoder (with target HVs)
    let mut encoder = LiquidProjection::new(&phonemes, config.clone());
    println!("\n🧠 Created Liquid Projection encoder:");
    println!("    Reservoir size: {}", cli.reservoir_size);
    println!("    Output dimension: {} bits", HDC_DIM);
    println!("    Ridge lambda: {}", cli.ridge_lambda);

    // Create audio frontend
    let audio_config = AudioConfig::default();
    let mut frontend = symthaea_stt::AudioFrontend::new(audio_config.clone());

    // Create LTC reservoir with stacked input size
    let mut ltc_config = LtcConfig::default();
    ltc_config.hidden_size = cli.reservoir_size;
    let n_mels = 40; // Standard mel filterbank size
    let input_size = n_mels * cli.context_frames;
    let mut reservoir = LtcCell::new(input_size, ltc_config);

    if cli.context_frames > 1 {
        println!(
            "    Context frames: {} ({} input dims)",
            cli.context_frames, input_size
        );
    }

    // Create Ridge Regression accumulator
    let mut accumulator = RidgeAccumulator::new(cli.reservoir_size, HDC_DIM);

    // Determine how many utterances to process
    let total = if cli.max_utterances > 0 {
        alignments.len().min(cli.max_utterances)
    } else {
        alignments.len()
    };

    println!("\n📂 Processing {} utterances...", total);

    // Progress bar
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

        // Extract mel features
        let features = frontend.extract_features(&samples);
        if features.is_empty() {
            skipped += 1;
            continue;
        }

        // Run through reservoir and collect states
        reservoir.reset();
        let mut reservoir_states: Vec<Vec<f32>> = Vec::with_capacity(features.len());

        for i in 0..features.len() {
            // Stack frames for temporal context
            let input = if cli.context_frames > 1 {
                stack_frames(&features, i, cli.context_frames)
            } else {
                features[i].clone()
            };
            reservoir.forward(&input, cli.frame_hop);
            reservoir_states.push(reservoir.state().to_vec());
        }

        // For each phoneme segment, add center frames to accumulator
        for segment in &alignment.phonemes {
            // Get target HV for this phoneme
            let target_hv = match encoder.get_target(&segment.phoneme) {
                Some(hv) => hv,
                None => continue,
            };

            // Convert time to frame index
            let start_frame = (segment.start / cli.frame_hop as f64) as usize;
            let end_frame = (segment.end / cli.frame_hop as f64) as usize;

            // Use center portion of segment
            let segment_len = end_frame.saturating_sub(start_frame);
            let margin = segment_len / 4;
            let center_start = (start_frame + margin).min(reservoir_states.len().saturating_sub(1));
            let center_end = end_frame
                .saturating_sub(margin)
                .min(reservoir_states.len())
                .max(center_start + 1);

            // Add center frames to accumulator
            for frame_idx in center_start..center_end {
                if frame_idx < reservoir_states.len() {
                    accumulator.add_with_hv(&reservoir_states[frame_idx], target_hv);
                    total_samples += 1;
                }
            }
        }

        processed += 1;

        if i % 100 == 0 {
            pb.set_message(format!("{} samples", total_samples));
        }
    }

    pb.finish_with_message("done");

    println!("\n  📊 Statistics:");
    println!("    Processed: {} utterances", processed);
    println!("    Skipped: {} utterances", skipped);
    println!("    Total training samples: {}", total_samples);

    // Solve Ridge Regression
    println!("\n🔧 Solving Ridge Regression...");
    println!("    Computing W = (X^T X + λI)^(-1) X^T Y");
    println!(
        "    Matrix size: {}x{} -> {}x{}",
        cli.reservoir_size, cli.reservoir_size, HDC_DIM, cli.reservoir_size
    );

    let weights = accumulator.solve(cli.ridge_lambda);
    encoder.set_readout_weights(weights);

    println!("  ✓ Readout matrix computed");

    // Diagnostic: Test on a few samples
    if cli.verbose {
        println!("\n🔬 Diagnostic: Testing encoder quality...");

        let mut correct = 0;
        let mut total_test = 0;
        let mut total_sim = 0.0f32;

        // Re-process a few utterances
        for (id, alignment) in alignments.iter().take(10) {
            let audio_path = match id_to_audio_path(id, &cli.audio_dir) {
                Some(p) if p.exists() => p,
                _ => continue,
            };

            let (samples, _) = match AudioFrontend::load_audio(&audio_path) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let features = frontend.extract_features(&samples);
            reservoir.reset();

            let mut states: Vec<Vec<f32>> = Vec::new();
            for i in 0..features.len() {
                let input = if cli.context_frames > 1 {
                    stack_frames(&features, i, cli.context_frames)
                } else {
                    features[i].clone()
                };
                reservoir.forward(&input, cli.frame_hop);
                states.push(reservoir.state().to_vec());
            }

            for segment in alignment.phonemes.iter().take(10) {
                let start_frame = (segment.start / cli.frame_hop as f64) as usize;
                let end_frame = (segment.end / cli.frame_hop as f64) as usize;
                let center = (start_frame + end_frame) / 2;

                if center < states.len() {
                    if let Some((decoded, sim)) = encoder.decode(&states[center]) {
                        total_sim += sim;
                        total_test += 1;
                        if decoded == segment.phoneme {
                            correct += 1;
                        }
                    }
                }
            }
        }

        if total_test > 0 {
            let accuracy = correct as f32 / total_test as f32 * 100.0;
            let avg_sim = total_sim / total_test as f32;
            println!(
                "    Frame accuracy: {:.1}% ({}/{})",
                accuracy, correct, total_test
            );
            println!("    Average similarity: {:.3}", avg_sim);
        }
    }

    // Save encoder
    println!("\n💾 Saving encoder to {:?}...", cli.output);

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    if let Err(e) = encoder.save(&cli.output) {
        eprintln!("{} Failed to save: {}", style("ERROR:").red().bold(), e);
        std::process::exit(1);
    }

    println!("  ✓ Saved Liquid Projection encoder");

    println!(
        "\n{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("             LIQUID PROJECTION TRAINING COMPLETE           ")
            .bold()
            .green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!("\n  Next: Evaluate with the trained encoder");
}
