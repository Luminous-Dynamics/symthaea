// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train Crystal Reservoir Classifier
//!
//! Architecture: Mel+Deltas -> Context Stack -> Gabor STRF Bank -> Online Prototype Classifier
//!
//! Uses structured 2D Gabor filters modeled after the mammalian auditory cortex
//! instead of random projections. Learning is online (incremental) via prototype averaging.

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::path::PathBuf;

use symthaea_stt::{
    AudioConfig, AudioFrontend, CrystalReservoir, OnlinePrototypeClassifier, RFActivation,
    RandomProjection, id_to_audio_path, load_alignments,
};

#[derive(Parser)]
#[command(
    name = "train-crystal",
    about = "Train Crystal Reservoir classifier with Gabor STRFs",
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
    #[arg(short, long, default_value = "data/models/crystal_classifier.bin")]
    output: PathBuf,

    /// Maximum utterances to process (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Number of Gabor filters (output features)
    #[arg(long, default_value = "512")]
    n_filters: usize,

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

    /// Per-utterance cepstral mean + variance normalization
    #[arg(long)]
    cmvn: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Use random features instead of Gabor (control experiment)
    #[arg(long)]
    use_random: bool,
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
        style("═══════════════════════════════════════════════════════════").magenta()
    );
    println!(
        "{}",
        style("    CRYSTAL RESERVOIR TRAINING                             ")
            .bold()
            .magenta()
    );
    println!(
        "{}",
        style("    Gabor STRFs + Online Prototype Learning                ").magenta()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );
    println!();
    println!("  Architecture: Mel+Δ+ΔΔ -> Context Stack -> Gabor Bank -> Prototype Classifier");
    println!("  Training: Online prototype averaging (one-shot geometric learning)");
    println!("  Output: Phoneme prototypes in Gabor feature space");
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

    // Create feature extractor (Gabor or Random)
    let crystal: Option<CrystalReservoir>;
    let random_proj: Option<RandomProjection>;

    if cli.use_random {
        println!("\n  Creating Random Projection (CONTROL EXPERIMENT)...");
        random_proj = Some(RandomProjection::with_activation(
            input_size,
            cli.n_filters,
            42,
            RFActivation::ReLU,
        ));
        crystal = None;
        println!("    Random features: {}", cli.n_filters);
        println!("    Input dimensions: {}", input_size);
        println!("    Activation: ReLU");
    } else {
        println!("\n  Creating Crystal Reservoir...");
        crystal = Some(CrystalReservoir::new(
            cli.n_filters,
            cli.context_frames,
            feature_dim,
            n_mels,
        ));
        random_proj = None;
        println!("    Gabor filters: {}", cli.n_filters);
        println!("    Input dimensions: {}", input_size);
        println!(
            "    Filter coverage: {} time steps × {} mel bins",
            cli.context_frames, n_mels
        );
    }

    // Create online prototype classifier
    let mut classifier = OnlinePrototypeClassifier::new(phonemes.clone(), cli.n_filters);

    println!("\n  Configuration:");
    if cli.use_random {
        println!("    Mode: RANDOM FEATURES + PROTOTYPE (Control)");
    } else {
        println!("    Mode: CRYSTAL RESERVOIR (Gabor STRFs)");
    }
    println!(
        "    Feature dim: {} (mel={}, deltas={})",
        feature_dim, n_mels, cli.use_deltas
    );
    println!(
        "    Context frames: {} -> {} input dims",
        cli.context_frames, input_size
    );
    println!("    Output features: {}", cli.n_filters);
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
            .template("{spinner:.magenta} [{bar:40.magenta/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("◆◇-"),
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

        // Per-utterance cepstral mean (and optionally variance) normalization
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

        // For each phoneme segment, project through crystal and update prototypes
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

                    // Project through feature extractor (Gabor or Random)
                    let projected_features = if let Some(ref c) = crystal {
                        c.forward(&stacked)
                    } else if let Some(ref r) = random_proj {
                        r.forward(&stacked)
                    } else {
                        stacked // fallback (shouldn't happen)
                    };

                    // Update prototype with online learning
                    classifier.update(&projected_features, target_idx);
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

    println!("\n  Statistics:");
    println!("    Processed: {} utterances", processed);
    println!("    Skipped: {} utterances", skipped);
    println!("    Total training samples: {}", total_samples);

    // Show prototype statistics
    println!("\n  Prototype Statistics:");
    let mut initialized = 0;
    let mut min_count = usize::MAX;
    let mut max_count = 0;
    for &count in &classifier.counts {
        if count > 0 {
            initialized += 1;
            min_count = min_count.min(count);
            max_count = max_count.max(count);
        }
    }
    println!(
        "    Initialized prototypes: {}/{}",
        initialized,
        phonemes.len()
    );
    if initialized > 0 {
        println!("    Sample counts: {} - {}", min_count, max_count);
    }

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
                    let projected_features = if let Some(ref c) = crystal {
                        c.forward(&stacked)
                    } else if let Some(ref r) = random_proj {
                        r.forward(&stacked)
                    } else {
                        stacked
                    };
                    let (class_idx, _sim) = classifier.classify(&projected_features);

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

    // Save classifier
    println!("\n  Saving Crystal classifier to {:?}...", cli.output);

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Serialize the classifier
    let serialized = CrystalClassifierState {
        phonemes: phonemes.clone(),
        prototypes: classifier.prototypes.clone(),
        counts: classifier.counts.clone(),
        n_filters: cli.n_filters,
        context_frames: cli.context_frames,
        feature_dim,
        n_mels,
    };

    let data = match bincode::serialize(&serialized) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "{} Failed to serialize classifier: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };
    if let Err(e) = std::fs::write(&cli.output, &data) {
        eprintln!(
            "{} Failed to write file {:?}: {}",
            style("ERROR:").red().bold(),
            cli.output,
            e
        );
        std::process::exit(1);
    }

    println!("  ✓ Saved Crystal Classifier");

    println!(
        "\n{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );
    println!(
        "{}",
        style("         CRYSTAL RESERVOIR TRAINING COMPLETE               ")
            .bold()
            .green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );
    println!("\n  Next: eval-crystal --classifier {:?}", cli.output);
}

/// Serializable state for the Crystal classifier
#[derive(serde::Serialize, serde::Deserialize)]
struct CrystalClassifierState {
    phonemes: Vec<String>,
    prototypes: Vec<Vec<f32>>,
    counts: Vec<usize>,
    n_filters: usize,
    context_frames: usize,
    feature_dim: usize,
    n_mels: usize,
}
