// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train Phoneme Prototypes with Ground Truth Alignments
//!
//! Uses pre-computed forced alignments from Parquet files to train
//! phoneme prototypes with exact phoneme boundaries.
//!
//! This replaces the unsupervised DTW approach with supervised training.
//!
//! Usage:
//! ```bash
//! cargo run --release --bin train-aligned -- \
//!     --alignments data/alignments/dev_clean.parquet \
//!     --audio-dir data/librispeech/LibriSpeech/dev-clean \
//!     --output data/models/aligned_prototypes.bin
//! ```

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use symthaea_stt::hdc::BundleAccumulator;
use symthaea_stt::{
    AudioFrontend, AudioProjector, BootstrapConfig, HV16, TrainedPrototypes, id_to_audio_path,
    load_alignments,
};

#[derive(Parser)]
#[command(
    name = "train-aligned",
    about = "Train phoneme prototypes with ground truth alignments",
    version,
    author
)]
struct Cli {
    /// Parquet file with alignments
    #[arg(short, long)]
    alignments: PathBuf,

    /// LibriSpeech audio directory (e.g., .../LibriSpeech/dev-clean)
    #[arg(short = 'd', long)]
    audio_dir: PathBuf,

    /// Output prototypes file
    #[arg(short, long, default_value = "data/models/aligned_prototypes.bin")]
    output: PathBuf,

    /// Maximum utterances to process (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Minimum instances per phoneme to include
    #[arg(long, default_value = "100")]
    min_instances: usize,

    /// Apply contrastive separation after training
    #[arg(long, default_value = "0.20")]
    min_separation: f32,

    /// Frame hop duration in seconds (should match AudioProjector)
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Accumulator for phoneme prototypes
struct PhonemeAccumulator {
    accumulators: HashMap<String, BundleAccumulator>,
    counts: HashMap<String, usize>,
}

impl PhonemeAccumulator {
    fn new() -> Self {
        Self {
            accumulators: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    fn add(&mut self, phoneme: &str, hv: &HV16) {
        let acc = self.accumulators.entry(phoneme.to_string()).or_default();
        acc.add(hv);

        *self.counts.entry(phoneme.to_string()).or_insert(0) += 1;
    }

    fn finalize(self, min_instances: usize) -> HashMap<String, HV16> {
        let mut prototypes = HashMap::new();

        for (phoneme, acc) in self.accumulators {
            let count = self.counts.get(&phoneme).copied().unwrap_or(0);
            if count >= min_instances {
                prototypes.insert(phoneme, acc.finalize());
            }
        }

        prototypes
    }

    fn counts(&self) -> &HashMap<String, usize> {
        &self.counts
    }
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("    SUPERVISED PHONEME PROTOTYPE TRAINING                  ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!();
    println!("  Uses ground truth phoneme boundaries from forced alignment");
    println!("  This is the GOLDEN PATH to accurate prototypes");
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

    // Create audio projector
    let mut projector = AudioProjector::default_config();

    // Accumulator for phoneme prototypes
    let mut accumulator = PhonemeAccumulator::new();

    // Determine how many utterances to process
    let total = if cli.max_utterances > 0 {
        alignments.len().min(cli.max_utterances)
    } else {
        alignments.len()
    };

    println!(
        "\n📂 Processing {} utterances from {:?}...",
        total, cli.audio_dir
    );

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
    let mut total_frames = 0usize;
    let mut total_segments = 0usize;

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

        // Project to hypervectors (using raw frames without temporal windowing)
        projector.reset();
        let hvs = projector.project_raw(&samples);
        if hvs.is_empty() {
            skipped += 1;
            continue;
        }

        total_frames += hvs.len();

        // For each phoneme segment, extract frames within its boundaries
        for segment in &alignment.phonemes {
            // Convert time to frame index
            let start_frame = (segment.start / cli.frame_hop as f64) as usize;
            let end_frame = (segment.end / cli.frame_hop as f64) as usize;

            // Clamp to valid range
            let start_frame = start_frame.min(hvs.len().saturating_sub(1));
            let end_frame = end_frame.min(hvs.len()).max(start_frame + 1);

            // Only use center portion of segment (skip boundaries)
            // Phoneme transitions at boundaries cause noise
            let segment_len = end_frame - start_frame;
            let margin = segment_len / 4; // Skip first and last 25%
            let center_start = start_frame + margin;
            let center_end = end_frame.saturating_sub(margin).max(center_start + 1);

            // Add center frames to the phoneme accumulator
            for frame_idx in center_start..center_end {
                accumulator.add(&segment.phoneme, &hvs[frame_idx]);
            }

            total_segments += 1;
        }

        processed += 1;

        // Update progress
        if i % 100 == 0 {
            pb.set_message(format!(
                "{} frames, {} segments",
                total_frames, total_segments
            ));
        }
    }

    pb.finish_with_message("done");

    println!("\n  📊 Statistics:");
    println!("    Processed: {} utterances", processed);
    println!("    Skipped: {} utterances (missing audio)", skipped);
    println!("    Total frames: {}", total_frames);
    println!("    Total phoneme segments: {}", total_segments);

    // Show phoneme coverage (clone counts before finalize consumes accumulator)
    let counts = accumulator.counts().clone();
    let mut sorted_counts: Vec<_> = counts.iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(a.1));

    println!("\n  Top phonemes:");
    for (ph, count) in sorted_counts.iter().take(10) {
        println!("    {} - {}", ph, count);
    }

    let phonemes_above_min = sorted_counts
        .iter()
        .filter(|(_, c)| **c >= cli.min_instances)
        .count();
    println!(
        "    Phonemes with ≥{} instances: {}",
        cli.min_instances, phonemes_above_min
    );

    // Finalize prototypes (consumes accumulator)
    let mut prototypes = accumulator.finalize(cli.min_instances);

    // Apply contrastive separation
    if cli.min_separation > 0.0 {
        println!(
            "\n🔧 Applying contrastive separation (target: <{})...",
            cli.min_separation
        );

        let mut trained = TrainedPrototypes {
            prototypes: prototypes.clone(),
            counts: HashMap::new(),
            config: BootstrapConfig::default(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        trained.enforce_separation(cli.min_separation);
        prototypes = trained.prototypes;
    }

    // Diagnostic: Test within-class similarity on a few samples
    if cli.verbose {
        println!("\n🔬 Diagnostic: Testing prototype quality...");

        // Re-process a few utterances and check similarity
        let mut within_class_sims: Vec<f32> = Vec::new();
        let mut cross_class_sims: Vec<f32> = Vec::new();

        for (id, alignment) in alignments.iter().take(10) {
            let audio_path = match id_to_audio_path(id, &cli.audio_dir) {
                Some(p) if p.exists() => p,
                _ => continue,
            };

            let (samples, _) = match AudioFrontend::load_audio(&audio_path) {
                Ok(a) => a,
                Err(_) => continue,
            };

            projector.reset();
            let hvs = projector.project_raw(&samples);

            for segment in alignment.phonemes.iter().take(10) {
                if let Some(proto) = prototypes.get(&segment.phoneme) {
                    let start_frame = (segment.start / cli.frame_hop as f64) as usize;
                    let end_frame = (segment.end / cli.frame_hop as f64) as usize;

                    let start_frame = start_frame.min(hvs.len().saturating_sub(1));
                    let end_frame = end_frame.min(hvs.len()).max(start_frame + 1);

                    // Check center frame similarity
                    let center = (start_frame + end_frame) / 2;
                    if center < hvs.len() {
                        let sim = hvs[center].similarity(proto);
                        within_class_sims.push(sim);

                        // Compare to wrong phoneme
                        for (other_ph, other_proto) in prototypes.iter().take(5) {
                            if other_ph != &segment.phoneme {
                                cross_class_sims.push(hvs[center].similarity(other_proto));
                            }
                        }
                    }
                }
            }
        }

        if !within_class_sims.is_empty() {
            let avg_within = within_class_sims.iter().sum::<f32>() / within_class_sims.len() as f32;
            let avg_cross = cross_class_sims.iter().sum::<f32>() / cross_class_sims.len() as f32;
            let max_within = within_class_sims
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_within = within_class_sims
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);

            println!(
                "    Within-class similarity: avg={:.3}, min={:.3}, max={:.3}",
                avg_within, min_within, max_within
            );
            println!("    Cross-class similarity: avg={:.3}", avg_cross);
            println!("    Separation margin: {:.3}", avg_within - avg_cross);
        }
    }

    // Save prototypes
    println!("\n💾 Saving prototypes to {:?}...", cli.output);

    // Ensure output directory exists
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent).ok();
    }

    let trained = TrainedPrototypes {
        prototypes,
        counts: counts.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        config: BootstrapConfig::default(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    if let Err(e) = trained.save(&cli.output) {
        eprintln!("{} Failed to save: {}", style("ERROR:").red().bold(), e);
        std::process::exit(1);
    }

    println!("  ✓ Saved {} phonemes", trained.len());

    println!(
        "\n{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("                SUPERVISED TRAINING COMPLETE               ")
            .bold()
            .green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!("\n  Next: stt-eval --prototypes {:?}", cli.output);
}
