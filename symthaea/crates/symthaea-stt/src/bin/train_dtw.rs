// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DTW-Aligned Prototype Training
//!
//! Trains phoneme prototypes using Dynamic Time Warping alignment
//! instead of salience-based alignment. This provides more accurate
//! frame-to-phoneme correspondence.
//!
//! The training proceeds in iterations:
//! 1. First pass: Uniform alignment (no prototypes yet)
//! 2. Subsequent passes: DTW alignment using current prototypes
//! 3. Each iteration refines the prototypes
//!
//! Usage:
//! ```bash
//! cargo run --release --bin train-dtw -- \
//!     --data-dir data/librispeech \
//!     --output data/models/dtw_prototypes.bin \
//!     --iterations 3
//! ```

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use symthaea_stt::{
    AudioFrontend, AudioProjector, BootstrapConfig, CmuDictionary, DtwTrainer, HV16,
    TextToPhonemes, TrainedPrototypes,
};

#[derive(Parser)]
#[command(
    name = "train-dtw",
    about = "Train phoneme prototypes with DTW alignment",
    version,
    author
)]
struct Cli {
    /// LibriSpeech data directory
    #[arg(short, long)]
    data_dir: PathBuf,

    /// Output prototypes file
    #[arg(short, long, default_value = "data/models/dtw_prototypes.bin")]
    output: PathBuf,

    /// Subset to train on
    #[arg(long, default_value = "dev-clean")]
    subset: String,

    /// Number of training iterations (1 = uniform, 2+ = DTW refinement)
    #[arg(long, default_value = "3")]
    iterations: usize,

    /// Maximum utterances per iteration (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Minimum instances per phoneme to include
    #[arg(long, default_value = "50")]
    min_instances: usize,

    /// Apply contrastive separation after training
    #[arg(long, default_value = "0.25")]
    min_separation: f32,

    /// Path to CMU dictionary
    #[arg(long, default_value = "data/dict/cmudict.dict")]
    dictionary: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("          DTW-ALIGNED PROTOTYPE TRAINING                   ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!();
    println!("  Uses Dynamic Time Warping for precise frame-phoneme alignment");
    println!(
        "  Iterations: {} (1=uniform, 2+=DTW refinement)",
        cli.iterations
    );
    println!();

    // Check paths
    let subset_path = cli.data_dir.join("LibriSpeech").join(&cli.subset);
    if !subset_path.exists() {
        eprintln!(
            "{} Subset not found: {:?}",
            style("ERROR:").red().bold(),
            subset_path
        );
        std::process::exit(1);
    }

    // Load dictionary
    println!("📖 Loading dictionary from {:?}...", cli.dictionary);
    let dictionary = match CmuDictionary::load(&cli.dictionary) {
        Ok(d) => d,
        Err(e) => {
            // Try alternate location
            let alt_path = cli.data_dir.join("dict").join("cmudict.dict");
            match CmuDictionary::load(&alt_path) {
                Ok(d) => d,
                Err(_) => {
                    eprintln!(
                        "{} Failed to load dictionary: {}",
                        style("ERROR:").red().bold(),
                        e
                    );
                    std::process::exit(1);
                }
            }
        }
    };
    let text_to_phonemes = TextToPhonemes::new(dictionary.clone()).with_silence(true);
    println!("  ✓ Loaded {} words", dictionary.len());

    // Scan for utterances
    println!("\n📂 Scanning {:?}...", subset_path);
    let utterances = scan_librispeech(&subset_path);
    let total = if cli.max_utterances > 0 {
        utterances.len().min(cli.max_utterances)
    } else {
        utterances.len()
    };
    println!(
        "  ✓ Found {} utterances, will process {}",
        utterances.len(),
        total
    );

    // Create audio projector
    let mut projector = AudioProjector::default_config();

    // Iterative training
    let mut prototypes: HashMap<String, HV16> = HashMap::new();

    for iteration in 0..cli.iterations {
        println!(
            "\n{}",
            style(format!(
                "══════════════ ITERATION {} ══════════════",
                iteration + 1
            ))
            .yellow()
        );

        let alignment_type = if iteration == 0 { "uniform" } else { "DTW" };
        println!("  Alignment: {}", alignment_type);
        println!("  Prototypes: {}", prototypes.len());

        // Create trainer with current prototypes
        let mut trainer = if prototypes.is_empty() {
            DtwTrainer::new()
        } else {
            DtwTrainer::with_prototypes(prototypes.clone())
        };

        // Progress bar
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut total_frames = 0usize;
        let mut total_phonemes = 0usize;

        for (i, (audio_path, transcript)) in utterances.iter().take(total).enumerate() {
            pb.set_position(i as u64);

            // Load audio
            let (samples, _sr) = match AudioFrontend::load_audio(audio_path) {
                Ok(a) => a,
                Err(_) => continue,
            };

            // Project to hypervectors
            projector.reset();
            let hvs = projector.project(&samples);

            // Convert transcript to phonemes
            let phonemes = text_to_phonemes.convert(&transcript.to_lowercase());
            if phonemes.is_empty() || hvs.is_empty() {
                continue;
            }

            // Train
            let pairs_added = trainer.train_utterance(&hvs, &phonemes);
            total_frames += hvs.len();
            total_phonemes += pairs_added;

            // Update progress
            if i % 100 == 0 {
                pb.set_message(format!("{} frames", total_frames));
            }
        }

        pb.finish_with_message("done");

        println!("\n  📊 Statistics:");
        println!("    Total frames: {}", total_frames);
        println!("    Frame-phoneme pairs: {}", total_phonemes);

        // Show phoneme coverage before finalizing (verbose only)
        if cli.verbose {
            let mut counts: Vec<_> = trainer.counts().iter().collect();
            counts.sort_by(|a, b| b.1.cmp(a.1));
            println!("\n  Top phonemes:");
            for (ph, count) in counts.iter().take(10) {
                println!("    {} - {}", ph, count);
            }
        }

        // Finalize this iteration
        prototypes = trainer.finalize(cli.min_instances);
        println!(
            "    Phonemes with ≥{} instances: {}",
            cli.min_instances,
            prototypes.len()
        );
    }

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

    // Save prototypes
    println!("\n💾 Saving prototypes to {:?}...", cli.output);

    // Ensure output directory exists
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent).ok();
    }

    let trained = TrainedPrototypes {
        prototypes,
        counts: HashMap::new(),
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
        style("                    TRAINING COMPLETE                       ")
            .bold()
            .green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!("\n  Next: stt-eval --prototypes {:?}", cli.output);
}

/// Scan LibriSpeech directory for utterances
fn scan_librispeech(path: &Path) -> Vec<(PathBuf, String)> {
    let mut utterances = Vec::new();

    // Walk directory structure: speaker/chapter/*.trans.txt
    if let Ok(speakers) = fs::read_dir(path) {
        for speaker_entry in speakers.filter_map(|e| e.ok()) {
            if !speaker_entry.path().is_dir() {
                continue;
            }

            if let Ok(chapters) = fs::read_dir(speaker_entry.path()) {
                for chapter_entry in chapters.filter_map(|e| e.ok()) {
                    if !chapter_entry.path().is_dir() {
                        continue;
                    }

                    // Find trans.txt file
                    if let Ok(files) = fs::read_dir(chapter_entry.path()) {
                        for file_entry in files.filter_map(|e| e.ok()) {
                            let file_path = file_entry.path();
                            if file_path.extension().map(|e| e == "txt").unwrap_or(false)
                                && file_path.to_string_lossy().contains("trans")
                            {
                                // Parse trans file
                                if let Ok(file) = File::open(&file_path) {
                                    let reader = BufReader::new(file);
                                    for line in reader.lines().filter_map(|l| l.ok()) {
                                        let parts: Vec<&str> = line.splitn(2, ' ').collect();
                                        if parts.len() == 2 {
                                            let audio_file = chapter_entry
                                                .path()
                                                .join(format!("{}.flac", parts[0]));
                                            utterances.push((audio_file, parts[1].to_string()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    utterances
}
