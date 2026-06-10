// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LibriSpeech Training Pipeline
//!
//! Download and train on LibriSpeech dataset:
//! ```bash
//! # Download test-clean subset (~350MB)
//! symthaea-librispeech download --subset test-clean
//!
//! # Train on downloaded data
//! symthaea-librispeech train --data data/librispeech/test-clean
//!
//! # Download and train in one step
//! symthaea-librispeech auto --subset dev-clean
//! ```

use clap::{Parser, Subcommand};
use console::{Emoji, style};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use symthaea_stt::{
    AudioFrontend, AudioProjector, BatchConfig, BatchTrainer, BootstrapConfig, CmuDictionary,
    PhonemeResonator, TextToPhonemes, TrainedPrototypes, phoneme_error_rate,
};

static DOWNLOAD: Emoji<'_, '_> = Emoji("📥 ", "");
static EXTRACT: Emoji<'_, '_> = Emoji("📦 ", "");
static TRAIN: Emoji<'_, '_> = Emoji("🎓 ", "");
static SUCCESS: Emoji<'_, '_> = Emoji("✅ ", "[OK] ");
static WARN: Emoji<'_, '_> = Emoji("⚠️  ", "[WARN] ");

#[derive(Parser)]
#[command(name = "symthaea-librispeech")]
#[command(about = "LibriSpeech training pipeline for Symthaea STT")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download LibriSpeech subset
    Download {
        /// Subset to download
        #[arg(short, long, default_value = "test-clean")]
        subset: String,

        /// Output directory
        #[arg(short, long, default_value = "data/librispeech")]
        output: PathBuf,

        /// Skip extraction
        #[arg(long)]
        no_extract: bool,
    },

    /// Train on LibriSpeech data
    Train {
        /// Data directory (containing speaker folders)
        #[arg(short, long)]
        data: PathBuf,

        /// Output model file
        #[arg(short, long, default_value = "models/librispeech_prototypes.bin")]
        output: PathBuf,

        /// Maximum files to process
        #[arg(long)]
        max_files: Option<usize>,

        /// Number of worker threads
        #[arg(short, long, default_value = "0")]
        workers: usize,

        /// Minimum instances per phoneme
        #[arg(long, default_value = "10")]
        min_instances: usize,

        /// CMU pronunciation dictionary path (for phoneme-level training)
        #[arg(long)]
        dict: Option<PathBuf>,

        /// Use adaptive allophone clustering (prevents prototype collapse)
        #[arg(long)]
        adaptive: bool,

        /// Similarity threshold for adaptive clustering (0.0-1.0)
        #[arg(long, default_value = "0.75")]
        threshold: f32,

        /// Maximum variants per phoneme for adaptive clustering
        #[arg(long, default_value = "5")]
        max_variants: usize,
    },

    /// Auto: download and train
    Auto {
        /// Subset to use
        #[arg(short, long, default_value = "test-clean")]
        subset: String,

        /// Base directory
        #[arg(short, long, default_value = "data")]
        base_dir: PathBuf,

        /// Number of worker threads
        #[arg(short, long, default_value = "0")]
        workers: usize,

        /// CMU pronunciation dictionary path (for phoneme-level training)
        #[arg(long)]
        dict: Option<PathBuf>,
    },

    /// List available subsets
    List,

    /// Show dataset info
    Info {
        /// Data directory
        #[arg(short, long)]
        data: PathBuf,
    },

    /// Evaluate model on LibriSpeech data
    Eval {
        /// Trained model file
        #[arg(short, long)]
        model: PathBuf,

        /// Test data directory
        #[arg(short, long)]
        data: PathBuf,

        /// Maximum files to evaluate
        #[arg(long)]
        max_files: Option<usize>,

        /// Output report file (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// CMU dictionary path (optional, for word-level eval)
        #[arg(long)]
        dict: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Download {
            subset,
            output,
            no_extract,
        } => {
            if let Err(e) = download_subset(&subset, &output, !no_extract) {
                eprintln!("{}Download failed: {}", WARN, e);
                std::process::exit(1);
            }
        }
        Commands::Train {
            data,
            output,
            max_files,
            workers,
            min_instances,
            dict,
            adaptive,
            threshold,
            max_variants,
        } => {
            if adaptive {
                if let Err(e) = train_adaptive(
                    &data,
                    &output,
                    max_files,
                    workers,
                    min_instances,
                    dict.as_deref(),
                    threshold,
                    max_variants,
                ) {
                    eprintln!("{}Training failed: {}", WARN, e);
                    std::process::exit(1);
                }
            } else if let Err(e) = train_on_data(
                &data,
                &output,
                max_files,
                workers,
                min_instances,
                dict.as_deref(),
            ) {
                eprintln!("{}Training failed: {}", WARN, e);
                std::process::exit(1);
            }
        }
        Commands::Auto {
            subset,
            base_dir,
            workers,
            dict,
        } => {
            if let Err(e) = auto_pipeline(&subset, &base_dir, workers, dict.as_deref()) {
                eprintln!("{}Pipeline failed: {}", WARN, e);
                std::process::exit(1);
            }
        }
        Commands::List => {
            list_subsets();
        }
        Commands::Info { data } => {
            if let Err(e) = show_info(&data) {
                eprintln!("{}Info failed: {}", WARN, e);
                std::process::exit(1);
            }
        }
        Commands::Eval {
            model,
            data,
            max_files,
            output,
            dict,
        } => {
            if let Err(e) =
                evaluate_model(&model, &data, max_files, output.as_deref(), dict.as_deref())
            {
                eprintln!("{}Evaluation failed: {}", WARN, e);
                std::process::exit(1);
            }
        }
    }
}

/// LibriSpeech subset info
struct SubsetInfo {
    name: &'static str,
    description: &'static str,
    size_mb: usize,
    hours: f32,
}

const SUBSETS: &[SubsetInfo] = &[
    SubsetInfo {
        name: "test-clean",
        description: "Clean test set (40 speakers)",
        size_mb: 346,
        hours: 5.4,
    },
    SubsetInfo {
        name: "test-other",
        description: "Other test set (33 speakers)",
        size_mb: 328,
        hours: 5.1,
    },
    SubsetInfo {
        name: "dev-clean",
        description: "Clean dev set (40 speakers)",
        size_mb: 338,
        hours: 5.4,
    },
    SubsetInfo {
        name: "dev-other",
        description: "Other dev set (33 speakers)",
        size_mb: 314,
        hours: 5.3,
    },
    SubsetInfo {
        name: "train-clean-100",
        description: "100 hours clean training",
        size_mb: 6300,
        hours: 100.6,
    },
    SubsetInfo {
        name: "train-clean-360",
        description: "360 hours clean training",
        size_mb: 23000,
        hours: 363.6,
    },
    SubsetInfo {
        name: "train-other-500",
        description: "500 hours other training",
        size_mb: 30000,
        hours: 496.7,
    },
];

fn list_subsets() {
    println!("\n{}Available LibriSpeech subsets:\n", style("").bold());

    println!(
        "{:<20} {:<35} {:>10} {:>10}",
        style("Subset").bold(),
        style("Description").bold(),
        style("Size").bold(),
        style("Hours").bold()
    );
    println!("{}", "-".repeat(75));

    for subset in SUBSETS {
        println!(
            "{:<20} {:<35} {:>7} MB {:>7.1} h",
            subset.name, subset.description, subset.size_mb, subset.hours
        );
    }

    println!(
        "\n{}Recommended for testing: {}",
        style("Tip: ").cyan(),
        style("test-clean").green()
    );
    println!(
        "{}Full training: {}\n",
        style("Tip: ").cyan(),
        style("train-clean-100").green()
    );
}

fn download_subset(subset: &str, output: &Path, extract: bool) -> Result<(), String> {
    // Validate subset
    if !SUBSETS.iter().any(|s| s.name == subset) {
        return Err(format!(
            "Unknown subset '{}'. Run 'list' to see available subsets.",
            subset
        ));
    }

    let url = format!(
        "https://www.openslr.org/resources/12/{}.tar.gz",
        subset.replace("-", "-")
    );

    println!("\n{}LibriSpeech Download Pipeline", style("").bold().cyan());
    println!("{}", "=".repeat(50));
    println!("Subset:  {}", style(subset).green());
    println!("URL:     {}", style(&url).dim());
    println!("Output:  {}", output.display());
    println!();

    // Create output directory
    fs::create_dir_all(output).map_err(|e| format!("Failed to create output directory: {}", e))?;

    let tar_path = output.join(format!("{}.tar.gz", subset));

    // Check if already downloaded
    if tar_path.exists() {
        println!(
            "{}Archive already exists, skipping download",
            style("Note: ").yellow()
        );
    } else {
        println!("{}Downloading {} ...", DOWNLOAD, subset);
        println!(
            "{}This may take a while for larger subsets",
            style("Note: ").dim()
        );

        // Use curl or wget for download
        download_file(&url, &tar_path)?;
    }

    // Extract
    if extract {
        let extract_dir = output.join(subset);
        if extract_dir.exists() {
            println!("{}Already extracted, skipping", style("Note: ").yellow());
        } else {
            println!("\n{}Extracting archive...", EXTRACT);
            extract_tar_gz(&tar_path, output)?;
        }
    }

    println!("\n{}Download complete!", SUCCESS);
    println!(
        "Data location: {}",
        output.join("LibriSpeech").join(subset).display()
    );

    Ok(())
}

fn download_file(url: &str, output: &Path) -> Result<(), String> {
    // Try curl first, then wget
    let result = std::process::Command::new("curl")
        .args(["-L", "-o", &output.to_string_lossy(), "--progress-bar", url])
        .status();

    match result {
        Ok(status) if status.success() => Ok(()),
        _ => {
            // Try wget
            let result = std::process::Command::new("wget")
                .args(["-O", &output.to_string_lossy(), "--progress=bar:force", url])
                .status();

            match result {
                Ok(status) if status.success() => Ok(()),
                _ => Err("Failed to download. Please install curl or wget.".to_string()),
            }
        }
    }
}

fn extract_tar_gz(archive: &Path, output: &Path) -> Result<(), String> {
    let result = std::process::Command::new("tar")
        .args([
            "-xzf",
            &archive.to_string_lossy(),
            "-C",
            &output.to_string_lossy(),
        ])
        .status();

    match result {
        Ok(status) if status.success() => Ok(()),
        _ => Err("Failed to extract archive. Please install tar.".to_string()),
    }
}

fn train_on_data(
    data_dir: &Path,
    output: &Path,
    max_files: Option<usize>,
    workers: usize,
    min_instances: usize,
    dict_path: Option<&Path>,
) -> Result<(), String> {
    println!("\n{}LibriSpeech Training Pipeline", style("").bold().cyan());
    println!("{}", "=".repeat(50));
    println!("Data:    {}", data_dir.display());
    println!("Output:  {}", output.display());
    println!(
        "Workers: {}",
        if workers == 0 {
            "auto".to_string()
        } else {
            workers.to_string()
        }
    );
    if let Some(dict) = dict_path {
        println!("Dict:    {} (phoneme-level training)", dict.display());
    } else {
        println!("Dict:    {} (word-level training)", style("none").yellow());
    }
    println!();

    // Scan for transcript files and audio
    println!("{}Scanning for transcripts...", style("").dim());
    let samples = scan_librispeech_data(data_dir)?;

    if samples.is_empty() {
        return Err("No audio/transcript pairs found".to_string());
    }

    let total_samples = max_files
        .map(|m| m.min(samples.len()))
        .unwrap_or(samples.len());
    println!(
        "Found {} audio files with transcripts",
        style(total_samples).green()
    );
    println!();

    // Configure training
    let batch_config = BatchConfig {
        num_workers: workers,
        max_files,
        shuffle: true,
        continue_on_error: true,
        progress_interval: 50,
    };

    let bootstrap_config = BootstrapConfig {
        min_instances,
        ..Default::default()
    };

    // Create trainer with optional dictionary
    let mut trainer = BatchTrainer::new(batch_config, bootstrap_config.clone());

    if let Some(dict) = dict_path {
        println!("{}Loading pronunciation dictionary...", style("").dim());
        trainer
            .load_dictionary(dict)
            .map_err(|e| format!("Failed to load dictionary: {}", e))?;
        println!("Dictionary loaded - training phoneme prototypes");
    }

    // Progress bar
    let multi = MultiProgress::new();
    let progress = multi.add(ProgressBar::new(total_samples as u64));
    progress.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
        .unwrap()
        .progress_chars("█▓▒░ "));

    let progress_clone = progress.clone();
    let callback = Box::new(move |current: usize, _total: usize, path: &str| {
        progress_clone.set_position(current as u64);
        progress_clone.set_message(format!(
            "Processing {}",
            path.split('/').next_back().unwrap_or(path)
        ));
    });

    // Train
    println!("{}Training phoneme prototypes...", TRAIN);
    let samples_to_train: Vec<_> = samples.into_iter().take(total_samples).collect();
    let (prototypes, stats) = trainer.train(&samples_to_train, Some(callback));

    progress.finish_with_message("Training complete!");

    // Save prototypes
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    prototypes
        .save(output)
        .map_err(|e| format!("Failed to save prototypes: {}", e))?;

    // Print stats
    println!("\n{}Training Statistics", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Files processed:    {}", stats.files_processed);
    println!("Files succeeded:    {}", style(stats.files_success).green());
    println!(
        "Files failed:       {}",
        if stats.files_failed > 0 {
            style(stats.files_failed).red().to_string()
        } else {
            stats.files_failed.to_string()
        }
    );
    println!(
        "Audio processed:    {:.1} hours",
        stats.total_audio_sec / 3600.0
    );
    println!("Processing time:    {:.1} seconds", stats.total_time_sec);
    println!("Realtime factor:    {:.1}x", stats.realtime_factor);
    println!();
    println!("{}Phoneme Prototypes", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Total prototypes:   {}", prototypes.len());

    // Show top phonemes
    let mut counts: Vec<_> = prototypes.counts.iter().collect();
    counts.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

    println!("\nTop phonemes by frequency:");
    for (phoneme, count) in counts.iter().take(10) {
        println!("  {:>15}: {:>6} instances", phoneme, count);
    }

    println!("\n{}Model saved to: {}", SUCCESS, output.display());

    Ok(())
}

fn train_adaptive(
    data_dir: &Path,
    output: &Path,
    max_files: Option<usize>,
    workers: usize,
    min_instances: usize,
    dict_path: Option<&Path>,
    threshold: f32,
    max_variants: usize,
) -> Result<(), String> {
    println!(
        "\n{}LibriSpeech Adaptive Training Pipeline",
        style("").bold().cyan()
    );
    println!("{}", "=".repeat(50));
    println!("Data:       {}", data_dir.display());
    println!("Output:     {}", output.display());
    println!(
        "Workers:    {}",
        if workers == 0 {
            "auto".to_string()
        } else {
            workers.to_string()
        }
    );
    println!("Mode:       {}", style("ADAPTIVE").green().bold());
    println!("Threshold:  {:.2}", threshold);
    println!("Max Variants: {}", max_variants);
    if let Some(dict) = dict_path {
        println!("Dict:       {} (phoneme-level)", dict.display());
    } else {
        println!("Dict:       {} (word-level)", style("none").yellow());
    }
    println!();

    // Scan for transcript files and audio
    println!("{}Scanning for transcripts...", style("").dim());
    let samples = scan_librispeech_data(data_dir)?;

    if samples.is_empty() {
        return Err("No audio/transcript pairs found".to_string());
    }

    let total_samples = max_files
        .map(|m| m.min(samples.len()))
        .unwrap_or(samples.len());
    println!(
        "Found {} audio files with transcripts",
        style(total_samples).green()
    );
    println!();

    // Configure training
    let batch_config = BatchConfig {
        num_workers: workers,
        max_files,
        shuffle: true,
        continue_on_error: true,
        progress_interval: 50,
    };

    let bootstrap_config = BootstrapConfig {
        min_instances,
        ..Default::default()
    };

    // Create trainer with optional dictionary
    let mut trainer = BatchTrainer::new(batch_config, bootstrap_config.clone());

    if let Some(dict) = dict_path {
        println!("{}Loading pronunciation dictionary...", style("").dim());
        trainer
            .load_dictionary(dict)
            .map_err(|e| format!("Failed to load dictionary: {}", e))?;
        println!("Dictionary loaded - training phoneme prototypes with allophone discovery");
    }

    // Progress bar
    let multi = MultiProgress::new();
    let progress = multi.add(ProgressBar::new(total_samples as u64));
    progress.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
        .unwrap()
        .progress_chars("█▓▒░ "));

    let progress_clone = progress.clone();
    let callback = Box::new(move |current: usize, _total: usize, path: &str| {
        progress_clone.set_position(current as u64);
        progress_clone.set_message(format!(
            "Processing {}",
            path.split('/').next_back().unwrap_or(path)
        ));
    });

    // Train with adaptive clustering
    println!("{}Training with adaptive allophone clustering...", TRAIN);
    let samples_to_train: Vec<_> = samples.into_iter().take(total_samples).collect();
    let (adaptive_set, stats, adaptive_stats) =
        trainer.train_adaptive(&samples_to_train, threshold, max_variants, Some(callback));

    progress.finish_with_message("Training complete!");

    // Convert to TrainedPrototypes for saving
    // Use EXPANDED prototypes: each variant becomes a separate prototype
    // This allows multi-prototype matching during inference
    let mut prototypes = adaptive_set.to_expanded_prototypes(bootstrap_config);

    // CONTRASTIVE SEPARATION: Push similar prototypes apart to prevent collapse
    // Must run AFTER expansion to separate all variant pairs
    println!(
        "{}Enforcing prototype separation (post-expansion)...",
        TRAIN
    );
    prototypes.enforce_separation(0.55); // Push apart pairs with >0.55 similarity (below mirror's max 0.67)
    println!("  Separation enforcement complete");

    // Save prototypes
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    prototypes
        .save(output)
        .map_err(|e| format!("Failed to save prototypes: {}", e))?;

    // Print stats
    println!("\n{}Training Statistics", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Files processed:    {}", stats.files_processed);
    println!("Files succeeded:    {}", style(stats.files_success).green());
    println!(
        "Files failed:       {}",
        if stats.files_failed > 0 {
            style(stats.files_failed).red().to_string()
        } else {
            stats.files_failed.to_string()
        }
    );
    println!(
        "Audio processed:    {:.1} hours",
        stats.total_audio_sec / 3600.0
    );
    println!("Processing time:    {:.1} seconds", stats.total_time_sec);
    println!("Realtime factor:    {:.1}x", stats.realtime_factor);

    println!("\n{}Adaptive Training Statistics", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Phonemes discovered: {}", adaptive_stats.total_phonemes);
    println!(
        "Allophone variants:  {} (avg {:.1} per phoneme)",
        adaptive_stats.total_variants, adaptive_stats.avg_variants_per_phoneme
    );
    println!("Max variants seen:   {}", adaptive_stats.max_variants_seen);
    println!("Total instances:     {}", adaptive_stats.total_instances);

    println!("\n{}Phoneme Prototypes", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Total prototypes:   {}", prototypes.len());

    // Show top phonemes
    let mut counts: Vec<_> = prototypes.counts.iter().collect();
    counts.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

    println!("\nTop phonemes by frequency:");
    for (phoneme, count) in counts.iter().take(10) {
        println!("  {:>15}: {:>6} instances", phoneme, count);
    }

    // Show allophone distribution for some high-frequency phonemes
    println!("\n{}Allophone Distribution (sample)", style("").bold());
    println!("{}", "-".repeat(40));
    for phoneme in ["T", "N", "AH0", "S", "D"].iter() {
        if let Some(proto) = adaptive_set.prototypes.get(*phoneme) {
            let variant_info: Vec<_> = proto
                .variant_counts
                .iter()
                .enumerate()
                .map(|(i, c)| format!("v{}: {}", i, c))
                .collect();
            println!(
                "  {:>6}: {} variants [{}]",
                phoneme,
                proto.num_variants(),
                variant_info.join(", ")
            );
        }
    }

    println!("\n{}Model saved to: {}", SUCCESS, output.display());

    Ok(())
}

fn scan_librispeech_data(data_dir: &Path) -> Result<Vec<(PathBuf, String)>, String> {
    let mut samples = Vec::new();

    // LibriSpeech structure: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
    // Transcripts: speaker_id/chapter_id/speaker_id-chapter_id.trans.txt

    fn scan_recursive(dir: &Path, samples: &mut Vec<(PathBuf, String)>) -> Result<(), String> {
        let entries = fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {}", e))?;

        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();

            if path.is_dir() {
                scan_recursive(&path, samples)?;
            } else if path.extension().map(|e| e == "txt").unwrap_or(false) {
                // Check if it's a transcript file
                let filename = path.file_name().unwrap().to_string_lossy();
                if filename.ends_with(".trans.txt") {
                    // Parse transcript file
                    if let Ok(transcripts) = parse_transcript_file(&path) {
                        for (utterance_id, text) in transcripts {
                            // Find corresponding audio file
                            let audio_path = path
                                .parent()
                                .unwrap()
                                .join(format!("{}.flac", utterance_id));
                            if audio_path.exists() {
                                samples.push((audio_path, text));
                            } else {
                                // Try .wav
                                let wav_path =
                                    path.parent().unwrap().join(format!("{}.wav", utterance_id));
                                if wav_path.exists() {
                                    samples.push((wav_path, text));
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    scan_recursive(data_dir, &mut samples)?;

    Ok(samples)
}

fn parse_transcript_file(path: &Path) -> Result<Vec<(String, String)>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut transcripts = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Some(idx) = line.find(' ') {
            let utterance_id = line[..idx].to_string();
            let text = line[idx + 1..].to_string();
            transcripts.push((utterance_id, text));
        }
    }

    Ok(transcripts)
}

fn auto_pipeline(
    subset: &str,
    base_dir: &Path,
    workers: usize,
    dict_path: Option<&Path>,
) -> Result<(), String> {
    println!(
        "\n{}Symthaea LibriSpeech Auto Pipeline",
        style("").bold().cyan()
    );
    println!("{}", "=".repeat(50));
    println!("Subset:  {}", style(subset).green());
    println!("Base:    {}", base_dir.display());
    println!();

    // Step 1: Download
    let librispeech_dir = base_dir.join("librispeech");
    download_subset(subset, &librispeech_dir, true)?;

    // Step 2: Train
    let data_dir = librispeech_dir.join("LibriSpeech").join(subset);
    let output = base_dir
        .join("models")
        .join(format!("{}_prototypes.bin", subset));

    train_on_data(&data_dir, &output, None, workers, 10, dict_path)?;

    println!("\n{}Auto pipeline complete!", SUCCESS);
    println!("Model: {}", output.display());

    Ok(())
}

fn show_info(data_dir: &Path) -> Result<(), String> {
    println!("\n{}LibriSpeech Dataset Info", style("").bold().cyan());
    println!("{}", "=".repeat(50));
    println!("Directory: {}", data_dir.display());
    println!();

    // Scan data
    let samples = scan_librispeech_data(data_dir)?;

    if samples.is_empty() {
        return Err("No audio/transcript pairs found".to_string());
    }

    // Collect statistics
    let mut speakers: HashMap<String, usize> = HashMap::new();
    let mut chapters: HashMap<String, usize> = HashMap::new();
    let mut total_words = 0;
    let mut total_chars = 0;

    for (path, transcript) in &samples {
        // Extract speaker and chapter from path
        let path_str = path.to_string_lossy();
        let parts: Vec<_> = path_str.split('/').collect();
        if parts.len() >= 2 {
            let filename = parts[parts.len() - 1];
            let components: Vec<_> = filename.split('-').collect();
            if components.len() >= 2 {
                *speakers.entry(components[0].to_string()).or_insert(0) += 1;
                *chapters
                    .entry(format!("{}-{}", components[0], components[1]))
                    .or_insert(0) += 1;
            }
        }

        total_words += transcript.split_whitespace().count();
        total_chars += transcript.len();
    }

    println!("{}Statistics", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Audio files:     {}", samples.len());
    println!("Speakers:        {}", speakers.len());
    println!("Chapters:        {}", chapters.len());
    println!("Total words:     {}", total_words);
    println!("Total chars:     {}", total_chars);
    println!(
        "Avg words/file:  {:.1}",
        total_words as f32 / samples.len() as f32
    );

    // Show speaker distribution
    println!("\n{}Speakers (top 10):", style("").bold());
    let mut speaker_list: Vec<_> = speakers.iter().collect();
    speaker_list.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (speaker, count) in speaker_list.iter().take(10) {
        println!("  {}: {} utterances", speaker, count);
    }

    // Sample transcripts
    println!("\n{}Sample transcripts:", style("").bold());
    for (path, transcript) in samples.iter().take(3) {
        println!("  {}", path.file_name().unwrap().to_string_lossy());
        println!(
            "    \"{}\"",
            if transcript.len() > 60 {
                format!("{}...", &transcript[..60])
            } else {
                transcript.clone()
            }
        );
    }

    Ok(())
}

static EVAL: Emoji<'_, '_> = Emoji("📊 ", "[EVAL] ");

fn evaluate_model(
    model_path: &Path,
    data_dir: &Path,
    max_files: Option<usize>,
    output: Option<&Path>,
    dict_path: Option<&Path>,
) -> Result<(), String> {
    println!(
        "\n{}LibriSpeech Evaluation Pipeline",
        style("").bold().cyan()
    );
    println!("{}", "=".repeat(50));
    println!("Model:   {}", model_path.display());
    println!("Data:    {}", data_dir.display());
    if let Some(dict) = dict_path {
        println!("Dict:    {}", dict.display());
    }
    println!();

    // Load prototypes
    println!("{}Loading model...", style("").dim());
    let prototypes =
        TrainedPrototypes::load(model_path).map_err(|e| format!("Failed to load model: {}", e))?;

    println!(
        "Loaded {} phoneme prototypes",
        style(prototypes.len()).green()
    );

    // Create resonator and load prototypes
    let mut resonator = PhonemeResonator::new();
    for (phoneme, hv) in prototypes.as_pairs() {
        resonator.store(&phoneme, hv);
    }

    // Build phoneme class HVs for linguistic constraints
    let phoneme_hvs: std::collections::HashMap<String, symthaea_stt::HV16> = prototypes
        .as_pairs()
        .iter()
        .map(|(l, h)| (l.clone(), *h))
        .collect();
    let phoneme_classes = symthaea_stt::PhonemeClasses::new(&phoneme_hvs);
    let phonotactics = symthaea_stt::PhonotacticConstraints::new(&phoneme_hvs);
    println!("{}Initialized linguistic constraints", style("").dim());

    // Load dictionary if provided
    let text_to_phonemes = if let Some(dict) = dict_path {
        println!("{}Loading dictionary...", style("").dim());
        let dict =
            CmuDictionary::load(dict).map_err(|e| format!("Failed to load dictionary: {}", e))?;
        Some(TextToPhonemes::new(dict))
    } else {
        None
    };

    // Scan test data
    println!("\n{}Scanning test data...", style("").dim());
    let samples = scan_librispeech_data(data_dir)?;

    if samples.is_empty() {
        return Err("No test samples found".to_string());
    }

    let total_samples = max_files
        .map(|m| m.min(samples.len()))
        .unwrap_or(samples.len());
    println!("Evaluating on {} samples", style(total_samples).green());

    // Progress bar
    let multi = MultiProgress::new();
    let progress = multi.add(ProgressBar::new(total_samples as u64));
    progress.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
        .unwrap()
        .progress_chars("█▓▒░ "));

    // Create projector
    let mut projector = AudioProjector::default_config();

    let mut total_per = 0.0;
    let mut total_per_nostress = 0.0; // Stress-agnostic PER
    let mut per_count = 0;
    let mut total_ref_phonemes = 0;
    let mut total_hyp_phonemes = 0;

    // Per-phoneme statistics (stress-stripped)
    let mut ref_phoneme_counts: HashMap<String, usize> = HashMap::new();
    let mut hyp_phoneme_counts: HashMap<String, usize> = HashMap::new();

    // Helper to strip stress markers (AH0 -> AH, IY1 -> IY)
    fn strip_stress(p: &str) -> String {
        p.trim_end_matches(|c: char| c.is_ascii_digit()).to_string()
    }

    // Evaluate each sample
    println!("\n{}Running evaluation...", EVAL);
    let mut debug_first_file = true; // Debug flag
    for (i, (audio_path, reference_text)) in samples.iter().take(total_samples).enumerate() {
        progress.set_position(i as u64);

        // Load and process audio
        let (audio, _sample_rate) = match AudioFrontend::load_audio(audio_path) {
            Ok(result) => result,
            Err(_) => continue,
        };

        projector.reset();
        let hvs = projector.project(&audio);

        // DEBUG: Print first file scores and phonemes
        if debug_first_file {
            debug_first_file = false;
            let mut above_threshold = 0;
            let mut below_threshold = 0;
            let mut max_score: f32 = -2.0;
            let mut phoneme_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for (fi, hv) in hvs.iter().take(100).enumerate() {
                let results = resonator.query(hv, 1);
                if let Some((phoneme, score)) = results.first() {
                    if fi < 10 {
                        println!("  DEBUG Frame {}: {} score={:.4}", fi, phoneme, score);
                    }
                    max_score = max_score.max(*score);
                    if *score > 0.1 {
                        above_threshold += 1;
                        *phoneme_counts.entry(phoneme.clone()).or_insert(0) += 1;
                    } else {
                        below_threshold += 1;
                    }
                }
            }
            println!(
                "  DEBUG: {} above 0.1, {} below 0.1, max={:.4}",
                above_threshold, below_threshold, max_score
            );
            // Show which phonemes are being predicted
            let mut counts: Vec<_> = phoneme_counts.into_iter().collect();
            counts.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            println!(
                "  DEBUG Phonemes above 0.1: {:?}",
                counts.iter().take(10).collect::<Vec<_>>()
            );
        }

        // Decode phonemes with LINGUISTIC CONSTRAINTS
        let mut predicted_phonemes: Vec<String> = Vec::new();
        let mut prev_phoneme = String::new();

        for hv in &hvs {
            // Get top-5 candidates for re-ranking with linguistic constraints
            let candidates = resonator.query(hv, 5);

            if candidates.is_empty() {
                continue;
            }

            // Find best candidate considering:
            // 1. Acoustic score (from resonator)
            // 2. Class consistency (manner, voicing, place)
            // 3. Phonotactic transition score
            let mut best_phoneme = String::new();
            let mut best_score: f32 = -999.0;

            for (phoneme, acoustic_score) in &candidates {
                // Strip variant suffix
                let base_phoneme = if let Some(idx) = phoneme.rfind('_') {
                    if phoneme[idx + 1..].chars().all(|c| c.is_ascii_digit()) {
                        phoneme[..idx].to_string()
                    } else {
                        phoneme.clone()
                    }
                } else {
                    phoneme.clone()
                };

                // Class consistency multiplier (0.5 to 1.5)
                let class_mult = phoneme_classes.class_consistency_score(hv, &base_phoneme);

                // Phonotactic transition score (0.5 to 1.5)
                let trans_mult = if prev_phoneme.is_empty() {
                    1.0 // No previous phoneme
                } else {
                    phonotactics.transition_score(&prev_phoneme, &base_phoneme)
                };

                // Combined score: acoustic + linguistic bonus
                // Use ADDITIVE scoring so weak acoustic can't be saved by linguistics
                let linguistic_bonus = (class_mult - 1.0) * 0.05 + (trans_mult - 1.0) * 0.03;
                let combined = acoustic_score + linguistic_bonus;

                if combined > best_score {
                    best_score = combined;
                    best_phoneme = base_phoneme;
                }
            }

            // CTC-style greedy decoding:
            // 1. Threshold gate: Ignore low-confidence frames
            // 2. Collapse repeats: Only emit on phoneme change
            const CONFIDENCE_THRESHOLD: f32 = 0.12; // Acoustic must be strong enough

            if best_score < CONFIDENCE_THRESHOLD {
                continue;
            }

            if best_phoneme != prev_phoneme {
                predicted_phonemes.push(best_phoneme.clone());
                prev_phoneme = best_phoneme;
            }
        }

        // Get reference phonemes
        let reference_phonemes: Vec<String> = if let Some(ref ttp) = text_to_phonemes {
            ttp.convert(reference_text)
                .into_iter()
                .filter(|p| p != "_" && p != "SIL") // Remove silence markers
                .collect()
        } else {
            // Without dictionary, use placeholder
            Vec::new()
        };

        // Calculate PER if we have reference phonemes
        if !reference_phonemes.is_empty() && !predicted_phonemes.is_empty() {
            // Convert to &str slices for phoneme_error_rate
            let ref_strs: Vec<&str> = reference_phonemes.iter().map(String::as_str).collect();
            let hyp_strs: Vec<&str> = predicted_phonemes.iter().map(String::as_str).collect();

            // DEBUG: Print first file's reference vs predicted
            if i == 0 {
                println!(
                    "  DEBUG REF ({} phonemes): {:?}",
                    reference_phonemes.len(),
                    &reference_phonemes[..reference_phonemes.len().min(30)]
                );
                println!(
                    "  DEBUG HYP ({} phonemes): {:?}",
                    predicted_phonemes.len(),
                    &predicted_phonemes[..predicted_phonemes.len().min(30)]
                );
            }

            let per = phoneme_error_rate(&ref_strs, &hyp_strs);
            total_per += per;

            // Also compute stress-agnostic PER
            let ref_nostress: Vec<String> =
                reference_phonemes.iter().map(|p| strip_stress(p)).collect();
            let hyp_nostress: Vec<String> =
                predicted_phonemes.iter().map(|p| strip_stress(p)).collect();
            let ref_ns: Vec<&str> = ref_nostress.iter().map(String::as_str).collect();
            let hyp_ns: Vec<&str> = hyp_nostress.iter().map(String::as_str).collect();
            let per_nostress = phoneme_error_rate(&ref_ns, &hyp_ns);
            total_per_nostress += per_nostress;

            // Track per-phoneme distribution
            for p in &ref_nostress {
                *ref_phoneme_counts.entry(p.clone()).or_insert(0) += 1;
            }
            for p in &hyp_nostress {
                *hyp_phoneme_counts.entry(p.clone()).or_insert(0) += 1;
            }

            per_count += 1;
            total_ref_phonemes += reference_phonemes.len();
            total_hyp_phonemes += predicted_phonemes.len();
        }
    }

    progress.finish_with_message("Evaluation complete!");

    // Compute averages
    let avg_per = if per_count > 0 {
        total_per / per_count as f32
    } else {
        1.0
    };
    let avg_per_nostress = if per_count > 0 {
        total_per_nostress / per_count as f32
    } else {
        1.0
    };

    // Print results
    println!("\n{}Evaluation Results", style("").bold().cyan());
    println!("{}", "=".repeat(50));
    println!("Samples evaluated:  {}", per_count);
    println!("Reference phonemes: {}", total_ref_phonemes);
    println!("Hypothesis phonemes:{}", total_hyp_phonemes);
    println!();
    println!("{}Phoneme Error Rate (PER)", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Average PER:           {:.1}%", avg_per * 100.0);
    println!("PER (no stress):       {:.1}%", avg_per_nostress * 100.0);
    println!(
        "Accuracy (no stress):  {:.1}%",
        (1.0 - avg_per_nostress) * 100.0
    );
    println!();

    if text_to_phonemes.is_none() {
        println!(
            "{}Note: Provide --dict for phoneme-level evaluation",
            style("Tip: ").yellow()
        );
    }

    // Print per-phoneme statistics
    if !ref_phoneme_counts.is_empty() {
        println!("{}Per-phoneme Analysis (stress-stripped)", style("").bold());
        println!("{}", "-".repeat(40));

        // Sort reference phonemes by frequency
        let mut ref_sorted: Vec<_> = ref_phoneme_counts.iter().collect();
        ref_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

        println!("Top phonemes - Reference vs Hypothesis:");
        for (phoneme, ref_count) in ref_sorted.iter().take(15) {
            let hyp_count = hyp_phoneme_counts.get(*phoneme).unwrap_or(&0);
            let ratio = *hyp_count as f32 / **ref_count as f32;
            let status = if ratio < 0.5 {
                "UNDER"
            } else if ratio > 2.0 {
                "OVER"
            } else {
                "~OK"
            };
            println!(
                "  {:4}: ref={:4} hyp={:4} ({:.2}x) {}",
                phoneme, ref_count, hyp_count, ratio, status
            );
        }

        // Show phonemes that are over-predicted (not in top reference)
        let mut hyp_only: Vec<_> = hyp_phoneme_counts
            .iter()
            .filter(|(p, c)| **c > 50 && ref_phoneme_counts.get(*p).unwrap_or(&0) < &(**c / 3))
            .collect();
        hyp_only.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

        if !hyp_only.is_empty() {
            println!("\nOver-predicted phonemes (hyp >> ref):");
            for (phoneme, hyp_count) in hyp_only.iter().take(5) {
                let ref_count = ref_phoneme_counts.get(*phoneme).unwrap_or(&0);
                println!("  {:4}: hyp={:4} ref={:4}", phoneme, hyp_count, ref_count);
            }
        }
        println!();
    }

    // Save report if output path provided
    if let Some(output_path) = output {
        let report_json = serde_json::json!({
            "samples_evaluated": per_count,
            "average_per": avg_per,
            "model": model_path.to_string_lossy(),
            "data_dir": data_dir.to_string_lossy(),
        });

        let mut file = File::create(output_path)
            .map_err(|e| format!("Failed to create output file: {}", e))?;
        file.write_all(
            serde_json::to_string_pretty(&report_json)
                .unwrap()
                .as_bytes(),
        )
        .map_err(|e| format!("Failed to write report: {}", e))?;

        println!("\n{}Report saved to: {}", SUCCESS, output_path.display());
    }

    println!("\n{}Evaluation complete!", SUCCESS);

    Ok(())
}
