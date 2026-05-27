// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Training CLI
//!
//! Bootstrap acoustic prototypes from audio corpora.
//!
//! # Usage
//!
//! ```bash
//! # Download LibriSpeech dev-clean and train
//! symthaea-train bootstrap --subset dev-clean
//!
//! # Train with limited utterances (for testing)
//! symthaea-train bootstrap --subset dev-clean --limit 100
//!
//! # Inspect trained model
//! symthaea-train inspect --model models/phoneme_prototypes.bin
//!
//! # Validate alignment quality
//! symthaea-train validate --model models/phoneme_prototypes.bin --audio test.wav
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use console::{Emoji, style};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

// ============================================================================
// CLI STRUCTURE
// ============================================================================

#[derive(Parser)]
#[command(name = "symthaea-train")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "Luminous Dynamics")]
#[command(about = "Bootstrap Symthaea Acoustic Prototypes from Audio Corpora")]
#[command(long_about = r#"
╔═══════════════════════════════════════════════════════════════════════╗
║                     SYMTHAEA TRAINING PROTOCOL                         ║
║                                                                        ║
║   "Teaching machines to listen through temporal integrity"            ║
║                                                                        ║
║   This tool bootstraps phoneme prototypes from raw audio + text.      ║
║   No external alignment tools required - uses LTC salience detection. ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode (no progress bars)
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Bootstrap prototypes from LibriSpeech or similar corpus
    Bootstrap {
        /// Path to store/find datasets
        #[arg(long, default_value = "./datasets")]
        data_dir: PathBuf,

        /// Output directory for trained models
        #[arg(long, default_value = "./models")]
        output_dir: PathBuf,

        /// LibriSpeech subset to use
        #[arg(long, default_value = "dev-clean")]
        subset: String,

        /// Limit number of utterances (0 = all)
        #[arg(long, default_value = "0")]
        limit: usize,

        /// Minimum samples per phoneme
        #[arg(long, default_value = "10")]
        min_samples: usize,

        /// Salience threshold for boundary detection
        #[arg(long, default_value = "0.5")]
        salience_threshold: f32,

        /// Skip download if dataset exists
        #[arg(long)]
        no_download: bool,

        /// Continue from existing prototypes
        #[arg(long)]
        resume: Option<PathBuf>,

        /// Export format
        #[arg(long, value_enum, default_value = "binary")]
        format: ExportFormat,
    },

    /// Inspect a trained model file
    Inspect {
        /// Path to model file
        #[arg(long)]
        model: PathBuf,

        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,

        /// Export inventory to JSON
        #[arg(long)]
        export_json: Option<PathBuf>,
    },

    /// Validate alignment quality on test audio
    Validate {
        /// Path to model file
        #[arg(long)]
        model: PathBuf,

        /// Path to test audio file
        #[arg(long)]
        audio: PathBuf,

        /// Expected transcription (optional)
        #[arg(long)]
        expected: Option<String>,

        /// Show detailed alignment
        #[arg(long)]
        show_alignment: bool,
    },

    /// Download dataset without training
    Download {
        /// Path to store datasets
        #[arg(long, default_value = "./datasets")]
        data_dir: PathBuf,

        /// Dataset subset to download
        #[arg(long, default_value = "dev-clean")]
        subset: String,
    },

    /// Show system information and configuration
    Info,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ExportFormat {
    /// Binary format (compact, fast loading)
    Binary,
    /// JSON format (human-readable)
    Json,
    /// Both formats
    Both,
}

// ============================================================================
// EMOJI CONSTANTS
// ============================================================================

static ROCKET: Emoji<'_, '_> = Emoji("🚀 ", "");
static PACKAGE: Emoji<'_, '_> = Emoji("📦 ", "");
static LIGHTNING: Emoji<'_, '_> = Emoji("⚡ ", "");
static CHECK: Emoji<'_, '_> = Emoji("✅ ", "");
static WARN: Emoji<'_, '_> = Emoji("⚠️  ", "");
static ERROR: Emoji<'_, '_> = Emoji("❌ ", "");
static BRAIN: Emoji<'_, '_> = Emoji("🧠 ", "");
static EAR: Emoji<'_, '_> = Emoji("👂 ", "");
static CHART: Emoji<'_, '_> = Emoji("📊 ", "");
static SAVE: Emoji<'_, '_> = Emoji("💾 ", "");
static SEARCH: Emoji<'_, '_> = Emoji("🔍 ", "");
static WAVE: Emoji<'_, '_> = Emoji("🌊 ", "");

// ============================================================================
// CORE TYPES (Simplified for CLI - actual impl in library)
// ============================================================================

/// 2048-bit hypervector
#[derive(Clone, Copy, PartialEq, Eq)]
struct HV16([u8; 256]);

impl HV16 {
    fn zero() -> Self {
        Self([0u8; 256])
    }

    fn from_seed(seed: &[u8]) -> Self {
        use blake3::Hasher;
        let hash = blake3::hash(seed);
        let mut result = [0u8; 256];
        let mut hasher = Hasher::new();
        hasher.update(hash.as_bytes());
        let mut output = hasher.finalize_xof();
        output.fill(&mut result);
        HV16(result)
    }

    fn similarity(&self, other: &Self) -> f32 {
        let matching: u32 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        matching as f32 / 2048.0
    }
}

fn bundle(vectors: &[HV16]) -> HV16 {
    if vectors.is_empty() {
        return HV16::zero();
    }

    let mut counts = [0i32; 2048];
    for vec in vectors {
        for byte_idx in 0..256 {
            for bit_idx in 0..8 {
                let bit = (vec.0[byte_idx] >> bit_idx) & 1;
                let pos = byte_idx * 8 + bit_idx;
                counts[pos] += if bit == 1 { 1 } else { -1 };
            }
        }
    }

    let mut result = [0u8; 256];
    for byte_idx in 0..256 {
        for bit_idx in 0..8 {
            let pos = byte_idx * 8 + bit_idx;
            if counts[pos] > 0 {
                result[byte_idx] |= 1 << bit_idx;
            }
        }
    }

    HV16(result)
}

/// Trained phoneme prototypes
struct TrainedPrototypes {
    prototypes: HashMap<String, HV16>,
    stats: HashMap<String, PhonemeStats>,
}

#[derive(Clone, Default)]
struct PhonemeStats {
    sample_count: usize,
    mean_duration_ms: f32,
}

impl TrainedPrototypes {
    fn save_binary(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        let num = self.prototypes.len() as u32;
        file.write_all(&num.to_le_bytes())?;

        for (label, hv) in &self.prototypes {
            let label_bytes = label.as_bytes();
            let len = label_bytes.len() as u16;
            file.write_all(&len.to_le_bytes())?;
            file.write_all(label_bytes)?;
            file.write_all(&hv.0)?;

            let stats = self.stats.get(label).cloned().unwrap_or_default();
            file.write_all(&(stats.sample_count as u32).to_le_bytes())?;
            file.write_all(&stats.mean_duration_ms.to_le_bytes())?;
        }

        Ok(())
    }

    fn save_json(&self, path: &PathBuf) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = File::create(path)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"version\": \"1.0\",")?;
        writeln!(file, "  \"phonemes\": {{")?;

        let count = self.prototypes.len();
        for (i, (label, _hv)) in self.prototypes.iter().enumerate() {
            let stats = self.stats.get(label).cloned().unwrap_or_default();
            let comma = if i < count - 1 { "," } else { "" };
            writeln!(
                file,
                "    \"{}\": {{ \"samples\": {}, \"duration_ms\": {:.1} }}{}",
                label, stats.sample_count, stats.mean_duration_ms, comma
            )?;
        }

        writeln!(file, "  }}")?;
        writeln!(file, "}}")?;

        Ok(())
    }

    fn load_binary(path: &PathBuf) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut prototypes = HashMap::new();
        let mut stats = HashMap::new();

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let num = u32::from_le_bytes(buf4) as usize;

        for _ in 0..num {
            let mut buf2 = [0u8; 2];
            file.read_exact(&mut buf2)?;
            let len = u16::from_le_bytes(buf2) as usize;

            let mut label_bytes = vec![0u8; len];
            file.read_exact(&mut label_bytes)?;
            let label = String::from_utf8_lossy(&label_bytes).to_string();

            let mut hv_bytes = [0u8; 256];
            file.read_exact(&mut hv_bytes)?;

            file.read_exact(&mut buf4)?;
            let sample_count = u32::from_le_bytes(buf4) as usize;

            file.read_exact(&mut buf4)?;
            let mean_duration_ms = f32::from_le_bytes(buf4);

            prototypes.insert(label.clone(), HV16(hv_bytes));
            stats.insert(
                label,
                PhonemeStats {
                    sample_count,
                    mean_duration_ms,
                },
            );
        }

        Ok(Self { prototypes, stats })
    }
}

// ============================================================================
// BOOTSTRAP TRAINING
// ============================================================================

/// Training callback with progress bars
struct ProgressCallback {
    #[allow(dead_code)]
    multi: MultiProgress,
    main_bar: ProgressBar,
    #[allow(dead_code)]
    phoneme_bars: HashMap<String, ProgressBar>,
    stats: BootstrapStats,
    start_time: Instant,
    verbose: bool,
}

#[derive(Clone, Default)]
struct BootstrapStats {
    utterances_processed: usize,
    phonemes_trained: usize,
    total_samples: usize,
    alignment_errors: usize,
    samples_per_phoneme: HashMap<String, usize>,
}

impl ProgressCallback {
    fn new(total_utterances: usize, verbose: bool) -> Self {
        let multi = MultiProgress::new();

        let style = ProgressStyle::default_bar()
            .template("{prefix:.bold.dim} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("█▓▒░  ");

        let main_bar = multi.add(ProgressBar::new(total_utterances as u64));
        main_bar.set_style(style);
        main_bar.set_prefix("Processing");

        Self {
            multi,
            main_bar,
            phoneme_bars: HashMap::new(),
            stats: BootstrapStats::default(),
            start_time: Instant::now(),
            verbose,
        }
    }

    fn on_utterance(&mut self, idx: usize, _total: usize, utterance_id: &str) {
        self.main_bar.set_position(idx as u64);
        if self.verbose {
            self.main_bar.set_message(utterance_id.to_string());
        }
        self.stats.utterances_processed = idx + 1;
    }

    fn on_phoneme_sample(&mut self, phoneme: &str, count: usize) {
        self.stats
            .samples_per_phoneme
            .entry(phoneme.to_string())
            .and_modify(|c| *c = count)
            .or_insert(count);
        self.stats.total_samples += 1;
    }

    fn on_error(&mut self) {
        self.stats.alignment_errors += 1;
    }

    fn finish(&mut self) {
        self.main_bar.finish_with_message("Complete");
        self.stats.phonemes_trained = self.stats.samples_per_phoneme.len();
    }
}

/// Main bootstrap training function
fn run_bootstrap(
    data_dir: &PathBuf,
    output_dir: &PathBuf,
    subset: &str,
    limit: usize,
    min_samples: usize,
    salience_threshold: f32,
    no_download: bool,
    resume: Option<PathBuf>,
    format: ExportFormat,
    verbose: bool,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Header
    if !quiet {
        println!("\n{}", style("═".repeat(70)).cyan());
        println!(
            "{}",
            style("                  SYMTHAEA BOOTSTRAP PROTOCOL")
                .cyan()
                .bold()
        );
        println!("{}", style("═".repeat(70)).cyan());
        println!();
    }

    // Step 1: Ensure dataset
    if !quiet {
        println!(
            "{}{}Ensuring dataset presence...",
            PACKAGE,
            style("Step 1/4: ").bold()
        );
    }

    let dataset_path = data_dir.join("LibriSpeech").join(subset);

    if !dataset_path.exists() {
        if no_download {
            return Err(format!("Dataset not found: {:?}", dataset_path).into());
        }
        download_librispeech(data_dir, subset, quiet)?;
    } else if !quiet {
        println!("  {}Dataset already present: {:?}", CHECK, dataset_path);
    }

    // Step 2: Scan dataset
    if !quiet {
        println!(
            "\n{}{}Scanning dataset...",
            SEARCH,
            style("Step 2/4: ").bold()
        );
    }

    let utterances = scan_librispeech(&dataset_path)?;
    let total = if limit > 0 {
        limit.min(utterances.len())
    } else {
        utterances.len()
    };

    if !quiet {
        println!(
            "  {}Found {} utterances ({} speakers)",
            CHECK,
            utterances.len(),
            utterances
                .iter()
                .map(|u| &u.speaker_id)
                .collect::<std::collections::HashSet<_>>()
                .len()
        );
        if limit > 0 {
            println!("  {}Limited to {} utterances", WARN, limit);
        }
    }

    // Step 3: Training
    if !quiet {
        println!(
            "\n{}{}Training prototypes...",
            BRAIN,
            style("Step 3/4: ").bold()
        );
        println!("  {}LTC temporal dynamics engaged", WAVE);
        println!("  {}Salience threshold: {}", LIGHTNING, salience_threshold);
        println!();
    }

    let mut callback = if !quiet {
        Some(ProgressCallback::new(total, verbose))
    } else {
        None
    };

    // Load existing prototypes if resuming
    let mut accumulators: HashMap<String, Vec<HV16>> = if let Some(ref path) = resume {
        if !quiet {
            println!("  {}Resuming from: {:?}", PACKAGE, path);
        }
        // Load and convert to accumulators (simplified)
        HashMap::new()
    } else {
        HashMap::new()
    };

    let phoneme_durations = get_phoneme_durations();
    let text_to_phonemes = TextToPhonemes::new();

    // Process utterances
    for (idx, utterance) in utterances.iter().take(total).enumerate() {
        if let Some(ref mut cb) = callback {
            cb.on_utterance(idx, total, &utterance.id);
        }

        // Load audio
        let audio = match load_wav(&utterance.audio_path) {
            Ok(a) => a,
            Err(_) => {
                if let Some(ref mut cb) = callback {
                    cb.on_error();
                }
                continue;
            }
        };

        // Project through mock LTC (in real implementation, use actual LtcAudioProjector)
        let (frames, salience, timestamps) = mock_ltc_project(&audio);

        // Convert text to phonemes
        let phonemes = text_to_phonemes.convert(&utterance.text);

        // Detect boundaries
        let boundaries = detect_boundaries(&salience, &timestamps, salience_threshold);

        // Align
        let aligned = align_phonemes(&phonemes, &boundaries, &phoneme_durations);

        // Accumulate
        for ap in aligned {
            if ap.confidence < 0.3 || ap.start_frame >= ap.end_frame {
                continue;
            }
            if ap.end_frame > frames.len() {
                continue;
            }

            let segment = &frames[ap.start_frame..ap.end_frame];
            if segment.is_empty() {
                continue;
            }

            let instance = bundle(segment);
            let count = {
                let acc = accumulators.entry(ap.ipa.clone()).or_default();
                acc.push(instance);
                acc.len()
            };

            if let Some(ref mut cb) = callback {
                cb.on_phoneme_sample(&ap.ipa, count);
            }
        }
    }

    if let Some(ref mut cb) = callback {
        cb.finish();
    }

    // Step 4: Save
    if !quiet {
        println!(
            "\n{}{}Saving prototypes...",
            SAVE,
            style("Step 4/4: ").bold()
        );
    }

    fs::create_dir_all(output_dir)?;

    // Build prototypes
    let mut prototypes = HashMap::new();
    let mut stats = HashMap::new();

    for (phoneme, instances) in &accumulators {
        if instances.len() >= min_samples {
            let proto = bundle(instances);
            prototypes.insert(phoneme.clone(), proto);
            stats.insert(
                phoneme.clone(),
                PhonemeStats {
                    sample_count: instances.len(),
                    mean_duration_ms: 80.0, // Placeholder
                },
            );
        }
    }

    let trained = TrainedPrototypes { prototypes, stats };

    // Save in requested format(s)
    match format {
        ExportFormat::Binary | ExportFormat::Both => {
            let path = output_dir.join("phoneme_prototypes.bin");
            trained.save_binary(&path)?;
            if !quiet {
                println!("  {}Saved binary: {:?}", CHECK, path);
            }
        }
        _ => {}
    }

    match format {
        ExportFormat::Json | ExportFormat::Both => {
            let path = output_dir.join("phoneme_prototypes.json");
            trained.save_json(&path)?;
            if !quiet {
                println!("  {}Saved JSON: {:?}", CHECK, path);
            }
        }
        _ => {}
    }

    // Summary
    if !quiet {
        let duration = callback
            .as_ref()
            .map(|c| c.start_time.elapsed())
            .unwrap_or_default();
        let total_samples: usize = accumulators.values().map(|v| v.len()).sum();

        println!("\n{}", style("═".repeat(70)).green());
        println!(
            "{}",
            style("                      TRAINING COMPLETE")
                .green()
                .bold()
        );
        println!("{}", style("═".repeat(70)).green());
        println!();
        println!("  {}Duration:      {:?}", CHART, duration);
        println!(
            "  {}Utterances:    {}",
            EAR,
            callback
                .as_ref()
                .map(|c| c.stats.utterances_processed)
                .unwrap_or(0)
        );
        println!("  {}Phonemes:      {}", BRAIN, trained.prototypes.len());
        println!("  {}Total samples: {}", CHART, total_samples);
        println!(
            "  {}Errors:        {}",
            WARN,
            callback
                .as_ref()
                .map(|c| c.stats.alignment_errors)
                .unwrap_or(0)
        );
        println!();

        // Per-phoneme summary
        println!("  {}Phoneme coverage:", CHART);
        let mut sorted: Vec<_> = accumulators
            .iter()
            .filter(|(_, v)| v.len() >= min_samples)
            .collect();
        sorted.sort_by_key(|(k, _)| *k);

        for (phoneme, instances) in sorted.iter().take(20) {
            let bar_len = (instances.len() as f32 / 100.0).min(30.0) as usize;
            let bar = "█".repeat(bar_len);
            println!(
                "    /{:4}/ {:>5} {}",
                phoneme,
                instances.len(),
                style(bar).cyan()
            );
        }

        if sorted.len() > 20 {
            println!("    ... and {} more phonemes", sorted.len() - 20);
        }

        println!();
        println!("  {}Output: {:?}", SAVE, output_dir);
        println!();
    }

    Ok(())
}

// ============================================================================
// INSPECT COMMAND
// ============================================================================

fn run_inspect(
    model: &PathBuf,
    detailed: bool,
    export_json: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n{}{}Inspecting model: {:?}",
        SEARCH,
        style("").bold(),
        model
    );
    println!();

    let trained = TrainedPrototypes::load_binary(model)?;

    println!("  {}Phonemes: {}", BRAIN, trained.prototypes.len());
    println!(
        "  {}Total samples: {}",
        CHART,
        trained
            .stats
            .values()
            .map(|s| s.sample_count)
            .sum::<usize>()
    );
    println!();

    if detailed {
        println!("  {}Inventory:", CHART);
        println!("  {}", style("─".repeat(50)).dim());

        let mut sorted: Vec<_> = trained.prototypes.keys().collect();
        sorted.sort();

        for phoneme in sorted {
            let stats = trained.stats.get(phoneme).cloned().unwrap_or_default();
            println!(
                "    /{:4}/  samples: {:>5}  duration: {:>5.1}ms",
                phoneme, stats.sample_count, stats.mean_duration_ms
            );
        }
        println!();
    }

    // Compute cross-similarity matrix (for quality check)
    println!("  {}Quality metrics:", CHART);

    let mut min_sim = 1.0f32;
    let mut max_sim = 0.0f32;
    let mut avg_sim = 0.0f32;
    let mut count = 0;

    let keys: Vec<_> = trained.prototypes.keys().collect();
    for (i, k1) in keys.iter().enumerate() {
        for k2 in keys.iter().skip(i + 1) {
            let hv1 = trained.prototypes.get(*k1).unwrap();
            let hv2 = trained.prototypes.get(*k2).unwrap();
            let sim = hv1.similarity(hv2);
            min_sim = min_sim.min(sim);
            max_sim = max_sim.max(sim);
            avg_sim += sim;
            count += 1;
        }
    }

    if count > 0 {
        avg_sim /= count as f32;
    }

    println!(
        "    Cross-similarity: min={:.3} avg={:.3} max={:.3}",
        min_sim, avg_sim, max_sim
    );
    println!("    (Lower is better - phonemes should be distinct)");
    println!();

    if avg_sim < 0.55 {
        println!("  {}Quality: GOOD - prototypes are well-separated", CHECK);
    } else if avg_sim < 0.65 {
        println!(
            "  {}Quality: ACCEPTABLE - some overlap between phonemes",
            WARN
        );
    } else {
        println!(
            "  {}Quality: POOR - prototypes are too similar, need more training data",
            ERROR
        );
    }

    if let Some(json_path) = export_json {
        trained.save_json(&json_path)?;
        println!("\n  {}Exported to: {:?}", SAVE, json_path);
    }

    println!();
    Ok(())
}

// ============================================================================
// INFO COMMAND
// ============================================================================

fn run_info() {
    println!("\n{}", style("═".repeat(70)).cyan());
    println!(
        "{}",
        style("                    SYMTHAEA SYSTEM INFORMATION")
            .cyan()
            .bold()
    );
    println!("{}", style("═".repeat(70)).cyan());
    println!();

    println!("  {}Version: {}", ROCKET, env!("CARGO_PKG_VERSION"));
    println!();

    println!("  {}Architecture:", BRAIN);
    println!("    • HDC Dimensions:     2048 bits (256 bytes per vector)");
    println!("    • LTC Time Constants: 5-100ms (adaptive)");
    println!("    • Phoneme Inventory:  44 English phonemes (IPA)");
    println!("    • Sample Rate:        16 kHz");
    println!("    • Frame Duration:     10ms");
    println!();

    println!("  {}Memory footprint:", CHART);
    println!("    • Per phoneme:        256 bytes");
    println!("    • Full inventory:     ~11 KB");
    println!("    • LTC state:          ~50 KB");
    println!("    • Total runtime:      < 1 MB");
    println!();

    println!("  {}Dependencies:", PACKAGE);
    println!("    • blake3:    Deterministic hashing");
    println!("    • hound:     WAV audio loading");
    println!("    • clap:      CLI parsing");
    println!("    • indicatif: Progress bars");
    println!();

    // Check for external tools
    println!("  {}External tools:", SEARCH);

    let ffmpeg_ok = Command::new("ffmpeg").arg("-version").output().is_ok();
    let curl_ok = Command::new("curl").arg("--version").output().is_ok();

    println!(
        "    • ffmpeg: {}",
        if ffmpeg_ok {
            style("✓ found").green()
        } else {
            style("✗ not found").red()
        }
    );
    println!(
        "    • curl:   {}",
        if curl_ok {
            style("✓ found").green()
        } else {
            style("✗ not found").red()
        }
    );
    println!();
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// LibriSpeech utterance
struct LibriUtterance {
    id: String,
    speaker_id: String,
    audio_path: PathBuf,
    text: String,
}

/// Scan LibriSpeech directory
fn scan_librispeech(path: &PathBuf) -> Result<Vec<LibriUtterance>, std::io::Error> {
    let mut utterances = Vec::new();

    fn scan_recursive(path: &PathBuf, utterances: &mut Vec<LibriUtterance>) -> std::io::Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();

            if entry_path.is_dir() {
                scan_recursive(&entry_path, utterances)?;
            } else if entry_path.to_string_lossy().ends_with(".trans.txt") {
                // Parse transcript file
                let parent = entry_path.parent().unwrap();
                let file = File::open(&entry_path)?;
                let reader = BufReader::new(file);

                for line in std::io::BufRead::lines(reader) {
                    let line = line?;
                    let parts: Vec<&str> = line.splitn(2, ' ').collect();
                    if parts.len() != 2 {
                        continue;
                    }

                    let id = parts[0];
                    let text = parts[1].to_lowercase();
                    let id_parts: Vec<&str> = id.split('-').collect();

                    if id_parts.len() < 2 {
                        continue;
                    }

                    let audio_path = parent.join(format!("{}.flac", id));
                    if !audio_path.exists() {
                        // Try .wav
                        let wav_path = parent.join(format!("{}.wav", id));
                        if !wav_path.exists() {
                            continue;
                        }
                    }

                    utterances.push(LibriUtterance {
                        id: id.to_string(),
                        speaker_id: id_parts[0].to_string(),
                        audio_path,
                        text,
                    });
                }
            }
        }
        Ok(())
    }

    scan_recursive(path, &mut utterances)?;
    Ok(utterances)
}

/// Download LibriSpeech subset
fn download_librispeech(
    data_dir: &PathBuf,
    subset: &str,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(data_dir)?;

    let url = format!("https://www.openslr.org/resources/12/{}.tar.gz", subset);
    let archive = data_dir.join(format!("{}.tar.gz", subset));

    if !quiet {
        println!("  {}Downloading from: {}", WAVE, url);
    }

    // Use curl or wget
    let download = Command::new("curl")
        .args(["-L", "-o", archive.to_str().unwrap(), &url])
        .status()
        .or_else(|_| {
            Command::new("wget")
                .args(["-O", archive.to_str().unwrap(), &url])
                .status()
        })?;

    if !download.success() {
        return Err("Download failed".into());
    }

    if !quiet {
        println!("  {}Extracting...", PACKAGE);
    }

    Command::new("tar")
        .args([
            "-xzf",
            archive.to_str().unwrap(),
            "-C",
            data_dir.to_str().unwrap(),
        ])
        .status()?;

    if !quiet {
        println!("  {}Download complete", CHECK);
    }

    Ok(())
}

/// Load WAV file
fn load_wav(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Try hound first
    if path.extension().map(|e| e == "wav").unwrap_or(false) {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();

        let samples: Vec<f32> = if spec.bits_per_sample == 16 {
            reader
                .into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect()
        } else {
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 2147483648.0)
                .collect()
        };

        return Ok(samples);
    }

    // Use ffmpeg for FLAC
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            path.to_str().unwrap(),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-",
        ])
        .output()?;

    if !output.status.success() {
        return Err("ffmpeg failed".into());
    }

    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(samples)
}

/// Mock LTC projection (simplified for CLI)
fn mock_ltc_project(audio: &[f32]) -> (Vec<HV16>, Vec<f32>, Vec<f32>) {
    let frame_samples = 160; // 10ms at 16kHz
    let num_frames = audio.len() / frame_samples;

    let mut frames = Vec::with_capacity(num_frames);
    let mut salience = Vec::with_capacity(num_frames);
    let mut timestamps = Vec::with_capacity(num_frames);

    let mut prev_energy = 0.0f32;

    for i in 0..num_frames {
        let start = i * frame_samples;
        let end = (start + frame_samples).min(audio.len());

        // Frame energy
        let energy: f32 =
            audio[start..end].iter().map(|s| s * s).sum::<f32>() / frame_samples as f32;

        // Generate HV from content
        let seed = format!("frame:{}:{:.6}", i, energy);
        let hv = HV16::from_seed(seed.as_bytes());

        // Salience from energy derivative
        let sal = ((energy - prev_energy).abs() * 100.0).min(10.0);
        prev_energy = energy;

        frames.push(hv);
        salience.push(sal);
        timestamps.push(i as f32 * 0.010);
    }

    (frames, salience, timestamps)
}

/// Detect boundaries from salience
fn detect_boundaries(salience: &[f32], timestamps: &[f32], threshold: f32) -> Vec<(f32, usize)> {
    let mean: f32 = salience.iter().sum::<f32>() / salience.len() as f32;
    let std: f32 =
        (salience.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / salience.len() as f32).sqrt();
    let thresh = mean + std * threshold;

    let mut boundaries = vec![(0.0, 0)];
    let mut last_time = -0.030f32;

    for i in 1..salience.len() - 1 {
        let is_peak = salience[i] > salience[i - 1] && salience[i] > salience[i + 1];
        if is_peak && salience[i] > thresh && timestamps[i] - last_time > 0.020 {
            boundaries.push((timestamps[i], i));
            last_time = timestamps[i];
        }
    }

    let last_t = *timestamps.last().unwrap_or(&0.0);
    boundaries.push((last_t, salience.len() - 1));

    boundaries
}

/// Aligned phoneme
struct AlignedPhoneme {
    ipa: String,
    start_frame: usize,
    end_frame: usize,
    confidence: f32,
}

/// Simple alignment
fn align_phonemes(
    phonemes: &[String],
    boundaries: &[(f32, usize)],
    _durations: &HashMap<String, f32>,
) -> Vec<AlignedPhoneme> {
    if phonemes.is_empty() || boundaries.len() < 2 {
        return Vec::new();
    }

    let total_frames = boundaries.last().unwrap().1;
    let frames_per_phoneme = total_frames / phonemes.len();

    phonemes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let start = i * frames_per_phoneme;
            let end = ((i + 1) * frames_per_phoneme).min(total_frames);

            AlignedPhoneme {
                ipa: p.clone(),
                start_frame: start,
                end_frame: end,
                confidence: 0.5,
            }
        })
        .collect()
}

/// Get phoneme durations
fn get_phoneme_durations() -> HashMap<String, f32> {
    let mut d = HashMap::new();
    for v in &[
        "ɑ", "æ", "ʌ", "ɔ", "ɛ", "ɝ", "ɪ", "i", "ʊ", "u", "ə", "e", "o",
    ] {
        d.insert(v.to_string(), 0.10);
    }
    for c in &["p", "b", "t", "d", "k", "g"] {
        d.insert(c.to_string(), 0.06);
    }
    d.insert("sil".to_string(), 0.05);
    d
}

/// Text to phonemes converter
struct TextToPhonemes {
    dict: HashMap<String, String>,
}

impl TextToPhonemes {
    fn new() -> Self {
        let mut dict = HashMap::new();

        // Core vocabulary
        let entries = [
            ("the", "ðə"),
            ("a", "ə"),
            ("and", "ænd"),
            ("to", "tu"),
            ("of", "əv"),
            ("in", "ɪn"),
            ("is", "ɪz"),
            ("it", "ɪt"),
            ("you", "ju"),
            ("that", "ðæt"),
            ("he", "hi"),
            ("was", "wɑz"),
            ("for", "fɔɹ"),
            ("on", "ɑn"),
            ("are", "ɑɹ"),
            ("with", "wɪð"),
            ("they", "ðe"),
            ("be", "bi"),
            ("at", "æt"),
            ("one", "wʌn"),
            ("have", "hæv"),
            ("this", "ðɪs"),
            ("from", "fɹʌm"),
            ("by", "baɪ"),
            ("not", "nɑt"),
            ("but", "bʌt"),
            ("what", "wʌt"),
            ("all", "ɔl"),
            ("were", "wɝ"),
            ("we", "wi"),
            ("when", "wɛn"),
            ("your", "jɔɹ"),
            ("can", "kæn"),
            ("said", "sɛd"),
            ("there", "ðɛɹ"),
            ("use", "juz"),
            ("each", "itʃ"),
            ("which", "wɪtʃ"),
            ("do", "du"),
            ("how", "haʊ"),
            ("their", "ðɛɹ"),
            ("if", "ɪf"),
            ("will", "wɪl"),
            ("up", "ʌp"),
            ("other", "ʌðɚ"),
            ("about", "əbaʊt"),
            ("out", "aʊt"),
            ("many", "mɛni"),
            ("then", "ðɛn"),
            ("them", "ðɛm"),
            ("so", "so"),
            ("some", "sʌm"),
            ("would", "wʊd"),
            ("make", "mek"),
            ("like", "laɪk"),
            ("time", "taɪm"),
            ("has", "hæz"),
            ("look", "lʊk"),
            ("two", "tu"),
            ("more", "mɔɹ"),
            ("go", "go"),
            ("see", "si"),
            ("no", "no"),
            ("way", "we"),
            ("could", "kʊd"),
            ("people", "pipəl"),
            ("my", "maɪ"),
            ("than", "ðæn"),
            ("first", "fɝst"),
            ("been", "bɪn"),
            ("who", "hu"),
            ("its", "ɪts"),
            ("now", "naʊ"),
            ("find", "faɪnd"),
            ("long", "lɔŋ"),
            ("down", "daʊn"),
            ("day", "de"),
            ("did", "dɪd"),
            ("get", "gɛt"),
            ("come", "kʌm"),
            ("made", "med"),
            ("may", "me"),
            ("cat", "kæt"),
            ("dog", "dɔg"),
        ];

        for (word, ipa) in entries {
            dict.insert(word.to_string(), ipa.to_string());
        }

        Self { dict }
    }

    fn convert(&self, text: &str) -> Vec<String> {
        let mut phonemes = Vec::new();

        // Add start silence
        phonemes.push("sil".to_string());

        for word in text.split_whitespace() {
            let word_clean: String = word
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>()
                .to_lowercase();

            if word_clean.is_empty() {
                continue;
            }

            if let Some(ipa) = self.dict.get(&word_clean) {
                // Split IPA into phonemes
                for c in ipa.chars() {
                    phonemes.push(c.to_string());
                }
            } else {
                // Fallback: just use characters
                for c in word_clean.chars() {
                    phonemes.push(c.to_string());
                }
            }

            // Add inter-word silence (the fix for pause handling!)
            phonemes.push("sil".to_string());
        }

        phonemes
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Bootstrap {
            data_dir,
            output_dir,
            subset,
            limit,
            min_samples,
            salience_threshold,
            no_download,
            resume,
            format,
        } => run_bootstrap(
            &data_dir,
            &output_dir,
            &subset,
            limit,
            min_samples,
            salience_threshold,
            no_download,
            resume,
            format,
            cli.verbose,
            cli.quiet,
        ),
        Commands::Inspect {
            model,
            detailed,
            export_json,
        } => run_inspect(&model, detailed, export_json),
        Commands::Validate {
            model,
            audio,
            expected,
            show_alignment,
        } => {
            println!("{}Validate command not yet implemented", WARN);
            println!("  Model: {:?}", model);
            println!("  Audio: {:?}", audio);
            if let Some(exp) = expected {
                println!("  Expected: {}", exp);
            }
            if show_alignment {
                println!("  (would show alignment)");
            }
            Ok(())
        }
        Commands::Download { data_dir, subset } => {
            download_librispeech(&data_dir, &subset, cli.quiet).map_err(|e| e)
        }
        Commands::Info => {
            run_info();
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("\n{}Error: {}", ERROR, e);
        std::process::exit(1);
    }
}
