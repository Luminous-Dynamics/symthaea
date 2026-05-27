// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bioacoustics Discovery Tool
//!
//! Download and analyze animal vocalizations (whales, birds, etc.)
//! to discover acoustic units without supervision.
//!
//! ```bash
//! # Download sample whale audio
//! symthaea-bioacoustics download --species whale --output data/whales
//!
//! # Run discovery on audio files
//! symthaea-bioacoustics discover --input data/whales --output results/whale_units.json
//!
//! # Generate synthetic test audio
//! symthaea-bioacoustics generate --type whale --duration 30 --output test_whale.wav
//! ```

use clap::{Parser, Subcommand};
use console::{Emoji, style};
use indicatif::{ProgressBar, ProgressStyle};
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use symthaea_stt::{BatchConfig, BatchDiscovery, DiscoveryConfig, batch::find_audio_files};

#[allow(dead_code)]
static DOWNLOAD: Emoji<'_, '_> = Emoji("📥 ", "");
static ANALYZE: Emoji<'_, '_> = Emoji("🔬 ", "");
#[allow(dead_code)]
static WHALE: Emoji<'_, '_> = Emoji("🐋 ", "[WHALE] ");
#[allow(dead_code)]
static BIRD: Emoji<'_, '_> = Emoji("🐦 ", "[BIRD] ");
static SUCCESS: Emoji<'_, '_> = Emoji("✅ ", "[OK] ");

#[derive(Parser)]
#[command(name = "symthaea-bioacoustics")]
#[command(about = "Bioacoustics discovery for Symthaea STT")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download animal audio from free sources
    Download {
        /// Species type (whale, bird, dolphin)
        #[arg(short, long, default_value = "whale")]
        species: String,

        /// Output directory
        #[arg(short, long, default_value = "data/bioacoustics")]
        output: PathBuf,

        /// Number of samples to download
        #[arg(short, long, default_value = "5")]
        count: usize,
    },

    /// Run acoustic unit discovery on audio files
    Discover {
        /// Input directory or file
        #[arg(short, long)]
        input: PathBuf,

        /// Output JSON file for results
        #[arg(short, long, default_value = "discovery_results.json")]
        output: PathBuf,

        /// Similarity threshold for clustering
        #[arg(long, default_value = "0.7")]
        threshold: f32,

        /// Minimum instances for stable unit
        #[arg(long, default_value = "3")]
        min_instances: usize,

        /// Number of worker threads
        #[arg(short, long, default_value = "0")]
        workers: usize,
    },

    /// Generate synthetic animal audio for testing
    Generate {
        /// Type of vocalization (whale, bird, dolphin)
        #[arg(short = 't', long, default_value = "whale")]
        vocalization_type: String,

        /// Duration in seconds
        #[arg(short, long, default_value = "10")]
        duration: f32,

        /// Output WAV file
        #[arg(short, long, default_value = "synthetic_animal.wav")]
        output: PathBuf,

        /// Sample rate
        #[arg(long, default_value = "16000")]
        sample_rate: u32,
    },

    /// Show info about discovered units
    Info {
        /// Results JSON file
        #[arg(short, long)]
        results: PathBuf,
    },

    /// List available species and sources
    List,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Download {
            species,
            output,
            count,
        } => {
            if let Err(e) = download_audio(&species, &output, count) {
                eprintln!("Download failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Discover {
            input,
            output,
            threshold,
            min_instances,
            workers,
        } => {
            if let Err(e) = run_discovery(&input, &output, threshold, min_instances, workers) {
                eprintln!("Discovery failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Generate {
            vocalization_type,
            duration,
            output,
            sample_rate,
        } => {
            if let Err(e) = generate_audio(&vocalization_type, duration, &output, sample_rate) {
                eprintln!("Generation failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Info { results } => {
            if let Err(e) = show_info(&results) {
                eprintln!("Info failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::List => {
            list_sources();
        }
    }
}

/// Species info
struct SpeciesInfo {
    name: &'static str,
    emoji: &'static str,
    description: &'static str,
    freq_range: (f32, f32),
    sources: &'static [&'static str],
}

const SPECIES: &[SpeciesInfo] = &[
    SpeciesInfo {
        name: "whale",
        emoji: "🐋",
        description: "Humpback, Blue, and other baleen whale songs",
        freq_range: (20.0, 4000.0),
        sources: &[
            "NOAA Fisheries Acoustic Library",
            "Watkins Marine Mammal Sound Database (WHOI)",
            "MBARI Soundscape Listening Room",
        ],
    },
    SpeciesInfo {
        name: "bird",
        emoji: "🐦",
        description: "Songbirds, raptors, and other avian vocalizations",
        freq_range: (1000.0, 8000.0),
        sources: &[
            "Xeno-canto (xeno-canto.org)",
            "Macaulay Library (Cornell Lab)",
            "British Library Sound Archive",
        ],
    },
    SpeciesInfo {
        name: "dolphin",
        emoji: "🐬",
        description: "Dolphin clicks and whistles",
        freq_range: (2000.0, 20000.0),
        sources: &[
            "Watkins Marine Mammal Sound Database",
            "Wild Dolphin Project",
        ],
    },
    SpeciesInfo {
        name: "bat",
        emoji: "🦇",
        description: "Echolocation calls (requires high sample rate)",
        freq_range: (20000.0, 100000.0),
        sources: &["Bat Library (batlib.org)", "ChiroVox database"],
    },
];

fn list_sources() {
    println!(
        "\n{}Available Species and Data Sources",
        style("").bold().cyan()
    );
    println!("{}", "=".repeat(60));

    for species in SPECIES {
        println!(
            "\n{} {} - {}",
            species.emoji,
            style(species.name).green().bold(),
            species.description
        );
        println!(
            "   Frequency range: {} - {} Hz",
            species.freq_range.0, species.freq_range.1
        );
        println!("   Data sources:");
        for source in species.sources {
            println!("     - {}", source);
        }
    }

    println!(
        "\n{}Note: For actual downloads, you may need to manually download",
        style("").dim()
    );
    println!(
        "{}from these sources as they require authentication or have",
        style("").dim()
    );
    println!(
        "{}usage restrictions. Use 'generate' to create synthetic test audio.",
        style("").dim()
    );
}

fn download_audio(species: &str, output: &Path, count: usize) -> Result<(), String> {
    let species_info = SPECIES.iter().find(|s| s.name == species).ok_or_else(|| {
        format!(
            "Unknown species: {}. Run 'list' for available species.",
            species
        )
    })?;

    println!(
        "\n{}{} Audio Download",
        style("").bold().cyan(),
        species_info.emoji
    );
    println!("{}", "=".repeat(50));
    println!("Species: {}", style(species).green());
    println!("Output:  {}", output.display());
    println!("Count:   {}", count);
    println!();

    // Create output directory
    fs::create_dir_all(output).map_err(|e| format!("Failed to create output directory: {}", e))?;

    println!(
        "{}Due to licensing restrictions, automatic downloads are not available.",
        style("Note: ").yellow()
    );
    println!(
        "{}Instead, generating {} synthetic {} calls for testing...",
        style("").dim(),
        count,
        species
    );
    println!();

    // Generate synthetic audio as placeholder
    let progress = ProgressBar::new(count as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap(),
    );

    for i in 0..count {
        let filename = output.join(format!("{}_{:03}.wav", species, i + 1));

        // Generate synthetic audio
        let duration = 5.0 + (i as f32 * 0.5); // Vary duration
        generate_audio(species, duration, &filename, 16000)?;

        progress.inc(1);
    }

    progress.finish_with_message("Done!");

    println!(
        "\n{}Generated {} synthetic {} audio files",
        SUCCESS, count, species
    );
    println!("Location: {}", output.display());
    println!();
    println!("{}For real data, please visit:", style("Tip: ").cyan());
    for source in species_info.sources {
        println!("  - {}", source);
    }

    Ok(())
}

fn generate_audio(
    vocalization_type: &str,
    duration: f32,
    output: &Path,
    sample_rate: u32,
) -> Result<(), String> {
    let samples = match vocalization_type {
        "whale" => generate_whale_calls(duration, sample_rate),
        "bird" => generate_bird_song(duration, sample_rate),
        "dolphin" => generate_dolphin_clicks(duration, sample_rate),
        "bat" => generate_bat_calls(duration, sample_rate),
        _ => return Err(format!("Unknown vocalization type: {}", vocalization_type)),
    };

    // Write WAV file
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output, spec)
        .map_err(|e| format!("Failed to create WAV file: {}", e))?;

    for sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(s)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

    Ok(())
}

/// Generate synthetic whale calls
fn generate_whale_calls(duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    let mut samples = vec![0.0; num_samples];
    let mut rng_state = 42u64;

    // Helper for pseudo-random
    let mut rand = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    // Generate several whale calls
    let num_calls = (duration / 3.0).max(1.0) as usize;

    for call_idx in 0..num_calls {
        let call_start = (call_idx as f32 * 3.0 + rand() * 0.5) * sample_rate as f32;
        let call_duration = (1.5 + rand() * 2.0) * sample_rate as f32;

        // Base frequency sweeps (characteristic of humpback songs)
        let f_start = 200.0 + rand() * 400.0; // 200-600 Hz
        let f_end = f_start * (0.5 + rand() * 1.0); // Frequency modulation

        for i in 0..(call_duration as usize) {
            let sample_idx = call_start as usize + i;
            if sample_idx >= num_samples {
                break;
            }

            let t = i as f32 / call_duration;
            let freq = f_start + (f_end - f_start) * t;

            // Envelope (whale calls have slow attack/release)
            let envelope = (t * PI).sin().powf(0.5);

            // Main tone with harmonics
            let phase = 2.0 * PI * freq * i as f32 / sample_rate as f32;
            let mut signal = phase.sin() * 0.6;
            signal += (phase * 2.0).sin() * 0.2; // Second harmonic
            signal += (phase * 3.0).sin() * 0.1; // Third harmonic

            // Add frequency modulation (warbling)
            let warble = (2.0 * PI * 2.0 * i as f32 / sample_rate as f32).sin() * 10.0;
            let mod_phase = 2.0 * PI * (freq + warble) * i as f32 / sample_rate as f32;
            signal += mod_phase.sin() * 0.1;

            samples[sample_idx] += signal * envelope * 0.4;
        }
    }

    // Add underwater ambience
    for i in 0..num_samples {
        let noise = (rand() - 0.5) * 0.02;
        samples[i] += noise;

        // Low frequency rumble
        let rumble = (2.0 * PI * 30.0 * i as f32 / sample_rate as f32).sin() * 0.01;
        samples[i] += rumble;
    }

    // Normalize
    let max_val = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if max_val > 0.0 {
        for s in &mut samples {
            *s /= max_val;
        }
    }

    samples
}

/// Generate synthetic bird song
fn generate_bird_song(duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    let mut samples = vec![0.0; num_samples];
    let mut rng_state = 123u64;

    let mut rand = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    // Generate bird calls (short chirps with complex modulation)
    let num_phrases = (duration / 1.5).max(1.0) as usize;

    for phrase_idx in 0..num_phrases {
        let phrase_start = phrase_idx as f32 * 1.5 + rand() * 0.3;
        let num_notes = (3.0 + rand() * 5.0) as usize;

        for note_idx in 0..num_notes {
            let note_start = (phrase_start + note_idx as f32 * 0.1) * sample_rate as f32;
            let note_duration = (0.05 + rand() * 0.15) * sample_rate as f32;

            let base_freq = 2000.0 + rand() * 4000.0; // 2-6 kHz
            let freq_sweep = (rand() - 0.5) * 2000.0; // Frequency sweep

            for i in 0..(note_duration as usize) {
                let sample_idx = note_start as usize + i;
                if sample_idx >= num_samples {
                    break;
                }

                let t = i as f32 / note_duration;
                let freq = base_freq + freq_sweep * t;

                // Quick attack, longer decay envelope
                let envelope = if t < 0.1 {
                    t / 0.1
                } else {
                    (1.0 - t).powf(0.5)
                };

                let phase = 2.0 * PI * freq * i as f32 / sample_rate as f32;
                let signal = phase.sin() + (phase * 2.0).sin() * 0.3;

                samples[sample_idx] += signal * envelope * 0.3;
            }
        }

        // Add a trill sometimes
        if rand() > 0.5 {
            let trill_start = (phrase_start + 0.5) * sample_rate as f32;
            let trill_freq = 3000.0 + rand() * 2000.0;
            let trill_rate = 30.0 + rand() * 20.0; // Modulation rate

            for i in 0..((0.3 * sample_rate as f32) as usize) {
                let sample_idx = trill_start as usize + i;
                if sample_idx >= num_samples {
                    break;
                }

                let t = i as f32 / sample_rate as f32;
                let envelope = (PI * t / 0.3).sin();
                let am = (2.0 * PI * trill_rate * t).sin() * 0.5 + 0.5;
                let phase = 2.0 * PI * trill_freq * t;

                samples[sample_idx] += phase.sin() * am * envelope * 0.2;
            }
        }
    }

    // Normalize
    let max_val = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if max_val > 0.0 {
        for s in &mut samples {
            *s /= max_val;
        }
    }

    samples
}

/// Generate synthetic dolphin clicks and whistles
fn generate_dolphin_clicks(duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    let mut samples = vec![0.0; num_samples];
    let mut rng_state = 456u64;

    let mut rand = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    // Click trains
    let num_trains = (duration / 2.0).max(1.0) as usize;

    for train_idx in 0..num_trains {
        let train_start = train_idx as f32 * 2.0 + rand() * 0.5;
        let num_clicks = (5.0 + rand() * 15.0) as usize;
        let click_interval = 0.02 + rand() * 0.05;

        for click_idx in 0..num_clicks {
            let click_start =
                (train_start + click_idx as f32 * click_interval) * sample_rate as f32;
            let click_duration = 0.001 * sample_rate as f32; // 1ms click

            for i in 0..(click_duration as usize) {
                let sample_idx = click_start as usize + i;
                if sample_idx >= num_samples {
                    break;
                }

                let t = i as f32 / click_duration;
                let envelope = (PI * t).sin();

                // Broadband click
                let freq = 5000.0 + rand() * 10000.0;
                let phase = 2.0 * PI * freq * i as f32 / sample_rate as f32;

                samples[sample_idx] += phase.sin() * envelope * 0.4;
            }
        }
    }

    // Signature whistles
    let num_whistles = (duration / 4.0).max(1.0) as usize;

    for whistle_idx in 0..num_whistles {
        let whistle_start = (whistle_idx as f32 * 4.0 + 1.0 + rand() * 0.5) * sample_rate as f32;
        let whistle_duration = (0.5 + rand() * 1.0) * sample_rate as f32;

        let f_start = 4000.0 + rand() * 4000.0;
        let f_peak = f_start + rand() * 4000.0;
        let f_end = f_start + rand() * 2000.0;

        for i in 0..(whistle_duration as usize) {
            let sample_idx = whistle_start as usize + i;
            if sample_idx >= num_samples {
                break;
            }

            let t = i as f32 / whistle_duration;

            // Characteristic up-down frequency contour
            let freq = if t < 0.5 {
                f_start + (f_peak - f_start) * (t * 2.0)
            } else {
                f_peak + (f_end - f_peak) * ((t - 0.5) * 2.0)
            };

            let envelope = (PI * t).sin();
            let phase = 2.0 * PI * freq * i as f32 / sample_rate as f32;

            samples[sample_idx] += phase.sin() * envelope * 0.3;
        }
    }

    // Normalize
    let max_val = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if max_val > 0.0 {
        for s in &mut samples {
            *s /= max_val;
        }
    }

    samples
}

/// Generate synthetic bat echolocation calls
fn generate_bat_calls(duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    let mut samples = vec![0.0; num_samples];
    let mut rng_state = 789u64;

    let mut rand = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    // Note: Real bat calls are ultrasonic (20-100 kHz)
    // We simulate them at audible frequencies for 16kHz sample rate
    let freq_scale = 0.25; // Scale down to audible range

    let num_calls = (duration / 0.5).max(1.0) as usize;

    for call_idx in 0..num_calls {
        let call_start = call_idx as f32 * 0.5 + rand() * 0.2;
        let call_duration = (0.005 + rand() * 0.015) * sample_rate as f32;

        // FM sweep (characteristic of many bat species)
        let f_start = (80000.0 * freq_scale) + rand() * (20000.0 * freq_scale);
        let f_end = (30000.0 * freq_scale) + rand() * (10000.0 * freq_scale);

        for i in 0..(call_duration as usize) {
            let sample_idx = (call_start * sample_rate as f32) as usize + i;
            if sample_idx >= num_samples {
                break;
            }

            let t = i as f32 / call_duration;
            let freq = f_start + (f_end - f_start) * t;

            // Sharp envelope
            let envelope = (PI * t).sin().powf(0.3);

            let phase = 2.0 * PI * freq * i as f32 / sample_rate as f32;
            samples[sample_idx] += phase.sin() * envelope * 0.5;
        }
    }

    // Normalize
    let max_val = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if max_val > 0.0 {
        for s in &mut samples {
            *s /= max_val;
        }
    }

    samples
}

fn run_discovery(
    input: &Path,
    output: &Path,
    threshold: f32,
    min_instances: usize,
    workers: usize,
) -> Result<(), String> {
    println!("\n{}Bioacoustics Unit Discovery", style("").bold().cyan());
    println!("{}", "=".repeat(50));
    println!("Input:      {}", input.display());
    println!("Output:     {}", output.display());
    println!("Threshold:  {}", threshold);
    println!("Min inst:   {}", min_instances);
    println!();

    // Find audio files
    let files = if input.is_file() {
        vec![input.to_path_buf()]
    } else {
        find_audio_files(input, &["wav", "flac"])
    };

    if files.is_empty() {
        return Err("No audio files found".to_string());
    }

    println!("Found {} audio files", style(files.len()).green());
    println!();

    // Configure discovery
    let batch_config = BatchConfig {
        num_workers: workers,
        shuffle: false,
        continue_on_error: true,
        progress_interval: 1,
        ..Default::default()
    };

    let discovery_config = DiscoveryConfig {
        similarity_threshold: threshold,
        min_instances,
        salience_threshold: 0.3, // Lower for better boundary detection
        min_segment_ms: 10.0,    // Allow short segments
        max_segment_ms: 5000.0,  // Allow longer segments for whale calls
        ..Default::default()
    };

    // Run discovery
    println!("{}Running acoustic unit discovery...", ANALYZE);

    let progress = ProgressBar::new(files.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap(),
    );

    let progress_clone = progress.clone();
    let callback = Box::new(move |current: usize, _total: usize, _path: &str| {
        progress_clone.set_position(current as u64);
    });

    let discovery = BatchDiscovery::new(batch_config, discovery_config);
    let (result, stats) = discovery.discover(&files, Some(callback));

    progress.finish_with_message("Discovery complete!");

    // Save results
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    let json = result.to_json();

    let mut file =
        File::create(output).map_err(|e| format!("Failed to create output file: {}", e))?;
    file.write_all(json.as_bytes())
        .map_err(|e| format!("Failed to write results: {}", e))?;

    // Print summary
    println!("\n{}Discovery Results", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Files processed:    {}", stats.files_processed);
    println!("Audio duration:     {:.1} seconds", stats.total_audio_sec);
    println!("Processing time:    {:.1} seconds", stats.total_time_sec);
    println!("Realtime factor:    {:.1}x", stats.realtime_factor);
    println!();
    println!("{}Discovered Units", style("").bold());
    println!("{}", "-".repeat(40));
    println!("Total units:        {}", result.units.len());
    println!("Stable units:       {}", result.stats.num_stable_units);

    // Show unit details
    if !result.units.is_empty() {
        println!("\n{}Unit Details:", style("").bold());
        for (i, unit) in result.units.iter().take(10).enumerate() {
            println!(
                "  {:>3}. {} - {} instances, {:.1}ms avg",
                i + 1,
                style(&unit.id).cyan(),
                unit.instance_count,
                unit.mean_duration_ms
            );
        }

        if result.units.len() > 10 {
            println!("  ... and {} more units", result.units.len() - 10);
        }
    }

    println!("\n{}Results saved to: {}", SUCCESS, output.display());

    Ok(())
}

fn show_info(results_path: &Path) -> Result<(), String> {
    // Read the JSON file
    let content =
        fs::read_to_string(results_path).map_err(|e| format!("Failed to read results: {}", e))?;

    // Parse as generic JSON since DiscoveryResult doesn't implement Deserialize
    let json: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse results: {}", e))?;

    println!("\n{}Discovery Results Info", style("").bold().cyan());
    println!("{}", "=".repeat(50));
    println!("File: {}", results_path.display());
    println!();

    // Extract stats
    if let Some(stats) = json.get("stats") {
        println!("{}Statistics", style("").bold());
        println!("{}", "-".repeat(40));

        if let Some(num_units) = stats.get("num_units").and_then(|v| v.as_u64()) {
            println!("Total units:        {}", num_units);
        }
        if let Some(num_stable) = stats.get("num_stable_units").and_then(|v| v.as_u64()) {
            println!("Stable units:       {}", num_stable);
        }
        if let Some(duration) = stats.get("total_duration_sec").and_then(|v| v.as_f64()) {
            println!("Total duration:     {:.1} seconds", duration);
        }
        if let Some(entropy) = stats.get("unit_entropy").and_then(|v| v.as_f64()) {
            println!("Unit entropy:       {:.3}", entropy);
        }
    }

    // Extract units
    if let Some(units) = json.get("units").and_then(|v| v.as_array()) {
        println!("\n{}All Discovered Units:", style("").bold());
        println!("{}", "-".repeat(60));
        println!("{:>10} {:>12} {:>12}", "ID", "Instances", "Duration");

        for unit in units {
            let id = unit.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let instances = unit.get("instances").and_then(|v| v.as_u64()).unwrap_or(0);
            let duration = unit
                .get("mean_duration_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            println!("{:>10} {:>12} {:>9.1} ms", id, instances, duration);
        }

        // Calculate summary stats
        let total_instances: u64 = units
            .iter()
            .filter_map(|u| u.get("instances").and_then(|v| v.as_u64()))
            .sum();
        let durations: Vec<f64> = units
            .iter()
            .filter_map(|u| u.get("mean_duration_ms").and_then(|v| v.as_f64()))
            .collect();

        if !durations.is_empty() {
            let mean_dur = durations.iter().sum::<f64>() / durations.len() as f64;
            println!("\nMean duration:      {:.1} ms", mean_dur);
        }
        println!("Total instances:    {}", total_instances);
    }

    // Show segments summary
    if let Some(segments) = json.get("segments").and_then(|v| v.as_array()) {
        println!("\nTotal segments:     {}", segments.len());
    }

    Ok(())
}
