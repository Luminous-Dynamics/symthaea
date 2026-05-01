// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extract HDC training pairs from the MAESTRO dataset.
//!
//! Iterates over all 1,276 MAESTRO performances, loads each (MIDI, WAV) pair,
//! reconstructs MusicalState frames from MIDI and mel spectrograms from WAV,
//! and emits binary training pair files for the HDC audio decoder.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --bin extract_maestro_pairs -- \
//!     /opt/datasets/maestro/maestro-v3.0.0 \
//!     /opt/datasets/maestro/training_pairs
//! ```

use std::path::{Path, PathBuf};
use symthaea_muse::midi_loader::load_midi;
use symthaea_muse::training_pairs::{build_pairs, save_pairs, PairConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: extract_maestro_pairs <maestro_dir> <output_dir>");
        eprintln!("  maestro_dir: path to maestro-v3.0.0/");
        eprintln!("  output_dir: where to save .bin training pair files");
        std::process::exit(1);
    }

    let maestro_dir = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&output_dir).expect("create output dir");

    println!("═══ MAESTRO → HDC Training Pair Extractor ═══");
    println!("  Source:      {}", maestro_dir.display());
    println!("  Destination: {}", output_dir.display());

    // Find all MIDI files (each has a matching WAV)
    let midi_files = find_midi_files(&maestro_dir);
    println!("  Found {} MIDI files", midi_files.len());

    let config = PairConfig::default();
    let mut total_pairs = 0usize;
    let mut processed = 0usize;
    let mut errors = 0usize;

    for (idx, midi_path) in midi_files.iter().enumerate() {
        let wav_path = midi_path.with_extension("wav");
        if !wav_path.exists() {
            eprintln!("  SKIP: no WAV for {}", midi_path.display());
            errors += 1;
            continue;
        }

        // Output filename: keep year subdirectory + base name
        let stem = midi_path.file_stem().unwrap().to_string_lossy();
        let year = midi_path
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let out_subdir = output_dir.join(&year);
        std::fs::create_dir_all(&out_subdir).ok();
        let out_path = out_subdir.join(format!("{}.pairs.bin", stem));

        // Skip if already extracted
        if out_path.exists() {
            processed += 1;
            continue;
        }

        // Load MIDI notes
        let notes = match load_midi(midi_path) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("  MIDI load error {}: {:?}", midi_path.display(), e);
                errors += 1;
                continue;
            }
        };

        // Load WAV samples
        let samples = match load_wav_mono(&wav_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  WAV load error {}: {}", wav_path.display(), e);
                errors += 1;
                continue;
            }
        };

        // Build training pairs
        let pairs = build_pairs(&notes, &samples, &config);
        if pairs.is_empty() {
            errors += 1;
            continue;
        }

        // Save
        if let Err(e) = save_pairs(&pairs, &out_path) {
            eprintln!("  Save error {}: {}", out_path.display(), e);
            errors += 1;
            continue;
        }

        total_pairs += pairs.len();
        processed += 1;

        if (idx + 1) % 10 == 0 || idx == midi_files.len() - 1 {
            println!(
                "  [{}/{}] processed, {} pairs total, {} errors",
                processed,
                midi_files.len(),
                total_pairs,
                errors
            );
        }
    }

    println!("\n═══ Extraction Complete ═══");
    println!("  Processed:   {} performances", processed);
    println!("  Errors:      {}", errors);
    println!("  Total pairs: {}", total_pairs);
    println!("  Output:      {}", output_dir.display());
}

/// Recursively find all `.midi` files under the given directory.
fn find_midi_files(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(find_midi_files(&path));
            } else if path.extension().and_then(|e| e.to_str()) == Some("midi") {
                results.push(path);
            }
        }
    }
    results
}

/// Load a WAV file and return mono f32 samples.
fn load_wav_mono(path: &Path) -> Result<Vec<f32>, String> {
    let reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bps = spec.bits_per_sample as f32;
            let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / scale)
                .collect::<Vec<f32>>()
                .into_iter()
                .map(|s| {
                    if bps == 24.0 {
                        s / 256.0 // hound returns i32 with 24-bit data in upper bits
                    } else {
                        s
                    }
                })
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    if channels == 1 {
        Ok(samples)
    } else {
        // Downmix to mono by averaging
        Ok(samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect())
    }
}
