// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Analyze WAV files using the external validation systems.
//!
//! Reports AudioQualityScore metrics (RMS, peak, crest, flatness, HNR,
//! dynamic range variation, clipping) for each WAV in a directory.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --example analyze_wavs -- audio_output/
//! ```

use std::env;
use std::path::PathBuf;
use symthaea_muse::creative_bench::AudioQualityScore;

fn main() {
    let args: Vec<String> = env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "audio_output".to_string());

    println!("═══ WAV Analysis: {} ═══\n", dir);
    println!(
        "{:<40} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>5} {:>6} {}",
        "File", "RMS dB", "Peak dB", "Crest", "Flat", "DynVar", "HNR", "Clip", "Score", "Issues"
    );
    println!("{}", "─".repeat(125));

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    files.sort();

    let mut all_scores: Vec<(String, AudioQualityScore)> = Vec::new();

    for path in &files {
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        // Read WAV
        let reader = match hound::WavReader::open(path) {
            Ok(r) => r,
            Err(e) => {
                println!("{:<40} ERROR: {}", filename, e);
                continue;
            }
        };

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels;

        // Decode samples to stereo f32
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
        };

        // Convert to stereo pairs
        let stereo: Vec<[f32; 2]> = if channels == 2 {
            samples.chunks(2).map(|c| [c[0], c.get(1).copied().unwrap_or(0.0)]).collect()
        } else {
            samples.iter().map(|&s| [s, s]).collect()
        };

        let score = AudioQualityScore::evaluate(&stereo, sample_rate);

        // Identify issues
        let mut issues = Vec::new();
        if score.clipped_samples > 0 {
            issues.push(format!("CLIP({})", score.clipped_samples));
        }
        if score.rms_db > -6.0 {
            issues.push("TOO_LOUD".to_string());
        }
        if score.rms_db < -30.0 {
            issues.push("TOO_QUIET".to_string());
        }
        if score.spectral_flatness > 0.5 {
            issues.push("NOISY".to_string()); // spectral flatness high = noise-like (static)
        }
        if score.crest_db < 6.0 {
            issues.push("NO_DYNAMICS".to_string());
        }
        if score.crest_db > 20.0 {
            issues.push("SPIKY".to_string());
        }
        if score.harmonic_to_noise_db < 0.0 {
            issues.push("HARSH".to_string());
        }
        if score.silence_ratio > 0.5 {
            issues.push("SILENT".to_string());
        }
        if score.dynamic_range_variation_db < 2.0 {
            issues.push("FLAT_DYN".to_string());
        }

        let issues_str = if issues.is_empty() {
            "-".to_string()
        } else {
            issues.join(",")
        };

        println!(
            "{:<40} {:>7.1} {:>7.1} {:>7.1} {:>7.3} {:>7.1} {:>7.1} {:>5} {:>6.3} {}",
            filename,
            score.rms_db,
            score.peak_db,
            score.crest_db,
            score.spectral_flatness,
            score.dynamic_range_variation_db,
            score.harmonic_to_noise_db,
            score.clipped_samples,
            score.composite,
            issues_str
        );

        all_scores.push((filename, score));
    }

    // Summary: which files have issues?
    println!("\n═══ Summary ═══");
    let with_issues: Vec<_> = all_scores
        .iter()
        .filter(|(_, s)| {
            s.clipped_samples > 0
                || s.spectral_flatness > 0.5
                || s.harmonic_to_noise_db < 0.0
                || s.crest_db < 6.0
        })
        .collect();

    if with_issues.is_empty() {
        println!("  No critical issues found.");
    } else {
        println!("  Files with critical issues (clip, noise, harsh, no-dynamics):");
        for (name, score) in &with_issues {
            println!(
                "    {}: clip={}, flat={:.3}, hnr={:.1}dB, crest={:.1}dB",
                name, score.clipped_samples, score.spectral_flatness,
                score.harmonic_to_noise_db, score.crest_db
            );
        }
    }

    // Best and worst
    if !all_scores.is_empty() {
        let mut sorted = all_scores.clone();
        sorted.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
        println!("\n  Best quality: {} ({:.3})", sorted[0].0, sorted[0].1.composite);
        println!("  Worst quality: {} ({:.3})",
            sorted.last().unwrap().0,
            sorted.last().unwrap().1.composite);
    }
}
