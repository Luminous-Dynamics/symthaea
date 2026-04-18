// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Click analyzer — measures crackling severity directly via 2nd derivative.
//!
//! This is the primary perceptual metric for crackling. Unlike spectral
//! flatness (which measures whole-signal distribution), click_score measures
//! localized discontinuities at sample level — exactly what makes audio crackle.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --example click_analyze -- audio_output/
//! ```

use std::env;
use std::path::PathBuf;
use symthaea_muse::param_tuner::{click_quality, click_score};

fn main() {
    let args: Vec<String> = env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "audio_output".to_string());

    println!("═══ Click Analysis: {} ═══\n", dir);
    println!(
        "{:<45} {:>10} {:>10} {:>10} {:>10} {}",
        "File", "Max 2d", "Clicks", "Density%", "Quality", "Severity"
    );
    println!("{}", "─".repeat(110));

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    files.sort();

    let mut results: Vec<(String, f32, usize, f32, f32)> = Vec::new();

    for path in &files {
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let reader = match hound::WavReader::open(path) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let spec = reader.spec();
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

        // Downmix to mono
        let mono: Vec<f32> = if spec.channels == 2 {
            samples.chunks(2).map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) * 0.5).collect()
        } else {
            samples
        };

        let metrics = click_score(&mono);
        let quality = click_quality(&metrics);

        let severity = if metrics.max_second_derivative > 0.5 {
            "SEVERE"
        } else if metrics.max_second_derivative > 0.2 {
            "AUDIBLE"
        } else if metrics.max_second_derivative > 0.08 {
            "MINOR"
        } else {
            "CLEAN"
        };

        println!(
            "{:<45} {:>10.4} {:>10} {:>10.4} {:>10.3} {}",
            filename,
            metrics.max_second_derivative,
            metrics.click_count,
            metrics.click_density * 100.0,
            quality,
            severity
        );

        results.push((
            filename,
            metrics.max_second_derivative,
            metrics.click_count,
            metrics.click_density,
            quality,
        ));
    }

    // Summary
    if !results.is_empty() {
        results.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
        println!("\n═══ Ranked by click quality (cleanest first) ═══");
        for (i, (name, _, _, _, q)) in results.iter().take(5).enumerate() {
            println!("  {}. {} — {:.3}", i + 1, name, q);
        }
        println!("\n═══ Worst (most crackling) ═══");
        for (i, (name, max_2d, count, _, q)) in results.iter().rev().take(5).enumerate() {
            println!(
                "  {}. {} — quality={:.3}, max_2d={:.3}, clicks={}",
                i + 1, name, q, max_2d, count
            );
        }
    }
}
