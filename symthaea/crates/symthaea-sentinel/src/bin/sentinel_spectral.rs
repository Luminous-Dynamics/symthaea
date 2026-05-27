// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analyze spectral characteristics of test sounds

use symthaea_sentinel::{
    FileAudioConfig, FileAudioPump, compute_spectral_centroid, compute_spectral_flatness,
};

fn analyze_file(path: &str) {
    let config = FileAudioConfig::default();
    let mut pump = match FileAudioPump::new(path, config) {
        Ok(p) => p,
        Err(e) => {
            println!("  Failed: {}", e);
            return;
        }
    };

    let mut centroid_sum = 0.0f32;
    let mut flatness_sum = 0.0f32;
    let mut count = 0;
    let mut flatness_values: Vec<f32> = Vec::new();
    let mut centroid_values: Vec<f32> = Vec::new();

    while let Some(spectrum) = pump.next_power_spectrum() {
        let centroid = compute_spectral_centroid(&spectrum, pump.sample_rate(), pump.window_size());
        let flatness = compute_spectral_flatness(&spectrum);

        centroid_sum += centroid;
        flatness_sum += flatness;
        centroid_values.push(centroid);
        flatness_values.push(flatness);
        count += 1;
    }

    if count == 0 {
        return;
    }

    let mean_centroid = centroid_sum / count as f32;
    let mean_flatness = flatness_sum / count as f32;

    // Compute variance
    let centroid_var: f32 = centroid_values
        .iter()
        .map(|&c| (c - mean_centroid).powi(2))
        .sum::<f32>()
        / count as f32;
    let flatness_var: f32 = flatness_values
        .iter()
        .map(|&f| (f - mean_flatness).powi(2))
        .sum::<f32>()
        / count as f32;

    let name = path.rsplit('/').next().unwrap_or(path);
    println!(
        "{:15} Centroid: {:7.1} Hz (σ={:6.1})  Flatness: {:.4} (σ={:.4})",
        name,
        mean_centroid,
        centroid_var.sqrt(),
        mean_flatness,
        flatness_var.sqrt()
    );
}

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║              SPECTRAL CHARACTERISTIC ANALYSIS                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let files = [
        // Synthetic sounds with known characteristics
        "test-data/cat/cat1.wav",       // ~800 Hz tonal
        "test-data/cat/cat2.wav",       // ~900 Hz tonal
        "test-data/dog/synth_dog1.wav", // ~200 Hz tonal
        "test-data/engine/engine1.wav", // ~80-160 Hz periodic
        "test-data/siren/siren1.wav",   // 800-1600 Hz sweep
        "test-data/bell/bell1.wav",     // 1000-3000 Hz harmonic
        "test-data/static/static1.wav", // Broadband noise
        "test-data/static/static2.wav", // Pink noise
        // Real ESC-50 samples
        "test-data/dog/dog1.wav",
        "test-data/dog/dog3.wav",
        "test-data/glass/glass1.wav",
        "test-data/rain/rain1.wav",
        "test-data/clock/clock1.wav",
    ];

    println!(
        "{:15} {:28} {:20}",
        "File", "Spectral Centroid", "Spectral Flatness"
    );
    println!("{}", "-".repeat(65));

    for path in files {
        analyze_file(path);
    }

    println!("\nNote: High flatness (~1.0) = noise-like, Low flatness (~0.0) = tonal/harmonic");
}
