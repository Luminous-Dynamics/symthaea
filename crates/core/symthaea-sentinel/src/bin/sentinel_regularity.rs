// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analyze temporal regularity of test sounds

use symthaea_sentinel::{
    FileAudioConfig, FileAudioPump, compute_onset_strength, compute_temporal_regularity,
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

    let mut prev_spectrum: Vec<f32> = Vec::new();
    let mut onset_history: Vec<f32> = Vec::new();
    let mut regularity_values: Vec<f32> = Vec::new();
    let mut onset_values: Vec<f32> = Vec::new();

    while let Some(spectrum) = pump.next_power_spectrum() {
        let onset_strength = if prev_spectrum.is_empty() {
            0.0
        } else {
            compute_onset_strength(&prev_spectrum, &spectrum)
        };
        prev_spectrum = spectrum;

        onset_history.push(onset_strength);
        onset_values.push(onset_strength);
        if onset_history.len() > 100 {
            onset_history.remove(0);
        }

        if onset_history.len() >= 20 {
            let regularity = compute_temporal_regularity(&onset_history);
            regularity_values.push(regularity);
        }
    }

    if regularity_values.is_empty() {
        return;
    }

    let mean_regularity: f32 =
        regularity_values.iter().sum::<f32>() / regularity_values.len() as f32;
    let max_regularity: f32 = regularity_values.iter().cloned().fold(0.0f32, f32::max);
    let mean_onset: f32 = onset_values.iter().sum::<f32>() / onset_values.len() as f32;
    let max_onset: f32 = onset_values.iter().cloned().fold(0.0f32, f32::max);

    let name = path.rsplit('/').next().unwrap_or(path);
    println!(
        "{:15} Regularity: mean={:.4} max={:.4}  Onset: mean={:.4} max={:.4}",
        name, mean_regularity, max_regularity, mean_onset, max_onset
    );
}

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║              TEMPORAL REGULARITY ANALYSIS                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let files = [
        "test-data/dog/dog1.wav",
        "test-data/dog/dog2.wav",
        "test-data/dog/dog3.wav",
        "test-data/glass/glass1.wav",
        "test-data/glass/glass2.wav",
        "test-data/clock/clock1.wav",
        "test-data/clock/clock2.wav",
        "test-data/rain/rain1.wav",
    ];

    for path in files {
        analyze_file(path);
    }

    println!("\nHigh regularity = periodic pattern, Low = aperiodic/random");
}
