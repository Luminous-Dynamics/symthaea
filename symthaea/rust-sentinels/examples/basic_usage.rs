// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Basic usage example for the Sentinels library

use sentinels::{AnalysisMode, analyze_consciousness};

fn main() {
    // Generate synthetic EEG-like data (30 seconds at 256 Hz)
    let sample_rate = 256.0;
    let duration = 30.0;
    let n_samples = (sample_rate * duration) as usize;

    // Create a signal with mixed frequencies
    let data: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let pi2 = 2.0 * std::f32::consts::PI;

            // Mix of alpha (10 Hz), theta (6 Hz), and some beta (20 Hz)
            0.5 * (pi2 * 10.0 * t).sin()  // Alpha
                + 0.3 * (pi2 * 6.0 * t).sin()   // Theta
                + 0.1 * (pi2 * 20.0 * t).sin()  // Beta
                + 0.1 * (pi2 * 2.0 * t).sin() // Delta
        })
        .collect();

    // Analyze consciousness state
    match analyze_consciousness(&data, sample_rate, AnalysisMode::Auto) {
        Ok(poc) => {
            println!("=== Proof of Consciousness ===");
            println!();
            println!("Overall State: {:?}", poc.state);
            println!(
                "Consciousness Level: {:.1}%",
                poc.consciousness_level * 100.0
            );
            println!("Wellbeing Score: {:.1}%", poc.wellbeing_score * 100.0);
            println!();

            if let Some(emotion) = &poc.emotion {
                println!("Emotion:");
                println!("  Valence: {:+.2}", emotion.valence);
                println!("  Arousal: {:.2}", emotion.arousal);
                println!("  Quadrant: {:?}", emotion.quadrant());
            }

            if let Some(sleep) = &poc.sleep {
                println!("Sleep:");
                println!("  Stage: {:?}", sleep.stage);
                println!("  Delta Power: {:.1}%", sleep.delta_power * 100.0);
            }

            if let Some(meditation) = &poc.meditation {
                println!("Meditation:");
                println!(
                    "  Depth: {:.2} ({:?})",
                    meditation.depth,
                    meditation.state()
                );
                println!("  Index: {:.2}", meditation.meditation_index);
            }

            println!();
            println!("JSON Output:");
            println!("{}", poc.to_json());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}