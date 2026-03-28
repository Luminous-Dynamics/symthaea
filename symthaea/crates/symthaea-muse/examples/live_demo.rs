// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live audio demo: consciousness-driven music through speakers.
//!
//! Run with:
//! ```sh
//! cargo run -p symthaea-muse --example live_demo --features muse-live
//! ```
//!
//! Sweeps arousal and valence over 60 seconds to demonstrate how
//! cognitive state drives timbre, tempo, and harmony in real time.

use symthaea_muse::live_output::LiveMuseOutput;
use symthaea_muse::{MuseConfig, MusicalState};

fn main() {
    println!("=== Symthaea Muse: Live Audio Demo ===");
    println!("Playing consciousness-driven music for 60 seconds...");
    println!("Sweeping: calm → agitated → calm");
    println!();

    let config = MuseConfig {
        num_partials: 12,
        ..Default::default()
    };

    let output = match LiveMuseOutput::new(config) {
        Ok(o) => {
            println!("Audio device opened at {}Hz", o.sample_rate());
            o
        }
        Err(e) => {
            eprintln!("Failed to open audio: {e}");
            std::process::exit(1);
        }
    };

    let start = std::time::Instant::now();
    let duration = std::time::Duration::from_secs(60);

    while start.elapsed() < duration {
        let t = start.elapsed().as_secs_f32();
        let progress = t / duration.as_secs_f32();

        // Sweep arousal: 0.2 → 0.9 → 0.2 (bell curve)
        let arousal = 0.2 + 0.7 * (progress * std::f32::consts::PI).sin();

        // Valence oscillates slowly
        let valence = 0.3 * (t * 0.2).sin();

        // Dopamine tracks arousal (more FM modulation when agitated)
        let dopamine = arousal * 0.8;

        // Serotonin inversely tracks arousal (harsh timbre when agitated)
        let serotonin = (1.0 - arousal) * 0.7 + 0.1;

        // Noradrenaline spikes with arousal (edgy overtones)
        let noradrenaline = arousal * 0.6;

        // Consciousness dips slightly at peak stress
        let consciousness = 0.7 - arousal * 0.2;

        // Prediction error rises with arousal
        let prediction_error = arousal * 0.4;

        // Harmonies shift with emotional state
        let mut harmonies = [0.3; 8];
        harmonies[0] = 0.5 * (1.0 - arousal); // ResonantCoherence (calm)
        harmonies[3] = arousal * 0.8; // InfinitePlay/minor 7th (tension)
        harmonies[7] = (1.0 - arousal) * 0.6; // SacredStillness (drone when calm)
        harmonies[6] = arousal * 0.5; // EvolutionaryProgression (urgency)

        let state = MusicalState {
            harmony_activations: harmonies,
            dopamine,
            serotonin,
            noradrenaline,
            arousal,
            valence,
            consciousness_level: consciousness,
            prediction_error,
        };

        output.update_state(&state);

        // Print telemetry
        print!(
            "\r  t={:5.1}s  arousal={:.2}  valence={:.2}  DA={:.2}  5HT={:.2}  NE={:.2}  Ψ={:.2}  ",
            t, arousal, valence, dopamine, serotonin, noradrenaline, consciousness,
        );

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!();
    println!("Done. Stopping audio...");
    output.stop();
}
