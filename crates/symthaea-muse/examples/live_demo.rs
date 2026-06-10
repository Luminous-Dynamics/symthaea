// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live audio demo: consciousness-driven music through speakers.
//!
//! Run with:
//! ```sh
//! cargo run -p symthaea-muse --example live_demo --features muse-live
//! ```
//!
//! Cycles through 6 emotional states (10s each) to demonstrate how
//! cognitive state drives timbre, tempo, harmony, and roughness.

use symthaea_muse::MusicalState;
use symthaea_muse::live_output::LiveMuseOutput;

/// Named emotional phase with full neuromodulator prescription.
struct Phase {
    name: &'static str,
    duration_secs: f32,
    valence: f32,
    arousal: f32,
    dopamine: f32,
    serotonin: f32,
    noradrenaline: f32,
    consciousness: f32,
    prediction_error: f32,
    harmonies: [f32; 8],
}

fn phases() -> Vec<Phase> {
    vec![
        Phase {
            name: "Contentment (warm, calm, major)",
            duration_secs: 12.0,
            valence: 0.7,
            arousal: 0.2,
            dopamine: 0.3,
            serotonin: 0.8,
            noradrenaline: 0.1,
            consciousness: 0.6,
            prediction_error: 0.1,
            harmonies: [0.7, 0.5, 0.6, 0.1, 0.4, 0.3, 0.2, 0.5],
        },
        Phase {
            name: "Excitement (bright, fast, triumphant)",
            duration_secs: 12.0,
            valence: 0.8,
            arousal: 0.85,
            dopamine: 0.9,
            serotonin: 0.3,
            noradrenaline: 0.4,
            consciousness: 0.8,
            prediction_error: 0.2,
            harmonies: [0.5, 0.7, 0.8, 0.3, 0.6, 0.5, 0.7, 0.1],
        },
        Phase {
            name: "Grief (dark, slow, minor, detuned)",
            duration_secs: 12.0,
            valence: -0.8,
            arousal: 0.15,
            dopamine: 0.1,
            serotonin: 0.2,
            noradrenaline: 0.2,
            consciousness: 0.4,
            prediction_error: 0.1,
            harmonies: [0.2, 0.1, 0.2, 0.4, 0.1, 0.1, 0.1, 0.6],
        },
        Phase {
            name: "Panic (harsh, fast, dissonant, jittery)",
            duration_secs: 12.0,
            valence: -0.7,
            arousal: 0.95,
            dopamine: 0.3,
            serotonin: 0.1,
            noradrenaline: 0.9,
            consciousness: 0.3,
            prediction_error: 0.7,
            harmonies: [0.1, 0.2, 0.1, 0.9, 0.3, 0.1, 0.5, 0.0],
        },
        Phase {
            name: "Wonder (ascending, spacious, high Phi)",
            duration_secs: 12.0,
            valence: 0.6,
            arousal: 0.5,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.2,
            consciousness: 0.9,
            prediction_error: 0.3,
            harmonies: [0.6, 0.5, 0.7, 0.2, 0.5, 0.4, 0.3, 0.3],
        },
        Phase {
            name: "Sacred Stillness (drone, silence, release)",
            duration_secs: 12.0,
            valence: 0.2,
            arousal: 0.05,
            dopamine: 0.1,
            serotonin: 0.6,
            noradrenaline: 0.0,
            consciousness: 0.5,
            prediction_error: 0.0,
            harmonies: [0.3, 0.2, 0.3, 0.0, 0.2, 0.2, 0.1, 0.9],
        },
    ]
}

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Symthaea Muse: Live Consciousness Audio Demo   ║");
    println!("║  6 emotional phases × 12s each = 72 seconds     ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let config = symthaea_muse::MuseConfig {
        num_partials: 12,
        enable_antialiasing: true,
        ..Default::default()
    };

    let output = match LiveMuseOutput::new(config) {
        Ok(o) => {
            println!("  Audio device: {}Hz stereo", o.sample_rate());
            o
        }
        Err(e) => {
            eprintln!("  Failed to open audio: {e}");
            eprintln!("  Ensure audio output is available (speakers/headphones).");
            std::process::exit(1);
        }
    };

    let all_phases = phases();
    let mut global_t = 0.0f32;

    for (i, phase) in all_phases.iter().enumerate() {
        println!();
        println!("  [{}/{}] {} ", i + 1, all_phases.len(), phase.name);
        println!(
            "        V={:+.1} A={:.1} DA={:.1} 5HT={:.1} NE={:.1} Ψ={:.1} PE={:.1}",
            phase.valence,
            phase.arousal,
            phase.dopamine,
            phase.serotonin,
            phase.noradrenaline,
            phase.consciousness,
            phase.prediction_error
        );

        let phase_start = std::time::Instant::now();
        let phase_dur = std::time::Duration::from_secs_f32(phase.duration_secs);

        while phase_start.elapsed() < phase_dur {
            let t = phase_start.elapsed().as_secs_f32();
            let blend = (t / 2.0).min(1.0); // 2s crossfade into each phase

            // Smooth interpolation for the first 2 seconds
            let v = phase.valence * blend;
            let a = phase.arousal * blend;

            let state = MusicalState {
                harmony_activations: phase.harmonies,
                dopamine: phase.dopamine * blend,
                serotonin: phase.serotonin * blend + 0.3 * (1.0 - blend),
                noradrenaline: phase.noradrenaline * blend,
                arousal: a,
                valence: v,
                consciousness_level: phase.consciousness,
                prediction_error: phase.prediction_error * blend,
            };

            output.update_state(&state);

            // Progress bar
            let pct = (t / phase.duration_secs * 40.0) as usize;
            print!(
                "\r        [{}{}] {:.0}s ",
                "█".repeat(pct.min(40)),
                "░".repeat(40 - pct.min(40)),
                t
            );

            std::thread::sleep(std::time::Duration::from_millis(100));
            global_t += 0.1;
        }
        println!();
    }

    println!();
    println!(
        "  Done. {} seconds of consciousness-driven audio.",
        global_t as u32
    );
    output.stop();
}
