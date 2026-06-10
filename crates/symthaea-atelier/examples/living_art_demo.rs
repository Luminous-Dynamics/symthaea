// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generate living art HTML pages — art that breathes and evolves in the browser.
//!
//! Run: cargo run -p symthaea-atelier --example living_art_demo

use symthaea_atelier::living_art::{LivingArtConfig, generate_living_art};
use symthaea_canvas::CognitiveSnapshot;

fn main() {
    let output_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("gallery")
        .join("living");
    std::fs::create_dir_all(&output_dir).expect("create dir");

    println!("╔════════════════════════════════════════════════╗");
    println!("║  Living Art — CfC Temporal Dynamics in Browser ║");
    println!("╚════════════════════════════════════════════════╝\n");

    let config = LivingArtConfig {
        width: 512,
        height: 512,
        initial_strokes: 40,
        lifespan_secs: 120.0,
        evolution_dt: 0.02,
        strokes_per_frame: 2,
    };

    let states: Vec<(&str, CognitiveSnapshot)> = vec![
        (
            "serene",
            CognitiveSnapshot {
                consciousness_level: 0.85,
                valence: 0.6,
                arousal: 0.25,
                dopamine: 0.5,
                serotonin: 0.8,
                noradrenaline: 0.15,
                harmony_activations: [0.7, 0.8, 0.6, 0.2, 0.7, 0.6, 0.4, 0.9],
                prediction_error: 0.03,
                thought_vector: vec![0.2, -0.1, 0.4, 0.1],
                ..CognitiveSnapshot::dormant()
            },
        ),
        (
            "turbulent",
            CognitiveSnapshot {
                consciousness_level: 0.55,
                valence: -0.7,
                arousal: 0.9,
                dopamine: 0.3,
                serotonin: 0.2,
                noradrenaline: 0.85,
                harmony_activations: [0.2, 0.15, 0.3, 0.9, 0.5, 0.15, 0.85, 0.05],
                prediction_error: 0.75,
                thought_vector: vec![0.8, -0.7, 0.9, -0.5],
                ..CognitiveSnapshot::dormant()
            },
        ),
        (
            "contemplative",
            CognitiveSnapshot {
                consciousness_level: 0.92,
                valence: 0.15,
                arousal: 0.1,
                dopamine: 0.35,
                serotonin: 0.75,
                noradrenaline: 0.08,
                harmony_activations: [0.8, 0.5, 0.9, 0.15, 0.4, 0.7, 0.3, 0.95],
                prediction_error: 0.01,
                thought_vector: vec![0.05, 0.0, 0.1, -0.05],
                ..CognitiveSnapshot::dormant()
            },
        ),
    ];

    for (name, snapshot) in &states {
        let html = generate_living_art(snapshot, &config);
        let path = output_dir.join(format!("{name}.html"));
        std::fs::write(&path, &html).expect("write HTML");

        println!(
            "  {name}: Ψ={:.2}, {} bytes → {}",
            snapshot.consciousness_level,
            html.len(),
            path.display()
        );
    }

    println!("\n  Open in browser:");
    println!("    firefox {}/serene.html", output_dir.display());
    println!("    firefox {}/turbulent.html", output_dir.display());
    println!("    firefox {}/contemplative.html", output_dir.display());
    println!("\n  Watch each piece evolve for 2 minutes, then rest.");
}
