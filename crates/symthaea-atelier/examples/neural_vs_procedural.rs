// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compare Neural (CfC-driven) vs Procedural (Composite) art generation.
//!
//! Run: cargo run -p symthaea-atelier --example neural_vs_procedural

use symthaea_atelier::{AtelierConfig, AtelierStyle};
use symthaea_canvas::CognitiveSnapshot;

fn main() {
    let output_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("gallery")
        .join("comparison");
    std::fs::create_dir_all(&output_dir).expect("create dir");

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║  Neural vs Procedural Art — Side-by-Side Compare  ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    let states = [
        ("serene", make_serene()),
        ("turbulent", make_turbulent()),
        ("playful", make_playful()),
        ("contemplative", make_contemplative()),
    ];

    for (name, snapshot) in &states {
        println!(
            "━━━ {name} (Ψ={:.2}, v={:.2}, a={:.2}) ━━━",
            snapshot.consciousness_level, snapshot.valence, snapshot.arousal
        );

        // Procedural composite
        let proc_config = AtelierConfig {
            style: AtelierStyle::Composite,
            iteration_budget: 5,
            ..AtelierConfig::default()
        };
        let proc_art = symthaea_atelier::create_artwork_iterative(&proc_config, snapshot, 42);

        // Neural CfC
        let neural_config = AtelierConfig {
            style: AtelierStyle::Neural,
            iteration_budget: 5,
            ..AtelierConfig::default()
        };
        let neural_art = symthaea_atelier::create_artwork_iterative(&neural_config, snapshot, 42);

        // Hybrid CfC-modulated procedural
        let hybrid_config = AtelierConfig {
            style: AtelierStyle::Hybrid,
            iteration_budget: 5,
            ..AtelierConfig::default()
        };
        let hybrid_art = symthaea_atelier::create_artwork_iterative(&hybrid_config, snapshot, 42);

        // Save all three
        let proc_path = output_dir.join(format!("{name}_procedural.svg"));
        let neural_path = output_dir.join(format!("{name}_neural.svg"));
        let hybrid_path = output_dir.join(format!("{name}_hybrid.svg"));
        std::fs::write(&proc_path, &proc_art.svg).expect("write");
        std::fs::write(&neural_path, &neural_art.svg).expect("write");
        std::fs::write(&hybrid_path, &hybrid_art.svg).expect("write");

        // Stats
        println!(
            "  Procedural: {:.3} score, {} nodes, {} bytes",
            proc_art.aesthetic_score.composite,
            proc_art.scene.node_count(),
            proc_art.svg.len()
        );
        println!(
            "  Neural:     {:.3} score, {} nodes, {} bytes",
            neural_art.aesthetic_score.composite,
            neural_art.scene.node_count(),
            neural_art.svg.len()
        );

        println!(
            "  Hybrid:     {:.3} score, {} nodes, {} bytes",
            hybrid_art.aesthetic_score.composite,
            hybrid_art.scene.node_count(),
            hybrid_art.svg.len()
        );
        println!();
    }

    println!("Output: {}", output_dir.display());
    println!("Compare all three:");
    println!("  firefox {p}/*_procedural.svg", p = output_dir.display());
    println!("  firefox {p}/*_hybrid.svg", p = output_dir.display());
    println!("  firefox {p}/*_neural.svg", p = output_dir.display());
}

fn make_serene() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.85,
        valence: 0.6,
        arousal: 0.3,
        dopamine: 0.5,
        serotonin: 0.8,
        noradrenaline: 0.2,
        harmony_activations: [0.7, 0.8, 0.6, 0.3, 0.7, 0.6, 0.5, 0.8],
        betti_0: 4,
        betti_1: 2,
        betti_2: 1,
        persistence_components: vec![[0.0, 0.7], [0.1, 0.9]],
        persistence_cycles: vec![[0.1, 0.6]],
        thought_vector: vec![0.3, -0.1, 0.5, 0.2, -0.3, 0.1, 0.4, -0.2],
        prediction_error: 0.05,
        ..CognitiveSnapshot::dormant()
    }
}

fn make_turbulent() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.6,
        valence: -0.6,
        arousal: 0.9,
        dopamine: 0.3,
        serotonin: 0.2,
        noradrenaline: 0.9,
        harmony_activations: [0.3, 0.2, 0.4, 0.9, 0.5, 0.2, 0.8, 0.1],
        betti_0: 7,
        betti_1: 3,
        betti_2: 2,
        persistence_components: vec![[0.0, 0.3], [0.1, 0.6], [0.2, 0.9]],
        persistence_cycles: vec![[0.1, 0.5], [0.2, 0.7]],
        thought_vector: vec![0.8, -0.7, 0.9, -0.5, 0.6, -0.8, 0.3, 0.7],
        prediction_error: 0.7,
        ..CognitiveSnapshot::dormant()
    }
}

fn make_playful() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.75,
        valence: 0.4,
        arousal: 0.7,
        dopamine: 0.8,
        serotonin: 0.5,
        noradrenaline: 0.5,
        harmony_activations: [0.4, 0.5, 0.3, 0.95, 0.6, 0.4, 0.7, 0.1],
        betti_0: 5,
        betti_1: 2,
        betti_2: 1,
        persistence_components: vec![[0.0, 0.6], [0.15, 0.8]],
        persistence_cycles: vec![[0.2, 0.7]],
        thought_vector: vec![0.5, 0.3, -0.2, 0.6, 0.1, -0.5, 0.4, 0.2],
        prediction_error: 0.3,
        ..CognitiveSnapshot::dormant()
    }
}

fn make_contemplative() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.9,
        valence: 0.2,
        arousal: 0.15,
        dopamine: 0.4,
        serotonin: 0.7,
        noradrenaline: 0.1,
        harmony_activations: [0.8, 0.6, 0.9, 0.2, 0.5, 0.7, 0.4, 0.95],
        betti_0: 2,
        betti_1: 1,
        betti_2: 0,
        persistence_components: vec![[0.0, 0.9]],
        persistence_cycles: vec![[0.1, 0.8]],
        thought_vector: vec![0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0],
        prediction_error: 0.02,
        ..CognitiveSnapshot::dormant()
    }
}
