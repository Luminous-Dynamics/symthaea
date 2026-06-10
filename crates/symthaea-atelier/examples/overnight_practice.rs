// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Overnight Practice Study
//!
//! Runs the autopoietic loop for 500 rounds and logs:
//! - Score trajectory (composite per round)
//! - Style evolution (16D embedding sampled every 50 rounds)
//! - Cognitive state drift over time
//! - Best and worst artworks saved as SVGs
//!
//! **Question:** Does the TasteModel develop non-trivial preferences?
//! Does the style vector drift into unexpected territory?
//!
//! ```bash
//! cargo run -p symthaea-atelier --example overnight_practice --release
//! ```

use symthaea_atelier::critic::{PerceptualInput, PracticeResult, SelfCritic, auto_improve_with};
use symthaea_atelier::{AtelierConfig, create_artwork};
use symthaea_canvas::CognitiveSnapshot;
use symthaea_gallery::style::{STYLE_DIM, StyleEmbedding, compute_style};
use symthaea_gallery::{ArtModality, GalleryEntry, GalleryIndex};

fn main() {
    let total_rounds = 500;
    let checkpoint_interval = 50;

    println!("═══════════════════════════════════════════════════════");
    println!("  Overnight Practice Study ({total_rounds} rounds)");
    println!("  Checkpoint every {checkpoint_interval} rounds");
    println!("═══════════════════════════════════════════════════════\n");

    let study_dir = std::path::Path::new("target/practice-study");
    std::fs::create_dir_all(study_dir).ok();

    let config = AtelierConfig {
        width: 512.0,
        height: 512.0,
        max_elements: 150,
        iteration_budget: 2,
        ..Default::default()
    };

    let mut snapshot = CognitiveSnapshot {
        consciousness_level: 0.7,
        harmony_activations: [0.5; 8],
        dopamine: 0.5,
        serotonin: 0.5,
        noradrenaline: 0.3,
        oxytocin: 0.4,
        allostatic_load: 0.1,
        arousal: 0.5,
        valence: 0.0,
        betti_0: 3,
        betti_1: 1,
        betti_2: 0,
        ..CognitiveSnapshot::dormant()
    };

    let initial_snapshot = snapshot.clone();

    // Track trajectory and style evolution
    let mut all_scores: Vec<f32> = Vec::new();
    let mut style_checkpoints: Vec<(usize, StyleEmbedding)> = Vec::new();
    let mut state_checkpoints: Vec<(usize, f32, f32, f32, f32)> = Vec::new(); // round, arousal, valence, serotonin, consciousness
    let mut gallery = GalleryIndex {
        entries: Vec::new(),
        max_entries: total_rounds,
    };
    let mut best_score = 0.0f32;
    let mut best_round = 0usize;
    let mut worst_score = 1.0f32;
    let mut worst_round = 0usize;

    let mut critic = SelfCritic::new();

    for round in 0..total_rounds {
        // Generate artwork
        let artwork = create_artwork(&config, &snapshot, round as u64);

        // Critique
        let features = PerceptualInput {
            creative_surprise: artwork.aesthetic_score.surprise,
            perceptual_coherence: artwork.aesthetic_score.order,
            visual_features: vec![
                artwork.aesthetic_score.order,
                artwork.aesthetic_score.complexity,
                artwork.aesthetic_score.surprise,
                artwork.aesthetic_score.harmony,
                artwork.aesthetic_score.birkhoff,
                artwork.aesthetic_score.composite,
            ],
        };
        let verdict = critic.evaluate(&features, &snapshot);
        all_scores.push(verdict.composite);

        // Track best/worst
        if verdict.composite > best_score {
            best_score = verdict.composite;
            best_round = round;
            let path = study_dir.join("best.svg");
            std::fs::write(&path, &artwork.svg).ok();
        }
        if verdict.composite < worst_score {
            worst_score = verdict.composite;
            worst_round = round;
            let path = study_dir.join("worst.svg");
            std::fs::write(&path, &artwork.svg).ok();
        }

        // Add to gallery for style computation
        gallery.entries.push(GalleryEntry {
            id: uuid::Uuid::new_v4(),
            created_at_cycle: round as u64,
            created_at: chrono::Utc::now(),
            modality: ArtModality::Visual {
                filename: format!("round_{round:04}.svg"),
            },
            aesthetic_score: artwork.aesthetic_score,
            harmony_activations: snapshot.harmony_activations,
            tags: vec![],
            protected: verdict.self_surprise > 0.7,
        });

        // Apply wisdom
        symthaea_atelier::critic::apply_verdict_wisdom(&mut snapshot, &verdict);

        // Checkpoint
        if (round + 1) % checkpoint_interval == 0 || round == 0 {
            let style = compute_style(&gallery, 50);
            let initial_style = if style_checkpoints.is_empty() {
                style.clone()
            } else {
                style_checkpoints[0].1.clone()
            };
            let drift = 1.0 - style.similarity(&initial_style);

            style_checkpoints.push((round, style.clone()));
            state_checkpoints.push((
                round,
                snapshot.arousal,
                snapshot.valence,
                snapshot.serotonin,
                snapshot.consciousness_level as f32,
            ));

            let recent_avg: f32 = if all_scores.len() >= 10 {
                all_scores[all_scores.len() - 10..].iter().sum::<f32>() / 10.0
            } else {
                all_scores.iter().sum::<f32>() / all_scores.len() as f32
            };

            println!(
                "  Round {:4}/{} | score={:.3} avg10={:.3} | drift={:.4} | aros={:.2} val={:+.2} ser={:.2} Ψ={:.2}",
                round + 1,
                total_rounds,
                verdict.composite,
                recent_avg,
                drift,
                snapshot.arousal,
                snapshot.valence,
                snapshot.serotonin,
                snapshot.consciousness_level
            );
        }
    }

    // ── Final Report ──

    println!("\n\n═══════════════════════════════════════════════════════");
    println!("  Practice Study Results ({total_rounds} rounds)");
    println!("═══════════════════════════════════════════════════════\n");

    let final_style = compute_style(&gallery, 50);
    let initial_style = &style_checkpoints[0].1;
    let total_drift = 1.0 - final_style.similarity(initial_style);

    println!(
        "  Score range: {:.4} (worst, round {worst_round}) → {:.4} (best, round {best_round})",
        worst_score, best_score
    );

    let first_50: f32 = all_scores[..50.min(all_scores.len())].iter().sum::<f32>()
        / 50.0f32.min(all_scores.len() as f32);
    let last_50: f32 = if all_scores.len() >= 50 {
        all_scores[all_scores.len() - 50..].iter().sum::<f32>() / 50.0
    } else {
        first_50
    };
    println!(
        "  First 50 avg: {:.4} | Last 50 avg: {:.4} | Delta: {:+.4}",
        first_50,
        last_50,
        last_50 - first_50
    );
    println!("  Total style drift: {:.4}", total_drift);

    println!("\n  Style evolution:");
    for (round, style) in &style_checkpoints {
        let drift = 1.0 - style.similarity(initial_style);
        println!(
            "    Round {:4}: drift={:.4} novelty_pref={:.3} trend={:.3}",
            round + 1,
            drift,
            style.vector[15],
            style.vector[14]
        );
    }

    println!("\n  Final cognitive state:");
    println!(
        "    Arousal:       {:.3} (was {:.3})",
        snapshot.arousal, initial_snapshot.arousal
    );
    println!(
        "    Valence:       {:+.3} (was {:+.3})",
        snapshot.valence, initial_snapshot.valence
    );
    println!(
        "    Serotonin:     {:.3} (was {:.3})",
        snapshot.serotonin, initial_snapshot.serotonin
    );
    println!(
        "    Consciousness: {:.3} (was {:.3})",
        snapshot.consciousness_level, initial_snapshot.consciousness_level
    );
    println!(
        "    Harmonies:     [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        snapshot.harmony_activations[0],
        snapshot.harmony_activations[1],
        snapshot.harmony_activations[2],
        snapshot.harmony_activations[3],
        snapshot.harmony_activations[4],
        snapshot.harmony_activations[5],
        snapshot.harmony_activations[6],
        snapshot.harmony_activations[7]
    );

    println!("\n  Files:");
    println!("    target/practice-study/best.svg (round {best_round}, score {best_score:.4})");
    println!("    target/practice-study/worst.svg (round {worst_round}, score {worst_score:.4})");

    if total_drift > 0.05 {
        println!(
            "\n  ✦ Style drifted significantly ({total_drift:.3}). The system developed distinct preferences."
        );
    } else {
        println!(
            "\n  ○ Style drift minimal ({total_drift:.3}). System converged early — needs more diverse input or lower stillness threshold."
        );
    }
}
