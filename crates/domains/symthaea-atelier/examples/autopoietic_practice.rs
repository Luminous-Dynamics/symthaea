// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Autopoietic Practice Session
//!
//! Demonstrates the full self-improving creative loop:
//! generate → self-critique → apply wisdom → repeat until Sacred Stillness.
//!
//! Outputs:
//! - SVG artworks for each round (target/practice-gallery/)
//! - Score trajectory showing aesthetic evolution
//! - Style embedding summary
//!
//! ```bash
//! cargo run -p symthaea-atelier --example autopoietic_practice
//! ```

use symthaea_atelier::critic::{PerceptualInput, SelfCritic, auto_improve};
use symthaea_atelier::{AtelierConfig, create_artwork};
use symthaea_canvas::CognitiveSnapshot;
use symthaea_gallery::evolution::analyze_evolution;
use symthaea_gallery::style::{StyleEmbedding, compute_style};
use symthaea_gallery::{ArtModality, GalleryEntry, GalleryIndex};

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Symthaea Autopoietic Practice Session");
    println!("  The system generates, critiques, and improves.");
    println!("═══════════════════════════════════════════════════════\n");

    // Create output directory
    let gallery_dir = std::path::Path::new("target/practice-gallery");
    std::fs::create_dir_all(gallery_dir).ok();

    // Initial cognitive state: balanced, moderate consciousness
    let mut snapshot = CognitiveSnapshot {
        consciousness_level: 0.7,
        harmony_activations: [0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.3],
        dopamine: 0.5,
        serotonin: 0.5,
        noradrenaline: 0.3,
        oxytocin: 0.4,
        allostatic_load: 0.1,
        arousal: 0.5,
        valence: 0.2,
        betti_0: 3,
        betti_1: 1,
        betti_2: 0,
        ..CognitiveSnapshot::dormant()
    };

    let config = AtelierConfig {
        width: 512.0,
        height: 512.0,
        max_elements: 200,
        iteration_budget: 3,
        ..Default::default()
    };

    // ── Phase 1: Autonomous Practice ─────────────────────────────────

    println!("Phase 1: Autonomous Practice (max 20 rounds)");
    println!("────────────────────────────────────────────\n");

    let result = auto_improve(&config, &mut snapshot, 20, 42);

    println!("  Rounds completed: {}", result.rounds);
    println!("  Reached stillness: {}", result.reached_stillness);
    println!("  Best score: {:.4}", result.best_score);
    println!("  Improvement: {:+.4}", result.improvement);
    println!("\n  Score trajectory:");

    for (i, &score) in result.trajectory.iter().enumerate() {
        let bar_len = (score * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        let marker = if i == 0 {
            " ← start"
        } else if Some(&score) == result.trajectory.last() && result.reached_stillness {
            " ← stillness"
        } else if score == result.best_score {
            " ← best"
        } else {
            ""
        };
        println!("    Round {:2}: {:.4} |{bar}{marker}", i + 1, score);
    }

    // ── Phase 2: Generate Gallery ────────────────────────────────────

    println!("\n\nPhase 2: Generating Gallery (10 artworks with evolved state)");
    println!("─────────────────────────────────────────────────────────\n");

    let mut gallery = GalleryIndex {
        entries: Vec::new(),
        max_entries: 100,
    };

    let mut critic = SelfCritic::new();

    for i in 0..10 {
        let artwork = create_artwork(&config, &snapshot, 100 + i);

        // Self-critique
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

        // Save SVG
        let filename = format!("artwork_{:02}.svg", i);
        let filepath = gallery_dir.join(&filename);
        std::fs::write(&filepath, &artwork.svg).ok();

        // Add to gallery
        gallery.entries.push(GalleryEntry {
            id: uuid::Uuid::new_v4(),
            created_at_cycle: i,
            created_at: chrono::Utc::now(),
            modality: ArtModality::Visual {
                filename: filename.clone(),
            },
            aesthetic_score: artwork.aesthetic_score,
            harmony_activations: snapshot.harmony_activations,
            tags: vec![format!("{:?}", artwork.style)],
            protected: verdict.self_surprise > 0.7,
        });

        println!(
            "  [{:02}] {} | composite={:.3} | novelty={:.3} | taste={:.3} | {}",
            i,
            filename,
            verdict.composite,
            verdict.novelty,
            verdict.taste_alignment,
            if verdict.reached_stillness {
                "STILL"
            } else {
                "evolving"
            }
        );
    }

    // ── Phase 3: Style Analysis ──────────────────────────────────────

    println!("\n\nPhase 3: Style Analysis");
    println!("───────────────────────\n");

    let style = compute_style(&gallery, 50);
    println!("  Style embedding (16D):");
    println!(
        "    Harmony means:  [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        style.vector[0],
        style.vector[1],
        style.vector[2],
        style.vector[3],
        style.vector[4],
        style.vector[5],
        style.vector[6],
        style.vector[7]
    );
    println!(
        "    Aesthetic means: order={:.2}, complexity={:.2}, surprise={:.2}, harmony={:.2}, birkhoff={:.2}, composite={:.2}",
        style.vector[8],
        style.vector[9],
        style.vector[10],
        style.vector[11],
        style.vector[12],
        style.vector[13]
    );
    println!(
        "    Score trend:     {:.3} (>0.5 = improving)",
        style.vector[14]
    );
    println!(
        "    Novelty pref:    {:.3} (>0.5 = explorer)",
        style.vector[15]
    );
    println!(
        "    Confidence:      {:.2} ({} samples)",
        style.confidence, style.sample_count
    );

    // Compare to neutral
    let neutral = StyleEmbedding::neutral();
    let divergence = 1.0 - style.similarity(&neutral);
    println!("\n    Divergence from neutral: {:.3}", divergence);
    if divergence > 0.1 {
        println!("    → The system has developed a distinct artistic identity.");
    } else {
        println!("    → Style is still close to neutral (needs more practice).");
    }

    // ── Phase 4: Evolution Report ────────────────────────────────────

    let evolution = analyze_evolution(&gallery, 3);
    println!("\n\nPhase 4: Evolution Report");
    println!("─────────────────────────\n");
    println!("  Score trend: {:+.4}", evolution.score_trend);
    println!("  Style periods: {}", evolution.periods.len());
    for period in &evolution.periods {
        println!(
            "    Cycles {}-{}: {} (avg score {:.3}, {} works)",
            period.start_cycle,
            period.end_cycle,
            period.harmony_name,
            period.average_score,
            period.artwork_count
        );
    }

    // ── Final State ──────────────────────────────────────────────────

    println!("\n\n═══════════════════════════════════════════════════════");
    println!("  Final Cognitive State (after practice + generation):");
    println!("═══════════════════════════════════════════════════════");
    println!("  Consciousness: {:.3}", snapshot.consciousness_level);
    println!("  Valence:       {:+.3}", snapshot.valence);
    println!("  Arousal:        {:.3}", snapshot.arousal);
    println!("  Dopamine:       {:.3}", snapshot.dopamine);
    println!("  Serotonin:      {:.3}", snapshot.serotonin);
    println!(
        "  Harmonies:     [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        snapshot.harmony_activations[0],
        snapshot.harmony_activations[1],
        snapshot.harmony_activations[2],
        snapshot.harmony_activations[3],
        snapshot.harmony_activations[4],
        snapshot.harmony_activations[5],
        snapshot.harmony_activations[6],
        snapshot.harmony_activations[7]
    );
    println!(
        "\n  Gallery: {} artworks in target/practice-gallery/",
        gallery.entries.len()
    );
    println!("  Open any .svg file in a browser to view.\n");
}
