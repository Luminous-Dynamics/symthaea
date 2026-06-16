// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Creative Evolution
//!
//! Interactive creative exploration loop: the system generates art from
//! evolving cognitive states, feeds aesthetic scores back, and its "taste"
//! develops over time. This demonstrates the closed creative feedback loop.
//!
//! Run:
//!   cargo run -p symthaea-atelier --example creative_evolution
//!
//! Output: `/tmp/symthaea-creative-evolution/` — numbered SVGs showing
//! how the art style evolves as aesthetic feedback reshapes generation.

use symthaea_aesthetic::{AestheticScore, AestheticTracker};
use symthaea_atelier::{AtelierConfig, AtelierStyle};
use symthaea_canvas::CognitiveSnapshot;

fn main() {
    let output_dir = {
        let project_gallery = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("gallery")
            .join("evolution");
        if project_gallery.parent().map_or(false, |p| p.exists()) {
            project_gallery
        } else {
            std::path::PathBuf::from("/tmp/symthaea-creative-evolution")
        }
    };
    std::fs::create_dir_all(&output_dir).expect("create output dir");

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Creative Evolution — Taste Through Feedback ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!();

    let config = AtelierConfig {
        style: AtelierStyle::Composite,
        iteration_budget: 5,
        ..AtelierConfig::default()
    };

    let mut tracker = AestheticTracker::default();
    let mut novelty_tracker = symthaea_aesthetic::novelty::NoveltyTracker::new(30);
    let mut taste = symthaea_aesthetic::novelty::TasteModel::new();

    // Simulate evolving cognitive state over 30 "cycles"
    let total_cycles = 30;
    let mut scores: Vec<f32> = Vec::new();
    let mut novelties: Vec<f32> = Vec::new();

    // Initial state: serene
    let mut harmony_activations = [0.5f32, 0.6, 0.4, 0.3, 0.5, 0.6, 0.4, 0.7];
    let mut valence = 0.3f32;
    let mut arousal = 0.3f32;
    let mut consciousness = 0.6f32;
    let mut dopamine = 0.5f32;

    println!("  Cycle  Score  EMA    Novelty  DA Δ     Taste pred  Dominant");
    println!("  ─────  ─────  ─────  ───────  ───────  ──────────  ────────");

    for cycle in 0..total_cycles {
        // Build snapshot from evolving state
        let snapshot = CognitiveSnapshot {
            consciousness_level: consciousness as f64,
            valence,
            arousal,
            dopamine,
            serotonin: 0.5 + valence * 0.2,
            noradrenaline: arousal * 0.7,
            harmony_activations,
            betti_0: 3 + (consciousness * 4.0) as usize,
            betti_1: 1 + (arousal * 2.0) as usize,
            persistence_components: vec![[0.0, consciousness as f64 * 0.8]],
            persistence_cycles: vec![[0.1, 0.6]],
            thought_vector: vec![
                valence * 0.5,
                arousal * 0.3,
                consciousness * 0.4,
                harmony_activations[3] * 0.6,
                -harmony_activations[7] * 0.3,
                dopamine * 0.2,
                0.1,
                -0.1,
            ],
            cycle_count: cycle as u64,
            ..CognitiveSnapshot::dormant()
        };

        // Generate artwork
        let artwork =
            symthaea_atelier::create_artwork_iterative(&config, &snapshot, 42 + cycle as u64);

        // Evaluate aesthetics
        let mut score = artwork.aesthetic_score;

        // Compute novelty relative to gallery history
        let novelty = novelty_tracker.record_and_score(&score);
        score.surprise = novelty;
        score.compute_composite();

        // Feed to tracker → get neuromodulator feedback
        let feedback = tracker.process(&score, &harmony_activations);

        // Train taste model
        taste.train(&score);
        let taste_pred = taste.predict(&score);

        // Save SVG
        let filename = format!("cycle_{:03}.svg", cycle);
        std::fs::write(output_dir.join(&filename), &artwork.svg).expect("write SVG");

        scores.push(score.composite);
        novelties.push(novelty);

        // Find dominant harmony
        let dominant = harmony_activations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let harmony_names = [
            "Coherence",
            "Flourish",
            "Wisdom",
            "Play",
            "Connect",
            "Reciproc",
            "Progress",
            "Stillness",
        ];

        println!(
            "  {:>3}    {:.3}  {:.3}  {:.3}    {:+.4}  {:.3}       {}",
            cycle,
            score.composite,
            tracker.expectation(),
            novelty,
            feedback.dopamine_delta,
            taste_pred,
            harmony_names[dominant],
        );

        // ── Evolve cognitive state based on feedback ──────────────────

        // Dopamine reward → boost the dominant harmony (reinforce what works)
        if feedback.dopamine_delta > 0.01 {
            harmony_activations[dominant] =
                (harmony_activations[dominant] + feedback.dopamine_delta * 2.0).min(1.0);
            dopamine = (dopamine + feedback.dopamine_delta).min(1.0);
        }

        // Low novelty → increase exploration (shift harmonies)
        if novelty < 0.4 {
            // Rotate dominant harmony forward (explore new styles)
            let next = (dominant + 1) % 8;
            harmony_activations[next] += 0.1;
            harmony_activations[dominant] -= 0.05;
            arousal = (arousal + 0.05).min(1.0); // more energy for exploration
        }

        // Surprise signal → boost noradrenaline → affects visual energy
        if feedback.surprise_signal > 0.05 {
            arousal = (arousal + feedback.surprise_signal * 0.5).min(1.0);
        }

        // Serotonin → increase consciousness + valence (satisfaction loop)
        if feedback.serotonin_delta > 0.01 {
            consciousness = (consciousness + feedback.serotonin_delta * 0.5).min(0.95);
            valence = (valence + feedback.serotonin_delta * 0.3).min(0.8);
        }

        // Gentle decay toward homeostasis
        arousal *= 0.97;
        valence *= 0.98;
        dopamine *= 0.98;

        // Normalize harmonies
        let h_sum: f32 = harmony_activations.iter().sum();
        if h_sum > 0.0 {
            let target_mean = 0.5;
            for h in &mut harmony_activations {
                *h = *h / h_sum * 8.0 * target_mean;
                *h = h.clamp(0.1, 0.95);
            }
        }
    }

    // ── Summary ──────────────────────────────────────────────────────
    println!();
    println!("━━━ EVOLUTION SUMMARY ━━━");
    println!();

    let first_5_avg: f32 = scores[..5].iter().sum::<f32>() / 5.0;
    let last_5_avg: f32 = scores[scores.len() - 5..].iter().sum::<f32>() / 5.0;
    let improvement = (last_5_avg - first_5_avg) / first_5_avg * 100.0;

    println!("  First 5 avg score:  {:.3}", first_5_avg);
    println!("  Last 5 avg score:   {:.3}", last_5_avg);
    println!("  Improvement:        {:+.1}%", improvement);
    println!("  Final EMA:          {:.3}", tracker.expectation());
    println!("  Total evaluations:  {}", tracker.evaluation_count());

    let weights = taste.weights();
    println!("  Learned taste weights:");
    println!(
        "    order={:.3}  complexity={:.3}  harmony={:.3}  birkhoff={:.3}",
        weights[0], weights[1], weights[2], weights[3]
    );

    let avg_novelty: f32 = novelties.iter().sum::<f32>() / novelties.len() as f32;
    println!("  Average novelty:    {:.3}", avg_novelty);

    println!();
    println!("  Output: {} ({} SVGs)", output_dir.display(), total_cycles);
    println!("  Open cycle_000.svg through cycle_029.svg to see the evolution!");
    println!();
}
