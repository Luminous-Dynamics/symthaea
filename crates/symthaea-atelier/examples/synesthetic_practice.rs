// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Synesthetic Practice Session
//!
//! The system composes music AND generates visual art simultaneously,
//! then checks cross-modal coherence: does the emotional quality of
//! the music match the visual output?
//!
//! This is the Gesamtkunstwerk loop — consciousness driving unified
//! multi-modal art with emotional honesty enforcement.
//!
//! ```bash
//! cargo run -p symthaea-atelier --example synesthetic_practice
//! ```

use symthaea_aesthetic::synesthesia;
use symthaea_atelier::critic::{PerceptualInput, SelfCritic};
use symthaea_atelier::{AtelierConfig, create_artwork};
use symthaea_canvas::CognitiveSnapshot;
use symthaea_muse::critic::evaluate_composition;
use symthaea_muse::{MuseConfig, MusicalState, compose};

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Symthaea Synesthetic Practice");
    println!("  Music + Visual + Cross-Modal Coherence");
    println!("═══════════════════════════════════════════════════════\n");

    let gallery_dir = std::path::Path::new("target/synesthetic-gallery");
    std::fs::create_dir_all(gallery_dir).ok();

    let visual_config = AtelierConfig {
        width: 512.0,
        height: 512.0,
        max_elements: 150,
        iteration_budget: 2,
        ..Default::default()
    };

    let music_config = MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        ..Default::default()
    };

    // Four emotional states to practice in
    let states: Vec<(&str, CognitiveSnapshot)> = vec![
        (
            "serene",
            CognitiveSnapshot {
                consciousness_level: 0.8,
                harmony_activations: [0.3, 0.7, 0.6, 0.2, 0.5, 0.6, 0.3, 0.8],
                serotonin: 0.8,
                oxytocin: 0.7,
                dopamine: 0.3,
                noradrenaline: 0.1,
                allostatic_load: 0.0,
                arousal: 0.2,
                valence: 0.6,
                ..CognitiveSnapshot::dormant()
            },
        ),
        (
            "turbulent",
            CognitiveSnapshot {
                consciousness_level: 0.6,
                harmony_activations: [0.2, 0.2, 0.3, 0.8, 0.4, 0.2, 0.7, 0.1],
                serotonin: 0.2,
                oxytocin: 0.1,
                dopamine: 0.8,
                noradrenaline: 0.9,
                allostatic_load: 0.6,
                arousal: 0.9,
                valence: -0.4,
                ..CognitiveSnapshot::dormant()
            },
        ),
        (
            "contemplative",
            CognitiveSnapshot {
                consciousness_level: 0.9,
                harmony_activations: [0.6, 0.4, 0.8, 0.1, 0.3, 0.4, 0.2, 0.9],
                serotonin: 0.7,
                oxytocin: 0.5,
                dopamine: 0.4,
                noradrenaline: 0.2,
                allostatic_load: 0.1,
                arousal: 0.15,
                valence: 0.1,
                ..CognitiveSnapshot::dormant()
            },
        ),
        (
            "joyful",
            CognitiveSnapshot {
                consciousness_level: 0.85,
                harmony_activations: [0.7, 0.9, 0.5, 0.7, 0.6, 0.8, 0.6, 0.2],
                serotonin: 0.6,
                oxytocin: 0.8,
                dopamine: 0.9,
                noradrenaline: 0.3,
                allostatic_load: 0.0,
                arousal: 0.7,
                valence: 0.8,
                ..CognitiveSnapshot::dormant()
            },
        ),
    ];

    let mut visual_critic = SelfCritic::new();

    for (name, snap) in &states {
        println!("── {} ──────────────────────────────────────", name);
        println!(
            "  Ψ={:.2}  val={:+.2}  aros={:.2}  ser={:.2}  NE={:.2}  allost={:.2}",
            snap.consciousness_level,
            snap.valence,
            snap.arousal,
            snap.serotonin,
            snap.noradrenaline,
            snap.allostatic_load
        );

        // Generate music
        let musical_state = MusicalState {
            harmony_activations: snap.harmony_activations,
            dopamine: snap.dopamine,
            serotonin: snap.serotonin,
            noradrenaline: snap.noradrenaline,
            arousal: snap.arousal,
            valence: snap.valence,
            consciousness_level: snap.consciousness_level as f32,
            prediction_error: 0.1,
        };
        let composition = compose(&music_config, &musical_state, 42);
        let music_verdict = evaluate_composition(&composition, &musical_state);

        // Extract synesthetic features from the music
        let note_tuples: Vec<(f32, f32, f32, f32)> = composition
            .notes
            .iter()
            .map(|n| (n.frequency, n.velocity, n.start_time, n.duration))
            .collect();
        let tempo = symthaea_muse::rhythm::compute_tempo(&music_config, &musical_state);
        let syn_frames = synesthesia::extract_synesthetic_features(
            &note_tuples,
            tempo,
            composition.duration_secs,
        );

        // Generate visual art
        let artwork = create_artwork(&visual_config, snap, 42);
        let visual_features = PerceptualInput {
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
        let visual_verdict = visual_critic.evaluate(&visual_features, snap);

        // Cross-modal coherence check
        let coherence = synesthesia::coherence_score(
            &syn_frames,
            snap.valence,
            snap.arousal,
            artwork.aesthetic_score.complexity,
        );

        // Save SVG
        let svg_path = gallery_dir.join(format!("{name}.svg"));
        std::fs::write(&svg_path, &artwork.svg).ok();

        // Save MIDI
        let midi_path = gallery_dir.join(format!("{name}.mid"));
        composition.to_midi_file(&midi_path, tempo).ok();

        // Report
        println!(
            "\n  Music:  melodic={:.3}  rhythm={:.3}  contour={:.3}  form={:.3}  → {:.3}",
            music_verdict.melodic_interest,
            music_verdict.rhythmic_regularity,
            music_verdict.melodic_contour,
            music_verdict.form_coherence,
            music_verdict.composite
        );
        println!(
            "  Visual: novelty={:.3}  golden={:.3}  taste={:.3}  → {:.3}",
            visual_verdict.novelty,
            visual_verdict.golden_ratio,
            visual_verdict.taste_alignment,
            visual_verdict.composite
        );
        println!("  Cross-modal coherence: {:.3}", coherence);

        let emotionally_honest = coherence > 0.5;
        println!(
            "  Emotionally honest: {}",
            if emotionally_honest {
                "YES"
            } else {
                "NO — mismatch"
            }
        );

        if !syn_frames.is_empty() {
            let avg_hue = syn_frames.iter().map(|f| f.hue).sum::<f32>() / syn_frames.len() as f32;
            let warmth = if avg_hue < 180.0 { "warm" } else { "cool" };
            println!(
                "  Synesthetic: avg_hue={:.0}° ({}) motion={:.2}",
                avg_hue,
                warmth,
                syn_frames.iter().map(|f| f.motion).sum::<f32>() / syn_frames.len() as f32
            );
        }

        println!("  Files: {}.svg + {}.mid", name, name);
        println!(
            "  Notes: {} | Section: {:?}\n",
            composition.notes.len(),
            composition.section
        );
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  Synesthetic gallery: target/synesthetic-gallery/");
    println!("  Open .svg in browser, .mid in any DAW.");
    println!("═══════════════════════════════════════════════════════\n");
}
