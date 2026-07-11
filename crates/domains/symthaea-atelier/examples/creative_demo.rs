// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Creative Demo
//!
//! End-to-end demonstration of Symthaea's art creation system.
//! Generates visual SVG artwork, musical WAV audio, and reports aesthetic scores.
//!
//! Run:
//!   cargo run -p symthaea-atelier --example creative_demo
//!
//! Output files are written to `/tmp/symthaea-creative-demo/`

use symthaea_atelier::{AtelierConfig, AtelierStyle};
use symthaea_canvas::CognitiveSnapshot;

fn main() {
    let output_dir = std::path::PathBuf::from("/tmp/symthaea-creative-demo");
    std::fs::create_dir_all(&output_dir).expect("create output dir");
    std::fs::create_dir_all(output_dir.join("visual")).expect("create visual dir");
    std::fs::create_dir_all(output_dir.join("music")).expect("create music dir");

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║    Symthaea Creative Art Demo — Consciousness → Art     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Cognitive States ─────────────────────────────────────────────
    let states = [
        ("serene", make_serene_state()),
        ("turbulent", make_turbulent_state()),
        ("playful", make_playful_state()),
        ("contemplative", make_contemplative_state()),
    ];

    // ── Visual Art Generation ────────────────────────────────────────
    println!("━━━ VISUAL ART ━━━");
    println!();

    for (name, snapshot) in &states {
        println!("  State: {name}");
        println!(
            "    Ψ = {:.2}, valence = {:.2}, arousal = {:.2}",
            snapshot.consciousness_level, snapshot.valence, snapshot.arousal
        );

        let styles = [
            ("lsystem", AtelierStyle::LSystem),
            ("curves", AtelierStyle::ParametricCurve),
            ("persistence", AtelierStyle::PersistenceTexture),
            ("colorfield", AtelierStyle::ColorField),
            ("composite", AtelierStyle::Composite),
        ];

        for (style_name, style) in &styles {
            let config = AtelierConfig {
                style: *style,
                iteration_budget: 5,
                ..AtelierConfig::default()
            };

            let artwork = symthaea_atelier::create_artwork_iterative(&config, snapshot, 42);

            let filename = format!("{name}_{style_name}.svg");
            let path = output_dir.join("visual").join(&filename);
            std::fs::write(&path, &artwork.svg).expect("write SVG");

            let s = &artwork.aesthetic_score;
            println!(
                "    {style_name:12} → score: {:.3} (order={:.2} complexity={:.2} harmony={:.2} birkhoff={:.2}) [{} nodes, {} cycles]",
                s.composite,
                s.order,
                s.complexity,
                s.harmony,
                s.birkhoff,
                artwork.scene.node_count(),
                artwork.generation_cycles
            );
        }
        println!();
    }

    // ── Music Generation ─────────────────────────────────────────────
    println!("━━━ MUSIC ━━━");
    println!();

    for (name, snapshot) in &states {
        let musical_state = symthaea_muse::MusicalState {
            harmony_activations: snapshot.harmony_activations,
            dopamine: snapshot.dopamine,
            serotonin: snapshot.serotonin,
            noradrenaline: snapshot.noradrenaline,
            arousal: snapshot.arousal,
            valence: snapshot.valence,
            consciousness_level: snapshot.consciousness_level as f32,
            prediction_error: snapshot.prediction_error,
        };

        let config = symthaea_muse::MuseConfig {
            duration_secs: 6.0,
            max_notes: 24,
            // This demo's WAV writer is mono 16-bit
            output_format: symthaea_muse::OutputFormat::Mono16,
            ..Default::default()
        };

        let composition = symthaea_muse::compose(&config, &musical_state, 42);

        // Write WAV
        let filename = format!("{name}.wav");
        let path = output_dir.join("music").join(&filename);
        let symthaea_muse::AudioData::I16(ref samples) = composition.audio else {
            unreachable!("Mono16 was requested above");
        };
        write_wav(&path, samples, composition.sample_rate);

        // Compute scale info
        let scale = symthaea_muse::pitch::build_scale(&musical_state);
        let tempo = symthaea_muse::rhythm::compute_tempo(&config, &musical_state);
        let section = symthaea_muse::structure::determine_section(&musical_state);

        println!(
            "  {name}: {:.1}s, {} notes, {:.0} BPM, {:?}, {} scale tones",
            composition.duration_secs,
            composition.notes.len(),
            tempo,
            section,
            scale.len()
        );

        // Note range
        if !composition.notes.is_empty() {
            let min_hz = composition
                .notes
                .iter()
                .map(|n| n.frequency)
                .fold(f32::INFINITY, f32::min);
            let max_hz = composition
                .notes
                .iter()
                .map(|n| n.frequency)
                .fold(0.0f32, f32::max);
            println!("    pitch range: {:.0}–{:.0} Hz", min_hz, max_hz);
        }
    }
    println!();

    // ── Gallery Demo ─────────────────────────────────────────────────
    println!("━━━ GALLERY ━━━");
    println!();

    let mut gallery = symthaea_gallery::GalleryIndex::new(100);

    for (name, snapshot) in &states {
        for style in [
            AtelierStyle::Composite,
            AtelierStyle::LSystem,
            AtelierStyle::ParametricCurve,
        ] {
            let config = AtelierConfig {
                style,
                iteration_budget: 3,
                ..AtelierConfig::default()
            };
            let artwork = symthaea_atelier::create_artwork_iterative(&config, snapshot, 42);

            let entry = symthaea_gallery::create_entry(
                symthaea_gallery::ArtModality::Visual {
                    filename: format!("{name}_{style:?}.svg"),
                },
                artwork.aesthetic_score,
                snapshot.harmony_activations,
                snapshot.cycle_count,
            );
            gallery.add(entry);
        }
    }

    println!("  {} artworks in gallery", gallery.len());
    println!("  average score: {:.3}", gallery.average_score());

    let top = gallery.top_n(3);
    println!("  top 3:");
    for entry in &top {
        println!(
            "    {:.3} — {:?} [{}]",
            entry.aesthetic_score.composite,
            entry.modality,
            entry.tags.join(", ")
        );
    }

    // Evolution analysis
    let evo = symthaea_gallery::evolution::analyze_evolution(&gallery, 5);
    println!("  style periods: {}", evo.periods.len());
    for period in &evo.periods {
        println!(
            "    {} (cycles {}-{}): {:.3} avg, {} artworks",
            period.harmony_name,
            period.start_cycle,
            period.end_cycle,
            period.average_score,
            period.artwork_count
        );
    }
    println!("  score trend: {:.4}", evo.score_trend);

    // ── Aesthetic Tracker Demo ────────────────────────────────────────
    println!();
    println!("━━━ AESTHETIC FEEDBACK LOOP ━━━");
    println!();

    let mut tracker = symthaea_aesthetic::AestheticTracker::default();
    let harmonies = [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2];

    for (name, snapshot) in &states {
        let config = AtelierConfig {
            style: AtelierStyle::Composite,
            iteration_budget: 3,
            ..AtelierConfig::default()
        };
        let artwork = symthaea_atelier::create_artwork_iterative(&config, snapshot, 42);
        let feedback = tracker.process(&artwork.aesthetic_score, &harmonies);

        println!(
            "  {name}: score={:.3}, EMA={:.3}, DA={:+.4}, 5-HT={:+.4}, surprise={:.4}",
            artwork.aesthetic_score.composite,
            tracker.expectation(),
            feedback.dopamine_delta,
            feedback.serotonin_delta,
            feedback.surprise_signal
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Output: {}", output_dir.display());
    println!(
        "  Visual: {}/visual/ ({} SVG files)",
        output_dir.display(),
        states.len() * 5
    );
    println!(
        "  Music:  {}/music/ ({} WAV files)",
        output_dir.display(),
        states.len()
    );
    println!("═══════════════════════════════════════════════════════════");
}

// ── Cognitive State Constructors ─────────────────────────────────────────

fn make_serene_state() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.85,
        prediction_error: 0.05,
        dopamine: 0.5,
        serotonin: 0.8,
        noradrenaline: 0.2,
        acetylcholine: 0.4,
        oxytocin: 0.6,
        gaba: 0.4,
        valence: 0.6,
        arousal: 0.3,
        harmony_activations: [0.7, 0.8, 0.6, 0.3, 0.7, 0.6, 0.5, 0.8],
        betti_0: 4,
        betti_1: 2,
        betti_2: 1,
        persistence_components: vec![[0.0, 0.7], [0.1, 0.9], [0.2, 0.5]],
        persistence_cycles: vec![[0.1, 0.6], [0.3, 0.8]],
        thought_vector: vec![
            0.3, -0.1, 0.5, 0.2, -0.3, 0.1, 0.4, -0.2, 0.1, 0.0, -0.1, 0.3, 0.2, -0.4, 0.1, 0.0,
        ],
        cantor_metacognitive_depth: 0.7,
        cantor_last_depth: 3,
        cycle_count: 100,
        ..CognitiveSnapshot::dormant()
    }
}

fn make_turbulent_state() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.6,
        prediction_error: 0.7,
        dopamine: 0.3,
        serotonin: 0.2,
        noradrenaline: 0.9,
        acetylcholine: 0.7,
        oxytocin: 0.2,
        gaba: 0.2,
        valence: -0.6,
        arousal: 0.9,
        harmony_activations: [0.3, 0.2, 0.4, 0.9, 0.5, 0.2, 0.8, 0.1],
        betti_0: 7,
        betti_1: 3,
        betti_2: 2,
        persistence_components: vec![[0.0, 0.3], [0.05, 0.4], [0.1, 0.6], [0.2, 0.9]],
        persistence_cycles: vec![[0.1, 0.5], [0.2, 0.7], [0.4, 0.8]],
        thought_vector: vec![
            0.8, -0.7, 0.9, -0.5, 0.6, -0.8, 0.3, 0.7, -0.4, 0.6, -0.9, 0.2, 0.7, -0.3, 0.8, -0.6,
        ],
        cantor_metacognitive_depth: 0.5,
        cantor_last_depth: 4,
        cycle_count: 200,
        ..CognitiveSnapshot::dormant()
    }
}

fn make_playful_state() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.75,
        prediction_error: 0.3,
        dopamine: 0.8,
        serotonin: 0.5,
        noradrenaline: 0.5,
        acetylcholine: 0.3,
        oxytocin: 0.5,
        gaba: 0.3,
        valence: 0.4,
        arousal: 0.7,
        harmony_activations: [0.4, 0.5, 0.3, 0.95, 0.6, 0.4, 0.7, 0.1],
        betti_0: 5,
        betti_1: 2,
        betti_2: 1,
        persistence_components: vec![[0.0, 0.6], [0.15, 0.8]],
        persistence_cycles: vec![[0.2, 0.7]],
        thought_vector: vec![
            0.5, 0.3, -0.2, 0.6, 0.1, -0.5, 0.4, 0.2, -0.1, 0.3, 0.6, -0.2, 0.1, 0.4, -0.3, 0.5,
        ],
        cantor_metacognitive_depth: 0.6,
        cantor_last_depth: 3,
        cycle_count: 300,
        ..CognitiveSnapshot::dormant()
    }
}

fn make_contemplative_state() -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: 0.9,
        prediction_error: 0.02,
        dopamine: 0.4,
        serotonin: 0.7,
        noradrenaline: 0.1,
        acetylcholine: 0.6,
        oxytocin: 0.5,
        gaba: 0.6,
        valence: 0.2,
        arousal: 0.15,
        harmony_activations: [0.8, 0.6, 0.9, 0.2, 0.5, 0.7, 0.4, 0.95],
        betti_0: 2,
        betti_1: 1,
        betti_2: 0,
        persistence_components: vec![[0.0, 0.9]],
        persistence_cycles: vec![[0.1, 0.8]],
        thought_vector: vec![
            0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.1, 0.0, 0.1, 0.0, -0.1,
        ],
        cantor_metacognitive_depth: 0.9,
        cantor_last_depth: 5,
        cycle_count: 400,
        ..CognitiveSnapshot::dormant()
    }
}

// ── WAV Writer ───────────────────────────────────────────────────────────

fn write_wav(path: &std::path::Path, samples: &[i16], sample_rate: u32) {
    let data_len = (samples.len() * 2) as u32;
    let file_len = 36 + data_len;
    let mut buf = Vec::with_capacity(44 + samples.len() * 2);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_len.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }

    std::fs::write(path, buf).expect("write WAV");
}
