// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Visual gallery: writes SVGs for different cognitive states to disk for manual inspection.
//!
//! Run: `cargo test -p symthaea-canvas --test visual_gallery -- --nocapture`
//! Then open `target/canvas-gallery/*.svg` in a browser.

use symthaea_canvas::{CognitiveSnapshot, render_snapshot};

fn gallery_dir() -> std::path::PathBuf {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/canvas-gallery");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn write_svg(name: &str, snap: &CognitiveSnapshot) {
    let (svg, state) = render_snapshot(snap);
    let path = gallery_dir().join(format!("{name}.svg"));
    std::fs::write(&path, &svg).expect("write SVG");
    eprintln!(
        "{name}: Ψ={:.2} lum={:.2} score={:.3} nodes={}bytes",
        snap.consciousness_level,
        state.luminosity,
        state.aesthetic_score,
        svg.len(),
    );
}

#[test]
fn gallery_dormant() {
    write_svg("01-dormant", &CognitiveSnapshot::dormant());
}

#[test]
fn gallery_waking() {
    let mut snap = CognitiveSnapshot::dormant();
    snap.consciousness_level = 0.25;
    snap.dopamine = 0.3;
    snap.serotonin = 0.4;
    snap.betti_0 = 2;
    snap.arousal = 0.2;
    snap.harmony_activations = [0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0];
    snap.cycle_count = 50;
    write_svg("02-waking", &snap);
}

#[test]
fn gallery_alert() {
    let snap = CognitiveSnapshot {
        consciousness_level: 0.7,
        prediction_error: 0.15,
        living_mind_vitality: 0.6,
        living_mind_coherence: 0.55,
        dopamine: 0.6,
        noradrenaline: 0.5,
        serotonin: 0.5,
        acetylcholine: 0.4,
        oxytocin: 0.3,
        gaba: 0.4,
        allostatic_load: 0.15,
        betti_0: 4,
        betti_1: 2,
        betti_2: 0,
        persistence_components: vec![[0.0, 0.5], [0.1, 0.8], [0.2, 0.6]],
        persistence_cycles: vec![[0.3, 0.7]],
        cantor_metacognitive_depth: 0.6,
        cantor_last_depth: 3,
        valence: 0.2,
        arousal: 0.5,
        harmony_activations: [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2],
        thought_vector: vec![0.3, -0.2],
        cycle_count: 500,
    };
    write_svg("03-alert", &snap);
}

#[test]
fn gallery_turbulent() {
    let snap = CognitiveSnapshot {
        consciousness_level: 0.5,
        prediction_error: 0.8,
        living_mind_vitality: 0.4,
        living_mind_coherence: 0.3,
        dopamine: 0.8,
        noradrenaline: 0.9,
        serotonin: 0.2,
        acetylcholine: 0.6,
        oxytocin: 0.1,
        gaba: 0.2,
        allostatic_load: 0.6,
        betti_0: 6,
        betti_1: 3,
        betti_2: 2,
        persistence_components: vec![[0.0, 0.3], [0.1, 0.4], [0.2, 0.5], [0.15, 0.35]],
        persistence_cycles: vec![[0.1, 0.6], [0.2, 0.4]],
        cantor_metacognitive_depth: 0.8,
        cantor_last_depth: 4,
        valence: -0.5,
        arousal: 0.9,
        harmony_activations: [0.9, 0.1, 0.8, 0.2, 0.7, 0.1, 0.6, 0.3],
        thought_vector: vec![0.8, 0.6],
        cycle_count: 1200,
    };
    write_svg("04-turbulent", &snap);
}

#[test]
fn gallery_harmonious() {
    let snap = CognitiveSnapshot {
        consciousness_level: 0.9,
        prediction_error: 0.05,
        living_mind_vitality: 0.85,
        living_mind_coherence: 0.8,
        dopamine: 0.5,
        noradrenaline: 0.3,
        serotonin: 0.7,
        acetylcholine: 0.5,
        oxytocin: 0.8,
        gaba: 0.6,
        allostatic_load: 0.05,
        betti_0: 3,
        betti_1: 2,
        betti_2: 1,
        persistence_components: vec![[0.0, 0.9], [0.1, 0.85]],
        persistence_cycles: vec![[0.2, 0.8]],
        cantor_metacognitive_depth: 0.7,
        cantor_last_depth: 4,
        valence: 0.6,
        arousal: 0.4,
        harmony_activations: [0.8, 0.85, 0.75, 0.9, 0.8, 0.85, 0.7, 0.8],
        thought_vector: vec![0.1, -0.1],
        cycle_count: 3000,
    };
    write_svg("05-harmonious", &snap);
}

#[test]
fn gallery_dreaming() {
    let snap = CognitiveSnapshot {
        consciousness_level: 0.3,
        prediction_error: 0.4,
        living_mind_vitality: 0.3,
        living_mind_coherence: 0.2,
        dopamine: 0.2,
        noradrenaline: 0.1,
        serotonin: 0.6,
        acetylcholine: 0.2,
        oxytocin: 0.4,
        gaba: 0.7,
        allostatic_load: 0.1,
        betti_0: 2,
        betti_1: 1,
        betti_2: 1,
        persistence_components: vec![[0.0, 0.6]],
        persistence_cycles: vec![[0.1, 0.5]],
        cantor_metacognitive_depth: 0.3,
        cantor_last_depth: 2,
        valence: 0.1,
        arousal: 0.15,
        harmony_activations: [0.2, 0.1, 0.3, 0.1, 0.2, 0.4, 0.1, 0.5],
        thought_vector: vec![-0.5, 0.4],
        cycle_count: 8000,
    };
    write_svg("06-dreaming", &snap);
}

#[test]
fn gallery_deterministic() {
    let snap = CognitiveSnapshot {
        consciousness_level: 0.6,
        prediction_error: 0.2,
        living_mind_vitality: 0.5,
        living_mind_coherence: 0.5,
        dopamine: 0.5,
        noradrenaline: 0.4,
        serotonin: 0.5,
        acetylcholine: 0.4,
        oxytocin: 0.3,
        gaba: 0.5,
        allostatic_load: 0.2,
        betti_0: 3,
        betti_1: 1,
        betti_2: 0,
        persistence_components: vec![[0.0, 0.5]],
        persistence_cycles: vec![],
        cantor_metacognitive_depth: 0.5,
        cantor_last_depth: 2,
        valence: 0.0,
        arousal: 0.4,
        harmony_activations: [0.5; 8],
        thought_vector: vec![0.0, 0.0],
        cycle_count: 200,
    };
    let (svg1, _) = render_snapshot(&snap);
    let (svg2, _) = render_snapshot(&snap);
    assert_eq!(svg1, svg2, "Same snapshot must produce identical SVG");
}
