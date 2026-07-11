// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Watch Symthaea "listen to" her own music: each round composes AND
//! realizes through the REAL pipeline (theory-crate structure + per-voice
//! instruments + mastering), scores it against `critic::evaluate_composition`,
//! and evolves `MusicalState` from the verdict before the next round.
//!
//! Honesty check (see `critic.rs`'s module doc): the critic is a
//! hand-authored statistical heuristic over note-sequence statistics, not a
//! perceptual/audio-domain judge. A rising trajectory means the loop is
//! doing something directional against these specific proxies -- not proof
//! the music got more beautiful.
//!
//! Run: cargo run --release -p symthaea-muse --features theory \
//!          --example self_critique_demo

use symthaea_muse::MusicalState;
use symthaea_muse::critic::{evaluate_composition, music_auto_improve_theory};
use symthaea_muse::theory_realize::compose_and_realize_styled;
use symthaea_music_theory::{MusicalIntent, PitchClass, Style};

fn main() {
    let intent = MusicalIntent {
        valence: 0.3,
        arousal: 0.4,
        energy: 0.5,
        bars: 4,
        seed: 7,
        tonic: PitchClass::C,
    };

    // Print one round's full verdict breakdown up front so it's clear what
    // "composite" is actually made of.
    let comp =
        compose_and_realize_styled(&intent, Style::Classical, &MusicalState::default(), 44100);
    let v = evaluate_composition(&comp, &MusicalState::default());
    println!(
        "=== One round's verdict breakdown (seed={}) ===\n",
        intent.seed
    );
    println!(
        "melodic_interest    = {:.3}  (pitch variety)",
        v.melodic_interest
    );
    println!(
        "rhythmic_regularity = {:.3}  (inter-onset timing CV)",
        v.rhythmic_regularity
    );
    println!(
        "harmonic_alignment  = {:.3}  (harmony_activations mean)",
        v.harmonic_alignment
    );
    println!(
        "voice_balance       = {:.3}  (velocity spread across notes)",
        v.voice_balance
    );
    println!(
        "form_coherence      = {:.3}  (section pitch-centroid contrast)",
        v.form_coherence
    );
    println!(
        "melodic_contour     = {:.3}  (rise-then-fall arc)",
        v.melodic_contour
    );
    println!(
        "composite           = {:.3}  (weighted sum of the above)\n",
        v.composite
    );

    println!("=== Auto-improve loop: generate -> score -> evolve state -> regenerate ===\n");
    let mut state = MusicalState::default();
    let result = music_auto_improve_theory(
        &intent,
        Style::Classical,
        &mut state,
        44100,
        20,
        intent.seed,
    );
    println!(
        "rounds={} reached_stillness={} best_score={:.3}",
        result.rounds, result.reached_stillness, result.best_score
    );
    println!("per-round composite: {:?}", result.trajectory);
}
