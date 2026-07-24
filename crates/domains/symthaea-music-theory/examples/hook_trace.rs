// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generalized version of the one-off `clip08_hook_trace`: prints exactly
//! which (hook_rhythm, hook_contour) pair `HookCell::generate_with` picks
//! for a given style/seed/arousal, the resulting `graft_hook` splice in
//! raw degree terms, and the style's own hook/motif banks for comparison
//! -- so a measured melodic anomaly can be traced to its real cause.
//!
//! Run: `cargo run --example hook_trace -p symthaea-music-theory -- <Style> <seed> <arousal>`

use symthaea_music_theory::{HookCell, Style, graft_hook};

fn parse_style(s: &str) -> Style {
    match s {
        "Tango" => Style::Tango,
        "Nocturne" => Style::Nocturne,
        "March" => Style::March,
        "Blues" => Style::Blues,
        "Minimalism" => Style::Minimalism,
        "Flamenco" => Style::Flamenco,
        "SacredChoral" => Style::SacredChoral,
        "Ambient" => Style::Ambient,
        other => panic!("unknown style {other:?} (add it to hook_trace.rs's parse_style)"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let style = parse_style(
        args.get(1)
            .expect("usage: hook_trace <Style> <seed> <arousal>"),
    );
    let seed: u64 = args
        .get(2)
        .expect("need seed")
        .parse()
        .expect("seed must be a u64");
    let arousal: f32 = args
        .get(3)
        .expect("need arousal")
        .parse()
        .expect("arousal must be a float");

    let spec = style.spec();
    let meter_beats = spec.meter as f64;

    let hook = HookCell::generate_with(&spec.melody, seed, meter_beats);
    println!("{style:?} seed={seed} arousal={arousal}\n");
    println!("Picked hook (raw degrees):");
    for (deg, dur) in &hook.notes {
        println!("  degree={deg:3}  duration={:.3} beats", dur.beats());
    }

    let motif = spec.motif(arousal, seed);
    println!("\nspec.motif(arousal={arousal}, seed={seed}) raw degrees (the grafting template):");
    for n in &motif.notes {
        println!(
            "  degree={:?}  duration={:.3} beats",
            n.degree,
            n.duration.beats()
        );
    }

    let grafted = graft_hook(&motif, &hook, meter_beats);
    println!("\ngraft_hook(template, hook, {meter_beats}) result (one bar, raw degrees):");
    for n in &grafted.notes {
        println!(
            "  degree={:?}  duration={:.3} beats",
            n.degree,
            n.duration.beats()
        );
    }

    println!("\n{style:?}'s own hook_contours bank:");
    for (i, c) in spec.melody.hook_contours.iter().enumerate() {
        println!("  [{i}] {c:?}");
    }
    println!("{style:?}'s own hook_rhythms bank:");
    for (i, r) in spec.melody.hook_rhythms.iter().enumerate() {
        println!("  [{i}] {r:?}");
    }
}
