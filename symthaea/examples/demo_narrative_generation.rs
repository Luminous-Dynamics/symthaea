// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end demo: Narrative Algebra -> Dynamics -> Compiler -> [LLM]
//!
//! Demonstrates the full narrative generation pipeline using a 5-scene
//! Hero's Journey arc. Each scene shows the Ghost Signal values and
//! compiled prompt. Optionally calls an LLM backend if available.
//!
//! Run: `cargo run --example demo_narrative_generation`

use symthaea::dynamics::narrative_dynamics::NarrativeSignal;
use symthaea::dynamics::story_session::StorySession;
use symthaea::hdc::narrative_algebra::NarrativeMood;
use symthaea::language::narrative_compiler::{
    NarrativeThought, PointOfView, TargetLength, Tense, generate_narrative,
};

fn print_signal(label: &str, signal: &NarrativeSignal) {
    println!("  {} Ghost Signal:", label);
    println!(
        "    energy={:.3}  surprise={:.3}  valence={:.3}  tension={:.3}  momentum={:.3}",
        signal.energy, signal.surprise, signal.valence, signal.tension, signal.momentum,
    );
    println!("    arc_phase={:?}", signal.arc_phase);
}

#[tokio::main]
async fn main() {
    println!("=== Symthaea Narrative Generation Demo ===\n");

    // --- Set up the story session ---
    let mut session = StorySession::new();

    // Register characters
    let protagonist_prim = session.algebra().primitives.protagonist.clone();
    let antagonist_prim = session.algebra().primitives.antagonist.clone();
    session.register_character("Kael", &protagonist_prim);
    session.register_character("Thira", &antagonist_prim);
    session.add_theme("The cost of knowledge");

    session.introduce_conflict("The Cipher threatens to unravel reality");

    // --- Define 5 Hero's Journey scenes ---
    #[allow(clippy::type_complexity)]
    let scenes: Vec<(&str, &str, &[&str], &str, NarrativeMood, &str)> = vec![
        (
            "The Ordinary World",
            "a quiet village at dawn",
            &["Kael"],
            "restless dreams of symbols",
            NarrativeMood::Peaceful,
            "Kael wakes from troubled sleep, sensing something has changed",
        ),
        (
            "The Call to Adventure",
            "the edge of the Whispering Forest",
            &["Kael", "Thira"],
            "Thira's warning about the Cipher",
            NarrativeMood::Mysterious,
            "Thira appears with a dire warning about the Cipher's awakening",
        ),
        (
            "The Ordeal",
            "a crumbling library deep underground",
            &["Kael"],
            "the Cipher's guardians attack",
            NarrativeMood::Tense,
            "Kael faces the guardians alone in the library's heart",
        ),
        (
            "The Reward",
            "the chamber of the Cipher itself",
            &["Kael"],
            "the Cipher reveals a terrible truth",
            NarrativeMood::Melancholy,
            "Kael deciphers the code but learns its cost",
        ),
        (
            "The Return",
            "the village, now changed",
            &["Kael", "Thira"],
            "choosing what to do with the knowledge",
            NarrativeMood::Hopeful,
            "Kael returns bearing the weight of what was learned",
        ),
    ];

    let mut signals: Vec<NarrativeSignal> = Vec::new();

    // Try to create an LLM backend (graceful fallback)
    let backend = symthaea::language::llm_backend::SimulatedBackend;

    for (title, setting, characters, conflict, mood, scene_goal) in &scenes {
        println!("--- Scene: {} ---", title);

        let signal = session.add_scene(title, setting, characters, conflict, *mood);
        print_signal(title, &signal);

        // Build a NarrativeThought for the compiler
        let thought = NarrativeThought {
            characters: characters
                .iter()
                .map(|&c| {
                    let role = if c == "Kael" {
                        "protagonist"
                    } else {
                        "antagonist"
                    };
                    (c.to_string(), role.to_string())
                })
                .collect(),
            setting: setting.to_string(),
            scene_goal: scene_goal.to_string(),
            theme: "The cost of knowledge".to_string(),
            signal: signal.clone(),
            pov: PointOfView::ThirdLimited,
            tense: Tense::Past,
            target_length: TargetLength::Paragraph,
            style_notes: vec!["Sparse, atmospheric".to_string()],
        };

        // Generate (uses SimulatedBackend for offline demo)
        let output = generate_narrative(&thought, Some(&backend)).await;
        println!("  LLM used: {}", output.used_llm());
        println!(
            "  Prompt (first 200 chars): {}...",
            &output.prompt[..output.prompt.len().min(200)]
        );
        println!(
            "  Prose (first 200 chars): {}...\n",
            &output.prose[..output.prose.len().min(200)]
        );

        signals.push(signal);
    }

    // --- Summary ---
    println!("=== Signal Trajectory Summary ===");
    println!(
        "{:<25} {:>7} {:>7} {:>7} {:>7} {:>7} {:?}",
        "Scene", "Energy", "Surpr.", "Valence", "Tension", "Moment.", "Phase"
    );
    println!("{}", "-".repeat(90));
    for (i, sig) in signals.iter().enumerate() {
        let title = scenes[i].0;
        println!(
            "{:<25} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:?}",
            title, sig.energy, sig.surprise, sig.valence, sig.tension, sig.momentum, sig.arc_phase,
        );
    }

    let state = session.get_story_state();
    println!(
        "\nStory state: {} scenes, {} characters, {} conflicts ({} unresolved)",
        state.total_scenes,
        state.character_count,
        state.total_conflicts,
        state.unresolved_conflicts
    );

    println!("\nDone.");
}