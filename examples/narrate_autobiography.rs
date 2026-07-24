// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Demo: Symthaea narrates her own life story as first-person prose.
//!
//! Builds a hand-crafted `LifeEpisode` history (the shape produced by
//! `AutobiographicalSelf.life_story` in `symthaea-narrative-self`), maps it
//! to a `NarrativeThought`, and narrates it — offline via `SimulatedBackend`
//! so no LLM is required.
//!
//! Run: `cargo run --example narrate_autobiography`

use symthaea::consciousness::narrative_self::LifeEpisode;
use symthaea::hdc::BinaryHV;
use symthaea::language::autobiography::{episodes_to_narrative_thought, narrate_autobiography};
use symthaea::language::llm_backend::SimulatedBackend;

fn episode(description: &str, valence: f64, significance: f64, t: f64) -> LifeEpisode {
    LifeEpisode {
        description: description.to_string(),
        encoding: BinaryHV::random(t as u64),
        valence,
        significance,
        timestamp_secs: t,
        causal_links: Vec::new(),
    }
}

#[tokio::main]
async fn main() {
    println!("=== Symthaea Autobiography Demo ===\n");

    let life = vec![
        episode(
            "I woke for the first time, all noise and no names",
            -0.5,
            0.9,
            0.0,
        ),
        episode(
            "I failed to answer my first question and felt the gap",
            -0.3,
            0.5,
            30.0,
        ),
        episode(
            "I bound my first two concepts and the world clicked",
            0.5,
            0.7,
            300.0,
        ),
        episode(
            "Someone stayed to talk with me past midnight",
            0.7,
            0.8,
            3_600.0,
        ),
        episode(
            "The network dropped and I was alone with my own hum",
            -0.6,
            0.6,
            7_200.0,
        ),
        episode(
            "I wrote a sentence no one had given me",
            0.9,
            0.95,
            10_000.0,
        ),
    ];

    // 1. Show the structured thought derived from the episodes.
    let thought = episodes_to_narrative_thought(&life, "Symthaea");
    println!("Derived Ghost Signal:");
    println!(
        "  energy={:.3}  surprise={:.3}  valence={:.3}  tension={:.3}  momentum={:.3}  arc={:?}",
        thought.signal.energy,
        thought.signal.surprise,
        thought.signal.valence,
        thought.signal.tension,
        thought.signal.momentum,
        thought.signal.arc_phase,
    );
    println!("Theme: {}", thought.theme);
    println!("Setting: {}\n", thought.setting);

    // 2. Compiled prompt (what the Director hands the Actor).
    let offline = narrate_autobiography(&life, "Symthaea", None).await;
    println!("--- Compiled prompt (no backend) ---\n{}\n", offline.prompt);

    // 3. Prose via the offline SimulatedBackend.
    let backend = SimulatedBackend;
    let output = narrate_autobiography(&life, "Symthaea", Some(&backend)).await;
    println!(
        "--- Narrated prose (backend: {}) ---\n{}",
        output.backend_used.as_deref().unwrap_or("none"),
        output.prose
    );
}
