// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Emotion + Attention Primitive Composition Demo
//!
//! Demonstrates how emotions (Plutchik's 8) and attention modes (Graziano's AST)
//! can compose together using NSM (Natural Semantic Metalanguage) primitives.
//!
//! ## Key Insights
//!
//! Both emotions and attention are grounded in the same NSM primitives:
//! - Emotions: joy = FEEL + GOOD + VERY, fear = FEEL + BAD + MAYBE + HAPPEN
//! - Attention: focused = SEE + THIS + VERY, vigilant = SEE + WANT + MAYBE + HAPPEN
//!
//! This shared grounding enables cross-domain composition via HDC binding (XOR).
//!
//! ## Run with:
//! ```bash
//! cargo run --example emotion_attention_composition
//! ```

use std::collections::HashMap;
use symthaea::consciousness::attention_schema::{
    AttentionMode, AttentionPrimitiveGrounding, AttentionSchema,
};
use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::language::emotional_core::{
    EmotionPrimitiveGrounding, EmotionalCore, EmotionalCoreConfig,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("    EMOTION + ATTENTION PRIMITIVE COMPOSITION SHOWCASE");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 1: Display the NSM groundings for emotions and attention
    // ═══════════════════════════════════════════════════════════════════════════

    println!("PART 1: NSM Primitive Groundings\n");

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  PLUTCHIK'S 8 EMOTIONS (grounded in NSM)                        │");
    println!("├─────────────────────────────────────────────────────────────────┤");

    let emotions = EmotionPrimitiveGrounding::plutchik_emotions();
    let mut emotion_names: Vec<_> = emotions.keys().collect();
    emotion_names.sort();

    for name in &emotion_names {
        let grounding = &emotions[*name];
        println!("│  {:12} = {} ", name, grounding.nsm_primitives.join(" + "));
        println!(
            "│               valence={:+.1}, arousal={:.1}",
            grounding.valence_weight, grounding.arousal_weight
        );
    }
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  GRAZIANO'S 7 ATTENTION MODES (grounded in NSM)                 │");
    println!("├─────────────────────────────────────────────────────────────────┤");

    let attention_modes = [
        AttentionMode::Focused,
        AttentionMode::Divided,
        AttentionMode::Diffuse,
        AttentionMode::Vigilant,
        AttentionMode::Scanning,
        AttentionMode::Reflexive,
        AttentionMode::Inhibited,
    ];

    for mode in &attention_modes {
        let grounding = AttentionPrimitiveGrounding::for_mode(*mode);
        println!(
            "│  {:10?} = {} ",
            mode,
            grounding.primitive_components.join(" + ")
        );
        println!("│              salience={:.2}", grounding.salience);
    }
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 2: Emotion similarity via primitive encodings
    // ═══════════════════════════════════════════════════════════════════════════

    println!("PART 2: Emotion Similarity via Primitive Encodings\n");

    let emotional_core = EmotionalCore::new(EmotionalCoreConfig::default());

    // Calculate similarity matrix for emotions
    println!("Similarity matrix (emotions sharing NSM primitives are more similar):\n");

    print!("{:12}", "");
    for name in &emotion_names {
        print!("{:>10}", &name[..name.len().min(8)]);
    }
    println!();

    for e1 in &emotion_names {
        print!("{:12}", e1);
        for e2 in &emotion_names {
            let sim = emotional_core.primitive_similarity(e1, e2);
            let indicator = if sim > 0.6 {
                "●"
            } else if sim > 0.4 {
                "◐"
            } else {
                "○"
            };
            print!("{:>9}{}", format!("{:.2}", sim), indicator);
        }
        println!();
    }

    println!("\n● = high similarity (>0.6)  ◐ = moderate (>0.4)  ○ = low (<0.4)\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 3: Attention mode similarity
    // ═══════════════════════════════════════════════════════════════════════════

    println!("PART 3: Attention Mode Similarity\n");

    let attention_schema = AttentionSchema::new();

    println!("Comparing attention modes using primitive encodings:");
    println!("(Modes sharing primitives like NSM_SEE will be more similar)\n");

    for m1 in &attention_modes {
        for m2 in &attention_modes {
            if *m1 as usize <= *m2 as usize {
                let sim = attention_schema.mode_similarity(*m1, *m2);
                if *m1 != *m2 {
                    println!("  {:10?} <-> {:10?}: {:.3}", m1, m2, sim);
                }
            }
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 4: Cross-domain composition - Emotion ⊗ Attention
    // ═══════════════════════════════════════════════════════════════════════════

    println!("PART 4: Cross-Domain Composition (Emotion ⊗ Attention)\n");

    println!("Composing emotional states with attention modes via HDC binding (XOR):");
    println!("This creates unique representations for 'how we attend while feeling'\n");

    // Select interesting emotion-attention pairs
    let pairs = [
        (
            "joy",
            AttentionMode::Focused,
            "Joyful focused attention - flow state",
        ),
        (
            "fear",
            AttentionMode::Vigilant,
            "Fearful vigilance - hyperawareness",
        ),
        (
            "anger",
            AttentionMode::Reflexive,
            "Angry reflexive response - rage",
        ),
        (
            "sadness",
            AttentionMode::Diffuse,
            "Sad diffuse attention - rumination",
        ),
        (
            "anticipation",
            AttentionMode::Scanning,
            "Anticipatory scanning - expectation",
        ),
        (
            "trust",
            AttentionMode::Focused,
            "Trusting focused attention - engagement",
        ),
        (
            "surprise",
            AttentionMode::Reflexive,
            "Surprised capture - startle response",
        ),
        (
            "disgust",
            AttentionMode::Inhibited,
            "Disgust with inhibition - avoidance",
        ),
    ];

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│  COMPOSED STATES                                                        │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");

    let mut composed_states: HashMap<String, BinaryHV> = HashMap::new();

    for (emotion, mode, description) in &pairs {
        let emotion_encoding = emotions.get(*emotion).unwrap().primitive_encoding;
        let attention_encoding = AttentionPrimitiveGrounding::for_mode(*mode).mode_encoding;

        // Compose via binding (XOR) - this creates a unique representation
        let composed = emotion_encoding.bind(&attention_encoding);

        let emotion_grounding = &emotions[*emotion];
        let attention_grounding = AttentionPrimitiveGrounding::for_mode(*mode);

        // Compute combined valence and salience
        let effective_valence = emotion_grounding.valence_weight;
        let effective_salience =
            attention_grounding.salience * (1.0 + emotion_grounding.arousal_weight) / 2.0;

        let state_name = format!("{}_{:?}", emotion, mode);
        composed_states.insert(state_name.clone(), composed);

        println!("│  {} + {:?}", emotion, mode);
        println!("│    => {}", description);
        println!(
            "│    valence={:+.2}, salience={:.2}, popcount={}",
            effective_valence,
            effective_salience,
            composed.popcount()
        );
        println!("│");
    }
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 5: Composed state similarity
    // ═══════════════════════════════════════════════════════════════════════════

    println!("PART 5: Similarity Between Composed States\n");

    println!("Finding semantically related emotion-attention compositions:");
    println!("(High similarity = shared primitive structure)\n");

    let mut similarities: Vec<(String, String, f32)> = Vec::new();

    let state_names: Vec<_> = composed_states.keys().cloned().collect();
    for (i, s1) in state_names.iter().enumerate() {
        for s2 in state_names.iter().skip(i + 1) {
            let sim = composed_states[s1].similarity(&composed_states[s2]);
            similarities.push((s1.clone(), s2.clone(), sim));
        }
    }

    // Sort by similarity
    similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("Top 5 most similar composed states:");
    for (s1, s2, sim) in similarities.iter().take(5) {
        println!("  {} <-> {}", s1, s2);
        println!("    similarity: {:.4}", sim);
    }

    println!("\nBottom 5 least similar (most distinct) composed states:");
    for (s1, s2, sim) in similarities.iter().rev().take(5) {
        println!("  {} <-> {}", s1, s2);
        println!("    similarity: {:.4}", sim);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 6: Primitive decomposition
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n");
    println!("PART 6: Primitive Decomposition Analysis\n");

    println!("Shared primitives between emotions and attention modes:");
    println!("(This explains why some compositions 'make sense' together)\n");

    // Find emotions and attention modes that share primitives
    for (emotion_name, emotion_grounding) in &emotions {
        for mode in &attention_modes {
            let attention_grounding = AttentionPrimitiveGrounding::for_mode(*mode);

            // Find shared primitives
            let shared: Vec<_> = emotion_grounding
                .nsm_primitives
                .iter()
                .filter(|p| attention_grounding.primitive_components.contains(p))
                .collect();

            if !shared.is_empty() {
                println!("  {} + {:?}:", emotion_name, mode);
                println!("    shared primitives: {:?}", shared);

                // Calculate binding similarity boost from shared primitives
                let base_sim = emotion_grounding
                    .primitive_encoding
                    .similarity(&attention_grounding.mode_encoding);
                println!("    base similarity: {:.3}", base_sim);
                println!();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("═══════════════════════════════════════════════════════════════════");
    println!("    SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("This demo showed how the unified primitive system enables:");
    println!();
    println!("  1. GROUNDING: Both emotions and attention are grounded in 65 NSM primes");
    println!("     - Emotions use: FEEL, GOOD, BAD, WANT, KNOW, HAPPEN, etc.");
    println!("     - Attention uses: SEE, THIS, VERY, MOVE, WHERE, etc.");
    println!();
    println!("  2. COMPOSITION: Emotion ⊗ Attention creates unique states via HDC binding");
    println!("     - \"Joyful focused attention\" = joy ⊗ Focused");
    println!("     - \"Fearful vigilance\" = fear ⊗ Vigilant");
    println!();
    println!("  3. SIMILARITY: Shared primitives create semantic relationships");
    println!("     - States sharing more primitives are more similar");
    println!("     - This enables analogical reasoning across domains");
    println!();
    println!("  4. INTERPRETABILITY: Every composed state traces back to NSM primes");
    println!("     - We know exactly what primitives contribute to each state");
    println!("     - This provides symbolic grounding for subsymbolic representations");
    println!();
    println!("Total primitive system: 65 NSM primes + 219 ontological primitives = 284 primitives");
    println!();
}