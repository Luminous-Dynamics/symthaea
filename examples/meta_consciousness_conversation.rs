// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal meta-consciousness conversational REPL.
//!
//! This example uses the `MetaConversationCore` helper to:
//! - Encode user input into HDC space using the TextEncoder + PrimitiveSystem
//! - Feed the resulting BinaryHV components into the MetaConsciousness engine
//! - Print Φ, meta-Φ, self-model confidence, and a short explanation each turn
//!
//! It does not use any external LLM; responses are simple summaries of the
//! current meta-conscious state. This keeps the example self-contained while
//! exercising the full meta-consciousness pipeline on natural language.

use std::io::{self, Write};

use symthaea::hdc::meta_conscious_conversation::MetaConversationCore;

fn main() -> anyhow::Result<()> {
    // Small number of components for meta-reflection.
    let mut core = MetaConversationCore::new(4)?;

    println!("🧠 Meta-Consciousness Conversational Demo");
    println!("Type a line of text and press enter. Type 'exit' or 'quit' to stop.");
    println!();

    loop {
        print!("you > ");
        io::stdout().flush().ok();

        let mut input = String::new();
        let bytes = io::stdin().read_line(&mut input)?;
        if bytes == 0 {
            break;
        }

        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("goodbye.");
            break;
        }
        if trimmed.is_empty() {
            continue;
        }

        let meta_state = match core.reflect_on_text(trimmed) {
            Ok(state) => state,
            Err(e) => {
                eprintln!("reflection error: {e}");
                continue;
            }
        };

        println!(
            "symthaea > Φ={:.3}, meta-Φ={:.3}, self-model confidence={:.3}",
            meta_state.phi, meta_state.meta_phi, meta_state.self_model.confidence,
        );
        println!("           {}", meta_state.explanation);

        // Optional: brief introspection summary after several turns could be added
        // by calling meta.introspect(), but we keep this example minimal.
    }

    Ok(())
}
