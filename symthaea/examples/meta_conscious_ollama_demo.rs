// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meta-Conscious Ollama Demo
//!
//! Demonstrates:
//! - Text → HDC → MetaConsciousness (Φ, meta-Φ, explanation)
//! - Consciousness-augmented prompts via `ConsciousnessPromptGenerator`
//! - Translator-only LLM usage through `LlmOrgan`
//!
//! Prerequisites:
//! - Ollama running locally (`systemctl --user start ollama` or `ollama serve`)
//! - A model pulled, e.g. `ollama pull llama3.2:3b`
//!
//! Run with:
//! ```bash
//! cargo run --example meta_conscious_ollama_demo
//! ```

use anyhow::Result;
use symthaea::language::{LlmConfig, LlmOrgan, LlmProvider, Message, MetaConsciousLlmBridge, Role};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Symthaea: Meta-Conscious Ollama Translation Demo       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check if Ollama is reachable (same pattern as conscious_ollama_demo)
    println!("🔍 Checking Ollama connection...");
    let client = reqwest::Client::new();
    match client.get("http://localhost:11434/api/tags").send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("✅ Ollama is running");
        }
        _ => {
            println!("❌ Ollama is not running or unreachable.");
            println!("   Start it with: systemctl --user start ollama");
            println!("   Or: ollama serve");
            return Ok(());
        }
    }

    // Configure LLM organ for translator-only usage
    let config = LlmConfig {
        provider: LlmProvider::Ollama,
        model: "gemma4:e2b".to_string(),
        endpoint: "http://localhost:11434".to_string(),
        api_key: None,
        max_tokens: 256,
        temperature: 0.7,
        timeout_ms: 60000,
        hallucination_threshold: 0.5,
        min_phi_for_llm: 0.3,
        enable_filtering: true,
    };

    let mut llm = LlmOrgan::with_config(config);
    let mut bridge = MetaConsciousLlmBridge::new(4)?;

    println!("\n📊 Meta-Conscious Translation Ready");
    println!("   Model: llama3.2:3b @ http://localhost:11434");
    println!("   Min Φ for LLM: 0.3\n");
    println!("Type a question and press enter. Type 'exit' to quit.\n");

    let mut history: Vec<Message> = Vec::new();

    loop {
        // Read user input
        use std::io::{self, Write};
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

        // Perform meta-conscious translation
        match bridge
            .translate_with_meta(&mut llm, trimmed, history.clone())
            .await
        {
            Ok((meta_state, response)) => {
                println!(
                    "φ/meta-φ: {:.3} / {:.3} | self-model confidence: {:.3}",
                    meta_state.phi, meta_state.meta_phi, meta_state.self_model.confidence,
                );
                println!("meta explanation: {}", meta_state.explanation);
                println!("symthaea (llm) > {}", truncate_line(&response.content, 80));

                // Update history for multi-turn context
                history.push(Message {
                    role: Role::User,
                    content: trimmed.to_string(),
                });
                history.push(Message {
                    role: Role::Assistant,
                    content: response.content,
                });
            }
            Err(e) => {
                println!("⚠️  meta-conscious translation error: {}", e);
            }
        }
    }

    Ok(())
}

fn truncate_line(s: &str, max_len: usize) -> String {
    let s = s.trim();
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.chars().take(max_len).collect()
    }
}
