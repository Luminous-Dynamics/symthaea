// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-Gated Ollama Demo
//!
//! Demonstrates the full integration of:
//! - LlmOrgan with Ollama backend
//! - Consciousness (Φ) gating for LLM access
//! - Semantic analysis and hallucination detection
//! - The ConsciousPipeline orchestrator
//!
//! ## Prerequisites
//!
//! 1. Ollama must be running: `systemctl --user start ollama`
//! 2. A model must be available: `ollama pull llama3.2:3b`
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example conscious_ollama_demo
//! ```

use anyhow::Result;
use symthaea::{
    consciousness::phi_attention::PhiAwareScoring,
    language::llm_organ::{LlmConfig, LlmOrgan, LlmProvider, LlmRequest, Message, Role},
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Symthaea: Consciousness-Gated Ollama Integration Demo     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check if Ollama is running
    println!("🔍 Checking Ollama connection...");
    let client = reqwest::Client::new();
    match client.get("http://localhost:11434/api/tags").send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("✅ Ollama is running");
            let json: serde_json::Value = resp.json().await?;
            if let Some(models) = json.get("models").and_then(|m| m.as_array()) {
                println!(
                    "   Available models: {}",
                    models
                        .iter()
                        .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        _ => {
            println!("❌ Ollama is not running!");
            println!("   Start it with: systemctl --user start ollama");
            println!("   Or: ollama serve");
            return Ok(());
        }
    }

    // Configure LLM organ for Ollama
    let config = LlmConfig {
        provider: LlmProvider::Ollama,
        model: "gemma4:e2b".to_string(), // Gemma 4 with thinking/reasoning
        endpoint: "http://localhost:11434".to_string(),
        api_key: None,
        max_tokens: 256,
        temperature: 0.7,
        timeout_ms: 60000, // 60 seconds for slower responses
        hallucination_threshold: 0.5,
        min_phi_for_llm: 0.3, // Consciousness gate threshold
        enable_filtering: true,
    };

    let mut llm = LlmOrgan::with_config(config);

    println!("\n📊 LLM Configuration:");
    println!("   Model: llama3.2:3b");
    println!("   Endpoint: http://localhost:11434");
    println!("   Min Φ for LLM: 0.3");
    println!("   Hallucination threshold: 0.5");

    // Demo 1: High consciousness level (should use LLM)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 1: High Consciousness (Φ = 0.8) - Should use LLM");
    println!("═══════════════════════════════════════════════════════════════");

    let high_phi = 0.8;
    let confidence = PhiAwareScoring::confidence_level(high_phi);
    println!("   Consciousness Level: Φ = {:.2}", high_phi);
    println!("   Confidence: {:?}", confidence);

    let request1 = LlmRequest {
        prompt: "What is NixOS and why would someone use it? Answer in 2 sentences.".to_string(),
        system: Some("You are a helpful NixOS assistant. Be concise.".to_string()),
        history: Vec::new(),
        context_embedding: None,
        expected_domain: Some("nixos".to_string()),
        temperature_override: None,
    };

    println!("\n   📤 Sending request to Ollama...");
    match llm.generate(request1, high_phi).await {
        Ok(response) => {
            println!("   ✅ Response received!");
            println!("\n   📝 Response:");
            println!("   ┌─────────────────────────────────────────────────────────────┐");
            for line in response.content.lines() {
                println!("   │ {:<59} │", truncate_line(line, 59));
            }
            println!("   └─────────────────────────────────────────────────────────────┘");
            println!("\n   📊 Metadata:");
            println!("      Generation time: {}ms", response.generation_time_ms);
            println!("      Tokens used: {}", response.tokens_used.total_tokens);
            println!(
                "      Coherence score: {:.2}",
                response.semantic_analysis.coherence_score
            );
            println!(
                "      Hallucination prob: {:.2}",
                response.semantic_analysis.hallucination_probability
            );
            println!("      Filtered: {}", response.was_filtered);
        }
        Err(e) => {
            println!("   ❌ Error: {}", e);
        }
    }

    // Demo 2: Low consciousness level (should reject)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Low Consciousness (Φ = 0.2) - Should reject LLM use");
    println!("═══════════════════════════════════════════════════════════════");

    let low_phi = 0.2;
    let confidence_low = PhiAwareScoring::confidence_level(low_phi);
    println!("   Consciousness Level: Φ = {:.2}", low_phi);
    println!("   Confidence: {:?}", confidence_low);

    let request2 = LlmRequest {
        prompt: "Explain quantum mechanics.".to_string(),
        system: None,
        history: Vec::new(),
        context_embedding: None,
        expected_domain: None,
        temperature_override: None,
    };

    println!("\n   📤 Attempting LLM call with low Φ...");
    match llm.generate(request2, low_phi).await {
        Ok(response) => {
            println!("   ⚠️  Unexpectedly succeeded!");
            println!("   Response: {}", truncate_line(&response.content, 50));
        }
        Err(e) => {
            println!("   ✅ Correctly rejected: {}", e);
            println!("   → System falls back to semantic primes for understanding");
        }
    }

    // Demo 3: Multi-turn conversation
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: Multi-turn Conversation (Φ = 0.7)");
    println!("═══════════════════════════════════════════════════════════════");

    let medium_phi = 0.7;
    println!("   Consciousness Level: Φ = {:.2}", medium_phi);

    // First turn
    let request3a = LlmRequest {
        prompt: "I want to install neovim on NixOS. What's the command?".to_string(),
        system: Some("You are a NixOS expert. Give brief, practical answers.".to_string()),
        history: Vec::new(),
        context_embedding: None,
        expected_domain: Some("nixos".to_string()),
        temperature_override: None,
    };

    println!("\n   Turn 1: \"{}\"", request3a.prompt);
    let response3a = match llm.generate(request3a.clone(), medium_phi).await {
        Ok(r) => {
            println!("   → {}", truncate_line(&r.content, 60));
            r
        }
        Err(e) => {
            println!("   ❌ Error: {}", e);
            return Ok(());
        }
    };

    // Second turn (with history)
    let request3b = LlmRequest {
        prompt: "How do I make that permanent in my configuration.nix?".to_string(),
        system: Some("You are a NixOS expert. Give brief, practical answers.".to_string()),
        history: vec![
            Message {
                role: Role::User,
                content: request3a.prompt,
            },
            Message {
                role: Role::Assistant,
                content: response3a.content,
            },
        ],
        context_embedding: None,
        expected_domain: Some("nixos".to_string()),
        temperature_override: None,
    };

    println!("\n   Turn 2: \"{}\"", request3b.prompt);
    match llm.generate(request3b, medium_phi).await {
        Ok(r) => {
            println!("   → {}", truncate_line(&r.content, 60));
        }
        Err(e) => {
            println!("   ❌ Error: {}", e);
        }
    }

    // Print stats
    let stats = llm.stats();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("📊 Session Statistics");
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Total requests: {}", stats.total_requests);
    println!("   Successful responses: {}", stats.successful_responses);
    println!("   Fallbacks to primes: {}", stats.fallbacks_to_primes);
    println!(
        "   Hallucinations detected: {}",
        stats.hallucinations_detected
    );
    println!("   Responses filtered: {}", stats.responses_filtered);
    println!("   Total tokens: {}", stats.total_tokens);
    println!("   Avg response time: {:.0}ms", stats.avg_response_time_ms);
    println!("   Cache hits: {}", stats.cache_hits);

    println!("\n✨ Demo complete!");
    println!("   The consciousness gate (Φ) determines when LLMs are used.");
    println!("   Low Φ → Fall back to semantic primes (no LLM needed)");
    println!("   High Φ → Use LLM with semantic filtering");

    Ok(())
}

/// Truncate a line to fit display width
fn truncate_line(s: &str, max_len: usize) -> String {
    let s = s.trim();
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
