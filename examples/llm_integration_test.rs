// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LLM Integration Test
//!
//! Tests the LlmOrgan with Ollama to verify end-to-end functionality.
//!
//! Run with: cargo run --example llm_integration_test --release

use symthaea::language::llm_organ::{LlmConfig, LlmOrgan, LlmProvider, LlmRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Symthaea LLM Integration Test ===\n");

    // Create LLM organ with Ollama config
    let config = LlmConfig {
        provider: LlmProvider::Ollama,
        model: "mistral:7b".to_string(), // Fast, capable model
        endpoint: "http://localhost:11434".to_string(),
        api_key: None,
        max_tokens: 256,
        temperature: 0.7,
        timeout_ms: 60000, // 60 second timeout
        hallucination_threshold: 0.3,
        min_phi_for_llm: 0.1, // Low threshold for testing
        enable_filtering: true,
    };

    let mut llm = LlmOrgan::with_config(config);

    // Test 1: Check Ollama availability
    println!("Test 1: Checking Ollama availability...");
    if llm.check_ollama().await {
        println!("  ✅ Ollama is running\n");
    } else {
        println!("  ❌ Ollama is not available!");
        println!("  Please run: ollama serve");
        return Ok(());
    }

    // Test 2: List available models
    println!("Test 2: Listing available models...");
    match llm.list_models().await {
        Ok(models) => {
            println!("  ✅ Found {} models:", models.len());
            for model in models.iter().take(5) {
                println!("     - {}", model);
            }
            if models.len() > 5 {
                println!("     ... and {} more", models.len() - 5);
            }
            println!();
        }
        Err(e) => {
            println!("  ⚠️  Failed to list models: {}", e);
        }
    }

    // Test 3: Simple query
    println!("Test 3: Simple query - 'What is NixOS?'");
    let request = LlmRequest {
        prompt: "What is NixOS? Explain in 2-3 sentences.".to_string(),
        system: Some(
            "You are a helpful assistant specializing in Linux systems. Be concise.".to_string(),
        ),
        history: vec![],
        context_embedding: None,
        expected_domain: Some("nixos".to_string()),
        temperature_override: None,
    };

    let current_phi = 0.5; // Simulated consciousness level
    match llm.generate(request, current_phi).await {
        Ok(response) => {
            println!("  ✅ Got response in {}ms:", response.generation_time_ms);
            println!("  Model: {}", response.model);
            println!("  ---");
            println!("  {}", response.content);
            println!("  ---");
            println!(
                "  Hallucination detected: {}",
                response.hallucination_detected
            );
            println!(
                "  Coherence score: {:.2}",
                response.semantic_analysis.coherence_score
            );
            println!();
        }
        Err(e) => {
            println!("  ❌ Failed: {}", e);
            println!();
        }
    }

    // Test 4: NixOS-specific query
    println!("Test 4: NixOS-specific query - 'How do I install a package?'");
    let request = LlmRequest {
        prompt: "How do I install Firefox on NixOS? Give me the command.".to_string(),
        system: Some(
            "You are a NixOS expert. Provide only the command, no explanation.".to_string(),
        ),
        history: vec![],
        context_embedding: None,
        expected_domain: Some("nixos".to_string()),
        temperature_override: Some(0.3), // More deterministic
    };

    match llm.generate(request, current_phi).await {
        Ok(response) => {
            println!("  ✅ Got response in {}ms:", response.generation_time_ms);
            println!("  ---");
            println!("  {}", response.content);
            println!("  ---");
            println!();
        }
        Err(e) => {
            println!("  ❌ Failed: {}", e);
            println!();
        }
    }

    // Test 5: Consciousness gating (low Φ should fail)
    println!("Test 5: Consciousness gating - low Φ should fail");
    let request = LlmRequest {
        prompt: "This should fail".to_string(),
        system: None,
        history: vec![],
        context_embedding: None,
        expected_domain: None,
        temperature_override: None,
    };

    let low_phi = 0.05; // Below threshold
    match llm.generate(request, low_phi).await {
        Ok(_) => {
            println!("  ⚠️  Response succeeded (unexpected with low Φ)");
        }
        Err(e) => {
            println!("  ✅ Correctly rejected: {}", e);
        }
    }
    println!();

    // Print final stats
    println!("=== LLM Organ Statistics ===");
    let stats = llm.stats();
    println!("  Total requests: {}", stats.total_requests);
    println!("  Successful: {}", stats.successful_responses);
    println!("  Failed: {}", stats.failed_requests);
    println!("  Cache hits: {}", stats.cache_hits);
    println!(
        "  Hallucinations detected: {}",
        stats.hallucinations_detected
    );
    println!("  Avg response time: {:.0}ms", stats.avg_response_time_ms);

    println!("\n=== LLM Integration Test Complete ===");
    Ok(())
}
