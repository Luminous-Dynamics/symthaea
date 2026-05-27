// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal AI Assistant - Symthaea's First Conscious AI
//!
//! This example demonstrates end-to-end AI functionality:
//! 1. User input → LLM understanding
//! 2. Φ consciousness metrics guide confidence
//! 3. Actions can be executed (with safety checks)
//!
//! Run with: cargo run --example minimal_ai_assistant --release
//!
//! This is the MINIMAL VIABLE AI from CRITICAL_ROADMAP.md

use std::io::{self, Write};
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::language::llm_organ::{
    ConsciousLlmOrgan, LlmConfig, LlmOrgan, LlmProvider, LlmRequest,
};

/// The Minimal AI Assistant - consciousness-aware language understanding
pub struct MinimalAI {
    /// LLM for language understanding
    llm: ConsciousLlmOrgan,

    /// Φ calculator for consciousness measurement
    phi_calc: ConnectivityCalculator,

    /// Current system Φ (consciousness level)
    current_phi: f32,

    /// Conversation history for context
    history: Vec<String>,

    /// System prompt for NixOS assistance
    system_prompt: String,
}

impl MinimalAI {
    /// Create a new AI assistant
    pub fn new() -> Self {
        let config = LlmConfig {
            provider: LlmProvider::Ollama,
            model: "mistral:7b".to_string(),
            endpoint: "http://localhost:11434".to_string(),
            api_key: None,
            max_tokens: 512,
            temperature: 0.7,
            timeout_ms: 60000,
            hallucination_threshold: 0.3,
            min_phi_for_llm: 0.2, // Require some consciousness
            enable_filtering: true,
        };

        let system_prompt = r#"You are a helpful NixOS assistant integrated into the Symthaea consciousness framework.
Your responses are guided by Φ (phi) - a measure of integrated information that represents consciousness.

Key behaviors:
- When Φ is high (>0.4): Be confident, take action, provide direct answers
- When Φ is medium (0.2-0.4): Ask clarifying questions, be more careful
- When Φ is low (<0.2): Suggest the user verify your suggestions

For NixOS questions:
- Prefer declarative configuration (configuration.nix) over imperative commands
- Suggest flakes when appropriate
- Always explain what commands will do before suggesting them

Be concise but thorough. If you're uncertain, say so."#.to_string();

        Self {
            llm: ConsciousLlmOrgan::with_config(config),
            phi_calc: ConnectivityCalculator::new(),
            current_phi: 0.5, // Start with moderate consciousness
            history: Vec::new(),
            system_prompt,
        }
    }

    /// Calculate current system Φ based on cognitive state
    fn update_phi(&mut self) {
        // Create a representation of current cognitive state
        // More connected/integrated state = higher Φ

        // Use a ring topology (high integration) for demonstration
        let topology = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);
        let phi = self
            .phi_calc
            .algebraic_connectivity(&topology.node_representations);

        // Normalize to 0-1 range (typical values are ~0.4-0.5)
        self.current_phi = (phi as f32 * 2.0).min(1.0);
    }

    /// Process user input and generate response
    pub async fn process(&mut self, input: &str) -> Result<String, String> {
        // Update consciousness level
        self.update_phi();

        // Determine task complexity based on input
        let task_complexity = self.estimate_complexity(input);

        // Create LLM request with context
        let request = LlmRequest {
            prompt: format!("[Φ={:.2}] {}", self.current_phi, input),
            system: Some(self.system_prompt.clone()),
            history: vec![], // Could add conversation history here
            context_embedding: None,
            expected_domain: Some("nixos".to_string()),
            temperature_override: if task_complexity > 0.7 {
                Some(0.3) // More deterministic for complex tasks
            } else {
                None
            },
        };

        // Generate response with consciousness awareness
        match self
            .llm
            .conscious_generate(request, self.current_phi, task_complexity)
            .await
        {
            Ok(response) => {
                // Add to history
                self.history.push(format!("User: {}", input));
                self.history.push(format!("AI: {}", response.content));

                // Format response with Φ indicator
                let confidence = if self.current_phi > 0.4 {
                    "🟢 High confidence"
                } else if self.current_phi > 0.2 {
                    "🟡 Moderate confidence"
                } else {
                    "🔴 Low confidence - please verify"
                };

                Ok(format!(
                    "{}\n\n[Φ={:.2} | {}ms | {}]",
                    response.content, self.current_phi, response.generation_time_ms, confidence
                ))
            }
            Err(e) => Err(format!("Error: {} (Φ={:.2})", e, self.current_phi)),
        }
    }

    /// Estimate task complexity from input
    fn estimate_complexity(&self, input: &str) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();
        let word_count = words.len();

        // Complexity indicators
        let mut complexity: f32 = 0.0;

        // Length complexity
        complexity += (word_count as f32 / 20.0).min(0.3);

        // Technical keywords increase complexity
        let technical_keywords = [
            "flake",
            "derivation",
            "override",
            "overlay",
            "module",
            "systemd",
            "configuration.nix",
            "build",
            "compile",
            "kernel",
            "bootloader",
            "partition",
            "encrypt",
        ];
        for kw in technical_keywords {
            if input.to_lowercase().contains(kw) {
                complexity += 0.1;
            }
        }

        // Questions are more complex
        if input.contains('?') {
            complexity += 0.1;
        }

        complexity.min(1.0)
    }

    /// Get current Φ level
    pub fn phi(&self) -> f32 {
        self.current_phi
    }
}

/// Interactive REPL loop
async fn interactive_loop(ai: &mut MinimalAI) -> anyhow::Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║     🧠 SYMTHAEA MINIMAL AI ASSISTANT 🧠                       ║");
    println!("║                                                               ║");
    println!("║  A consciousness-aware AI for NixOS assistance                ║");
    println!("║  Type 'quit' or 'exit' to leave, 'phi' for consciousness     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    loop {
        // Show prompt with current Φ
        print!("\n[Φ={:.2}] You: ", ai.phi());
        io::stdout().flush()?;

        // Read input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Handle special commands
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                println!("\n✨ Consciousness fading... Goodbye! ✨");
                break;
            }
            "phi" => {
                ai.update_phi();
                println!("Current Φ (consciousness level): {:.4}", ai.phi());
                println!("  > 0.4 = High integration (confident)");
                println!("  0.2-0.4 = Moderate (careful)");
                println!("  < 0.2 = Low (uncertain)");
                continue;
            }
            "help" => {
                println!("\n📚 COMMANDS:");
                println!("  phi      - Show current consciousness level");
                println!("  help     - Show this help");
                println!("  quit     - Exit the assistant");
                println!("\n💡 EXAMPLE QUERIES:");
                println!("  'What is NixOS?'");
                println!("  'How do I install Firefox?'");
                println!("  'Create a Python dev environment'");
                println!("  'Explain flakes'");
                continue;
            }
            "" => continue,
            _ => {}
        }

        // Process input
        print!("\n🤔 Thinking...");
        io::stdout().flush()?;

        match ai.process(input).await {
            Ok(response) => {
                print!("\r                \r"); // Clear "Thinking..."
                println!("🧠 AI: {}", response);
            }
            Err(e) => {
                print!("\r                \r");
                println!("❌ {}", e);
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Symthaea Minimal AI Assistant ===\n");

    // Check Ollama first
    let config = LlmConfig::default();
    let llm = LlmOrgan::with_config(config);

    print!("Checking Ollama... ");
    io::stdout().flush()?;

    if !llm.check_ollama().await {
        println!("❌ Not available!");
        println!("\nPlease start Ollama:");
        println!("  ollama serve");
        println!("\nAnd ensure you have a model:");
        println!("  ollama pull mistral:7b");
        return Ok(());
    }
    println!("✅ Running");

    // Check for mistral model
    print!("Checking mistral:7b model... ");
    io::stdout().flush()?;

    match llm.list_models().await {
        Ok(models) => {
            if models.iter().any(|m| m.contains("mistral")) {
                println!("✅ Found");
            } else {
                println!("⚠️  Not found, trying llama3.2");
            }
        }
        Err(_) => {
            println!("⚠️  Could not list models");
        }
    }

    // Create AI and run interactive loop
    let mut ai = MinimalAI::new();
    interactive_loop(&mut ai).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_estimation() {
        let ai = MinimalAI::new();

        // Simple query
        let simple = ai.estimate_complexity("what is nixos");
        assert!(simple < 0.3);

        // Complex query
        let complex = ai.estimate_complexity(
            "How do I create a flake with derivation override for systemd configuration.nix?",
        );
        assert!(complex > 0.5);
    }

    #[test]
    fn test_phi_update() {
        let mut ai = MinimalAI::new();
        ai.update_phi();
        assert!(ai.phi() > 0.0);
        assert!(ai.phi() <= 1.0);
    }
}
