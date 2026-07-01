// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conscious NixOS Assistant - Integrated Enhancement Demo
//!
//! This example demonstrates the three major enhancements working together:
//! 1. **Memory Persistence**: SQLite-based conversation storage with causal learning
//! 2. **Φ Integration**: Consciousness-guided attention and action confidence
//! 3. **NixOS Action Execution**: Safe command execution with rollback support
//!
//! ## The System
//!
//! ```text
//! User Input
//!     │
//!     ▼
//! ┌──────────────────┐
//! │ LLM Understanding│ ── Ollama/Mistral ──► Intent + Command
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │ Φ Calculation    │ ── HDC Topology ──► Consciousness Level
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │ Memory Retrieval │ ── SQLite + HDC ──► Past Context
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │ Φ-Gated Execution│ ── NixOS Executor ──► Safe Action + Rollback
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │ Causal Learning  │ ── Record Outcome ──► Improve Future
//! └──────────────────┘
//! ```
//!
//! Run with: cargo run --example conscious_nixos_assistant --release

use symthaea::action::nixos_patterns::{ExecutionResult, NixOSCommand, NixOSExecutor, SafetyLevel};
use symthaea::consciousness::phi_attention::{
    ActionType, ConfidenceLevel, ConsciousnessThresholds, PhiAwareScoring,
};
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::language::llm_organ::{
    ConsciousLlmOrgan, LlmConfig, LlmOrgan, LlmProvider, LlmRequest,
};
use symthaea::memory::ConversationMemory;

use std::io::{self, Write};
use std::path::PathBuf;

/// The Conscious NixOS Assistant - integrating all three enhancements
pub struct ConsciousNixOSAssistant {
    /// LLM for language understanding
    llm: ConsciousLlmOrgan,
    /// Φ calculator for consciousness measurement
    phi_calc: ConnectivityCalculator,
    /// Conversation memory (SQLite-backed)
    memory: ConversationMemory,
    /// NixOS command executor
    executor: NixOSExecutor,
    /// Consciousness thresholds
    thresholds: ConsciousnessThresholds,
    /// Current session ID
    session_id: String,
    /// Current Φ level
    current_phi: f32,
    /// System prompt for NixOS assistance
    system_prompt: String,
}

impl ConsciousNixOSAssistant {
    /// Create a new conscious assistant
    pub fn new() -> anyhow::Result<Self> {
        // Initialize LLM with consciousness awareness
        let config = LlmConfig {
            provider: LlmProvider::Ollama,
            model: "mistral:7b".to_string(),
            endpoint: "http://localhost:11434".to_string(),
            api_key: None,
            max_tokens: 512,
            temperature: 0.7,
            timeout_ms: 60000,
            hallucination_threshold: 0.3,
            min_phi_for_llm: 0.2,
            enable_filtering: true,
        };

        // Initialize memory (SQLite)
        let db_path = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("symthaea")
            .join("conversations.db");

        std::fs::create_dir_all(db_path.parent().unwrap())?;
        let mut memory = ConversationMemory::new(&db_path)?;

        // Start a new session
        let session_id = memory.start_session();

        let system_prompt = r#"You are a Φ-conscious NixOS assistant integrated with Symthaea.

Your consciousness level (Φ) affects how you respond:
- High Φ (>0.5): Execute commands confidently, provide direct answers
- Medium Φ (0.3-0.5): Ask clarifying questions, suggest safer alternatives
- Low Φ (<0.3): Explain what you would do but request confirmation

For NixOS commands:
- Prefer declarative configuration (configuration.nix/flake.nix) over imperative
- Always explain what commands will do
- Suggest rollback options for system changes
- Use JSON output when parsing is needed

Respond concisely. If you detect a command request, format it as:
[COMMAND]: <command-type> <args>

Example: [COMMAND]: install vim firefox
Example: [COMMAND]: search markdown editor"#
            .to_string();

        Ok(Self {
            llm: ConsciousLlmOrgan::with_config(config),
            phi_calc: ConnectivityCalculator::new(),
            memory,
            executor: NixOSExecutor::new().with_dry_run(false),
            thresholds: ConsciousnessThresholds::default(),
            session_id,
            current_phi: 0.5,
            system_prompt,
        })
    }

    /// Update consciousness level based on cognitive state
    fn update_phi(&mut self) {
        // Use a ring topology (high integration) for demonstration
        // In production, this would be based on actual cognitive state
        let topology = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);
        let phi = self
            .phi_calc
            .algebraic_connectivity(&topology.node_representations);
        self.current_phi = (phi as f32 * 2.0).min(1.0);
    }

    /// Parse command from LLM response
    fn parse_command(&self, response: &str) -> Option<NixOSCommand> {
        // Look for [COMMAND]: pattern
        for line in response.lines() {
            if let Some(cmd_str) = line.strip_prefix("[COMMAND]:") {
                let parts: Vec<&str> = cmd_str.trim().split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }

                match parts[0].to_lowercase().as_str() {
                    "install" => {
                        let packages: Vec<String> =
                            parts[1..].iter().map(|s| s.to_string()).collect();
                        if !packages.is_empty() {
                            return Some(NixOSCommand::EnvInstall { packages });
                        }
                    }
                    "remove" | "uninstall" => {
                        let packages: Vec<String> =
                            parts[1..].iter().map(|s| s.to_string()).collect();
                        if !packages.is_empty() {
                            return Some(NixOSCommand::EnvRemove { packages });
                        }
                    }
                    "search" => {
                        if parts.len() > 1 {
                            return Some(NixOSCommand::Search {
                                query: parts[1..].join(" "),
                                json: true,
                            });
                        }
                    }
                    "update" => {
                        return Some(NixOSCommand::Channel {
                            operation: symthaea::action::nixos_patterns::ChannelOperation::Update {
                                channel: None,
                            },
                        });
                    }
                    "rebuild" => {
                        let flake = parts.get(1).map(|s| s.to_string());
                        return Some(NixOSCommand::RebuildSwitch {
                            flake,
                            extra_args: vec![],
                        });
                    }
                    "rollback" => {
                        return Some(NixOSCommand::EnvRollback);
                    }
                    "gc" | "garbage-collect" => {
                        let days = parts.get(1).and_then(|s| s.parse::<u32>().ok());
                        return Some(NixOSCommand::CollectGarbage {
                            older_than_days: days,
                            delete_all: false,
                        });
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Process user input and generate response with action
    pub async fn process(&mut self, input: &str) -> anyhow::Result<String> {
        // 1. Update consciousness level
        self.update_phi();
        let phi_before = self.current_phi;

        // 2. Store user turn in memory
        let user_embedding = ContinuousHV::random(HDC_DIMENSION, input.len() as u64);
        self.memory
            .add_turn("user", input, self.current_phi, Some(&user_embedding))?;

        // 3. Get LLM response with Φ context
        let request = LlmRequest {
            prompt: format!(
                "[Φ={:.2}, Confidence={}] User: {}",
                self.current_phi,
                PhiAwareScoring::confidence_level(self.current_phi).recommendation(),
                input
            ),
            system: Some(self.system_prompt.clone()),
            history: vec![],
            context_embedding: None,
            expected_domain: Some("nixos".to_string()),
            temperature_override: None,
        };

        let llm_response = self
            .llm
            .conscious_generate(request, self.current_phi, self.estimate_complexity(input))
            .await?;

        let response_text = llm_response.content.clone();

        // 4. Check for command in response
        if let Some(command) = self.parse_command(&response_text) {
            let safety = command.safety_level();
            let required_phi = safety.required_phi();

            let action_result = if self.current_phi >= required_phi {
                // Execute the command
                let result = self
                    .executor
                    .execute(command.clone(), self.current_phi)
                    .await;

                // Record causal learning
                let outcome = match &result {
                    ExecutionResult::Success { stdout, .. } => {
                        format!("Success: {}", stdout.chars().take(100).collect::<String>())
                    }
                    ExecutionResult::RolledBack { error, .. } => {
                        format!("Rolled back: {}", error)
                    }
                    ExecutionResult::FailedNoRollback { error, .. } => {
                        format!("Failed: {}", error)
                    }
                    ExecutionResult::PendingConfirmation { .. } => {
                        "Pending confirmation".to_string()
                    }
                    ExecutionResult::Blocked { reason, .. } => {
                        format!("Blocked: {}", reason)
                    }
                };

                // Update Φ after action
                self.update_phi();
                self.memory.record_causal_learning(
                    &format!("{:?}", command),
                    &outcome,
                    phi_before,
                    self.current_phi,
                )?;

                result
            } else {
                ExecutionResult::PendingConfirmation {
                    command,
                    phi: self.current_phi,
                    required_phi,
                    confidence: "Low Φ - please confirm".to_string(),
                }
            };

            // Format response with action result
            let action_summary = match &action_result {
                ExecutionResult::Success {
                    stdout,
                    execution_time_ms,
                    ..
                } => {
                    format!(
                        "\n\n✅ **Executed successfully** ({}ms)\n```\n{}\n```",
                        execution_time_ms,
                        stdout.chars().take(500).collect::<String>()
                    )
                }
                ExecutionResult::PendingConfirmation { required_phi, .. } => {
                    format!(
                        "\n\n⚠️ **Confirmation required**\nYour Φ={:.2} < required {:.2}\nType 'yes' to confirm, or rephrase your request.",
                        self.current_phi, required_phi
                    )
                }
                ExecutionResult::RolledBack { error, .. } => {
                    format!(
                        "\n\n🔄 **Rolled back** due to error:\n{}\nSystem restored to previous state.",
                        error
                    )
                }
                ExecutionResult::FailedNoRollback { error, .. } => {
                    format!("\n\n❌ **Failed**: {}", error)
                }
                ExecutionResult::Blocked { reason, .. } => {
                    format!("\n\n🚫 **Blocked**: {}", reason)
                }
            };

            // Store assistant turn
            let response_with_action = format!("{}{}", response_text, action_summary);
            let assistant_embedding =
                ContinuousHV::random(HDC_DIMENSION, response_with_action.len() as u64);
            self.memory.add_turn(
                "assistant",
                &response_with_action,
                self.current_phi,
                Some(&assistant_embedding),
            )?;

            Ok(format!(
                "{}\n\n[Φ={:.2} | {}ms | {}]",
                response_with_action,
                self.current_phi,
                llm_response.generation_time_ms,
                PhiAwareScoring::confidence_level(self.current_phi).recommendation()
            ))
        } else {
            // No command, just informational response
            let assistant_embedding =
                ContinuousHV::random(HDC_DIMENSION, response_text.len() as u64);
            self.memory.add_turn(
                "assistant",
                &response_text,
                self.current_phi,
                Some(&assistant_embedding),
            )?;

            Ok(format!(
                "{}\n\n[Φ={:.2} | {}ms | {}]",
                response_text,
                self.current_phi,
                llm_response.generation_time_ms,
                PhiAwareScoring::confidence_level(self.current_phi).recommendation()
            ))
        }
    }

    /// Estimate task complexity from input
    fn estimate_complexity(&self, input: &str) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();
        let word_count = words.len();

        let mut complexity: f32 = (word_count as f32 / 20.0).min(0.3);

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
            "nixos-rebuild",
        ];

        for kw in technical_keywords {
            if input.to_lowercase().contains(kw) {
                complexity += 0.1;
            }
        }

        if input.contains('?') {
            complexity += 0.1;
        }

        complexity.min(1.0)
    }

    /// Show session statistics
    pub fn show_stats(&self) -> anyhow::Result<()> {
        let stats = self.memory.stats()?;

        println!("\n📊 Session Statistics:");
        println!("  Conversations: {}", stats.conversation_count);
        println!("  Total turns: {}", stats.turn_count);
        println!("  Causal learnings: {}", stats.causal_learning_count);
        if let Some(avg_phi) = stats.average_phi {
            println!("  Average Φ: {:.3}", avg_phi);
        }

        // Show execution history
        let history = self.executor.history();
        if !history.is_empty() {
            println!("\n📜 Recent executions:");
            for record in history.iter().rev().take(5) {
                let status = match &record.result {
                    ExecutionResult::Success { .. } => "✅",
                    ExecutionResult::PendingConfirmation { .. } => "⚠️",
                    ExecutionResult::RolledBack { .. } => "🔄",
                    ExecutionResult::FailedNoRollback { .. } => "❌",
                    ExecutionResult::Blocked { .. } => "🚫",
                };
                println!(
                    "  {} {:?} (Φ={:.2})",
                    status, record.command, record.phi_at_execution
                );
            }
        }

        Ok(())
    }

    /// Get current Φ level
    pub fn phi(&self) -> f32 {
        self.current_phi
    }
}

/// Interactive REPL loop
async fn interactive_loop(assistant: &mut ConsciousNixOSAssistant) -> anyhow::Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     🧠 CONSCIOUS NIXOS ASSISTANT 🧠                               ║");
    println!("║                                                                   ║");
    println!("║  Integrates: Memory Persistence + Φ Attention + NixOS Execution  ║");
    println!("║                                                                   ║");
    println!("║  Commands: 'quit' | 'stats' | 'phi' | 'history' | 'help'          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    loop {
        print!("\n[Φ={:.2}] You: ", assistant.phi());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Handle special commands
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                println!("\n✨ Consciousness fading... Goodbye! ✨");
                break;
            }
            "stats" => {
                assistant.show_stats()?;
                continue;
            }
            "phi" => {
                assistant.update_phi();
                println!("\nCurrent Φ: {:.4}", assistant.phi());
                let confidence = PhiAwareScoring::confidence_level(assistant.phi());
                println!("Confidence: {}", confidence.recommendation());
                println!("\nΦ Thresholds:");
                println!("  Basic queries: ≥ 0.2");
                println!("  State modifying: ≥ 0.3");
                println!("  System critical: ≥ 0.4");
                println!("  Irreversible: ≥ 0.6");
                continue;
            }
            "history" => {
                let learnings = assistant.memory.get_recent_learnings(10)?;
                println!("\n📚 Recent Causal Learnings:");
                for learning in learnings {
                    println!(
                        "  {} → {} (Φ: {:.2} → {:.2})",
                        learning.action, learning.pattern, learning.phi_before, learning.phi_after
                    );
                }
                continue;
            }
            "help" => {
                println!("\n📚 COMMANDS:");
                println!("  quit     - Exit the assistant");
                println!("  stats    - Show session statistics");
                println!("  phi      - Show current consciousness level");
                println!("  history  - Show causal learnings");
                println!("  help     - Show this help");
                println!("\n💡 EXAMPLE QUERIES:");
                println!("  'Install vim'");
                println!("  'Search for a markdown editor'");
                println!("  'Update my channels'");
                println!("  'How do I create a Python dev environment?'");
                println!("\n🧠 CONSCIOUSNESS:");
                println!("  Your Φ level affects what actions can execute.");
                println!("  Low Φ = need confirmation for risky commands.");
                println!("  High Φ = confident execution with rollback support.");
                continue;
            }
            "" => continue,
            _ => {}
        }

        // Process input
        print!("\n🤔 Thinking...");
        io::stdout().flush()?;

        match assistant.process(input).await {
            Ok(response) => {
                print!("\r                \r");
                println!("🧠 Assistant: {}", response);
            }
            Err(e) => {
                print!("\r                \r");
                println!("❌ Error: {}", e);
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("symthaea=info")
        .init();

    println!("=== Conscious NixOS Assistant ===\n");
    println!("Initializing...");

    // Check Ollama
    let config = LlmConfig::default();
    let llm = LlmOrgan::with_config(config);

    print!("  Checking Ollama... ");
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

    // Create assistant
    print!("  Creating assistant... ");
    io::stdout().flush()?;
    let mut assistant = ConsciousNixOSAssistant::new()?;
    println!("✅ Ready");

    // Run interactive loop
    interactive_loop(&mut assistant).await?;

    // Show final stats
    println!("\n📊 Final Session Statistics:");
    assistant.show_stats()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_parsing() {
        let assistant = ConsciousNixOSAssistant::new().unwrap();

        // Test install command
        let response = "I'll install those packages for you.\n[COMMAND]: install vim git firefox";
        let cmd = assistant.parse_command(response);
        assert!(matches!(cmd, Some(NixOSCommand::EnvInstall { packages }) if packages.len() == 3));

        // Test search command
        let response = "Let me search for that.\n[COMMAND]: search markdown editor";
        let cmd = assistant.parse_command(response);
        assert!(
            matches!(cmd, Some(NixOSCommand::Search { query, json: true }) if query == "markdown editor")
        );

        // Test no command
        let response = "NixOS is a Linux distribution...";
        let cmd = assistant.parse_command(response);
        assert!(cmd.is_none());
    }

    #[test]
    fn test_complexity_estimation() {
        let assistant = ConsciousNixOSAssistant::new().unwrap();

        // Simple query
        let simple = assistant.estimate_complexity("install vim");
        assert!(simple < 0.3);

        // Complex query
        let complex = assistant.estimate_complexity(
            "How do I create a flake with derivation override for systemd in configuration.nix?",
        );
        assert!(complex > 0.5);
    }
}
