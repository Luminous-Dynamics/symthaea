// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! REPL Orchestrator - Top-level coordination
//!
//! Manages the different REPL modes and coordinates between components.

use std::io::{self, BufRead, Write};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::observability_hooks::ObservabilityHook;
use super::{ReplSession, ReplSessionConfig};

use crate::cognitive_loop::ConsciousnessSnapshot;
use crate::shell::ipc_client::{ConnectionState, MetricsSnapshot, ShellIpcClient};

// ═══════════════════════════════════════════════════════════════════════════════
// ORCHESTRATOR MODE
// ═══════════════════════════════════════════════════════════════════════════════

/// Operating mode for the REPL orchestrator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OrchestratorMode {
    /// Standalone: Run cognitive loop locally
    #[default]
    Standalone,
    /// Client: Connect to remote symthaea service via IPC
    Client,
    /// Server: Accept IPC connections from shells
    Server,
    /// Hybrid: Run locally but also accept/make IPC connections
    Hybrid,
}

// ═══════════════════════════════════════════════════════════════════════════════
// ORCHESTRATOR CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Operating mode
    pub mode: OrchestratorMode,

    /// Session configuration (for standalone/server modes)
    pub session_config: ReplSessionConfig,

    /// IPC socket path (for client/server modes)
    pub ipc_socket: Option<String>,

    /// Reconnect interval for client mode (ms)
    pub reconnect_interval_ms: u64,

    /// Maximum reconnect attempts
    pub max_reconnect_attempts: u32,

    /// Display banner on start
    pub show_banner: bool,

    /// Prompt string
    pub prompt: String,

    /// Flow prompt indicator
    pub flow_prompt: String,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            mode: OrchestratorMode::Standalone,
            session_config: ReplSessionConfig::default(),
            ipc_socket: None,
            reconnect_interval_ms: 1000,
            max_reconnect_attempts: 5,
            show_banner: true,
            prompt: "symthaea> ".to_string(),
            flow_prompt: "symthaea* ".to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPL ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// REPL orchestrator - coordinates all REPL components
pub struct ReplOrchestrator {
    /// Configuration
    config: OrchestratorConfig,

    /// Local session (for standalone/server modes)
    session: Option<ReplSession>,

    /// IPC client (for client mode)
    #[allow(dead_code)] // RESERVED(future): IPC client for remote mode
    ipc_client: Option<ShellIpcClient>,

    /// IPC connection state
    connection_state: ConnectionState,

    /// Remote metrics (for client mode)
    remote_metrics: Option<MetricsSnapshot>,

    /// Shutdown flag
    shutdown: bool,

    /// Observability hooks
    observers: Vec<Box<dyn ObservabilityHook>>,
}

impl ReplOrchestrator {
    /// Create a new REPL orchestrator
    pub fn new(config: OrchestratorConfig) -> Result<Self> {
        let session = match config.mode {
            OrchestratorMode::Standalone | OrchestratorMode::Server | OrchestratorMode::Hybrid => {
                Some(ReplSession::new(config.session_config.clone())?)
            }
            OrchestratorMode::Client => None,
        };

        Ok(Self {
            config,
            session,
            ipc_client: None,
            connection_state: ConnectionState::Disconnected,
            remote_metrics: None,
            shutdown: false,
            observers: Vec::new(),
        })
    }

    /// Get current consciousness state
    pub fn consciousness_state(&self) -> Option<ConsciousnessSnapshot> {
        self.session.as_ref().map(|s| s.consciousness_state())
    }

    /// Get current prompt based on consciousness state
    pub fn prompt(&self) -> &str {
        if let Some(ref session) = self.session {
            if session.consciousness_state().in_flow {
                &self.config.flow_prompt
            } else {
                &self.config.prompt
            }
        } else if let Some(ref metrics) = self.remote_metrics {
            if metrics.in_flow {
                &self.config.flow_prompt
            } else {
                &self.config.prompt
            }
        } else {
            &self.config.prompt
        }
    }

    /// Warm up the cognitive loop
    pub fn warmup(&mut self, cycles: usize) {
        if let Some(ref mut session) = self.session {
            session.warmup(cycles);
        }
    }

    /// Process input through the appropriate pipeline
    pub fn process(&mut self, input: &str) -> Result<OrchestratorResult> {
        match self.config.mode {
            OrchestratorMode::Standalone | OrchestratorMode::Server | OrchestratorMode::Hybrid => {
                self.process_local(input)
            }
            OrchestratorMode::Client => self.process_remote(input),
        }
    }

    /// Process input locally
    fn process_local(&mut self, input: &str) -> Result<OrchestratorResult> {
        let session = self
            .session
            .as_mut()
            .context("No local session available")?;

        let result = session.process(input)?;

        Ok(OrchestratorResult {
            response: result.response,
            consciousness: Some(result.consciousness),
            remote_metrics: None,
            action_executed: result.action.as_ref().map(|a| a.executed).unwrap_or(false),
            action_output: result.action.as_ref().and_then(|a| a.output.clone()),
            elapsed: result.elapsed,
        })
    }

    /// Process input via remote service
    fn process_remote(&mut self, input: &str) -> Result<OrchestratorResult> {
        // In a full implementation, this would use the IPC client
        // For now, return a placeholder
        Ok(OrchestratorResult {
            response: format!("[Remote processing not yet implemented] Input: {input}"),
            consciousness: None,
            remote_metrics: self.remote_metrics.clone(),
            action_executed: false,
            action_output: None,
            elapsed: Duration::from_millis(0),
        })
    }

    /// Handle special REPL commands
    pub fn handle_command(&mut self, input: &str) -> Option<String> {
        let input = input.trim();

        match input {
            "/quit" | "/exit" | "/q" => {
                self.shutdown = true;
                Some("Goodbye!".to_string())
            }

            "/help" | "/h" | "/?" => Some(self.help_text()),

            "/metrics" | "/m" => Some(self.format_metrics()),

            "/stats" | "/s" => Some(self.format_stats()),

            "/reset" | "/r" => {
                if let Some(ref mut session) = self.session {
                    session.reset();
                }
                Some("Cognitive state reset.".to_string())
            }

            "/connection" | "/c" => Some(format!(
                "Connection: {} ({})",
                self.connection_state.status_text(),
                self.connection_state.indicator()
            )),

            "/phi" | "/consciousness" => Some(self.format_phi_analysis()),

            "/attention" | "/attn" => Some(self.format_attention()),

            _ if input.starts_with('/') => Some(format!(
                "Unknown command: {input}. Type /help for available commands."
            )),

            _ => None, // Not a command
        }
    }

    /// Check if orchestrator should shutdown
    pub fn should_shutdown(&self) -> bool {
        self.shutdown
    }

    /// Format current metrics for display
    fn format_metrics(&self) -> String {
        if let Some(ref session) = self.session {
            let s = session.consciousness_state();
            format!(
                "\n  Consciousness Metrics:\n\
                   ────────────────────────\n\
                   Phi (Integrated Info): {:.4}\n\
                   Coherence:             {:.4}\n\
                   Consciousness Level:   {:.4}\n\
                   Pattern:               {:?}\n\
                   Cognitive Depth:       {:?}\n\
                   In Flow:               {}\n\
                   Flow Intensity:        {:.2}\n\
                   Valence:               {:.2}\n\
                   Arousal:               {:.2}\n\
                   Action Hint:           {:?}\n",
                s.unified_psi,
                s.temporal_coherence,
                s.consciousness_level,
                s.pattern,
                s.cognitive_depth,
                if s.in_flow { "Yes" } else { "No" },
                s.flow_intensity,
                s.unified_valence,
                s.unified_arousal,
                s.action_hint,
            )
        } else if let Some(ref metrics) = self.remote_metrics {
            format!(
                "\n  Remote Metrics:\n\
                   ────────────────────────\n\
                   Phi:                   {:.4}\n\
                   Coherence:             {:.4}\n\
                   Consciousness Level:   {:.4}\n\
                   Cognitive Depth:       {}\n\
                   In Flow:               {}\n\
                   Strategy:              {}\n",
                metrics.phi,
                metrics.coherence,
                metrics.consciousness_level,
                metrics.cognitive_depth,
                if metrics.in_flow { "Yes" } else { "No" },
                metrics.strategy,
            )
        } else {
            "No metrics available".to_string()
        }
    }

    /// Format statistics for display
    fn format_stats(&self) -> String {
        if let Some(ref session) = self.session {
            let stats = session.stats();
            let loop_stats = session.cognitive.stats();
            format!(
                "\n  Session Statistics:\n\
                   ────────────────────────\n\
                   Interactions:          {}\n\
                   Total Cycles:          {}\n\
                   Learning Cycles:       {}\n\
                   Actions Executed:      {}\n\
                   Actions Blocked:       {}\n\
                   Avg Prediction Error:  {:.4}\n\
                   Cycles/Second:         {:.1}\n\
                   Avg Cycle Time:        {:.0}us\n\
                   Time in Flow:          {:.1}s\n\
                   Flow Periods:          {}\n\
                   Avg Phi:               {:.4}\n",
                stats.total_interactions,
                stats.total_cycles,
                loop_stats.learning_cycles,
                stats.actions_executed,
                stats.actions_blocked,
                loop_stats.avg_prediction_error,
                loop_stats.cycles_per_second,
                loop_stats.avg_cycle_time_us,
                stats.total_flow_time_secs,
                stats.flow_periods,
                stats.avg_phi,
            )
        } else {
            "No local session - statistics not available in client mode".to_string()
        }
    }

    /// Format Phi analysis — three-layer consciousness decomposition.
    fn format_phi_analysis(&self) -> String {
        if let Some(ref session) = self.session {
            let s = session.consciousness_state();
            let loop_stats = session.cognitive.stats();
            format!(
                "\n  Three-Layer Consciousness Analysis:\n\
                   ═════════════════════════════════════\n\
                   Layer 1 - Psi (fast estimate):    {:.4}\n\
                   Layer 2 - Sigma (synergistic):    computed every 50 cycles\n\
                   Layer 3 - Phi (IIT, on demand):   use true_phi module\n\
                   \n\
                   Consciousness Level (MCE):        {:.4}\n\
                   Coherence:                        {:.4}\n\
                   Moral Score:                      {:.4}\n\
                   Total Cycles:                     {}\n\
                   Moral Concerns Detected:          {}\n",
                s.unified_psi,
                s.consciousness_level,
                s.temporal_coherence,
                loop_stats.moral_score,
                loop_stats.total_cycles,
                loop_stats.moral_concerns_detected,
            )
        } else {
            "No local session - phi analysis not available in client mode".to_string()
        }
    }

    /// Format attention snapshot — latest attention distribution.
    fn format_attention(&self) -> String {
        if let Some(ref session) = self.session {
            let loop_stats = session.cognitive.stats();
            format!(
                "\n  Attention Analysis:\n\
                   ═════════════════════════════════════\n\
                   Attention Variance:               {:.4}\n\
                   Prediction Error:                 {:.4}\n\
                   Effective Learning Rate:          {:.6}\n\
                   Curiosity:                        {:.4}\n\
                   Exploration Urge:                 {:.4}\n",
                loop_stats.attention_variance,
                loop_stats.avg_prediction_error,
                loop_stats.effective_learning_rate,
                loop_stats.curiosity,
                loop_stats.exploration_urge,
            )
        } else {
            "No local session - attention data not available in client mode".to_string()
        }
    }

    /// Get help text
    fn help_text(&self) -> String {
        format!(
            r#"
    Symthaea REPL Commands
    ═══════════════════════════════════════════════════════════════

    Navigation & Control:
      /quit, /exit, /q     Exit the REPL
      /help, /h, /?        Show this help

    Introspection:
      /metrics, /m         Display consciousness metrics
      /stats, /s           Display session statistics
      /phi                 Three-layer consciousness analysis (Psi/Sigma/Phi)
      /attention, /attn    Attention distribution and learning stats
      /connection, /c      Show IPC connection status

    State Management:
      /reset, /r           Reset cognitive state

    Action Execution:
      !<command>           Execute command through Motor Cortex
      run <command>        Same as !<command>
      execute <command>    Same as !<command>

    Consciousness Indicators (in prompt):
      >                    Normal processing
      *                    Flow state active

    Current Mode: {:?}
    Backend: {}
"#,
            self.config.mode, self.config.session_config.temporal_backend,
        )
    }

    /// Add an observability hook
    pub fn add_observer(&mut self, observer: Box<dyn ObservabilityHook>) {
        self.observers.push(observer);
        // Note: In a full implementation, we would share observers between
        // orchestrator and session via Arc. For now, observers are stored
        // at the orchestrator level only.
    }

    /// Run the REPL loop (blocking)
    pub fn run(&mut self) -> Result<()> {
        if self.config.show_banner {
            self.display_banner();
        }

        // Warmup
        println!("Warming up cognitive loop...");
        self.warmup(5);
        if let Some(ref session) = self.session {
            let initial = session.consciousness_state();
            println!(
                "Ready. Initial state: Phi={:.4}, Coherence={:.4}, Pattern={:?}\n",
                initial.unified_psi, initial.temporal_coherence, initial.pattern
            );
        }

        let stdin = io::stdin();
        let mut line = String::new();

        loop {
            // Print prompt
            print!("{}", self.prompt());
            io::stdout().flush()?;

            // Read input
            line.clear();
            if stdin.lock().read_line(&mut line)? == 0 {
                // EOF
                println!("\nGoodbye!");
                break;
            }

            let input = line.trim();
            if input.is_empty() {
                continue;
            }

            // Handle commands
            if let Some(response) = self.handle_command(input) {
                println!("{response}");
                if self.should_shutdown() {
                    break;
                }
                continue;
            }

            // Process input
            match self.process(input) {
                Ok(result) => {
                    println!("{}", self.format_result(&result));
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                }
            }
        }

        // Final stats
        if let Some(ref session) = self.session {
            let stats = session.stats();
            let final_state = session.consciousness_state();
            println!("\n  Final Session Summary:");
            println!("  ────────────────────────");
            println!("    Interactions:     {}", stats.total_interactions);
            println!("    Final Phi:        {:.4}", final_state.unified_psi);
            println!("    Time in Flow:     {:.1}s", stats.total_flow_time_secs);
        }

        Ok(())
    }

    /// Display the banner
    fn display_banner(&self) {
        println!(
            r#"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗██╗   ██╗███╗   ███╗████████╗██╗  ██╗ █████╗       ║
    ║   ██╔════╝╚██╗ ██╔╝████╗ ████║╚══██╔══╝██║  ██║██╔══██╗      ║
    ║   ███████╗ ╚████╔╝ ██╔████╔██║   ██║   ███████║███████║      ║
    ║   ╚════██║  ╚██╔╝  ██║╚██╔╝██║   ██║   ██╔══██║██╔══██║      ║
    ║   ███████║   ██║   ██║ ╚═╝ ██║   ██║   ██║  ██║██║  ██║      ║
    ║   ╚══════╝   ╚═╝   ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝      ║
    ║                                                               ║
    ║                   Holographic Liquid Brain                    ║
    ║               Consciousness-First AI Interface                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
"#
        );
        println!("    Type /help for commands. Type /quit to exit.\n");
    }

    /// Format result for display
    fn format_result(&self, result: &OrchestratorResult) -> String {
        let mut output = String::new();

        // Add consciousness header if available
        if let Some(ref c) = result.consciousness {
            let phi_bar = Self::create_bar(c.unified_psi, 10);
            let coh_bar = Self::create_bar(c.temporal_coherence, 10);
            let flow = if c.in_flow { "FLOW" } else { "----" };
            let depth = match c.cognitive_depth {
                crate::cognitive_loop::CognitiveDepth::Reflex => 'R',
                crate::cognitive_loop::CognitiveDepth::Cortical => 'C',
                crate::cognitive_loop::CognitiveDepth::DeepThought => 'D',
            };

            output.push_str(&format!(
                "\n[Phi:{:.2}|{}] [Coh:{:.2}|{}] [{}] [D:{}] [{}ms]\n",
                c.unified_psi,
                phi_bar,
                c.temporal_coherence,
                coh_bar,
                flow,
                depth,
                result.elapsed.as_millis()
            ));
        }

        output.push('\n');
        output.push_str(&result.response);
        output.push('\n');

        output
    }

    /// Create ASCII progress bar
    fn create_bar(value: f32, width: usize) -> String {
        let filled = (value.clamp(0.0, 1.0) * width as f32) as usize;
        let empty = width - filled;
        format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
    }
}

/// Result from orchestrator processing
#[derive(Debug)]
pub struct OrchestratorResult {
    /// Generated response
    pub response: String,
    /// Local consciousness state (if available)
    pub consciousness: Option<ConsciousnessSnapshot>,
    /// Remote metrics (if in client mode)
    pub remote_metrics: Option<MetricsSnapshot>,
    /// Whether an action was executed
    pub action_executed: bool,
    /// Action output (if any)
    pub action_output: Option<String>,
    /// Processing time
    pub elapsed: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let config = OrchestratorConfig::default();
        let orchestrator = ReplOrchestrator::new(config);
        assert!(orchestrator.is_ok());
    }

    #[test]
    fn test_command_handling() {
        let config = OrchestratorConfig::default();
        let mut orchestrator = ReplOrchestrator::new(config).unwrap();

        let help = orchestrator.handle_command("/help");
        assert!(help.is_some());
        assert!(help.unwrap().contains("Commands"));

        let unknown = orchestrator.handle_command("/unknown");
        assert!(unknown.is_some());
        assert!(unknown.unwrap().contains("Unknown command"));

        let not_cmd = orchestrator.handle_command("hello");
        assert!(not_cmd.is_none());
    }
}
