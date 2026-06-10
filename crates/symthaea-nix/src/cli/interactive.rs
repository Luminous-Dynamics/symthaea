// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interactive Conscious REPL
//!
//! Provides a conversational interface for NixOS management with:
//! - Working memory display (what the system is thinking about)
//! - Consciousness metrics (Φ, Confidence, Free Energy)
//! - Multi-turn conversation that accumulates context
//! - Command history

use std::io::{self, BufRead, Write};

use super::commands::OutputFormat;
use crate::action::executor::NixOSExecutor;
use crate::mind::active_inference::NixActiveInference;

/// Default threshold for Φ and confidence in `ConsciousnessQuadrant::from_metrics`.
///
/// Values above this are "high", values at or below are "low".
/// Based on IIT: Φ > 0.5 indicates non-trivial integrated information.
pub const QUADRANT_THRESHOLD: f64 = 0.5;

/// Consciousness quadrant based on Φ and Confidence.
#[derive(Debug, Clone, Copy)]
pub enum ConsciousnessQuadrant {
    /// High Φ, High Confidence — ready to act.
    Confident,
    /// High Φ, Low Confidence — explore and ask.
    Curious,
    /// Low Φ, High Confidence — habitual/automatic.
    Habitual,
    /// Low Φ, Low Confidence — confused/overwhelmed.
    Confused,
}

impl ConsciousnessQuadrant {
    /// Determine quadrant from Φ and confidence.
    pub fn from_metrics(phi: f64, confidence: f64) -> Self {
        match (phi > QUADRANT_THRESHOLD, confidence > QUADRANT_THRESHOLD) {
            (true, true) => Self::Confident,
            (true, false) => Self::Curious,
            (false, true) => Self::Habitual,
            (false, false) => Self::Confused,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Confident => "Confident",
            Self::Curious => "Curious",
            Self::Habitual => "Habitual",
            Self::Confused => "Confused",
        }
    }
}

impl std::fmt::Display for ConsciousnessQuadrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The interactive REPL state.
pub struct ConsciousRepl {
    /// The active inference engine (cognitive core).
    engine: NixActiveInference,
    /// The Φ-gated executor.
    executor: NixOSExecutor,
    /// Output format.
    format: OutputFormat,
    /// Current Φ level (simulated for now).
    phi: f64,
    /// Turn counter.
    turn: usize,
}

impl ConsciousRepl {
    /// Create a new REPL.
    pub fn new(dry_run: bool, format: OutputFormat) -> Self {
        Self {
            engine: NixActiveInference::new(),
            executor: NixOSExecutor::new().with_dry_run(dry_run),
            format,
            phi: 0.7, // Default Φ
            turn: 0,
        }
    }

    /// Set the Φ level override.
    pub fn with_phi(mut self, phi: f64) -> Self {
        self.phi = phi;
        self
    }

    /// Run the REPL loop.
    pub fn run(&mut self) -> io::Result<()> {
        self.print_banner();

        let stdin = io::stdin();
        let mut reader = stdin.lock();

        loop {
            self.print_prompt();
            io::stdout().flush()?;

            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                // EOF
                println!();
                break;
            }

            let input = line.trim();
            if input.is_empty() {
                continue;
            }

            // Handle REPL commands
            match input {
                "/quit" | "/exit" | "/q" => break,
                "/reset" => {
                    self.engine.reset();
                    self.turn = 0;
                    println!("  Context reset. Starting fresh conversation.");
                    continue;
                }
                "/memory" | "/wm" => {
                    self.print_working_memory();
                    continue;
                }
                "/status" => {
                    self.print_status();
                    continue;
                }
                "/help" | "/?" => {
                    self.print_help();
                    continue;
                }
                _ => {}
            }

            self.turn += 1;
            self.process_input(input);
        }

        println!("Goodbye.");
        Ok(())
    }

    /// Process a single natural language input.
    fn process_input(&mut self, input: &str) {
        let plan = self.engine.process_input(input);

        // Determine consciousness quadrant
        let quadrant = ConsciousnessQuadrant::from_metrics(self.phi, plan.goal.confidence);

        match self.format {
            OutputFormat::Json => {
                self.print_json_response(&plan, quadrant);
            }
            OutputFormat::Minimal => {
                self.print_minimal_response(&plan);
            }
            OutputFormat::Human => {
                self.print_human_response(&plan, quadrant);
            }
        }
    }

    /// Print human-readable response with consciousness metrics.
    fn print_human_response(
        &self,
        plan: &crate::mind::active_inference::ActionPlan,
        quadrant: ConsciousnessQuadrant,
    ) {
        // Consciousness bar
        println!();
        println!(
            "  Phi: {:.2}  Confidence: {:.2}  Quadrant: {}  FE: {:.3}",
            self.phi,
            plan.goal.confidence,
            quadrant.name(),
            plan.current_free_energy,
        );

        if plan.needs_clarification {
            println!();
            println!("  I'm not sure I understand. Could you be more specific?");
            println!("  Goal: {}", plan.goal.description);
            return;
        }

        println!("  Goal: {}", plan.goal.description);
        println!();

        // Show top actions
        let top_n = plan.actions.len().min(3);
        if top_n == 0 {
            println!("  No actions generated.");
            return;
        }

        println!("  Recommended actions:");
        for (i, action) in plan.actions.iter().take(top_n).enumerate() {
            let marker = if i == 0 { ">>>" } else { "   " };
            println!(
                "  {} {:?}  (pragmatic={:.2}, epistemic={:.2}, EFE={:.3})",
                marker,
                action.action,
                action.pragmatic_value,
                action.epistemic_value,
                action.expected_free_energy,
            );
        }
        println!();
    }

    /// Print JSON response.
    fn print_json_response(
        &self,
        plan: &crate::mind::active_inference::ActionPlan,
        quadrant: ConsciousnessQuadrant,
    ) {
        let response = serde_json::json!({
            "phi": self.phi,
            "confidence": plan.goal.confidence,
            "quadrant": quadrant.name(),
            "free_energy": plan.current_free_energy,
            "goal": plan.goal.description,
            "needs_clarification": plan.needs_clarification,
            "actions": plan.actions.iter().take(5).map(|a| {
                serde_json::json!({
                    "action": format!("{:?}", a.action),
                    "efe": a.expected_free_energy,
                    "pragmatic": a.pragmatic_value,
                    "epistemic": a.epistemic_value,
                })
            }).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }

    /// Print minimal response (just the action).
    fn print_minimal_response(&self, plan: &crate::mind::active_inference::ActionPlan) {
        if plan.needs_clarification {
            println!("CLARIFY: {}", plan.goal.description);
        } else if let Some(action) = plan.actions.first() {
            println!("{:?}", action.action);
        } else {
            println!("NO_ACTION");
        }
    }

    /// Print working memory contents.
    fn print_working_memory(&self) {
        let wm = self.engine.goal_inference().working_memory();
        println!();
        println!("  Working Memory ({}/{} items):", wm.len(), 7);
        println!("  Mean activation: {:.3}", wm.mean_activation());
        println!();
        for item in wm.items() {
            println!(
                "    [{:.2}] {:?}: {}",
                item.activation, item.source, item.label,
            );
        }
        println!();
    }

    /// Print system status.
    fn print_status(&self) {
        println!();
        println!("  Phi: {:.2}", self.phi);
        println!("  Free Energy: {:.3}", self.engine.free_energy());
        println!("  Episodes: {}", self.engine.episode_count());
        println!(
            "  Learned Actions: {}",
            self.engine.world_model().learned_action_count()
        );
        println!(
            "  Total Observations: {}",
            self.engine.world_model().total_observations()
        );
        println!("  Turn: {}", self.turn);
        println!();
    }

    /// Print the welcome banner.
    fn print_banner(&self) {
        println!();
        println!("  nix-mind: Conscious NixOS Management");
        println!("  Powered by HDC + Active Inference");
        println!();
        println!("  Type natural language or /help for commands.");
        if self.executor.history().is_empty() {
            // First time
        }
        println!();
    }

    /// Print the input prompt.
    fn print_prompt(&self) {
        let wm_count = self.engine.goal_inference().working_memory().len();
        print!("  [{}|wm:{}] > ", self.turn, wm_count);
    }

    /// Print help.
    fn print_help(&self) {
        println!();
        println!("  Natural Language Commands:");
        println!("    install <package>      Install a package");
        println!("    remove <package>       Remove a package");
        println!("    enable <service>       Enable a service");
        println!("    rebuild                Apply system configuration");
        println!("    why did X fail?        Diagnose issues");
        println!("    make it faster         Optimize system");
        println!();
        println!("  REPL Commands:");
        println!("    /memory, /wm           Show working memory");
        println!("    /status                Show consciousness metrics");
        println!("    /reset                 Clear context (new conversation)");
        println!("    /help, /?              Show this help");
        println!("    /quit, /exit, /q       Exit");
        println!();
    }
}

/// Process a single non-interactive input and print the result.
///
/// Used when the user passes natural language on the command line:
///   nix-mind "install firefox"
pub fn process_oneshot(
    input: &str,
    dry_run: bool,
    format: OutputFormat,
    phi_override: Option<f64>,
) {
    let mut repl = ConsciousRepl::new(dry_run, format);
    if let Some(phi) = phi_override {
        repl = repl.with_phi(phi);
    }
    repl.process_input(input);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_quadrant() {
        assert!(matches!(
            ConsciousnessQuadrant::from_metrics(0.8, 0.8),
            ConsciousnessQuadrant::Confident
        ));
        assert!(matches!(
            ConsciousnessQuadrant::from_metrics(0.8, 0.2),
            ConsciousnessQuadrant::Curious
        ));
        assert!(matches!(
            ConsciousnessQuadrant::from_metrics(0.2, 0.8),
            ConsciousnessQuadrant::Habitual
        ));
        assert!(matches!(
            ConsciousnessQuadrant::from_metrics(0.2, 0.2),
            ConsciousnessQuadrant::Confused
        ));
    }

    #[test]
    fn test_process_oneshot_human() {
        // Just verify it doesn't panic
        process_oneshot("install firefox", true, OutputFormat::Human, None);
    }

    #[test]
    fn test_process_oneshot_json() {
        process_oneshot("install firefox", true, OutputFormat::Json, None);
    }

    #[test]
    fn test_process_oneshot_minimal() {
        process_oneshot("install firefox", true, OutputFormat::Minimal, None);
    }

    #[test]
    fn test_process_oneshot_ambiguous() {
        process_oneshot("help", true, OutputFormat::Human, None);
    }

    #[test]
    fn test_repl_creation() {
        let repl = ConsciousRepl::new(true, OutputFormat::Human);
        assert_eq!(repl.turn, 0);
    }

    #[test]
    fn test_quadrant_threshold_boundary() {
        // Exactly at threshold should be "low" (> not >=)
        assert!(matches!(
            ConsciousnessQuadrant::from_metrics(QUADRANT_THRESHOLD, QUADRANT_THRESHOLD),
            ConsciousnessQuadrant::Confused
        ));
        // Just above should be "high"
        assert!(matches!(
            ConsciousnessQuadrant::from_metrics(
                QUADRANT_THRESHOLD + 0.01,
                QUADRANT_THRESHOLD + 0.01
            ),
            ConsciousnessQuadrant::Confident
        ));
    }

    #[test]
    fn test_quadrant_display() {
        let q = ConsciousnessQuadrant::Confident;
        assert_eq!(format!("{q}"), "Confident");
        let q = ConsciousnessQuadrant::Curious;
        assert_eq!(format!("{q}"), "Curious");
    }
}
