//! Command executor with Phi-gating for the cognitive loop.
//!
//! Extracted from `mod.rs` to reduce file size.

use crate::action::DestructivenessLevel;
use crate::shell::context::{Completion, CompletionKind};
use crate::shell::ipc_server::{CommandExecutor, ExecutionResult, ValidationResult};

/// CommandExecutor implementation for CognitiveLoopService
///
/// Wraps the cognitive loop to provide command execution with Phi-gating.
#[allow(dead_code)]
pub(crate) struct CognitiveLoopExecutor {
    /// Reference to cognitive loop for Phi checks
    min_phi: f64,
    /// Current Phi value (updated from metrics)
    current_phi: f64,
}

#[allow(dead_code)]
impl CognitiveLoopExecutor {
    pub fn new(min_phi: f64) -> Self {
        Self {
            min_phi,
            current_phi: 0.5,
        }
    }

    /// Update current Phi from service
    pub fn update_phi(&mut self, phi: f64) {
        self.current_phi = phi;
    }
}

impl Default for CognitiveLoopExecutor {
    fn default() -> Self {
        Self::new(0.3)
    }
}

impl CommandExecutor for CognitiveLoopExecutor {
    fn execute(&self, command: &str, require_phi: Option<f64>) -> ExecutionResult {
        let required = require_phi.unwrap_or(self.min_phi);

        // Check Phi threshold
        if self.current_phi < required {
            return ExecutionResult {
                success: false,
                output: String::new(),
                phi_at_execution: self.current_phi,
                vetoed: true,
                veto_reason: Some(format!(
                    "Consciousness level ({:.2}) below required threshold ({:.2})",
                    self.current_phi, required
                )),
            };
        }

        // For now, return a stub result - actual execution would delegate to NixOS
        ExecutionResult {
            success: true,
            output: format!("[Phi={:.2}] Command queued: {}", self.current_phi, command),
            phi_at_execution: self.current_phi,
            vetoed: false,
            veto_reason: None,
        }
    }

    fn validate(&self, command: &str) -> ValidationResult {
        // Classify command destructiveness
        let cmd_lower = command.to_lowercase();

        let (safety_level, warnings) = if cmd_lower.contains("rm")
            || cmd_lower.contains("delete")
            || cmd_lower.contains("gc -d")
            || cmd_lower.contains("--delete")
        {
            (
                DestructivenessLevel::Destructive,
                vec!["This command may permanently delete data".to_string()],
            )
        } else if cmd_lower.contains("rebuild")
            || cmd_lower.contains("switch")
            || cmd_lower.contains("restart")
        {
            (
                DestructivenessLevel::NeedsConfirmation,
                vec!["This command may modify system state".to_string()],
            )
        } else if cmd_lower.contains("install")
            || cmd_lower.contains("remove")
            || cmd_lower.contains("update")
        {
            (DestructivenessLevel::Reversible, Vec::new())
        } else {
            (DestructivenessLevel::ReadOnly, Vec::new())
        };

        ValidationResult {
            valid: true,
            safety_level: format!("{safety_level:?}"),
            preview: Some(format!("Would execute: {command}")),
            warnings,
        }
    }

    fn get_completions(&self, input: &str, _cursor_pos: usize) -> Vec<Completion> {
        // Common NixOS commands for completion
        let commands = [
            ("install", "Install packages to environment"),
            ("remove", "Remove packages from environment"),
            ("search", "Search for packages"),
            ("rebuild", "Rebuild NixOS configuration"),
            ("switch", "Switch to new configuration"),
            ("rollback", "Rollback to previous generation"),
            ("gc", "Run garbage collection"),
            ("flake", "Flake management commands"),
            ("profile", "Profile management"),
            ("doctor", "Check system health"),
        ];

        commands
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(input))
            .map(|(cmd, desc)| {
                Completion::new(*cmd, CompletionKind::Command)
                    .with_docs(*desc)
                    .with_similarity(if *cmd == input { 1.0 } else { 0.8 })
            })
            .collect()
    }
}
