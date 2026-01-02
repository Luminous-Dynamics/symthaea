//! Shell Module - AI-Native Sidecar Shell Infrastructure
//!
//! Provides consciousness-aware shell functionality:
//! - IntelliSense: HDC-based semantic command completion
//! - PhiGate: Consciousness-gated execution with DestructivenessLevel
//! - IPC Client: Async Unix socket communication with symthaea-service
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SHELL SIDECAR                             │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ IntelliSense│  │  PhiGate    │  │   IPC Client        │  │
//! │  │ (completions)│  │  (gating)   │  │   (service comms)   │  │
//! │  └──────┬───────┘  └──────┬──────┘  └──────────┬──────────┘  │
//! │         └──────────────────┴───────────────────┘             │
//! │                              │                                │
//! │                 Unix Socket (symthaea.sock)                   │
//! └──────────────────────────────┼────────────────────────────────┘
//!                                │
//!                     ┌──────────▼──────────┐
//!                     │  SYMTHAEA SERVICE   │
//!                     │  (consciousness,    │
//!                     │   safety, HDC)      │
//!                     └─────────────────────┘
//! ```

pub mod intellisense;
pub mod phi_gate;
pub mod ipc_client;

pub use intellisense::{IntelliSenseEngine, Completion, CompletionKind};
pub use phi_gate::{PhiGate, GateDecision, GateReason, ExecutionRequest};
pub use ipc_client::{ShellIpcClient, IpcError};

use crate::action::DestructivenessLevel;

/// Shell context for tracking state across interactions
#[derive(Debug, Clone)]
pub struct ShellContext {
    /// Current working directory
    pub cwd: String,

    /// Command history (most recent first)
    pub history: Vec<String>,

    /// Maximum history size
    pub max_history: usize,

    /// Current Phi level from last service query
    pub current_phi: f64,

    /// Current coherence level
    pub current_coherence: f64,

    /// Whether consciousness is active
    pub is_conscious: bool,
}

impl Default for ShellContext {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "/".to_string()),
            history: Vec::new(),
            max_history: 1000,
            current_phi: 0.0,
            current_coherence: 0.0,
            is_conscious: false,
        }
    }
}

impl ShellContext {
    /// Create new shell context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add command to history
    pub fn add_to_history(&mut self, command: String) {
        // Don't add duplicates of last command
        if self.history.first() != Some(&command) {
            self.history.insert(0, command);

            // Trim history if too long
            if self.history.len() > self.max_history {
                self.history.truncate(self.max_history);
            }
        }
    }

    /// Update consciousness metrics
    pub fn update_metrics(&mut self, phi: f64, coherence: f64, is_conscious: bool) {
        self.current_phi = phi;
        self.current_coherence = coherence;
        self.is_conscious = is_conscious;
    }

    /// Get consciousness indicator for prompt
    pub fn consciousness_indicator(&self) -> &'static str {
        if self.is_conscious { "●" } else { "○" }
    }

    /// Get status color (ANSI escape code)
    pub fn status_color(&self) -> &'static str {
        if self.current_phi >= 0.7 {
            "\x1b[32m" // Green - high consciousness
        } else if self.current_phi >= 0.4 {
            "\x1b[33m" // Yellow - moderate
        } else {
            "\x1b[31m" // Red - low
        }
    }

    /// Format prompt string with metrics
    pub fn format_prompt(&self) -> String {
        format!(
            "{}{}[Φ:{:.2}|C:{:.0}%]\x1b[0m symthaea> ",
            self.status_color(),
            self.consciousness_indicator(),
            self.current_phi,
            self.current_coherence * 100.0
        )
    }
}

/// Command classification result
#[derive(Debug, Clone)]
pub struct CommandClassification {
    /// The command being classified
    pub command: String,

    /// Destructiveness level
    pub destructiveness: DestructivenessLevel,

    /// Required Phi threshold for execution
    pub required_phi: f64,

    /// Whether confirmation is needed
    pub needs_confirmation: bool,

    /// Rollback hint if available
    pub rollback_hint: Option<String>,

    /// Safety warnings
    pub warnings: Vec<String>,
}

impl CommandClassification {
    /// Create from command string
    pub fn from_command(command: &str) -> Self {
        use crate::action::{classify_command_destructiveness, get_rollback_hint};

        let parts: Vec<&str> = command.split_whitespace().collect();
        let (program, args) = if parts.is_empty() {
            ("", Vec::new())
        } else {
            (parts[0], parts[1..].to_vec())
        };

        let args_string: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let destructiveness = classify_command_destructiveness(program, &args_string);
        let rollback_hint = get_rollback_hint(program, &args_string);

        // Determine required Phi based on destructiveness
        let required_phi = match destructiveness {
            DestructivenessLevel::ReadOnly => 0.3,
            DestructivenessLevel::Reversible => 0.5,
            DestructivenessLevel::NeedsConfirmation => 0.7,
            DestructivenessLevel::Destructive => 0.9,
        };

        let needs_confirmation = destructiveness.requires_confirmation();

        // Generate warnings based on command analysis
        let mut warnings = Vec::new();
        if destructiveness >= DestructivenessLevel::NeedsConfirmation {
            warnings.push(format!(
                "This command is classified as {:?}",
                destructiveness
            ));
        }

        Self {
            command: command.to_string(),
            destructiveness,
            required_phi,
            needs_confirmation,
            rollback_hint,
            warnings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_context_default() {
        let ctx = ShellContext::default();
        assert!(ctx.history.is_empty());
        assert_eq!(ctx.max_history, 1000);
        assert_eq!(ctx.current_phi, 0.0);
    }

    #[test]
    fn test_add_to_history() {
        let mut ctx = ShellContext::new();
        ctx.add_to_history("ls".to_string());
        ctx.add_to_history("pwd".to_string());

        assert_eq!(ctx.history.len(), 2);
        assert_eq!(ctx.history[0], "pwd");
        assert_eq!(ctx.history[1], "ls");
    }

    #[test]
    fn test_no_duplicate_history() {
        let mut ctx = ShellContext::new();
        ctx.add_to_history("ls".to_string());
        ctx.add_to_history("ls".to_string());

        assert_eq!(ctx.history.len(), 1);
    }

    #[test]
    fn test_command_classification() {
        let safe = CommandClassification::from_command("nix search nixpkgs#firefox");
        assert_eq!(safe.destructiveness, DestructivenessLevel::ReadOnly);
        assert!(!safe.needs_confirmation);

        let destructive = CommandClassification::from_command("nix-collect-garbage -d");
        assert_eq!(destructive.destructiveness, DestructivenessLevel::Destructive);
        assert!(destructive.needs_confirmation);
    }
}
