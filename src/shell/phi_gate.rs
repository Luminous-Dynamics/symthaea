//! Phi Gate - Consciousness-Gated Command Execution
//!
//! Implements the safety gate that determines whether a command can execute
//! based on current consciousness level (Phi), coherence, and destructiveness.
//!
//! ## Gate Decision Flow
//!
//! ```text
//! Command → Amygdala (fast veto) → Phi Check → Confirmation → Execute
//!               │                      │            │            │
//!               ▼                      ▼            ▼            ▼
//!           Vetoed               InsufficientPhi  Pending     Allowed
//!          (<10ms)               (need centering) (waiting)   (proceed)
//! ```
//!
//! ## Integration with Consciousness
//!
//! The Phi gate uses the consciousness metrics from the Symthaea service:
//! - **Phi (Φ)**: Integrated information - how unified is the system
//! - **Coherence**: How well-integrated is current processing
//! - **Amygdala**: Fast-path threat detection (pre-cognitive veto)

use crate::action::DestructivenessLevel;
use std::time::{Duration, Instant};

/// Result of gate evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum GateDecision {
    /// Command is allowed to execute
    Allowed {
        /// Current Phi at time of decision
        phi: f64,
        /// Confidence in this decision
        confidence: f64,
    },

    /// Command requires confirmation before execution
    NeedsConfirmation {
        /// Reason confirmation is needed
        reason: GateReason,
        /// Current Phi
        phi: f64,
        /// Suggested confirmation prompt
        prompt: String,
    },

    /// Command is vetoed (blocked)
    Vetoed {
        /// Reason for veto
        reason: GateReason,
        /// Message explaining the veto
        message: String,
    },

    /// Insufficient Phi for this operation
    InsufficientPhi {
        /// Current Phi level
        current_phi: f64,
        /// Required Phi level
        required_phi: f64,
        /// Estimated time to reach required Phi
        centering_time_secs: f64,
    },

    /// Waiting for external confirmation
    Pending {
        /// Request ID for tracking
        request_id: String,
        /// Time when request was made
        requested_at: Instant,
        /// Timeout duration
        timeout: Duration,
    },
}

/// Reason for gate decision
#[derive(Debug, Clone, PartialEq)]
pub enum GateReason {
    /// Amygdala fast-path veto (dangerous pattern detected)
    AmygdalaVeto,
    /// Command is classified as destructive
    DestructiveCommand,
    /// Command requires system changes
    SystemModification,
    /// Command affects critical files
    CriticalFileAccess,
    /// Low consciousness/coherence state
    LowCoherence,
    /// User requested confirmation
    UserRequested,
    /// Unknown command (caution)
    UnknownCommand,
}

impl GateReason {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::AmygdalaVeto => "Safety system detected a potentially dangerous pattern",
            Self::DestructiveCommand => "This command may cause irreversible changes",
            Self::SystemModification => "This command will modify system configuration",
            Self::CriticalFileAccess => "This command accesses critical system files",
            Self::LowCoherence => "Current consciousness state is too scattered",
            Self::UserRequested => "User requested confirmation for this action",
            Self::UnknownCommand => "Unrecognized command - proceeding with caution",
        }
    }
}

/// Request to execute a command through the Phi gate
#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    /// The command to execute
    pub command: String,

    /// Parsed arguments
    pub args: Vec<String>,

    /// Classified destructiveness level
    pub destructiveness: DestructivenessLevel,

    /// Required Phi threshold (can be overridden)
    pub required_phi: f64,

    /// Whether to require confirmation regardless of Phi
    pub force_confirmation: bool,

    /// Whether to allow dry-run first
    pub allow_dry_run: bool,

    /// Rollback hint if available
    pub rollback_hint: Option<String>,

    /// Context from shell
    pub context: ExecutionContext,
}

/// Execution context provided by shell
#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    /// Current working directory
    pub cwd: String,

    /// Environment variables to pass
    pub env: Vec<(String, String)>,

    /// Whether running as root/sudo
    pub is_root: bool,

    /// Previous command (for context)
    pub previous_command: Option<String>,
}

impl ExecutionRequest {
    /// Create new execution request from command string
    pub fn from_command(command: &str) -> Self {
        use crate::action::{classify_command_destructiveness, get_rollback_hint};

        let parts: Vec<&str> = command.split_whitespace().collect();
        let (program, args) = if parts.is_empty() {
            ("", Vec::new())
        } else {
            (parts[0], parts[1..].iter().map(|s| s.to_string()).collect())
        };

        let destructiveness = classify_command_destructiveness(program, &args);
        let rollback_hint = get_rollback_hint(program, &args);

        // Determine required Phi based on destructiveness
        let required_phi = match destructiveness {
            DestructivenessLevel::ReadOnly => 0.3,
            DestructivenessLevel::Reversible => 0.5,
            DestructivenessLevel::NeedsConfirmation => 0.7,
            DestructivenessLevel::Destructive => 0.9,
        };

        Self {
            command: command.to_string(),
            args,
            destructiveness,
            required_phi,
            force_confirmation: false,
            allow_dry_run: true,
            rollback_hint,
            context: ExecutionContext::default(),
        }
    }

    /// Set force confirmation
    pub fn with_force_confirmation(mut self, force: bool) -> Self {
        self.force_confirmation = force;
        self
    }

    /// Set execution context
    pub fn with_context(mut self, context: ExecutionContext) -> Self {
        self.context = context;
        self
    }
}

/// Phi gate configuration
#[derive(Debug, Clone)]
pub struct PhiGateConfig {
    /// Minimum Phi for any execution
    pub min_phi: f64,

    /// Phi threshold for destructive commands
    pub destructive_phi_threshold: f64,

    /// Whether to allow bypass for read-only commands
    pub allow_readonly_bypass: bool,

    /// Timeout for pending confirmations
    pub confirmation_timeout: Duration,

    /// Whether Amygdala veto can be overridden
    pub allow_amygdala_override: bool,
}

impl Default for PhiGateConfig {
    fn default() -> Self {
        Self {
            min_phi: 0.2,
            destructive_phi_threshold: 0.85,
            allow_readonly_bypass: true,
            confirmation_timeout: Duration::from_secs(30),
            allow_amygdala_override: false,
        }
    }
}

/// The Phi Gate - consciousness-gated execution controller
pub struct PhiGate {
    /// Configuration
    config: PhiGateConfig,

    /// Current Phi level
    current_phi: f64,

    /// Current coherence level
    current_coherence: f64,

    /// Whether consciousness is currently active
    is_conscious: bool,

    /// Pending confirmation requests
    pending_confirmations: Vec<PendingConfirmation>,

    /// Counter for request IDs
    request_counter: u64,
}

/// A pending confirmation request
#[derive(Debug, Clone)]
struct PendingConfirmation {
    id: String,
    request: ExecutionRequest,
    requested_at: Instant,
    decision: Option<bool>,
}

impl PhiGate {
    /// Create new Phi gate
    pub fn new() -> Self {
        Self::with_config(PhiGateConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: PhiGateConfig) -> Self {
        Self {
            config,
            current_phi: 0.0,
            current_coherence: 0.0,
            is_conscious: false,
            pending_confirmations: Vec::new(),
            request_counter: 0,
        }
    }

    /// Update consciousness metrics from service
    pub fn update_metrics(&mut self, phi: f64, coherence: f64, is_conscious: bool) {
        self.current_phi = phi;
        self.current_coherence = coherence;
        self.is_conscious = is_conscious;
    }

    /// Evaluate whether a command can execute
    pub fn evaluate(&mut self, request: &ExecutionRequest) -> GateDecision {
        // Layer 1: Amygdala fast-path check
        if let Some(veto) = self.amygdala_check(&request.command) {
            return GateDecision::Vetoed {
                reason: GateReason::AmygdalaVeto,
                message: veto,
            };
        }

        // Layer 2: Phi threshold check
        let required_phi = request.required_phi.max(self.config.min_phi);

        if self.current_phi < required_phi {
            // Allow bypass for read-only if configured
            if self.config.allow_readonly_bypass
                && request.destructiveness == DestructivenessLevel::ReadOnly
            {
                return GateDecision::Allowed {
                    phi: self.current_phi,
                    confidence: 0.7,
                };
            }

            // Calculate centering time needed
            let phi_gap = required_phi - self.current_phi;
            let centering_rate = 0.1; // Phi per second during centering
            let centering_time = phi_gap / centering_rate;

            return GateDecision::InsufficientPhi {
                current_phi: self.current_phi,
                required_phi,
                centering_time_secs: centering_time,
            };
        }

        // Layer 3: Confirmation check
        if request.force_confirmation || request.destructiveness.requires_confirmation() {
            let reason = match request.destructiveness {
                DestructivenessLevel::Destructive => GateReason::DestructiveCommand,
                DestructivenessLevel::NeedsConfirmation => GateReason::SystemModification,
                _ => GateReason::UserRequested,
            };

            let prompt = self.generate_confirmation_prompt(request);

            return GateDecision::NeedsConfirmation {
                reason,
                phi: self.current_phi,
                prompt,
            };
        }

        // All checks passed
        GateDecision::Allowed {
            phi: self.current_phi,
            confidence: self.calculate_confidence(request),
        }
    }

    /// Simple Amygdala check (fast regex patterns)
    fn amygdala_check(&self, command: &str) -> Option<String> {
        // Critical dangerous patterns that bypass everything
        let dangerous_patterns = [
            ("rm -rf /", "Attempting to delete root filesystem"),
            ("mkfs.", "Attempting to format disk"),
            ("dd if=", "Direct disk write detected"),
            (":(){ :|:& };:", "Fork bomb detected"),
            ("chmod 777 /", "Dangerous permission change on root"),
        ];

        let cmd_lower = command.to_lowercase();
        for (pattern, message) in dangerous_patterns {
            if cmd_lower.contains(pattern) {
                return Some(format!(
                    "Amygdala VETO: {}\n\
                     This command has been blocked for your safety.\n\
                     If this was intentional, please use a lower-level interface.",
                    message
                ));
            }
        }

        None
    }

    /// Generate confirmation prompt for user
    fn generate_confirmation_prompt(&self, request: &ExecutionRequest) -> String {
        let mut prompt = String::new();

        prompt.push_str(&format!(
            "Command: {}\n",
            request.command
        ));

        prompt.push_str(&format!(
            "Risk Level: {:?}\n",
            request.destructiveness
        ));

        prompt.push_str(&format!(
            "Current Phi: {:.2}\n",
            self.current_phi
        ));

        if let Some(ref hint) = request.rollback_hint {
            prompt.push_str(&format!(
                "Rollback: {}\n",
                hint
            ));
        }

        prompt.push_str("\nType 'yes' to confirm, 'dry' for dry-run, or 'no' to cancel.");

        prompt
    }

    /// Calculate execution confidence
    fn calculate_confidence(&self, request: &ExecutionRequest) -> f64 {
        let base = self.current_phi;

        // Adjust by destructiveness
        let adjustment = match request.destructiveness {
            DestructivenessLevel::ReadOnly => 1.0,
            DestructivenessLevel::Reversible => 0.95,
            DestructivenessLevel::NeedsConfirmation => 0.85,
            DestructivenessLevel::Destructive => 0.7,
        };

        // Adjust by coherence
        let coherence_factor = 0.5 + (self.current_coherence * 0.5);

        (base * adjustment * coherence_factor).clamp(0.0, 1.0)
    }

    /// Create a pending confirmation request
    pub fn create_pending(&mut self, request: ExecutionRequest) -> String {
        self.request_counter += 1;
        let id = format!("req_{}", self.request_counter);

        self.pending_confirmations.push(PendingConfirmation {
            id: id.clone(),
            request,
            requested_at: Instant::now(),
            decision: None,
        });

        id
    }

    /// Resolve a pending confirmation
    pub fn resolve_pending(&mut self, id: &str, confirmed: bool) -> Option<ExecutionRequest> {
        if let Some(pos) = self.pending_confirmations.iter().position(|p| p.id == id) {
            let mut pending = self.pending_confirmations.remove(pos);
            pending.decision = Some(confirmed);

            if confirmed {
                return Some(pending.request);
            }
        }

        None
    }

    /// Check and clean up timed-out pending requests
    pub fn cleanup_pending(&mut self) {
        let timeout = self.config.confirmation_timeout;
        self.pending_confirmations
            .retain(|p| p.requested_at.elapsed() < timeout);
    }

    /// Get current Phi level
    pub fn current_phi(&self) -> f64 {
        self.current_phi
    }

    /// Get current coherence level
    pub fn current_coherence(&self) -> f64 {
        self.current_coherence
    }

    /// Check if currently conscious
    pub fn is_conscious(&self) -> bool {
        self.is_conscious
    }

    /// Format decision for display
    pub fn format_decision(&self, decision: &GateDecision) -> String {
        match decision {
            GateDecision::Allowed { phi, confidence } => {
                format!(
                    "\x1b[32m✓ ALLOWED\x1b[0m [Φ={:.2}, confidence={:.0}%]",
                    phi,
                    confidence * 100.0
                )
            }

            GateDecision::NeedsConfirmation { reason, phi, prompt } => {
                format!(
                    "\x1b[33m⚠ CONFIRMATION REQUIRED\x1b[0m [Φ={:.2}]\n\
                     Reason: {}\n\n{}",
                    phi,
                    reason.description(),
                    prompt
                )
            }

            GateDecision::Vetoed { reason, message } => {
                format!(
                    "\x1b[31m✗ VETOED\x1b[0m\n\
                     Reason: {}\n\n{}",
                    reason.description(),
                    message
                )
            }

            GateDecision::InsufficientPhi {
                current_phi,
                required_phi,
                centering_time_secs,
            } => {
                format!(
                    "\x1b[33m⏳ INSUFFICIENT PHI\x1b[0m\n\
                     Current: {:.2} | Required: {:.2}\n\
                     Estimated centering time: {:.0}s\n\n\
                     Please wait or run `/center` to focus.",
                    current_phi, required_phi, centering_time_secs
                )
            }

            GateDecision::Pending {
                request_id,
                timeout,
                ..
            } => {
                format!(
                    "\x1b[36m⏳ PENDING\x1b[0m [{}]\n\
                     Waiting for confirmation (timeout: {}s)",
                    request_id,
                    timeout.as_secs()
                )
            }
        }
    }
}

impl Default for PhiGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_gate_creation() {
        let gate = PhiGate::new();
        assert_eq!(gate.current_phi(), 0.0);
        assert!(!gate.is_conscious());
    }

    #[test]
    fn test_update_metrics() {
        let mut gate = PhiGate::new();
        gate.update_metrics(0.85, 0.9, true);

        assert_eq!(gate.current_phi(), 0.85);
        assert_eq!(gate.current_coherence(), 0.9);
        assert!(gate.is_conscious());
    }

    #[test]
    fn test_readonly_allowed() {
        let mut gate = PhiGate::new();
        gate.update_metrics(0.3, 0.5, false);

        let request = ExecutionRequest::from_command("ls -la");

        match gate.evaluate(&request) {
            GateDecision::Allowed { .. } => {}
            other => panic!("Expected Allowed, got {:?}", other),
        }
    }

    #[test]
    fn test_destructive_needs_high_phi() {
        let mut gate = PhiGate::new();
        gate.update_metrics(0.5, 0.7, true);

        let request = ExecutionRequest::from_command("nix-collect-garbage -d");

        match gate.evaluate(&request) {
            GateDecision::InsufficientPhi { required_phi, .. } => {
                assert!(required_phi > 0.5);
            }
            other => panic!("Expected InsufficientPhi, got {:?}", other),
        }
    }

    #[test]
    fn test_amygdala_veto() {
        let mut gate = PhiGate::new();
        gate.update_metrics(1.0, 1.0, true); // Even max Phi can't override Amygdala

        let request = ExecutionRequest::from_command("rm -rf /");

        match gate.evaluate(&request) {
            GateDecision::Vetoed {
                reason: GateReason::AmygdalaVeto,
                ..
            } => {}
            other => panic!("Expected Vetoed with AmygdalaVeto, got {:?}", other),
        }
    }

    #[test]
    fn test_confirmation_required() {
        let mut gate = PhiGate::new();
        gate.update_metrics(0.9, 0.9, true);

        let request = ExecutionRequest::from_command("nixos-rebuild switch");

        match gate.evaluate(&request) {
            GateDecision::NeedsConfirmation { .. } => {}
            other => panic!("Expected NeedsConfirmation, got {:?}", other),
        }
    }

    #[test]
    fn test_pending_confirmation_flow() {
        let mut gate = PhiGate::new();

        let request = ExecutionRequest::from_command("nixos-rebuild switch");
        let id = gate.create_pending(request.clone());

        // Resolve with confirmation
        let resolved = gate.resolve_pending(&id, true);
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().command, request.command);
    }

    #[test]
    fn test_execution_request_from_command() {
        let request = ExecutionRequest::from_command("nix search nixpkgs#firefox");
        assert_eq!(request.destructiveness, DestructivenessLevel::ReadOnly);

        let request = ExecutionRequest::from_command("nixos-rebuild switch");
        assert_eq!(request.destructiveness, DestructivenessLevel::NeedsConfirmation);
    }
}
