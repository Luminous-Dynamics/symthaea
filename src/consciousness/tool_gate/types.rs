// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tool Gate Types
//!
//! Risk lattice, tool descriptors, gate results, and fallback strategies
//! for the consciousness-gated tool use subsystem.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Risk Level (Lattice)
// ─────────────────────────────────────────────────────────────────────────────

/// Risk levels forming a lattice: ReadOnly < Reversible < Elevated < High < Critical.
///
/// Escalation rules (normative, not self-modifiable):
/// - Missing rollback → Critical (INV-7)
/// - Unknown domain → Elevated
/// - Calibration cold (< 10 outcomes) → +1 tier
/// - Missing metadata → +1 tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Read-only operations: queries, reads, searches.
    ReadOnly,
    /// Reversible operations with known rollback.
    Reversible,
    /// Operations requiring elevated consciousness.
    Elevated,
    /// High-risk operations with significant consequences.
    High,
    /// Critical/irreversible operations requiring maximum safeguards.
    Critical,
}

impl RiskLevel {
    /// Minimum Φ_eff required for this risk level.
    pub fn phi_threshold(self) -> f64 {
        match self {
            RiskLevel::ReadOnly => 0.1,
            RiskLevel::Reversible => 0.3,
            RiskLevel::Elevated => 0.5,
            RiskLevel::High => 0.7,
            RiskLevel::Critical => 0.9,
        }
    }

    /// Minimum plan confidence required (INV-8).
    pub fn confidence_threshold(self) -> f64 {
        match self {
            RiskLevel::ReadOnly => 0.0, // no confidence needed
            RiskLevel::Reversible => 0.2,
            RiskLevel::Elevated => 0.4,
            RiskLevel::High => 0.6,
            RiskLevel::Critical => 0.8,
        }
    }

    /// Escalate one tier (for missing metadata, cold calibration).
    pub fn escalate(self) -> Self {
        match self {
            RiskLevel::ReadOnly => RiskLevel::Reversible,
            RiskLevel::Reversible => RiskLevel::Elevated,
            RiskLevel::Elevated => RiskLevel::High,
            RiskLevel::High => RiskLevel::Critical,
            RiskLevel::Critical => RiskLevel::Critical, // already max
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool Descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Description of a tool or command for risk classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// Tool/command name.
    pub name: String,

    /// Domain (e.g., "nixos", "filesystem", "network", "system").
    pub domain: Option<String>,

    /// Whether the operation is read-only.
    pub is_read_only: bool,

    /// Rollback command, if known.
    pub rollback_command: Option<String>,

    /// Whether the tool has been calibrated (has posthoc outcome data).
    pub calibration_count: usize,

    /// Raw command string for pattern matching.
    pub raw_command: Option<String>,
}

impl ToolDescriptor {
    /// Create a read-only tool descriptor.
    pub fn read_only(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            domain: None,
            is_read_only: true,
            rollback_command: None,
            calibration_count: 0,
            raw_command: None,
        }
    }

    /// Create from a raw command string.
    pub fn from_command(command: impl Into<String>) -> Self {
        let cmd = command.into();
        let name = cmd.split_whitespace().next().unwrap_or("").to_string();
        Self {
            name,
            domain: None,
            is_read_only: false,
            rollback_command: None,
            calibration_count: 0,
            raw_command: Some(cmd),
        }
    }

    /// Builder: set domain.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Builder: set rollback command.
    pub fn with_rollback(mut self, rollback: impl Into<String>) -> Self {
        self.rollback_command = Some(rollback.into());
        self
    }

    /// Builder: set calibration count.
    pub fn with_calibration_count(mut self, count: usize) -> Self {
        self.calibration_count = count;
        self
    }

    /// Whether this tool has a known rollback.
    pub fn has_rollback(&self) -> bool {
        self.rollback_command.is_some()
    }

    /// Create a builder for constructing a ToolDescriptor.
    pub fn builder(name: impl Into<String>) -> ToolDescriptorBuilder {
        ToolDescriptorBuilder::new(name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolDescriptor Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for ergonomic construction of ToolDescriptor.
///
/// Provides sensible defaults:
/// - `domain`: None (triggers escalation to Elevated)
/// - `is_read_only`: false
/// - `rollback_command`: None (triggers escalation to Critical if not read-only)
/// - `calibration_count`: 0 (cold start)
/// - `raw_command`: None
///
/// # Example
///
/// ```ignore
/// let tool = ToolDescriptor::builder("nixos-rebuild")
///     .command("nixos-rebuild switch")
///     .domain("nixos")
///     .rollback("nixos-rebuild switch --rollback")
///     .calibration_count(100)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ToolDescriptorBuilder {
    name: String,
    domain: Option<String>,
    is_read_only: bool,
    rollback_command: Option<String>,
    calibration_count: usize,
    raw_command: Option<String>,
}

impl ToolDescriptorBuilder {
    /// Create a new builder with the given tool name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            domain: None,
            is_read_only: false,
            rollback_command: None,
            calibration_count: 0,
            raw_command: None,
        }
    }

    /// Set the raw command string.
    pub fn command(mut self, command: impl Into<String>) -> Self {
        self.raw_command = Some(command.into());
        self
    }

    /// Set the domain (e.g., "nixos", "filesystem", "network").
    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Mark as read-only.
    pub fn read_only(mut self) -> Self {
        self.is_read_only = true;
        self
    }

    /// Set the rollback command.
    pub fn rollback(mut self, rollback: impl Into<String>) -> Self {
        self.rollback_command = Some(rollback.into());
        self
    }

    /// Set the calibration count (number of historical outcomes).
    pub fn calibration_count(mut self, count: usize) -> Self {
        self.calibration_count = count;
        self
    }

    /// Build the ToolDescriptor.
    pub fn build(self) -> ToolDescriptor {
        ToolDescriptor {
            name: self.name,
            domain: self.domain,
            is_read_only: self.is_read_only,
            rollback_command: self.rollback_command,
            calibration_count: self.calibration_count,
            raw_command: self.raw_command,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate Decision
// ─────────────────────────────────────────────────────────────────────────────

/// Decision from the tool gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateDecision {
    /// Action is allowed.
    Allowed,
    /// Blocked: Φ_eff is insufficient.
    InsufficientPhi { current: f64, required: f64 },
    /// Blocked: plan confidence is insufficient (INV-8).
    InsufficientConfidence { current: f64, required: f64 },
}

impl GateDecision {
    /// Whether the action is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, GateDecision::Allowed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate Result (unified decision + fallback)
// ─────────────────────────────────────────────────────────────────────────────

/// Unified gate result: decision + fallback in one object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// The gate's decision.
    pub decision: GateDecision,
    /// Required Φ_eff for this risk level.
    pub required_phi: f64,
    /// Required plan confidence for this risk level.
    pub required_confidence: f64,
    /// Actual Φ_eff at decision time.
    pub actual_phi_eff: f64,
    /// Actual plan confidence at decision time.
    pub actual_confidence: f64,
    /// Classified risk level.
    pub risk_level: RiskLevel,
    /// Fallback strategy (always computed when not Allowed).
    pub fallback: Option<FallbackStrategy>,
}

impl GateResult {
    /// Whether the action was allowed.
    pub fn is_allowed(&self) -> bool {
        self.decision.is_allowed()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback Strategy
// ─────────────────────────────────────────────────────────────────────────────

/// Fallback when the tool gate blocks an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// Downgrade to a safer (read-only) variant of the command.
    DowngradeToReadOnly { alternative_command: String },
    /// Defer until Φ_eff increases.
    DeferUntilHigherPhi { required_phi: f64 },
    /// Request human confirmation.
    RequestHumanConfirmation { reason: String },
    /// Suggest an epistemic action instead.
    EpistemicAction {
        action: super::super::epistemic_conflict::EpistemicAction,
        reason: String,
    },
    /// No fallback available; action is fully blocked.
    NoFallback { reason: String },
}

impl FallbackStrategy {
    /// Variant label without inner data (avoids Debug formatting allocation).
    pub fn label(&self) -> &'static str {
        match self {
            FallbackStrategy::DowngradeToReadOnly { .. } => "DowngradeToReadOnly",
            FallbackStrategy::DeferUntilHigherPhi { .. } => "DeferUntilHigherPhi",
            FallbackStrategy::RequestHumanConfirmation { .. } => "RequestHumanConfirmation",
            FallbackStrategy::EpistemicAction { .. } => "EpistemicAction",
            FallbackStrategy::NoFallback { .. } => "NoFallback",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool Calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Per-tool calibration tracking (success rate, bounded updates).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCalibration {
    /// Tool name / pattern.
    pub tool_name: String,
    /// Total invocations.
    pub invocations: usize,
    /// Successful outcomes.
    pub successes: usize,
    /// Success rate = successes / invocations.
    pub success_rate: f64,
}

impl ToolCalibration {
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            invocations: 0,
            successes: 0,
            success_rate: 0.5, // uninformative prior
        }
    }

    /// Record an outcome (bounded step per INV-9 spirit).
    pub fn record(&mut self, success: bool) {
        self.invocations += 1;
        if success {
            self.successes += 1;
        }
        self.success_rate = self.successes as f64 / self.invocations as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_lattice_ordering() {
        assert!(RiskLevel::ReadOnly < RiskLevel::Reversible);
        assert!(RiskLevel::Reversible < RiskLevel::Elevated);
        assert!(RiskLevel::Elevated < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }

    #[test]
    fn test_escalation() {
        assert_eq!(RiskLevel::ReadOnly.escalate(), RiskLevel::Reversible);
        assert_eq!(RiskLevel::High.escalate(), RiskLevel::Critical);
        assert_eq!(RiskLevel::Critical.escalate(), RiskLevel::Critical);
    }

    #[test]
    fn test_phi_thresholds_monotonic() {
        let levels = [
            RiskLevel::ReadOnly,
            RiskLevel::Reversible,
            RiskLevel::Elevated,
            RiskLevel::High,
            RiskLevel::Critical,
        ];
        for i in 1..levels.len() {
            assert!(
                levels[i].phi_threshold() >= levels[i - 1].phi_threshold(),
                "Phi thresholds must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_confidence_thresholds_monotonic() {
        let levels = [
            RiskLevel::ReadOnly,
            RiskLevel::Reversible,
            RiskLevel::Elevated,
            RiskLevel::High,
            RiskLevel::Critical,
        ];
        for i in 1..levels.len() {
            assert!(
                levels[i].confidence_threshold() >= levels[i - 1].confidence_threshold(),
                "Confidence thresholds must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_tool_descriptor_read_only() {
        let td = ToolDescriptor::read_only("nix search");
        assert!(td.is_read_only);
        assert!(!td.has_rollback());
    }

    #[test]
    fn test_tool_descriptor_from_command() {
        let td = ToolDescriptor::from_command("nixos-rebuild switch")
            .with_domain("nixos")
            .with_rollback("nixos-rebuild switch --rollback");
        assert!(td.has_rollback());
        assert_eq!(td.domain, Some("nixos".to_string()));
    }

    #[test]
    fn test_tool_descriptor_builder_defaults() {
        let td = ToolDescriptor::builder("test-tool").build();
        assert_eq!(td.name, "test-tool");
        assert!(td.domain.is_none());
        assert!(!td.is_read_only);
        assert!(td.rollback_command.is_none());
        assert_eq!(td.calibration_count, 0);
        assert!(td.raw_command.is_none());
    }

    #[test]
    fn test_tool_descriptor_builder_full() {
        let td = ToolDescriptor::builder("nixos-rebuild")
            .command("nixos-rebuild switch")
            .domain("nixos")
            .rollback("nixos-rebuild switch --rollback")
            .calibration_count(50)
            .build();

        assert_eq!(td.name, "nixos-rebuild");
        assert_eq!(td.raw_command, Some("nixos-rebuild switch".to_string()));
        assert_eq!(td.domain, Some("nixos".to_string()));
        assert!(td.has_rollback());
        assert_eq!(td.calibration_count, 50);
        assert!(!td.is_read_only);
    }

    #[test]
    fn test_tool_descriptor_builder_read_only() {
        let td = ToolDescriptor::builder("nix-search")
            .command("nix search nixpkgs#emacs")
            .domain("nixos")
            .read_only()
            .build();

        assert!(td.is_read_only);
        assert!(!td.has_rollback()); // read-only doesn't need rollback
    }
}
