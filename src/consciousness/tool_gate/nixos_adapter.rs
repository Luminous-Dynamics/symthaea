// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Adapter: Backward-Compatible PhiGate Wrapper
//!
//! Provides `From<tool_gate::GateDecision>` for the existing
//! `shell::context::GateDecision`, ensuring backward compatibility
//! with the existing PhiGate system while adding the new tool gate.

use super::types::{GateDecision as ToolGateDecision, GateResult, RiskLevel, ToolDescriptor};
use crate::action::DestructivenessLevel;
use crate::shell::context::GateDecision as ShellGateDecision;

/// Convert a tool gate decision to a shell gate decision.
impl From<&GateResult> for ShellGateDecision {
    fn from(result: &GateResult) -> Self {
        match &result.decision {
            ToolGateDecision::Allowed => ShellGateDecision::Allowed {
                phi: result.actual_phi_eff,
                confidence: result.actual_confidence,
            },
            ToolGateDecision::InsufficientPhi { current, required } => {
                if *current < 0.3 {
                    ShellGateDecision::Vetoed {
                        reason: format!(
                            "Consciousness level ({:.2}) too low (requires {:.2})",
                            current, required,
                        ),
                        message: "Wait for consciousness to stabilize".to_string(),
                    }
                } else {
                    ShellGateDecision::NeedsConfirmation {
                        phi: *current,
                        reason: format!(
                            "Φ_eff ({:.2}) below threshold ({:.2}) for {:?} risk",
                            current, required, result.risk_level,
                        ),
                        prompt: format!("Confirm action at risk level {:?}?", result.risk_level),
                    }
                }
            }
            ToolGateDecision::InsufficientConfidence { current, required } => {
                ShellGateDecision::NeedsConfirmation {
                    phi: result.actual_phi_eff,
                    reason: format!(
                        "Plan confidence ({:.2}) below threshold ({:.2}) for {:?} risk",
                        current, required, result.risk_level,
                    ),
                    prompt: "Confirm action despite low plan confidence?".to_string(),
                }
            }
        }
    }
}

/// Convert an existing `DestructivenessLevel` to a `RiskLevel`.
impl From<DestructivenessLevel> for RiskLevel {
    fn from(level: DestructivenessLevel) -> Self {
        match level {
            DestructivenessLevel::ReadOnly => RiskLevel::ReadOnly,
            DestructivenessLevel::Reversible => RiskLevel::Reversible,
            DestructivenessLevel::NeedsConfirmation => RiskLevel::Elevated,
            DestructivenessLevel::Destructive => RiskLevel::Critical,
        }
    }
}

/// Create a `ToolDescriptor` from a shell command with existing classification.
pub fn tool_descriptor_from_shell_command(
    command: &str,
    destructiveness: DestructivenessLevel,
) -> ToolDescriptor {
    let mut td = ToolDescriptor::from_command(command);
    td.domain = Some("nixos".to_string());
    td.is_read_only = destructiveness == DestructivenessLevel::ReadOnly;

    // Set rollback for known reversible commands
    if let Some(rollback) = get_nixos_rollback(command) {
        td.rollback_command = Some(rollback);
    }

    td
}

/// Get the rollback command for known NixOS operations.
fn get_nixos_rollback(command: &str) -> Option<String> {
    let cmd = command.trim().to_lowercase();

    if cmd.starts_with("nixos-rebuild switch") || cmd.starts_with("nixos-rebuild boot") {
        Some("nixos-rebuild switch --rollback".to_string())
    } else if cmd.contains("nix-env -i") || cmd.contains("nix profile install") {
        let pkg = cmd.split_whitespace().last().unwrap_or("package");
        Some(format!("nix-env -e {} || nix profile remove {}", pkg, pkg))
    } else if cmd.starts_with("nix build")
        || cmd.starts_with("nix develop")
        || cmd.starts_with("nix shell")
    {
        Some("exit".to_string()) // nix environments are ephemeral
    } else if cmd.starts_with("systemctl restart") || cmd.starts_with("systemctl stop") {
        let svc = cmd.split_whitespace().last().unwrap_or("service");
        Some(format!("systemctl start {}", svc))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_result_to_shell_allowed() {
        let result = GateResult {
            decision: ToolGateDecision::Allowed,
            required_phi: 0.3,
            required_confidence: 0.2,
            actual_phi_eff: 0.8,
            actual_confidence: 0.9,
            risk_level: RiskLevel::Reversible,
            fallback: None,
        };
        let shell: ShellGateDecision = (&result).into();
        assert!(shell.is_allowed());
    }

    #[test]
    fn test_gate_result_to_shell_vetoed() {
        let result = GateResult {
            decision: ToolGateDecision::InsufficientPhi {
                current: 0.1,
                required: 0.7,
            },
            required_phi: 0.7,
            required_confidence: 0.6,
            actual_phi_eff: 0.1,
            actual_confidence: 0.9,
            risk_level: RiskLevel::High,
            fallback: None,
        };
        let shell: ShellGateDecision = (&result).into();
        assert!(shell.is_vetoed());
    }

    #[test]
    fn test_destructiveness_to_risk_level() {
        assert_eq!(
            RiskLevel::from(DestructivenessLevel::ReadOnly),
            RiskLevel::ReadOnly
        );
        assert_eq!(
            RiskLevel::from(DestructivenessLevel::Destructive),
            RiskLevel::Critical
        );
    }

    #[test]
    fn test_nixos_rollback_known() {
        assert!(get_nixos_rollback("nixos-rebuild switch").is_some());
        assert!(get_nixos_rollback("nix-env -i firefox").is_some());
        assert!(get_nixos_rollback("nix build .#package").is_some());
    }

    #[test]
    fn test_tool_descriptor_from_shell() {
        let td = tool_descriptor_from_shell_command(
            "nix search nixpkgs firefox",
            DestructivenessLevel::ReadOnly,
        );
        assert!(td.is_read_only);
        assert_eq!(td.domain, Some("nixos".to_string()));
    }
}
