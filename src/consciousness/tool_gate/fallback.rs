// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fallback Strategy Selection
//!
//! When the tool gate blocks an action, this module selects the
//! best alternative strategy based on the current state.

use super::super::epistemic_conflict::ConflictMatrix;
use super::types::*;

/// Select the best fallback strategy based on conflict state and gate result.
pub fn select_contextual_fallback(
    tool: &ToolDescriptor,
    gate_result: &GateResult,
    conflicts: &ConflictMatrix,
) -> FallbackStrategy {
    // If we already have a fallback from the gate, use it
    if let Some(ref fallback) = gate_result.fallback {
        return fallback.clone();
    }

    // Use conflict information to suggest epistemic actions
    if let Some(dominant_kind) = conflicts.dominant_kind {
        let action = dominant_kind.recommended_action();
        if action.is_external() {
            return FallbackStrategy::EpistemicAction {
                action,
                reason: format!(
                    "Dominant conflict ({:?}) suggests gathering more information",
                    dominant_kind
                ),
            };
        }
    }

    // Default: if we have a read-only alternative, use it
    if let Some(cmd) = &tool.raw_command {
        if let Some(alt) = suggest_safe_alternative(cmd) {
            return FallbackStrategy::DowngradeToReadOnly {
                alternative_command: alt,
            };
        }
    }

    // Last resort: request human confirmation
    FallbackStrategy::RequestHumanConfirmation {
        reason: format!(
            "Action '{}' blocked (Φ_eff={:.2}, confidence={:.2}). Risk: {:?}",
            tool.name,
            gate_result.actual_phi_eff,
            gate_result.actual_confidence,
            gate_result.risk_level,
        ),
    }
}

/// Suggest a safe read-only alternative for common commands.
fn suggest_safe_alternative(command: &str) -> Option<String> {
    let cmd = command.trim().to_lowercase();

    if cmd.contains("install") {
        let pkg = cmd.split_whitespace().last().unwrap_or("package");
        Some(format!("nix search nixpkgs {}", pkg))
    } else if cmd.contains("rebuild") {
        Some("nixos-rebuild dry-build".to_string())
    } else if cmd.contains("delete") || cmd.contains("remove") {
        Some("nix-store --query --referrers /nix/store/*".to_string())
    } else if cmd.contains("restart") || cmd.contains("stop") {
        let svc = cmd.split_whitespace().last().unwrap_or("service");
        Some(format!("systemctl status {}", svc))
    } else {
        None
    }
}

/// Compute a "frustration cost" for repeated fallbacks.
/// Higher values indicate the user may be getting frustrated.
pub fn fallback_frustration_cost(consecutive_blocks: usize) -> f64 {
    // Exponential frustration: blocking the same action 3+ times
    // means we should suggest human escalation
    match consecutive_blocks {
        0 => 0.0,
        1 => 0.1,
        2 => 0.3,
        3 => 0.6,
        _ => 1.0, // max frustration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_conflict::{ConflictKind, ConflictScore, TheoryId};

    fn empty_conflicts() -> ConflictMatrix {
        let conflicts: Vec<ConflictScore> = (0..15)
            .map(|i| {
                let a = TheoryId::ALL[i % 6];
                let b = TheoryId::ALL[(i + 1) % 6];
                ConflictScore::new(a, b, 0.0, ConflictKind::IntegrationCollapse)
            })
            .collect();
        ConflictMatrix::new(conflicts)
    }

    #[test]
    fn test_contextual_fallback_with_conflicts() {
        let tool = ToolDescriptor::from_command("nixos-rebuild switch");
        let gate_result = GateResult {
            decision: GateDecision::InsufficientPhi {
                current: 0.3,
                required: 0.7,
            },
            required_phi: 0.7,
            required_confidence: 0.6,
            actual_phi_eff: 0.3,
            actual_confidence: 0.8,
            risk_level: RiskLevel::High,
            fallback: None,
        };

        let mut conflicts_vec: Vec<ConflictScore> = (0..15)
            .map(|i| {
                let a = TheoryId::ALL[i % 6];
                let b = TheoryId::ALL[(i + 1) % 6];
                ConflictScore::new(a, b, 0.1, ConflictKind::IntegrationCollapse)
            })
            .collect();
        // Make one conflict dominant with NoBroadcast (which recommends Verify)
        conflicts_vec[1] =
            ConflictScore::new(TheoryId::IIT, TheoryId::AST, 0.8, ConflictKind::NoBroadcast);
        let conflicts = ConflictMatrix::new(conflicts_vec);

        let fallback = select_contextual_fallback(&tool, &gate_result, &conflicts);
        match fallback {
            FallbackStrategy::EpistemicAction { action, .. } => {
                assert!(action.is_external());
            }
            _ => panic!("Expected EpistemicAction fallback for dominant conflict"),
        }
    }

    #[test]
    fn test_frustration_cost() {
        assert_eq!(fallback_frustration_cost(0), 0.0);
        assert!(fallback_frustration_cost(3) > 0.5);
        assert_eq!(fallback_frustration_cost(10), 1.0);
    }
}
