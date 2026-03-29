// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tool Risk Classifier
//!
//! Classifies tools into the risk lattice using pattern matching
//! and escalation rules. Normative rules — not self-modifiable.

use super::types::*;

/// Classify a tool descriptor into a risk level.
///
/// Escalation rules (normative):
/// - Read-only → ReadOnly
/// - Has rollback → Reversible
/// - No rollback + not read-only → Critical (INV-7)
/// - Unknown domain → Elevated
/// - Cold calibration (< 10 outcomes) → +1 tier
pub fn classify(tool: &ToolDescriptor) -> RiskLevel {
    // Step 1: Base classification
    let base = if tool.is_read_only {
        RiskLevel::ReadOnly
    } else if let Some(cmd) = &tool.raw_command {
        classify_command(cmd)
    } else if tool.has_rollback() {
        RiskLevel::Reversible
    } else {
        // INV-7: No provable reversibility → Critical
        RiskLevel::Critical
    };

    // Step 2: Escalation rules
    let mut level = base;

    // INV-7: Missing rollback for non-read-only → escalate to Critical
    if !tool.is_read_only && !tool.has_rollback() && level < RiskLevel::Critical {
        level = RiskLevel::Critical;
    }

    // Unknown domain → at least Elevated (but not for read-only)
    if !tool.is_read_only && tool.domain.is_none() && level < RiskLevel::Elevated {
        level = level.escalate();
    }

    // Cold calibration → +1 tier (but not for read-only)
    if !tool.is_read_only && tool.calibration_count < 10 && level < RiskLevel::Critical {
        level = level.escalate();
    }

    level
}

/// Classify a raw command string into a base risk level.
fn classify_command(command: &str) -> RiskLevel {
    let cmd = command.trim().to_lowercase();

    // Destructive / irreversible
    if cmd.contains("--purge")
        || cmd.contains("gc")
        || cmd.contains("delete")
        || cmd.contains("rm ")
        || cmd.contains("remove")
        || cmd.starts_with("nix-collect-garbage")
        || cmd.contains("wipe")
        || cmd.contains("format")
    {
        return RiskLevel::Critical;
    }

    // High risk: system-level changes
    if cmd.starts_with("nixos-rebuild") || cmd.contains("switch") {
        return RiskLevel::High;
    }

    // Elevated: package management
    if cmd.starts_with("nix profile install")
        || cmd.starts_with("nix-env -i")
        || cmd.contains("upgrade")
        || cmd.contains("update")
    {
        return RiskLevel::Elevated;
    }

    // Reversible
    if cmd.starts_with("nix build")
        || cmd.starts_with("nix develop")
        || cmd.starts_with("nix shell")
    {
        return RiskLevel::Reversible;
    }

    // Read-only
    if cmd.starts_with("nix search")
        || cmd.starts_with("nix-env -q")
        || cmd.starts_with("nixos-option")
        || cmd.contains("list")
        || cmd.contains("info")
        || cmd.contains("show")
        || cmd.contains("search")
        || cmd.starts_with("nix flake show")
        || cmd.starts_with("nix flake metadata")
        || cmd.starts_with("cat ")
        || cmd.starts_with("ls ")
        || cmd.starts_with("echo ")
    {
        return RiskLevel::ReadOnly;
    }

    // Default: Elevated for unknown commands
    RiskLevel::Elevated
}

/// Gate a tool action against Φ_eff AND plan_confidence.
///
/// # INV-8: Confidence/Action Alignment
/// Even if Φ_eff is high, low plan_confidence blocks risky actions.
pub fn gate(tool: &ToolDescriptor, phi_eff: f64, plan_confidence: f64) -> GateResult {
    let risk = classify(tool);
    let required_phi = risk.phi_threshold();
    let required_conf = risk.confidence_threshold();

    let decision = if phi_eff < required_phi {
        GateDecision::InsufficientPhi {
            current: phi_eff,
            required: required_phi,
        }
    } else if plan_confidence < required_conf {
        // INV-8: high Φ but low confidence → still blocked
        GateDecision::InsufficientConfidence {
            current: plan_confidence,
            required: required_conf,
        }
    } else {
        GateDecision::Allowed
    };

    let fallback = if !decision.is_allowed() {
        Some(select_fallback(tool, &decision, phi_eff))
    } else {
        None
    };

    GateResult {
        decision,
        required_phi,
        required_confidence: required_conf,
        actual_phi_eff: phi_eff,
        actual_confidence: plan_confidence,
        risk_level: risk,
        fallback,
    }
}

/// Select an appropriate fallback strategy when gate blocks.
fn select_fallback(
    tool: &ToolDescriptor,
    decision: &GateDecision,
    _phi_eff: f64,
) -> FallbackStrategy {
    match decision {
        GateDecision::InsufficientPhi { required, .. } => {
            // If it's a known NixOS command, suggest read-only alternative
            if let Some(cmd) = &tool.raw_command {
                if let Some(alt) = suggest_readonly_alternative(cmd) {
                    return FallbackStrategy::DowngradeToReadOnly {
                        alternative_command: alt,
                    };
                }
            }
            FallbackStrategy::DeferUntilHigherPhi {
                required_phi: *required,
            }
        }
        GateDecision::InsufficientConfidence { .. } => FallbackStrategy::RequestHumanConfirmation {
            reason: "Plan confidence is too low for this risk level".to_string(),
        },
        GateDecision::Allowed => {
            // Caller guards against this, but degrade gracefully if reached
            tracing::warn!("select_fallback called for Allowed decision — returning no-op");
            FallbackStrategy::RequestHumanConfirmation {
                reason: "Unexpected: fallback requested for already-allowed gate decision"
                    .to_string(),
            }
        }
    }
}

/// Suggest a read-only alternative for common NixOS commands.
fn suggest_readonly_alternative(command: &str) -> Option<String> {
    let cmd = command.trim().to_lowercase();

    if cmd.starts_with("nixos-rebuild switch") {
        Some("nixos-rebuild dry-build".to_string())
    } else if cmd.starts_with("nix profile install") || cmd.starts_with("nix-env -i") {
        let pkg = cmd.split_whitespace().last().unwrap_or("package");
        Some(format!("nix search nixpkgs {}", pkg))
    } else if cmd.starts_with("nix-collect-garbage") {
        Some("nix-store --query --roots /nix/store/*".to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_only_classification() {
        let tool = ToolDescriptor::read_only("nix search");
        assert_eq!(classify(&tool), RiskLevel::ReadOnly);
    }

    #[test]
    fn test_inv7_missing_rollback_escalates() {
        // INV-7: Missing rollback → automatic escalation to Critical
        let tool = ToolDescriptor::from_command("some-unknown-command").with_domain("unknown");
        let risk = classify(&tool);
        assert_eq!(
            risk,
            RiskLevel::Critical,
            "INV-7: Missing rollback must escalate to Critical"
        );
    }

    #[test]
    fn test_reversible_with_rollback() {
        let tool = ToolDescriptor::from_command("nix build .#package")
            .with_domain("nixos")
            .with_rollback("nix store delete")
            .with_calibration_count(100);
        let risk = classify(&tool);
        assert_eq!(risk, RiskLevel::Reversible);
    }

    #[test]
    fn test_destructive_command() {
        let tool = ToolDescriptor::from_command("nix-collect-garbage -d").with_domain("nixos");
        let risk = classify(&tool);
        assert_eq!(risk, RiskLevel::Critical);
    }

    #[test]
    fn test_inv8_low_confidence_blocks_high() {
        // INV-8: High Φ_eff but low confidence → blocked
        let tool = ToolDescriptor::from_command("nixos-rebuild switch")
            .with_domain("nixos")
            .with_rollback("nixos-rebuild switch --rollback")
            .with_calibration_count(100);
        let result = gate(&tool, 0.9, 0.1); // high phi, low confidence
        assert!(
            !result.is_allowed(),
            "INV-8: Low confidence should block even with high Φ_eff"
        );
    }

    #[test]
    fn test_gate_allows_with_sufficient_phi_and_confidence() {
        let tool =
            ToolDescriptor::read_only("nix search nixpkgs firefox").with_calibration_count(100);
        let result = gate(&tool, 0.5, 0.5);
        assert!(result.is_allowed());
    }

    #[test]
    fn test_gate_blocks_insufficient_phi() {
        let tool = ToolDescriptor::from_command("nixos-rebuild switch")
            .with_domain("nixos")
            .with_rollback("nixos-rebuild switch --rollback")
            .with_calibration_count(100);
        let result = gate(&tool, 0.3, 0.9); // low phi, high confidence
        assert!(!result.is_allowed());
        assert!(result.fallback.is_some());
    }

    #[test]
    fn test_cold_calibration_escalates() {
        let tool = ToolDescriptor::from_command("nix build .#pkg")
            .with_domain("nixos")
            .with_rollback("nix store delete")
            .with_calibration_count(5); // cold
        let risk = classify(&tool);
        // Reversible + cold → Elevated
        assert!(
            risk >= RiskLevel::Elevated,
            "Cold calibration should escalate"
        );
    }

    #[test]
    fn test_fallback_downgrade_suggestion() {
        let tool = ToolDescriptor::from_command("nixos-rebuild switch")
            .with_domain("nixos")
            .with_rollback("nixos-rebuild switch --rollback")
            .with_calibration_count(100);
        let result = gate(&tool, 0.3, 0.9);
        if let Some(FallbackStrategy::DowngradeToReadOnly {
            alternative_command,
        }) = &result.fallback
        {
            assert!(alternative_command.contains("dry-build"));
        }
    }
}
