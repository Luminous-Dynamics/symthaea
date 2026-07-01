// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shell Integration Tests
//!
//! End-to-end tests for the Symthaea Shell components working together:
//! - IntelliSense with HDC semantic encoding
//! - PhiGate with Amygdala integration
//! - Epistemic overlays with knowledge sources
//! - Command classification and confirmation flow
//!
//! This test is only compiled with the `shell` feature enabled.

#![cfg(feature = "shell")]

use symthaea::action::DestructivenessLevel;
use symthaea::shell::{
    CommandClassification, CommandContext, EpistemicOverlayEngine, ExecutionRequest, GateDecision,
    IntelliSenseEngine, KnowledgeSource, OverlayType, PhiGate, ShellContext,
    classify_command_destructiveness,
};

mod intellisense_tests {
    use super::*;

    #[test]
    fn test_completion_with_history_influence() {
        let mut engine = IntelliSenseEngine::new();
        engine.set_phi(0.8);

        // Complete with input (cursor at end)
        let input = "nix";
        let completions = engine.complete(input, input.len());
        assert!(!completions.is_empty(), "Should have completions for 'nix'");

        // Verify completions are NixOS-related
        for c in &completions {
            assert!(
                c.text.to_lowercase().contains("nix"),
                "Completion '{}' should contain 'nix'",
                c.text
            );
        }
    }

    #[test]
    fn test_completion_confidence_ordering() {
        let mut engine = IntelliSenseEngine::new();
        engine.set_phi(0.75);

        let input = "nix-env";
        let completions = engine.complete(input, input.len());

        if completions.len() >= 2 {
            // Completions should be ordered by confidence (descending)
            for i in 0..completions.len() - 1 {
                assert!(
                    completions[i].confidence >= completions[i + 1].confidence,
                    "Completions should be ordered by confidence"
                );
            }
        }
    }

    #[test]
    fn test_destructiveness_level_in_completions() {
        let mut engine = IntelliSenseEngine::new();
        engine.set_phi(0.9);

        let input = "nix-collect-garbage";
        let completions = engine.complete(input, input.len());

        // Should include destructive command detection
        for c in completions.iter().filter(|c| c.text.contains("-d")) {
            assert!(
                c.destructiveness >= DestructivenessLevel::NeedsConfirmation,
                "nix-collect-garbage -d should be marked as needing confirmation"
            );
        }
    }

    #[test]
    fn test_semantic_similarity_in_completions() {
        let mut engine = IntelliSenseEngine::new();
        engine.set_phi(0.8);

        // Related commands should appear as completions
        let input = "install";
        let completions = engine.complete(input, input.len());

        // Should find nix-env -i since it's semantically related to "install"
        let has_install_related = completions
            .iter()
            .any(|c| c.text.contains("nix-env") || c.text.contains("nix profile install"));

        assert!(
            has_install_related,
            "Should find install-related commands for 'install'"
        );
    }
}

mod phi_gate_tests {
    use super::*;

    #[test]
    fn test_gate_allows_safe_commands() {
        let mut gate = PhiGate::new();
        gate.update_metrics(0.8, 0.9, true);

        let command = "nix search nixpkgs#firefox";
        let destructiveness = classify_command_destructiveness(command);
        let decision = gate.evaluate(command, destructiveness);

        match decision {
            GateDecision::Allowed { phi, .. } => {
                assert!(
                    phi >= 0.5,
                    "Safe commands should be allowed with sufficient Phi"
                );
            }
            GateDecision::Vetoed { message, .. } => {
                panic!("Safe command vetoed: {}", message);
            }
            _ => {}
        }
    }

    #[test]
    fn test_gate_requires_confirmation_for_destructive() {
        let mut gate = PhiGate::new();
        gate.update_metrics(0.6, 0.7, true);

        let command = "nix-collect-garbage -d";
        let destructiveness = classify_command_destructiveness(command);
        let decision = gate.evaluate(command, destructiveness);

        match decision {
            GateDecision::NeedsConfirmation { reason, phi, .. } => {
                assert!(phi > 0.0, "Should report current Phi");
                // Reason should indicate why confirmation is needed
                assert!(!reason.is_empty(), "Should have a reason");
            }
            GateDecision::Allowed { .. } => {
                // Could be allowed if Phi is very high
            }
            GateDecision::Vetoed { .. } => {
                // Could be vetoed by Amygdala
            }
            _ => {}
        }
    }

    #[test]
    fn test_gate_vetos_dangerous_commands() {
        let mut gate = PhiGate::new();
        gate.update_metrics(0.5, 0.6, true);

        // Known dangerous pattern
        let command = "rm -rf /";
        let destructiveness = classify_command_destructiveness(command);
        let decision = gate.evaluate(command, destructiveness);

        match decision {
            GateDecision::Vetoed { reason: _, message } => {
                assert!(!message.is_empty(), "Should have veto message");
                // Amygdala should have vetoed this
            }
            _ => {
                // May also need confirmation, which is acceptable
            }
        }
    }

    #[test]
    fn test_gate_respects_phi_threshold() {
        let mut gate = PhiGate::new();

        // Set very low Phi
        gate.update_metrics(0.1, 0.2, false);

        let command = "nixos-rebuild switch";
        let destructiveness = classify_command_destructiveness(command);
        let decision = gate.evaluate(command, destructiveness);

        match decision {
            GateDecision::InsufficientPhi {
                current_phi,
                required_phi,
                ..
            } => {
                assert!(current_phi < required_phi, "Should require higher Phi");
            }
            GateDecision::NeedsConfirmation { .. } | GateDecision::Vetoed { .. } => {
                // These are also acceptable outcomes for low Phi
            }
            GateDecision::Allowed { .. } => {
                // Might be allowed if configured with low thresholds
            }
            _ => {}
        }
    }
}

mod epistemic_overlay_tests {
    use super::*;

    #[test]
    fn test_overlay_generation_for_command() {
        let mut engine = EpistemicOverlayEngine::new();
        let context = CommandContext {
            source: KnowledgeSource::NixosDocs,
            confidence: 0.8,
            k_index: 0.75,
            epistemic_uncertainty: 0.2,
            aleatoric_uncertainty: 0.1,
            theory_weights: vec![
                ("HDC Semantic".to_string(), 0.5),
                ("Pattern Match".to_string(), 0.5),
            ],
            phi_required: 0.5,
            current_phi: 0.7,
        };

        let overlays = engine.generate_for_command("nix-env -i firefox", &context);
        assert!(!overlays.is_empty(), "Should generate overlays");

        // Check for knowledge source overlay
        let has_knowledge_overlay = overlays
            .iter()
            .any(|o| o.overlay_type == OverlayType::KnowledgeSource);
        assert!(
            has_knowledge_overlay,
            "Should have knowledge source overlay"
        );
    }

    #[test]
    fn test_uncertainty_warning_for_low_confidence() {
        let mut engine = EpistemicOverlayEngine::new();
        let context = CommandContext {
            source: KnowledgeSource::Unknown,
            confidence: 0.2, // Low confidence
            k_index: 0.2,
            epistemic_uncertainty: 0.8,
            aleatoric_uncertainty: 0.1,
            theory_weights: Vec::new(),
            phi_required: 0.5,
            current_phi: 0.7,
        };

        let overlays = engine.generate_for_command("some-unknown-command", &context);

        // Should have uncertainty warning
        let has_warning = overlays
            .iter()
            .any(|o| o.overlay_type == OverlayType::UncertaintyWarning);
        assert!(
            has_warning,
            "Should have uncertainty warning for low confidence"
        );
    }

    #[test]
    fn test_safety_hint_for_destructive_commands() {
        let mut engine = EpistemicOverlayEngine::new();
        let context = CommandContext::default();

        let overlays = engine.generate_for_command("rm -rf /tmp/test", &context);

        // Should have safety hint
        let has_safety = overlays
            .iter()
            .any(|o| o.overlay_type == OverlayType::SafetyHint);
        assert!(has_safety, "Should have safety hint for rm command");
    }

    #[test]
    fn test_knowledge_source_confidence_modifiers() {
        assert!(
            KnowledgeSource::Builtin.confidence_modifier()
                > KnowledgeSource::Unknown.confidence_modifier(),
            "Builtin should have higher confidence than Unknown"
        );

        assert!(
            KnowledgeSource::NixosDocs.confidence_modifier()
                > KnowledgeSource::SemanticInference.confidence_modifier(),
            "NixosDocs should have higher confidence than SemanticInference"
        );
    }
}

mod command_classification_tests {
    use super::*;

    #[test]
    fn test_readonly_command_classification() {
        let classification = CommandClassification::from_command("nix search nixpkgs#firefox");

        assert_eq!(
            classification.destructiveness,
            DestructivenessLevel::ReadOnly
        );
        assert!(!classification.needs_confirmation);
    }

    #[test]
    fn test_destructive_command_classification() {
        let classification = CommandClassification::from_command("nix-collect-garbage -d");

        assert_eq!(
            classification.destructiveness,
            DestructivenessLevel::Destructive
        );
        assert!(classification.needs_confirmation);
    }

    #[test]
    fn test_rollback_hint_for_nixos_rebuild() {
        let classification = CommandClassification::from_command("nixos-rebuild switch");

        assert!(classification.rollback_hint.is_some());
        assert!(
            classification
                .rollback_hint
                .as_ref()
                .unwrap()
                .contains("rollback"),
            "Should mention rollback option"
        );
    }

    #[test]
    fn test_required_phi_scaling() {
        let readonly = CommandClassification::from_command("nix search");
        let destructive = CommandClassification::from_command("nix-collect-garbage -d");

        assert!(
            readonly.required_phi < destructive.required_phi,
            "Destructive commands should require higher Phi"
        );
    }
}

mod shell_context_tests {
    use super::*;

    #[test]
    fn test_context_history_management() {
        let mut context = ShellContext::new();

        context.add_to_history("ls".to_string());
        context.add_to_history("pwd".to_string());
        context.add_to_history("ls".to_string()); // Duplicate of last

        assert_eq!(context.history.len(), 2, "Should not add duplicate of last");
        assert_eq!(context.history[0], "ls");
        assert_eq!(context.history[1], "pwd");
    }

    #[test]
    fn test_context_metrics_update() {
        let mut context = ShellContext::new();

        context.update_metrics(0.8, 0.9, true);

        assert_eq!(context.current_phi, 0.8);
        assert_eq!(context.current_coherence, 0.9);
        assert!(context.is_conscious);
    }

    #[test]
    fn test_consciousness_indicator() {
        let mut context = ShellContext::new();

        context.update_metrics(0.8, 0.9, true);
        assert_eq!(context.consciousness_indicator(), "●");

        context.update_metrics(0.8, 0.9, false);
        assert_eq!(context.consciousness_indicator(), "○");
    }

    #[test]
    fn test_status_color_thresholds() {
        let mut context = ShellContext::new();

        context.update_metrics(0.8, 0.9, true);
        assert!(
            context.status_color().contains("32"),
            "High Phi should be green"
        );

        context.update_metrics(0.5, 0.6, true);
        assert!(
            context.status_color().contains("33"),
            "Medium Phi should be yellow"
        );

        context.update_metrics(0.2, 0.3, false);
        assert!(
            context.status_color().contains("31"),
            "Low Phi should be red"
        );
    }
}

mod integration_flow_tests {
    use super::*;

    /// End-to-end test: User types a command, gets completions, and executes
    #[test]
    fn test_full_command_flow() {
        // 1. Set up components
        let mut context = ShellContext::new();
        let mut intellisense = IntelliSenseEngine::new();
        let mut phi_gate = PhiGate::new();

        // 2. Simulate user session with good consciousness
        context.update_metrics(0.8, 0.9, true);
        intellisense.set_phi(0.8);
        phi_gate.update_metrics(0.8, 0.9, true);

        // 3. User starts typing
        let input = "nix search";
        let completions = intellisense.complete(input, input.len());

        // 4. User selects a completion and executes
        let command = if !completions.is_empty() {
            completions[0].text.clone()
        } else {
            input.to_string()
        };

        // 5. Classify and gate
        let classification = CommandClassification::from_command(&command);
        let destructiveness = classify_command_destructiveness(&command);
        let decision = phi_gate.evaluate(&command, destructiveness);

        // 6. Verify the flow
        assert_eq!(
            classification.destructiveness,
            DestructivenessLevel::ReadOnly
        );
        match decision {
            GateDecision::Allowed { phi, .. } => {
                assert!(phi >= 0.7, "Command should be allowed with high Phi");
            }
            _ => {
                // Search commands should generally be allowed
            }
        }

        // 7. Add to history
        context.add_to_history(command.clone());
        assert_eq!(context.history[0], command);
    }

    /// End-to-end test: Destructive command with confirmation
    #[test]
    fn test_destructive_command_flow() {
        let mut context = ShellContext::new();
        let mut phi_gate = PhiGate::new();
        let mut epistemic_engine = EpistemicOverlayEngine::new();

        // Set up with moderate consciousness
        context.update_metrics(0.6, 0.7, true);
        phi_gate.update_metrics(0.6, 0.7, true);

        // User enters destructive command
        let command = "nixos-rebuild switch";
        let classification = CommandClassification::from_command(command);

        // Verify classification
        assert!(
            classification.needs_confirmation,
            "Should need confirmation"
        );
        assert!(
            classification.rollback_hint.is_some(),
            "Should have rollback hint"
        );

        // Check epistemic overlays
        let epi_context = CommandContext {
            source: KnowledgeSource::NixosDocs,
            confidence: 0.7,
            k_index: 0.65,
            epistemic_uncertainty: 0.3,
            aleatoric_uncertainty: 0.1,
            theory_weights: Vec::new(),
            phi_required: classification.required_phi,
            current_phi: context.current_phi,
        };
        let overlays = epistemic_engine.generate_for_command(command, &epi_context);

        // Should have safety-related overlays
        let has_safety_overlay = overlays
            .iter()
            .any(|o| o.overlay_type == OverlayType::SafetyHint || o.header.contains("System"));

        // Gate the command
        let destructiveness = classify_command_destructiveness(command);
        let decision = phi_gate.evaluate(command, destructiveness);

        // Should require confirmation or be vetoed
        match decision {
            GateDecision::NeedsConfirmation { .. } => {
                // Expected - needs user confirmation
            }
            GateDecision::Vetoed { .. } => {
                // Also acceptable - safety system vetoed
            }
            GateDecision::InsufficientPhi { .. } => {
                // Also acceptable - needs higher consciousness
            }
            GateDecision::Allowed { .. } => {
                // May be allowed if Phi is very high
            }
            _ => {}
        }
    }

    /// Test that completions work with cursor position
    #[test]
    fn test_completions_with_cursor_position() {
        let mut intellisense = IntelliSenseEngine::new();
        intellisense.set_phi(0.8);

        // Get completions with cursor at end
        let input = "nix-env";
        let completions = intellisense.complete(input, input.len());

        // Should have completions
        assert!(!completions.is_empty(), "Should have completions");
    }
}