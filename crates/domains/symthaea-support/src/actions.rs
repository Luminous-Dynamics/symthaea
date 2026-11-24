// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Action Engine — Propose, execute, and rollback autonomous actions
//!
//! This module's autonomy check is a conservative local allowlist, not an
//! authorization boundary. Consequential mutations must pass the downstream
//! act-bound policy and participant-side effect gate for the exact tenant,
//! target, operation, and attempt.

use crate::types::*;

#[derive(Debug)]
pub struct ActionEngine {
    /// Actions safe enough for this legacy local gate. Keep this read-only.
    /// Mutating actions require a separate, act-bound authorization path.
    permitted: Vec<ActionType>,
}

impl ActionEngine {
    pub fn new() -> Self {
        Self {
            // Diagnostics are observational. Do not add RestartService,
            // ClearCache, UpdateConfig, or Custom here: this type has no
            // tenant, target, resource, expiry, or authorization context.
            permitted: vec![ActionType::RunDiagnostic],
        }
    }

    /// Apply the conservative local allowlist for the requested autonomy tier.
    ///
    /// FullAutonomous is deliberately not a bypass. This helper does not
    /// authorize operating-system effects; callers must separately obtain
    /// exact act authorization and enforce it at the target participant.
    pub fn can_execute(&self, action: &ActionType, level: &AutonomyLevel) -> bool {
        match level {
            AutonomyLevel::Advisory => false,
            AutonomyLevel::SemiAutonomous | AutonomyLevel::FullAutonomous => {
                self.permitted.contains(action)
            }
        }
    }

    /// Propose an action with human-readable recovery notes.
    ///
    /// These notes are not an executable rollback plan and do not establish
    /// that recovery is possible. The effect executor must bind a verified
    /// recovery strategy to the exact change before a mutation is authorized.
    pub fn propose_action(&self, action_type: ActionType, description: String) -> ProposedAction {
        let rollback_steps = match &action_type {
            ActionType::RestartService => vec![
                "Check if service was running".to_string(),
                "Restart service with previous config".to_string(),
            ],
            ActionType::ClearCache => vec![
                "Note cache size before clearing".to_string(),
                "Clear was performed, cache will rebuild".to_string(),
            ],
            ActionType::UpdateConfig => vec![
                "Backup current config".to_string(),
                "Restore backed-up config".to_string(),
            ],
            ActionType::RunDiagnostic => {
                vec!["No rollback needed for read-only diagnostic".to_string()]
            }
            ActionType::Custom(_) => vec!["Manual rollback required".to_string()],
        };

        ProposedAction {
            action_type,
            description,
            rollback_steps,
            confidence: 0.8,
        }
    }
}

impl Default for ActionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advisory_never_allows_execution() {
        let engine = ActionEngine::new();
        let actions = [
            ActionType::RestartService,
            ActionType::ClearCache,
            ActionType::UpdateConfig,
            ActionType::RunDiagnostic,
            ActionType::Custom("test".to_string()),
        ];

        for action in &actions {
            assert!(
                !engine.can_execute(action, &AutonomyLevel::Advisory),
                "Advisory must block {action:?}"
            );
        }
    }

    #[test]
    fn semi_autonomous_allows_only_read_only_diagnostics() {
        let engine = ActionEngine::new();

        assert!(engine.can_execute(
            &ActionType::RunDiagnostic,
            &AutonomyLevel::SemiAutonomous
        ));

        for action in [
            ActionType::RestartService,
            ActionType::ClearCache,
            ActionType::UpdateConfig,
            ActionType::Custom("deploy-hotfix".to_string()),
        ] {
            assert!(
                !engine.can_execute(&action, &AutonomyLevel::SemiAutonomous),
                "SemiAutonomous must not authorize mutation {action:?}"
            );
        }
    }

    #[test]
    fn full_autonomous_cannot_bypass_the_allowlist() {
        let engine = ActionEngine::new();

        assert!(engine.can_execute(
            &ActionType::RunDiagnostic,
            &AutonomyLevel::FullAutonomous
        ));

        for action in [
            ActionType::RestartService,
            ActionType::ClearCache,
            ActionType::UpdateConfig,
            ActionType::Custom("anything".to_string()),
        ] {
            assert!(
                !engine.can_execute(&action, &AutonomyLevel::FullAutonomous),
                "FullAutonomous must not bypass the allowlist for {action:?}"
            );
        }
    }

    #[test]
    fn propose_action_includes_non_authoritative_recovery_notes() {
        let engine = ActionEngine::new();
        let proposed =
            engine.propose_action(ActionType::RestartService, "Restart conductor".to_string());
        assert_eq!(proposed.action_type, ActionType::RestartService);
        assert!(!proposed.rollback_steps.is_empty());
        assert_eq!(proposed.rollback_steps.len(), 2);

        let diag =
            engine.propose_action(ActionType::RunDiagnostic, "Run network check".to_string());
        assert_eq!(diag.rollback_steps.len(), 1);
        assert!(diag.rollback_steps[0].contains("No rollback"));
    }
}
