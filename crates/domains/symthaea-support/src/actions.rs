// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Action Engine — propose support actions and legacy autonomy eligibility.
//!
//! IMPORTANT: `ActionEngine::can_execute` is a coarse legacy eligibility hint.
//! It is **not** an execution authorization boundary. New governed IT execution
//! paths must require `crate::it_authority::ItCommandAuthorityV1` (or a future
//! stricter successor) bound to exact state/evidence/policy prerequisites.

use crate::types::*;

#[derive(Debug)]
pub struct ActionEngine {
    permitted: Vec<ActionType>,
}

impl ActionEngine {
    pub fn new() -> Self {
        Self {
            permitted: vec![
                ActionType::RestartService,
                ActionType::ClearCache,
                ActionType::RunDiagnostic,
            ],
        }
    }

    /// Legacy coarse automation eligibility only.
    ///
    /// `true` preserves historical API behavior; it does **not** establish
    /// currentness, blast radius, rollback readiness, operator/organizational
    /// authority, or a live capability. Do not use this method as permission to
    /// execute a governed IT action. New execution paths must require the bounded
    /// capability/revalidation path in `it_authority`.
    pub fn can_execute(&self, action: &ActionType, level: &AutonomyLevel) -> bool {
        match level {
            AutonomyLevel::Advisory => false,
            AutonomyLevel::SemiAutonomous => self.permitted.contains(action),
            AutonomyLevel::FullAutonomous => true,
        }
    }

    /// Propose an action with rollback steps. A proposal carries no execution
    /// authority and must pass the bounded authority/revalidation boundary before
    /// any governed mutating operation is handed to an executor.
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
        assert!(!engine.can_execute(&ActionType::RestartService, &AutonomyLevel::Advisory));
        assert!(!engine.can_execute(&ActionType::ClearCache, &AutonomyLevel::Advisory));
        assert!(!engine.can_execute(&ActionType::UpdateConfig, &AutonomyLevel::Advisory));
        assert!(!engine.can_execute(
            &ActionType::Custom("test".to_string()),
            &AutonomyLevel::Advisory
        ));
    }

    #[test]
    fn semi_autonomous_allows_permitted_actions() {
        let engine = ActionEngine::new();
        assert!(engine.can_execute(&ActionType::RestartService, &AutonomyLevel::SemiAutonomous));
        assert!(engine.can_execute(&ActionType::ClearCache, &AutonomyLevel::SemiAutonomous));
        assert!(engine.can_execute(&ActionType::RunDiagnostic, &AutonomyLevel::SemiAutonomous));
        // UpdateConfig is NOT in permitted list
        assert!(!engine.can_execute(&ActionType::UpdateConfig, &AutonomyLevel::SemiAutonomous));
    }

    #[test]
    fn full_autonomous_legacy_eligibility_is_not_runtime_authority() {
        // Historical eligibility behavior retained for compatibility. Governed
        // execution must still present an ItCommandAuthorityV1.
        let engine = ActionEngine::new();
        assert!(engine.can_execute(&ActionType::RestartService, &AutonomyLevel::FullAutonomous));
        assert!(engine.can_execute(&ActionType::UpdateConfig, &AutonomyLevel::FullAutonomous));
        assert!(engine.can_execute(
            &ActionType::Custom("anything".to_string()),
            &AutonomyLevel::FullAutonomous
        ));
    }

    #[test]
    fn propose_action_includes_rollback_steps() {
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
