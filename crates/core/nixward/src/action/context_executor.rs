// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Context-authorized execution facade for Nixward.
//!
//! This is the migration boundary between the new explicit authority model and
//! the legacy numeric executor. Callers present an `ExecutionContext`; cognitive
//! telemetry is never consulted when deciding whether an action is permitted.
//!
//! The legacy backend still expects a numeric value in order to preserve its
//! mature rollback/outcome behavior. After this facade has independently
//! authorized the exact request, it passes an internal bypass constant to that
//! backend. That constant is NOT authority and MUST NOT be consumed as cognitive
//! telemetry. The facade records truthful optional Phi telemetry separately.

use super::execution_context::{AuthoritySource, ExecutionContext};
use super::executor::{ExecutionResult, NixOSCommand, NixOSExecutor, SafetyLevel};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Transitional value used only to neutralize the legacy numeric gate *after*
/// explicit authority has already been checked by this facade.
const LEGACY_AUTHORIZATION_BYPASS_LEVEL: f32 = 1.0;

/// A truthful execution record for the context-aware path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextExecutionRecord {
    pub command: NixOSCommand,
    pub command_digest: String,
    pub authority_source: AuthoritySource,
    pub safety_ceiling: SafetyLevel,
    pub measured_phi: Option<f32>,
    pub result: ExecutionResult,
    pub timestamp_ms: u64,
}

/// Executor facade whose permission decision is based solely on explicit
/// authority. Cognitive telemetry may be recorded, but it cannot grant access.
pub struct ContextualExecutor {
    legacy: NixOSExecutor,
    history: VecDeque<ContextExecutionRecord>,
}

impl Default for ContextualExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextualExecutor {
    pub fn new() -> Self {
        Self {
            legacy: NixOSExecutor::new(),
            history: VecDeque::with_capacity(1000),
        }
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.legacy = self.legacy.with_dry_run(dry_run);
        self
    }

    /// Deterministic digest used to bind authority to an exact serialized
    /// command. The domain tag prevents accidental reuse with another schema.
    pub fn command_digest(command: &NixOSCommand) -> Result<String, String> {
        let encoded = serde_json::to_vec(command)
            .map_err(|error| format!("serialize command for authority digest: {error}"))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"nixward-command-authority-v1\0");
        hasher.update(&encoded);
        Ok(hasher.finalize().to_hex().to_string())
    }

    /// Execute an action only when the supplied context has sufficient explicit
    /// authority. Phi/confidence/free-energy never participate in the allow
    /// decision.
    pub async fn execute(
        &mut self,
        command: NixOSCommand,
        context: &ExecutionContext,
    ) -> ExecutionResult {
        let safety = command.safety_level();

        if let Err(error) = context.validate() {
            return ExecutionResult::Blocked {
                reason: format!("invalid execution context: {error}"),
                safety_level: safety,
            };
        }

        if !context.allows(safety) {
            return ExecutionResult::Blocked {
                reason: format!(
                    "authority {:?} with ceiling {:?} does not permit {:?}",
                    context.authority.source(),
                    context.authority.safety_ceiling(),
                    safety
                ),
                safety_level: safety,
            };
        }

        let digest = match Self::command_digest(&command) {
            Ok(digest) => digest,
            Err(error) => {
                return ExecutionResult::Blocked {
                    reason: error,
                    safety_level: safety,
                };
            }
        };

        // Deterministic policy authority is never accepted as a floating
        // capability. It must be bound to the exact command being executed.
        if context.authority.source() == AuthoritySource::PolicyDecision
            && context.authority.action_digest().is_none()
        {
            return ExecutionResult::Blocked {
                reason: "policy authority must be bound to an exact command digest".into(),
                safety_level: safety,
            };
        }

        if let Some(expected) = context.authority.action_digest()
            && expected != digest
        {
            return ExecutionResult::Blocked {
                reason: "authority digest does not match the proposed command".into(),
                safety_level: safety,
            };
        }

        // Authority has already been decided above. The legacy numeric executor
        // is now only a compatibility backend for command execution and rollback.
        let result = self
            .legacy
            .execute(command.clone(), LEGACY_AUTHORIZATION_BYPASS_LEVEL)
            .await;

        self.record(
            command,
            digest,
            context,
            result.clone(),
        );
        result
    }

    fn record(
        &mut self,
        command: NixOSCommand,
        command_digest: String,
        context: &ExecutionContext,
        result: ExecutionResult,
    ) {
        self.history.push_back(ContextExecutionRecord {
            command,
            command_digest,
            authority_source: context.authority.source(),
            safety_ceiling: context.authority.safety_ceiling(),
            measured_phi: context.cognition.phi.value(),
            result,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis() as u64)
                .unwrap_or(0),
        });

        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    pub fn history(&self) -> &VecDeque<ContextExecutionRecord> {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{
        AuthorityContext, CognitiveContext, PhiMeasurement,
    };

    fn rebuild() -> NixOSCommand {
        NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        }
    }

    #[tokio::test]
    async fn high_phi_cannot_create_authority() {
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let context = ExecutionContext::new(
            AuthorityContext::observe_only(),
            CognitiveContext {
                phi: PhiMeasurement::measured(1.0, "test").unwrap(),
                confidence: Some(1.0),
                free_energy: Some(0.0),
                prediction_error: Some(0.0),
                causal_support: Some(1.0),
            },
        );

        let result = executor.execute(rebuild(), &context).await;
        assert!(matches!(result, ExecutionResult::Blocked { .. }));
        assert!(executor.history().is_empty());
    }

    #[tokio::test]
    async fn explicit_authority_works_without_phi() {
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let context = ExecutionContext::new(
            AuthorityContext::explicit_operator_confirmation(SafetyLevel::SystemCritical),
            CognitiveContext::default(),
        );

        let result = executor.execute(rebuild(), &context).await;
        assert!(matches!(result, ExecutionResult::Success { .. }));
        assert_eq!(executor.history().len(), 1);
        assert_eq!(executor.history()[0].measured_phi, None);
    }

    #[tokio::test]
    async fn policy_authority_requires_exact_digest() {
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let floating = ExecutionContext::new(
            AuthorityContext::policy_decision(SafetyLevel::SystemCritical, None),
            CognitiveContext::default(),
        );
        assert!(matches!(
            executor.execute(rebuild(), &floating).await,
            ExecutionResult::Blocked { .. }
        ));

        let command = rebuild();
        let digest = ContextualExecutor::command_digest(&command).unwrap();
        let bound = ExecutionContext::new(
            AuthorityContext::policy_decision(SafetyLevel::SystemCritical, Some(digest)),
            CognitiveContext::default(),
        );
        assert!(matches!(
            executor.execute(command, &bound).await,
            ExecutionResult::Success { .. }
        ));
    }

    #[tokio::test]
    async fn mismatched_digest_is_rejected() {
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let context = ExecutionContext::new(
            AuthorityContext::upstream_human_gate(
                SafetyLevel::SystemCritical,
                Some("not-the-command".into()),
            ),
            CognitiveContext::default(),
        );
        assert!(matches!(
            executor.execute(rebuild(), &context).await,
            ExecutionResult::Blocked { .. }
        ));
    }

    #[tokio::test]
    async fn destructive_action_needs_destructive_ceiling() {
        let destructive = NixOSCommand::CollectGarbage {
            older_than_days: None,
            delete_all: true,
        };
        let critical_only = ExecutionContext::new(
            AuthorityContext::explicit_operator_confirmation(SafetyLevel::SystemCritical),
            CognitiveContext::default(),
        );
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        assert!(matches!(
            executor.execute(destructive.clone(), &critical_only).await,
            ExecutionResult::Blocked { .. }
        ));

        let destructive_ok = ExecutionContext::new(
            AuthorityContext::explicit_operator_confirmation(SafetyLevel::Destructive),
            CognitiveContext::default(),
        );
        assert!(matches!(
            executor.execute(destructive, &destructive_ok).await,
            ExecutionResult::Success { .. }
        ));
    }

    #[test]
    fn command_digest_is_deterministic_and_command_specific() {
        let a = rebuild();
        let b = NixOSCommand::RebuildBoot {
            flake: None,
            extra_args: vec![],
        };
        assert_eq!(
            ContextualExecutor::command_digest(&a).unwrap(),
            ContextualExecutor::command_digest(&a).unwrap()
        );
        assert_ne!(
            ContextualExecutor::command_digest(&a).unwrap(),
            ContextualExecutor::command_digest(&b).unwrap()
        );
    }
}
