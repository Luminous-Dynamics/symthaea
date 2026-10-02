//! Provenance-aware action gating for GIS.
//!
//! This module separates historical authorization from current authorization.
//! A frame revision can require fresh support for future high-risk execution
//! without rewriting the historical record of an action that already happened.

use super::ignorance_types::{ConclusionDependencyGraph, EpistemicFrameImpact, EpistemicFrameRevision};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionRisk {
    Informational,
    Low,
    High,
    Critical,
}

impl ActionRisk {
    pub const fn requires_current_support(self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionDependencyKind {
    ConclusionSupport,
    CausalBasis,
    OntologyBasis,
    EvidenceBasis,
    AssumptionBasis,
}

impl ActionDependencyKind {
    pub const fn affects(self, impact: EpistemicFrameImpact) -> bool {
        match self {
            Self::ConclusionSupport => impact.evidence_boundary || impact.ontology || impact.causal_model,
            Self::CausalBasis => impact.causal_model,
            Self::OntologyBasis => impact.ontology,
            Self::EvidenceBasis => impact.evidence_boundary,
            Self::AssumptionBasis => impact.evidence_boundary
                || impact.ontology
                || impact.causal_model
                || impact.exclusions
                || impact.blind_spots,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    Ready,
    RequiresReevaluation,
    Deferred,
    Executed,
    Superseded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDependency {
    pub conclusion_id: String,
    pub kind: ActionDependencyKind,
}

/// Orthogonal assessment of whether a conclusion is currently usable as support.
///
/// ConclusionStatus is lifecycle state only; it must not be treated as a generic
/// authority or freshness signal. These dimensions remain explicit at the action gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentConclusionSupport {
    pub lifecycle: super::ignorance_types::ConclusionStatus,
    pub stale: bool,
    pub conflicted: bool,
    pub provenance_complete: bool,
}

impl CurrentConclusionSupport {
    pub const fn active() -> Self {
        Self {
            lifecycle: super::ignorance_types::ConclusionStatus::Active,
            stale: false,
            conflicted: false,
            provenance_complete: true,
        }
    }

    pub const fn is_currently_authoritative(self) -> bool {
        matches!(self.lifecycle, super::ignorance_types::ConclusionStatus::Active)
            && !self.stale
            && !self.conflicted
            && self.provenance_complete
    }
}

/// Immutable witness binding current authorization to one exact action instance.
/// Kept separate from the epistemic decision witness so authorization cannot be
/// replayed merely because its supporting conclusions remain available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionAuthorizationWitness {
    pub action_id: String,
    pub action_digest: String,
    pub frame: String,
    pub support_digest: String,
    pub policy: String,
    pub decision: String,
    pub issued_at: String,
    pub expires_at: Option<String>,
    pub authority_epoch: u64,
}

impl ActionAuthorizationWitness {
    pub fn is_bound_to(
        &self,
        action: &EpistemicAction,
        current_frame: &str,
        expected_support_digest: &str,
        expected_policy: &str,
    ) -> bool {
        self.action_id == action.id
            && !self.action_digest.is_empty()
            && self.frame == current_frame
            && self.support_digest == expected_support_digest
            && self.policy == expected_policy
            && !self.issued_at.is_empty()
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorizationLeaseState {
    Ready,
    Prepared { attempt_id: String },
    Indeterminate { attempt_id: String },
    Exhausted,
    Revoked,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionOutcome {
    Succeeded,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReceipt {
    pub action_id: String,
    pub action_digest: String,
    pub attempt_id: String,
    pub authority_epoch: u64,
    pub outcome: ExecutionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorizationConsumptionError {
    InvalidBinding,
    NotReady,
    BudgetExhausted,
    AttemptMismatch,
    IndeterminateRequiresReconciliation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizationLease {
    pub action_id: String,
    pub action_digest: String,
    pub support_digest: String,
    pub policy: String,
    pub authority_epoch: u64,
    pub remaining_executions: u32,
    pub state: AuthorizationLeaseState,
}

impl AuthorizationLease {
    pub fn new(
        action_id: impl Into<String>,
        action_digest: impl Into<String>,
        support_digest: impl Into<String>,
        policy: impl Into<String>,
        authority_epoch: u64,
        execution_budget: u32,
    ) -> Self {
        Self {
            action_id: action_id.into(),
            action_digest: action_digest.into(),
            support_digest: support_digest.into(),
            policy: policy.into(),
            authority_epoch,
            remaining_executions: execution_budget,
            state: if execution_budget == 0 {
                AuthorizationLeaseState::Exhausted
            } else {
                AuthorizationLeaseState::Ready
            },
        }
    }

    pub fn prepare_for_execution(
        &mut self,
        witness: &ActionAuthorizationWitness,
        action: &EpistemicAction,
        current_frame: &str,
        attempt_id: impl Into<String>,
    ) -> Result<(), AuthorizationConsumptionError> {
        if !witness.is_bound_to(action, current_frame, &self.support_digest, &self.policy)
            || witness.action_digest != self.action_digest
            || witness.authority_epoch != self.authority_epoch
        {
            return Err(AuthorizationConsumptionError::InvalidBinding);
        }

        if self.remaining_executions == 0
            || matches!(self.state, AuthorizationLeaseState::Exhausted)
        {
            return Err(AuthorizationConsumptionError::BudgetExhausted);
        }

        if !matches!(self.state, AuthorizationLeaseState::Ready) {
            return match self.state {
                AuthorizationLeaseState::Indeterminate { .. } => {
                    Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation)
                }
                _ => Err(AuthorizationConsumptionError::NotReady),
            };
        }

        self.state = AuthorizationLeaseState::Prepared {
            attempt_id: attempt_id.into(),
        };
        Ok(())
    }

    pub fn commit(
        &mut self,
        attempt_id: &str,
        outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationConsumptionError> {
        let prepared = matches!(
            &self.state,
            AuthorizationLeaseState::Prepared { attempt_id: id } if id == attempt_id
        );
        if !prepared {
            return if matches!(self.state, AuthorizationLeaseState::Indeterminate { .. }) {
                Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation)
            } else if matches!(self.state, AuthorizationLeaseState::Exhausted) {
                Err(AuthorizationConsumptionError::BudgetExhausted)
            } else {
                Err(AuthorizationConsumptionError::AttemptMismatch)
            };
        }

        let receipt = ExecutionReceipt {
            action_id: self.action_id.clone(),
            action_digest: self.action_digest.clone(),
            attempt_id: attempt_id.to_owned(),
            authority_epoch: self.authority_epoch,
            outcome,
        };

        match outcome {
            ExecutionOutcome::Indeterminate => {
                self.state = AuthorizationLeaseState::Indeterminate {
                    attempt_id: attempt_id.to_owned(),
                };
            }
            ExecutionOutcome::Succeeded | ExecutionOutcome::Failed => {
                self.remaining_executions -= 1;
                self.state = if self.remaining_executions == 0 {
                    AuthorizationLeaseState::Exhausted
                } else {
                    AuthorizationLeaseState::Ready
                };
            }
        }

        Ok(receipt)
    }

    pub fn reconcile_indeterminate(
        &mut self,
        attempt_id: &str,
        outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationConsumptionError> {
        if !matches!(
            &self.state,
            AuthorizationLeaseState::Indeterminate { attempt_id: id } if id == attempt_id
        ) || matches!(outcome, ExecutionOutcome::Indeterminate)
        {
            return Err(AuthorizationConsumptionError::AttemptMismatch);
        }

        self.remaining_executions = self.remaining_executions.saturating_sub(1);
        self.state = if self.remaining_executions == 0 {
            AuthorizationLeaseState::Exhausted
        } else {
            AuthorizationLeaseState::Ready
        };

        Ok(ExecutionReceipt {
            action_id: self.action_id.clone(),
            action_digest: self.action_digest.clone(),
            attempt_id: attempt_id.to_owned(),
            authority_epoch: self.authority_epoch,
            outcome,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDecisionWitness {
    pub frame: String,
    pub conclusions: Vec<String>,
    pub evidence: Vec<String>,
    pub policy: String,
    pub decision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionReevaluationWitness {
    pub prior_frame: String,
    pub revised_frame: String,
    pub affected_conclusions: Vec<String>,
    pub reasons: Vec<ActionDependencyKind>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicAction {
    pub id: String,
    pub description: String,
    pub risk: ActionRisk,
    pub dependencies: Vec<ActionDependency>,
    pub status: ActionStatus,
    pub historical_decisions: Vec<ActionDecisionWitness>,
    pub reevaluation: Option<ActionReevaluationWitness>,
}

impl EpistemicAction {
    pub fn new(id: impl Into<String>, description: impl Into<String>, risk: ActionRisk) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            risk,
            dependencies: Vec::new(),
            status: ActionStatus::Ready,
            historical_decisions: Vec::new(),
            reevaluation: None,
        }
    }

    pub fn require_reevaluation(
        &mut self,
        revision: &EpistemicFrameRevision,
        affected_conclusions: Vec<String>,
        reasons: Vec<ActionDependencyKind>,
    ) {
        if matches!(self.status, ActionStatus::Executed | ActionStatus::Superseded) {
            return;
        }

        self.status = ActionStatus::RequiresReevaluation;
        self.reevaluation = Some(ActionReevaluationWitness {
            prior_frame: revision.prior_frame.clone(),
            revised_frame: revision.revised_frame.clone(),
            affected_conclusions,
            reasons,
            reason: "epistemic prerequisite changed; current support required by policy".into(),
        });
    }

    pub fn record_decision(&mut self, witness: ActionDecisionWitness) {
        self.historical_decisions.push(witness);
        self.status = ActionStatus::Executed;
    }

    /// Authorize a current high-risk decision only when every declared conclusion
    /// prerequisite is currently active in the authoritative conclusion store.
    ///
    /// This is intentionally separate from `record_decision`: historical witnesses
    /// remain append-only, while current authorization must be re-established.
    pub fn try_record_current_decision(
        &mut self,
        witness: ActionDecisionWitness,
        current_frame: &str,
        conclusion_support: &std::collections::HashMap<String, CurrentConclusionSupport>,
    ) -> Result<(), ActionStatus> {
        if self.risk.requires_current_support() {
            // The authorization witness must be bound to the frame that is current
            // at the execution boundary. A valid historical witness is not reusable.
            let frame_matches = witness.frame == current_frame;
            let dependencies_are_witnessed = self.dependencies.iter().all(|dependency| {
                witness.conclusions.iter().any(|id| id == &dependency.conclusion_id)
            });
            let all_current = self.dependencies.iter().all(|dependency| {
                conclusion_support
                    .get(&dependency.conclusion_id)
                    .copied()
                    .is_some_and(CurrentConclusionSupport::is_currently_authoritative)
            });

            if !frame_matches || !dependencies_are_witnessed || !all_current {
                self.status = if self.status == ActionStatus::RequiresReevaluation {
                    ActionStatus::RequiresReevaluation
                } else {
                    ActionStatus::Deferred
                };
                return Err(self.status);
            }
        }

        self.record_decision(witness);
        Ok(())
    }

    /// Fail closed for high-risk actions when any declared prerequisite is absent.
    /// The supplied set must come from the authoritative conclusion store.
    pub fn defer_for_unresolved_prerequisites(
        &mut self,
        known_conclusions: &std::collections::HashSet<String>,
    ) -> bool {
        if !self.risk.requires_current_support()
            || matches!(self.status, ActionStatus::Executed | ActionStatus::Superseded)
        {
            return false;
        }

        let unresolved = self.dependencies.iter().any(|dependency| {
            !known_conclusions.contains(&dependency.conclusion_id)
        });
        if unresolved {
            self.status = ActionStatus::Deferred;
            return true;
        }
        false
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActionDependencyGraph {
    pub actions: Vec<EpistemicAction>,
}

impl ActionDependencyGraph {
    pub fn add(&mut self, action: EpistemicAction) {
        self.actions.push(action);
    }

    /// Connect the action graph directly to the canonical conclusion-impact traversal.
    pub fn reevaluate_with_conclusion_graph(
        &mut self,
        conclusions: &mut ConclusionDependencyGraph,
        revision: &EpistemicFrameRevision,
    ) -> Vec<String> {
        let affected = conclusions.reopen_from_frame_revision(revision);
        self.reevaluate_from_frame_revision(revision, &affected)
    }

    pub fn reevaluate_from_frame_revision(
        &mut self,
        revision: &EpistemicFrameRevision,
        affected_conclusions: &[String],
    ) -> Vec<String> {
        let affected: std::collections::HashSet<&str> =
            affected_conclusions.iter().map(String::as_str).collect();
        let mut gated = Vec::new();

        for action in &mut self.actions {
            if !action.risk.requires_current_support() {
                continue;
            }

            let impacted: Vec<_> = action
                .dependencies
                .iter()
                .filter(|dependency| affected.contains(dependency.conclusion_id.as_str()) && dependency.kind.affects(revision.impact))
                .collect();

            if impacted.is_empty() {
                continue;
            }

            action.require_reevaluation(
                revision,
                impacted.iter().map(|d| d.conclusion_id.clone()).collect(),
                impacted.iter().map(|d| d.kind).collect(),
            );

            if action.status == ActionStatus::RequiresReevaluation {
                gated.push(action.id.clone());
            }
        }

        gated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_risk_action_defers_when_prerequisite_is_missing() {
        let mut action = EpistemicAction::new("a-missing", "intervention", ActionRisk::Critical);
        action.dependencies.push(ActionDependency {
            conclusion_id: "missing-c".into(),
            kind: ActionDependencyKind::CausalBasis,
        });
        let known = std::collections::HashSet::new();
        assert!(action.defer_for_unresolved_prerequisites(&known));
        assert_eq!(action.status, ActionStatus::Deferred);
    }

    #[test]
    fn frame_revision_connects_conclusion_and_action_graphs() {
        use super::super::ignorance_types::{
            ConclusionDependency, ConclusionDependencyKind, EpistemicConclusion,
        };
        let revision = EpistemicFrameRevision {
            prior_frame: "f1".into(),
            revised_frame: "f2".into(),
            trigger: "ontology expanded".into(),
            newly_represented: Some("institution".into()),
            scope_change: "ontology".into(),
            affected_conclusions: vec!["c-root".into()],
            impact: EpistemicFrameImpact {
                evidence_boundary: false,
                ontology: true,
                causal_model: false,
                exclusions: false,
                blind_spots: false,
            },
        };
        let mut conclusions = ConclusionDependencyGraph::default();
        conclusions.add(EpistemicConclusion::new("c-root", "root", "f1"));
        conclusions.add(EpistemicConclusion::new("c-child", "child", "f1"));
        conclusions.add_dependency(ConclusionDependency::new(
            "c-root", "c-child", ConclusionDependencyKind::OntologyDependency,
        ));

        let mut action = EpistemicAction::new("a-ontology", "policy intervention", ActionRisk::High);
        action.dependencies.push(ActionDependency {
            conclusion_id: "c-child".into(),
            kind: ActionDependencyKind::OntologyBasis,
        });
        let mut actions = ActionDependencyGraph::default();
        actions.add(action);

        assert_eq!(
            actions.reevaluate_with_conclusion_graph(&mut conclusions, &revision),
            vec!["a-ontology"]
        );
        assert_eq!(conclusions.conclusions[1].status, super::super::ignorance_types::ConclusionStatus::Reopened);
    }

    #[test]
    fn current_high_risk_decision_requires_active_prerequisites() {
        let mut action = EpistemicAction::new("a-current", "intervention", ActionRisk::Critical);
        action.dependencies.push(ActionDependency {
            conclusion_id: "c1".into(),
            kind: ActionDependencyKind::CausalBasis,
        });

        let witness = ActionDecisionWitness {
            frame: "f1".into(),
            conclusions: vec!["c1".into()],
            evidence: vec!["e1".into()],
            policy: "policy-v1".into(),
            decision: "execute".into(),
        };

        let mut statuses = std::collections::HashMap::new();
        statuses.insert("c1".into(), CurrentConclusionSupport {
            lifecycle: super::super::ignorance_types::ConclusionStatus::Reopened,
            stale: false,
            conflicted: false,
            provenance_complete: true,
        });
        assert_eq!(
            action.try_record_current_decision(witness.clone(), "f1", &statuses),
            Err(ActionStatus::Deferred)
        );
        assert!(action.historical_decisions.is_empty());

        statuses.insert("c1".into(), CurrentConclusionSupport::active());
        assert_eq!(action.try_record_current_decision(witness, "f1", &statuses), Ok(()));
        assert_eq!(action.status, ActionStatus::Executed);
        assert_eq!(action.historical_decisions.len(), 1);
    }

    #[test]
    fn current_witness_cannot_cross_frame_or_dependency_boundary() {
        let mut action = EpistemicAction::new("a-bound", "intervention", ActionRisk::High);
        action.dependencies.push(ActionDependency {
            conclusion_id: "c1".into(),
            kind: ActionDependencyKind::CausalBasis,
        });

        let witness = ActionDecisionWitness {
            frame: "frame@1".into(),
            conclusions: vec!["unrelated".into()],
            evidence: vec!["e1".into()],
            policy: "policy-v1".into(),
            decision: "execute".into(),
        };
        let mut statuses = std::collections::HashMap::new();
        statuses.insert("c1".into(), CurrentConclusionSupport::active());

        assert_eq!(
            action.try_record_current_decision(witness, "frame@2", &statuses),
            Err(ActionStatus::Deferred)
        );
        assert!(action.historical_decisions.is_empty());
    }


    #[test]
    fn authorization_witness_is_bound_to_exact_action_frame_support_and_policy() {
        let action = EpistemicAction::new("a-bound", "intervention", ActionRisk::High);
        let witness = ActionAuthorizationWitness {
            action_id: "a-bound".into(),
            action_digest: "sha256:action".into(),
            frame: "f2".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v2".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: Some("2026-10-02T20:05:00Z".into()),
        };
        assert!(witness.is_bound_to(&action, "f2", "sha256:support", "policy-v2"));
        assert!(!witness.is_bound_to(&action, "f1", "sha256:support", "policy-v2"));
        assert!(!witness.is_bound_to(&action, "f2", "sha256:other", "policy-v2"));
        assert!(!witness.is_bound_to(&action, "f2", "sha256:support", "policy-v1"));
    }

    #[test]
    fn current_support_does_not_collapse_staleness_conflict_or_provenance_into_lifecycle() {
        let mut action = EpistemicAction::new("a-support", "intervention", ActionRisk::Critical);
        action.dependencies.push(ActionDependency {
            conclusion_id: "c1".into(),
            kind: ActionDependencyKind::ConclusionSupport,
        });
        let witness = ActionDecisionWitness {
            frame: "f1".into(),
            conclusions: vec!["c1".into()],
            evidence: vec!["e1".into()],
            policy: "policy-v1".into(),
            decision: "execute".into(),
        };
        for support in [
            CurrentConclusionSupport { lifecycle: super::super::ignorance_types::ConclusionStatus::Active, stale: true, conflicted: false, provenance_complete: true },
            CurrentConclusionSupport { lifecycle: super::super::ignorance_types::ConclusionStatus::Active, stale: false, conflicted: true, provenance_complete: true },
            CurrentConclusionSupport { lifecycle: super::super::ignorance_types::ConclusionStatus::Active, stale: false, conflicted: false, provenance_complete: false },
        ] {
            let mut statuses = std::collections::HashMap::new();
            statuses.insert("c1".into(), support);
            assert_eq!(action.try_record_current_decision(witness.clone(), "f1", &statuses), Err(ActionStatus::Deferred));
            assert!(action.historical_decisions.is_empty());
        }
    }


    #[test]
    fn authorization_lease_blocks_replay_and_fresh_witness_reissuance() {
        let action = EpistemicAction::new("a-lease", "intervention", ActionRisk::Critical);
        let witness = ActionAuthorizationWitness {
            action_id: "a-lease".into(),
            action_digest: "sha256:canonical-action".into(),
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 7,
        };
        let mut lease = AuthorizationLease::new("a-lease", "sha256:canonical-action", "sha256:support", "policy-v1", 7, 1);
        lease.prepare_for_execution(&witness, &action, "f1", "attempt-1").unwrap();
        let receipt = lease.commit("attempt-1", ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(receipt.outcome, ExecutionOutcome::Succeeded);
        assert_eq!(lease.state, AuthorizationLeaseState::Exhausted);
        let fresh_witness = ActionAuthorizationWitness { issued_at: "2026-10-02T20:01:00Z".into(), ..witness };
        assert_eq!(
            lease.prepare_for_execution(&fresh_witness, &action, "f1", "attempt-2"),
            Err(AuthorizationConsumptionError::BudgetExhausted)
        );
    }

    #[test]
    fn authorization_lease_serializes_prepare_and_commit() {
        let action = EpistemicAction::new("a-concurrent", "intervention", ActionRisk::High);
        let witness = ActionAuthorizationWitness {
            action_id: "a-concurrent".into(), action_digest: "sha256:canonical".into(),
            frame: "f1".into(), support_digest: "sha256:support".into(),
            policy: "policy-v1".into(), decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(), expires_at: None, authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new("a-concurrent", "sha256:canonical", "sha256:support", "policy-v1", 1, 1);
        lease.prepare_for_execution(&witness, &action, "f1", "attempt-1").unwrap();
        assert_eq!(
            lease.prepare_for_execution(&witness, &action, "f1", "attempt-2"),
            Err(AuthorizationConsumptionError::NotReady)
        );
    }

    #[test]
    fn indeterminate_commit_requires_reconciliation_before_retry() {
        let action = EpistemicAction::new("a-crash", "intervention", ActionRisk::Critical);
        let witness = ActionAuthorizationWitness {
            action_id: "a-crash".into(), action_digest: "sha256:canonical".into(),
            frame: "f1".into(), support_digest: "sha256:support".into(),
            policy: "policy-v1".into(), decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(), expires_at: None, authority_epoch: 3,
        };
        let mut lease = AuthorizationLease::new("a-crash", "sha256:canonical", "sha256:support", "policy-v1", 3, 1);
        lease.prepare_for_execution(&witness, &action, "f1", "attempt-1").unwrap();
        let receipt = lease.commit("attempt-1", ExecutionOutcome::Indeterminate).unwrap();
        assert_eq!(receipt.outcome, ExecutionOutcome::Indeterminate);
        assert_eq!(
            lease.prepare_for_execution(&witness, &action, "f1", "attempt-2"),
            Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation)
        );
        let reconciled = lease.reconcile_indeterminate("attempt-1", ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(reconciled.outcome, ExecutionOutcome::Succeeded);
        assert_eq!(lease.state, AuthorizationLeaseState::Exhausted);
    }

    #[test]
    fn authorization_lease_rejects_frame_support_policy_or_epoch_changes() {
        let action = EpistemicAction::new("a-binding", "intervention", ActionRisk::High);
        let base = ActionAuthorizationWitness {
            action_id: "a-binding".into(), action_digest: "sha256:canonical".into(),
            frame: "f1".into(), support_digest: "sha256:support".into(),
            policy: "policy-v1".into(), decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(), expires_at: None, authority_epoch: 9,
        };
        for witness in [
            ActionAuthorizationWitness { frame: "f2".into(), ..base.clone() },
            ActionAuthorizationWitness { support_digest: "sha256:other".into(), ..base.clone() },
            ActionAuthorizationWitness { policy: "policy-v2".into(), ..base.clone() },
            ActionAuthorizationWitness { authority_epoch: 10, ..base.clone() },
        ] {
            let mut lease = AuthorizationLease::new("a-binding", "sha256:canonical", "sha256:support", "policy-v1", 9, 1);
            assert_eq!(
                lease.prepare_for_execution(&witness, &action, "f1", "attempt"),
                Err(AuthorizationConsumptionError::InvalidBinding)
            );
        }
    }

    #[test]
    fn execution_receipt_is_not_an_authorization_witness() {
        let receipt = ExecutionReceipt {
            action_id: "a-receipt".into(), action_digest: "sha256:canonical".into(),
            attempt_id: "attempt-1".into(), authority_epoch: 1, outcome: ExecutionOutcome::Succeeded,
        };
        assert_eq!(receipt.outcome, ExecutionOutcome::Succeeded);
    }

    #[test]
    fn high_risk_action_is_gated_but_executed_history_is_preserved() {
        let revision = EpistemicFrameRevision {
            prior_frame: "frame@1".into(),
            revised_frame: "frame@2".into(),
            trigger: "causal model changed".into(),
            newly_represented: None,
            scope_change: "causal model".into(),
            affected_conclusions: vec!["c1".into()],
            impact: EpistemicFrameImpact {
                evidence_boundary: false,
                ontology: false,
                causal_model: true,
                exclusions: false,
                blind_spots: false,
            },
        };

        let mut action = EpistemicAction::new("a1", "intervention", ActionRisk::High);
        action.dependencies.push(ActionDependency {
            conclusion_id: "c1".into(),
            kind: ActionDependencyKind::CausalBasis,
        });

        let mut graph = ActionDependencyGraph::default();
        graph.add(action);

        assert_eq!(
            graph.reevaluate_from_frame_revision(&revision, &["c1".into()]),
            vec!["a1"]
        );
        assert_eq!(graph.actions[0].status, ActionStatus::RequiresReevaluation);
        assert_eq!(
            graph.actions[0].reevaluation.as_ref().unwrap().revised_frame,
            "frame@2"
        );
    }
}
