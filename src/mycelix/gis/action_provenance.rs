//! Provenance-aware action gating for GIS.
//!
//! This module separates historical authorization from current authorization.
//! A frame revision can require fresh support for future high-risk execution
//! without rewriting the historical record of an action that already happened.

use super::ignorance_types::{ConclusionDependencyGraph, EpistemicFrameImpact, EpistemicFrameRevision};
use sha2::{Digest, Sha256};

fn append_len_prefixed<H: Digest>(hasher: &mut H, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

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
    /// Stable authorization instance identity. Fresh presentation identifiers
    /// must not silently create a fresh spendable authority for the same action.
    pub authorization_instance: String,
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
            && !self.authorization_instance.is_empty()
            && self.action_digest == action.canonical_action_digest()
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
    /// The durable pre-dispatch fence. No effect sink may be entered until
    /// this state is durably recorded.
    DispatchPending { attempt_id: String },
    /// The external sink has been entered; this is durable dispatch evidence,
    /// not proof that the protected effect succeeded.
    Invoked { attempt_id: String },
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
    pub authorization_instance: String,
    pub action_digest: String,
    /// Stable downstream replay identity for the exact authorized action.
    ///
    /// This intentionally excludes the executor attempt ID: retrying or
    /// replacing an executor must reuse the same provider identity, while a
    /// freshly issued authorization instance gets a distinct identity.
    pub provider_idempotency_key: String,
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
    PreDispatchRecoveryNotAllowed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizationLease {
    pub action_id: String,
    /// Durable identity of this authorization issuance. It is intentionally
    /// distinct from the canonical action identity so a later explicit
    /// authorization can exist without making the old authority replayable.
    pub authorization_instance: String,
    pub action_digest: String,
    pub support_digest: String,
    pub policy: String,
    pub authority_epoch: u64,
    pub remaining_executions: u32,
    pub state: AuthorizationLeaseState,
}

impl AuthorizationLease {
    /// Derive the stable downstream replay identity from the native grant.
    ///
    /// The derivation is intentionally independent of authorization presentation,
    /// operation IDs, attempt IDs, wrappers, sessions, and provider-selected
    /// identifiers. The exact action digest and effecting target are included so
    /// one native grant cannot be rebound to a materially different sink.
    pub fn provider_idempotency_key_for_native_replay(
        &self,
        native_replay_identity: &str,
        target_identity: &str,
    ) -> Result<String, AuthorizationConsumptionError> {
        if native_replay_identity.is_empty() || target_identity.is_empty() || self.action_digest.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:gis:provider-idempotency:v2\n");
        append_len_prefixed(&mut hasher, native_replay_identity.as_bytes());
        append_len_prefixed(&mut hasher, target_identity.as_bytes());
        append_len_prefixed(&mut hasher, self.action_digest.as_bytes());
        Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
    }

    /// Legacy authorization-instance based derivation retained exactly for
    /// backward-compatible reconstruction of historical unbound receipts.
    /// Effectful execution MUST use `provider_idempotency_key_for_native_replay`.
    #[deprecated(note = "effectful execution must derive provider idempotency from native replay identity")]
    pub fn provider_idempotency_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:gis:provider-idempotency:v1\n");
        hasher.update((self.authorization_instance.len() as u64).to_be_bytes());
        hasher.update(self.authorization_instance.as_bytes());
        hasher.update((self.action_digest.len() as u64).to_be_bytes());
        hasher.update(self.action_digest.as_bytes());
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    pub fn new(
        action_id: impl Into<String>,
        action_digest: impl Into<String>,
        support_digest: impl Into<String>,
        policy: impl Into<String>,
        authority_epoch: u64,
        execution_budget: u32,
    ) -> Self {
        let action_id = action_id.into();
        Self::new_with_instance(
            action_id.clone(),
            action_id,
            action_digest,
            support_digest,
            policy,
            authority_epoch,
            execution_budget,
        )
    }

    /// Construct an explicitly identified authorization issuance.
    ///
    /// The instance ID is caller-supplied and must be persisted before it can
    /// become spendable. This prevents fresh token/presentation IDs from
    /// becoming implicit fresh authority.
    pub fn new_with_instance(
        authorization_instance: impl Into<String>,
        action_id: impl Into<String>,
        action_digest: impl Into<String>,
        support_digest: impl Into<String>,
        policy: impl Into<String>,
        authority_epoch: u64,
        execution_budget: u32,
    ) -> Self {
        Self {
            action_id: action_id.into(),
            authorization_instance: authorization_instance.into(),
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
            || witness.authorization_instance != self.authorization_instance
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

    /// Effectful execution must use an action carrying an exact sink binding.
    /// The witness already commits to the action digest, so the sink cannot be
    /// swapped after authorization without invalidating that witness.
    pub fn prepare_for_effect_execution(
        &mut self,
        witness: &ActionAuthorizationWitness,
        action: &EpistemicAction,
        expected_effect: &ActionEffectBinding,
        current_frame: &str,
        attempt_id: impl Into<String>,
    ) -> Result<(), AuthorizationConsumptionError> {
        if action.effect_binding.as_ref() != Some(expected_effect) {
            return Err(AuthorizationConsumptionError::InvalidBinding);
        }
        self.prepare_for_execution(witness, action, current_frame, attempt_id)
    }

    /// Cross the durable effect boundary. The caller must persist this state
    /// before invoking any external effect sink.
    pub fn mark_dispatch_pending(
        &mut self,
        attempt_id: &str,
    ) -> Result<(), AuthorizationConsumptionError> {
        if matches!(
            &self.state,
            AuthorizationLeaseState::Prepared { attempt_id: id } if id == attempt_id
        ) {
            self.state = AuthorizationLeaseState::DispatchPending {
                attempt_id: attempt_id.to_owned(),
            };
            Ok(())
        } else if matches!(
            &self.state,
            AuthorizationLeaseState::DispatchPending { attempt_id: id } if id == attempt_id
        ) {
            Ok(())
        } else {
            Err(AuthorizationConsumptionError::AttemptMismatch)
        }
    }

    /// Record that provider entry has begun after a durable DispatchPending fence.
    ///
    /// This transition is deliberately separate from the outcome: entering the
    /// sink does not establish success. A crash after entry but before an
    /// authoritative outcome remains recoverable as Indeterminate.
    pub fn mark_invoked(
        &mut self,
        attempt_id: &str,
    ) -> Result<(), AuthorizationConsumptionError> {
        if matches!(
            &self.state,
            AuthorizationLeaseState::DispatchPending { attempt_id: id } if id == attempt_id
        ) {
            self.state = AuthorizationLeaseState::Invoked {
                attempt_id: attempt_id.to_owned(),
            };
            Ok(())
        } else if matches!(
            &self.state,
            AuthorizationLeaseState::Invoked { attempt_id: id } if id == attempt_id
        ) {
            Ok(())
        } else {
            Err(AuthorizationConsumptionError::AttemptMismatch)
        }
    }

    pub fn commit(
        &mut self,
        attempt_id: &str,
        outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationConsumptionError> {
        let prepared = matches!(
            &self.state,
            AuthorizationLeaseState::DispatchPending { attempt_id: id }
                | AuthorizationLeaseState::Invoked { attempt_id: id }
                if id == attempt_id
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
            authorization_instance: self.authorization_instance.clone(),
            action_digest: self.action_digest.clone(),
            provider_idempotency_key: self.provider_idempotency_key(),
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

        /// Revoke authority before the effect boundary. Revocation is terminal and
    /// cannot be undone by presenting the old authorization witness again.
    pub fn revoke(&mut self) -> Result<(), AuthorizationConsumptionError> {
        if matches!(self.state, AuthorizationLeaseState::Ready) {
            self.state = AuthorizationLeaseState::Revoked;
            Ok(())
        } else {
            Err(AuthorizationConsumptionError::NotReady)
        }
    }

    /// Expire authority when its validity window is no longer acceptable.
    /// Like revocation, expiry is terminal and does not replenish budget.
    pub fn expire(&mut self) -> Result<(), AuthorizationConsumptionError> {
        if matches!(self.state, AuthorizationLeaseState::Ready) {
            self.state = AuthorizationLeaseState::Expired;
            Ok(())
        } else {
            Err(AuthorizationConsumptionError::NotReady)
        }
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
            authorization_instance: self.authorization_instance.clone(),
            action_digest: self.action_digest.clone(),
            provider_idempotency_key: self.provider_idempotency_key(),
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
pub struct ActionEffectBinding {
    /// Executor-observed identity of the effect target.
    pub target_identity: String,
    /// Intended audience/environment of the effecting interface.
    pub audience: String,
    /// Exact adapter/finality sink that is permitted to effect the action.
    pub adapter: String,
}

impl ActionEffectBinding {
    pub fn new(
        target_identity: impl Into<String>,
        audience: impl Into<String>,
        adapter: impl Into<String>,
    ) -> Self {
        Self {
            target_identity: target_identity.into(),
            audience: audience.into(),
            adapter: adapter.into(),
        }
    }

    pub fn canonical_digest(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SYMTHEA-GIS-EFFECT-V1");
        fn append_field(bytes: &mut Vec<u8>, value: &[u8]) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value);
        }
        append_field(&mut bytes, self.target_identity.as_bytes());
        append_field(&mut bytes, self.audience.as_bytes());
        append_field(&mut bytes, self.adapter.as_bytes());

        let digest = Sha256::digest(bytes);
        let mut encoded = String::with_capacity(71);
        encoded.push_str("sha256:");
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        encoded
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicAction {
    pub id: String,
    pub description: String,
    pub risk: ActionRisk,
    pub dependencies: Vec<ActionDependency>,
    /// Exact effect boundary covered by the action digest. None is retained for
    /// informational/control-plane actions that do not cross an effect sink.
    pub effect_binding: Option<ActionEffectBinding>,
    pub status: ActionStatus,
    pub historical_decisions: Vec<ActionDecisionWitness>,
    pub reevaluation: Option<ActionReevaluationWitness>,
}

impl EpistemicAction {
    /// Digest the immutable action contract, excluding lifecycle/history fields.
    /// Dependency order is canonicalized because the dependency set is semantic.
    pub fn canonical_action_digest(&self) -> String {
        let mut dependencies: Vec<_> = self
            .dependencies
            .iter()
            .map(|dependency| {
                (
                    dependency.conclusion_id.as_str(),
                    match dependency.kind {
                        ActionDependencyKind::ConclusionSupport => 0u8,
                        ActionDependencyKind::CausalBasis => 1,
                        ActionDependencyKind::OntologyBasis => 2,
                        ActionDependencyKind::EvidenceBasis => 3,
                        ActionDependencyKind::AssumptionBasis => 4,
                    },
                )
            })
            .collect();
        dependencies.sort_unstable();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SYMTHEA-GIS-ACTION-V1");
        fn append_field(bytes: &mut Vec<u8>, value: &[u8]) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value);
        }
        append_field(&mut bytes, self.id.as_bytes());
        append_field(&mut bytes, self.description.as_bytes());
        append_field(
            &mut bytes,
            &[match self.risk {
                ActionRisk::Informational => 0,
                ActionRisk::Low => 1,
                ActionRisk::High => 2,
                ActionRisk::Critical => 3,
            }],
        );
        for (conclusion_id, kind) in dependencies {
            append_field(&mut bytes, conclusion_id.as_bytes());
            append_field(&mut bytes, &[kind]);
        }
        match &self.effect_binding {
            Some(effect) => {
                append_field(&mut bytes, b"effect-bound");
                append_field(&mut bytes, effect.canonical_digest().as_bytes());
            }
            None => append_field(&mut bytes, b"no-effect-binding"),
        }

        let digest = Sha256::digest(bytes);
        let mut encoded = String::with_capacity(71);
        encoded.push_str("sha256:");
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        encoded
    }

    pub fn new(id: impl Into<String>, description: impl Into<String>, risk: ActionRisk) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            risk,
            dependencies: Vec::new(),
            effect_binding: None,
            status: ActionStatus::Ready,
            historical_decisions: Vec::new(),
            reevaluation: None,
        }
    }

    /// Bind the immutable executable action to one exact effect boundary.
    /// The binding becomes part of the canonical action digest, so changing the
    /// target, audience, or adapter invalidates prior authorization.
    pub fn with_effect_binding(mut self, effect: ActionEffectBinding) -> Self {
        self.effect_binding = Some(effect);
        self
    }

    pub fn effect_binding(&self) -> Option<&ActionEffectBinding> {
        self.effect_binding.as_ref()
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

    #[test]
    fn provider_idempotency_key_is_stable_across_attempt_metadata() {
        let lease_a=AuthorizationLease::new_with_instance(
            "authorization-A","action-A","sha256:action","support","policy",1,1
        );
        let lease_b=AuthorizationLease::new_with_instance(
            "authorization-B","action-A","sha256:action","support","policy",1,1
        );

        let key_a=lease_a.provider_idempotency_key_for_native_replay("native-grant-1","target-A").unwrap();
        let key_b=lease_b.provider_idempotency_key_for_native_replay("native-grant-1","target-A").unwrap();
        assert_eq!(key_a,key_b);

        let different_native=lease_a.provider_idempotency_key_for_native_replay("native-grant-2","target-A").unwrap();
        let different_target=lease_a.provider_idempotency_key_for_native_replay("native-grant-1","target-B").unwrap();
        assert_ne!(key_a,different_native);
        assert_ne!(key_a,different_target);
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
        let action_digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: "a-bound".into(),
            authorization_instance: "a-bound".into(), action_digest: action_digest.clone(),
            frame: "f2".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v2".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: Some("2026-10-02T20:05:00Z".into()),
            authority_epoch: 1,
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
    fn dispatch_pending_is_the_only_effect_entry_state() {
        let mut action = EpistemicAction::new("dispatch-action", "effect", ActionRisk::Critical);
        action = action.with_effect_binding(ActionEffectBinding::new("target-a", "prod", "adapter-a"));
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: action.id.clone(),
            authorization_instance: "approval-1".into(),
            action_digest: digest.clone(),
            frame: "frame@1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-1", action.id.clone(), digest, "sha256:support", "policy-v1", 1, 1,
        );
        lease.prepare_for_effect_execution(
            &witness, &action, action.effect_binding().unwrap(), "frame@1", "attempt-1",
        ).unwrap();
        assert!(matches!(lease.state, AuthorizationLeaseState::Prepared { .. }));
        lease.mark_dispatch_pending("attempt-1").unwrap();
        assert!(matches!(lease.state, AuthorizationLeaseState::DispatchPending { .. }));
        assert_eq!(
            lease.commit("attempt-1", ExecutionOutcome::Succeeded).unwrap().outcome,
            ExecutionOutcome::Succeeded
        );
    }

    #[test]
    fn invoked_is_a_distinct_durable_dispatch_evidence_state() {
        let action = EpistemicAction::new("invoked-action", "effect", ActionRisk::Critical);
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: action.id.clone(),
            authorization_instance: "approval-1".into(),
            action_digest: digest.clone(),
            frame: "frame-1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-1", action.id.clone(), digest, "sha256:support", "policy-v1", 1, 1,
        );
        lease.prepare_for_execution(&witness, &action, "frame-1", "attempt-1").unwrap();
        lease.mark_dispatch_pending("attempt-1").unwrap();
        lease.mark_invoked("attempt-1").unwrap();
        assert!(matches!(
            lease.state,
            AuthorizationLeaseState::Invoked { ref attempt_id } if attempt_id == "attempt-1"
        ));
        let receipt = lease.commit("attempt-1", ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(receipt.outcome, ExecutionOutcome::Succeeded);
    }

    #[test]
    fn invoked_wrong_attempt_is_fenced() {
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-1", "action-1", "sha256:action", "sha256:support", "policy-v1", 1, 1,
        );
        lease.state = AuthorizationLeaseState::DispatchPending { attempt_id: "attempt-1".into() };
        assert_eq!(
            lease.mark_invoked("attempt-2"),
            Err(AuthorizationConsumptionError::AttemptMismatch)
        );
    }

    #[test]
    fn provider_idempotency_identity_is_stable_across_attempts() {
        let lease = AuthorizationLease::new_with_instance(
            "approval-1", "action-1", "sha256:action", "sha256:support", "policy-v1", 1, 3,
        );
        let first = lease.provider_idempotency_key();
        let second = lease.provider_idempotency_key();
        assert_eq!(first, second);
        assert!(first.starts_with("sha256:"));
    }

    #[test]
    fn provider_idempotency_identity_changes_with_authorization_instance() {
        let old = AuthorizationLease::new_with_instance(
            "approval-old", "action-1", "sha256:action", "sha256:support", "policy-v1", 1, 1,
        );
        let fresh = AuthorizationLease::new_with_instance(
            "approval-new", "action-1", "sha256:action", "sha256:support", "policy-v1", 1, 1,
        );
        assert_ne!(old.provider_idempotency_key(), fresh.provider_idempotency_key());
    }

    #[test]
    fn provider_idempotency_identity_does_not_depend_on_attempt_id() {
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-1", "action-1", "sha256:action", "sha256:support", "policy-v1", 1, 1,
        );
        let key = lease.provider_idempotency_key();
        lease.state = AuthorizationLeaseState::Prepared { attempt_id: "attempt-a".into() };
        assert_eq!(lease.provider_idempotency_key(), key);
        lease.state = AuthorizationLeaseState::DispatchPending { attempt_id: "attempt-b".into() };
        assert_eq!(lease.provider_idempotency_key(), key);
    }

    #[test]
    fn dispatch_pending_wrong_attempt_cannot_cross_effect_boundary() {
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-1", "action-1", "sha256:action", "sha256:support", "policy-v1", 1, 1,
        );
        lease.state = AuthorizationLeaseState::Prepared { attempt_id: "attempt-1".into() };
        assert_eq!(
            lease.mark_dispatch_pending("attempt-2"),
            Err(AuthorizationConsumptionError::AttemptMismatch)
        );
        assert!(matches!(lease.state, AuthorizationLeaseState::Prepared { .. }));
    }

    #[test]
    fn authorization_lease_blocks_replay_and_fresh_witness_reissuance() {
        let action = EpistemicAction::new("a-lease", "intervention", ActionRisk::Critical);
        let action_digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: "a-lease".into(),
            authorization_instance: "a-lease".into(), action_digest: action_digest.clone(),
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 7,
        };
        let mut lease = AuthorizationLease::new("a-lease", action_digest.clone(), "sha256:support", "policy-v1", 7, 1);
        lease.&witness, &action, "f1", "attempt-1").unwrap();

        lease.mark_dispatch_pending("attempt-1").unwrap();
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
        let action_digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: "a-concurrent".into(), authorization_instance: "a-concurrent".into(), action_digest: action_digest.clone(),
            frame: "f1".into(), support_digest: "sha256:support".into(),
            policy: "policy-v1".into(), decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(), expires_at: None, authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new("a-concurrent", action_digest.clone(), "sha256:support", "policy-v1", 1, 1);
        lease.&witness, &action, "f1", "attempt-1").unwrap();

        lease.mark_dispatch_pending("attempt-1").unwrap();
        assert_eq!(
            lease.prepare_for_execution(&witness, &action, "f1", "attempt-2"),
            Err(AuthorizationConsumptionError::NotReady)
        );
    }

    #[test]
    fn indeterminate_commit_requires_reconciliation_before_retry() {
        let action = EpistemicAction::new("a-crash", "intervention", ActionRisk::Critical);
        let action_digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: "a-crash".into(), authorization_instance: "a-crash".into(), action_digest: action_digest.clone(),
            frame: "f1".into(), support_digest: "sha256:support".into(),
            policy: "policy-v1".into(), decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(), expires_at: None, authority_epoch: 3,
        };
        let mut lease = AuthorizationLease::new("a-crash", action_digest.clone(), "sha256:support", "policy-v1", 3, 1);
        lease.&witness, &action, "f1", "attempt-1").unwrap();

        lease.mark_dispatch_pending("attempt-1").unwrap();
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
    fn fresh_presentation_cannot_replay_an_existing_authorization_instance() {
        let action = EpistemicAction::new("a-semantic-replay", "intervention", ActionRisk::Critical);
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: action.id.clone(),
            authorization_instance: "approval-2026-10-02-001".into(),
            action_digest: digest.clone(),
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-2026-10-02-001", action.id.clone(), digest,
            "sha256:support", "policy-v1", 1, 1,
        );
        lease.&witness, &action, "f1", "attempt-1").unwrap();

        lease.mark_dispatch_pending("attempt-1").unwrap();
        lease.commit("attempt-1", ExecutionOutcome::Succeeded).unwrap();

        let replay = ActionAuthorizationWitness {
            issued_at: "2026-10-02T20:01:00Z".into(),
            ..witness
        };
        assert_eq!(
            lease.prepare_for_execution(&replay, &action, "f1", "attempt-2"),
            Err(AuthorizationConsumptionError::BudgetExhausted)
        );
    }

    #[test]
    fn explicit_new_authorization_instance_is_distinct_authority() {
        let action = EpistemicAction::new("a-new-issuance", "intervention", ActionRisk::Critical);
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: action.id.clone(),
            authorization_instance: "approval-old".into(),
            action_digest: digest.clone(),
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        let mut old = AuthorizationLease::new_with_instance(
            "approval-old", action.id.clone(), digest.clone(),
            "sha256:support", "policy-v1", 1, 1,
        );
        old.&witness, &action, "f1", "attempt-old").unwrap();

        old.mark_dispatch_pending("attempt-old").unwrap();
        old.commit("attempt-old", ExecutionOutcome::Succeeded).unwrap();

        let fresh_witness = ActionAuthorizationWitness {
            authorization_instance: "approval-new".into(),
            issued_at: "2026-10-02T20:02:00Z".into(),
            ..witness
        };
        let mut fresh = AuthorizationLease::new_with_instance(
            "approval-new", action.id.clone(), digest,
            "sha256:support", "policy-v1", 1, 1,
        );
        assert!(fresh.prepare_for_execution(&fresh_witness, &action, "f1", "attempt-new").is_ok());
    }

    #[test]
    fn authorization_lease_rejects_frame_support_policy_or_epoch_changes() {
        let action = EpistemicAction::new("a-binding", "intervention", ActionRisk::High);
        let action_digest = action.canonical_action_digest();
        let base = ActionAuthorizationWitness {
            action_id: "a-binding".into(), authorization_instance: "a-binding".into(), action_digest: action_digest.clone(),
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
            let mut lease = AuthorizationLease::new("a-binding", action_digest.clone(), "sha256:support", "policy-v1", 9, 1);
            assert_eq!(
                lease.prepare_for_execution(&witness, &action, "f1", "attempt"),
                Err(AuthorizationConsumptionError::InvalidBinding)
            );
        }
    }

    #[test]
    fn revoked_or_expired_leases_cannot_be_resurrected() {
        let action = EpistemicAction::new("a-terminal", "intervention", ActionRisk::High);
        let action_digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: "a-terminal".into(), authorization_instance: "a-terminal".into(), action_digest: action_digest.clone(),
            frame: "f1".into(), support_digest: "sha256:support".into(),
            policy: "policy-v1".into(), decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(), expires_at: None, authority_epoch: 1,
        };

        let mut revoked = AuthorizationLease::new(
            "a-terminal", action_digest.clone(), "sha256:support", "policy-v1", 1, 1,
        );
        revoked.revoke().unwrap();
        assert_eq!(
            revoked.prepare_for_execution(&witness, &action, "f1", "attempt"),
            Err(AuthorizationConsumptionError::NotReady)
        );
        assert_eq!(revoked.revoke(), Err(AuthorizationConsumptionError::NotReady));

        let mut expired = AuthorizationLease::new(
            "a-terminal", action_digest.clone(), "sha256:support", "policy-v1", 1, 1,
        );
        expired.expire().unwrap();
        assert_eq!(
            expired.prepare_for_execution(&witness, &action, "f1", "attempt"),
            Err(AuthorizationConsumptionError::NotReady)
        );
        assert_eq!(expired.expire(), Err(AuthorizationConsumptionError::NotReady));
    }

    #[test]
    fn prepared_lease_cannot_be_revoked_or_expired_without_reconciliation() {
        let action = EpistemicAction::new("a-race", "intervention", ActionRisk::Critical);
        let action_digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: "a-race".into(),
            authorization_instance: "a-race".into(), action_digest: action_digest.clone(),
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new(
            "a-race",
            action_digest,
            "sha256:support",
            "policy-v1",
            1,
            1,
        );
        lease.&witness, &action, "f1", "attempt-1").unwrap();

        lease.mark_dispatch_pending("attempt-1").unwrap();

        assert_eq!(
            lease.revoke(),
            Err(AuthorizationConsumptionError::NotReady)
        );
        assert_eq!(
            lease.expire(),
            Err(AuthorizationConsumptionError::NotReady)
        );
        assert_eq!(
            lease.commit("attempt-1", ExecutionOutcome::Indeterminate).unwrap().outcome,
            ExecutionOutcome::Indeterminate
        );
    }

    #[test]
    fn execution_receipt_is_not_an_authorization_witness() {
        let receipt = ExecutionReceipt {
            action_id: "a-receipt".into(), authorization_instance: "approval-1".into(),
            action_digest: "sha256:execution-only".into(),
            provider_idempotency_key: "sha256:provider".into(),
            attempt_id: "attempt-1".into(), authority_epoch: 1, outcome: ExecutionOutcome::Succeeded,
        };
        assert_eq!(receipt.outcome, ExecutionOutcome::Succeeded);
    }

    #[test]
    fn effect_boundary_is_part_of_canonical_authorization_identity() {
        let effect = ActionEffectBinding::new("target:payments/ledger-7", "audience:ledger", "adapter:ledger-v2");
        let action = EpistemicAction::new("a-effect", "transfer", ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest = action.canonical_action_digest();

        let witness = ActionAuthorizationWitness {
            action_id: action.id.clone(),
            authorization_instance: "approval-effect-1".into(),
            action_digest: digest.clone(),
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-effect-1", action.id.clone(), digest,
            "sha256:support", "policy-v1", 1, 1,
        );
        lease.&witness, &action, &effect, "f1", "attempt-1").unwrap();

        lease.mark_dispatch_pending("attempt-1").unwrap();

        let wrong_effect = ActionEffectBinding::new("target:payments/ledger-8", "audience:ledger", "adapter:ledger-v2");
        assert_eq!(
            lease.commit("attempt-1", ExecutionOutcome::Succeeded).unwrap().outcome,
            ExecutionOutcome::Succeeded
        );
        assert_ne!(wrong_effect.canonical_digest(), effect.canonical_digest());
    }

    #[test]
    fn effect_rebinding_invalidates_existing_authorization() {
        let effect = ActionEffectBinding::new("target:device-1", "audience:actuator", "adapter:v1");
        let mut action = EpistemicAction::new("a-effect-rebind", "actuate", ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: action.id.clone(),
            authorization_instance: "approval-rebind-1".into(),
            action_digest: digest.clone(),
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        let mut lease = AuthorizationLease::new_with_instance(
            "approval-rebind-1", action.id.clone(), digest,
            "sha256:support", "policy-v1", 1, 1,
        );

        action.effect_binding = Some(ActionEffectBinding::new("target:device-2", "audience:actuator", "adapter:v1"));
        assert_eq!(
            lease.prepare_for_execution(&witness, &action, "f1", "attempt-1"),
            Err(AuthorizationConsumptionError::InvalidBinding)
        );
    }

    #[test]
    fn canonical_action_digest_changes_when_executable_contract_changes() {
        let mut action = EpistemicAction::new("a-canonical", "intervention", ActionRisk::High);
        let original = action.canonical_action_digest();

        action.description = "different intervention".into();
        assert_ne!(original, action.canonical_action_digest());

        action.description = "intervention".into();
        action.dependencies.push(ActionDependency {
            conclusion_id: "c1".into(),
            kind: ActionDependencyKind::CausalBasis,
        });
        assert_ne!(original, action.canonical_action_digest());

        let with_dependency = action.canonical_action_digest();
        action.dependencies.reverse();
        assert_eq!(with_dependency, action.canonical_action_digest());
    }

    #[test]
    fn authorization_rejects_witness_for_mutated_action_contract() {
        let mut action = EpistemicAction::new("a-mutation", "intervention", ActionRisk::Critical);
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id: "a-mutation".into(),
            authorization_instance: "legacy-instance".into(), action_digest: digest,
            frame: "f1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: None,
            authority_epoch: 1,
        };
        assert!(witness.is_bound_to(&action, "f1", "sha256:support", "policy-v1"));

        action.description = "mutated intervention".into();
        assert!(!witness.is_bound_to(&action, "f1", "sha256:support", "policy-v1"));
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
