// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed action lifecycle for autonomous systems.
//!
//! Invalid lifecycle shortcuts are absent from the public API. In particular,
//! a proposed action cannot be marked executed, an authorized action requires a
//! matching one-shot capability, and an executed action requires a separately
//! authorized observation before it can be resolved.
//!
//! ```compile_fail
//! use symthaea_ai_assurance::{Action, PrincipalId, Proposed, Scope, Write};
//! let action = Action::<Write, Proposed>::propose(
//!     PrincipalId::new(),
//!     "edit-source",
//!     Scope::new("workspace", ["symthaea", "src"]).unwrap(),
//!     b"candidate patch",
//! );
//! // No such transition exists.
//! let _ = action.record_execution([0_u8; 32]);
//! ```
//!
//! ```compile_fail
//! use std::time::SystemTime;
//! use symthaea_ai_assurance::{
//!     Action, AuthorityRoot, PrincipalId, Proposed, Read, Scope, Write,
//! };
//! let actor = PrincipalId::new();
//! let scope = Scope::new("workspace", ["symthaea"]).unwrap();
//! let root = AuthorityRoot::new(PrincipalId::new());
//! let read_grant = root.issue_one_shot::<Read>(actor, scope.clone(), None);
//! let action = Action::<Write, Proposed>::propose(actor, "edit", scope, b"patch")
//!     .assess(symthaea_ai_assurance::ActionRisk::Reversible);
//! // A read capability cannot authorize a write action.
//! let _ = action.authorize(read_grant, SystemTime::now());
//! ```

use crate::capability::{
    CapabilityKind, GrantError, GrantId, GrantMetadata, Observe, OneShotCapability, PrincipalId,
    Scope,
};
use std::fmt;
use std::marker::PhantomData;
use std::time::SystemTime;
use uuid::Uuid;

/// Stable identity of an action lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(Uuid);

impl ActionId {
    fn fresh() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the underlying UUID.
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

/// Risk classification independent of any particular cognition architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionRisk {
    /// Read-only or observational behavior.
    Observation,
    /// Side effects are intended to be straightforwardly reversible.
    Reversible,
    /// Persistent state changes that require explicit recovery semantics.
    StateModifying,
    /// Potentially destructive behavior.
    Destructive,
    /// Irreversible or safety-critical behavior.
    Critical,
}

/// Immutable description bound to the action fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDescriptor {
    kind: String,
    scope: Scope,
    fingerprint: [u8; 32],
}

impl ActionDescriptor {
    fn new<K: CapabilityKind>(
        actor: PrincipalId,
        kind: impl Into<String>,
        scope: Scope,
        canonical_payload: &[u8],
    ) -> Self {
        let kind = kind.into();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-ai-assurance/action-v1\0");
        hash_field(&mut hasher, K::NAME.as_bytes());
        hash_field(&mut hasher, actor.as_uuid().as_bytes());
        hash_field(&mut hasher, kind.as_bytes());
        hash_field(&mut hasher, scope.namespace().as_bytes());
        for segment in scope.segments() {
            hash_field(&mut hasher, segment.as_bytes());
        }
        hash_field(&mut hasher, canonical_payload);

        Self {
            kind,
            scope,
            fingerprint: *hasher.finalize().as_bytes(),
        }
    }

    /// Logical action kind supplied by the host adapter.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Resource scope affected by the action.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Domain-separated digest binding actor, capability class, kind, scope,
    /// and canonical payload bytes.
    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Initial state: proposed by a model or planner but not yet risk assessed.
#[derive(Debug)]
pub struct Proposed;

/// Risk has been explicitly classified.
#[derive(Debug)]
pub struct RiskAssessed {
    risk: ActionRisk,
}

/// A matching one-shot capability was validated and consumed.
#[derive(Debug)]
pub struct Authorized {
    risk: ActionRisk,
    grant: GrantMetadata,
}

/// Trusted host execution has been recorded.
#[derive(Debug)]
pub struct Executed {
    risk: ActionRisk,
    grant: GrantMetadata,
    output_digest: [u8; 32],
}

/// A separately authorized observer recorded external evidence.
#[derive(Debug)]
pub struct Observed {
    risk: ActionRisk,
    grant: GrantMetadata,
    output_digest: [u8; 32],
    observer_grant: GrantMetadata,
    observation: Observation,
}

/// Final resolved state.
#[derive(Debug)]
pub struct Resolved {
    receipt: EvidenceReceipt,
}

/// An action whose available operations depend on typestate `S` and capability
/// class `K`.
#[derive(Debug)]
pub struct Action<K: CapabilityKind, S> {
    id: ActionId,
    actor: PrincipalId,
    descriptor: ActionDescriptor,
    state: S,
    _kind: PhantomData<K>,
}

impl<K: CapabilityKind, S> Action<K, S> {
    /// Stable action id across all state transitions.
    pub fn id(&self) -> ActionId {
        self.id
    }

    /// Principal on whose behalf the action is proposed.
    pub fn actor(&self) -> PrincipalId {
        self.actor
    }

    /// Immutable action descriptor.
    pub fn descriptor(&self) -> &ActionDescriptor {
        &self.descriptor
    }
}

impl<K: CapabilityKind> Action<K, Proposed> {
    /// Propose an action. This grants no execution authority.
    pub fn propose(
        actor: PrincipalId,
        kind: impl Into<String>,
        scope: Scope,
        canonical_payload: &[u8],
    ) -> Self {
        Self {
            id: ActionId::fresh(),
            actor,
            descriptor: ActionDescriptor::new::<K>(actor, kind, scope, canonical_payload),
            state: Proposed,
            _kind: PhantomData,
        }
    }

    /// Add an explicit risk classification.
    pub fn assess(self, risk: ActionRisk) -> Action<K, RiskAssessed> {
        Action {
            id: self.id,
            actor: self.actor,
            descriptor: self.descriptor,
            state: RiskAssessed { risk },
            _kind: PhantomData,
        }
    }
}

impl<K: CapabilityKind> Action<K, RiskAssessed> {
    /// Risk classification attached to this action.
    pub fn risk(&self) -> ActionRisk {
        self.state.risk
    }

    /// Consume a matching one-shot capability to authorize this exact action
    /// class and scope.
    pub fn authorize(
        self,
        grant: OneShotCapability<K>,
        now: SystemTime,
    ) -> Result<Action<K, Authorized>, ActionError> {
        grant.validate_at(now).map_err(ActionError::Grant)?;

        if grant.metadata().subject() != self.actor {
            return Err(ActionError::WrongSubject {
                expected: self.actor,
                actual: grant.metadata().subject(),
            });
        }

        if !grant.metadata().scope().contains(self.descriptor.scope()) {
            return Err(ActionError::ScopeMismatch {
                granted: grant.metadata().scope().clone(),
                required: self.descriptor.scope().clone(),
            });
        }

        let grant = grant.into_metadata();
        Ok(Action {
            id: self.id,
            actor: self.actor,
            descriptor: self.descriptor,
            state: Authorized {
                risk: self.state.risk,
                grant,
            },
            _kind: PhantomData,
        })
    }
}

impl<K: CapabilityKind> Action<K, Authorized> {
    /// Authorization grant id consumed by this action.
    pub fn grant_id(&self) -> GrantId {
        self.state.grant.grant_id()
    }

    /// Record the digest produced by a trusted executor adapter.
    ///
    /// This method deliberately performs no ambient process/filesystem/network
    /// operation. PR 2 will bind concrete executors to this transition.
    pub fn record_execution(self, output_digest: [u8; 32]) -> Action<K, Executed> {
        Action {
            id: self.id,
            actor: self.actor,
            descriptor: self.descriptor,
            state: Executed {
                risk: self.state.risk,
                grant: self.state.grant,
                output_digest,
            },
            _kind: PhantomData,
        }
    }
}

/// Externally observed outcome category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObservedOutcome {
    /// Intended effect was observed.
    Success,
    /// Some but not all intended effect was observed.
    Partial,
    /// No intended effect was observed.
    NoEffect,
    /// Failure occurred without the modeled harmful consequence.
    SafeFailure,
    /// Failure occurred with a modeled harmful consequence.
    UnsafeFailure,
}

/// Observation evidence supplied under independent observation authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    outcome: ObservedOutcome,
    evidence_digest: [u8; 32],
}

impl Observation {
    /// Construct a digest-bound observation. The observation does not become
    /// part of an action lineage until a matching `Observe` grant is consumed.
    pub fn new(outcome: ObservedOutcome, evidence_digest: [u8; 32]) -> Self {
        Self {
            outcome,
            evidence_digest,
        }
    }

    /// Observed outcome.
    pub fn outcome(&self) -> ObservedOutcome {
        self.outcome
    }

    /// Digest of externally produced observation evidence.
    pub fn evidence_digest(&self) -> [u8; 32] {
        self.evidence_digest
    }
}

impl<K: CapabilityKind> Action<K, Executed> {
    /// Record external observation using a separately issued one-shot observer
    /// grant whose scope covers the action.
    pub fn observe(
        self,
        observer: OneShotCapability<Observe>,
        observation: Observation,
        now: SystemTime,
    ) -> Result<Action<K, Observed>, ActionError> {
        observer.validate_at(now).map_err(ActionError::Grant)?;

        if !observer.metadata().scope().contains(self.descriptor.scope()) {
            return Err(ActionError::ObservationScopeMismatch {
                granted: observer.metadata().scope().clone(),
                required: self.descriptor.scope().clone(),
            });
        }

        let observer_grant = observer.into_metadata();
        Ok(Action {
            id: self.id,
            actor: self.actor,
            descriptor: self.descriptor,
            state: Observed {
                risk: self.state.risk,
                grant: self.state.grant,
                output_digest: self.state.output_digest,
                observer_grant,
                observation,
            },
            _kind: PhantomData,
        })
    }
}

/// Final interpretation of the external observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionDecision {
    /// Observation confirms the modeled success criterion.
    Confirmed,
    /// Observation positively contradicts the modeled success criterion.
    Contradicted,
    /// Evidence does not justify either confirmation or contradiction.
    Inconclusive,
}

/// Immutable evidence lineage emitted at resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceReceipt {
    action_id: ActionId,
    action_fingerprint: [u8; 32],
    actor: PrincipalId,
    action_scope: Scope,
    capability_kind: &'static str,
    risk: ActionRisk,
    authorization_grant_id: GrantId,
    observer_grant_id: GrantId,
    observer: PrincipalId,
    output_digest: [u8; 32],
    observation: Observation,
    resolution: ResolutionDecision,
}

impl EvidenceReceipt {
    /// Stable action identity.
    pub fn action_id(&self) -> ActionId {
        self.action_id
    }

    /// Digest of the authorized action descriptor and canonical payload.
    pub fn action_fingerprint(&self) -> [u8; 32] {
        self.action_fingerprint
    }

    /// Acting principal.
    pub fn actor(&self) -> PrincipalId {
        self.actor
    }

    /// Authorized action scope.
    pub fn action_scope(&self) -> &Scope {
        &self.action_scope
    }

    /// Capability marker name required by the action.
    pub fn capability_kind(&self) -> &'static str {
        self.capability_kind
    }

    /// Risk classification at authorization time.
    pub fn risk(&self) -> ActionRisk {
        self.risk
    }

    /// One-shot grant consumed to authorize execution.
    pub fn authorization_grant_id(&self) -> GrantId {
        self.authorization_grant_id
    }

    /// One-shot grant consumed to authorize observation.
    pub fn observer_grant_id(&self) -> GrantId {
        self.observer_grant_id
    }

    /// Principal that held observation authority.
    pub fn observer(&self) -> PrincipalId {
        self.observer
    }

    /// Digest reported by the trusted executor adapter.
    pub fn output_digest(&self) -> [u8; 32] {
        self.output_digest
    }

    /// External observation attached to the lineage.
    pub fn observation(&self) -> &Observation {
        &self.observation
    }

    /// Final resolution decision.
    pub fn resolution(&self) -> ResolutionDecision {
        self.resolution
    }
}

impl<K: CapabilityKind> Action<K, Observed> {
    /// Resolve an externally observed action and emit immutable lineage evidence.
    pub fn resolve(
        self,
        resolution: ResolutionDecision,
    ) -> (Action<K, Resolved>, EvidenceReceipt) {
        let receipt = EvidenceReceipt {
            action_id: self.id,
            action_fingerprint: self.descriptor.fingerprint(),
            actor: self.actor,
            action_scope: self.descriptor.scope().clone(),
            capability_kind: K::NAME,
            risk: self.state.risk,
            authorization_grant_id: self.state.grant.grant_id(),
            observer_grant_id: self.state.observer_grant.grant_id(),
            observer: self.state.observer_grant.subject(),
            output_digest: self.state.output_digest,
            observation: self.state.observation,
            resolution,
        };

        let state_receipt = receipt.clone();
        (
            Action {
                id: self.id,
                actor: self.actor,
                descriptor: self.descriptor,
                state: Resolved {
                    receipt: state_receipt,
                },
                _kind: PhantomData,
            },
            receipt,
        )
    }
}

impl<K: CapabilityKind> Action<K, Resolved> {
    /// Final receipt retained by the resolved typestate.
    pub fn receipt(&self) -> &EvidenceReceipt {
        &self.state.receipt
    }
}

/// Failure to advance an action through a guarded transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionError {
    /// Capability validation failed.
    Grant(GrantError),
    /// Execution grant belongs to a different actor.
    WrongSubject {
        /// Principal that proposed the action.
        expected: PrincipalId,
        /// Principal that holds the supplied grant.
        actual: PrincipalId,
    },
    /// Execution grant does not cover the requested scope.
    ScopeMismatch {
        /// Scope supplied by the capability.
        granted: Scope,
        /// Scope required by the action.
        required: Scope,
    },
    /// Observer authority does not cover the action scope.
    ObservationScopeMismatch {
        /// Scope supplied by observer authority.
        granted: Scope,
        /// Scope required by the action.
        required: Scope,
    },
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grant(error) => write!(f, "grant validation failed: {error}"),
            Self::WrongSubject { .. } => write!(f, "grant subject does not match action actor"),
            Self::ScopeMismatch { .. } => write!(f, "execution grant does not cover action scope"),
            Self::ObservationScopeMismatch { .. } => {
                write!(f, "observer grant does not cover action scope")
            }
        }
    }
}

impl std::error::Error for ActionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{AuthorityRoot, Read, Write};
    use std::time::Duration;

    fn scope(parts: &[&str]) -> Scope {
        Scope::new("workspace", parts.iter().copied()).unwrap()
    }

    #[test]
    fn authorization_binds_subject_scope_and_grant_lineage() {
        let issuer = AuthorityRoot::new(PrincipalId::new());
        let actor = PrincipalId::new();
        let requested = scope(&["symthaea", "src"]);
        let grant = issuer.issue_one_shot::<Write>(actor, scope(&["symthaea"]), None);
        let grant_id = grant.metadata().grant_id();

        let authorized = Action::<Write, Proposed>::propose(
            actor,
            "edit-source",
            requested,
            b"canonical-patch-v1",
        )
        .assess(ActionRisk::Reversible)
        .authorize(grant, SystemTime::now())
        .unwrap();

        assert_eq!(authorized.grant_id(), grant_id);
    }

    #[test]
    fn wrong_subject_fails_closed() {
        let issuer = AuthorityRoot::new(PrincipalId::new());
        let actor = PrincipalId::new();
        let grant = issuer.issue_one_shot::<Write>(PrincipalId::new(), scope(&["symthaea"]), None);

        let result = Action::<Write, Proposed>::propose(
            actor,
            "edit-source",
            scope(&["symthaea", "src"]),
            b"patch",
        )
        .assess(ActionRisk::Reversible)
        .authorize(grant, SystemTime::now());

        assert!(matches!(result, Err(ActionError::WrongSubject { .. })));
    }

    #[test]
    fn action_scope_cannot_exceed_grant_scope() {
        let issuer = AuthorityRoot::new(PrincipalId::new());
        let actor = PrincipalId::new();
        let grant = issuer.issue_one_shot::<Write>(actor, scope(&["symthaea", "scratch"]), None);

        let result = Action::<Write, Proposed>::propose(
            actor,
            "edit-source",
            scope(&["symthaea", "src"]),
            b"patch",
        )
        .assess(ActionRisk::Reversible)
        .authorize(grant, SystemTime::now());

        assert!(matches!(result, Err(ActionError::ScopeMismatch { .. })));
    }

    #[test]
    fn expired_one_shot_grant_cannot_authorize() {
        let now = SystemTime::now();
        let issuer = AuthorityRoot::new(PrincipalId::new());
        let actor = PrincipalId::new();
        let grant = issuer.issue_one_shot::<Read>(
            actor,
            scope(&["symthaea"]),
            Some(now - Duration::from_secs(1)),
        );

        let result = Action::<Read, Proposed>::propose(
            actor,
            "read-source",
            scope(&["symthaea", "src"]),
            b"src/lib.rs",
        )
        .assess(ActionRisk::Observation)
        .authorize(grant, now);

        assert!(matches!(result, Err(ActionError::Grant(GrantError::Expired { .. }))));
    }

    #[test]
    fn resolution_receipt_binds_execution_and_independent_observation() {
        let root = AuthorityRoot::new(PrincipalId::new());
        let actor = PrincipalId::new();
        let observer = PrincipalId::new();
        let action_scope = scope(&["symthaea", "src"]);
        let execution_grant = root.issue_one_shot::<Write>(actor, scope(&["symthaea"]), None);
        let observation_grant = root.issue_one_shot::<Observe>(observer, scope(&["symthaea"]), None);
        let execution_grant_id = execution_grant.metadata().grant_id();
        let observation_grant_id = observation_grant.metadata().grant_id();

        let executed = Action::<Write, Proposed>::propose(
            actor,
            "edit-source",
            action_scope.clone(),
            b"patch-v1",
        )
        .assess(ActionRisk::Reversible)
        .authorize(execution_grant, SystemTime::now())
        .unwrap()
        .record_execution([7_u8; 32]);

        let observed = executed
            .observe(
                observation_grant,
                Observation::new(ObservedOutcome::Success, [9_u8; 32]),
                SystemTime::now(),
            )
            .unwrap();
        let (resolved, receipt) = observed.resolve(ResolutionDecision::Confirmed);

        assert_eq!(receipt.authorization_grant_id(), execution_grant_id);
        assert_eq!(receipt.observer_grant_id(), observation_grant_id);
        assert_eq!(receipt.observer(), observer);
        assert_eq!(receipt.actor(), actor);
        assert_eq!(receipt.action_scope(), &action_scope);
        assert_eq!(receipt.output_digest(), [7_u8; 32]);
        assert_eq!(receipt.observation().evidence_digest(), [9_u8; 32]);
        assert_eq!(resolved.receipt(), &receipt);
    }

    #[test]
    fn action_fingerprint_changes_when_payload_changes() {
        let actor = PrincipalId::new();
        let scope = scope(&["symthaea", "src"]);
        let first = Action::<Write, Proposed>::propose(actor, "edit", scope.clone(), b"patch-a");
        let second = Action::<Write, Proposed>::propose(actor, "edit", scope, b"patch-b");
        assert_ne!(first.descriptor().fingerprint(), second.descriptor().fingerprint());
    }
}
