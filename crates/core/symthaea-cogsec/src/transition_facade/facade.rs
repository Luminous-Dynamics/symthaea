// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symthaea Cognitive Security Core
//!
//! Public, monitor-domain-sealed facade for the deterministic CogSec algebra.
//!
//! The underlying label/policy/state-transition implementation lives in a
//! private module. Security-relevant facts are deliberately *not* ordinary
//! serializable structs at this boundary: a trusted adapter must possess the
//! [`TrustedFactAuthority`] paired with one [`ReferenceMonitor`] domain before
//! it can create facts that the monitor will accept.
//!
//! A second independently bootstrapped monitor may evaluate its own facts, but
//! its permits are cryptographically-neutral **in-process capabilities** for a
//! different domain and must not authorize a protected sink owned by the first
//! monitor. Protected sinks therefore validate commit-permit domain affinity in
//! addition to consuming the one-use typestate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod legacy;

use std::sync::Arc;

pub use legacy::{
    ArtifactIntegrity, CognitiveSecurityLabel, Confidentiality, Consequence, ControlIntegrity,
    DecisionOutcome, DelegationError, Digest32, MonitorDecision, MutationKind, MutationReceipt,
    MutationRequest, OriginState, PolicyRule, PolicySnapshot, PrincipalId, ReasonCode, ResourceId,
    ResourceScope, TaintLevel,
};

/// Private identity shared only by one monitor and its trusted fact authority.
///
/// Pointer identity is sufficient for the in-process safe-Rust boundary: the
/// protected monitor keeps the allocation alive for its lifetime, so another
/// independently bootstrapped monitor cannot obtain the same identity by
/// constructing ordinary application data.
#[derive(Debug)]
struct MonitorDomainSeal;

/// Error returned when a security object is used outside the monitor domain
/// that issued it, or when the deterministic inner monitor rejects a request.
#[derive(Debug, PartialEq, Eq)]
pub enum AuthorityError {
    /// Facts, capabilities, or permits belong to another monitor domain.
    MonitorDomainMismatch,
    /// The deterministic reference monitor rejected or deferred the request.
    Monitor(MonitorDecision),
    /// A requested delegated capability would widen its parent.
    Delegation(DelegationError),
}

impl From<MonitorDecision> for AuthorityError {
    fn from(value: MonitorDecision) -> Self {
        Self::Monitor(value)
    }
}

impl From<DelegationError> for AuthorityError {
    fn from(value: DelegationError) -> Self {
        Self::Delegation(value)
    }
}

/// Opaque verified capability fact belonging to exactly one monitor domain.
///
/// This type intentionally has no serde implementation and no public struct
/// fields or constructor. Network/signed capability envelopes are ordinary
/// data outside this type; a trusted adapter converts them only after
/// authentication, signature, revocation, and local-policy verification.
///
/// ```compile_fail
/// use symthaea_cogsec::CapabilityFact;
/// let _ = CapabilityFact {};
/// ```
pub struct CapabilityFact {
    inner: legacy::CapabilityFact,
    seal: Arc<MonitorDomainSeal>,
}

impl std::fmt::Debug for CapabilityFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapabilityFact")
            .field("capability_id", &self.inner.capability_id)
            .field("subject", &self.inner.subject)
            .field("mutation", &self.inner.mutation)
            .field("resource_scope", &self.inner.resource_scope)
            .field("max_consequence", &self.inner.max_consequence)
            .field("authorization_epoch", &self.inner.authorization_epoch)
            .field("revocation_epoch", &self.inner.revocation_epoch)
            .field("valid_from_sequence", &self.inner.valid_from_sequence)
            .field("valid_until_sequence", &self.inner.valid_until_sequence)
            .field("revoked", &self.inner.revoked)
            .finish_non_exhaustive()
    }
}

impl CapabilityFact {
    /// Stable capability identity.
    pub fn capability_id(&self) -> Digest32 {
        self.inner.capability_id
    }

    /// Principal to whom the verified fact applies.
    pub fn subject(&self) -> &PrincipalId {
        &self.inner.subject
    }

    /// Authorized mutation class.
    pub fn mutation(&self) -> MutationKind {
        self.inner.mutation
    }

    /// Authorized resource scope.
    pub fn resource_scope(&self) -> &ResourceScope {
        &self.inner.resource_scope
    }

    /// Maximum consequence authorized by this fact.
    pub fn max_consequence(&self) -> Consequence {
        self.inner.max_consequence
    }

    /// Authorization epoch bound to this fact.
    pub fn authorization_epoch(&self) -> u64 {
        self.inner.authorization_epoch
    }

    /// Revocation epoch bound to this fact.
    pub fn revocation_epoch(&self) -> u64 {
        self.inner.revocation_epoch
    }

    /// First sequence at which this fact is valid.
    pub fn valid_from_sequence(&self) -> u64 {
        self.inner.valid_from_sequence
    }

    /// Optional last sequence at which this fact is valid.
    pub fn valid_until_sequence(&self) -> Option<u64> {
        self.inner.valid_until_sequence
    }

    /// Whether the trusted adapter marked this fact revoked.
    pub fn revoked(&self) -> bool {
        self.inner.revoked
    }
}

/// Opaque snapshot of trusted state, policy, authorization, revocation, and
/// verified capabilities for one monitor domain.
///
/// Unlike [`MutationRequest`] and policy IR, this is not a wire type. Its
/// fields are private and it is not deserializable.
///
/// ```compile_fail
/// use symthaea_cogsec::TrustedFacts;
/// let _ = TrustedFacts {};
/// ```
pub struct TrustedFacts {
    inner: legacy::TrustedFacts,
    seal: Arc<MonitorDomainSeal>,
}

impl std::fmt::Debug for TrustedFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustedFacts")
            .field("resource_state_root", &self.inner.resource_state_root)
            .field("policy_root", &self.inner.policy_root)
            .field("policy_epoch", &self.inner.policy_epoch)
            .field("authorization_epoch", &self.inner.authorization_epoch)
            .field("revocation_epoch", &self.inner.revocation_epoch)
            .field("capability_count", &self.inner.capabilities.len())
            .finish_non_exhaustive()
    }
}

impl TrustedFacts {
    /// Current protected-resource state root.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root
    }

    /// Current trusted policy root.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root
    }

    /// Current policy epoch.
    pub fn policy_epoch(&self) -> u64 {
        self.inner.policy_epoch
    }

    /// Current authorization epoch.
    pub fn authorization_epoch(&self) -> u64 {
        self.inner.authorization_epoch
    }

    /// Current revocation epoch.
    pub fn revocation_epoch(&self) -> u64 {
        self.inner.revocation_epoch
    }

    /// Number of verified capability facts in this snapshot.
    pub fn capability_count(&self) -> usize {
        self.inner.capabilities.len()
    }
}

/// Trusted adapter capability for issuing security facts in one monitor domain.
///
/// Possession of this object is itself privileged. It is deliberately neither
/// cloneable nor serializable and should be held only by the authenticated
/// identity/policy/state adapter owned by the protected runtime boundary.
pub struct TrustedFactAuthority {
    seal: Arc<MonitorDomainSeal>,
}

impl std::fmt::Debug for TrustedFactAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustedFactAuthority")
            .finish_non_exhaustive()
    }
}

impl TrustedFactAuthority {
    /// Convert already-verified adapter results into an opaque capability fact.
    ///
    /// This function does not perform cryptography. Calling it asserts that the
    /// holder has independently verified the identity/signature/delegation and
    /// current revocation status represented by these fields.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_capability(
        &self,
        capability_id: Digest32,
        subject: PrincipalId,
        mutation: MutationKind,
        resource_scope: ResourceScope,
        max_consequence: Consequence,
        authorization_epoch: u64,
        revocation_epoch: u64,
        valid_from_sequence: u64,
        valid_until_sequence: Option<u64>,
        revoked: bool,
    ) -> CapabilityFact {
        CapabilityFact {
            inner: legacy::CapabilityFact {
                capability_id,
                subject,
                mutation,
                resource_scope,
                max_consequence,
                authorization_epoch,
                revocation_epoch,
                valid_from_sequence,
                valid_until_sequence,
                revoked,
            },
            seal: Arc::clone(&self.seal),
        }
    }

    /// Build an opaque trusted-facts snapshot from fresh authoritative state.
    ///
    /// Every capability must have been issued inside this same monitor domain.
    /// Passing a fact from another independently bootstrapped domain fails
    /// closed before the deterministic policy engine sees its contents.
    pub fn snapshot(
        &self,
        resource_state_root: Digest32,
        policy_root: Digest32,
        policy_epoch: u64,
        authorization_epoch: u64,
        revocation_epoch: u64,
        capabilities: &[&CapabilityFact],
    ) -> Result<TrustedFacts, AuthorityError> {
        let mut inner_capabilities = Vec::with_capacity(capabilities.len());
        for capability in capabilities {
            if !Arc::ptr_eq(&self.seal, &capability.seal) {
                return Err(AuthorityError::MonitorDomainMismatch);
            }
            inner_capabilities.push(capability.inner.clone());
        }

        Ok(TrustedFacts {
            inner: legacy::TrustedFacts {
                resource_state_root,
                policy_root,
                policy_epoch,
                authorization_epoch,
                revocation_epoch,
                capabilities: inner_capabilities,
            },
            seal: Arc::clone(&self.seal),
        })
    }

    /// Issue a verified structurally-attenuated child capability.
    ///
    /// The inner algebra checks that scope, consequence, and validity only
    /// narrow. This method is intentionally on the trusted issuer rather than
    /// on [`CapabilityFact`] itself: possession of a verified parent fact does
    /// not by itself confer authority to manufacture another verified subject.
    /// The adapter remains responsible for verifying that the parent actually
    /// authorized the delegation and for recording delegation ancestry.
    #[allow(clippy::too_many_arguments)]
    pub fn derive_capability(
        &self,
        parent: &CapabilityFact,
        capability_id: Digest32,
        subject: PrincipalId,
        resource_scope: ResourceScope,
        max_consequence: Consequence,
        valid_from_sequence: u64,
        valid_until_sequence: Option<u64>,
    ) -> Result<CapabilityFact, AuthorityError> {
        if !Arc::ptr_eq(&self.seal, &parent.seal) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }

        let inner = parent.inner.attenuate(
            capability_id,
            subject,
            resource_scope,
            max_consequence,
            valid_from_sequence,
            valid_until_sequence,
        )?;

        Ok(CapabilityFact {
            inner,
            seal: Arc::clone(&self.seal),
        })
    }
}

/// Authorization-time token bound to one monitor domain.
///
/// This is not commit-ready. It is non-cloneable and non-serializable and must
/// be consumed by [`ReferenceMonitor::precommit`].
pub struct MutationPermit {
    inner: legacy::MutationPermit,
    seal: Arc<MonitorDomainSeal>,
}

impl std::fmt::Debug for MutationPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutationPermit")
            .field("request_id", &self.inner.request_id())
            .field("kind", &self.inner.kind())
            .field("resource", &self.inner.resource())
            .field("mutation_digest", &self.inner.mutation_digest())
            .finish_non_exhaustive()
    }
}

impl MutationPermit {
    /// Request identity bound into this permit.
    pub fn request_id(&self) -> Digest32 {
        self.inner.request_id()
    }

    /// Mutation class bound into this permit.
    pub fn kind(&self) -> MutationKind {
        self.inner.kind()
    }

    /// Principal bound into this permit.
    pub fn subject(&self) -> &PrincipalId {
        self.inner.subject()
    }

    /// Protected resource bound into this permit.
    pub fn resource(&self) -> &ResourceId {
        self.inner.resource()
    }

    /// Exact effect commitment bound into this permit.
    pub fn mutation_digest(&self) -> Digest32 {
        self.inner.mutation_digest()
    }

    /// Consequence class bound into this permit.
    pub fn consequence(&self) -> Consequence {
        self.inner.consequence()
    }

    /// Verified capability identity used to authorize this permit, if required.
    pub fn capability_id(&self) -> Option<Digest32> {
        self.inner.capability_id()
    }

    /// Resource state root observed at authorization.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root()
    }

    /// Policy root observed at authorization.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root()
    }

    /// Policy epoch observed at authorization.
    pub fn policy_epoch(&self) -> u64 {
        self.inner.policy_epoch()
    }

    /// Authorization epoch observed at authorization.
    pub fn authorization_epoch(&self) -> u64 {
        self.inner.authorization_epoch()
    }

    /// Revocation epoch observed at authorization.
    pub fn revocation_epoch(&self) -> u64 {
        self.inner.revocation_epoch()
    }

    /// Logical sequence bound into this permit.
    pub fn sequence(&self) -> u64 {
        self.inner.sequence()
    }
}

/// Commit-ready one-use token bound to exactly one monitor domain.
///
/// Future P0 sinks must both consume this type and verify that it belongs to
/// the protected owner's monitor via [`ReferenceMonitor::accepts_commit_permit`]
/// (or encapsulate that check in the protected owner itself).
pub struct CommitPermit {
    inner: legacy::CommitPermit,
    seal: Arc<MonitorDomainSeal>,
}

impl std::fmt::Debug for CommitPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitPermit")
            .field("request_id", &self.inner.request_id())
            .field("kind", &self.inner.kind())
            .field("resource", &self.inner.resource())
            .field("mutation_digest", &self.inner.mutation_digest())
            .finish_non_exhaustive()
    }
}

impl CommitPermit {
    /// Request identity bound into this permit.
    pub fn request_id(&self) -> Digest32 {
        self.inner.request_id()
    }

    /// Mutation class bound into this permit.
    pub fn kind(&self) -> MutationKind {
        self.inner.kind()
    }

    /// Principal bound into this permit.
    pub fn subject(&self) -> &PrincipalId {
        self.inner.subject()
    }

    /// Protected resource bound into this permit.
    pub fn resource(&self) -> &ResourceId {
        self.inner.resource()
    }

    /// Exact effect commitment bound into this permit.
    pub fn mutation_digest(&self) -> Digest32 {
        self.inner.mutation_digest()
    }

    /// Consequence class bound into this permit.
    pub fn consequence(&self) -> Consequence {
        self.inner.consequence()
    }

    /// Verified capability identity used to authorize this permit, if required.
    pub fn capability_id(&self) -> Option<Digest32> {
        self.inner.capability_id()
    }

    /// Freshly revalidated resource state root.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root()
    }

    /// Freshly revalidated policy root.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root()
    }

    /// Freshly revalidated policy epoch.
    pub fn policy_epoch(&self) -> u64 {
        self.inner.policy_epoch()
    }

    /// Freshly revalidated authorization epoch.
    pub fn authorization_epoch(&self) -> u64 {
        self.inner.authorization_epoch()
    }

    /// Freshly revalidated revocation epoch.
    pub fn revocation_epoch(&self) -> u64 {
        self.inner.revocation_epoch()
    }

    /// Logical sequence bound into this permit.
    pub fn sequence(&self) -> u64 {
        self.inner.sequence()
    }
}

/// Deterministic reference monitor bound to one private in-process authority
/// domain.
///
/// `bootstrap()` creates a fresh domain and returns the two role-separated
/// capabilities required by the protected runtime: the monitor and its trusted
/// fact issuer. The monitor is intentionally not `Copy`, `Clone`, `Default`, or
/// serializable.
pub struct ReferenceMonitor {
    inner: legacy::ReferenceMonitor,
    seal: Arc<MonitorDomainSeal>,
}

impl std::fmt::Debug for ReferenceMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceMonitor").finish_non_exhaustive()
    }
}

impl ReferenceMonitor {
    /// Bootstrap one isolated monitor domain and its trusted fact authority.
    pub fn bootstrap() -> (Self, TrustedFactAuthority) {
        let seal = Arc::new(MonitorDomainSeal);
        (
            Self {
                inner: legacy::ReferenceMonitor,
                seal: Arc::clone(&seal),
            },
            TrustedFactAuthority { seal },
        )
    }

    fn facts_match_domain(&self, facts: &TrustedFacts) -> bool {
        Arc::ptr_eq(&self.seal, &facts.seal)
    }

    /// Evaluate one mutation using only facts issued for this monitor domain.
    pub fn evaluate(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MonitorDecision, AuthorityError> {
        if !self.facts_match_domain(facts) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }
        Ok(self.inner.evaluate(request, &facts.inner, policy))
    }

    /// Evaluate and mint a one-use authorization-time permit in this monitor
    /// domain.
    pub fn authorize(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MutationPermit, AuthorityError> {
        if !self.facts_match_domain(facts) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }

        let inner = self
            .inner
            .authorize(request, &facts.inner, policy)
            .map_err(AuthorityError::Monitor)?;
        Ok(MutationPermit {
            inner,
            seal: Arc::clone(&self.seal),
        })
    }

    /// Revalidate an authorization-time permit against fresh facts and produce
    /// commit typestate only when both objects belong to this same monitor
    /// domain.
    pub fn precommit(
        &self,
        permit: MutationPermit,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<CommitPermit, AuthorityError> {
        if !Arc::ptr_eq(&self.seal, &permit.seal) || !self.facts_match_domain(facts) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }

        let inner = self
            .inner
            .precommit(permit.inner, &facts.inner, policy)
            .map_err(AuthorityError::Monitor)?;
        Ok(CommitPermit {
            inner,
            seal: Arc::clone(&self.seal),
        })
    }

    /// Whether a commit-ready permit belongs to this exact monitor domain.
    ///
    /// Protected P0 owners should perform this check inside the same serialized
    /// state-owner boundary as the mutation itself. A permit from another
    /// monitor instance is rejected regardless of otherwise identical fields.
    pub fn accepts_commit_permit(&self, permit: &CommitPermit) -> bool {
        Arc::ptr_eq(&self.seal, &permit.seal)
    }

    /// Produce a deterministic audit receipt for one monitor decision.
    pub fn receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
        decision: &MonitorDecision,
    ) -> Result<MutationReceipt, AuthorityError> {
        if !self.facts_match_domain(facts) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }
        Ok(self
            .inner
            .receipt(request, &facts.inner, policy, decision))
    }
}

#[cfg(test)]
mod facade_tests {
    use super::*;
    use std::collections::BTreeSet;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn label() -> CognitiveSecurityLabel {
        let mut provenance_roots = BTreeSet::new();
        provenance_roots.insert(d(3));
        CognitiveSecurityLabel {
            control_integrity: ControlIntegrity::Authenticated,
            confidentiality: Confidentiality::Local,
            origin: OriginState::Authenticated,
            artifact_integrity: ArtifactIntegrity::Authenticated,
            taint: TaintLevel::Clean,
            provenance_roots,
        }
    }

    fn request(policy_root: Digest32, state_root: Digest32) -> MutationRequest {
        MutationRequest {
            request_id: d(1),
            kind: MutationKind::GoalActivation,
            subject: PrincipalId("local-user".into()),
            resource: ResourceId("mind/goals".into()),
            mutation_digest: d(2),
            expected_resource_state_root: state_root,
            expected_policy_root: policy_root,
            input_label: label(),
            consequence: Consequence::High,
            sequence: 42,
        }
    }

    fn policy(root: Digest32) -> PolicySnapshot {
        PolicySnapshot {
            root,
            epoch: 7,
            rules: vec![PolicyRule {
                kind: MutationKind::GoalActivation,
                minimum_control_integrity: ControlIntegrity::Authenticated,
                maximum_taint: TaintLevel::Clean,
                capability_required: true,
            }],
        }
    }

    fn capability(authority: &TrustedFactAuthority) -> CapabilityFact {
        authority.issue_capability(
            d(4),
            PrincipalId("local-user".into()),
            MutationKind::GoalActivation,
            ResourceScope::Exact(ResourceId("mind/goals".into())),
            Consequence::High,
            11,
            13,
            40,
            Some(50),
            false,
        )
    }

    #[test]
    fn same_domain_facts_authorize_and_precommit() {
        let (monitor, authority) = ReferenceMonitor::bootstrap();
        let cap = capability(&authority);
        let facts = authority
            .snapshot(d(8), d(9), 7, 11, 13, &[&cap])
            .unwrap();
        let request = request(d(9), d(8));
        let policy = policy(d(9));

        let permit = monitor.authorize(&request, &facts, &policy).unwrap();
        let commit = monitor.precommit(permit, &facts, &policy).unwrap();
        assert!(monitor.accepts_commit_permit(&commit));
        assert_eq!(commit.capability_id(), Some(d(4)));
    }

    #[test]
    fn cross_domain_trusted_facts_are_rejected_before_policy_evaluation() {
        let (monitor_a, _authority_a) = ReferenceMonitor::bootstrap();
        let (_monitor_b, authority_b) = ReferenceMonitor::bootstrap();
        let cap_b = capability(&authority_b);
        let facts_b = authority_b
            .snapshot(d(8), d(9), 7, 11, 13, &[&cap_b])
            .unwrap();

        let error = monitor_a
            .authorize(&request(d(9), d(8)), &facts_b, &policy(d(9)))
            .unwrap_err();
        assert_eq!(error, AuthorityError::MonitorDomainMismatch);
    }

    #[test]
    fn commit_permit_from_monitor_b_is_not_valid_for_monitor_a() {
        let (monitor_a, _authority_a) = ReferenceMonitor::bootstrap();
        let (monitor_b, authority_b) = ReferenceMonitor::bootstrap();
        let cap_b = capability(&authority_b);
        let facts_b = authority_b
            .snapshot(d(8), d(9), 7, 11, 13, &[&cap_b])
            .unwrap();
        let request = request(d(9), d(8));
        let policy = policy(d(9));
        let permit_b = monitor_b.authorize(&request, &facts_b, &policy).unwrap();
        let commit_b = monitor_b.precommit(permit_b, &facts_b, &policy).unwrap();

        assert!(!monitor_a.accepts_commit_permit(&commit_b));
        assert!(monitor_b.accepts_commit_permit(&commit_b));
    }

    #[test]
    fn trusted_fact_authority_rejects_foreign_capability_in_snapshot() {
        let (_monitor_a, authority_a) = ReferenceMonitor::bootstrap();
        let (_monitor_b, authority_b) = ReferenceMonitor::bootstrap();
        let cap_b = capability(&authority_b);

        let error = authority_a
            .snapshot(d(8), d(9), 7, 11, 13, &[&cap_b])
            .unwrap_err();
        assert_eq!(error, AuthorityError::MonitorDomainMismatch);
    }

    #[test]
    fn only_trusted_issuer_can_turn_attenuation_into_verified_child_fact() {
        let (_monitor, authority) = ReferenceMonitor::bootstrap();
        let parent = capability(&authority);
        let child = authority
            .derive_capability(
                &parent,
                d(21),
                PrincipalId("delegate".into()),
                ResourceScope::Exact(ResourceId("mind/goals".into())),
                Consequence::Moderate,
                43,
                Some(48),
            )
            .unwrap();

        assert_eq!(child.mutation(), parent.mutation());
        assert!(child.max_consequence() <= parent.max_consequence());
        assert!(child.valid_from_sequence() >= parent.valid_from_sequence());
        assert!(child.valid_until_sequence() <= parent.valid_until_sequence());
    }

    #[test]
    fn foreign_issuer_cannot_derive_child_from_verified_parent() {
        let (_monitor_a, authority_a) = ReferenceMonitor::bootstrap();
        let (_monitor_b, authority_b) = ReferenceMonitor::bootstrap();
        let parent_a = capability(&authority_a);

        let error = authority_b
            .derive_capability(
                &parent_a,
                d(21),
                PrincipalId("delegate".into()),
                ResourceScope::Exact(ResourceId("mind/goals".into())),
                Consequence::Moderate,
                43,
                Some(48),
            )
            .unwrap_err();
        assert_eq!(error, AuthorityError::MonitorDomainMismatch);
    }
}
