// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public CogSec API boundary.
//!
//! The detailed deterministic implementation is private. This wrapper adds the
//! authority topology needed for safe runtime use: verified transition facts,
//! trusted state snapshots, authorization permits, and commit permits are
//! opaque, non-serializable, and bound to one monitor domain. Caller-controlled
//! request annotations cannot become security facts merely because they have
//! the expected Rust shape.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod facade;

use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub use facade::{
    ArtifactIntegrity, CapabilityFact, CognitiveSecurityLabel, Confidentiality, Consequence,
    ControlIntegrity, DecisionOutcome, DelegationError, Digest32, MonitorDecision, MutationKind,
    MutationRequest, OriginState, PolicyRule, PolicySnapshot, PrincipalId, ReasonCode, ResourceId,
    ResourceScope, TaintLevel,
};

/// Private identity for the outer public monitor boundary.
#[derive(Debug)]
struct PublicMonitorSeal;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TransitionBinding {
    subject: PrincipalId,
    kind: MutationKind,
    resource: ResourceId,
    mutation_digest: Digest32,
    consequence: Consequence,
    input_label: CognitiveSecurityLabel,
    sequence: u64,
}

impl TransitionBinding {
    fn mismatch_with_request(&self, request: &MutationRequest) -> Option<TransitionField> {
        if self.subject != request.subject {
            return Some(TransitionField::Subject);
        }
        if self.kind != request.kind {
            return Some(TransitionField::MutationKind);
        }
        if self.resource != request.resource {
            return Some(TransitionField::Resource);
        }
        if self.mutation_digest != request.mutation_digest {
            return Some(TransitionField::MutationDigest);
        }
        if self.consequence != request.consequence {
            return Some(TransitionField::Consequence);
        }
        if self.input_label != request.input_label {
            return Some(TransitionField::InputSecurityLabel);
        }
        if self.sequence != request.sequence {
            return Some(TransitionField::Sequence);
        }
        None
    }

    fn mismatch_with_binding(&self, other: &Self) -> Option<TransitionField> {
        if self.subject != other.subject {
            return Some(TransitionField::Subject);
        }
        if self.kind != other.kind {
            return Some(TransitionField::MutationKind);
        }
        if self.resource != other.resource {
            return Some(TransitionField::Resource);
        }
        if self.mutation_digest != other.mutation_digest {
            return Some(TransitionField::MutationDigest);
        }
        if self.consequence != other.consequence {
            return Some(TransitionField::Consequence);
        }
        if self.input_label != other.input_label {
            return Some(TransitionField::InputSecurityLabel);
        }
        if self.sequence != other.sequence {
            return Some(TransitionField::Sequence);
        }
        None
    }
}

/// Security-relevant transition field that disagreed with independently
/// verified facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionField {
    /// Authenticated/authorized principal identity.
    Subject,
    /// Canonical mutation class.
    MutationKind,
    /// Protected resource identity.
    Resource,
    /// Exact proposed-effect commitment.
    MutationDigest,
    /// Trusted consequence classification.
    Consequence,
    /// Trusted cognitive security label/provenance.
    InputSecurityLabel,
    /// State-owner logical sequence.
    Sequence,
}

/// Public authority-boundary error.
#[derive(Debug, PartialEq, Eq)]
pub enum AuthorityError {
    /// Facts, transitions, or permits belong to another monitor domain.
    MonitorDomainMismatch,
    /// A caller-controlled request field disagrees with trusted transition facts.
    TransitionBindingMismatch(TransitionField),
    /// The deterministic policy engine rejected/deferred the request.
    Monitor(MonitorDecision),
    /// A requested delegated capability would widen its verified parent.
    Delegation(DelegationError),
}

impl From<facade::AuthorityError> for AuthorityError {
    fn from(value: facade::AuthorityError) -> Self {
        match value {
            facade::AuthorityError::MonitorDomainMismatch => Self::MonitorDomainMismatch,
            facade::AuthorityError::Monitor(decision) => Self::Monitor(decision),
            facade::AuthorityError::Delegation(error) => Self::Delegation(error),
        }
    }
}

/// Opaque trusted classification of one proposed state transition.
///
/// The fields cannot be constructed or deserialized by ordinary application
/// data. A trusted adapter possessing [`TrustedFactAuthority`] creates this only
/// after independently establishing the principal, resource/effect identity,
/// consequence, security label/provenance, and logical sequence.
///
/// ```compile_fail
/// use symthaea_cogsec::VerifiedTransition;
/// let _ = VerifiedTransition {};
/// ```
pub struct VerifiedTransition {
    binding: TransitionBinding,
    seal: Arc<PublicMonitorSeal>,
}

impl std::fmt::Debug for VerifiedTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedTransition")
            .field("subject", &self.binding.subject)
            .field("kind", &self.binding.kind)
            .field("resource", &self.binding.resource)
            .field("mutation_digest", &self.binding.mutation_digest)
            .field("consequence", &self.binding.consequence)
            .field("sequence", &self.binding.sequence)
            .finish_non_exhaustive()
    }
}

impl VerifiedTransition {
    /// Verified principal.
    pub fn subject(&self) -> &PrincipalId {
        &self.binding.subject
    }

    /// Verified mutation class.
    pub fn kind(&self) -> MutationKind {
        self.binding.kind
    }

    /// Verified protected resource.
    pub fn resource(&self) -> &ResourceId {
        &self.binding.resource
    }

    /// Verified effect commitment.
    pub fn mutation_digest(&self) -> Digest32 {
        self.binding.mutation_digest
    }

    /// Verified consequence classification.
    pub fn consequence(&self) -> Consequence {
        self.binding.consequence
    }

    /// Verified input security label/provenance.
    pub fn input_label(&self) -> &CognitiveSecurityLabel {
        &self.binding.input_label
    }

    /// Verified logical sequence.
    pub fn sequence(&self) -> u64 {
        self.binding.sequence
    }
}

/// Trusted adapter capability for one public monitor domain.
///
/// Possession of this object is privileged. It is neither cloneable nor
/// serializable and should remain inside the authenticated state/identity/policy
/// adapter owned by the protected runtime.
pub struct TrustedFactAuthority {
    inner: facade::TrustedFactAuthority,
    seal: Arc<PublicMonitorSeal>,
}

impl std::fmt::Debug for TrustedFactAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustedFactAuthority")
            .finish_non_exhaustive()
    }
}

impl TrustedFactAuthority {
    /// Convert already-verified capability information into an opaque fact.
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
        self.inner.issue_capability(
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
        )
    }

    /// Issue the independently verified transition facts for one proposal.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_transition(
        &self,
        subject: PrincipalId,
        kind: MutationKind,
        resource: ResourceId,
        mutation_digest: Digest32,
        consequence: Consequence,
        input_label: CognitiveSecurityLabel,
        sequence: u64,
    ) -> VerifiedTransition {
        VerifiedTransition {
            binding: TransitionBinding {
                subject,
                kind,
                resource,
                mutation_digest,
                consequence,
                input_label,
                sequence,
            },
            seal: Arc::clone(&self.seal),
        }
    }

    /// Issue a verified structurally attenuated child capability.
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
        self.inner
            .derive_capability(
                parent,
                capability_id,
                subject,
                resource_scope,
                max_consequence,
                valid_from_sequence,
                valid_until_sequence,
            )
            .map_err(AuthorityError::from)
    }

    /// Build an opaque trusted snapshot for exactly one verified transition.
    #[allow(clippy::too_many_arguments)]
    pub fn snapshot(
        &self,
        transition: &VerifiedTransition,
        resource_state_root: Digest32,
        policy_root: Digest32,
        policy_epoch: u64,
        authorization_epoch: u64,
        revocation_epoch: u64,
        capabilities: &[&CapabilityFact],
    ) -> Result<TrustedFacts, AuthorityError> {
        if !Arc::ptr_eq(&self.seal, &transition.seal) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }

        let inner = self
            .inner
            .snapshot(
                resource_state_root,
                policy_root,
                policy_epoch,
                authorization_epoch,
                revocation_epoch,
                capabilities,
            )
            .map_err(AuthorityError::from)?;

        Ok(TrustedFacts {
            inner,
            binding: transition.binding.clone(),
            seal: Arc::clone(&self.seal),
        })
    }
}

/// Opaque trusted state/policy snapshot bound to one verified transition.
pub struct TrustedFacts {
    inner: facade::TrustedFacts,
    binding: TransitionBinding,
    seal: Arc<PublicMonitorSeal>,
}

impl std::fmt::Debug for TrustedFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustedFacts")
            .field("subject", &self.binding.subject)
            .field("kind", &self.binding.kind)
            .field("resource", &self.binding.resource)
            .field("consequence", &self.binding.consequence)
            .field("resource_state_root", &self.inner.resource_state_root())
            .field("policy_root", &self.inner.policy_root())
            .field("policy_epoch", &self.inner.policy_epoch())
            .field("authorization_epoch", &self.inner.authorization_epoch())
            .field("revocation_epoch", &self.inner.revocation_epoch())
            .finish_non_exhaustive()
    }
}

impl TrustedFacts {
    /// Verified principal.
    pub fn subject(&self) -> &PrincipalId {
        &self.binding.subject
    }

    /// Verified mutation class.
    pub fn kind(&self) -> MutationKind {
        self.binding.kind
    }

    /// Verified protected resource.
    pub fn resource(&self) -> &ResourceId {
        &self.binding.resource
    }

    /// Verified mutation commitment.
    pub fn mutation_digest(&self) -> Digest32 {
        self.binding.mutation_digest
    }

    /// Verified consequence classification.
    pub fn consequence(&self) -> Consequence {
        self.binding.consequence
    }

    /// Verified security label/provenance.
    pub fn input_label(&self) -> &CognitiveSecurityLabel {
        &self.binding.input_label
    }

    /// Verified logical sequence.
    pub fn sequence(&self) -> u64 {
        self.binding.sequence
    }

    /// Current protected-resource state root.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root()
    }

    /// Current trusted policy root.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root()
    }

    /// Current policy epoch.
    pub fn policy_epoch(&self) -> u64 {
        self.inner.policy_epoch()
    }

    /// Current authorization epoch.
    pub fn authorization_epoch(&self) -> u64 {
        self.inner.authorization_epoch()
    }

    /// Current revocation epoch.
    pub fn revocation_epoch(&self) -> u64 {
        self.inner.revocation_epoch()
    }

    /// Number of verified capability facts in the snapshot.
    pub fn capability_count(&self) -> usize {
        self.inner.capability_count()
    }
}

/// Authorization-time token for exactly one trusted transition.
pub struct MutationPermit {
    inner: facade::MutationPermit,
    binding: TransitionBinding,
    seal: Arc<PublicMonitorSeal>,
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
    /// Request identity.
    pub fn request_id(&self) -> Digest32 {
        self.inner.request_id()
    }
    /// Trusted mutation class.
    pub fn kind(&self) -> MutationKind {
        self.binding.kind
    }
    /// Trusted principal.
    pub fn subject(&self) -> &PrincipalId {
        &self.binding.subject
    }
    /// Trusted resource.
    pub fn resource(&self) -> &ResourceId {
        &self.binding.resource
    }
    /// Trusted effect commitment.
    pub fn mutation_digest(&self) -> Digest32 {
        self.binding.mutation_digest
    }
    /// Trusted consequence classification.
    pub fn consequence(&self) -> Consequence {
        self.binding.consequence
    }
    /// Capability selected at authorization, if required.
    pub fn capability_id(&self) -> Option<Digest32> {
        self.inner.capability_id()
    }
    /// Resource state root at authorization.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root()
    }
    /// Policy root at authorization.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root()
    }
    /// Policy epoch at authorization.
    pub fn policy_epoch(&self) -> u64 {
        self.inner.policy_epoch()
    }
    /// Authorization epoch at authorization.
    pub fn authorization_epoch(&self) -> u64 {
        self.inner.authorization_epoch()
    }
    /// Revocation epoch at authorization.
    pub fn revocation_epoch(&self) -> u64 {
        self.inner.revocation_epoch()
    }
    /// Trusted logical sequence.
    pub fn sequence(&self) -> u64 {
        self.binding.sequence
    }
}

/// Commit-ready one-use token for exactly one trusted transition.
pub struct CommitPermit {
    inner: facade::CommitPermit,
    binding: TransitionBinding,
    seal: Arc<PublicMonitorSeal>,
}

impl std::fmt::Debug for CommitPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitPermit")
            .field("request_id", &self.inner.request_id())
            .field("kind", &self.binding.kind)
            .field("resource", &self.binding.resource)
            .field("mutation_digest", &self.binding.mutation_digest)
            .finish_non_exhaustive()
    }
}

impl CommitPermit {
    /// Request identity.
    pub fn request_id(&self) -> Digest32 {
        self.inner.request_id()
    }
    /// Trusted mutation class.
    pub fn kind(&self) -> MutationKind {
        self.binding.kind
    }
    /// Trusted principal.
    pub fn subject(&self) -> &PrincipalId {
        &self.binding.subject
    }
    /// Trusted resource.
    pub fn resource(&self) -> &ResourceId {
        &self.binding.resource
    }
    /// Trusted effect commitment.
    pub fn mutation_digest(&self) -> Digest32 {
        self.binding.mutation_digest
    }
    /// Trusted consequence classification.
    pub fn consequence(&self) -> Consequence {
        self.binding.consequence
    }
    /// Capability selected at authorization, if required.
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
    /// Trusted logical sequence.
    pub fn sequence(&self) -> u64 {
        self.binding.sequence
    }
}

/// Stage represented by a durable CogSec evidence record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptStage {
    /// Deterministic policy evaluation against one trusted-facts snapshot.
    Evaluation,
}

/// Serializable evidence for one monitor evaluation.
///
/// This is evidence, not authority. It deliberately carries no private domain
/// seal; exported receipts require an authenticated evidence-plane signature or
/// equivalent provenance envelope when used across trust boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationReceipt {
    /// Evidence stage.
    pub stage: ReceiptStage,
    /// Request identity chosen by the proposer.
    pub request_id: Digest32,
    /// Independently verified principal.
    pub subject: PrincipalId,
    /// Independently verified mutation class.
    pub kind: MutationKind,
    /// Independently verified protected resource.
    pub resource: ResourceId,
    /// Independently verified exact effect commitment.
    pub mutation_digest: Digest32,
    /// Independently verified consequence classification.
    pub consequence: Consequence,
    /// Independently verified security label/provenance.
    pub input_label: CognitiveSecurityLabel,
    /// Resource root expected by the proposer.
    pub expected_resource_state_root: Digest32,
    /// Resource root supplied by the trusted state owner.
    pub observed_resource_state_root: Digest32,
    /// Policy root expected by the proposer.
    pub expected_policy_root: Digest32,
    /// Canonical policy root evaluated by the monitor.
    pub evaluated_policy_root: Digest32,
    /// Trusted policy root supplied independently in the snapshot.
    pub trusted_policy_root: Digest32,
    /// Trusted policy epoch.
    pub policy_epoch: u64,
    /// Trusted authorization epoch.
    pub authorization_epoch: u64,
    /// Trusted revocation epoch.
    pub revocation_epoch: u64,
    /// Independently verified logical sequence.
    pub sequence: u64,
    /// Capability selected for an allowed request, if policy required one.
    pub capability_id: Option<Digest32>,
    /// Monitor outcome.
    pub outcome: DecisionOutcome,
    /// Stable reason codes.
    pub reasons: Vec<ReasonCode>,
}

/// Public deterministic reference monitor for one private authority domain.
///
/// This type is neither cloneable, default-constructible, nor serializable.
#[derive(Debug)]
pub struct ReferenceMonitor {
    inner: facade::ReferenceMonitor,
    seal: Arc<PublicMonitorSeal>,
}

impl ReferenceMonitor {
    /// Bootstrap one isolated monitor domain and its trusted fact issuer.
    pub fn bootstrap() -> (Self, TrustedFactAuthority) {
        let (inner, inner_authority) = facade::ReferenceMonitor::bootstrap();
        let seal = Arc::new(PublicMonitorSeal);
        (
            Self {
                inner,
                seal: Arc::clone(&seal),
            },
            TrustedFactAuthority {
                inner: inner_authority,
                seal,
            },
        )
    }

    fn facts_match_domain(&self, facts: &TrustedFacts) -> bool {
        Arc::ptr_eq(&self.seal, &facts.seal)
    }

    fn validate_request_binding(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
    ) -> Result<(), AuthorityError> {
        if !self.facts_match_domain(facts) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }
        if let Some(field) = facts.binding.mismatch_with_request(request) {
            return Err(AuthorityError::TransitionBindingMismatch(field));
        }
        Ok(())
    }

    fn validate_permit_binding(
        &self,
        permit: &MutationPermit,
        facts: &TrustedFacts,
    ) -> Result<(), AuthorityError> {
        if !Arc::ptr_eq(&self.seal, &permit.seal) || !self.facts_match_domain(facts) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }
        if let Some(field) = permit.binding.mismatch_with_binding(&facts.binding) {
            return Err(AuthorityError::TransitionBindingMismatch(field));
        }
        Ok(())
    }

    /// Evaluate one proposed mutation only after its security-relevant request
    /// fields match independently verified transition facts exactly.
    pub fn evaluate(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MonitorDecision, AuthorityError> {
        self.validate_request_binding(request, facts)?;
        self.inner
            .evaluate(request, &facts.inner, policy)
            .map_err(AuthorityError::from)
    }

    /// Evaluate and mint a one-use authorization token for the verified transition.
    pub fn authorize(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MutationPermit, AuthorityError> {
        self.validate_request_binding(request, facts)?;
        let inner = self
            .inner
            .authorize(request, &facts.inner, policy)
            .map_err(AuthorityError::from)?;
        Ok(MutationPermit {
            inner,
            binding: facts.binding.clone(),
            seal: Arc::clone(&self.seal),
        })
    }

    /// Revalidate a one-use authorization token against fresh same-transition facts.
    pub fn precommit(
        &self,
        permit: MutationPermit,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<CommitPermit, AuthorityError> {
        self.validate_permit_binding(&permit, facts)?;
        let MutationPermit {
            inner,
            binding,
            seal: _,
        } = permit;
        let inner = self
            .inner
            .precommit(inner, &facts.inner, policy)
            .map_err(AuthorityError::from)?;
        Ok(CommitPermit {
            inner,
            binding,
            seal: Arc::clone(&self.seal),
        })
    }

    /// Whether a commit-ready permit belongs to this exact public and private
    /// monitor domain.
    ///
    /// Protected owners must still check the permit's resource, kind, effect
    /// digest, consequence, and current state binding as part of the sink's
    /// serialized commit contract.
    pub fn accepts_commit_permit(&self, permit: &CommitPermit) -> bool {
        Arc::ptr_eq(&self.seal, &permit.seal) && self.inner.accepts_commit_permit(&permit.inner)
    }

    /// Evaluate and produce a receipt whose security fields come from trusted
    /// transition facts rather than caller annotations.
    pub fn evaluate_with_receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<(MonitorDecision, MutationReceipt), AuthorityError> {
        self.validate_request_binding(request, facts)?;
        let decision = self
            .inner
            .evaluate(request, &facts.inner, policy)
            .map_err(AuthorityError::from)?;
        let capability_id = if decision.outcome == DecisionOutcome::Allow {
            self.inner
                .authorize(request, &facts.inner, policy)
                .map_err(AuthorityError::from)?
                .capability_id()
        } else {
            None
        };

        let receipt = MutationReceipt {
            stage: ReceiptStage::Evaluation,
            request_id: request.request_id,
            subject: facts.binding.subject.clone(),
            kind: facts.binding.kind,
            resource: facts.binding.resource.clone(),
            mutation_digest: facts.binding.mutation_digest,
            consequence: facts.binding.consequence,
            input_label: facts.binding.input_label.clone(),
            expected_resource_state_root: request.expected_resource_state_root,
            observed_resource_state_root: facts.resource_state_root(),
            expected_policy_root: request.expected_policy_root,
            evaluated_policy_root: policy.root,
            trusted_policy_root: facts.policy_root(),
            policy_epoch: facts.policy_epoch(),
            authorization_epoch: facts.authorization_epoch(),
            revocation_epoch: facts.revocation_epoch(),
            sequence: facts.binding.sequence,
            capability_id,
            outcome: decision.outcome,
            reasons: decision.reasons.clone(),
        };
        Ok((decision, receipt))
    }

    /// Produce evidence for the monitor's own freshly evaluated decision.
    pub fn receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MutationReceipt, AuthorityError> {
        self.evaluate_with_receipt(request, facts, policy)
            .map(|(_, receipt)| receipt)
    }
}

#[cfg(test)]
mod public_api_tests {
    use super::*;
    use std::collections::BTreeSet;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn label() -> CognitiveSecurityLabel {
        let mut roots = BTreeSet::new();
        roots.insert(d(3));
        CognitiveSecurityLabel {
            control_integrity: ControlIntegrity::Authenticated,
            confidentiality: Confidentiality::Local,
            origin: OriginState::Authenticated,
            artifact_integrity: ArtifactIntegrity::Authenticated,
            taint: TaintLevel::Clean,
            provenance_roots: roots,
        }
    }

    struct Fixture {
        monitor: ReferenceMonitor,
        authority: TrustedFactAuthority,
        capability: CapabilityFact,
        request: MutationRequest,
        policy: PolicySnapshot,
    }

    impl Fixture {
        fn new() -> Self {
            let (monitor, authority) = ReferenceMonitor::bootstrap();
            let subject = PrincipalId("local-user".into());
            let resource = ResourceId("mind/goals".into());
            let capability = authority.issue_capability(
                d(4),
                subject.clone(),
                MutationKind::GoalActivation,
                ResourceScope::Exact(resource.clone()),
                Consequence::High,
                11,
                13,
                40,
                Some(50),
                false,
            );
            let request = MutationRequest {
                request_id: d(1),
                kind: MutationKind::GoalActivation,
                subject,
                resource,
                mutation_digest: d(2),
                expected_resource_state_root: d(8),
                expected_policy_root: d(9),
                input_label: label(),
                consequence: Consequence::High,
                sequence: 42,
            };
            let policy = PolicySnapshot {
                root: d(9),
                epoch: 7,
                rules: vec![PolicyRule {
                    kind: MutationKind::GoalActivation,
                    minimum_control_integrity: ControlIntegrity::Authenticated,
                    maximum_taint: TaintLevel::Clean,
                    capability_required: true,
                }],
            };
            Self {
                monitor,
                authority,
                capability,
                request,
                policy,
            }
        }

        fn transition(&self) -> VerifiedTransition {
            self.authority.issue_transition(
                self.request.subject.clone(),
                self.request.kind,
                self.request.resource.clone(),
                self.request.mutation_digest,
                self.request.consequence,
                self.request.input_label.clone(),
                self.request.sequence,
            )
        }

        fn facts(&self, transition: &VerifiedTransition) -> TrustedFacts {
            self.authority
                .snapshot(transition, d(8), d(9), 7, 11, 13, &[&self.capability])
                .unwrap()
        }
    }

    #[test]
    fn verified_transition_allows_matching_request() {
        let fixture = Fixture::new();
        let transition = fixture.transition();
        let facts = fixture.facts(&transition);
        let permit = fixture
            .monitor
            .authorize(&fixture.request, &facts, &fixture.policy)
            .unwrap();
        assert_eq!(permit.subject(), &fixture.request.subject);
        assert_eq!(permit.consequence(), Consequence::High);
    }

    #[test]
    fn caller_cannot_understate_consequence_after_verification() {
        let mut fixture = Fixture::new();
        let transition = fixture.transition();
        let facts = fixture.facts(&transition);
        fixture.request.consequence = Consequence::Low;
        assert_eq!(
            fixture
                .monitor
                .authorize(&fixture.request, &facts, &fixture.policy)
                .unwrap_err(),
            AuthorityError::TransitionBindingMismatch(TransitionField::Consequence)
        );
    }

    #[test]
    fn caller_cannot_upgrade_integrity_or_clear_taint_after_verification() {
        let mut fixture = Fixture::new();
        let mut verified_label = fixture.request.input_label.clone();
        verified_label.taint = TaintLevel::Tainted;
        let transition = fixture.authority.issue_transition(
            fixture.request.subject.clone(),
            fixture.request.kind,
            fixture.request.resource.clone(),
            fixture.request.mutation_digest,
            fixture.request.consequence,
            verified_label,
            fixture.request.sequence,
        );
        let facts = fixture.facts(&transition);
        fixture.request.input_label.taint = TaintLevel::Clean;
        fixture.request.input_label.control_integrity = ControlIntegrity::PolicyEndorsed;
        assert_eq!(
            fixture
                .monitor
                .evaluate(&fixture.request, &facts, &fixture.policy)
                .unwrap_err(),
            AuthorityError::TransitionBindingMismatch(TransitionField::InputSecurityLabel)
        );
    }

    #[test]
    fn caller_cannot_claim_another_subject_after_verification() {
        let mut fixture = Fixture::new();
        let transition = fixture.transition();
        let facts = fixture.facts(&transition);
        fixture.request.subject = PrincipalId("other-user".into());
        assert_eq!(
            fixture
                .monitor
                .evaluate(&fixture.request, &facts, &fixture.policy)
                .unwrap_err(),
            AuthorityError::TransitionBindingMismatch(TransitionField::Subject)
        );
    }

    #[test]
    fn foreign_transition_cannot_build_local_trusted_snapshot() {
        let fixture_a = Fixture::new();
        let fixture_b = Fixture::new();
        let transition_b = fixture_b.transition();
        assert_eq!(
            fixture_a
                .authority
                .snapshot(
                    &transition_b,
                    d(8),
                    d(9),
                    7,
                    11,
                    13,
                    &[&fixture_a.capability],
                )
                .unwrap_err(),
            AuthorityError::MonitorDomainMismatch
        );
    }

    #[test]
    fn precommit_rejects_changed_verified_transition_context() {
        let fixture = Fixture::new();
        let transition = fixture.transition();
        let facts = fixture.facts(&transition);
        let permit = fixture
            .monitor
            .authorize(&fixture.request, &facts, &fixture.policy)
            .unwrap();

        let changed_transition = fixture.authority.issue_transition(
            fixture.request.subject.clone(),
            fixture.request.kind,
            fixture.request.resource.clone(),
            fixture.request.mutation_digest,
            Consequence::Moderate,
            fixture.request.input_label.clone(),
            fixture.request.sequence,
        );
        let changed_facts = fixture.facts(&changed_transition);
        assert_eq!(
            fixture
                .monitor
                .precommit(permit, &changed_facts, &fixture.policy)
                .unwrap_err(),
            AuthorityError::TransitionBindingMismatch(TransitionField::Consequence)
        );
    }

    #[test]
    fn receipt_uses_verified_transition_security_fields() {
        let fixture = Fixture::new();
        let transition = fixture.transition();
        let facts = fixture.facts(&transition);
        let receipt = fixture
            .monitor
            .receipt(&fixture.request, &facts, &fixture.policy)
            .unwrap();

        assert_eq!(receipt.stage, ReceiptStage::Evaluation);
        assert_eq!(receipt.subject, fixture.request.subject);
        assert_eq!(receipt.consequence, Consequence::High);
        assert_eq!(receipt.input_label, fixture.request.input_label);
        assert_eq!(receipt.capability_id, Some(d(4)));
        assert_eq!(receipt.outcome, DecisionOutcome::Allow);
    }
}
