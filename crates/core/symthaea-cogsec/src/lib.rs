// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public CogSec authority boundary.
//!
//! The lower transition/fact/permit implementation is private. This final
//! public layer also seals the canonical policy IR and distinguishes opaque
//! monitor-origin receipts from serializable evidence records.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod transition_facade;

use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub use transition_facade::{
    ArtifactIntegrity, AuthorityError, CapabilityFact, CognitiveSecurityLabel, Confidentiality,
    Consequence, ControlIntegrity, DecisionOutcome, DelegationError, Digest32, MonitorDecision,
    MutationKind, MutationRequest, OriginState, PolicyRule, PolicySnapshot, PrincipalId,
    ReasonCode, ReceiptStage, ResourceId, ResourceScope, TaintLevel, TransitionField,
    VerifiedTransition,
};

/// Private identity for the final public policy/evidence boundary.
#[derive(Debug)]
struct PolicyDomainSeal;

/// Opaque canonical policy accepted by one monitor domain.
///
/// A raw [`PolicySnapshot`] is ordinary data. It cannot be used for public
/// authorization until a trusted policy compiler/owner possessing
/// [`TrustedFactAuthority`] admits it as `VerifiedPolicy`.
///
/// ```compile_fail
/// use symthaea_cogsec::VerifiedPolicy;
/// let _ = VerifiedPolicy {};
/// ```
pub struct VerifiedPolicy {
    inner: PolicySnapshot,
    seal: Arc<PolicyDomainSeal>,
}

impl std::fmt::Debug for VerifiedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedPolicy")
            .field("root", &self.inner.root)
            .field("epoch", &self.inner.epoch)
            .field("rule_count", &self.inner.rules.len())
            .finish_non_exhaustive()
    }
}

impl VerifiedPolicy {
    /// Commitment to the canonical policy IR.
    pub fn root(&self) -> Digest32 {
        self.inner.root
    }

    /// Monotonic policy epoch.
    pub fn epoch(&self) -> u64 {
        self.inner.epoch
    }

    /// Number of canonical policy rules.
    pub fn rule_count(&self) -> usize {
        self.inner.rules.len()
    }
}

/// Trusted fact/policy authority for one protected monitor domain.
///
/// This object is intentionally non-cloneable and non-serializable. Runtime
/// topology must keep it inside the trusted identity/state/policy adapter.
pub struct TrustedFactAuthority {
    inner: transition_facade::TrustedFactAuthority,
    seal: Arc<PolicyDomainSeal>,
}

impl std::fmt::Debug for TrustedFactAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustedFactAuthority")
            .finish_non_exhaustive()
    }
}

impl TrustedFactAuthority {
    /// Admit an already-validated canonical policy IR into this monitor domain.
    ///
    /// The caller holding this privileged authority asserts that the policy
    /// compiler validated the IR and that `policy.root` actually commits to the
    /// supplied canonical contents. CogSec intentionally does not implement
    /// hashing or text-policy compilation inside the logical TCB.
    pub fn issue_policy(&self, policy: PolicySnapshot) -> VerifiedPolicy {
        VerifiedPolicy {
            inner: policy,
            seal: Arc::clone(&self.seal),
        }
    }

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

    /// Issue independently verified facts for one proposed transition.
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
        self.inner.issue_transition(
            subject,
            kind,
            resource,
            mutation_digest,
            consequence,
            input_label,
            sequence,
        )
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
        self.inner.derive_capability(
            parent,
            capability_id,
            subject,
            resource_scope,
            max_consequence,
            valid_from_sequence,
            valid_until_sequence,
        )
    }

    /// Build a trusted snapshot for exactly one transition and one sealed policy.
    ///
    /// Policy root/epoch are derived from `VerifiedPolicy`; callers cannot pass
    /// those security facts independently of the policy contents.
    #[allow(clippy::too_many_arguments)]
    pub fn snapshot(
        &self,
        transition: &VerifiedTransition,
        resource_state_root: Digest32,
        policy: &VerifiedPolicy,
        authorization_epoch: u64,
        revocation_epoch: u64,
        capabilities: &[&CapabilityFact],
    ) -> Result<TrustedFacts, AuthorityError> {
        if !Arc::ptr_eq(&self.seal, &policy.seal) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }

        let inner = self.inner.snapshot(
            transition,
            resource_state_root,
            policy.inner.root,
            policy.inner.epoch,
            authorization_epoch,
            revocation_epoch,
            capabilities,
        )?;

        Ok(TrustedFacts {
            inner,
            seal: Arc::clone(&self.seal),
        })
    }
}

/// Opaque trusted state/transition snapshot bound to the sealed policy domain.
pub struct TrustedFacts {
    inner: transition_facade::TrustedFacts,
    seal: Arc<PolicyDomainSeal>,
}

impl std::fmt::Debug for TrustedFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustedFacts")
            .field("subject", &self.inner.subject())
            .field("kind", &self.inner.kind())
            .field("resource", &self.inner.resource())
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
        self.inner.subject()
    }

    /// Verified mutation class.
    pub fn kind(&self) -> MutationKind {
        self.inner.kind()
    }

    /// Verified protected resource.
    pub fn resource(&self) -> &ResourceId {
        self.inner.resource()
    }

    /// Verified exact effect commitment.
    pub fn mutation_digest(&self) -> Digest32 {
        self.inner.mutation_digest()
    }

    /// Verified consequence classification.
    pub fn consequence(&self) -> Consequence {
        self.inner.consequence()
    }

    /// Verified security label/provenance.
    pub fn input_label(&self) -> &CognitiveSecurityLabel {
        self.inner.input_label()
    }

    /// Verified logical sequence.
    pub fn sequence(&self) -> u64 {
        self.inner.sequence()
    }

    /// Current protected-resource state root.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root()
    }

    /// Current trusted policy root.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root()
    }

    /// Current trusted policy epoch.
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

    /// Number of verified capabilities in the snapshot.
    pub fn capability_count(&self) -> usize {
        self.inner.capability_count()
    }
}

/// Authorization-time one-use token bound to the sealed policy domain.
pub struct MutationPermit {
    inner: transition_facade::MutationPermit,
    seal: Arc<PolicyDomainSeal>,
}

impl std::fmt::Debug for MutationPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutationPermit")
            .field("request_id", &self.inner.request_id())
            .field("kind", &self.inner.kind())
            .field("resource", &self.inner.resource())
            .field("mutation_digest", &self.inner.mutation_digest())
            .field("policy_root", &self.inner.policy_root())
            .finish_non_exhaustive()
    }
}

impl MutationPermit {
    /// Request identity.
    pub fn request_id(&self) -> Digest32 {
        self.inner.request_id()
    }

    /// Verified mutation class.
    pub fn kind(&self) -> MutationKind {
        self.inner.kind()
    }

    /// Verified principal.
    pub fn subject(&self) -> &PrincipalId {
        self.inner.subject()
    }

    /// Verified resource.
    pub fn resource(&self) -> &ResourceId {
        self.inner.resource()
    }

    /// Verified effect commitment.
    pub fn mutation_digest(&self) -> Digest32 {
        self.inner.mutation_digest()
    }

    /// Verified consequence.
    pub fn consequence(&self) -> Consequence {
        self.inner.consequence()
    }

    /// Capability selected at authorization, if required.
    pub fn capability_id(&self) -> Option<Digest32> {
        self.inner.capability_id()
    }

    /// Resource root at authorization.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root()
    }

    /// Sealed policy root at authorization.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root()
    }

    /// Sealed policy epoch at authorization.
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

    /// Verified logical sequence.
    pub fn sequence(&self) -> u64 {
        self.inner.sequence()
    }
}

/// Commit-ready one-use token bound to the sealed policy domain.
pub struct CommitPermit {
    inner: transition_facade::CommitPermit,
    seal: Arc<PolicyDomainSeal>,
}

impl std::fmt::Debug for CommitPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitPermit")
            .field("request_id", &self.inner.request_id())
            .field("kind", &self.inner.kind())
            .field("resource", &self.inner.resource())
            .field("mutation_digest", &self.inner.mutation_digest())
            .field("policy_root", &self.inner.policy_root())
            .finish_non_exhaustive()
    }
}

impl CommitPermit {
    /// Request identity.
    pub fn request_id(&self) -> Digest32 {
        self.inner.request_id()
    }

    /// Verified mutation class.
    pub fn kind(&self) -> MutationKind {
        self.inner.kind()
    }

    /// Verified principal.
    pub fn subject(&self) -> &PrincipalId {
        self.inner.subject()
    }

    /// Verified resource.
    pub fn resource(&self) -> &ResourceId {
        self.inner.resource()
    }

    /// Verified effect commitment.
    pub fn mutation_digest(&self) -> Digest32 {
        self.inner.mutation_digest()
    }

    /// Verified consequence.
    pub fn consequence(&self) -> Consequence {
        self.inner.consequence()
    }

    /// Capability selected at authorization, if required.
    pub fn capability_id(&self) -> Option<Digest32> {
        self.inner.capability_id()
    }

    /// Freshly revalidated resource root.
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

    /// Verified logical sequence.
    pub fn sequence(&self) -> u64 {
        self.inner.sequence()
    }
}

/// Serializable export representation of a CogSec evaluation receipt.
///
/// This type is deliberately ordinary data and can be constructed or
/// deserialized by untrusted code. It is **not proof of monitor origin** until
/// wrapped in an authenticated evidence-plane envelope/signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationReceiptRecord {
    /// Evidence stage represented by this record.
    pub stage: ReceiptStage,
    /// Request identity.
    pub request_id: Digest32,
    /// Independently verified principal.
    pub subject: PrincipalId,
    /// Independently verified mutation class.
    pub kind: MutationKind,
    /// Independently verified protected resource.
    pub resource: ResourceId,
    /// Independently verified effect commitment.
    pub mutation_digest: Digest32,
    /// Independently verified consequence classification.
    pub consequence: Consequence,
    /// Independently verified security label/provenance.
    pub input_label: CognitiveSecurityLabel,
    /// Resource root expected by the proposer.
    pub expected_resource_state_root: Digest32,
    /// Resource root observed through trusted state facts.
    pub observed_resource_state_root: Digest32,
    /// Policy root expected by the proposer.
    pub expected_policy_root: Digest32,
    /// Root of the sealed policy actually evaluated.
    pub evaluated_policy_root: Digest32,
    /// Trusted policy root carried by the state snapshot.
    pub trusted_policy_root: Digest32,
    /// Policy epoch used for evaluation.
    pub policy_epoch: u64,
    /// Authorization epoch observed during evaluation.
    pub authorization_epoch: u64,
    /// Revocation epoch observed during evaluation.
    pub revocation_epoch: u64,
    /// Independently verified logical sequence.
    pub sequence: u64,
    /// Capability selected for an allowed evaluation, if required.
    pub capability_id: Option<Digest32>,
    /// Monitor outcome.
    pub outcome: DecisionOutcome,
    /// Stable reason codes.
    pub reasons: Vec<ReasonCode>,
}

impl MutationReceiptRecord {
    fn from_inner(inner: transition_facade::MutationReceipt) -> Self {
        Self {
            stage: inner.stage,
            request_id: inner.request_id,
            subject: inner.subject,
            kind: inner.kind,
            resource: inner.resource,
            mutation_digest: inner.mutation_digest,
            consequence: inner.consequence,
            input_label: inner.input_label,
            expected_resource_state_root: inner.expected_resource_state_root,
            observed_resource_state_root: inner.observed_resource_state_root,
            expected_policy_root: inner.expected_policy_root,
            evaluated_policy_root: inner.evaluated_policy_root,
            trusted_policy_root: inner.trusted_policy_root,
            policy_epoch: inner.policy_epoch,
            authorization_epoch: inner.authorization_epoch,
            revocation_epoch: inner.revocation_epoch,
            sequence: inner.sequence,
            capability_id: inner.capability_id,
            outcome: inner.outcome,
            reasons: inner.reasons,
        }
    }
}

/// Opaque same-domain proof that this monitor instance produced an evaluation receipt.
///
/// The type is intentionally non-cloneable and non-serde. Use
/// [`MutationReceipt::export_record`] only when preparing data for an external
/// authenticated evidence envelope.
///
/// ```compile_fail
/// use symthaea_cogsec::MutationReceipt;
/// let _ = MutationReceipt {};
/// ```
pub struct MutationReceipt {
    record: MutationReceiptRecord,
    seal: Arc<PolicyDomainSeal>,
}

impl std::fmt::Debug for MutationReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutationReceipt")
            .field("stage", &self.record.stage)
            .field("request_id", &self.record.request_id)
            .field("kind", &self.record.kind)
            .field("resource", &self.record.resource)
            .field("outcome", &self.record.outcome)
            .finish_non_exhaustive()
    }
}

impl MutationReceipt {
    /// Read the monitor-produced record while retaining opaque origin proof.
    pub fn record(&self) -> &MutationReceiptRecord {
        &self.record
    }

    /// Export a serializable copy for a later authenticated evidence envelope.
    ///
    /// The returned value is ordinary data and must not be treated as proof of
    /// monitor origin by itself.
    pub fn export_record(&self) -> MutationReceiptRecord {
        self.record.clone()
    }
}

/// Public deterministic reference monitor for one sealed policy/authority domain.
#[derive(Debug)]
pub struct ReferenceMonitor {
    inner: transition_facade::ReferenceMonitor,
    seal: Arc<PolicyDomainSeal>,
}

impl ReferenceMonitor {
    /// Bootstrap one protected monitor domain and its trusted authority.
    pub fn bootstrap() -> (Self, TrustedFactAuthority) {
        let (inner, inner_authority) = transition_facade::ReferenceMonitor::bootstrap();
        let seal = Arc::new(PolicyDomainSeal);
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

    fn validate_context(
        &self,
        facts: &TrustedFacts,
        policy: &VerifiedPolicy,
    ) -> Result<(), AuthorityError> {
        if !Arc::ptr_eq(&self.seal, &facts.seal) || !Arc::ptr_eq(&self.seal, &policy.seal) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }
        Ok(())
    }

    /// Evaluate one proposed mutation against the sealed canonical policy.
    pub fn evaluate(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &VerifiedPolicy,
    ) -> Result<MonitorDecision, AuthorityError> {
        self.validate_context(facts, policy)?;
        self.inner.evaluate(request, &facts.inner, &policy.inner)
    }

    /// Authorize one verified transition under the sealed canonical policy.
    pub fn authorize(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &VerifiedPolicy,
    ) -> Result<MutationPermit, AuthorityError> {
        self.validate_context(facts, policy)?;
        let inner = self.inner.authorize(request, &facts.inner, &policy.inner)?;
        Ok(MutationPermit {
            inner,
            seal: Arc::clone(&self.seal),
        })
    }

    /// Revalidate authorization against fresh trusted facts and sealed policy.
    pub fn precommit(
        &self,
        permit: MutationPermit,
        facts: &TrustedFacts,
        policy: &VerifiedPolicy,
    ) -> Result<CommitPermit, AuthorityError> {
        self.validate_context(facts, policy)?;
        if !Arc::ptr_eq(&self.seal, &permit.seal) {
            return Err(AuthorityError::MonitorDomainMismatch);
        }
        let inner = self
            .inner
            .precommit(permit.inner, &facts.inner, &policy.inner)?;
        Ok(CommitPermit {
            inner,
            seal: Arc::clone(&self.seal),
        })
    }

    /// Whether a commit permit belongs to this complete public monitor domain.
    pub fn accepts_commit_permit(&self, permit: &CommitPermit) -> bool {
        Arc::ptr_eq(&self.seal, &permit.seal) && self.inner.accepts_commit_permit(&permit.inner)
    }

    /// Whether an opaque receipt was produced by this exact public monitor domain.
    pub fn accepts_receipt(&self, receipt: &MutationReceipt) -> bool {
        Arc::ptr_eq(&self.seal, &receipt.seal)
    }

    /// Evaluate and emit an opaque receipt for the exact sealed policy used.
    pub fn evaluate_with_receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &VerifiedPolicy,
    ) -> Result<(MonitorDecision, MutationReceipt), AuthorityError> {
        self.validate_context(facts, policy)?;
        let (decision, inner_receipt) =
            self.inner
                .evaluate_with_receipt(request, &facts.inner, &policy.inner)?;
        Ok((
            decision,
            MutationReceipt {
                record: MutationReceiptRecord::from_inner(inner_receipt),
                seal: Arc::clone(&self.seal),
            },
        ))
    }

    /// Emit an opaque evaluation receipt for the exact sealed policy used.
    pub fn receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &VerifiedPolicy,
    ) -> Result<MutationReceipt, AuthorityError> {
        self.evaluate_with_receipt(request, facts, policy)
            .map(|(_, receipt)| receipt)
    }
}

#[cfg(test)]
mod public_policy_and_evidence_tests {
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
        transition: VerifiedTransition,
        request: MutationRequest,
        policy: VerifiedPolicy,
    }

    impl Fixture {
        fn new() -> Self {
            let (monitor, authority) = ReferenceMonitor::bootstrap();
            let subject = PrincipalId("local-user".into());
            let resource = ResourceId("mind/goals".into());
            let request = MutationRequest {
                request_id: d(1),
                kind: MutationKind::GoalActivation,
                subject: subject.clone(),
                resource: resource.clone(),
                mutation_digest: d(2),
                expected_resource_state_root: d(8),
                expected_policy_root: d(9),
                input_label: label(),
                consequence: Consequence::High,
                sequence: 42,
            };
            let transition = authority.issue_transition(
                subject.clone(),
                request.kind,
                resource.clone(),
                request.mutation_digest,
                request.consequence,
                request.input_label.clone(),
                request.sequence,
            );
            let capability = authority.issue_capability(
                d(4),
                subject,
                MutationKind::GoalActivation,
                ResourceScope::Exact(resource),
                Consequence::High,
                11,
                13,
                40,
                Some(50),
                false,
            );
            let policy = authority.issue_policy(PolicySnapshot {
                root: d(9),
                epoch: 7,
                rules: vec![PolicyRule {
                    kind: MutationKind::GoalActivation,
                    minimum_control_integrity: ControlIntegrity::Authenticated,
                    maximum_taint: TaintLevel::Clean,
                    capability_required: true,
                }],
            });
            Self {
                monitor,
                authority,
                capability,
                transition,
                request,
                policy,
            }
        }

        fn facts(&self, state_root: Digest32) -> TrustedFacts {
            self.authority
                .snapshot(
                    &self.transition,
                    state_root,
                    &self.policy,
                    11,
                    13,
                    &[&self.capability],
                )
                .unwrap()
        }
    }

    #[test]
    fn verified_policy_allows_matching_request() {
        let fixture = Fixture::new();
        let facts = fixture.facts(d(8));
        assert!(fixture
            .monitor
            .authorize(&fixture.request, &facts, &fixture.policy)
            .is_ok());
    }

    #[test]
    fn foreign_verified_policy_is_rejected_before_policy_evaluation() {
        let fixture_a = Fixture::new();
        let fixture_b = Fixture::new();
        let facts_a = fixture_a.facts(d(8));
        let result = fixture_a
            .monitor
            .evaluate(&fixture_a.request, &facts_a, &fixture_b.policy);
        assert!(matches!(result, Err(AuthorityError::MonitorDomainMismatch)));
    }

    #[test]
    fn mutating_raw_policy_copy_after_admission_cannot_change_verified_policy() {
        let (monitor, authority) = ReferenceMonitor::bootstrap();
        let subject = PrincipalId("local-user".into());
        let resource = ResourceId("mind/goals".into());
        let request = MutationRequest {
            request_id: d(1),
            kind: MutationKind::GoalActivation,
            subject: subject.clone(),
            resource: resource.clone(),
            mutation_digest: d(2),
            expected_resource_state_root: d(8),
            expected_policy_root: d(9),
            input_label: label(),
            consequence: Consequence::High,
            sequence: 42,
        };
        let transition = authority.issue_transition(
            subject.clone(),
            request.kind,
            resource.clone(),
            request.mutation_digest,
            request.consequence,
            request.input_label.clone(),
            request.sequence,
        );
        let capability = authority.issue_capability(
            d(4),
            subject,
            request.kind,
            ResourceScope::Exact(resource),
            Consequence::High,
            11,
            13,
            40,
            Some(50),
            false,
        );
        let mut raw = PolicySnapshot {
            root: d(9),
            epoch: 7,
            rules: vec![PolicyRule {
                kind: MutationKind::GoalActivation,
                minimum_control_integrity: ControlIntegrity::Authenticated,
                maximum_taint: TaintLevel::Clean,
                capability_required: true,
            }],
        };
        let verified = authority.issue_policy(raw.clone());
        raw.rules[0].minimum_control_integrity = ControlIntegrity::Untrusted;
        raw.rules[0].capability_required = false;

        let facts = authority
            .snapshot(&transition, d(8), &verified, 11, 13, &[&capability])
            .unwrap();
        let permit = monitor.authorize(&request, &facts, &verified).unwrap();
        assert_eq!(permit.capability_id(), Some(d(4)));
        assert_eq!(verified.rule_count(), 1);
    }

    #[test]
    fn trusted_snapshot_policy_identity_is_derived_from_verified_policy() {
        let fixture = Fixture::new();
        let facts = fixture.facts(d(8));
        assert_eq!(facts.policy_root(), fixture.policy.root());
        assert_eq!(facts.policy_epoch(), fixture.policy.epoch());
        let receipt = fixture
            .monitor
            .receipt(&fixture.request, &facts, &fixture.policy)
            .unwrap();
        assert_eq!(
            receipt.record().evaluated_policy_root,
            fixture.policy.root()
        );
        assert_eq!(receipt.record().trusted_policy_root, fixture.policy.root());
        assert_eq!(receipt.record().policy_epoch, fixture.policy.epoch());
    }

    #[test]
    fn receipt_origin_is_domain_bound_but_export_record_is_plain_data() {
        let fixture_a = Fixture::new();
        let fixture_b = Fixture::new();
        let facts_b = fixture_b.facts(d(8));
        let receipt_b = fixture_b
            .monitor
            .receipt(&fixture_b.request, &facts_b, &fixture_b.policy)
            .unwrap();

        assert!(fixture_b.monitor.accepts_receipt(&receipt_b));
        assert!(!fixture_a.monitor.accepts_receipt(&receipt_b));

        let record = receipt_b.export_record();
        let encoded = serde_json::to_string(&record).unwrap();
        let decoded: MutationReceiptRecord = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(decoded.stage, ReceiptStage::Evaluation);
        assert_eq!(decoded.outcome, DecisionOutcome::Allow);
    }
}
