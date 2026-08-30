// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symthaea Cognitive Security Core
//!
//! A deliberately small, deterministic reference-monitor foundation for
//! privileged cognitive state transitions.
//!
//! This crate is intentionally free of LLM, HDC, networking, filesystem,
//! wall-clock, random-number, and cryptographic implementation dependencies.
//! Callers provide already-established security facts; this crate evaluates
//! those facts against canonical policy and can mint a one-use mutation permit.
//!
//! The crate does **not** decide whether a proposition is true. Authentication,
//! epistemic support, confidence, consensus, and authorization are distinct.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Fixed-size commitment used for policy, state, provenance, capability, and
/// mutation identities.
///
/// CogSec does not hash data itself in this core crate. Trusted adapters are
/// responsible for supplying cryptographically appropriate commitments.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Digest32(pub [u8; 32]);

/// Stable principal identity supplied by a trusted identity adapter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PrincipalId(pub String);

/// Stable resource identity for the state being mutated.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceId(pub String);

/// Control-integrity level attached to information.
///
/// This is **not** factual truth. It describes how much privileged influence
/// may safely derive from the information under local policy.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum ControlIntegrity {
    /// Unknown, unauthenticated, or otherwise untrusted influence.
    Untrusted = 0,
    /// Origin is authenticated, but no local endorsement has been granted.
    Authenticated = 1,
    /// A trusted local policy/authority explicitly endorsed the information for
    /// a bounded class of cognitive influence.
    PolicyEndorsed = 2,
}

impl Default for ControlIntegrity {
    fn default() -> Self {
        Self::Untrusted
    }
}

/// Confidentiality classification. Larger values are more restrictive.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum Confidentiality {
    /// May be released publicly.
    Public = 0,
    /// Intended to remain local to the user/device unless explicitly released.
    Local = 1,
    /// Sensitive information requiring an authorized sink.
    Sensitive = 2,
    /// Most restrictive application-defined information class.
    Restricted = 3,
}

impl Default for Confidentiality {
    fn default() -> Self {
        Self::Public
    }
}

/// Strength of the known origin statement for an object.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum OriginState {
    /// No authoritative origin information is available.
    Unknown = 0,
    /// An origin was claimed but not independently authenticated.
    Claimed = 1,
    /// Origin was authenticated by a trusted adapter.
    Authenticated = 2,
}

impl Default for OriginState {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Integrity state of the artifact/bytes carrying the information.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum ArtifactIntegrity {
    /// Artifact integrity has not been checked.
    Unchecked = 0,
    /// Bytes matched an expected digest supplied by a trusted context.
    DigestMatched = 1,
    /// A trusted adapter verified an authenticated integrity statement.
    Authenticated = 2,
}

impl Default for ArtifactIntegrity {
    fn default() -> Self {
        Self::Unchecked
    }
}

/// Monotonic contamination level for security-relevant dependencies.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum TaintLevel {
    /// No known security contamination.
    Clean = 0,
    /// Dependency requires additional review or revalidation.
    Suspect = 1,
    /// Dependency is known to be contaminated for the requested use.
    Tainted = 2,
    /// Dependency was explicitly revoked.
    Revoked = 3,
}

impl Default for TaintLevel {
    fn default() -> Self {
        Self::Clean
    }
}

/// Security metadata that remains attached to cognitively meaningful data.
///
/// Authority is intentionally absent. Authority is represented separately by
/// capability facts and never propagates through ordinary data transformations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveSecurityLabel {
    /// Minimum control integrity of the information lineage.
    pub control_integrity: ControlIntegrity,
    /// Maximum confidentiality restriction of the information lineage.
    pub confidentiality: Confidentiality,
    /// Strength of origin authentication.
    pub origin: OriginState,
    /// Strength of artifact integrity verification.
    pub artifact_integrity: ArtifactIntegrity,
    /// Highest known taint level across dependencies.
    pub taint: TaintLevel,
    /// Security-relevant provenance commitments.
    pub provenance_roots: BTreeSet<Digest32>,
}

impl Default for CognitiveSecurityLabel {
    fn default() -> Self {
        Self {
            control_integrity: ControlIntegrity::Untrusted,
            confidentiality: Confidentiality::Public,
            origin: OriginState::Unknown,
            artifact_integrity: ArtifactIntegrity::Unchecked,
            taint: TaintLevel::Clean,
            provenance_roots: BTreeSet::new(),
        }
    }
}

impl CognitiveSecurityLabel {
    /// Conservative combination for ordinary data transformation.
    ///
    /// - control integrity can only stay equal or decrease;
    /// - confidentiality can only stay equal or become more restrictive;
    /// - origin/artifact-integrity strength can only stay equal or decrease;
    /// - taint can only stay equal or increase;
    /// - provenance commitments are unioned.
    ///
    /// Deliberate endorsement/declassification are separate privileged
    /// operations and therefore are not expressible through this method.
    pub fn combine(&self, other: &Self) -> Self {
        let mut provenance_roots = self.provenance_roots.clone();
        provenance_roots.extend(other.provenance_roots.iter().copied());

        Self {
            control_integrity: self.control_integrity.min(other.control_integrity),
            confidentiality: self.confidentiality.max(other.confidentiality),
            origin: self.origin.min(other.origin),
            artifact_integrity: self.artifact_integrity.min(other.artifact_integrity),
            taint: self.taint.max(other.taint),
            provenance_roots,
        }
    }
}

/// Security-relevant cognitive or external state-transition class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MutationKind {
    /// Receive/inspect information without granting it persistent or active influence.
    Observe,
    /// Change attention, priority, or salience.
    Attention,
    /// Change affective or neuromodulatory state.
    Affect,
    /// Admit an item into working memory.
    WorkingMemoryAdmission,
    /// Commit information to persistent episodic/conversation memory.
    PersistentMemoryCommit,
    /// Promote information into semantic/operational knowledge.
    SemanticPromotion,
    /// Promote a model, gradient, LoRA delta, or learned adaptation.
    LearningPromotion,
    /// Activate or materially modify an active goal.
    GoalActivation,
    /// Modify source/identity trust policy.
    TrustPolicyChange,
    /// Modify CogSec or another security policy.
    SecurityPolicyChange,
    /// Invoke a tool or capability-bearing internal service.
    ToolInvocation,
    /// Cause a consequential effect outside the cognitive process.
    ExternalAction,
    /// Release information to a less restrictive confidentiality domain.
    Declassify,
}

/// Coarse consequence class used by deterministic policy.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum Consequence {
    /// Negligible lasting consequence.
    Low = 0,
    /// Bounded/reversible consequence.
    Moderate = 1,
    /// Significant persistent, financial, privacy, or operational consequence.
    High = 2,
    /// Safety-, sovereignty-, or infrastructure-critical consequence.
    Critical = 3,
}

/// Exact or global resource scope for a capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceScope {
    /// Capability is bound to exactly one resource.
    Exact(ResourceId),
    /// Capability applies to any resource accepted by the surrounding policy.
    /// High-assurance profiles should prefer exact scopes.
    Any,
}

impl ResourceScope {
    fn contains(&self, resource: &ResourceId) -> bool {
        match self {
            Self::Exact(expected) => expected == resource,
            Self::Any => true,
        }
    }

    fn is_subset_of(&self, parent: &Self) -> bool {
        match (self, parent) {
            (Self::Exact(child), Self::Exact(parent)) => child == parent,
            (Self::Exact(_), Self::Any) | (Self::Any, Self::Any) => true,
            (Self::Any, Self::Exact(_)) => false,
        }
    }
}

/// Reason a requested capability delegation would widen its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DelegationError {
    /// A revoked capability cannot delegate further authority.
    ParentRevoked,
    /// The requested resource scope is broader than the parent scope.
    ResourceScopeWidened,
    /// The requested consequence ceiling is higher than the parent ceiling.
    ConsequenceWidened,
    /// The requested sequence-validity window extends outside the parent window.
    ValidityWidened,
}

/// Capability statement after identity/signature verification by a trusted
/// adapter.
///
/// This type is a security *fact*, not a live permit. The reference monitor
/// revalidates scope, epochs, sequence bounds, consequence, and revocation
/// before minting a one-use permit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityFact {
    /// Stable identity of the capability/delegation.
    pub capability_id: Digest32,
    /// Principal allowed to exercise the capability.
    pub subject: PrincipalId,
    /// Mutation class authorized by the capability.
    pub mutation: MutationKind,
    /// Resource scope authorized by the capability.
    pub resource_scope: ResourceScope,
    /// Highest consequence class authorized by the capability.
    pub max_consequence: Consequence,
    /// Authorization epoch at issuance/verification.
    pub authorization_epoch: u64,
    /// Revocation epoch at issuance/verification.
    pub revocation_epoch: u64,
    /// First logical sequence at which this fact is valid.
    pub valid_from_sequence: u64,
    /// Optional last logical sequence at which this fact is valid.
    pub valid_until_sequence: Option<u64>,
    /// Trusted adapter's current revocation result.
    pub revoked: bool,
}

impl CapabilityFact {
    fn valid_for(&self, request: &MutationRequest, facts: &TrustedFacts) -> bool {
        if self.revoked
            || self.subject != request.subject
            || self.mutation != request.kind
            || !self.resource_scope.contains(&request.resource)
            || self.max_consequence < request.consequence
            || self.authorization_epoch != facts.authorization_epoch
            || self.revocation_epoch != facts.revocation_epoch
            || request.sequence < self.valid_from_sequence
        {
            return false;
        }

        if let Some(until) = self.valid_until_sequence {
            if request.sequence > until {
                return false;
            }
        }

        true
    }

    /// Derive a structurally narrower child capability.
    ///
    /// This validates attenuation only. The surrounding authorization adapter
    /// remains responsible for proving that the parent principal actually
    /// authorized the delegation and for binding the child to its parent in
    /// the signed/auditable capability chain.
    pub fn attenuate(
        &self,
        capability_id: Digest32,
        subject: PrincipalId,
        resource_scope: ResourceScope,
        max_consequence: Consequence,
        valid_from_sequence: u64,
        valid_until_sequence: Option<u64>,
    ) -> Result<Self, DelegationError> {
        if self.revoked {
            return Err(DelegationError::ParentRevoked);
        }
        if !resource_scope.is_subset_of(&self.resource_scope) {
            return Err(DelegationError::ResourceScopeWidened);
        }
        if max_consequence > self.max_consequence {
            return Err(DelegationError::ConsequenceWidened);
        }
        if valid_from_sequence < self.valid_from_sequence {
            return Err(DelegationError::ValidityWidened);
        }

        match (self.valid_until_sequence, valid_until_sequence) {
            (Some(parent_until), Some(child_until))
                if child_until <= parent_until && child_until >= valid_from_sequence => {}
            (Some(_), _) => return Err(DelegationError::ValidityWidened),
            (None, Some(child_until)) if child_until >= valid_from_sequence => {}
            (None, Some(_)) => return Err(DelegationError::ValidityWidened),
            (None, None) => {}
        }

        Ok(Self {
            capability_id,
            subject,
            mutation: self.mutation,
            resource_scope,
            max_consequence,
            authorization_epoch: self.authorization_epoch,
            revocation_epoch: self.revocation_epoch,
            valid_from_sequence,
            valid_until_sequence,
            revoked: false,
        })
    }
}

/// Canonical request describing one proposed cognitive mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationRequest {
    /// Caller-chosen stable request identity/commitment.
    pub request_id: Digest32,
    /// Mutation class.
    pub kind: MutationKind,
    /// Principal on whose authority the mutation is requested.
    pub subject: PrincipalId,
    /// Protected resource to mutate.
    pub resource: ResourceId,
    /// Commitment to the exact intended effect.
    pub mutation_digest: Digest32,
    /// Resource-state root the caller observed when preparing the request.
    pub expected_resource_state_root: Digest32,
    /// Policy root the caller observed when preparing the request.
    pub expected_policy_root: Digest32,
    /// Security label of the information driving the proposed mutation.
    pub input_label: CognitiveSecurityLabel,
    /// Consequence class of the requested effect.
    pub consequence: Consequence,
    /// Monotonic logical sequence supplied by the state owner.
    pub sequence: u64,
}

/// Trusted security facts supplied independently of the untrusted request
/// payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedFacts {
    /// Current protected-resource state root.
    pub resource_state_root: Digest32,
    /// Current canonical policy root.
    pub policy_root: Digest32,
    /// Current policy epoch.
    pub policy_epoch: u64,
    /// Current authorization epoch.
    pub authorization_epoch: u64,
    /// Current security-relevant revocation epoch. Any revocation capable of
    /// invalidating an outstanding permit must advance this epoch.
    pub revocation_epoch: u64,
    /// Verified capability facts available to the requesting subject.
    pub capabilities: Vec<CapabilityFact>,
}

/// One canonical policy rule for a mutation class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Mutation class governed by this rule.
    pub kind: MutationKind,
    /// Minimum control integrity needed for direct influence on this mutation.
    pub minimum_control_integrity: ControlIntegrity,
    /// Highest taint level accepted without quarantine/revalidation.
    pub maximum_taint: TaintLevel,
    /// Whether a valid capability is required.
    pub capability_required: bool,
}

/// Canonical deterministic policy snapshot consumed by the reference monitor.
///
/// Parsing human-readable policy belongs outside the TCB. The surrounding
/// system should validate/compile policy into this IR and provide its root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicySnapshot {
    /// Commitment to the exact policy IR.
    pub root: Digest32,
    /// Monotonic policy epoch.
    pub epoch: u64,
    /// Rules. Duplicate entries are invalid and fail closed at evaluation time.
    pub rules: Vec<PolicyRule>,
}

impl PolicySnapshot {
    fn rule_for(&self, kind: MutationKind) -> Option<PolicyRule> {
        let mut matches = self.rules.iter().copied().filter(|r| r.kind == kind);
        let first = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        Some(first)
    }
}

/// Deterministic monitor outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionOutcome {
    /// Request is permitted and may be converted into a one-use permit.
    Allow,
    /// Request is forbidden by policy or malformed security context.
    Deny,
    /// Content may be retained for analysis but must not influence the protected sink.
    Quarantine,
    /// A valid capability/authorization is missing.
    RequireAuthorization,
    /// State, policy, or authorization context changed and must be re-evaluated.
    RequireRevalidation,
    /// Trusted facts are temporarily insufficient to make a safe decision.
    Defer,
}

/// Stable machine-readable reason code for a monitor decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReasonCode {
    /// Request's policy commitment differs from the current policy.
    PolicyRootMismatch,
    /// Trusted policy facts disagree with the supplied policy snapshot.
    PolicyFactsMismatch,
    /// Policy epoch changed.
    PolicyEpochMismatch,
    /// Authorization context changed after evaluation/permit issuance.
    AuthorizationEpochMismatch,
    /// Revocation context changed after evaluation/permit issuance.
    RevocationEpochMismatch,
    /// Resource changed after request preparation or permit issuance.
    ResourceStateMismatch,
    /// No unique rule exists for the mutation class.
    MissingOrDuplicatePolicyRule,
    /// Input control integrity is below the policy requirement.
    InsufficientControlIntegrity,
    /// Input carries too much taint for the requested mutation.
    TaintedDependency,
    /// Policy requires a capability and no valid matching capability exists.
    MissingCapability,
    /// At least one apparently relevant capability is revoked.
    RevokedCapability,
    /// At least one apparently relevant capability is stale for current epochs/sequence.
    StaleCapability,
    /// Capability does not cover the requested resource/effect class.
    ScopeMismatch,
}

/// Complete deterministic decision returned by the reference monitor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonitorDecision {
    /// High-level outcome.
    pub outcome: DecisionOutcome,
    /// Stable, sorted reason codes supporting the outcome.
    pub reasons: Vec<ReasonCode>,
}

impl MonitorDecision {
    fn single(outcome: DecisionOutcome, reason: ReasonCode) -> Self {
        Self {
            outcome,
            reasons: vec![reason],
        }
    }
}

/// Authorization-time token for exactly one already-evaluated mutation.
///
/// This is deliberately **not** commit-ready. A protected sink should accept
/// only [`CommitPermit`], which can be minted only by
/// [`ReferenceMonitor::precommit`] from a still-valid `MutationPermit`.
///
/// The fields are private, the type is not `Clone`, and it is deliberately not
/// serializable or default-constructible.
///
/// ```compile_fail
/// use symthaea_cogsec::MutationPermit;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<MutationPermit>();
/// ```
///
/// ```compile_fail
/// use symthaea_cogsec::MutationPermit;
/// fn requires_default<T: Default>() {}
/// requires_default::<MutationPermit>();
/// ```
///
/// ```compile_fail
/// use serde::de::DeserializeOwned;
/// use symthaea_cogsec::MutationPermit;
/// fn requires_deserialize<T: DeserializeOwned>() {}
/// requires_deserialize::<MutationPermit>();
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct MutationPermit {
    request_id: Digest32,
    kind: MutationKind,
    subject: PrincipalId,
    resource: ResourceId,
    mutation_digest: Digest32,
    consequence: Consequence,
    capability_id: Option<Digest32>,
    resource_state_root: Digest32,
    policy_root: Digest32,
    policy_epoch: u64,
    authorization_epoch: u64,
    revocation_epoch: u64,
    sequence: u64,
}

impl MutationPermit {
    /// Request identity bound into this permit.
    pub fn request_id(&self) -> Digest32 {
        self.request_id
    }

    /// Mutation class bound into this permit.
    pub fn kind(&self) -> MutationKind {
        self.kind
    }

    /// Principal bound into this permit.
    pub fn subject(&self) -> &PrincipalId {
        &self.subject
    }

    /// Resource bound into this permit.
    pub fn resource(&self) -> &ResourceId {
        &self.resource
    }

    /// Exact mutation commitment bound into this permit.
    pub fn mutation_digest(&self) -> Digest32 {
        self.mutation_digest
    }

    /// Consequence class bound into this permit.
    pub fn consequence(&self) -> Consequence {
        self.consequence
    }

    /// Capability identity used to authorize the mutation, if policy required one.
    pub fn capability_id(&self) -> Option<Digest32> {
        self.capability_id
    }

    /// Resource state that must still be current at commit time.
    pub fn resource_state_root(&self) -> Digest32 {
        self.resource_state_root
    }

    /// Policy root used to issue this permit.
    pub fn policy_root(&self) -> Digest32 {
        self.policy_root
    }

    /// Policy epoch used to issue this permit.
    pub fn policy_epoch(&self) -> u64 {
        self.policy_epoch
    }

    /// Authorization epoch used to issue this permit.
    pub fn authorization_epoch(&self) -> u64 {
        self.authorization_epoch
    }

    /// Revocation epoch used to issue this permit.
    pub fn revocation_epoch(&self) -> u64 {
        self.revocation_epoch
    }

    /// Logical sequence bound into this permit.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

/// Commit-ready one-use token produced only after fresh-state revalidation.
///
/// Protected P0 mutation sinks should require this type rather than
/// [`MutationPermit`]. That makes the precommit boundary part of the Rust API
/// topology instead of a convention that callers can accidentally skip.
///
/// ```compile_fail
/// use symthaea_cogsec::CommitPermit;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<CommitPermit>();
/// ```
///
/// ```compile_fail
/// use symthaea_cogsec::CommitPermit;
/// fn requires_default<T: Default>() {}
/// requires_default::<CommitPermit>();
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct CommitPermit {
    inner: MutationPermit,
}

impl CommitPermit {
    /// Request identity bound into this commit permit.
    pub fn request_id(&self) -> Digest32 {
        self.inner.request_id()
    }

    /// Mutation class bound into this commit permit.
    pub fn kind(&self) -> MutationKind {
        self.inner.kind()
    }

    /// Principal bound into this commit permit.
    pub fn subject(&self) -> &PrincipalId {
        self.inner.subject()
    }

    /// Resource bound into this commit permit.
    pub fn resource(&self) -> &ResourceId {
        self.inner.resource()
    }

    /// Exact mutation commitment bound into this commit permit.
    pub fn mutation_digest(&self) -> Digest32 {
        self.inner.mutation_digest()
    }

    /// Consequence class bound into this commit permit.
    pub fn consequence(&self) -> Consequence {
        self.inner.consequence()
    }

    /// Capability identity used to authorize the mutation, if required.
    pub fn capability_id(&self) -> Option<Digest32> {
        self.inner.capability_id()
    }

    /// Resource-state root freshly revalidated immediately before commit.
    pub fn resource_state_root(&self) -> Digest32 {
        self.inner.resource_state_root()
    }

    /// Policy root freshly revalidated immediately before commit.
    pub fn policy_root(&self) -> Digest32 {
        self.inner.policy_root()
    }

    /// Policy epoch freshly revalidated immediately before commit.
    pub fn policy_epoch(&self) -> u64 {
        self.inner.policy_epoch()
    }

    /// Authorization epoch freshly revalidated immediately before commit.
    pub fn authorization_epoch(&self) -> u64 {
        self.inner.authorization_epoch()
    }

    /// Revocation epoch freshly revalidated immediately before commit.
    pub fn revocation_epoch(&self) -> u64 {
        self.inner.revocation_epoch()
    }

    /// Logical sequence bound into this commit permit.
    pub fn sequence(&self) -> u64 {
        self.inner.sequence()
    }
}

/// Audit receipt for a monitor evaluation or later state commit.
///
/// This initial core receipt contains only commitments and reason codes. Runtime
/// integrations can wrap it with signatures/timestamps without moving those
/// dependencies into the logical kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationReceipt {
    /// Request identity.
    pub request_id: Digest32,
    /// Mutation class.
    pub kind: MutationKind,
    /// Protected resource.
    pub resource: ResourceId,
    /// Exact proposed/committed effect commitment.
    pub mutation_digest: Digest32,
    /// Resource-state root observed by the monitor.
    pub resource_state_root: Digest32,
    /// Policy root used by the monitor.
    pub policy_root: Digest32,
    /// Policy epoch used by the monitor.
    pub policy_epoch: u64,
    /// Authorization epoch used by the monitor.
    pub authorization_epoch: u64,
    /// Revocation epoch used by the monitor.
    pub revocation_epoch: u64,
    /// Logical sequence.
    pub sequence: u64,
    /// Monitor outcome.
    pub outcome: DecisionOutcome,
    /// Stable reason codes.
    pub reasons: Vec<ReasonCode>,
}

/// Deterministic cognitive-security reference monitor.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceMonitor;

impl ReferenceMonitor {
    /// Evaluate a proposed mutation without minting a permit.
    pub fn evaluate(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> MonitorDecision {
        if request.expected_policy_root != policy.root {
            return MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::PolicyRootMismatch,
            );
        }

        if facts.policy_root != policy.root {
            return MonitorDecision::single(
                DecisionOutcome::Defer,
                ReasonCode::PolicyFactsMismatch,
            );
        }

        if facts.policy_epoch != policy.epoch {
            return MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::PolicyEpochMismatch,
            );
        }

        if request.expected_resource_state_root != facts.resource_state_root {
            return MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::ResourceStateMismatch,
            );
        }

        let Some(rule) = policy.rule_for(request.kind) else {
            return MonitorDecision::single(
                DecisionOutcome::Deny,
                ReasonCode::MissingOrDuplicatePolicyRule,
            );
        };

        if request.input_label.taint > rule.maximum_taint {
            return MonitorDecision::single(
                DecisionOutcome::Quarantine,
                ReasonCode::TaintedDependency,
            );
        }

        if request.input_label.control_integrity < rule.minimum_control_integrity {
            return MonitorDecision::single(
                DecisionOutcome::Quarantine,
                ReasonCode::InsufficientControlIntegrity,
            );
        }

        if !rule.capability_required {
            return MonitorDecision {
                outcome: DecisionOutcome::Allow,
                reasons: Vec::new(),
            };
        }

        if facts
            .capabilities
            .iter()
            .any(|cap| cap.valid_for(request, facts))
        {
            return MonitorDecision {
                outcome: DecisionOutcome::Allow,
                reasons: Vec::new(),
            };
        }

        let mut reasons = BTreeSet::new();
        let mut saw_relevant = false;
        for cap in &facts.capabilities {
            if cap.subject != request.subject || cap.mutation != request.kind {
                continue;
            }
            saw_relevant = true;
            if cap.revoked {
                reasons.insert(ReasonCode::RevokedCapability);
            }
            if cap.authorization_epoch != facts.authorization_epoch
                || cap.revocation_epoch != facts.revocation_epoch
                || request.sequence < cap.valid_from_sequence
                || cap
                    .valid_until_sequence
                    .is_some_and(|until| request.sequence > until)
            {
                reasons.insert(ReasonCode::StaleCapability);
            }
            if !cap.resource_scope.contains(&request.resource)
                || cap.max_consequence < request.consequence
            {
                reasons.insert(ReasonCode::ScopeMismatch);
            }
        }

        if !saw_relevant || reasons.is_empty() {
            reasons.insert(ReasonCode::MissingCapability);
        }

        MonitorDecision {
            outcome: DecisionOutcome::RequireAuthorization,
            reasons: reasons.into_iter().collect(),
        }
    }

    /// Evaluate and, only on `Allow`, mint an authorization-time mutation permit.
    pub fn authorize(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MutationPermit, MonitorDecision> {
        let decision = self.evaluate(request, facts, policy);
        if decision.outcome != DecisionOutcome::Allow {
            return Err(decision);
        }

        let Some(rule) = policy.rule_for(request.kind) else {
            return Err(MonitorDecision::single(
                DecisionOutcome::Deny,
                ReasonCode::MissingOrDuplicatePolicyRule,
            ));
        };

        let capability_id = if rule.capability_required {
            match facts
                .capabilities
                .iter()
                .find(|cap| cap.valid_for(request, facts))
            {
                Some(cap) => Some(cap.capability_id),
                None => {
                    return Err(MonitorDecision::single(
                        DecisionOutcome::RequireAuthorization,
                        ReasonCode::MissingCapability,
                    ));
                }
            }
        } else {
            None
        };

        Ok(MutationPermit {
            request_id: request.request_id,
            kind: request.kind,
            subject: request.subject.clone(),
            resource: request.resource.clone(),
            mutation_digest: request.mutation_digest,
            consequence: request.consequence,
            capability_id,
            resource_state_root: facts.resource_state_root,
            policy_root: policy.root,
            policy_epoch: policy.epoch,
            authorization_epoch: facts.authorization_epoch,
            revocation_epoch: facts.revocation_epoch,
            sequence: request.sequence,
        })
    }

    /// Revalidate an authorization-time permit immediately before the protected
    /// state owner consumes it, returning a distinct commit-ready typestate.
    ///
    /// The authorization permit is taken by value. Any failed precommit destroys
    /// that token; callers must prepare and authorize a new request against the
    /// fresh state rather than retrying stale authority. The protected state
    /// owner should perform this check and the subsequent commit under the same
    /// serialization/transaction boundary.
    pub fn precommit(
        &self,
        permit: MutationPermit,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<CommitPermit, MonitorDecision> {
        if facts.policy_root != policy.root {
            return Err(MonitorDecision::single(
                DecisionOutcome::Defer,
                ReasonCode::PolicyFactsMismatch,
            ));
        }
        if permit.policy_root != policy.root {
            return Err(MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::PolicyRootMismatch,
            ));
        }
        if permit.policy_epoch != policy.epoch || facts.policy_epoch != policy.epoch {
            return Err(MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::PolicyEpochMismatch,
            ));
        }
        if permit.resource_state_root != facts.resource_state_root {
            return Err(MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::ResourceStateMismatch,
            ));
        }
        if permit.authorization_epoch != facts.authorization_epoch {
            return Err(MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::AuthorizationEpochMismatch,
            ));
        }
        if permit.revocation_epoch != facts.revocation_epoch {
            return Err(MonitorDecision::single(
                DecisionOutcome::RequireRevalidation,
                ReasonCode::RevocationEpochMismatch,
            ));
        }

        if let Some(capability_id) = permit.capability_id {
            let Some(capability) = facts
                .capabilities
                .iter()
                .find(|cap| cap.capability_id == capability_id)
            else {
                return Err(MonitorDecision::single(
                    DecisionOutcome::RequireAuthorization,
                    ReasonCode::MissingCapability,
                ));
            };

            if capability.revoked {
                return Err(MonitorDecision::single(
                    DecisionOutcome::RequireAuthorization,
                    ReasonCode::RevokedCapability,
                ));
            }
            if capability.subject != permit.subject
                || capability.mutation != permit.kind
                || !capability.resource_scope.contains(&permit.resource)
                || capability.max_consequence < permit.consequence
            {
                return Err(MonitorDecision::single(
                    DecisionOutcome::RequireAuthorization,
                    ReasonCode::ScopeMismatch,
                ));
            }
            if capability.authorization_epoch != facts.authorization_epoch
                || capability.revocation_epoch != facts.revocation_epoch
                || permit.sequence < capability.valid_from_sequence
                || capability
                    .valid_until_sequence
                    .is_some_and(|until| permit.sequence > until)
            {
                return Err(MonitorDecision::single(
                    DecisionOutcome::RequireAuthorization,
                    ReasonCode::StaleCapability,
                ));
            }
        }

        Ok(CommitPermit { inner: permit })
    }

    /// Produce an audit receipt for a monitor decision.
    pub fn receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
        decision: &MonitorDecision,
    ) -> MutationReceipt {
        MutationReceipt {
            request_id: request.request_id,
            kind: request.kind,
            resource: request.resource.clone(),
            mutation_digest: request.mutation_digest,
            resource_state_root: facts.resource_state_root,
            policy_root: policy.root,
            policy_epoch: policy.epoch,
            authorization_epoch: facts.authorization_epoch,
            revocation_epoch: facts.revocation_epoch,
            sequence: request.sequence,
            outcome: decision.outcome,
            reasons: decision.reasons.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn label(
        integrity: ControlIntegrity,
        confidentiality: Confidentiality,
        taint: TaintLevel,
        provenance: u8,
    ) -> CognitiveSecurityLabel {
        let mut roots = BTreeSet::new();
        roots.insert(d(provenance));
        CognitiveSecurityLabel {
            control_integrity: integrity,
            confidentiality,
            origin: OriginState::Authenticated,
            artifact_integrity: ArtifactIntegrity::Authenticated,
            taint,
            provenance_roots: roots,
        }
    }

    fn goal_fixture() -> (MutationRequest, TrustedFacts, PolicySnapshot) {
        let subject = PrincipalId("local-user".into());
        let resource = ResourceId("mind/goals".into());
        let policy_root = d(9);
        let state_root = d(8);
        let request = MutationRequest {
            request_id: d(1),
            kind: MutationKind::GoalActivation,
            subject: subject.clone(),
            resource: resource.clone(),
            mutation_digest: d(2),
            expected_resource_state_root: state_root,
            expected_policy_root: policy_root,
            input_label: label(
                ControlIntegrity::Authenticated,
                Confidentiality::Local,
                TaintLevel::Clean,
                3,
            ),
            consequence: Consequence::High,
            sequence: 42,
        };
        let facts = TrustedFacts {
            resource_state_root: state_root,
            policy_root,
            policy_epoch: 7,
            authorization_epoch: 11,
            revocation_epoch: 13,
            capabilities: vec![CapabilityFact {
                capability_id: d(4),
                subject,
                mutation: MutationKind::GoalActivation,
                resource_scope: ResourceScope::Exact(resource),
                max_consequence: Consequence::High,
                authorization_epoch: 11,
                revocation_epoch: 13,
                valid_from_sequence: 40,
                valid_until_sequence: Some(50),
                revoked: false,
            }],
        };
        let policy = PolicySnapshot {
            root: policy_root,
            epoch: 7,
            rules: vec![PolicyRule {
                kind: MutationKind::GoalActivation,
                minimum_control_integrity: ControlIntegrity::Authenticated,
                maximum_taint: TaintLevel::Clean,
                capability_required: true,
            }],
        };
        (request, facts, policy)
    }

    #[test]
    fn default_label_is_low_privilege() {
        let label = CognitiveSecurityLabel::default();
        assert_eq!(label.control_integrity, ControlIntegrity::Untrusted);
        assert_eq!(label.origin, OriginState::Unknown);
        assert_eq!(label.artifact_integrity, ArtifactIntegrity::Unchecked);
    }

    #[test]
    fn provenance_union_is_preserved() {
        let a = label(
            ControlIntegrity::PolicyEndorsed,
            Confidentiality::Public,
            TaintLevel::Clean,
            1,
        );
        let b = label(
            ControlIntegrity::Authenticated,
            Confidentiality::Sensitive,
            TaintLevel::Suspect,
            2,
        );
        let combined = a.combine(&b);
        assert!(combined.provenance_roots.contains(&d(1)));
        assert!(combined.provenance_roots.contains(&d(2)));
        assert_eq!(combined.control_integrity, ControlIntegrity::Authenticated);
        assert_eq!(combined.confidentiality, Confidentiality::Sensitive);
        assert_eq!(combined.taint, TaintLevel::Suspect);
    }

    #[test]
    fn valid_capability_allows_exact_goal_mutation() {
        let (request, facts, policy) = goal_fixture();
        let monitor = ReferenceMonitor;
        let permit = monitor.authorize(&request, &facts, &policy).unwrap();
        assert_eq!(permit.kind(), MutationKind::GoalActivation);
        assert_eq!(permit.resource(), &ResourceId("mind/goals".into()));
        assert_eq!(permit.mutation_digest(), d(2));
        assert_eq!(permit.resource_state_root(), d(8));
        assert_eq!(permit.capability_id(), Some(d(4)));
    }

    #[test]
    fn revocation_epoch_change_fails_closed() {
        let (request, mut facts, policy) = goal_fixture();
        facts.revocation_epoch += 1;
        let decision = ReferenceMonitor.evaluate(&request, &facts, &policy);
        assert_eq!(decision.outcome, DecisionOutcome::RequireAuthorization);
        assert!(decision.reasons.contains(&ReasonCode::StaleCapability));
    }

    #[test]
    fn revoked_capability_fails_closed() {
        let (request, mut facts, policy) = goal_fixture();
        facts.capabilities[0].revoked = true;
        let decision = ReferenceMonitor.evaluate(&request, &facts, &policy);
        assert_eq!(decision.outcome, DecisionOutcome::RequireAuthorization);
        assert!(decision.reasons.contains(&ReasonCode::RevokedCapability));
    }

    #[test]
    fn resource_state_change_requires_revalidation() {
        let (request, mut facts, policy) = goal_fixture();
        facts.resource_state_root = d(99);
        let decision = ReferenceMonitor.evaluate(&request, &facts, &policy);
        assert_eq!(decision.outcome, DecisionOutcome::RequireRevalidation);
        assert_eq!(decision.reasons, vec![ReasonCode::ResourceStateMismatch]);
    }

    #[test]
    fn tainted_dependency_is_quarantined_before_capability_check() {
        let (mut request, facts, policy) = goal_fixture();
        request.input_label.taint = TaintLevel::Tainted;
        let decision = ReferenceMonitor.evaluate(&request, &facts, &policy);
        assert_eq!(decision.outcome, DecisionOutcome::Quarantine);
        assert_eq!(decision.reasons, vec![ReasonCode::TaintedDependency]);
    }

    #[test]
    fn insufficient_integrity_is_quarantined() {
        let (mut request, facts, policy) = goal_fixture();
        request.input_label.control_integrity = ControlIntegrity::Untrusted;
        let decision = ReferenceMonitor.evaluate(&request, &facts, &policy);
        assert_eq!(decision.outcome, DecisionOutcome::Quarantine);
        assert_eq!(
            decision.reasons,
            vec![ReasonCode::InsufficientControlIntegrity]
        );
    }

    #[test]
    fn permit_is_bound_to_policy_and_epochs() {
        let (request, facts, policy) = goal_fixture();
        let permit = ReferenceMonitor.authorize(&request, &facts, &policy).unwrap();
        assert_eq!(permit.policy_root(), policy.root);
        assert_eq!(permit.policy_epoch(), policy.epoch);
        assert_eq!(permit.authorization_epoch(), facts.authorization_epoch);
        assert_eq!(permit.revocation_epoch(), facts.revocation_epoch);
        assert_eq!(permit.sequence(), request.sequence);
    }

    #[test]
    fn precommit_accepts_unchanged_context_and_changes_typestate() {
        let (request, facts, policy) = goal_fixture();
        let permit = ReferenceMonitor.authorize(&request, &facts, &policy).unwrap();
        let commit = ReferenceMonitor.precommit(permit, &facts, &policy).unwrap();
        assert_eq!(commit.request_id(), request.request_id);
        assert_eq!(commit.resource_state_root(), facts.resource_state_root);
        assert_eq!(commit.capability_id(), Some(d(4)));
    }

    #[test]
    fn precommit_destroys_stale_resource_authorization() {
        let (request, mut facts, policy) = goal_fixture();
        let permit = ReferenceMonitor.authorize(&request, &facts, &policy).unwrap();
        facts.resource_state_root = d(99);
        let decision = ReferenceMonitor.precommit(permit, &facts, &policy).unwrap_err();
        assert_eq!(decision.outcome, DecisionOutcome::RequireRevalidation);
        assert_eq!(decision.reasons, vec![ReasonCode::ResourceStateMismatch]);
    }

    #[test]
    fn precommit_rechecks_revocation_context() {
        let (request, mut facts, policy) = goal_fixture();
        let permit = ReferenceMonitor.authorize(&request, &facts, &policy).unwrap();
        facts.revocation_epoch += 1;
        let decision = ReferenceMonitor.precommit(permit, &facts, &policy).unwrap_err();
        assert_eq!(decision.outcome, DecisionOutcome::RequireRevalidation);
        assert_eq!(decision.reasons, vec![ReasonCode::RevocationEpochMismatch]);
    }

    #[test]
    fn precommit_rechecks_exact_capability_identity() {
        let (request, mut facts, policy) = goal_fixture();
        let permit = ReferenceMonitor.authorize(&request, &facts, &policy).unwrap();
        facts.capabilities.clear();
        let decision = ReferenceMonitor.precommit(permit, &facts, &policy).unwrap_err();
        assert_eq!(decision.outcome, DecisionOutcome::RequireAuthorization);
        assert_eq!(decision.reasons, vec![ReasonCode::MissingCapability]);
    }

    #[test]
    fn attenuation_can_only_narrow_parent_authority() {
        let (_, facts, _) = goal_fixture();
        let parent = &facts.capabilities[0];
        let child = parent
            .attenuate(
                d(21),
                PrincipalId("delegate".into()),
                ResourceScope::Exact(ResourceId("mind/goals".into())),
                Consequence::Moderate,
                43,
                Some(48),
            )
            .unwrap();
        assert_eq!(child.mutation, parent.mutation);
        assert!(child.max_consequence <= parent.max_consequence);
        assert!(child.valid_from_sequence >= parent.valid_from_sequence);
        assert!(child.valid_until_sequence <= parent.valid_until_sequence);
        assert_eq!(child.authorization_epoch, parent.authorization_epoch);
        assert_eq!(child.revocation_epoch, parent.revocation_epoch);
    }

    #[test]
    fn attenuation_rejects_resource_scope_widening() {
        let (_, facts, _) = goal_fixture();
        let parent = &facts.capabilities[0];
        let result = parent.attenuate(
            d(21),
            PrincipalId("delegate".into()),
            ResourceScope::Any,
            Consequence::Moderate,
            43,
            Some(48),
        );
        assert_eq!(result, Err(DelegationError::ResourceScopeWidened));
    }

    #[test]
    fn attenuation_rejects_consequence_widening() {
        let (_, facts, _) = goal_fixture();
        let parent = &facts.capabilities[0];
        let result = parent.attenuate(
            d(21),
            PrincipalId("delegate".into()),
            ResourceScope::Exact(ResourceId("mind/goals".into())),
            Consequence::Critical,
            43,
            Some(48),
        );
        assert_eq!(result, Err(DelegationError::ConsequenceWidened));
    }

    #[test]
    fn attenuation_rejects_validity_widening() {
        let (_, facts, _) = goal_fixture();
        let parent = &facts.capabilities[0];
        let result = parent.attenuate(
            d(21),
            PrincipalId("delegate".into()),
            ResourceScope::Exact(ResourceId("mind/goals".into())),
            Consequence::Moderate,
            39,
            Some(48),
        );
        assert_eq!(result, Err(DelegationError::ValidityWidened));
    }

    #[test]
    fn decisions_round_trip_but_live_permits_are_not_serde_types() {
        let (request, facts, policy) = goal_fixture();
        let decision = ReferenceMonitor.evaluate(&request, &facts, &policy);
        let encoded = serde_json::to_string(&decision).unwrap();
        let decoded: MonitorDecision = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, decision);
        // MutationPermit and CommitPermit intentionally have no serde implementations.
    }

    proptest! {
        #[test]
        fn combine_never_increases_control_integrity(
            ia in 0u8..3,
            ib in 0u8..3,
            ca in 0u8..4,
            cb in 0u8..4,
            ta in 0u8..4,
            tb in 0u8..4,
        ) {
            let integrity = [
                ControlIntegrity::Untrusted,
                ControlIntegrity::Authenticated,
                ControlIntegrity::PolicyEndorsed,
            ];
            let confidentiality = [
                Confidentiality::Public,
                Confidentiality::Local,
                Confidentiality::Sensitive,
                Confidentiality::Restricted,
            ];
            let taint = [
                TaintLevel::Clean,
                TaintLevel::Suspect,
                TaintLevel::Tainted,
                TaintLevel::Revoked,
            ];

            let a = label(
                integrity[ia as usize],
                confidentiality[ca as usize],
                taint[ta as usize],
                1,
            );
            let b = label(
                integrity[ib as usize],
                confidentiality[cb as usize],
                taint[tb as usize],
                2,
            );
            let merged = a.combine(&b);

            prop_assert!(merged.control_integrity <= a.control_integrity);
            prop_assert!(merged.control_integrity <= b.control_integrity);
            prop_assert!(merged.confidentiality >= a.confidentiality);
            prop_assert!(merged.confidentiality >= b.confidentiality);
            prop_assert!(merged.taint >= a.taint);
            prop_assert!(merged.taint >= b.taint);
            prop_assert!(merged.provenance_roots.is_superset(&a.provenance_roots));
            prop_assert!(merged.provenance_roots.is_superset(&b.provenance_roots));
        }

        #[test]
        fn attenuation_never_raises_consequence(parent_idx in 0usize..4, child_idx in 0usize..4) {
            let levels = [
                Consequence::Low,
                Consequence::Moderate,
                Consequence::High,
                Consequence::Critical,
            ];
            let (_, mut facts, _) = goal_fixture();
            let parent = &mut facts.capabilities[0];
            parent.max_consequence = levels[parent_idx];
            let result = parent.attenuate(
                d(22),
                PrincipalId("delegate".into()),
                ResourceScope::Exact(ResourceId("mind/goals".into())),
                levels[child_idx],
                43,
                Some(48),
            );
            match result {
                Ok(child) => prop_assert!(child.max_consequence <= parent.max_consequence),
                Err(error) if child_idx > parent_idx => {
                    prop_assert_eq!(error, DelegationError::ConsequenceWidened)
                }
                Err(_) => {}
            }
        }
    }
}
