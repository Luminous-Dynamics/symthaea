// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bearing policy admission for exact autonomous execution authority.
//!
//! Exact action grants prove *what* may execute and under which trust domain,
//! but a production autonomous system also needs provenance for *why* trusted
//! policy minted that authority. This module keeps policy semantics outside the
//! cognition layer while binding a deterministic policy-admission receipt to the
//! exact execution grant consumed by the resource-bound host path.
//!
//! The core deliberately does not decide whether a policy is wise. A trusted
//! [`PolicyEvaluatorDomain`] records the policy descriptor, risk, mode,
//! approvals, evidence snapshot, obligations, emergency state, and host-owned
//! decision time. A separate [`PolicyExecutionDomain`] is pinned to one policy
//! evaluator domain and can mint execution grants only from validated opaque
//! [`PolicyAdmission`] values.
//!
//! The public state-changing path is [`PolicyResourceAction`]. It accepts a
//! [`PolicyGrant`] rather than a raw exact execution capability and preserves the
//! complete admission lineage into [`PolicyResourceEvidenceReceipt`].

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::capability::{CapabilityKind, GrantId, PrincipalId, Read, Scope};
use crate::host::ResolutionError;
use crate::resolution::ResolutionGrant;
use crate::resource::{
    ResolvedResource, ResourceAction, ResourceEvidenceReceipt, ResourceExecutionError,
    ResourceRuntime,
};
use crate::trusted::{
    AuthorityDomain, AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError,
    TrustedBoundOneShotCapability,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Immutable identity of the host policy implementation that made an admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDescriptor {
    family: String,
    version: u32,
    policy_digest: [u8; 32],
    evaluator_schema_version: u32,
}

impl PolicyDescriptor {
    /// Construct a policy descriptor from an immutable policy digest.
    pub fn new(
        family: impl Into<String>,
        version: u32,
        policy_digest: [u8; 32],
        evaluator_schema_version: u32,
    ) -> Result<Self, PolicyError> {
        let family = family.into();
        if !valid_label(&family) {
            return Err(PolicyError::InvalidPolicyFamily(family));
        }
        Ok(Self {
            family,
            version,
            policy_digest,
            evaluator_schema_version,
        })
    }

    /// Stable policy-family label.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Policy version supplied by trusted policy configuration.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Immutable digest of the policy definition/configuration.
    pub fn policy_digest(&self) -> [u8; 32] {
        self.policy_digest
    }

    /// Schema version of the evaluator that interprets this policy.
    pub fn evaluator_schema_version(&self) -> u32 {
        self.evaluator_schema_version
    }
}

/// Execution disposition recorded by trusted policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyMode {
    /// Policy explicitly denied execution.
    Denied,
    /// Policy permits only a non-production/dry-run path.
    DryRun,
    /// Execution requires satisfied external supervision/approval policy.
    Supervised,
    /// Policy permits autonomous execution within the other bound constraints.
    Autonomous,
}

impl PolicyMode {
    fn code(self) -> u8 {
        match self {
            Self::Denied => 0,
            Self::DryRun => 1,
            Self::Supervised => 2,
            Self::Autonomous => 3,
        }
    }
}

/// Canonical summary of policy-specific human/institutional approval evidence.
///
/// The assurance core does not interpret quorum semantics. `satisfied` is a
/// statement made by trusted policy evaluation and is committed alongside the
/// approval-policy and approval-set digests so later evidence can be audited by
/// the domain-specific verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalEvidence {
    policy_digest: [u8; 32],
    approval_set_digest: [u8; 32],
    satisfied: bool,
}

impl ApprovalEvidence {
    /// Construct a canonical approval summary.
    pub fn new(policy_digest: [u8; 32], approval_set_digest: [u8; 32], satisfied: bool) -> Self {
        Self {
            policy_digest,
            approval_set_digest,
            satisfied,
        }
    }

    /// Digest of the approval/quorum policy used by the evaluator.
    pub fn policy_digest(&self) -> [u8; 32] {
        self.policy_digest
    }

    /// Digest committing to the exact observed approval set/evidence references.
    pub fn approval_set_digest(&self) -> [u8; 32] {
        self.approval_set_digest
    }

    /// Whether trusted policy evaluation considered the approval requirement satisfied.
    pub fn satisfied(&self) -> bool {
        self.satisfied
    }
}

/// Immutable policy decision receipt prior to execution-authority minting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyAdmissionReceipt {
    action_binding: [u8; 32],
    scope: Scope,
    risk: ActionRisk,
    mode: PolicyMode,
    policy: PolicyDescriptor,
    evaluator: PrincipalId,
    approvals: ApprovalEvidence,
    evidence_snapshot_digest: [u8; 32],
    obligations_digest: [u8; 32],
    emergency_state_digest: [u8; 32],
    decided_at: SystemTime,
    digest: [u8; 32],
}

impl PolicyAdmissionReceipt {
    /// Exact action authorization binding evaluated by policy.
    pub fn action_binding(&self) -> [u8; 32] {
        self.action_binding
    }

    /// Exact logical action scope evaluated by policy.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Risk classification evaluated by policy.
    pub fn risk(&self) -> ActionRisk {
        self.risk
    }

    /// Policy execution disposition.
    pub fn mode(&self) -> PolicyMode {
        self.mode
    }

    /// Immutable policy descriptor.
    pub fn policy(&self) -> &PolicyDescriptor {
        &self.policy
    }

    /// Trusted evaluator principal.
    pub fn evaluator(&self) -> PrincipalId {
        self.evaluator
    }

    /// Approval/quorum evidence summary.
    pub fn approvals(&self) -> &ApprovalEvidence {
        &self.approvals
    }

    /// Digest of calibration/evidence snapshot considered by policy.
    pub fn evidence_snapshot_digest(&self) -> [u8; 32] {
        self.evidence_snapshot_digest
    }

    /// Digest of obligations/constraints attached to the decision.
    pub fn obligations_digest(&self) -> [u8; 32] {
        self.obligations_digest
    }

    /// Digest of emergency/stop/revocation state observed by policy.
    pub fn emergency_state_digest(&self) -> [u8; 32] {
        self.emergency_state_digest
    }

    /// Host-owned wall-clock decision time recorded by the evaluator.
    pub fn decided_at(&self) -> SystemTime {
        self.decided_at
    }

    /// Domain-separated digest of the complete policy admission receipt.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Trusted policy evaluator domain that can create opaque policy admissions.
#[derive(Debug)]
pub struct PolicyEvaluatorDomain {
    inner: AuthorityDomain,
    descriptor: PolicyDescriptor,
    clock: Arc<PolicyClock>,
}

impl PolicyEvaluatorDomain {
    /// Create a trusted evaluator for one immutable policy descriptor.
    pub fn new(principal: PrincipalId, descriptor: PolicyDescriptor) -> Self {
        Self {
            inner: AuthorityDomain::new(principal),
            descriptor,
            clock: Arc::new(PolicyClock::new()),
        }
    }

    /// Policy evaluator trust-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Evaluator principal identity.
    pub fn principal(&self) -> PrincipalId {
        self.inner.principal()
    }

    /// Immutable policy descriptor used by this evaluator.
    pub fn descriptor(&self) -> &PolicyDescriptor {
        &self.descriptor
    }

    /// Create a verifier retained by execution-authority policy code.
    pub fn verifier(&self) -> PolicyVerifier {
        PolicyVerifier {
            inner: self.inner.verifier(),
        }
    }

    /// Record a trusted policy decision for one exact action binding.
    ///
    /// Domain-specific code is responsible for calculating `mode`, approval
    /// evidence, evidence/calibration snapshot, obligations, and emergency-state
    /// digests. This method supplies host-owned time and attests the resulting
    /// deterministic receipt in the evaluator's revocation epoch.
    #[allow(clippy::too_many_arguments)]
    pub fn admit(
        &self,
        action_binding: [u8; 32],
        scope: Scope,
        risk: ActionRisk,
        mode: PolicyMode,
        approvals: ApprovalEvidence,
        evidence_snapshot_digest: [u8; 32],
        obligations_digest: [u8; 32],
        emergency_state_digest: [u8; 32],
        expires_at: Option<SystemTime>,
    ) -> PolicyAdmission {
        let decided_at = self.clock.now();
        let digest = compute_admission_digest(
            action_binding,
            &scope,
            risk,
            mode,
            &self.descriptor,
            self.inner.principal(),
            &approvals,
            evidence_snapshot_digest,
            obligations_digest,
            emergency_state_digest,
            decided_at,
        );
        let receipt = PolicyAdmissionReceipt {
            action_binding,
            scope: scope.clone(),
            risk,
            mode,
            policy: self.descriptor.clone(),
            evaluator: self.inner.principal(),
            approvals,
            evidence_snapshot_digest,
            obligations_digest,
            emergency_state_digest,
            decided_at,
            digest,
        };
        let attestation = self.inner.issue_bound_one_shot::<Read>(
            self.inner.principal(),
            scope,
            expires_at,
            digest,
        );
        PolicyAdmission {
            receipt,
            attestation,
        }
    }

    /// Revoke admissions from earlier evaluator epochs.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        self.inner.revoke_all()
    }
}

/// Host-retained trust anchor for policy-admission provenance.
#[derive(Debug, Clone)]
pub struct PolicyVerifier {
    inner: AuthorityVerifier,
}

impl PolicyVerifier {
    /// Policy evaluator trust-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Current policy evaluator revocation epoch.
    pub fn current_epoch(&self) -> AuthorityEpoch {
        self.inner.current_epoch()
    }
}

/// Opaque trusted policy admission.
#[derive(Debug)]
pub struct PolicyAdmission {
    receipt: PolicyAdmissionReceipt,
    attestation: TrustedBoundOneShotCapability<Read>,
}

impl PolicyAdmission {
    /// Immutable decision receipt carried by this admission.
    pub fn receipt(&self) -> &PolicyAdmissionReceipt {
        &self.receipt
    }

    /// Policy evaluator domain that attested the receipt.
    pub fn evaluator_domain(&self) -> AuthorityDomainId {
        self.attestation.domain_id()
    }

    /// Policy evaluator revocation epoch.
    pub fn evaluator_epoch(&self) -> AuthorityEpoch {
        self.attestation.epoch()
    }

    /// Policy-admission attestation grant id.
    pub fn attestation_grant_id(&self) -> GrantId {
        self.attestation.metadata().grant_id()
    }

    fn validate_with(&self, verifier: &PolicyVerifier, now: SystemTime) -> Result<(), PolicyError> {
        self.attestation
            .validate_with(&verifier.inner, now)
            .map_err(PolicyError::Trust)?;
        if self.attestation.metadata().scope() != self.receipt.scope() {
            return Err(PolicyError::AdmissionScopeMismatch);
        }
        if self.attestation.binding() != self.receipt.digest() {
            return Err(PolicyError::AdmissionBindingMismatch);
        }
        Ok(())
    }
}

/// Execution-authority root that accepts admissions only from one pinned policy evaluator domain.
#[derive(Debug)]
pub struct PolicyExecutionDomain {
    inner: AuthorityDomain,
    policy_verifier: PolicyVerifier,
    clock: Arc<PolicyClock>,
}

impl PolicyExecutionDomain {
    /// Create a state-changing execution authority domain pinned to one trusted policy evaluator.
    pub fn new(principal: PrincipalId, policy_verifier: PolicyVerifier) -> Self {
        Self {
            inner: AuthorityDomain::new(principal),
            policy_verifier,
            clock: Arc::new(PolicyClock::new()),
        }
    }

    /// Execution authority-domain identity used to configure [`TrustedRuntime`].
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Policy evaluator domain this execution authority trusts.
    pub fn policy_domain(&self) -> AuthorityDomainId {
        self.policy_verifier.domain_id()
    }

    /// Verifier supplied to the strict host runtime.
    pub fn verifier(&self) -> AuthorityVerifier {
        self.inner.verifier()
    }

    /// Mint exact state-changing authority only from a valid pinned policy admission.
    pub fn issue<K: CapabilityKind>(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
        action_binding: [u8; 32],
        admission: PolicyAdmission,
    ) -> Result<PolicyGrant<K>, PolicyError> {
        let now = self.clock.now();
        admission.validate_with(&self.policy_verifier, now)?;

        if admission.receipt.action_binding() != action_binding {
            return Err(PolicyError::ActionBindingMismatch);
        }
        if admission.receipt.scope() != &scope {
            return Err(PolicyError::PolicyScopeMismatch {
                admitted: admission.receipt.scope().clone(),
                requested: scope,
            });
        }

        match admission.receipt.mode() {
            PolicyMode::Denied => return Err(PolicyError::ExecutionDenied),
            PolicyMode::DryRun => return Err(PolicyError::DryRunOnly),
            PolicyMode::Supervised if !admission.receipt.approvals().satisfied() => {
                return Err(PolicyError::SupervisionUnsatisfied);
            }
            PolicyMode::Supervised | PolicyMode::Autonomous => {}
        }

        let policy_domain = admission.evaluator_domain();
        let policy_epoch = admission.evaluator_epoch();
        let policy_attestation_grant_id = admission.attestation_grant_id();
        let receipt = admission.receipt.clone();
        let grant =
            self.inner
                .issue_bound_one_shot::<K>(subject, scope, expires_at, action_binding);
        let policy_binding = compute_policy_grant_binding(
            action_binding,
            receipt.digest(),
            self.inner.domain_id(),
            grant.epoch(),
            grant.metadata().grant_id(),
            policy_domain,
            policy_epoch,
            policy_attestation_grant_id,
        );
        let evidence = PolicyAuthorizationEvidence {
            receipt,
            policy_domain,
            policy_epoch,
            policy_attestation_grant_id,
            execution_domain: self.inner.domain_id(),
            execution_epoch: grant.epoch(),
            execution_grant_id: grant.metadata().grant_id(),
            policy_binding,
        };
        Ok(PolicyGrant { grant, evidence })
    }

    /// Revoke state-changing execution grants from earlier execution epochs.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        self.inner.revoke_all()
    }
}

/// Opaque exact execution authority carrying its policy-admission provenance.
#[derive(Debug)]
pub struct PolicyGrant<K: CapabilityKind> {
    grant: TrustedBoundOneShotCapability<K>,
    evidence: PolicyAuthorizationEvidence,
}

impl<K: CapabilityKind> PolicyGrant<K> {
    /// Deterministic binding between exact action authority and its policy admission.
    pub fn policy_binding(&self) -> [u8; 32] {
        self.evidence.policy_binding
    }

    /// Immutable policy authorization evidence carried by this grant.
    pub fn evidence(&self) -> &PolicyAuthorizationEvidence {
        &self.evidence
    }

    fn validate_for(
        &self,
        action_binding: [u8; 32],
        scope: &Scope,
        risk: ActionRisk,
    ) -> Result<(), PolicyError> {
        if self.grant.binding() != action_binding
            || self.evidence.receipt.action_binding() != action_binding
        {
            return Err(PolicyError::ActionBindingMismatch);
        }
        if self.grant.metadata().scope() != scope || self.evidence.receipt.scope() != scope {
            return Err(PolicyError::PolicyScopeMismatch {
                admitted: self.evidence.receipt.scope().clone(),
                requested: scope.clone(),
            });
        }
        if self.evidence.receipt.risk() != risk {
            return Err(PolicyError::RiskMismatch {
                admitted: self.evidence.receipt.risk(),
                required: risk,
            });
        }
        let expected = compute_policy_grant_binding(
            action_binding,
            self.evidence.receipt.digest(),
            self.grant.domain_id(),
            self.grant.epoch(),
            self.grant.metadata().grant_id(),
            self.evidence.policy_domain,
            self.evidence.policy_epoch,
            self.evidence.policy_attestation_grant_id,
        );
        if expected != self.evidence.policy_binding {
            return Err(PolicyError::PolicyBindingMismatch);
        }
        Ok(())
    }

    fn into_parts(
        self,
    ) -> (
        TrustedBoundOneShotCapability<K>,
        PolicyAuthorizationEvidence,
    ) {
        (self.grant, self.evidence)
    }
}

/// Immutable provenance linking exact execution authority to trusted policy admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyAuthorizationEvidence {
    receipt: PolicyAdmissionReceipt,
    policy_domain: AuthorityDomainId,
    policy_epoch: AuthorityEpoch,
    policy_attestation_grant_id: GrantId,
    execution_domain: AuthorityDomainId,
    execution_epoch: AuthorityEpoch,
    execution_grant_id: GrantId,
    policy_binding: [u8; 32],
}

impl PolicyAuthorizationEvidence {
    /// Policy admission receipt that justified authority minting.
    pub fn receipt(&self) -> &PolicyAdmissionReceipt {
        &self.receipt
    }

    /// Trusted policy evaluator domain.
    pub fn policy_domain(&self) -> AuthorityDomainId {
        self.policy_domain
    }

    /// Policy evaluator epoch consumed during minting.
    pub fn policy_epoch(&self) -> AuthorityEpoch {
        self.policy_epoch
    }

    /// Grant id attesting the policy admission receipt.
    pub fn policy_attestation_grant_id(&self) -> GrantId {
        self.policy_attestation_grant_id
    }

    /// Exact execution authority domain.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_domain
    }

    /// Execution authority epoch.
    pub fn execution_epoch(&self) -> AuthorityEpoch {
        self.execution_epoch
    }

    /// Exact execution grant id.
    pub fn execution_grant_id(&self) -> GrantId {
        self.execution_grant_id
    }

    /// Deterministic digest joining policy receipt and exact execution authority.
    pub fn policy_binding(&self) -> [u8; 32] {
        self.policy_binding
    }
}

/// Resource-bound runtime wrapper that requires policy-bound execution grants.
#[derive(Debug, Clone)]
pub struct PolicyResourceRuntime {
    inner: ResourceRuntime,
}

impl PolicyResourceRuntime {
    /// Wrap an existing resource-bound host runtime.
    pub fn new(inner: ResourceRuntime) -> Self {
        Self { inner }
    }

    /// Admit a resolved resource action into the policy-bound lifecycle.
    pub fn admit_resolved<K: CapabilityKind, H>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        resource: ResolvedResource<H>,
        canonical_payload: &[u8],
    ) -> Result<PolicyResourceAction<K, Proposed, H>, crate::ResourceError> {
        let inner = self
            .inner
            .admit_resolved::<K, H>(actor, kind, resource, canonical_payload)?;
        Ok(PolicyResourceAction {
            inner,
            policy_evidence: None,
        })
    }
}

/// Resource-bound action lifecycle whose execution transition requires policy provenance.
pub struct PolicyResourceAction<K: CapabilityKind, S, H> {
    inner: ResourceAction<K, S, H>,
    policy_evidence: Option<PolicyAuthorizationEvidence>,
}

impl<K: CapabilityKind, S, H> PolicyResourceAction<K, S, H> {
    /// Stable action identity.
    pub fn id(&self) -> ActionId {
        self.inner.id()
    }

    /// Acting principal.
    pub fn actor(&self) -> PrincipalId {
        self.inner.actor()
    }

    /// Immutable resource-bound action descriptor.
    pub fn descriptor(&self) -> &ActionDescriptor {
        self.inner.descriptor()
    }

    /// Concrete resource identity committed into this action.
    pub fn resource_identity(&self) -> &crate::ResourceIdentity {
        self.inner.resource_identity()
    }
}

impl<K: CapabilityKind, H> PolicyResourceAction<K, Proposed, H> {
    /// Attach explicit risk before policy evaluation.
    pub fn assess(self, risk: ActionRisk) -> PolicyResourceAction<K, RiskAssessed, H> {
        PolicyResourceAction {
            inner: self.inner.assess(risk),
            policy_evidence: None,
        }
    }
}

impl<K: CapabilityKind, H> PolicyResourceAction<K, RiskAssessed, H> {
    /// Risk classification evaluated by policy.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact action binding supplied to trusted policy evaluation.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Consume a policy-bound exact execution grant.
    pub fn authorize(
        self,
        grant: PolicyGrant<K>,
    ) -> Result<PolicyResourceAction<K, Authorized, H>, PolicyError> {
        grant.validate_for(
            self.inner.authorization_binding(),
            self.inner.descriptor().scope(),
            self.inner.risk(),
        )?;
        let (grant, evidence) = grant.into_parts();
        let inner = self.inner.authorize(grant).map_err(PolicyError::Trust)?;
        Ok(PolicyResourceAction {
            inner,
            policy_evidence: Some(evidence),
        })
    }
}

impl<K: CapabilityKind, H> PolicyResourceAction<K, Authorized, H> {
    /// Policy provenance consumed by this authorized action.
    pub fn policy_evidence(&self) -> &PolicyAuthorizationEvidence {
        self.policy_evidence
            .as_ref()
            .expect("Authorized policy action always carries policy evidence")
    }

    /// Execute against the retained concrete resource handle.
    pub fn execute_with<F, E>(
        self,
        execute: F,
    ) -> Result<PolicyResourceAction<K, Executed, H>, ResourceExecutionError<E>>
    where
        F: FnOnce(&mut H) -> Result<[u8; 32], E>,
    {
        let inner = self.inner.execute_with(execute)?;
        Ok(PolicyResourceAction {
            inner,
            policy_evidence: self.policy_evidence,
        })
    }
}

impl<K: CapabilityKind, H> PolicyResourceAction<K, Executed, H> {
    /// Exact independent-observation binding.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Attach independently authorized observation.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<crate::Observe>,
        observation: Observation,
    ) -> Result<PolicyResourceAction<K, Observed, H>, TrustError> {
        let inner = self.inner.observe(observer, observation)?;
        Ok(PolicyResourceAction {
            inner,
            policy_evidence: self.policy_evidence,
        })
    }
}

impl<K: CapabilityKind, H> PolicyResourceAction<K, Observed, H> {
    /// Exact final-resolution binding.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        self.inner.resolution_binding(decision)
    }

    /// Consume exact final-resolution authority and emit complete policy/resource evidence.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<
        (
            PolicyResourceAction<K, Resolved, H>,
            PolicyResourceEvidenceReceipt,
        ),
        ResolutionError,
    > {
        let (inner, resource_receipt) = self.inner.resolve(grant, decision)?;
        let policy_evidence = self
            .policy_evidence
            .expect("Observed policy action always carries policy evidence");
        let receipt = PolicyResourceEvidenceReceipt {
            resource_receipt,
            policy_evidence: policy_evidence.clone(),
        };
        Ok((
            PolicyResourceAction {
                inner,
                policy_evidence: Some(policy_evidence),
            },
            receipt,
        ))
    }
}

impl<K: CapabilityKind, H> PolicyResourceAction<K, Resolved, H> {
    /// Policy authorization evidence retained by the completed action.
    pub fn policy_evidence(&self) -> &PolicyAuthorizationEvidence {
        self.policy_evidence
            .as_ref()
            .expect("Resolved policy action always carries policy evidence")
    }
}

/// Final evidence joining policy admission with resource/execution/observation/resolution lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyResourceEvidenceReceipt {
    resource_receipt: ResourceEvidenceReceipt,
    policy_evidence: PolicyAuthorizationEvidence,
}

impl PolicyResourceEvidenceReceipt {
    /// Concrete-resource + execution/observation/resolution evidence.
    pub fn resource_receipt(&self) -> &ResourceEvidenceReceipt {
        &self.resource_receipt
    }

    /// Policy admission and execution-authority provenance.
    pub fn policy_evidence(&self) -> &PolicyAuthorizationEvidence {
        &self.policy_evidence
    }
}

/// Policy admission or policy-bound execution-authority failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// Invalid policy family label.
    InvalidPolicyFamily(String),
    /// Policy admission trust-domain/epoch/expiry validation failed.
    Trust(TrustError),
    /// Admission attestation scope did not match the policy receipt.
    AdmissionScopeMismatch,
    /// Admission attestation did not bind the exact policy receipt.
    AdmissionBindingMismatch,
    /// Policy receipt was created for another action binding.
    ActionBindingMismatch,
    /// Policy receipt and requested execution scope differ.
    PolicyScopeMismatch {
        /// Scope recorded by policy admission.
        admitted: Scope,
        /// Scope requested for exact execution authority.
        requested: Scope,
    },
    /// Policy receipt risk differs from the action risk.
    RiskMismatch {
        /// Risk recorded by policy.
        admitted: ActionRisk,
        /// Risk required by the action.
        required: ActionRisk,
    },
    /// Policy denied execution.
    ExecutionDenied,
    /// Policy permits only a dry-run path, not ordinary execution authority.
    DryRunOnly,
    /// Supervised mode did not contain satisfied approval evidence.
    SupervisionUnsatisfied,
    /// Internal policy/action binding evidence was inconsistent.
    PolicyBindingMismatch,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicyFamily(value) => write!(f, "invalid policy family: {value:?}"),
            Self::Trust(error) => write!(f, "policy trust validation failed: {error}"),
            Self::AdmissionScopeMismatch => write!(f, "policy admission scope mismatch"),
            Self::AdmissionBindingMismatch => write!(f, "policy admission binding mismatch"),
            Self::ActionBindingMismatch => write!(f, "policy admission targets another action"),
            Self::PolicyScopeMismatch { .. } => {
                write!(f, "policy admission scope differs from execution scope")
            }
            Self::RiskMismatch { .. } => {
                write!(f, "policy admission risk differs from action risk")
            }
            Self::ExecutionDenied => write!(f, "policy denied execution"),
            Self::DryRunOnly => write!(f, "policy permits only dry-run execution"),
            Self::SupervisionUnsatisfied => {
                write!(f, "supervised execution lacks satisfied approval evidence")
            }
            Self::PolicyBindingMismatch => {
                write!(f, "policy grant provenance binding is inconsistent")
            }
        }
    }
}

impl std::error::Error for PolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Trust(error) => Some(error),
            _ => None,
        }
    }
}

fn compute_admission_digest(
    action_binding: [u8; 32],
    scope: &Scope,
    risk: ActionRisk,
    mode: PolicyMode,
    policy: &PolicyDescriptor,
    evaluator: PrincipalId,
    approvals: &ApprovalEvidence,
    evidence_snapshot_digest: [u8; 32],
    obligations_digest: [u8; 32],
    emergency_state_digest: [u8; 32],
    decided_at: SystemTime,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/policy-admission-v1\0");
    hash_field(&mut hasher, &action_binding);
    hash_field(&mut hasher, scope.namespace().as_bytes());
    for segment in scope.segments() {
        hash_field(&mut hasher, segment.as_bytes());
    }
    hash_field(&mut hasher, &[risk_code(risk)]);
    hash_field(&mut hasher, &[mode.code()]);
    hash_field(&mut hasher, policy.family().as_bytes());
    hash_field(&mut hasher, &policy.version().to_le_bytes());
    hash_field(&mut hasher, &policy.policy_digest());
    hash_field(
        &mut hasher,
        &policy.evaluator_schema_version().to_le_bytes(),
    );
    hash_field(&mut hasher, evaluator.as_uuid().as_bytes());
    hash_field(&mut hasher, &approvals.policy_digest());
    hash_field(&mut hasher, &approvals.approval_set_digest());
    hash_field(&mut hasher, &[u8::from(approvals.satisfied())]);
    hash_field(&mut hasher, &evidence_snapshot_digest);
    hash_field(&mut hasher, &obligations_digest);
    hash_field(&mut hasher, &emergency_state_digest);
    hash_system_time(&mut hasher, decided_at);
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn compute_policy_grant_binding(
    action_binding: [u8; 32],
    admission_digest: [u8; 32],
    execution_domain: AuthorityDomainId,
    execution_epoch: AuthorityEpoch,
    execution_grant_id: GrantId,
    policy_domain: AuthorityDomainId,
    policy_epoch: AuthorityEpoch,
    policy_attestation_grant_id: GrantId,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/policy-grant-v1\0");
    hash_field(&mut hasher, &action_binding);
    hash_field(&mut hasher, &admission_digest);
    hash_field(&mut hasher, execution_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, &execution_epoch.value().to_le_bytes());
    hash_field(&mut hasher, execution_grant_id.as_uuid().as_bytes());
    hash_field(&mut hasher, policy_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, &policy_epoch.value().to_le_bytes());
    hash_field(
        &mut hasher,
        policy_attestation_grant_id.as_uuid().as_bytes(),
    );
    *hasher.finalize().as_bytes()
}

fn risk_code(risk: ActionRisk) -> u8 {
    match risk {
        ActionRisk::Observation => 0,
        ActionRisk::Reversible => 1,
        ActionRisk::StateModifying => 2,
        ActionRisk::Destructive => 3,
        ActionRisk::Critical => 4,
    }
}

fn hash_system_time(hasher: &mut blake3::Hasher, time: SystemTime) {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            hash_field(hasher, &[0]);
            hash_field(hasher, &duration.as_secs().to_le_bytes());
            hash_field(hasher, &duration.subsec_nanos().to_le_bytes());
        }
        Err(error) => {
            let duration = error.duration();
            hash_field(hasher, &[1]);
            hash_field(hasher, &duration.as_secs().to_le_bytes());
            hash_field(hasher, &duration.subsec_nanos().to_le_bytes());
        }
    }
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn valid_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

#[derive(Debug)]
struct PolicyClock {
    last: Mutex<SystemTime>,
}

impl PolicyClock {
    fn new() -> Self {
        Self {
            last: Mutex::new(SystemTime::now()),
        }
    }

    fn now(&self) -> SystemTime {
        let observed = SystemTime::now();
        let mut last = self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if observed > *last {
            *last = observed;
        }
        *last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::TrustedRuntime;
    use crate::resolution::ResolutionAuthorityDomain;
    use crate::resource::ResourceResolverDomain;
    use crate::{AdapterSchema, ObservedOutcome, ResourceIdentity, Write};

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    fn descriptor(byte: u8) -> PolicyDescriptor {
        PolicyDescriptor::new("magi-gate", 1, [byte; 32], 1).unwrap()
    }

    fn approvals(satisfied: bool) -> ApprovalEvidence {
        ApprovalEvidence::new([1; 32], [2; 32], satisfied)
    }

    fn resource_identity() -> ResourceIdentity {
        ResourceIdentity::new(
            scope(),
            "worktree-file",
            [3; 32],
            [4; 32],
            AdapterSchema::new("policy-test", 1).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn denied_and_dry_run_admissions_cannot_mint_normal_execution() {
        for mode in [PolicyMode::Denied, PolicyMode::DryRun] {
            let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(5));
            let execution = PolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier());
            let binding = [6; 32];
            let admission = evaluator.admit(
                binding,
                scope(),
                ActionRisk::Reversible,
                mode,
                approvals(true),
                [7; 32],
                [8; 32],
                [9; 32],
                None,
            );
            assert!(
                execution
                    .issue::<Write>(PrincipalId::new(), scope(), None, binding, admission)
                    .is_err()
            );
        }
    }

    #[test]
    fn supervised_admission_requires_satisfied_approval_evidence() {
        let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
        let execution = PolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier());
        let binding = [2; 32];
        let admission = evaluator.admit(
            binding,
            scope(),
            ActionRisk::StateModifying,
            PolicyMode::Supervised,
            approvals(false),
            [3; 32],
            [4; 32],
            [5; 32],
            None,
        );
        assert!(matches!(
            execution.issue::<Write>(PrincipalId::new(), scope(), None, binding, admission),
            Err(PolicyError::SupervisionUnsatisfied)
        ));
    }

    #[test]
    fn unrelated_policy_evaluator_cannot_justify_execution_authority() {
        let trusted = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
        let attacker = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
        let execution = PolicyExecutionDomain::new(PrincipalId::new(), trusted.verifier());
        let binding = [2; 32];
        let admission = attacker.admit(
            binding,
            scope(),
            ActionRisk::Reversible,
            PolicyMode::Autonomous,
            approvals(true),
            [3; 32],
            [4; 32],
            [5; 32],
            None,
        );
        assert!(
            execution
                .issue::<Write>(PrincipalId::new(), scope(), None, binding, admission)
                .is_err()
        );
    }

    #[test]
    fn policy_evidence_survives_resource_execution_and_resolution() {
        let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(9));
        let execution = PolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier());
        let observation = AuthorityDomain::new(PrincipalId::new());
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let strict = TrustedRuntime::new(
            execution.verifier(),
            observation.verifier(),
            resolution.verifier(),
        );
        let resource_runtime = ResourceRuntime::new(strict, resources.verifier());
        let runtime = PolicyResourceRuntime::new(resource_runtime);
        let actor = PrincipalId::new();
        let observer = PrincipalId::new();
        let resolver = PrincipalId::new();
        let action = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit-source",
                resources.resolve(0_u64, resource_identity(), None),
                b"patch",
            )
            .unwrap()
            .assess(ActionRisk::Reversible);
        let admission = evaluator.admit(
            action.authorization_binding(),
            scope(),
            action.risk(),
            PolicyMode::Autonomous,
            approvals(true),
            [10; 32],
            [11; 32],
            [12; 32],
            None,
        );
        let policy_grant = execution
            .issue::<Write>(
                actor,
                scope(),
                None,
                action.authorization_binding(),
                admission,
            )
            .unwrap();
        let expected_policy_digest = policy_grant.evidence().receipt().digest();
        let action = action
            .authorize(policy_grant)
            .unwrap()
            .execute_with(|handle| -> Result<[u8; 32], &'static str> {
                *handle += 1;
                Ok([13; 32])
            })
            .unwrap();
        let observer_grant = observation.issue_bound_one_shot::<crate::Observe>(
            observer,
            scope(),
            None,
            action.observation_binding(),
        );
        let action = action
            .observe(
                observer_grant,
                Observation::new(ObservedOutcome::Success, [14; 32]),
            )
            .unwrap();
        let decision = ResolutionDecision::Confirmed;
        let resolver_grant = resolution.issue_bound_one_shot(
            resolver,
            scope(),
            None,
            action.resolution_binding(decision),
        );
        let (_, receipt) = action.resolve(resolver_grant, decision).unwrap();
        assert_eq!(
            receipt.policy_evidence().receipt().digest(),
            expected_policy_digest
        );
        assert_eq!(
            receipt.policy_evidence().policy_domain(),
            evaluator.domain_id()
        );
        assert_eq!(
            receipt.policy_evidence().execution_domain(),
            execution.domain_id()
        );
    }
}
