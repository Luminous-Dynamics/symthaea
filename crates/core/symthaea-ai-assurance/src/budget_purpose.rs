// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bearing quantitative-purpose admission.
//!
//! The budget substrate proves that a conserved quantity was reserved for an
//! exact action. The policy substrate proves why that action was admitted. Those
//! statements are deliberately distinct, but a production autonomous system
//! also needs to prove which trusted quantitative policy approved *this exact
//! budget envelope for this exact purpose*.
//!
//! This module adds that missing cross-binding without pretending the earlier
//! general execution policy already contained quantitative semantics. A
//! [`BudgetPurposeAuthorityDomain`] is an independent trusted policy root. It
//! approves one existing [`crate::BudgetLease`] only after cross-checking the
//! temporally bounded execution-policy lineage and the budget lease against
//! host-pinned policy, execution, and budget verifiers.
//!
//! The resulting [`PurposeBoundBudgetLease`] remains affine because it owns the
//! original non-`Clone` budget lease. Its admission receipt commits to:
//!
//! - exact action binding and principal;
//! - exact logical scope;
//! - exact purpose digest;
//! - exact allocation and budget profile;
//! - original budget lease/domain/epoch/lifetime;
//! - exact policy-admission digest and policy binding;
//! - exact temporal policy/execution lineage;
//! - quantitative-policy descriptor, approver, decision time, and lifetime.
//!
//! [`PurposeGuardedRuntime`] composes this approval with the current strongest
//! [`crate::IndependenceGuardedRuntime`] path. Final evidence therefore answers
//! two different questions explicitly:
//!
//! 1. which general policy authorized the effectful action; and
//! 2. which quantitative-purpose policy approved the resource envelope.
//!
//! This does **not** yet make child-budget purpose delegation implicit. A child
//! action must receive a fresh purpose admission (or a future explicit delegated
//! purpose transition) rather than treating a raw affine `split` as proof that
//! the parent policy approved the child's purpose.

use crate::Observe;
use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::budget::{
    BudgetDimension, BudgetError, BudgetLease, BudgetQuantities, BudgetReleaseReceipt,
    BudgetVerifier,
};
use crate::capability::{CapabilityKind, GrantId, PrincipalId, Read, Scope};
use crate::effect_guard::{
    EffectAttemptEvidence, EffectAttemptOutcome, EffectGuardedAuthorizeError,
};
use crate::independence::{
    IndependenceEffectAttemptFailure, IndependenceEvidenceReceipt, IndependenceGuardedAction,
    IndependenceGuardedRuntime, IndependenceObservationError, IndependenceResolutionError,
};
use crate::policy::{PolicyAuthorizationEvidence, PolicyVerifier};
use crate::resolution::ResolutionGrant;
use crate::resource::{ResolvedResource, ResourceError};
use crate::temporal_policy::{TemporalDerivationEvidence, TemporalPolicyGrant};
use crate::trusted::{
    AuthorityDomain, AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError,
    TrustedBoundOneShotCapability,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Immutable identity of the trusted quantitative-purpose policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetPurposeDescriptor {
    family: String,
    version: u32,
    policy_digest: [u8; 32],
    schema_version: u32,
}

impl BudgetPurposeDescriptor {
    /// Construct a stable quantitative-purpose policy descriptor.
    pub fn new(
        family: impl Into<String>,
        version: u32,
        policy_digest: [u8; 32],
        schema_version: u32,
    ) -> Result<Self, BudgetPurposeError> {
        let family = family.into();
        if !valid_label(&family) {
            return Err(BudgetPurposeError::InvalidPolicyFamily(family));
        }
        Ok(Self {
            family,
            version,
            policy_digest,
            schema_version,
        })
    }

    /// Stable policy family label.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Policy version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Immutable digest of the quantitative-purpose policy/configuration.
    pub fn policy_digest(&self) -> [u8; 32] {
        self.policy_digest
    }

    /// Schema version defining how this policy interprets purpose admissions.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// Trusted lifetime configuration for quantitative-purpose admissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetPurposeRules {
    allow_unbounded_purpose: bool,
}

impl BudgetPurposeRules {
    /// Strict production-oriented rules: purpose admissions require finite
    /// lifetimes even when every parent authority is unbounded.
    pub const fn strict() -> Self {
        Self {
            allow_unbounded_purpose: false,
        }
    }

    /// Construct explicit trusted purpose-lifetime rules.
    pub const fn new(allow_unbounded_purpose: bool) -> Self {
        Self {
            allow_unbounded_purpose,
        }
    }

    /// Whether an unbounded purpose admission may be derived when every parent
    /// authority is also unbounded.
    pub const fn allow_unbounded_purpose(self) -> bool {
        self.allow_unbounded_purpose
    }
}

impl Default for BudgetPurposeRules {
    fn default() -> Self {
        Self::strict()
    }
}

/// Immutable quantitative-policy admission joining exact budget and policy lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetPurposeAdmissionReceipt {
    action_binding: [u8; 32],
    subject: PrincipalId,
    scope: Scope,
    purpose_digest: [u8; 32],
    allocation: BudgetQuantities,
    budget_profile_digest: [u8; 32],
    budget_lease_id: GrantId,
    budget_domain: AuthorityDomainId,
    budget_epoch: AuthorityEpoch,
    budget_expires_at: Option<SystemTime>,
    policy_admission_digest: [u8; 32],
    policy_binding: [u8; 32],
    policy_domain: AuthorityDomainId,
    policy_epoch: AuthorityEpoch,
    policy_attestation_grant_id: GrantId,
    execution_domain: AuthorityDomainId,
    execution_epoch: AuthorityEpoch,
    execution_grant_id: GrantId,
    execution_expires_at: Option<SystemTime>,
    descriptor: BudgetPurposeDescriptor,
    approver: PrincipalId,
    approved_at: SystemTime,
    expires_at: Option<SystemTime>,
    digest: [u8; 32],
}

impl BudgetPurposeAdmissionReceipt {
    /// Exact action authorization binding receiving quantitative approval.
    pub fn action_binding(&self) -> [u8; 32] {
        self.action_binding
    }

    /// Principal receiving the approved budget envelope.
    pub fn subject(&self) -> PrincipalId {
        self.subject
    }

    /// Exact logical scope approved by quantitative policy.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Domain-specific purpose commitment.
    pub fn purpose_digest(&self) -> [u8; 32] {
        self.purpose_digest
    }

    /// Exact approved quantitative envelope.
    pub fn allocation(&self) -> BudgetQuantities {
        self.allocation
    }

    /// Immutable budget profile digest governing the envelope.
    pub fn budget_profile_digest(&self) -> [u8; 32] {
        self.budget_profile_digest
    }

    /// Original conserved budget lease id.
    pub fn budget_lease_id(&self) -> GrantId {
        self.budget_lease_id
    }

    /// Budget authority domain.
    pub fn budget_domain(&self) -> AuthorityDomainId {
        self.budget_domain
    }

    /// Budget revocation epoch captured at approval.
    pub fn budget_epoch(&self) -> AuthorityEpoch {
        self.budget_epoch
    }

    /// Original budget-lease lifetime.
    pub fn budget_expires_at(&self) -> Option<SystemTime> {
        self.budget_expires_at
    }

    /// General execution-policy admission digest.
    pub fn policy_admission_digest(&self) -> [u8; 32] {
        self.policy_admission_digest
    }

    /// Deterministic binding between the general policy admission and exact execution grant.
    pub fn policy_binding(&self) -> [u8; 32] {
        self.policy_binding
    }

    /// General policy evaluator domain.
    pub fn policy_domain(&self) -> AuthorityDomainId {
        self.policy_domain
    }

    /// General policy evaluator epoch.
    pub fn policy_epoch(&self) -> AuthorityEpoch {
        self.policy_epoch
    }

    /// General policy admission attestation id.
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

    /// Temporally derived execution-grant lifetime.
    pub fn execution_expires_at(&self) -> Option<SystemTime> {
        self.execution_expires_at
    }

    /// Quantitative-purpose policy descriptor.
    pub fn descriptor(&self) -> &BudgetPurposeDescriptor {
        &self.descriptor
    }

    /// Trusted quantitative-policy principal that approved the envelope.
    pub fn approver(&self) -> PrincipalId {
        self.approver
    }

    /// Host-owned purpose approval time.
    pub fn approved_at(&self) -> SystemTime {
        self.approved_at
    }

    /// Purpose admission lifetime.
    pub fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    /// Domain-separated digest of the complete quantitative-purpose admission.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Trusted domain allowed to approve an exact quantitative envelope for a purpose.
#[derive(Debug)]
pub struct BudgetPurposeAuthorityDomain {
    inner: AuthorityDomain,
    descriptor: BudgetPurposeDescriptor,
    rules: BudgetPurposeRules,
    budget_verifier: BudgetVerifier,
    policy_verifier: PolicyVerifier,
    execution_verifier: AuthorityVerifier,
    clock: Arc<BudgetPurposeClock>,
}

impl BudgetPurposeAuthorityDomain {
    /// Construct a quantitative-purpose policy root pinned to the exact budget,
    /// general-policy, and execution trust roots used by the host.
    pub fn new(
        principal: PrincipalId,
        descriptor: BudgetPurposeDescriptor,
        rules: BudgetPurposeRules,
        budget_verifier: BudgetVerifier,
        policy_verifier: PolicyVerifier,
        execution_verifier: AuthorityVerifier,
    ) -> Self {
        Self {
            inner: AuthorityDomain::new(principal),
            descriptor,
            rules,
            budget_verifier,
            policy_verifier,
            execution_verifier,
            clock: Arc::new(BudgetPurposeClock::new()),
        }
    }

    /// Quantitative-purpose trust-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Quantitative-purpose policy principal.
    pub fn principal(&self) -> PrincipalId {
        self.inner.principal()
    }

    /// Immutable quantitative-purpose policy descriptor.
    pub fn descriptor(&self) -> &BudgetPurposeDescriptor {
        &self.descriptor
    }

    /// Purpose lifetime rules.
    pub fn rules(&self) -> BudgetPurposeRules {
        self.rules
    }

    /// Verifier retained by the strongest host path.
    pub fn verifier(&self) -> BudgetPurposeVerifier {
        BudgetPurposeVerifier {
            inner: self.inner.verifier(),
            descriptor_digest: descriptor_digest(&self.descriptor, self.rules),
            clock: Arc::clone(&self.clock),
        }
    }

    /// Approve one exact conserved budget envelope for one exact policy/action/purpose.
    ///
    /// The existing budget lease is consumed into the returned affine wrapper;
    /// it is not cloned or recreated.
    #[allow(clippy::too_many_arguments)]
    pub fn approve<K: CapabilityKind>(
        &self,
        temporal_grant: &TemporalPolicyGrant<K>,
        budget_lease: BudgetLease,
        subject: PrincipalId,
        purpose_digest: [u8; 32],
        expires_at: Option<SystemTime>,
    ) -> Result<PurposeBoundBudgetLease, BudgetPurposeError> {
        let now = self.clock.now();
        let policy_evidence = temporal_grant.policy_grant().evidence();
        let temporal = temporal_grant.temporal_evidence();
        validate_policy_lineage(
            policy_evidence,
            temporal,
            &self.policy_verifier,
            &self.execution_verifier,
        )?;

        let action_binding = policy_evidence.receipt().action_binding();
        let scope = policy_evidence.receipt().scope().clone();
        budget_lease
            .validate_for(&self.budget_verifier, subject, &scope, action_binding)
            .map_err(BudgetPurposeError::Budget)?;
        if budget_lease.scope() != &scope {
            return Err(BudgetPurposeError::BudgetScopeNotExact {
                budget: budget_lease.scope().clone(),
                policy: scope,
            });
        }

        validate_derived_expiry(
            expires_at,
            budget_lease.expires_at(),
            temporal.execution_expires_at(),
            temporal.policy_expires_at(),
            self.rules,
            now,
        )?;

        let approved_at = now;
        let digest = compute_admission_digest(
            action_binding,
            subject,
            budget_lease.scope(),
            purpose_digest,
            budget_lease.allocation(),
            budget_lease.profile().digest(),
            budget_lease.lease_id(),
            budget_lease.domain_id(),
            budget_lease.epoch(),
            budget_lease.expires_at(),
            policy_evidence,
            temporal,
            &self.descriptor,
            self.inner.principal(),
            approved_at,
            expires_at,
        );
        let receipt = BudgetPurposeAdmissionReceipt {
            action_binding,
            subject,
            scope: budget_lease.scope().clone(),
            purpose_digest,
            allocation: budget_lease.allocation(),
            budget_profile_digest: budget_lease.profile().digest(),
            budget_lease_id: budget_lease.lease_id(),
            budget_domain: budget_lease.domain_id(),
            budget_epoch: budget_lease.epoch(),
            budget_expires_at: budget_lease.expires_at(),
            policy_admission_digest: policy_evidence.receipt().digest(),
            policy_binding: temporal_grant.policy_grant().policy_binding(),
            policy_domain: policy_evidence.policy_domain(),
            policy_epoch: policy_evidence.policy_epoch(),
            policy_attestation_grant_id: policy_evidence.policy_attestation_grant_id(),
            execution_domain: policy_evidence.execution_domain(),
            execution_epoch: policy_evidence.execution_epoch(),
            execution_grant_id: policy_evidence.execution_grant_id(),
            execution_expires_at: temporal.execution_expires_at(),
            descriptor: self.descriptor.clone(),
            approver: self.inner.principal(),
            approved_at,
            expires_at,
            digest,
        };
        let attestation = self.inner.issue_bound_one_shot::<Read>(
            subject,
            budget_lease.scope().clone(),
            expires_at,
            digest,
        );
        Ok(PurposeBoundBudgetLease {
            lease: budget_lease,
            receipt,
            attestation,
            descriptor_digest: descriptor_digest(&self.descriptor, self.rules),
        })
    }

    /// Revoke quantitative-purpose admissions from earlier epochs.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        self.inner.revoke_all()
    }
}

/// Host-retained quantitative-purpose trust anchor.
#[derive(Debug, Clone)]
pub struct BudgetPurposeVerifier {
    inner: AuthorityVerifier,
    descriptor_digest: [u8; 32],
    clock: Arc<BudgetPurposeClock>,
}

impl BudgetPurposeVerifier {
    /// Purpose authority-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Current purpose-policy revocation epoch.
    pub fn current_epoch(&self) -> AuthorityEpoch {
        self.inner.current_epoch()
    }

    /// Digest naming the quantitative-purpose policy + lifetime-rule configuration.
    pub fn descriptor_digest(&self) -> [u8; 32] {
        self.descriptor_digest
    }
}

/// Affine conserved budget lease carrying exact quantitative-purpose approval.
///
/// This type is neither `Clone` nor `Copy` because it owns the original budget
/// lease and one-shot purpose attestation.
#[derive(Debug)]
pub struct PurposeBoundBudgetLease {
    lease: BudgetLease,
    receipt: BudgetPurposeAdmissionReceipt,
    attestation: TrustedBoundOneShotCapability<Read>,
    descriptor_digest: [u8; 32],
}

impl PurposeBoundBudgetLease {
    /// Exact quantitative-purpose admission receipt.
    pub fn receipt(&self) -> &BudgetPurposeAdmissionReceipt {
        &self.receipt
    }

    /// Original conserved budget lease.
    pub fn budget_lease(&self) -> &BudgetLease {
        &self.lease
    }

    /// Purpose trust-domain identity.
    pub fn purpose_domain(&self) -> AuthorityDomainId {
        self.attestation.domain_id()
    }

    /// Purpose revocation epoch.
    pub fn purpose_epoch(&self) -> AuthorityEpoch {
        self.attestation.epoch()
    }

    /// Purpose-attestation grant id.
    pub fn purpose_attestation_grant_id(&self) -> GrantId {
        self.attestation.metadata().grant_id()
    }

    /// Stable digest naming the purpose policy + lifetime-rule configuration.
    pub fn descriptor_digest(&self) -> [u8; 32] {
        self.descriptor_digest
    }

    /// Validate the exact quantitative-purpose admission for the actual action
    /// and temporal policy grant about to be consumed.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_for<K: CapabilityKind>(
        &self,
        purpose_verifier: &BudgetPurposeVerifier,
        budget_verifier: &BudgetVerifier,
        temporal_grant: &TemporalPolicyGrant<K>,
        subject: PrincipalId,
        scope: &Scope,
        action_binding: [u8; 32],
    ) -> Result<(), BudgetPurposeError> {
        let now = purpose_verifier.clock.now();
        self.attestation
            .validate_with(&purpose_verifier.inner, now)
            .map_err(BudgetPurposeError::Trust)?;
        if self.descriptor_digest != purpose_verifier.descriptor_digest {
            return Err(BudgetPurposeError::DescriptorMismatch);
        }
        self.lease
            .validate_for(budget_verifier, subject, scope, action_binding)
            .map_err(BudgetPurposeError::Budget)?;
        if self.lease.scope() != scope || self.receipt.scope() != scope {
            return Err(BudgetPurposeError::BudgetScopeNotExact {
                budget: self.lease.scope().clone(),
                policy: scope.clone(),
            });
        }
        if self.receipt.subject() != subject
            || self.receipt.action_binding() != action_binding
            || self.receipt.budget_lease_id() != self.lease.lease_id()
            || self.receipt.budget_domain() != self.lease.domain_id()
            || self.receipt.budget_epoch() != self.lease.epoch()
            || self.receipt.budget_profile_digest() != self.lease.profile().digest()
            || self.receipt.allocation() != self.lease.allocation()
            || self.receipt.budget_expires_at() != self.lease.expires_at()
        {
            return Err(BudgetPurposeError::BudgetLineageMismatch);
        }

        let policy = temporal_grant.policy_grant().evidence();
        let temporal = temporal_grant.temporal_evidence();
        if self.receipt.policy_admission_digest() != policy.receipt().digest()
            || self.receipt.policy_binding() != temporal_grant.policy_grant().policy_binding()
            || self.receipt.policy_domain() != policy.policy_domain()
            || self.receipt.policy_epoch() != policy.policy_epoch()
            || self.receipt.policy_attestation_grant_id() != policy.policy_attestation_grant_id()
            || self.receipt.execution_domain() != temporal.execution_domain()
            || self.receipt.execution_epoch() != temporal.execution_epoch()
            || self.receipt.execution_grant_id() != temporal.execution_grant_id()
            || self.receipt.execution_expires_at() != temporal.execution_expires_at()
        {
            return Err(BudgetPurposeError::PolicyLineageMismatch);
        }

        let expected = compute_admission_digest(
            self.receipt.action_binding(),
            self.receipt.subject(),
            self.receipt.scope(),
            self.receipt.purpose_digest(),
            self.receipt.allocation(),
            self.receipt.budget_profile_digest(),
            self.receipt.budget_lease_id(),
            self.receipt.budget_domain(),
            self.receipt.budget_epoch(),
            self.receipt.budget_expires_at(),
            policy,
            temporal,
            self.receipt.descriptor(),
            self.receipt.approver(),
            self.receipt.approved_at(),
            self.receipt.expires_at(),
        );
        if expected != self.receipt.digest() || self.attestation.binding() != expected {
            return Err(BudgetPurposeError::AdmissionBindingMismatch);
        }
        Ok(())
    }

    fn into_parts(self) -> (BudgetLease, BudgetPurposeEvidence) {
        let evidence = BudgetPurposeEvidence {
            receipt: self.receipt,
            purpose_domain: self.attestation.domain_id(),
            purpose_epoch: self.attestation.epoch(),
            purpose_attestation_grant_id: self.attestation.metadata().grant_id(),
            descriptor_digest: self.descriptor_digest,
        };
        (self.lease, evidence)
    }
}

/// Immutable purpose-policy evidence retained after the affine lease is consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetPurposeEvidence {
    receipt: BudgetPurposeAdmissionReceipt,
    purpose_domain: AuthorityDomainId,
    purpose_epoch: AuthorityEpoch,
    purpose_attestation_grant_id: GrantId,
    descriptor_digest: [u8; 32],
}

impl BudgetPurposeEvidence {
    /// Exact quantitative-purpose admission receipt.
    pub fn receipt(&self) -> &BudgetPurposeAdmissionReceipt {
        &self.receipt
    }

    /// Purpose trust-domain identity.
    pub fn purpose_domain(&self) -> AuthorityDomainId {
        self.purpose_domain
    }

    /// Purpose revocation epoch consumed by the action.
    pub fn purpose_epoch(&self) -> AuthorityEpoch {
        self.purpose_epoch
    }

    /// Purpose-attestation grant id.
    pub fn purpose_attestation_grant_id(&self) -> GrantId {
        self.purpose_attestation_grant_id
    }

    /// Stable quantitative-purpose policy/rules digest.
    pub fn descriptor_digest(&self) -> [u8; 32] {
        self.descriptor_digest
    }
}

/// Strongest host wrapper requiring exact purpose-approved quantitative authority.
#[derive(Debug, Clone)]
pub struct PurposeGuardedRuntime {
    inner: IndependenceGuardedRuntime,
    purpose_verifier: BudgetPurposeVerifier,
    budget_verifier: BudgetVerifier,
}

impl PurposeGuardedRuntime {
    /// Construct the purpose-aware strongest host path.
    pub fn new(
        inner: IndependenceGuardedRuntime,
        purpose_verifier: BudgetPurposeVerifier,
        budget_verifier: BudgetVerifier,
    ) -> Self {
        Self {
            inner,
            purpose_verifier,
            budget_verifier,
        }
    }

    /// Quantitative-purpose trust domain selected by the host.
    pub fn purpose_domain(&self) -> AuthorityDomainId {
        self.purpose_verifier.domain_id()
    }

    /// Admit a resolved resource action into the purpose-aware lifecycle.
    pub fn admit_resolved<K: CapabilityKind, H>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        resource: ResolvedResource<H>,
        canonical_payload: &[u8],
    ) -> Result<PurposeGuardedAction<K, Proposed, H>, ResourceError> {
        let inner = self
            .inner
            .admit_resolved::<K, H>(actor, kind, resource, canonical_payload)?;
        Ok(PurposeGuardedAction {
            inner,
            purpose_verifier: self.purpose_verifier.clone(),
            budget_verifier: self.budget_verifier.clone(),
            purpose_evidence: None,
            final_receipt: None,
        })
    }
}

/// Action lifecycle carrying quantitative-purpose evidence through final resolution.
pub struct PurposeGuardedAction<K: CapabilityKind, S, H> {
    inner: IndependenceGuardedAction<K, S, H>,
    purpose_verifier: BudgetPurposeVerifier,
    budget_verifier: BudgetVerifier,
    purpose_evidence: Option<BudgetPurposeEvidence>,
    final_receipt: Option<PurposeBoundEvidenceReceipt>,
}

impl<K: CapabilityKind, S, H> PurposeGuardedAction<K, S, H> {
    /// Stable exact action identity.
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
}

impl<K: CapabilityKind, H> PurposeGuardedAction<K, Proposed, H> {
    /// Attach explicit action risk before policy and quantitative-purpose approval.
    pub fn assess(self, risk: ActionRisk) -> PurposeGuardedAction<K, RiskAssessed, H> {
        PurposeGuardedAction {
            inner: self.inner.assess(risk),
            purpose_verifier: self.purpose_verifier,
            budget_verifier: self.budget_verifier,
            purpose_evidence: None,
            final_receipt: None,
        }
    }
}

impl<K: CapabilityKind, H> PurposeGuardedAction<K, RiskAssessed, H> {
    /// Exact action binding targeted by both general and quantitative-purpose policy.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Risk classification evaluated by the general policy path.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Consume a temporally bounded exact policy grant plus a purpose-approved
    /// conserved budget lease.
    pub fn authorize(
        self,
        temporal_grant: TemporalPolicyGrant<K>,
        purpose_lease: PurposeBoundBudgetLease,
    ) -> Result<PurposeGuardedAction<K, Authorized, H>, BudgetPurposeAuthorizeError> {
        purpose_lease.validate_for(
            &self.purpose_verifier,
            &self.budget_verifier,
            &temporal_grant,
            self.inner.actor(),
            self.inner.descriptor().scope(),
            self.inner.authorization_binding(),
        )?;
        let (budget_lease, purpose_evidence) = purpose_lease.into_parts();
        let inner = self
            .inner
            .authorize(temporal_grant, budget_lease)
            .map_err(BudgetPurposeAuthorizeError::Inner)?;
        Ok(PurposeGuardedAction {
            inner,
            purpose_verifier: self.purpose_verifier,
            budget_verifier: self.budget_verifier,
            purpose_evidence: Some(purpose_evidence),
            final_receipt: None,
        })
    }
}

impl<K: CapabilityKind, H> PurposeGuardedAction<K, Authorized, H> {
    /// Quantitative-purpose evidence consumed at authorization.
    pub fn purpose_evidence(&self) -> &BudgetPurposeEvidence {
        self.purpose_evidence
            .as_ref()
            .expect("Authorized purpose action always carries purpose evidence")
    }

    /// Execute through the existing exact-preflight/effect-attempt boundary.
    pub fn execute_attempt_with<F>(
        self,
        attempt: F,
    ) -> Result<PurposeGuardedAction<K, Executed, H>, PurposeEffectAttemptFailure<K, H>>
    where
        F: FnOnce(&mut H) -> EffectAttemptOutcome,
    {
        let PurposeGuardedAction {
            inner,
            purpose_verifier,
            budget_verifier,
            purpose_evidence,
            final_receipt: _,
        } = self;
        match inner.execute_attempt_with(attempt) {
            Ok(inner) => Ok(PurposeGuardedAction {
                inner,
                purpose_verifier,
                budget_verifier,
                purpose_evidence,
                final_receipt: None,
            }),
            Err(IndependenceEffectAttemptFailure::Preflight { action, error }) => {
                Err(PurposeEffectAttemptFailure::Preflight {
                    action: PurposeGuardedAction {
                        inner: action,
                        purpose_verifier,
                        budget_verifier,
                        purpose_evidence,
                        final_receipt: None,
                    },
                    error,
                })
            }
            Err(IndependenceEffectAttemptFailure::RejectedBeforeAttempt { error }) => {
                Err(PurposeEffectAttemptFailure::RejectedBeforeAttempt { error })
            }
            Err(IndependenceEffectAttemptFailure::LineageFailedAfterAttempt {
                evidence,
                error,
            }) => Err(PurposeEffectAttemptFailure::LineageFailedAfterAttempt { evidence, error }),
        }
    }
}

impl<K: CapabilityKind, H> PurposeGuardedAction<K, Executed, H> {
    /// Exact observation binding committing transitively to the effect-attempt record.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Consume observation authority under the selected independence policy.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<Observe>,
        observation: Observation,
    ) -> Result<PurposeGuardedAction<K, Observed, H>, IndependenceObservationError> {
        let inner = self.inner.observe(observer, observation)?;
        Ok(PurposeGuardedAction {
            inner,
            purpose_verifier: self.purpose_verifier,
            budget_verifier: self.budget_verifier,
            purpose_evidence: self.purpose_evidence,
            final_receipt: None,
        })
    }
}

impl<K: CapabilityKind, H> PurposeGuardedAction<K, Observed, H> {
    /// Exact final-resolution binding.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        self.inner.resolution_binding(decision)
    }

    /// Resolve independently observed evidence and emit quantitative-purpose lineage.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<
        (
            PurposeGuardedAction<K, Resolved, H>,
            PurposeBoundEvidenceReceipt,
        ),
        IndependenceResolutionError,
    > {
        let purpose_evidence = self
            .purpose_evidence
            .expect("Observed purpose action always carries purpose evidence");
        let (inner, independence) = self.inner.resolve(grant, decision)?;
        let digest = compute_final_digest(&independence, &purpose_evidence);
        let receipt = PurposeBoundEvidenceReceipt {
            independence,
            purpose: purpose_evidence.clone(),
            digest,
        };
        Ok((
            PurposeGuardedAction {
                inner,
                purpose_verifier: self.purpose_verifier,
                budget_verifier: self.budget_verifier,
                purpose_evidence: Some(purpose_evidence),
                final_receipt: Some(receipt.clone()),
            },
            receipt,
        ))
    }
}

impl<K: CapabilityKind, H> PurposeGuardedAction<K, Resolved, H> {
    /// Final evidence joining general policy, quantity, purpose, effect, and independence lineage.
    pub fn purpose_receipt(&self) -> &PurposeBoundEvidenceReceipt {
        self.final_receipt
            .as_ref()
            .expect("Resolved purpose action always carries final evidence")
    }

    /// Return reserved quantitative capacity after final evidence has been retained.
    pub fn release_budget(self) -> Result<BudgetReleaseReceipt, BudgetError> {
        self.inner.release_budget()
    }
}

/// Final strongest evidence joining quantitative-purpose admission with the existing
/// policy/resource/budget/effect/independence lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurposeBoundEvidenceReceipt {
    independence: IndependenceEvidenceReceipt,
    purpose: BudgetPurposeEvidence,
    digest: [u8; 32],
}

impl PurposeBoundEvidenceReceipt {
    /// Existing separation-aware effect evidence.
    pub fn independence_receipt(&self) -> &IndependenceEvidenceReceipt {
        &self.independence
    }

    /// Quantitative-purpose approval evidence.
    pub fn purpose_evidence(&self) -> &BudgetPurposeEvidence {
        &self.purpose
    }

    /// Domain-separated digest joining the purpose admission to final effect evidence.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Failure while validating/consuming purpose-approved quantitative authority.
#[derive(Debug)]
pub enum BudgetPurposeAuthorizeError {
    /// Purpose/budget/policy cross-binding failed before lower authorization.
    Purpose(BudgetPurposeError),
    /// Existing strongest action authorization failed after purpose validation.
    /// Issue #140 tracks transactional recovery of quantitative authority on this path.
    Inner(EffectGuardedAuthorizeError),
}

impl From<BudgetPurposeError> for BudgetPurposeAuthorizeError {
    fn from(value: BudgetPurposeError) -> Self {
        Self::Purpose(value)
    }
}

impl fmt::Display for BudgetPurposeAuthorizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Purpose(error) => write!(f, "quantitative-purpose validation failed: {error}"),
            Self::Inner(error) => write!(f, "lower authorization failed: {error}"),
        }
    }
}

impl std::error::Error for BudgetPurposeAuthorizeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Purpose(error) => Some(error),
            Self::Inner(error) => Some(error),
        }
    }
}

/// Effect-attempt failure preserving purpose context on recoverable exact preflight rejection.
pub enum PurposeEffectAttemptFailure<K: CapabilityKind, H> {
    /// Exact execution preflight rejected before lower effect delegation.
    Preflight {
        /// Recoverable authorized action retaining quantitative-purpose evidence.
        action: PurposeGuardedAction<K, Authorized, H>,
        /// Exact preflight error.
        error: crate::ExecutionPreflightError,
    },
    /// Lower policy/resource/budget guard rejected before user adapter entry.
    RejectedBeforeAttempt {
        /// Existing lower error.
        error: crate::EffectInnerExecutionError,
    },
    /// Adapter boundary was entered and attempt evidence exists, but lower lineage failed afterward.
    LineageFailedAfterAttempt {
        /// Preserved attempt evidence.
        evidence: EffectAttemptEvidence,
        /// Existing lower error.
        error: crate::EffectInnerExecutionError,
    },
}

impl<K: CapabilityKind, H> PurposeEffectAttemptFailure<K, H> {
    /// Whether the user adapter boundary was entered.
    pub fn adapter_was_entered(&self) -> bool {
        matches!(self, Self::LineageFailedAfterAttempt { .. })
    }

    /// Preserved effect-attempt evidence when entry occurred.
    pub fn attempt_evidence(&self) -> Option<&EffectAttemptEvidence> {
        match self {
            Self::LineageFailedAfterAttempt { evidence, .. } => Some(evidence),
            Self::Preflight { .. } | Self::RejectedBeforeAttempt { .. } => None,
        }
    }
}

impl<K: CapabilityKind, H> fmt::Debug for PurposeEffectAttemptFailure<K, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preflight { error, .. } => f
                .debug_struct("PurposeEffectAttemptFailure::Preflight")
                .field("error", error)
                .field("action", &"<authority-bearing action retained>")
                .finish(),
            Self::RejectedBeforeAttempt { error } => f
                .debug_struct("PurposeEffectAttemptFailure::RejectedBeforeAttempt")
                .field("error", error)
                .finish(),
            Self::LineageFailedAfterAttempt { evidence, error } => f
                .debug_struct("PurposeEffectAttemptFailure::LineageFailedAfterAttempt")
                .field("evidence", evidence)
                .field("error", error)
                .finish(),
        }
    }
}

/// Quantitative-purpose admission failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetPurposeError {
    /// Invalid quantitative-policy family label.
    InvalidPolicyFamily(String),
    /// Purpose attestation belongs to another domain/epoch or is expired.
    Trust(TrustError),
    /// Conserved budget validation failed.
    Budget(BudgetError),
    /// Budget lease scope was not exactly the policy-admitted action scope.
    BudgetScopeNotExact {
        /// Budget lease scope.
        budget: Scope,
        /// General-policy action scope.
        policy: Scope,
    },
    /// Purpose policy configuration digest differs from the host-selected verifier.
    DescriptorMismatch,
    /// Budget fields in the purpose receipt do not match the owned original lease.
    BudgetLineageMismatch,
    /// General/temporal policy fields in the purpose receipt do not match the grant being consumed.
    PolicyLineageMismatch,
    /// General policy evidence comes from another policy domain/epoch.
    WrongPolicyLineage,
    /// Execution evidence comes from another execution domain/epoch.
    WrongExecutionLineage,
    /// Temporal and ordinary policy evidence disagree internally.
    TemporalPolicyMismatch,
    /// Purpose admission expiry is already stale.
    PurposeAlreadyExpired {
        /// Requested purpose expiry.
        expiry: SystemTime,
        /// Host-owned validation time.
        now: SystemTime,
    },
    /// Purpose admission lifetime would widen one of its finite parent bounds.
    PurposeExpiryWidening {
        /// Requested purpose expiry.
        requested: Option<SystemTime>,
        /// Finite parent bound that would be exceeded.
        parent: SystemTime,
    },
    /// Strict purpose policy forbids an unbounded purpose admission.
    UnboundedPurposeForbidden,
    /// Purpose receipt digest/attestation binding is inconsistent.
    AdmissionBindingMismatch,
}

impl fmt::Display for BudgetPurposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicyFamily(value) => {
                write!(f, "invalid budget-purpose policy family: {value:?}")
            }
            Self::Trust(error) => write!(f, "budget-purpose trust validation failed: {error}"),
            Self::Budget(error) => write!(f, "budget validation failed: {error}"),
            Self::BudgetScopeNotExact { .. } => write!(
                f,
                "budget scope is not the exact policy-admitted action scope"
            ),
            Self::DescriptorMismatch => write!(f, "budget-purpose policy descriptor mismatch"),
            Self::BudgetLineageMismatch => write!(
                f,
                "budget-purpose receipt does not match owned budget lease"
            ),
            Self::PolicyLineageMismatch => write!(
                f,
                "budget-purpose receipt does not match consumed policy lineage"
            ),
            Self::WrongPolicyLineage => write!(
                f,
                "general policy lineage is not current for the pinned policy root"
            ),
            Self::WrongExecutionLineage => write!(
                f,
                "execution lineage is not current for the pinned execution root"
            ),
            Self::TemporalPolicyMismatch => {
                write!(f, "temporal and ordinary policy evidence disagree")
            }
            Self::PurposeAlreadyExpired { .. } => {
                write!(f, "requested budget-purpose admission is already expired")
            }
            Self::PurposeExpiryWidening { .. } => write!(
                f,
                "budget-purpose lifetime would widen a finite parent authority"
            ),
            Self::UnboundedPurposeForbidden => {
                write!(f, "strict budget-purpose policy requires a finite lifetime")
            }
            Self::AdmissionBindingMismatch => {
                write!(f, "budget-purpose admission binding is inconsistent")
            }
        }
    }
}

impl std::error::Error for BudgetPurposeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Trust(error) => Some(error),
            Self::Budget(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_policy_lineage(
    policy: &PolicyAuthorizationEvidence,
    temporal: &TemporalDerivationEvidence,
    policy_verifier: &PolicyVerifier,
    execution_verifier: &AuthorityVerifier,
) -> Result<(), BudgetPurposeError> {
    if policy.policy_domain() != policy_verifier.domain_id()
        || policy.policy_epoch() != policy_verifier.current_epoch()
    {
        return Err(BudgetPurposeError::WrongPolicyLineage);
    }
    if policy.execution_domain() != execution_verifier.domain_id()
        || policy.execution_epoch() != execution_verifier.current_epoch()
    {
        return Err(BudgetPurposeError::WrongExecutionLineage);
    }
    if temporal.policy_domain() != policy.policy_domain()
        || temporal.policy_epoch() != policy.policy_epoch()
        || temporal.policy_attestation_grant_id() != policy.policy_attestation_grant_id()
        || temporal.execution_domain() != policy.execution_domain()
        || temporal.execution_epoch() != policy.execution_epoch()
        || temporal.execution_grant_id() != policy.execution_grant_id()
        || !temporal.preserves_finite_parent_bound()
    {
        return Err(BudgetPurposeError::TemporalPolicyMismatch);
    }
    Ok(())
}

fn validate_derived_expiry(
    requested: Option<SystemTime>,
    budget_expiry: Option<SystemTime>,
    execution_expiry: Option<SystemTime>,
    policy_expiry: Option<SystemTime>,
    rules: BudgetPurposeRules,
    now: SystemTime,
) -> Result<(), BudgetPurposeError> {
    if let Some(expiry) = requested {
        if expiry < now {
            return Err(BudgetPurposeError::PurposeAlreadyExpired { expiry, now });
        }
    }
    for parent in [budget_expiry, execution_expiry, policy_expiry]
        .into_iter()
        .flatten()
    {
        match requested {
            Some(child) if child <= parent => {}
            _ => return Err(BudgetPurposeError::PurposeExpiryWidening { requested, parent }),
        }
    }
    if requested.is_none()
        && budget_expiry.is_none()
        && execution_expiry.is_none()
        && policy_expiry.is_none()
        && !rules.allow_unbounded_purpose()
    {
        return Err(BudgetPurposeError::UnboundedPurposeForbidden);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_admission_digest(
    action_binding: [u8; 32],
    subject: PrincipalId,
    scope: &Scope,
    purpose_digest: [u8; 32],
    allocation: BudgetQuantities,
    budget_profile_digest: [u8; 32],
    budget_lease_id: GrantId,
    budget_domain: AuthorityDomainId,
    budget_epoch: AuthorityEpoch,
    budget_expires_at: Option<SystemTime>,
    policy: &PolicyAuthorizationEvidence,
    temporal: &TemporalDerivationEvidence,
    descriptor: &BudgetPurposeDescriptor,
    approver: PrincipalId,
    approved_at: SystemTime,
    expires_at: Option<SystemTime>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/budget-purpose-admission-v1\0");
    hash_field(&mut hasher, &action_binding);
    hash_field(&mut hasher, subject.as_uuid().as_bytes());
    hash_scope(&mut hasher, scope);
    hash_field(&mut hasher, &purpose_digest);
    for dimension in BudgetDimension::ALL {
        hash_field(&mut hasher, &[dimension as u8]);
        hash_field(&mut hasher, &allocation.get(dimension).to_le_bytes());
    }
    hash_field(&mut hasher, &budget_profile_digest);
    hash_field(&mut hasher, budget_lease_id.as_uuid().as_bytes());
    hash_field(&mut hasher, budget_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, &budget_epoch.value().to_le_bytes());
    hash_optional_time(&mut hasher, budget_expires_at);
    hash_field(&mut hasher, &policy.receipt().digest());
    hash_field(&mut hasher, &policy.policy_binding());
    hash_field(&mut hasher, policy.policy_domain().as_uuid().as_bytes());
    hash_field(&mut hasher, &policy.policy_epoch().value().to_le_bytes());
    hash_field(
        &mut hasher,
        policy.policy_attestation_grant_id().as_uuid().as_bytes(),
    );
    hash_field(
        &mut hasher,
        temporal.execution_domain().as_uuid().as_bytes(),
    );
    hash_field(
        &mut hasher,
        &temporal.execution_epoch().value().to_le_bytes(),
    );
    hash_field(
        &mut hasher,
        temporal.execution_grant_id().as_uuid().as_bytes(),
    );
    hash_optional_time(&mut hasher, temporal.execution_expires_at());
    hash_field(&mut hasher, descriptor.family().as_bytes());
    hash_field(&mut hasher, &descriptor.version().to_le_bytes());
    hash_field(&mut hasher, &descriptor.policy_digest());
    hash_field(&mut hasher, &descriptor.schema_version().to_le_bytes());
    hash_field(&mut hasher, approver.as_uuid().as_bytes());
    hash_system_time(&mut hasher, approved_at);
    hash_optional_time(&mut hasher, expires_at);
    *hasher.finalize().as_bytes()
}

fn descriptor_digest(descriptor: &BudgetPurposeDescriptor, rules: BudgetPurposeRules) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/budget-purpose-policy-v1\0");
    hash_field(&mut hasher, descriptor.family().as_bytes());
    hash_field(&mut hasher, &descriptor.version().to_le_bytes());
    hash_field(&mut hasher, &descriptor.policy_digest());
    hash_field(&mut hasher, &descriptor.schema_version().to_le_bytes());
    hash_field(&mut hasher, &[u8::from(rules.allow_unbounded_purpose())]);
    *hasher.finalize().as_bytes()
}

fn compute_final_digest(
    independence: &IndependenceEvidenceReceipt,
    purpose: &BudgetPurposeEvidence,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/purpose-bound-final-evidence-v1\0");
    hash_field(&mut hasher, &independence.digest());
    hash_field(&mut hasher, &purpose.receipt().digest());
    hash_field(&mut hasher, purpose.purpose_domain().as_uuid().as_bytes());
    hash_field(&mut hasher, &purpose.purpose_epoch().value().to_le_bytes());
    hash_field(
        &mut hasher,
        purpose.purpose_attestation_grant_id().as_uuid().as_bytes(),
    );
    hash_field(&mut hasher, &purpose.descriptor_digest());
    *hasher.finalize().as_bytes()
}

fn hash_scope(hasher: &mut blake3::Hasher, scope: &Scope) {
    hash_field(hasher, scope.namespace().as_bytes());
    for segment in scope.segments() {
        hash_field(hasher, segment.as_bytes());
    }
}

fn hash_optional_time(hasher: &mut blake3::Hasher, time: Option<SystemTime>) {
    match time {
        Some(time) => {
            hash_field(hasher, &[1]);
            hash_system_time(hasher, time);
        }
        None => hash_field(hasher, &[0]),
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
struct BudgetPurposeClock {
    last: Mutex<SystemTime>,
}

impl BudgetPurposeClock {
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
