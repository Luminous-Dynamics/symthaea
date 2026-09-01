// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict temporal derivation for policy-bound execution authority.
//!
//! The lower-level [`crate::policy`] module deliberately exposes deterministic
//! policy mechanics. This module adds the stricter host-facing rule required by
//! state-changing autonomous execution: authority derived from a policy
//! admission may narrow its lifetime, but may never silently outlive the
//! admission that justified it.
//!
//! The facade also rejects already-stale admission/execution requests using a
//! host-owned non-decreasing wall-clock floor. Unbounded lifetimes are explicit
//! trusted configuration rather than an accidental consequence of `None`.
//!
//! Temporal derivation evidence is retained alongside the underlying
//! [`crate::PolicyGrant`]. Existing resource/budget layers can consume the inner
//! grant after trusted code explicitly separates the pair; a later composition
//! tranche can carry [`TemporalDerivationEvidence`] directly into final receipts.

use crate::ActionRisk;
use crate::capability::{CapabilityKind, GrantId, PrincipalId, Scope};
use crate::policy::{
    ApprovalEvidence, PolicyAdmission, PolicyAdmissionReceipt, PolicyDescriptor, PolicyError,
    PolicyEvaluatorDomain, PolicyExecutionDomain, PolicyGrant, PolicyMode, PolicyVerifier,
};
use crate::trusted::{AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Trusted configuration governing whether unbounded policy/execution lifetimes
/// are permitted at the strict temporal boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalPolicyRules {
    allow_unbounded_admission: bool,
    allow_unbounded_execution: bool,
}

impl TemporalPolicyRules {
    /// Strict production-oriented rules: both policy admissions and derived
    /// execution grants require finite lifetimes.
    pub const fn strict() -> Self {
        Self {
            allow_unbounded_admission: false,
            allow_unbounded_execution: false,
        }
    }

    /// Construct explicit trusted lifetime rules.
    pub const fn new(allow_unbounded_admission: bool, allow_unbounded_execution: bool) -> Self {
        Self {
            allow_unbounded_admission,
            allow_unbounded_execution,
        }
    }

    /// Whether policy admissions may omit an expiry.
    pub const fn allow_unbounded_admission(self) -> bool {
        self.allow_unbounded_admission
    }

    /// Whether execution grants derived from an unbounded admission may omit an expiry.
    pub const fn allow_unbounded_execution(self) -> bool {
        self.allow_unbounded_execution
    }
}

impl Default for TemporalPolicyRules {
    fn default() -> Self {
        Self::strict()
    }
}

/// Policy admission whose attestation lifetime is retained explicitly for later
/// derivation checks.
#[derive(Debug)]
pub struct TemporalPolicyAdmission {
    inner: PolicyAdmission,
    expires_at: Option<SystemTime>,
}

impl TemporalPolicyAdmission {
    /// Immutable policy decision receipt.
    pub fn receipt(&self) -> &PolicyAdmissionReceipt {
        self.inner.receipt()
    }

    /// Policy evaluator trust domain that attested the admission.
    pub fn evaluator_domain(&self) -> AuthorityDomainId {
        self.inner.evaluator_domain()
    }

    /// Policy evaluator revocation epoch in which the admission was created.
    pub fn evaluator_epoch(&self) -> AuthorityEpoch {
        self.inner.evaluator_epoch()
    }

    /// One-shot policy-attestation grant identity.
    pub fn attestation_grant_id(&self) -> GrantId {
        self.inner.attestation_grant_id()
    }

    /// Upper lifetime bound of the policy admission.
    pub fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }
}

/// Evidence that an exact execution grant was derived without widening the
/// temporal bound of its policy admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalDerivationEvidence {
    policy_domain: AuthorityDomainId,
    policy_epoch: AuthorityEpoch,
    policy_attestation_grant_id: GrantId,
    policy_decided_at: SystemTime,
    policy_expires_at: Option<SystemTime>,
    execution_domain: AuthorityDomainId,
    execution_epoch: AuthorityEpoch,
    execution_grant_id: GrantId,
    execution_expires_at: Option<SystemTime>,
    derived_at: SystemTime,
}

impl TemporalDerivationEvidence {
    /// Policy evaluator trust domain.
    pub fn policy_domain(&self) -> AuthorityDomainId {
        self.policy_domain
    }

    /// Policy evaluator revocation epoch.
    pub fn policy_epoch(&self) -> AuthorityEpoch {
        self.policy_epoch
    }

    /// Policy admission attestation grant identity.
    pub fn policy_attestation_grant_id(&self) -> GrantId {
        self.policy_attestation_grant_id
    }

    /// Host-owned policy decision time.
    pub fn policy_decided_at(&self) -> SystemTime {
        self.policy_decided_at
    }

    /// Maximum lifetime of the policy admission.
    pub fn policy_expires_at(&self) -> Option<SystemTime> {
        self.policy_expires_at
    }

    /// Execution authority trust domain.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_domain
    }

    /// Execution authority revocation epoch.
    pub fn execution_epoch(&self) -> AuthorityEpoch {
        self.execution_epoch
    }

    /// Exact execution grant identity.
    pub fn execution_grant_id(&self) -> GrantId {
        self.execution_grant_id
    }

    /// Effective lifetime of the derived execution grant.
    pub fn execution_expires_at(&self) -> Option<SystemTime> {
        self.execution_expires_at
    }

    /// Host-owned time at which temporal derivation was admitted.
    pub fn derived_at(&self) -> SystemTime {
        self.derived_at
    }

    /// Return true when the recorded execution lifetime is demonstrably no
    /// wider than the recorded finite policy lifetime.
    pub fn preserves_finite_parent_bound(&self) -> bool {
        match self.policy_expires_at {
            Some(parent) => self
                .execution_expires_at
                .is_some_and(|child| child <= parent),
            None => true,
        }
    }
}

/// Exact execution grant plus its temporal-derivation evidence.
///
/// This type intentionally implements neither `Copy` nor `Clone` because the
/// underlying policy grant is one-shot authority.
#[derive(Debug)]
pub struct TemporalPolicyGrant<K: CapabilityKind> {
    inner: PolicyGrant<K>,
    temporal_evidence: TemporalDerivationEvidence,
}

impl<K: CapabilityKind> TemporalPolicyGrant<K> {
    /// Existing policy/execution provenance retained by the underlying grant.
    pub fn policy_grant(&self) -> &PolicyGrant<K> {
        &self.inner
    }

    /// Temporal derivation evidence retained by this strict facade.
    pub fn temporal_evidence(&self) -> &TemporalDerivationEvidence {
        &self.temporal_evidence
    }

    /// Explicitly separate the already-validated underlying execution grant and
    /// temporal evidence for integration with existing resource/budget layers.
    ///
    /// Trusted integration code should preserve the returned evidence alongside
    /// the action lineage rather than discarding it.
    pub fn into_parts(self) -> (PolicyGrant<K>, TemporalDerivationEvidence) {
        (self.inner, self.temporal_evidence)
    }
}

/// Strict policy evaluator that remembers the exact lifetime supplied to each
/// opaque admission and rejects already-stale/unbounded requests according to
/// trusted host configuration.
#[derive(Debug)]
pub struct TemporalPolicyEvaluatorDomain {
    inner: PolicyEvaluatorDomain,
    rules: TemporalPolicyRules,
    clock: Arc<TemporalClock>,
}

impl TemporalPolicyEvaluatorDomain {
    /// Construct a strict temporal evaluator around a normal policy evaluator.
    pub fn new(
        principal: PrincipalId,
        descriptor: PolicyDescriptor,
        rules: TemporalPolicyRules,
    ) -> Self {
        Self {
            inner: PolicyEvaluatorDomain::new(principal, descriptor),
            rules,
            clock: Arc::new(TemporalClock::new()),
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

    /// Immutable policy descriptor.
    pub fn descriptor(&self) -> &PolicyDescriptor {
        self.inner.descriptor()
    }

    /// Trusted temporal rules used by this evaluator.
    pub fn rules(&self) -> TemporalPolicyRules {
        self.rules
    }

    /// Policy verifier supplied to the execution-authority side of the boundary.
    pub fn verifier(&self) -> PolicyVerifier {
        self.inner.verifier()
    }

    /// Record one exact policy admission while retaining and validating its
    /// explicit lifetime bound.
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
    ) -> Result<TemporalPolicyAdmission, TemporalPolicyError> {
        let now = self.clock.now();
        match expires_at {
            None if !self.rules.allow_unbounded_admission => {
                return Err(TemporalPolicyError::UnboundedAdmissionForbidden);
            }
            Some(expiry) if expiry < now => {
                return Err(TemporalPolicyError::AdmissionAlreadyExpired { expiry, now });
            }
            _ => {}
        }

        let inner = self.inner.admit(
            action_binding,
            scope,
            risk,
            mode,
            approvals,
            evidence_snapshot_digest,
            obligations_digest,
            emergency_state_digest,
            expires_at,
        );
        Ok(TemporalPolicyAdmission { inner, expires_at })
    }

    /// Revoke admissions from earlier evaluator epochs.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        self.inner.revoke_all()
    }
}

/// Strict execution-authority root that prevents temporal widening when deriving
/// an exact [`PolicyGrant`] from a [`TemporalPolicyAdmission`].
#[derive(Debug)]
pub struct TemporalPolicyExecutionDomain {
    inner: PolicyExecutionDomain,
    rules: TemporalPolicyRules,
    clock: Arc<TemporalClock>,
}

impl TemporalPolicyExecutionDomain {
    /// Construct a temporal execution domain pinned to one policy evaluator.
    pub fn new(
        principal: PrincipalId,
        policy_verifier: PolicyVerifier,
        rules: TemporalPolicyRules,
    ) -> Self {
        Self {
            inner: PolicyExecutionDomain::new(principal, policy_verifier),
            rules,
            clock: Arc::new(TemporalClock::new()),
        }
    }

    /// Execution authority trust-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Policy evaluator domain trusted by this execution authority.
    pub fn policy_domain(&self) -> AuthorityDomainId {
        self.inner.policy_domain()
    }

    /// Verifier supplied to the strict runtime's execution boundary.
    pub fn verifier(&self) -> AuthorityVerifier {
        self.inner.verifier()
    }

    /// Trusted temporal rules used by this execution authority.
    pub fn rules(&self) -> TemporalPolicyRules {
        self.rules
    }

    /// Mint exact execution authority only when the requested lifetime is
    /// current and no wider than the policy admission that justifies it.
    pub fn issue<K: CapabilityKind>(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
        action_binding: [u8; 32],
        admission: TemporalPolicyAdmission,
    ) -> Result<TemporalPolicyGrant<K>, TemporalPolicyError> {
        let now = self.clock.now();

        if let Some(policy_expiry) = admission.expires_at {
            if policy_expiry < now {
                return Err(TemporalPolicyError::AdmissionAlreadyExpired {
                    expiry: policy_expiry,
                    now,
                });
            }
        }

        if let Some(execution_expiry) = expires_at {
            if execution_expiry < now {
                return Err(TemporalPolicyError::ExecutionAlreadyExpired {
                    expiry: execution_expiry,
                    now,
                });
            }
        }

        match admission.expires_at {
            Some(policy_expiry) => match expires_at {
                Some(execution_expiry) if execution_expiry <= policy_expiry => {}
                _ => {
                    return Err(TemporalPolicyError::ExpiryWidening {
                        policy_expiry: Some(policy_expiry),
                        requested_execution_expiry: expires_at,
                    });
                }
            },
            None if expires_at.is_none() && !self.rules.allow_unbounded_execution => {
                return Err(TemporalPolicyError::UnboundedExecutionForbidden);
            }
            None => {}
        }

        let policy_domain = admission.evaluator_domain();
        let policy_epoch = admission.evaluator_epoch();
        let policy_attestation_grant_id = admission.attestation_grant_id();
        let policy_decided_at = admission.receipt().decided_at();
        let policy_expires_at = admission.expires_at;

        let inner = self
            .inner
            .issue::<K>(subject, scope, expires_at, action_binding, admission.inner)
            .map_err(TemporalPolicyError::Policy)?;
        let evidence = inner.evidence();
        let temporal_evidence = TemporalDerivationEvidence {
            policy_domain,
            policy_epoch,
            policy_attestation_grant_id,
            policy_decided_at,
            policy_expires_at,
            execution_domain: evidence.execution_domain(),
            execution_epoch: evidence.execution_epoch(),
            execution_grant_id: evidence.execution_grant_id(),
            execution_expires_at: expires_at,
            derived_at: now,
        };

        debug_assert_eq!(temporal_evidence.policy_domain(), evidence.policy_domain());
        debug_assert_eq!(
            temporal_evidence.policy_attestation_grant_id(),
            evidence.policy_attestation_grant_id()
        );
        debug_assert!(temporal_evidence.preserves_finite_parent_bound());

        Ok(TemporalPolicyGrant {
            inner,
            temporal_evidence,
        })
    }

    /// Revoke state-changing execution grants from earlier execution epochs.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        self.inner.revoke_all()
    }
}

/// Failure to create or derive temporally bounded policy authority.
#[derive(Debug)]
pub enum TemporalPolicyError {
    /// Strict evaluator configuration requires a finite policy admission lifetime.
    UnboundedAdmissionForbidden,
    /// Strict execution configuration forbids an unbounded derived grant.
    UnboundedExecutionForbidden,
    /// Requested policy admission expiry was already stale.
    AdmissionAlreadyExpired {
        /// Stale requested/admission expiry.
        expiry: SystemTime,
        /// Host-owned validation time.
        now: SystemTime,
    },
    /// Requested execution expiry was already stale.
    ExecutionAlreadyExpired {
        /// Stale execution expiry.
        expiry: SystemTime,
        /// Host-owned validation time.
        now: SystemTime,
    },
    /// A derived execution lifetime would exceed a finite policy lifetime.
    ExpiryWidening {
        /// Finite policy admission expiry, when present.
        policy_expiry: Option<SystemTime>,
        /// Requested execution expiry.
        requested_execution_expiry: Option<SystemTime>,
    },
    /// Lower-level policy validation or minting failed.
    Policy(PolicyError),
}

impl fmt::Display for TemporalPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnboundedAdmissionForbidden => {
                write!(
                    f,
                    "strict temporal policy requires a finite admission lifetime"
                )
            }
            Self::UnboundedExecutionForbidden => {
                write!(
                    f,
                    "strict temporal policy forbids an unbounded execution grant"
                )
            }
            Self::AdmissionAlreadyExpired { .. } => {
                write!(f, "policy admission lifetime is already expired")
            }
            Self::ExecutionAlreadyExpired { .. } => {
                write!(f, "requested execution lifetime is already expired")
            }
            Self::ExpiryWidening { .. } => {
                write!(
                    f,
                    "derived execution authority would outlive policy admission"
                )
            }
            Self::Policy(error) => write!(f, "policy derivation failed: {error}"),
        }
    }
}

impl std::error::Error for TemporalPolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Policy(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PolicyError> for TemporalPolicyError {
    fn from(value: PolicyError) -> Self {
        Self::Policy(value)
    }
}

#[derive(Debug)]
struct TemporalClock {
    last: Mutex<SystemTime>,
}

impl TemporalClock {
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
    use crate::Write;
    use std::time::Duration;

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    fn descriptor() -> PolicyDescriptor {
        PolicyDescriptor::new("temporal-test", 1, [1; 32], 1).unwrap()
    }

    fn approvals() -> ApprovalEvidence {
        ApprovalEvidence::new([2; 32], [3; 32], true)
    }

    fn admission(
        evaluator: &TemporalPolicyEvaluatorDomain,
        binding: [u8; 32],
        expires_at: Option<SystemTime>,
    ) -> TemporalPolicyAdmission {
        evaluator
            .admit(
                binding,
                scope(),
                ActionRisk::Reversible,
                PolicyMode::Autonomous,
                approvals(),
                [4; 32],
                [5; 32],
                [6; 32],
                expires_at,
            )
            .unwrap()
    }

    #[test]
    fn strict_rules_reject_unbounded_policy_admission() {
        let evaluator = TemporalPolicyEvaluatorDomain::new(
            PrincipalId::new(),
            descriptor(),
            TemporalPolicyRules::strict(),
        );
        let result = evaluator.admit(
            [7; 32],
            scope(),
            ActionRisk::Reversible,
            PolicyMode::Autonomous,
            approvals(),
            [8; 32],
            [9; 32],
            [10; 32],
            None,
        );
        assert!(matches!(
            result,
            Err(TemporalPolicyError::UnboundedAdmissionForbidden)
        ));
    }

    #[test]
    fn finite_admission_cannot_mint_unbounded_execution() {
        let rules = TemporalPolicyRules::new(true, true);
        let evaluator = TemporalPolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(), rules);
        let execution =
            TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
        let binding = [11; 32];
        let policy_expiry = SystemTime::now() + Duration::from_secs(60);
        let result = execution.issue::<Write>(
            PrincipalId::new(),
            scope(),
            None,
            binding,
            admission(&evaluator, binding, Some(policy_expiry)),
        );
        assert!(matches!(
            result,
            Err(TemporalPolicyError::ExpiryWidening { .. })
        ));
    }

    #[test]
    fn finite_admission_cannot_mint_later_execution_expiry() {
        let rules = TemporalPolicyRules::new(true, true);
        let evaluator = TemporalPolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(), rules);
        let execution =
            TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
        let binding = [12; 32];
        let policy_expiry = SystemTime::now() + Duration::from_secs(60);
        let result = execution.issue::<Write>(
            PrincipalId::new(),
            scope(),
            Some(policy_expiry + Duration::from_secs(1)),
            binding,
            admission(&evaluator, binding, Some(policy_expiry)),
        );
        assert!(matches!(
            result,
            Err(TemporalPolicyError::ExpiryWidening { .. })
        ));
    }

    #[test]
    fn equal_or_earlier_finite_execution_expiry_is_accepted() {
        for offset in [0_u64, 30_u64] {
            let rules = TemporalPolicyRules::new(true, true);
            let evaluator =
                TemporalPolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(), rules);
            let execution =
                TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
            let binding = [13 + offset as u8; 32];
            let policy_expiry = SystemTime::now() + Duration::from_secs(90);
            let execution_expiry = policy_expiry - Duration::from_secs(offset);
            let grant = execution
                .issue::<Write>(
                    PrincipalId::new(),
                    scope(),
                    Some(execution_expiry),
                    binding,
                    admission(&evaluator, binding, Some(policy_expiry)),
                )
                .unwrap();
            assert_eq!(
                grant.temporal_evidence().policy_expires_at(),
                Some(policy_expiry)
            );
            assert_eq!(
                grant.temporal_evidence().execution_expires_at(),
                Some(execution_expiry)
            );
            assert!(grant.temporal_evidence().preserves_finite_parent_bound());
        }
    }

    #[test]
    fn already_stale_execution_request_is_rejected_before_minting() {
        let rules = TemporalPolicyRules::new(true, true);
        let evaluator = TemporalPolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(), rules);
        let execution =
            TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
        let binding = [15; 32];
        let policy_expiry = SystemTime::now() + Duration::from_secs(60);
        let stale_execution = SystemTime::now() - Duration::from_secs(1);
        let result = execution.issue::<Write>(
            PrincipalId::new(),
            scope(),
            Some(stale_execution),
            binding,
            admission(&evaluator, binding, Some(policy_expiry)),
        );
        assert!(matches!(
            result,
            Err(TemporalPolicyError::ExecutionAlreadyExpired { .. })
        ));
    }

    #[test]
    fn explicit_unbounded_policy_can_derive_finite_or_unbounded_execution() {
        let rules = TemporalPolicyRules::new(true, true);

        for execution_expiry in [Some(SystemTime::now() + Duration::from_secs(60)), None] {
            let evaluator =
                TemporalPolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(), rules);
            let execution =
                TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
            let binding = if execution_expiry.is_some() {
                [16; 32]
            } else {
                [17; 32]
            };
            let grant = execution
                .issue::<Write>(
                    PrincipalId::new(),
                    scope(),
                    execution_expiry,
                    binding,
                    admission(&evaluator, binding, None),
                )
                .unwrap();
            assert_eq!(
                grant.temporal_evidence().execution_expires_at(),
                execution_expiry
            );
        }
    }

    #[test]
    fn unbounded_execution_requires_explicit_execution_rule() {
        let evaluator_rules = TemporalPolicyRules::new(true, false);
        let evaluator =
            TemporalPolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(), evaluator_rules);
        let execution = TemporalPolicyExecutionDomain::new(
            PrincipalId::new(),
            evaluator.verifier(),
            evaluator_rules,
        );
        let binding = [18; 32];
        let result = execution.issue::<Write>(
            PrincipalId::new(),
            scope(),
            None,
            binding,
            admission(&evaluator, binding, None),
        );
        assert!(matches!(
            result,
            Err(TemporalPolicyError::UnboundedExecutionForbidden)
        ));
    }

    #[test]
    fn temporal_evidence_matches_underlying_policy_lineage() {
        let rules = TemporalPolicyRules::new(true, true);
        let evaluator = TemporalPolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(), rules);
        let execution =
            TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
        let binding = [19; 32];
        let policy_expiry = SystemTime::now() + Duration::from_secs(60);
        let execution_expiry = policy_expiry - Duration::from_secs(1);
        let grant = execution
            .issue::<Write>(
                PrincipalId::new(),
                scope(),
                Some(execution_expiry),
                binding,
                admission(&evaluator, binding, Some(policy_expiry)),
            )
            .unwrap();

        assert_eq!(
            grant.temporal_evidence().policy_domain(),
            grant.policy_grant().evidence().policy_domain()
        );
        assert_eq!(
            grant.temporal_evidence().execution_domain(),
            grant.policy_grant().evidence().execution_domain()
        );
        assert_eq!(
            grant.temporal_evidence().execution_grant_id(),
            grant.policy_grant().evidence().execution_grant_id()
        );
    }
}
