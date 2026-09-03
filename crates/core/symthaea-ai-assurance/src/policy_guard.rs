// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict host guard for policy-domain revocation after authority minting.
//!
//! [`crate::PolicyExecutionDomain`] proves that an exact execution grant was
//! minted from a valid policy admission. Policy state can still change after
//! that minting event. This module adds the strict host layer that pins the
//! policy verifier and rechecks its domain/epoch when the grant is consumed and
//! again immediately before entering the concrete resource adapter closure.
//!
//! Deployments should rotate the policy evaluator epoch whenever a policy,
//! emergency-stop state, approval revocation, or other host condition changes in
//! a way that invalidates outstanding admissions/grants. Rotation then fails
//! closed at the intended side-effect boundary.

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::capability::{CapabilityKind, PrincipalId};
use crate::host::ResolutionError;
use crate::policy::{
    PolicyAuthorizationEvidence, PolicyError, PolicyGrant, PolicyResourceAction,
    PolicyResourceEvidenceReceipt, PolicyResourceRuntime, PolicyVerifier,
};
use crate::resolution::ResolutionGrant;
use crate::resource::{ResolvedResource, ResourceError, ResourceExecutionError, ResourceIdentity};
use crate::trusted::{
    AuthorityDomainId, AuthorityEpoch, TrustError, TrustedBoundOneShotCapability,
};
use std::fmt;

/// Strict policy-revocation-aware wrapper around the policy/resource runtime.
#[derive(Debug, Clone)]
pub struct PolicyGuardedRuntime {
    inner: PolicyResourceRuntime,
    policy_verifier: PolicyVerifier,
}

impl PolicyGuardedRuntime {
    /// Construct a host runtime pinned to the same policy evaluator verifier used
    /// by the corresponding [`crate::PolicyExecutionDomain`].
    pub fn new(inner: PolicyResourceRuntime, policy_verifier: PolicyVerifier) -> Self {
        Self {
            inner,
            policy_verifier,
        }
    }

    /// Policy evaluator domain pinned by the host.
    pub fn policy_domain(&self) -> AuthorityDomainId {
        self.policy_verifier.domain_id()
    }

    /// Admit a concrete resource action into the revocation-aware policy path.
    pub fn admit_resolved<K: CapabilityKind, H>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        resource: ResolvedResource<H>,
        canonical_payload: &[u8],
    ) -> Result<PolicyGuardedAction<K, Proposed, H>, ResourceError> {
        let inner = self
            .inner
            .admit_resolved::<K, H>(actor, kind, resource, canonical_payload)?;
        Ok(PolicyGuardedAction {
            inner,
            policy_verifier: self.policy_verifier.clone(),
        })
    }
}

/// Policy/resource action lifecycle with host-pinned policy epoch revalidation.
pub struct PolicyGuardedAction<K: CapabilityKind, S, H> {
    inner: PolicyResourceAction<K, S, H>,
    policy_verifier: PolicyVerifier,
}

impl<K: CapabilityKind, S, H> PolicyGuardedAction<K, S, H> {
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
    pub fn resource_identity(&self) -> &ResourceIdentity {
        self.inner.resource_identity()
    }

    /// Host-pinned policy evaluator domain.
    pub fn policy_domain(&self) -> AuthorityDomainId {
        self.policy_verifier.domain_id()
    }
}

impl<K: CapabilityKind, H> PolicyGuardedAction<K, Proposed, H> {
    /// Attach explicit risk before policy evaluation.
    pub fn assess(self, risk: ActionRisk) -> PolicyGuardedAction<K, RiskAssessed, H> {
        PolicyGuardedAction {
            inner: self.inner.assess(risk),
            policy_verifier: self.policy_verifier,
        }
    }
}

impl<K: CapabilityKind, H> PolicyGuardedAction<K, RiskAssessed, H> {
    /// Risk classification evaluated by policy.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact action authorization binding supplied to trusted policy evaluation.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Consume a policy-bound grant only while its policy evaluator epoch is
    /// still the host-pinned current epoch.
    pub fn authorize(
        self,
        grant: PolicyGrant<K>,
    ) -> Result<PolicyGuardedAction<K, Authorized, H>, PolicyGuardedAuthorizeError> {
        validate_policy_epoch(grant.evidence(), &self.policy_verifier)
            .map_err(PolicyGuardedAuthorizeError::Guard)?;
        let inner = self
            .inner
            .authorize(grant)
            .map_err(PolicyGuardedAuthorizeError::Policy)?;
        Ok(PolicyGuardedAction {
            inner,
            policy_verifier: self.policy_verifier,
        })
    }
}

impl<K: CapabilityKind, H> PolicyGuardedAction<K, Authorized, H> {
    /// Policy evidence consumed by this authorized action.
    pub fn policy_evidence(&self) -> &PolicyAuthorizationEvidence {
        self.inner.policy_evidence()
    }

    /// Execute only while the policy evaluator epoch that justified authority
    /// remains current. Validation occurs before the resource adapter closure is
    /// entered.
    pub fn execute_with<F, E>(
        self,
        execute: F,
    ) -> Result<PolicyGuardedAction<K, Executed, H>, PolicyGuardedExecutionError<E>>
    where
        F: FnOnce(&mut H) -> Result<[u8; 32], E>,
    {
        validate_policy_epoch(self.inner.policy_evidence(), &self.policy_verifier)
            .map_err(PolicyGuardedExecutionError::Guard)?;
        let inner = self
            .inner
            .execute_with(execute)
            .map_err(PolicyGuardedExecutionError::Resource)?;
        Ok(PolicyGuardedAction {
            inner,
            policy_verifier: self.policy_verifier,
        })
    }
}

impl<K: CapabilityKind, H> PolicyGuardedAction<K, Executed, H> {
    /// Exact independent-observation binding.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Attach independently authorized observation after the side effect.
    ///
    /// Policy epoch is intentionally not rechecked here: a later policy change
    /// should not erase the ability to collect evidence about an effect that
    /// already occurred.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<crate::Observe>,
        observation: Observation,
    ) -> Result<PolicyGuardedAction<K, Observed, H>, TrustError> {
        let inner = self.inner.observe(observer, observation)?;
        Ok(PolicyGuardedAction {
            inner,
            policy_verifier: self.policy_verifier,
        })
    }
}

impl<K: CapabilityKind, H> PolicyGuardedAction<K, Observed, H> {
    /// Exact final-resolution binding.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        self.inner.resolution_binding(decision)
    }

    /// Resolve independently observed evidence using exact resolver authority.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<
        (
            PolicyGuardedAction<K, Resolved, H>,
            PolicyResourceEvidenceReceipt,
        ),
        ResolutionError,
    > {
        let (inner, receipt) = self.inner.resolve(grant, decision)?;
        Ok((
            PolicyGuardedAction {
                inner,
                policy_verifier: self.policy_verifier,
            },
            receipt,
        ))
    }
}

impl<K: CapabilityKind, H> PolicyGuardedAction<K, Resolved, H> {
    /// Policy provenance retained by the completed action.
    pub fn policy_evidence(&self) -> &PolicyAuthorizationEvidence {
        self.inner.policy_evidence()
    }
}

/// Policy-domain/epoch guard failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyGuardError {
    /// Policy grant came from a policy evaluator domain other than the one pinned by the host.
    WrongPolicyDomain {
        /// Host-pinned policy evaluator domain.
        expected: AuthorityDomainId,
        /// Policy evaluator domain carried by the grant lineage.
        actual: AuthorityDomainId,
    },
    /// Policy evaluator epoch changed after admission/grant minting.
    RevokedPolicyEpoch {
        /// Policy evaluator domain.
        domain: AuthorityDomainId,
        /// Epoch carried by the policy grant lineage.
        lineage_epoch: AuthorityEpoch,
        /// Current epoch required by the host.
        current_epoch: AuthorityEpoch,
    },
}

impl fmt::Display for PolicyGuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPolicyDomain { .. } => {
                write!(f, "policy grant belongs to another evaluator domain")
            }
            Self::RevokedPolicyEpoch { .. } => {
                write!(
                    f,
                    "policy admission was revoked by evaluator epoch rotation"
                )
            }
        }
    }
}

impl std::error::Error for PolicyGuardError {}

/// Failure while consuming a policy-bound grant on the strict guarded path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyGuardedAuthorizeError {
    /// Host-pinned policy-domain/epoch validation failed.
    Guard(PolicyGuardError),
    /// Policy/action authority validation failed.
    Policy(PolicyError),
}

impl fmt::Display for PolicyGuardedAuthorizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Guard(error) => write!(f, "policy guard rejected authorization: {error}"),
            Self::Policy(error) => write!(f, "policy authorization failed: {error}"),
        }
    }
}

impl std::error::Error for PolicyGuardedAuthorizeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Guard(error) => Some(error),
            Self::Policy(error) => Some(error),
        }
    }
}

/// Failure while crossing the guarded state-changing execution boundary.
#[derive(Debug)]
pub enum PolicyGuardedExecutionError<E> {
    /// Policy evaluator domain/epoch is no longer valid.
    Guard(PolicyGuardError),
    /// Concrete resource/execution adapter failed.
    Resource(ResourceExecutionError<E>),
}

impl<E: fmt::Display> fmt::Display for PolicyGuardedExecutionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Guard(error) => write!(f, "policy guard rejected execution: {error}"),
            Self::Resource(error) => write!(f, "resource execution failed: {error}"),
        }
    }
}

impl<E> std::error::Error for PolicyGuardedExecutionError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Guard(error) => Some(error),
            Self::Resource(error) => Some(error),
        }
    }
}

fn validate_policy_epoch(
    evidence: &PolicyAuthorizationEvidence,
    verifier: &PolicyVerifier,
) -> Result<(), PolicyGuardError> {
    if evidence.policy_domain() != verifier.domain_id() {
        return Err(PolicyGuardError::WrongPolicyDomain {
            expected: verifier.domain_id(),
            actual: evidence.policy_domain(),
        });
    }
    let current = verifier.current_epoch();
    if evidence.policy_epoch() != current {
        return Err(PolicyGuardError::RevokedPolicyEpoch {
            domain: verifier.domain_id(),
            lineage_epoch: evidence.policy_epoch(),
            current_epoch: current,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdapterSchema, ApprovalEvidence, AuthorityDomain, PolicyDescriptor, PolicyEvaluatorDomain,
        PolicyExecutionDomain, PolicyMode, ResolutionAuthorityDomain, ResourceIdentity,
        ResourceResolverDomain, ResourceRuntime, Scope, TrustedRuntime, Write,
    };
    use std::sync::atomic::{AtomicBool, Ordering};

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    fn identity() -> ResourceIdentity {
        ResourceIdentity::new(
            scope(),
            "worktree-file",
            [1; 32],
            [2; 32],
            AdapterSchema::new("policy-guard-test", 1).unwrap(),
        )
        .unwrap()
    }

    fn setup() -> (
        PolicyEvaluatorDomain,
        PolicyExecutionDomain,
        ResourceResolverDomain,
        PolicyGuardedRuntime,
    ) {
        let descriptor = PolicyDescriptor::new("magi-gate", 1, [3; 32], 1).unwrap();
        let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor);
        let policy_verifier = evaluator.verifier();
        let execution = PolicyExecutionDomain::new(PrincipalId::new(), policy_verifier.clone());
        let observation = AuthorityDomain::new(PrincipalId::new());
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let strict = TrustedRuntime::new(
            execution.verifier(),
            observation.verifier(),
            resolution.verifier(),
        );
        let resource_runtime = ResourceRuntime::new(strict, resources.verifier());
        let policy_runtime = PolicyResourceRuntime::new(resource_runtime);
        (
            evaluator,
            execution,
            resources,
            PolicyGuardedRuntime::new(policy_runtime, policy_verifier),
        )
    }

    fn grant(
        evaluator: &PolicyEvaluatorDomain,
        execution: &PolicyExecutionDomain,
        actor: PrincipalId,
        binding: [u8; 32],
        risk: ActionRisk,
    ) -> PolicyGrant<Write> {
        let admission = evaluator.admit(
            binding,
            scope(),
            risk,
            PolicyMode::Autonomous,
            ApprovalEvidence::new([4; 32], [5; 32], true),
            [6; 32],
            [7; 32],
            [8; 32],
            None,
        );
        execution
            .issue::<Write>(actor, scope(), None, binding, admission)
            .unwrap()
    }

    #[test]
    fn revocation_after_grant_minting_blocks_authorization() {
        let (evaluator, execution, resources, runtime) = setup();
        let actor = PrincipalId::new();
        let action = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve((), identity(), None),
                b"patch",
            )
            .unwrap()
            .assess(ActionRisk::Reversible);
        let grant = grant(
            &evaluator,
            &execution,
            actor,
            action.authorization_binding(),
            action.risk(),
        );
        evaluator.revoke_all().unwrap();
        assert!(matches!(
            action.authorize(grant),
            Err(PolicyGuardedAuthorizeError::Guard(
                PolicyGuardError::RevokedPolicyEpoch { .. }
            ))
        ));
    }

    #[test]
    fn revocation_after_authorization_blocks_adapter_entry() {
        let (evaluator, execution, resources, runtime) = setup();
        let actor = PrincipalId::new();
        let action = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve(0_u64, identity(), None),
                b"patch",
            )
            .unwrap()
            .assess(ActionRisk::StateModifying);
        let grant = grant(
            &evaluator,
            &execution,
            actor,
            action.authorization_binding(),
            action.risk(),
        );
        let action = action.authorize(grant).unwrap();
        evaluator.revoke_all().unwrap();
        let entered = AtomicBool::new(false);
        let result = action.execute_with(|handle| -> Result<[u8; 32], &'static str> {
            entered.store(true, Ordering::SeqCst);
            *handle += 1;
            Ok([9; 32])
        });
        assert!(matches!(
            result,
            Err(PolicyGuardedExecutionError::Guard(
                PolicyGuardError::RevokedPolicyEpoch { .. }
            ))
        ));
        assert!(!entered.load(Ordering::SeqCst));
    }
}
