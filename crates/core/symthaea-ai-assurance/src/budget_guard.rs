// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quantitative-budget guard composed with the strongest state-changing host path.
//!
//! [`crate::PolicyGuardedAction`] already binds exact policy provenance,
//! revocation-aware execution authority, and a retained concrete resource. This
//! module adds a separately trusted [`crate::BudgetLease`] requirement without
//! conflating permission with quantity.
//!
//! The budget lease is validated when authorization is consumed and again
//! *inside* the retained-resource adapter closure, after policy/resource guards
//! have run but immediately before user adapter code. A budget epoch rotation
//! between authorization and effect therefore fails closed on this path.
//!
//! Pre-effect authorization can use the recoverable API so a policy rejection
//! returns the exact quantitative lease instead of silently stranding conserved
//! capacity. This does not imply that post-entry failures are refundable.
//!
//! This module still does not claim that every dimension is physically hard
//! enforced. It proves that the host admitted an exact quantitative reservation
//! and that the reservation remained valid at adapter entry. Concrete adapters
//! must implement the enforcement class recorded by the budget profile and
//! later report actual usage/enforcement evidence.

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::budget::{
    BudgetError, BudgetLease, BudgetQuantities, BudgetReleaseReceipt, BudgetVerifier,
};
use crate::capability::{CapabilityKind, GrantId, PrincipalId};
use crate::host::ResolutionError;
use crate::policy::{PolicyGrant, PolicyResourceEvidenceReceipt};
use crate::policy_guard::{
    PolicyGuardedAction, PolicyGuardedAuthorizeError, PolicyGuardedExecutionError,
    PolicyGuardedRuntime,
};
use crate::resolution::ResolutionGrant;
use crate::resource::{ResolvedResource, ResourceError};
use crate::trusted::{AuthorityDomainId, AuthorityEpoch, TrustError, TrustedBoundOneShotCapability};
use std::fmt;

/// Host wrapper that pins one budget verifier alongside the policy/resource host boundary.
#[derive(Debug, Clone)]
pub struct BudgetGuardedRuntime {
    inner: PolicyGuardedRuntime,
    budget_verifier: BudgetVerifier,
}

impl BudgetGuardedRuntime {
    /// Construct a state-changing host path pinned to one quantitative budget domain/profile.
    pub fn new(inner: PolicyGuardedRuntime, budget_verifier: BudgetVerifier) -> Self {
        Self {
            inner,
            budget_verifier,
        }
    }

    /// Budget authority domain accepted by this runtime.
    pub fn budget_domain(&self) -> AuthorityDomainId {
        self.budget_verifier.domain_id()
    }

    /// Immutable budget profile digest accepted by this runtime.
    pub fn budget_profile_digest(&self) -> [u8; 32] {
        self.budget_verifier.profile_digest()
    }

    /// Admit a pre-resolved concrete resource into the budget-aware guarded lifecycle.
    pub fn admit_resolved<K: CapabilityKind, H>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        resource: ResolvedResource<H>,
        canonical_payload: &[u8],
    ) -> Result<BudgetGuardedAction<K, Proposed, H>, ResourceError> {
        let inner = self
            .inner
            .admit_resolved::<K, H>(actor, kind, resource, canonical_payload)?;
        Ok(BudgetGuardedAction {
            inner,
            budget_verifier: self.budget_verifier.clone(),
            action_binding: None,
            budget_lease: None,
        })
    }
}

/// Strong state-changing action lifecycle with separate permission and budget authority.
pub struct BudgetGuardedAction<K: CapabilityKind, S, H> {
    inner: PolicyGuardedAction<K, S, H>,
    budget_verifier: BudgetVerifier,
    action_binding: Option<[u8; 32]>,
    budget_lease: Option<BudgetLease>,
}

impl<K: CapabilityKind, S, H> BudgetGuardedAction<K, S, H> {
    /// Stable action identity.
    pub fn id(&self) -> ActionId {
        self.inner.id()
    }

    /// Acting principal.
    pub fn actor(&self) -> PrincipalId {
        self.inner.actor()
    }

    /// Immutable action descriptor whose fingerprint commits to the concrete resource identity.
    pub fn descriptor(&self) -> &ActionDescriptor {
        self.inner.descriptor()
    }

    /// Budget trust domain pinned by the host.
    pub fn budget_domain(&self) -> AuthorityDomainId {
        self.budget_verifier.domain_id()
    }
}

impl<K: CapabilityKind, H> BudgetGuardedAction<K, Proposed, H> {
    /// Attach explicit risk and establish the exact action binding used by both
    /// policy and quantitative budget authority.
    pub fn assess(self, risk: ActionRisk) -> BudgetGuardedAction<K, RiskAssessed, H> {
        let inner = self.inner.assess(risk);
        let action_binding = inner.authorization_binding();
        BudgetGuardedAction {
            inner,
            budget_verifier: self.budget_verifier,
            action_binding: Some(action_binding),
            budget_lease: None,
        }
    }
}

impl<K: CapabilityKind, H> BudgetGuardedAction<K, RiskAssessed, H> {
    /// Risk classification evaluated by policy.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact binding that policy admission and budget reservation must both target.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.action_binding
            .expect("RiskAssessed budget action always carries action binding")
    }

    /// Compatibility authorization API retaining the original lossy error shape.
    ///
    /// Trusted pre-effect callers should prefer [`Self::authorize_recoverable`]
    /// so a rejected join cannot strand conserved budget capacity.
    pub fn authorize(
        self,
        policy_grant: PolicyGrant<K>,
        budget_lease: BudgetLease,
    ) -> Result<BudgetGuardedAction<K, Authorized, H>, BudgetGuardedAuthorizeError> {
        self.authorize_recoverable(policy_grant, budget_lease)
            .map_err(BudgetGuardedAuthorizeFailure::into_error)
    }

    /// Consume policy-bound permission and quantitative authority while
    /// returning the exact budget lease on any rejection before authorization
    /// succeeds.
    ///
    /// The policy grant/action may remain consumed when the underlying one-shot
    /// policy transition rejects. The quantitative lease is independent affine
    /// authority, so it is returned unchanged because no external effect has
    /// occurred at this stage.
    pub fn authorize_recoverable(
        self,
        policy_grant: PolicyGrant<K>,
        budget_lease: BudgetLease,
    ) -> Result<BudgetGuardedAction<K, Authorized, H>, BudgetGuardedAuthorizeFailure> {
        let action_binding = self.authorization_binding();
        if let Err(error) = budget_lease.validate_for(
            &self.budget_verifier,
            self.inner.actor(),
            self.inner.descriptor().scope(),
            action_binding,
        ) {
            return Err(BudgetGuardedAuthorizeFailure::new(
                budget_lease,
                BudgetGuardedAuthorizeError::Budget(error),
            ));
        }

        let BudgetGuardedAction {
            inner,
            budget_verifier,
            action_binding: _,
            budget_lease: _,
        } = self;
        let inner = match inner.authorize(policy_grant) {
            Ok(inner) => inner,
            Err(error) => {
                return Err(BudgetGuardedAuthorizeFailure::new(
                    budget_lease,
                    BudgetGuardedAuthorizeError::Policy(error),
                ));
            }
        };

        Ok(BudgetGuardedAction {
            inner,
            budget_verifier,
            action_binding: Some(action_binding),
            budget_lease: Some(budget_lease),
        })
    }
}

impl<K: CapabilityKind, H> BudgetGuardedAction<K, Authorized, H> {
    /// Quantities reserved for this exact authorized action.
    pub fn budget_allocation(&self) -> BudgetQuantities {
        self.budget_lease
            .as_ref()
            .expect("Authorized budget action always carries a lease")
            .allocation()
    }

    /// Budget lease identity reserved for this exact action.
    pub fn budget_lease_id(&self) -> GrantId {
        self.budget_lease
            .as_ref()
            .expect("Authorized budget action always carries a lease")
            .lease_id()
    }

    /// Cross the effect boundary only while policy, concrete-resource, and
    /// quantitative-budget authority all remain valid.
    ///
    /// Policy and resource checks are performed by the wrapped host path. The
    /// budget check occurs inside that retained-resource closure immediately
    /// before `execute` is called.
    pub fn execute_with<F, E>(
        self,
        execute: F,
    ) -> Result<
        BudgetGuardedAction<K, Executed, H>,
        PolicyGuardedExecutionError<BudgetAdapterError<E>>,
    >
    where
        F: FnOnce(&mut H) -> Result<[u8; 32], E>,
    {
        let BudgetGuardedAction {
            inner,
            budget_verifier,
            action_binding,
            budget_lease,
        } = self;
        let binding = action_binding.expect("Authorized budget action always carries binding");
        let lease = budget_lease.expect("Authorized budget action always carries lease");
        let actor = inner.actor();
        let scope = inner.descriptor().scope().clone();

        let inner = inner.execute_with(|handle| {
            lease
                .validate_for(&budget_verifier, actor, &scope, binding)
                .map_err(BudgetAdapterError::Budget)?;
            execute(handle).map_err(BudgetAdapterError::Adapter)
        })?;

        Ok(BudgetGuardedAction {
            inner,
            budget_verifier,
            action_binding: Some(binding),
            budget_lease: Some(lease),
        })
    }
}

impl<K: CapabilityKind, H> BudgetGuardedAction<K, Executed, H> {
    /// Exact independent-observation binding.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Attach independent observation. Budget revocation after an already
    /// executed effect does not block evidence collection.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<crate::Observe>,
        observation: Observation,
    ) -> Result<BudgetGuardedAction<K, Observed, H>, TrustError> {
        let inner = self.inner.observe(observer, observation)?;
        Ok(BudgetGuardedAction {
            inner,
            budget_verifier: self.budget_verifier,
            action_binding: self.action_binding,
            budget_lease: self.budget_lease,
        })
    }
}

impl<K: CapabilityKind, H> BudgetGuardedAction<K, Observed, H> {
    /// Exact final-resolution binding.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        self.inner.resolution_binding(decision)
    }

    /// Resolve observed evidence and emit policy/resource/budget provenance.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<
        (
            BudgetGuardedAction<K, Resolved, H>,
            BudgetedEvidenceReceipt,
        ),
        ResolutionError,
    > {
        let BudgetGuardedAction {
            inner,
            budget_verifier,
            action_binding,
            budget_lease,
        } = self;
        let lease = budget_lease.expect("Observed budget action always carries lease");
        let budget_evidence = BudgetLeaseEvidence::from_lease(&lease);
        let (inner, policy_resource_receipt) = inner.resolve(grant, decision)?;
        let receipt = BudgetedEvidenceReceipt {
            policy_resource_receipt,
            budget: budget_evidence,
        };
        Ok((
            BudgetGuardedAction {
                inner,
                budget_verifier,
                action_binding,
                budget_lease: Some(lease),
            },
            receipt,
        ))
    }
}

impl<K: CapabilityKind, H> BudgetGuardedAction<K, Resolved, H> {
    /// Immutable budget lineage retained by this completed action.
    pub fn budget_evidence(&self) -> BudgetLeaseEvidence {
        BudgetLeaseEvidence::from_lease(
            self.budget_lease
                .as_ref()
                .expect("Resolved budget action always carries lease"),
        )
    }

    /// Consume the completed wrapper and return the reserved capacity to the
    /// root pool. The final evidence receipt should be retained by the caller
    /// before invoking this method.
    pub fn release_budget(self) -> Result<BudgetReleaseReceipt, BudgetError> {
        self.budget_lease
            .expect("Resolved budget action always carries lease")
            .release()
    }
}

/// Immutable budget provenance preserved in final evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetLeaseEvidence {
    lease_id: GrantId,
    parent_lease_id: Option<GrantId>,
    domain: AuthorityDomainId,
    epoch: AuthorityEpoch,
    profile_digest: [u8; 32],
    allocation: BudgetQuantities,
    action_binding: [u8; 32],
}

impl BudgetLeaseEvidence {
    fn from_lease(lease: &BudgetLease) -> Self {
        Self {
            lease_id: lease.lease_id(),
            parent_lease_id: lease.parent_lease_id(),
            domain: lease.domain_id(),
            epoch: lease.epoch(),
            profile_digest: lease.profile().digest(),
            allocation: lease.allocation(),
            action_binding: lease.action_binding(),
        }
    }

    /// Exact budget lease id.
    pub fn lease_id(&self) -> GrantId {
        self.lease_id
    }

    /// Parent quantitative lease, if this was delegated via affine split.
    pub fn parent_lease_id(&self) -> Option<GrantId> {
        self.parent_lease_id
    }

    /// Budget authority domain.
    pub fn domain(&self) -> AuthorityDomainId {
        self.domain
    }

    /// Budget revocation epoch captured by the lease.
    pub fn epoch(&self) -> AuthorityEpoch {
        self.epoch
    }

    /// Immutable budget profile digest.
    pub fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    /// Quantities reserved for the exact action.
    pub fn allocation(&self) -> BudgetQuantities {
        self.allocation
    }

    /// Exact action authorization binding targeted by this lease.
    pub fn action_binding(&self) -> [u8; 32] {
        self.action_binding
    }
}

/// Final evidence joining quantitative budget provenance with the strongest
/// policy/resource/execution/observation/resolution lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedEvidenceReceipt {
    policy_resource_receipt: PolicyResourceEvidenceReceipt,
    budget: BudgetLeaseEvidence,
}

impl BudgetedEvidenceReceipt {
    /// Policy/resource/execution/observation/resolution evidence.
    pub fn policy_resource_receipt(&self) -> &PolicyResourceEvidenceReceipt {
        &self.policy_resource_receipt
    }

    /// Quantitative reservation evidence.
    pub fn budget(&self) -> &BudgetLeaseEvidence {
        &self.budget
    }
}

/// Failure while consuming the permission + quantitative authority pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetGuardedAuthorizeError {
    /// Quantitative lease validation failed.
    Budget(BudgetError),
    /// Policy-bound permission validation failed.
    Policy(PolicyGuardedAuthorizeError),
}

impl fmt::Display for BudgetGuardedAuthorizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget(error) => write!(f, "budget authorization failed: {error}"),
            Self::Policy(error) => write!(f, "policy authorization failed: {error}"),
        }
    }
}

impl std::error::Error for BudgetGuardedAuthorizeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Budget(error) => Some(error),
            Self::Policy(error) => Some(error),
        }
    }
}

/// Recoverable pre-effect authorization failure that retains the exact budget lease.
///
/// This failure is intentionally not `Clone`: it owns affine quantitative
/// authority that may be retried, retained, or explicitly released by the host.
#[derive(Debug)]
pub struct BudgetGuardedAuthorizeFailure {
    budget_lease: BudgetLease,
    error: BudgetGuardedAuthorizeError,
}

impl BudgetGuardedAuthorizeFailure {
    fn new(budget_lease: BudgetLease, error: BudgetGuardedAuthorizeError) -> Self {
        Self {
            budget_lease,
            error,
        }
    }

    /// Authorization rejection reason.
    pub fn error(&self) -> &BudgetGuardedAuthorizeError {
        &self.error
    }

    /// Exact original lease recovered from the failed join.
    pub fn budget_lease(&self) -> &BudgetLease {
        &self.budget_lease
    }

    /// Consume the failure and recover the exact original lease.
    pub fn into_budget_lease(self) -> BudgetLease {
        self.budget_lease
    }

    /// Consume the failure into the original lease plus rejection reason.
    pub fn into_parts(self) -> (BudgetLease, BudgetGuardedAuthorizeError) {
        (self.budget_lease, self.error)
    }

    /// Consume the failure and retain only the compatibility error.
    pub fn into_error(self) -> BudgetGuardedAuthorizeError {
        self.error
    }
}

impl fmt::Display for BudgetGuardedAuthorizeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "recoverable budget authorization rejected: {}", self.error)
    }
}

impl std::error::Error for BudgetGuardedAuthorizeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Error produced inside the retained-resource adapter boundary.
#[derive(Debug)]
pub enum BudgetAdapterError<E> {
    /// Budget lease was revoked/expired/substituted before adapter entry.
    Budget(BudgetError),
    /// Concrete adapter returned its own error.
    Adapter(E),
}

impl<E: fmt::Display> fmt::Display for BudgetAdapterError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget(error) => write!(f, "budget guard rejected adapter entry: {error}"),
            Self::Adapter(error) => write!(f, "adapter failed: {error}"),
        }
    }
}

impl<E> std::error::Error for BudgetAdapterError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Budget(error) => Some(error),
            Self::Adapter(error) => Some(error),
        }
    }
}
