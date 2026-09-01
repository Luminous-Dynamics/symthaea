// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit, evidence-bearing separation-of-duties policy for observation and
//! final resolution.
//!
//! The lower assurance stack already provides separately attributable execution,
//! observation, and resolution authority. It intentionally does not infer that
//! different principals or random authority-domain ids imply organizational,
//! process, hardware, or operator independence.
//!
//! This module makes the *actual* separation requirement a host-selected policy:
//!
//! - [`IndependencePolicy::DistinctPrincipal`] requires the observer principal to
//!   differ from the action actor. Trust roots may overlap.
//! - [`IndependencePolicy::DistinctAuthorityDomains`] additionally requires the
//!   execution, observation, and resolution authority-domain ids selected by the
//!   host to be pairwise distinct.
//! - [`IndependencePolicy::SeparationOfDuties`] adds resolver-principal separation:
//!   resolver must differ from both actor and observer.
//!
//! The selected policy is preserved in [`IndependenceEvidenceReceipt`] together
//! with the exact role principals/domains and a deterministic evidence digest.
//! No level in this module claims external organizational or hardware
//! independence. A future externally-attested profile requires real adapter /
//! HAL / Xenia attestation evidence rather than UUID inequality.
//!
//! [`IndependenceGuardedRuntime`] wraps the current strongest
//! [`crate::EffectGuardedRuntime`]. Observation and resolution grants are first
//! validated against host-retained verifiers owned by this wrapper and then are
//! delegated to the lower host path, which independently validates its own
//! pinned verifier. A successful transition therefore satisfies both layers.
//! Deployments should configure both layers from the same selected trust roots;
//! a mismatch fails closed rather than silently weakening the policy.

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::budget::{BudgetError, BudgetLease, BudgetReleaseReceipt};
use crate::capability::{CapabilityKind, Observe, PrincipalId};
use crate::effect_guard::{
    EffectAssuredEvidenceReceipt, EffectAttemptEvidence, EffectAttemptFailure,
    EffectAttemptOutcome, EffectGuardedAction, EffectGuardedAuthorizeError, EffectGuardedRuntime,
    EffectInnerExecutionError, ExecutionPreflightError,
};
use crate::host::ResolutionError;
use crate::resolution::{ResolutionGrant, ResolutionVerifier};
use crate::resource::{ResolvedResource, ResourceError};
use crate::temporal_policy::TemporalPolicyGrant;
use crate::trusted::{
    AuthorityDomainId, AuthorityVerifier, TrustError, TrustedBoundOneShotCapability,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Explicit host-selected separation requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndependencePolicy {
    /// Observer principal must differ from the actor. Authority domains may
    /// overlap; this level must not be described as independent trust roots.
    DistinctPrincipal,
    /// Execution, observation, and resolution trust-domain ids must be pairwise
    /// distinct. Resolver principal may still equal another role principal.
    DistinctAuthorityDomains,
    /// Pairwise-distinct trust domains plus resolver principal distinct from both
    /// actor and observer.
    SeparationOfDuties,
}

impl IndependencePolicy {
    fn code(self) -> u8 {
        match self {
            Self::DistinctPrincipal => 0,
            Self::DistinctAuthorityDomains => 1,
            Self::SeparationOfDuties => 2,
        }
    }

    fn requires_distinct_domains(self) -> bool {
        matches!(
            self,
            Self::DistinctAuthorityDomains | Self::SeparationOfDuties
        )
    }

    fn requires_distinct_resolver_principal(self) -> bool {
        self == Self::SeparationOfDuties
    }

    /// Domain-separated digest naming this core independence policy level.
    pub fn digest(self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-ai-assurance/independence-policy-v1\0");
        hash_field(&mut hasher, &[self.code()]);
        *hasher.finalize().as_bytes()
    }
}

/// Role name used in configuration diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndependenceRole {
    /// Exact execution authority.
    Execution,
    /// External observation authority.
    Observation,
    /// Final-resolution authority.
    Resolution,
}

/// Invalid host trust-root configuration for the selected independence policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndependenceConfigError {
    /// A policy requiring pairwise-distinct trust roots was configured with the
    /// same authority-domain id for two roles.
    DomainReuse {
        /// First role using the domain.
        first: IndependenceRole,
        /// Second role using the same domain.
        second: IndependenceRole,
        /// Reused authority-domain id.
        domain: AuthorityDomainId,
    },
}

impl fmt::Display for IndependenceConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainReuse { first, second, .. } => write!(
                f,
                "independence policy requires distinct domains for {first:?} and {second:?}"
            ),
        }
    }
}

impl std::error::Error for IndependenceConfigError {}

/// Final evidence declaring exactly which separation policy was enforced and
/// which principals/domains occupied each role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndependenceEvidenceReceipt {
    effect_assured: EffectAssuredEvidenceReceipt,
    policy: IndependencePolicy,
    policy_digest: [u8; 32],
    actor: PrincipalId,
    execution_domain: AuthorityDomainId,
    observer: PrincipalId,
    observer_domain: AuthorityDomainId,
    resolver: PrincipalId,
    resolver_domain: AuthorityDomainId,
    digest: [u8; 32],
}

impl IndependenceEvidenceReceipt {
    /// Existing temporal/budget/resource/effect/observation/resolution evidence.
    pub fn effect_assured_receipt(&self) -> &EffectAssuredEvidenceReceipt {
        &self.effect_assured
    }

    /// Host-selected separation policy actually enforced by this wrapper.
    pub fn policy(&self) -> IndependencePolicy {
        self.policy
    }

    /// Stable digest naming the selected policy level.
    pub fn policy_digest(&self) -> [u8; 32] {
        self.policy_digest
    }

    /// Acting principal.
    pub fn actor(&self) -> PrincipalId {
        self.actor
    }

    /// Exact execution trust domain.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_domain
    }

    /// Observer principal whose grant was consumed.
    pub fn observer(&self) -> PrincipalId {
        self.observer
    }

    /// Observer authority domain selected by the host wrapper.
    pub fn observer_domain(&self) -> AuthorityDomainId {
        self.observer_domain
    }

    /// Resolver principal whose exact final-decision grant was consumed.
    pub fn resolver(&self) -> PrincipalId {
        self.resolver
    }

    /// Resolver authority domain selected by the host wrapper.
    pub fn resolver_domain(&self) -> AuthorityDomainId {
        self.resolver_domain
    }

    /// Domain-separated digest joining the separation policy and exact role
    /// lineage to the underlying effect-attempt + final-decision lineage.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Strongest current host wrapper with an explicit observer/resolver separation
/// contract.
#[derive(Debug, Clone)]
pub struct IndependenceGuardedRuntime {
    inner: EffectGuardedRuntime,
    observer_verifier: AuthorityVerifier,
    resolver_verifier: ResolutionVerifier,
    policy: IndependencePolicy,
    clock: Arc<IndependenceClock>,
}

impl IndependenceGuardedRuntime {
    /// Construct a wrapper with host-selected observation/resolution trust roots.
    ///
    /// Policies requiring distinct authority domains fail at construction if any
    /// selected role pair reuses a domain id. This checks the wrapper-selected
    /// trust roots. The lower host path independently validates the same grants;
    /// deployments should configure both layers from the same verifiers.
    pub fn new(
        inner: EffectGuardedRuntime,
        observer_verifier: AuthorityVerifier,
        resolver_verifier: ResolutionVerifier,
        policy: IndependencePolicy,
    ) -> Result<Self, IndependenceConfigError> {
        let execution = inner.execution_domain();
        let observation = observer_verifier.domain_id();
        let resolution = resolver_verifier.domain_id();

        if policy.requires_distinct_domains() {
            require_distinct(
                IndependenceRole::Execution,
                execution,
                IndependenceRole::Observation,
                observation,
            )?;
            require_distinct(
                IndependenceRole::Execution,
                execution,
                IndependenceRole::Resolution,
                resolution,
            )?;
            require_distinct(
                IndependenceRole::Observation,
                observation,
                IndependenceRole::Resolution,
                resolution,
            )?;
        }

        Ok(Self {
            inner,
            observer_verifier,
            resolver_verifier,
            policy,
            clock: Arc::new(IndependenceClock::new()),
        })
    }

    /// Selected independence policy.
    pub fn policy(&self) -> IndependencePolicy {
        self.policy
    }

    /// Exact execution authority domain.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.inner.execution_domain()
    }

    /// Host-selected observer authority domain.
    pub fn observer_domain(&self) -> AuthorityDomainId {
        self.observer_verifier.domain_id()
    }

    /// Host-selected resolver authority domain.
    pub fn resolver_domain(&self) -> AuthorityDomainId {
        self.resolver_verifier.domain_id()
    }

    /// Admit a concrete resource action into the separation-aware lifecycle.
    pub fn admit_resolved<K: CapabilityKind, H>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        resource: ResolvedResource<H>,
        canonical_payload: &[u8],
    ) -> Result<IndependenceGuardedAction<K, Proposed, H>, ResourceError> {
        let inner = self
            .inner
            .admit_resolved::<K, H>(actor, kind, resource, canonical_payload)?;
        Ok(IndependenceGuardedAction {
            inner,
            observer_verifier: self.observer_verifier.clone(),
            resolver_verifier: self.resolver_verifier.clone(),
            policy: self.policy,
            clock: Arc::clone(&self.clock),
            observer_principal: None,
            observer_domain: None,
            final_receipt: None,
        })
    }
}

/// Action lifecycle carrying the selected independence policy through observation
/// and final resolution.
pub struct IndependenceGuardedAction<K: CapabilityKind, S, H> {
    inner: EffectGuardedAction<K, S, H>,
    observer_verifier: AuthorityVerifier,
    resolver_verifier: ResolutionVerifier,
    policy: IndependencePolicy,
    clock: Arc<IndependenceClock>,
    observer_principal: Option<PrincipalId>,
    observer_domain: Option<AuthorityDomainId>,
    final_receipt: Option<IndependenceEvidenceReceipt>,
}

impl<K: CapabilityKind, S, H> IndependenceGuardedAction<K, S, H> {
    /// Stable exact action identity.
    pub fn id(&self) -> ActionId {
        self.inner.id()
    }

    /// Acting principal.
    pub fn actor(&self) -> PrincipalId {
        self.inner.actor()
    }

    /// Immutable concrete-resource-bound action descriptor.
    pub fn descriptor(&self) -> &ActionDescriptor {
        self.inner.descriptor()
    }

    /// Selected separation policy.
    pub fn independence_policy(&self) -> IndependencePolicy {
        self.policy
    }

    /// Host-selected observation authority domain.
    pub fn observer_domain(&self) -> AuthorityDomainId {
        self.observer_verifier.domain_id()
    }

    /// Host-selected final-resolution authority domain.
    pub fn resolver_domain(&self) -> AuthorityDomainId {
        self.resolver_verifier.domain_id()
    }
}

impl<K: CapabilityKind, H> IndependenceGuardedAction<K, Proposed, H> {
    /// Attach explicit risk before trusted policy evaluation.
    pub fn assess(self, risk: ActionRisk) -> IndependenceGuardedAction<K, RiskAssessed, H> {
        IndependenceGuardedAction {
            inner: self.inner.assess(risk),
            observer_verifier: self.observer_verifier,
            resolver_verifier: self.resolver_verifier,
            policy: self.policy,
            clock: self.clock,
            observer_principal: None,
            observer_domain: None,
            final_receipt: None,
        }
    }
}

impl<K: CapabilityKind, H> IndependenceGuardedAction<K, RiskAssessed, H> {
    /// Risk classification evaluated by policy.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact action binding targeted by policy and quantitative authority.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Consume temporally bounded exact policy authority plus quantitative authority.
    pub fn authorize(
        self,
        temporal_grant: TemporalPolicyGrant<K>,
        budget_lease: BudgetLease,
    ) -> Result<IndependenceGuardedAction<K, Authorized, H>, EffectGuardedAuthorizeError> {
        let inner = self.inner.authorize(temporal_grant, budget_lease)?;
        Ok(IndependenceGuardedAction {
            inner,
            observer_verifier: self.observer_verifier,
            resolver_verifier: self.resolver_verifier,
            policy: self.policy,
            clock: self.clock,
            observer_principal: None,
            observer_domain: None,
            final_receipt: None,
        })
    }
}

impl<K: CapabilityKind, H> IndependenceGuardedAction<K, Authorized, H> {
    /// Execute through the v0.8 exact-preflight/effect-attempt boundary.
    pub fn execute_attempt_with<F>(
        self,
        attempt: F,
    ) -> Result<IndependenceGuardedAction<K, Executed, H>, IndependenceEffectAttemptFailure<K, H>>
    where
        F: FnOnce(&mut H) -> EffectAttemptOutcome,
    {
        let IndependenceGuardedAction {
            inner,
            observer_verifier,
            resolver_verifier,
            policy,
            clock,
            observer_principal: _,
            observer_domain: _,
            final_receipt: _,
        } = self;

        match inner.execute_attempt_with(attempt) {
            Ok(inner) => Ok(IndependenceGuardedAction {
                inner,
                observer_verifier,
                resolver_verifier,
                policy,
                clock,
                observer_principal: None,
                observer_domain: None,
                final_receipt: None,
            }),
            Err(EffectAttemptFailure::Preflight { action, error }) => {
                Err(IndependenceEffectAttemptFailure::Preflight {
                    action: IndependenceGuardedAction {
                        inner: action,
                        observer_verifier,
                        resolver_verifier,
                        policy,
                        clock,
                        observer_principal: None,
                        observer_domain: None,
                        final_receipt: None,
                    },
                    error,
                })
            }
            Err(EffectAttemptFailure::RejectedBeforeAttempt { error }) => {
                Err(IndependenceEffectAttemptFailure::RejectedBeforeAttempt { error })
            }
            Err(EffectAttemptFailure::LineageFailedAfterAttempt { evidence, error }) => {
                Err(IndependenceEffectAttemptFailure::LineageFailedAfterAttempt { evidence, error })
            }
        }
    }
}

impl<K: CapabilityKind, H> IndependenceGuardedAction<K, Executed, H> {
    /// Exact observation binding committing to the v0.8 effect-attempt digest.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Consume observer authority under the selected separation policy.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<Observe>,
        observation: Observation,
    ) -> Result<IndependenceGuardedAction<K, Observed, H>, IndependenceObservationError> {
        let now = self.clock.now();
        observer
            .validate_with(&self.observer_verifier, now)
            .map_err(IndependenceObservationError::ObserverGrant)?;

        let observer_principal = observer.metadata().subject();
        if observer_principal == self.inner.actor() {
            return Err(IndependenceObservationError::ObserverPrincipalConflict {
                actor: self.inner.actor(),
                observer: observer_principal,
            });
        }
        let observer_domain = observer.domain_id();

        let inner = self
            .inner
            .observe(observer, observation)
            .map_err(IndependenceObservationError::Inner)?;
        Ok(IndependenceGuardedAction {
            inner,
            observer_verifier: self.observer_verifier,
            resolver_verifier: self.resolver_verifier,
            policy: self.policy,
            clock: self.clock,
            observer_principal: Some(observer_principal),
            observer_domain: Some(observer_domain),
            final_receipt: None,
        })
    }
}

impl<K: CapabilityKind, H> IndependenceGuardedAction<K, Observed, H> {
    /// Exact final-resolution binding for this observed effect lineage.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        self.inner.resolution_binding(decision)
    }

    /// Consume exact final-resolution authority under the selected separation policy.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<
        (
            IndependenceGuardedAction<K, Resolved, H>,
            IndependenceEvidenceReceipt,
        ),
        IndependenceResolutionError,
    > {
        let now = self.clock.now();
        grant
            .validate_with(&self.resolver_verifier, now)
            .map_err(IndependenceResolutionError::ResolverGrant)?;

        let resolver = grant.metadata().subject();
        let actor = self.inner.actor();
        let observer = self
            .observer_principal
            .expect("Observed independence action always carries observer principal");
        let observer_domain = self
            .observer_domain
            .expect("Observed independence action always carries observer domain");

        if self.policy.requires_distinct_resolver_principal()
            && (resolver == actor || resolver == observer)
        {
            return Err(IndependenceResolutionError::ResolverPrincipalConflict {
                actor,
                observer,
                resolver,
            });
        }

        let execution_domain = self.inner.execution_domain();
        let resolver_domain = grant.domain_id();
        let policy = self.policy;
        let action_id = self.inner.id();
        let effect_attempt_digest = self
            .inner
            .effect_attempt()
            .expect("Observed effect action always carries attempt evidence")
            .digest();
        let digest = compute_independence_digest(
            action_id,
            policy,
            actor,
            execution_domain,
            observer,
            observer_domain,
            resolver,
            resolver_domain,
            effect_attempt_digest,
            decision,
        );

        let (inner, effect_assured) = self
            .inner
            .resolve(grant, decision)
            .map_err(IndependenceResolutionError::Inner)?;
        let receipt = IndependenceEvidenceReceipt {
            effect_assured,
            policy,
            policy_digest: policy.digest(),
            actor,
            execution_domain,
            observer,
            observer_domain,
            resolver,
            resolver_domain,
            digest,
        };
        Ok((
            IndependenceGuardedAction {
                inner,
                observer_verifier: self.observer_verifier,
                resolver_verifier: self.resolver_verifier,
                policy,
                clock: self.clock,
                observer_principal: Some(observer),
                observer_domain: Some(observer_domain),
                final_receipt: Some(receipt.clone()),
            },
            receipt,
        ))
    }
}

impl<K: CapabilityKind, H> IndependenceGuardedAction<K, Resolved, H> {
    /// Final evidence including the exact separation policy and role lineage.
    pub fn independence_receipt(&self) -> &IndependenceEvidenceReceipt {
        self.final_receipt
            .as_ref()
            .expect("Resolved independence action always carries final evidence")
    }

    /// Return reserved quantitative capacity after final evidence has been retained.
    pub fn release_budget(self) -> Result<BudgetReleaseReceipt, BudgetError> {
        self.inner.release_budget()
    }
}

/// Observation-stage separation failure.
#[derive(Debug)]
pub enum IndependenceObservationError {
    /// Observer grant failed validation against the wrapper-selected trust root.
    ObserverGrant(TrustError),
    /// Observer principal equals the acting principal.
    ObserverPrincipalConflict {
        /// Acting principal.
        actor: PrincipalId,
        /// Presented observer principal.
        observer: PrincipalId,
    },
    /// Lower host path rejected the observer grant/evidence after wrapper validation.
    Inner(TrustError),
}

impl fmt::Display for IndependenceObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObserverGrant(error) => write!(f, "observer trust validation failed: {error}"),
            Self::ObserverPrincipalConflict { .. } => {
                write!(f, "observer principal must differ from action actor")
            }
            Self::Inner(error) => write!(f, "lower observation boundary rejected grant: {error}"),
        }
    }
}

impl std::error::Error for IndependenceObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ObserverGrant(error) | Self::Inner(error) => Some(error),
            Self::ObserverPrincipalConflict { .. } => None,
        }
    }
}

/// Final-resolution separation failure.
#[derive(Debug)]
pub enum IndependenceResolutionError {
    /// Resolver grant failed validation against the wrapper-selected resolver trust root.
    ResolverGrant(TrustError),
    /// Resolver principal violates the selected separation-of-duties profile.
    ResolverPrincipalConflict {
        /// Acting principal.
        actor: PrincipalId,
        /// Observer principal already consumed by the observed lineage.
        observer: PrincipalId,
        /// Presented resolver principal.
        resolver: PrincipalId,
    },
    /// Lower exact-resolution boundary rejected the grant/decision.
    Inner(ResolutionError),
}

impl fmt::Display for IndependenceResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResolverGrant(error) => write!(f, "resolver trust validation failed: {error}"),
            Self::ResolverPrincipalConflict { .. } => {
                write!(f, "resolver principal violates separation-of-duties policy")
            }
            Self::Inner(error) => write!(f, "lower resolution boundary rejected grant: {error}"),
        }
    }
}

impl std::error::Error for IndependenceResolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ResolverGrant(error) => Some(error),
            Self::ResolverPrincipalConflict { .. } => None,
            Self::Inner(error) => Some(error),
        }
    }
}

/// Effect-attempt failure preserving the separation wrapper when v0.8's exact
/// execution preflight rejects before adapter entry.
pub enum IndependenceEffectAttemptFailure<K: CapabilityKind, H> {
    /// Exact execution preflight failed before lower effect delegation; the
    /// original separation-aware authorized action is recoverable.
    Preflight {
        /// Recoverable authorized action.
        action: IndependenceGuardedAction<K, Authorized, H>,
        /// Exact execution preflight failure.
        error: ExecutionPreflightError,
    },
    /// A lower policy/resource/budget check rejected before user adapter entry.
    RejectedBeforeAttempt {
        /// Existing lower execution error.
        error: EffectInnerExecutionError,
    },
    /// Adapter returned attempt evidence but lower lineage failed afterward.
    LineageFailedAfterAttempt {
        /// Preserved adapter-attempt evidence.
        evidence: EffectAttemptEvidence,
        /// Existing lower execution error.
        error: EffectInnerExecutionError,
    },
}

impl<K: CapabilityKind, H> IndependenceEffectAttemptFailure<K, H> {
    /// Whether the user adapter boundary was entered.
    pub fn adapter_was_entered(&self) -> bool {
        matches!(self, Self::LineageFailedAfterAttempt { .. })
    }

    /// Preserved adapter-attempt evidence when entry occurred.
    pub fn attempt_evidence(&self) -> Option<&EffectAttemptEvidence> {
        match self {
            Self::LineageFailedAfterAttempt { evidence, .. } => Some(evidence),
            Self::Preflight { .. } | Self::RejectedBeforeAttempt { .. } => None,
        }
    }
}

impl<K: CapabilityKind, H> fmt::Debug for IndependenceEffectAttemptFailure<K, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preflight { error, .. } => f
                .debug_struct("IndependenceEffectAttemptFailure::Preflight")
                .field("error", error)
                .field("action", &"<authority-bearing action retained>")
                .finish(),
            Self::RejectedBeforeAttempt { error } => f
                .debug_struct("IndependenceEffectAttemptFailure::RejectedBeforeAttempt")
                .field("error", error)
                .finish(),
            Self::LineageFailedAfterAttempt { evidence, error } => f
                .debug_struct("IndependenceEffectAttemptFailure::LineageFailedAfterAttempt")
                .field("evidence", evidence)
                .field("error", error)
                .finish(),
        }
    }
}

fn require_distinct(
    first: IndependenceRole,
    first_domain: AuthorityDomainId,
    second: IndependenceRole,
    second_domain: AuthorityDomainId,
) -> Result<(), IndependenceConfigError> {
    if first_domain == second_domain {
        return Err(IndependenceConfigError::DomainReuse {
            first,
            second,
            domain: first_domain,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_independence_digest(
    action_id: ActionId,
    policy: IndependencePolicy,
    actor: PrincipalId,
    execution_domain: AuthorityDomainId,
    observer: PrincipalId,
    observer_domain: AuthorityDomainId,
    resolver: PrincipalId,
    resolver_domain: AuthorityDomainId,
    effect_attempt_digest: [u8; 32],
    decision: ResolutionDecision,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/independence-evidence-v1\0");
    hash_field(&mut hasher, action_id.as_uuid().as_bytes());
    hash_field(&mut hasher, &[policy.code()]);
    hash_field(&mut hasher, &policy.digest());
    hash_field(&mut hasher, actor.as_uuid().as_bytes());
    hash_field(&mut hasher, execution_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, observer.as_uuid().as_bytes());
    hash_field(&mut hasher, observer_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, resolver.as_uuid().as_bytes());
    hash_field(&mut hasher, resolver_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, &effect_attempt_digest);
    hash_field(&mut hasher, &[resolution_code(decision)]);
    *hasher.finalize().as_bytes()
}

fn resolution_code(decision: ResolutionDecision) -> u8 {
    match decision {
        ResolutionDecision::Confirmed => 0,
        ResolutionDecision::Contradicted => 1,
        ResolutionDecision::Inconclusive => 2,
    }
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug)]
struct IndependenceClock {
    last: Mutex<SystemTime>,
}

impl IndependenceClock {
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
    use crate::{
        AdapterSchema, ApprovalEvidence, AuthorityDomain, BudgetAuthorityDomain, BudgetDimension,
        BudgetEnforcement, BudgetGuardedRuntime, BudgetProfile, BudgetQuantities,
        EffectGuardedRuntime, EnforcementClass, ObservedOutcome, PolicyDescriptor,
        PolicyGuardedRuntime, PolicyMode, PolicyResourceRuntime, ResolutionAuthorityDomain,
        ResourceIdentity, ResourceResolverDomain, ResourceRuntime, Scope,
        TemporalPolicyEvaluatorDomain, TemporalPolicyExecutionDomain, TemporalPolicyRules,
        TrustedRuntime, Write,
    };
    use std::time::Duration;

    fn scope() -> crate::Scope {
        crate::Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    fn resource_identity() -> ResourceIdentity {
        ResourceIdentity::new(
            scope(),
            "worktree-file",
            [1; 32],
            [2; 32],
            AdapterSchema::new("independence-test", 1).unwrap(),
        )
        .unwrap()
    }

    fn budget_profile() -> BudgetProfile {
        let limits = BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 10);
        let enforcement = BudgetEnforcement::soft()
            .with(BudgetDimension::ComputeUnits, EnforcementClass::CoreMetered);
        BudgetProfile::new(limits, enforcement, None).unwrap()
    }

    struct Harness {
        evaluator: TemporalPolicyEvaluatorDomain,
        execution: TemporalPolicyExecutionDomain,
        observation: AuthorityDomain,
        resolution: ResolutionAuthorityDomain,
        resources: ResourceResolverDomain,
        budgets: BudgetAuthorityDomain,
        runtime: IndependenceGuardedRuntime,
    }

    fn effect_runtime(
        evaluator: &TemporalPolicyEvaluatorDomain,
        execution: &TemporalPolicyExecutionDomain,
        observation_verifier: AuthorityVerifier,
        resolution: &ResolutionAuthorityDomain,
        resources: &ResourceResolverDomain,
        budgets: &BudgetAuthorityDomain,
    ) -> EffectGuardedRuntime {
        let strict = TrustedRuntime::new(
            execution.verifier(),
            observation_verifier,
            resolution.verifier(),
        );
        let resource_runtime = ResourceRuntime::new(strict, resources.verifier());
        let policy_runtime = PolicyResourceRuntime::new(resource_runtime);
        let policy_guard = PolicyGuardedRuntime::new(policy_runtime, evaluator.verifier());
        let budget_runtime = BudgetGuardedRuntime::new(policy_guard, budgets.verifier());
        EffectGuardedRuntime::new(budget_runtime, execution.verifier())
    }

    fn harness(policy: IndependencePolicy) -> Harness {
        let rules = TemporalPolicyRules::strict();
        let evaluator = TemporalPolicyEvaluatorDomain::new(
            PrincipalId::new(),
            PolicyDescriptor::new("independence", 1, [3; 32], 1).unwrap(),
            rules,
        );
        let execution =
            TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
        let observation = AuthorityDomain::new(PrincipalId::new());
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let budgets = BudgetAuthorityDomain::new(PrincipalId::new(), budget_profile());
        let effect = effect_runtime(
            &evaluator,
            &execution,
            observation.verifier(),
            &resolution,
            &resources,
            &budgets,
        );
        let runtime = IndependenceGuardedRuntime::new(
            effect,
            observation.verifier(),
            resolution.verifier(),
            policy,
        )
        .unwrap();
        Harness {
            evaluator,
            execution,
            observation,
            resolution,
            resources,
            budgets,
            runtime,
        }
    }

    fn executed_action(
        harness: &Harness,
        expiry: SystemTime,
    ) -> IndependenceGuardedAction<Write, Executed, u64> {
        let actor = PrincipalId::new();
        let action = harness
            .runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit-source",
                harness
                    .resources
                    .resolve(0_u64, resource_identity(), Some(expiry)),
                b"patch-v1",
            )
            .unwrap()
            .assess(ActionRisk::Reversible);
        let binding = action.authorization_binding();
        let admission = harness
            .evaluator
            .admit(
                binding,
                scope(),
                action.risk(),
                PolicyMode::Autonomous,
                ApprovalEvidence::new([4; 32], [5; 32], true),
                [6; 32],
                [7; 32],
                [8; 32],
                Some(expiry),
            )
            .unwrap();
        let grant = harness
            .execution
            .issue::<Write>(actor, scope(), Some(expiry), binding, admission)
            .unwrap();
        let lease = harness
            .budgets
            .reserve(
                actor,
                scope(),
                binding,
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
                Some(expiry),
            )
            .unwrap();
        action
            .authorize(grant, lease)
            .unwrap()
            .execute_attempt_with(|handle| {
                *handle += 1;
                EffectAttemptOutcome::Succeeded {
                    evidence_digest: [9; 32],
                }
            })
            .unwrap()
    }

    #[test]
    fn distinct_domain_policy_rejects_execution_observation_reuse() {
        let rules = TemporalPolicyRules::strict();
        let evaluator = TemporalPolicyEvaluatorDomain::new(
            PrincipalId::new(),
            PolicyDescriptor::new("independence", 1, [10; 32], 1).unwrap(),
            rules,
        );
        let execution =
            TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let budgets = BudgetAuthorityDomain::new(PrincipalId::new(), budget_profile());
        let shared = execution.verifier();
        let effect = effect_runtime(
            &evaluator,
            &execution,
            shared.clone(),
            &resolution,
            &resources,
            &budgets,
        );
        let result = IndependenceGuardedRuntime::new(
            effect,
            shared,
            resolution.verifier(),
            IndependencePolicy::DistinctAuthorityDomains,
        );
        assert!(matches!(
            result,
            Err(IndependenceConfigError::DomainReuse {
                first: IndependenceRole::Execution,
                second: IndependenceRole::Observation,
                ..
            })
        ));
    }

    #[test]
    fn distinct_principal_policy_allows_shared_execution_observation_root_configuration() {
        let rules = TemporalPolicyRules::strict();
        let evaluator = TemporalPolicyEvaluatorDomain::new(
            PrincipalId::new(),
            PolicyDescriptor::new("independence", 1, [11; 32], 1).unwrap(),
            rules,
        );
        let execution =
            TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let budgets = BudgetAuthorityDomain::new(PrincipalId::new(), budget_profile());
        let shared = execution.verifier();
        let effect = effect_runtime(
            &evaluator,
            &execution,
            shared.clone(),
            &resolution,
            &resources,
            &budgets,
        );
        let runtime = IndependenceGuardedRuntime::new(
            effect,
            shared,
            resolution.verifier(),
            IndependencePolicy::DistinctPrincipal,
        )
        .unwrap();
        assert_eq!(runtime.execution_domain(), runtime.observer_domain());
        assert_eq!(runtime.policy(), IndependencePolicy::DistinctPrincipal);
    }

    #[test]
    fn separation_of_duties_rejects_resolver_principal_reuse() {
        let harness = harness(IndependencePolicy::SeparationOfDuties);
        let expiry = SystemTime::now() + Duration::from_secs(60);
        let action = executed_action(&harness, expiry);
        let observer = PrincipalId::new();
        let observer_grant = harness.observation.issue_bound_one_shot::<Observe>(
            observer,
            scope(),
            Some(expiry),
            action.observation_binding(),
        );
        let action = action
            .observe(
                observer_grant,
                Observation::new(ObservedOutcome::Success, [12; 32]),
            )
            .unwrap();
        let decision = ResolutionDecision::Confirmed;
        let grant = harness.resolution.issue_bound_one_shot(
            observer,
            scope(),
            Some(expiry),
            action.resolution_binding(decision),
        );
        assert!(matches!(
            action.resolve(grant, decision),
            Err(IndependenceResolutionError::ResolverPrincipalConflict { .. })
        ));
    }

    #[test]
    fn three_way_separation_is_preserved_in_final_evidence() {
        let harness = harness(IndependencePolicy::SeparationOfDuties);
        let expiry = SystemTime::now() + Duration::from_secs(60);
        let action = executed_action(&harness, expiry);
        let actor = action.actor();
        let observer = PrincipalId::new();
        let observer_grant = harness.observation.issue_bound_one_shot::<Observe>(
            observer,
            scope(),
            Some(expiry),
            action.observation_binding(),
        );
        let action = action
            .observe(
                observer_grant,
                Observation::new(ObservedOutcome::Success, [13; 32]),
            )
            .unwrap();
        let resolver = PrincipalId::new();
        let decision = ResolutionDecision::Confirmed;
        let grant = harness.resolution.issue_bound_one_shot(
            resolver,
            scope(),
            Some(expiry),
            action.resolution_binding(decision),
        );
        let (resolved, receipt) = action.resolve(grant, decision).unwrap();

        assert_eq!(receipt.policy(), IndependencePolicy::SeparationOfDuties);
        assert_eq!(receipt.actor(), actor);
        assert_eq!(receipt.observer(), observer);
        assert_eq!(receipt.resolver(), resolver);
        assert_ne!(receipt.execution_domain(), receipt.observer_domain());
        assert_ne!(receipt.execution_domain(), receipt.resolver_domain());
        assert_ne!(receipt.observer_domain(), receipt.resolver_domain());
        assert_ne!(receipt.actor(), receipt.observer());
        assert_ne!(receipt.actor(), receipt.resolver());
        assert_ne!(receipt.observer(), receipt.resolver());
        assert_eq!(resolved.independence_receipt(), &receipt);
    }

    #[test]
    fn wrapper_selected_wrong_observer_domain_fails_closed() {
        let harness = harness(IndependencePolicy::DistinctPrincipal);
        let expiry = SystemTime::now() + Duration::from_secs(60);
        let action = executed_action(&harness, expiry);
        let wrong = AuthorityDomain::new(PrincipalId::new());
        let observer = PrincipalId::new();
        let wrong_grant = wrong.issue_bound_one_shot::<Observe>(
            observer,
            scope(),
            Some(expiry),
            action.observation_binding(),
        );
        assert!(matches!(
            action.observe(
                wrong_grant,
                Observation::new(ObservedOutcome::Success, [14; 32])
            ),
            Err(IndependenceObservationError::ObserverGrant(_))
        ));
    }
}
