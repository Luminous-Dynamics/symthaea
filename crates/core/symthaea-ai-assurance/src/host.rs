// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Host-owned trusted runtime for security-sensitive agent execution.
//!
//! [`crate::trusted`] binds actions to authority domains and revocation epochs,
//! but its lower-level transition methods intentionally accept verifier and time
//! values from the caller. That is useful for deterministic assurance mechanics,
//! but too flexible for a concrete autonomous-agent execution boundary.
//!
//! This module provides the stricter integration surface. [`TrustedRuntime`]
//! stores host-selected execution, observation, and final-resolution trust
//! anchors internally. Actions admitted through it carry those choices across
//! typestate transitions, and all expiry checks use a host-owned wall-clock
//! floor. Model/planner call data never chooses a verifier or validation time.
//!
//! Final resolution is separately authorized. An independently observed action
//! exposes a deterministic resolution binding over the exact action lineage,
//! observation commitment, and proposed [`ResolutionDecision`]. The transition
//! to `Resolved` consumes an opaque one-shot [`crate::ResolutionGrant`] from the
//! runtime-pinned resolver domain. A grant for one decision or observed lineage
//! cannot be substituted onto another.
//!
//! The clock floor is non-decreasing for the lifetime of the runtime. A local
//! wall-clock rollback therefore cannot resurrect authority that has already
//! aged past a later observed time. Durable anti-rollback across process restart
//! remains a deployment responsibility for a later HAL/Xenia-backed trusted
//! clock or persisted monotonic epoch.

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed,
    ObservedOutcome, Proposed, ResolutionDecision, Resolved, RiskAssessed,
};
use crate::capability::{CapabilityKind, GrantId, Observe, PrincipalId, Scope};
use crate::resolution::{ResolutionGrant, ResolutionVerifier};
use crate::trusted::{
    AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError, TrustedAction,
    TrustedBoundOneShotCapability, TrustedEvidenceReceipt,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Host-owned runtime boundary for configured execution, observation, and final
/// resolution trust domains.
///
/// The runtime can verify and admit authority, but cannot mint it. Capability
/// minting remains in trusted policy code, preserving separation between grant
/// issuance and concrete tool execution.
#[derive(Debug, Clone)]
pub struct TrustedRuntime {
    execution_verifier: AuthorityVerifier,
    observer_verifier: AuthorityVerifier,
    resolver_verifier: ResolutionVerifier,
    clock: Arc<MonotonicWallClock>,
}

impl TrustedRuntime {
    /// Construct a host runtime from trust anchors selected by trusted policy.
    ///
    /// The resulting runtime should be retained by concrete host/tool adapters.
    /// Model or planner output should never replace any of these verifiers.
    pub fn new(
        execution_verifier: AuthorityVerifier,
        observer_verifier: AuthorityVerifier,
        resolver_verifier: ResolutionVerifier,
    ) -> Self {
        Self {
            execution_verifier,
            observer_verifier,
            resolver_verifier,
            clock: Arc::new(MonotonicWallClock::new()),
        }
    }

    /// Configured execution authority domain.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_verifier.domain_id()
    }

    /// Configured external-observation authority domain.
    pub fn observer_domain(&self) -> AuthorityDomainId {
        self.observer_verifier.domain_id()
    }

    /// Configured final-resolution authority domain.
    pub fn resolver_domain(&self) -> AuthorityDomainId {
        self.resolver_verifier.domain_id()
    }

    /// Admit proposal data into the host-selected execution trust domain.
    pub fn admit<K: CapabilityKind>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        scope: Scope,
        canonical_payload: &[u8],
    ) -> RuntimeAction<K, Proposed> {
        RuntimeAction {
            inner: TrustedAction::<K, Proposed>::propose(
                &self.execution_verifier,
                actor,
                kind,
                scope,
                canonical_payload,
            ),
            execution_verifier: self.execution_verifier.clone(),
            observer_verifier: self.observer_verifier.clone(),
            resolver_verifier: self.resolver_verifier.clone(),
            clock: Arc::clone(&self.clock),
            resolution_context: None,
            resolution_receipt: None,
        }
    }
}

/// Strict host action lifecycle whose trust anchors and validation clock cannot
/// be substituted by model-provided transition arguments.
#[derive(Debug)]
pub struct RuntimeAction<K: CapabilityKind, S> {
    inner: TrustedAction<K, S>,
    execution_verifier: AuthorityVerifier,
    observer_verifier: AuthorityVerifier,
    resolver_verifier: ResolutionVerifier,
    clock: Arc<MonotonicWallClock>,
    resolution_context: Option<ResolutionContext>,
    resolution_receipt: Option<ResolutionEvidenceReceipt>,
}

impl<K: CapabilityKind, S> RuntimeAction<K, S> {
    /// Stable action identity.
    pub fn id(&self) -> ActionId {
        self.inner.id()
    }

    /// Principal on whose behalf the action was proposed.
    pub fn actor(&self) -> PrincipalId {
        self.inner.actor()
    }

    /// Immutable action descriptor.
    pub fn descriptor(&self) -> &ActionDescriptor {
        self.inner.descriptor()
    }

    /// Execution trust domain retained by the concrete host path.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_verifier.domain_id()
    }

    /// Observation trust domain retained by the concrete host path.
    pub fn observer_domain(&self) -> AuthorityDomainId {
        self.observer_verifier.domain_id()
    }

    /// Final-resolution trust domain retained by the concrete host path.
    pub fn resolver_domain(&self) -> AuthorityDomainId {
        self.resolver_verifier.domain_id()
    }
}

impl<K: CapabilityKind> RuntimeAction<K, Proposed> {
    /// Attach explicit risk without exposing host trust configuration.
    pub fn assess(self, risk: ActionRisk) -> RuntimeAction<K, RiskAssessed> {
        RuntimeAction {
            inner: self.inner.assess(risk),
            execution_verifier: self.execution_verifier,
            observer_verifier: self.observer_verifier,
            resolver_verifier: self.resolver_verifier,
            clock: self.clock,
            resolution_context: None,
            resolution_receipt: None,
        }
    }
}

impl<K: CapabilityKind> RuntimeAction<K, RiskAssessed> {
    /// Risk classification attached to this action.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact digest trusted policy must bind into the execution grant.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Consume exact execution authority using the runtime's retained verifier
    /// and host-owned validation time.
    pub fn authorize(
        self,
        grant: TrustedBoundOneShotCapability<K>,
    ) -> Result<RuntimeAction<K, Authorized>, TrustError> {
        let now = self.clock.now();
        let inner = self
            .inner
            .authorize(grant, &self.execution_verifier, now)?;
        Ok(RuntimeAction {
            inner,
            execution_verifier: self.execution_verifier,
            observer_verifier: self.observer_verifier,
            resolver_verifier: self.resolver_verifier,
            clock: self.clock,
            resolution_context: None,
            resolution_receipt: None,
        })
    }
}

impl<K: CapabilityKind> RuntimeAction<K, Authorized> {
    /// Exact authorization binding consumed by this action.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Cross the side-effect boundary using the runtime's retained verifier and
    /// host-owned validation time.
    pub fn record_execution(
        self,
        output_digest: [u8; 32],
    ) -> Result<RuntimeAction<K, Executed>, TrustError> {
        let now = self.clock.now();
        let inner = self
            .inner
            .record_execution(&self.execution_verifier, output_digest, now)?;
        Ok(RuntimeAction {
            inner,
            execution_verifier: self.execution_verifier,
            observer_verifier: self.observer_verifier,
            resolver_verifier: self.resolver_verifier,
            clock: self.clock,
            resolution_context: None,
            resolution_receipt: None,
        })
    }
}

impl<K: CapabilityKind> RuntimeAction<K, Executed> {
    /// Exact digest independent observation authority must bind.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Attach independent observation using only the runtime-configured observer
    /// trust domain and host-owned validation time.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<Observe>,
        observation: Observation,
    ) -> Result<RuntimeAction<K, Observed>, TrustError> {
        let now = self.clock.now();
        let observation_binding = self.inner.observation_binding();
        let resolution_context = ResolutionContext {
            observation_binding,
            observation: observation.clone(),
        };
        let inner = self
            .inner
            .observe(observer, &self.observer_verifier, observation, now)?;
        Ok(RuntimeAction {
            inner,
            execution_verifier: self.execution_verifier,
            observer_verifier: self.observer_verifier,
            resolver_verifier: self.resolver_verifier,
            clock: self.clock,
            resolution_context: Some(resolution_context),
            resolution_receipt: None,
        })
    }
}

impl<K: CapabilityKind> RuntimeAction<K, Observed> {
    /// Exact digest resolver policy must bind for this observed lineage and
    /// proposed final decision.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        let context = self
            .resolution_context
            .as_ref()
            .expect("Observed runtime action always carries resolution context");
        compute_resolution_binding(
            self.inner.id(),
            self.inner.descriptor().fingerprint(),
            self.execution_verifier.domain_id(),
            self.observer_verifier.domain_id(),
            self.resolver_verifier.domain_id(),
            context,
            decision,
        )
    }

    /// Resolve an independently observed action using exact one-shot authority
    /// from the runtime-pinned resolver domain.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<(RuntimeAction<K, Resolved>, ResolutionEvidenceReceipt), ResolutionError> {
        let now = self.clock.now();
        grant
            .validate_with(&self.resolver_verifier, now)
            .map_err(ResolutionError::Trust)?;

        if !grant.metadata().scope().contains(self.inner.descriptor().scope()) {
            return Err(ResolutionError::ScopeMismatch {
                granted: grant.metadata().scope().clone(),
                required: self.inner.descriptor().scope().clone(),
            });
        }

        let expected_binding = self.resolution_binding(decision);
        if grant.binding() != expected_binding {
            return Err(ResolutionError::BindingMismatch {
                expected: expected_binding,
                actual: grant.binding(),
            });
        }

        let resolver_grant_id = grant.metadata().grant_id();
        let resolver = grant.metadata().subject();
        let resolver_domain = grant.domain_id();
        let resolver_epoch = grant.epoch();
        let resolution_binding = grant.binding();

        let (inner, trusted_receipt) = self.inner.resolve(decision);
        let resolution_receipt = ResolutionEvidenceReceipt {
            trusted_receipt,
            resolver_grant_id,
            resolver,
            resolver_domain,
            resolver_epoch,
            resolution_binding,
            decision,
        };

        let retained_receipt = resolution_receipt.clone();
        Ok((
            RuntimeAction {
                inner,
                execution_verifier: self.execution_verifier,
                observer_verifier: self.observer_verifier,
                resolver_verifier: self.resolver_verifier,
                clock: self.clock,
                resolution_context: self.resolution_context,
                resolution_receipt: Some(retained_receipt),
            },
            resolution_receipt,
        ))
    }
}

impl<K: CapabilityKind> RuntimeAction<K, Resolved> {
    /// Lower-level trusted execution/observation receipt retained by the action.
    pub fn trusted_receipt(&self) -> &TrustedEvidenceReceipt {
        self.inner.trusted_receipt()
    }

    /// Final receipt including independently attributable resolver lineage.
    pub fn resolution_receipt(&self) -> &ResolutionEvidenceReceipt {
        self.resolution_receipt
            .as_ref()
            .expect("Resolved runtime action always carries resolution evidence")
    }
}

#[derive(Debug, Clone)]
struct ResolutionContext {
    observation_binding: [u8; 32],
    observation: Observation,
}

/// Immutable final evidence augmented with resolver authority lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionEvidenceReceipt {
    trusted_receipt: TrustedEvidenceReceipt,
    resolver_grant_id: GrantId,
    resolver: PrincipalId,
    resolver_domain: AuthorityDomainId,
    resolver_epoch: AuthorityEpoch,
    resolution_binding: [u8; 32],
    decision: ResolutionDecision,
}

impl ResolutionEvidenceReceipt {
    /// Execution/observation evidence produced by the trusted lower layer.
    pub fn trusted_receipt(&self) -> &TrustedEvidenceReceipt {
        &self.trusted_receipt
    }

    /// One-shot resolver grant consumed by the final transition.
    pub fn resolver_grant_id(&self) -> GrantId {
        self.resolver_grant_id
    }

    /// Principal that held final-resolution authority.
    pub fn resolver(&self) -> PrincipalId {
        self.resolver
    }

    /// Resolver trust domain selected by the host runtime.
    pub fn resolver_domain(&self) -> AuthorityDomainId {
        self.resolver_domain
    }

    /// Resolver revocation epoch consumed by this decision.
    pub fn resolver_epoch(&self) -> AuthorityEpoch {
        self.resolver_epoch
    }

    /// Exact observed-lineage + decision digest authorized by resolver policy.
    pub fn resolution_binding(&self) -> [u8; 32] {
        self.resolution_binding
    }

    /// Final authorized interpretation.
    pub fn decision(&self) -> ResolutionDecision {
        self.decision
    }
}

/// Failure to cross the strict final-resolution boundary.
#[derive(Debug)]
pub enum ResolutionError {
    /// Resolver domain, epoch, or expiry validation failed.
    Trust(TrustError),
    /// Resolver authority did not cover the action scope.
    ScopeMismatch {
        /// Scope carried by resolver authority.
        granted: Scope,
        /// Scope required by the observed action.
        required: Scope,
    },
    /// Resolver grant was minted for another observed lineage or decision.
    BindingMismatch {
        /// Exact digest required for this observed lineage and decision.
        expected: [u8; 32],
        /// Digest carried by the supplied resolver grant.
        actual: [u8; 32],
    },
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trust(error) => write!(f, "resolution authority validation failed: {error}"),
            Self::ScopeMismatch { .. } => {
                write!(f, "resolution authority does not cover action scope")
            }
            Self::BindingMismatch { .. } => {
                write!(f, "resolution grant is bound to another lineage or decision")
            }
        }
    }
}

impl std::error::Error for ResolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Trust(error) => Some(error),
            Self::ScopeMismatch { .. } | Self::BindingMismatch { .. } => None,
        }
    }
}

fn compute_resolution_binding(
    action_id: ActionId,
    action_fingerprint: [u8; 32],
    execution_domain: AuthorityDomainId,
    observer_domain: AuthorityDomainId,
    resolver_domain: AuthorityDomainId,
    context: &ResolutionContext,
    decision: ResolutionDecision,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/resolve-v1\0");
    hash_field(&mut hasher, action_id.as_uuid().as_bytes());
    hash_field(&mut hasher, &action_fingerprint);
    hash_field(&mut hasher, execution_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, observer_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, resolver_domain.as_uuid().as_bytes());
    hash_field(&mut hasher, &context.observation_binding);
    hash_field(&mut hasher, &[observed_outcome_code(context.observation.outcome())]);
    hash_field(&mut hasher, &context.observation.evidence_digest());
    hash_field(&mut hasher, &[resolution_decision_code(decision)]);
    *hasher.finalize().as_bytes()
}

fn observed_outcome_code(outcome: ObservedOutcome) -> u8 {
    match outcome {
        ObservedOutcome::Success => 0,
        ObservedOutcome::Partial => 1,
        ObservedOutcome::NoEffect => 2,
        ObservedOutcome::SafeFailure => 3,
        ObservedOutcome::UnsafeFailure => 4,
    }
}

fn resolution_decision_code(decision: ResolutionDecision) -> u8 {
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
struct MonotonicWallClock {
    last: Mutex<SystemTime>,
}

impl MonotonicWallClock {
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

    #[cfg(test)]
    fn raise_floor(&self, floor: SystemTime) {
        let mut last = self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if floor > *last {
            *last = floor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActionRisk, AuthorityDomain, ObservedOutcome, ResolutionAuthorityDomain, Write,
    };
    use std::time::Duration;

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    fn runtime(
    ) -> (
        AuthorityDomain,
        AuthorityDomain,
        ResolutionAuthorityDomain,
        TrustedRuntime,
    ) {
        let execution = AuthorityDomain::new(PrincipalId::new());
        let observation = AuthorityDomain::new(PrincipalId::new());
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let runtime = TrustedRuntime::new(
            execution.verifier(),
            observation.verifier(),
            resolution.verifier(),
        );
        (execution, observation, resolution, runtime)
    }

    fn observed_action(
        execution: &AuthorityDomain,
        observation_domain: &AuthorityDomain,
        runtime: &TrustedRuntime,
        actor: PrincipalId,
        observer: PrincipalId,
        action_scope: Scope,
        outcome: ObservedOutcome,
        evidence_digest: [u8; 32],
    ) -> RuntimeAction<Write, Observed> {
        let action = runtime
            .admit::<Write>(actor, "edit-source", action_scope.clone(), b"patch-v1")
            .assess(ActionRisk::Reversible);
        let execution_grant = execution.issue_bound_one_shot::<Write>(
            actor,
            action_scope.clone(),
            None,
            action.authorization_binding(),
        );
        let action = action
            .authorize(execution_grant)
            .unwrap()
            .record_execution([7; 32])
            .unwrap();
        let observer_grant = observation_domain.issue_bound_one_shot::<Observe>(
            observer,
            action_scope,
            None,
            action.observation_binding(),
        );
        action
            .observe(observer_grant, Observation::new(outcome, evidence_digest))
            .unwrap()
    }

    #[test]
    fn strict_runtime_requires_exact_resolution_authority() {
        let (execution, observation, resolution, runtime) = runtime();
        let actor = PrincipalId::new();
        let observer = PrincipalId::new();
        let resolver = PrincipalId::new();
        let action_scope = scope();
        let observed = observed_action(
            &execution,
            &observation,
            &runtime,
            actor,
            observer,
            action_scope.clone(),
            ObservedOutcome::Success,
            [8; 32],
        );
        let binding = observed.resolution_binding(ResolutionDecision::Confirmed);
        let resolution_grant =
            resolution.issue_bound_one_shot(resolver, action_scope, None, binding);

        let (resolved, receipt) = observed
            .resolve(resolution_grant, ResolutionDecision::Confirmed)
            .unwrap();

        assert_eq!(resolved.execution_domain(), execution.domain_id());
        assert_eq!(resolved.observer_domain(), observation.domain_id());
        assert_eq!(resolved.resolver_domain(), resolution.domain_id());
        assert_eq!(receipt.resolver(), resolver);
        assert_eq!(receipt.decision(), ResolutionDecision::Confirmed);
        assert_eq!(resolved.resolution_receipt(), &receipt);
    }

    #[test]
    fn grant_for_confirmed_cannot_authorize_contradicted() {
        let (execution, observation, resolution, runtime) = runtime();
        let action_scope = scope();
        let observed = observed_action(
            &execution,
            &observation,
            &runtime,
            PrincipalId::new(),
            PrincipalId::new(),
            action_scope.clone(),
            ObservedOutcome::UnsafeFailure,
            [9; 32],
        );
        let grant = resolution.issue_bound_one_shot(
            PrincipalId::new(),
            action_scope,
            None,
            observed.resolution_binding(ResolutionDecision::Confirmed),
        );

        let result = observed.resolve(grant, ResolutionDecision::Contradicted);
        assert!(matches!(result, Err(ResolutionError::BindingMismatch { .. })));
    }

    #[test]
    fn runtime_pins_resolver_domain() {
        let (execution, observation, expected_resolution, runtime) = runtime();
        let wrong_resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let action_scope = scope();
        let observed = observed_action(
            &execution,
            &observation,
            &runtime,
            PrincipalId::new(),
            PrincipalId::new(),
            action_scope.clone(),
            ObservedOutcome::Success,
            [2; 32],
        );
        let grant = wrong_resolution.issue_bound_one_shot(
            PrincipalId::new(),
            action_scope,
            None,
            observed.resolution_binding(ResolutionDecision::Confirmed),
        );

        assert!(matches!(
            observed.resolve(grant, ResolutionDecision::Confirmed),
            Err(ResolutionError::Trust(_))
        ));
        assert_ne!(wrong_resolution.domain_id(), expected_resolution.domain_id());
    }

    #[test]
    fn resolver_epoch_revocation_blocks_final_decision() {
        let (execution, observation, resolution, runtime) = runtime();
        let action_scope = scope();
        let observed = observed_action(
            &execution,
            &observation,
            &runtime,
            PrincipalId::new(),
            PrincipalId::new(),
            action_scope.clone(),
            ObservedOutcome::Success,
            [3; 32],
        );
        let grant = resolution.issue_bound_one_shot(
            PrincipalId::new(),
            action_scope,
            None,
            observed.resolution_binding(ResolutionDecision::Confirmed),
        );
        resolution.revoke_all().unwrap();

        assert!(matches!(
            observed.resolve(grant, ResolutionDecision::Confirmed),
            Err(ResolutionError::Trust(_))
        ));
    }

    #[test]
    fn caller_cannot_resurrect_expired_execution_grant_by_selecting_old_time() {
        let (execution, _observation, _resolution, runtime) = runtime();
        let actor = PrincipalId::new();
        let action_scope = scope();
        let action = runtime
            .admit::<Write>(actor, "edit", action_scope.clone(), b"patch")
            .assess(ActionRisk::Reversible);
        let grant = execution.issue_bound_one_shot::<Write>(
            actor,
            action_scope,
            Some(SystemTime::UNIX_EPOCH),
            action.authorization_binding(),
        );

        assert!(action.authorize(grant).is_err());
    }

    #[test]
    fn runtime_pins_observer_verifier() {
        let (execution, expected_observer, resolution, _runtime) = runtime();
        let wrong_observer = AuthorityDomain::new(PrincipalId::new());
        let runtime = TrustedRuntime::new(
            execution.verifier(),
            expected_observer.verifier(),
            resolution.verifier(),
        );
        let actor = PrincipalId::new();
        let observer = PrincipalId::new();
        let action_scope = scope();

        let action = runtime
            .admit::<Write>(actor, "edit", action_scope.clone(), b"patch")
            .assess(ActionRisk::Reversible);
        let execution_grant = execution.issue_bound_one_shot::<Write>(
            actor,
            action_scope.clone(),
            None,
            action.authorization_binding(),
        );
        let action = action
            .authorize(execution_grant)
            .unwrap()
            .record_execution([1; 32])
            .unwrap();

        let wrong_grant = wrong_observer.issue_bound_one_shot::<Observe>(
            observer,
            action_scope,
            None,
            action.observation_binding(),
        );
        let result = action.observe(
            wrong_grant,
            Observation::new(ObservedOutcome::Success, [2; 32]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn wall_clock_floor_is_non_decreasing() {
        let clock = MonotonicWallClock::new();
        let future = SystemTime::now() + Duration::from_secs(3600);
        clock.raise_floor(future);
        assert_eq!(clock.now(), future);
    }
}
