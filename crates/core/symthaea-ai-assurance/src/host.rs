// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Host-owned trusted runtime for security-sensitive agent execution.
//!
//! [`crate::trusted`] binds actions to authority domains and revocation epochs,
//! but its low-level transition methods intentionally accept an
//! [`crate::AuthorityVerifier`] and [`std::time::SystemTime`] from the caller.
//! That is useful for deterministic assurance mechanics, but it is too flexible
//! for a concrete autonomous-agent execution boundary: model-controlled call
//! data must not choose either the trust anchor or the time used for expiry.
//!
//! This module adds the stricter integration surface. [`TrustedRuntime`] stores
//! the host-selected execution and observation verifiers internally. Actions
//! admitted through it carry those verifiers across typestate transitions, and
//! all expiry checks use a host-owned wall-clock floor. Callers supply proposal
//! data, grants, executor output digests, and observation evidence -- never a
//! verifier or a validation timestamp.
//!
//! The clock floor is non-decreasing for the lifetime of the runtime. A local
//! wall-clock rollback therefore cannot resurrect authority that has already
//! aged past a later observed time. Durable anti-rollback across process restart
//! remains a deployment responsibility for a later HAL/Xenia-backed trusted
//! clock or persisted monotonic epoch.
//!
//! ```compile_fail
//! use std::time::SystemTime;
//! use symthaea_ai_assurance::{
//!     ActionRisk, AuthorityDomain, PrincipalId, Scope, TrustedRuntime, Write,
//! };
//!
//! let execution = AuthorityDomain::new(PrincipalId::new());
//! let observation = AuthorityDomain::new(PrincipalId::new());
//! let runtime = TrustedRuntime::new(execution.verifier(), observation.verifier());
//! let actor = PrincipalId::new();
//! let scope = Scope::new("workspace", ["symthaea"]).unwrap();
//! let action = runtime
//!     .admit::<Write>(actor, "edit", scope.clone(), b"patch")
//!     .assess(ActionRisk::Reversible);
//! let grant = execution.issue_bound_one_shot::<Write>(
//!     actor,
//!     scope,
//!     None,
//!     action.authorization_binding(),
//! );
//!
//! // The strict host API deliberately accepts neither a verifier nor a time.
//! let _ = action.authorize(grant, SystemTime::UNIX_EPOCH);
//! ```

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::capability::{CapabilityKind, Observe, PrincipalId, Scope};
use crate::trusted::{
    AuthorityDomainId, AuthorityVerifier, TrustError, TrustedAction, TrustedBoundOneShotCapability,
    TrustedEvidenceReceipt,
};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Host-owned runtime boundary for one configured execution trust domain and
/// one configured observation trust domain.
///
/// The runtime can verify and admit authority, but cannot mint it. Capability
/// minting remains in [`crate::AuthorityDomain`] policy code, preserving a
/// separation between grant issuance and concrete tool execution.
#[derive(Debug, Clone)]
pub struct TrustedRuntime {
    execution_verifier: AuthorityVerifier,
    observer_verifier: AuthorityVerifier,
    clock: Arc<MonotonicWallClock>,
}

impl TrustedRuntime {
    /// Construct a host runtime from verifiers selected by trusted host policy.
    ///
    /// The resulting runtime should be retained by the concrete executor. Model
    /// or planner output should never be permitted to replace these verifiers.
    pub fn new(
        execution_verifier: AuthorityVerifier,
        observer_verifier: AuthorityVerifier,
    ) -> Self {
        Self {
            execution_verifier,
            observer_verifier,
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
            clock: Arc::clone(&self.clock),
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
    clock: Arc<MonotonicWallClock>,
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
}

impl<K: CapabilityKind> RuntimeAction<K, Proposed> {
    /// Attach explicit risk without exposing host trust configuration.
    pub fn assess(self, risk: ActionRisk) -> RuntimeAction<K, RiskAssessed> {
        RuntimeAction {
            inner: self.inner.assess(risk),
            execution_verifier: self.execution_verifier,
            observer_verifier: self.observer_verifier,
            clock: self.clock,
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
            clock: self.clock,
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
            clock: self.clock,
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
        let inner = self
            .inner
            .observe(observer, &self.observer_verifier, observation, now)?;
        Ok(RuntimeAction {
            inner,
            execution_verifier: self.execution_verifier,
            observer_verifier: self.observer_verifier,
            clock: self.clock,
        })
    }
}

impl<K: CapabilityKind> RuntimeAction<K, Observed> {
    /// Resolve an independently observed action and emit trusted-domain evidence.
    pub fn resolve(
        self,
        resolution: ResolutionDecision,
    ) -> (RuntimeAction<K, Resolved>, TrustedEvidenceReceipt) {
        let (inner, receipt) = self.inner.resolve(resolution);
        (
            RuntimeAction {
                inner,
                execution_verifier: self.execution_verifier,
                observer_verifier: self.observer_verifier,
                clock: self.clock,
            },
            receipt,
        )
    }
}

impl<K: CapabilityKind> RuntimeAction<K, Resolved> {
    /// Final trusted-domain receipt retained by the resolved state.
    pub fn trusted_receipt(&self) -> &TrustedEvidenceReceipt {
        self.inner.trusted_receipt()
    }
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
        ActionRisk, AuthorityDomain, ObservedOutcome, ResolutionDecision, Write,
    };
    use std::time::Duration;

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    #[test]
    fn strict_runtime_executes_and_observes_without_caller_time_or_verifier() {
        let execution = AuthorityDomain::new(PrincipalId::new());
        let observation_domain = AuthorityDomain::new(PrincipalId::new());
        let runtime = TrustedRuntime::new(execution.verifier(), observation_domain.verifier());
        let actor = PrincipalId::new();
        let observer = PrincipalId::new();
        let action_scope = scope();
        let expires = SystemTime::now() + Duration::from_secs(60);

        let action = runtime
            .admit::<Write>(actor, "edit-source", action_scope.clone(), b"patch-v1")
            .assess(ActionRisk::Reversible);
        let execution_grant = execution.issue_bound_one_shot::<Write>(
            actor,
            action_scope.clone(),
            Some(expires),
            action.authorization_binding(),
        );
        let action = action.authorize(execution_grant).unwrap();
        let action = action.record_execution([7; 32]).unwrap();

        let observer_grant = observation_domain.issue_bound_one_shot::<Observe>(
            observer,
            action_scope,
            Some(expires),
            action.observation_binding(),
        );
        let observed = action
            .observe(
                observer_grant,
                Observation::new(ObservedOutcome::Success, [8; 32]),
            )
            .unwrap();
        let (resolved, receipt) = observed.resolve(ResolutionDecision::Confirmed);

        assert_eq!(resolved.execution_domain(), execution.domain_id());
        assert_eq!(resolved.observer_domain(), observation_domain.domain_id());
        assert_eq!(resolved.trusted_receipt(), &receipt);
    }

    #[test]
    fn caller_cannot_resurrect_expired_grant_by_selecting_old_time() {
        let execution = AuthorityDomain::new(PrincipalId::new());
        let observation_domain = AuthorityDomain::new(PrincipalId::new());
        let runtime = TrustedRuntime::new(execution.verifier(), observation_domain.verifier());
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
        let execution = AuthorityDomain::new(PrincipalId::new());
        let expected_observer = AuthorityDomain::new(PrincipalId::new());
        let wrong_observer = AuthorityDomain::new(PrincipalId::new());
        let runtime = TrustedRuntime::new(execution.verifier(), expected_observer.verifier());
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
