// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trusted-host authority domains layered over the low-level capability core.
//!
//! The low-level [`crate::AuthorityRoot`] API models affine authority values but
//! intentionally does not answer *which root a host trusts*. This module adds
//! that missing trust-anchor layer. A trusted action is bound to one random
//! authority domain and one revocation epoch; unrelated roots cannot supply the
//! typed grant required by the trusted path, and rotating the epoch invalidates
//! outstanding execution grants before side effects occur.
//!
//! Concrete executors should retain their [`AuthorityVerifier`] inside trusted
//! host/tool-adapter state. Model-provided data should never choose the verifier
//! used to admit execution.
//!
//! ```compile_fail
//! use std::time::SystemTime;
//! use symthaea_ai_assurance::{
//!     ActionRisk, AuthorityDomain, AuthorityRoot, PrincipalId, Scope,
//!     TrustedAction, Proposed, Write,
//! };
//!
//! let host = AuthorityDomain::new(PrincipalId::new());
//! let verifier = host.verifier();
//! let actor = PrincipalId::new();
//! let scope = Scope::new("workspace", ["symthaea"]).unwrap();
//! let action = TrustedAction::<Write, Proposed>::propose(
//!     &verifier,
//!     actor,
//!     "edit",
//!     scope.clone(),
//!     b"patch",
//! ).assess(ActionRisk::Reversible);
//!
//! // A self-created low-level root can mint a raw grant, but the trusted path
//! // does not accept that raw capability type.
//! let forged = AuthorityRoot::new(PrincipalId::new()).issue_bound_one_shot::<Write>(
//!     actor,
//!     scope,
//!     None,
//!     action.authorization_binding(),
//! );
//! let _ = action.authorize(forged, &verifier, SystemTime::now());
//! ```

use crate::action::{
    Action, ActionDescriptor, ActionError, ActionId, ActionRisk, Authorized, EvidenceReceipt,
    Executed, Observation, Observed, Proposed, ResolutionDecision, Resolved, RiskAssessed,
};
use crate::capability::{
    AuthorityRoot, BoundOneShotCapability, CapabilityKind, GrantError, GrantMetadata, Observe,
    PrincipalId, Scope,
};
use std::fmt;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::SystemTime;
use uuid::Uuid;

/// Opaque identity of one trusted authority domain.
///
/// IDs are generated internally by [`AuthorityDomain::new`]; callers cannot
/// choose an existing domain id when constructing another domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorityDomainId(Uuid);

impl AuthorityDomainId {
    fn fresh() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the underlying UUID for evidence and diagnostics.
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

/// Monotonic revocation epoch within one authority domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuthorityEpoch(u64);

impl AuthorityEpoch {
    /// Numeric epoch value for evidence and diagnostics.
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Cloneable trust anchor retained by trusted host/tool-adapter code.
///
/// Possession of a verifier does not mint authority. It identifies the domain
/// the host trusts and supplies the current revocation epoch used to reject
/// stale lineages.
#[derive(Debug, Clone)]
pub struct AuthorityVerifier {
    domain_id: AuthorityDomainId,
    epoch: Arc<AtomicU64>,
}

impl AuthorityVerifier {
    /// Trusted authority domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.domain_id
    }

    /// Current revocation epoch.
    pub fn current_epoch(&self) -> AuthorityEpoch {
        AuthorityEpoch(self.epoch.load(Ordering::SeqCst))
    }

    fn validate_lineage(
        &self,
        domain_id: AuthorityDomainId,
        epoch: AuthorityEpoch,
    ) -> Result<(), TrustError> {
        if domain_id != self.domain_id {
            return Err(TrustError::WrongDomain {
                expected: self.domain_id,
                actual: domain_id,
            });
        }

        let current = self.current_epoch();
        if epoch != current {
            return Err(TrustError::RevokedEpoch {
                domain_id,
                lineage_epoch: epoch,
                current_epoch: current,
            });
        }

        Ok(())
    }
}

/// Trusted host authority root plus revocation state.
///
/// The domain should live only in trusted host policy code. Less-trusted code
/// receives narrowly scoped [`TrustedBoundOneShotCapability`] values, while
/// concrete executors retain an [`AuthorityVerifier`] chosen by the host.
#[derive(Debug)]
pub struct AuthorityDomain {
    domain_id: AuthorityDomainId,
    root: AuthorityRoot,
    epoch: Arc<AtomicU64>,
}

impl AuthorityDomain {
    /// Create a fresh trust domain for a host principal.
    pub fn new(principal: PrincipalId) -> Self {
        Self {
            domain_id: AuthorityDomainId::fresh(),
            root: AuthorityRoot::new(principal),
            epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Trusted domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.domain_id
    }

    /// Root host principal.
    pub fn principal(&self) -> PrincipalId {
        self.root.principal()
    }

    /// Create a verifier for trusted adapters in this domain.
    pub fn verifier(&self) -> AuthorityVerifier {
        AuthorityVerifier {
            domain_id: self.domain_id,
            epoch: Arc::clone(&self.epoch),
        }
    }

    /// Mint exact one-shot authority in the current revocation epoch.
    pub fn issue_bound_one_shot<K: CapabilityKind>(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
        binding: [u8; 32],
    ) -> TrustedBoundOneShotCapability<K> {
        let epoch = AuthorityEpoch(self.epoch.load(Ordering::SeqCst));
        TrustedBoundOneShotCapability {
            domain_id: self.domain_id,
            epoch,
            inner: self
                .root
                .issue_bound_one_shot::<K>(subject, scope, expires_at, binding),
        }
    }

    /// Revoke all capabilities and unexecuted action lineages from older epochs.
    ///
    /// New grants and proposals use the returned epoch. Already executed actions
    /// can still be *observed and resolved* so evidence about past side effects
    /// is not lost, but old authorization cannot cross the execution boundary.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        loop {
            let current = self.epoch.load(Ordering::SeqCst);
            let next = current.checked_add(1).ok_or(TrustError::EpochExhausted {
                domain_id: self.domain_id,
            })?;
            match self
                .epoch
                .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return Ok(AuthorityEpoch(next)),
                Err(_) => continue,
            }
        }
    }
}

/// Exact one-shot grant carrying trusted-domain and revocation provenance.
///
/// This type intentionally implements neither `Copy` nor `Clone`.
#[derive(Debug)]
pub struct TrustedBoundOneShotCapability<K: CapabilityKind> {
    domain_id: AuthorityDomainId,
    epoch: AuthorityEpoch,
    inner: BoundOneShotCapability<K>,
}

impl<K: CapabilityKind> TrustedBoundOneShotCapability<K> {
    /// Authority domain that minted this grant.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.domain_id
    }

    /// Revocation epoch in which this grant was minted.
    pub fn epoch(&self) -> AuthorityEpoch {
        self.epoch
    }

    /// Immutable low-level grant metadata.
    pub fn metadata(&self) -> &GrantMetadata {
        self.inner.metadata()
    }

    /// Exact transition digest carried by the grant.
    pub fn binding(&self) -> [u8; 32] {
        self.inner.binding()
    }

    /// Validate domain, revocation epoch, and expiry against a host verifier.
    pub fn validate_with(
        &self,
        verifier: &AuthorityVerifier,
        now: SystemTime,
    ) -> Result<(), TrustError> {
        verifier.validate_lineage(self.domain_id, self.epoch)?;
        self.inner.validate_at(now).map_err(TrustError::Grant)
    }

    fn into_inner(self) -> BoundOneShotCapability<K> {
        self.inner
    }
}

/// Domain-bound action lifecycle for trusted host integrations.
///
/// This wrapper leaves cognition/model semantics outside the assurance kernel
/// while ensuring execution admission is tied to a host-selected trust domain
/// and revocation epoch.
#[derive(Debug)]
pub struct TrustedAction<K: CapabilityKind, S> {
    inner: Action<K, S>,
    execution_domain: AuthorityDomainId,
    execution_epoch: AuthorityEpoch,
    observer_lineage: Option<(AuthorityDomainId, AuthorityEpoch)>,
    trusted_receipt: Option<TrustedEvidenceReceipt>,
}

impl<K: CapabilityKind, S> TrustedAction<K, S> {
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

    /// Trusted execution domain selected when the host admitted the proposal.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_domain
    }

    /// Revocation epoch selected when the host admitted the proposal.
    pub fn execution_epoch(&self) -> AuthorityEpoch {
        self.execution_epoch
    }
}

impl<K: CapabilityKind> TrustedAction<K, Proposed> {
    /// Admit a model/planner proposal into a host-selected trust domain.
    pub fn propose(
        verifier: &AuthorityVerifier,
        actor: PrincipalId,
        kind: impl Into<String>,
        scope: Scope,
        canonical_payload: &[u8],
    ) -> Self {
        Self {
            inner: Action::<K, Proposed>::propose(actor, kind, scope, canonical_payload),
            execution_domain: verifier.domain_id(),
            execution_epoch: verifier.current_epoch(),
            observer_lineage: None,
            trusted_receipt: None,
        }
    }

    /// Attach explicit risk without changing trust provenance.
    pub fn assess(self, risk: ActionRisk) -> TrustedAction<K, RiskAssessed> {
        TrustedAction {
            inner: self.inner.assess(risk),
            execution_domain: self.execution_domain,
            execution_epoch: self.execution_epoch,
            observer_lineage: None,
            trusted_receipt: None,
        }
    }
}

impl<K: CapabilityKind> TrustedAction<K, RiskAssessed> {
    /// Risk classification attached to this action.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact digest a trusted domain grant must bind.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Consume exact authority from the host-selected trust domain.
    pub fn authorize(
        self,
        grant: TrustedBoundOneShotCapability<K>,
        verifier: &AuthorityVerifier,
        now: SystemTime,
    ) -> Result<TrustedAction<K, Authorized>, TrustError> {
        verifier.validate_lineage(self.execution_domain, self.execution_epoch)?;
        grant.validate_with(verifier, now)?;

        let inner = self
            .inner
            .authorize(grant.into_inner(), now)
            .map_err(TrustError::Action)?;

        Ok(TrustedAction {
            inner,
            execution_domain: self.execution_domain,
            execution_epoch: self.execution_epoch,
            observer_lineage: None,
            trusted_receipt: None,
        })
    }
}

impl<K: CapabilityKind> TrustedAction<K, Authorized> {
    /// Exact authorization binding consumed by this action.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Cross the side-effect boundary only while the host-selected epoch remains current.
    pub fn record_execution(
        self,
        verifier: &AuthorityVerifier,
        output_digest: [u8; 32],
        now: SystemTime,
    ) -> Result<TrustedAction<K, Executed>, TrustError> {
        verifier.validate_lineage(self.execution_domain, self.execution_epoch)?;
        let inner = self
            .inner
            .record_execution(output_digest, now)
            .map_err(TrustError::Action)?;

        Ok(TrustedAction {
            inner,
            execution_domain: self.execution_domain,
            execution_epoch: self.execution_epoch,
            observer_lineage: None,
            trusted_receipt: None,
        })
    }
}

impl<K: CapabilityKind> TrustedAction<K, Executed> {
    /// Exact digest that independent observation authority must bind.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Attach independently authorized external observation.
    ///
    /// The observer principal must differ from the action actor. This is stronger
    /// than merely using a second grant and prevents the action principal from
    /// self-grading an externally resolved outcome through this trusted path.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<Observe>,
        observer_verifier: &AuthorityVerifier,
        observation: Observation,
        now: SystemTime,
    ) -> Result<TrustedAction<K, Observed>, TrustError> {
        observer.validate_with(observer_verifier, now)?;

        if observer.metadata().subject() == self.inner.actor() {
            return Err(TrustError::ObserverNotIndependent {
                actor: self.inner.actor(),
                observer: observer.metadata().subject(),
            });
        }

        let observer_lineage = (observer.domain_id(), observer.epoch());
        let inner = self
            .inner
            .observe(observer.into_inner(), observation, now)
            .map_err(TrustError::Action)?;

        Ok(TrustedAction {
            inner,
            execution_domain: self.execution_domain,
            execution_epoch: self.execution_epoch,
            observer_lineage: Some(observer_lineage),
            trusted_receipt: None,
        })
    }
}

impl<K: CapabilityKind> TrustedAction<K, Observed> {
    /// Resolve an observed action and emit domain-aware immutable evidence.
    pub fn resolve(
        self,
        resolution: ResolutionDecision,
    ) -> (TrustedAction<K, Resolved>, TrustedEvidenceReceipt) {
        let observer_lineage = self
            .observer_lineage
            .expect("Observed trusted action always carries observer lineage");
        let (inner, receipt) = self.inner.resolve(resolution);
        let trusted_receipt = TrustedEvidenceReceipt {
            receipt,
            execution_domain: self.execution_domain,
            execution_epoch: self.execution_epoch,
            observer_domain: observer_lineage.0,
            observer_epoch: observer_lineage.1,
        };

        (
            TrustedAction {
                inner,
                execution_domain: self.execution_domain,
                execution_epoch: self.execution_epoch,
                observer_lineage: Some(observer_lineage),
                trusted_receipt: Some(trusted_receipt.clone()),
            },
            trusted_receipt,
        )
    }
}

impl<K: CapabilityKind> TrustedAction<K, Resolved> {
    /// Domain-aware final receipt retained by the resolved typestate.
    pub fn trusted_receipt(&self) -> &TrustedEvidenceReceipt {
        self.trusted_receipt
            .as_ref()
            .expect("Resolved trusted action always carries a trusted receipt")
    }
}

/// Evidence receipt augmented with execution and observer trust-domain lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedEvidenceReceipt {
    receipt: EvidenceReceipt,
    execution_domain: AuthorityDomainId,
    execution_epoch: AuthorityEpoch,
    observer_domain: AuthorityDomainId,
    observer_epoch: AuthorityEpoch,
}

impl TrustedEvidenceReceipt {
    /// Underlying exact-action evidence receipt.
    pub fn receipt(&self) -> &EvidenceReceipt {
        &self.receipt
    }

    /// Trust domain that authorized execution.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_domain
    }

    /// Execution-domain revocation epoch.
    pub fn execution_epoch(&self) -> AuthorityEpoch {
        self.execution_epoch
    }

    /// Trust domain that authorized independent observation.
    pub fn observer_domain(&self) -> AuthorityDomainId {
        self.observer_domain
    }

    /// Observer-domain revocation epoch.
    pub fn observer_epoch(&self) -> AuthorityEpoch {
        self.observer_epoch
    }
}

/// Trust-domain validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustError {
    /// A lineage or grant belongs to a different root trust domain.
    WrongDomain {
        /// Domain the trusted adapter expects.
        expected: AuthorityDomainId,
        /// Domain supplied by the lineage/grant.
        actual: AuthorityDomainId,
    },
    /// A grant/action belongs to an older revocation epoch.
    RevokedEpoch {
        /// Authority domain whose epoch changed.
        domain_id: AuthorityDomainId,
        /// Epoch carried by the stale lineage.
        lineage_epoch: AuthorityEpoch,
        /// Current epoch required by the verifier.
        current_epoch: AuthorityEpoch,
    },
    /// The epoch counter cannot be advanced without overflow.
    EpochExhausted {
        /// Authority domain whose epoch counter is exhausted.
        domain_id: AuthorityDomainId,
    },
    /// The external observer is the same principal as the acting principal.
    ObserverNotIndependent {
        /// Principal that performed/proposed the action.
        actor: PrincipalId,
        /// Principal presented as observer.
        observer: PrincipalId,
    },
    /// Low-level capability validation failed.
    Grant(GrantError),
    /// Low-level action transition validation failed.
    Action(ActionError),
}

impl fmt::Display for TrustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDomain { .. } => write!(f, "authority belongs to a different trust domain"),
            Self::RevokedEpoch { .. } => {
                write!(f, "authority lineage was revoked by epoch rotation")
            }
            Self::EpochExhausted { .. } => write!(f, "authority revocation epoch exhausted"),
            Self::ObserverNotIndependent { .. } => {
                write!(f, "external observer must differ from the acting principal")
            }
            Self::Grant(error) => write!(f, "trusted grant validation failed: {error}"),
            Self::Action(error) => write!(f, "trusted action transition failed: {error}"),
        }
    }
}

impl std::error::Error for TrustError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObservedOutcome, Read, Write};
    use std::time::Duration;

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    #[test]
    fn unrelated_trust_domain_cannot_authorize_action() {
        let trusted = AuthorityDomain::new(PrincipalId::new());
        let attacker = AuthorityDomain::new(PrincipalId::new());
        let trusted_verifier = trusted.verifier();
        let actor = PrincipalId::new();
        let action = TrustedAction::<Write, Proposed>::propose(
            &trusted_verifier,
            actor,
            "edit",
            scope(),
            b"patch",
        )
        .assess(ActionRisk::Reversible);
        let forged = attacker.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
        );

        let result = action.authorize(forged, &trusted_verifier, SystemTime::now());
        assert!(matches!(result, Err(TrustError::WrongDomain { .. })));
    }

    #[test]
    fn epoch_rotation_revokes_unspent_grant() {
        let domain = AuthorityDomain::new(PrincipalId::new());
        let verifier = domain.verifier();
        let actor = PrincipalId::new();
        let action =
            TrustedAction::<Write, Proposed>::propose(&verifier, actor, "edit", scope(), b"patch")
                .assess(ActionRisk::Reversible);
        let grant = domain.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
        );
        domain.revoke_all().unwrap();

        let result = action.authorize(grant, &verifier, SystemTime::now());
        assert!(matches!(result, Err(TrustError::RevokedEpoch { .. })));
    }

    #[test]
    fn epoch_rotation_after_authorization_blocks_execution() {
        let domain = AuthorityDomain::new(PrincipalId::new());
        let verifier = domain.verifier();
        let actor = PrincipalId::new();
        let action =
            TrustedAction::<Write, Proposed>::propose(&verifier, actor, "edit", scope(), b"patch")
                .assess(ActionRisk::Reversible);
        let grant = domain.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
        );
        let authorized = action
            .authorize(grant, &verifier, SystemTime::now())
            .unwrap();
        domain.revoke_all().unwrap();

        let result = authorized.record_execution(&verifier, [9; 32], SystemTime::now());
        assert!(matches!(result, Err(TrustError::RevokedEpoch { .. })));
    }

    #[test]
    fn same_principal_cannot_self_grade_external_observation() {
        let execution = AuthorityDomain::new(PrincipalId::new());
        let observation = AuthorityDomain::new(PrincipalId::new());
        let exec_verifier = execution.verifier();
        let obs_verifier = observation.verifier();
        let actor = PrincipalId::new();
        let action = TrustedAction::<Write, Proposed>::propose(
            &exec_verifier,
            actor,
            "edit",
            scope(),
            b"patch",
        )
        .assess(ActionRisk::Reversible);
        let grant = execution.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
        );
        let executed = action
            .authorize(grant, &exec_verifier, SystemTime::now())
            .unwrap()
            .record_execution(&exec_verifier, [1; 32], SystemTime::now())
            .unwrap();
        let observer = observation.issue_bound_one_shot::<Observe>(
            actor,
            scope(),
            None,
            executed.observation_binding(),
        );
        let result = executed.observe(
            observer,
            &obs_verifier,
            Observation::new(ObservedOutcome::Success, [2; 32]),
            SystemTime::now(),
        );
        assert!(matches!(
            result,
            Err(TrustError::ObserverNotIndependent { .. })
        ));
    }

    #[test]
    fn trusted_receipt_preserves_execution_and_observer_domains() {
        let execution = AuthorityDomain::new(PrincipalId::new());
        let observation = AuthorityDomain::new(PrincipalId::new());
        let exec_verifier = execution.verifier();
        let obs_verifier = observation.verifier();
        let actor = PrincipalId::new();
        let observer_principal = PrincipalId::new();
        let action = TrustedAction::<Write, Proposed>::propose(
            &exec_verifier,
            actor,
            "edit",
            scope(),
            b"patch",
        )
        .assess(ActionRisk::Reversible);
        let grant = execution.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            Some(SystemTime::now() + Duration::from_secs(60)),
            action.authorization_binding(),
        );
        let executed = action
            .authorize(grant, &exec_verifier, SystemTime::now())
            .unwrap()
            .record_execution(&exec_verifier, [3; 32], SystemTime::now())
            .unwrap();
        let observer = observation.issue_bound_one_shot::<Observe>(
            observer_principal,
            scope(),
            None,
            executed.observation_binding(),
        );
        let observed = executed
            .observe(
                observer,
                &obs_verifier,
                Observation::new(ObservedOutcome::Success, [4; 32]),
                SystemTime::now(),
            )
            .unwrap();
        let (resolved, receipt) = observed.resolve(ResolutionDecision::Confirmed);

        assert_eq!(receipt.execution_domain(), execution.domain_id());
        assert_eq!(receipt.observer_domain(), observation.domain_id());
        assert_eq!(resolved.trusted_receipt(), &receipt);
    }

    #[test]
    fn capability_kind_stays_static_through_trusted_wrapper() {
        let domain = AuthorityDomain::new(PrincipalId::new());
        let verifier = domain.verifier();
        let actor = PrincipalId::new();
        let action =
            TrustedAction::<Read, Proposed>::propose(&verifier, actor, "inspect", scope(), b"read")
                .assess(ActionRisk::Observation);
        let grant = domain.issue_bound_one_shot::<Read>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
        );
        assert!(
            action
                .authorize(grant, &verifier, SystemTime::now())
                .is_ok()
        );
    }
}
