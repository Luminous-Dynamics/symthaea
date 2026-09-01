// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conserved multidimensional resource authority for autonomous execution.
//!
//! Capability authority answers whether an action may perform an effect. Budget
//! authority answers how much resource may be reserved for that action. The two
//! are deliberately separate: possession of `Write` or `Execute` authority does
//! not imply unlimited compute, memory, network, storage, subprocess, or model
//! inference capacity.
//!
//! This module implements the accounting substrate before concrete platform
//! enforcement adapters:
//!
//! - a host-owned [`BudgetAuthorityDomain`] with an atomically conserved pool;
//! - deterministic [`BudgetProfile`] and per-dimension enforcement classes;
//! - opaque non-`Clone` [`BudgetLease`] values bound to an exact action digest;
//! - host-owned validation time and revocation epochs;
//! - race-safe root reservations;
//! - affine split semantics that consume one parent lease and produce conserved
//!   parent-remainder + child leases;
//! - recoverable affine split rejection that returns the exact parent lease;
//! - explicit release of unused reservations back to the pool.
//!
//! Dropping a lease without releasing it intentionally leaks *capacity*, not
//! authority. That is fail-safe: abandoned reservations can reduce availability
//! but cannot mint additional resource authority. Trusted pre-effect transitions
//! should prefer recoverable APIs where the runtime can prove no effect occurred.
//!
//! Concrete execution integration must still distinguish accounting from real
//! enforcement. `CoreMetered` means the future adapter routes usage through a
//! core meter; `ExternalHard` requires an external enforcement-evidence digest;
//! `Measured` and `Soft` must never be presented as hard ceilings.

use crate::capability::{GrantId, PrincipalId, Read, Scope};
use crate::trusted::{
    AuthorityDomain, AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError,
    TrustedBoundOneShotCapability,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

const DIMENSION_COUNT: usize = 8;

/// Generic resource dimensions governed by the first budget profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BudgetDimension {
    /// Elapsed wall-clock duration in milliseconds.
    WallTimeMillis = 0,
    /// Abstract compute/CPU units supplied by the adapter.
    ComputeUnits = 1,
    /// Memory ceiling/accounting in bytes.
    MemoryBytes = 2,
    /// Bytes written to persistent or externally visible storage.
    BytesWritten = 3,
    /// Network payload/accounting bytes.
    NetworkBytes = 4,
    /// Network request count.
    NetworkRequests = 5,
    /// Child process or child-agent count.
    Subprocesses = 6,
    /// Model/inference token units.
    ModelTokens = 7,
}

impl BudgetDimension {
    /// All dimensions in canonical hashing/accounting order.
    pub const ALL: [Self; DIMENSION_COUNT] = [
        Self::WallTimeMillis,
        Self::ComputeUnits,
        Self::MemoryBytes,
        Self::BytesWritten,
        Self::NetworkBytes,
        Self::NetworkRequests,
        Self::Subprocesses,
        Self::ModelTokens,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// Quantities for all budget dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BudgetQuantities {
    values: [u64; DIMENSION_COUNT],
}

impl BudgetQuantities {
    /// Construct an all-zero quantity vector.
    pub const fn zero() -> Self {
        Self {
            values: [0; DIMENSION_COUNT],
        }
    }

    /// Return one dimension's quantity.
    pub fn get(self, dimension: BudgetDimension) -> u64 {
        self.values[dimension.index()]
    }

    /// Return a copy with one dimension replaced.
    pub fn with(mut self, dimension: BudgetDimension, value: u64) -> Self {
        self.values[dimension.index()] = value;
        self
    }

    /// Return true when every dimension fits within `available`.
    pub fn fits_within(self, available: Self) -> bool {
        BudgetDimension::ALL
            .iter()
            .all(|dimension| self.get(*dimension) <= available.get(*dimension))
    }

    fn first_excess(self, available: Self) -> Option<(BudgetDimension, u64, u64)> {
        BudgetDimension::ALL.iter().find_map(|dimension| {
            let requested = self.get(*dimension);
            let remaining = available.get(*dimension);
            (requested > remaining).then_some((*dimension, requested, remaining))
        })
    }

    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let mut values = [0; DIMENSION_COUNT];
        for dimension in BudgetDimension::ALL {
            values[dimension.index()] = self.get(dimension).checked_sub(rhs.get(dimension))?;
        }
        Some(Self { values })
    }

    fn checked_add(self, rhs: Self) -> Option<Self> {
        let mut values = [0; DIMENSION_COUNT];
        for dimension in BudgetDimension::ALL {
            values[dimension.index()] = self.get(dimension).checked_add(rhs.get(dimension))?;
        }
        Some(Self { values })
    }
}

/// Truth class for how a concrete adapter enforces or observes a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnforcementClass {
    /// Usage is intended to be admitted through a core accounting meter.
    CoreMetered,
    /// A platform mechanism outside this crate claims a hard ceiling.
    ExternalHard,
    /// Usage is measured/reported but not guaranteed to be prevented.
    Measured,
    /// Policy target only; no enforcement claim is made.
    Soft,
}

impl EnforcementClass {
    fn code(self) -> u8 {
        match self {
            Self::CoreMetered => 0,
            Self::ExternalHard => 1,
            Self::Measured => 2,
            Self::Soft => 3,
        }
    }
}

/// Per-dimension enforcement truth labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetEnforcement {
    classes: [EnforcementClass; DIMENSION_COUNT],
}

impl BudgetEnforcement {
    /// Construct a profile that makes only soft policy-target claims.
    pub const fn soft() -> Self {
        Self {
            classes: [EnforcementClass::Soft; DIMENSION_COUNT],
        }
    }

    /// Return the enforcement class for one dimension.
    pub fn get(self, dimension: BudgetDimension) -> EnforcementClass {
        self.classes[dimension.index()]
    }

    /// Return a copy with one dimension's class replaced.
    pub fn with(mut self, dimension: BudgetDimension, class: EnforcementClass) -> Self {
        self.classes[dimension.index()] = class;
        self
    }

    fn requires_external_evidence(self) -> bool {
        BudgetDimension::ALL
            .iter()
            .any(|dimension| self.get(*dimension) == EnforcementClass::ExternalHard)
    }
}

impl Default for BudgetEnforcement {
    fn default() -> Self {
        Self::soft()
    }
}

/// Immutable budget profile shared by a conserved pool and all leases from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetProfile {
    limits: BudgetQuantities,
    enforcement: BudgetEnforcement,
    external_enforcement_evidence_digest: Option<[u8; 32]>,
    digest: [u8; 32],
}

impl BudgetProfile {
    /// Construct an immutable pool profile.
    ///
    /// Any `ExternalHard` dimension requires an explicit evidence digest naming
    /// the external enforcement configuration/mechanism. The core records this
    /// claim but does not independently prove the external mechanism works.
    pub fn new(
        limits: BudgetQuantities,
        enforcement: BudgetEnforcement,
        external_enforcement_evidence_digest: Option<[u8; 32]>,
    ) -> Result<Self, BudgetError> {
        if enforcement.requires_external_evidence()
            && external_enforcement_evidence_digest.is_none()
        {
            return Err(BudgetError::MissingExternalEnforcementEvidence);
        }
        let digest =
            compute_profile_digest(limits, enforcement, external_enforcement_evidence_digest);
        Ok(Self {
            limits,
            enforcement,
            external_enforcement_evidence_digest,
            digest,
        })
    }

    /// Maximum quantities conserved by the root pool.
    pub fn limits(&self) -> BudgetQuantities {
        self.limits
    }

    /// Enforcement truth labels for this profile.
    pub fn enforcement(&self) -> BudgetEnforcement {
        self.enforcement
    }

    /// Optional digest naming the external hard-enforcement configuration.
    pub fn external_enforcement_evidence_digest(&self) -> Option<[u8; 32]> {
        self.external_enforcement_evidence_digest
    }

    /// Domain-separated profile digest preserved in lease evidence.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Host-owned quantitative authority domain and conserved reservation pool.
#[derive(Debug)]
pub struct BudgetAuthorityDomain {
    inner: AuthorityDomain,
    profile: BudgetProfile,
    ledger: Arc<Mutex<BudgetLedger>>,
    clock: Arc<BudgetClock>,
    control: Mutex<()>,
}

impl BudgetAuthorityDomain {
    /// Create a fresh budget domain whose initial remaining capacity equals the
    /// profile limits.
    pub fn new(principal: PrincipalId, profile: BudgetProfile) -> Self {
        Self {
            inner: AuthorityDomain::new(principal),
            ledger: Arc::new(Mutex::new(BudgetLedger {
                total: profile.limits(),
                remaining: profile.limits(),
            })),
            profile,
            clock: Arc::new(BudgetClock::new()),
            control: Mutex::new(()),
        }
    }

    /// Budget trust-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Budget root principal.
    pub fn principal(&self) -> PrincipalId {
        self.inner.principal()
    }

    /// Immutable pool profile.
    pub fn profile(&self) -> &BudgetProfile {
        &self.profile
    }

    /// Current unreserved root-pool capacity.
    pub fn remaining(&self) -> BudgetQuantities {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remaining
    }

    /// Create a verifier retained by a strict execution host.
    pub fn verifier(&self) -> BudgetVerifier {
        BudgetVerifier {
            inner: self.inner.verifier(),
            profile_digest: self.profile.digest(),
            clock: Arc::clone(&self.clock),
        }
    }

    /// Atomically reserve root-pool capacity for one exact action.
    pub fn reserve(
        &self,
        subject: PrincipalId,
        scope: Scope,
        action_binding: [u8; 32],
        allocation: BudgetQuantities,
        expires_at: Option<SystemTime>,
    ) -> Result<BudgetLease, BudgetError> {
        let _control = self
            .control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = self.clock.now();
        if expires_at.is_some_and(|expiry| expiry < now) {
            return Err(BudgetError::ExpiredReservationRequest);
        }

        {
            let mut ledger = self
                .ledger
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((dimension, requested, remaining)) =
                allocation.first_excess(ledger.remaining)
            {
                return Err(BudgetError::InsufficientBudget {
                    dimension,
                    requested,
                    remaining,
                });
            }
            ledger.remaining = ledger
                .remaining
                .checked_sub(allocation)
                .ok_or(BudgetError::ArithmeticInvariant)?;
        }

        Ok(self.issue_lease(subject, scope, action_binding, allocation, expires_at, None))
    }

    /// Compatibility affine split API.
    ///
    /// This preserves the original lossy error shape. Trusted callers that need
    /// to avoid capacity-burning pre-effect failures should use
    /// [`Self::split_recoverable`] instead.
    pub fn split(
        &self,
        parent: BudgetLease,
        child_subject: PrincipalId,
        child_scope: Scope,
        child_action_binding: [u8; 32],
        child_allocation: BudgetQuantities,
        child_expires_at: Option<SystemTime>,
    ) -> Result<(BudgetLease, BudgetLease), BudgetError> {
        self.split_recoverable(
            parent,
            child_subject,
            child_scope,
            child_action_binding,
            child_allocation,
            child_expires_at,
        )
        .map_err(BudgetSplitFailure::into_error)
    }

    /// Affinely split one lease while returning the exact original parent on
    /// any rejection before child/remainder issuance.
    ///
    /// Validation and issuance are serialized under the budget domain's private
    /// control lock. One host-owned time snapshot is used for parent validation
    /// and child-expiry checks. On failure no output lease is minted, the root
    /// ledger is unchanged, and [`BudgetSplitFailure`] owns the same parent lease
    /// object supplied by the caller.
    pub fn split_recoverable(
        &self,
        parent: BudgetLease,
        child_subject: PrincipalId,
        child_scope: Scope,
        child_action_binding: [u8; 32],
        child_allocation: BudgetQuantities,
        child_expires_at: Option<SystemTime>,
    ) -> Result<(BudgetLease, BudgetLease), BudgetSplitFailure> {
        let _control = self
            .control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = self.clock.now();
        let verifier = self.verifier();

        if let Err(error) = parent.validate_with_at(&verifier, now) {
            return Err(BudgetSplitFailure::new(parent, error));
        }
        if !parent.scope().contains(&child_scope) {
            let error = BudgetError::ScopeWidening {
                parent: parent.scope().clone(),
                requested: child_scope.clone(),
            };
            return Err(BudgetSplitFailure::new(parent, error));
        }
        if let Some(child_expiry) = child_expires_at {
            if child_expiry < now {
                return Err(BudgetSplitFailure::new(
                    parent,
                    BudgetError::ExpiredDelegationRequest {
                        requested: child_expiry,
                        now,
                    },
                ));
            }
        }
        if let Some(parent_expiry) = parent.expires_at() {
            match child_expires_at {
                Some(child_expiry) if child_expiry <= parent_expiry => {}
                _ => {
                    return Err(BudgetSplitFailure::new(
                        parent,
                        BudgetError::ExpiryWidening {
                            parent: Some(parent_expiry),
                            requested: child_expires_at,
                        },
                    ));
                }
            }
        }
        if let Some((dimension, requested, remaining)) =
            child_allocation.first_excess(parent.allocation)
        {
            return Err(BudgetSplitFailure::new(
                parent,
                BudgetError::InsufficientBudget {
                    dimension,
                    requested,
                    remaining,
                },
            ));
        }

        let Some(remainder) = parent.allocation.checked_sub(child_allocation) else {
            return Err(BudgetSplitFailure::new(
                parent,
                BudgetError::ArithmeticInvariant,
            ));
        };
        let parent_id = parent.lease_id();
        let parent_subject = parent.subject();
        let parent_scope = parent.scope().clone();
        let parent_action_binding = parent.action_binding;
        let parent_expires_at = parent.expires_at();

        let remainder_lease = self.issue_lease(
            parent_subject,
            parent_scope,
            parent_action_binding,
            remainder,
            parent_expires_at,
            Some(parent_id),
        );
        let child_lease = self.issue_lease(
            child_subject,
            child_scope,
            child_action_binding,
            child_allocation,
            child_expires_at,
            Some(parent_id),
        );
        Ok((remainder_lease, child_lease))
    }

    /// Rotate the budget epoch so outstanding leases fail validation before use.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        let _control = self
            .control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.inner.revoke_all()
    }

    fn issue_lease(
        &self,
        subject: PrincipalId,
        scope: Scope,
        action_binding: [u8; 32],
        allocation: BudgetQuantities,
        expires_at: Option<SystemTime>,
        parent_lease_id: Option<GrantId>,
    ) -> BudgetLease {
        let binding = compute_lease_binding(
            subject,
            &scope,
            action_binding,
            allocation,
            self.profile.digest(),
            parent_lease_id,
        );
        let attestation = self
            .inner
            .issue_bound_one_shot::<Read>(subject, scope, expires_at, binding);
        BudgetLease {
            attestation,
            profile: self.profile.clone(),
            allocation,
            action_binding,
            parent_lease_id,
            ledger: Arc::clone(&self.ledger),
        }
    }
}

/// Host-retained verifier for one budget domain/profile.
#[derive(Debug, Clone)]
pub struct BudgetVerifier {
    inner: AuthorityVerifier,
    profile_digest: [u8; 32],
    clock: Arc<BudgetClock>,
}

impl BudgetVerifier {
    /// Budget authority-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Current budget revocation epoch.
    pub fn current_epoch(&self) -> AuthorityEpoch {
        self.inner.current_epoch()
    }

    /// Immutable budget profile digest expected by this verifier.
    pub fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }
}

/// Non-`Clone`, non-`Copy` exact-action quantitative reservation.
#[derive(Debug)]
pub struct BudgetLease {
    attestation: TrustedBoundOneShotCapability<Read>,
    profile: BudgetProfile,
    allocation: BudgetQuantities,
    action_binding: [u8; 32],
    parent_lease_id: Option<GrantId>,
    ledger: Arc<Mutex<BudgetLedger>>,
}

impl BudgetLease {
    /// Unique lease identity, represented by the exact attestation grant id.
    pub fn lease_id(&self) -> GrantId {
        self.attestation.metadata().grant_id()
    }

    /// Parent lease consumed to create this lease, if this is a split lineage.
    pub fn parent_lease_id(&self) -> Option<GrantId> {
        self.parent_lease_id
    }

    /// Principal that holds this quantitative authority.
    pub fn subject(&self) -> PrincipalId {
        self.attestation.metadata().subject()
    }

    /// Logical resource/action scope covered by this reservation.
    pub fn scope(&self) -> &Scope {
        self.attestation.metadata().scope()
    }

    /// Optional lease expiry.
    pub fn expires_at(&self) -> Option<SystemTime> {
        self.attestation.metadata().expires_at()
    }

    /// Budget authority domain that reserved this capacity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.attestation.domain_id()
    }

    /// Budget revocation epoch in which the lease was created.
    pub fn epoch(&self) -> AuthorityEpoch {
        self.attestation.epoch()
    }

    /// Immutable profile governing this lease.
    pub fn profile(&self) -> &BudgetProfile {
        &self.profile
    }

    /// Reserved quantitative envelope.
    pub fn allocation(&self) -> BudgetQuantities {
        self.allocation
    }

    /// Exact action authorization binding this budget was reserved for.
    pub fn action_binding(&self) -> [u8; 32] {
        self.action_binding
    }

    /// Validate trust domain, epoch, expiry, profile identity, attestation
    /// binding, holder, scope coverage, and exact action binding.
    pub fn validate_for(
        &self,
        verifier: &BudgetVerifier,
        subject: PrincipalId,
        required_scope: &Scope,
        action_binding: [u8; 32],
    ) -> Result<(), BudgetError> {
        self.validate_with(verifier)?;
        if self.subject() != subject {
            return Err(BudgetError::WrongSubject {
                expected: subject,
                actual: self.subject(),
            });
        }
        if !self.scope().contains(required_scope) {
            return Err(BudgetError::ScopeMismatch {
                granted: self.scope().clone(),
                required: required_scope.clone(),
            });
        }
        if self.action_binding != action_binding {
            return Err(BudgetError::ActionBindingMismatch);
        }
        Ok(())
    }

    /// Explicitly release this reservation back to the root pool.
    ///
    /// The lease is consumed so safe code cannot release the same reservation
    /// twice. Release is allowed even after revocation because returning capacity
    /// cannot create additional authority.
    pub fn release(self) -> Result<BudgetReleaseReceipt, BudgetError> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let updated = ledger
            .remaining
            .checked_add(self.allocation)
            .ok_or(BudgetError::ArithmeticInvariant)?;
        if !updated.fits_within(ledger.total) {
            return Err(BudgetError::ReleaseExceedsPool);
        }
        ledger.remaining = updated;
        Ok(BudgetReleaseReceipt {
            lease_id: self.lease_id(),
            released: self.allocation,
        })
    }

    fn validate_with(&self, verifier: &BudgetVerifier) -> Result<(), BudgetError> {
        self.validate_with_at(verifier, verifier.clock.now())
    }

    fn validate_with_at(
        &self,
        verifier: &BudgetVerifier,
        now: SystemTime,
    ) -> Result<(), BudgetError> {
        self.attestation
            .validate_with(&verifier.inner, now)
            .map_err(BudgetError::Trust)?;
        if self.profile.digest() != verifier.profile_digest {
            return Err(BudgetError::ProfileMismatch);
        }
        let expected = compute_lease_binding(
            self.subject(),
            self.scope(),
            self.action_binding,
            self.allocation,
            self.profile.digest(),
            self.parent_lease_id,
        );
        if self.attestation.binding() != expected {
            return Err(BudgetError::LeaseBindingMismatch);
        }
        Ok(())
    }
}

/// Recoverable affine split rejection.
///
/// The original parent lease is returned by value so callers can correct the
/// request, retain the reservation, or explicitly release it. This type is not
/// `Clone` because it owns affine quantitative authority.
#[derive(Debug)]
pub struct BudgetSplitFailure {
    parent: BudgetLease,
    error: BudgetError,
}

impl BudgetSplitFailure {
    fn new(parent: BudgetLease, error: BudgetError) -> Self {
        Self { parent, error }
    }

    /// Reason the split was rejected before any output lease was minted.
    pub fn error(&self) -> &BudgetError {
        &self.error
    }

    /// Exact original parent lease recovered from the failed transition.
    pub fn parent(&self) -> &BudgetLease {
        &self.parent
    }

    /// Consume the failure and recover the exact original parent lease.
    pub fn into_parent(self) -> BudgetLease {
        self.parent
    }

    /// Consume the failure and return both the original parent and rejection.
    pub fn into_parts(self) -> (BudgetLease, BudgetError) {
        (self.parent, self.error)
    }

    /// Consume the failure and retain only the compatibility error.
    pub fn into_error(self) -> BudgetError {
        self.error
    }
}

impl fmt::Display for BudgetSplitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "recoverable budget split rejected: {}", self.error)
    }
}

impl std::error::Error for BudgetSplitFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Evidence emitted when unused/reserved capacity is explicitly returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetReleaseReceipt {
    lease_id: GrantId,
    released: BudgetQuantities,
}

impl BudgetReleaseReceipt {
    /// Lease whose capacity was released.
    pub fn lease_id(self) -> GrantId {
        self.lease_id
    }

    /// Quantities returned to the root pool.
    pub fn released(self) -> BudgetQuantities {
        self.released
    }
}

/// Budget reservation, delegation, validation, or accounting failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// `ExternalHard` was claimed without naming an external enforcement configuration.
    MissingExternalEnforcementEvidence,
    /// Requested reservation already expired according to the host-owned clock.
    ExpiredReservationRequest,
    /// Requested child delegation expiry was already stale at split validation.
    ExpiredDelegationRequest {
        /// Requested stale child expiry.
        requested: SystemTime,
        /// Single host-owned validation-time snapshot.
        now: SystemTime,
    },
    /// Not enough conserved capacity remained for one dimension.
    InsufficientBudget {
        /// Dimension that failed.
        dimension: BudgetDimension,
        /// Requested quantity.
        requested: u64,
        /// Remaining quantity.
        remaining: u64,
    },
    /// Child logical scope widened beyond the parent lease.
    ScopeWidening {
        /// Parent lease scope.
        parent: Scope,
        /// Requested child scope.
        requested: Scope,
    },
    /// Child expiry was absent or later than a finite parent expiry.
    ExpiryWidening {
        /// Parent expiry.
        parent: Option<SystemTime>,
        /// Requested child expiry.
        requested: Option<SystemTime>,
    },
    /// Lease belongs to another trust domain/epoch or is expired.
    Trust(TrustError),
    /// Lease profile differs from the verifier's immutable pool profile.
    ProfileMismatch,
    /// Lease attestation did not bind the reconstructed lease fields.
    LeaseBindingMismatch,
    /// Lease holder did not match the action principal.
    WrongSubject {
        /// Expected action principal.
        expected: PrincipalId,
        /// Actual lease holder.
        actual: PrincipalId,
    },
    /// Lease scope did not cover the required action scope.
    ScopeMismatch {
        /// Lease scope.
        granted: Scope,
        /// Required action scope.
        required: Scope,
    },
    /// Lease was reserved for another exact action binding.
    ActionBindingMismatch,
    /// Checked accounting arithmetic failed; this indicates an internal invariant violation.
    ArithmeticInvariant,
    /// Release would increase root capacity above its immutable profile total.
    ReleaseExceedsPool,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingExternalEnforcementEvidence => {
                write!(
                    f,
                    "external hard budget claim requires enforcement evidence"
                )
            }
            Self::ExpiredReservationRequest => {
                write!(f, "requested budget lease is already expired")
            }
            Self::ExpiredDelegationRequest { .. } => {
                write!(f, "requested child budget lease is already expired")
            }
            Self::InsufficientBudget { dimension, .. } => {
                write!(f, "insufficient remaining budget for {dimension:?}")
            }
            Self::ScopeWidening { .. } => write!(f, "child budget scope would widen parent scope"),
            Self::ExpiryWidening { .. } => {
                write!(f, "child budget expiry would widen parent expiry")
            }
            Self::Trust(error) => write!(f, "budget trust validation failed: {error}"),
            Self::ProfileMismatch => write!(f, "budget lease belongs to another profile"),
            Self::LeaseBindingMismatch => write!(f, "budget lease binding is inconsistent"),
            Self::WrongSubject { .. } => write!(f, "budget lease belongs to another principal"),
            Self::ScopeMismatch { .. } => write!(f, "budget lease does not cover action scope"),
            Self::ActionBindingMismatch => write!(f, "budget lease targets another action"),
            Self::ArithmeticInvariant => write!(f, "budget arithmetic invariant failed"),
            Self::ReleaseExceedsPool => write!(f, "budget release would exceed pool total"),
        }
    }
}

impl std::error::Error for BudgetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Trust(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct BudgetLedger {
    total: BudgetQuantities,
    remaining: BudgetQuantities,
}

#[derive(Debug)]
struct BudgetClock {
    last: Mutex<SystemTime>,
}

impl BudgetClock {
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

fn compute_profile_digest(
    limits: BudgetQuantities,
    enforcement: BudgetEnforcement,
    external_evidence: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/budget-profile-v1\0");
    for dimension in BudgetDimension::ALL {
        hash_field(&mut hasher, &[dimension as u8]);
        hash_field(&mut hasher, &limits.get(dimension).to_le_bytes());
        hash_field(&mut hasher, &[enforcement.get(dimension).code()]);
    }
    match external_evidence {
        Some(digest) => {
            hash_field(&mut hasher, &[1]);
            hash_field(&mut hasher, &digest);
        }
        None => hash_field(&mut hasher, &[0]),
    }
    *hasher.finalize().as_bytes()
}

fn compute_lease_binding(
    subject: PrincipalId,
    scope: &Scope,
    action_binding: [u8; 32],
    allocation: BudgetQuantities,
    profile_digest: [u8; 32],
    parent_lease_id: Option<GrantId>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/budget-lease-v1\0");
    hash_field(&mut hasher, subject.as_uuid().as_bytes());
    hash_field(&mut hasher, scope.namespace().as_bytes());
    for segment in scope.segments() {
        hash_field(&mut hasher, segment.as_bytes());
    }
    hash_field(&mut hasher, &action_binding);
    for dimension in BudgetDimension::ALL {
        hash_field(&mut hasher, &[dimension as u8]);
        hash_field(&mut hasher, &allocation.get(dimension).to_le_bytes());
    }
    hash_field(&mut hasher, &profile_digest);
    match parent_lease_id {
        Some(parent) => {
            hash_field(&mut hasher, &[1]);
            hash_field(&mut hasher, parent.as_uuid().as_bytes());
        }
        None => hash_field(&mut hasher, &[0]),
    }
    *hasher.finalize().as_bytes()
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    fn scope(parts: &[&str]) -> Scope {
        Scope::new("agent", parts.iter().copied()).unwrap()
    }

    fn profile(compute_units: u64, subprocesses: u64) -> BudgetProfile {
        let limits = BudgetQuantities::zero()
            .with(BudgetDimension::ComputeUnits, compute_units)
            .with(BudgetDimension::Subprocesses, subprocesses);
        let enforcement = BudgetEnforcement::soft()
            .with(BudgetDimension::ComputeUnits, EnforcementClass::CoreMetered)
            .with(BudgetDimension::Subprocesses, EnforcementClass::CoreMetered);
        BudgetProfile::new(limits, enforcement, None).unwrap()
    }

    #[test]
    fn external_hard_claim_requires_evidence_digest() {
        let limits = BudgetQuantities::zero().with(BudgetDimension::MemoryBytes, 1024);
        let enforcement = BudgetEnforcement::soft()
            .with(BudgetDimension::MemoryBytes, EnforcementClass::ExternalHard);
        assert!(matches!(
            BudgetProfile::new(limits, enforcement, None),
            Err(BudgetError::MissingExternalEnforcementEvidence)
        ));
    }

    #[test]
    fn root_pool_conserves_capacity_across_reservations() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
        let actor = PrincipalId::new();
        let first = BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 7);
        let second = BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 4);
        domain
            .reserve(actor, scope(&["root"]), [1; 32], first, None)
            .unwrap();
        assert!(matches!(
            domain.reserve(actor, scope(&["root"]), [2; 32], second, None),
            Err(BudgetError::InsufficientBudget {
                dimension: BudgetDimension::ComputeUnits,
                ..
            })
        ));
    }

    #[test]
    fn concurrent_last_unit_can_only_be_reserved_once() {
        let domain = Arc::new(BudgetAuthorityDomain::new(
            PrincipalId::new(),
            profile(1, 0),
        ));
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for tag in [1_u8, 2_u8] {
            let domain = Arc::clone(&domain);
            let barrier = Arc::clone(&barrier);
            threads.push(thread::spawn(move || {
                barrier.wait();
                domain.reserve(
                    PrincipalId::new(),
                    scope(&["root"]),
                    [tag; 32],
                    BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 1),
                    None,
                )
            }));
        }
        barrier.wait();
        let successes = threads
            .into_iter()
            .filter(|thread| thread.join().unwrap().is_ok())
            .count();
        assert_eq!(successes, 1);
        assert_eq!(domain.remaining().get(BudgetDimension::ComputeUnits), 0);
    }

    #[test]
    fn lease_is_exact_action_and_subject_bound() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(4, 1));
        let verifier = domain.verifier();
        let actor = PrincipalId::new();
        let lease = domain
            .reserve(
                actor,
                scope(&["root"]),
                [4; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
                None,
            )
            .unwrap();
        assert!(
            lease
                .validate_for(&verifier, actor, &scope(&["root"]), [4; 32])
                .is_ok()
        );
        assert!(matches!(
            lease.validate_for(&verifier, actor, &scope(&["root"]), [5; 32]),
            Err(BudgetError::ActionBindingMismatch)
        ));
        assert!(matches!(
            lease.validate_for(&verifier, PrincipalId::new(), &scope(&["root"]), [4; 32]),
            Err(BudgetError::WrongSubject { .. })
        ));
    }

    #[test]
    fn unrelated_domain_and_revoked_epoch_are_rejected() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(4, 1));
        let other = BudgetAuthorityDomain::new(PrincipalId::new(), profile(4, 1));
        let actor = PrincipalId::new();
        let lease = domain
            .reserve(
                actor,
                scope(&["root"]),
                [6; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 1),
                None,
            )
            .unwrap();
        assert!(
            lease
                .validate_for(&other.verifier(), actor, &scope(&["root"]), [6; 32])
                .is_err()
        );
        domain.revoke_all().unwrap();
        assert!(
            lease
                .validate_for(&domain.verifier(), actor, &scope(&["root"]), [6; 32])
                .is_err()
        );
    }

    #[test]
    fn affine_split_conserves_allocation_and_lineage() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 3));
        let parent_actor = PrincipalId::new();
        let child_actor = PrincipalId::new();
        let parent = domain
            .reserve(
                parent_actor,
                scope(&["root"]),
                [7; 32],
                BudgetQuantities::zero()
                    .with(BudgetDimension::ComputeUnits, 8)
                    .with(BudgetDimension::Subprocesses, 2),
                Some(SystemTime::now() + Duration::from_secs(60)),
            )
            .unwrap();
        let source_id = parent.lease_id();
        let (remainder, child) = domain
            .split_recoverable(
                parent,
                child_actor,
                scope(&["root", "child"]),
                [8; 32],
                BudgetQuantities::zero()
                    .with(BudgetDimension::ComputeUnits, 3)
                    .with(BudgetDimension::Subprocesses, 1),
                Some(SystemTime::now() + Duration::from_secs(30)),
            )
            .unwrap();
        assert_eq!(remainder.parent_lease_id(), Some(source_id));
        assert_eq!(child.parent_lease_id(), Some(source_id));
        assert_eq!(remainder.allocation().get(BudgetDimension::ComputeUnits), 5);
        assert_eq!(child.allocation().get(BudgetDimension::ComputeUnits), 3);
        assert_eq!(remainder.allocation().get(BudgetDimension::Subprocesses), 1);
        assert_eq!(child.allocation().get(BudgetDimension::Subprocesses), 1);
    }

    #[test]
    fn recoverable_split_returns_exact_parent_on_scope_rejection() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
        let actor = PrincipalId::new();
        let allocation = BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 6);
        let parent = domain
            .reserve(
                actor,
                scope(&["root"]),
                [10; 32],
                allocation,
                Some(SystemTime::now() + Duration::from_secs(60)),
            )
            .unwrap();
        let id = parent.lease_id();
        let epoch = parent.epoch();
        let budget_domain = parent.domain_id();
        let remaining_before = domain.remaining();

        let failure = domain
            .split_recoverable(
                parent,
                PrincipalId::new(),
                scope(&["other"]),
                [11; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
                Some(SystemTime::now() + Duration::from_secs(30)),
            )
            .unwrap_err();
        assert!(matches!(failure.error(), BudgetError::ScopeWidening { .. }));
        assert_eq!(failure.parent().lease_id(), id);
        assert_eq!(failure.parent().allocation(), allocation);
        assert_eq!(failure.parent().epoch(), epoch);
        assert_eq!(failure.parent().domain_id(), budget_domain);
        assert_eq!(domain.remaining(), remaining_before);

        failure.into_parent().release().unwrap();
        assert_eq!(domain.remaining(), domain.profile().limits());
    }

    #[test]
    fn recoverable_split_rejects_stale_child_without_consuming_parent() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
        let actor = PrincipalId::new();
        let parent = domain
            .reserve(
                actor,
                scope(&["root"]),
                [12; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 6),
                Some(SystemTime::now() + Duration::from_secs(60)),
            )
            .unwrap();
        let id = parent.lease_id();
        let remaining_before = domain.remaining();

        let failure = domain
            .split_recoverable(
                parent,
                PrincipalId::new(),
                scope(&["root", "child"]),
                [13; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
                Some(SystemTime::UNIX_EPOCH),
            )
            .unwrap_err();
        assert!(matches!(
            failure.error(),
            BudgetError::ExpiredDelegationRequest { .. }
        ));
        assert_eq!(failure.parent().lease_id(), id);
        assert_eq!(domain.remaining(), remaining_before);
        failure.into_parent().release().unwrap();
    }

    #[test]
    fn recoverable_split_returns_parent_when_child_allocation_exceeds_source() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
        let actor = PrincipalId::new();
        let parent = domain
            .reserve(
                actor,
                scope(&["root"]),
                [14; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 4),
                Some(SystemTime::now() + Duration::from_secs(60)),
            )
            .unwrap();
        let id = parent.lease_id();
        let remaining_before = domain.remaining();

        let failure = domain
            .split_recoverable(
                parent,
                PrincipalId::new(),
                scope(&["root", "child"]),
                [15; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 5),
                Some(SystemTime::now() + Duration::from_secs(30)),
            )
            .unwrap_err();
        assert!(matches!(
            failure.error(),
            BudgetError::InsufficientBudget { .. }
        ));
        assert_eq!(failure.parent().lease_id(), id);
        assert_eq!(domain.remaining(), remaining_before);
        failure.into_parent().release().unwrap();
    }

    #[test]
    fn release_returns_reserved_capacity_exactly_once() {
        let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(5, 1));
        let actor = PrincipalId::new();
        let lease = domain
            .reserve(
                actor,
                scope(&["root"]),
                [9; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 3),
                None,
            )
            .unwrap();
        assert_eq!(domain.remaining().get(BudgetDimension::ComputeUnits), 2);
        let receipt = lease.release().unwrap();
        assert_eq!(receipt.released().get(BudgetDimension::ComputeUnits), 3);
        assert_eq!(domain.remaining().get(BudgetDimension::ComputeUnits), 5);
    }
}
