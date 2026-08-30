// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strongest state-changing host facade with exact execution preflight and
//! evidence-bearing effect-attempt outcomes.
//!
//! The lower resource/policy/budget stack already validates concrete resource,
//! policy, and quantitative-budget authority around adapter entry. A later
//! adversarial review found two remaining semantics that should not be left to a
//! generic `Result`:
//!
//! 1. exact execution-domain epoch/expiry must be checked *before* entering the
//!    real adapter, not only by post-adapter execution recording;
//! 2. an effectful adapter reporting failure does not prove that no effect
//!    occurred. Failure/unknown outcomes must remain in an observable action
//!    lineage rather than disappearing through `Err(E)`.
//!
//! [`EffectGuardedRuntime`] composes [`crate::BudgetGuardedRuntime`] with the
//! strict temporal-policy grant from [`crate::temporal_policy`]. The authorized
//! wrapper retains the host-selected execution verifier plus exact derived
//! execution lifetime and preflights both immediately before delegating into the
//! existing retained-handle path.
//!
//! Adapters on this surface return [`EffectAttemptOutcome`] *as data*. The
//! closure is infallible from the assurance wrapper's perspective: reported
//! failure, uncertainty, and proven transactional no-effect are distinct outcome
//! classes whose evidence digest is committed into the existing execution /
//! observation lineage.
//!
//! This tranche does **not** claim atomic concurrent revocation/effect ordering.
//! A revocation may still race after this facade's preflight but before/while the
//! lower adapter enters; issue #134 defines the required linearization permit.
//! If such a race causes the lower post-adapter execution check to fail after the
//! adapter returned, this facade preserves [`EffectAttemptEvidence`] in a
//! [`EffectAttemptFailure::LineageFailedAfterAttempt`] result rather than losing
//! all evidence that the boundary was entered.
//!
//! Likewise, lower policy/resource/budget failures that happen before the user
//! adapter are still subject to issue #140's recoverable-authority work because
//! the current lower wrappers consume themselves on error. Panic/cancellation
//! after adapter entry also requires an isolated adapter/task/process boundary if
//! a deployment wants durable attempt journaling across abnormal termination.

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::budget::{
    BudgetError, BudgetLease, BudgetQuantities, BudgetReleaseReceipt,
};
use crate::budget_guard::{
    BudgetAdapterError, BudgetGuardedAction, BudgetGuardedAuthorizeError, BudgetGuardedRuntime,
    BudgetedEvidenceReceipt,
};
use crate::capability::{CapabilityKind, GrantId, PrincipalId, Scope};
use crate::host::ResolutionError;
use crate::policy_guard::PolicyGuardedExecutionError;
use crate::resolution::ResolutionGrant;
use crate::resource::{ResolvedResource, ResourceError};
use crate::temporal_policy::{TemporalDerivationEvidence, TemporalPolicyGrant};
use crate::trusted::{
    AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError, TrustedBoundOneShotCapability,
};
use std::convert::Infallible;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Explicit outcome reported by an adapter after the effect boundary was entered.
///
/// The classification is evidence, not objective truth. `ProvenNoEffect` should
/// be used only by adapters with a genuine transactional/rollback/no-commit
/// guarantee. Ordinary application errors should normally be
/// `ReportedFailure` or `OutcomeUnknown` and later be independently observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectAttemptOutcome {
    /// Adapter reports that the intended operation completed.
    Succeeded {
        /// Digest of adapter-produced outcome evidence.
        evidence_digest: [u8; 32],
    },
    /// Adapter reports failure, but the effect boundary was entered and partial
    /// side effects may have occurred.
    ReportedFailure {
        /// Digest of adapter error/outcome evidence.
        evidence_digest: [u8; 32],
    },
    /// Adapter cannot determine the externally visible outcome after entry.
    OutcomeUnknown {
        /// Digest of the evidence explaining the uncertainty.
        evidence_digest: [u8; 32],
    },
    /// Adapter proves that its transaction did not commit any external effect.
    ///
    /// This is a strong adapter-specific claim and must not be inferred merely
    /// from a Rust error return.
    ProvenNoEffect {
        /// Digest of rollback/no-commit evidence.
        evidence_digest: [u8; 32],
    },
}

impl EffectAttemptOutcome {
    /// Adapter evidence digest carried by this outcome.
    pub fn evidence_digest(self) -> [u8; 32] {
        match self {
            Self::Succeeded { evidence_digest }
            | Self::ReportedFailure { evidence_digest }
            | Self::OutcomeUnknown { evidence_digest }
            | Self::ProvenNoEffect { evidence_digest } => evidence_digest,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Succeeded { .. } => 0,
            Self::ReportedFailure { .. } => 1,
            Self::OutcomeUnknown { .. } => 2,
            Self::ProvenNoEffect { .. } => 3,
        }
    }
}

/// Host-owned evidence that the real adapter boundary was entered and returned
/// an explicit outcome classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectAttemptEvidence {
    action_id: ActionId,
    actor: PrincipalId,
    action_fingerprint: [u8; 32],
    action_scope: Scope,
    action_binding: [u8; 32],
    budget_lease_id: GrantId,
    budget_allocation: BudgetQuantities,
    temporal_derivation: TemporalDerivationEvidence,
    preflight_at: SystemTime,
    adapter_entered_at: SystemTime,
    adapter_returned_at: SystemTime,
    outcome: EffectAttemptOutcome,
    digest: [u8; 32],
}

impl EffectAttemptEvidence {
    /// Stable exact action identity.
    pub fn action_id(&self) -> ActionId {
        self.action_id
    }

    /// Principal on whose behalf the effect was attempted.
    pub fn actor(&self) -> PrincipalId {
        self.actor
    }

    /// Exact action fingerprint, including concrete resource identity through
    /// the lower resource-bound payload.
    pub fn action_fingerprint(&self) -> [u8; 32] {
        self.action_fingerprint
    }

    /// Logical action scope.
    pub fn action_scope(&self) -> &Scope {
        &self.action_scope
    }

    /// Exact action authorization binding.
    pub fn action_binding(&self) -> [u8; 32] {
        self.action_binding
    }

    /// Quantitative lease held when the adapter was entered.
    pub fn budget_lease_id(&self) -> GrantId {
        self.budget_lease_id
    }

    /// Quantitative reservation in force for the attempt.
    pub fn budget_allocation(&self) -> BudgetQuantities {
        self.budget_allocation
    }

    /// Policy-to-execution temporal derivation in force for the attempt.
    pub fn temporal_derivation(&self) -> &TemporalDerivationEvidence {
        &self.temporal_derivation
    }

    /// Host-owned time when exact execution preflight completed.
    pub fn preflight_at(&self) -> SystemTime {
        self.preflight_at
    }

    /// Host-owned time immediately before user adapter code was entered.
    pub fn adapter_entered_at(&self) -> SystemTime {
        self.adapter_entered_at
    }

    /// Host-owned time immediately after user adapter code returned.
    pub fn adapter_returned_at(&self) -> SystemTime {
        self.adapter_returned_at
    }

    /// Adapter-reported outcome classification.
    pub fn outcome(&self) -> EffectAttemptOutcome {
        self.outcome
    }

    /// Domain-separated digest committed as the lower execution output digest.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Final evidence from the strongest current facade: existing
/// policy/resource/budget/execution/observation/resolution evidence plus temporal
/// derivation and explicit effect-attempt evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectAssuredEvidenceReceipt {
    budgeted: BudgetedEvidenceReceipt,
    temporal_derivation: TemporalDerivationEvidence,
    effect_attempt: EffectAttemptEvidence,
}

impl EffectAssuredEvidenceReceipt {
    /// Existing policy/resource/budget/execution/observation/resolution evidence.
    pub fn budgeted_receipt(&self) -> &BudgetedEvidenceReceipt {
        &self.budgeted
    }

    /// Policy-to-execution lifetime derivation consumed by the action.
    pub fn temporal_derivation(&self) -> &TemporalDerivationEvidence {
        &self.temporal_derivation
    }

    /// Evidence that the concrete adapter boundary was entered and returned.
    pub fn effect_attempt(&self) -> &EffectAttemptEvidence {
        &self.effect_attempt
    }
}

/// Host wrapper that adds exact execution preflight and explicit attempt semantics
/// to the current strongest policy/resource/budget runtime.
#[derive(Debug, Clone)]
pub struct EffectGuardedRuntime {
    inner: BudgetGuardedRuntime,
    execution_verifier: AuthorityVerifier,
    clock: Arc<EffectClock>,
}

impl EffectGuardedRuntime {
    /// Construct the strongest current host facade.
    ///
    /// `execution_verifier` must be the verifier corresponding to the execution
    /// authority that produced [`TemporalPolicyGrant`] values. A mismatch fails
    /// closed when the grant is consumed.
    pub fn new(inner: BudgetGuardedRuntime, execution_verifier: AuthorityVerifier) -> Self {
        Self {
            inner,
            execution_verifier,
            clock: Arc::new(EffectClock::new()),
        }
    }

    /// Exact execution authority domain pinned by this facade.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_verifier.domain_id()
    }

    /// Admit a pre-resolved concrete resource into the effect-attempt lifecycle.
    pub fn admit_resolved<K: CapabilityKind, H>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        resource: ResolvedResource<H>,
        canonical_payload: &[u8],
    ) -> Result<EffectGuardedAction<K, Proposed, H>, ResourceError> {
        let inner = self
            .inner
            .admit_resolved::<K, H>(actor, kind, resource, canonical_payload)?;
        Ok(EffectGuardedAction {
            inner,
            execution_verifier: self.execution_verifier.clone(),
            clock: Arc::clone(&self.clock),
            action_binding: None,
            temporal_derivation: None,
            effect_attempt: None,
            final_receipt: None,
        })
    }
}

/// Strong state-changing action lifecycle that requires temporal policy authority
/// and explicit effect-attempt outcomes.
pub struct EffectGuardedAction<K: CapabilityKind, S, H> {
    inner: BudgetGuardedAction<K, S, H>,
    execution_verifier: AuthorityVerifier,
    clock: Arc<EffectClock>,
    action_binding: Option<[u8; 32]>,
    temporal_derivation: Option<TemporalDerivationEvidence>,
    effect_attempt: Option<EffectAttemptEvidence>,
    final_receipt: Option<EffectAssuredEvidenceReceipt>,
}

impl<K: CapabilityKind, S, H> EffectGuardedAction<K, S, H> {
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

    /// Execution authority domain pinned by the host facade.
    pub fn execution_domain(&self) -> AuthorityDomainId {
        self.execution_verifier.domain_id()
    }

    /// Temporal derivation evidence, once exact authority has been consumed.
    pub fn temporal_derivation(&self) -> Option<&TemporalDerivationEvidence> {
        self.temporal_derivation.as_ref()
    }

    /// Effect-attempt evidence, once the adapter boundary has been entered and returned.
    pub fn effect_attempt(&self) -> Option<&EffectAttemptEvidence> {
        self.effect_attempt.as_ref()
    }
}

impl<K: CapabilityKind, H> EffectGuardedAction<K, Proposed, H> {
    /// Attach explicit risk and establish the exact action binding.
    pub fn assess(self, risk: ActionRisk) -> EffectGuardedAction<K, RiskAssessed, H> {
        let inner = self.inner.assess(risk);
        let action_binding = inner.authorization_binding();
        EffectGuardedAction {
            inner,
            execution_verifier: self.execution_verifier,
            clock: self.clock,
            action_binding: Some(action_binding),
            temporal_derivation: None,
            effect_attempt: None,
            final_receipt: None,
        }
    }
}

impl<K: CapabilityKind, H> EffectGuardedAction<K, RiskAssessed, H> {
    /// Risk classification evaluated by trusted policy.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact action authorization binding targeted by policy and budget authority.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.action_binding
            .expect("RiskAssessed effect action always carries action binding")
    }

    /// Consume temporally bounded policy authority plus quantitative authority.
    pub fn authorize(
        self,
        temporal_grant: TemporalPolicyGrant<K>,
        budget_lease: BudgetLease,
    ) -> Result<EffectGuardedAction<K, Authorized, H>, EffectGuardedAuthorizeError> {
        let now = self.clock.now();
        validate_temporal_grant(&temporal_grant, &self.execution_verifier, now)
            .map_err(EffectGuardedAuthorizeError::Execution)?;

        let temporal_derivation = temporal_grant.temporal_evidence().clone();
        let (policy_grant, retained_temporal) = temporal_grant.into_parts();
        debug_assert_eq!(temporal_derivation, retained_temporal);
        let inner = self
            .inner
            .authorize(policy_grant, budget_lease)
            .map_err(EffectGuardedAuthorizeError::Inner)?;

        Ok(EffectGuardedAction {
            inner,
            execution_verifier: self.execution_verifier,
            clock: self.clock,
            action_binding: self.action_binding,
            temporal_derivation: Some(temporal_derivation),
            effect_attempt: None,
            final_receipt: None,
        })
    }
}

impl<K: CapabilityKind, H> EffectGuardedAction<K, Authorized, H> {
    /// Quantitative allocation reserved for this exact action.
    pub fn budget_allocation(&self) -> BudgetQuantities {
        self.inner.budget_allocation()
    }

    /// Quantitative lease identity reserved for this exact action.
    pub fn budget_lease_id(&self) -> GrantId {
        self.inner.budget_lease_id()
    }

    /// Preflight exact execution domain/epoch/expiry, then enter the retained
    /// resource adapter and record its outcome as evidence rather than control
    /// flow.
    ///
    /// The adapter callback cannot return an assurance-level `Err`: it must
    /// classify what happened using [`EffectAttemptOutcome`]. This means
    /// application-level failure and uncertainty still produce an `Executed`
    /// lineage that can be independently observed and resolved.
    pub fn execute_attempt_with<F>(
        self,
        attempt: F,
    ) -> Result<EffectGuardedAction<K, Executed, H>, EffectAttemptFailure<K, H>>
    where
        F: FnOnce(&mut H) -> EffectAttemptOutcome,
    {
        let temporal = self
            .temporal_derivation
            .as_ref()
            .expect("Authorized effect action always carries temporal derivation");
        let preflight_at = self.clock.now();
        if let Err(error) =
            validate_temporal_evidence(temporal, &self.execution_verifier, preflight_at)
        {
            return Err(EffectAttemptFailure::Preflight {
                action: self,
                error,
            });
        }

        let action_id = self.inner.id();
        let actor = self.inner.actor();
        let action_fingerprint = self.inner.descriptor().fingerprint();
        let action_scope = self.inner.descriptor().scope().clone();
        let action_binding = self
            .action_binding
            .expect("Authorized effect action always carries action binding");
        let budget_lease_id = self.inner.budget_lease_id();
        let budget_allocation = self.inner.budget_allocation();
        let temporal_derivation = temporal.clone();

        let EffectGuardedAction {
            inner,
            execution_verifier,
            clock,
            action_binding: _,
            temporal_derivation: _,
            effect_attempt: _,
            final_receipt: _,
        } = self;

        let attempt_clock = Arc::clone(&clock);
        let mut attempt_evidence = None;
        let result = inner.execute_with(|handle| -> Result<[u8; 32], Infallible> {
            let adapter_entered_at = attempt_clock.now();
            let outcome = attempt(handle);
            let adapter_returned_at = attempt_clock.now();
            let digest = compute_effect_attempt_digest(
                action_id,
                actor,
                action_fingerprint,
                &action_scope,
                action_binding,
                budget_lease_id,
                &temporal_derivation,
                preflight_at,
                adapter_entered_at,
                adapter_returned_at,
                outcome,
            );
            attempt_evidence = Some(EffectAttemptEvidence {
                action_id,
                actor,
                action_fingerprint,
                action_scope: action_scope.clone(),
                action_binding,
                budget_lease_id,
                budget_allocation,
                temporal_derivation: temporal_derivation.clone(),
                preflight_at,
                adapter_entered_at,
                adapter_returned_at,
                outcome,
                digest,
            });
            Ok(digest)
        });

        match result {
            Ok(inner) => {
                let effect_attempt = attempt_evidence
                    .expect("successful lower execution implies adapter closure was entered");
                Ok(EffectGuardedAction {
                    inner,
                    execution_verifier,
                    clock,
                    action_binding: Some(action_binding),
                    temporal_derivation: Some(temporal_derivation),
                    effect_attempt: Some(effect_attempt),
                    final_receipt: None,
                })
            }
            Err(error) => match attempt_evidence {
                Some(evidence) => Err(EffectAttemptFailure::LineageFailedAfterAttempt {
                    evidence,
                    error,
                }),
                None => Err(EffectAttemptFailure::RejectedBeforeAttempt { error }),
            },
        }
    }
}

impl<K: CapabilityKind, H> EffectGuardedAction<K, Executed, H> {
    /// Exact observation binding, which commits to the effect-attempt digest as
    /// the lower execution output digest.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Attach independently authorized external observation.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<crate::Observe>,
        observation: Observation,
    ) -> Result<EffectGuardedAction<K, Observed, H>, TrustError> {
        let inner = self.inner.observe(observer, observation)?;
        Ok(EffectGuardedAction {
            inner,
            execution_verifier: self.execution_verifier,
            clock: self.clock,
            action_binding: self.action_binding,
            temporal_derivation: self.temporal_derivation,
            effect_attempt: self.effect_attempt,
            final_receipt: None,
        })
    }
}

impl<K: CapabilityKind, H> EffectGuardedAction<K, Observed, H> {
    /// Exact final-resolution binding for this independently observed lineage.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        self.inner.resolution_binding(decision)
    }

    /// Consume exact resolution authority and emit the strongest composed receipt.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<
        (
            EffectGuardedAction<K, Resolved, H>,
            EffectAssuredEvidenceReceipt,
        ),
        ResolutionError,
    > {
        let temporal_derivation = self
            .temporal_derivation
            .as_ref()
            .expect("Observed effect action always carries temporal derivation")
            .clone();
        let effect_attempt = self
            .effect_attempt
            .as_ref()
            .expect("Observed effect action always carries attempt evidence")
            .clone();
        let (inner, budgeted) = self.inner.resolve(grant, decision)?;
        let receipt = EffectAssuredEvidenceReceipt {
            budgeted,
            temporal_derivation,
            effect_attempt,
        };
        Ok((
            EffectGuardedAction {
                inner,
                execution_verifier: self.execution_verifier,
                clock: self.clock,
                action_binding: self.action_binding,
                temporal_derivation: self.temporal_derivation,
                effect_attempt: self.effect_attempt,
                final_receipt: Some(receipt.clone()),
            },
            receipt,
        ))
    }
}

impl<K: CapabilityKind, H> EffectGuardedAction<K, Resolved, H> {
    /// Strongest composed final evidence retained by the resolved action.
    pub fn assurance_receipt(&self) -> &EffectAssuredEvidenceReceipt {
        self.final_receipt
            .as_ref()
            .expect("Resolved effect action always carries final evidence")
    }

    /// Return reserved quantitative capacity after final evidence has been retained.
    pub fn release_budget(self) -> Result<BudgetReleaseReceipt, BudgetError> {
        self.inner.release_budget()
    }
}

/// Exact execution preflight failure before the user adapter is intentionally entered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionPreflightError {
    /// Temporal evidence disagreed with the underlying policy/execution grant lineage.
    TemporalLineageMismatch,
    /// Temporal evidence itself records a widened finite lifetime.
    TemporalBoundViolation,
    /// Exact execution authority came from another domain than the host-pinned verifier.
    WrongExecutionDomain {
        /// Host-pinned execution domain.
        expected: AuthorityDomainId,
        /// Execution domain recorded by temporal authority.
        actual: AuthorityDomainId,
    },
    /// Exact execution authority belongs to an older/newer epoch than the host's current epoch.
    RevokedExecutionEpoch {
        /// Execution authority domain.
        domain: AuthorityDomainId,
        /// Epoch carried by the derived grant.
        lineage_epoch: AuthorityEpoch,
        /// Current host-required epoch.
        current_epoch: AuthorityEpoch,
    },
    /// Exact execution grant expired before the attempted side-effect boundary.
    ExpiredExecutionGrant {
        /// Finite execution-grant expiry.
        expires_at: SystemTime,
        /// Host-owned validation time.
        now: SystemTime,
    },
}

impl fmt::Display for ExecutionPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TemporalLineageMismatch => {
                write!(f, "temporal evidence does not match policy/execution lineage")
            }
            Self::TemporalBoundViolation => {
                write!(f, "temporal evidence widens a finite policy lifetime")
            }
            Self::WrongExecutionDomain { .. } => {
                write!(f, "execution authority belongs to another host domain")
            }
            Self::RevokedExecutionEpoch { .. } => {
                write!(f, "execution authority epoch is no longer current")
            }
            Self::ExpiredExecutionGrant { .. } => {
                write!(f, "execution authority expired before adapter entry")
            }
        }
    }
}

impl std::error::Error for ExecutionPreflightError {}

/// Failure while consuming strict temporal policy authority plus a budget lease.
#[derive(Debug)]
pub enum EffectGuardedAuthorizeError {
    /// Exact execution temporal/domain preflight failed before lower authorization.
    Execution(ExecutionPreflightError),
    /// Existing policy/resource/budget authorization failed.
    Inner(BudgetGuardedAuthorizeError),
}

impl fmt::Display for EffectGuardedAuthorizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Execution(error) => write!(f, "execution preflight rejected authority: {error}"),
            Self::Inner(error) => write!(f, "guarded authorization failed: {error}"),
        }
    }
}

impl std::error::Error for EffectGuardedAuthorizeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            Self::Inner(error) => Some(error),
        }
    }
}

/// Existing lower execution error type when the adapter callback itself is
/// assurance-level infallible and reports outcomes through [`EffectAttemptOutcome`].
pub type EffectInnerExecutionError =
    PolicyGuardedExecutionError<BudgetAdapterError<Infallible>>;

/// Failure while attempting an effect through the strongest current facade.
///
/// This enum deliberately distinguishes whether the user adapter was entered.
/// Only the `Preflight` variant can currently return the original authorized
/// wrapper intact. `RejectedBeforeAttempt` is still tracked by issue #140 because
/// lower wrappers currently consume themselves on pre-adapter error.
pub enum EffectAttemptFailure<K: CapabilityKind, H> {
    /// This facade rejected exact execution authority before delegating into the
    /// lower effect path. The original authorized action is recoverable.
    Preflight {
        /// Original authorized action; user adapter was not entered.
        action: EffectGuardedAction<K, Authorized, H>,
        /// Exact preflight reason.
        error: ExecutionPreflightError,
    },
    /// A lower policy/resource/budget check rejected before user adapter entry.
    ///
    /// The current lower API consumes the wrapper here; issue #140 tracks
    /// transactional recovery of the conserved budget/action state.
    RejectedBeforeAttempt {
        /// Existing lower guarded error.
        error: EffectInnerExecutionError,
    },
    /// User adapter returned explicit attempt evidence, but a lower post-adapter
    /// lineage transition failed (for example a concurrent execution epoch
    /// rotation observed by the existing post-adapter execution check).
    LineageFailedAfterAttempt {
        /// Preserved evidence that the adapter was entered and returned.
        evidence: EffectAttemptEvidence,
        /// Existing lower error that prevented a normal `Executed` action lineage.
        error: EffectInnerExecutionError,
    },
}

impl<K: CapabilityKind, H> EffectAttemptFailure<K, H> {
    /// Whether the user adapter boundary was entered according to this failure.
    pub fn adapter_was_entered(&self) -> bool {
        matches!(self, Self::LineageFailedAfterAttempt { .. })
    }

    /// Preserved adapter-attempt evidence when the boundary was entered.
    pub fn attempt_evidence(&self) -> Option<&EffectAttemptEvidence> {
        match self {
            Self::LineageFailedAfterAttempt { evidence, .. } => Some(evidence),
            Self::Preflight { .. } | Self::RejectedBeforeAttempt { .. } => None,
        }
    }

    /// Exact preflight error when this facade rejected before lower execution.
    pub fn preflight_error(&self) -> Option<&ExecutionPreflightError> {
        match self {
            Self::Preflight { error, .. } => Some(error),
            Self::RejectedBeforeAttempt { .. } | Self::LineageFailedAfterAttempt { .. } => None,
        }
    }
}

fn validate_temporal_grant<K: CapabilityKind>(
    grant: &TemporalPolicyGrant<K>,
    verifier: &AuthorityVerifier,
    now: SystemTime,
) -> Result<(), ExecutionPreflightError> {
    let temporal = grant.temporal_evidence();
    let policy = grant.policy_grant().evidence();
    if temporal.policy_domain() != policy.policy_domain()
        || temporal.policy_epoch() != policy.policy_epoch()
        || temporal.policy_attestation_grant_id() != policy.policy_attestation_grant_id()
        || temporal.execution_domain() != policy.execution_domain()
        || temporal.execution_epoch() != policy.execution_epoch()
        || temporal.execution_grant_id() != policy.execution_grant_id()
    {
        return Err(ExecutionPreflightError::TemporalLineageMismatch);
    }
    validate_temporal_evidence(temporal, verifier, now)
}

fn validate_temporal_evidence(
    temporal: &TemporalDerivationEvidence,
    verifier: &AuthorityVerifier,
    now: SystemTime,
) -> Result<(), ExecutionPreflightError> {
    if !temporal.preserves_finite_parent_bound() {
        return Err(ExecutionPreflightError::TemporalBoundViolation);
    }
    if temporal.execution_domain() != verifier.domain_id() {
        return Err(ExecutionPreflightError::WrongExecutionDomain {
            expected: verifier.domain_id(),
            actual: temporal.execution_domain(),
        });
    }
    let current_epoch = verifier.current_epoch();
    if temporal.execution_epoch() != current_epoch {
        return Err(ExecutionPreflightError::RevokedExecutionEpoch {
            domain: verifier.domain_id(),
            lineage_epoch: temporal.execution_epoch(),
            current_epoch,
        });
    }
    if let Some(expires_at) = temporal.execution_expires_at() {
        if expires_at < now {
            return Err(ExecutionPreflightError::ExpiredExecutionGrant { expires_at, now });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_effect_attempt_digest(
    action_id: ActionId,
    actor: PrincipalId,
    action_fingerprint: [u8; 32],
    scope: &Scope,
    action_binding: [u8; 32],
    budget_lease_id: GrantId,
    temporal: &TemporalDerivationEvidence,
    preflight_at: SystemTime,
    adapter_entered_at: SystemTime,
    adapter_returned_at: SystemTime,
    outcome: EffectAttemptOutcome,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/effect-attempt-v1\0");
    hash_field(&mut hasher, action_id.as_uuid().as_bytes());
    hash_field(&mut hasher, actor.as_uuid().as_bytes());
    hash_field(&mut hasher, &action_fingerprint);
    hash_field(&mut hasher, scope.namespace().as_bytes());
    for segment in scope.segments() {
        hash_field(&mut hasher, segment.as_bytes());
    }
    hash_field(&mut hasher, &action_binding);
    hash_field(&mut hasher, budget_lease_id.as_uuid().as_bytes());
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
    hash_system_time(&mut hasher, preflight_at);
    hash_system_time(&mut hasher, adapter_entered_at);
    hash_system_time(&mut hasher, adapter_returned_at);
    hash_field(&mut hasher, &[outcome.code()]);
    hash_field(&mut hasher, &outcome.evidence_digest());
    *hasher.finalize().as_bytes()
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

#[derive(Debug)]
struct EffectClock {
    last: Mutex<SystemTime>,
}

impl EffectClock {
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
        AdapterSchema, ApprovalEvidence, AuthorityDomain, BudgetAuthorityDomain, BudgetDimension,
        BudgetEnforcement, BudgetProfile, BudgetQuantities, EnforcementClass, Observe,
        ObservedOutcome, PolicyDescriptor, PolicyGuardedRuntime, PolicyMode, PolicyResourceRuntime,
        ResolutionAuthorityDomain, ResourceIdentity, ResourceResolverDomain, ResourceRuntime,
        TemporalPolicyEvaluatorDomain, TemporalPolicyExecutionDomain, TemporalPolicyRules,
        TrustedRuntime, Write,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    fn resource_identity() -> ResourceIdentity {
        ResourceIdentity::new(
            scope(),
            "worktree-file",
            [1; 32],
            [2; 32],
            AdapterSchema::new("effect-guard-test", 1).unwrap(),
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
        runtime: EffectGuardedRuntime,
    }

    fn harness() -> Harness {
        let rules = TemporalPolicyRules::strict();
        let evaluator = TemporalPolicyEvaluatorDomain::new(
            PrincipalId::new(),
            PolicyDescriptor::new("effect-guard", 1, [3; 32], 1).unwrap(),
            rules,
        );
        let execution = TemporalPolicyExecutionDomain::new(
            PrincipalId::new(),
            evaluator.verifier(),
            rules,
        );
        let observation = AuthorityDomain::new(PrincipalId::new());
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let budgets = BudgetAuthorityDomain::new(PrincipalId::new(), budget_profile());

        let strict = TrustedRuntime::new(
            execution.verifier(),
            observation.verifier(),
            resolution.verifier(),
        );
        let resource_runtime = ResourceRuntime::new(strict, resources.verifier());
        let policy_runtime = PolicyResourceRuntime::new(resource_runtime);
        let policy_guard = PolicyGuardedRuntime::new(policy_runtime, evaluator.verifier());
        let budget_runtime = BudgetGuardedRuntime::new(policy_guard, budgets.verifier());
        let runtime = EffectGuardedRuntime::new(budget_runtime, execution.verifier());

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

    fn authorized_action(
        harness: &Harness,
        expiry: SystemTime,
    ) -> EffectGuardedAction<Write, Authorized, u64> {
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
        action.authorize(grant, lease).unwrap()
    }

    #[test]
    fn revoked_execution_epoch_before_attempt_never_enters_adapter() {
        let harness = harness();
        let expiry = SystemTime::now() + Duration::from_secs(60);
        let action = authorized_action(&harness, expiry);
        harness.execution.revoke_all().unwrap();

        let entered = AtomicBool::new(false);
        let result = action.execute_attempt_with(|_| {
            entered.store(true, Ordering::SeqCst);
            EffectAttemptOutcome::Succeeded {
                evidence_digest: [9; 32],
            }
        });

        assert!(!entered.load(Ordering::SeqCst));
        assert!(matches!(
            result,
            Err(EffectAttemptFailure::Preflight {
                error: ExecutionPreflightError::RevokedExecutionEpoch { .. },
                ..
            })
        ));
    }

    #[test]
    fn expired_execution_lifetime_before_attempt_never_enters_adapter() {
        let harness = harness();
        let expiry = SystemTime::now() + Duration::from_secs(60);
        let action = authorized_action(&harness, expiry);
        action.clock.raise_floor(expiry + Duration::from_secs(1));

        let entered = AtomicBool::new(false);
        let result = action.execute_attempt_with(|_| {
            entered.store(true, Ordering::SeqCst);
            EffectAttemptOutcome::Succeeded {
                evidence_digest: [10; 32],
            }
        });

        assert!(!entered.load(Ordering::SeqCst));
        assert!(matches!(
            result,
            Err(EffectAttemptFailure::Preflight {
                error: ExecutionPreflightError::ExpiredExecutionGrant { .. },
                ..
            })
        ));
    }

    #[test]
    fn reported_failure_remains_observable_and_resolvable() {
        let harness = harness();
        let expiry = SystemTime::now() + Duration::from_secs(60);
        let action = authorized_action(&harness, expiry)
            .execute_attempt_with(|handle| {
                *handle += 1;
                EffectAttemptOutcome::ReportedFailure {
                    evidence_digest: [11; 32],
                }
            })
            .unwrap();

        assert!(matches!(
            action.effect_attempt().unwrap().outcome(),
            EffectAttemptOutcome::ReportedFailure { .. }
        ));

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
                Observation::new(ObservedOutcome::Partial, [12; 32]),
            )
            .unwrap();
        let decision = ResolutionDecision::Inconclusive;
        let resolver_grant = harness.resolution.issue_bound_one_shot(
            PrincipalId::new(),
            scope(),
            Some(expiry),
            action.resolution_binding(decision),
        );
        let (resolved, receipt) = action.resolve(resolver_grant, decision).unwrap();

        assert!(matches!(
            receipt.effect_attempt().outcome(),
            EffectAttemptOutcome::ReportedFailure { .. }
        ));
        assert!(receipt
            .temporal_derivation()
            .preserves_finite_parent_bound());
        assert_eq!(resolved.assurance_receipt(), &receipt);
    }

    #[test]
    fn concurrent_revocation_after_adapter_entry_preserves_orphan_attempt_evidence() {
        let harness = harness();
        let expiry = SystemTime::now() + Duration::from_secs(60);
        let action = authorized_action(&harness, expiry);

        let result = action.execute_attempt_with(|handle| {
            *handle += 1;
            harness.execution.revoke_all().unwrap();
            EffectAttemptOutcome::OutcomeUnknown {
                evidence_digest: [13; 32],
            }
        });

        match result {
            Err(EffectAttemptFailure::LineageFailedAfterAttempt { evidence, .. }) => {
                assert!(matches!(
                    evidence.outcome(),
                    EffectAttemptOutcome::OutcomeUnknown { .. }
                ));
                assert_eq!(evidence.action_scope(), &scope());
            }
            _ => panic!("expected preserved post-entry attempt evidence"),
        }
    }
}
