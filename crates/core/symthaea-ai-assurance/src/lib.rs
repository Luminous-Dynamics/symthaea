// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Project-neutral assurance primitives for autonomous systems.
//!
//! The crate is intentionally independent of Symthaea cognition. Models and
//! planners propose actions; trusted host code decides which authority values
//! they receive.
//!
//! [`capability`] and [`action`] expose the low-level affine capability and
//! typestate mechanics. [`trusted`] binds actions and grants to host-selected
//! authority domains and revocation epochs. [`resolution`] provides distinct
//! final-interpretation authority. [`resource`] binds logical scopes to concrete
//! adapter-resolved identities and retained handles. [`policy`] records why
//! trusted policy admitted an action and carries that provenance into exact
//! resource-bound execution authority. [`temporal_policy`] adds a strict
//! production-facing derivation rule so execution authority cannot silently
//! outlive the policy admission that justified it. [`policy_guard`] additionally
//! rechecks policy-domain revocation immediately before the concrete side-effect
//! boundary. [`budget`] separates permission from quantitative resource authority
//! using conserved exact-action leases and explicit enforcement truth labels.
//! [`budget_guard`] composes those leases with the strongest policy/resource
//! boundary so budget authority is revalidated immediately before adapter entry.
//! [`effect_guard`] adds exact execution-domain/expiry preflight and forces
//! effectful adapters to report success, failure, uncertainty, or proven
//! transactional no-effect as evidence-bearing outcome data rather than using a
//! generic `Err` as proof that nothing happened. [`independence`] makes
//! observer/resolver separation an explicit host policy and records the exact
//! principal/domain guarantee actually enforced rather than overloading the word
//! “independent.” [`budget_purpose`] adds a separate quantitative-policy
//! admission proving which trusted purpose policy approved the exact conserved
//! resource envelope for the exact action.
//!
//! Security-sensitive state-changing integrations should compose these layers so
//! trust anchors, validation time, concrete resources, policy admission,
//! revocation, temporal bounds, quantitative capacity, approved purpose,
//! effect-attempt evidence, and separation-of-duties claims remain host-owned
//! rather than model-selected.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod action;
pub mod budget;
pub mod budget_guard;
pub mod budget_purpose;
pub mod capability;
pub mod effect_guard;
pub mod host;
pub mod independence;
pub mod policy;
pub mod policy_guard;
pub mod resolution;
pub mod resource;
pub mod temporal_policy;
pub mod trusted;

pub use action::{
    Action, ActionDescriptor, ActionError, ActionId, ActionRisk, Authorized, EvidenceReceipt,
    Executed, Observation, Observed, ObservedOutcome, Proposed, ResolutionDecision, Resolved,
    RiskAssessed,
};
pub use budget::{
    BudgetAuthorityDomain, BudgetDimension, BudgetEnforcement, BudgetError, BudgetLease,
    BudgetProfile, BudgetQuantities, BudgetReleaseReceipt, BudgetSplitFailure, BudgetVerifier,
    EnforcementClass,
};
pub use budget_guard::{
    BudgetAdapterError, BudgetGuardedAction, BudgetGuardedAuthorizeError, BudgetGuardedRuntime,
    BudgetLeaseEvidence, BudgetedEvidenceReceipt,
};
pub use budget_purpose::{
    BudgetPurposeAdmissionReceipt, BudgetPurposeAuthorityDomain, BudgetPurposeAuthorizeError,
    BudgetPurposeDescriptor, BudgetPurposeError, BudgetPurposeEvidence, BudgetPurposeRules,
    BudgetPurposeVerifier, PurposeBoundBudgetLease, PurposeBoundEvidenceReceipt,
    PurposeEffectAttemptFailure, PurposeGuardedAction, PurposeGuardedRuntime,
};
pub use capability::{
    AuthorityRoot, BoundOneShotCapability, Capability, CapabilityKind, Deploy, Execute, GrantError,
    GrantId, GrantMetadata, Network, Observe, OneShotCapability, PrincipalId, Read, Scope,
    ScopeError, UpdateModel, Write,
};
pub use effect_guard::{
    EffectAssuredEvidenceReceipt, EffectAttemptEvidence, EffectAttemptFailure,
    EffectAttemptOutcome, EffectGuardedAction, EffectGuardedAuthorizeError, EffectGuardedRuntime,
    EffectInnerExecutionError, ExecutionPreflightError,
};
pub use host::{ResolutionError, ResolutionEvidenceReceipt, RuntimeAction, TrustedRuntime};
pub use independence::{
    IndependenceConfigError, IndependenceEffectAttemptFailure, IndependenceEvidenceReceipt,
    IndependenceGuardedAction, IndependenceGuardedRuntime, IndependenceObservationError,
    IndependencePolicy, IndependenceResolutionError, IndependenceRole,
};
pub use policy::{
    ApprovalEvidence, PolicyAdmission, PolicyAdmissionReceipt, PolicyAuthorizationEvidence,
    PolicyDescriptor, PolicyError, PolicyEvaluatorDomain, PolicyExecutionDomain, PolicyGrant,
    PolicyMode, PolicyResourceAction, PolicyResourceEvidenceReceipt, PolicyResourceRuntime,
    PolicyVerifier,
};
pub use policy_guard::{
    PolicyGuardError, PolicyGuardedAction, PolicyGuardedAuthorizeError,
    PolicyGuardedExecutionError, PolicyGuardedRuntime,
};
pub use resolution::{ResolutionAuthorityDomain, ResolutionGrant, ResolutionVerifier};
pub use resource::{
    AdapterSchema, ResolvedResource, ResourceAction, ResourceError, ResourceEvidenceReceipt,
    ResourceExecutionError, ResourceIdentity, ResourceIdentityError, ResourceResolverDomain,
    ResourceRuntime, ResourceVerifier,
};
pub use temporal_policy::{
    TemporalDerivationEvidence, TemporalPolicyAdmission, TemporalPolicyError,
    TemporalPolicyEvaluatorDomain, TemporalPolicyExecutionDomain, TemporalPolicyGrant,
    TemporalPolicyRules,
};
pub use trusted::{
    AuthorityDomain, AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError,
    TrustedAction, TrustedBoundOneShotCapability, TrustedEvidenceReceipt,
};

impl<K: CapabilityKind, H> std::fmt::Debug for EffectAttemptFailure<K, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preflight { error, .. } => f
                .debug_struct("EffectAttemptFailure::Preflight")
                .field("error", error)
                .field("action", &"<authority-bearing action retained>")
                .finish(),
            Self::RejectedBeforeAttempt { error } => f
                .debug_struct("EffectAttemptFailure::RejectedBeforeAttempt")
                .field("error", error)
                .finish(),
            Self::LineageFailedAfterAttempt { evidence, error } => f
                .debug_struct("EffectAttemptFailure::LineageFailedAfterAttempt")
                .field("evidence", evidence)
                .field("error", error)
                .finish(),
        }
    }
}
