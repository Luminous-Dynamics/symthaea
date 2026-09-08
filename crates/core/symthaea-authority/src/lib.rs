// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic bounded-authority primitives for Symthaea agency.
//!
//! This crate is intentionally cognition-free, I/O-free, and crypto-key-free.
//! It defines the small reference semantics that higher layers can sign,
//! transport, persist, or map onto external policy systems. In particular:
//!
//! - confidence or Phi never creates authority;
//! - delegated grants may only attenuate parent authority;
//! - negative authority facts dominate positive grants;
//! - authority epochs prevent stale-grant resurrection;
//! - use and consequence budgets are explicit and conservatively accounted.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Current schema version for [`CapabilityGrant`].
pub const CAPABILITY_GRANT_SCHEMA_VERSION: u16 = 1;
/// Domain separator for capability commitments.
pub const CAPABILITY_GRANT_DOMAIN: &[u8] = b"symthaea-capability-grant-v1";

/// Stable identity of a human, service, agent, workload, or authority issuer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PrincipalId(pub String);

/// Stable identity of the task whose intent bounds a grant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

/// Canonical resource identifier.
///
/// v0.1 deliberately uses exact resource matching. Hierarchical resource
/// scopes must be introduced explicitly in a later schema rather than inferred
/// from string prefixes, which avoids accidental scope broadening.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceRef(pub String);

/// Stable semantic operation name, such as `service.restart` or `repo.branch.create`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Operation(pub String);

/// Fixed-size cryptographic commitment carried by security objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Digest32(pub [u8; 32]);

/// Monotonic authority generation for a protected domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityEpoch(pub u64);

/// Bounded cumulative consequence budget.
///
/// All fields are integers so signed/committed authority objects never depend
/// on floating-point canonicalization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskBudget {
    /// Generic bounded mutation units assigned by the enclosing policy.
    pub mutation_units: u64,
    /// Number of irreversible-effect units permitted.
    pub irreversible_units: u64,
    /// Maximum externally disclosed bytes charged to this grant.
    pub external_disclosure_bytes: u64,
    /// Monetary ceiling in millionths of the configured currency unit.
    pub monetary_microunits: u64,
}

impl RiskBudget {
    /// Returns true when `self` is no broader than `parent` in every dimension.
    pub fn attenuates(self, parent: Self) -> bool {
        self.mutation_units <= parent.mutation_units
            && self.irreversible_units <= parent.irreversible_units
            && self.external_disclosure_bytes <= parent.external_disclosure_bytes
            && self.monetary_microunits <= parent.monetary_microunits
    }
}

/// Positive authority delegated to one subject for one bounded purpose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityGrant {
    /// Schema version. Unknown versions must fail closed at enforcement points.
    pub schema_version: u16,
    /// Human-readable stable grant identifier for operations and audit UX.
    pub grant_id: String,
    /// Principal that issued this grant.
    pub issuer: PrincipalId,
    /// Principal receiving authority.
    pub subject: PrincipalId,
    /// Optional exact executor/audience restriction.
    pub audience: Option<PrincipalId>,
    /// Optional exact task binding.
    pub task: Option<TaskId>,
    /// Authority epoch in which this grant was issued.
    pub authority_epoch: AuthorityEpoch,
    /// Exact resources this grant may affect.
    pub resources: BTreeSet<ResourceRef>,
    /// Semantic operations allowed on those resources.
    pub operations: BTreeSet<Operation>,
    /// Optional plan binding. A child may add but never remove this restriction.
    pub plan_digest: Option<Digest32>,
    /// Optional observed-world binding. A child may add but never remove it.
    pub world_digest: Option<Digest32>,
    /// Optional wall-clock expiry in Unix seconds.
    pub expires_at_unix_s: Option<u64>,
    /// Maximum committed plus reserved uses.
    pub max_uses: u32,
    /// Number of further delegation edges permitted after this grant.
    pub delegation_depth_remaining: u8,
    /// Cumulative consequence ceiling.
    pub risk_budget: RiskBudget,
    /// Parent commitment for delegated grants.
    pub parent_digest: Option<Digest32>,
}

impl CapabilityGrant {
    /// Construct a minimally initialized grant.
    pub fn new(
        grant_id: impl Into<String>,
        issuer: PrincipalId,
        subject: PrincipalId,
        authority_epoch: AuthorityEpoch,
    ) -> Self {
        Self {
            schema_version: CAPABILITY_GRANT_SCHEMA_VERSION,
            grant_id: grant_id.into(),
            issuer,
            subject,
            audience: None,
            task: None,
            authority_epoch,
            resources: BTreeSet::new(),
            operations: BTreeSet::new(),
            plan_digest: None,
            world_digest: None,
            expires_at_unix_s: None,
            max_uses: 1,
            delegation_depth_remaining: 0,
            risk_budget: RiskBudget::default(),
            parent_digest: None,
        }
    }

    /// Deterministic domain-separated commitment to every security-relevant field.
    pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(CAPABILITY_GRANT_DOMAIN);
        t.u16(self.schema_version);
        t.string(&self.grant_id);
        t.string(&self.issuer.0);
        t.string(&self.subject.0);
        t.optional_string(self.audience.as_ref().map(|v| v.0.as_str()));
        t.optional_string(self.task.as_ref().map(|v| v.0.as_str()));
        t.u64(self.authority_epoch.0);
        t.u32(self.resources.len() as u32);
        for resource in &self.resources {
            t.string(&resource.0);
        }
        t.u32(self.operations.len() as u32);
        for operation in &self.operations {
            t.string(&operation.0);
        }
        t.optional_digest(self.plan_digest);
        t.optional_digest(self.world_digest);
        t.optional_u64(self.expires_at_unix_s);
        t.u32(self.max_uses);
        t.byte(self.delegation_depth_remaining);
        t.u64(self.risk_budget.mutation_units);
        t.u64(self.risk_budget.irreversible_units);
        t.u64(self.risk_budget.external_disclosure_bytes);
        t.u64(self.risk_budget.monetary_microunits);
        t.optional_digest(self.parent_digest);
        Digest32(*t.finish().as_bytes())
    }

    /// Verify that this delegated grant is no broader than `parent`.
    pub fn validate_attenuation(&self, parent: &CapabilityGrant) -> Result<(), AttenuationError> {
        if self.schema_version != CAPABILITY_GRANT_SCHEMA_VERSION
            || parent.schema_version != CAPABILITY_GRANT_SCHEMA_VERSION
        {
            return Err(AttenuationError::UnsupportedSchema);
        }
        if self.issuer != parent.subject {
            return Err(AttenuationError::IssuerNotParentSubject);
        }
        if self.parent_digest != Some(parent.digest()) {
            return Err(AttenuationError::ParentDigestMismatch);
        }
        if self.authority_epoch != parent.authority_epoch {
            return Err(AttenuationError::EpochChanged);
        }
        if !restriction_attenuates(&self.audience, &parent.audience) {
            return Err(AttenuationError::AudienceBroadened);
        }
        if !restriction_attenuates(&self.task, &parent.task) {
            return Err(AttenuationError::TaskBroadened);
        }
        if !self.resources.is_subset(&parent.resources) {
            return Err(AttenuationError::ResourcesBroadened);
        }
        if !self.operations.is_subset(&parent.operations) {
            return Err(AttenuationError::OperationsBroadened);
        }
        if !digest_restriction_attenuates(self.plan_digest, parent.plan_digest) {
            return Err(AttenuationError::PlanBindingRemovedOrChanged);
        }
        if !digest_restriction_attenuates(self.world_digest, parent.world_digest) {
            return Err(AttenuationError::WorldBindingRemovedOrChanged);
        }
        if !expiry_attenuates(self.expires_at_unix_s, parent.expires_at_unix_s) {
            return Err(AttenuationError::ExpiryBroadened);
        }
        if self.max_uses > parent.max_uses {
            return Err(AttenuationError::UsesBroadened);
        }
        if parent.delegation_depth_remaining == 0
            || self.delegation_depth_remaining >= parent.delegation_depth_remaining
        {
            return Err(AttenuationError::DelegationDepthBroadened);
        }
        if !self.risk_budget.attenuates(parent.risk_budget) {
            return Err(AttenuationError::RiskBroadened);
        }
        Ok(())
    }
}

fn restriction_attenuates<T: PartialEq>(child: &Option<T>, parent: &Option<T>) -> bool {
    match parent {
        Some(parent_value) => child.as_ref() == Some(parent_value),
        None => true,
    }
}

fn digest_restriction_attenuates(child: Option<Digest32>, parent: Option<Digest32>) -> bool {
    match parent {
        Some(parent_value) => child == Some(parent_value),
        None => true,
    }
}

fn expiry_attenuates(child: Option<u64>, parent: Option<u64>) -> bool {
    match parent {
        Some(parent_expiry) => child.is_some_and(|child_expiry| child_expiry <= parent_expiry),
        None => true,
    }
}

/// Why delegated authority failed monotonic attenuation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AttenuationError {
    #[error("unsupported capability schema")]
    UnsupportedSchema,
    #[error("delegated grant issuer must equal the parent subject")]
    IssuerNotParentSubject,
    #[error("delegated grant does not bind the parent digest")]
    ParentDigestMismatch,
    #[error("delegation changed the authority epoch")]
    EpochChanged,
    #[error("delegation broadened the audience")]
    AudienceBroadened,
    #[error("delegation broadened the task")]
    TaskBroadened,
    #[error("delegation broadened resource scope")]
    ResourcesBroadened,
    #[error("delegation broadened allowed operations")]
    OperationsBroadened,
    #[error("delegation removed or changed a required plan binding")]
    PlanBindingRemovedOrChanged,
    #[error("delegation removed or changed a required world binding")]
    WorldBindingRemovedOrChanged,
    #[error("delegation broadened expiry")]
    ExpiryBroadened,
    #[error("delegation increased permitted uses")]
    UsesBroadened,
    #[error("delegation depth was not strictly attenuated")]
    DelegationDepthBroadened,
    #[error("delegation broadened the cumulative risk budget")]
    RiskBroadened,
}

/// Crash-safe accounting state supplied by the runtime when evaluating a grant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantUseState {
    /// Uses durably committed as having taken effect.
    pub committed: u32,
    /// Uses durably reserved for in-flight executions.
    pub reserved: u32,
}

impl GrantUseState {
    /// Conservative number of consumed-or-potentially-consumed uses.
    pub fn charged(self) -> u32 {
        self.committed.saturating_add(self.reserved)
    }
}

/// Durable negative authority fact. Applicable facts override positive grants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NegativeAuthorityFact {
    /// Revoke one exact capability commitment.
    RevokeGrant { grant_digest: Digest32 },
    /// Permanently retire a principal identity.
    TombstonePrincipal { principal: PrincipalId },
    /// Freeze an exact resource against all granted mutations at this layer.
    FreezeResource { resource: ResourceRef },
    /// Require grants for an exact resource to come from at least this epoch.
    MinimumResourceEpoch {
        resource: ResourceRef,
        minimum_epoch: AuthorityEpoch,
    },
}

/// Runtime admission context independent of cognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityContext {
    /// Current trusted wall-clock time when available.
    pub now_unix_s: u64,
    /// Current authority epoch for the protected domain.
    pub current_epoch: AuthorityEpoch,
    /// Durable use accounting for the candidate grant.
    pub use_state: GrantUseState,
}

/// Stable reason an authority decision failed closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenyReason {
    UnsupportedSchema,
    EpochStale,
    Expired,
    UseBudgetExhausted,
    ExplicitlyRevoked,
    SubjectTombstoned,
    ResourceFrozen,
    ResourceEpochStale,
}

/// Deterministic authority decision. Higher layers may add admission obligations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityDecision {
    Allow,
    Deny(DenyReason),
}

/// Evaluate positive authority against monotonic epoch, use accounting, expiry,
/// and negative facts. Negative facts always dominate an otherwise valid grant.
pub fn evaluate_authority(
    grant: &CapabilityGrant,
    context: AuthorityContext,
    negative_facts: &[NegativeAuthorityFact],
) -> AuthorityDecision {
    if grant.schema_version != CAPABILITY_GRANT_SCHEMA_VERSION {
        return AuthorityDecision::Deny(DenyReason::UnsupportedSchema);
    }
    if grant.authority_epoch != context.current_epoch {
        return AuthorityDecision::Deny(DenyReason::EpochStale);
    }
    if grant
        .expires_at_unix_s
        .is_some_and(|expiry| context.now_unix_s > expiry)
    {
        return AuthorityDecision::Deny(DenyReason::Expired);
    }
    if context.use_state.charged() >= grant.max_uses {
        return AuthorityDecision::Deny(DenyReason::UseBudgetExhausted);
    }

    let digest = grant.digest();
    for fact in negative_facts {
        match fact {
            NegativeAuthorityFact::RevokeGrant { grant_digest } if *grant_digest == digest => {
                return AuthorityDecision::Deny(DenyReason::ExplicitlyRevoked);
            }
            NegativeAuthorityFact::TombstonePrincipal { principal }
                if *principal == grant.subject || grant.audience.as_ref() == Some(principal) =>
            {
                return AuthorityDecision::Deny(DenyReason::SubjectTombstoned);
            }
            NegativeAuthorityFact::FreezeResource { resource }
                if grant.resources.contains(resource) =>
            {
                return AuthorityDecision::Deny(DenyReason::ResourceFrozen);
            }
            NegativeAuthorityFact::MinimumResourceEpoch {
                resource,
                minimum_epoch,
            } if grant.resources.contains(resource) && grant.authority_epoch < *minimum_epoch => {
                return AuthorityDecision::Deny(DenyReason::ResourceEpochStale);
            }
            _ => {}
        }
    }

    AuthorityDecision::Allow
}

struct Transcript(blake3::Hasher);

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(domain.len() as u32).to_be_bytes());
        hasher.update(domain);
        Self(hasher)
    }

    fn byte(&mut self, value: u8) {
        self.0.update(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.0.update(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.0.update(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(&value.to_be_bytes());
    }

    fn string(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.0.update(value.as_bytes());
    }

    fn optional_string(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.byte(1);
                self.string(value);
            }
            None => self.byte(0),
        }
    }

    fn optional_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.byte(1);
                self.u64(value);
            }
            None => self.byte(0),
        }
    }

    fn optional_digest(&mut self, value: Option<Digest32>) {
        match value {
            Some(Digest32(value)) => {
                self.byte(1);
                self.0.update(&value);
            }
            None => self.byte(0),
        }
    }

    fn finish(self) -> blake3::Hash {
        self.0.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn parent_grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "parent",
            PrincipalId("user:alice".into()),
            PrincipalId("agent:root".into()),
            AuthorityEpoch(7),
        );
        grant.resources = [
            ResourceRef("host:alpha/service:postgresql".into()),
            ResourceRef("host:alpha/service:nginx".into()),
        ]
        .into_iter()
        .collect();
        grant.operations = [Operation("service.query".into()), Operation("service.restart".into())]
            .into_iter()
            .collect();
        grant.task = Some(TaskId("repair-db".into()));
        grant.expires_at_unix_s = Some(10_000);
        grant.max_uses = 6;
        grant.delegation_depth_remaining = 3;
        grant.risk_budget = RiskBudget {
            mutation_units: 10,
            irreversible_units: 0,
            external_disclosure_bytes: 0,
            monetary_microunits: 0,
        };
        grant
    }

    fn valid_child(parent: &CapabilityGrant) -> CapabilityGrant {
        let mut child = CapabilityGrant::new(
            "child",
            parent.subject.clone(),
            PrincipalId("agent:worker".into()),
            parent.authority_epoch,
        );
        child.resources = [ResourceRef("host:alpha/service:postgresql".into())]
            .into_iter()
            .collect();
        child.operations = [Operation("service.restart".into())].into_iter().collect();
        child.task = parent.task.clone();
        child.expires_at_unix_s = Some(9_000);
        child.max_uses = 2;
        child.delegation_depth_remaining = 2;
        child.risk_budget = RiskBudget {
            mutation_units: 2,
            ..RiskBudget::default()
        };
        child.parent_digest = Some(parent.digest());
        child
    }

    #[test]
    fn valid_child_attenuates_parent() {
        let parent = parent_grant();
        assert_eq!(valid_child(&parent).validate_attenuation(&parent), Ok(()));
    }

    #[test]
    fn high_confidence_is_not_an_authority_input() {
        let grant = parent_grant();
        let decision = evaluate_authority(
            &grant,
            AuthorityContext {
                now_unix_s: 9_000,
                current_epoch: AuthorityEpoch(8),
                use_state: GrantUseState::default(),
            },
            &[],
        );
        assert_eq!(decision, AuthorityDecision::Deny(DenyReason::EpochStale));
    }

    #[test]
    fn negative_fact_dominates_positive_grant() {
        let grant = parent_grant();
        let decision = evaluate_authority(
            &grant,
            AuthorityContext {
                now_unix_s: 9_000,
                current_epoch: grant.authority_epoch,
                use_state: GrantUseState::default(),
            },
            &[NegativeAuthorityFact::RevokeGrant {
                grant_digest: grant.digest(),
            }],
        );
        assert_eq!(
            decision,
            AuthorityDecision::Deny(DenyReason::ExplicitlyRevoked)
        );
    }

    #[test]
    fn reserved_uses_are_charged_before_dispatch() {
        let mut grant = parent_grant();
        grant.max_uses = 2;
        let decision = evaluate_authority(
            &grant,
            AuthorityContext {
                now_unix_s: 9_000,
                current_epoch: grant.authority_epoch,
                use_state: GrantUseState {
                    committed: 1,
                    reserved: 1,
                },
            },
            &[],
        );
        assert_eq!(
            decision,
            AuthorityDecision::Deny(DenyReason::UseBudgetExhausted)
        );
    }

    #[test]
    fn digest_changes_when_security_field_changes() {
        let grant = parent_grant();
        let mut changed = grant.clone();
        changed.max_uses += 1;
        assert_ne!(grant.digest(), changed.digest());
    }

    proptest! {
        #[test]
        fn risk_budget_attenuation_never_accepts_larger_mutation_budget(
            parent_units in 0u64..10_000,
            extra in 1u64..10_000,
        ) {
            let parent = RiskBudget { mutation_units: parent_units, ..RiskBudget::default() };
            let child = RiskBudget {
                mutation_units: parent_units.saturating_add(extra),
                ..RiskBudget::default()
            };
            prop_assert!(!child.attenuates(parent));
        }

        #[test]
        fn use_count_amplification_is_rejected(parent_uses in 1u32..10_000, extra in 1u32..10_000) {
            let mut parent = parent_grant();
            parent.max_uses = parent_uses;
            let mut child = valid_child(&parent);
            child.max_uses = parent_uses.saturating_add(extra);
            child.parent_digest = Some(parent.digest());
            prop_assert_eq!(
                child.validate_attenuation(&parent),
                Err(AttenuationError::UsesBroadened)
            );
        }
    }
}
