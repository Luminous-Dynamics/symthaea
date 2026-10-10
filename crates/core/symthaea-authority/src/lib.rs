// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic bounded-authority reference semantics for Symthaea agency.
//!
//! This crate is intentionally cognition-free, I/O-free, transport-free, and
//! crypto-key-free. It defines what a bounded positive authority record means
//! and how such a record evaluates against a supplied authority snapshot.
//!
//! It deliberately does **not** authenticate that snapshot and does not mint
//! runtime/live authority. Fresh trusted time, current verified negative facts,
//! verified delegation ancestry, admission, reservation, and effect execution
//! belong in separate verifier-owned layers.
//!
//! Core separation:
//!
//! ```text
//! capability record
//!     != verified current authority state
//!     != verified delegation ancestry
//!     != execution admission
//!     != effect
//! ```
//!
//! In particular, Phi, confidence, posterior probability, scientific support,
//! reputation, urgency, anomaly scores, or predicted benefit never create
//! authority here.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Current schema version for [`CapabilityGrant`].
pub const CAPABILITY_GRANT_SCHEMA_VERSION: u16 = 2;
/// Domain separator for capability commitments.
pub const CAPABILITY_GRANT_DOMAIN: &[u8] = b"symthaea-capability-grant-v2";

/// Stable identity of a human, service, agent, workload, or authority issuer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PrincipalId(pub String);

/// Stable semantic purpose of a grant.
///
/// Purpose is deliberately separate from task identity. Delegation is never
/// allowed to silently move authority into another purpose class.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PurposeId(pub String);

/// Stable identity of the task whose intent further bounds a grant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

/// Canonical exact resource identifier.
///
/// v0.2 deliberately uses exact matching. Hierarchical resource semantics must
/// be introduced by an explicit future schema, never inferred from prefixes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceRef(pub String);

/// Stable semantic operation name, such as `robot.move` or `repo.branch.create`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Operation(pub String);

/// Namespace for a domain-defined authority context.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityNamespace(pub String);

/// Fixed-size cryptographic commitment carried by authority objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Digest32(pub [u8; 32]);

/// Monotonic grant generation for a protected authority domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityEpoch(pub u64);

/// Exact domain-defined authority context against which a grant was issued.
///
/// The shared core does not interpret the digest. A swarm adapter may bind a
/// controller-lease context; fabrication may bind its richer monotonic authority
/// vector; another domain may bind a different exact qualified context root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityContextRef {
    pub namespace: AuthorityNamespace,
    pub digest: Digest32,
}

impl AuthorityContextRef {
    pub fn new(namespace: impl Into<String>, digest: Digest32) -> Self {
        Self {
            namespace: AuthorityNamespace(namespace.into()),
            digest,
        }
    }
}

/// Bounded cumulative consequence budget.
///
/// All fields are integers so committed authority semantics do not depend on
/// floating-point canonicalization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskBudget {
    pub mutation_units: u64,
    pub irreversible_units: u64,
    pub external_disclosure_bytes: u64,
    pub monetary_microunits: u64,
}

impl RiskBudget {
    /// True when `self` is no broader than `parent` in every dimension.
    pub fn attenuates(self, parent: Self) -> bool {
        self.mutation_units <= parent.mutation_units
            && self.irreversible_units <= parent.irreversible_units
            && self.external_disclosure_bytes <= parent.external_disclosure_bytes
            && self.monetary_microunits <= parent.monetary_microunits
    }
}

/// Serializable positive-authority record.
///
/// This type is authority **data**, not a live capability. Cloning,
/// deserializing, caching, or replaying it cannot by itself authorize execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityGrant {
    pub schema_version: u16,
    pub grant_id: String,
    pub issuer: PrincipalId,
    pub subject: PrincipalId,
    /// Optional exact executor/audience restriction.
    pub audience: Option<PrincipalId>,
    /// Exact semantic purpose. Delegation may not change it.
    pub purpose: PurposeId,
    /// Optional exact task binding.
    pub task: Option<TaskId>,
    /// Grant succession/replay epoch.
    pub authority_epoch: AuthorityEpoch,
    /// Exact domain authority snapshot against which this record was issued.
    pub authority_context: AuthorityContextRef,
    /// Exact resources this record may affect.
    pub resources: BTreeSet<ResourceRef>,
    /// Semantic operations permitted on those resources.
    pub operations: BTreeSet<Operation>,
    /// Optional plan binding. A child may add but never remove this restriction.
    pub plan_digest: Option<Digest32>,
    /// Optional observed-world binding. A child may add but never remove it.
    pub world_digest: Option<Digest32>,
    /// Optional trusted wall-clock expiry in Unix seconds.
    pub expires_at_unix_s: Option<u64>,
    /// Maximum committed plus durably reserved uses.
    pub max_uses: u32,
    /// Number of further delegation edges permitted after this grant.
    pub delegation_depth_remaining: u8,
    /// Cumulative consequence ceiling.
    pub risk_budget: RiskBudget,
    /// Parent commitment for delegated grants.
    ///
    /// Presence of this field is not proof that the parent or full ancestry is
    /// current. Single-record evaluation fails closed when this field is set.
    pub parent_digest: Option<Digest32>,
}

impl CapabilityGrant {
    /// Construct a minimally initialized root record. Callers must still populate
    /// at least one resource and operation before evaluation can succeed.
    pub fn new(
        grant_id: impl Into<String>,
        issuer: PrincipalId,
        subject: PrincipalId,
        purpose: PurposeId,
        authority_epoch: AuthorityEpoch,
        authority_context: AuthorityContextRef,
    ) -> Self {
        Self {
            schema_version: CAPABILITY_GRANT_SCHEMA_VERSION,
            grant_id: grant_id.into(),
            issuer,
            subject,
            audience: None,
            purpose,
            task: None,
            authority_epoch,
            authority_context,
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

    /// Validate closed structural requirements before authority evaluation.
    pub fn validate(&self) -> Result<(), GrantValidationError> {
        if self.schema_version != CAPABILITY_GRANT_SCHEMA_VERSION {
            return Err(GrantValidationError::UnsupportedSchema);
        }
        if self.grant_id.is_empty() {
            return Err(GrantValidationError::EmptyGrantId);
        }
        if self.issuer.0.is_empty()
            || self.subject.0.is_empty()
            || self.audience.as_ref().is_some_and(|value| value.0.is_empty())
        {
            return Err(GrantValidationError::EmptyPrincipal);
        }
        if self.purpose.0.is_empty() {
            return Err(GrantValidationError::EmptyPurpose);
        }
        if self.task.as_ref().is_some_and(|value| value.0.is_empty()) {
            return Err(GrantValidationError::EmptyTask);
        }
        if self.authority_epoch.0 == 0 {
            return Err(GrantValidationError::ZeroAuthorityEpoch);
        }
        if self.authority_context.namespace.0.is_empty() {
            return Err(GrantValidationError::EmptyAuthorityNamespace);
        }
        if self.authority_context.digest.0 == [0; 32] {
            return Err(GrantValidationError::ZeroAuthorityContextDigest);
        }
        if self.resources.is_empty() {
            return Err(GrantValidationError::EmptyResources);
        }
        if self.resources.iter().any(|value| value.0.is_empty()) {
            return Err(GrantValidationError::EmptyResourceIdentifier);
        }
        if self.operations.is_empty() {
            return Err(GrantValidationError::EmptyOperations);
        }
        if self.operations.iter().any(|value| value.0.is_empty()) {
            return Err(GrantValidationError::EmptyOperationIdentifier);
        }
        if self.max_uses == 0 {
            return Err(GrantValidationError::ZeroMaxUses);
        }
        Ok(())
    }

    /// Deterministic domain-separated commitment to every authority-relevant field.
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(CAPABILITY_GRANT_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.string(&self.grant_id);
        transcript.string(&self.issuer.0);
        transcript.string(&self.subject.0);
        transcript.optional_string(self.audience.as_ref().map(|value| value.0.as_str()));
        transcript.string(&self.purpose.0);
        transcript.optional_string(self.task.as_ref().map(|value| value.0.as_str()));
        transcript.u64(self.authority_epoch.0);
        transcript.string(&self.authority_context.namespace.0);
        transcript.digest(self.authority_context.digest);
        transcript.u32(self.resources.len() as u32);
        for resource in &self.resources {
            transcript.string(&resource.0);
        }
        transcript.u32(self.operations.len() as u32);
        for operation in &self.operations {
            transcript.string(&operation.0);
        }
        transcript.optional_digest(self.plan_digest);
        transcript.optional_digest(self.world_digest);
        transcript.optional_u64(self.expires_at_unix_s);
        transcript.u32(self.max_uses);
        transcript.byte(self.delegation_depth_remaining);
        transcript.u64(self.risk_budget.mutation_units);
        transcript.u64(self.risk_budget.irreversible_units);
        transcript.u64(self.risk_budget.external_disclosure_bytes);
        transcript.u64(self.risk_budget.monetary_microunits);
        transcript.optional_digest(self.parent_digest);
        Digest32(*transcript.finish().as_bytes())
    }

    /// Verify that a delegated record is no broader than `parent`.
    ///
    /// A successful result proves only this static attenuation edge. It does not
    /// prove that the parent or any earlier ancestor is currently valid.
    pub fn validate_attenuation(&self, parent: &Self) -> Result<(), AttenuationError> {
        self.validate().map_err(AttenuationError::InvalidGrant)?;
        parent.validate().map_err(AttenuationError::InvalidGrant)?;
        if self.issuer != parent.subject {
            return Err(AttenuationError::IssuerNotParentSubject);
        }
        if self.parent_digest != Some(parent.digest()) {
            return Err(AttenuationError::ParentDigestMismatch);
        }
        if self.authority_epoch != parent.authority_epoch {
            return Err(AttenuationError::EpochChanged);
        }
        if self.authority_context != parent.authority_context {
            return Err(AttenuationError::ContextChanged);
        }
        if self.purpose != parent.purpose {
            return Err(AttenuationError::PurposeChanged);
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum GrantValidationError {
    #[error("unsupported capability schema")]
    UnsupportedSchema,
    #[error("grant ID must not be empty")]
    EmptyGrantId,
    #[error("issuer, subject, and explicit audience identities must not be empty")]
    EmptyPrincipal,
    #[error("purpose must not be empty")]
    EmptyPurpose,
    #[error("explicit task binding must not be empty")]
    EmptyTask,
    #[error("authority epoch zero is reserved")]
    ZeroAuthorityEpoch,
    #[error("authority context namespace must not be empty")]
    EmptyAuthorityNamespace,
    #[error("authority context digest must not be the all-zero placeholder")]
    ZeroAuthorityContextDigest,
    #[error("grant must name at least one exact resource")]
    EmptyResources,
    #[error("resource identifiers must not be empty")]
    EmptyResourceIdentifier,
    #[error("grant must name at least one semantic operation")]
    EmptyOperations,
    #[error("operation identifiers must not be empty")]
    EmptyOperationIdentifier,
    #[error("max_uses must be greater than zero")]
    ZeroMaxUses,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AttenuationError {
    #[error("invalid grant: {0}")]
    InvalidGrant(GrantValidationError),
    #[error("delegated grant issuer must equal the parent subject")]
    IssuerNotParentSubject,
    #[error("delegated grant does not bind the parent digest")]
    ParentDigestMismatch,
    #[error("delegation changed the authority epoch")]
    EpochChanged,
    #[error("delegation changed the authority context")]
    ContextChanged,
    #[error("delegation changed the semantic purpose")]
    PurposeChanged,
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

/// Crash-conservative accounting supplied to the pure evaluator.
///
/// Higher layers are responsible for proving that these counters came from the
/// correct durable grant account. The pure core only applies their semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantUseState {
    pub committed: u32,
    pub reserved: u32,
}

impl GrantUseState {
    pub fn charged(self) -> u32 {
        self.committed.saturating_add(self.reserved)
    }
}

/// Negative authority fact. Applicable facts dominate positive grants.
///
/// Authenticity, completeness, freshness, and frontier provenance of this set
/// are deliberately outside this crate. The existing verified-authority-state
/// lineage supplies a stronger challenge/witness boundary for that job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NegativeAuthorityFact {
    RevokeGrant { grant_digest: Digest32 },
    RevokeContext { context: AuthorityContextRef },
    TombstonePrincipal { principal: PrincipalId },
    FreezeResource { resource: ResourceRef },
    MinimumResourceEpoch {
        resource: ResourceRef,
        minimum_epoch: AuthorityEpoch,
    },
}

/// Inputs to the pure authority evaluator.
///
/// **Security boundary:** constructing this value does not prove that the time,
/// epoch, context, use state, or negative facts are current/authentic. It is a
/// deterministic input object, not a verified authority-state capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityEvaluationInput {
    pub now_unix_s: u64,
    pub current_epoch: AuthorityEpoch,
    pub current_authority_context: AuthorityContextRef,
    pub use_state: GrantUseState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenyReason {
    InvalidGrant(GrantValidationError),
    DelegationChainRequired,
    EpochStale,
    ContextMismatch,
    Expired,
    UseBudgetExhausted,
    ExplicitlyRevoked,
    ContextRevoked,
    PrincipalTombstoned,
    ResourceFrozen,
    ResourceEpochStale,
}

/// Pure semantic result only.
///
/// `Allow` means a **root** record is eligible under the supplied evaluator
/// inputs. It does **not** mean those inputs were authenticated, that a current
/// verified authority state exists, or that execution is admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityDecision {
    Allow,
    Deny(DenyReason),
}

/// Evaluate one root positive-authority record against supplied current-state facts.
///
/// Delegated records fail closed here because a `parent_digest` does not prove
/// the full ancestry is current. A separately qualified delegation-chain/admission
/// layer must validate that ancestry before delegated authority can become live.
///
/// The function is deliberately pure and deterministic. Security-sensitive
/// callers must first obtain trusted/current authority inputs from verifier-owned
/// layers and must perform execution admission separately after `Allow`.
pub fn evaluate_authority(
    grant: &CapabilityGrant,
    input: &AuthorityEvaluationInput,
    negative_facts: &[NegativeAuthorityFact],
) -> AuthorityDecision {
    if let Err(error) = grant.validate() {
        return AuthorityDecision::Deny(DenyReason::InvalidGrant(error));
    }
    if grant.parent_digest.is_some() {
        return AuthorityDecision::Deny(DenyReason::DelegationChainRequired);
    }
    if grant.authority_epoch != input.current_epoch {
        return AuthorityDecision::Deny(DenyReason::EpochStale);
    }
    if grant.authority_context != input.current_authority_context {
        return AuthorityDecision::Deny(DenyReason::ContextMismatch);
    }
    if grant
        .expires_at_unix_s
        .is_some_and(|expiry| input.now_unix_s > expiry)
    {
        return AuthorityDecision::Deny(DenyReason::Expired);
    }
    if input.use_state.charged() >= grant.max_uses {
        return AuthorityDecision::Deny(DenyReason::UseBudgetExhausted);
    }

    let grant_digest = grant.digest();
    for fact in negative_facts {
        match fact {
            NegativeAuthorityFact::RevokeGrant { grant_digest: revoked }
                if *revoked == grant_digest =>
            {
                return AuthorityDecision::Deny(DenyReason::ExplicitlyRevoked);
            }
            NegativeAuthorityFact::RevokeContext { context }
                if context == &grant.authority_context =>
            {
                return AuthorityDecision::Deny(DenyReason::ContextRevoked);
            }
            NegativeAuthorityFact::TombstonePrincipal { principal }
                if principal == &grant.subject
                    || principal == &grant.issuer
                    || grant.audience.as_ref() == Some(principal) =>
            {
                return AuthorityDecision::Deny(DenyReason::PrincipalTombstoned);
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

    fn bytes(&mut self, value: &[u8]) {
        self.u32(value.len() as u32);
        self.0.update(value);
    }

    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.0.update(&value.0);
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
            Some(value) => {
                self.byte(1);
                self.digest(value);
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

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn authority_context(byte: u8) -> AuthorityContextRef {
        AuthorityContextRef::new("symthaea.test.authority-context.v1", digest(byte))
    }

    fn robot_grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "auth-tiny-001",
            PrincipalId("issuer-1".into()),
            PrincipalId("controller-1".into()),
            PurposeId("goal-directed-actuation".into()),
            AuthorityEpoch(5),
            authority_context(1),
        );
        grant.resources.insert(ResourceRef("robot-1".into()));
        grant.operations.insert(Operation("move".into()));
        grant.max_uses = 2;
        grant.delegation_depth_remaining = 2;
        grant.risk_budget = RiskBudget {
            mutation_units: 10,
            irreversible_units: 0,
            external_disclosure_bytes: 0,
            monetary_microunits: 0,
        };
        grant
    }

    fn evaluation_input() -> AuthorityEvaluationInput {
        AuthorityEvaluationInput {
            now_unix_s: 100,
            current_epoch: AuthorityEpoch(5),
            current_authority_context: authority_context(1),
            use_state: GrantUseState::default(),
        }
    }

    #[test]
    fn auth_tiny_001_exact_root_record_is_eligible_under_matching_inputs() {
        assert_eq!(
            evaluate_authority(&robot_grant(), &evaluation_input(), &[]),
            AuthorityDecision::Allow
        );
    }

    #[test]
    fn historical_v1_schema_fails_closed() {
        let mut grant = robot_grant();
        grant.schema_version = 1;
        assert_eq!(
            evaluate_authority(&grant, &evaluation_input(), &[]),
            AuthorityDecision::Deny(DenyReason::InvalidGrant(
                GrantValidationError::UnsupportedSchema
            ))
        );
    }

    #[test]
    fn exact_context_binding_fails_closed() {
        let mut input = evaluation_input();
        input.current_authority_context = authority_context(2);
        assert_eq!(
            evaluate_authority(&robot_grant(), &input, &[]),
            AuthorityDecision::Deny(DenyReason::ContextMismatch)
        );
    }

    #[test]
    fn revocation_dominates_otherwise_eligible_record() {
        let grant = robot_grant();
        let negative = NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        };
        assert_eq!(
            evaluate_authority(&grant, &evaluation_input(), &[negative]),
            AuthorityDecision::Deny(DenyReason::ExplicitlyRevoked)
        );
    }

    #[test]
    fn context_revocation_dominates_matching_context() {
        let grant = robot_grant();
        let negative = NegativeAuthorityFact::RevokeContext {
            context: grant.authority_context.clone(),
        };
        assert_eq!(
            evaluate_authority(&grant, &evaluation_input(), &[negative]),
            AuthorityDecision::Deny(DenyReason::ContextRevoked)
        );
    }

    #[test]
    fn reserved_uses_are_charged_before_dispatch() {
        let mut input = evaluation_input();
        input.use_state = GrantUseState {
            committed: 1,
            reserved: 1,
        };
        assert_eq!(
            evaluate_authority(&robot_grant(), &input, &[]),
            AuthorityDecision::Deny(DenyReason::UseBudgetExhausted)
        );
    }

    #[test]
    fn stale_authority_epoch_is_denied() {
        let mut input = evaluation_input();
        input.current_epoch = AuthorityEpoch(6);
        assert_eq!(
            evaluate_authority(&robot_grant(), &input, &[]),
            AuthorityDecision::Deny(DenyReason::EpochStale)
        );
    }

    #[test]
    fn delegated_record_requires_separate_chain_verification() {
        let parent = robot_grant();
        let mut child = parent.clone();
        child.grant_id = "auth-tiny-001-child".into();
        child.issuer = parent.subject.clone();
        child.subject = PrincipalId("controller-2".into());
        child.parent_digest = Some(parent.digest());
        child.delegation_depth_remaining = 1;
        child.max_uses = 1;
        child.risk_budget.mutation_units = 5;
        assert!(child.validate_attenuation(&parent).is_ok());
        assert_eq!(
            evaluate_authority(&child, &evaluation_input(), &[]),
            AuthorityDecision::Deny(DenyReason::DelegationChainRequired)
        );
    }

    #[test]
    fn delegation_attenuates_but_cannot_change_purpose_or_context() {
        let parent = robot_grant();
        let mut child = parent.clone();
        child.grant_id = "auth-tiny-001-child".into();
        child.issuer = parent.subject.clone();
        child.subject = PrincipalId("controller-2".into());
        child.parent_digest = Some(parent.digest());
        child.delegation_depth_remaining = 1;
        child.max_uses = 1;
        child.risk_budget.mutation_units = 5;
        assert!(child.validate_attenuation(&parent).is_ok());

        let mut changed_purpose = child.clone();
        changed_purpose.purpose = PurposeId("software-deployment".into());
        assert_eq!(
            changed_purpose.validate_attenuation(&parent),
            Err(AttenuationError::PurposeChanged)
        );

        let mut changed_context = child;
        changed_context.authority_context = authority_context(3);
        assert_eq!(
            changed_context.validate_attenuation(&parent),
            Err(AttenuationError::ContextChanged)
        );
    }

    #[test]
    fn purpose_and_context_are_commitment_sensitive() {
        let grant = robot_grant();
        let original = grant.digest();

        let mut changed_purpose = grant.clone();
        changed_purpose.purpose = PurposeId("safety-fallback".into());
        assert_ne!(original, changed_purpose.digest());

        let mut changed_context = grant;
        changed_context.authority_context = authority_context(9);
        assert_ne!(original, changed_context.digest());
    }

    #[test]
    fn structural_invalidity_fails_before_positive_decision() {
        let mut grant = robot_grant();
        grant.operations.clear();
        assert_eq!(
            evaluate_authority(&grant, &evaluation_input(), &[]),
            AuthorityDecision::Deny(DenyReason::InvalidGrant(
                GrantValidationError::EmptyOperations
            ))
        );
    }

    #[test]
    fn empty_exact_identifiers_and_placeholder_context_are_rejected() {
        let mut grant = robot_grant();
        grant.resources.clear();
        grant.resources.insert(ResourceRef(String::new()));
        assert_eq!(
            grant.validate(),
            Err(GrantValidationError::EmptyResourceIdentifier)
        );

        let mut grant = robot_grant();
        grant.operations.clear();
        grant.operations.insert(Operation(String::new()));
        assert_eq!(
            grant.validate(),
            Err(GrantValidationError::EmptyOperationIdentifier)
        );

        let mut grant = robot_grant();
        grant.authority_context.digest = Digest32([0; 32]);
        assert_eq!(
            grant.validate(),
            Err(GrantValidationError::ZeroAuthorityContextDigest)
        );
    }

    #[test]
    fn issuer_tombstone_is_negative_authority() {
        let grant = robot_grant();
        let negative = NegativeAuthorityFact::TombstonePrincipal {
            principal: grant.issuer.clone(),
        };
        assert_eq!(
            evaluate_authority(&grant, &evaluation_input(), &[negative]),
            AuthorityDecision::Deny(DenyReason::PrincipalTombstoned)
        );
    }

    proptest! {
        #[test]
        fn risk_budget_componentwise_attenuation_is_monotonic(
            a in 0u64..10_000,
            b in 0u64..10_000,
            c in 0u64..10_000,
            d in 0u64..10_000,
            da in 0u64..10_000,
            db in 0u64..10_000,
            dc in 0u64..10_000,
            dd in 0u64..10_000,
        ) {
            let parent = RiskBudget {
                mutation_units: a.saturating_add(da),
                irreversible_units: b.saturating_add(db),
                external_disclosure_bytes: c.saturating_add(dc),
                monetary_microunits: d.saturating_add(dd),
            };
            let child = RiskBudget {
                mutation_units: a,
                irreversible_units: b,
                external_disclosure_bytes: c,
                monetary_microunits: d,
            };
            prop_assert!(child.attenuates(parent));
        }
    }
}
