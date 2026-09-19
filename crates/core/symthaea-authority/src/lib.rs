// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic bounded-authority and runtime-admission primitives.
//!
//! The crate is intentionally cognition-free, I/O-free, transport-free, and
//! crypto-key-free. It defines small reference semantics that higher layers can
//! sign, transport, persist, or map onto domain policy systems.
//!
//! Core rule:
//!
//! ```text
//! grant record != runtime admission != live authority != effect
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
/// Purpose is deliberately separate from task identity: two tasks may share one
/// purpose, while delegation is never allowed to silently move authority into a
/// different purpose class.
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

/// Fixed-size cryptographic commitment carried by security objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Digest32(pub [u8; 32]);

/// Monotonic grant generation for a protected domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityEpoch(pub u64);

/// Process/session generation used to fence live authority across restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RuntimeEpoch(pub u64);

/// Exact domain-defined authority context against which a grant is evaluated.
///
/// The shared core does not interpret the digest. Swarm, fabrication, browser,
/// HAL, or other domains may commit different native authority-state models.
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
/// This type is deliberately **not** live authority. Deserializing, cloning, or
/// replaying it does not authorize execution. Current admission must produce a
/// non-serializable [`LiveGrant`].
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
    pub parent_digest: Option<Digest32>,
}

impl CapabilityGrant {
    /// Construct a minimally initialized record. Callers must still populate at
    /// least one resource and operation before admission can succeed.
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

    /// Validate closed structural requirements before any authority evaluation.
    pub fn validate(&self) -> Result<(), GrantValidationError> {
        if self.schema_version != CAPABILITY_GRANT_SCHEMA_VERSION {
            return Err(GrantValidationError::UnsupportedSchema);
        }
        if self.grant_id.is_empty() {
            return Err(GrantValidationError::EmptyGrantId);
        }
        if self.issuer.0.is_empty() || self.subject.0.is_empty() {
            return Err(GrantValidationError::EmptyPrincipal);
        }
        if self.purpose.0.is_empty() {
            return Err(GrantValidationError::EmptyPurpose);
        }
        if self.authority_epoch.0 == 0 {
            return Err(GrantValidationError::ZeroAuthorityEpoch);
        }
        if self.authority_context.namespace.0.is_empty() {
            return Err(GrantValidationError::EmptyAuthorityNamespace);
        }
        if self.resources.is_empty() {
            return Err(GrantValidationError::EmptyResources);
        }
        if self.operations.is_empty() {
            return Err(GrantValidationError::EmptyOperations);
        }
        if self.max_uses == 0 {
            return Err(GrantValidationError::ZeroMaxUses);
        }
        Ok(())
    }

    /// Deterministic domain-separated commitment to every authority-relevant field.
    pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(CAPABILITY_GRANT_DOMAIN);
        t.u16(self.schema_version);
        t.string(&self.grant_id);
        t.string(&self.issuer.0);
        t.string(&self.subject.0);
        t.optional_string(self.audience.as_ref().map(|value| value.0.as_str()));
        t.string(&self.purpose.0);
        t.optional_string(self.task.as_ref().map(|value| value.0.as_str()));
        t.u64(self.authority_epoch.0);
        t.string(&self.authority_context.namespace.0);
        t.digest(self.authority_context.digest);
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

    /// Verify that a delegated record is no broader than `parent`.
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

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GrantValidationError {
    #[error("unsupported capability schema")]
    UnsupportedSchema,
    #[error("grant ID must not be empty")]
    EmptyGrantId,
    #[error("issuer and subject identities must not be empty")]
    EmptyPrincipal,
    #[error("purpose must not be empty")]
    EmptyPurpose,
    #[error("authority epoch zero is reserved")]
    ZeroAuthorityEpoch,
    #[error("authority context namespace must not be empty")]
    EmptyAuthorityNamespace,
    #[error("grant must name at least one exact resource")]
    EmptyResources,
    #[error("grant must name at least one semantic operation")]
    EmptyOperations,
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

/// Crash-safe accounting state supplied by the runtime when evaluating a record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantUseState {
    /// Uses durably committed as having taken effect.
    pub committed: u32,
    /// Uses durably reserved for in-flight executions.
    pub reserved: u32,
}

impl GrantUseState {
    /// Conservative consumed-or-potentially-consumed count.
    pub fn charged(self) -> u32 {
        self.committed.saturating_add(self.reserved)
    }
}

/// Durable negative authority fact. Applicable facts dominate positive records.
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

/// Current authority state used for deterministic record evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityEvaluationContext {
    pub now_unix_s: u64,
    pub current_epoch: AuthorityEpoch,
    pub current_authority_context: AuthorityContextRef,
    pub use_state: GrantUseState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenyReason {
    InvalidGrant(GrantValidationError),
    EpochStale,
    ContextMismatch,
    Expired,
    UseBudgetExhausted,
    ExplicitlyRevoked,
    ContextRevoked,
    SubjectTombstoned,
    ResourceFrozen,
    ResourceEpochStale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityDecision {
    Allow,
    Deny(DenyReason),
}

/// Evaluate a positive record against current epoch/context, durable use state,
/// expiry, and negative facts.
pub fn evaluate_authority(
    grant: &CapabilityGrant,
    context: &AuthorityEvaluationContext,
    negative_facts: &[NegativeAuthorityFact],
) -> AuthorityDecision {
    if let Err(error) = grant.validate() {
        return AuthorityDecision::Deny(DenyReason::InvalidGrant(error));
    }
    if grant.authority_epoch != context.current_epoch {
        return AuthorityDecision::Deny(DenyReason::EpochStale);
    }
    if grant.authority_context != context.current_authority_context {
        return AuthorityDecision::Deny(DenyReason::ContextMismatch);
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
            NegativeAuthorityFact::RevokeContext { context: revoked }
                if *revoked == grant.authority_context =>
            {
                return AuthorityDecision::Deny(DenyReason::ContextRevoked);
            }
            NegativeAuthorityFact::TombstonePrincipal { principal }
                if *principal == grant.subject
                    || *principal == grant.issuer
                    || grant.audience.as_ref() == Some(principal) =>
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

/// Runtime facts that fence a successful authority evaluation to one process/session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAdmissionContext {
    pub runtime_epoch: RuntimeEpoch,
    pub admitted_at_tick: u64,
    pub authority: AuthorityEvaluationContext,
}

/// Serializable evidence that a grant record passed admission at one exact runtime state.
///
/// This receipt is audit data only. It is **not** a live capability and cannot be
/// converted back into [`LiveGrant`] through a public constructor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionReceipt {
    pub grant_digest: Digest32,
    pub runtime_epoch: RuntimeEpoch,
    pub admitted_at_tick: u64,
    pub authority_epoch: AuthorityEpoch,
    pub authority_context: AuthorityContextRef,
}

/// Currently admitted positive authority.
///
/// `LiveGrant` is intentionally neither `Clone` nor serializable. Its fields are
/// private so persisted grant/receipt bytes cannot reconstruct live authority.
#[derive(Debug)]
pub struct LiveGrant {
    grant: CapabilityGrant,
    receipt: AdmissionReceipt,
}

impl LiveGrant {
    pub fn grant(&self) -> &CapabilityGrant {
        &self.grant
    }

    pub fn receipt(&self) -> &AdmissionReceipt {
        &self.receipt
    }

    /// Verify that this in-memory admission still belongs to the current runtime
    /// and authority state. Higher layers should call this immediately before
    /// effect-specific admission or reservation.
    pub fn validate_current(
        &self,
        runtime_epoch: RuntimeEpoch,
        authority_epoch: AuthorityEpoch,
        authority_context: &AuthorityContextRef,
    ) -> Result<(), LiveGrantStaleReason> {
        if runtime_epoch != self.receipt.runtime_epoch {
            return Err(LiveGrantStaleReason::RuntimeEpochChanged);
        }
        if authority_epoch != self.receipt.authority_epoch {
            return Err(LiveGrantStaleReason::AuthorityEpochChanged);
        }
        if authority_context != &self.receipt.authority_context {
            return Err(LiveGrantStaleReason::AuthorityContextChanged);
        }
        if self.grant.digest() != self.receipt.grant_digest {
            return Err(LiveGrantStaleReason::GrantCommitmentChanged);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LiveGrantStaleReason {
    #[error("runtime epoch changed")]
    RuntimeEpochChanged,
    #[error("authority epoch changed")]
    AuthorityEpochChanged,
    #[error("authority context changed")]
    AuthorityContextChanged,
    #[error("grant commitment no longer matches admission receipt")]
    GrantCommitmentChanged,
}

/// Admit one serialized grant record into runtime-local live authority.
///
/// This does not execute an effect and does not reserve a use. Consequential
/// consumers remain responsible for durable reservation and domain safety gates.
pub fn admit_live_grant(
    grant: CapabilityGrant,
    context: &RuntimeAdmissionContext,
    negative_facts: &[NegativeAuthorityFact],
) -> Result<LiveGrant, DenyReason> {
    match evaluate_authority(&grant, &context.authority, negative_facts) {
        AuthorityDecision::Allow => {
            let receipt = AdmissionReceipt {
                grant_digest: grant.digest(),
                runtime_epoch: context.runtime_epoch,
                admitted_at_tick: context.admitted_at_tick,
                authority_epoch: context.authority.current_epoch,
                authority_context: context.authority.current_authority_context.clone(),
            };
            Ok(LiveGrant { grant, receipt })
        }
        AuthorityDecision::Deny(reason) => Err(reason),
    }
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

    fn admission_context() -> RuntimeAdmissionContext {
        RuntimeAdmissionContext {
            runtime_epoch: RuntimeEpoch(7),
            admitted_at_tick: 150,
            authority: AuthorityEvaluationContext {
                now_unix_s: 100,
                current_epoch: AuthorityEpoch(5),
                current_authority_context: authority_context(1),
                use_state: GrantUseState::default(),
            },
        }
    }

    #[test]
    fn auth_tiny_001_exact_grant_admits() {
        let grant = robot_grant();
        let live = admit_live_grant(grant.clone(), &admission_context(), &[]).unwrap();
        assert_eq!(live.grant(), &grant);
        assert_eq!(live.receipt().runtime_epoch, RuntimeEpoch(7));
        assert_eq!(live.receipt().grant_digest, grant.digest());
    }

    #[test]
    fn context_mismatch_fails_closed() {
        let grant = robot_grant();
        let mut context = admission_context();
        context.authority.current_authority_context = authority_context(2);
        assert_eq!(
            admit_live_grant(grant, &context, &[]).unwrap_err(),
            DenyReason::ContextMismatch
        );
    }

    #[test]
    fn live_authority_is_runtime_epoch_bound() {
        let context = admission_context();
        let authority_ref = context.authority.current_authority_context.clone();
        let live = admit_live_grant(robot_grant(), &context, &[]).unwrap();
        assert_eq!(
            live.validate_current(RuntimeEpoch(8), AuthorityEpoch(5), &authority_ref),
            Err(LiveGrantStaleReason::RuntimeEpochChanged)
        );
        assert!(
            live.validate_current(RuntimeEpoch(7), AuthorityEpoch(5), &authority_ref)
                .is_ok()
        );
    }

    #[test]
    fn revocation_dominates_otherwise_valid_grant() {
        let grant = robot_grant();
        let negative = NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        };
        assert_eq!(
            admit_live_grant(grant, &admission_context(), &[negative]).unwrap_err(),
            DenyReason::ExplicitlyRevoked
        );
    }

    #[test]
    fn reserved_uses_are_charged_before_dispatch() {
        let grant = robot_grant();
        let mut context = admission_context();
        context.authority.use_state = GrantUseState {
            committed: 1,
            reserved: 1,
        };
        assert_eq!(
            admit_live_grant(grant, &context, &[]).unwrap_err(),
            DenyReason::UseBudgetExhausted
        );
    }

    #[test]
    fn stale_authority_epoch_is_denied() {
        let grant = robot_grant();
        let mut context = admission_context();
        context.authority.current_epoch = AuthorityEpoch(6);
        assert_eq!(
            admit_live_grant(grant, &context, &[]).unwrap_err(),
            DenyReason::EpochStale
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
    fn structural_invalidity_fails_before_positive_authority() {
        let mut grant = robot_grant();
        grant.operations.clear();
        assert_eq!(
            admit_live_grant(grant, &admission_context(), &[]).unwrap_err(),
            DenyReason::InvalidGrant(GrantValidationError::EmptyOperations)
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
