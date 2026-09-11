// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded, revocable IT execution authority for support operations.
//!
//! This module applies the authority theorem from the broader control architecture
//! to IT support without claiming that authorization is itself a safety proof.
//!
//! ```text
//! proposed action != authorized action
//! low-risk action != authorized action
//! diagnostic value != permission to intervene
//! authorization record != live runtime capability
//! capability possession != perpetual permission
//! capability revalidation != execution evidence
//! ```
//!
//! Runtime capabilities and permits intentionally do not implement Serde or Clone.
//! Persisted decision records are audit material only and cannot recreate authority.

use crate::system_state::{
    CurrentnessStatusV1, EntityId, ObservationId, StateValueV1, SystemStateGraphV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ItActionKindV1 {
    ReadOnlyDiagnostic,
    ActiveDiagnostic,
    RestartService,
    ClearCache,
    UpdateConfig,
    Custom(String),
}

impl ItActionKindV1 {
    pub fn is_mutating_or_active(&self) -> bool {
        !matches!(self, Self::ReadOnlyDiagnostic)
    }

    fn validate(&self) -> Result<(), ItAuthorityErrorV1> {
        if let Self::Custom(name) = self {
            require_nonempty(name, "custom action kind")?;
        }
        Ok(())
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum ItOperationalRiskV1 {
    Passive,
    Low,
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ItAuthorityEvidenceRequirementV1 {
    AnyRecorded,
    FreshOnly,
    FreshOrIndeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItActionRequestV1 {
    pub request_id: String,
    pub subject: EntityId,
    pub action: ItActionKindV1,
    pub risk: ItOperationalRiskV1,
    #[serde(default)]
    pub parameters: BTreeMap<String, StateValueV1>,
    /// Exact live-state revision the proposal was formed against.
    pub graph_revision: u64,
    /// Evidence that must remain valid for authorization/revalidation.
    #[serde(default)]
    pub required_evidence: BTreeSet<ObservationId>,
    /// Digest of the exact impact/blast-radius analysis when policy requires it.
    pub impact_analysis_digest: Option<String>,
    /// Digest of the exact rollback/recovery plan when policy requires it.
    pub rollback_plan_digest: Option<String>,
    pub principal: String,
    pub requested_at_unix_ms: u64,
}

impl ItActionRequestV1 {
    pub fn validate(&self) -> Result<(), ItAuthorityErrorV1> {
        require_nonempty(&self.request_id, "action request id")?;
        require_nonempty(&self.subject.0, "action subject")?;
        require_nonempty(&self.principal, "authorization principal")?;
        self.action.validate()?;
        validate_optional_nonempty(
            self.impact_analysis_digest.as_deref(),
            "impact analysis digest",
        )?;
        validate_optional_nonempty(
            self.rollback_plan_digest.as_deref(),
            "rollback plan digest",
        )?;
        for (key, value) in &self.parameters {
            require_nonempty(key, "action parameter key")?;
            validate_state_value(value, "action parameter")?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ItAuthorityErrorV1> {
        self.validate()?;
        digest_serializable("symthaea-it-action-request-v1", self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItAuthorizationPolicyV1 {
    pub policy_id: String,
    pub generation: u64,
    #[serde(default)]
    pub allowed_actions: BTreeSet<ItActionKindV1>,
    #[serde(default)]
    pub allowed_principals: BTreeSet<String>,
    pub max_risk: ItOperationalRiskV1,
    pub evidence_requirement: ItAuthorityEvidenceRequirementV1,
    pub require_evidence_for_active_actions: bool,
    pub require_impact_for_active_actions: bool,
    pub require_rollback_for_active_actions: bool,
    pub capability_ttl_ms: u64,
    pub handoff_window_ms: u64,
}

impl ItAuthorizationPolicyV1 {
    pub fn validate(&self) -> Result<(), ItAuthorityErrorV1> {
        require_nonempty(&self.policy_id, "authorization policy id")?;
        if self.generation == 0 {
            return Err(ItAuthorityErrorV1::InvalidPolicy(
                "policy generation must be non-zero".into(),
            ));
        }
        if self.allowed_actions.is_empty() {
            return Err(ItAuthorityErrorV1::InvalidPolicy(
                "policy must allow at least one exact action kind".into(),
            ));
        }
        if self.allowed_principals.is_empty() {
            return Err(ItAuthorityErrorV1::InvalidPolicy(
                "policy must allow at least one principal".into(),
            ));
        }
        for action in &self.allowed_actions {
            action.validate()?;
        }
        for principal in &self.allowed_principals {
            require_nonempty(principal, "allowed principal")?;
        }
        if self.capability_ttl_ms == 0 {
            return Err(ItAuthorityErrorV1::InvalidPolicy(
                "capability TTL must be non-zero".into(),
            ));
        }
        if self.handoff_window_ms == 0 || self.handoff_window_ms > self.capability_ttl_ms {
            return Err(ItAuthorityErrorV1::InvalidPolicy(
                "handoff window must be non-zero and no larger than capability TTL".into(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ItAuthorityErrorV1> {
        self.validate()?;
        digest_serializable("symthaea-it-authorization-policy-v1", self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ItAuthorizationDenialV1 {
    InvalidRequest(String),
    UnknownSubject(EntityId),
    StaleGraphRevision {
        request_revision: u64,
        current_revision: u64,
    },
    PrincipalNotAllowed(String),
    ActionNotAllowed(ItActionKindV1),
    RiskAbovePolicy {
        requested: ItOperationalRiskV1,
        maximum: ItOperationalRiskV1,
    },
    MissingRequiredEvidence,
    UnknownEvidence(ObservationId),
    EvidenceNotCurrent {
        observation: ObservationId,
        currentness: CurrentnessStatusV1,
    },
    MissingImpactAnalysis,
    MissingRollbackPlan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItAuthorizationDecisionRecordV1 {
    pub decision_id: String,
    pub request_id: String,
    pub request_digest: String,
    pub policy_id: String,
    pub policy_generation: u64,
    pub policy_digest: String,
    pub revocation_generation: u64,
    pub decided_at_unix_ms: u64,
    pub granted: bool,
    #[serde(default)]
    pub denials: Vec<ItAuthorizationDenialV1>,
}

/// Live runtime authority. This type is intentionally non-Serde and non-Clone.
#[derive(Debug)]
pub struct AuthorizedItActionCapabilityV1 {
    decision_id: String,
    request_digest: String,
    subject: EntityId,
    action: ItActionKindV1,
    graph_revision: u64,
    policy_id: String,
    policy_generation: u64,
    revocation_generation: u64,
    expires_at_unix_ms: u64,
    consumed: bool,
    _seal: CapabilitySeal,
}

impl AuthorizedItActionCapabilityV1 {
    pub fn decision_id(&self) -> &str {
        &self.decision_id
    }

    pub fn request_digest(&self) -> &str {
        &self.request_digest
    }

    pub fn subject(&self) -> &EntityId {
        &self.subject
    }

    pub fn action(&self) -> &ItActionKindV1 {
        &self.action
    }

    pub fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }

    pub fn is_consumed(&self) -> bool {
        self.consumed
    }
}

#[derive(Debug)]
struct CapabilitySeal;

#[derive(Debug)]
pub struct ItAuthorizationOutcomeV1 {
    pub record: ItAuthorizationDecisionRecordV1,
    pub capability: Option<AuthorizedItActionCapabilityV1>,
}

/// A successfully revalidated capability. Non-Serde/non-Clone and short-lived.
#[derive(Debug)]
pub struct RevalidatedItExecutionPermitV1 {
    decision_id: String,
    request_digest: String,
    subject: EntityId,
    action: ItActionKindV1,
    graph_revision: u64,
    policy_id: String,
    policy_generation: u64,
    revocation_generation: u64,
    handoff_deadline_unix_ms: u64,
    _seal: CapabilitySeal,
}

impl RevalidatedItExecutionPermitV1 {
    pub fn handoff_deadline_unix_ms(&self) -> u64 {
        self.handoff_deadline_unix_ms
    }

    /// Consume the permit into the exact command-boundary authority. Moving `self`
    /// enforces one-use semantics in safe Rust.
    pub fn into_command_authority(
        self,
        manager: &ItAuthorityManagerV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ItCommandAuthorityV1, ItAuthorityErrorV1> {
        manager.validate_handoff_context(&self, graph, now_unix_ms)?;
        Ok(ItCommandAuthorityV1 {
            decision_id: self.decision_id,
            request_digest: self.request_digest,
            subject: self.subject,
            action: self.action,
            graph_revision: self.graph_revision,
            policy_id: self.policy_id,
            policy_generation: self.policy_generation,
            revocation_generation: self.revocation_generation,
            _seal: CapabilitySeal,
        })
    }
}

/// Final non-Serde/non-Clone authority object intended to be consumed by an
/// executor/controller boundary. It is not execution evidence.
#[derive(Debug)]
pub struct ItCommandAuthorityV1 {
    decision_id: String,
    request_digest: String,
    subject: EntityId,
    action: ItActionKindV1,
    graph_revision: u64,
    policy_id: String,
    policy_generation: u64,
    revocation_generation: u64,
    _seal: CapabilitySeal,
}

impl ItCommandAuthorityV1 {
    pub fn decision_id(&self) -> &str {
        &self.decision_id
    }

    pub fn request_digest(&self) -> &str {
        &self.request_digest
    }

    pub fn subject(&self) -> &EntityId {
        &self.subject
    }

    pub fn action(&self) -> &ItActionKindV1 {
        &self.action
    }

    pub fn graph_revision(&self) -> u64 {
        self.graph_revision
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub fn policy_generation(&self) -> u64 {
        self.policy_generation
    }

    pub fn revocation_generation(&self) -> u64 {
        self.revocation_generation
    }
}

#[derive(Debug)]
pub struct ItAuthorityManagerV1 {
    policy: ItAuthorizationPolicyV1,
    revocation_generation: u64,
    next_decision_generation: u64,
}

impl ItAuthorityManagerV1 {
    pub fn new(policy: ItAuthorizationPolicyV1) -> Result<Self, ItAuthorityErrorV1> {
        policy.validate()?;
        Ok(Self {
            policy,
            revocation_generation: 1,
            next_decision_generation: 1,
        })
    }

    pub fn policy(&self) -> &ItAuthorizationPolicyV1 {
        &self.policy
    }

    pub fn revocation_generation(&self) -> u64 {
        self.revocation_generation
    }

    /// Replace policy only with a strictly newer generation. Existing capabilities
    /// become unusable because their policy generation no longer matches.
    pub fn replace_policy(
        &mut self,
        policy: ItAuthorizationPolicyV1,
    ) -> Result<(), ItAuthorityErrorV1> {
        policy.validate()?;
        if policy.policy_id != self.policy.policy_id {
            return Err(ItAuthorityErrorV1::PolicyIdentityChanged {
                current: self.policy.policy_id.clone(),
                proposed: policy.policy_id,
            });
        }
        if policy.generation <= self.policy.generation {
            return Err(ItAuthorityErrorV1::PolicyGenerationNotAdvanced {
                current: self.policy.generation,
                proposed: policy.generation,
            });
        }
        self.policy = policy;
        Ok(())
    }

    /// Revoke all outstanding runtime capabilities/permits issued under the prior
    /// revocation generation.
    pub fn revoke_all(&mut self) {
        self.revocation_generation = self.revocation_generation.saturating_add(1);
    }

    pub fn authorize(
        &mut self,
        request: &ItActionRequestV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ItAuthorizationOutcomeV1, ItAuthorityErrorV1> {
        let request_digest = request.digest()?;
        let policy_digest = self.policy.digest()?;
        let denials = self.evaluate_request(request, graph, now_unix_ms);
        let decision_generation = self.next_decision_generation;
        self.next_decision_generation = self.next_decision_generation.saturating_add(1);

        let decision_id = digest_serializable(
            "symthaea-it-authorization-decision-v1",
            &(
                &request_digest,
                &policy_digest,
                self.revocation_generation,
                decision_generation,
                now_unix_ms,
            ),
        )?;

        let record = ItAuthorizationDecisionRecordV1 {
            decision_id: decision_id.clone(),
            request_id: request.request_id.clone(),
            request_digest: request_digest.clone(),
            policy_id: self.policy.policy_id.clone(),
            policy_generation: self.policy.generation,
            policy_digest,
            revocation_generation: self.revocation_generation,
            decided_at_unix_ms: now_unix_ms,
            granted: denials.is_empty(),
            denials,
        };

        if !record.granted {
            return Ok(ItAuthorizationOutcomeV1 {
                record,
                capability: None,
            });
        }

        let expires_at_unix_ms = now_unix_ms
            .checked_add(self.policy.capability_ttl_ms)
            .ok_or(ItAuthorityErrorV1::TimeOverflow)?;
        let capability = AuthorizedItActionCapabilityV1 {
            decision_id,
            request_digest,
            subject: request.subject.clone(),
            action: request.action.clone(),
            graph_revision: request.graph_revision,
            policy_id: self.policy.policy_id.clone(),
            policy_generation: self.policy.generation,
            revocation_generation: self.revocation_generation,
            expires_at_unix_ms,
            consumed: false,
            _seal: CapabilitySeal,
        };

        Ok(ItAuthorizationOutcomeV1 {
            record,
            capability: Some(capability),
        })
    }

    /// Revalidate exact prerequisites and consume the live capability. Failed
    /// revalidation leaves it unconsumed so callers can distinguish revocation,
    /// staleness, expiry, or request drift in audit logic.
    pub fn revalidate_and_consume(
        &self,
        capability: &mut AuthorizedItActionCapabilityV1,
        request: &ItActionRequestV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<RevalidatedItExecutionPermitV1, ItAuthorityErrorV1> {
        if capability.consumed {
            return Err(ItAuthorityErrorV1::CapabilityAlreadyConsumed);
        }
        if capability.policy_id != self.policy.policy_id
            || capability.policy_generation != self.policy.generation
        {
            return Err(ItAuthorityErrorV1::PolicyChanged);
        }
        if capability.revocation_generation != self.revocation_generation {
            return Err(ItAuthorityErrorV1::CapabilityRevoked);
        }
        if now_unix_ms > capability.expires_at_unix_ms {
            return Err(ItAuthorityErrorV1::CapabilityExpired);
        }

        let request_digest = request.digest()?;
        if request_digest != capability.request_digest {
            return Err(ItAuthorityErrorV1::RequestDigestMismatch);
        }
        if request.subject != capability.subject || request.action != capability.action {
            return Err(ItAuthorityErrorV1::RequestDigestMismatch);
        }

        let denials = self.evaluate_request(request, graph, now_unix_ms);
        if !denials.is_empty() {
            return Err(ItAuthorityErrorV1::RevalidationDenied(denials));
        }
        if graph.revision != capability.graph_revision {
            return Err(ItAuthorityErrorV1::GraphRevisionChanged {
                authorized: capability.graph_revision,
                current: graph.revision,
            });
        }

        let handoff_deadline_unix_ms = now_unix_ms
            .checked_add(self.policy.handoff_window_ms)
            .ok_or(ItAuthorityErrorV1::TimeOverflow)?
            .min(capability.expires_at_unix_ms);

        capability.consumed = true;
        Ok(RevalidatedItExecutionPermitV1 {
            decision_id: capability.decision_id.clone(),
            request_digest: capability.request_digest.clone(),
            subject: capability.subject.clone(),
            action: capability.action.clone(),
            graph_revision: capability.graph_revision,
            policy_id: capability.policy_id.clone(),
            policy_generation: capability.policy_generation,
            revocation_generation: capability.revocation_generation,
            handoff_deadline_unix_ms,
            _seal: CapabilitySeal,
        })
    }

    fn validate_handoff_context(
        &self,
        permit: &RevalidatedItExecutionPermitV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<(), ItAuthorityErrorV1> {
        if permit.policy_id != self.policy.policy_id
            || permit.policy_generation != self.policy.generation
        {
            return Err(ItAuthorityErrorV1::PolicyChanged);
        }
        if permit.revocation_generation != self.revocation_generation {
            return Err(ItAuthorityErrorV1::CapabilityRevoked);
        }
        if now_unix_ms > permit.handoff_deadline_unix_ms {
            return Err(ItAuthorityErrorV1::HandoffWindowExpired);
        }
        if graph.revision != permit.graph_revision {
            return Err(ItAuthorityErrorV1::GraphRevisionChanged {
                authorized: permit.graph_revision,
                current: graph.revision,
            });
        }
        Ok(())
    }

    fn evaluate_request(
        &self,
        request: &ItActionRequestV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Vec<ItAuthorizationDenialV1> {
        let mut denials = Vec::new();
        if let Err(err) = request.validate() {
            denials.push(ItAuthorizationDenialV1::InvalidRequest(err.to_string()));
            return denials;
        }
        if graph.entity(&request.subject).is_none() {
            denials.push(ItAuthorizationDenialV1::UnknownSubject(
                request.subject.clone(),
            ));
        }
        if request.graph_revision != graph.revision {
            denials.push(ItAuthorizationDenialV1::StaleGraphRevision {
                request_revision: request.graph_revision,
                current_revision: graph.revision,
            });
        }
        if !self.policy.allowed_principals.contains(&request.principal) {
            denials.push(ItAuthorizationDenialV1::PrincipalNotAllowed(
                request.principal.clone(),
            ));
        }
        if !self.policy.allowed_actions.contains(&request.action) {
            denials.push(ItAuthorizationDenialV1::ActionNotAllowed(
                request.action.clone(),
            ));
        }
        if request.risk > self.policy.max_risk {
            denials.push(ItAuthorizationDenialV1::RiskAbovePolicy {
                requested: request.risk,
                maximum: self.policy.max_risk,
            });
        }

        let active = request.action.is_mutating_or_active();
        if active && self.policy.require_evidence_for_active_actions && request.required_evidence.is_empty()
        {
            denials.push(ItAuthorizationDenialV1::MissingRequiredEvidence);
        }
        for evidence_id in &request.required_evidence {
            let Some(observation) = graph.observation(evidence_id) else {
                denials.push(ItAuthorizationDenialV1::UnknownEvidence(
                    evidence_id.clone(),
                ));
                continue;
            };
            let currentness = observation.clock.currentness_at(now_unix_ms);
            let allowed = match self.policy.evidence_requirement {
                ItAuthorityEvidenceRequirementV1::AnyRecorded => true,
                ItAuthorityEvidenceRequirementV1::FreshOnly => {
                    currentness == CurrentnessStatusV1::Fresh
                }
                ItAuthorityEvidenceRequirementV1::FreshOrIndeterminate => {
                    currentness != CurrentnessStatusV1::Stale
                }
            };
            if !allowed {
                denials.push(ItAuthorizationDenialV1::EvidenceNotCurrent {
                    observation: evidence_id.clone(),
                    currentness,
                });
            }
        }
        if active
            && self.policy.require_impact_for_active_actions
            && request.impact_analysis_digest.is_none()
        {
            denials.push(ItAuthorizationDenialV1::MissingImpactAnalysis);
        }
        if active
            && self.policy.require_rollback_for_active_actions
            && request.rollback_plan_digest.is_none()
        {
            denials.push(ItAuthorizationDenialV1::MissingRollbackPlan);
        }
        denials
    }
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, ItAuthorityErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| ItAuthorityErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), ItAuthorityErrorV1> {
    if value.trim().is_empty() {
        Err(ItAuthorityErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_optional_nonempty(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), ItAuthorityErrorV1> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        Err(ItAuthorityErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_state_value(
    value: &StateValueV1,
    field: &'static str,
) -> Result<(), ItAuthorityErrorV1> {
    if let StateValueV1::F64(value) = value {
        if !value.is_finite() {
            return Err(ItAuthorityErrorV1::InvalidNumericField(field));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum ItAuthorityErrorV1 {
    EmptyField(&'static str),
    InvalidNumericField(&'static str),
    InvalidPolicy(String),
    Serialization(String),
    TimeOverflow,
    PolicyIdentityChanged { current: String, proposed: String },
    PolicyGenerationNotAdvanced { current: u64, proposed: u64 },
    CapabilityAlreadyConsumed,
    CapabilityRevoked,
    CapabilityExpired,
    PolicyChanged,
    RequestDigestMismatch,
    RevalidationDenied(Vec<ItAuthorizationDenialV1>),
    GraphRevisionChanged { authorized: u64, current: u64 },
    HandoffWindowExpired,
}

impl fmt::Display for ItAuthorityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty IT authority field {field}"),
            Self::InvalidNumericField(field) => write!(f, "non-finite IT authority field {field}"),
            Self::InvalidPolicy(message) => write!(f, "invalid IT authorization policy: {message}"),
            Self::Serialization(message) => write!(f, "IT authority serialization failed: {message}"),
            Self::TimeOverflow => write!(f, "IT authority validity time overflow"),
            Self::PolicyIdentityChanged { current, proposed } => write!(
                f,
                "authorization policy identity cannot change in-place: {current} -> {proposed}"
            ),
            Self::PolicyGenerationNotAdvanced { current, proposed } => write!(
                f,
                "authorization policy generation must advance beyond {current}; got {proposed}"
            ),
            Self::CapabilityAlreadyConsumed => write!(f, "IT action capability already consumed"),
            Self::CapabilityRevoked => write!(f, "IT action capability/permit was revoked"),
            Self::CapabilityExpired => write!(f, "IT action capability expired"),
            Self::PolicyChanged => write!(f, "authorization policy changed since capability issuance"),
            Self::RequestDigestMismatch => write!(f, "action request no longer matches authorized digest"),
            Self::RevalidationDenied(denials) => {
                write!(f, "IT action revalidation denied by {} prerequisite(s)", denials.len())
            }
            Self::GraphRevisionChanged { authorized, current } => write!(
                f,
                "live state revision changed since authorization: {authorized} -> {current}"
            ),
            Self::HandoffWindowExpired => write!(f, "revalidated command handoff window expired"),
        }
    }
}

impl Error for ItAuthorityErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_state::{
        EntityKindV1, ObservationClockV1, ObservationProvenanceV1, ObservationSourceKindV1,
        SystemObservationV1,
    };

    fn fixture_graph(now: u64) -> (SystemStateGraphV1, EntityId, ObservationId) {
        let subject = EntityId("service:dns".into());
        let evidence = ObservationId("obs:fresh".into());
        let mut graph = SystemStateGraphV1::new();
        graph
            .record_observation(SystemObservationV1 {
                id: evidence.clone(),
                subject: subject.clone(),
                provenance: ObservationProvenanceV1 {
                    source_id: "fixture".into(),
                    source_kind: ObservationSourceKindV1::SyntheticTest,
                    collector: "authority-test".into(),
                    collector_version: Some("1".into()),
                    schema_version: None,
                    artifact_digest: None,
                },
                clock: ObservationClockV1 {
                    event_time_unix_ms: Some(now),
                    observed_at_unix_ms: now,
                    ingested_at_unix_ms: Some(now),
                    max_age_ms: Some(10_000),
                    clock_uncertainty_ms: Some(1),
                },
                confidence: 1.0,
                facts: BTreeMap::new(),
            })
            .unwrap();
        graph
            .upsert_entity(
                subject.clone(),
                EntityKindV1::Service,
                BTreeMap::new(),
                &evidence,
            )
            .unwrap();
        (graph, subject, evidence)
    }

    fn policy() -> ItAuthorizationPolicyV1 {
        ItAuthorizationPolicyV1 {
            policy_id: "support-prod-v1".into(),
            generation: 1,
            allowed_actions: BTreeSet::from([
                ItActionKindV1::ReadOnlyDiagnostic,
                ItActionKindV1::RestartService,
            ]),
            allowed_principals: BTreeSet::from(["operator:alice".into()]),
            max_risk: ItOperationalRiskV1::Moderate,
            evidence_requirement: ItAuthorityEvidenceRequirementV1::FreshOnly,
            require_evidence_for_active_actions: true,
            require_impact_for_active_actions: true,
            require_rollback_for_active_actions: true,
            capability_ttl_ms: 5_000,
            handoff_window_ms: 500,
        }
    }

    fn restart_request(
        graph: &SystemStateGraphV1,
        subject: EntityId,
        evidence: ObservationId,
        now: u64,
    ) -> ItActionRequestV1 {
        ItActionRequestV1 {
            request_id: "req-1".into(),
            subject,
            action: ItActionKindV1::RestartService,
            risk: ItOperationalRiskV1::Moderate,
            parameters: BTreeMap::from([(
                "service".into(),
                StateValueV1::Text("dns".into()),
            )]),
            graph_revision: graph.revision,
            required_evidence: BTreeSet::from([evidence]),
            impact_analysis_digest: Some("impact:abc".into()),
            rollback_plan_digest: Some("rollback:def".into()),
            principal: "operator:alice".into(),
            requested_at_unix_ms: now,
        }
    }

    #[test]
    fn exact_current_request_mints_runtime_capability() {
        let now = 1_000_000;
        let (graph, subject, evidence) = fixture_graph(now);
        let request = restart_request(&graph, subject, evidence, now);
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let outcome = manager.authorize(&request, &graph, now).unwrap();
        assert!(outcome.record.granted);
        assert!(outcome.record.denials.is_empty());
        let capability = outcome.capability.unwrap();
        assert_eq!(capability.subject(), &request.subject);
        assert_eq!(capability.action(), &ItActionKindV1::RestartService);
    }

    #[test]
    fn missing_impact_or_rollback_fails_closed() {
        let now = 1_000_000;
        let (graph, subject, evidence) = fixture_graph(now);
        let mut request = restart_request(&graph, subject, evidence, now);
        request.impact_analysis_digest = None;
        request.rollback_plan_digest = None;
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let outcome = manager.authorize(&request, &graph, now).unwrap();
        assert!(!outcome.record.granted);
        assert!(outcome.capability.is_none());
        assert!(outcome
            .record
            .denials
            .contains(&ItAuthorizationDenialV1::MissingImpactAnalysis));
        assert!(outcome
            .record
            .denials
            .contains(&ItAuthorizationDenialV1::MissingRollbackPlan));
    }

    #[test]
    fn stale_evidence_cannot_satisfy_fresh_policy() {
        let now = 1_000_000;
        let (graph, subject, evidence) = fixture_graph(now);
        let request = restart_request(&graph, subject, evidence.clone(), now);
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let outcome = manager.authorize(&request, &graph, now + 20_000).unwrap();
        assert!(!outcome.record.granted);
        assert!(outcome.record.denials.iter().any(|denial| matches!(
            denial,
            ItAuthorizationDenialV1::EvidenceNotCurrent {
                observation,
                currentness: CurrentnessStatusV1::Stale,
            } if observation == &evidence
        )));
    }

    #[test]
    fn capability_is_one_use_and_revalidated_before_handoff() {
        let now = 1_000_000;
        let (graph, subject, evidence) = fixture_graph(now);
        let request = restart_request(&graph, subject, evidence, now);
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let mut capability = manager
            .authorize(&request, &graph, now)
            .unwrap()
            .capability
            .unwrap();
        let permit = manager
            .revalidate_and_consume(&mut capability, &request, &graph, now + 100)
            .unwrap();
        assert!(capability.is_consumed());
        assert!(matches!(
            manager.revalidate_and_consume(&mut capability, &request, &graph, now + 101),
            Err(ItAuthorityErrorV1::CapabilityAlreadyConsumed)
        ));
        let command_authority = permit
            .into_command_authority(&manager, &graph, now + 200)
            .unwrap();
        assert_eq!(command_authority.subject(), &request.subject);
        assert_eq!(command_authority.action(), &request.action);
    }

    #[test]
    fn revocation_invalidates_capability_before_revalidation() {
        let now = 1_000_000;
        let (graph, subject, evidence) = fixture_graph(now);
        let request = restart_request(&graph, subject, evidence, now);
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let mut capability = manager
            .authorize(&request, &graph, now)
            .unwrap()
            .capability
            .unwrap();
        manager.revoke_all();
        assert!(matches!(
            manager.revalidate_and_consume(&mut capability, &request, &graph, now + 100),
            Err(ItAuthorityErrorV1::CapabilityRevoked)
        ));
    }

    #[test]
    fn graph_change_invalidates_capability() {
        let now = 1_000_000;
        let (mut graph, subject, evidence) = fixture_graph(now);
        let request = restart_request(&graph, subject.clone(), evidence.clone(), now);
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let mut capability = manager
            .authorize(&request, &graph, now)
            .unwrap()
            .capability
            .unwrap();
        graph
            .upsert_entity(
                subject,
                EntityKindV1::Service,
                BTreeMap::from([("status".into(), StateValueV1::Text("changed".into()))]),
                &evidence,
            )
            .unwrap();
        assert!(matches!(
            manager.revalidate_and_consume(&mut capability, &request, &graph, now + 100),
            Err(ItAuthorityErrorV1::RevalidationDenied(_))
                | Err(ItAuthorityErrorV1::GraphRevisionChanged { .. })
        ));
    }

    #[test]
    fn policy_generation_change_invalidates_existing_capability() {
        let now = 1_000_000;
        let (graph, subject, evidence) = fixture_graph(now);
        let request = restart_request(&graph, subject, evidence, now);
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let mut capability = manager
            .authorize(&request, &graph, now)
            .unwrap()
            .capability
            .unwrap();
        let mut next = policy();
        next.generation = 2;
        manager.replace_policy(next).unwrap();
        assert!(matches!(
            manager.revalidate_and_consume(&mut capability, &request, &graph, now + 100),
            Err(ItAuthorityErrorV1::PolicyChanged)
        ));
    }

    #[test]
    fn handoff_window_is_short_and_rechecks_revocation() {
        let now = 1_000_000;
        let (graph, subject, evidence) = fixture_graph(now);
        let request = restart_request(&graph, subject, evidence, now);
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let mut capability = manager
            .authorize(&request, &graph, now)
            .unwrap()
            .capability
            .unwrap();
        let permit = manager
            .revalidate_and_consume(&mut capability, &request, &graph, now + 100)
            .unwrap();
        manager.revoke_all();
        assert!(matches!(
            permit.into_command_authority(&manager, &graph, now + 200),
            Err(ItAuthorityErrorV1::CapabilityRevoked)
        ));
    }

    #[test]
    fn read_only_request_can_be_policy_bounded_without_impact_or_rollback() {
        let now = 1_000_000;
        let (graph, subject, _) = fixture_graph(now);
        let request = ItActionRequestV1 {
            request_id: "read-only".into(),
            subject,
            action: ItActionKindV1::ReadOnlyDiagnostic,
            risk: ItOperationalRiskV1::Passive,
            parameters: BTreeMap::new(),
            graph_revision: graph.revision,
            required_evidence: BTreeSet::new(),
            impact_analysis_digest: None,
            rollback_plan_digest: None,
            principal: "operator:alice".into(),
            requested_at_unix_ms: now,
        };
        let mut manager = ItAuthorityManagerV1::new(policy()).unwrap();
        let outcome = manager.authorize(&request, &graph, now).unwrap();
        assert!(outcome.record.granted);
        assert!(outcome.capability.is_some());
    }
}
