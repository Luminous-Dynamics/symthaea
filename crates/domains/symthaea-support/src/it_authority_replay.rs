// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Durable anti-replay ledger for governed IT authority.
//!
//! The runtime admission manager prevents duplicate grants within one process.
//! This layer persists *non-authority* grant history so restart cannot erase the
//! fact that a request or mutating semantic intent already produced authority.
//!
//! ```text
//! replay ledger != live authority
//! ledger deserialization != capability recreation
//! new request UUID != new mutating intent
//! process restart != replay reset
//! ledger digest verification != trusted storage by itself
//! ```
//!
//! A caller restoring a ledger must supply the expected ledger digest from an
//! independently trusted persistence/audit root. The ledger itself is audit and
//! anti-replay state only; it never recreates a capability.

use crate::it_authority::{
    ItActionRequestV1, ItAuthorizationPolicyV1, ItAuthorityErrorV1, ItCommandAuthorityV1,
};
use crate::it_authority_admission::{
    ItAuthorityAdmissionErrorV1, ItAuthorityAdmissionPolicyV1,
    ReplayProtectedItActionCapabilityV1, ReplayProtectedItAuthorizationOutcomeV1,
    ReplayProtectedItAuthorityManagerV1, ReplayProtectedItExecutionPermitV1,
};
use crate::system_state::SystemStateGraphV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub use crate::it_authority_admission::{
    ReplayProtectedItActionCapabilityV1 as DurableItActionCapabilityV1,
    ReplayProtectedItAuthorizationOutcomeV1 as DurableItAuthorizationOutcomeV1,
    ReplayProtectedItExecutionPermitV1 as DurableItExecutionPermitV1,
};

pub const IT_AUTHORITY_REPLAY_LEDGER_SCHEMA_V1: &str = "symthaea-it-authority-replay-ledger-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItAuthorityReplayGrantV1 {
    pub request_id: String,
    pub request_digest: String,
    /// Semantic action identity excluding request ID and request timestamp.
    pub intent_digest: String,
    pub decision_id: String,
    pub policy_id: String,
    pub policy_generation: u64,
    pub graph_revision: u64,
    pub granted_at_unix_ms: u64,
    pub mutating_or_active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItAuthorityReplayLedgerV1 {
    pub schema_version: String,
    /// Monotonic append generation. V1 requires this to equal grant count.
    pub generation: u64,
    #[serde(default)]
    pub grants: Vec<ItAuthorityReplayGrantV1>,
}

impl Default for ItAuthorityReplayLedgerV1 {
    fn default() -> Self {
        Self {
            schema_version: IT_AUTHORITY_REPLAY_LEDGER_SCHEMA_V1.into(),
            generation: 0,
            grants: Vec::new(),
        }
    }
}

impl ItAuthorityReplayLedgerV1 {
    pub fn validate(&self) -> Result<(), ItAuthorityReplayErrorV1> {
        if self.schema_version != IT_AUTHORITY_REPLAY_LEDGER_SCHEMA_V1 {
            return Err(ItAuthorityReplayErrorV1::UnsupportedLedgerSchema(
                self.schema_version.clone(),
            ));
        }
        if self.generation != self.grants.len() as u64 {
            return Err(ItAuthorityReplayErrorV1::LedgerGenerationMismatch {
                generation: self.generation,
                grants: self.grants.len(),
            });
        }

        let mut request_ids = BTreeSet::new();
        let mut request_digests = BTreeSet::new();
        let mut decision_ids = BTreeSet::new();
        let mut mutating_intents = BTreeSet::new();
        for grant in &self.grants {
            require_nonempty(&grant.request_id, "replay request id")?;
            require_digest(&grant.request_digest, "replay request digest")?;
            require_digest(&grant.intent_digest, "replay intent digest")?;
            require_digest(&grant.decision_id, "replay decision id")?;
            require_nonempty(&grant.policy_id, "replay policy id")?;
            if grant.policy_generation == 0 {
                return Err(ItAuthorityReplayErrorV1::InvalidGrant(
                    "policy generation must be non-zero".into(),
                ));
            }
            if !request_ids.insert(grant.request_id.as_str()) {
                return Err(ItAuthorityReplayErrorV1::DuplicateRequestId(
                    grant.request_id.clone(),
                ));
            }
            if !request_digests.insert(grant.request_digest.as_str()) {
                return Err(ItAuthorityReplayErrorV1::DuplicateRequestDigest(
                    grant.request_digest.clone(),
                ));
            }
            if !decision_ids.insert(grant.decision_id.as_str()) {
                return Err(ItAuthorityReplayErrorV1::DuplicateDecisionId(
                    grant.decision_id.clone(),
                ));
            }
            if grant.mutating_or_active && !mutating_intents.insert(grant.intent_digest.as_str()) {
                return Err(ItAuthorityReplayErrorV1::DuplicateMutatingIntent(
                    grant.intent_digest.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ItAuthorityReplayErrorV1> {
        self.validate()?;
        digest_serializable("symthaea-it-authority-replay-ledger-v1", self)
    }

    fn contains_request_id(&self, request_id: &str) -> bool {
        self.grants.iter().any(|grant| grant.request_id == request_id)
    }

    fn contains_request_digest(&self, request_digest: &str) -> bool {
        self.grants
            .iter()
            .any(|grant| grant.request_digest == request_digest)
    }

    fn contains_mutating_intent(&self, intent_digest: &str) -> bool {
        self.grants.iter().any(|grant| {
            grant.mutating_or_active && grant.intent_digest == intent_digest
        })
    }

    fn append_grant(
        &mut self,
        request: &ItActionRequestV1,
        request_digest: String,
        intent_digest: String,
        record: &crate::it_authority::ItAuthorizationDecisionRecordV1,
    ) -> Result<(), ItAuthorityReplayErrorV1> {
        if !record.granted {
            return Err(ItAuthorityReplayErrorV1::InvalidGrant(
                "cannot append a denied authorization decision".into(),
            ));
        }
        let grant = ItAuthorityReplayGrantV1 {
            request_id: request.request_id.clone(),
            request_digest,
            intent_digest,
            decision_id: record.decision_id.clone(),
            policy_id: record.policy_id.clone(),
            policy_generation: record.policy_generation,
            graph_revision: request.graph_revision,
            granted_at_unix_ms: record.decided_at_unix_ms,
            mutating_or_active: request.action.is_mutating_or_active(),
        };
        self.grants.push(grant);
        self.generation = self.generation.saturating_add(1);
        self.validate()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct DurableReplayProtectedItAuthorityManagerV1 {
    inner: ReplayProtectedItAuthorityManagerV1,
    ledger: ItAuthorityReplayLedgerV1,
}

impl DurableReplayProtectedItAuthorityManagerV1 {
    pub fn new(
        policy: ItAuthorizationPolicyV1,
        admission_policy: ItAuthorityAdmissionPolicyV1,
    ) -> Result<Self, ItAuthorityReplayErrorV1> {
        Ok(Self {
            inner: ReplayProtectedItAuthorityManagerV1::new(policy, admission_policy)?,
            ledger: ItAuthorityReplayLedgerV1::default(),
        })
    }

    /// Restore anti-replay history only after matching a separately trusted digest.
    /// Deserializing a ledger alone is intentionally insufficient.
    pub fn restore_verified(
        policy: ItAuthorizationPolicyV1,
        admission_policy: ItAuthorityAdmissionPolicyV1,
        ledger: ItAuthorityReplayLedgerV1,
        expected_ledger_digest: &str,
    ) -> Result<Self, ItAuthorityReplayErrorV1> {
        require_digest(expected_ledger_digest, "expected replay ledger digest")?;
        let actual = ledger.digest()?;
        if actual != expected_ledger_digest {
            return Err(ItAuthorityReplayErrorV1::LedgerDigestMismatch {
                expected: expected_ledger_digest.into(),
                actual,
            });
        }
        Ok(Self {
            inner: ReplayProtectedItAuthorityManagerV1::new(policy, admission_policy)?,
            ledger,
        })
    }

    pub fn policy(&self) -> &ItAuthorizationPolicyV1 {
        self.inner.policy()
    }

    pub fn replace_policy(
        &mut self,
        policy: ItAuthorizationPolicyV1,
    ) -> Result<(), ItAuthorityReplayErrorV1> {
        self.inner.replace_policy(policy)?;
        Ok(())
    }

    pub fn revoke_all(&mut self) {
        self.inner.revoke_all();
    }

    pub fn ledger_snapshot(&self) -> ItAuthorityReplayLedgerV1 {
        self.ledger.clone()
    }

    pub fn ledger_digest(&self) -> Result<String, ItAuthorityReplayErrorV1> {
        self.ledger.digest()
    }

    pub fn authorize_once(
        &mut self,
        request: &ItActionRequestV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ReplayProtectedItAuthorizationOutcomeV1, ItAuthorityReplayErrorV1> {
        let request_digest = request.digest()?;
        let intent_digest = semantic_intent_digest_v1(request)?;

        if self.ledger.contains_request_id(&request.request_id) {
            return Err(ItAuthorityReplayErrorV1::RequestIdAlreadyGranted(
                request.request_id.clone(),
            ));
        }
        if self.ledger.contains_request_digest(&request_digest) {
            return Err(ItAuthorityReplayErrorV1::RequestDigestAlreadyGranted(
                request_digest,
            ));
        }
        if request.action.is_mutating_or_active()
            && self.ledger.contains_mutating_intent(&intent_digest)
        {
            return Err(ItAuthorityReplayErrorV1::MutatingIntentAlreadyGranted(
                intent_digest,
            ));
        }

        let outcome = self.inner.authorize_once(request, graph, now_unix_ms)?;
        if outcome.record.granted {
            self.ledger.append_grant(
                request,
                request_digest,
                intent_digest,
                &outcome.record,
            )?;
        }
        Ok(outcome)
    }

    pub fn revalidate_and_consume(
        &self,
        capability: &mut ReplayProtectedItActionCapabilityV1,
        request: &ItActionRequestV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ReplayProtectedItExecutionPermitV1, ItAuthorityReplayErrorV1> {
        Ok(self
            .inner
            .revalidate_and_consume(capability, request, graph, now_unix_ms)?)
    }

    pub fn into_command_authority(
        &self,
        permit: ReplayProtectedItExecutionPermitV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ItCommandAuthorityV1, ItAuthorityReplayErrorV1> {
        Ok(self
            .inner
            .into_command_authority(permit, graph, now_unix_ms)?)
    }
}

/// Semantic identity for a proposed action excluding request ID and timestamp.
///
/// For mutating/active actions this makes `new UUID + same exact intent` a replay
/// until material prerequisites change (for example graph revision/evidence,
/// parameters, impact analysis, rollback plan, principal, or action).
pub fn semantic_intent_digest_v1(
    request: &ItActionRequestV1,
) -> Result<String, ItAuthorityReplayErrorV1> {
    request.validate()?;
    digest_serializable(
        "symthaea-it-authority-semantic-intent-v1",
        &(
            &request.subject,
            &request.action,
            request.risk,
            &request.parameters,
            request.graph_revision,
            &request.required_evidence,
            &request.impact_analysis_digest,
            &request.rollback_plan_digest,
            &request.principal,
        ),
    )
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, ItAuthorityReplayErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| ItAuthorityReplayErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn require_nonempty(
    value: &str,
    field: &'static str,
) -> Result<(), ItAuthorityReplayErrorV1> {
    if value.trim().is_empty() {
        Err(ItAuthorityReplayErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_digest(
    value: &str,
    field: &'static str,
) -> Result<(), ItAuthorityReplayErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(ItAuthorityReplayErrorV1::InvalidDigest(field))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum ItAuthorityReplayErrorV1 {
    EmptyField(&'static str),
    InvalidDigest(&'static str),
    UnsupportedLedgerSchema(String),
    LedgerGenerationMismatch { generation: u64, grants: usize },
    DuplicateRequestId(String),
    DuplicateRequestDigest(String),
    DuplicateDecisionId(String),
    DuplicateMutatingIntent(String),
    InvalidGrant(String),
    LedgerDigestMismatch { expected: String, actual: String },
    RequestIdAlreadyGranted(String),
    RequestDigestAlreadyGranted(String),
    MutatingIntentAlreadyGranted(String),
    Serialization(String),
    Admission(ItAuthorityAdmissionErrorV1),
    Authority(ItAuthorityErrorV1),
}

impl fmt::Display for ItAuthorityReplayErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty IT authority replay field {field}"),
            Self::InvalidDigest(field) => write!(f, "invalid 32-byte hex digest for {field}"),
            Self::UnsupportedLedgerSchema(schema) => {
                write!(f, "unsupported IT authority replay ledger schema {schema}")
            }
            Self::LedgerGenerationMismatch { generation, grants } => write!(
                f,
                "IT authority replay ledger generation {generation} does not match {grants} grants"
            ),
            Self::DuplicateRequestId(id) => write!(f, "duplicate granted request id {id:?}"),
            Self::DuplicateRequestDigest(digest) => {
                write!(f, "duplicate granted request digest {digest}")
            }
            Self::DuplicateDecisionId(id) => write!(f, "duplicate authorization decision id {id}"),
            Self::DuplicateMutatingIntent(digest) => {
                write!(f, "duplicate mutating semantic intent {digest}")
            }
            Self::InvalidGrant(message) => write!(f, "invalid replay ledger grant: {message}"),
            Self::LedgerDigestMismatch { expected, actual } => write!(
                f,
                "IT authority replay ledger digest mismatch: expected {expected}, got {actual}"
            ),
            Self::RequestIdAlreadyGranted(id) => {
                write!(f, "request id {id:?} has already produced authority")
            }
            Self::RequestDigestAlreadyGranted(digest) => {
                write!(f, "request digest {digest} has already produced authority")
            }
            Self::MutatingIntentAlreadyGranted(digest) => write!(
                f,
                "mutating semantic intent {digest} has already produced authority against these exact prerequisites"
            ),
            Self::Serialization(message) => {
                write!(f, "IT authority replay serialization failed: {message}")
            }
            Self::Admission(error) => write!(f, "IT authority admission error: {error}"),
            Self::Authority(error) => write!(f, "IT authority error: {error}"),
        }
    }
}

impl Error for ItAuthorityReplayErrorV1 {}

impl From<ItAuthorityAdmissionErrorV1> for ItAuthorityReplayErrorV1 {
    fn from(value: ItAuthorityAdmissionErrorV1) -> Self {
        Self::Admission(value)
    }
}

impl From<ItAuthorityErrorV1> for ItAuthorityReplayErrorV1 {
    fn from(value: ItAuthorityErrorV1) -> Self {
        Self::Authority(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EntityId, EntityKindV1, ItActionKindV1, ItAuthorityEvidenceRequirementV1,
        ItOperationalRiskV1, ObservationClockV1, ObservationId, ObservationProvenanceV1,
        ObservationSourceKindV1, StateValueV1, SystemObservationV1,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn fixture(now: u64) -> (SystemStateGraphV1, ItActionRequestV1, ItAuthorizationPolicyV1) {
        let subject = EntityId("service:dns".into());
        let evidence = ObservationId("obs:dns".into());
        let mut graph = SystemStateGraphV1::new();
        graph
            .record_observation(SystemObservationV1 {
                id: evidence.clone(),
                subject: subject.clone(),
                provenance: ObservationProvenanceV1 {
                    source_id: "fixture".into(),
                    source_kind: ObservationSourceKindV1::SyntheticTest,
                    collector: "replay-ledger-test".into(),
                    collector_version: Some("1".into()),
                    schema_version: None,
                    artifact_digest: None,
                },
                clock: ObservationClockV1 {
                    event_time_unix_ms: Some(now),
                    observed_at_unix_ms: now,
                    ingested_at_unix_ms: Some(now),
                    max_age_ms: Some(60_000),
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
        let request = ItActionRequestV1 {
            request_id: "request-1".into(),
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
        };
        let policy = ItAuthorizationPolicyV1 {
            policy_id: "support-prod-v1".into(),
            generation: 1,
            allowed_actions: BTreeSet::from([ItActionKindV1::RestartService]),
            allowed_principals: BTreeSet::from(["operator:alice".into()]),
            max_risk: ItOperationalRiskV1::Moderate,
            evidence_requirement: ItAuthorityEvidenceRequirementV1::FreshOnly,
            require_evidence_for_active_actions: true,
            require_impact_for_active_actions: true,
            require_rollback_for_active_actions: true,
            capability_ttl_ms: 5_000,
            handoff_window_ms: 500,
        };
        (graph, request, policy)
    }

    #[test]
    fn semantic_intent_ignores_uuid_and_timestamp_but_binds_prerequisites() {
        let now = 1_000_000;
        let (_, request, _) = fixture(now);
        let mut same_intent = request.clone();
        same_intent.request_id = "request-2".into();
        same_intent.requested_at_unix_ms = now + 1;
        assert_eq!(
            semantic_intent_digest_v1(&request).unwrap(),
            semantic_intent_digest_v1(&same_intent).unwrap()
        );
        same_intent.graph_revision += 1;
        assert_ne!(
            semantic_intent_digest_v1(&request).unwrap(),
            semantic_intent_digest_v1(&same_intent).unwrap()
        );
    }

    #[test]
    fn new_uuid_cannot_multiply_same_mutating_intent() {
        let now = 1_000_000;
        let (graph, request, policy) = fixture(now);
        let mut manager = DurableReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        assert!(manager
            .authorize_once(&request, &graph, now)
            .unwrap()
            .record
            .granted);
        let mut replay = request.clone();
        replay.request_id = "request-new-uuid".into();
        replay.requested_at_unix_ms = now + 1;
        assert!(matches!(
            manager.authorize_once(&replay, &graph, now + 1),
            Err(ItAuthorityReplayErrorV1::MutatingIntentAlreadyGranted(_))
        ));
    }

    #[test]
    fn verified_ledger_survives_restart_and_blocks_replay() {
        let now = 1_000_000;
        let (graph, request, policy) = fixture(now);
        let admission = ItAuthorityAdmissionPolicyV1::default();
        let mut first = DurableReplayProtectedItAuthorityManagerV1::new(
            policy.clone(),
            admission.clone(),
        )
        .unwrap();
        assert!(first
            .authorize_once(&request, &graph, now)
            .unwrap()
            .record
            .granted);
        let ledger = first.ledger_snapshot();
        let expected_digest = first.ledger_digest().unwrap();

        let mut restored = DurableReplayProtectedItAuthorityManagerV1::restore_verified(
            policy,
            admission,
            ledger,
            &expected_digest,
        )
        .unwrap();
        assert!(matches!(
            restored.authorize_once(&request, &graph, now + 1),
            Err(ItAuthorityReplayErrorV1::RequestIdAlreadyGranted(_))
        ));
    }

    #[test]
    fn tampered_ledger_cannot_restore_under_old_trusted_digest() {
        let now = 1_000_000;
        let (graph, request, policy) = fixture(now);
        let admission = ItAuthorityAdmissionPolicyV1::default();
        let mut first = DurableReplayProtectedItAuthorityManagerV1::new(
            policy.clone(),
            admission.clone(),
        )
        .unwrap();
        first.authorize_once(&request, &graph, now).unwrap();
        let expected_digest = first.ledger_digest().unwrap();
        let mut ledger = first.ledger_snapshot();
        ledger.grants[0].request_id = "tampered".into();
        assert!(matches!(
            DurableReplayProtectedItAuthorityManagerV1::restore_verified(
                policy,
                admission,
                ledger,
                &expected_digest,
            ),
            Err(ItAuthorityReplayErrorV1::LedgerDigestMismatch { .. })
        ));
    }

    #[test]
    fn denied_request_is_not_appended_to_durable_ledger() {
        let now = 1_000_000;
        let (graph, request, mut policy) = fixture(now);
        policy.allowed_principals = BTreeSet::from(["operator:bob".into()]);
        let mut manager = DurableReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        let denied = manager.authorize_once(&request, &graph, now).unwrap();
        assert!(!denied.record.granted);
        assert_eq!(manager.ledger_snapshot().generation, 0);
    }
}
