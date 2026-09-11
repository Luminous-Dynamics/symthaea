// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public replay-protected admission boundary for IT execution authority.
//!
//! `it_authority` owns the exact capability/revalidation mechanics. This module
//! is the public manager boundary and adds request-time validity plus anti-replay
//! admission so one already-granted request cannot mint multiple fresh runtime
//! capabilities.
//!
//! ```text
//! capability one-use != request one-use
//! request replay != fresh authorization request
//! old request timestamp != current authority
//! future-dated request != current authority
//! denied request != consumed authority
//! persisted audit record != replay ticket
//! ```

use crate::it_authority::{
    AuthorizedItActionCapabilityV1, ItActionRequestV1, ItAuthorizationDecisionRecordV1,
    ItAuthorizationPolicyV1, ItAuthorityErrorV1, ItAuthorityManagerV1, ItCommandAuthorityV1,
    RevalidatedItExecutionPermitV1,
};
use crate::system_state::SystemStateGraphV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItAuthorityAdmissionPolicyV1 {
    /// Maximum age of a request at initial authorization admission.
    pub max_request_age_ms: u64,
    /// Maximum tolerated amount by which a request timestamp may appear ahead of
    /// the authority clock. This is clock-skew tolerance, not future authority.
    pub max_future_request_skew_ms: u64,
}

impl ItAuthorityAdmissionPolicyV1 {
    pub fn validate(&self) -> Result<(), ItAuthorityAdmissionErrorV1> {
        if self.max_request_age_ms == 0 {
            return Err(ItAuthorityAdmissionErrorV1::InvalidAdmissionPolicy(
                "maximum request age must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

impl Default for ItAuthorityAdmissionPolicyV1 {
    fn default() -> Self {
        Self {
            max_request_age_ms: 30_000,
            max_future_request_skew_ms: 2_000,
        }
    }
}

/// Public runtime capability wrapper. Intentionally non-Serde and non-Clone.
#[derive(Debug)]
pub struct ReplayProtectedItActionCapabilityV1 {
    inner: AuthorizedItActionCapabilityV1,
}

impl ReplayProtectedItActionCapabilityV1 {
    pub fn decision_id(&self) -> &str {
        self.inner.decision_id()
    }

    pub fn request_digest(&self) -> &str {
        self.inner.request_digest()
    }

    pub fn is_consumed(&self) -> bool {
        self.inner.is_consumed()
    }
}

/// Public short-lived revalidated permit wrapper. Non-Serde and non-Clone.
#[derive(Debug)]
pub struct ReplayProtectedItExecutionPermitV1 {
    inner: RevalidatedItExecutionPermitV1,
}

impl ReplayProtectedItExecutionPermitV1 {
    pub fn handoff_deadline_unix_ms(&self) -> u64 {
        self.inner.handoff_deadline_unix_ms()
    }
}

#[derive(Debug)]
pub struct ReplayProtectedItAuthorizationOutcomeV1 {
    pub record: ItAuthorizationDecisionRecordV1,
    pub capability: Option<ReplayProtectedItActionCapabilityV1>,
}

/// Sole public manager for governed IT authority in V1.
///
/// The inner manager remains crate-private so callers cannot bypass replay/time
/// admission and directly mint another capability from an already-granted request.
#[derive(Debug)]
pub struct ReplayProtectedItAuthorityManagerV1 {
    inner: ItAuthorityManagerV1,
    admission_policy: ItAuthorityAdmissionPolicyV1,
    granted_request_ids: BTreeSet<String>,
    granted_request_digests: BTreeSet<String>,
}

impl ReplayProtectedItAuthorityManagerV1 {
    pub fn new(
        policy: ItAuthorizationPolicyV1,
        admission_policy: ItAuthorityAdmissionPolicyV1,
    ) -> Result<Self, ItAuthorityAdmissionErrorV1> {
        admission_policy.validate()?;
        let inner = ItAuthorityManagerV1::new(policy)?;
        Ok(Self {
            inner,
            admission_policy,
            granted_request_ids: BTreeSet::new(),
            granted_request_digests: BTreeSet::new(),
        })
    }

    pub fn policy(&self) -> &ItAuthorizationPolicyV1 {
        self.inner.policy()
    }

    pub fn admission_policy(&self) -> &ItAuthorityAdmissionPolicyV1 {
        &self.admission_policy
    }

    pub fn revocation_generation(&self) -> u64 {
        self.inner.revocation_generation()
    }

    pub fn replace_policy(
        &mut self,
        policy: ItAuthorizationPolicyV1,
    ) -> Result<(), ItAuthorityAdmissionErrorV1> {
        self.inner.replace_policy(policy)?;
        Ok(())
    }

    pub fn revoke_all(&mut self) {
        self.inner.revoke_all();
    }

    /// Initial authorization admission. A request that has ever been granted by
    /// this runtime manager cannot be granted again under the same request ID or
    /// exact request digest. Denied attempts do not consume the request identity.
    pub fn authorize_once(
        &mut self,
        request: &ItActionRequestV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ReplayProtectedItAuthorizationOutcomeV1, ItAuthorityAdmissionErrorV1> {
        self.validate_request_time(request, now_unix_ms)?;
        let request_digest = request.digest()?;

        if self.granted_request_ids.contains(&request.request_id) {
            return Err(ItAuthorityAdmissionErrorV1::RequestIdAlreadyGranted(
                request.request_id.clone(),
            ));
        }
        if self.granted_request_digests.contains(&request_digest) {
            return Err(ItAuthorityAdmissionErrorV1::RequestDigestAlreadyGranted(
                request_digest,
            ));
        }

        let outcome = self.inner.authorize(request, graph, now_unix_ms)?;
        let granted = outcome.record.granted;
        let record = outcome.record;
        let capability = outcome
            .capability
            .map(|inner| ReplayProtectedItActionCapabilityV1 { inner });

        if granted {
            // Insert only after the inner authority decision is positively granted.
            // A failed/denied request therefore cannot burn an identity and force a
            // caller to invent a new request merely to retry after prerequisites fix.
            self.granted_request_ids.insert(request.request_id.clone());
            self.granted_request_digests.insert(request_digest);
        }

        Ok(ReplayProtectedItAuthorizationOutcomeV1 { record, capability })
    }

    pub fn revalidate_and_consume(
        &self,
        capability: &mut ReplayProtectedItActionCapabilityV1,
        request: &ItActionRequestV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ReplayProtectedItExecutionPermitV1, ItAuthorityAdmissionErrorV1> {
        let inner = self.inner.revalidate_and_consume(
            &mut capability.inner,
            request,
            graph,
            now_unix_ms,
        )?;
        Ok(ReplayProtectedItExecutionPermitV1 { inner })
    }

    pub fn into_command_authority(
        &self,
        permit: ReplayProtectedItExecutionPermitV1,
        graph: &SystemStateGraphV1,
        now_unix_ms: u64,
    ) -> Result<ItCommandAuthorityV1, ItAuthorityAdmissionErrorV1> {
        Ok(permit
            .inner
            .into_command_authority(&self.inner, graph, now_unix_ms)?)
    }

    pub fn granted_request_count(&self) -> usize {
        debug_assert_eq!(
            self.granted_request_ids.len(),
            self.granted_request_digests.len()
        );
        self.granted_request_ids.len()
    }

    fn validate_request_time(
        &self,
        request: &ItActionRequestV1,
        now_unix_ms: u64,
    ) -> Result<(), ItAuthorityAdmissionErrorV1> {
        if request.requested_at_unix_ms > now_unix_ms {
            let ahead_ms = request.requested_at_unix_ms - now_unix_ms;
            if ahead_ms > self.admission_policy.max_future_request_skew_ms {
                return Err(ItAuthorityAdmissionErrorV1::RequestFromFuture {
                    requested_at_unix_ms: request.requested_at_unix_ms,
                    now_unix_ms,
                    allowed_skew_ms: self.admission_policy.max_future_request_skew_ms,
                });
            }
            return Ok(());
        }

        let age_ms = now_unix_ms - request.requested_at_unix_ms;
        if age_ms > self.admission_policy.max_request_age_ms {
            return Err(ItAuthorityAdmissionErrorV1::RequestTooOld {
                requested_at_unix_ms: request.requested_at_unix_ms,
                now_unix_ms,
                maximum_age_ms: self.admission_policy.max_request_age_ms,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ItAuthorityAdmissionErrorV1 {
    InvalidAdmissionPolicy(String),
    RequestTooOld {
        requested_at_unix_ms: u64,
        now_unix_ms: u64,
        maximum_age_ms: u64,
    },
    RequestFromFuture {
        requested_at_unix_ms: u64,
        now_unix_ms: u64,
        allowed_skew_ms: u64,
    },
    RequestIdAlreadyGranted(String),
    RequestDigestAlreadyGranted(String),
    Authority(ItAuthorityErrorV1),
}

impl fmt::Display for ItAuthorityAdmissionErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAdmissionPolicy(message) => {
                write!(f, "invalid IT authority admission policy: {message}")
            }
            Self::RequestTooOld {
                requested_at_unix_ms,
                now_unix_ms,
                maximum_age_ms,
            } => write!(
                f,
                "IT authority request at {requested_at_unix_ms} is too old at {now_unix_ms}; maximum age is {maximum_age_ms} ms"
            ),
            Self::RequestFromFuture {
                requested_at_unix_ms,
                now_unix_ms,
                allowed_skew_ms,
            } => write!(
                f,
                "IT authority request at {requested_at_unix_ms} is too far ahead of authority time {now_unix_ms}; allowed skew is {allowed_skew_ms} ms"
            ),
            Self::RequestIdAlreadyGranted(request_id) => write!(
                f,
                "IT authority request id {request_id:?} has already produced a granted capability"
            ),
            Self::RequestDigestAlreadyGranted(digest) => write!(
                f,
                "IT authority request digest {digest} has already produced a granted capability"
            ),
            Self::Authority(error) => write!(f, "IT authority error: {error}"),
        }
    }
}

impl Error for ItAuthorityAdmissionErrorV1 {}

impl From<ItAuthorityErrorV1> for ItAuthorityAdmissionErrorV1 {
    fn from(value: ItAuthorityErrorV1) -> Self {
        Self::Authority(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::it_authority::{
        ItActionKindV1, ItAuthorityEvidenceRequirementV1, ItOperationalRiskV1,
    };
    use crate::system_state::{
        EntityId, EntityKindV1, ObservationClockV1, ObservationId,
        ObservationProvenanceV1, ObservationSourceKindV1, StateValueV1,
        SystemObservationV1,
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
                    collector: "admission-test".into(),
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
    fn same_granted_request_cannot_mint_second_capability() {
        let now = 1_000_000;
        let (graph, request, policy) = fixture(now);
        let mut manager = ReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        let first = manager.authorize_once(&request, &graph, now).unwrap();
        assert!(first.record.granted);
        assert!(first.capability.is_some());
        assert!(matches!(
            manager.authorize_once(&request, &graph, now + 1),
            Err(ItAuthorityAdmissionErrorV1::RequestIdAlreadyGranted(_))
        ));
        assert_eq!(manager.granted_request_count(), 1);
    }

    #[test]
    fn changed_payload_with_same_granted_request_id_is_rejected() {
        let now = 1_000_000;
        let (graph, mut request, policy) = fixture(now);
        let mut manager = ReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        assert!(manager
            .authorize_once(&request, &graph, now)
            .unwrap()
            .record
            .granted);
        request.parameters.insert(
            "service".into(),
            StateValueV1::Text("different-service".into()),
        );
        assert!(matches!(
            manager.authorize_once(&request, &graph, now + 1),
            Err(ItAuthorityAdmissionErrorV1::RequestIdAlreadyGranted(_))
        ));
    }

    #[test]
    fn stale_request_is_rejected_before_inner_authorization() {
        let now = 1_000_000;
        let (graph, mut request, policy) = fixture(now);
        request.requested_at_unix_ms = now - 31_000;
        let mut manager = ReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        assert!(matches!(
            manager.authorize_once(&request, &graph, now),
            Err(ItAuthorityAdmissionErrorV1::RequestTooOld { .. })
        ));
        assert_eq!(manager.granted_request_count(), 0);
    }

    #[test]
    fn future_request_beyond_skew_is_rejected() {
        let now = 1_000_000;
        let (graph, mut request, policy) = fixture(now);
        request.requested_at_unix_ms = now + 2_001;
        let mut manager = ReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        assert!(matches!(
            manager.authorize_once(&request, &graph, now),
            Err(ItAuthorityAdmissionErrorV1::RequestFromFuture { .. })
        ));
    }

    #[test]
    fn denied_request_does_not_burn_identity() {
        let now = 1_000_000;
        let (graph, request, mut policy) = fixture(now);
        policy.allowed_principals = BTreeSet::from(["operator:bob".into()]);
        let mut manager = ReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        let denied = manager.authorize_once(&request, &graph, now).unwrap();
        assert!(!denied.record.granted);
        assert!(denied.capability.is_none());
        assert_eq!(manager.granted_request_count(), 0);

        let mut corrected = manager.policy().clone();
        corrected.generation = 2;
        corrected.allowed_principals = BTreeSet::from(["operator:alice".into()]);
        manager.replace_policy(corrected).unwrap();
        let granted = manager.authorize_once(&request, &graph, now + 1).unwrap();
        assert!(granted.record.granted);
        assert!(granted.capability.is_some());
    }

    #[test]
    fn wrapper_preserves_one_use_revalidation_and_handoff() {
        let now = 1_000_000;
        let (graph, request, policy) = fixture(now);
        let mut manager = ReplayProtectedItAuthorityManagerV1::new(
            policy,
            ItAuthorityAdmissionPolicyV1::default(),
        )
        .unwrap();
        let mut capability = manager
            .authorize_once(&request, &graph, now)
            .unwrap()
            .capability
            .unwrap();
        let permit = manager
            .revalidate_and_consume(&mut capability, &request, &graph, now + 100)
            .unwrap();
        assert!(capability.is_consumed());
        let command = manager
            .into_command_authority(permit, &graph, now + 200)
            .unwrap();
        assert_eq!(command.request_digest(), capability.request_digest());
    }
}
