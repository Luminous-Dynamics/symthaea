// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Request-bound local human-approval evidence for Nixward.
//!
//! These types fill the ceremony boundary immediately above
//! `LiveNixAuthorizationV1` without becoming execution authority themselves.
//! A serialized approval decision is audit/input evidence only: it cannot be
//! deserialized into a live authorization capability or a generic dispatch permit.
//!
//! Persisted fields retain explicit `*_unix_ms` names for an unambiguous wire/audit
//! representation. Runtime construction and evaluation use the shared temporal
//! newtypes from `temporal`, so naked timestamp units do not cross the API boundary.

use super::authorization::{NixActionIntentV1, NixAuthorizationErrorV1};
use super::temporal::{
    EvidenceCurrentnessV1, EvidenceTemporalEvaluationV1, EvidenceTemporalStatusV1,
    EvidenceWindowMillisV1, UnixMillisV1,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const APPROVAL_REQUEST_DOMAIN: &[u8] = b"nixward-local-approval-request-v1";
const APPROVAL_DECISION_DOMAIN: &[u8] = b"nixward-local-approval-decision-v1";
const DISPLAY_DOMAIN: &[u8] = b"nixward-local-approval-display-v1";
const MAX_IDENTIFIER_BYTES: usize = 1024;

/// One exact local interactive approval request.
///
/// `daemon_incarnation_id` deliberately makes outstanding approval requests
/// invalid across daemon restarts unless a future separately-qualified resume
/// protocol says otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingNixApprovalRequestV1 {
    pub daemon_incarnation_id: String,
    pub action_intent_digest: String,
    pub machine_target_ref: String,
    pub displayed_action_digest: String,
    pub authority_profile_ref: String,
    pub created_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    /// Verifier/runtime supplied uniqueness. All-zero is rejected.
    pub request_nonce: [u8; 32],
}

impl PendingNixApprovalRequestV1 {
    pub fn from_intent(
        intent: &NixActionIntentV1,
        daemon_incarnation_id: impl Into<String>,
        displayed_action: &str,
        authority_profile_ref: impl Into<String>,
        created_at: UnixMillisV1,
        expires_at: UnixMillisV1,
        request_nonce: [u8; 32],
    ) -> Result<Self, LocalApprovalErrorV1> {
        if displayed_action.trim().is_empty() {
            return Err(LocalApprovalErrorV1::EmptyDisplayedAction);
        }
        let request = Self {
            daemon_incarnation_id: daemon_incarnation_id.into(),
            action_intent_digest: intent.digest()?,
            // #4929 V1 uses `subject_identity` for the machine/installation target.
            // Derive it rather than accepting a second caller-selected target truth.
            // A future intent schema can rename this field explicitly under #5197.
            machine_target_ref: intent.subject_identity.clone(),
            displayed_action_digest: digest_display(displayed_action),
            authority_profile_ref: authority_profile_ref.into(),
            created_at_unix_ms: created_at.as_u64(),
            expires_at_unix_ms: expires_at.as_u64(),
            request_nonce,
        };
        request.validate_shape()?;
        Ok(request)
    }

    pub fn validate_shape(&self) -> Result<(), LocalApprovalErrorV1> {
        validate_identifier(&self.daemon_incarnation_id, "daemon incarnation id")?;
        validate_digest(&self.action_intent_digest, "action intent digest")?;
        validate_identifier(&self.machine_target_ref, "machine target ref")?;
        validate_digest(&self.displayed_action_digest, "displayed action digest")?;
        validate_identifier(&self.authority_profile_ref, "authority profile ref")?;
        self.currentness()?;

        if self.request_nonce == [0; 32] {
            return Err(LocalApprovalErrorV1::ZeroRequestNonce);
        }
        Ok(())
    }

    /// Stable identity of this exact pending request.
    pub fn request_id(&self) -> Result<String, LocalApprovalErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(APPROVAL_REQUEST_DOMAIN);
        put_str(&mut h, &self.daemon_incarnation_id);
        put_str(&mut h, &self.action_intent_digest);
        put_str(&mut h, &self.machine_target_ref);
        put_str(&mut h, &self.displayed_action_digest);
        put_str(&mut h, &self.authority_profile_ref);
        put_u64(&mut h, self.created_at_unix_ms);
        put_u64(&mut h, self.expires_at_unix_ms);
        h.update(&self.request_nonce);
        Ok(h.finalize().to_hex().to_string())
    }

    /// Replayable temporal evaluation of this request at one exact wall-clock time.
    pub fn temporal_evaluation(
        &self,
        evaluated_at: UnixMillisV1,
    ) -> Result<EvidenceTemporalEvaluationV1, LocalApprovalErrorV1> {
        self.validate_shape()?;
        Ok(EvidenceTemporalEvaluationV1::evaluate(
            self.currentness()?,
            evaluated_at,
        ))
    }

    pub fn is_current_at(&self, now: UnixMillisV1) -> Result<bool, LocalApprovalErrorV1> {
        Ok(self.temporal_evaluation(now)?.is_current())
    }

    fn currentness(&self) -> Result<EvidenceCurrentnessV1, LocalApprovalErrorV1> {
        let window = EvidenceWindowMillisV1::new(
            UnixMillisV1::new(self.created_at_unix_ms),
            UnixMillisV1::new(self.expires_at_unix_ms),
        )
        .map_err(|_| LocalApprovalErrorV1::InvalidRequestWindow)?;
        Ok(EvidenceCurrentnessV1::Observed(window))
    }
}

/// Human decision over one exact pending request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalApprovalDecisionKindV1 {
    Approved,
    Denied,
}

/// Serializable local approval evidence.
///
/// This record is intentionally ordinary data. Matching `Approved` evidence is
/// still not a live capability. A runtime bridge must additionally prove that the
/// request is still live/current and then mint the separate transient authority
/// object through the owning authority layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalNixApprovalDecisionV1 {
    pub request_id: String,
    pub daemon_incarnation_id: String,
    pub action_intent_digest: String,
    pub decision: LocalApprovalDecisionKindV1,
    pub decided_at_unix_ms: u64,
    pub approver_ref: String,
}

impl LocalNixApprovalDecisionV1 {
    pub fn for_request(
        request: &PendingNixApprovalRequestV1,
        decision: LocalApprovalDecisionKindV1,
        decided_at: UnixMillisV1,
        approver_ref: impl Into<String>,
    ) -> Result<Self, LocalApprovalErrorV1> {
        request.validate_shape()?;
        match request.temporal_evaluation(decided_at)?.status() {
            EvidenceTemporalStatusV1::Current => {}
            EvidenceTemporalStatusV1::NotYetValid => {
                return Err(LocalApprovalErrorV1::DecisionBeforeRequest);
            }
            EvidenceTemporalStatusV1::Expired => {
                return Err(LocalApprovalErrorV1::DecisionAfterExpiry);
            }
            EvidenceTemporalStatusV1::Unknown => {
                return Err(LocalApprovalErrorV1::TemporalStatusUnknown);
            }
        }

        let record = Self {
            request_id: request.request_id()?,
            daemon_incarnation_id: request.daemon_incarnation_id.clone(),
            action_intent_digest: request.action_intent_digest.clone(),
            decision,
            decided_at_unix_ms: decided_at.as_u64(),
            approver_ref: approver_ref.into(),
        };
        record.validate_shape()?;
        Ok(record)
    }

    pub fn validate_shape(&self) -> Result<(), LocalApprovalErrorV1> {
        validate_digest(&self.request_id, "request id")?;
        validate_identifier(&self.daemon_incarnation_id, "daemon incarnation id")?;
        validate_digest(&self.action_intent_digest, "action intent digest")?;
        validate_identifier(&self.approver_ref, "approver ref")?;
        Ok(())
    }

    pub fn digest(&self) -> Result<String, LocalApprovalErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(APPROVAL_DECISION_DOMAIN);
        put_str(&mut h, &self.request_id);
        put_str(&mut h, &self.daemon_incarnation_id);
        put_str(&mut h, &self.action_intent_digest);
        put_u8(
            &mut h,
            match self.decision {
                LocalApprovalDecisionKindV1::Approved => 0,
                LocalApprovalDecisionKindV1::Denied => 1,
            },
        );
        put_u64(&mut h, self.decided_at_unix_ms);
        put_str(&mut h, &self.approver_ref);
        Ok(h.finalize().to_hex().to_string())
    }

    /// Evaluate this persisted decision against the exact still-live request.
    ///
    /// A positive result is **approval evidence**, not execution authority.
    pub fn evaluate_against(
        &self,
        request: &PendingNixApprovalRequestV1,
        now: UnixMillisV1,
    ) -> Result<LocalApprovalDecisionKindV1, LocalApprovalErrorV1> {
        self.validate_shape()?;
        request.validate_shape()?;

        // Preserve precise diagnostics for the two semantically important
        // mismatches before the aggregate request commitment check.
        if self.daemon_incarnation_id != request.daemon_incarnation_id {
            return Err(LocalApprovalErrorV1::DaemonIncarnationMismatch);
        }
        if self.action_intent_digest != request.action_intent_digest {
            return Err(LocalApprovalErrorV1::IntentMismatch);
        }
        if self.request_id != request.request_id()? {
            return Err(LocalApprovalErrorV1::RequestMismatch);
        }

        let decided_at = UnixMillisV1::new(self.decided_at_unix_ms);
        match request.temporal_evaluation(decided_at)?.status() {
            EvidenceTemporalStatusV1::Current => {}
            EvidenceTemporalStatusV1::NotYetValid => {
                return Err(LocalApprovalErrorV1::DecisionBeforeRequest);
            }
            EvidenceTemporalStatusV1::Expired => {
                return Err(LocalApprovalErrorV1::DecisionAfterExpiry);
            }
            EvidenceTemporalStatusV1::Unknown => {
                return Err(LocalApprovalErrorV1::TemporalStatusUnknown);
            }
        }

        if decided_at > now {
            return Err(LocalApprovalErrorV1::DecisionFromFuture);
        }

        match request.temporal_evaluation(now)?.status() {
            EvidenceTemporalStatusV1::Current => Ok(self.decision),
            EvidenceTemporalStatusV1::NotYetValid => {
                Err(LocalApprovalErrorV1::RequestNotYetValid)
            }
            EvidenceTemporalStatusV1::Expired => Err(LocalApprovalErrorV1::RequestExpired),
            EvidenceTemporalStatusV1::Unknown => Err(LocalApprovalErrorV1::TemporalStatusUnknown),
        }
    }
}

pub fn digest_display(displayed_action: &str) -> String {
    let mut h = Hasher::new();
    h.update(DISPLAY_DOMAIN);
    put_str(&mut h, displayed_action);
    h.finalize().to_hex().to_string()
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LocalApprovalErrorV1 {
    #[error("action-intent construction failed: {0}")]
    ActionIntent(#[from] NixAuthorizationErrorV1),
    #[error("empty or oversized field: {0}")]
    InvalidIdentifier(&'static str),
    #[error("invalid canonical digest field: {0}")]
    InvalidDigest(&'static str),
    #[error("operator-visible action display must not be blank")]
    EmptyDisplayedAction,
    #[error("approval request expires before it is created")]
    InvalidRequestWindow,
    #[error("approval request nonce must not be all-zero")]
    ZeroRequestNonce,
    #[error("approval decision predates its request")]
    DecisionBeforeRequest,
    #[error("approval decision was issued after request expiry")]
    DecisionAfterExpiry,
    #[error("approval decision names another request")]
    RequestMismatch,
    #[error("approval decision belongs to another daemon incarnation")]
    DaemonIncarnationMismatch,
    #[error("approval decision is bound to another action intent")]
    IntentMismatch,
    #[error("approval request is not yet valid")]
    RequestNotYetValid,
    #[error("approval request has expired")]
    RequestExpired,
    #[error("approval decision timestamp is later than evaluation time")]
    DecisionFromFuture,
    #[error("approval request has unknown temporal status")]
    TemporalStatusUnknown,
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), LocalApprovalErrorV1> {
    if value.trim().is_empty() || value.len() > MAX_IDENTIFIER_BYTES {
        Err(LocalApprovalErrorV1::InvalidIdentifier(field))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), LocalApprovalErrorV1> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(LocalApprovalErrorV1::InvalidDigest(field));
    }
    Ok(())
}

fn put_u8(h: &mut Hasher, value: u8) {
    h.update(&[value]);
}

fn put_u64(h: &mut Hasher, value: u64) {
    h.update(&value.to_be_bytes());
}

fn put_str(h: &mut Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::executor::NixOSCommand;

    fn ms(value: u64) -> UnixMillisV1 {
        UnixMillisV1::new(value)
    }

    fn intent(host: &str, generation: &str) -> NixActionIntentV1 {
        NixActionIntentV1::from_command(
            host,
            Some(generation.to_string()),
            &NixOSCommand::RebuildSwitch {
                flake: Some(".#workstation".to_string()),
                extra_args: vec![],
            },
        )
        .unwrap()
    }

    fn request(intent: &NixActionIntentV1, nonce_byte: u8) -> PendingNixApprovalRequestV1 {
        PendingNixApprovalRequestV1::from_intent(
            intent,
            "daemon-incarnation:1",
            "nixos-rebuild switch --flake .#workstation",
            "local-human-v1",
            ms(1_000),
            ms(2_000),
            [nonce_byte; 32],
        )
        .unwrap()
    }

    #[test]
    fn request_identity_is_deterministic_and_binds_display() {
        let intent = intent("machine:workstation", "generation:42");
        let a = request(&intent, 1);
        let mut b = request(&intent, 1);
        assert_eq!(a.request_id().unwrap(), b.request_id().unwrap());

        b.displayed_action_digest = digest_display("different operator-visible action");
        assert_ne!(a.request_id().unwrap(), b.request_id().unwrap());
    }

    #[test]
    fn machine_target_is_derived_from_exact_intent() {
        let a_intent = intent("machine:workstation-a", "generation:42");
        let b_intent = intent("machine:workstation-b", "generation:42");
        let a = request(&a_intent, 1);
        let b = request(&b_intent, 1);

        assert_eq!(a.machine_target_ref, "machine:workstation-a");
        assert_eq!(b.machine_target_ref, "machine:workstation-b");
        assert_ne!(a.request_id().unwrap(), b.request_id().unwrap());
    }

    #[test]
    fn blank_operator_display_is_rejected() {
        let intent = intent("machine:workstation", "generation:42");
        assert_eq!(
            PendingNixApprovalRequestV1::from_intent(
                &intent,
                "daemon-incarnation:1",
                "   ",
                "local-human-v1",
                ms(1_000),
                ms(2_000),
                [1; 32],
            )
            .unwrap_err(),
            LocalApprovalErrorV1::EmptyDisplayedAction
        );
    }

    #[test]
    fn request_identity_changes_with_daemon_incarnation_or_nonce() {
        let intent = intent("machine:workstation", "generation:42");
        let a = request(&intent, 1);
        let mut b = request(&intent, 1);
        b.daemon_incarnation_id = "daemon-incarnation:2".to_string();
        assert_ne!(a.request_id().unwrap(), b.request_id().unwrap());

        let c = request(&intent, 2);
        assert_ne!(a.request_id().unwrap(), c.request_id().unwrap());
    }

    #[test]
    fn request_identity_changes_when_intent_or_prestate_changes() {
        let a_intent = intent("machine:workstation", "generation:42");
        let b_intent = intent("machine:workstation", "generation:43");
        let a = request(&a_intent, 1);
        let b = request(&b_intent, 1);
        assert_ne!(a.request_id().unwrap(), b.request_id().unwrap());
    }

    #[test]
    fn request_temporal_status_distinguishes_future_current_and_expired() {
        let intent = intent("machine:workstation", "generation:42");
        let request = request(&intent, 1);

        assert_eq!(
            request.temporal_evaluation(ms(999)).unwrap().status(),
            EvidenceTemporalStatusV1::NotYetValid
        );
        assert_eq!(
            request.temporal_evaluation(ms(1_000)).unwrap().status(),
            EvidenceTemporalStatusV1::Current
        );
        assert_eq!(
            request.temporal_evaluation(ms(2_000)).unwrap().status(),
            EvidenceTemporalStatusV1::Current
        );
        assert_eq!(
            request.temporal_evaluation(ms(2_001)).unwrap().status(),
            EvidenceTemporalStatusV1::Expired
        );
    }

    #[test]
    fn exact_approval_matches_only_its_live_request() {
        let intent = intent("machine:workstation", "generation:42");
        let request = request(&intent, 1);
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_100),
            "local-user:test",
        )
        .unwrap();

        assert_eq!(
            decision.evaluate_against(&request, ms(1_200)).unwrap(),
            LocalApprovalDecisionKindV1::Approved
        );
    }

    #[test]
    fn copied_approval_cannot_satisfy_new_request_nonce() {
        let intent = intent("machine:workstation", "generation:42");
        let request_a = request(&intent, 1);
        let request_b = request(&intent, 2);
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request_a,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_100),
            "local-user:test",
        )
        .unwrap();

        assert_eq!(
            decision.evaluate_against(&request_b, ms(1_200)).unwrap_err(),
            LocalApprovalErrorV1::RequestMismatch
        );
    }

    #[test]
    fn approval_cannot_cross_daemon_restart() {
        let intent = intent("machine:workstation", "generation:42");
        let request_a = request(&intent, 1);
        let mut request_b = request_a.clone();
        request_b.daemon_incarnation_id = "daemon-incarnation:2".to_string();
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request_a,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_100),
            "local-user:test",
        )
        .unwrap();

        assert_eq!(
            decision.evaluate_against(&request_b, ms(1_200)).unwrap_err(),
            LocalApprovalErrorV1::DaemonIncarnationMismatch
        );
    }

    #[test]
    fn changed_intent_reports_intent_mismatch() {
        let intent_a = intent("machine:workstation", "generation:42");
        let intent_b = intent("machine:workstation", "generation:43");
        let request_a = request(&intent_a, 1);
        let request_b = request(&intent_b, 1);
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request_a,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_100),
            "local-user:test",
        )
        .unwrap();

        assert_eq!(
            decision.evaluate_against(&request_b, ms(1_200)).unwrap_err(),
            LocalApprovalErrorV1::IntentMismatch
        );
    }

    #[test]
    fn expired_request_rejects_previously_valid_approval() {
        let intent = intent("machine:workstation", "generation:42");
        let request = request(&intent, 1);
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_100),
            "local-user:test",
        )
        .unwrap();

        assert_eq!(
            decision.evaluate_against(&request, ms(2_001)).unwrap_err(),
            LocalApprovalErrorV1::RequestExpired
        );
    }

    #[test]
    fn decision_from_future_is_rejected() {
        let intent = intent("machine:workstation", "generation:42");
        let request = request(&intent, 1);
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_500),
            "local-user:test",
        )
        .unwrap();

        assert_eq!(
            decision.evaluate_against(&request, ms(1_400)).unwrap_err(),
            LocalApprovalErrorV1::DecisionFromFuture
        );
    }

    #[test]
    fn denied_decision_remains_denied() {
        let intent = intent("machine:workstation", "generation:42");
        let request = request(&intent, 1);
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Denied,
            ms(1_100),
            "local-user:test",
        )
        .unwrap();
        assert_eq!(
            decision.evaluate_against(&request, ms(1_200)).unwrap(),
            LocalApprovalDecisionKindV1::Denied
        );
    }

    #[test]
    fn decision_after_expiry_is_rejected_at_construction() {
        let intent = intent("machine:workstation", "generation:42");
        let request = request(&intent, 1);
        assert_eq!(
            LocalNixApprovalDecisionV1::for_request(
                &request,
                LocalApprovalDecisionKindV1::Approved,
                ms(2_001),
                "local-user:test",
            )
            .unwrap_err(),
            LocalApprovalErrorV1::DecisionAfterExpiry
        );
    }

    #[test]
    fn all_zero_nonce_is_rejected() {
        let intent = intent("machine:workstation", "generation:42");
        assert_eq!(
            PendingNixApprovalRequestV1::from_intent(
                &intent,
                "daemon-incarnation:1",
                "nixos-rebuild switch --flake .#workstation",
                "local-human-v1",
                ms(1_000),
                ms(2_000),
                [0; 32],
            )
            .unwrap_err(),
            LocalApprovalErrorV1::ZeroRequestNonce
        );
    }

    #[test]
    fn persisted_decision_roundtrip_is_still_only_data() {
        let intent = intent("machine:workstation", "generation:42");
        let request = request(&intent, 1);
        let decision = LocalNixApprovalDecisionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_100),
            "local-user:test",
        )
        .unwrap();

        let json = serde_json::to_string(&decision).unwrap();
        let restored: LocalNixApprovalDecisionV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decision, restored);
        assert_eq!(decision.digest().unwrap(), restored.digest().unwrap());
        // Intentionally no conversion from this persisted data type into
        // LiveNixAuthorizationV1 exists in this module.
    }
}
