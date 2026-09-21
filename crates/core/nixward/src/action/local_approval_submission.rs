// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Identity-free client submissions for local interactive Nixward approval.
//!
//! The local client is allowed to submit only a decision over one exact pending
//! request. It is not allowed to claim its UID/GID/PID, approver identity, Xenia
//! identity, role, or authority. The daemon obtains local peer credentials from
//! the kernel-backed IPC adapter and combines the two facts here.
//!
//! A successful admission returns ordinary `LocalNixApprovalDecisionV1` evidence.
//! It does not consume the pending request and is not execution authority. The
//! owning request store / authority runtime must still perform atomic single-use
//! consumption before minting or exercising any live effect capability.

use super::approver_evidence::{
    ApproverEvidenceErrorV1, ApproverEvidenceProfileV1, VerifiedLocalUnixPeerCredentialV1,
};
use super::local_approval::{
    LocalApprovalDecisionKindV1, LocalApprovalErrorV1, LocalNixApprovalDecisionV1,
    PendingNixApprovalRequestV1,
};
use super::temporal::UnixMillisV1;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const LOCAL_PEER_APPROVER_REF_PREFIX: &str =
    "nixward-approver-evidence-v1:local-unix-peer-credential-v1:";

/// What a local approval client may send to the daemon.
///
/// There are intentionally no identity or authority fields. Unknown fields are
/// rejected during deserialization so a caller cannot smuggle a self-asserted
/// `uid`, `approver_ref`, Xenia principal, role, or permit into this protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalApprovalSubmissionV1 {
    pub request_id: String,
    pub daemon_incarnation_id: String,
    pub action_intent_digest: String,
    pub decision: LocalApprovalDecisionKindV1,
    pub decided_at_unix_ms: u64,
}

impl LocalApprovalSubmissionV1 {
    /// Build a client submission from the exact request the operator reviewed.
    pub fn for_request(
        request: &PendingNixApprovalRequestV1,
        decision: LocalApprovalDecisionKindV1,
        decided_at: UnixMillisV1,
    ) -> Result<Self, LocalApprovalErrorV1> {
        request.validate_shape()?;
        let submission = Self {
            request_id: request.request_id()?,
            daemon_incarnation_id: request.daemon_incarnation_id.clone(),
            action_intent_digest: request.action_intent_digest.clone(),
            decision,
            decided_at_unix_ms: decided_at.as_u64(),
        };
        submission.validate_against(request)?;
        Ok(submission)
    }

    /// Verify that the transport input names the exact still-pending request.
    ///
    /// This validates request binding and the decision timestamp against the
    /// request's own validity window. Current-time evaluation is performed by
    /// `admit_verified_local_submission_v1` immediately after daemon-observed
    /// peer identity is attached.
    pub fn validate_against(
        &self,
        request: &PendingNixApprovalRequestV1,
    ) -> Result<(), LocalApprovalErrorV1> {
        request.validate_shape()?;

        if self.daemon_incarnation_id != request.daemon_incarnation_id {
            return Err(LocalApprovalErrorV1::DaemonIncarnationMismatch);
        }
        if self.action_intent_digest != request.action_intent_digest {
            return Err(LocalApprovalErrorV1::IntentMismatch);
        }
        if self.request_id != request.request_id()? {
            return Err(LocalApprovalErrorV1::RequestMismatch);
        }

        // Reuse the existing decision constructor for the exact temporal
        // semantics instead of defining a second approval clock policy here.
        let _ = LocalNixApprovalDecisionV1::for_request(
            request,
            self.decision,
            UnixMillisV1::new(self.decided_at_unix_ms),
            "nixward-local-submission-validation-v1",
        )?;

        Ok(())
    }
}

/// Combine identity-free client input with a kernel-verified local peer.
///
/// This is the preferred local daemon admission seam. The approver reference in
/// the resulting audit record is derived exclusively from the verified peer
/// evidence. No client-controlled identity string is accepted by this function.
pub fn admit_verified_local_submission_v1(
    submission: &LocalApprovalSubmissionV1,
    request: &PendingNixApprovalRequestV1,
    verified_peer: &VerifiedLocalUnixPeerCredentialV1,
    now: UnixMillisV1,
) -> Result<LocalNixApprovalDecisionV1, LocalApprovalAdmissionErrorV1> {
    submission.validate_against(request)?;

    let approver_ref = local_verified_peer_approver_ref_v1(verified_peer)?;
    let decision = LocalNixApprovalDecisionV1::for_request(
        request,
        submission.decision,
        UnixMillisV1::new(submission.decided_at_unix_ms),
        approver_ref,
    )?;

    // Re-evaluate immediately against the daemon's current view so a decision
    // from the future or an expired request cannot become admitted evidence.
    let evaluated = decision.evaluate_against(request, now)?;
    debug_assert_eq!(evaluated, submission.decision);

    Ok(decision)
}

/// Stable audit encoding for the daemon-observed local peer evidence reference.
///
/// This string remains provenance only. The positive theorem comes from the
/// non-Serde `VerifiedLocalUnixPeerCredentialV1` argument to the admission
/// function, not from possession of this string.
fn local_verified_peer_approver_ref_v1(
    verified_peer: &VerifiedLocalUnixPeerCredentialV1,
) -> Result<String, LocalApprovalAdmissionErrorV1> {
    let evidence_ref = verified_peer.evidence_ref();
    evidence_ref.validate_shape()?;
    if evidence_ref.profile != ApproverEvidenceProfileV1::LocalUnixPeerCredentialV1 {
        return Err(LocalApprovalAdmissionErrorV1::UnexpectedApproverEvidenceProfile);
    }
    Ok(format!(
        "{LOCAL_PEER_APPROVER_REF_PREFIX}{}",
        evidence_ref.evidence_digest
    ))
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LocalApprovalAdmissionErrorV1 {
    #[error("local approval request/decision validation failed: {0}")]
    Approval(#[from] LocalApprovalErrorV1),
    #[error("approver evidence reference construction failed: {0}")]
    ApproverEvidence(#[from] ApproverEvidenceErrorV1),
    #[error("verified local peer carried a non-local approver evidence profile")]
    UnexpectedApproverEvidenceProfile,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::authorization::NixActionIntentV1;
    use crate::action::executor::NixOSCommand;
    use crate::action::VerifiedLocalUnixPeerCredentialV1;

    fn ms(value: u64) -> UnixMillisV1 {
        UnixMillisV1::new(value)
    }

    fn intent() -> NixActionIntentV1 {
        NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &NixOSCommand::RebuildSwitch {
                flake: Some(".#workstation".to_string()),
                extra_args: vec![],
            },
        )
        .unwrap()
    }

    fn request() -> PendingNixApprovalRequestV1 {
        PendingNixApprovalRequestV1::from_intent(
            &intent(),
            "nixward-daemon-incarnation-v1:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "nixos-rebuild switch --flake .#workstation",
            "local-human-v1",
            ms(1_000),
            ms(2_000),
            [7; 32],
        )
        .unwrap()
    }

    fn peer(uid: u32, pid: u32) -> VerifiedLocalUnixPeerCredentialV1 {
        VerifiedLocalUnixPeerCredentialV1::from_kernel_peer_observation(
            uid,
            100,
            Some(pid),
            "unix-socket-instance:test-1",
            ms(1_100),
        )
        .unwrap()
    }

    #[test]
    fn exact_submission_plus_verified_peer_produces_bound_existing_decision_record() {
        let request = request();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let peer = peer(1000, 4242);

        let admitted =
            admit_verified_local_submission_v1(&submission, &request, &peer, ms(1_300)).unwrap();

        assert_eq!(admitted.request_id, request.request_id().unwrap());
        assert_eq!(admitted.daemon_incarnation_id, request.daemon_incarnation_id);
        assert_eq!(admitted.action_intent_digest, request.action_intent_digest);
        assert_eq!(admitted.decision, LocalApprovalDecisionKindV1::Approved);
        assert_eq!(admitted.decided_at_unix_ms, 1_200);
        assert_eq!(
            admitted.approver_ref,
            local_verified_peer_approver_ref_v1(&peer).unwrap()
        );
        assert!(
            admitted
                .approver_ref
                .starts_with(LOCAL_PEER_APPROVER_REF_PREFIX)
        );
    }

    #[test]
    fn submission_with_wrong_request_id_is_rejected() {
        let request = request();
        let mut submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        submission.request_id = "00".repeat(32);

        let err = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1000, 1),
            ms(1_300),
        )
        .unwrap_err();
        assert_eq!(
            err,
            LocalApprovalAdmissionErrorV1::Approval(LocalApprovalErrorV1::RequestMismatch)
        );
    }

    #[test]
    fn submission_for_another_daemon_incarnation_is_rejected() {
        let request = request();
        let mut submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        submission.daemon_incarnation_id = "another-daemon-incarnation".to_string();

        let err = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1000, 1),
            ms(1_300),
        )
        .unwrap_err();
        assert_eq!(
            err,
            LocalApprovalAdmissionErrorV1::Approval(
                LocalApprovalErrorV1::DaemonIncarnationMismatch
            )
        );
    }

    #[test]
    fn submission_for_another_action_intent_is_rejected() {
        let request = request();
        let mut submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        submission.action_intent_digest = "11".repeat(32);

        let err = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1000, 1),
            ms(1_300),
        )
        .unwrap_err();
        assert_eq!(
            err,
            LocalApprovalAdmissionErrorV1::Approval(LocalApprovalErrorV1::IntentMismatch)
        );
    }

    #[test]
    fn decision_from_future_is_rejected_at_daemon_admission_time() {
        let request = request();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_500),
        )
        .unwrap();

        let err = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1000, 1),
            ms(1_400),
        )
        .unwrap_err();
        assert_eq!(
            err,
            LocalApprovalAdmissionErrorV1::Approval(LocalApprovalErrorV1::DecisionFromFuture)
        );
    }

    #[test]
    fn expired_request_is_rejected_at_daemon_admission_time() {
        let request = request();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_500),
        )
        .unwrap();

        let err = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1000, 1),
            ms(2_001),
        )
        .unwrap_err();
        assert_eq!(
            err,
            LocalApprovalAdmissionErrorV1::Approval(LocalApprovalErrorV1::RequestExpired)
        );
    }

    #[test]
    fn client_wire_shape_rejects_identity_and_authority_fields() {
        let request = request();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let value = serde_json::to_value(&submission).unwrap();
        let object = value.as_object().unwrap();

        for (field, value) in [
            ("uid", serde_json::json!(0)),
            ("gid", serde_json::json!(0)),
            ("pid", serde_json::json!(1)),
            ("approver_ref", serde_json::json!("root")),
            ("xenia_principal", serde_json::json!("operator:admin")),
            ("role", serde_json::json!("admin")),
            ("permit", serde_json::json!("allow-all")),
        ] {
            let mut hostile = object.clone();
            hostile.insert(field.to_string(), value);
            assert!(
                serde_json::from_value::<LocalApprovalSubmissionV1>(
                    serde_json::Value::Object(hostile)
                )
                .is_err(),
                "identity/authority field {field} must be rejected"
            );
        }
    }

    #[test]
    fn different_verified_peer_changes_only_daemon_derived_approver_binding() {
        let request = request();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();

        let first = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1000, 10),
            ms(1_300),
        )
        .unwrap();
        let second = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1001, 11),
            ms(1_300),
        )
        .unwrap();

        assert_eq!(first.request_id, second.request_id);
        assert_eq!(first.decision, second.decision);
        assert_ne!(first.approver_ref, second.approver_ref);
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn denial_is_admitted_as_denial_evidence_not_authority() {
        let request = request();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Denied,
            ms(1_200),
        )
        .unwrap();
        let admitted = admit_verified_local_submission_v1(
            &submission,
            &request,
            &peer(1000, 10),
            ms(1_300),
        )
        .unwrap();

        assert_eq!(admitted.decision, LocalApprovalDecisionKindV1::Denied);
        assert_eq!(
            admitted.evaluate_against(&request, ms(1_300)).unwrap(),
            LocalApprovalDecisionKindV1::Denied
        );
    }

    #[test]
    fn admission_layer_deliberately_does_not_claim_single_use() {
        let request = request();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let peer = peer(1000, 10);

        let first =
            admit_verified_local_submission_v1(&submission, &request, &peer, ms(1_300)).unwrap();
        let second =
            admit_verified_local_submission_v1(&submission, &request, &peer, ms(1_300)).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
        // Atomic request consumption belongs to the owning request store / live
        // authority runtime. This pure evidence-admission layer intentionally has
        // no mutable nonce/request-consumption state and makes no single-use claim.
    }
}
