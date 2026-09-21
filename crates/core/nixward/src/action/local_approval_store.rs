// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Daemon-incarnation-bound pending local approval request store.
//!
//! This module owns one narrow live-state theorem:
//!
//! - pending requests belong to one non-rehydratable daemon incarnation;
//! - at most one request for one exact action-intent digest is pending at a time;
//! - replacement/supersession and consumption occur under one mutex;
//! - a successful consume combines the exact pending request, an identity-free
//!   client submission, and a kernel-verified local peer, then removes that
//!   request before returning a live consumed-decision token.
//!
//! A consumed decision is still not Nix execution authority. A later authority
//! layer must consume that live token under the applicable policy/state checks.

use super::daemon_incarnation::LiveDaemonIncarnationV1;
use super::local_approval::{
    LocalApprovalDecisionKindV1, LocalApprovalErrorV1, LocalNixApprovalDecisionV1,
    PendingNixApprovalRequestV1,
};
use super::local_approval_submission::{
    LocalApprovalAdmissionErrorV1, LocalApprovalSubmissionV1,
    admit_verified_local_submission_v1,
};
use super::approver_evidence::VerifiedLocalUnixPeerCredentialV1;
use super::temporal::UnixMillisV1;
use std::collections::HashMap;
use std::sync::Mutex;
use thiserror::Error;

/// Live, process-incarnation-bound pending request store.
///
/// There is deliberately no constructor from a persisted incarnation reference,
/// no Serialize/Deserialize implementation, and no Clone implementation.
pub struct LocalApprovalRequestStoreV1 {
    daemon_incarnation_ref: String,
    pending: Mutex<HashMap<String, PendingNixApprovalRequestV1>>,
}

impl LocalApprovalRequestStoreV1 {
    pub fn new(daemon_incarnation: &LiveDaemonIncarnationV1) -> Self {
        Self {
            daemon_incarnation_ref: daemon_incarnation.reference(),
            pending: Mutex::new(HashMap::new()),
        }
    }

    pub fn daemon_incarnation_ref(&self) -> &str {
        &self.daemon_incarnation_ref
    }

    /// Atomically install one pending request for its exact action intent.
    ///
    /// Any older request for the same `action_intent_digest` is superseded under
    /// the same mutex before the new request becomes visible. A submission for a
    /// superseded request therefore cannot race its way into a positive consume
    /// after replacement has completed.
    pub fn install_pending(
        &self,
        request: PendingNixApprovalRequestV1,
    ) -> Result<PendingRequestInstallV1, LocalApprovalRequestStoreErrorV1> {
        request.validate_shape()?;
        if request.daemon_incarnation_id != self.daemon_incarnation_ref {
            return Err(LocalApprovalRequestStoreErrorV1::DaemonIncarnationMismatch);
        }

        let request_id = request.request_id()?;
        let intent_digest = request.action_intent_digest.clone();
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| LocalApprovalRequestStoreErrorV1::StorePoisoned)?;

        if pending.contains_key(&request_id) {
            return Err(LocalApprovalRequestStoreErrorV1::DuplicateRequest);
        }

        let superseded_request_ids: Vec<String> = pending
            .iter()
            .filter_map(|(id, existing)| {
                (existing.action_intent_digest == intent_digest).then(|| id.clone())
            })
            .collect();
        for id in &superseded_request_ids {
            pending.remove(id);
        }
        pending.insert(request_id.clone(), request);

        Ok(PendingRequestInstallV1 {
            request_id,
            superseded_request_ids,
        })
    }

    /// Atomically admit and consume one exact pending local approval request.
    ///
    /// The mutex remains held from lookup through admission and removal. Two
    /// concurrent consumers of the same request cannot both observe and remove
    /// the same pending request successfully.
    pub fn consume_verified_submission(
        &self,
        submission: &LocalApprovalSubmissionV1,
        verified_peer: &VerifiedLocalUnixPeerCredentialV1,
        now: UnixMillisV1,
    ) -> Result<ConsumedLocalApprovalDecisionV1, LocalApprovalRequestStoreErrorV1> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| LocalApprovalRequestStoreErrorV1::StorePoisoned)?;

        let decision = {
            let request = pending
                .get(&submission.request_id)
                .ok_or(LocalApprovalRequestStoreErrorV1::RequestNotPending)?;
            admit_verified_local_submission_v1(submission, request, verified_peer, now)?
        };

        let removed = pending
            .remove(&submission.request_id)
            .ok_or(LocalApprovalRequestStoreErrorV1::RequestDisappearedDuringConsume)?;
        debug_assert_eq!(removed.request_id().ok().as_deref(), Some(submission.request_id.as_str()));

        Ok(ConsumedLocalApprovalDecisionV1 {
            request_id: submission.request_id.clone(),
            decision_evidence: decision,
            consumed_at_unix_ms: now.as_u64(),
        })
    }

    pub fn pending_count(&self) -> Result<usize, LocalApprovalRequestStoreErrorV1> {
        Ok(self
            .pending
            .lock()
            .map_err(|_| LocalApprovalRequestStoreErrorV1::StorePoisoned)?
            .len())
    }

    pub fn is_pending(
        &self,
        request_id: &str,
    ) -> Result<bool, LocalApprovalRequestStoreErrorV1> {
        Ok(self
            .pending
            .lock()
            .map_err(|_| LocalApprovalRequestStoreErrorV1::StorePoisoned)?
            .contains_key(request_id))
    }
}

impl std::fmt::Debug for LocalApprovalRequestStoreV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending_count = self.pending.lock().map(|p| p.len()).ok();
        f.debug_struct("LocalApprovalRequestStoreV1")
            .field("daemon_incarnation_ref", &self.daemon_incarnation_ref)
            .field("pending_count", &pending_count)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRequestInstallV1 {
    pub request_id: String,
    pub superseded_request_ids: Vec<String>,
}

/// Live proof that one exact pending request was admitted and atomically removed.
///
/// This type is intentionally private-field, non-Serde, non-Clone and non-Copy.
/// Persisted `LocalNixApprovalDecisionV1` remains audit evidence only; a later live
/// authority layer should accept this token by value if it needs the store's
/// single-use theorem.
pub struct ConsumedLocalApprovalDecisionV1 {
    request_id: String,
    decision_evidence: LocalNixApprovalDecisionV1,
    consumed_at_unix_ms: u64,
}

impl ConsumedLocalApprovalDecisionV1 {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn decision_kind(&self) -> LocalApprovalDecisionKindV1 {
        self.decision_evidence.decision
    }

    pub fn decision_evidence(&self) -> &LocalNixApprovalDecisionV1 {
        &self.decision_evidence
    }

    pub fn consumed_at(&self) -> UnixMillisV1 {
        UnixMillisV1::new(self.consumed_at_unix_ms)
    }
}

impl std::fmt::Debug for ConsumedLocalApprovalDecisionV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsumedLocalApprovalDecisionV1")
            .field("request_id", &self.request_id)
            .field("decision", &self.decision_evidence.decision)
            .field("consumed_at_unix_ms", &self.consumed_at_unix_ms)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LocalApprovalRequestStoreErrorV1 {
    #[error("approval request validation failed: {0}")]
    Request(#[from] LocalApprovalErrorV1),
    #[error("approval submission admission failed: {0}")]
    Admission(#[from] LocalApprovalAdmissionErrorV1),
    #[error("pending approval store mutex is poisoned")]
    StorePoisoned,
    #[error("request belongs to another daemon incarnation")]
    DaemonIncarnationMismatch,
    #[error("exact request is already pending")]
    DuplicateRequest,
    #[error("approval request is not pending in this live daemon store")]
    RequestNotPending,
    #[error("pending request disappeared while its consume mutex was held")]
    RequestDisappearedDuringConsume,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::authorization::NixActionIntentV1;
    use crate::action::executor::NixOSCommand;
    use crate::action::VerifiedLocalUnixPeerCredentialV1;
    use std::sync::{Arc, Barrier};
    use std::thread;

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

    fn request_for(
        store: &LocalApprovalRequestStoreV1,
        nonce_byte: u8,
    ) -> PendingNixApprovalRequestV1 {
        PendingNixApprovalRequestV1::from_intent(
            &intent(),
            store.daemon_incarnation_ref().to_string(),
            "nixos-rebuild switch --flake .#workstation",
            "local-human-v1",
            ms(1_000),
            ms(2_000),
            [nonce_byte; 32],
        )
        .unwrap()
    }

    fn peer(uid: u32, pid: u32) -> VerifiedLocalUnixPeerCredentialV1 {
        VerifiedLocalUnixPeerCredentialV1::from_kernel_peer_observation(
            uid,
            100,
            Some(pid),
            format!("unix-socket-instance:test-{pid}"),
            ms(1_100),
        )
        .unwrap()
    }

    #[test]
    fn store_is_bound_to_live_daemon_incarnation() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = LocalApprovalRequestStoreV1::new(&daemon);
        let mut request = request_for(&store, 1);
        request.daemon_incarnation_id = format!("{}-old", store.daemon_incarnation_ref());

        assert_eq!(
            store.install_pending(request).unwrap_err(),
            LocalApprovalRequestStoreErrorV1::DaemonIncarnationMismatch
        );
        assert_eq!(store.pending_count().unwrap(), 0);
    }

    #[test]
    fn duplicate_exact_request_is_rejected() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = LocalApprovalRequestStoreV1::new(&daemon);
        let request = request_for(&store, 1);
        store.install_pending(request.clone()).unwrap();

        assert_eq!(
            store.install_pending(request).unwrap_err(),
            LocalApprovalRequestStoreErrorV1::DuplicateRequest
        );
        assert_eq!(store.pending_count().unwrap(), 1);
    }

    #[test]
    fn newer_request_for_same_intent_atomically_supersedes_old_request() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = LocalApprovalRequestStoreV1::new(&daemon);
        let old_request = request_for(&store, 1);
        let old_id = old_request.request_id().unwrap();
        let old_submission = LocalApprovalSubmissionV1::for_request(
            &old_request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        store.install_pending(old_request).unwrap();

        let new_request = request_for(&store, 2);
        let new_id = new_request.request_id().unwrap();
        let install = store.install_pending(new_request.clone()).unwrap();
        assert_eq!(install.request_id, new_id);
        assert_eq!(install.superseded_request_ids, vec![old_id.clone()]);
        assert!(!store.is_pending(&old_id).unwrap());
        assert!(store.is_pending(&new_id).unwrap());

        assert_eq!(
            store
                .consume_verified_submission(&old_submission, &peer(1000, 1), ms(1_300))
                .unwrap_err(),
            LocalApprovalRequestStoreErrorV1::RequestNotPending
        );

        let new_submission = LocalApprovalSubmissionV1::for_request(
            &new_request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let consumed = store
            .consume_verified_submission(&new_submission, &peer(1000, 1), ms(1_300))
            .unwrap();
        assert_eq!(consumed.request_id(), new_id);
    }

    #[test]
    fn successful_consume_removes_request_and_returns_live_token() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = LocalApprovalRequestStoreV1::new(&daemon);
        let request = request_for(&store, 1);
        let request_id = request.request_id().unwrap();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        store.install_pending(request).unwrap();

        let consumed = store
            .consume_verified_submission(&submission, &peer(1000, 1), ms(1_300))
            .unwrap();

        assert_eq!(consumed.request_id(), request_id);
        assert_eq!(consumed.decision_kind(), LocalApprovalDecisionKindV1::Approved);
        assert_eq!(consumed.consumed_at(), ms(1_300));
        assert_eq!(store.pending_count().unwrap(), 0);
        assert!(!store.is_pending(&request_id).unwrap());
    }

    #[test]
    fn invalid_submission_does_not_consume_legitimate_pending_request() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = LocalApprovalRequestStoreV1::new(&daemon);
        let request = request_for(&store, 1);
        let request_id = request.request_id().unwrap();
        let valid_submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let mut hostile = valid_submission.clone();
        hostile.action_intent_digest = "11".repeat(32);
        store.install_pending(request).unwrap();

        assert!(matches!(
            store.consume_verified_submission(&hostile, &peer(1000, 1), ms(1_300)),
            Err(LocalApprovalRequestStoreErrorV1::Admission(
                LocalApprovalAdmissionErrorV1::Approval(LocalApprovalErrorV1::IntentMismatch)
            ))
        ));
        assert!(store.is_pending(&request_id).unwrap());

        store
            .consume_verified_submission(&valid_submission, &peer(1000, 1), ms(1_300))
            .unwrap();
        assert!(!store.is_pending(&request_id).unwrap());
    }

    #[test]
    fn denial_is_terminal_and_consumes_pending_request() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = LocalApprovalRequestStoreV1::new(&daemon);
        let request = request_for(&store, 1);
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Denied,
            ms(1_200),
        )
        .unwrap();
        store.install_pending(request).unwrap();

        let consumed = store
            .consume_verified_submission(&submission, &peer(1000, 1), ms(1_300))
            .unwrap();
        assert_eq!(consumed.decision_kind(), LocalApprovalDecisionKindV1::Denied);
        assert_eq!(store.pending_count().unwrap(), 0);
    }

    #[test]
    fn concurrent_duplicate_consumers_cannot_both_succeed() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = Arc::new(LocalApprovalRequestStoreV1::new(&daemon));
        let request = request_for(&store, 1);
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        store.install_pending(request).unwrap();

        let barrier = Arc::new(Barrier::new(3));
        let mut handles = Vec::new();
        for (uid, pid) in [(1000, 10), (1001, 11)] {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let submission = submission.clone();
            handles.push(thread::spawn(move || {
                let peer = peer(uid, pid);
                barrier.wait();
                store.consume_verified_submission(&submission, &peer, ms(1_300))
            }));
        }
        barrier.wait();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|r| matches!(r, Err(LocalApprovalRequestStoreErrorV1::RequestNotPending)))
                .count(),
            1
        );
        assert_eq!(store.pending_count().unwrap(), 0);
    }
}
