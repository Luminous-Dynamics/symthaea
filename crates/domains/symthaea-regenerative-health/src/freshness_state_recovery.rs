// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Crash-consistent receiver freshness-state recovery.
//!
//! The recovery record is rollbackable storage. The anchor is deliberately a
//! separate deployment boundary and must be backed by a rollback-resistant
//! monotonic mechanism or trusted authority. A content hash detects mutation;
//! it does not by itself prevent rollback.

use serde::{Deserialize, Serialize};

use crate::freshness_anchor_assurance::{
    FreshnessAnchorAssuranceError, FreshnessAnchorCapabilities, FreshnessAnchorProfile,
    VerifiedFreshnessAnchor,
};
use crate::freshness_reconciliation::{FreshnessAcceptancePolicy, FreshnessAcceptanceState};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessRecoveryRecord {
    pub schema_version: String,
    pub receiver_id: String,
    pub policy_fingerprint: String,
    pub generation: u64,
    pub state: FreshnessAcceptanceState,
    pub state_fingerprint: String,
    pub previous_state_fingerprint: Option<String>,
    pub authority_reference: String,
    pub authority_statement_digest: String,
    pub authentication_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessRecoveryAnchor {
    pub schema_version: String,
    pub receiver_id: String,
    pub policy_fingerprint: String,
    pub generation: u64,
    pub state_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessRecoveryIssue {
    InvalidRecord,
    ReceiverMismatch,
    PolicyMismatch,
    StateFingerprintMismatch,
    AnchorMismatch,
    GenerationNotMonotonic,
    PreviousStateMismatch,
    AuthenticationFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessRecoveryOutcome {
    Restored,
    RestoredConflicted,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessRecoveryDecision {
    pub outcome: FreshnessRecoveryOutcome,
    pub state: Option<FreshnessAcceptanceState>,
    pub issues: Vec<FreshnessRecoveryIssue>,
}

/// Authentication is intentionally abstract. Deployments may bind this to a
/// signature, MAC, TPM-backed key, remote authority, or another trust anchor.
pub trait FreshnessRecoveryAuthenticator {
    fn authenticate(&self, record: &FreshnessRecoveryRecord) -> bool;
}

/// The external anchor is intentionally abstract. Its implementation must
/// provide an atomic compare-and-swap operation over the complete anchor.
pub trait FreshnessRecoveryAnchorStore {
    fn load(&self) -> Option<FreshnessRecoveryAnchor>;
    fn compare_and_swap(
        &self,
        expected: Option<&FreshnessRecoveryAnchor>,
        next: &FreshnessRecoveryAnchor,
    ) -> bool;

    /// Existing implementations default to non-authoritative software
    /// persistence. Backends must opt in by declaring all required properties.
    fn profile(&self) -> FreshnessAnchorProfile {
        FreshnessAnchorProfile::new(
            crate::freshness_anchor_assurance::FreshnessAnchorBacking::SoftwareOnly,
            FreshnessAnchorCapabilities::software_only(),
            "software-only"
        )
        .expect("static software-only anchor profile must be valid")
    }
}

impl FreshnessRecoveryRecord {
    pub fn new(
        receiver_id: impl Into<String>,
        policy: &FreshnessAcceptancePolicy,
        generation: u64,
        state: FreshnessAcceptanceState,
        previous_state_fingerprint: Option<String>,
        authority_reference: impl Into<String>,
        authority_statement_digest: impl Into<String>,
        authentication_binding: impl Into<String>,
    ) -> Result<Self, &'static str> {
        policy.validate()?;
        let receiver_id = receiver_id.into();
        let authority_reference = authority_reference.into();
        let authority_statement_digest = authority_statement_digest.into();
        let authentication_binding = authentication_binding.into();

        if receiver_id.trim().is_empty()
            || authority_reference.trim().is_empty()
            || authority_statement_digest.trim().is_empty()
            || authentication_binding.trim().is_empty()
        {
            return Err("incomplete recovery identity or authority binding");
        }
        if state.policy_fingerprint != policy.fingerprint() {
            return Err("state policy fingerprint mismatch");
        }

        Ok(Self {
            schema_version: "0.1".into(),
            receiver_id,
            policy_fingerprint: policy.fingerprint(),
            generation,
            state_fingerprint: state.fingerprint(),
            state,
            previous_state_fingerprint,
            authority_reference,
            authority_statement_digest,
            authentication_binding,
        })
    }

    pub fn anchor(&self) -> FreshnessRecoveryAnchor {
        FreshnessRecoveryAnchor {
            schema_version: self.schema_version.clone(),
            receiver_id: self.receiver_id.clone(),
            policy_fingerprint: self.policy_fingerprint.clone(),
            generation: self.generation,
            state_fingerprint: self.state_fingerprint.clone(),
        }
    }

    pub fn recover<A: FreshnessRecoveryAuthenticator>(
        &self,
        policy: &FreshnessAcceptancePolicy,
        anchor: &FreshnessRecoveryAnchor,
        authenticator: &A,
    ) -> FreshnessRecoveryDecision {
        let mut issues = Vec::new();
        let policy_fingerprint = policy.fingerprint();

        if self.schema_version != "0.1"
            || self.receiver_id.trim().is_empty()
            || self.authority_reference.trim().is_empty()
            || self.authority_statement_digest.trim().is_empty()
            || self.authentication_binding.trim().is_empty()
        {
            issues.push(FreshnessRecoveryIssue::InvalidRecord);
        }
        if self.receiver_id != anchor.receiver_id {
            issues.push(FreshnessRecoveryIssue::ReceiverMismatch);
        }
        if self.policy_fingerprint != policy_fingerprint
            || anchor.policy_fingerprint != policy_fingerprint
            || self.state.policy_fingerprint != policy_fingerprint
        {
            issues.push(FreshnessRecoveryIssue::PolicyMismatch);
        }
        if self.state.fingerprint() != self.state_fingerprint {
            issues.push(FreshnessRecoveryIssue::StateFingerprintMismatch);
        }
        if self.schema_version != anchor.schema_version
            || self.generation != anchor.generation
            || self.state_fingerprint != anchor.state_fingerprint
        {
            issues.push(FreshnessRecoveryIssue::AnchorMismatch);
        }
        if !authenticator.authenticate(self) {
            issues.push(FreshnessRecoveryIssue::AuthenticationFailure);
        }

        if !issues.is_empty() {
            return FreshnessRecoveryDecision {
                outcome: FreshnessRecoveryOutcome::Quarantined,
                state: None,
                issues,
            };
        }

        FreshnessRecoveryDecision {
            outcome: if self.state.status_conflicted() {
                FreshnessRecoveryOutcome::RestoredConflicted
            } else {
                FreshnessRecoveryOutcome::Restored
            },
            state: Some(self.state.clone()),
            issues,
        }
    }
}

pub fn prepare_record(
    receiver_id: impl Into<String>,
    policy: &FreshnessAcceptancePolicy,
    generation: u64,
    state: FreshnessAcceptanceState,
    previous_state_fingerprint: Option<String>,
    authority_reference: impl Into<String>,
    authority_statement_digest: impl Into<String>,
    authentication_binding: impl Into<String>,
) -> Result<FreshnessRecoveryRecord, &'static str> {
    FreshnessRecoveryRecord::new(
        receiver_id,
        policy,
        generation,
        state,
        previous_state_fingerprint,
        authority_reference,
        authority_statement_digest,
        authentication_binding,
    )
}

/// Commit an anchor using the mechanical store interface only.
///
/// This validates the recovery chain but does not establish that the backing
/// store is rollback-resistant. Use commit_authoritative_anchor when the
/// anchor is intended to authorize recovery after receiver rollback.
pub fn commit_anchor<S: FreshnessRecoveryAnchorStore>(
    store: &S,
    expected: Option<&FreshnessRecoveryAnchor>,
    record: &FreshnessRecoveryRecord,
) -> Result<(), &'static str> {
    if let Some(expected) = expected {
        if record.generation <= expected.generation {
            return Err("recovery generation is not strictly monotonic");
        }
        if record.previous_state_fingerprint.as_deref()
            != Some(expected.state_fingerprint.as_str())
        {
            return Err("recovery predecessor fingerprint mismatch");
        }
    } else if record.generation != 0 {
        return Err("initial recovery generation must be zero");
    }

    let next = record.anchor();
    if store.compare_and_swap(expected, &next) {
        Ok(())
    } else {
        Err("recovery anchor compare-and-swap failed")
    }
}

/// Commit an anchor only when the backing store explicitly satisfies the
/// authoritative anti-rollback contract.
pub fn commit_authoritative_anchor<S: FreshnessRecoveryAnchorStore>(
    store: &S,
    verified_anchor: &VerifiedFreshnessAnchor,
    expected: Option<&FreshnessRecoveryAnchor>,
    record: &FreshnessRecoveryRecord,
) -> Result<(), FreshnessAnchorAssuranceError> {
    let store_profile = store.profile();
    let verified_profile = verified_anchor.profile();

    if store_profile.schema_version != verified_profile.schema_version
        || store_profile.fingerprint().map_err(|_| FreshnessAnchorAssuranceError::InvalidProfile)?
            != verified_profile.fingerprint().map_err(|_| FreshnessAnchorAssuranceError::InvalidProfile)?
    {
        return Err(FreshnessAnchorAssuranceError::ProfileBindingMismatch);
    }

    verified_profile.require_authoritative()?;
    commit_anchor(store, expected, record)
        .map_err(|message| FreshnessAnchorAssuranceError::CommitRejected(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freshness_reconciliation::FreshnessMarker;

    struct Accept;
    impl FreshnessRecoveryAuthenticator for Accept {
        fn authenticate(&self, _: &FreshnessRecoveryRecord) -> bool { true }
    }

    struct Reject;
    impl FreshnessRecoveryAuthenticator for Reject {
        fn authenticate(&self, _: &FreshnessRecoveryRecord) -> bool { false }
    }

    fn policy() -> FreshnessAcceptancePolicy {
        FreshnessAcceptancePolicy {
            schema_version: "0.1".into(),
            policy_id: "freshness-window-v1".into(),
            expected_scheme: "epoch-marker-v1".into(),
            expected_source_id: "epoch-bell-1".into(),
            minimum_accepted_epoch: 7,
            acceptance_window: 2,
        }
    }

    fn state() -> FreshnessAcceptanceState {
        FreshnessAcceptanceState::new(&policy()).unwrap()
    }

    fn record(generation: u64, state: FreshnessAcceptanceState, previous: Option<String>) -> FreshnessRecoveryRecord {
        let p = policy();
        FreshnessRecoveryRecord::new(
            "receiver-1", &p, generation, state, previous,
            "authority-1", "statement-1", "binding-1",
        ).unwrap()
    }

    #[test]
    fn exact_anchor_restores() {
        let r = record(0, state(), None);
        let d = r.recover(&policy(), &r.anchor(), &Accept);
        assert_eq!(d.outcome, FreshnessRecoveryOutcome::Restored);
        assert_eq!(d.state, Some(r.state));
    }

    #[test]
    fn modified_state_is_rejected() {
        let r = record(0, state(), None);
        let mut modified = r.clone();
        modified.state.highest_accepted_epoch = Some(99);
        let d = modified.recover(&policy(), &r.anchor(), &Accept);
        assert_eq!(d.outcome, FreshnessRecoveryOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessRecoveryIssue::StateFingerprintMismatch));
    }

    #[test]
    fn old_anchor_is_rejected() {
        let r = record(3, state(), Some("previous".into()));
        let mut anchor = r.anchor();
        anchor.generation = 2;
        let d = r.recover(&policy(), &anchor, &Accept);
        assert_eq!(d.outcome, FreshnessRecoveryOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessRecoveryIssue::AnchorMismatch));
    }

    #[test]
    fn unauthenticated_record_is_rejected() {
        let r = record(0, state(), None);
        let d = r.recover(&policy(), &r.anchor(), &Reject);
        assert_eq!(d.outcome, FreshnessRecoveryOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessRecoveryIssue::AuthenticationFailure));
    }

    #[test]
    fn conflict_survives_recovery() {
        let p = policy();
        let s = state();
        let a = FreshnessMarker {
            scheme: p.expected_scheme.clone(),
            source_id: p.expected_source_id.clone(),
            epoch: 7,
            marker_digest: "a".into(),
        };
        let b = FreshnessMarker { marker_digest: "b".into(), ..a.clone() };
        let s = s.apply(&p, &a).state.apply(&p, &b).state;
        let r = record(0, s, None);
        let d = r.recover(&p, &r.anchor(), &Accept);
        assert_eq!(d.outcome, FreshnessRecoveryOutcome::RestoredConflicted);
        assert!(d.state.unwrap().status_conflicted());
    }

    #[test]
    fn non_monotonic_generation_is_rejected() {
        let p = policy();
        let base = record(4, state(), None);
        let next = record(4, state(), Some(base.state_fingerprint.clone()));
        struct Store;
        impl FreshnessRecoveryAnchorStore for Store {
            fn load(&self) -> Option<FreshnessRecoveryAnchor> { None }
            fn compare_and_swap(
                &self,
                _: Option<&FreshnessRecoveryAnchor>,
                _: &FreshnessRecoveryAnchor,
            ) -> bool { true }
        }
        assert_eq!(
            commit_anchor(&Store, Some(&base.anchor()), &next),
            Err("recovery generation is not strictly monotonic")
        );
    }
    #[test]
    fn software_only_store_cannot_make_authoritative_commit() {
        struct SoftwareStore;
        impl FreshnessRecoveryAnchorStore for SoftwareStore {
            fn load(&self) -> Option<FreshnessRecoveryAnchor> { None }
            fn compare_and_swap(
                &self,
                _: Option<&FreshnessRecoveryAnchor>,
                _: &FreshnessRecoveryAnchor,
            ) -> bool { true }
        }

        let r = record(0, state(), None);
        let profile = SoftwareStore.profile();
        let receipt = crate::freshness_anchor_assurance::FreshnessAnchorVerificationReceipt {
            schema_version: "0.1".into(),
            profile_fingerprint: profile.fingerprint().unwrap(),
            verifier_reference: "verifier-1".into(),
            evidence_reference: "evidence-1".into(),
            evidence_digest: "digest-1".into(),
        };
        struct Accept;
        impl crate::freshness_anchor_assurance::FreshnessAnchorEvidenceVerifier for Accept {
            fn verify(
                &self,
                _: &FreshnessAnchorProfile,
                _: &crate::freshness_anchor_assurance::FreshnessAnchorVerificationReceipt,
            ) -> bool { true }
        }
        assert!(VerifiedFreshnessAnchor::verify(profile, receipt, &Accept).is_err());
        let err = FreshnessAnchorAssuranceError::InvalidProfile;
        assert_eq!(
            err,
            FreshnessAnchorAssuranceError::InsufficientCapabilities {
                missing: vec![
                    crate::freshness_anchor_assurance::FreshnessAnchorCapability::IntegrityProtection,
                    crate::freshness_anchor_assurance::FreshnessAnchorCapability::Authentication,
                    crate::freshness_anchor_assurance::FreshnessAnchorCapability::Monotonicity,
                    crate::freshness_anchor_assurance::FreshnessAnchorCapability::RollbackResistance,
                    crate::freshness_anchor_assurance::FreshnessAnchorCapability::AtomicUpdate,
                    crate::freshness_anchor_assurance::FreshnessAnchorCapability::CrashPersistence,
                ],
            }
        );
    }

    #[test]
    fn fully_capable_store_can_make_authoritative_commit() {
        struct AuthoritativeStore;
        impl FreshnessRecoveryAnchorStore for AuthoritativeStore {
            fn load(&self) -> Option<FreshnessRecoveryAnchor> { None }
            fn compare_and_swap(
                &self,
                _: Option<&FreshnessRecoveryAnchor>,
                _: &FreshnessRecoveryAnchor,
            ) -> bool { true }
            fn profile(&self) -> FreshnessAnchorProfile {
                FreshnessAnchorProfile::new(
                    crate::freshness_anchor_assurance::FreshnessAnchorBacking::RemoteAuthority,
                    FreshnessAnchorCapabilities::authoritative(),
                    "remote://authority-a",
                ).unwrap()
            }
        }

        struct Accept;
        impl crate::freshness_anchor_assurance::FreshnessAnchorEvidenceVerifier for Accept {
            fn verify(
                &self,
                _: &FreshnessAnchorProfile,
                _: &crate::freshness_anchor_assurance::FreshnessAnchorVerificationReceipt,
            ) -> bool { true }
        }

        let profile = AuthoritativeStore.profile();
        let receipt = crate::freshness_anchor_assurance::FreshnessAnchorVerificationReceipt {
            schema_version: "0.1".into(),
            profile_fingerprint: profile.fingerprint().unwrap(),
            verifier_reference: "verifier-1".into(),
            evidence_reference: "evidence-1".into(),
            evidence_digest: "digest-1".into(),
        };
        let verified = VerifiedFreshnessAnchor::verify(profile, receipt, &Accept).unwrap();
        let r = record(0, state(), None);
        assert!(commit_authoritative_anchor(&AuthoritativeStore, &verified, None, &r).is_ok());
    }

}
