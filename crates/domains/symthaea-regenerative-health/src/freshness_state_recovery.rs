// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Authenticated recovery of receiver-local freshness state.
//!
//! The recovery record is separate from the evidence envelope. Its hash
//! provides integrity; deployment supplies authentication and a rollback-
//! resistant anchor (for example protected NVRAM, a TPM-backed counter, or a
//! trusted higher-level service). A hash alone is not authentication.

use serde::{Deserialize, Serialize};

use crate::freshness_reconciliation::{FreshnessAcceptancePolicy, FreshnessAcceptanceState};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessStateRecoveryRecord {
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
pub struct FreshnessStateRecoveryAnchor {
    pub schema_version: String,
    pub receiver_id: String,
    pub policy_fingerprint: String,
    pub generation: u64,
    pub state_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessStateRecoveryOutcome {
    Restored,
    RestoredConflicted,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessStateRecoveryIssue {
    InvalidRecord,
    ReceiverMismatch,
    PolicyMismatch,
    StateFingerprintMismatch,
    AnchorMismatch,
    AuthenticationFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessStateRecoveryDecision {
    pub outcome: FreshnessStateRecoveryOutcome,
    pub state: Option<FreshnessAcceptanceState>,
    pub issues: Vec<FreshnessStateRecoveryIssue>,
}

/// Authentication is a deployment boundary: TPM, signature, MAC, or remote
/// authority implementations can be supplied without changing freshness
/// semantics.
pub trait FreshnessStateRecoveryAuthenticator {
    fn authenticate(&self, record: &FreshnessStateRecoveryRecord) -> bool;
}

impl FreshnessStateRecoveryRecord {
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
        let policy_fingerprint = policy.fingerprint();
        if state.policy_fingerprint != policy_fingerprint {
            return Err("state policy fingerprint mismatch");
        }
        let state_fingerprint = state.fingerprint();
        Ok(Self {
            schema_version: "0.1".into(),
            receiver_id,
            policy_fingerprint,
            generation,
            state,
            state_fingerprint,
            previous_state_fingerprint,
            authority_reference,
            authority_statement_digest,
            authentication_binding,
        })
    }

    pub fn anchor(&self) -> FreshnessStateRecoveryAnchor {
        FreshnessStateRecoveryAnchor {
            schema_version: self.schema_version.clone(),
            receiver_id: self.receiver_id.clone(),
            policy_fingerprint: self.policy_fingerprint.clone(),
            generation: self.generation,
            state_fingerprint: self.state_fingerprint.clone(),
        }
    }

    pub fn recover<A: FreshnessStateRecoveryAuthenticator>(
        &self,
        policy: &FreshnessAcceptancePolicy,
        anchor: &FreshnessStateRecoveryAnchor,
        authenticator: &A,
    ) -> FreshnessStateRecoveryDecision {
        let mut issues = Vec::new();
        let policy_fingerprint = policy.fingerprint();
        if self.schema_version != "0.1"
            || self.receiver_id.trim().is_empty()
            || self.authority_reference.trim().is_empty()
            || self.authority_statement_digest.trim().is_empty()
            || self.authentication_binding.trim().is_empty()
        {
            issues.push(FreshnessStateRecoveryIssue::InvalidRecord);
        }
        if self.receiver_id != anchor.receiver_id {
            issues.push(FreshnessStateRecoveryIssue::ReceiverMismatch);
        }
        if self.policy_fingerprint != policy_fingerprint
            || anchor.policy_fingerprint != policy_fingerprint
        {
            issues.push(FreshnessStateRecoveryIssue::PolicyMismatch);
        }
        if self.state.policy_fingerprint != policy_fingerprint
            || self.state.fingerprint() != self.state_fingerprint
        {
            issues.push(FreshnessStateRecoveryIssue::StateFingerprintMismatch);
        }
        if self.schema_version != anchor.schema_version
            || self.generation != anchor.generation
            || self.state_fingerprint != anchor.state_fingerprint
        {
            issues.push(FreshnessStateRecoveryIssue::AnchorMismatch);
        }
        if !authenticator.authenticate(self) {
            issues.push(FreshnessStateRecoveryIssue::AuthenticationFailure);
        }
        if !issues.is_empty() {
            return FreshnessStateRecoveryDecision {
                outcome: FreshnessStateRecoveryOutcome::Quarantined,
                state: None,
                issues,
            };
        }
        FreshnessStateRecoveryDecision {
            outcome: if self.state.status_conflicted() {
                FreshnessStateRecoveryOutcome::RestoredConflicted
            } else {
                FreshnessStateRecoveryOutcome::Restored
            },
            state: Some(self.state.clone()),
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freshness_reconciliation::{
        FreshnessAcceptancePolicy, FreshnessMarker,
    };

    struct AcceptAll;
    impl FreshnessStateRecoveryAuthenticator for AcceptAll {
        fn authenticate(&self, _: &FreshnessStateRecoveryRecord) -> bool { true }
    }
    struct Reject;
    impl FreshnessStateRecoveryAuthenticator for Reject {
        fn authenticate(&self, _: &FreshnessStateRecoveryRecord) -> bool { false }
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

    fn record() -> FreshnessStateRecoveryRecord {
        let p = policy();
        FreshnessStateRecoveryRecord::new(
            "receiver-1", &p, 4, FreshnessAcceptanceState::new(&p).unwrap(),
            None, "authority-1", "statement-1", "binding-1",
        ).unwrap()
    }

    #[test]
    fn exact_anchor_restores() {
        let r = record();
        let d = r.recover(&policy(), &r.anchor(), &AcceptAll);
        assert_eq!(d.outcome, FreshnessStateRecoveryOutcome::Restored);
        assert_eq!(d.state, Some(r.state));
    }

    #[test]
    fn modified_state_is_rejected() {
        let mut r = record();
        r.state.highest_accepted_epoch = Some(99);
        let d = r.recover(&policy(), &r.anchor(), &AcceptAll);
        assert_eq!(d.outcome, FreshnessStateRecoveryOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessStateRecoveryIssue::StateFingerprintMismatch));
    }

    #[test]
    fn older_anchor_is_rejected() {
        let r = record();
        let mut a = r.anchor();
        a.generation -= 1;
        let d = r.recover(&policy(), &a, &AcceptAll);
        assert_eq!(d.outcome, FreshnessStateRecoveryOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessStateRecoveryIssue::AnchorMismatch));
    }

    #[test]
    fn unauthenticated_record_is_rejected() {
        let r = record();
        let d = r.recover(&policy(), &r.anchor(), &Reject);
        assert_eq!(d.outcome, FreshnessStateRecoveryOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessStateRecoveryIssue::AuthenticationFailure));
    }

    #[test]
    fn conflict_survives_recovery() {
        let p = policy();
        let s = FreshnessAcceptanceState::new(&p).unwrap();
        let a = FreshnessMarker {
            scheme: p.expected_scheme.clone(),
            source_id: p.expected_source_id.clone(),
            epoch: 7,
            marker_digest: "a".into(),
        };
        let b = FreshnessMarker { marker_digest: "b".into(), ..a.clone() };
        let s = s.apply(&p, &a).state.apply(&p, &b).state;
        let r = FreshnessStateRecoveryRecord::new(
            "receiver-1", &p, 5, s, None, "authority-1", "statement-1", "binding-1",
        ).unwrap();
        let d = r.recover(&p, &r.anchor(), &AcceptAll);
        assert_eq!(d.outcome, FreshnessStateRecoveryOutcome::RestoredConflicted);
        assert!(d.state.unwrap().status_conflicted());
    }
}
