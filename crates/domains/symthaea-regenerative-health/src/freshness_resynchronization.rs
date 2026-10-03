// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit authority-driven freshness resynchronization.
//!
//! Resynchronization is deliberately not ordinary marker acceptance. It is an
//! authenticated lifecycle transition that may replace a conflicted receiver
//! state only when a trusted authority explicitly binds the replacement to
//! the receiver's prior state.

use serde::{Deserialize, Serialize};

use crate::freshness_reconciliation::{FreshnessAcceptancePolicy, FreshnessAcceptanceState};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessResynchronizationRequest {
    pub schema_version: String,
    pub receiver_id: String,
    pub policy_fingerprint: String,
    pub current_generation: u64,
    pub current_state_fingerprint: String,
    pub replacement_generation: u64,
    pub replacement_state: FreshnessAcceptanceState,
    pub replacement_state_fingerprint: String,
    pub authority_reference: String,
    pub authority_statement_digest: String,
    pub reason_code: String,
    pub authentication_binding: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessResynchronizationIssue {
    InvalidRequest,
    ReceiverMismatch,
    PolicyMismatch,
    CurrentStateMismatch,
    ReplacementFingerprintMismatch,
    GenerationNotMonotonic,
    EmptyAuthorityBinding,
    AuthenticationFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessResynchronizationOutcome {
    Applied,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessResynchronizationDecision {
    pub outcome: FreshnessResynchronizationOutcome,
    pub state: Option<FreshnessAcceptanceState>,
    pub issues: Vec<FreshnessResynchronizationIssue>,
}

pub trait FreshnessResynchronizationAuthenticator {
    fn authenticate(&self, request: &FreshnessResynchronizationRequest) -> bool;
}

impl FreshnessResynchronizationRequest {
    pub fn new(
        receiver_id: impl Into<String>,
        policy: &FreshnessAcceptancePolicy,
        current_generation: u64,
        current_state_fingerprint: impl Into<String>,
        replacement_generation: u64,
        replacement_state: FreshnessAcceptanceState,
        authority_reference: impl Into<String>,
        authority_statement_digest: impl Into<String>,
        reason_code: impl Into<String>,
        authentication_binding: impl Into<String>,
    ) -> Result<Self, &'static str> {
        policy.validate()?;
        let receiver_id = receiver_id.into();
        let current_state_fingerprint = current_state_fingerprint.into();
        let authority_reference = authority_reference.into();
        let authority_statement_digest = authority_statement_digest.into();
        let reason_code = reason_code.into();
        let authentication_binding = authentication_binding.into();
        if receiver_id.trim().is_empty()
            || current_state_fingerprint.trim().is_empty()
            || authority_reference.trim().is_empty()
            || authority_statement_digest.trim().is_empty()
            || reason_code.trim().is_empty()
            || authentication_binding.trim().is_empty()
        {
            return Err("incomplete resynchronization binding");
        }
        if replacement_state.policy_fingerprint != policy.fingerprint() {
            return Err("replacement state policy fingerprint mismatch");
        }
        if replacement_generation <= current_generation {
            return Err("replacement generation is not strictly monotonic");
        }
        let replacement_state_fingerprint = replacement_state.fingerprint();
        Ok(Self {
            schema_version: "0.1".into(),
            receiver_id,
            policy_fingerprint: policy.fingerprint(),
            current_generation,
            current_state_fingerprint,
            replacement_generation,
            replacement_state,
            replacement_state_fingerprint,
            authority_reference,
            authority_statement_digest,
            reason_code,
            authentication_binding,
        })
    }

    pub fn apply<A: FreshnessResynchronizationAuthenticator>(
        &self,
        policy: &FreshnessAcceptancePolicy,
        receiver_id: &str,
        current_generation: u64,
        current_state_fingerprint: &str,
        authenticator: &A,
    ) -> FreshnessResynchronizationDecision {
        let mut issues = Vec::new();
        let policy_fingerprint = policy.fingerprint();

        if self.schema_version != "0.1"
            || self.receiver_id.trim().is_empty()
            || self.current_state_fingerprint.trim().is_empty()
            || self.authority_reference.trim().is_empty()
            || self.authority_statement_digest.trim().is_empty()
            || self.reason_code.trim().is_empty()
            || self.authentication_binding.trim().is_empty()
        {
            issues.push(FreshnessResynchronizationIssue::InvalidRequest);
        }
        if self.receiver_id != receiver_id {
            issues.push(FreshnessResynchronizationIssue::ReceiverMismatch);
        }
        if self.policy_fingerprint != policy_fingerprint
            || self.replacement_state.policy_fingerprint != policy_fingerprint
        {
            issues.push(FreshnessResynchronizationIssue::PolicyMismatch);
        }
        if self.current_generation != current_generation
            || self.current_state_fingerprint != current_state_fingerprint
        {
            issues.push(FreshnessResynchronizationIssue::CurrentStateMismatch);
        }
        if self.replacement_state.fingerprint() != self.replacement_state_fingerprint {
            issues.push(FreshnessResynchronizationIssue::ReplacementFingerprintMismatch);
        }
        if self.replacement_generation <= current_generation {
            issues.push(FreshnessResynchronizationIssue::GenerationNotMonotonic);
        }
        if self.authority_reference.trim().is_empty()
            || self.authority_statement_digest.trim().is_empty()
            || self.authentication_binding.trim().is_empty()
        {
            issues.push(FreshnessResynchronizationIssue::EmptyAuthorityBinding);
        }
        if !authenticator.authenticate(self) {
            issues.push(FreshnessResynchronizationIssue::AuthenticationFailure);
        }

        if !issues.is_empty() {
            return FreshnessResynchronizationDecision {
                outcome: FreshnessResynchronizationOutcome::Quarantined,
                state: None,
                issues,
            };
        }

        FreshnessResynchronizationDecision {
            outcome: FreshnessResynchronizationOutcome::Applied,
            state: Some(self.replacement_state.clone()),
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freshness_reconciliation::FreshnessMarker;

    struct Accept;
    impl FreshnessResynchronizationAuthenticator for Accept {
        fn authenticate(&self, _: &FreshnessResynchronizationRequest) -> bool { true }
    }

    struct Reject;
    impl FreshnessResynchronizationAuthenticator for Reject {
        fn authenticate(&self, _: &FreshnessResynchronizationRequest) -> bool { false }
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

    #[test]
    fn authorized_resync_can_replace_conflicted_state() {
        let p = policy();
        let current = state()
            .apply(&p, &FreshnessMarker {
                scheme: p.expected_scheme.clone(),
                source_id: p.expected_source_id.clone(),
                epoch: 7,
                marker_digest: "a".into(),
            }).state
            .apply(&p, &FreshnessMarker {
                scheme: p.expected_scheme.clone(),
                source_id: p.expected_source_id.clone(),
                epoch: 7,
                marker_digest: "b".into(),
            }).state;
        assert!(current.status_conflicted());

        let request = FreshnessResynchronizationRequest::new(
            "receiver-1", &p, 4, current.fingerprint(), 5, state(),
            "authority-1", "statement-5", "trusted-resync", "binding-5",
        ).unwrap();
        let d = request.apply(&p, "receiver-1", 4, &current.fingerprint(), &Accept);
        assert_eq!(d.outcome, FreshnessResynchronizationOutcome::Applied);
        assert!(!d.state.unwrap().status_conflicted());
    }

    #[test]
    fn stale_resync_cannot_overwrite_current_state() {
        let p = policy();
        let current = state();
        let request = FreshnessResynchronizationRequest::new(
            "receiver-1", &p, 4, current.fingerprint(), 5, state(),
            "authority-1", "statement-5", "trusted-resync", "binding-5",
        ).unwrap();
        let d = request.apply(&p, "receiver-1", 5, &current.fingerprint(), &Accept);
        assert_eq!(d.outcome, FreshnessResynchronizationOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessResynchronizationIssue::CurrentStateMismatch));
    }

    #[test]
    fn replacement_mutation_is_rejected() {
        let p = policy();
        let current = state();
        let mut request = FreshnessResynchronizationRequest::new(
            "receiver-1", &p, 4, current.fingerprint(), 5, state(),
            "authority-1", "statement-5", "trusted-resync", "binding-5",
        ).unwrap();
        request.replacement_state.highest_accepted_epoch = Some(99);
        let d = request.apply(&p, "receiver-1", 4, &current.fingerprint(), &Accept);
        assert_eq!(d.outcome, FreshnessResynchronizationOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessResynchronizationIssue::ReplacementFingerprintMismatch));
    }

    #[test]
    fn unauthenticated_resync_is_rejected() {
        let p = policy();
        let current = state();
        let request = FreshnessResynchronizationRequest::new(
            "receiver-1", &p, 4, current.fingerprint(), 5, state(),
            "authority-1", "statement-5", "trusted-resync", "binding-5",
        ).unwrap();
        let d = request.apply(&p, "receiver-1", 4, &current.fingerprint(), &Reject);
        assert_eq!(d.outcome, FreshnessResynchronizationOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessResynchronizationIssue::AuthenticationFailure));
    }

    #[test]
    fn generation_must_advance() {
        let p = policy();
        let current = state();
        assert!(FreshnessResynchronizationRequest::new(
            "receiver-1", &p, 4, current.fingerprint(), 4, state(),
            "authority-1", "statement-4", "trusted-resync", "binding-4",
        ).is_err());
    }
}
