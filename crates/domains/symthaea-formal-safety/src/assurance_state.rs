// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical, scope-specific assurance state vocabulary.
//!
//! These types deliberately prevent unrelated assurance concepts from collapsing
//! into a single `verified: bool`. Observation, verification, attestation,
//! admission, qualification, authorization, execution, and outcome are separate
//! state families. Moving between them requires explicit caller logic and, where
//! appropriate, independently validated evidence.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// State of information before it has any assurance authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationState {
    Generated,
    Parsed,
    Observed,
    Converged,
}

/// Scoped conclusion about one declared property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyState {
    Unexamined,
    Checked,
    Supported,
    Refuted,
    Proved,
    Indeterminate,
}

/// Cryptographic/identity state of an external attestation.
///
/// `SignatureValid` does not imply signer authorization, and
/// `SignerAuthorized` does not imply that the evidence is admitted or that a
/// claim is qualified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationState {
    Absent,
    Present,
    SignatureValid,
    SignerAuthorized {
        authority_scope: String,
    },
    Invalid,
    Revoked,
    Expired,
}

/// Whether evidence has been admitted by a particular assurance profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceAdmissionState {
    NotAdmitted,
    AdmittedUnderProfile {
        profile_id: String,
    },
    Rejected {
        reason: String,
    },
    Stale,
}

/// Qualification state for a claim under an explicit profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationState {
    Unqualified,
    QualifiedUnderProfile {
        profile_id: String,
    },
    CounterexampleFound,
    Invalidated,
    Stale,
    Indeterminate,
}

/// Authority to perform an action.
///
/// This state is independent of assurance qualification: a qualified claim does
/// not authorize an action, and an authorized action is not thereby safe or
/// correct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionAuthorizationState {
    NotAuthorized,
    Authorized {
        scope: String,
        valid_until_unix_s: Option<u64>,
    },
    Revoked,
    Expired,
}

/// Outcome of an execution. Execution itself is not evidence of success.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionOutcome {
    Unknown,
    Succeeded,
    Failed,
    PartiallySucceeded,
}

/// Whether an authorized/proposed action actually executed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionState {
    NotExecuted,
    Executed {
        execution_id: Uuid,
        outcome: ExecutionOutcome,
    },
}

/// Snapshot of independently scoped assurance states for one subject/action.
///
/// This intentionally has no `is_verified`, `is_safe`, `is_trusted`, or
/// `is_guaranteed` shortcut. Consumers must ask the exact question they mean.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceStateSnapshot {
    pub observation: Option<ObservationState>,
    pub property: PropertyState,
    pub attestation: AttestationState,
    pub admission: EvidenceAdmissionState,
    pub qualification: QualificationState,
    pub authorization: ActionAuthorizationState,
    pub execution: ExecutionState,
}

impl Default for AssuranceStateSnapshot {
    fn default() -> Self {
        Self {
            observation: None,
            property: PropertyState::Unexamined,
            attestation: AttestationState::Absent,
            admission: EvidenceAdmissionState::NotAdmitted,
            qualification: QualificationState::Unqualified,
            authorization: ActionAuthorizationState::NotAuthorized,
            execution: ExecutionState::NotExecuted,
        }
    }
}

impl AssuranceStateSnapshot {
    /// Exact profile-scoped qualification predicate.
    pub fn is_qualified_under(&self, profile_id: &str) -> bool {
        matches!(
            &self.qualification,
            QualificationState::QualifiedUnderProfile { profile_id: qualified }
                if qualified == profile_id
        )
    }

    /// Exact profile-scoped evidence-admission predicate.
    pub fn is_evidence_admitted_under(&self, profile_id: &str) -> bool {
        matches!(
            &self.admission,
            EvidenceAdmissionState::AdmittedUnderProfile { profile_id: admitted }
                if admitted == profile_id
        )
    }

    /// Whether the current action authorization is active at `now_unix_s`.
    ///
    /// This does not inspect or imply qualification, execution, or success.
    pub fn is_action_authorized_at(&self, now_unix_s: u64) -> bool {
        match &self.authorization {
            ActionAuthorizationState::Authorized {
                scope,
                valid_until_unix_s,
            } => {
                !scope.trim().is_empty()
                    && valid_until_unix_s.is_none_or(|deadline| now_unix_s <= deadline)
            }
            _ => false,
        }
    }

    /// Whether an execution receipt is present, regardless of outcome.
    pub fn was_executed(&self) -> bool {
        matches!(self.execution, ExecutionState::Executed { .. })
    }

    /// Whether an execution explicitly reports success.
    ///
    /// `was_executed()` and `execution_succeeded()` are intentionally separate.
    pub fn execution_succeeded(&self) -> bool {
        matches!(
            self.execution,
            ExecutionState::Executed {
                outcome: ExecutionOutcome::Succeeded,
                ..
            }
        )
    }

    /// Structural validation only. This does not establish any cryptographic or
    /// assurance conclusion.
    pub fn validate(&self) -> bool {
        let attestation_valid = match &self.attestation {
            AttestationState::SignerAuthorized { authority_scope } => {
                !authority_scope.trim().is_empty()
            }
            _ => true,
        };
        let admission_valid = match &self.admission {
            EvidenceAdmissionState::AdmittedUnderProfile { profile_id } => {
                !profile_id.trim().is_empty()
            }
            EvidenceAdmissionState::Rejected { reason } => !reason.trim().is_empty(),
            _ => true,
        };
        let qualification_valid = match &self.qualification {
            QualificationState::QualifiedUnderProfile { profile_id } => {
                !profile_id.trim().is_empty()
            }
            _ => true,
        };
        let authorization_valid = match &self.authorization {
            ActionAuthorizationState::Authorized { scope, .. } => !scope.trim().is_empty(),
            _ => true,
        };

        attestation_valid && admission_valid && qualification_valid && authorization_valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_valid_does_not_imply_qualification() {
        let state = AssuranceStateSnapshot {
            attestation: AttestationState::SignatureValid,
            ..Default::default()
        };

        assert!(!state.is_qualified_under("production"));
        assert!(!state.is_evidence_admitted_under("production"));
        assert!(!state.is_action_authorized_at(10));
    }

    #[test]
    fn signer_authorized_does_not_widen_claim_or_authorize_action() {
        let state = AssuranceStateSnapshot {
            attestation: AttestationState::SignerAuthorized {
                authority_scope: "attest:proof-receipt".into(),
            },
            property: PropertyState::Supported,
            ..Default::default()
        };

        assert_eq!(state.property, PropertyState::Supported);
        assert!(!state.is_qualified_under("production"));
        assert!(!state.is_action_authorized_at(10));
    }

    #[test]
    fn qualification_does_not_authorize_or_execute() {
        let state = AssuranceStateSnapshot {
            qualification: QualificationState::QualifiedUnderProfile {
                profile_id: "production".into(),
            },
            ..Default::default()
        };

        assert!(state.is_qualified_under("production"));
        assert!(!state.is_action_authorized_at(10));
        assert!(!state.was_executed());
    }

    #[test]
    fn authorization_is_time_and_scope_bounded() {
        let state = AssuranceStateSnapshot {
            authorization: ActionAuthorizationState::Authorized {
                scope: "deploy:service-a".into(),
                valid_until_unix_s: Some(100),
            },
            ..Default::default()
        };

        assert!(state.is_action_authorized_at(100));
        assert!(!state.is_action_authorized_at(101));
        assert!(!state.was_executed());
    }

    #[test]
    fn execution_is_not_success() {
        let execution_id = Uuid::new_v4();
        let state = AssuranceStateSnapshot {
            execution: ExecutionState::Executed {
                execution_id,
                outcome: ExecutionOutcome::Failed,
            },
            ..Default::default()
        };

        assert!(state.was_executed());
        assert!(!state.execution_succeeded());
    }

    #[test]
    fn profile_scopes_do_not_cross() {
        let state = AssuranceStateSnapshot {
            admission: EvidenceAdmissionState::AdmittedUnderProfile {
                profile_id: "lab".into(),
            },
            qualification: QualificationState::QualifiedUnderProfile {
                profile_id: "lab".into(),
            },
            ..Default::default()
        };

        assert!(state.is_evidence_admitted_under("lab"));
        assert!(state.is_qualified_under("lab"));
        assert!(!state.is_evidence_admitted_under("production"));
        assert!(!state.is_qualified_under("production"));
    }

    #[test]
    fn malformed_authority_bearing_labels_fail_structural_validation() {
        let bad_profile = AssuranceStateSnapshot {
            qualification: QualificationState::QualifiedUnderProfile {
                profile_id: "   ".into(),
            },
            ..Default::default()
        };
        assert!(!bad_profile.validate());

        let bad_scope = AssuranceStateSnapshot {
            authorization: ActionAuthorizationState::Authorized {
                scope: "".into(),
                valid_until_unix_s: None,
            },
            ..Default::default()
        };
        assert!(!bad_scope.validate());
    }
}