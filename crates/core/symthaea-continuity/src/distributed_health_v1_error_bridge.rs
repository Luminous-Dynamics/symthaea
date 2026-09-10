// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compatibility mapping from the protocol-neutral shared distributed evaluator to
//! the historical post-transition V1 error vocabulary.
//!
//! Keeping this mapping isolated prevents V1 terminology from leaking back into the
//! common evaluator while preserving the existing V1 public error type.

use crate::distributed_health_common::DistributedHealthEvaluationError;
use crate::post_transition_distributed_health::PostTransitionDistributedHealthError;

impl From<DistributedHealthEvaluationError> for PostTransitionDistributedHealthError {
    fn from(error: DistributedHealthEvaluationError) -> Self {
        match error {
            DistributedHealthEvaluationError::ZeroEvaluationTime => Self::ZeroEvaluationTime,
            DistributedHealthEvaluationError::CurrentnessContextMismatch => {
                Self::CurrentnessContextMismatch
            }
            DistributedHealthEvaluationError::FailureDomainPolicyContextMismatch { policy_id } => {
                Self::FailureDomainPolicyContextMismatch { policy_id }
            }
            DistributedHealthEvaluationError::UnexpectedFailureDomainPolicy { policy_id } => {
                Self::UnexpectedFailureDomainPolicy { policy_id }
            }
            DistributedHealthEvaluationError::DuplicateFailureDomainPolicy { policy_id } => {
                Self::DuplicateFailureDomainPolicy { policy_id }
            }
            DistributedHealthEvaluationError::IncompleteFailureDomainPolicies {
                expected,
                observed,
            } => Self::IncompleteFailureDomainPolicies { expected, observed },
            DistributedHealthEvaluationError::EvidenceContextMismatch { role } => {
                Self::EvidenceContextMismatch { role }
            }
            DistributedHealthEvaluationError::UnapprovedVerifierProfile { role, profile_id } => {
                Self::UnapprovedVerifierProfile { role, profile_id }
            }
            DistributedHealthEvaluationError::ParticipantOutsideBudget { participant } => {
                Self::ParticipantOutsideBudget { participant }
            }
            DistributedHealthEvaluationError::DuplicateParticipantEvidence { participant } => {
                Self::DuplicateParticipantEvidence { participant }
            }
            DistributedHealthEvaluationError::IncompleteParticipantEvidence {
                expected,
                observed,
            } => Self::IncompleteParticipantEvidence { expected, observed },
            DistributedHealthEvaluationError::MissingParticipantEvidence { participant } => {
                Self::MissingParticipantEvidence { participant }
            }
            DistributedHealthEvaluationError::UnknownParticipantState { participant } => {
                Self::UnknownParticipantState { participant }
            }
            DistributedHealthEvaluationError::LocalSubjectNotHealthy { participant } => {
                Self::TransitionedSubjectNotHealthy { participant }
            }
            DistributedHealthEvaluationError::AvailabilityBudgetExceeded {
                unavailable,
                allowed,
            } => Self::AvailabilityBudgetExceeded {
                unavailable,
                allowed,
            },
            DistributedHealthEvaluationError::MinimumHealthyViolated { observed, required } => {
                Self::MinimumHealthyViolated { observed, required }
            }
            DistributedHealthEvaluationError::MutualExclusionViolated => {
                Self::MutualExclusionViolated
            }
            DistributedHealthEvaluationError::UnexpectedFailureDomainEvidence { policy_id } => {
                Self::UnexpectedFailureDomainEvidence { policy_id }
            }
            DistributedHealthEvaluationError::DuplicateFailureDomainEvidence { policy_id } => {
                Self::DuplicateFailureDomainEvidence { policy_id }
            }
            DistributedHealthEvaluationError::IncompleteFailureDomainEvidence {
                expected,
                observed,
            } => Self::IncompleteFailureDomainEvidence { expected, observed },
            DistributedHealthEvaluationError::MissingFailureDomainEvidence { policy_id } => {
                Self::MissingFailureDomainEvidence { policy_id }
            }
            DistributedHealthEvaluationError::FailureDomainNotCurrent { policy_id } => {
                Self::FailureDomainNotCurrent { policy_id }
            }
            DistributedHealthEvaluationError::FailureDomainFloorViolated {
                policy_id,
                observed,
                required,
            } => Self::FailureDomainFloorViolated {
                policy_id,
                observed,
                required,
            },
            DistributedHealthEvaluationError::RecoveryClassOutsideBudget => {
                Self::RecoveryClassOutsideBudget
            }
            DistributedHealthEvaluationError::DuplicateRecoveryEvidence => {
                Self::DuplicateRecoveryEvidence
            }
            DistributedHealthEvaluationError::AvailableRecoveryMissingIdentity => {
                Self::AvailableRecoveryMissingIdentity
            }
            DistributedHealthEvaluationError::NoAvailableRecoveryPath => {
                Self::NoAvailableRecoveryPath
            }
            DistributedHealthEvaluationError::StaleEvidence {
                role,
                age_ms,
                allowed_ms,
            } => Self::StaleEvidence {
                role,
                age_ms,
                allowed_ms,
            },
            DistributedHealthEvaluationError::EvidenceFromFuture {
                role,
                skew_ms,
                allowed_ms,
            } => Self::EvidenceFromFuture {
                role,
                skew_ms,
                allowed_ms,
            },
            DistributedHealthEvaluationError::NoCurrentnessEvidence => Self::NoCurrentnessEvidence,
            DistributedHealthEvaluationError::CrossEvidenceSkewExceeded {
                observed_ms,
                allowed_ms,
            } => Self::CrossEvidenceSkewExceeded {
                observed_ms,
                allowed_ms,
            },
        }
    }
}
