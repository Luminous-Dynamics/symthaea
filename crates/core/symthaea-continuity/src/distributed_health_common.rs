// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared closed-world distributed-health evaluator.
//!
//! This module owns the policy/evidence evaluation common to the original
//! post-transition V1 proof and the generic healthy-local-snapshot V2 proof. It does
//! not define either proof identity. The error vocabulary is deliberately neutral:
//! the evaluated local subject may be a new target, a crash-reconciled source, or a
//! future independently admitted baseline.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::distributed::RecoveryPathClassV1;
use crate::distributed_currentness::ValidatedDistributedCurrentnessPolicyV1;
use crate::distributed_evidence::{
    AuthenticatedFailureDomainStateEvidenceId, AuthenticatedFailureDomainStateEvidenceV1,
    AuthenticatedRecoveryPathStateEvidenceId, AuthenticatedRecoveryPathStateEvidenceV1,
    FailureDomainObservationOutcomeV1, RecoveryPathObservationOutcomeV1,
};
use crate::distributed_state::{
    AuthenticatedParticipantStateEvidenceId, AuthenticatedParticipantStateEvidenceV1,
    DistributedStateContextId, ParticipantOperationalStateV1,
    ValidatedDistributedStateContextV1,
};
use crate::failure_domain::{FailureDomainPolicyId, ValidatedFailureDomainPolicyV1};
use crate::scope::ContinuitySubjectId;
use crate::verifier::VerifierProfileId;

#[derive(Debug, Clone)]
pub(crate) struct EvaluatedDistributedHealthV1 {
    pub participant_evidence_ids: Vec<AuthenticatedParticipantStateEvidenceId>,
    pub failure_evidence_ids: Vec<AuthenticatedFailureDomainStateEvidenceId>,
    pub recovery_evidence_ids: Vec<AuthenticatedRecoveryPathStateEvidenceId>,
    pub verifier_snapshots: Vec<(VerifierProfileId, u64)>,
    pub recovery_paths: Vec<(RecoveryPathClassV1, [u8; 32])>,
    pub unavailable_count: u32,
    pub healthy_count: u32,
    pub healthy_domains: Vec<(FailureDomainPolicyId, u32)>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_distributed_health(
    local_subject_id: ContinuitySubjectId,
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    failure_domain_policies: &[ValidatedFailureDomainPolicyV1],
    participant_evidence: &[AuthenticatedParticipantStateEvidenceV1],
    failure_domain_evidence: &[AuthenticatedFailureDomainStateEvidenceV1],
    recovery_evidence: &[AuthenticatedRecoveryPathStateEvidenceV1],
    evaluated_at_unix_ms: u64,
) -> Result<EvaluatedDistributedHealthV1, DistributedHealthEvaluationError> {
    if evaluated_at_unix_ms == 0 {
        return Err(DistributedHealthEvaluationError::ZeroEvaluationTime);
    }
    if currentness.budget_id() != context.budget_id()
        || currentness.budget_generation() != context.budget_generation()
    {
        return Err(DistributedHealthEvaluationError::CurrentnessContextMismatch);
    }

    let failure_policy_by_id =
        validate_failure_policies(context, currentness, failure_domain_policies)?;

    let mut observation_times = Vec::new();
    let mut verifier_snapshots = BTreeSet::new();
    let mut participant_evidence_ids = Vec::new();
    let mut states = BTreeMap::new();

    for evidence in participant_evidence {
        require_context("participant", evidence.context_id(), context.id())?;
        require_profile(
            "participant",
            evidence.profile_id(),
            currentness.participant_verifier_profile_ids(),
        )?;
        check_freshness(
            "participant",
            evidence.observed_at_unix_ms(),
            evaluated_at_unix_ms,
            currentness.max_participant_state_age_ms(),
            currentness.max_future_skew_ms(),
        )?;
        if context
            .budget()
            .participant_subject_ids()
            .binary_search(&evidence.participant_subject_id())
            .is_err()
        {
            return Err(DistributedHealthEvaluationError::ParticipantOutsideBudget {
                participant: evidence.participant_subject_id(),
            });
        }
        if states
            .insert(evidence.participant_subject_id(), evidence.state())
            .is_some()
        {
            return Err(DistributedHealthEvaluationError::DuplicateParticipantEvidence {
                participant: evidence.participant_subject_id(),
            });
        }
        if evidence.state() == ParticipantOperationalStateV1::Unknown {
            return Err(DistributedHealthEvaluationError::UnknownParticipantState {
                participant: evidence.participant_subject_id(),
            });
        }
        observation_times.push(evidence.observed_at_unix_ms());
        participant_evidence_ids.push(evidence.id());
        verifier_snapshots.insert((evidence.profile_id(), evidence.root_epoch()));
    }

    let expected_participants = context.budget().participant_subject_ids();
    if states.len() != expected_participants.len() {
        return Err(DistributedHealthEvaluationError::IncompleteParticipantEvidence {
            expected: expected_participants.len(),
            observed: states.len(),
        });
    }
    for participant in expected_participants {
        if !states.contains_key(participant) {
            return Err(DistributedHealthEvaluationError::MissingParticipantEvidence {
                participant: *participant,
            });
        }
    }

    if states.get(&local_subject_id) != Some(&ParticipantOperationalStateV1::Healthy) {
        return Err(DistributedHealthEvaluationError::LocalSubjectNotHealthy {
            participant: local_subject_id,
        });
    }

    let unavailable: BTreeSet<ContinuitySubjectId> = states
        .iter()
        .filter_map(|(participant, state)| {
            matches!(
                state,
                ParticipantOperationalStateV1::Unhealthy
                    | ParticipantOperationalStateV1::Transitioning
            )
            .then_some(*participant)
        })
        .collect();
    let unavailable_count = unavailable.len() as u32;
    if unavailable_count > context.budget().max_concurrent_unavailable() {
        return Err(DistributedHealthEvaluationError::AvailabilityBudgetExceeded {
            unavailable: unavailable_count,
            allowed: context.budget().max_concurrent_unavailable(),
        });
    }
    let healthy_count = expected_participants.len() as u32 - unavailable_count;
    if healthy_count < context.budget().minimum_healthy() {
        return Err(DistributedHealthEvaluationError::MinimumHealthyViolated {
            observed: healthy_count,
            required: context.budget().minimum_healthy(),
        });
    }
    for exclusion in context.budget().mutual_exclusion_sets() {
        let unavailable_members = exclusion
            .members()
            .iter()
            .filter(|member| unavailable.contains(member))
            .count();
        if unavailable_members > 1 {
            return Err(DistributedHealthEvaluationError::MutualExclusionViolated);
        }
    }

    let mut failure_outcomes = BTreeMap::new();
    let mut failure_evidence_ids = Vec::new();
    for evidence in failure_domain_evidence {
        require_context("failure-domain", evidence.context_id(), context.id())?;
        require_profile(
            "failure-domain",
            evidence.profile_id(),
            currentness.failure_domain_verifier_profile_ids(),
        )?;
        check_freshness(
            "failure-domain",
            evidence.observed_at_unix_ms(),
            evaluated_at_unix_ms,
            currentness.max_failure_domain_age_ms(),
            currentness.max_future_skew_ms(),
        )?;
        if !failure_policy_by_id.contains_key(&evidence.policy_id()) {
            return Err(DistributedHealthEvaluationError::UnexpectedFailureDomainEvidence {
                policy_id: evidence.policy_id(),
            });
        }
        if failure_outcomes
            .insert(evidence.policy_id(), evidence.outcome())
            .is_some()
        {
            return Err(DistributedHealthEvaluationError::DuplicateFailureDomainEvidence {
                policy_id: evidence.policy_id(),
            });
        }
        observation_times.push(evidence.observed_at_unix_ms());
        failure_evidence_ids.push(evidence.id());
        verifier_snapshots.insert((evidence.profile_id(), evidence.root_epoch()));
    }

    if failure_outcomes.len() != currentness.required_failure_domain_policy_ids().len() {
        return Err(DistributedHealthEvaluationError::IncompleteFailureDomainEvidence {
            expected: currentness.required_failure_domain_policy_ids().len(),
            observed: failure_outcomes.len(),
        });
    }

    let mut healthy_domains = Vec::new();
    for policy_id in currentness.required_failure_domain_policy_ids() {
        let outcome = failure_outcomes.get(policy_id).copied().ok_or(
            DistributedHealthEvaluationError::MissingFailureDomainEvidence {
                policy_id: *policy_id,
            },
        )?;
        if outcome != FailureDomainObservationOutcomeV1::MatchesPolicy {
            return Err(DistributedHealthEvaluationError::FailureDomainNotCurrent {
                policy_id: *policy_id,
            });
        }
        let policy = failure_policy_by_id
            .get(policy_id)
            .expect("validated required failure-domain policy is present");
        let count = policy
            .groups()
            .iter()
            .filter(|group| {
                group
                    .members()
                    .iter()
                    .any(|member| !unavailable.contains(member))
            })
            .count() as u32;
        if count < policy.minimum_healthy_domains() {
            return Err(DistributedHealthEvaluationError::FailureDomainFloorViolated {
                policy_id: *policy_id,
                observed: count,
                required: policy.minimum_healthy_domains(),
            });
        }
        healthy_domains.push((*policy_id, count));
    }

    let mut recovery_by_class = BTreeMap::<RecoveryPathClassV1, [u8; 32]>::new();
    let mut seen_recovery_classes = BTreeSet::<RecoveryPathClassV1>::new();
    let mut recovery_evidence_ids = Vec::new();
    for evidence in recovery_evidence {
        require_context("recovery", evidence.context_id(), context.id())?;
        require_profile(
            "recovery",
            evidence.profile_id(),
            currentness.recovery_verifier_profile_ids(),
        )?;
        check_freshness(
            "recovery",
            evidence.observed_at_unix_ms(),
            evaluated_at_unix_ms,
            currentness.max_recovery_path_age_ms(),
            currentness.max_future_skew_ms(),
        )?;
        if context
            .budget()
            .recovery_path_any_of()
            .binary_search(evidence.recovery_path_class())
            .is_err()
        {
            return Err(DistributedHealthEvaluationError::RecoveryClassOutsideBudget);
        }
        if !seen_recovery_classes.insert(evidence.recovery_path_class().clone()) {
            return Err(DistributedHealthEvaluationError::DuplicateRecoveryEvidence);
        }
        if evidence.outcome() == RecoveryPathObservationOutcomeV1::Available {
            let identity = evidence
                .recovery_path_identity_digest()
                .ok_or(DistributedHealthEvaluationError::AvailableRecoveryMissingIdentity)?;
            recovery_by_class.insert(evidence.recovery_path_class().clone(), identity);
        }
        observation_times.push(evidence.observed_at_unix_ms());
        recovery_evidence_ids.push(evidence.id());
        verifier_snapshots.insert((evidence.profile_id(), evidence.root_epoch()));
    }

    if recovery_by_class.is_empty() {
        return Err(DistributedHealthEvaluationError::NoAvailableRecoveryPath);
    }
    validate_cross_evidence_skew(
        &observation_times,
        currentness.max_cross_evidence_skew_ms(),
    )?;

    participant_evidence_ids.sort();
    failure_evidence_ids.sort();
    recovery_evidence_ids.sort();
    healthy_domains.sort();
    let recovery_paths = recovery_by_class.into_iter().collect::<Vec<_>>();
    let verifier_snapshots = verifier_snapshots.into_iter().collect::<Vec<_>>();

    Ok(EvaluatedDistributedHealthV1 {
        participant_evidence_ids,
        failure_evidence_ids,
        recovery_evidence_ids,
        verifier_snapshots,
        recovery_paths,
        unavailable_count,
        healthy_count,
        healthy_domains,
    })
}

fn validate_failure_policies<'a>(
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    policies: &'a [ValidatedFailureDomainPolicyV1],
) -> Result<
    BTreeMap<FailureDomainPolicyId, &'a ValidatedFailureDomainPolicyV1>,
    DistributedHealthEvaluationError,
> {
    let mut by_id = BTreeMap::new();
    for policy in policies {
        if policy.budget_id() != context.budget_id()
            || policy.budget_generation() != context.budget_generation()
            || policy.aggregate_subject_id() != context.aggregate_subject_id()
        {
            return Err(DistributedHealthEvaluationError::FailureDomainPolicyContextMismatch {
                policy_id: policy.id(),
            });
        }
        if currentness
            .required_failure_domain_policy_ids()
            .binary_search(&policy.id())
            .is_err()
        {
            return Err(DistributedHealthEvaluationError::UnexpectedFailureDomainPolicy {
                policy_id: policy.id(),
            });
        }
        if by_id.insert(policy.id(), policy).is_some() {
            return Err(DistributedHealthEvaluationError::DuplicateFailureDomainPolicy {
                policy_id: policy.id(),
            });
        }
    }
    if by_id.len() != currentness.required_failure_domain_policy_ids().len() {
        return Err(DistributedHealthEvaluationError::IncompleteFailureDomainPolicies {
            expected: currentness.required_failure_domain_policy_ids().len(),
            observed: by_id.len(),
        });
    }
    Ok(by_id)
}

fn require_context(
    role: &'static str,
    observed: DistributedStateContextId,
    expected: DistributedStateContextId,
) -> Result<(), DistributedHealthEvaluationError> {
    if observed != expected {
        return Err(DistributedHealthEvaluationError::EvidenceContextMismatch { role });
    }
    Ok(())
}

fn require_profile(
    role: &'static str,
    profile_id: VerifierProfileId,
    allowed: &[VerifierProfileId],
) -> Result<(), DistributedHealthEvaluationError> {
    if allowed.binary_search(&profile_id).is_err() {
        return Err(DistributedHealthEvaluationError::UnapprovedVerifierProfile {
            role,
            profile_id,
        });
    }
    Ok(())
}

fn check_freshness(
    role: &'static str,
    observed_at_unix_ms: u64,
    evaluated_at_unix_ms: u64,
    maximum_age_ms: u64,
    maximum_future_skew_ms: u64,
) -> Result<(), DistributedHealthEvaluationError> {
    if observed_at_unix_ms > evaluated_at_unix_ms {
        let skew = observed_at_unix_ms - evaluated_at_unix_ms;
        if skew > maximum_future_skew_ms {
            return Err(DistributedHealthEvaluationError::EvidenceFromFuture {
                role,
                skew_ms: skew,
                allowed_ms: maximum_future_skew_ms,
            });
        }
        return Ok(());
    }
    let age = evaluated_at_unix_ms - observed_at_unix_ms;
    if age > maximum_age_ms {
        return Err(DistributedHealthEvaluationError::StaleEvidence {
            role,
            age_ms: age,
            allowed_ms: maximum_age_ms,
        });
    }
    Ok(())
}

fn validate_cross_evidence_skew(
    times: &[u64],
    maximum_skew_ms: u64,
) -> Result<(), DistributedHealthEvaluationError> {
    let Some(minimum) = times.iter().min().copied() else {
        return Err(DistributedHealthEvaluationError::NoCurrentnessEvidence);
    };
    let maximum = times.iter().max().copied().unwrap_or(minimum);
    let skew = maximum - minimum;
    if skew > maximum_skew_ms {
        return Err(DistributedHealthEvaluationError::CrossEvidenceSkewExceeded {
            observed_ms: skew,
            allowed_ms: maximum_skew_ms,
        });
    }
    Ok(())
}

/// Protocol-neutral errors produced by the shared distributed-health evaluator.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DistributedHealthEvaluationError {
    #[error("distributed-health evaluation time must be non-zero")]
    ZeroEvaluationTime,
    #[error("distributed currentness policy is bound to a different budget/generation")]
    CurrentnessContextMismatch,
    #[error("failure-domain policy {policy_id:?} belongs to a different distributed context")]
    FailureDomainPolicyContextMismatch { policy_id: FailureDomainPolicyId },
    #[error("unexpected failure-domain policy {policy_id:?}")]
    UnexpectedFailureDomainPolicy { policy_id: FailureDomainPolicyId },
    #[error("duplicate failure-domain policy {policy_id:?}")]
    DuplicateFailureDomainPolicy { policy_id: FailureDomainPolicyId },
    #[error("failure-domain policy set incomplete: expected {expected}, observed {observed}")]
    IncompleteFailureDomainPolicies { expected: usize, observed: usize },
    #[error("authenticated {role} evidence belongs to a different distributed context")]
    EvidenceContextMismatch { role: &'static str },
    #[error("{role} verifier profile {profile_id:?} is not approved by currentness policy")]
    UnapprovedVerifierProfile {
        role: &'static str,
        profile_id: VerifierProfileId,
    },
    #[error("participant evidence references subject outside budget: {participant:?}")]
    ParticipantOutsideBudget { participant: ContinuitySubjectId },
    #[error("duplicate participant evidence for {participant:?}")]
    DuplicateParticipantEvidence { participant: ContinuitySubjectId },
    #[error("participant evidence incomplete: expected {expected}, observed {observed}")]
    IncompleteParticipantEvidence { expected: usize, observed: usize },
    #[error("missing participant evidence for {participant:?}")]
    MissingParticipantEvidence { participant: ContinuitySubjectId },
    #[error("participant state is UNKNOWN: {participant:?}")]
    UnknownParticipantState { participant: ContinuitySubjectId },
    #[error("exact local subject is not currently healthy in participant evidence: {participant:?}")]
    LocalSubjectNotHealthy { participant: ContinuitySubjectId },
    #[error("current unavailable count {unavailable} exceeds distributed budget {allowed}")]
    AvailabilityBudgetExceeded { unavailable: u32, allowed: u32 },
    #[error("current healthy count {observed} is below required minimum {required}")]
    MinimumHealthyViolated { observed: u32, required: u32 },
    #[error("current unavailable set violates a protected mutual-exclusion set")]
    MutualExclusionViolated,
    #[error("unexpected failure-domain evidence for policy {policy_id:?}")]
    UnexpectedFailureDomainEvidence { policy_id: FailureDomainPolicyId },
    #[error("duplicate failure-domain evidence for policy {policy_id:?}")]
    DuplicateFailureDomainEvidence { policy_id: FailureDomainPolicyId },
    #[error("failure-domain evidence incomplete: expected {expected}, observed {observed}")]
    IncompleteFailureDomainEvidence { expected: usize, observed: usize },
    #[error("missing failure-domain evidence for policy {policy_id:?}")]
    MissingFailureDomainEvidence { policy_id: FailureDomainPolicyId },
    #[error("failure-domain policy {policy_id:?} is not currently observed to match")]
    FailureDomainNotCurrent { policy_id: FailureDomainPolicyId },
    #[error("failure-domain policy {policy_id:?} has {observed} healthy domains below required {required}")]
    FailureDomainFloorViolated {
        policy_id: FailureDomainPolicyId,
        observed: u32,
        required: u32,
    },
    #[error("recovery evidence names a class outside the distributed budget")]
    RecoveryClassOutsideBudget,
    #[error("duplicate recovery evidence for the same recovery class")]
    DuplicateRecoveryEvidence,
    #[error("available recovery evidence is missing its exact path identity")]
    AvailableRecoveryMissingIdentity,
    #[error("no authenticated current independent recovery path is available")]
    NoAvailableRecoveryPath,
    #[error("authenticated {role} evidence is stale: age {age_ms} ms > {allowed_ms} ms")]
    StaleEvidence {
        role: &'static str,
        age_ms: u64,
        allowed_ms: u64,
    },
    #[error("authenticated {role} evidence is too far in the future: skew {skew_ms} ms > {allowed_ms} ms")]
    EvidenceFromFuture {
        role: &'static str,
        skew_ms: u64,
        allowed_ms: u64,
    },
    #[error("distributed currentness evidence set is empty")]
    NoCurrentnessEvidence,
    #[error("cross-evidence observation skew {observed_ms} ms exceeds allowed {allowed_ms} ms")]
    CrossEvidenceSkewExceeded { observed_ms: u64, allowed_ms: u64 },
}
