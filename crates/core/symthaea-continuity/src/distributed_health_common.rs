// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared closed-world distributed-health evaluator.
//!
//! This module owns the policy/evidence evaluation that is common to the original
//! post-transition V1 proof and the generic healthy-local-snapshot V2 proof. It does
//! not define either proof identity. Callers retain their own domain-separated hash
//! contracts so extracting this evaluator does not silently rewrite V1 identities.

use std::collections::{BTreeMap, BTreeSet};

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
use crate::post_transition_distributed_health::PostTransitionDistributedHealthError;
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
) -> Result<EvaluatedDistributedHealthV1, PostTransitionDistributedHealthError> {
    if evaluated_at_unix_ms == 0 {
        return Err(PostTransitionDistributedHealthError::ZeroEvaluationTime);
    }
    if currentness.budget_id() != context.budget_id()
        || currentness.budget_generation() != context.budget_generation()
    {
        return Err(PostTransitionDistributedHealthError::CurrentnessContextMismatch);
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
            return Err(PostTransitionDistributedHealthError::ParticipantOutsideBudget {
                participant: evidence.participant_subject_id(),
            });
        }
        if states
            .insert(evidence.participant_subject_id(), evidence.state())
            .is_some()
        {
            return Err(PostTransitionDistributedHealthError::DuplicateParticipantEvidence {
                participant: evidence.participant_subject_id(),
            });
        }
        if evidence.state() == ParticipantOperationalStateV1::Unknown {
            return Err(PostTransitionDistributedHealthError::UnknownParticipantState {
                participant: evidence.participant_subject_id(),
            });
        }
        observation_times.push(evidence.observed_at_unix_ms());
        participant_evidence_ids.push(evidence.id());
        verifier_snapshots.insert((evidence.profile_id(), evidence.root_epoch()));
    }

    let expected_participants = context.budget().participant_subject_ids();
    if states.len() != expected_participants.len() {
        return Err(PostTransitionDistributedHealthError::IncompleteParticipantEvidence {
            expected: expected_participants.len(),
            observed: states.len(),
        });
    }
    for participant in expected_participants {
        if !states.contains_key(participant) {
            return Err(PostTransitionDistributedHealthError::MissingParticipantEvidence {
                participant: *participant,
            });
        }
    }

    if states.get(&local_subject_id) != Some(&ParticipantOperationalStateV1::Healthy) {
        return Err(PostTransitionDistributedHealthError::TransitionedSubjectNotHealthy {
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
        return Err(PostTransitionDistributedHealthError::AvailabilityBudgetExceeded {
            unavailable: unavailable_count,
            allowed: context.budget().max_concurrent_unavailable(),
        });
    }
    let healthy_count = expected_participants.len() as u32 - unavailable_count;
    if healthy_count < context.budget().minimum_healthy() {
        return Err(PostTransitionDistributedHealthError::MinimumHealthyViolated {
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
            return Err(PostTransitionDistributedHealthError::MutualExclusionViolated);
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
            return Err(PostTransitionDistributedHealthError::UnexpectedFailureDomainEvidence {
                policy_id: evidence.policy_id(),
            });
        }
        if failure_outcomes
            .insert(evidence.policy_id(), evidence.outcome())
            .is_some()
        {
            return Err(PostTransitionDistributedHealthError::DuplicateFailureDomainEvidence {
                policy_id: evidence.policy_id(),
            });
        }
        observation_times.push(evidence.observed_at_unix_ms());
        failure_evidence_ids.push(evidence.id());
        verifier_snapshots.insert((evidence.profile_id(), evidence.root_epoch()));
    }

    if failure_outcomes.len() != currentness.required_failure_domain_policy_ids().len() {
        return Err(PostTransitionDistributedHealthError::IncompleteFailureDomainEvidence {
            expected: currentness.required_failure_domain_policy_ids().len(),
            observed: failure_outcomes.len(),
        });
    }

    let mut healthy_domains = Vec::new();
    for policy_id in currentness.required_failure_domain_policy_ids() {
        let outcome = failure_outcomes.get(policy_id).copied().ok_or(
            PostTransitionDistributedHealthError::MissingFailureDomainEvidence {
                policy_id: *policy_id,
            },
        )?;
        if outcome != FailureDomainObservationOutcomeV1::MatchesPolicy {
            return Err(PostTransitionDistributedHealthError::FailureDomainNotCurrent {
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
            return Err(PostTransitionDistributedHealthError::FailureDomainFloorViolated {
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
            return Err(PostTransitionDistributedHealthError::RecoveryClassOutsideBudget);
        }
        if !seen_recovery_classes.insert(evidence.recovery_path_class().clone()) {
            return Err(PostTransitionDistributedHealthError::DuplicateRecoveryEvidence);
        }
        if evidence.outcome() == RecoveryPathObservationOutcomeV1::Available {
            let identity = evidence
                .recovery_path_identity_digest()
                .ok_or(PostTransitionDistributedHealthError::AvailableRecoveryMissingIdentity)?;
            recovery_by_class.insert(evidence.recovery_path_class().clone(), identity);
        }
        observation_times.push(evidence.observed_at_unix_ms());
        recovery_evidence_ids.push(evidence.id());
        verifier_snapshots.insert((evidence.profile_id(), evidence.root_epoch()));
    }

    if recovery_by_class.is_empty() {
        return Err(PostTransitionDistributedHealthError::NoAvailableRecoveryPath);
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
    PostTransitionDistributedHealthError,
> {
    let mut by_id = BTreeMap::new();
    for policy in policies {
        if policy.budget_id() != context.budget_id()
            || policy.budget_generation() != context.budget_generation()
            || policy.aggregate_subject_id() != context.aggregate_subject_id()
        {
            return Err(PostTransitionDistributedHealthError::FailureDomainPolicyContextMismatch {
                policy_id: policy.id(),
            });
        }
        if currentness
            .required_failure_domain_policy_ids()
            .binary_search(&policy.id())
            .is_err()
        {
            return Err(PostTransitionDistributedHealthError::UnexpectedFailureDomainPolicy {
                policy_id: policy.id(),
            });
        }
        if by_id.insert(policy.id(), policy).is_some() {
            return Err(PostTransitionDistributedHealthError::DuplicateFailureDomainPolicy {
                policy_id: policy.id(),
            });
        }
    }
    if by_id.len() != currentness.required_failure_domain_policy_ids().len() {
        return Err(PostTransitionDistributedHealthError::IncompleteFailureDomainPolicies {
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
) -> Result<(), PostTransitionDistributedHealthError> {
    if observed != expected {
        return Err(PostTransitionDistributedHealthError::EvidenceContextMismatch { role });
    }
    Ok(())
}

fn require_profile(
    role: &'static str,
    profile_id: VerifierProfileId,
    allowed: &[VerifierProfileId],
) -> Result<(), PostTransitionDistributedHealthError> {
    if allowed.binary_search(&profile_id).is_err() {
        return Err(PostTransitionDistributedHealthError::UnapprovedVerifierProfile {
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
) -> Result<(), PostTransitionDistributedHealthError> {
    if observed_at_unix_ms > evaluated_at_unix_ms {
        let skew = observed_at_unix_ms - evaluated_at_unix_ms;
        if skew > maximum_future_skew_ms {
            return Err(PostTransitionDistributedHealthError::EvidenceFromFuture {
                role,
                skew_ms: skew,
                allowed_ms: maximum_future_skew_ms,
            });
        }
        return Ok(());
    }
    let age = evaluated_at_unix_ms - observed_at_unix_ms;
    if age > maximum_age_ms {
        return Err(PostTransitionDistributedHealthError::StaleEvidence {
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
) -> Result<(), PostTransitionDistributedHealthError> {
    let Some(minimum) = times.iter().min().copied() else {
        return Err(PostTransitionDistributedHealthError::NoCurrentnessEvidence);
    };
    let maximum = times.iter().max().copied().unwrap_or(minimum);
    let skew = maximum - minimum;
    if skew > maximum_skew_ms {
        return Err(PostTransitionDistributedHealthError::CrossEvidenceSkewExceeded {
            observed_ms: skew,
            allowed_ms: maximum_skew_ms,
        });
    }
    Ok(())
}
