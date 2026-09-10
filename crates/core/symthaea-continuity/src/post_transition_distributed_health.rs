// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh distributed-health qualification after a local transition.
//!
//! The pre-transition distributed witness answers whether taking the candidate
//! through a transition is safe. It deliberately treats candidates as projected
//! unavailable. That witness therefore cannot prove that the candidate successfully
//! returned to a healthy distributed world.
//!
//! `PreTransitionSafety != PostTransitionDistributedHealth != LastKnownGood`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::{DistributedChangeBudgetId, RecoveryPathClassV1};
use crate::distributed_currentness::{
    DistributedCurrentnessPolicyId, ValidatedDistributedCurrentnessPolicyV1,
};
use crate::distributed_evidence::{
    AuthenticatedFailureDomainStateEvidenceId, AuthenticatedFailureDomainStateEvidenceV1,
    AuthenticatedRecoveryPathStateEvidenceId, AuthenticatedRecoveryPathStateEvidenceV1,
    FailureDomainObservationOutcomeV1, RecoveryPathObservationOutcomeV1,
};
use crate::distributed_state::{
    AuthenticatedParticipantStateEvidenceId, AuthenticatedParticipantStateEvidenceV1,
    DistributedStateContextId, ParticipantOperationalStateV1, ParticipantSetDigest,
    ValidatedDistributedStateContextV1,
};
use crate::failure_domain::{FailureDomainPolicyId, ValidatedFailureDomainPolicyV1};
use crate::post_execution_health::{
    PostExecutionHealthOutcomeV1, QualifiedPostExecutionHealthId,
    QualifiedPostExecutionHealthV1,
};
use crate::scope::ContinuitySubjectId;
use crate::verifier::VerifierProfileId;

const CURRENT_STATE_DOMAIN: &[u8] =
    b"symthaea.continuity.post-transition-distributed-current-state.v1\0";
const QUALIFIED_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-post-transition-distributed-health.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PostTransitionDistributedStateDigest([u8; 32]);
impl PostTransitionDistributedStateDigest {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedPostTransitionDistributedHealthId([u8; 32]);
impl QualifiedPostTransitionDistributedHealthId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PostTransitionVerifierSnapshotV1 {
    profile_id: VerifierProfileId,
    root_epoch: u64,
}

impl PostTransitionVerifierSnapshotV1 {
    pub fn profile_id(&self) -> VerifierProfileId {
        self.profile_id
    }

    pub fn root_epoch(&self) -> u64 {
        self.root_epoch
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedRecoveryPathSnapshotV1 {
    recovery_path_class: RecoveryPathClassV1,
    recovery_path_identity_digest: [u8; 32],
}

impl QualifiedRecoveryPathSnapshotV1 {
    pub fn recovery_path_class(&self) -> &RecoveryPathClassV1 {
        &self.recovery_path_class
    }

    pub fn recovery_path_identity_digest(&self) -> [u8; 32] {
        self.recovery_path_identity_digest
    }
}

/// Non-Serde exact post-transition distributed-health proof.
///
/// The candidate is no longer projected unavailable. Its fresh participant state is
/// evaluated like every other member and must be `Healthy` before this value exists.
#[derive(Debug, Clone)]
pub struct QualifiedPostTransitionDistributedHealthV1 {
    qualified_id: QualifiedPostTransitionDistributedHealthId,
    local_health_id: QualifiedPostExecutionHealthId,
    context_id: DistributedStateContextId,
    aggregate_subject_id: ContinuitySubjectId,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    participant_set_digest: ParticipantSetDigest,
    currentness_policy_id: DistributedCurrentnessPolicyId,
    currentness_policy_generation: u64,
    transitioned_subject_id: ContinuitySubjectId,
    current_state_digest: PostTransitionDistributedStateDigest,
    verifier_snapshots: Vec<PostTransitionVerifierSnapshotV1>,
    recovery_paths: Vec<QualifiedRecoveryPathSnapshotV1>,
    unavailable_count: u32,
    healthy_count: u32,
    healthy_domains: Vec<(FailureDomainPolicyId, u32)>,
    evaluated_at_unix_ms: u64,
}

impl QualifiedPostTransitionDistributedHealthV1 {
    pub fn id(&self) -> QualifiedPostTransitionDistributedHealthId {
        self.qualified_id
    }

    pub fn local_health_id(&self) -> QualifiedPostExecutionHealthId {
        self.local_health_id
    }

    pub fn context_id(&self) -> DistributedStateContextId {
        self.context_id
    }

    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId {
        self.aggregate_subject_id
    }

    pub fn transitioned_subject_id(&self) -> ContinuitySubjectId {
        self.transitioned_subject_id
    }

    pub fn current_state_digest(&self) -> PostTransitionDistributedStateDigest {
        self.current_state_digest
    }

    pub fn recovery_paths(&self) -> &[QualifiedRecoveryPathSnapshotV1] {
        &self.recovery_paths
    }

    pub fn unavailable_count(&self) -> u32 {
        self.unavailable_count
    }

    pub fn healthy_count(&self) -> u32 {
        self.healthy_count
    }

    pub fn healthy_domains(&self) -> &[(FailureDomainPolicyId, u32)] {
        &self.healthy_domains
    }

    pub fn evaluated_at_unix_ms(&self) -> u64 {
        self.evaluated_at_unix_ms
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compose_post_transition_distributed_health(
    local_health: &QualifiedPostExecutionHealthV1,
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    failure_domain_policies: &[ValidatedFailureDomainPolicyV1],
    participant_evidence: &[AuthenticatedParticipantStateEvidenceV1],
    failure_domain_evidence: &[AuthenticatedFailureDomainStateEvidenceV1],
    recovery_evidence: &[AuthenticatedRecoveryPathStateEvidenceV1],
    evaluated_at_unix_ms: u64,
) -> Result<QualifiedPostTransitionDistributedHealthV1, PostTransitionDistributedHealthError> {
    if evaluated_at_unix_ms == 0 {
        return Err(PostTransitionDistributedHealthError::ZeroEvaluationTime);
    }
    if local_health.outcome() != PostExecutionHealthOutcomeV1::Healthy {
        return Err(PostTransitionDistributedHealthError::LocalTargetNotHealthy);
    }
    if local_health.distributed_context_id() != context.id() {
        return Err(PostTransitionDistributedHealthError::LocalHealthContextMismatch);
    }
    if local_health.qualified_at_unix_ms() != evaluated_at_unix_ms {
        return Err(PostTransitionDistributedHealthError::LocalHealthEvaluationTimeMismatch);
    }
    if context
        .candidate_subject_ids()
        .binary_search(&local_health.subject_id())
        .is_err()
    {
        return Err(PostTransitionDistributedHealthError::LocalSubjectOutsideCandidateSet);
    }
    if currentness.budget_id() != context.budget_id()
        || currentness.budget_generation() != context.budget_generation()
    {
        return Err(PostTransitionDistributedHealthError::CurrentnessContextMismatch);
    }

    let failure_policy_by_id = validate_failure_policies(
        context,
        currentness,
        failure_domain_policies,
    )?;

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
        verifier_snapshots.insert(PostTransitionVerifierSnapshotV1 {
            profile_id: evidence.profile_id(),
            root_epoch: evidence.root_epoch(),
        });
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

    if states.get(&local_health.subject_id()) != Some(&ParticipantOperationalStateV1::Healthy) {
        return Err(PostTransitionDistributedHealthError::TransitionedSubjectNotHealthy {
            participant: local_health.subject_id(),
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
        verifier_snapshots.insert(PostTransitionVerifierSnapshotV1 {
            profile_id: evidence.profile_id(),
            root_epoch: evidence.root_epoch(),
        });
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

    let mut recovery_by_class = BTreeMap::<RecoveryPathClassV1, QualifiedRecoveryPathSnapshotV1>::new();
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
        if recovery_by_class.contains_key(evidence.recovery_path_class()) {
            return Err(PostTransitionDistributedHealthError::DuplicateRecoveryEvidence);
        }
        if evidence.outcome() == RecoveryPathObservationOutcomeV1::Available {
            let identity = evidence
                .recovery_path_identity_digest()
                .ok_or(PostTransitionDistributedHealthError::AvailableRecoveryMissingIdentity)?;
            recovery_by_class.insert(
                evidence.recovery_path_class().clone(),
                QualifiedRecoveryPathSnapshotV1 {
                    recovery_path_class: evidence.recovery_path_class().clone(),
                    recovery_path_identity_digest: identity,
                },
            );
        }
        observation_times.push(evidence.observed_at_unix_ms());
        recovery_evidence_ids.push(evidence.id());
        verifier_snapshots.insert(PostTransitionVerifierSnapshotV1 {
            profile_id: evidence.profile_id(),
            root_epoch: evidence.root_epoch(),
        });
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
    let recovery_paths = recovery_by_class.into_values().collect::<Vec<_>>();
    let verifier_snapshots = verifier_snapshots.into_iter().collect::<Vec<_>>();

    let current_state_digest = PostTransitionDistributedStateDigest(hash_current_state(
        local_health.id(),
        context.id(),
        currentness.id(),
        &participant_evidence_ids,
        &failure_evidence_ids,
        &recovery_evidence_ids,
    ));
    let qualified_id = QualifiedPostTransitionDistributedHealthId(hash_qualified(
        current_state_digest,
        local_health,
        context,
        currentness,
        &verifier_snapshots,
        &recovery_paths,
        unavailable_count,
        healthy_count,
        &healthy_domains,
        evaluated_at_unix_ms,
    ));

    Ok(QualifiedPostTransitionDistributedHealthV1 {
        qualified_id,
        local_health_id: local_health.id(),
        context_id: context.id(),
        aggregate_subject_id: context.aggregate_subject_id(),
        budget_id: context.budget_id(),
        budget_generation: context.budget_generation(),
        participant_set_digest: context.participant_set_digest(),
        currentness_policy_id: currentness.id(),
        currentness_policy_generation: currentness.policy_generation(),
        transitioned_subject_id: local_health.subject_id(),
        current_state_digest,
        verifier_snapshots,
        recovery_paths,
        unavailable_count,
        healthy_count,
        healthy_domains,
        evaluated_at_unix_ms,
    })
}

fn validate_failure_policies<'a>(
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    policies: &'a [ValidatedFailureDomainPolicyV1],
) -> Result<BTreeMap<FailureDomainPolicyId, &'a ValidatedFailureDomainPolicyV1>, PostTransitionDistributedHealthError>
{
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

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PostTransitionDistributedHealthError {
    #[error("post-transition distributed evaluation time must be non-zero")]
    ZeroEvaluationTime,
    #[error("local post-execution target is not qualified healthy")]
    LocalTargetNotHealthy,
    #[error("local post-execution health belongs to a different distributed context")]
    LocalHealthContextMismatch,
    #[error("local post-execution health and distributed evaluation must share one exact logical evaluation time")]
    LocalHealthEvaluationTimeMismatch,
    #[error("local transitioned subject is outside the exact distributed candidate set")]
    LocalSubjectOutsideCandidateSet,
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
    #[error("transitioned subject is not currently healthy: {participant:?}")]
    TransitionedSubjectNotHealthy { participant: ContinuitySubjectId },
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
    #[error("no authenticated current independent recovery path is available after transition")]
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
    #[error("post-transition distributed currentness evidence set is empty")]
    NoCurrentnessEvidence,
    #[error("cross-evidence observation skew {observed_ms} ms exceeds allowed {allowed_ms} ms")]
    CrossEvidenceSkewExceeded { observed_ms: u64, allowed_ms: u64 },
}

fn hash_current_state(
    local_health_id: QualifiedPostExecutionHealthId,
    context_id: DistributedStateContextId,
    currentness_policy_id: DistributedCurrentnessPolicyId,
    participant_ids: &[AuthenticatedParticipantStateEvidenceId],
    failure_ids: &[AuthenticatedFailureDomainStateEvidenceId],
    recovery_ids: &[AuthenticatedRecoveryPathStateEvidenceId],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CURRENT_STATE_DOMAIN);
    hasher.update(local_health_id.as_bytes());
    hasher.update(context_id.as_bytes());
    hasher.update(currentness_policy_id.as_bytes());
    hash_ids(&mut hasher, participant_ids.iter().map(|id| id.as_bytes()));
    hash_ids(&mut hasher, failure_ids.iter().map(|id| id.as_bytes()));
    hash_ids(&mut hasher, recovery_ids.iter().map(|id| id.as_bytes()));
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn hash_qualified(
    current_state_digest: PostTransitionDistributedStateDigest,
    local_health: &QualifiedPostExecutionHealthV1,
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    verifier_snapshots: &[PostTransitionVerifierSnapshotV1],
    recovery_paths: &[QualifiedRecoveryPathSnapshotV1],
    unavailable_count: u32,
    healthy_count: u32,
    healthy_domains: &[(FailureDomainPolicyId, u32)],
    evaluated_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFIED_DOMAIN);
    hasher.update(current_state_digest.as_bytes());
    hasher.update(local_health.id().as_bytes());
    hasher.update(context.id().as_bytes());
    hasher.update(context.aggregate_subject_id().as_bytes());
    hasher.update(context.budget_id().as_bytes());
    hasher.update(&context.budget_generation().to_le_bytes());
    hasher.update(context.participant_set_digest().as_bytes());
    hasher.update(currentness.id().as_bytes());
    hasher.update(&currentness.policy_generation().to_le_bytes());
    hasher.update(local_health.subject_id().as_bytes());
    hasher.update(&unavailable_count.to_le_bytes());
    hasher.update(&healthy_count.to_le_bytes());
    hasher.update(&evaluated_at_unix_ms.to_le_bytes());
    hasher.update(&(verifier_snapshots.len() as u64).to_le_bytes());
    for snapshot in verifier_snapshots {
        hasher.update(snapshot.profile_id.as_bytes());
        hasher.update(&snapshot.root_epoch.to_le_bytes());
    }
    hasher.update(&(recovery_paths.len() as u64).to_le_bytes());
    for path in recovery_paths {
        encode_recovery_class(&mut hasher, &path.recovery_path_class);
        hasher.update(&path.recovery_path_identity_digest);
    }
    hasher.update(&(healthy_domains.len() as u64).to_le_bytes());
    for (policy_id, count) in healthy_domains {
        hasher.update(policy_id.as_bytes());
        hasher.update(&count.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_ids<'a>(
    hasher: &mut blake3::Hasher,
    ids: impl Iterator<Item = &'a [u8; 32]>,
) {
    let ids = ids.collect::<Vec<_>>();
    hasher.update(&(ids.len() as u64).to_le_bytes());
    for id in ids {
        hasher.update(id);
    }
}

fn encode_recovery_class(hasher: &mut blake3::Hasher, class: &RecoveryPathClassV1) {
    match class {
        RecoveryPathClassV1::DeviceLocalAutomaticRollback => {
            hasher.update(&[1]);
        }
        RecoveryPathClassV1::OutOfBandManagement => {
            hasher.update(&[2]);
        }
        RecoveryPathClassV1::IndependentNetworkPath => {
            hasher.update(&[3]);
        }
        RecoveryPathClassV1::LocalPhysicalIntervention => {
            hasher.update(&[4]);
        }
        RecoveryPathClassV1::Custom { kind_id } => {
            hasher.update(&[255]);
            hasher.update(&(kind_id.len() as u64).to_le_bytes());
            hasher.update(kind_id.as_bytes());
        }
    }
}
