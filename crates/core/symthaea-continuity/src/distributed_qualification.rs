// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-world composition of authenticated distributed currentness evidence.
//!
//! This is the first value in the distributed continuity stack that says a proposed
//! participant set is safe under the exact generic distributed policy world that was
//! evaluated. It is still not commit eligibility and never grants execution authority.
//!
//! Core theorem:
//!
//! `QualifiedDistributedTransitionWitnessV1 != CommitEligibility != ExecutionAuthority`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::{DistributedChangeBudgetId, RecoveryPathClassV1};
use crate::distributed_currentness::{
    DistributedCurrentnessPolicyId, ValidatedDistributedCurrentnessPolicyV1,
};
use crate::distributed_evidence::{
    AuthenticatedFailureDomainStateEvidenceV1, AuthenticatedFailureDomainStateEvidenceId,
    AuthenticatedRecoveryPathStateEvidenceV1, AuthenticatedRecoveryPathStateEvidenceId,
    FailureDomainObservationOutcomeV1, RecoveryPathObservationOutcomeV1,
};
use crate::distributed_state::{
    AuthenticatedParticipantStateEvidenceId, AuthenticatedParticipantStateEvidenceV1,
    DistributedStateContextId, ParticipantOperationalStateV1, ParticipantSetDigest,
    ValidatedDistributedStateContextV1,
};
use crate::failure_domain::{FailureDomainPolicyId, ValidatedFailureDomainPolicyV1};
use crate::scope::ContinuitySubjectId;
use crate::verifier::VerifierProfileId;

const CURRENT_STATE_DOMAIN: &[u8] = b"symthaea.continuity.distributed-current-state.v1\0";
const QUALIFIED_WITNESS_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-distributed-transition.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DistributedCurrentStateDigest([u8; 32]);

impl DistributedCurrentStateDigest {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedDistributedTransitionWitnessId([u8; 32]);

impl QualifiedDistributedTransitionWitnessId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact verifier profile/root-epoch snapshot that contributed authenticated
/// currentness evidence to the distributed decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DistributedVerifierSnapshotV1 {
    profile_id: VerifierProfileId,
    root_epoch: u64,
}

impl DistributedVerifierSnapshotV1 {
    pub fn profile_id(&self) -> VerifierProfileId {
        self.profile_id
    }

    pub fn root_epoch(&self) -> u64 {
        self.root_epoch
    }
}

/// Non-Serde verifier/composer-owned proof that one exact distributed transition
/// satisfied the generic distributed budget, required failure-domain floors,
/// recovery-path availability, verifier allowlists, and freshness policy.
#[derive(Debug, Clone)]
pub struct QualifiedDistributedTransitionWitnessV1 {
    witness_id: QualifiedDistributedTransitionWitnessId,
    context_id: DistributedStateContextId,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    participant_set_digest: ParticipantSetDigest,
    currentness_policy_id: DistributedCurrentnessPolicyId,
    currentness_policy_generation: u64,
    candidate_subject_ids: Vec<ContinuitySubjectId>,
    required_failure_domain_policy_ids: Vec<FailureDomainPolicyId>,
    available_recovery_path_classes: Vec<RecoveryPathClassV1>,
    verifier_snapshots: Vec<DistributedVerifierSnapshotV1>,
    current_state_digest: DistributedCurrentStateDigest,
    evaluated_at_unix_ms: u64,
    current_unavailable_count: u32,
    projected_healthy_count: u32,
    projected_healthy_domains: Vec<(FailureDomainPolicyId, u32)>,
}

impl QualifiedDistributedTransitionWitnessV1 {
    pub fn id(&self) -> QualifiedDistributedTransitionWitnessId {
        self.witness_id
    }

    pub fn context_id(&self) -> DistributedStateContextId {
        self.context_id
    }

    pub fn budget_id(&self) -> DistributedChangeBudgetId {
        self.budget_id
    }

    pub fn budget_generation(&self) -> u64 {
        self.budget_generation
    }

    pub fn participant_set_digest(&self) -> ParticipantSetDigest {
        self.participant_set_digest
    }

    pub fn currentness_policy_id(&self) -> DistributedCurrentnessPolicyId {
        self.currentness_policy_id
    }

    pub fn currentness_policy_generation(&self) -> u64 {
        self.currentness_policy_generation
    }

    pub fn candidate_subject_ids(&self) -> &[ContinuitySubjectId] {
        &self.candidate_subject_ids
    }

    pub fn required_failure_domain_policy_ids(&self) -> &[FailureDomainPolicyId] {
        &self.required_failure_domain_policy_ids
    }

    pub fn available_recovery_path_classes(&self) -> &[RecoveryPathClassV1] {
        &self.available_recovery_path_classes
    }

    pub fn verifier_snapshots(&self) -> &[DistributedVerifierSnapshotV1] {
        &self.verifier_snapshots
    }

    pub fn current_state_digest(&self) -> DistributedCurrentStateDigest {
        self.current_state_digest
    }

    pub fn evaluated_at_unix_ms(&self) -> u64 {
        self.evaluated_at_unix_ms
    }

    pub fn current_unavailable_count(&self) -> u32 {
        self.current_unavailable_count
    }

    pub fn projected_healthy_count(&self) -> u32 {
        self.projected_healthy_count
    }

    pub fn projected_healthy_domains(&self) -> &[(FailureDomainPolicyId, u32)] {
        &self.projected_healthy_domains
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compose_qualified_distributed_transition(
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    failure_domain_policies: &[ValidatedFailureDomainPolicyV1],
    participant_evidence: &[AuthenticatedParticipantStateEvidenceV1],
    failure_domain_evidence: &[AuthenticatedFailureDomainStateEvidenceV1],
    recovery_evidence: &[AuthenticatedRecoveryPathStateEvidenceV1],
    evaluated_at_unix_ms: u64,
) -> Result<QualifiedDistributedTransitionWitnessV1, DistributedQualificationError> {
    if evaluated_at_unix_ms == 0 {
        return Err(DistributedQualificationError::ZeroEvaluationTime);
    }
    validate_currentness_context(context, currentness)?;

    let failure_policy_by_id = validate_failure_policy_inputs(
        context,
        currentness,
        failure_domain_policies,
    )?;

    let mut observation_times = Vec::new();
    let mut verifier_snapshots = BTreeSet::new();
    let mut participant_evidence_ids = Vec::new();
    let mut states = BTreeMap::new();

    for evidence in participant_evidence {
        if evidence.context_id() != context.id() {
            return Err(DistributedQualificationError::EvidenceContextMismatch {
                role: "participant",
            });
        }
        if currentness
            .participant_verifier_profile_ids()
            .binary_search(&evidence.profile_id())
            .is_err()
        {
            return Err(DistributedQualificationError::UnapprovedVerifierProfile {
                role: "participant",
                profile_id: evidence.profile_id(),
            });
        }
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
            return Err(DistributedQualificationError::ParticipantOutsideBudget {
                participant: evidence.participant_subject_id(),
            });
        }
        if states
            .insert(evidence.participant_subject_id(), evidence.state())
            .is_some()
        {
            return Err(DistributedQualificationError::DuplicateParticipantEvidence {
                participant: evidence.participant_subject_id(),
            });
        }
        if evidence.state() == ParticipantOperationalStateV1::Unknown {
            return Err(DistributedQualificationError::UnknownParticipantState {
                participant: evidence.participant_subject_id(),
            });
        }
        observation_times.push(evidence.observed_at_unix_ms());
        participant_evidence_ids.push(evidence.id());
        verifier_snapshots.insert(DistributedVerifierSnapshotV1 {
            profile_id: evidence.profile_id(),
            root_epoch: evidence.root_epoch(),
        });
    }

    if states.len() != context.budget().participant_subject_ids().len() {
        return Err(DistributedQualificationError::IncompleteParticipantEvidence {
            expected: context.budget().participant_subject_ids().len(),
            observed: states.len(),
        });
    }

    for participant in context.budget().participant_subject_ids() {
        if !states.contains_key(participant) {
            return Err(DistributedQualificationError::MissingParticipantEvidence {
                participant: *participant,
            });
        }
    }

    for candidate in context.candidate_subject_ids() {
        if states.get(candidate) != Some(&ParticipantOperationalStateV1::Healthy) {
            return Err(DistributedQualificationError::CandidateNotHealthy {
                candidate: *candidate,
            });
        }
    }

    let mut current_unavailable = BTreeSet::new();
    for (participant, state) in &states {
        if matches!(
            state,
            ParticipantOperationalStateV1::Unhealthy
                | ParticipantOperationalStateV1::Transitioning
        ) {
            current_unavailable.insert(*participant);
        }
    }
    let current_unavailable_count = current_unavailable.len() as u32;

    let mut unavailable_after = current_unavailable.clone();
    unavailable_after.extend(context.candidate_subject_ids().iter().copied());
    if unavailable_after.len() as u32 > context.budget().max_concurrent_unavailable() {
        return Err(DistributedQualificationError::AvailabilityBudgetExceeded {
            unavailable: unavailable_after.len() as u32,
            allowed: context.budget().max_concurrent_unavailable(),
        });
    }

    let projected_healthy_count =
        context.budget().participant_subject_ids().len() as u32 - unavailable_after.len() as u32;
    if projected_healthy_count < context.budget().minimum_healthy() {
        return Err(DistributedQualificationError::MinimumHealthyViolated {
            projected: projected_healthy_count,
            required: context.budget().minimum_healthy(),
        });
    }

    for exclusion in context.budget().mutual_exclusion_sets() {
        let unavailable = exclusion
            .members()
            .iter()
            .filter(|member| unavailable_after.contains(member))
            .count();
        if unavailable > 1 {
            return Err(DistributedQualificationError::MutualExclusionViolated);
        }
    }

    let mut failure_evidence_by_policy = BTreeMap::new();
    let mut failure_evidence_ids = Vec::new();
    for evidence in failure_domain_evidence {
        if evidence.context_id() != context.id() {
            return Err(DistributedQualificationError::EvidenceContextMismatch {
                role: "failure-domain",
            });
        }
        if currentness
            .failure_domain_verifier_profile_ids()
            .binary_search(&evidence.profile_id())
            .is_err()
        {
            return Err(DistributedQualificationError::UnapprovedVerifierProfile {
                role: "failure-domain",
                profile_id: evidence.profile_id(),
            });
        }
        check_freshness(
            "failure-domain",
            evidence.observed_at_unix_ms(),
            evaluated_at_unix_ms,
            currentness.max_failure_domain_age_ms(),
            currentness.max_future_skew_ms(),
        )?;
        if !failure_policy_by_id.contains_key(&evidence.policy_id()) {
            return Err(DistributedQualificationError::UnexpectedFailureDomainEvidence {
                policy_id: evidence.policy_id(),
            });
        }
        if failure_evidence_by_policy
            .insert(evidence.policy_id(), evidence.outcome())
            .is_some()
        {
            return Err(DistributedQualificationError::DuplicateFailureDomainEvidence {
                policy_id: evidence.policy_id(),
            });
        }
        observation_times.push(evidence.observed_at_unix_ms());
        failure_evidence_ids.push(evidence.id());
        verifier_snapshots.insert(DistributedVerifierSnapshotV1 {
            profile_id: evidence.profile_id(),
            root_epoch: evidence.root_epoch(),
        });
    }

    if failure_evidence_by_policy.len()
        != currentness.required_failure_domain_policy_ids().len()
    {
        return Err(DistributedQualificationError::IncompleteFailureDomainEvidence {
            expected: currentness.required_failure_domain_policy_ids().len(),
            observed: failure_evidence_by_policy.len(),
        });
    }

    let mut projected_healthy_domains = Vec::new();
    for policy_id in currentness.required_failure_domain_policy_ids() {
        let outcome = failure_evidence_by_policy
            .get(policy_id)
            .copied()
            .ok_or(DistributedQualificationError::MissingFailureDomainEvidence {
                policy_id: *policy_id,
            })?;
        if outcome != FailureDomainObservationOutcomeV1::MatchesPolicy {
            return Err(DistributedQualificationError::FailureDomainNotCurrent {
                policy_id: *policy_id,
            });
        }
        let policy = failure_policy_by_id
            .get(policy_id)
            .expect("validated required failure-domain policy is present");
        let healthy_domains = policy
            .groups()
            .iter()
            .filter(|group| {
                group
                    .members()
                    .iter()
                    .any(|member| !unavailable_after.contains(member))
            })
            .count() as u32;
        if healthy_domains < policy.minimum_healthy_domains() {
            return Err(DistributedQualificationError::FailureDomainFloorViolated {
                policy_id: *policy_id,
                projected: healthy_domains,
                required: policy.minimum_healthy_domains(),
            });
        }
        projected_healthy_domains.push((*policy_id, healthy_domains));
    }

    let mut recovery_by_class = BTreeMap::new();
    let mut recovery_evidence_ids = Vec::new();
    for evidence in recovery_evidence {
        if evidence.context_id() != context.id() {
            return Err(DistributedQualificationError::EvidenceContextMismatch {
                role: "recovery",
            });
        }
        if currentness
            .recovery_verifier_profile_ids()
            .binary_search(&evidence.profile_id())
            .is_err()
        {
            return Err(DistributedQualificationError::UnapprovedVerifierProfile {
                role: "recovery",
                profile_id: evidence.profile_id(),
            });
        }
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
            return Err(DistributedQualificationError::RecoveryClassOutsideBudget);
        }
        if recovery_by_class
            .insert(evidence.recovery_path_class().clone(), evidence.outcome())
            .is_some()
        {
            return Err(DistributedQualificationError::DuplicateRecoveryEvidence);
        }
        observation_times.push(evidence.observed_at_unix_ms());
        recovery_evidence_ids.push(evidence.id());
        verifier_snapshots.insert(DistributedVerifierSnapshotV1 {
            profile_id: evidence.profile_id(),
            root_epoch: evidence.root_epoch(),
        });
    }

    let mut available_recovery_path_classes: Vec<RecoveryPathClassV1> = recovery_by_class
        .iter()
        .filter_map(|(class, outcome)| {
            (*outcome == RecoveryPathObservationOutcomeV1::Available).then(|| class.clone())
        })
        .collect();
    available_recovery_path_classes.sort();
    if available_recovery_path_classes.is_empty() {
        return Err(DistributedQualificationError::NoAvailableRecoveryPath);
    }

    validate_cross_evidence_skew(
        &observation_times,
        currentness.max_cross_evidence_skew_ms(),
    )?;

    participant_evidence_ids.sort();
    failure_evidence_ids.sort();
    recovery_evidence_ids.sort();
    let verifier_snapshots: Vec<DistributedVerifierSnapshotV1> =
        verifier_snapshots.into_iter().collect();

    let current_state_digest = DistributedCurrentStateDigest(hash_current_state(
        context.id(),
        currentness.id(),
        &participant_evidence_ids,
        &failure_evidence_ids,
        &recovery_evidence_ids,
    ));
    let witness_id = QualifiedDistributedTransitionWitnessId(hash_qualified_witness(
        current_state_digest,
        context,
        currentness,
        &available_recovery_path_classes,
        &verifier_snapshots,
        evaluated_at_unix_ms,
        current_unavailable_count,
        projected_healthy_count,
        &projected_healthy_domains,
    ));

    Ok(QualifiedDistributedTransitionWitnessV1 {
        witness_id,
        context_id: context.id(),
        budget_id: context.budget_id(),
        budget_generation: context.budget_generation(),
        participant_set_digest: context.participant_set_digest(),
        currentness_policy_id: currentness.id(),
        currentness_policy_generation: currentness.policy_generation(),
        candidate_subject_ids: context.candidate_subject_ids().to_vec(),
        required_failure_domain_policy_ids: currentness
            .required_failure_domain_policy_ids()
            .to_vec(),
        available_recovery_path_classes,
        verifier_snapshots,
        current_state_digest,
        evaluated_at_unix_ms,
        current_unavailable_count,
        projected_healthy_count,
        projected_healthy_domains,
    })
}

fn validate_currentness_context(
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
) -> Result<(), DistributedQualificationError> {
    if currentness.budget_id() != context.budget_id()
        || currentness.budget_generation() != context.budget_generation()
    {
        return Err(DistributedQualificationError::CurrentnessPolicyContextMismatch);
    }
    Ok(())
}

fn validate_failure_policy_inputs<'a>(
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    policies: &'a [ValidatedFailureDomainPolicyV1],
) -> Result<BTreeMap<FailureDomainPolicyId, &'a ValidatedFailureDomainPolicyV1>, DistributedQualificationError>
{
    let mut by_id = BTreeMap::new();
    for policy in policies {
        if policy.budget_id() != context.budget_id()
            || policy.budget_generation() != context.budget_generation()
            || policy.aggregate_subject_id() != context.aggregate_subject_id()
        {
            return Err(DistributedQualificationError::FailureDomainPolicyContextMismatch {
                policy_id: policy.id(),
            });
        }
        if currentness
            .required_failure_domain_policy_ids()
            .binary_search(&policy.id())
            .is_err()
        {
            return Err(DistributedQualificationError::UnexpectedFailureDomainPolicy {
                policy_id: policy.id(),
            });
        }
        if by_id.insert(policy.id(), policy).is_some() {
            return Err(DistributedQualificationError::DuplicateFailureDomainPolicy {
                policy_id: policy.id(),
            });
        }
    }
    if by_id.len() != currentness.required_failure_domain_policy_ids().len() {
        return Err(DistributedQualificationError::IncompleteFailureDomainPolicies {
            expected: currentness.required_failure_domain_policy_ids().len(),
            observed: by_id.len(),
        });
    }
    Ok(by_id)
}

fn check_freshness(
    role: &'static str,
    observed_at_unix_ms: u64,
    evaluated_at_unix_ms: u64,
    max_age_ms: u64,
    max_future_skew_ms: u64,
) -> Result<(), DistributedQualificationError> {
    if observed_at_unix_ms > evaluated_at_unix_ms {
        let skew = observed_at_unix_ms - evaluated_at_unix_ms;
        if skew > max_future_skew_ms {
            return Err(DistributedQualificationError::EvidenceFromFuture {
                role,
                skew_ms: skew,
                allowed_ms: max_future_skew_ms,
            });
        }
        return Ok(());
    }
    let age = evaluated_at_unix_ms - observed_at_unix_ms;
    if age > max_age_ms {
        return Err(DistributedQualificationError::StaleEvidence {
            role,
            age_ms: age,
            allowed_ms: max_age_ms,
        });
    }
    Ok(())
}

fn validate_cross_evidence_skew(
    observation_times: &[u64],
    max_skew_ms: u64,
) -> Result<(), DistributedQualificationError> {
    let Some(minimum) = observation_times.iter().min().copied() else {
        return Err(DistributedQualificationError::NoCurrentnessEvidence);
    };
    let maximum = observation_times.iter().max().copied().unwrap_or(minimum);
    let skew = maximum - minimum;
    if skew > max_skew_ms {
        return Err(DistributedQualificationError::CrossEvidenceSkewExceeded {
            observed_ms: skew,
            allowed_ms: max_skew_ms,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DistributedQualificationError {
    #[error("distributed transition evaluation time must be non-zero")]
    ZeroEvaluationTime,
    #[error("distributed currentness policy is bound to a different context budget/generation")]
    CurrentnessPolicyContextMismatch,
    #[error("failure-domain policy {policy_id:?} is bound to a different distributed context")]
    FailureDomainPolicyContextMismatch { policy_id: FailureDomainPolicyId },
    #[error("unexpected failure-domain policy {policy_id:?} outside currentness policy")]
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
    #[error("authenticated participant evidence references subject outside budget: {participant:?}")]
    ParticipantOutsideBudget { participant: ContinuitySubjectId },
    #[error("duplicate authenticated participant evidence for {participant:?}")]
    DuplicateParticipantEvidence { participant: ContinuitySubjectId },
    #[error("participant evidence incomplete: expected {expected}, observed {observed}")]
    IncompleteParticipantEvidence { expected: usize, observed: usize },
    #[error("missing authenticated participant evidence for {participant:?}")]
    MissingParticipantEvidence { participant: ContinuitySubjectId },
    #[error("participant state is UNKNOWN and cannot be counted safely: {participant:?}")]
    UnknownParticipantState { participant: ContinuitySubjectId },
    #[error("transition candidate is not currently healthy: {candidate:?}")]
    CandidateNotHealthy { candidate: ContinuitySubjectId },
    #[error("projected unavailable count {unavailable} exceeds distributed budget {allowed}")]
    AvailabilityBudgetExceeded { unavailable: u32, allowed: u32 },
    #[error("projected healthy count {projected} is below required minimum {required}")]
    MinimumHealthyViolated { projected: u32, required: u32 },
    #[error("projected unavailable set violates a distributed mutual-exclusion set")]
    MutualExclusionViolated,
    #[error("unexpected failure-domain evidence for policy {policy_id:?}")]
    UnexpectedFailureDomainEvidence { policy_id: FailureDomainPolicyId },
    #[error("duplicate failure-domain evidence for policy {policy_id:?}")]
    DuplicateFailureDomainEvidence { policy_id: FailureDomainPolicyId },
    #[error("failure-domain evidence incomplete: expected {expected}, observed {observed}")]
    IncompleteFailureDomainEvidence { expected: usize, observed: usize },
    #[error("missing authenticated failure-domain evidence for policy {policy_id:?}")]
    MissingFailureDomainEvidence { policy_id: FailureDomainPolicyId },
    #[error("failure-domain policy {policy_id:?} is not currently observed to match")]
    FailureDomainNotCurrent { policy_id: FailureDomainPolicyId },
    #[error("failure-domain policy {policy_id:?} projected healthy domains {projected} below required {required}")]
    FailureDomainFloorViolated {
        policy_id: FailureDomainPolicyId,
        projected: u32,
        required: u32,
    },
    #[error("recovery evidence names a class outside the distributed budget")]
    RecoveryClassOutsideBudget,
    #[error("duplicate recovery evidence for the same recovery-path class")]
    DuplicateRecoveryEvidence,
    #[error("no authenticated current independent recovery path is available")]
    NoAvailableRecoveryPath,
    #[error("authenticated {role} evidence is stale by policy: age {age_ms} ms > {allowed_ms} ms")]
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

fn hash_current_state(
    context_id: DistributedStateContextId,
    currentness_policy_id: DistributedCurrentnessPolicyId,
    participant_evidence_ids: &[AuthenticatedParticipantStateEvidenceId],
    failure_domain_evidence_ids: &[AuthenticatedFailureDomainStateEvidenceId],
    recovery_evidence_ids: &[AuthenticatedRecoveryPathStateEvidenceId],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(512);
    bytes.extend_from_slice(context_id.as_bytes());
    bytes.extend_from_slice(currentness_policy_id.as_bytes());
    put_len(&mut bytes, participant_evidence_ids.len());
    for id in participant_evidence_ids {
        bytes.extend_from_slice(id.as_bytes());
    }
    put_len(&mut bytes, failure_domain_evidence_ids.len());
    for id in failure_domain_evidence_ids {
        bytes.extend_from_slice(id.as_bytes());
    }
    put_len(&mut bytes, recovery_evidence_ids.len());
    for id in recovery_evidence_ids {
        bytes.extend_from_slice(id.as_bytes());
    }
    domain_hash(CURRENT_STATE_DOMAIN, &bytes)
}

#[allow(clippy::too_many_arguments)]
fn hash_qualified_witness(
    current_state_digest: DistributedCurrentStateDigest,
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    available_recovery_path_classes: &[RecoveryPathClassV1],
    verifier_snapshots: &[DistributedVerifierSnapshotV1],
    evaluated_at_unix_ms: u64,
    current_unavailable_count: u32,
    projected_healthy_count: u32,
    projected_healthy_domains: &[(FailureDomainPolicyId, u32)],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(512);
    bytes.extend_from_slice(current_state_digest.as_bytes());
    bytes.extend_from_slice(context.id().as_bytes());
    bytes.extend_from_slice(currentness.id().as_bytes());
    bytes.extend_from_slice(&evaluated_at_unix_ms.to_le_bytes());
    bytes.extend_from_slice(&current_unavailable_count.to_le_bytes());
    bytes.extend_from_slice(&projected_healthy_count.to_le_bytes());
    put_len(&mut bytes, available_recovery_path_classes.len());
    for class in available_recovery_path_classes {
        encode_recovery_class(&mut bytes, class);
    }
    put_len(&mut bytes, verifier_snapshots.len());
    for snapshot in verifier_snapshots {
        bytes.extend_from_slice(snapshot.profile_id.as_bytes());
        bytes.extend_from_slice(&snapshot.root_epoch.to_le_bytes());
    }
    put_len(&mut bytes, projected_healthy_domains.len());
    for (policy_id, count) in projected_healthy_domains {
        bytes.extend_from_slice(policy_id.as_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    domain_hash(QUALIFIED_WITNESS_DOMAIN, &bytes)
}

fn encode_recovery_class(out: &mut Vec<u8>, class: &RecoveryPathClassV1) {
    match class {
        RecoveryPathClassV1::DeviceLocalAutomaticRollback => out.push(1),
        RecoveryPathClassV1::OutOfBandManagement => out.push(2),
        RecoveryPathClassV1::IndependentNetworkPath => out.push(3),
        RecoveryPathClassV1::LocalPhysicalIntervention => out.push(4),
        RecoveryPathClassV1::Custom { kind_id } => {
            out.push(255);
            put_len(out, kind_id.len());
            out.extend_from_slice(kind_id.as_bytes());
        }
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        DistributedChangeBudgetV1, MutualExclusionSetV1, RecoveryPathClassV1,
    };
    use crate::distributed_currentness::DistributedCurrentnessPolicyV1;
    use crate::distributed_evidence::{
        FailureDomainStateClaimV1, RecoveryPathStateClaimV1,
        policy_check_failure_domain_state_claim, policy_check_recovery_path_state_claim,
    };
    use crate::distributed_state::{
        DistributedStateContextV1, ParticipantStateClaimV1,
        policy_check_participant_state_claim,
    };
    use crate::failure_domain::{
        FailureDomainGroupV1, FailureDomainKindV1, FailureDomainPolicyV1,
    };
    use crate::scope::{ContinuityScopeV1, ContinuitySubjectV1};
    use crate::verifier::VerifierProfileV1;
    use crate::witness::EvidenceClass;

    struct Fixture {
        participants: Vec<ContinuitySubjectV1>,
        context: ValidatedDistributedStateContextV1,
        domain_policy: ValidatedFailureDomainPolicyV1,
        currentness: ValidatedDistributedCurrentnessPolicyV1,
        participant_profile: VerifierProfileV1,
        domain_profile: VerifierProfileV1,
        recovery_profile: VerifierProfileV1,
    }

    fn subject(logical_id: &str, scope: ContinuityScopeV1) -> ContinuitySubjectV1 {
        ContinuitySubjectV1::new("org.example", logical_id, scope, None).unwrap()
    }

    fn profile(name: &str, seed: u8) -> VerifierProfileV1 {
        VerifierProfileV1::new(name, [seed; 32], seed as u64, EvidenceClass::HardwareVerified)
            .unwrap()
    }

    fn fixture(
        max_unavailable: u32,
        minimum_healthy: u32,
        minimum_healthy_domains: u32,
        exclusions: Vec<MutualExclusionSetV1>,
    ) -> Fixture {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let participants = vec![
            subject("node-a", ContinuityScopeV1::Machine),
            subject("node-b", ContinuityScopeV1::Machine),
            subject("node-c", ContinuityScopeV1::Machine),
        ];
        let raw_budget = DistributedChangeBudgetV1::new(
            &aggregate,
            5,
            participants.iter().map(ContinuitySubjectV1::id).collect(),
            max_unavailable,
            minimum_healthy,
            exclusions,
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let budget = raw_budget.validate_against_subject(&aggregate).unwrap();
        let context = DistributedStateContextV1::new(&budget, vec![participants[0].id()], [9; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let raw_domain = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "rack-placement",
            vec![
                FailureDomainGroupV1::new(
                    "rack-a",
                    vec![participants[0].id(), participants[1].id()],
                )
                .unwrap(),
                FailureDomainGroupV1::new("rack-b", vec![participants[2].id()]).unwrap(),
            ],
            minimum_healthy_domains,
        )
        .unwrap();
        let domain_policy = raw_domain.validate_against_budget(&budget).unwrap();
        let participant_profile = profile("participant-state", 1);
        let domain_profile = profile("failure-domain-state", 2);
        let recovery_profile = profile("recovery-state", 3);
        let raw_currentness = DistributedCurrentnessPolicyV1::new(
            &budget,
            2,
            vec![participant_profile.id()],
            vec![domain_profile.id()],
            vec![recovery_profile.id()],
            vec![domain_policy.id()],
            1_000,
            1_000,
            1_000,
            100,
            500,
        )
        .unwrap();
        let currentness = raw_currentness.validate_against_budget(&budget).unwrap();
        Fixture {
            participants,
            context,
            domain_policy,
            currentness,
            participant_profile,
            domain_profile,
            recovery_profile,
        }
    }

    fn participant_evidence(
        fixture: &Fixture,
        index: usize,
        state: ParticipantOperationalStateV1,
        observed_at: u64,
    ) -> AuthenticatedParticipantStateEvidenceV1 {
        let protocol = (!matches!(state, ParticipantOperationalStateV1::Unknown))
            .then_some([20 + index as u8; 32]);
        let claim = ParticipantStateClaimV1::new(
            &fixture.context,
            fixture.participants[index].id(),
            fixture.participant_profile.id(),
            observed_at,
            state,
            protocol,
            [30 + index as u8; 32],
        )
        .unwrap();
        let checked = policy_check_participant_state_claim(
            &fixture.context,
            &fixture.participant_profile,
            claim,
        )
        .unwrap();
        AuthenticatedParticipantStateEvidenceV1::authenticate_for_test(checked, [40 + index as u8; 32])
            .unwrap()
    }

    fn domain_evidence(
        fixture: &Fixture,
        outcome: FailureDomainObservationOutcomeV1,
        observed_at: u64,
    ) -> AuthenticatedFailureDomainStateEvidenceV1 {
        let claim = FailureDomainStateClaimV1::new(
            &fixture.context,
            &fixture.domain_policy,
            fixture.domain_profile.id(),
            observed_at,
            outcome,
            [51; 32],
        )
        .unwrap();
        let checked = policy_check_failure_domain_state_claim(
            &fixture.context,
            &fixture.domain_policy,
            &fixture.domain_profile,
            claim,
        )
        .unwrap();
        AuthenticatedFailureDomainStateEvidenceV1::authenticate_for_test(checked, [52; 32])
            .unwrap()
    }

    fn recovery_evidence(
        fixture: &Fixture,
        outcome: RecoveryPathObservationOutcomeV1,
        observed_at: u64,
    ) -> AuthenticatedRecoveryPathStateEvidenceV1 {
        let (identity, independence) = if outcome == RecoveryPathObservationOutcomeV1::Available {
            (Some([61; 32]), Some([62; 32]))
        } else if outcome == RecoveryPathObservationOutcomeV1::Unavailable {
            (Some([61; 32]), None)
        } else {
            (None, None)
        };
        let claim = RecoveryPathStateClaimV1::new(
            &fixture.context,
            RecoveryPathClassV1::OutOfBandManagement,
            fixture.recovery_profile.id(),
            observed_at,
            outcome,
            identity,
            independence,
            [63; 32],
        )
        .unwrap();
        let checked = policy_check_recovery_path_state_claim(
            &fixture.context,
            &fixture.recovery_profile,
            claim,
        )
        .unwrap();
        AuthenticatedRecoveryPathStateEvidenceV1::authenticate_for_test(checked, [64; 32])
            .unwrap()
    }

    fn healthy_participants(fixture: &Fixture, observed_at: u64) -> Vec<AuthenticatedParticipantStateEvidenceV1> {
        (0..fixture.participants.len())
            .map(|index| {
                participant_evidence(
                    fixture,
                    index,
                    ParticipantOperationalStateV1::Healthy,
                    observed_at,
                )
            })
            .collect()
    }

    #[test]
    fn exact_current_world_qualifies_distributed_transition() {
        let fixture = fixture(1, 2, 1, vec![]);
        let witness = compose_qualified_distributed_transition(
            &fixture.context,
            &fixture.currentness,
            std::slice::from_ref(&fixture.domain_policy),
            &healthy_participants(&fixture, 10_000),
            &[domain_evidence(
                &fixture,
                FailureDomainObservationOutcomeV1::MatchesPolicy,
                10_050,
            )],
            &[recovery_evidence(
                &fixture,
                RecoveryPathObservationOutcomeV1::Available,
                10_100,
            )],
            10_500,
        )
        .unwrap();
        assert_eq!(witness.projected_healthy_count(), 2);
        assert_eq!(witness.current_unavailable_count(), 0);
        assert_eq!(
            witness.available_recovery_path_classes(),
            &[RecoveryPathClassV1::OutOfBandManagement]
        );
    }

    #[test]
    fn locally_healthy_candidate_is_denied_when_another_member_is_down() {
        let fixture = fixture(1, 2, 1, vec![]);
        let evidence = vec![
            participant_evidence(&fixture, 0, ParticipantOperationalStateV1::Healthy, 10_000),
            participant_evidence(&fixture, 1, ParticipantOperationalStateV1::Unhealthy, 10_000),
            participant_evidence(&fixture, 2, ParticipantOperationalStateV1::Healthy, 10_000),
        ];
        assert_eq!(
            compose_qualified_distributed_transition(
                &fixture.context,
                &fixture.currentness,
                std::slice::from_ref(&fixture.domain_policy),
                &evidence,
                &[domain_evidence(
                    &fixture,
                    FailureDomainObservationOutcomeV1::MatchesPolicy,
                    10_000,
                )],
                &[recovery_evidence(
                    &fixture,
                    RecoveryPathObservationOutcomeV1::Available,
                    10_000,
                )],
                10_500,
            )
            .unwrap_err(),
            DistributedQualificationError::AvailabilityBudgetExceeded {
                unavailable: 2,
                allowed: 1,
            }
        );
    }

    #[test]
    fn unknown_participant_state_fails_closed() {
        let fixture = fixture(1, 2, 1, vec![]);
        let evidence = vec![
            participant_evidence(&fixture, 0, ParticipantOperationalStateV1::Healthy, 10_000),
            participant_evidence(&fixture, 1, ParticipantOperationalStateV1::Unknown, 10_000),
            participant_evidence(&fixture, 2, ParticipantOperationalStateV1::Healthy, 10_000),
        ];
        assert!(matches!(
            compose_qualified_distributed_transition(
                &fixture.context,
                &fixture.currentness,
                std::slice::from_ref(&fixture.domain_policy),
                &evidence,
                &[domain_evidence(
                    &fixture,
                    FailureDomainObservationOutcomeV1::MatchesPolicy,
                    10_000,
                )],
                &[recovery_evidence(
                    &fixture,
                    RecoveryPathObservationOutcomeV1::Available,
                    10_000,
                )],
                10_500,
            ),
            Err(DistributedQualificationError::UnknownParticipantState { .. })
        ));
    }

    #[test]
    fn failure_domain_mismatch_cannot_qualify() {
        let fixture = fixture(1, 2, 1, vec![]);
        assert!(matches!(
            compose_qualified_distributed_transition(
                &fixture.context,
                &fixture.currentness,
                std::slice::from_ref(&fixture.domain_policy),
                &healthy_participants(&fixture, 10_000),
                &[domain_evidence(
                    &fixture,
                    FailureDomainObservationOutcomeV1::Mismatch,
                    10_000,
                )],
                &[recovery_evidence(
                    &fixture,
                    RecoveryPathObservationOutcomeV1::Available,
                    10_000,
                )],
                10_500,
            ),
            Err(DistributedQualificationError::FailureDomainNotCurrent { .. })
        ));
    }

    #[test]
    fn failure_domain_floor_can_deny_even_when_global_healthy_floor_passes() {
        let fixture = fixture(2, 1, 2, vec![]);
        let evidence = vec![
            participant_evidence(&fixture, 0, ParticipantOperationalStateV1::Healthy, 10_000),
            participant_evidence(&fixture, 1, ParticipantOperationalStateV1::Unhealthy, 10_000),
            participant_evidence(&fixture, 2, ParticipantOperationalStateV1::Healthy, 10_000),
        ];
        assert!(matches!(
            compose_qualified_distributed_transition(
                &fixture.context,
                &fixture.currentness,
                std::slice::from_ref(&fixture.domain_policy),
                &evidence,
                &[domain_evidence(
                    &fixture,
                    FailureDomainObservationOutcomeV1::MatchesPolicy,
                    10_000,
                )],
                &[recovery_evidence(
                    &fixture,
                    RecoveryPathObservationOutcomeV1::Available,
                    10_000,
                )],
                10_500,
            ),
            Err(DistributedQualificationError::FailureDomainFloorViolated { .. })
        ));
    }

    #[test]
    fn unavailable_recovery_path_fails_closed() {
        let fixture = fixture(1, 2, 1, vec![]);
        assert_eq!(
            compose_qualified_distributed_transition(
                &fixture.context,
                &fixture.currentness,
                std::slice::from_ref(&fixture.domain_policy),
                &healthy_participants(&fixture, 10_000),
                &[domain_evidence(
                    &fixture,
                    FailureDomainObservationOutcomeV1::MatchesPolicy,
                    10_000,
                )],
                &[recovery_evidence(
                    &fixture,
                    RecoveryPathObservationOutcomeV1::Unavailable,
                    10_000,
                )],
                10_500,
            )
            .unwrap_err(),
            DistributedQualificationError::NoAvailableRecoveryPath
        );
    }

    #[test]
    fn stale_authenticated_evidence_is_not_current_evidence() {
        let fixture = fixture(1, 2, 1, vec![]);
        assert!(matches!(
            compose_qualified_distributed_transition(
                &fixture.context,
                &fixture.currentness,
                std::slice::from_ref(&fixture.domain_policy),
                &healthy_participants(&fixture, 8_000),
                &[domain_evidence(
                    &fixture,
                    FailureDomainObservationOutcomeV1::MatchesPolicy,
                    10_000,
                )],
                &[recovery_evidence(
                    &fixture,
                    RecoveryPathObservationOutcomeV1::Available,
                    10_000,
                )],
                10_500,
            ),
            Err(DistributedQualificationError::StaleEvidence {
                role: "participant",
                ..
            })
        ));
    }

    #[test]
    fn existing_unavailable_peer_plus_candidate_violates_mutual_exclusion() {
        let a = subject("node-a", ContinuityScopeV1::Machine);
        let b = subject("node-b", ContinuityScopeV1::Machine);
        let exclusion = MutualExclusionSetV1::new(vec![a.id(), b.id()]).unwrap();

        let aggregate = subject("cluster-x", ContinuityScopeV1::Cluster);
        let c = subject("node-c", ContinuityScopeV1::Machine);
        let raw_budget = DistributedChangeBudgetV1::new(
            &aggregate,
            6,
            vec![a.id(), b.id(), c.id()],
            2,
            1,
            vec![exclusion],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let budget = raw_budget.validate_against_subject(&aggregate).unwrap();
        let context = DistributedStateContextV1::new(&budget, vec![a.id()], [9; 32])
            .unwrap()
            .validate_against_budget(&budget)
            .unwrap();
        let raw_domain = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "rack-placement",
            vec![
                FailureDomainGroupV1::new("rack-a", vec![a.id(), b.id()]).unwrap(),
                FailureDomainGroupV1::new("rack-b", vec![c.id()]).unwrap(),
            ],
            1,
        )
        .unwrap();
        let domain_policy = raw_domain.validate_against_budget(&budget).unwrap();
        let participant_profile = profile("participant-state-x", 4);
        let domain_profile = profile("domain-state-x", 5);
        let recovery_profile = profile("recovery-state-x", 6);
        let raw_currentness = DistributedCurrentnessPolicyV1::new(
            &budget,
            1,
            vec![participant_profile.id()],
            vec![domain_profile.id()],
            vec![recovery_profile.id()],
            vec![domain_policy.id()],
            1_000,
            1_000,
            1_000,
            100,
            500,
        )
        .unwrap();
        let currentness = raw_currentness.validate_against_budget(&budget).unwrap();
        let participants = vec![a, b, c];
        let fixture = Fixture {
            participants,
            context,
            domain_policy,
            currentness,
            participant_profile,
            domain_profile,
            recovery_profile,
        };
        let evidence = vec![
            participant_evidence(&fixture, 0, ParticipantOperationalStateV1::Healthy, 10_000),
            participant_evidence(&fixture, 1, ParticipantOperationalStateV1::Unhealthy, 10_000),
            participant_evidence(&fixture, 2, ParticipantOperationalStateV1::Healthy, 10_000),
        ];
        assert_eq!(
            compose_qualified_distributed_transition(
                &fixture.context,
                &fixture.currentness,
                std::slice::from_ref(&fixture.domain_policy),
                &evidence,
                &[domain_evidence(
                    &fixture,
                    FailureDomainObservationOutcomeV1::MatchesPolicy,
                    10_000,
                )],
                &[recovery_evidence(
                    &fixture,
                    RecoveryPathObservationOutcomeV1::Available,
                    10_000,
                )],
                10_500,
            )
            .unwrap_err(),
            DistributedQualificationError::MutualExclusionViolated
        );
    }
}
