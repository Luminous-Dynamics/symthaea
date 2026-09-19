// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cadence-conformant institutional currentness at one later evaluation horizon.
//!
//! This capability does not create a fresh scientific judgment. It composes:
//! - one previously authenticated positive institutional-currentness decision;
//! - replay-bound proof of the exact policies that produced that decision;
//! - a complete non-weakening authorized cadence-policy lineage whose genesis was
//!   already authorized when the positive currentness decision was made;
//! - proof that the latest cadence policy is the latest publication through the
//!   exact later federated head;
//! - cross-stream proof that neither a newer currentness observation nor a newer
//!   lifecycle event appears through that same head; and
//! - a replay of the later fresh-head federation under an exact policy satisfying
//!   the authorized cadence limits.
//!
//! The resulting type is intentionally scoped to one authenticated later
//! evaluation interval. It does not promise currentness after that interval,
//! globally latest visibility, global evidence exhaustiveness, or scientific
//! truth.

use serde::Serialize;
use symthaea_trust_core::{
    federate_fresh_witnessed_heads, NamespacedTransparencyMonitorReceipt,
    NamespacedWitnessedTransparencyCheckpoint, Sha256Digest as TrustSha256Digest,
    TransparencyHeadFederationFinding, TransparencyHeadFederationPolicy,
    TransparencyHeadFederationReceipt, TrustedTime, VerifiedTransparencyWitnessQuorum,
};

use crate::{
    evidence_freshness_policy_digest, head_federation_policy_digest,
    lifecycle_freshness_policy_digest, AuthorizedQualificationCurrentnessCadencePolicy,
    EvidenceFreshnessPolicy, InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    LifecycleHeadFreshnessPolicy, NonWeakeningQualificationCurrentnessCadenceLineage,
    PolicyBoundQualificationValidityReadiness, QualificationCurrentnessCadenceHeadAssessment,
    QualificationCurrentnessCadenceHeadClosure, QualificationCurrentnessGuardHeadBundle,
    QualificationCurrentnessGuardHeadClosure, ResearchId, Sha256Digest,
};

const CADENCE_CONFORMANT_CURRENTNESS_DOMAIN: &str =
    "symthaea.cadence-conformant-currentness-at-evaluation.identity.v1";

pub struct LaterHeadFederationReplayInputs<'a> {
    pub views: &'a [NamespacedWitnessedTransparencyCheckpoint],
    pub quorums: &'a [VerifiedTransparencyWitnessQuorum],
    pub monitor: &'a NamespacedTransparencyMonitorReceipt,
    pub policy: &'a TransparencyHeadFederationPolicy,
}

pub struct QualificationCadenceCurrentnessInputs<'a> {
    pub current: &'a InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    pub policy_bound_readiness: &'a PolicyBoundQualificationValidityReadiness,
    pub original_head_policy: &'a TransparencyHeadFederationPolicy,
    pub original_lifecycle_policy: &'a LifecycleHeadFreshnessPolicy,
    pub original_evidence_policy: &'a EvidenceFreshnessPolicy,
    pub cadence: &'a AuthorizedQualificationCurrentnessCadencePolicy,
    pub cadence_lineage: &'a NonWeakeningQualificationCurrentnessCadenceLineage,
    /// Exact authorized policies used to derive `cadence_lineage`, ordered from
    /// genesis through `cadence`.
    pub cadence_policies: &'a [AuthorizedQualificationCurrentnessCadencePolicy],
    pub cadence_head: &'a QualificationCurrentnessCadenceHeadAssessment,
    pub currentness_guard: &'a QualificationCurrentnessGuardHeadBundle,
    pub later_head: LaterHeadFederationReplayInputs<'a>,
    pub evaluation_time: &'a TrustedTime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationCadenceCurrentnessError {
    QualificationMismatch,
    CurrentnessReadinessMismatch,
    PolicyBoundReadinessNotEstablished,
    OriginalHeadPolicyMismatch,
    OriginalLifecyclePolicyMismatch,
    OriginalEvidencePolicyMismatch,
    EmptyCadencePolicyLineage,
    CadenceLineageMismatch,
    CadencePolicyListMismatch,
    CadenceGenesisAuthorizedAfterOriginalEvaluation,
    CadenceLatestAuthorizedAfterLaterEvaluation,
    CadenceHeadNotLatest,
    CadenceHeadMismatch,
    CurrentnessGuardNotEstablished,
    CurrentnessGuardMismatch,
    HeadAssessmentMismatch,
    RootAuthorityMismatch,
    EvaluationIntervalsOverlapOrRegress,
    CurrentnessAgeExceeded,
    OriginalHeadAgePolicyTooWeak,
    OriginalLifecycleAgePolicyTooWeak,
    OriginalEvidenceLagPolicyTooWeak,
    OriginalConvergedViewsTooWeak,
    OriginalDistinctQuorumsTooWeak,
    OriginalDistinctPrincipalsTooWeak,
    OriginalDistinctOrganizationsTooWeak,
    OriginalDistinctRegionsTooWeak,
    LaterHeadAgePolicyTooWeak,
    LaterConvergedViewsTooWeak,
    LaterDistinctQuorumsTooWeak,
    LaterDistinctPrincipalsTooWeak,
    LaterDistinctOrganizationsTooWeak,
    LaterDistinctRegionsTooWeak,
    LaterHeadFederationRejected(Vec<TransparencyHeadFederationFinding>),
    LaterHeadFederationMismatch,
    TimeDependsOnComposedCurrentnessEvidence,
}

/// Non-deserializable capability establishing cadence-conformant institutional
/// currentness only at one exact later authenticated evaluation interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CadenceConformantInstitutionalCurrentnessAtEvaluation {
    qualification_sha256: Sha256Digest,
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    original_currentness_sha256: TrustSha256Digest,
    policy_bound_readiness_sha256: TrustSha256Digest,
    cadence_policy_sha256: TrustSha256Digest,
    cadence_policy_authority_sha256: TrustSha256Digest,
    cadence_lineage_sha256: TrustSha256Digest,
    cadence_head_assessment_sha256: TrustSha256Digest,
    currentness_guard_assessment_sha256: TrustSha256Digest,
    later_head_federation_receipt_sha256: TrustSha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    currentness_age_upper_bound_s: u64,
    capability_sha256: TrustSha256Digest,
}

impl CadenceConformantInstitutionalCurrentnessAtEvaluation {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn claim_id(&self) -> &ResearchId { &self.claim_id }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn original_currentness_sha256(&self) -> &TrustSha256Digest {
        &self.original_currentness_sha256
    }
    pub fn policy_bound_readiness_sha256(&self) -> &TrustSha256Digest {
        &self.policy_bound_readiness_sha256
    }
    pub fn cadence_policy_sha256(&self) -> &TrustSha256Digest { &self.cadence_policy_sha256 }
    pub fn cadence_policy_authority_sha256(&self) -> &TrustSha256Digest {
        &self.cadence_policy_authority_sha256
    }
    pub fn cadence_lineage_sha256(&self) -> &TrustSha256Digest { &self.cadence_lineage_sha256 }
    pub fn cadence_head_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.cadence_head_assessment_sha256
    }
    pub fn currentness_guard_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.currentness_guard_assessment_sha256
    }
    pub fn later_head_federation_receipt_sha256(&self) -> &TrustSha256Digest {
        &self.later_head_federation_receipt_sha256
    }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn currentness_age_upper_bound_s(&self) -> u64 { self.currentness_age_upper_bound_s }
    pub fn capability_sha256(&self) -> &TrustSha256Digest { &self.capability_sha256 }

    pub const fn cadence_conformant_currentness_at_evaluation_established(&self) -> bool { true }
    pub const fn currentness_at_later_time_established(&self) -> bool { false }
    pub const fn globally_latest_head_established(&self) -> bool { false }
    pub const fn globally_latest_policy_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn global_evidence_exhaustiveness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn establish_cadence_conformant_currentness_at_evaluation(
    inputs: QualificationCadenceCurrentnessInputs<'_>,
) -> Result<CadenceConformantInstitutionalCurrentnessAtEvaluation, QualificationCadenceCurrentnessError> {
    let current = inputs.current;
    let qualification = current.qualification_sha256();
    if inputs.policy_bound_readiness.qualification_sha256() != qualification
        || inputs.cadence.qualification_sha256() != qualification
        || inputs.cadence_lineage.qualification_sha256() != qualification
        || inputs.cadence_head.qualification_sha256() != qualification
        || inputs.currentness_guard.guard().qualification_sha256() != qualification
    {
        return Err(QualificationCadenceCurrentnessError::QualificationMismatch);
    }

    if inputs.policy_bound_readiness.readiness().readiness().assessment_sha256()
        != current.readiness_assessment_sha256()
    {
        return Err(QualificationCadenceCurrentnessError::CurrentnessReadinessMismatch);
    }
    if !inputs.policy_bound_readiness.policy_bound_readiness_established() {
        return Err(QualificationCadenceCurrentnessError::PolicyBoundReadinessNotEstablished);
    }
    if inputs.policy_bound_readiness.head_federation_policy_sha256()
        != &head_federation_policy_digest(inputs.original_head_policy)
    {
        return Err(QualificationCadenceCurrentnessError::OriginalHeadPolicyMismatch);
    }
    if inputs.policy_bound_readiness.lifecycle_freshness_policy_sha256()
        != &lifecycle_freshness_policy_digest(inputs.original_lifecycle_policy)
    {
        return Err(QualificationCadenceCurrentnessError::OriginalLifecyclePolicyMismatch);
    }
    if inputs.policy_bound_readiness.evidence_freshness_policy_sha256()
        != &evidence_freshness_policy_digest(inputs.original_evidence_policy)
    {
        return Err(QualificationCadenceCurrentnessError::OriginalEvidencePolicyMismatch);
    }

    if inputs.cadence_policies.is_empty() {
        return Err(QualificationCadenceCurrentnessError::EmptyCadencePolicyLineage);
    }
    let policy_sha256s: Vec<_> = inputs
        .cadence_policies
        .iter()
        .map(|policy| policy.policy_sha256().clone())
        .collect();
    let authority_sha256s: Vec<_> = inputs
        .cadence_policies
        .iter()
        .map(|policy| policy.authority_sha256().clone())
        .collect();
    if policy_sha256s != inputs.cadence_lineage.policy_sha256s()
        || authority_sha256s != inputs.cadence_lineage.policy_authority_sha256s()
    {
        return Err(QualificationCadenceCurrentnessError::CadencePolicyListMismatch);
    }
    if inputs.cadence_lineage.latest_policy_sha256() != inputs.cadence.policy_sha256()
        || inputs.cadence_lineage.latest_policy_authority_sha256()
            != inputs.cadence.authority_sha256()
        || inputs.cadence_lineage.latest_policy_version() != inputs.cadence.version()
    {
        return Err(QualificationCadenceCurrentnessError::CadenceLineageMismatch);
    }

    let (current_earliest, current_latest) = current.evaluation_interval();
    let (evaluation_earliest, evaluation_latest) = inputs.evaluation_time.consensus_interval();
    if inputs.cadence_policies[0].authorized_at_unix_s() > current_earliest {
        return Err(
            QualificationCadenceCurrentnessError::CadenceGenesisAuthorizedAfterOriginalEvaluation,
        );
    }
    if inputs.cadence.authorized_at_unix_s() > evaluation_earliest {
        return Err(
            QualificationCadenceCurrentnessError::CadenceLatestAuthorizedAfterLaterEvaluation,
        );
    }
    if evaluation_earliest < current_latest {
        return Err(QualificationCadenceCurrentnessError::EvaluationIntervalsOverlapOrRegress);
    }

    if inputs.cadence_head.closure()
        != QualificationCurrentnessCadenceHeadClosure::LatestPublishedPolicyThroughFederatedHead
        || !inputs
            .cadence_head
            .latest_published_policy_through_observed_head_established()
    {
        return Err(QualificationCadenceCurrentnessError::CadenceHeadNotLatest);
    }
    if inputs.cadence_head.policy_sha256() != inputs.cadence.policy_sha256()
        || inputs.cadence_head.policy_authority_sha256() != inputs.cadence.authority_sha256()
        || inputs.cadence_head.policy_version() != inputs.cadence.version()
    {
        return Err(QualificationCadenceCurrentnessError::CadenceHeadMismatch);
    }

    if inputs.currentness_guard.guard().closure()
        != QualificationCurrentnessGuardHeadClosure::GuardedThroughFederatedHead
        || !inputs
            .currentness_guard
            .guard()
            .no_later_currentness_or_lifecycle_publication_through_observed_head_established()
    {
        return Err(QualificationCadenceCurrentnessError::CurrentnessGuardNotEstablished);
    }
    if inputs.currentness_guard.guard().currentness_sha256() != current.currentness_sha256()
        || inputs.currentness_guard.guard().lifecycle_event_sha256()
            != current.lifecycle_event_sha256()
    {
        return Err(QualificationCadenceCurrentnessError::CurrentnessGuardMismatch);
    }

    if inputs.currentness_guard.guard().observed_head_namespaced_view_sha256()
        != inputs.cadence_head.observed_head_namespaced_view_sha256()
        || inputs.currentness_guard.guard().observed_head_tree_size()
            != inputs.cadence_head.observed_head_tree_size()
        || inputs.currentness_guard.currentness_head().head_federation_receipt_sha256()
            != inputs.cadence_head.head_federation_receipt_sha256()
    {
        return Err(QualificationCadenceCurrentnessError::HeadAssessmentMismatch);
    }

    if current.root_authority_sha256() != inputs.cadence.root_authority_sha256()
        || current.root_authority_sha256() != inputs.evaluation_time.root_authority_sha256()
    {
        return Err(QualificationCadenceCurrentnessError::RootAuthorityMismatch);
    }

    let limits = inputs.cadence.policy().limits();
    validate_policy_strength(
        inputs.original_head_policy,
        inputs.original_lifecycle_policy,
        inputs.original_evidence_policy,
        limits,
        true,
    )?;
    validate_head_policy_strength(inputs.later_head.policy, limits, false)?;

    let later_head_federation = federate_fresh_witnessed_heads(
        inputs.later_head.views,
        inputs.later_head.quorums,
        inputs.later_head.monitor,
        inputs.evaluation_time,
        inputs.later_head.policy,
    )
    .map_err(QualificationCadenceCurrentnessError::LaterHeadFederationRejected)?;
    if later_head_federation.receipt_sha256()
        != inputs.currentness_guard.currentness_head().head_federation_receipt_sha256()
        || later_head_federation.receipt_sha256()
            != inputs.cadence_head.head_federation_receipt_sha256()
    {
        return Err(QualificationCadenceCurrentnessError::LaterHeadFederationMismatch);
    }

    let currentness_age_upper_bound_s = evaluation_latest.saturating_sub(current_earliest);
    if currentness_age_upper_bound_s > limits.maximum_currentness_age_s {
        return Err(QualificationCadenceCurrentnessError::CurrentnessAgeExceeded);
    }

    let forbidden_time_sources = [
        current.currentness_sha256(),
        inputs.policy_bound_readiness.binding_sha256(),
        inputs.cadence.authority_sha256(),
        inputs.cadence_lineage.lineage_sha256(),
        inputs.cadence_head.assessment_sha256(),
        inputs.currentness_guard.guard().assessment_sha256(),
        inputs.currentness_guard.currentness_head().assessment_sha256(),
        later_head_federation.receipt_sha256(),
    ];
    if inputs.evaluation_time.bindings().iter().any(|binding| {
        forbidden_time_sources
            .iter()
            .any(|forbidden| binding.source_artifact_sha256() == *forbidden)
    }) {
        return Err(
            QualificationCadenceCurrentnessError::TimeDependsOnComposedCurrentnessEvidence,
        );
    }

    let capability_sha256 = cadence_conformant_currentness_digest(
        current,
        inputs.policy_bound_readiness,
        inputs.cadence,
        inputs.cadence_lineage,
        inputs.cadence_head,
        inputs.currentness_guard,
        &later_head_federation,
        inputs.evaluation_time,
        currentness_age_upper_bound_s,
    );
    Ok(CadenceConformantInstitutionalCurrentnessAtEvaluation {
        qualification_sha256: current.qualification_sha256().clone(),
        claim_id: current.claim_id().clone(),
        subject_sha256: current.subject_sha256().clone(),
        original_currentness_sha256: current.currentness_sha256().clone(),
        policy_bound_readiness_sha256: inputs.policy_bound_readiness.binding_sha256().clone(),
        cadence_policy_sha256: inputs.cadence.policy_sha256().clone(),
        cadence_policy_authority_sha256: inputs.cadence.authority_sha256().clone(),
        cadence_lineage_sha256: inputs.cadence_lineage.lineage_sha256().clone(),
        cadence_head_assessment_sha256: inputs.cadence_head.assessment_sha256().clone(),
        currentness_guard_assessment_sha256: inputs
            .currentness_guard
            .guard()
            .assessment_sha256()
            .clone(),
        later_head_federation_receipt_sha256: later_head_federation.receipt_sha256().clone(),
        evaluation_time_authority_sha256: inputs.evaluation_time.authority_sha256().clone(),
        evaluation_earliest_unix_s: evaluation_earliest,
        evaluation_latest_unix_s: evaluation_latest,
        currentness_age_upper_bound_s,
        capability_sha256,
    })
}

fn validate_policy_strength(
    head: &TransparencyHeadFederationPolicy,
    lifecycle: &LifecycleHeadFreshnessPolicy,
    evidence: &EvidenceFreshnessPolicy,
    limits: &crate::QualificationCurrentnessCadenceLimits,
    original: bool,
) -> Result<(), QualificationCadenceCurrentnessError> {
    validate_head_policy_strength(head, limits, original)?;
    if lifecycle.maximum_checkpoint_age_s > limits.maximum_lifecycle_checkpoint_age_s {
        return Err(QualificationCadenceCurrentnessError::OriginalLifecycleAgePolicyTooWeak);
    }
    if evidence.maximum_search_lag_ms > limits.maximum_evidence_search_lag_ms {
        return Err(QualificationCadenceCurrentnessError::OriginalEvidenceLagPolicyTooWeak);
    }
    Ok(())
}

fn validate_head_policy_strength(
    policy: &TransparencyHeadFederationPolicy,
    limits: &crate::QualificationCurrentnessCadenceLimits,
    original: bool,
) -> Result<(), QualificationCadenceCurrentnessError> {
    if policy.maximum_head_age_s > limits.maximum_head_age_s {
        return Err(if original {
            QualificationCadenceCurrentnessError::OriginalHeadAgePolicyTooWeak
        } else {
            QualificationCadenceCurrentnessError::LaterHeadAgePolicyTooWeak
        });
    }
    if policy.minimum_converged_views < limits.minimum_converged_views {
        return Err(if original {
            QualificationCadenceCurrentnessError::OriginalConvergedViewsTooWeak
        } else {
            QualificationCadenceCurrentnessError::LaterConvergedViewsTooWeak
        });
    }
    if policy.minimum_distinct_quorums < limits.minimum_distinct_quorums {
        return Err(if original {
            QualificationCadenceCurrentnessError::OriginalDistinctQuorumsTooWeak
        } else {
            QualificationCadenceCurrentnessError::LaterDistinctQuorumsTooWeak
        });
    }
    if policy.minimum_distinct_principals < limits.minimum_distinct_principals {
        return Err(if original {
            QualificationCadenceCurrentnessError::OriginalDistinctPrincipalsTooWeak
        } else {
            QualificationCadenceCurrentnessError::LaterDistinctPrincipalsTooWeak
        });
    }
    if policy.minimum_distinct_organizations < limits.minimum_distinct_organizations {
        return Err(if original {
            QualificationCadenceCurrentnessError::OriginalDistinctOrganizationsTooWeak
        } else {
            QualificationCadenceCurrentnessError::LaterDistinctOrganizationsTooWeak
        });
    }
    if policy.minimum_distinct_regions < limits.minimum_distinct_regions {
        return Err(if original {
            QualificationCadenceCurrentnessError::OriginalDistinctRegionsTooWeak
        } else {
            QualificationCadenceCurrentnessError::LaterDistinctRegionsTooWeak
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cadence_conformant_currentness_digest(
    current: &InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    policy_bound_readiness: &PolicyBoundQualificationValidityReadiness,
    cadence: &AuthorizedQualificationCurrentnessCadencePolicy,
    cadence_lineage: &NonWeakeningQualificationCurrentnessCadenceLineage,
    cadence_head: &QualificationCurrentnessCadenceHeadAssessment,
    currentness_guard: &QualificationCurrentnessGuardHeadBundle,
    later_head_federation: &TransparencyHeadFederationReceipt,
    evaluation_time: &TrustedTime,
    currentness_age_upper_bound_s: u64,
) -> TrustSha256Digest {
    let mut digest = symthaea_trust_core::FramedDigest::new(
        CADENCE_CONFORMANT_CURRENTNESS_DOMAIN,
    );
    digest.text(current.qualification_sha256().as_str());
    digest.text(current.claim_id().as_str());
    digest.text(current.subject_sha256().as_str());
    digest.text(current.currentness_sha256().as_str());
    digest.text(policy_bound_readiness.binding_sha256().as_str());
    digest.text(cadence.policy_sha256().as_str());
    digest.text(cadence.authority_sha256().as_str());
    digest.text(cadence_lineage.lineage_sha256().as_str());
    digest.text(cadence_head.assessment_sha256().as_str());
    digest.text(currentness_guard.guard().assessment_sha256().as_str());
    digest.text(currentness_guard.currentness_head().assessment_sha256().as_str());
    digest.text(later_head_federation.receipt_sha256().as_str());
    digest.text(evaluation_time.authority_sha256().as_str());
    let (earliest, latest) = evaluation_time.consensus_interval();
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.text(&currentness_age_upper_bound_s.to_string());
    digest.text("cadence-conformant-currentness-at-evaluation-established");
    digest.text("currentness-at-later-time-not-established");
    digest.text("globally-latest-head-not-established");
    digest.text("globally-latest-policy-not-established");
    digest.text("global-log-consistency-not-established");
    digest.text("global-evidence-exhaustiveness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_name_preserves_evaluation_scope() {
        fn _assert_api(value: &CadenceConformantInstitutionalCurrentnessAtEvaluation) {
            if value.cadence_conformant_currentness_at_evaluation_established() {
                assert!(!value.currentness_at_later_time_established());
                assert!(!value.globally_latest_head_established());
                assert!(!value.globally_latest_policy_established());
                assert!(!value.scientific_truth_established());
            }
        }
    }
}
