// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound requalification after an emergency release rollback.
//!
//! A rollback restores a previously authorized release, but it does not prove
//! that the restored release is safe under the current hardware, trust, or
//! incident context. Requalification therefore requires resolved trigger
//! incidents, intact lineage, fresh assurance, and a clean supervised hardware
//! observation window before broader authority can be restored.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::incident_ledger::{IncidentLedger, digest_incident_ledger};
use crate::release_assurance::AssuredReleasePromotion;
use crate::release_lineage::{ReleaseLineage, digest_release_lineage};
use crate::release_rollback::AuthorizedReleaseRollback;
use crate::rollout::{RolloutObservation, digest_rollout_observation};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const POST_ROLLBACK_REQUALIFICATION_SCHEMA: &str =
    "symthaea.fabrication.post-rollback-requalification.v1";
pub const MAX_REQUALIFICATION_OBSERVATIONS: usize = 1024;
pub const MAX_REQUALIFIED_MACHINES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostRollbackRequalificationEvidence {
    pub schema_version: String,
    pub requalification_sequence: u64,
    pub rollback_digest: Sha256Digest,
    pub target_promotion_digest: Sha256Digest,
    pub target_assurance_digest: Sha256Digest,
    pub release_lineage_digest: Sha256Digest,
    pub incident_ledger_digest: Sha256Digest,
    pub resolved_triggering_incident_digests: Vec<Sha256Digest>,
    pub observation_digests: Vec<Sha256Digest>,
    pub authorized_machine_ids: BTreeSet<String>,
    pub observation_started_at_unix_s: u64,
    pub observation_ended_at_unix_s: u64,
    pub attempted_jobs: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub uncertain_jobs: u64,
    pub emergency_stops: u64,
    pub authorized_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostRollbackRequalificationPolicy {
    pub minimum_observation_duration_s: u64,
    pub minimum_successful_jobs: u64,
    pub maximum_authorization_duration_s: u64,
    pub require_zero_adverse_outcomes: bool,
}

impl Default for PostRollbackRequalificationPolicy {
    fn default() -> Self {
        Self {
            minimum_observation_duration_s: 24 * 3_600,
            minimum_successful_jobs: 10,
            maximum_authorization_duration_s: 24 * 3_600,
            require_zero_adverse_outcomes: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostRollbackRequalificationError {
    UnsupportedSchema,
    InvalidPolicy,
    SequenceZero,
    TargetAssuranceMismatch,
    LineageTargetMismatch,
    LineageInvalid,
    IncidentLedgerInvalid,
    TriggerIncidentUnresolved(Sha256Digest),
    TooManyObservations,
    NoObservations,
    ObservationInvalid,
    ObservationPromotionMismatch,
    ObservationWindowMismatch,
    NonCanonicalObservationOrder,
    ObservationOverlap,
    DuplicateObservation,
    ObservationTooShort,
    InsufficientSuccessfulJobs,
    AdverseOutcomeObserved,
    CountOverflow,
    InvalidMachineScope,
    InvalidWindow,
    AssuranceExpired,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedPostRollbackRequalification {
    evidence: PostRollbackRequalificationEvidence,
    requalification_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedPostRollbackRequalification {
    pub fn evidence(&self) -> &PostRollbackRequalificationEvidence {
        &self.evidence
    }
    pub fn requalification_digest(&self) -> Sha256Digest {
        self.requalification_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }

    pub fn permits_machine(&self, machine_id: &str, unix_s: u64) -> bool {
        unix_s >= self.evidence.authorized_at_unix_s
            && unix_s < self.evidence.expires_at_unix_s
            && self.evidence.authorized_machine_ids.contains(machine_id)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_post_rollback_requalification_evidence(
    requalification_sequence: u64,
    rollback: &AuthorizedReleaseRollback,
    target_assurance: &AssuredReleasePromotion,
    release_lineage: &ReleaseLineage,
    incident_ledger: &IncidentLedger,
    observations: &[RolloutObservation],
    authorized_machine_ids: BTreeSet<String>,
    authorized_at_unix_s: u64,
    expires_at_unix_s: u64,
    policy: &PostRollbackRequalificationPolicy,
) -> Result<PostRollbackRequalificationEvidence, PostRollbackRequalificationError> {
    validate_policy(policy)?;
    if requalification_sequence == 0 {
        return Err(PostRollbackRequalificationError::SequenceZero);
    }
    if target_assurance.evidence().promotion_digest != rollback.evidence().target_promotion_digest {
        return Err(PostRollbackRequalificationError::TargetAssuranceMismatch);
    }
    release_lineage
        .validate()
        .map_err(|_| PostRollbackRequalificationError::LineageInvalid)?;
    if release_lineage.active_promotion_digest()
        != Some(rollback.evidence().target_promotion_digest)
    {
        return Err(PostRollbackRequalificationError::LineageTargetMismatch);
    }
    let ledger_report = incident_ledger.verify();
    if !ledger_report.intact() {
        return Err(PostRollbackRequalificationError::IncidentLedgerInvalid);
    }
    let resolved: BTreeSet<_> = ledger_report.resolved_incidents.iter().copied().collect();
    for incident in &rollback.evidence().triggering_incident_digests {
        if !resolved.contains(incident) {
            return Err(PostRollbackRequalificationError::TriggerIncidentUnresolved(
                *incident,
            ));
        }
    }
    if observations.is_empty() {
        return Err(PostRollbackRequalificationError::NoObservations);
    }
    if observations.len() > MAX_REQUALIFICATION_OBSERVATIONS {
        return Err(PostRollbackRequalificationError::TooManyObservations);
    }
    validate_machine_scope(&authorized_machine_ids)?;
    let mut observation_digests = Vec::with_capacity(observations.len());
    let mut started_at = u64::MAX;
    let mut ended_at = 0;
    let mut attempted = 0_u64;
    let mut successful = 0_u64;
    let mut failed = 0_u64;
    let mut uncertain = 0_u64;
    let mut emergency = 0_u64;
    let mut previous_started_at = None;
    let mut previous_ended_at = None;
    for observation in observations {
        observation
            .validate()
            .map_err(|_| PostRollbackRequalificationError::ObservationInvalid)?;
        if observation.promotion_digest != rollback.evidence().target_promotion_digest {
            return Err(PostRollbackRequalificationError::ObservationPromotionMismatch);
        }
        if previous_started_at.is_some_and(|previous| observation.started_at_unix_s < previous) {
            return Err(PostRollbackRequalificationError::NonCanonicalObservationOrder);
        }
        if previous_ended_at.is_some_and(|previous| observation.started_at_unix_s < previous) {
            return Err(PostRollbackRequalificationError::ObservationOverlap);
        }
        previous_started_at = Some(observation.started_at_unix_s);
        previous_ended_at = Some(observation.ended_at_unix_s);
        started_at = started_at.min(observation.started_at_unix_s);
        ended_at = ended_at.max(observation.ended_at_unix_s);
        attempted = attempted
            .checked_add(u64::from(observation.attempted_jobs))
            .ok_or(PostRollbackRequalificationError::CountOverflow)?;
        successful = successful
            .checked_add(u64::from(observation.successful_jobs))
            .ok_or(PostRollbackRequalificationError::CountOverflow)?;
        failed = failed
            .checked_add(u64::from(observation.failed_jobs))
            .ok_or(PostRollbackRequalificationError::CountOverflow)?;
        uncertain = uncertain
            .checked_add(u64::from(observation.uncertain_jobs))
            .ok_or(PostRollbackRequalificationError::CountOverflow)?;
        emergency = emergency
            .checked_add(u64::from(observation.emergency_stops))
            .ok_or(PostRollbackRequalificationError::CountOverflow)?;
        observation_digests.push(
            digest_rollout_observation(observation)
                .map_err(|_| PostRollbackRequalificationError::ObservationInvalid)?,
        );
    }
    observation_digests.sort();
    if observation_digests
        .windows(2)
        .any(|pair| pair[0] == pair[1])
    {
        return Err(PostRollbackRequalificationError::DuplicateObservation);
    }
    if started_at >= ended_at || authorized_at_unix_s < ended_at {
        return Err(PostRollbackRequalificationError::ObservationWindowMismatch);
    }
    if ended_at.saturating_sub(started_at) < policy.minimum_observation_duration_s {
        return Err(PostRollbackRequalificationError::ObservationTooShort);
    }
    if successful < policy.minimum_successful_jobs {
        return Err(PostRollbackRequalificationError::InsufficientSuccessfulJobs);
    }
    if policy.require_zero_adverse_outcomes && (failed != 0 || uncertain != 0 || emergency != 0) {
        return Err(PostRollbackRequalificationError::AdverseOutcomeObserved);
    }
    if attempted != successful.saturating_add(failed).saturating_add(uncertain) {
        return Err(PostRollbackRequalificationError::ObservationInvalid);
    }
    if authorized_at_unix_s >= expires_at_unix_s
        || expires_at_unix_s.saturating_sub(authorized_at_unix_s)
            > policy.maximum_authorization_duration_s
    {
        return Err(PostRollbackRequalificationError::InvalidWindow);
    }
    if authorized_at_unix_s >= target_assurance.evidence().expires_at_unix_s {
        return Err(PostRollbackRequalificationError::AssuranceExpired);
    }
    Ok(PostRollbackRequalificationEvidence {
        schema_version: POST_ROLLBACK_REQUALIFICATION_SCHEMA.into(),
        requalification_sequence,
        rollback_digest: rollback.rollback_digest(),
        target_promotion_digest: rollback.evidence().target_promotion_digest,
        target_assurance_digest: target_assurance.assurance_digest(),
        release_lineage_digest: digest_release_lineage(release_lineage)
            .map_err(|_| PostRollbackRequalificationError::LineageInvalid)?,
        incident_ledger_digest: digest_incident_ledger(incident_ledger)
            .map_err(|_| PostRollbackRequalificationError::IncidentLedgerInvalid)?,
        resolved_triggering_incident_digests: rollback
            .evidence()
            .triggering_incident_digests
            .clone(),
        observation_digests,
        authorized_machine_ids,
        observation_started_at_unix_s: started_at,
        observation_ended_at_unix_s: ended_at,
        attempted_jobs: attempted,
        successful_jobs: successful,
        failed_jobs: failed,
        uncertain_jobs: uncertain,
        emergency_stops: emergency,
        authorized_at_unix_s,
        expires_at_unix_s,
    })
}

pub fn digest_post_rollback_requalification(
    evidence: &PostRollbackRequalificationEvidence,
) -> Result<Sha256Digest, PostRollbackRequalificationError> {
    validate_evidence(evidence)?;
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| PostRollbackRequalificationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.post-rollback-requalification-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_post_rollback_requalification(
    evidence: PostRollbackRequalificationEvidence,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedPostRollbackRequalification, PostRollbackRequalificationError> {
    let requalification_digest = digest_post_rollback_requalification(&evidence)?;
    if ceremony.purpose() != "post-rollback-requalification" {
        return Err(PostRollbackRequalificationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != requalification_digest {
        return Err(PostRollbackRequalificationError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedPostRollbackRequalification {
        evidence,
        requalification_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_policy(
    policy: &PostRollbackRequalificationPolicy,
) -> Result<(), PostRollbackRequalificationError> {
    if policy.minimum_observation_duration_s == 0
        || policy.minimum_successful_jobs == 0
        || policy.maximum_authorization_duration_s == 0
    {
        return Err(PostRollbackRequalificationError::InvalidPolicy);
    }
    Ok(())
}

fn validate_machine_scope(
    machine_ids: &BTreeSet<String>,
) -> Result<(), PostRollbackRequalificationError> {
    if machine_ids.is_empty() || machine_ids.len() > MAX_REQUALIFIED_MACHINES {
        return Err(PostRollbackRequalificationError::InvalidMachineScope);
    }
    if machine_ids.iter().any(|machine_id| {
        machine_id.trim().is_empty()
            || machine_id != machine_id.trim()
            || machine_id.len() > 256
            || machine_id.chars().any(char::is_control)
    }) {
        return Err(PostRollbackRequalificationError::InvalidMachineScope);
    }
    Ok(())
}

fn validate_evidence(
    evidence: &PostRollbackRequalificationEvidence,
) -> Result<(), PostRollbackRequalificationError> {
    if evidence.schema_version != POST_ROLLBACK_REQUALIFICATION_SCHEMA {
        return Err(PostRollbackRequalificationError::UnsupportedSchema);
    }
    if evidence.requalification_sequence == 0 {
        return Err(PostRollbackRequalificationError::SequenceZero);
    }
    validate_machine_scope(&evidence.authorized_machine_ids)?;
    if evidence.observation_digests.is_empty()
        || evidence.observation_digests.len() > MAX_REQUALIFICATION_OBSERVATIONS
    {
        return Err(PostRollbackRequalificationError::NoObservations);
    }
    if evidence.observation_started_at_unix_s >= evidence.observation_ended_at_unix_s
        || evidence.authorized_at_unix_s < evidence.observation_ended_at_unix_s
    {
        return Err(PostRollbackRequalificationError::ObservationWindowMismatch);
    }
    if evidence.authorized_at_unix_s >= evidence.expires_at_unix_s {
        return Err(PostRollbackRequalificationError::InvalidWindow);
    }
    if evidence.attempted_jobs
        != evidence
            .successful_jobs
            .saturating_add(evidence.failed_jobs)
            .saturating_add(evidence.uncertain_jobs)
    {
        return Err(PostRollbackRequalificationError::ObservationInvalid);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_rejects_administrative_empty_machine_scope() {
        let evidence = PostRollbackRequalificationEvidence {
            schema_version: POST_ROLLBACK_REQUALIFICATION_SCHEMA.into(),
            requalification_sequence: 1,
            rollback_digest: Sha256Digest([1; 32]),
            target_promotion_digest: Sha256Digest([2; 32]),
            target_assurance_digest: Sha256Digest([3; 32]),
            release_lineage_digest: Sha256Digest([4; 32]),
            incident_ledger_digest: Sha256Digest([5; 32]),
            resolved_triggering_incident_digests: vec![Sha256Digest([6; 32])],
            observation_digests: vec![Sha256Digest([7; 32])],
            authorized_machine_ids: BTreeSet::new(),
            observation_started_at_unix_s: 10,
            observation_ended_at_unix_s: 20,
            attempted_jobs: 1,
            successful_jobs: 1,
            failed_jobs: 0,
            uncertain_jobs: 0,
            emergency_stops: 0,
            authorized_at_unix_s: 21,
            expires_at_unix_s: 30,
        };
        assert_eq!(
            digest_post_rollback_requalification(&evidence),
            Err(PostRollbackRequalificationError::InvalidMachineScope)
        );
    }
}
