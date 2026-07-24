// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded post-activation probation before an upgrade may be finalized.
//!
//! Activation is not proof that a successor is safe under sustained physical
//! operation. This module aggregates canonical machine observations and grants
//! a short-lived finalization capability only when the configured operational
//! evidence budget is satisfied.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::threshold::VerifiedThresholdCeremony;
use crate::upgrade_handoff::AuthorizedUpgradeHandoff;
use crate::upgrade_state::FabricationUpgradeState;
use crate::upgrade_tracker::UpgradeStage;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const UPGRADE_PROBATION_OBSERVATION_SCHEMA: &str =
    "symthaea.fabrication.upgrade-probation-observation.v1";
pub const UPGRADE_PROBATION_SCHEMA: &str = "symthaea.fabrication.upgrade-probation.v1";
pub const MAX_PROBATION_OBSERVATIONS: usize = 16_384;
pub const MAX_PROBATION_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeProbationObservation {
    pub schema_version: String,
    pub handoff_digest: Sha256Digest,
    pub successor_state_digest: Sha256Digest,
    pub machine_id: String,
    pub region_id: String,
    pub started_at_unix_ms: u64,
    pub ended_at_unix_ms: u64,
    pub attempted_jobs: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub uncertain_jobs: u64,
    pub emergency_stops: u64,
    pub containment_actions: u64,
    pub telemetry_evidence_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeProbationPolicy {
    pub minimum_observation_duration_ms: u64,
    pub minimum_distinct_machines: usize,
    pub minimum_distinct_regions: usize,
    pub minimum_successful_jobs: u64,
    pub maximum_failure_basis_points: u32,
    pub maximum_uncertain_basis_points: u32,
    pub maximum_emergency_stops: u64,
    pub maximum_containment_actions: u64,
    pub maximum_observations: usize,
    pub maximum_clearance_duration_ms: u64,
}

impl Default for UpgradeProbationPolicy {
    fn default() -> Self {
        Self {
            minimum_observation_duration_ms: 24 * 60 * 60 * 1_000,
            minimum_distinct_machines: 2,
            minimum_distinct_regions: 2,
            minimum_successful_jobs: 20,
            maximum_failure_basis_points: 100,
            maximum_uncertain_basis_points: 0,
            maximum_emergency_stops: 0,
            maximum_containment_actions: 0,
            maximum_observations: 1_024,
            maximum_clearance_duration_ms: 60 * 60 * 1_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeProbationEvidence {
    pub schema_version: String,
    pub probation_sequence: u64,
    pub handoff_digest: Sha256Digest,
    pub successor_state_digest: Sha256Digest,
    pub upgrade_state_digest: Sha256Digest,
    pub observation_digests: Vec<Sha256Digest>,
    pub machine_ids: BTreeSet<String>,
    pub region_ids: BTreeSet<String>,
    pub observation_started_at_unix_ms: u64,
    pub observation_ended_at_unix_ms: u64,
    pub attempted_jobs: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub uncertain_jobs: u64,
    pub emergency_stops: u64,
    pub containment_actions: u64,
    pub cleared_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeProbationError {
    UnsupportedSchema,
    InvalidPolicy,
    InvalidIdentifier,
    ZeroDigest(&'static str),
    InvalidObservationWindow,
    InvalidJobAccounting,
    SequenceZero,
    HandoffMismatch,
    SuccessorStateMismatch,
    UpgradeNotActivated,
    UpgradeStateInvalid(String),
    NoObservations,
    TooManyObservations { actual: usize, maximum: usize },
    DuplicateObservation,
    ObservationTooShort,
    InsufficientMachines,
    InsufficientRegions,
    InsufficientSuccessfulJobs,
    FailureBudgetExceeded,
    UncertainBudgetExceeded,
    EmergencyStopBudgetExceeded,
    ContainmentBudgetExceeded,
    InvalidClearanceWindow,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedUpgradeProbationClearance {
    evidence: UpgradeProbationEvidence,
    evidence_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedUpgradeProbationClearance {
    pub fn evidence(&self) -> &UpgradeProbationEvidence {
        &self.evidence
    }
    pub fn evidence_digest(&self) -> Sha256Digest {
        self.evidence_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }

    pub fn permits_finalization(&self, handoff_digest: Sha256Digest, unix_ms: u64) -> bool {
        self.evidence.handoff_digest == handoff_digest
            && unix_ms >= self.evidence.cleared_at_unix_ms
            && unix_ms < self.evidence.expires_at_unix_ms
    }
}

impl UpgradeProbationObservation {
    pub fn validate(&self) -> Result<(), UpgradeProbationError> {
        if self.schema_version != UPGRADE_PROBATION_OBSERVATION_SCHEMA {
            return Err(UpgradeProbationError::UnsupportedSchema);
        }
        if self.handoff_digest.0 == [0; 32] {
            return Err(UpgradeProbationError::ZeroDigest("handoff_digest"));
        }
        if self.successor_state_digest.0 == [0; 32] {
            return Err(UpgradeProbationError::ZeroDigest("successor_state_digest"));
        }
        if self.telemetry_evidence_digest.0 == [0; 32] {
            return Err(UpgradeProbationError::ZeroDigest(
                "telemetry_evidence_digest",
            ));
        }
        validate_id(&self.machine_id)?;
        validate_id(&self.region_id)?;
        if self.started_at_unix_ms >= self.ended_at_unix_ms {
            return Err(UpgradeProbationError::InvalidObservationWindow);
        }
        let accounted = self
            .successful_jobs
            .checked_add(self.failed_jobs)
            .and_then(|value| value.checked_add(self.uncertain_jobs))
            .ok_or(UpgradeProbationError::InvalidJobAccounting)?;
        if self.attempted_jobs == 0 || accounted != self.attempted_jobs {
            return Err(UpgradeProbationError::InvalidJobAccounting);
        }
        Ok(())
    }
}

pub fn digest_upgrade_probation_observation(
    observation: &UpgradeProbationObservation,
) -> Result<Sha256Digest, UpgradeProbationError> {
    observation.validate()?;
    let bytes = serde_json::to_vec(observation)
        .map_err(|error| UpgradeProbationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-probation-observation-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[allow(clippy::too_many_arguments)]
pub fn build_upgrade_probation_evidence(
    probation_sequence: u64,
    handoff: &AuthorizedUpgradeHandoff,
    upgrade_state: &FabricationUpgradeState,
    observations: &[UpgradeProbationObservation],
    cleared_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    policy: &UpgradeProbationPolicy,
) -> Result<UpgradeProbationEvidence, UpgradeProbationError> {
    validate_policy(policy)?;
    if probation_sequence == 0 {
        return Err(UpgradeProbationError::SequenceZero);
    }
    upgrade_state
        .validate_shape()
        .map_err(|error| UpgradeProbationError::UpgradeStateInvalid(format!("{error:?}")))?;
    if upgrade_state.active_stage != UpgradeStage::Activated {
        return Err(UpgradeProbationError::UpgradeNotActivated);
    }
    if upgrade_state.evidence.handoff_digest != handoff.plan_digest {
        return Err(UpgradeProbationError::HandoffMismatch);
    }
    if observations.is_empty() {
        return Err(UpgradeProbationError::NoObservations);
    }
    if observations.len() > policy.maximum_observations {
        return Err(UpgradeProbationError::TooManyObservations {
            actual: observations.len(),
            maximum: policy.maximum_observations,
        });
    }

    let successor_state_digest = handoff.plan.successor.durable_state_digest;
    let mut observation_digests = Vec::with_capacity(observations.len());
    let mut machine_ids = BTreeSet::new();
    let mut region_ids = BTreeSet::new();
    let mut started_at = u64::MAX;
    let mut ended_at = 0_u64;
    let mut attempted = 0_u64;
    let mut successful = 0_u64;
    let mut failed = 0_u64;
    let mut uncertain = 0_u64;
    let mut emergency = 0_u64;
    let mut containment = 0_u64;

    for observation in observations {
        observation.validate()?;
        if observation.handoff_digest != handoff.plan_digest {
            return Err(UpgradeProbationError::HandoffMismatch);
        }
        if observation.successor_state_digest != successor_state_digest {
            return Err(UpgradeProbationError::SuccessorStateMismatch);
        }
        started_at = started_at.min(observation.started_at_unix_ms);
        ended_at = ended_at.max(observation.ended_at_unix_ms);
        machine_ids.insert(observation.machine_id.clone());
        region_ids.insert(observation.region_id.clone());
        attempted = checked_add(attempted, observation.attempted_jobs)?;
        successful = checked_add(successful, observation.successful_jobs)?;
        failed = checked_add(failed, observation.failed_jobs)?;
        uncertain = checked_add(uncertain, observation.uncertain_jobs)?;
        emergency = checked_add(emergency, observation.emergency_stops)?;
        containment = checked_add(containment, observation.containment_actions)?;
        observation_digests.push(digest_upgrade_probation_observation(observation)?);
    }
    observation_digests.sort();
    if observation_digests
        .windows(2)
        .any(|pair| pair[0] == pair[1])
    {
        return Err(UpgradeProbationError::DuplicateObservation);
    }
    if ended_at.saturating_sub(started_at) < policy.minimum_observation_duration_ms {
        return Err(UpgradeProbationError::ObservationTooShort);
    }
    if machine_ids.len() < policy.minimum_distinct_machines {
        return Err(UpgradeProbationError::InsufficientMachines);
    }
    if region_ids.len() < policy.minimum_distinct_regions {
        return Err(UpgradeProbationError::InsufficientRegions);
    }
    if successful < policy.minimum_successful_jobs {
        return Err(UpgradeProbationError::InsufficientSuccessfulJobs);
    }
    if basis_points(failed, attempted) > u64::from(policy.maximum_failure_basis_points) {
        return Err(UpgradeProbationError::FailureBudgetExceeded);
    }
    if basis_points(uncertain, attempted) > u64::from(policy.maximum_uncertain_basis_points) {
        return Err(UpgradeProbationError::UncertainBudgetExceeded);
    }
    if emergency > policy.maximum_emergency_stops {
        return Err(UpgradeProbationError::EmergencyStopBudgetExceeded);
    }
    if containment > policy.maximum_containment_actions {
        return Err(UpgradeProbationError::ContainmentBudgetExceeded);
    }
    if cleared_at_unix_ms < ended_at
        || cleared_at_unix_ms >= expires_at_unix_ms
        || expires_at_unix_ms.saturating_sub(cleared_at_unix_ms)
            > policy.maximum_clearance_duration_ms
    {
        return Err(UpgradeProbationError::InvalidClearanceWindow);
    }

    let upgrade_state_digest = crate::upgrade_state::digest_upgrade_state(upgrade_state)
        .map_err(|error| UpgradeProbationError::UpgradeStateInvalid(format!("{error:?}")))?;
    Ok(UpgradeProbationEvidence {
        schema_version: UPGRADE_PROBATION_SCHEMA.into(),
        probation_sequence,
        handoff_digest: handoff.plan_digest,
        successor_state_digest,
        upgrade_state_digest,
        observation_digests,
        machine_ids,
        region_ids,
        observation_started_at_unix_ms: started_at,
        observation_ended_at_unix_ms: ended_at,
        attempted_jobs: attempted,
        successful_jobs: successful,
        failed_jobs: failed,
        uncertain_jobs: uncertain,
        emergency_stops: emergency,
        containment_actions: containment,
        cleared_at_unix_ms,
        expires_at_unix_ms,
    })
}

pub fn digest_upgrade_probation_evidence(
    evidence: &UpgradeProbationEvidence,
) -> Result<Sha256Digest, UpgradeProbationError> {
    validate_evidence(evidence)?;
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| UpgradeProbationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-probation-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_upgrade_probation_clearance(
    evidence: UpgradeProbationEvidence,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedUpgradeProbationClearance, UpgradeProbationError> {
    let evidence_digest = digest_upgrade_probation_evidence(&evidence)?;
    if ceremony.purpose() != "upgrade-probation-clearance" {
        return Err(UpgradeProbationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != evidence_digest {
        return Err(UpgradeProbationError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedUpgradeProbationClearance {
        evidence,
        evidence_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_policy(policy: &UpgradeProbationPolicy) -> Result<(), UpgradeProbationError> {
    if policy.minimum_observation_duration_ms == 0
        || policy.minimum_distinct_machines == 0
        || policy.minimum_distinct_regions == 0
        || policy.minimum_successful_jobs == 0
        || policy.maximum_failure_basis_points > 10_000
        || policy.maximum_uncertain_basis_points > 10_000
        || policy.maximum_observations == 0
        || policy.maximum_observations > MAX_PROBATION_OBSERVATIONS
        || policy.maximum_clearance_duration_ms == 0
    {
        return Err(UpgradeProbationError::InvalidPolicy);
    }
    Ok(())
}

fn validate_evidence(evidence: &UpgradeProbationEvidence) -> Result<(), UpgradeProbationError> {
    if evidence.schema_version != UPGRADE_PROBATION_SCHEMA {
        return Err(UpgradeProbationError::UnsupportedSchema);
    }
    if evidence.probation_sequence == 0 {
        return Err(UpgradeProbationError::SequenceZero);
    }
    for (name, digest) in [
        ("handoff_digest", evidence.handoff_digest),
        ("successor_state_digest", evidence.successor_state_digest),
        ("upgrade_state_digest", evidence.upgrade_state_digest),
    ] {
        if digest.0 == [0; 32] {
            return Err(UpgradeProbationError::ZeroDigest(name));
        }
    }
    if evidence.observation_digests.is_empty()
        || evidence
            .observation_digests
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || evidence.machine_ids.is_empty()
        || evidence.region_ids.is_empty()
        || evidence.observation_started_at_unix_ms >= evidence.observation_ended_at_unix_ms
        || evidence.cleared_at_unix_ms < evidence.observation_ended_at_unix_ms
        || evidence.cleared_at_unix_ms >= evidence.expires_at_unix_ms
    {
        return Err(UpgradeProbationError::InvalidObservationWindow);
    }
    let accounted = evidence
        .successful_jobs
        .checked_add(evidence.failed_jobs)
        .and_then(|value| value.checked_add(evidence.uncertain_jobs))
        .ok_or(UpgradeProbationError::InvalidJobAccounting)?;
    if evidence.attempted_jobs == 0 || accounted != evidence.attempted_jobs {
        return Err(UpgradeProbationError::InvalidJobAccounting);
    }
    for id in evidence
        .machine_ids
        .iter()
        .chain(evidence.region_ids.iter())
    {
        validate_id(id)?;
    }
    Ok(())
}

fn validate_id(value: &str) -> Result<(), UpgradeProbationError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_PROBATION_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(UpgradeProbationError::InvalidIdentifier);
    }
    Ok(())
}

fn checked_add(left: u64, right: u64) -> Result<u64, UpgradeProbationError> {
    left.checked_add(right)
        .ok_or(UpgradeProbationError::InvalidJobAccounting)
}

fn basis_points(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 10_000;
    }
    numerator.saturating_mul(10_000) / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn observation(start: u64, machine: &str, region: &str) -> UpgradeProbationObservation {
        UpgradeProbationObservation {
            schema_version: UPGRADE_PROBATION_OBSERVATION_SCHEMA.into(),
            handoff_digest: sha256(b"handoff"),
            successor_state_digest: sha256(b"state"),
            machine_id: machine.into(),
            region_id: region.into(),
            started_at_unix_ms: start,
            ended_at_unix_ms: start + 100,
            attempted_jobs: 10,
            successful_jobs: 10,
            failed_jobs: 0,
            uncertain_jobs: 0,
            emergency_stops: 0,
            containment_actions: 0,
            telemetry_evidence_digest: sha256(&start.to_le_bytes()),
        }
    }

    #[test]
    fn observation_digest_is_identity_sensitive() {
        let first = observation(10, "machine-a", "region-a");
        let second = observation(10, "machine-b", "region-a");
        assert_ne!(
            digest_upgrade_probation_observation(&first).unwrap(),
            digest_upgrade_probation_observation(&second).unwrap()
        );
    }

    #[test]
    fn accounting_must_close() {
        let mut value = observation(10, "machine-a", "region-a");
        value.failed_jobs = 1;
        assert_eq!(
            value.validate(),
            Err(UpgradeProbationError::InvalidJobAccounting)
        );
    }

    #[test]
    fn basis_point_budget_is_integer_deterministic() {
        assert_eq!(basis_points(1, 100), 100);
        assert_eq!(basis_points(0, 100), 0);
    }
}
