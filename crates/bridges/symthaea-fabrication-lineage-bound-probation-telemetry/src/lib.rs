// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finalization-grade exact telemetry binding for lineage-bound upgrade probation.
//!
//! Exact telemetry bundles remain independently qualified by the existing raw-telemetry theorem.
//! This bridge binds the exact signed observation set approved by lineage-bound probation to those
//! bundles, closes the outer signed-wrapper schema canonicality boundary, and preserves predecessor
//! provenance for downstream hardware/finalization authority.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::upgrade_probation::digest_upgrade_probation_observation;
use symthaea_fabrication_lineage_bound_upgrade_probation::{
    LineageBoundUpgradeProbationClearanceIdV1, LineageBoundUpgradeProbationClearanceV1,
};
use symthaea_fabrication_upgrade_probation_authority::SIGNED_UPGRADE_PROBATION_OBSERVATION_SCHEMA;
use symthaea_fabrication_upgrade_probation_telemetry::ProbationTelemetryBindingInputV1;

pub const LINEAGE_BOUND_TELEMETRY_PROBATION_CLEARANCE_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-telemetry-probation-clearance.v1";

const SIGNED_OBSERVATION_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-upgrade-probation-observation-evidence.v1\0";
const LINEAGE_OBSERVATION_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-probation-observation-set.v1\0";
const TELEMETRY_BINDING_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-probation-telemetry-binding-set.v1\0";
const CAPABILITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-telemetry-probation-clearance.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundTelemetryProbationClearanceIdV1(Sha256Digest);

impl LineageBoundTelemetryProbationClearanceIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundTelemetryProbationClearanceV1 {
    id: LineageBoundTelemetryProbationClearanceIdV1,
    clearance_id: LineageBoundUpgradeProbationClearanceIdV1,
    lineage_handoff_digest: Sha256Digest,
    activation_permit_digest: Sha256Digest,
    predecessor_root_digest: Sha256Digest,
    current_head_digest: Sha256Digest,
    governance_view_digest: Sha256Digest,
    registry_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    observation_set_digest: Sha256Digest,
    telemetry_binding_set_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    observation_count: usize,
    total_frame_count: usize,
    total_attempted_job_count: u64,
}

impl LineageBoundTelemetryProbationClearanceV1 {
    pub fn id(&self) -> LineageBoundTelemetryProbationClearanceIdV1 { self.id }
    pub fn clearance_id(&self) -> LineageBoundUpgradeProbationClearanceIdV1 { self.clearance_id }
    pub fn lineage_handoff_digest(&self) -> Sha256Digest { self.lineage_handoff_digest }
    pub fn activation_permit_digest(&self) -> Sha256Digest { self.activation_permit_digest }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn current_head_digest(&self) -> Sha256Digest { self.current_head_digest }
    pub fn governance_view_digest(&self) -> Sha256Digest { self.governance_view_digest }
    pub fn registry_head_digest(&self) -> Sha256Digest { self.registry_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn observation_set_digest(&self) -> Sha256Digest { self.observation_set_digest }
    pub fn telemetry_binding_set_digest(&self) -> Sha256Digest { self.telemetry_binding_set_digest }
    pub fn containment_state_digest(&self) -> Sha256Digest { self.containment_state_digest }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest { self.compromise_tracker_digest }
    pub fn observation_count(&self) -> usize { self.observation_count }
    pub fn total_frame_count(&self) -> usize { self.total_frame_count }
    pub fn total_attempted_job_count(&self) -> u64 { self.total_attempted_job_count }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundTelemetryBindingError {
    EmptyBindings,
    BindingCountMismatch { actual: usize, expected: usize },
    NonCanonicalSignedObservationSchema(String),
    ObservationDigestMismatch(String),
    ObservationSetMismatch,
    DuplicateObservation,
    DuplicateTelemetryBundle,
    TelemetryReferenceMismatch(String),
    TelemetryMachineMismatch(String),
    TelemetryOutsideObservationWindow(String),
    TelemetryContainmentMismatch(String),
    AttemptedJobCoverageMismatch { key_id: String, attempted: u64, covered: usize },
    CountOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct BindingCommitment {
    observation_digest: String,
    telemetry_bundle_id: String,
    frame_set_digest: String,
    replay_tracker_digest: String,
    verifier_set_digest: String,
    frame_count: usize,
    attempted_job_count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CapabilityCommitment {
    schema: &'static str,
    clearance_id: String,
    lineage_handoff_digest: String,
    activation_permit_digest: String,
    predecessor_root_digest: String,
    current_head_digest: String,
    governance_view_digest: String,
    registry_head_digest: String,
    predecessor_finalization_sequence: u64,
    handoff_plan_digest: String,
    observation_set_digest: String,
    telemetry_binding_set_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    observation_count: usize,
    total_frame_count: usize,
    total_attempted_job_count: u64,
}

pub fn bind_lineage_probation_to_exact_telemetry_v1(
    clearance: &LineageBoundUpgradeProbationClearanceV1,
    bindings: &[ProbationTelemetryBindingInputV1<'_>],
) -> Result<LineageBoundTelemetryProbationClearanceV1, Vec<LineageBoundTelemetryBindingError>> {
    let mut violations = Vec::new();
    if bindings.is_empty() {
        violations.push(LineageBoundTelemetryBindingError::EmptyBindings);
    }
    if bindings.len() != clearance.observation_count() {
        violations.push(LineageBoundTelemetryBindingError::BindingCountMismatch {
            actual: bindings.len(), expected: clearance.observation_count(),
        });
    }

    let mut observation_evidence_digests = Vec::with_capacity(bindings.len());
    let mut seen_observations = BTreeSet::new();
    let mut seen_bundles = BTreeSet::new();
    let mut commitments = Vec::with_capacity(bindings.len());
    let mut total_frame_count = 0usize;
    let mut total_attempted_job_count = 0u64;

    for binding in bindings {
        let signed = binding.signed_observation;
        let observation = &signed.observation;
        if signed.schema_version != SIGNED_UPGRADE_PROBATION_OBSERVATION_SCHEMA {
            violations.push(LineageBoundTelemetryBindingError::NonCanonicalSignedObservationSchema(
                signed.key_id.clone(),
            ));
        }
        let expected = match digest_upgrade_probation_observation(observation) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundTelemetryBindingError::ObservationDigestMismatch(
                    format!("{}: {error:?}", signed.key_id),
                ));
                continue;
            }
        };
        if expected != signed.observation_digest {
            violations.push(LineageBoundTelemetryBindingError::ObservationDigestMismatch(
                signed.key_id.clone(),
            ));
        }
        if !seen_observations.insert(signed.observation_digest) {
            violations.push(LineageBoundTelemetryBindingError::DuplicateObservation);
        }
        if !seen_bundles.insert(binding.telemetry_bundle.id().as_digest()) {
            violations.push(LineageBoundTelemetryBindingError::DuplicateTelemetryBundle);
        }
        if observation.telemetry_evidence_digest != binding.telemetry_bundle.evidence_digest() {
            violations.push(LineageBoundTelemetryBindingError::TelemetryReferenceMismatch(
                signed.key_id.clone(),
            ));
        }
        if observation.machine_id != binding.telemetry_bundle.machine_id() {
            violations.push(LineageBoundTelemetryBindingError::TelemetryMachineMismatch(
                signed.key_id.clone(),
            ));
        }
        if binding.telemetry_bundle.observed_started_at_unix_ms() < observation.started_at_unix_ms
            || binding.telemetry_bundle.observed_ended_at_unix_ms() > observation.ended_at_unix_ms
        {
            violations.push(LineageBoundTelemetryBindingError::TelemetryOutsideObservationWindow(
                signed.key_id.clone(),
            ));
        }
        if binding.telemetry_bundle.containment_state_digest() != clearance.containment_state_digest()
            || binding.telemetry_bundle.compromise_tracker_digest()
                != clearance.compromise_tracker_digest()
        {
            violations.push(LineageBoundTelemetryBindingError::TelemetryContainmentMismatch(
                signed.key_id.clone(),
            ));
        }
        if u64::try_from(binding.telemetry_bundle.distinct_job_count()).ok()
            != Some(observation.attempted_jobs)
        {
            violations.push(LineageBoundTelemetryBindingError::AttemptedJobCoverageMismatch {
                key_id: signed.key_id.clone(), attempted: observation.attempted_jobs,
                covered: binding.telemetry_bundle.distinct_job_count(),
            });
        }

        match hash_serializable(SIGNED_OBSERVATION_EVIDENCE_DOMAIN, signed) {
            Ok(value) => observation_evidence_digests.push(value),
            Err(error) => violations.push(error),
        }
        total_frame_count = match total_frame_count.checked_add(binding.telemetry_bundle.frame_count()) {
            Some(value) => value,
            None => { violations.push(LineageBoundTelemetryBindingError::CountOverflow); total_frame_count }
        };
        total_attempted_job_count = match total_attempted_job_count.checked_add(observation.attempted_jobs) {
            Some(value) => value,
            None => { violations.push(LineageBoundTelemetryBindingError::CountOverflow); total_attempted_job_count }
        };
        commitments.push(BindingCommitment {
            observation_digest: signed.observation_digest.to_hex(),
            telemetry_bundle_id: binding.telemetry_bundle.id().to_hex(),
            frame_set_digest: binding.telemetry_bundle.frame_set_digest().to_hex(),
            replay_tracker_digest: binding.telemetry_bundle.replay_tracker_digest().to_hex(),
            verifier_set_digest: binding.telemetry_bundle.verifier_set_digest().to_hex(),
            frame_count: binding.telemetry_bundle.frame_count(),
            attempted_job_count: observation.attempted_jobs,
        });
    }

    observation_evidence_digests.sort();
    let observation_set_digest = match hash_serializable(
        LINEAGE_OBSERVATION_SET_DOMAIN, &observation_evidence_digests,
    ) {
        Ok(value) => value,
        Err(error) => { violations.push(error); Sha256Digest([0; 32]) }
    };
    if observation_set_digest != clearance.observation_set_digest() {
        violations.push(LineageBoundTelemetryBindingError::ObservationSetMismatch);
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    commitments.sort_by(|left, right| left.observation_digest.cmp(&right.observation_digest));
    let telemetry_binding_set_digest = hash_serializable(TELEMETRY_BINDING_SET_DOMAIN, &commitments)
        .map_err(|error| vec![error])?;
    let commitment = CapabilityCommitment {
        schema: LINEAGE_BOUND_TELEMETRY_PROBATION_CLEARANCE_SCHEMA,
        clearance_id: clearance.id().to_hex(),
        lineage_handoff_digest: clearance.lineage_handoff_id().as_digest().to_hex(),
        activation_permit_digest: clearance.activation_permit_id().as_digest().to_hex(),
        predecessor_root_digest: clearance.predecessor_root_digest().to_hex(),
        current_head_digest: clearance.current_head_digest().to_hex(),
        governance_view_digest: clearance.governance_view_digest().to_hex(),
        registry_head_digest: clearance.registry_head_digest().to_hex(),
        predecessor_finalization_sequence: clearance.predecessor_finalization_sequence(),
        handoff_plan_digest: clearance.handoff_plan_digest().to_hex(),
        observation_set_digest: observation_set_digest.to_hex(),
        telemetry_binding_set_digest: telemetry_binding_set_digest.to_hex(),
        containment_state_digest: clearance.containment_state_digest().to_hex(),
        compromise_tracker_digest: clearance.compromise_tracker_digest().to_hex(),
        observation_count: bindings.len(), total_frame_count, total_attempted_job_count,
    };
    let id = LineageBoundTelemetryProbationClearanceIdV1(
        hash_serializable(CAPABILITY_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(LineageBoundTelemetryProbationClearanceV1 {
        id,
        clearance_id: clearance.id(),
        lineage_handoff_digest: clearance.lineage_handoff_id().as_digest(),
        activation_permit_digest: clearance.activation_permit_id().as_digest(),
        predecessor_root_digest: clearance.predecessor_root_digest(),
        current_head_digest: clearance.current_head_digest(),
        governance_view_digest: clearance.governance_view_digest(),
        registry_head_digest: clearance.registry_head_digest(),
        predecessor_finalization_sequence: clearance.predecessor_finalization_sequence(),
        handoff_plan_digest: clearance.handoff_plan_digest(),
        observation_set_digest,
        telemetry_binding_set_digest,
        containment_state_digest: clearance.containment_state_digest(),
        compromise_tracker_digest: clearance.compromise_tracker_digest(),
        observation_count: bindings.len(),
        total_frame_count,
        total_attempted_job_count,
    })
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8], value: &T,
) -> Result<Sha256Digest, LineageBoundTelemetryBindingError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundTelemetryBindingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
