// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe live authority for fabrication evidence-retention policy.
//!
//! The kernel's serializable retention policy remains portable evidence. This bridge proves that one
//! exact policy is valid, definitely effective for every possible true time in a trusted operational
//! clock envelope, and authorized by the exact interval-qualified EvidenceRetention governance
//! quorum. It intentionally does not claim that no later policy sequence exists elsewhere.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::evidence_retention::{
    EvidenceRetentionPolicy, digest_evidence_retention_policy,
};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::{KeyUsage, TrustSnapshot, digest_trust_snapshot};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_EVIDENCE_RETENTION_PURPOSE: &str =
    "clock-governed-evidence-retention-policy-v1";
pub const PREPARED_CLOCK_GOVERNED_EVIDENCE_RETENTION_SCHEMA: &str =
    "symthaea.fabrication.prepared-clock-governed-evidence-retention-policy.v1";
pub const CLOCK_GOVERNED_EVIDENCE_RETENTION_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-evidence-retention-policy.v1";

const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const PREPARED_RETENTION_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-clock-governed-evidence-retention-policy.v1\0";
const AUTHORIZED_RETENTION_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-evidence-retention-policy.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedClockGovernedEvidenceRetentionPolicyIdV1(Sha256Digest);

impl PreparedClockGovernedEvidenceRetentionPolicyIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedEvidenceRetentionPolicyIdV1(Sha256Digest);

impl ClockGovernedEvidenceRetentionPolicyIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proposal proving one exact retention policy is definitely effective under one exact
/// trusted interval. This is not yet governance authorization.
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedClockGovernedEvidenceRetentionPolicyV1 {
    id: PreparedClockGovernedEvidenceRetentionPolicyIdV1,
    policy: EvidenceRetentionPolicy,
    policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl PreparedClockGovernedEvidenceRetentionPolicyV1 {
    pub fn id(&self) -> PreparedClockGovernedEvidenceRetentionPolicyIdV1 {
        self.id
    }
    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }
    pub fn policy(&self) -> &EvidenceRetentionPolicy {
        &self.policy
    }
    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

/// Opaque live authority for one exact, definitely-effective retention policy.
///
/// This proves authorization/effectivity for this exact sequence. It does not prove that a later
/// retention sequence has not been authorized elsewhere.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedEvidenceRetentionPolicyV1 {
    id: ClockGovernedEvidenceRetentionPolicyIdV1,
    prepared_id: PreparedClockGovernedEvidenceRetentionPolicyIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
    policy: EvidenceRetentionPolicy,
    policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl ClockGovernedEvidenceRetentionPolicyV1 {
    pub fn id(&self) -> ClockGovernedEvidenceRetentionPolicyIdV1 {
        self.id
    }
    pub fn prepared_id(&self) -> PreparedClockGovernedEvidenceRetentionPolicyIdV1 {
        self.prepared_id
    }
    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.threshold_ceremony_id
    }
    pub fn threshold_ceremony_digest(&self) -> Sha256Digest {
        self.threshold_ceremony_digest
    }
    pub fn policy(&self) -> &EvidenceRetentionPolicy {
        &self.policy
    }
    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedEvidenceRetentionError {
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    RetentionPolicyInvalid(String),
    PolicyEffectiveTimeOverflow,
    PolicyMayNotBeEffective,
    TrustSnapshotInvalid(String),
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    ContainmentStateInvalid(String),
    Clock(ClockGovernanceTimeError),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    Encoding(String),
}

/// Validate and freeze the exact policy/governance context the retention quorum must approve.
///
/// `effective_at_unix_s` must be no later than the trusted clock lower bound, so the policy is
/// already effective for every possible true time in the envelope. There is no caller `now`.
pub fn prepare_clock_governed_evidence_retention_policy_v1(
    policy: EvidenceRetentionPolicy,
    threshold_policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
) -> Result<PreparedClockGovernedEvidenceRetentionPolicyV1, Vec<ClockGovernedEvidenceRetentionError>> {
    let mut violations = Vec::new();

    if !valid_threshold_policy(threshold_policy) {
        violations.push(ClockGovernedEvidenceRetentionError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::EvidenceRetention {
        violations.push(ClockGovernedEvidenceRetentionError::ThresholdPolicyUsageMismatch);
    }
    if let Err(error) = policy.validate() {
        violations.push(ClockGovernedEvidenceRetentionError::RetentionPolicyInvalid(format!(
            "{error:?}"
        )));
    }

    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedEvidenceRetentionError::Clock(error));
            return Err(violations);
        }
    };
    match policy.effective_at_unix_s.checked_mul(1_000) {
        Some(effective_ms) if effective_ms <= clock.lower_unix_ms() => {}
        Some(_) => violations.push(ClockGovernedEvidenceRetentionError::PolicyMayNotBeEffective),
        None => violations.push(ClockGovernedEvidenceRetentionError::PolicyEffectiveTimeOverflow),
    }

    if let Err(error) = trust_snapshot.validate() {
        violations.push(ClockGovernedEvidenceRetentionError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            ClockGovernedEvidenceRetentionError::TrustSnapshotNotValidAcrossEnvelope(reason),
        );
    }
    if let Err(error) = containment_state.validate() {
        violations.push(ClockGovernedEvidenceRetentionError::ContainmentStateInvalid(format!(
            "{error:?}"
        )));
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let policy_digest = digest_evidence_retention_policy(&policy).map_err(|error| {
        vec![ClockGovernedEvidenceRetentionError::RetentionPolicyInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let threshold_policy_digest =
        digest_threshold_policy(threshold_policy).map_err(|error| vec![error])?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        vec![ClockGovernedEvidenceRetentionError::TrustSnapshotInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        vec![ClockGovernedEvidenceRetentionError::ContainmentStateInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| {
                vec![ClockGovernedEvidenceRetentionError::ContainmentStateInvalid(format!(
                    "{error:?}"
                ))]
            },
        )?;

    let id = PreparedClockGovernedEvidenceRetentionPolicyIdV1(
        digest_prepared_retention(
            policy_digest,
            policy.sequence,
            policy.effective_at_unix_s,
            threshold_policy_digest,
            trust_snapshot_digest,
            containment_state_digest,
            compromise_tracker_digest,
            containment_state.generation,
            clock.id(),
            operational_basis.id(),
        )
        .map_err(|error| vec![error])?,
    );

    Ok(PreparedClockGovernedEvidenceRetentionPolicyV1 {
        id,
        policy,
        policy_digest,
        threshold_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        containment_generation: containment_state.generation,
        clock_envelope_id: clock.id(),
        operational_basis_id: operational_basis.id(),
    })
}

pub fn authorize_clock_governed_evidence_retention_policy_v1(
    prepared: PreparedClockGovernedEvidenceRetentionPolicyV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<ClockGovernedEvidenceRetentionPolicyV1, ClockGovernedEvidenceRetentionError> {
    if ceremony.purpose() != CLOCK_GOVERNED_EVIDENCE_RETENTION_PURPOSE {
        return Err(ClockGovernedEvidenceRetentionError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(ClockGovernedEvidenceRetentionError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(ClockGovernedEvidenceRetentionError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(ClockGovernedEvidenceRetentionError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest {
        return Err(ClockGovernedEvidenceRetentionError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(ClockGovernedEvidenceRetentionError::ClockEnvelopeMismatch);
    }

    let id = ClockGovernedEvidenceRetentionPolicyIdV1(
        digest_authorized_retention(prepared.id, ceremony.id(), ceremony.ceremony_digest())?,
    );
    Ok(ClockGovernedEvidenceRetentionPolicyV1 {
        id,
        prepared_id: prepared.id,
        threshold_ceremony_id: ceremony.id(),
        threshold_ceremony_digest: ceremony.ceremony_digest(),
        policy: prepared.policy,
        policy_digest: prepared.policy_digest,
        threshold_policy_digest: prepared.threshold_policy_digest,
        trust_snapshot_digest: prepared.trust_snapshot_digest,
        containment_state_digest: prepared.containment_state_digest,
        compromise_tracker_digest: prepared.compromise_tracker_digest,
        containment_generation: prepared.containment_generation,
        clock_envelope_id: prepared.clock_envelope_id,
        operational_basis_id: prepared.operational_basis_id,
    })
}

fn valid_threshold_policy(policy: &ThresholdCeremonyPolicy) -> bool {
    policy.minimum_distinct_signers > 0
        && policy.maximum_approvals > 0
        && policy.minimum_distinct_signers <= policy.maximum_approvals
        && policy.maximum_approvals <= MAX_THRESHOLD_APPROVALS
        && policy
            .required_algorithms
            .iter()
            .all(SignatureAlgorithm::is_canonical)
        && policy.allowed_key_ids.as_ref().is_none_or(|ids| {
            ids.iter().all(|id| {
                !id.trim().is_empty()
                    && id == id.trim()
                    && id.len() <= MAX_THRESHOLD_KEY_ID_BYTES
                    && !id.chars().any(char::is_control)
            })
        })
}

#[derive(Serialize)]
struct ThresholdPolicyCommitment<'a> {
    minimum_distinct_signers: usize,
    maximum_approvals: usize,
    require_algorithm_diversity: bool,
    required_algorithms: &'a BTreeSet<SignatureAlgorithm>,
    allowed_key_ids: &'a Option<BTreeSet<String>>,
    key_usage: KeyUsage,
}

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, ClockGovernedEvidenceRetentionError> {
    if !valid_threshold_policy(policy) {
        return Err(ClockGovernedEvidenceRetentionError::InvalidThresholdPolicy);
    }
    hash_serializable(
        THRESHOLD_POLICY_DOMAIN,
        &ThresholdPolicyCommitment {
            minimum_distinct_signers: policy.minimum_distinct_signers,
            maximum_approvals: policy.maximum_approvals,
            require_algorithm_diversity: policy.require_algorithm_diversity,
            required_algorithms: &policy.required_algorithms,
            allowed_key_ids: &policy.allowed_key_ids,
            key_usage: policy.key_usage,
        },
    )
}

#[derive(Serialize)]
struct PreparedRetentionCommitment {
    schema: &'static str,
    policy_digest: String,
    policy_sequence: u64,
    effective_at_unix_s: u64,
    threshold_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    containment_generation: u64,
    clock_envelope_id: String,
    operational_basis_id: String,
}

#[allow(clippy::too_many_arguments)]
fn digest_prepared_retention(
    policy_digest: Sha256Digest,
    policy_sequence: u64,
    effective_at_unix_s: u64,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
) -> Result<Sha256Digest, ClockGovernedEvidenceRetentionError> {
    hash_serializable(
        PREPARED_RETENTION_DOMAIN,
        &PreparedRetentionCommitment {
            schema: PREPARED_CLOCK_GOVERNED_EVIDENCE_RETENTION_SCHEMA,
            policy_digest: policy_digest.to_hex(),
            policy_sequence,
            effective_at_unix_s,
            threshold_policy_digest: threshold_policy_digest.to_hex(),
            trust_snapshot_digest: trust_snapshot_digest.to_hex(),
            containment_state_digest: containment_state_digest.to_hex(),
            compromise_tracker_digest: compromise_tracker_digest.to_hex(),
            containment_generation,
            clock_envelope_id: clock_envelope_id.to_hex(),
            operational_basis_id: operational_basis_id.to_hex(),
        },
    )
}

#[derive(Serialize)]
struct AuthorizedRetentionCommitment {
    schema: &'static str,
    prepared_id: String,
    threshold_ceremony_id: String,
    threshold_ceremony_digest: String,
}

fn digest_authorized_retention(
    prepared_id: PreparedClockGovernedEvidenceRetentionPolicyIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
) -> Result<Sha256Digest, ClockGovernedEvidenceRetentionError> {
    hash_serializable(
        AUTHORIZED_RETENTION_DOMAIN,
        &AuthorizedRetentionCommitment {
            schema: CLOCK_GOVERNED_EVIDENCE_RETENTION_SCHEMA,
            prepared_id: prepared_id.to_hex(),
            threshold_ceremony_id: threshold_ceremony_id.to_hex(),
            threshold_ceremony_digest: threshold_ceremony_digest.to_hex(),
        },
    )
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ClockGovernedEvidenceRetentionError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ClockGovernedEvidenceRetentionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
