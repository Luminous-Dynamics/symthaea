// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transparency-anchored observation of clock-governed policy lineage heads.
//!
//! This crate deliberately proves a bounded claim: one exact policy-lineage head is the latest
//! publication for its policy domain inside one exact append-only transparency-log view, and that
//! log view is covered by an interval-safe signed checkpoint plus an independently verified witness
//! quorum. It does not claim that no later checkpoint exists elsewhere.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::transparency::{
    TransparencyLog, digest_transparency_log,
};
use symthaea_fabrication_kernel::transparency_checkpoint::{
    SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA, SignedTransparencyCheckpoint,
    VerifiedTransparencyCheckpoint, digest_transparency_checkpoint,
};
use symthaea_fabrication_kernel::transparency_witness::{
    MAX_TRANSPARENCY_WITNESSES, SIGNED_TRANSPARENCY_WITNESS_SCHEMA,
    TRANSPARENCY_WITNESS_SCHEMA, SignedTransparencyWitness, TransparencyWitnessPolicy,
    VerifiedTransparencyWitnessQuorum, digest_transparency_witness_statement,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_policy_lineage::{
    ClockGovernedPolicyLineageIdV1, ClockGovernedPolicyLineageV1,
};
use symthaea_fabrication_policy_temporal_validity::{
    ClockGovernedPolicyTemporalValidityPermitIdV1,
    ClockGovernedPolicyTemporalValidityPermitV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const POLICY_LINEAGE_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.policy-lineage-head-publication.v1";
pub const QUORUM_OBSERVED_POLICY_HEAD_SCHEMA: &str =
    "symthaea.fabrication.quorum-observed-policy-head.v1";
pub const POLICY_LINEAGE_HEAD_LOG_KIND_PREFIX: &str = "policy-lineage-head-v1:";

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-publication.v1\0";
const LOG_KIND_DOMAIN: &[u8] = b"symthaea.fabrication.policy-lineage-head-kind.v1\0";
const SIGNED_CHECKPOINT_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-checkpoint-evidence.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-witness-set.v1\0";
const OBSERVED_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.quorum-observed-policy-head.v1\0";
const LEGACY_WITNESS_QUORUM_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-quorum.v1\0";

/// Portable publication evidence. This is intentionally serializable and is not live authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyLineageHeadPublicationV1 {
    pub schema_version: String,
    pub policy_domain: String,
    pub lineage_id: String,
    pub lineage_sequence: u64,
    pub current_policy_binding_digest: Sha256Digest,
    pub temporal_validity_permit_id: String,
    pub current_operational_basis_id: String,
    pub current_clock_envelope_id: String,
    pub active_waiver_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuorumObservedPolicyHeadIdV1(Sha256Digest);

impl QuorumObservedPolicyHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one exact lineage head is the latest publication for its policy domain in one
/// exact witnessed transparency checkpoint view and remains temporally valid under the same clock.
#[derive(Debug, Clone)]
#[must_use]
pub struct QuorumObservedPolicyHeadV1 {
    id: QuorumObservedPolicyHeadIdV1,
    lineage_id: ClockGovernedPolicyLineageIdV1,
    lineage_sequence: u64,
    temporal_validity_permit_id: ClockGovernedPolicyTemporalValidityPermitIdV1,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    transparency_log_size: u64,
    transparency_root_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    signed_checkpoint_evidence_digest: Sha256Digest,
    witness_quorum_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
}

impl QuorumObservedPolicyHeadV1 {
    pub fn id(&self) -> QuorumObservedPolicyHeadIdV1 {
        self.id
    }
    pub fn lineage_id(&self) -> ClockGovernedPolicyLineageIdV1 {
        self.lineage_id
    }
    pub fn lineage_sequence(&self) -> u64 {
        self.lineage_sequence
    }
    pub fn temporal_validity_permit_id(&self) -> ClockGovernedPolicyTemporalValidityPermitIdV1 {
        self.temporal_validity_permit_id
    }
    pub fn publication_digest(&self) -> Sha256Digest {
        self.publication_digest
    }
    pub fn publication_entry_sequence(&self) -> u64 {
        self.publication_entry_sequence
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn transparency_log_size(&self) -> u64 {
        self.transparency_log_size
    }
    pub fn transparency_root_digest(&self) -> Sha256Digest {
        self.transparency_root_digest
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn signed_checkpoint_evidence_digest(&self) -> Sha256Digest {
        self.signed_checkpoint_evidence_digest
    }
    pub fn witness_quorum_digest(&self) -> Sha256Digest {
        self.witness_quorum_digest
    }
    pub fn witness_set_evidence_digest(&self) -> Sha256Digest {
        self.witness_set_evidence_digest
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
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyHeadObservationError {
    PublicationMismatch,
    InvalidPublication,
    TemporalPermitMismatch,
    CurrentClockMismatch,
    Clock(ClockGovernanceTimeError),
    TransparencyLogInvalid(String),
    PublicationNotFound,
    PublicationNotLatestInCheckpointView,
    PublicationMayBeFuture,
    CheckpointInvalid(String),
    CheckpointVerifiedEvidenceMismatch,
    CheckpointLogMismatch,
    TrustSnapshotInvalid(String),
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    SignerUnknown(String),
    SignerNotActive(String),
    SignerUsageNotAllowed(String),
    SignerNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    SignerCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    ContainmentStateInvalid(String),
    WitnessPolicyInvalid,
    TooManyWitnesses,
    WitnessMalformed(String),
    WitnessCheckpointMismatch(String),
    WitnessDigestMismatch(String),
    WitnessMayBeFuture(String),
    WitnessMayBeStale(String),
    DuplicateWitnessSigner(String),
    InsufficientWitnesses { actual: usize, required: usize },
    InsufficientOrganizations { actual: usize, required: usize },
    InsufficientRegions { actual: usize, required: usize },
    MissingAlgorithmDiversity,
    WitnessVerifiedEvidenceMismatch,
    TimeScaleOverflow,
    Encoding(String),
}

/// Build the exact portable value that should be appended to the transparency log by digest.
pub fn build_policy_lineage_head_publication_v1(
    lineage: &ClockGovernedPolicyLineageV1,
    temporal: &ClockGovernedPolicyTemporalValidityPermitV1,
) -> Result<PolicyLineageHeadPublicationV1, PolicyHeadObservationError> {
    require_temporal_matches_lineage(lineage, temporal)?;
    Ok(PolicyLineageHeadPublicationV1 {
        schema_version: POLICY_LINEAGE_HEAD_PUBLICATION_SCHEMA.into(),
        policy_domain: lineage.domain().to_string(),
        lineage_id: lineage.id().to_hex(),
        lineage_sequence: lineage.sequence(),
        current_policy_binding_digest: lineage.current_policy_binding_digest(),
        temporal_validity_permit_id: temporal.id().to_hex(),
        current_operational_basis_id: temporal.current_operational_basis_id().to_hex(),
        current_clock_envelope_id: temporal.current_clock_envelope_id().to_hex(),
        active_waiver_count: lineage.active_waivers().len(),
    })
}

pub fn digest_policy_lineage_head_publication_v1(
    publication: &PolicyLineageHeadPublicationV1,
) -> Result<Sha256Digest, PolicyHeadObservationError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

/// Domain-specific transparency-log kind. The full domain is hashed so multiple policy domains can
/// share one log while each still has an independently detectable latest publication.
pub fn policy_lineage_head_log_kind(policy_domain: &str) -> Result<String, PolicyHeadObservationError> {
    validate_identifier(policy_domain)?;
    let mut hasher = Sha256::new();
    hasher.update(LOG_KIND_DOMAIN);
    hasher.update(policy_domain.as_bytes());
    Ok(format!("{POLICY_LINEAGE_HEAD_LOG_KIND_PREFIX}{}", hasher.finalize().to_hex()))
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_quorum_observed_policy_head_v1(
    lineage: &ClockGovernedPolicyLineageV1,
    temporal: &ClockGovernedPolicyTemporalValidityPermitV1,
    current_basis: &OperationalClockBasisV1,
    publication: &PolicyLineageHeadPublicationV1,
    log: &TransparencyLog,
    signed_checkpoint: &SignedTransparencyCheckpoint,
    verified_checkpoint: &VerifiedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    verified_witness_quorum: &VerifiedTransparencyWitnessQuorum,
    witness_policy: &TransparencyWitnessPolicy,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
) -> Result<QuorumObservedPolicyHeadV1, Vec<PolicyHeadObservationError>> {
    let mut violations = Vec::new();

    if let Err(error) = require_temporal_matches_lineage(lineage, temporal) {
        violations.push(error);
    }
    let clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(PolicyHeadObservationError::Clock(error));
            return Err(violations);
        }
    };
    if temporal.current_operational_basis_id() != current_basis.id()
        || temporal.current_clock_envelope_id() != clock.id()
    {
        violations.push(PolicyHeadObservationError::CurrentClockMismatch);
    }

    let expected_publication = match build_policy_lineage_head_publication_v1(lineage, temporal) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if publication != &expected_publication {
        violations.push(PolicyHeadObservationError::PublicationMismatch);
    }
    let publication_digest = match digest_policy_lineage_head_publication_v1(publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };

    if let Err(error) = log.validate() {
        violations.push(PolicyHeadObservationError::TransparencyLogInvalid(format!("{error:?}")));
    }
    let log_kind = match policy_lineage_head_log_kind(lineage.domain()) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    let matching = log
        .entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.kind == log_kind)
        .collect::<Vec<_>>();
    let Some((publication_index, publication_entry)) = matching.last().copied() else {
        violations.push(PolicyHeadObservationError::PublicationNotFound);
        return Err(violations);
    };
    if publication_entry.subject_digest != publication_digest {
        violations.push(PolicyHeadObservationError::PublicationNotLatestInCheckpointView);
    }
    if matching
        .iter()
        .any(|(index, _)| *index > publication_index)
    {
        violations.push(PolicyHeadObservationError::PublicationNotLatestInCheckpointView);
    }
    match seconds_to_millis(publication_entry.recorded_at_unix_s) {
        Ok(recorded_at_ms) if recorded_at_ms <= clock.lower_unix_ms() => {}
        Ok(_) => violations.push(PolicyHeadObservationError::PublicationMayBeFuture),
        Err(error) => violations.push(error),
    }

    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(PolicyHeadObservationError::TrustSnapshotInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(PolicyHeadObservationError::TrustSnapshotInvalid(format!("{error:?}")));
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(PolicyHeadObservationError::TrustSnapshotNotValidAcrossEnvelope(reason));
    }

    if let Err(error) = containment_state.validate() {
        violations.push(PolicyHeadObservationError::ContainmentStateInvalid(format!("{error:?}")));
    }
    let containment_state_digest = match digest_containment_state(containment_state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(PolicyHeadObservationError::ContainmentStateInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    let compromise_tracker_digest = match digest_signer_compromise_tracker(
        &containment_state.signer_compromise_tracker,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(PolicyHeadObservationError::ContainmentStateInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };

    if signed_checkpoint.schema_version != SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA {
        violations.push(PolicyHeadObservationError::CheckpointInvalid("unsupported signed checkpoint schema".into()));
    }
    if let Err(error) = signed_checkpoint.checkpoint.validate() {
        violations.push(PolicyHeadObservationError::CheckpointInvalid(format!("{error:?}")));
    }
    match digest_transparency_checkpoint(&signed_checkpoint.checkpoint) {
        Ok(value) if value == signed_checkpoint.checkpoint_digest => {}
        Ok(_) => violations.push(PolicyHeadObservationError::CheckpointInvalid("checkpoint digest mismatch".into())),
        Err(error) => violations.push(PolicyHeadObservationError::CheckpointInvalid(format!("{error:?}"))),
    }
    if verified_checkpoint.checkpoint() != &signed_checkpoint.checkpoint
        || verified_checkpoint.checkpoint_digest() != signed_checkpoint.checkpoint_digest
        || verified_checkpoint.signer()
            != &(signed_checkpoint.signature.algorithm.clone(), signed_checkpoint.signature.key_id.clone())
    {
        violations.push(PolicyHeadObservationError::CheckpointVerifiedEvidenceMismatch);
    }
    let log_root = match log.root() {
        Ok(value) => value,
        Err(error) => {
            violations.push(PolicyHeadObservationError::TransparencyLogInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    if signed_checkpoint.checkpoint.log_size != log.entries.len() as u64
        || signed_checkpoint.checkpoint.root_digest != log_root
    {
        violations.push(PolicyHeadObservationError::CheckpointLogMismatch);
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        signed_checkpoint.checkpoint.issued_at_unix_s,
        signed_checkpoint.checkpoint.expires_at_unix_s,
    ) {
        violations.push(PolicyHeadObservationError::CheckpointInvalid(format!("{reason:?}")));
    }
    requalify_signer(
        &signed_checkpoint.signature.algorithm,
        &signed_checkpoint.signature.key_id,
        KeyUsage::TransparencyLog,
        trust_snapshot,
        containment_state,
        &clock,
        &mut violations,
    );

    validate_witness_policy(witness_policy, &mut violations);
    if signed_witnesses.len() > witness_policy.maximum_witnesses {
        violations.push(PolicyHeadObservationError::TooManyWitnesses);
    }
    let mut signer_ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut regions = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut witness_identities = Vec::new();
    let mut statement_digests = Vec::new();
    let mut witness_evidence_digests = Vec::new();

    for signed in signed_witnesses.iter().take(witness_policy.maximum_witnesses) {
        let key_id = signed.signature.key_id.clone();
        if signed.schema_version != SIGNED_TRANSPARENCY_WITNESS_SCHEMA
            || signed.statement.schema_version != TRANSPARENCY_WITNESS_SCHEMA
            || validate_identifier(&signed.statement.witness_organization).is_err()
            || validate_identifier(&signed.statement.witness_region).is_err()
            || signed.statement.checkpoint_log_size == 0
            || !signed.signature.algorithm.is_canonical()
            || validate_identifier(&signed.signature.key_id).is_err()
            || signed.signature.signature.is_empty()
            || signed.signature.signature.len() > 64 * 1024
        {
            violations.push(PolicyHeadObservationError::WitnessMalformed(key_id));
            continue;
        }
        if signed.statement.checkpoint_digest != signed_checkpoint.checkpoint_digest
            || signed.statement.checkpoint_log_size != signed_checkpoint.checkpoint.log_size
            || signed.statement.checkpoint_root_digest != signed_checkpoint.checkpoint.root_digest
        {
            violations.push(PolicyHeadObservationError::WitnessCheckpointMismatch(key_id));
            continue;
        }
        match digest_transparency_witness_statement(&signed.statement) {
            Ok(value) if value == signed.statement_digest => {}
            Ok(_) => {
                violations.push(PolicyHeadObservationError::WitnessDigestMismatch(key_id));
                continue;
            }
            Err(error) => {
                violations.push(PolicyHeadObservationError::WitnessMalformed(format!("{key_id}: {error:?}")));
                continue;
            }
        }
        let observed_at_ms = match seconds_to_millis(signed.statement.observed_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if observed_at_ms > clock.lower_unix_ms() {
            violations.push(PolicyHeadObservationError::WitnessMayBeFuture(key_id.clone()));
        }
        let max_age_ms = match seconds_to_millis(witness_policy.maximum_observation_age_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        let freshness_deadline = match observed_at_ms.checked_add(max_age_ms) {
            Some(value) => value,
            None => {
                violations.push(PolicyHeadObservationError::TimeScaleOverflow);
                continue;
            }
        };
        if clock.upper_unix_ms() > freshness_deadline {
            violations.push(PolicyHeadObservationError::WitnessMayBeStale(key_id.clone()));
        }
        let signer = (signed.signature.algorithm.clone(), key_id.clone());
        if !signer_ids.insert(signer.clone()) {
            violations.push(PolicyHeadObservationError::DuplicateWitnessSigner(key_id));
            continue;
        }
        requalify_signer(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            KeyUsage::TransparencyWitness,
            trust_snapshot,
            containment_state,
            &clock,
            &mut violations,
        );
        algorithms.insert(signed.signature.algorithm.clone());
        organizations.insert(signed.statement.witness_organization.clone());
        regions.insert(signed.statement.witness_region.clone());
        witness_identities.push((
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
            signed.statement.witness_organization.clone(),
            signed.statement.witness_region.clone(),
        ));
        statement_digests.push(signed.statement_digest);
        match hash_serializable(SIGNED_WITNESS_EVIDENCE_DOMAIN, signed) {
            Ok(value) => witness_evidence_digests.push(value),
            Err(error) => violations.push(error),
        }
    }

    if witness_identities.len() < witness_policy.minimum_distinct_witnesses {
        violations.push(PolicyHeadObservationError::InsufficientWitnesses {
            actual: witness_identities.len(),
            required: witness_policy.minimum_distinct_witnesses,
        });
    }
    if organizations.len() < witness_policy.minimum_distinct_organizations {
        violations.push(PolicyHeadObservationError::InsufficientOrganizations {
            actual: organizations.len(),
            required: witness_policy.minimum_distinct_organizations,
        });
    }
    if regions.len() < witness_policy.minimum_distinct_regions {
        violations.push(PolicyHeadObservationError::InsufficientRegions {
            actual: regions.len(),
            required: witness_policy.minimum_distinct_regions,
        });
    }
    if witness_policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(PolicyHeadObservationError::MissingAlgorithmDiversity);
    }

    witness_identities.sort();
    statement_digests.sort();
    let witness_quorum_digest = digest_legacy_witness_quorum(
        signed_checkpoint.checkpoint_digest,
        &statement_digests,
    );
    if verified_witness_quorum.checkpoint_digest() != signed_checkpoint.checkpoint_digest
        || verified_witness_quorum.witness_quorum_digest() != witness_quorum_digest
        || verified_witness_quorum.trust_snapshot_digest() != trust_snapshot_digest
        || verified_witness_quorum.witnesses() != witness_identities.as_slice()
    {
        violations.push(PolicyHeadObservationError::WitnessVerifiedEvidenceMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let transparency_log_digest = digest_transparency_log(log)
        .map_err(|error| vec![PolicyHeadObservationError::TransparencyLogInvalid(format!("{error:?}"))])?;
    let signed_checkpoint_evidence_digest = hash_serializable(
        SIGNED_CHECKPOINT_EVIDENCE_DOMAIN,
        signed_checkpoint,
    )
    .map_err(|error| vec![error])?;
    witness_evidence_digests.sort();
    let witness_set_evidence_digest = digest_witness_evidence_set(&witness_evidence_digests);
    let id = QuorumObservedPolicyHeadIdV1(
        digest_observed_head(
            lineage.id(),
            lineage.sequence(),
            temporal.id(),
            publication_digest,
            publication_entry.sequence,
            transparency_log_digest,
            log.entries.len() as u64,
            log_root,
            signed_checkpoint.checkpoint_digest,
            signed_checkpoint_evidence_digest,
            witness_quorum_digest,
            witness_set_evidence_digest,
            trust_snapshot_digest,
            containment_state_digest,
            compromise_tracker_digest,
            clock.id(),
        )
        .map_err(|error| vec![error])?,
    );

    Ok(QuorumObservedPolicyHeadV1 {
        id,
        lineage_id: lineage.id(),
        lineage_sequence: lineage.sequence(),
        temporal_validity_permit_id: temporal.id(),
        publication_digest,
        publication_entry_sequence: publication_entry.sequence,
        transparency_log_digest,
        transparency_log_size: log.entries.len() as u64,
        transparency_root_digest: log_root,
        checkpoint_digest: signed_checkpoint.checkpoint_digest,
        signed_checkpoint_evidence_digest,
        witness_quorum_digest,
        witness_set_evidence_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        clock_envelope_id: clock.id(),
    })
}

fn require_temporal_matches_lineage(
    lineage: &ClockGovernedPolicyLineageV1,
    temporal: &ClockGovernedPolicyTemporalValidityPermitV1,
) -> Result<(), PolicyHeadObservationError> {
    if temporal.lineage_id() != lineage.id()
        || temporal.lineage_sequence() != lineage.sequence()
        || temporal.current_policy_binding_digest() != lineage.current_policy_binding_digest()
        || temporal.active_waiver_count() != lineage.active_waivers().len()
    {
        return Err(PolicyHeadObservationError::TemporalPermitMismatch);
    }
    Ok(())
}

fn validate_publication(
    publication: &PolicyLineageHeadPublicationV1,
) -> Result<(), PolicyHeadObservationError> {
    if publication.schema_version != POLICY_LINEAGE_HEAD_PUBLICATION_SCHEMA
        || publication.lineage_sequence == 0
        || publication.lineage_id.len() != 64
        || publication.temporal_validity_permit_id.len() != 64
        || publication.current_operational_basis_id.len() != 64
        || publication.current_clock_envelope_id.len() != 64
        || publication.current_policy_binding_digest == Sha256Digest([0; 32])
    {
        return Err(PolicyHeadObservationError::InvalidPublication);
    }
    validate_identifier(&publication.policy_domain)?;
    Ok(())
}

fn validate_witness_policy(
    policy: &TransparencyWitnessPolicy,
    violations: &mut Vec<PolicyHeadObservationError>,
) {
    if policy.minimum_distinct_witnesses == 0
        || policy.minimum_distinct_organizations == 0
        || policy.minimum_distinct_regions == 0
        || policy.maximum_observation_age_s == 0
        || policy.maximum_witnesses == 0
        || policy.maximum_witnesses > MAX_TRANSPARENCY_WITNESSES
        || policy.minimum_distinct_witnesses > policy.maximum_witnesses
        || policy.minimum_distinct_organizations > policy.maximum_witnesses
        || policy.minimum_distinct_regions > policy.maximum_witnesses
    {
        violations.push(PolicyHeadObservationError::WitnessPolicyInvalid);
    }
}

fn requalify_signer(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    usage: KeyUsage,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<PolicyHeadObservationError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(PolicyHeadObservationError::SignerUnknown(key_id.to_string()));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(PolicyHeadObservationError::SignerNotActive(key_id.to_string()));
    }
    if !record.usages.contains(&usage) {
        violations.push(PolicyHeadObservationError::SignerUsageNotAllowed(key_id.to_string()));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(PolicyHeadObservationError::SignerNotValidAcrossEnvelope {
            key_id: key_id.to_string(),
            reason,
        });
    }
    for compromise in containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|compromise| {
            &compromise.signer.algorithm == algorithm
                && compromise.signer.key_id == key_id
                && compromise.affected_usages.contains(&usage)
        })
    {
        match clock.require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s) {
            Ok(()) => {}
            Err(ClockGovernanceTimeError::EventMayAlreadyBeEffective) => violations.push(
                PolicyHeadObservationError::SignerCompromisedAcrossEnvelope(key_id.to_string()),
            ),
            Err(reason) => violations.push(PolicyHeadObservationError::CompromiseTimeInvalid {
                key_id: key_id.to_string(),
                reason,
            }),
        }
    }
}

fn validate_identifier(value: &str) -> Result<(), PolicyHeadObservationError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(PolicyHeadObservationError::InvalidPublication);
    }
    Ok(())
}

fn seconds_to_millis(value: u64) -> Result<u64, PolicyHeadObservationError> {
    value
        .checked_mul(1_000)
        .ok_or(PolicyHeadObservationError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, PolicyHeadObservationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| PolicyHeadObservationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn digest_legacy_witness_quorum(
    checkpoint_digest: Sha256Digest,
    statement_digests: &[Sha256Digest],
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(LEGACY_WITNESS_QUORUM_DOMAIN);
    hasher.update(&checkpoint_digest.0);
    for digest in statement_digests {
        hasher.update(&digest.0);
    }
    hasher.finalize()
}

fn digest_witness_evidence_set(witness_digests: &[Sha256Digest]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(WITNESS_SET_EVIDENCE_DOMAIN);
    hasher.update(&(witness_digests.len() as u64).to_le_bytes());
    for digest in witness_digests {
        hasher.update(&digest.0);
    }
    hasher.finalize()
}

#[derive(Serialize)]
struct ObservedHeadCommitment {
    schema: &'static str,
    lineage_id: String,
    lineage_sequence: u64,
    temporal_validity_permit_id: String,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    transparency_log_size: u64,
    transparency_root_digest: String,
    checkpoint_digest: String,
    signed_checkpoint_evidence_digest: String,
    witness_quorum_digest: String,
    witness_set_evidence_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
}

#[allow(clippy::too_many_arguments)]
fn digest_observed_head(
    lineage_id: ClockGovernedPolicyLineageIdV1,
    lineage_sequence: u64,
    temporal_validity_permit_id: ClockGovernedPolicyTemporalValidityPermitIdV1,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    transparency_log_size: u64,
    transparency_root_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    signed_checkpoint_evidence_digest: Sha256Digest,
    witness_quorum_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
) -> Result<Sha256Digest, PolicyHeadObservationError> {
    hash_serializable(
        OBSERVED_HEAD_DOMAIN,
        &ObservedHeadCommitment {
            schema: QUORUM_OBSERVED_POLICY_HEAD_SCHEMA,
            lineage_id: lineage_id.to_hex(),
            lineage_sequence,
            temporal_validity_permit_id: temporal_validity_permit_id.to_hex(),
            publication_digest: publication_digest.to_hex(),
            publication_entry_sequence,
            transparency_log_digest: transparency_log_digest.to_hex(),
            transparency_log_size,
            transparency_root_digest: transparency_root_digest.to_hex(),
            checkpoint_digest: checkpoint_digest.to_hex(),
            signed_checkpoint_evidence_digest: signed_checkpoint_evidence_digest.to_hex(),
            witness_quorum_digest: witness_quorum_digest.to_hex(),
            witness_set_evidence_digest: witness_set_evidence_digest.to_hex(),
            trust_snapshot_digest: trust_snapshot_digest.to_hex(),
            containment_state_digest: containment_state_digest.to_hex(),
            compromise_tracker_digest: compromise_tracker_digest.to_hex(),
            clock_envelope_id: clock_envelope_id.to_hex(),
        },
    )
}
