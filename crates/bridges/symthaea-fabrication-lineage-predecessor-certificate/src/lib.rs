// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable predecessor certificates for crossing the finalized-upgrade -> next-handoff crate cycle.
//!
//! The certificate itself is audit data, not authority. Live authority exists only after a distinct
//! interval-qualified threshold ceremony signs the exact certificate digest under the *same* threshold
//! policy, trust snapshot, compromise tracker, and trusted-clock envelope as one exact prepared handoff.
//! The portable record also carries the exact witnessed governance/registry view that established the
//! predecessor as current, reducing the facts that the notarizing quorum must accept by convention.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::upgrade_handoff::{UpgradeEndpoint, digest_upgrade_endpoint};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_fabrication_upgrade_authority::{
    PreparedClockGovernedUpgradeHandoffIdV1, PreparedClockGovernedUpgradeHandoffV1,
};

pub const LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA: &str =
    "symthaea.fabrication.lineage-finalized-predecessor-certificate.v1";
pub const VERIFIED_LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA: &str =
    "symthaea.fabrication.verified-lineage-finalized-predecessor-certificate.v1";
pub const LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_PURPOSE: &str =
    "lineage-finalized-predecessor-certificate-v1";

const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-finalized-predecessor-certificate.v1\0";
const VERIFIED_CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.fabrication.verified-lineage-finalized-predecessor-certificate.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageFinalizedPredecessorCertificateV1 {
    pub schema_version: String,
    pub predecessor_root_id: String,
    pub global_head_id: String,
    pub current_head_id: String,
    pub governance_view_id: String,
    pub registry_head_id: String,
    pub finalized_upgrade_id: String,
    pub finalization_record_digest: Sha256Digest,
    pub prior_predecessor_root_digest: Sha256Digest,
    pub finalization_sequence: u64,
    pub endpoint: UpgradeEndpoint,
    pub endpoint_digest: Sha256Digest,
    pub rollback_target_digest: Sha256Digest,
    pub evidence_checkpoint_digest: Sha256Digest,
    pub transparency_log_digest: Sha256Digest,
    pub registry_digest: Sha256Digest,
    pub registry_sequence: u64,
    pub trust_snapshot_digest: Sha256Digest,
    pub containment_state_digest: Sha256Digest,
    pub compromise_tracker_digest: Sha256Digest,
    pub containment_generation: u64,
    pub clock_envelope_id: String,
    pub operational_basis_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerifiedLineageFinalizedPredecessorCertificateIdV1(Sha256Digest);
impl VerifiedLineageFinalizedPredecessorCertificateIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct VerifiedLineageFinalizedPredecessorCertificateV1 {
    id: VerifiedLineageFinalizedPredecessorCertificateIdV1,
    certificate_digest: Sha256Digest,
    certificate_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    certificate_ceremony_digest: Sha256Digest,
    prepared_handoff_id: PreparedClockGovernedUpgradeHandoffIdV1,
    predecessor_root_digest: Sha256Digest,
    current_head_digest: Sha256Digest,
    finalization_sequence: u64,
    endpoint_digest: Sha256Digest,
    rollback_target_digest: Sha256Digest,
    evidence_checkpoint_digest: Sha256Digest,
    transparency_log_digest: Sha256Digest,
}

impl VerifiedLineageFinalizedPredecessorCertificateV1 {
    pub fn id(&self) -> VerifiedLineageFinalizedPredecessorCertificateIdV1 { self.id }
    pub fn certificate_digest(&self) -> Sha256Digest { self.certificate_digest }
    pub fn certificate_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 { self.certificate_ceremony_id }
    pub fn certificate_ceremony_digest(&self) -> Sha256Digest { self.certificate_ceremony_digest }
    pub fn prepared_handoff_id(&self) -> PreparedClockGovernedUpgradeHandoffIdV1 { self.prepared_handoff_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn current_head_digest(&self) -> Sha256Digest { self.current_head_digest }
    pub fn finalization_sequence(&self) -> u64 { self.finalization_sequence }
    pub fn endpoint_digest(&self) -> Sha256Digest { self.endpoint_digest }
    pub fn rollback_target_digest(&self) -> Sha256Digest { self.rollback_target_digest }
    pub fn evidence_checkpoint_digest(&self) -> Sha256Digest { self.evidence_checkpoint_digest }
    pub fn transparency_log_digest(&self) -> Sha256Digest { self.transparency_log_digest }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineagePredecessorCertificateError {
    InvalidSchema,
    InvalidCanonicalId(&'static str),
    ZeroDigest(&'static str),
    InvalidFinalizationSequence,
    InvalidRegistrySequence,
    InvalidContainmentGeneration,
    EndpointInvalid(String),
    EndpointDigestMismatch,
    RollbackTargetMismatch,
    PreparedPlanPredecessorMismatch,
    PreparedPlanRollbackMismatch,
    PreparedPlanCheckpointMismatch,
    CertificateTrustSnapshotMismatch,
    CertificateContainmentMismatch,
    CertificateCompromiseTrackerMismatch,
    CertificateClockMismatch,
    CertificateBasisMismatch,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    CeremonyPolicyMismatch,
    CeremonyTrustSnapshotMismatch,
    CeremonyCompromiseTrackerMismatch,
    CeremonyClockMismatch,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct VerifiedCertificateCommitment {
    schema: &'static str,
    certificate_digest: String,
    certificate_ceremony_id: String,
    certificate_ceremony_digest: String,
    prepared_handoff_id: String,
    predecessor_root_id: String,
    current_head_id: String,
    governance_view_id: String,
    registry_head_id: String,
    finalization_sequence: u64,
    endpoint_digest: String,
    rollback_target_digest: String,
    evidence_checkpoint_digest: String,
    transparency_log_digest: String,
}

pub fn digest_lineage_finalized_predecessor_certificate_v1(
    certificate: &LineageFinalizedPredecessorCertificateV1,
) -> Result<Sha256Digest, LineagePredecessorCertificateError> {
    validate_lineage_finalized_predecessor_certificate_v1(certificate)?;
    hash_serializable(CERTIFICATE_DOMAIN, certificate)
}

pub fn validate_lineage_finalized_predecessor_certificate_v1(
    certificate: &LineageFinalizedPredecessorCertificateV1,
) -> Result<(), LineagePredecessorCertificateError> {
    if certificate.schema_version != LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA {
        return Err(LineagePredecessorCertificateError::InvalidSchema);
    }
    for (name, value) in [
        ("predecessor_root_id", certificate.predecessor_root_id.as_str()),
        ("global_head_id", certificate.global_head_id.as_str()),
        ("current_head_id", certificate.current_head_id.as_str()),
        ("governance_view_id", certificate.governance_view_id.as_str()),
        ("registry_head_id", certificate.registry_head_id.as_str()),
        ("finalized_upgrade_id", certificate.finalized_upgrade_id.as_str()),
        ("clock_envelope_id", certificate.clock_envelope_id.as_str()),
        ("operational_basis_id", certificate.operational_basis_id.as_str()),
    ] {
        if !canonical_hex_id(value) {
            return Err(LineagePredecessorCertificateError::InvalidCanonicalId(name));
        }
    }
    if certificate.finalization_sequence == 0 {
        return Err(LineagePredecessorCertificateError::InvalidFinalizationSequence);
    }
    if certificate.registry_sequence == 0 {
        return Err(LineagePredecessorCertificateError::InvalidRegistrySequence);
    }
    if certificate.containment_generation == 0 {
        return Err(LineagePredecessorCertificateError::InvalidContainmentGeneration);
    }
    for (name, digest) in [
        ("finalization_record_digest", certificate.finalization_record_digest),
        ("prior_predecessor_root_digest", certificate.prior_predecessor_root_digest),
        ("endpoint_digest", certificate.endpoint_digest),
        ("rollback_target_digest", certificate.rollback_target_digest),
        ("evidence_checkpoint_digest", certificate.evidence_checkpoint_digest),
        ("transparency_log_digest", certificate.transparency_log_digest),
        ("registry_digest", certificate.registry_digest),
        ("trust_snapshot_digest", certificate.trust_snapshot_digest),
        ("containment_state_digest", certificate.containment_state_digest),
        ("compromise_tracker_digest", certificate.compromise_tracker_digest),
    ] {
        if digest == Sha256Digest([0; 32]) {
            return Err(LineagePredecessorCertificateError::ZeroDigest(name));
        }
    }
    certificate.endpoint.validate()
        .map_err(|error| LineagePredecessorCertificateError::EndpointInvalid(format!("{error:?}")))?;
    let endpoint_digest = digest_upgrade_endpoint(&certificate.endpoint)
        .map_err(|error| LineagePredecessorCertificateError::EndpointInvalid(format!("{error:?}")))?;
    if endpoint_digest != certificate.endpoint_digest {
        return Err(LineagePredecessorCertificateError::EndpointDigestMismatch);
    }
    if certificate.rollback_target_digest != certificate.endpoint.durable_state_digest {
        return Err(LineagePredecessorCertificateError::RollbackTargetMismatch);
    }
    Ok(())
}

pub fn verify_lineage_finalized_predecessor_certificate_v1(
    certificate: &LineageFinalizedPredecessorCertificateV1,
    certificate_ceremony: &ClockGovernedThresholdCeremonyV1,
    prepared_handoff: &PreparedClockGovernedUpgradeHandoffV1,
) -> Result<VerifiedLineageFinalizedPredecessorCertificateV1, LineagePredecessorCertificateError> {
    let certificate_digest = digest_lineage_finalized_predecessor_certificate_v1(certificate)?;
    if prepared_handoff.plan().predecessor != certificate.endpoint {
        return Err(LineagePredecessorCertificateError::PreparedPlanPredecessorMismatch);
    }
    if prepared_handoff.plan().rollback_target_digest != certificate.rollback_target_digest {
        return Err(LineagePredecessorCertificateError::PreparedPlanRollbackMismatch);
    }
    if prepared_handoff.plan().evidence_checkpoint_digest != certificate.evidence_checkpoint_digest {
        return Err(LineagePredecessorCertificateError::PreparedPlanCheckpointMismatch);
    }
    if prepared_handoff.trust_snapshot_digest() != certificate.trust_snapshot_digest {
        return Err(LineagePredecessorCertificateError::CertificateTrustSnapshotMismatch);
    }
    if prepared_handoff.containment_state_digest() != certificate.containment_state_digest {
        return Err(LineagePredecessorCertificateError::CertificateContainmentMismatch);
    }
    if prepared_handoff.compromise_tracker_digest() != certificate.compromise_tracker_digest {
        return Err(LineagePredecessorCertificateError::CertificateCompromiseTrackerMismatch);
    }
    if prepared_handoff.clock_envelope_id().to_hex() != certificate.clock_envelope_id {
        return Err(LineagePredecessorCertificateError::CertificateClockMismatch);
    }
    if prepared_handoff.operational_basis_id().to_hex() != certificate.operational_basis_id {
        return Err(LineagePredecessorCertificateError::CertificateBasisMismatch);
    }
    if certificate_ceremony.purpose() != LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_PURPOSE {
        return Err(LineagePredecessorCertificateError::CeremonyPurposeMismatch);
    }
    if certificate_ceremony.payload_digest() != certificate_digest {
        return Err(LineagePredecessorCertificateError::CeremonyPayloadMismatch);
    }
    if certificate_ceremony.policy_digest() != prepared_handoff.threshold_policy_digest() {
        return Err(LineagePredecessorCertificateError::CeremonyPolicyMismatch);
    }
    if certificate_ceremony.trust_snapshot_digest() != prepared_handoff.trust_snapshot_digest() {
        return Err(LineagePredecessorCertificateError::CeremonyTrustSnapshotMismatch);
    }
    if certificate_ceremony.compromise_tracker_digest() != prepared_handoff.compromise_tracker_digest() {
        return Err(LineagePredecessorCertificateError::CeremonyCompromiseTrackerMismatch);
    }
    if certificate_ceremony.clock_envelope_id() != prepared_handoff.clock_envelope_id() {
        return Err(LineagePredecessorCertificateError::CeremonyClockMismatch);
    }

    let predecessor_root_digest = parse_hex_digest(&certificate.predecessor_root_id)
        .ok_or(LineagePredecessorCertificateError::InvalidCanonicalId("predecessor_root_id"))?;
    let current_head_digest = parse_hex_digest(&certificate.current_head_id)
        .ok_or(LineagePredecessorCertificateError::InvalidCanonicalId("current_head_id"))?;
    let commitment = VerifiedCertificateCommitment {
        schema: VERIFIED_LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA,
        certificate_digest: certificate_digest.to_hex(),
        certificate_ceremony_id: certificate_ceremony.id().to_hex(),
        certificate_ceremony_digest: certificate_ceremony.ceremony_digest().to_hex(),
        prepared_handoff_id: prepared_handoff.id().to_hex(),
        predecessor_root_id: certificate.predecessor_root_id.clone(),
        current_head_id: certificate.current_head_id.clone(),
        governance_view_id: certificate.governance_view_id.clone(),
        registry_head_id: certificate.registry_head_id.clone(),
        finalization_sequence: certificate.finalization_sequence,
        endpoint_digest: certificate.endpoint_digest.to_hex(),
        rollback_target_digest: certificate.rollback_target_digest.to_hex(),
        evidence_checkpoint_digest: certificate.evidence_checkpoint_digest.to_hex(),
        transparency_log_digest: certificate.transparency_log_digest.to_hex(),
    };
    let id = VerifiedLineageFinalizedPredecessorCertificateIdV1(
        hash_serializable(VERIFIED_CERTIFICATE_DOMAIN, &commitment)?,
    );
    Ok(VerifiedLineageFinalizedPredecessorCertificateV1 {
        id,
        certificate_digest,
        certificate_ceremony_id: certificate_ceremony.id(),
        certificate_ceremony_digest: certificate_ceremony.ceremony_digest(),
        prepared_handoff_id: prepared_handoff.id(),
        predecessor_root_digest,
        current_head_digest,
        finalization_sequence: certificate.finalization_sequence,
        endpoint_digest: certificate.endpoint_digest,
        rollback_target_digest: certificate.rollback_target_digest,
        evidence_checkpoint_digest: certificate.evidence_checkpoint_digest,
        transparency_log_digest: certificate.transparency_log_digest,
    })
}

fn canonical_hex_id(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_hex_digest(value: &str) -> Option<Sha256Digest> {
    if !canonical_hex_id(value) { return None; }
    let mut bytes = [0u8; 32];
    for (index, slot) in bytes.iter_mut().enumerate() {
        let offset = index * 2;
        *slot = u8::from_str_radix(&value[offset..offset + 2], 16).ok()?;
    }
    Some(Sha256Digest(bytes))
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8], value: &T,
) -> Result<Sha256Digest, LineagePredecessorCertificateError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineagePredecessorCertificateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
