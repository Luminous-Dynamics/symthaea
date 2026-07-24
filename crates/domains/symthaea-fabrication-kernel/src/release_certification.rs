// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signed release-candidate certification over exact fabrication and gateway evidence.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_consensus::VerifiedGatewayConsensus;
use crate::gateway_recovery::{GatewayRecoveryBundle, GatewayRecoveryError};
use crate::gateway_state::{FabricationGatewayState, GatewayStateError};
use crate::incident_ledger::{IncidentLedger, IncidentLedgerError};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const RELEASE_CANDIDATE_SCHEMA: &str = "symthaea.fabrication.release-candidate.v1";
pub const SIGNED_RELEASE_CANDIDATE_SCHEMA: &str =
    "symthaea.fabrication.signed-release-candidate.v1";
pub const MAX_RELEASE_CANDIDATE_SIGNATURES: usize = 32;
pub const MAX_RELEASE_CANDIDATE_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_RELEASE_CANDIDATE_ID_BYTES: usize = 256;
pub const MAX_RELEASE_VERSION_BYTES: usize = 128;
pub const MAX_INCIDENT_REFERENCES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseCandidateEvidence {
    pub schema_version: String,
    pub candidate_id: String,
    pub software_version: String,
    pub created_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub source_tree_digest: Sha256Digest,
    pub manifest_digest: Sha256Digest,
    pub governed_replay_digest: Sha256Digest,
    pub gateway_replay_digest: Sha256Digest,
    pub gateway_state_digest: Sha256Digest,
    pub gateway_generation: u64,
    pub gateway_consensus_digest: Sha256Digest,
    pub recovery_bundle_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub unresolved_incident_digests: Vec<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedReleaseCandidate {
    pub schema_version: String,
    pub candidate: ReleaseCandidateEvidence,
    pub candidate_digest: Sha256Digest,
    pub signatures: Vec<DetachedSignature>,
}

pub trait ReleaseCandidateSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_release_candidate(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait ReleaseCandidateVerifier {
    fn verify_release_candidate(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseCertificationError {
    UnsupportedSchema,
    EmptyCandidateId,
    NonCanonicalCandidateId,
    CandidateIdTooLong,
    InvalidSoftwareVersion,
    InvalidWindow,
    GatewayState(GatewayStateError),
    Recovery(GatewayRecoveryError),
    ConsensusStateMismatch,
    ConsensusGenerationMismatch,
    ConsensusTrustMismatch,
    RecoveryStateMismatch,
    TooManyIncidentReferences {
        actual: usize,
        maximum: usize,
    },
    DuplicateIncidentReference,
    InvalidAlgorithm,
    EmptyKeyId,
    NonCanonicalKeyId,
    EmptySignature,
    SignatureTooLarge {
        actual: usize,
        maximum: usize,
    },
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    Signing {
        key_id: String,
        reason: String,
    },
    IncidentLedger(IncidentLedgerError),
    Encoding(String),
}

impl ReleaseCandidateEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        candidate_id: impl Into<String>,
        software_version: impl Into<String>,
        created_at_unix_s: u64,
        expires_at_unix_s: u64,
        source_tree_digest: Sha256Digest,
        manifest_digest: Sha256Digest,
        governed_replay_digest: Sha256Digest,
        gateway_replay_digest: Sha256Digest,
        gateway_state: &FabricationGatewayState,
        consensus: &VerifiedGatewayConsensus,
        recovery: &GatewayRecoveryBundle,
        unresolved_incident_digests: Vec<Sha256Digest>,
    ) -> Result<Self, ReleaseCertificationError> {
        gateway_state
            .validate()
            .map_err(ReleaseCertificationError::GatewayState)?;
        recovery
            .validate()
            .map_err(ReleaseCertificationError::Recovery)?;
        let gateway_state_digest = gateway_state
            .digest()
            .map_err(ReleaseCertificationError::GatewayState)?;
        if consensus.state_digest() != gateway_state_digest {
            return Err(ReleaseCertificationError::ConsensusStateMismatch);
        }
        if consensus.generation() != gateway_state.generation {
            return Err(ReleaseCertificationError::ConsensusGenerationMismatch);
        }
        let trust_snapshot_digest =
            digest_trust_snapshot(&gateway_state.trust_snapshot).map_err(|error| {
                ReleaseCertificationError::GatewayState(GatewayStateError::TrustSnapshot(error))
            })?;
        if consensus.trust_snapshot_digest() != trust_snapshot_digest {
            return Err(ReleaseCertificationError::ConsensusTrustMismatch);
        }
        if recovery
            .latest_state()
            .map_err(ReleaseCertificationError::Recovery)?
            .digest()
            .map_err(ReleaseCertificationError::GatewayState)?
            != gateway_state_digest
        {
            return Err(ReleaseCertificationError::RecoveryStateMismatch);
        }
        if recovery.checkpoints.last().is_none_or(|checkpoint| {
            checkpoint.consensus.consensus_digest != consensus.consensus_digest()
                || checkpoint.consensus.state_digest != consensus.state_digest()
                || checkpoint.consensus.generation != consensus.generation()
                || checkpoint.consensus.trust_snapshot_digest != consensus.trust_snapshot_digest()
        }) {
            return Err(ReleaseCertificationError::RecoveryStateMismatch);
        }
        let mut candidate = Self {
            schema_version: RELEASE_CANDIDATE_SCHEMA.into(),
            candidate_id: candidate_id.into(),
            software_version: software_version.into(),
            created_at_unix_s,
            expires_at_unix_s,
            source_tree_digest,
            manifest_digest,
            governed_replay_digest,
            gateway_replay_digest,
            gateway_state_digest,
            gateway_generation: gateway_state.generation,
            gateway_consensus_digest: consensus.consensus_digest(),
            recovery_bundle_digest: recovery.bundle_digest,
            trust_snapshot_digest,
            unresolved_incident_digests,
        };
        candidate.canonicalize();
        candidate.validate()?;
        Ok(candidate)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_from_incident_ledger(
        candidate_id: impl Into<String>,
        software_version: impl Into<String>,
        created_at_unix_s: u64,
        expires_at_unix_s: u64,
        source_tree_digest: Sha256Digest,
        manifest_digest: Sha256Digest,
        governed_replay_digest: Sha256Digest,
        gateway_replay_digest: Sha256Digest,
        gateway_state: &FabricationGatewayState,
        consensus: &VerifiedGatewayConsensus,
        recovery: &GatewayRecoveryBundle,
        incidents: &IncidentLedger,
    ) -> Result<Self, ReleaseCertificationError> {
        let unresolved = incidents
            .unresolved_digests()
            .map_err(ReleaseCertificationError::IncidentLedger)?;
        Self::build(
            candidate_id,
            software_version,
            created_at_unix_s,
            expires_at_unix_s,
            source_tree_digest,
            manifest_digest,
            governed_replay_digest,
            gateway_replay_digest,
            gateway_state,
            consensus,
            recovery,
            unresolved,
        )
    }

    pub fn canonicalize(&mut self) {
        self.unresolved_incident_digests.sort();
    }

    pub fn validate(&self) -> Result<(), ReleaseCertificationError> {
        if self.schema_version != RELEASE_CANDIDATE_SCHEMA {
            return Err(ReleaseCertificationError::UnsupportedSchema);
        }
        validate_text(
            &self.candidate_id,
            MAX_RELEASE_CANDIDATE_ID_BYTES,
            ReleaseCertificationError::EmptyCandidateId,
            ReleaseCertificationError::NonCanonicalCandidateId,
            ReleaseCertificationError::CandidateIdTooLong,
        )?;
        if self.software_version.trim().is_empty()
            || self.software_version != self.software_version.trim()
            || self.software_version.len() > MAX_RELEASE_VERSION_BYTES
        {
            return Err(ReleaseCertificationError::InvalidSoftwareVersion);
        }
        if self.created_at_unix_s >= self.expires_at_unix_s {
            return Err(ReleaseCertificationError::InvalidWindow);
        }
        if self.unresolved_incident_digests.len() > MAX_INCIDENT_REFERENCES {
            return Err(ReleaseCertificationError::TooManyIncidentReferences {
                actual: self.unresolved_incident_digests.len(),
                maximum: MAX_INCIDENT_REFERENCES,
            });
        }
        if self
            .unresolved_incident_digests
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(ReleaseCertificationError::DuplicateIncidentReference);
        }
        Ok(())
    }
}

pub fn digest_release_candidate(
    candidate: &ReleaseCandidateEvidence,
) -> Result<Sha256Digest, ReleaseCertificationError> {
    let mut canonical = candidate.clone();
    canonical.canonicalize();
    canonical.validate()?;
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| ReleaseCertificationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-candidate-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn sign_release_candidate(
    candidate: ReleaseCandidateEvidence,
    signers: &[&dyn ReleaseCandidateSigner],
) -> Result<SignedReleaseCandidate, ReleaseCertificationError> {
    candidate.validate()?;
    if signers.len() > MAX_RELEASE_CANDIDATE_SIGNATURES {
        return Err(ReleaseCertificationError::TooManySignatures {
            actual: signers.len(),
            maximum: MAX_RELEASE_CANDIDATE_SIGNATURES,
        });
    }
    let candidate_digest = digest_release_candidate(&candidate)?;
    let message = release_candidate_signature_message(candidate_digest);
    let mut identities = BTreeSet::new();
    let mut signatures = Vec::with_capacity(signers.len());
    for signer in signers {
        let algorithm = signer.algorithm();
        if !algorithm.is_canonical() {
            return Err(ReleaseCertificationError::InvalidAlgorithm);
        }
        let key_id = signer.key_id();
        if key_id.trim().is_empty() {
            return Err(ReleaseCertificationError::EmptyKeyId);
        }
        if key_id != key_id.trim() {
            return Err(ReleaseCertificationError::NonCanonicalKeyId);
        }
        if !identities.insert((algorithm.clone(), key_id.to_string())) {
            return Err(ReleaseCertificationError::DuplicateSigner {
                algorithm,
                key_id: key_id.to_string(),
            });
        }
        let signature = signer.sign_release_candidate(&message).map_err(|reason| {
            ReleaseCertificationError::Signing {
                key_id: key_id.to_string(),
                reason,
            }
        })?;
        if signature.is_empty() {
            return Err(ReleaseCertificationError::EmptySignature);
        }
        if signature.len() > MAX_RELEASE_CANDIDATE_SIGNATURE_BYTES {
            return Err(ReleaseCertificationError::SignatureTooLarge {
                actual: signature.len(),
                maximum: MAX_RELEASE_CANDIDATE_SIGNATURE_BYTES,
            });
        }
        signatures.push(DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature,
        });
    }
    Ok(SignedReleaseCandidate {
        schema_version: SIGNED_RELEASE_CANDIDATE_SCHEMA.into(),
        candidate,
        candidate_digest,
        signatures,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseCertificationPolicy {
    pub minimum_valid_signatures: usize,
    pub maximum_signatures: usize,
    pub require_algorithm_diversity: bool,
    pub require_no_unresolved_incidents: bool,
}

impl Default for ReleaseCertificationPolicy {
    fn default() -> Self {
        Self {
            minimum_valid_signatures: 2,
            maximum_signatures: MAX_RELEASE_CANDIDATE_SIGNATURES,
            require_algorithm_diversity: true,
            require_no_unresolved_incidents: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseCertificationViolation {
    InvalidPolicy,
    UnsupportedSchema,
    InvalidCandidate(ReleaseCertificationError),
    DigestMismatch,
    NotYetValid,
    Expired,
    UnresolvedIncidents(usize),
    TooManySignatures,
    DuplicateSigner(String),
    SignatureTooLarge {
        key_id: String,
        actual: usize,
        maximum: usize,
    },
    TrustSnapshotInvalid,
    TrustSnapshotStale,
    TrustSnapshotMismatch,
    SignerIneligible(String),
    InvalidSignature(String),
    VerificationProviderError {
        key_id: String,
        reason: String,
    },
    InsufficientSignatures {
        actual: usize,
        required: usize,
    },
    MissingAlgorithmDiversity,
}

#[derive(Debug, Clone)]
pub struct CertifiedReleaseCandidate {
    signed: SignedReleaseCandidate,
    valid_signers: Vec<(SignatureAlgorithm, String)>,
}

impl CertifiedReleaseCandidate {
    pub fn candidate(&self) -> &ReleaseCandidateEvidence {
        &self.signed.candidate
    }
    pub fn candidate_digest(&self) -> Sha256Digest {
        self.signed.candidate_digest
    }
    pub fn valid_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.valid_signers
    }
}

pub fn verify_release_candidate(
    signed: SignedReleaseCandidate,
    policy: &ReleaseCertificationPolicy,
    trust_snapshot: &TrustSnapshot,
    now_unix_s: u64,
    verifier: &dyn ReleaseCandidateVerifier,
) -> Result<CertifiedReleaseCandidate, Vec<ReleaseCertificationViolation>> {
    let mut violations = Vec::new();
    if policy.minimum_valid_signatures == 0
        || policy.maximum_signatures == 0
        || policy.minimum_valid_signatures > policy.maximum_signatures
    {
        violations.push(ReleaseCertificationViolation::InvalidPolicy);
    }
    if signed.schema_version != SIGNED_RELEASE_CANDIDATE_SCHEMA {
        violations.push(ReleaseCertificationViolation::UnsupportedSchema);
    }
    if let Err(error) = signed.candidate.validate() {
        violations.push(ReleaseCertificationViolation::InvalidCandidate(error));
    }
    match digest_release_candidate(&signed.candidate) {
        Ok(digest) if digest != signed.candidate_digest => {
            violations.push(ReleaseCertificationViolation::DigestMismatch)
        }
        Err(error) => violations.push(ReleaseCertificationViolation::InvalidCandidate(error)),
        Ok(_) => {}
    }
    if now_unix_s < signed.candidate.created_at_unix_s {
        violations.push(ReleaseCertificationViolation::NotYetValid);
    }
    if now_unix_s >= signed.candidate.expires_at_unix_s {
        violations.push(ReleaseCertificationViolation::Expired);
    }
    if policy.require_no_unresolved_incidents
        && !signed.candidate.unresolved_incident_digests.is_empty()
    {
        violations.push(ReleaseCertificationViolation::UnresolvedIncidents(
            signed.candidate.unresolved_incident_digests.len(),
        ));
    }
    if signed.signatures.len() > policy.maximum_signatures {
        violations.push(ReleaseCertificationViolation::TooManySignatures);
    }
    if trust_snapshot.validate().is_err() {
        violations.push(ReleaseCertificationViolation::TrustSnapshotInvalid);
    }
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        violations.push(ReleaseCertificationViolation::TrustSnapshotStale);
    }
    match digest_trust_snapshot(trust_snapshot) {
        Ok(digest) if digest != signed.candidate.trust_snapshot_digest => {
            violations.push(ReleaseCertificationViolation::TrustSnapshotMismatch)
        }
        Err(_) => violations.push(ReleaseCertificationViolation::TrustSnapshotInvalid),
        Ok(_) => {}
    }
    let message = release_candidate_signature_message(signed.candidate_digest);
    let mut seen = BTreeSet::new();
    let mut valid_signers = Vec::new();
    let mut algorithms = BTreeSet::new();
    for signature in &signed.signatures {
        if signature.signature.len() > MAX_RELEASE_CANDIDATE_SIGNATURE_BYTES {
            violations.push(ReleaseCertificationViolation::SignatureTooLarge {
                key_id: signature.key_id.clone(),
                actual: signature.signature.len(),
                maximum: MAX_RELEASE_CANDIDATE_SIGNATURE_BYTES,
            });
            continue;
        }
        let identity = (signature.algorithm.clone(), signature.key_id.clone());
        if !seen.insert(identity.clone()) {
            violations.push(ReleaseCertificationViolation::DuplicateSigner(
                signature.key_id.clone(),
            ));
            continue;
        }
        if trust_snapshot.key_eligibility(
            &signature.algorithm,
            &signature.key_id,
            KeyUsage::ReleaseCertification,
            now_unix_s,
        ) != KeyEligibility::Eligible
        {
            violations.push(ReleaseCertificationViolation::SignerIneligible(
                signature.key_id.clone(),
            ));
            continue;
        }
        match verifier.verify_release_candidate(
            &signature.algorithm,
            &signature.key_id,
            &message,
            &signature.signature,
        ) {
            Ok(true) => {
                algorithms.insert(signature.algorithm.clone());
                valid_signers.push(identity);
            }
            Ok(false) => violations.push(ReleaseCertificationViolation::InvalidSignature(
                signature.key_id.clone(),
            )),
            Err(reason) => {
                violations.push(ReleaseCertificationViolation::VerificationProviderError {
                    key_id: signature.key_id.clone(),
                    reason,
                })
            }
        }
    }
    if valid_signers.len() < policy.minimum_valid_signatures {
        violations.push(ReleaseCertificationViolation::InsufficientSignatures {
            actual: valid_signers.len(),
            required: policy.minimum_valid_signatures,
        });
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(ReleaseCertificationViolation::MissingAlgorithmDiversity);
    }
    if !violations.is_empty() {
        return Err(violations);
    }
    Ok(CertifiedReleaseCandidate {
        signed,
        valid_signers,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseEvidenceMismatch {
    SourceTree,
    Manifest,
    GovernedReplay,
    GatewayReplay,
    GatewayState,
    GatewayGeneration,
    GatewayConsensus,
    RecoveryBundle,
    TrustSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseEvidenceVerificationReport {
    pub mismatches: Vec<ReleaseEvidenceMismatch>,
}

impl ReleaseEvidenceVerificationReport {
    pub fn exact(&self) -> bool {
        self.mismatches.is_empty()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_release_candidate_evidence(
    candidate: &ReleaseCandidateEvidence,
    source_tree_digest: Sha256Digest,
    manifest_digest: Sha256Digest,
    governed_replay_digest: Sha256Digest,
    gateway_replay_digest: Sha256Digest,
    gateway_state: &FabricationGatewayState,
    consensus: &VerifiedGatewayConsensus,
    recovery: &GatewayRecoveryBundle,
) -> Result<ReleaseEvidenceVerificationReport, ReleaseCertificationError> {
    candidate.validate()?;
    gateway_state
        .validate()
        .map_err(ReleaseCertificationError::GatewayState)?;
    recovery
        .validate()
        .map_err(ReleaseCertificationError::Recovery)?;
    let gateway_state_digest = gateway_state
        .digest()
        .map_err(ReleaseCertificationError::GatewayState)?;
    let trust_snapshot_digest =
        digest_trust_snapshot(&gateway_state.trust_snapshot).map_err(|error| {
            ReleaseCertificationError::GatewayState(GatewayStateError::TrustSnapshot(error))
        })?;
    let mut mismatches = Vec::new();
    if candidate.source_tree_digest != source_tree_digest {
        mismatches.push(ReleaseEvidenceMismatch::SourceTree);
    }
    if candidate.manifest_digest != manifest_digest {
        mismatches.push(ReleaseEvidenceMismatch::Manifest);
    }
    if candidate.governed_replay_digest != governed_replay_digest {
        mismatches.push(ReleaseEvidenceMismatch::GovernedReplay);
    }
    if candidate.gateway_replay_digest != gateway_replay_digest {
        mismatches.push(ReleaseEvidenceMismatch::GatewayReplay);
    }
    if candidate.gateway_state_digest != gateway_state_digest
        || consensus.state_digest() != gateway_state_digest
    {
        mismatches.push(ReleaseEvidenceMismatch::GatewayState);
    }
    if candidate.gateway_generation != gateway_state.generation
        || consensus.generation() != gateway_state.generation
    {
        mismatches.push(ReleaseEvidenceMismatch::GatewayGeneration);
    }
    if candidate.gateway_consensus_digest != consensus.consensus_digest() {
        mismatches.push(ReleaseEvidenceMismatch::GatewayConsensus);
    }
    if candidate.recovery_bundle_digest != recovery.bundle_digest
        || recovery
            .latest_state()
            .map_err(ReleaseCertificationError::Recovery)?
            .digest()
            .map_err(ReleaseCertificationError::GatewayState)?
            != gateway_state_digest
        || recovery.checkpoints.last().is_none_or(|checkpoint| {
            checkpoint.consensus.consensus_digest != consensus.consensus_digest()
                || checkpoint.consensus.state_digest != consensus.state_digest()
                || checkpoint.consensus.generation != consensus.generation()
                || checkpoint.consensus.trust_snapshot_digest != consensus.trust_snapshot_digest()
        })
    {
        mismatches.push(ReleaseEvidenceMismatch::RecoveryBundle);
    }
    if candidate.trust_snapshot_digest != trust_snapshot_digest
        || consensus.trust_snapshot_digest() != trust_snapshot_digest
    {
        mismatches.push(ReleaseEvidenceMismatch::TrustSnapshot);
    }
    Ok(ReleaseEvidenceVerificationReport { mismatches })
}

fn release_candidate_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.release-candidate-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn validate_text(
    value: &str,
    maximum: usize,
    empty: ReleaseCertificationError,
    noncanonical: ReleaseCertificationError,
    too_long: ReleaseCertificationError,
) -> Result<(), ReleaseCertificationError> {
    if value.trim().is_empty() {
        return Err(empty);
    }
    if value != value.trim() {
        return Err(noncanonical);
    }
    if value.len() > maximum {
        return Err(too_long);
    }
    Ok(())
}
