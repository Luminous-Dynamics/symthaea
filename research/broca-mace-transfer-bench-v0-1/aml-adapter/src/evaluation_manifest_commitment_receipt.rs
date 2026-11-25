// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consumer-side verification for the detached MACE evaluation-manifest witness.
//!
//! A valid signature alone does not establish pre-run commitment. This module
//! also requires an independently configured append-only-log verifier and
//! consumer-owned trusted timestamps for candidate access and episode start.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::evidence_qualification_gate::{
    EvidenceQualificationRejectV1, EvidenceQualificationSignatureVerifierV1,
};

pub const EVALUATION_MANIFEST_COMMITMENT_RECEIPT_SCHEMA_V1: &str =
    "mace_evaluation_manifest_commitment_receipt_v1";
pub const EVALUATION_MANIFEST_COMMITMENT_PAYLOAD_SCHEMA_V1: &str =
    "mace_evaluation_manifest_commitment_payload_v1";
pub const MAX_EVALUATION_MANIFEST_COMMITMENT_RECEIPT_BYTES_V1: usize = 1_048_576;
pub const MAX_EVALUATION_MANIFEST_UNIX_SECONDS_V1: u64 = 253_402_300_799;
pub const MAX_EVALUATION_MANIFEST_IDENTIFIER_BYTES_V1: usize = 256;

/// Strict detached witness receipt. Unknown fields and duplicate fields are
/// rejected by Serde at every nested object boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationManifestCommitmentReceiptV1 {
    pub schema_version: String,
    pub payload: EvaluationManifestCommitmentPayloadV1,
    pub attestation: EvaluationManifestCommitmentAttestationV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationManifestCommitmentPayloadV1 {
    pub schema_version: String,
    pub suite_id: String,
    pub suite_version: String,
    pub episode_manifest_root_sha256: String,
    pub sealed_evaluation_manifest_sha256: String,
    pub arm_assignment_root_sha256: String,
    pub pre_registration_root_sha256: String,
    pub sealed_artifacts_root_sha256: String,
    pub committed_at_unix: u64,
    pub candidate_access_profile_sha256: String,
    pub reveal_protocol_sha256: String,
    pub anchor_log_id: String,
    pub anchor_entry_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationManifestCommitmentAttestationV1 {
    pub witness_profile_sha256: String,
    pub verifier_profile_sha256: String,
    pub key_id: String,
    pub issued_at_unix: u64,
    pub payload_sha256: String,
    pub attestation_sha256: String,
    pub signature_base64: String,
}

/// Consumer-owned bindings. In particular, the two start times must come
/// from the trusted runner/custody record, not from the candidate or witness.
pub struct EvaluationManifestCommitmentExpectedBindingV1<'a> {
    pub suite_id: &'a str,
    pub suite_version: &'a str,
    pub episode_manifest_root_sha256: &'a str,
    pub sealed_evaluation_manifest_sha256: &'a str,
    pub arm_assignment_root_sha256: &'a str,
    pub pre_registration_root_sha256: &'a str,
    pub sealed_artifacts_root_sha256: &'a str,
    pub candidate_access_profile_sha256: &'a str,
    pub reveal_protocol_sha256: &'a str,
    pub anchor_log_id: &'a str,
    pub anchor_entry_id: &'a str,
    pub witness_profile_sha256: &'a str,
    pub verifier_profile_sha256: &'a str,
    pub first_episode_started_unix: u64,
    pub candidate_access_started_unix: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManifestWitnessLogRejectV1 {
    InclusionProofMissing,
    InclusionProofInvalid,
    CheckpointUntrusted,
    LogForkDetected,
}

/// Implementations must verify both inclusion of the committed values at the
/// referenced log entry and consistency from a previously pinned log
/// checkpoint. A lookup result or an unanchored log response is insufficient.
pub trait AppendOnlyManifestWitnessLogVerifierV1 {
    fn verify_inclusion_and_consistency(
        &self,
        anchor_log_id: &str,
        anchor_entry_id: &str,
        sealed_evaluation_manifest_sha256: &str,
        episode_manifest_root_sha256: &str,
        arm_assignment_root_sha256: &str,
        pre_registration_root_sha256: &str,
        sealed_artifacts_root_sha256: &str,
        committed_at_unix: u64,
        trusted_now_unix: u64,
    ) -> Result<(), ManifestWitnessLogRejectV1>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationManifestCommitmentReceiptRejectV1 {
    ReceiptTooLarge,
    InvalidJson,
    UnsupportedSchemaVersion,
    InvalidIdentifier,
    InvalidDigest,
    BindingMismatch,
    InvalidTrustedTime,
    WitnessCommitmentAfterCandidateAccess,
    WitnessCommitmentAfterEpisodeStart,
    WitnessIssuedBeforeCommit,
    WitnessIssuedInFuture,
    InvalidSignatureEncoding,
    PayloadDigestMismatch,
    AttestationDigestMismatch,
    CanonicalizationFailed,
    LogVerificationRejected(ManifestWitnessLogRejectV1),
    SignatureRejected(EvidenceQualificationRejectV1),
}

/// Private-field capability. Consumers should pass this object—not an
/// arbitrary digest string—when binding an outcome receipt to its evaluator
/// commitment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedEvaluationManifestCommitmentV1 {
    receipt_sha256: String,
    payload_sha256: String,
    attestation_sha256: String,
    suite_id: String,
    suite_version: String,
    episode_manifest_root_sha256: String,
    sealed_evaluation_manifest_sha256: String,
    arm_assignment_root_sha256: String,
    pre_registration_root_sha256: String,
    sealed_artifacts_root_sha256: String,
    committed_at_unix: u64,
    anchor_log_id: String,
    anchor_entry_id: String,
    witness_profile_sha256: String,
}

impl VerifiedEvaluationManifestCommitmentV1 {
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn attestation_sha256(&self) -> &str {
        &self.attestation_sha256
    }

    pub fn suite_id(&self) -> &str {
        &self.suite_id
    }

    pub fn suite_version(&self) -> &str {
        &self.suite_version
    }

    pub fn episode_manifest_root_sha256(&self) -> &str {
        &self.episode_manifest_root_sha256
    }

    pub fn sealed_evaluation_manifest_sha256(&self) -> &str {
        &self.sealed_evaluation_manifest_sha256
    }

    pub fn arm_assignment_root_sha256(&self) -> &str {
        &self.arm_assignment_root_sha256
    }

    pub fn pre_registration_root_sha256(&self) -> &str {
        &self.pre_registration_root_sha256
    }

    pub fn sealed_artifacts_root_sha256(&self) -> &str {
        &self.sealed_artifacts_root_sha256
    }

    pub fn committed_at_unix(&self) -> u64 {
        self.committed_at_unix
    }

    pub fn anchor_log_id(&self) -> &str {
        &self.anchor_log_id
    }

    pub fn anchor_entry_id(&self) -> &str {
        &self.anchor_entry_id
    }

    pub fn witness_profile_sha256(&self) -> &str {
        &self.witness_profile_sha256
    }
}

pub fn parse_and_verify_evaluation_manifest_commitment_receipt_v1(
    wire: &[u8],
    expected: &EvaluationManifestCommitmentExpectedBindingV1<'_>,
    trusted_now_unix: u64,
    verifier: &impl EvidenceQualificationSignatureVerifierV1,
    log_verifier: &impl AppendOnlyManifestWitnessLogVerifierV1,
) -> Result<VerifiedEvaluationManifestCommitmentV1, EvaluationManifestCommitmentReceiptRejectV1> {
    if wire.len() > MAX_EVALUATION_MANIFEST_COMMITMENT_RECEIPT_BYTES_V1 {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::ReceiptTooLarge);
    }

    let receipt: EvaluationManifestCommitmentReceiptV1 =
        serde_json::from_slice(wire)
            .map_err(|_| EvaluationManifestCommitmentReceiptRejectV1::InvalidJson)?;

    validate_evaluation_manifest_commitment_receipt_v1(
        &receipt,
        expected,
        trusted_now_unix,
        verifier,
        log_verifier,
    )
}

pub fn validate_evaluation_manifest_commitment_receipt_v1(
    receipt: &EvaluationManifestCommitmentReceiptV1,
    expected: &EvaluationManifestCommitmentExpectedBindingV1<'_>,
    trusted_now_unix: u64,
    verifier: &impl EvidenceQualificationSignatureVerifierV1,
    log_verifier: &impl AppendOnlyManifestWitnessLogVerifierV1,
) -> Result<VerifiedEvaluationManifestCommitmentV1, EvaluationManifestCommitmentReceiptRejectV1> {
    if receipt.schema_version != EVALUATION_MANIFEST_COMMITMENT_RECEIPT_SCHEMA_V1
        || receipt.payload.schema_version != EVALUATION_MANIFEST_COMMITMENT_PAYLOAD_SCHEMA_V1
    {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::UnsupportedSchemaVersion);
    }

    for atom in [
        receipt.payload.suite_id.as_str(),
        receipt.payload.suite_version.as_str(),
        receipt.payload.anchor_log_id.as_str(),
        receipt.payload.anchor_entry_id.as_str(),
        expected.suite_id,
        expected.suite_version,
        expected.anchor_log_id,
        expected.anchor_entry_id,
        receipt.attestation.key_id.as_str(),
    ] {
        validate_atom(atom)?;
    }

    for digest in [
        receipt.payload.episode_manifest_root_sha256.as_str(),
        receipt.payload.sealed_evaluation_manifest_sha256.as_str(),
        receipt.payload.arm_assignment_root_sha256.as_str(),
        receipt.payload.pre_registration_root_sha256.as_str(),
        receipt.payload.sealed_artifacts_root_sha256.as_str(),
        receipt.payload.candidate_access_profile_sha256.as_str(),
        receipt.payload.reveal_protocol_sha256.as_str(),
        receipt.attestation.witness_profile_sha256.as_str(),
        receipt.attestation.verifier_profile_sha256.as_str(),
        receipt.attestation.payload_sha256.as_str(),
        receipt.attestation.attestation_sha256.as_str(),
        expected.episode_manifest_root_sha256,
        expected.sealed_evaluation_manifest_sha256,
        expected.arm_assignment_root_sha256,
        expected.pre_registration_root_sha256,
        expected.sealed_artifacts_root_sha256,
        expected.candidate_access_profile_sha256,
        expected.reveal_protocol_sha256,
        expected.witness_profile_sha256,
        expected.verifier_profile_sha256,
    ] {
        validate_hash(digest)?;
    }

    if receipt.payload.suite_id != expected.suite_id
        || receipt.payload.suite_version != expected.suite_version
        || receipt.payload.episode_manifest_root_sha256 != expected.episode_manifest_root_sha256
        || receipt.payload.sealed_evaluation_manifest_sha256
            != expected.sealed_evaluation_manifest_sha256
        || receipt.payload.arm_assignment_root_sha256 != expected.arm_assignment_root_sha256
        || receipt.payload.pre_registration_root_sha256 != expected.pre_registration_root_sha256
        || receipt.payload.sealed_artifacts_root_sha256 != expected.sealed_artifacts_root_sha256
        || receipt.payload.candidate_access_profile_sha256
            != expected.candidate_access_profile_sha256
        || receipt.payload.reveal_protocol_sha256 != expected.reveal_protocol_sha256
        || receipt.payload.anchor_log_id != expected.anchor_log_id
        || receipt.payload.anchor_entry_id != expected.anchor_entry_id
        || receipt.attestation.witness_profile_sha256 != expected.witness_profile_sha256
        || receipt.attestation.verifier_profile_sha256 != expected.verifier_profile_sha256
    {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::BindingMismatch);
    }

    if trusted_now_unix > MAX_EVALUATION_MANIFEST_UNIX_SECONDS_V1
        || expected.first_episode_started_unix > MAX_EVALUATION_MANIFEST_UNIX_SECONDS_V1
        || expected.candidate_access_started_unix > MAX_EVALUATION_MANIFEST_UNIX_SECONDS_V1
    {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::InvalidTrustedTime);
    }

    if receipt.payload.committed_at_unix >= expected.candidate_access_started_unix {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::WitnessCommitmentAfterCandidateAccess);
    }
    if receipt.payload.committed_at_unix >= expected.first_episode_started_unix {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::WitnessCommitmentAfterEpisodeStart);
    }
    if receipt.payload.committed_at_unix > receipt.attestation.issued_at_unix {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::WitnessIssuedBeforeCommit);
    }
    if receipt.attestation.issued_at_unix > trusted_now_unix {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::WitnessIssuedInFuture);
    }

    let payload_sha256 = evaluation_manifest_commitment_payload_sha256_v1(&receipt.payload)?;
    if payload_sha256 != receipt.attestation.payload_sha256 {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::PayloadDigestMismatch);
    }

    let attestation_sha256 = evaluation_manifest_commitment_attestation_sha256_v1(
        &payload_sha256,
        &receipt.attestation,
    )?;
    if attestation_sha256 != receipt.attestation.attestation_sha256 {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::AttestationDigestMismatch);
    }

    if receipt.attestation.signature_base64.len() != 88 {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::InvalidSignatureEncoding);
    }
    let signature = STANDARD
        .decode(&receipt.attestation.signature_base64)
        .map_err(|_| EvaluationManifestCommitmentReceiptRejectV1::InvalidSignatureEncoding)?;
    if signature.len() != 64 || STANDARD.encode(&signature) != receipt.attestation.signature_base64 {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::InvalidSignatureEncoding);
    }

    verifier
        .verify_signature(
            &receipt.attestation.verifier_profile_sha256,
            &receipt.attestation.key_id,
            &attestation_sha256,
            &signature,
            receipt.attestation.issued_at_unix,
            trusted_now_unix,
        )
        .map_err(EvaluationManifestCommitmentReceiptRejectV1::SignatureRejected)?;

    let receipt_sha256 = evaluation_manifest_commitment_receipt_sha256_v1(receipt)?;

    log_verifier
        .verify_inclusion_and_consistency(
            &receipt.payload.anchor_log_id,
            &receipt.payload.anchor_entry_id,
            &receipt.payload.sealed_evaluation_manifest_sha256,
            &receipt.payload.episode_manifest_root_sha256,
            &receipt.payload.arm_assignment_root_sha256,
            &receipt.payload.pre_registration_root_sha256,
            &receipt.payload.sealed_artifacts_root_sha256,
            receipt.payload.committed_at_unix,
            trusted_now_unix,
        )
        .map_err(EvaluationManifestCommitmentReceiptRejectV1::LogVerificationRejected)?;

    Ok(VerifiedEvaluationManifestCommitmentV1 {
        receipt_sha256,
        payload_sha256,
        attestation_sha256,
        suite_id: receipt.payload.suite_id.clone(),
        suite_version: receipt.payload.suite_version.clone(),
        episode_manifest_root_sha256: receipt.payload.episode_manifest_root_sha256.clone(),
        sealed_evaluation_manifest_sha256: receipt.payload.sealed_evaluation_manifest_sha256.clone(),
        arm_assignment_root_sha256: receipt.payload.arm_assignment_root_sha256.clone(),
        pre_registration_root_sha256: receipt.payload.pre_registration_root_sha256.clone(),
        sealed_artifacts_root_sha256: receipt.payload.sealed_artifacts_root_sha256.clone(),
        committed_at_unix: receipt.payload.committed_at_unix,
        anchor_log_id: receipt.payload.anchor_log_id.clone(),
        anchor_entry_id: receipt.payload.anchor_entry_id.clone(),
        witness_profile_sha256: receipt.attestation.witness_profile_sha256.clone(),
    })
}

pub fn evaluation_manifest_commitment_payload_sha256_v1(
    payload: &EvaluationManifestCommitmentPayloadV1,
) -> Result<String, EvaluationManifestCommitmentReceiptRejectV1> {
    let canonical = serde_json_canonicalizer::to_vec(payload)
        .map_err(|_| EvaluationManifestCommitmentReceiptRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:evaluation-manifest-commitment-payload:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}

pub fn evaluation_manifest_commitment_attestation_sha256_v1(
    payload_sha256: &str,
    attestation: &EvaluationManifestCommitmentAttestationV1,
) -> Result<String, EvaluationManifestCommitmentReceiptRejectV1> {
    validate_hash(payload_sha256)?;
    let preimage = json!({
        "schema_version": "mace_evaluation_manifest_commitment_attestation_preimage_v1",
        "payload_sha256": payload_sha256,
        "witness_profile_sha256": attestation.witness_profile_sha256,
        "verifier_profile_sha256": attestation.verifier_profile_sha256,
        "key_id": attestation.key_id,
        "issued_at_unix": attestation.issued_at_unix
    });
    let canonical = serde_json_canonicalizer::to_vec(&preimage)
        .map_err(|_| EvaluationManifestCommitmentReceiptRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:evaluation-manifest-commitment-attestation:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}

pub fn evaluation_manifest_commitment_receipt_sha256_v1(
    receipt: &EvaluationManifestCommitmentReceiptV1,
) -> Result<String, EvaluationManifestCommitmentReceiptRejectV1> {
    let canonical = serde_json_canonicalizer::to_vec(receipt)
        .map_err(|_| EvaluationManifestCommitmentReceiptRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:evaluation-manifest-commitment-receipt:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}

fn validate_atom(value: &str) -> Result<(), EvaluationManifestCommitmentReceiptRejectV1> {
    if value.is_empty()
        || value.len() > MAX_EVALUATION_MANIFEST_IDENTIFIER_BYTES_V1
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), EvaluationManifestCommitmentReceiptRejectV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EvaluationManifestCommitmentReceiptRejectV1::InvalidDigest);
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qualification_trust::{
        qualification_key_id_v1, PinnedEd25519QualificationVerifierV1,
        QualificationKeyStatusV1, QualificationTrustKeyV1, QualificationTrustProfileV1,
    };
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    struct AcceptLog;
    impl AppendOnlyManifestWitnessLogVerifierV1 for AcceptLog {
        fn verify_inclusion_and_consistency(
            &self,
            _anchor_log_id: &str,
            _anchor_entry_id: &str,
            _sealed_evaluation_manifest_sha256: &str,
            _episode_manifest_root_sha256: &str,
            _arm_assignment_root_sha256: &str,
            _pre_registration_root_sha256: &str,
            _sealed_artifacts_root_sha256: &str,
            _committed_at_unix: u64,
            _trusted_now_unix: u64,
        ) -> Result<(), ManifestWitnessLogRejectV1> {
            Ok(())
        }
    }

    struct RejectLog;
    impl AppendOnlyManifestWitnessLogVerifierV1 for RejectLog {
        fn verify_inclusion_and_consistency(
            &self,
            _anchor_log_id: &str,
            _anchor_entry_id: &str,
            _sealed_evaluation_manifest_sha256: &str,
            _episode_manifest_root_sha256: &str,
            _arm_assignment_root_sha256: &str,
            _pre_registration_root_sha256: &str,
            _sealed_artifacts_root_sha256: &str,
            _committed_at_unix: u64,
            _trusted_now_unix: u64,
        ) -> Result<(), ManifestWitnessLogRejectV1> {
            Err(ManifestWitnessLogRejectV1::InclusionProofInvalid)
        }
    }

    struct Fixture {
        receipt: EvaluationManifestCommitmentReceiptV1,
        verifier: PinnedEd25519QualificationVerifierV1,
        signer: SigningKey,
        suite_id: String,
        suite_version: String,
        episode_manifest_root_sha256: String,
        sealed_evaluation_manifest_sha256: String,
        arm_assignment_root_sha256: String,
        pre_registration_root_sha256: String,
        sealed_artifacts_root_sha256: String,
        candidate_access_profile_sha256: String,
        reveal_protocol_sha256: String,
        anchor_log_id: String,
        anchor_entry_id: String,
        witness_profile_sha256: String,
        verifier_profile_sha256: String,
    }

    impl Fixture {
        fn expected(&self) -> EvaluationManifestCommitmentExpectedBindingV1<'_> {
            EvaluationManifestCommitmentExpectedBindingV1 {
                suite_id: &self.suite_id,
                suite_version: &self.suite_version,
                episode_manifest_root_sha256: &self.episode_manifest_root_sha256,
                sealed_evaluation_manifest_sha256: &self.sealed_evaluation_manifest_sha256,
                arm_assignment_root_sha256: &self.arm_assignment_root_sha256,
                pre_registration_root_sha256: &self.pre_registration_root_sha256,
                sealed_artifacts_root_sha256: &self.sealed_artifacts_root_sha256,
                candidate_access_profile_sha256: &self.candidate_access_profile_sha256,
                reveal_protocol_sha256: &self.reveal_protocol_sha256,
                anchor_log_id: &self.anchor_log_id,
                anchor_entry_id: &self.anchor_entry_id,
                witness_profile_sha256: &self.witness_profile_sha256,
                verifier_profile_sha256: &self.verifier_profile_sha256,
                first_episode_started_unix: 200,
                candidate_access_started_unix: 150,
            }
        }

        fn resign(&mut self) {
            let payload_sha256 =
                evaluation_manifest_commitment_payload_sha256_v1(&self.receipt.payload).unwrap();
            self.receipt.attestation.payload_sha256 = payload_sha256.clone();
            let attestation_sha256 = evaluation_manifest_commitment_attestation_sha256_v1(
                &payload_sha256,
                &self.receipt.attestation,
            )
            .unwrap();
            self.receipt.attestation.attestation_sha256 = attestation_sha256.clone();
            self.receipt.attestation.signature_base64 =
                STANDARD.encode(self.signer.sign(attestation_sha256.as_bytes()).to_bytes());
        }

        fn verify(
            &self,
            now: u64,
            expected: &EvaluationManifestCommitmentExpectedBindingV1<'_>,
            log: &impl AppendOnlyManifestWitnessLogVerifierV1,
        ) -> Result<VerifiedEvaluationManifestCommitmentV1, EvaluationManifestCommitmentReceiptRejectV1> {
            validate_evaluation_manifest_commitment_receipt_v1(
                &self.receipt,
                expected,
                now,
                &self.verifier,
                log,
            )
        }
    }

    fn fixture() -> Fixture {
        let signer = SigningKey::from_bytes(&[9; 32]);
        let public_key = signer.verifying_key().to_bytes();
        let key_id = qualification_key_id_v1(&public_key);
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-manifest-witness-test-verifier".to_owned(),
            profile_version: 1,
            keys: vec![QualificationTrustKeyV1 {
                key_id: key_id.clone(),
                public_key,
                status: QualificationKeyStatusV1::Active,
                not_before_unix: 10,
                not_after_unix: 1_000,
            }],
        };
        let verifier_profile_sha256 = profile.profile_sha256().unwrap();
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile,
            &verifier_profile_sha256,
            1,
        )
        .unwrap();

        let suite_id = "mace-test".to_owned();
        let suite_version = "v1".to_owned();
        let episode_manifest_root_sha256 = digest('b');
        let sealed_evaluation_manifest_sha256 = digest('c');
        let arm_assignment_root_sha256 = digest('d');
        let pre_registration_root_sha256 = digest('e');
        let sealed_artifacts_root_sha256 = digest('f');
        let candidate_access_profile_sha256 = digest('1');
        let reveal_protocol_sha256 = digest('2');
        let anchor_log_id = "test-log".to_owned();
        let anchor_entry_id = "entry-17".to_owned();
        let witness_profile_sha256 = digest('a');
        let verifier_profile_sha256 = verifier_profile_sha256.clone();
        let payload = EvaluationManifestCommitmentPayloadV1 {
            schema_version: EVALUATION_MANIFEST_COMMITMENT_PAYLOAD_SCHEMA_V1.to_owned(),
            suite_id: suite_id.clone(),
            suite_version: suite_version.clone(),
            episode_manifest_root_sha256: episode_manifest_root_sha256.clone(),
            sealed_evaluation_manifest_sha256: sealed_evaluation_manifest_sha256.clone(),
            arm_assignment_root_sha256: arm_assignment_root_sha256.clone(),
            pre_registration_root_sha256: pre_registration_root_sha256.clone(),
            sealed_artifacts_root_sha256: sealed_artifacts_root_sha256.clone(),
            committed_at_unix: 100,
            candidate_access_profile_sha256: candidate_access_profile_sha256.clone(),
            reveal_protocol_sha256: reveal_protocol_sha256.clone(),
            anchor_log_id: anchor_log_id.clone(),
            anchor_entry_id: anchor_entry_id.clone(),
        };
        let mut fixture = Fixture {
            receipt: EvaluationManifestCommitmentReceiptV1 {
                schema_version: EVALUATION_MANIFEST_COMMITMENT_RECEIPT_SCHEMA_V1.to_owned(),
                payload,
                attestation: EvaluationManifestCommitmentAttestationV1 {
                    witness_profile_sha256: witness_profile_sha256.clone(),
                    verifier_profile_sha256: verifier_profile_sha256.clone(),
                    key_id,
                    issued_at_unix: 110,
                    payload_sha256: digest('3'),
                    attestation_sha256: digest('4'),
                    signature_base64: String::new(),
                },
            },
            verifier,
            signer,
            suite_id,
            suite_version,
            episode_manifest_root_sha256,
            sealed_evaluation_manifest_sha256,
            arm_assignment_root_sha256,
            pre_registration_root_sha256,
            sealed_artifacts_root_sha256,
            candidate_access_profile_sha256,
            reveal_protocol_sha256,
            anchor_log_id,
            anchor_entry_id,
            witness_profile_sha256,
            verifier_profile_sha256,
        };
        fixture.resign();
        fixture
    }

    #[test]
    fn valid_witness_requires_signature_and_log_inclusion() {
        let fixture = fixture();
        let expected = fixture.expected();
        let verified = fixture.verify(250, &expected, &AcceptLog).unwrap();
        assert_eq!(verified.committed_at_unix(), 100);
        assert_eq!(verified.suite_id(), "mace-test");
        assert_eq!(verified.sealed_evaluation_manifest_sha256(), digest('c'));
        assert_eq!(verified.receipt_sha256().len(), 64);
    }

    #[test]
    fn commit_after_candidate_access_is_rejected() {
        let fixture = fixture();
        let mut expected = fixture.expected();
        expected.candidate_access_started_unix = 100;
        // Keep the test binding independent of the receipt's committed time.
        assert_eq!(
            fixture.verify(250, &expected, &AcceptLog),
            Err(EvaluationManifestCommitmentReceiptRejectV1::WitnessCommitmentAfterCandidateAccess)
        );
        expected.candidate_access_started_unix = 150;
    }

    #[test]
    fn commit_after_episode_start_is_rejected() {
        let fixture = fixture();
        let mut expected = fixture.expected();
        expected.first_episode_started_unix = 100;
        assert_eq!(
            fixture.verify(250, &expected, &AcceptLog),
            Err(EvaluationManifestCommitmentReceiptRejectV1::WitnessCommitmentAfterEpisodeStart)
        );
    }

    #[test]
    fn invalid_append_only_log_proof_is_rejected() {
        let fixture = fixture();
        assert_eq!(
            fixture.verify(250, &fixture.expected, &RejectLog),
            Err(EvaluationManifestCommitmentReceiptRejectV1::LogVerificationRejected(
                ManifestWitnessLogRejectV1::InclusionProofInvalid
            ))
        );
    }

    #[test]
    fn commitment_metadata_cannot_be_rebound_without_resigning() {
        let mut fixture = fixture();
        fixture.receipt.attestation.issued_at_unix = 111;
        assert_eq!(
            fixture.verify(250, &fixture.expected(), &AcceptLog),
            Err(EvaluationManifestCommitmentReceiptRejectV1::AttestationDigestMismatch)
        );
    }

    #[test]
    fn wrong_manifest_root_is_rejected() {
        let fixture = fixture();
        let wrong_root = digest('8');
        let mut expected = fixture.expected();
        expected.episode_manifest_root_sha256 = &wrong_root;
        assert_eq!(
            fixture.verify(250, &expected, &AcceptLog),
            Err(EvaluationManifestCommitmentReceiptRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn witness_issue_time_must_not_be_in_the_future() {
        let mut fixture = fixture();
        fixture.receipt.attestation.issued_at_unix = 300;
        // Re-sign so the only expected failure is the trusted-time boundary.
        fixture.resign();
        assert_eq!(
            fixture.verify(250, &fixture.expected, &AcceptLog),
            Err(EvaluationManifestCommitmentReceiptRejectV1::WitnessIssuedInFuture)
        );
    }

    #[test]
    fn duplicate_json_fields_are_rejected() {
        let fixture = fixture();
        let wire = serde_json::to_string(&fixture.receipt).unwrap();
        let duplicated = wire.replacen(
            "{\"schema_version\":\"mace_evaluation_manifest_commitment_receipt_v1\",",
            "{\"schema_version\":\"mace_evaluation_manifest_commitment_receipt_v1\",\"schema_version\":\"mace_evaluation_manifest_commitment_receipt_v1\",",
            1,
        );
        assert_eq!(
            parse_and_verify_evaluation_manifest_commitment_receipt_v1(
                duplicated.as_bytes(),
                &fixture.expected(),
                250,
                &fixture.verifier,
                &AcceptLog,
            ),
            Err(EvaluationManifestCommitmentReceiptRejectV1::InvalidJson)
        );
    }
}
