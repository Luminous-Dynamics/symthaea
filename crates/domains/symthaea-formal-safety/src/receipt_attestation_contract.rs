// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical signing contract for verification-receipt attestations.
//!
//! This module freezes the exact metadata and bytes that an external authority
//! such as Xenia signs for a Symthaea verification receipt. It does not perform
//! signature verification, enrollment lookup, authorization, policy admission,
//! or assurance qualification.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

use crate::verification_receipt::{
    VerificationReceipt, MAX_RECEIPT_STRING_BYTES, VERIFICATION_RECEIPT_SCHEMA_VERSION,
};

/// Current attestation transcript protocol version.
pub const RECEIPT_ATTESTATION_PROTOCOL_VERSION_V1: u16 = 1;
/// Domain separator for the bytes an external signer must sign.
pub const RECEIPT_ATTESTATION_TRANSCRIPT_DOMAIN_V1: &[u8] =
    b"symthaea-verification-receipt-attestation-v1\0";
/// Digest size fixed by the v1 contract.
pub const SHA256_DIGEST_BYTES: usize = 32;
/// Ed25519 signature size fixed by RFC 8032.
pub const ED25519_SIGNATURE_BYTES: usize = 64;
/// ML-DSA-65 signature size fixed by FIPS 204 and Xenia's current implementation.
pub const ML_DSA_65_SIGNATURE_BYTES: usize = 3309;

/// Digest algorithm fixed by the v1 receipt-attestation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReceiptDigestAlgorithmV1 {
    /// SHA-256 over exact canonical receipt bytes.
    #[serde(rename = "sha256")]
    Sha256,
}

impl ReceiptDigestAlgorithmV1 {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
        }
    }
}

/// Algorithm-qualified digest of exact canonical receipt bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptDigestV1 {
    pub algorithm: ReceiptDigestAlgorithmV1,
    pub bytes: [u8; SHA256_DIGEST_BYTES],
}

impl ReceiptDigestV1 {
    /// Hash the exact canonical transcript of a structurally valid receipt.
    pub fn from_receipt(receipt: &VerificationReceipt) -> Option<Self> {
        let transcript = receipt.canonical_transcript()?;
        let bytes: [u8; SHA256_DIGEST_BYTES] = Sha256::digest(transcript).into();
        Some(Self {
            algorithm: ReceiptDigestAlgorithmV1::Sha256,
            bytes,
        })
    }
}

/// Cryptographic suite accepted by attestation protocol v1.
///
/// The wire spelling intentionally matches Xenia's existing
/// `XeniaHybridSuite::Ed25519MlDsa65V1` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReceiptAttestationSuiteV1 {
    /// Both Ed25519 and ML-DSA-65 authenticate the same canonical transcript.
    Ed25519MlDsa65V1,
}

impl ReceiptAttestationSuiteV1 {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Ed25519MlDsa65V1 => "Ed25519MlDsa65V1",
        }
    }
}

/// Exact signature bundle for the v1 hybrid suite.
///
/// The two signatures independently authenticate the same canonical transcript.
/// Their bytes are not themselves part of that transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaHybridSignatureBundleV1 {
    pub ed25519: [u8; ED25519_SIGNATURE_BYTES],
    pub ml_dsa_65: Vec<u8>,
}

impl XeniaHybridSignatureBundleV1 {
    pub fn validate_structure(&self) -> bool {
        self.ml_dsa_65.len() == ML_DSA_65_SIGNATURE_BYTES
    }
}

/// Portable attestation envelope whose metadata is itself covered by the
/// canonical signing transcript.
///
/// `signatures` is deliberately excluded from `canonical_signing_transcript()`:
/// those are the signatures *over* the transcript. Presence of this envelope
/// still does not mean either signature is valid or the signer is authorized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptAttestationContractV1 {
    pub protocol_version: u16,
    pub receipt_schema_version: u16,
    pub receipt_id: Uuid,
    pub payload_digest: ReceiptDigestV1,
    pub provider_namespace: String,
    pub signer_id: String,
    /// Stable key identifier or key-lineage commitment supplied by the authority
    /// provider. Xenia should use its exact current hybrid lineage commitment.
    pub key_lineage_commitment: String,
    pub suite: ReceiptAttestationSuiteV1,
    pub signatures: XeniaHybridSignatureBundleV1,
}

impl ReceiptAttestationContractV1 {
    /// Construct an envelope bound to the exact canonical bytes of `receipt`.
    pub fn new_for_receipt(
        receipt: &VerificationReceipt,
        provider_namespace: impl Into<String>,
        signer_id: impl Into<String>,
        key_lineage_commitment: impl Into<String>,
        suite: ReceiptAttestationSuiteV1,
        signatures: XeniaHybridSignatureBundleV1,
    ) -> Option<Self> {
        let value = Self {
            protocol_version: RECEIPT_ATTESTATION_PROTOCOL_VERSION_V1,
            receipt_schema_version: receipt.schema_version,
            receipt_id: receipt.receipt_id,
            payload_digest: ReceiptDigestV1::from_receipt(receipt)?,
            provider_namespace: provider_namespace.into(),
            signer_id: signer_id.into(),
            key_lineage_commitment: key_lineage_commitment.into(),
            suite,
            signatures,
        };
        value.validate_structure().then_some(value)
    }

    /// Structural validity only. This does not check either signature.
    pub fn validate_structure(&self) -> bool {
        self.protocol_version == RECEIPT_ATTESTATION_PROTOCOL_VERSION_V1
            && self.receipt_schema_version == VERIFICATION_RECEIPT_SCHEMA_VERSION
            && self.receipt_id != Uuid::nil()
            && bounded_nonempty(&self.provider_namespace)
            && bounded_nonempty(&self.signer_id)
            && bounded_nonempty(&self.key_lineage_commitment)
            && self.signatures.validate_structure()
    }

    /// Whether this envelope's payload identity exactly matches `receipt`.
    ///
    /// This checks only canonical-byte binding. It does not authenticate the
    /// envelope or authorize its signer.
    pub fn matches_receipt(&self, receipt: &VerificationReceipt) -> bool {
        self.validate_structure()
            && receipt.validate()
            && self.receipt_schema_version == receipt.schema_version
            && self.receipt_id == receipt.receipt_id
            && ReceiptDigestV1::from_receipt(receipt).as_ref() == Some(&self.payload_digest)
    }

    /// Exact, domain-separated bytes both external signatures must cover.
    ///
    /// All authority-bearing metadata is included, so signer, provider,
    /// key-lineage, suite, receipt-id, or payload substitution changes the bytes.
    pub fn canonical_signing_transcript(&self) -> Option<Vec<u8>> {
        if !self.validate_structure() {
            return None;
        }

        let mut out = Vec::new();
        out.extend_from_slice(RECEIPT_ATTESTATION_TRANSCRIPT_DOMAIN_V1);
        push_field(
            &mut out,
            "protocol_version",
            &self.protocol_version.to_be_bytes(),
        );
        push_field(
            &mut out,
            "receipt_schema_version",
            &self.receipt_schema_version.to_be_bytes(),
        );
        push_field(&mut out, "receipt_id", self.receipt_id.as_bytes());
        push_field(
            &mut out,
            "payload_digest.algorithm",
            self.payload_digest.algorithm.id().as_bytes(),
        );
        push_field(
            &mut out,
            "payload_digest.bytes",
            &self.payload_digest.bytes,
        );
        push_field(
            &mut out,
            "provider_namespace",
            self.provider_namespace.as_bytes(),
        );
        push_field(&mut out, "signer_id", self.signer_id.as_bytes());
        push_field(
            &mut out,
            "key_lineage_commitment",
            self.key_lineage_commitment.as_bytes(),
        );
        push_field(&mut out, "suite", self.suite.id().as_bytes());
        Some(out)
    }

    /// Canonical signing transcript only when the payload is bound to `receipt`.
    pub fn canonical_signing_transcript_for(
        &self,
        receipt: &VerificationReceipt,
    ) -> Option<Vec<u8>> {
        if self.matches_receipt(receipt) {
            self.canonical_signing_transcript()
        } else {
            None
        }
    }
}

fn bounded_nonempty(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_RECEIPT_STRING_BYTES
}

fn push_field(out: &mut Vec<u8>, label: &str, value: &[u8]) {
    let label_len = u16::try_from(label.len()).expect("static transcript label length fits u16");
    let value_len = u32::try_from(value.len()).expect("validated attestation field fits u32");
    out.extend_from_slice(&label_len.to_be_bytes());
    out.extend_from_slice(label.as_bytes());
    out.extend_from_slice(&value_len.to_be_bytes());
    out.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_case::{EvidenceMethod, SubjectRef};
    use crate::verification_receipt::{
        VerificationConclusion, VerifierIdentity, VERIFICATION_RECEIPT_SCHEMA_VERSION,
    };

    fn receipt() -> VerificationReceipt {
        VerificationReceipt {
            schema_version: VERIFICATION_RECEIPT_SCHEMA_VERSION,
            receipt_id: Uuid::from_bytes([0x11; 16]),
            subject: SubjectRef {
                namespace: "symthaea.source".into(),
                subject_id: "crate-a".into(),
                digest: "sha256:subject".into(),
            },
            method: EvidenceMethod::FormalProof,
            conclusion: VerificationConclusion::Supports,
            properties: vec!["property.memory_safe".into()],
            does_not_establish: vec!["property.side_channel_free".into()],
            verifier: VerifierIdentity {
                verifier_id: "verifier.example".into(),
                version: "1.2.3".into(),
                artifact_digest: "sha256:verifier".into(),
            },
            environment_digest: "sha256:environment".into(),
            input_digest: "sha256:input".into(),
            output_digest: "sha256:output".into(),
            assumptions_digest: Some("sha256:assumptions".into()),
            issued_unix_s: 100,
            valid_until_unix_s: Some(200),
        }
    }

    fn signatures() -> XeniaHybridSignatureBundleV1 {
        XeniaHybridSignatureBundleV1 {
            ed25519: [0xAA; ED25519_SIGNATURE_BYTES],
            ml_dsa_65: vec![0xBB; ML_DSA_65_SIGNATURE_BYTES],
        }
    }

    fn contract() -> ReceiptAttestationContractV1 {
        ReceiptAttestationContractV1::new_for_receipt(
            &receipt(),
            "luminous-dynamics/xenia",
            "operator:alice",
            "sha256:lineage",
            ReceiptAttestationSuiteV1::Ed25519MlDsa65V1,
            signatures(),
        )
        .unwrap()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn frozen_receipt_and_attestation_vectors() {
        let receipt = receipt();
        let payload = ReceiptDigestV1::from_receipt(&receipt).unwrap();
        assert_eq!(
            hex(&payload.bytes),
            "ce4c2c8873da8c165ad2e60941836c1f92c5cab3d930c83f3761a85de120f6dc"
        );

        let transcript = contract().canonical_signing_transcript_for(&receipt).unwrap();
        let transcript_digest: [u8; 32] = Sha256::digest(&transcript).into();
        assert_eq!(
            hex(&transcript_digest),
            "b77c2d5e2f6c36a7f93fb3883af6d5d0d405e1c9f02e80b97a0eb55d657aeeac"
        );
    }

    #[test]
    fn authority_metadata_substitution_changes_transcript() {
        let baseline = contract().canonical_signing_transcript().unwrap();

        let mut signer = contract();
        signer.signer_id = "operator:bob".into();
        assert_ne!(baseline, signer.canonical_signing_transcript().unwrap());

        let mut provider = contract();
        provider.provider_namespace = "other-provider".into();
        assert_ne!(baseline, provider.canonical_signing_transcript().unwrap());

        let mut lineage = contract();
        lineage.key_lineage_commitment = "sha256:rotated-lineage".into();
        assert_ne!(baseline, lineage.canonical_signing_transcript().unwrap());
    }

    #[test]
    fn receipt_substitution_breaks_payload_binding() {
        let contract = contract();
        let mut other = receipt();
        other.output_digest = "sha256:different-output".into();

        assert!(!contract.matches_receipt(&other));
        assert!(contract.canonical_signing_transcript_for(&other).is_none());
    }

    #[test]
    fn signature_bytes_are_not_part_of_the_signed_message() {
        let a = contract();
        let mut b = a.clone();
        b.signatures.ed25519 = [0xCC; ED25519_SIGNATURE_BYTES];
        b.signatures.ml_dsa_65 = vec![0xDD; ML_DSA_65_SIGNATURE_BYTES];

        assert_ne!(a.signatures, b.signatures);
        assert_eq!(
            a.canonical_signing_transcript().unwrap(),
            b.canonical_signing_transcript().unwrap()
        );
    }

    #[test]
    fn unsupported_suite_fails_closed_at_deserialization() {
        let json = r#""UnknownSuite""#;
        assert!(serde_json::from_str::<ReceiptAttestationSuiteV1>(json).is_err());
    }

    #[test]
    fn malformed_authority_labels_and_signature_bundle_fail_structurally() {
        let mut value = contract();
        value.provider_namespace = "   ".into();
        assert!(!value.validate_structure());

        let mut value = contract();
        value.signatures.ml_dsa_65.pop();
        assert!(!value.validate_structure());
    }
}
