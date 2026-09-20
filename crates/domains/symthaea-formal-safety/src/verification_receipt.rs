// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed receipts for externally performed verification.
//!
//! A receipt records what an external verifier claims to have checked. It is
//! deliberately **not** proof authority by itself. In particular:
//!
//! - a digest match does not authenticate an issuer;
//! - a signature does not establish that the signer was authorized;
//! - an authorized signer does not widen the receipt's property scope;
//! - a receipt that is structurally valid has not thereby been cryptographically
//!   verified;
//! - an attested receipt must still be admitted under an explicit trust/profile
//!   policy before it can replace local verification.
//!
//! Cryptographic verification and authorization belong in adapters such as
//! Xenia. Forge/Nixward/Spore may provide provenance or environment evidence.
//! This module only defines the evidence envelope that Symthaea can reason over.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::assurance_case::{EvidenceMethod, SubjectRef};

pub const VERIFICATION_RECEIPT_SCHEMA_VERSION: u16 = 1;
pub const VERIFICATION_RECEIPT_TRANSCRIPT_DOMAIN: &str = "symthaea-verification-receipt-v1";
pub const VERIFICATION_RECEIPT_ATTESTATION_DOMAIN: &str =
    "symthaea-verification-receipt-attestation-v1";

/// Defensive structural bounds. Receipts are evidence metadata, not bulk
/// artifact transport; large proof/test outputs belong behind digests.
pub const MAX_RECEIPT_STRING_BYTES: usize = 4 * 1024;
pub const MAX_RECEIPT_PROPERTIES: usize = 128;
pub const MAX_RECEIPT_SIGNATURE_BYTES: usize = 16 * 1024;

/// Exact identity of the tool that produced a verification conclusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierIdentity {
    /// Stable logical verifier name, e.g. `z3`, `lean4`, `cargo-test`.
    pub verifier_id: String,
    /// Exact reported tool version.
    pub version: String,
    /// Digest of the executable/package/closure used for verification.
    pub artifact_digest: String,
}

impl VerifierIdentity {
    pub fn validate(&self) -> bool {
        bounded_nonempty(&self.verifier_id)
            && bounded_nonempty(&self.version)
            && bounded_nonempty(&self.artifact_digest)
    }
}

/// Outcome reported by the verifier for the listed property scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationConclusion {
    /// Evidence supports the listed properties under the recorded assumptions
    /// and environment.
    Supports,
    /// Evidence contradicts/refutes the listed properties.
    Refutes,
    /// The verifier did not establish either support or refutation.
    Inconclusive,
}

impl VerificationConclusion {
    const fn transcript_label(self) -> &'static str {
        match self {
            Self::Supports => "supports",
            Self::Refutes => "refutes",
            Self::Inconclusive => "inconclusive",
        }
    }
}

/// A typed, scope-limited record of one external verification operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReceipt {
    pub schema_version: u16,
    pub receipt_id: Uuid,
    pub subject: SubjectRef,
    pub method: EvidenceMethod,
    pub conclusion: VerificationConclusion,
    /// Exact claim/property identifiers to which the conclusion applies.
    pub properties: Vec<String>,
    /// Explicit limitations that downstream reasoning must preserve.
    pub does_not_establish: Vec<String>,
    /// Exact verifier identity and artifact provenance.
    pub verifier: VerifierIdentity,
    /// Digest of the environment/closure/configuration in which verification ran.
    pub environment_digest: String,
    /// Digest of the exact verifier input (query, source, test plan, model, etc.).
    pub input_digest: String,
    /// Digest of the raw verifier output, proof object, test report, or transcript.
    pub output_digest: String,
    /// Optional digest of explicit assumptions/axioms used by the verifier.
    pub assumptions_digest: Option<String>,
    pub issued_unix_s: u64,
    pub valid_until_unix_s: Option<u64>,
}

impl VerificationReceipt {
    pub fn validate(&self) -> bool {
        if self.schema_version != VERIFICATION_RECEIPT_SCHEMA_VERSION
            || !self.subject.validate()
            || !bounded_nonempty(&self.subject.namespace)
            || !bounded_nonempty(&self.subject.subject_id)
            || !bounded_nonempty(&self.subject.digest)
            || !self.verifier.validate()
            || !bounded_nonempty(&self.environment_digest)
            || !bounded_nonempty(&self.input_digest)
            || !bounded_nonempty(&self.output_digest)
            || self.properties.is_empty()
            || self.properties.len() > MAX_RECEIPT_PROPERTIES
            || self.does_not_establish.len() > MAX_RECEIPT_PROPERTIES
            || !all_bounded_nonempty_unique(&self.properties)
            || !all_bounded_nonempty_unique(&self.does_not_establish)
            || self
                .assumptions_digest
                .as_ref()
                .is_some_and(|digest| !bounded_nonempty(digest))
            || self
                .valid_until_unix_s
                .is_some_and(|deadline| deadline < self.issued_unix_s)
        {
            return false;
        }

        let properties: BTreeSet<_> = self.properties.iter().collect();
        let limitations: BTreeSet<_> = self.does_not_establish.iter().collect();
        properties.is_disjoint(&limitations)
    }

    pub fn is_expired_at(&self, now_unix_s: u64) -> bool {
        self.valid_until_unix_s
            .is_some_and(|deadline| now_unix_s > deadline)
    }

    /// True only when this receipt reports a positive conclusion for `property`.
    ///
    /// This is a statement about the *receipt contents*, not a trust decision.
    /// Callers must still independently authenticate/admit the receipt.
    pub fn claims_support_for(&self, property: &str) -> bool {
        self.validate()
            && self.conclusion == VerificationConclusion::Supports
            && self.properties.iter().any(|value| value == property)
    }

    /// True only when this receipt reports a refuting conclusion for `property`.
    /// This does not authenticate the issuer or establish local proof authority.
    pub fn claims_refutation_of(&self, property: &str) -> bool {
        self.validate()
            && self.conclusion == VerificationConclusion::Refutes
            && self.properties.iter().any(|value| value == property)
    }

    /// Deterministic, domain-separated bytes for external hashing/signing.
    ///
    /// The encoding is deliberately independent of serde/JSON map ordering. Set-like
    /// property fields are sorted before encoding so semantically identical receipts
    /// yield identical transcripts. This method performs **no** hashing or signing.
    pub fn canonical_transcript(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }

        let mut out = Vec::new();
        push_field(&mut out, "domain", VERIFICATION_RECEIPT_TRANSCRIPT_DOMAIN.as_bytes());
        push_field(&mut out, "schema_version", &self.schema_version.to_be_bytes());
        push_field(&mut out, "receipt_id", self.receipt_id.as_bytes());
        push_field(&mut out, "subject.namespace", self.subject.namespace.as_bytes());
        push_field(&mut out, "subject.id", self.subject.subject_id.as_bytes());
        push_field(&mut out, "subject.digest", self.subject.digest.as_bytes());
        push_field(&mut out, "method", method_label(self.method).as_bytes());
        push_field(
            &mut out,
            "conclusion",
            self.conclusion.transcript_label().as_bytes(),
        );

        let mut properties = self.properties.iter().map(String::as_str).collect::<Vec<_>>();
        properties.sort_unstable();
        for property in properties {
            push_field(&mut out, "property", property.as_bytes());
        }

        let mut limitations = self
            .does_not_establish
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        limitations.sort_unstable();
        for limitation in limitations {
            push_field(&mut out, "does_not_establish", limitation.as_bytes());
        }

        push_field(&mut out, "verifier.id", self.verifier.verifier_id.as_bytes());
        push_field(&mut out, "verifier.version", self.verifier.version.as_bytes());
        push_field(
            &mut out,
            "verifier.artifact_digest",
            self.verifier.artifact_digest.as_bytes(),
        );
        push_field(
            &mut out,
            "environment_digest",
            self.environment_digest.as_bytes(),
        );
        push_field(&mut out, "input_digest", self.input_digest.as_bytes());
        push_field(&mut out, "output_digest", self.output_digest.as_bytes());
        match &self.assumptions_digest {
            Some(digest) => {
                push_field(&mut out, "assumptions.present", &[1]);
                push_field(&mut out, "assumptions.digest", digest.as_bytes());
            }
            None => push_field(&mut out, "assumptions.present", &[0]),
        }
        push_field(&mut out, "issued_unix_s", &self.issued_unix_s.to_be_bytes());
        match self.valid_until_unix_s {
            Some(deadline) => {
                push_field(&mut out, "valid_until.present", &[1]);
                push_field(&mut out, "valid_until_unix_s", &deadline.to_be_bytes());
            }
            None => push_field(&mut out, "valid_until.present", &[0]),
        }

        Some(out)
    }
}

/// Signature/attestation bytes associated with a receipt.
///
/// Structural validity means only that the envelope is well-formed. It does not
/// mean the signature has been checked or that the signer was authorized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptAttestation {
    /// Domain separator/transcript version used by the external signer.
    pub domain: String,
    pub signer_id: String,
    pub key_id: String,
    pub algorithm: String,
    /// Digest of `VerificationReceipt::canonical_transcript()` that was signed.
    pub payload_digest: String,
    /// Signature bytes as supplied by the external signing system.
    pub signature: Vec<u8>,
}

impl ReceiptAttestation {
    pub fn validate_structure(&self) -> bool {
        self.domain == VERIFICATION_RECEIPT_ATTESTATION_DOMAIN
            && bounded_nonempty(&self.signer_id)
            && bounded_nonempty(&self.key_id)
            && bounded_nonempty(&self.algorithm)
            && bounded_nonempty(&self.payload_digest)
            && !self.signature.is_empty()
            && self.signature.len() <= MAX_RECEIPT_SIGNATURE_BYTES
    }
}

/// Receipt plus an optional external attestation.
///
/// Presence of `attestation` is intentionally not exposed as an `is_verified`
/// boolean. Consumers must route it through an independently configured
/// verifier/trust policy and record that admission separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestedVerificationReceipt {
    pub receipt: VerificationReceipt,
    pub attestation: Option<ReceiptAttestation>,
}

impl AttestedVerificationReceipt {
    pub fn validate_structure(&self) -> bool {
        self.receipt.validate()
            && self
                .attestation
                .as_ref()
                .is_none_or(ReceiptAttestation::validate_structure)
    }

    /// Whether an attestation envelope is present. This is intentionally named
    /// `has_attestation`, not `is_verified`.
    pub fn has_attestation(&self) -> bool {
        self.attestation.is_some()
    }
}

fn bounded_nonempty(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_RECEIPT_STRING_BYTES
}

fn all_bounded_nonempty_unique(values: &[String]) -> bool {
    values.iter().all(|value| bounded_nonempty(value))
        && BTreeSet::<_>::from_iter(values.iter()).len() == values.len()
}

fn push_field(out: &mut Vec<u8>, label: &str, value: &[u8]) {
    let label_len = u16::try_from(label.len()).expect("static transcript label length fits u16");
    let value_len = u32::try_from(value.len()).expect("validated receipt field length fits u32");
    out.extend_from_slice(&label_len.to_be_bytes());
    out.extend_from_slice(label.as_bytes());
    out.extend_from_slice(&value_len.to_be_bytes());
    out.extend_from_slice(value);
}

const fn method_label(method: EvidenceMethod) -> &'static str {
    match method {
        EvidenceMethod::Inspection => "inspection",
        EvidenceMethod::StaticAnalysis => "static_analysis",
        EvidenceMethod::ExampleTest => "example_test",
        EvidenceMethod::PropertyTest => "property_test",
        EvidenceMethod::FuzzCampaign => "fuzz_campaign",
        EvidenceMethod::AdversarialCampaign => "adversarial_campaign",
        EvidenceMethod::Simulation => "simulation",
        EvidenceMethod::RuntimeObservation => "runtime_observation",
        EvidenceMethod::ModelCheck => "model_check",
        EvidenceMethod::FormalProof => "formal_proof",
        EvidenceMethod::StandardReference => "standard_reference",
        EvidenceMethod::SignedAttestation => "signed_attestation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> SubjectRef {
        SubjectRef {
            namespace: "symthaea.source".into(),
            subject_id: "crate-a".into(),
            digest: "sha256:subject".into(),
        }
    }

    fn receipt() -> VerificationReceipt {
        VerificationReceipt {
            schema_version: VERIFICATION_RECEIPT_SCHEMA_VERSION,
            receipt_id: Uuid::from_bytes([0x11; 16]),
            subject: subject(),
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

    #[test]
    fn valid_receipt_is_scope_limited() {
        let value = receipt();
        assert!(value.validate());
        assert!(value.claims_support_for("property.memory_safe"));
        assert!(!value.claims_support_for("property.side_channel_free"));
        assert!(!value.claims_refutation_of("property.memory_safe"));
    }

    #[test]
    fn overlapping_property_and_limitation_is_invalid() {
        let mut value = receipt();
        value
            .does_not_establish
            .push("property.memory_safe".into());
        assert!(!value.validate());
    }

    #[test]
    fn expiry_is_not_silently_ignored() {
        let value = receipt();
        assert!(!value.is_expired_at(200));
        assert!(value.is_expired_at(201));

        let mut impossible = value;
        impossible.valid_until_unix_s = Some(99);
        assert!(!impossible.validate());
    }

    #[test]
    fn inconclusive_receipt_never_claims_support_or_refutation() {
        let mut value = receipt();
        value.conclusion = VerificationConclusion::Inconclusive;
        assert!(value.validate());
        assert!(!value.claims_support_for("property.memory_safe"));
        assert!(!value.claims_refutation_of("property.memory_safe"));
    }

    #[test]
    fn canonical_transcript_is_order_independent_for_set_like_fields() {
        let mut a = receipt();
        a.properties = vec!["property.b".into(), "property.a".into()];
        a.does_not_establish = vec!["property.y".into(), "property.x".into()];

        let mut b = a.clone();
        b.properties.reverse();
        b.does_not_establish.reverse();

        let transcript_a = a.canonical_transcript().unwrap();
        let transcript_b = b.canonical_transcript().unwrap();
        assert_eq!(transcript_a, transcript_b);
        assert!(transcript_a.windows(VERIFICATION_RECEIPT_TRANSCRIPT_DOMAIN.len()).any(|w| {
            w == VERIFICATION_RECEIPT_TRANSCRIPT_DOMAIN.as_bytes()
        }));
    }

    #[test]
    fn oversized_fields_fail_before_transcript_allocation() {
        let mut value = receipt();
        value.properties[0] = "x".repeat(MAX_RECEIPT_STRING_BYTES + 1);
        assert!(!value.validate());
        assert!(value.canonical_transcript().is_none());
    }

    #[test]
    fn attestation_presence_is_not_verification() {
        let value = AttestedVerificationReceipt {
            receipt: receipt(),
            attestation: Some(ReceiptAttestation {
                domain: VERIFICATION_RECEIPT_ATTESTATION_DOMAIN.into(),
                signer_id: "operator-a".into(),
                key_id: "key-1".into(),
                algorithm: "example-signature".into(),
                payload_digest: "sha256:payload".into(),
                signature: vec![1, 2, 3],
            }),
        };

        assert!(value.validate_structure());
        assert!(value.has_attestation());
        // There is deliberately no `is_verified()` API: cryptographic validity,
        // trust, and authorization must be established externally.
    }

    #[test]
    fn malformed_or_wrong_domain_attestation_is_rejected_structurally() {
        let mut value = AttestedVerificationReceipt {
            receipt: receipt(),
            attestation: Some(ReceiptAttestation {
                domain: "wrong-domain".into(),
                signer_id: "operator-a".into(),
                key_id: "key-1".into(),
                algorithm: "example-signature".into(),
                payload_digest: "sha256:payload".into(),
                signature: vec![1],
            }),
        };
        assert!(!value.validate_structure());

        value.attestation.as_mut().unwrap().domain = VERIFICATION_RECEIPT_ATTESTATION_DOMAIN.into();
        value.attestation.as_mut().unwrap().signature =
            vec![0; MAX_RECEIPT_SIGNATURE_BYTES + 1];
        assert!(!value.validate_structure());
    }
}