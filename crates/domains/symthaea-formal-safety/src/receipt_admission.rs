// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit admission boundary for externally produced verification receipts.
//!
//! Authorization and admission are represented as replay-resistant grants bound
//! to the exact receipt and subject. This module deliberately returns admitted
//! property evidence, not a qualified claim or action capability.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::assurance_case::{EvidenceMethod, SubjectRef};
use crate::assurance_state::PropertyState;
use crate::verification_receipt::{AttestedVerificationReceipt, VerificationConclusion};

pub const MAX_ADMISSION_GRANT_STRING_BYTES: usize = 4 * 1024;
pub const MAX_ADMISSION_GRANT_PROPERTIES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptAdmissionError {
    EmptyProfileId,
    MalformedReceiptEnvelope,
    MissingAttestation,
    ReceiptExpired,
    InvalidSignerAuthorizationGrant,
    SignerAuthorizationExpired,
    SignerAuthorizationReceiptMismatch,
    SignerAuthorizationSubjectMismatch,
    SignerAuthorizationAttestationMismatch,
    InvalidEvidenceAdmissionGrant,
    EvidenceAdmissionExpired,
    EvidenceAdmissionReceiptMismatch,
    EvidenceAdmissionSubjectMismatch,
    AdmissionProfileMismatch {
        admitted_profile_id: String,
        requested_profile_id: String,
    },
    AdmissionPropertyOutOfReceiptScope {
        property: String,
    },
}

/// Authorization evidence produced by an external identity/authority system.
///
/// This is intentionally bound to the exact receipt, subject, signer/key,
/// algorithm, and attestation payload digest so an authorization decision cannot
/// be replayed against an unrelated receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignerAuthorizationGrant {
    pub grant_id: Uuid,
    pub receipt_id: Uuid,
    pub subject: SubjectRef,
    pub signer_id: String,
    pub key_id: String,
    pub algorithm: String,
    pub attestation_payload_digest: String,
    pub authority_scope: String,
    pub issued_unix_s: u64,
    pub valid_until_unix_s: Option<u64>,
}

impl SignerAuthorizationGrant {
    pub fn validate(&self) -> bool {
        self.grant_id != Uuid::nil()
            && self.receipt_id != Uuid::nil()
            && self.subject.validate()
            && bounded_nonempty(&self.signer_id)
            && bounded_nonempty(&self.key_id)
            && bounded_nonempty(&self.algorithm)
            && bounded_nonempty(&self.attestation_payload_digest)
            && bounded_nonempty(&self.authority_scope)
            && self
                .valid_until_unix_s
                .is_none_or(|deadline| deadline >= self.issued_unix_s)
    }

    pub fn is_expired_at(&self, now_unix_s: u64) -> bool {
        self.valid_until_unix_s
            .is_some_and(|deadline| now_unix_s > deadline)
    }
}

/// Profile-local permission to admit a specific subset of one receipt's
/// property conclusions as evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAdmissionGrant {
    pub grant_id: Uuid,
    pub receipt_id: Uuid,
    pub subject: SubjectRef,
    pub profile_id: String,
    pub properties: Vec<String>,
    pub issued_unix_s: u64,
    pub valid_until_unix_s: Option<u64>,
}

impl EvidenceAdmissionGrant {
    pub fn validate(&self) -> bool {
        self.grant_id != Uuid::nil()
            && self.receipt_id != Uuid::nil()
            && self.subject.validate()
            && bounded_nonempty(&self.profile_id)
            && !self.properties.is_empty()
            && self.properties.len() <= MAX_ADMISSION_GRANT_PROPERTIES
            && self.properties.iter().all(|property| bounded_nonempty(property))
            && BTreeSet::<_>::from_iter(self.properties.iter()).len() == self.properties.len()
            && self
                .valid_until_unix_s
                .is_none_or(|deadline| deadline >= self.issued_unix_s)
    }

    pub fn is_expired_at(&self, now_unix_s: u64) -> bool {
        self.valid_until_unix_s
            .is_some_and(|deadline| now_unix_s > deadline)
    }
}

/// One property conclusion admitted from an external verifier receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmittedPropertyEvidence {
    pub receipt_id: Uuid,
    pub subject: SubjectRef,
    pub profile_id: String,
    pub signer_authorization_grant_id: Uuid,
    pub evidence_admission_grant_id: Uuid,
    pub authority_scope: String,
    pub method: EvidenceMethod,
    pub property: String,
    pub state: PropertyState,
    pub verifier_id: String,
    pub verifier_version: String,
    pub verifier_artifact_digest: String,
    pub environment_digest: String,
    pub input_digest: String,
    pub output_digest: String,
    pub assumptions_digest: Option<String>,
    /// Earliest expiry across the receipt, signer authorization, and policy
    /// admission. `None` means none of those inputs declared an expiry.
    pub valid_until_unix_s: Option<u64>,
}

/// Profile-local admission result for one external receipt.
///
/// This type intentionally contains no qualification or action-authorization
/// field. Consumers must still evaluate assurance claims separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmittedVerificationEvidence {
    pub receipt_id: Uuid,
    pub subject: SubjectRef,
    pub profile_id: String,
    pub signer_authorization_grant_id: Uuid,
    pub evidence_admission_grant_id: Uuid,
    pub authority_scope: String,
    pub method: EvidenceMethod,
    pub receipt_issued_unix_s: u64,
    pub admitted_at_unix_s: u64,
    /// Earliest expiry across every authority/evidence input used for admission.
    pub valid_until_unix_s: Option<u64>,
    pub properties: Vec<AdmittedPropertyEvidence>,
    pub does_not_establish: Vec<String>,
}

impl AdmittedVerificationEvidence {
    pub fn property_state(&self, property: &str) -> Option<PropertyState> {
        self.properties
            .iter()
            .find(|value| value.property == property)
            .map(|value| value.state)
    }

    pub fn is_expired_at(&self, now_unix_s: u64) -> bool {
        self.valid_until_unix_s
            .is_some_and(|deadline| now_unix_s > deadline)
    }
}

/// Admit an externally attested verifier receipt as profile-local evidence.
///
/// This function verifies *binding and policy state*, not cryptography. The
/// caller must obtain `signer_authorization` from an independently trusted
/// identity/authority adapter (for example Xenia) and `evidence_admission` from
/// an independently evaluated assurance policy.
///
/// Success still does **not** qualify a claim or authorize an action.
pub fn admit_verification_receipt(
    envelope: &AttestedVerificationReceipt,
    signer_authorization: &SignerAuthorizationGrant,
    evidence_admission: &EvidenceAdmissionGrant,
    requested_profile_id: &str,
    now_unix_s: u64,
) -> Result<AdmittedVerificationEvidence, ReceiptAdmissionError> {
    if requested_profile_id.trim().is_empty() {
        return Err(ReceiptAdmissionError::EmptyProfileId);
    }
    if !envelope.validate_structure() {
        return Err(ReceiptAdmissionError::MalformedReceiptEnvelope);
    }
    let Some(attestation) = envelope.attestation.as_ref() else {
        return Err(ReceiptAdmissionError::MissingAttestation);
    };
    if envelope.receipt.is_expired_at(now_unix_s) {
        return Err(ReceiptAdmissionError::ReceiptExpired);
    }

    if !signer_authorization.validate() {
        return Err(ReceiptAdmissionError::InvalidSignerAuthorizationGrant);
    }
    if signer_authorization.is_expired_at(now_unix_s) {
        return Err(ReceiptAdmissionError::SignerAuthorizationExpired);
    }
    if signer_authorization.receipt_id != envelope.receipt.receipt_id {
        return Err(ReceiptAdmissionError::SignerAuthorizationReceiptMismatch);
    }
    if signer_authorization.subject != envelope.receipt.subject {
        return Err(ReceiptAdmissionError::SignerAuthorizationSubjectMismatch);
    }
    if signer_authorization.signer_id != attestation.signer_id
        || signer_authorization.key_id != attestation.key_id
        || signer_authorization.algorithm != attestation.algorithm
        || signer_authorization.attestation_payload_digest != attestation.payload_digest
    {
        return Err(ReceiptAdmissionError::SignerAuthorizationAttestationMismatch);
    }

    if !evidence_admission.validate() {
        return Err(ReceiptAdmissionError::InvalidEvidenceAdmissionGrant);
    }
    if evidence_admission.is_expired_at(now_unix_s) {
        return Err(ReceiptAdmissionError::EvidenceAdmissionExpired);
    }
    if evidence_admission.receipt_id != envelope.receipt.receipt_id {
        return Err(ReceiptAdmissionError::EvidenceAdmissionReceiptMismatch);
    }
    if evidence_admission.subject != envelope.receipt.subject {
        return Err(ReceiptAdmissionError::EvidenceAdmissionSubjectMismatch);
    }
    if evidence_admission.profile_id != requested_profile_id {
        return Err(ReceiptAdmissionError::AdmissionProfileMismatch {
            admitted_profile_id: evidence_admission.profile_id.clone(),
            requested_profile_id: requested_profile_id.to_string(),
        });
    }

    let receipt_properties: BTreeSet<_> = envelope.receipt.properties.iter().collect();
    for property in &evidence_admission.properties {
        if !receipt_properties.contains(property) {
            return Err(ReceiptAdmissionError::AdmissionPropertyOutOfReceiptScope {
                property: property.clone(),
            });
        }
    }

    let property_state = match envelope.receipt.conclusion {
        VerificationConclusion::Supports => PropertyState::Supported,
        VerificationConclusion::Refutes => PropertyState::Refuted,
        VerificationConclusion::Inconclusive => PropertyState::Indeterminate,
    };
    let effective_valid_until = earliest_deadline([
        envelope.receipt.valid_until_unix_s,
        signer_authorization.valid_until_unix_s,
        evidence_admission.valid_until_unix_s,
    ]);

    let properties = evidence_admission
        .properties
        .iter()
        .map(|property| AdmittedPropertyEvidence {
            receipt_id: envelope.receipt.receipt_id,
            subject: envelope.receipt.subject.clone(),
            profile_id: requested_profile_id.to_string(),
            signer_authorization_grant_id: signer_authorization.grant_id,
            evidence_admission_grant_id: evidence_admission.grant_id,
            authority_scope: signer_authorization.authority_scope.clone(),
            method: envelope.receipt.method,
            property: property.clone(),
            state: property_state,
            verifier_id: envelope.receipt.verifier.verifier_id.clone(),
            verifier_version: envelope.receipt.verifier.version.clone(),
            verifier_artifact_digest: envelope.receipt.verifier.artifact_digest.clone(),
            environment_digest: envelope.receipt.environment_digest.clone(),
            input_digest: envelope.receipt.input_digest.clone(),
            output_digest: envelope.receipt.output_digest.clone(),
            assumptions_digest: envelope.receipt.assumptions_digest.clone(),
            valid_until_unix_s: effective_valid_until,
        })
        .collect();

    Ok(AdmittedVerificationEvidence {
        receipt_id: envelope.receipt.receipt_id,
        subject: envelope.receipt.subject.clone(),
        profile_id: requested_profile_id.to_string(),
        signer_authorization_grant_id: signer_authorization.grant_id,
        evidence_admission_grant_id: evidence_admission.grant_id,
        authority_scope: signer_authorization.authority_scope.clone(),
        method: envelope.receipt.method,
        receipt_issued_unix_s: envelope.receipt.issued_unix_s,
        admitted_at_unix_s: now_unix_s,
        valid_until_unix_s: effective_valid_until,
        properties,
        does_not_establish: envelope.receipt.does_not_establish.clone(),
    })
}

fn bounded_nonempty(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_ADMISSION_GRANT_STRING_BYTES
}

fn earliest_deadline(values: [Option<u64>; 3]) -> Option<u64> {
    values.into_iter().flatten().min()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_case::{EvidenceMethod, SubjectRef};
    use crate::verification_receipt::{
        ReceiptAttestation, VerificationReceipt, VerifierIdentity,
        VERIFICATION_RECEIPT_ATTESTATION_DOMAIN, VERIFICATION_RECEIPT_SCHEMA_VERSION,
    };

    fn subject() -> SubjectRef {
        SubjectRef {
            namespace: "symthaea.source".into(),
            subject_id: "crate-a".into(),
            digest: "sha256:subject".into(),
        }
    }

    fn envelope(conclusion: VerificationConclusion) -> AttestedVerificationReceipt {
        AttestedVerificationReceipt {
            receipt: VerificationReceipt {
                schema_version: VERIFICATION_RECEIPT_SCHEMA_VERSION,
                receipt_id: Uuid::from_bytes([0x44; 16]),
                subject: subject(),
                method: EvidenceMethod::FormalProof,
                conclusion,
                properties: vec!["property.memory_safe".into(), "property.bounds_safe".into()],
                does_not_establish: vec!["property.side_channel_free".into()],
                verifier: VerifierIdentity {
                    verifier_id: "lean4".into(),
                    version: "4.x".into(),
                    artifact_digest: "sha256:lean".into(),
                },
                environment_digest: "sha256:env".into(),
                input_digest: "sha256:input".into(),
                output_digest: "sha256:proof".into(),
                assumptions_digest: Some("sha256:assumptions".into()),
                issued_unix_s: 100,
                valid_until_unix_s: Some(200),
            },
            attestation: Some(ReceiptAttestation {
                domain: VERIFICATION_RECEIPT_ATTESTATION_DOMAIN.into(),
                signer_id: "xenia.operator".into(),
                key_id: "key-1".into(),
                algorithm: "hybrid-profile".into(),
                payload_digest: "sha256:receipt-transcript".into(),
                signature: vec![1, 2, 3],
            }),
        }
    }

    fn signer_grant(value: &AttestedVerificationReceipt) -> SignerAuthorizationGrant {
        let attestation = value.attestation.as_ref().unwrap();
        SignerAuthorizationGrant {
            grant_id: Uuid::from_bytes([0x55; 16]),
            receipt_id: value.receipt.receipt_id,
            subject: value.receipt.subject.clone(),
            signer_id: attestation.signer_id.clone(),
            key_id: attestation.key_id.clone(),
            algorithm: attestation.algorithm.clone(),
            attestation_payload_digest: attestation.payload_digest.clone(),
            authority_scope: "attest:verification-receipt".into(),
            issued_unix_s: 100,
            valid_until_unix_s: Some(180),
        }
    }

    fn admission_grant(
        value: &AttestedVerificationReceipt,
        profile: &str,
        properties: &[&str],
    ) -> EvidenceAdmissionGrant {
        EvidenceAdmissionGrant {
            grant_id: Uuid::from_bytes([0x66; 16]),
            receipt_id: value.receipt.receipt_id,
            subject: value.receipt.subject.clone(),
            profile_id: profile.into(),
            properties: properties.iter().map(|value| (*value).to_string()).collect(),
            issued_unix_s: 100,
            valid_until_unix_s: Some(180),
        }
    }

    #[test]
    fn signer_authorization_cannot_be_replayed_across_receipts() {
        let first = envelope(VerificationConclusion::Supports);
        let grant = signer_grant(&first);
        let mut second = first.clone();
        second.receipt.receipt_id = Uuid::from_bytes([0x77; 16]);
        let admission = admission_grant(&second, "production", &["property.memory_safe"]);

        let result = admit_verification_receipt(&second, &grant, &admission, "production", 150);
        assert_eq!(
            result,
            Err(ReceiptAdmissionError::SignerAuthorizationReceiptMismatch)
        );
    }

    #[test]
    fn signer_authorization_must_match_exact_attestation_identity() {
        let value = envelope(VerificationConclusion::Supports);
        let mut grant = signer_grant(&value);
        grant.key_id = "different-key".into();
        let admission = admission_grant(&value, "production", &["property.memory_safe"]);

        let result = admit_verification_receipt(&value, &grant, &admission, "production", 150);
        assert_eq!(
            result,
            Err(ReceiptAdmissionError::SignerAuthorizationAttestationMismatch)
        );
    }

    #[test]
    fn admission_is_exact_profile_and_receipt_scoped() {
        let value = envelope(VerificationConclusion::Supports);
        let signer = signer_grant(&value);
        let admission = admission_grant(&value, "lab", &["property.memory_safe"]);

        let result = admit_verification_receipt(&value, &signer, &admission, "production", 150);
        assert_eq!(
            result,
            Err(ReceiptAdmissionError::AdmissionProfileMismatch {
                admitted_profile_id: "lab".into(),
                requested_profile_id: "production".into(),
            })
        );
    }

    #[test]
    fn admission_cannot_widen_receipt_property_scope() {
        let value = envelope(VerificationConclusion::Supports);
        let signer = signer_grant(&value);
        let admission = admission_grant(&value, "production", &["property.side_channel_free"]);

        let result = admit_verification_receipt(&value, &signer, &admission, "production", 150);
        assert_eq!(
            result,
            Err(ReceiptAdmissionError::AdmissionPropertyOutOfReceiptScope {
                property: "property.side_channel_free".into(),
            })
        );
    }

    #[test]
    fn expired_authority_or_admission_grants_fail_closed() {
        let value = envelope(VerificationConclusion::Supports);
        let signer = signer_grant(&value);
        let admission = admission_grant(&value, "production", &["property.memory_safe"]);

        assert_eq!(
            admit_verification_receipt(&value, &signer, &admission, "production", 181),
            Err(ReceiptAdmissionError::SignerAuthorizationExpired)
        );
    }

    #[test]
    fn admitted_support_preserves_exact_subset_and_limitations() {
        let value = envelope(VerificationConclusion::Supports);
        let signer = signer_grant(&value);
        let admission = admission_grant(&value, "production", &["property.memory_safe"]);

        let result = admit_verification_receipt(&value, &signer, &admission, "production", 150)
            .unwrap();

        assert_eq!(result.method, EvidenceMethod::FormalProof);
        assert_eq!(result.receipt_issued_unix_s, 100);
        assert_eq!(result.properties.len(), 1);
        assert_eq!(result.properties[0].method, EvidenceMethod::FormalProof);
        assert_eq!(
            result.property_state("property.memory_safe"),
            Some(PropertyState::Supported)
        );
        assert_eq!(result.property_state("property.bounds_safe"), None);
        assert_eq!(result.property_state("property.side_channel_free"), None);
        assert!(
            result
                .does_not_establish
                .contains(&"property.side_channel_free".to_string())
        );
        assert_eq!(result.valid_until_unix_s, Some(180));
        assert!(!result.is_expired_at(180));
        assert!(result.is_expired_at(181));
    }

    #[test]
    fn effective_expiry_is_earliest_input_deadline() {
        let value = envelope(VerificationConclusion::Supports);
        let mut signer = signer_grant(&value);
        signer.valid_until_unix_s = Some(190);
        let mut admission = admission_grant(&value, "production", &["property.memory_safe"]);
        admission.valid_until_unix_s = Some(170);

        let result = admit_verification_receipt(&value, &signer, &admission, "production", 150)
            .unwrap();
        assert_eq!(result.valid_until_unix_s, Some(170));
        assert_eq!(result.properties[0].valid_until_unix_s, Some(170));
    }

    #[test]
    fn refutation_and_inconclusive_remain_distinct() {
        let refuted_value = envelope(VerificationConclusion::Refutes);
        let refuted = admit_verification_receipt(
            &refuted_value,
            &signer_grant(&refuted_value),
            &admission_grant(&refuted_value, "production", &["property.memory_safe"]),
            "production",
            150,
        )
        .unwrap();
        assert_eq!(
            refuted.property_state("property.memory_safe"),
            Some(PropertyState::Refuted)
        );

        let inconclusive_value = envelope(VerificationConclusion::Inconclusive);
        let inconclusive = admit_verification_receipt(
            &inconclusive_value,
            &signer_grant(&inconclusive_value),
            &admission_grant(
                &inconclusive_value,
                "production",
                &["property.memory_safe"],
            ),
            "production",
            150,
        )
        .unwrap();
        assert_eq!(
            inconclusive.property_state("property.memory_safe"),
            Some(PropertyState::Indeterminate)
        );
    }
}