//! Standards-oriented Subterra science-return attestation profile.
//!
//! Wire format:
//! - in-toto Statement v1 JSON as the COSE payload;
//! - COSE_Sign1 with CBOR tag 18 (RFC 9943 Signed_Statement);
//! - protected algorithm Ed25519 (-19), key ID, content type and CWT iss/sub claims.
//!
//! This module creates/verifies signed statements. It does not register them
//! with a SCITT Transparency Service or verify a SCITT inclusion receipt.

use std::collections::BTreeMap;

use coset::{
    cbor::Value,
    iana,
    AsCborValue,
    CborSerializable,
    CoseSign1,
    CoseSign1Builder,
    HeaderBuilder,
    Label,
};
use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::science_return::{
    ScienceReturnEvent,
    ScienceReturnPayload,
    ScienceReturnRecord,
    ScienceReturnIssue,
};

pub const IN_TOTO_STATEMENT_V1: &str = "https://in-toto.io/Statement/v1";
pub const SUBTERRA_SCIENCE_RETURN_PREDICATE_V1: &str =
    "https://luminousdynamics.io/attestations/subterra/science-return/v0.1";
pub const IN_TOTO_JSON_CONTENT_TYPE: &str = "application/vnd.in-toto+json";
pub const SCITT_COSE_SIGN1_TAG: u64 = 18;
pub const IN_TOTO_SUBJECT_NAME: &str = "subterra-observation-v0.1";
const CWT_CLAIMS_HEADER_LABEL: i64 = 15;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InTotoResourceDescriptorV1 {
    pub name: String,
    pub digest: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubterraScienceReturnPredicateV1 {
    pub site_id: String,
    pub observation_id: String,
    /// Full ordered event log, including acquisition, verification, custody,
    /// delivery and admission states. Opaque refs remain opaque.
    pub events: Vec<ScienceReturnEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InTotoStatementV1 {
    #[serde(rename = "_type")]
    pub type_uri: String,
    pub subject: Vec<InTotoResourceDescriptorV1>,
    #[serde(rename = "predicateType")]
    pub predicate_type: String,
    pub predicate: SubterraScienceReturnPredicateV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScittAttestationError {
    EmptyIssuerOrSubjectUri,
    InvalidKeyId,
    InvalidScienceReturnRecord(Vec<ScienceReturnIssue>),
    CborEncoding,
    CborDecoding,
    InvalidCoseTag,
    InvalidCoseStructure,
    MissingCosePayload,
    WrongCoseAlgorithm,
    MissingProtectedKeyId,
    ProtectedKeyIdMismatch,
    WrongCoseContentType,
    UnexpectedUnprotectedSecurityHeader,
    UnsupportedProtectedCriticalHeader,
    MissingCwtClaims,
    DuplicateCwtClaims,
    InvalidCwtClaims,
    IssuerMismatch,
    SubjectMismatch,
    InvalidPublicKey,
    InvalidSignature,
    WrongInTotoStatementType,
    WrongPredicateType,
    WrongSubjectCount,
    WrongSubjectName,
    MissingSha256Digest,
    MalformedSha256Digest,
    PayloadDigestMismatch,
    InvalidStatementPayload,
}

impl std::fmt::Display for ScittAttestationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ScittAttestationError {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

/// Narrow URI gate for the v0.1 profile. Full URI authorization and ownership
/// must be enforced by the caller's issuer/key registry.
fn valid_absolute_uri(value: &str) -> bool {
    nonempty(value)
        && value.contains(':')
        && !value.chars().any(char::is_whitespace)
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn cwt_issuer_subject_claims(issuer_uri: &str, subject_uri: &str) -> Value {
    Value::Map(vec![
        (
            Value::Integer(1.into()),
            Value::Text(issuer_uri.to_string()),
        ),
        (
            Value::Integer(2.into()),
            Value::Text(subject_uri.to_string()),
        ),
    ])
}

fn cwt_claim_text<'a>(claims: &'a Value, label: i64) -> Result<&'a str, ScittAttestationError> {
    let entries = claims
        .as_map()
        .ok_or(ScittAttestationError::InvalidCwtClaims)?;
    let key = Value::Integer(label.into());
    let matches = entries
        .iter()
        .filter(|(candidate, _)| candidate == &key)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(ScittAttestationError::InvalidCwtClaims);
    }
    matches[0]
        .1
        .as_text()
        .ok_or(ScittAttestationError::InvalidCwtClaims)
}

fn protected_cwt_claims(sign1: &CoseSign1) -> Result<&Value, ScittAttestationError> {
    let matches = sign1
        .protected
        .header
        .rest
        .iter()
        .filter(|(label, _)| label == &Label::Int(CWT_CLAIMS_HEADER_LABEL))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [(_, claims)] => Ok(claims),
        [] => Err(ScittAttestationError::MissingCwtClaims),
        _ => Err(ScittAttestationError::DuplicateCwtClaims),
    }
}

fn check_cose_profile(
    sign1: &CoseSign1,
    expected_key_id: &str,
    expected_issuer_uri: &str,
    expected_subject_uri: &str,
) -> Result<(), ScittAttestationError> {
    if sign1.protected.header.alg != Some(iana::Algorithm::Ed25519) {
        return Err(ScittAttestationError::WrongCoseAlgorithm);
    }
    if sign1.protected.header.key_id.is_empty() {
        return Err(ScittAttestationError::MissingProtectedKeyId);
    }
    if sign1.protected.header.key_id.as_slice() != expected_key_id.as_bytes() {
        return Err(ScittAttestationError::ProtectedKeyIdMismatch);
    }
    if !sign1.protected.header.crit.is_empty() {
        return Err(ScittAttestationError::UnsupportedProtectedCriticalHeader);
    }
    if sign1.protected.header.content_type.as_ref()
        != Some(&coset::ContentType::Text(IN_TOTO_JSON_CONTENT_TYPE.to_string()))
    {
        return Err(ScittAttestationError::WrongCoseContentType);
    }

    // Security-relevant fields must be protected. A conflicting unprotected
    // algorithm, key ID or CWT claim is rejected rather than merged.
    if sign1.unprotected.alg.is_some()
        || sign1.unprotected.content_type.is_some()
        || !sign1.unprotected.key_id.is_empty()
        || sign1.unprotected.rest.iter().any(|(label, _)| {
            label == &Label::Int(CWT_CLAIMS_HEADER_LABEL)
        })
    {
        return Err(ScittAttestationError::UnexpectedUnprotectedSecurityHeader);
    }

    let claims = protected_cwt_claims(sign1)?;
    if cwt_claim_text(claims, 1)? != expected_issuer_uri {
        return Err(ScittAttestationError::IssuerMismatch);
    }
    if cwt_claim_text(claims, 2)? != expected_subject_uri {
        return Err(ScittAttestationError::SubjectMismatch);
    }
    Ok(())
}

/// Construct an in-toto Statement v1 carried by an RFC 9943 COSE_Sign1
/// Signed_Statement (CBOR tag 18). The key ID and issuer URI are signed
/// protected-header values; they do not establish trust by themselves.
pub fn create_scitt_signed_science_return_statement(
    record: &ScienceReturnRecord,
    payload_bytes: &[u8],
    issuer_uri: &str,
    subject_uri: &str,
    key_id: &str,
    signing_key: &SigningKey,
) -> Result<Vec<u8>, ScittAttestationError> {
    if !valid_absolute_uri(issuer_uri) || !valid_absolute_uri(subject_uri) {
        return Err(ScittAttestationError::EmptyIssuerOrSubjectUri);
    }
    if !nonempty(key_id) || key_id.len() > 256 {
        return Err(ScittAttestationError::InvalidKeyId);
    }
    record
        .verify_payload_bytes(payload_bytes)
        .map_err(ScittAttestationError::InvalidScienceReturnRecord)?;

    let declared_payload = record
        .payload
        .as_ref()
        .ok_or(ScittAttestationError::InvalidScienceReturnRecord(vec![
            ScienceReturnIssue::PayloadNotDeclaredForVerification,
        ]))?;
    if !is_canonical_sha256(&declared_payload.sha256) {
        return Err(ScittAttestationError::MalformedSha256Digest);
    }

    let mut digest = BTreeMap::new();
    digest.insert("sha256".to_string(), declared_payload.sha256.clone());
    let statement = InTotoStatementV1 {
        type_uri: IN_TOTO_STATEMENT_V1.to_string(),
        subject: vec![InTotoResourceDescriptorV1 {
            name: IN_TOTO_SUBJECT_NAME.to_string(),
            digest,
        }],
        predicate_type: SUBTERRA_SCIENCE_RETURN_PREDICATE_V1.to_string(),
        predicate: SubterraScienceReturnPredicateV1 {
            site_id: record.site_id.clone(),
            observation_id: record.observation_id.clone(),
            events: record.events.clone(),
        },
    };
    let statement_bytes = serde_json::to_vec(&statement)
        .map_err(|_| ScittAttestationError::InvalidStatementPayload)?;

    let protected = HeaderBuilder::new()
        .algorithm(iana::Algorithm::Ed25519)
        .key_id(key_id.as_bytes().to_vec())
        .content_type(IN_TOTO_JSON_CONTENT_TYPE.to_string())
        .value(
            CWT_CLAIMS_HEADER_LABEL,
            cwt_issuer_subject_claims(issuer_uri, subject_uri),
        )
        .build();
    let signed_statement = CoseSign1Builder::new()
        .protected(protected)
        .payload(statement_bytes)
        .create_signature(&[], |to_be_signed| {
            use ed25519_dalek::Signer;
            signing_key.sign(to_be_signed).to_bytes().to_vec()
        })
        .build();

    let cose_value = signed_statement
        .to_cbor_value()
        .map_err(|_| ScittAttestationError::CborEncoding)?;
    let tagged_value = Value::Tag(SCITT_COSE_SIGN1_TAG, Box::new(cose_value));
    tagged_value
        .to_vec()
        .map_err(|_| ScittAttestationError::CborEncoding)
}

/// Verify a tagged RFC 9943 Signed_Statement against caller-trusted issuer,
/// subject, key ID and public key. Returns the parsed in-toto Statement only
/// after strict signature validation, digest checking and event-log validation.
///
/// No SCITT transparency receipt is accepted here; receipt inclusion is a
/// separate verifier and does not prove the scientific predicate is true.
pub fn verify_scitt_signed_science_return_statement(
    signed_statement_bytes: &[u8],
    payload_bytes: &[u8],
    trusted_issuer_uri: &str,
    trusted_subject_uri: &str,
    trusted_key_id: &str,
    trusted_public_key_bytes: &[u8],
) -> Result<InTotoStatementV1, ScittAttestationError> {
    if !valid_absolute_uri(trusted_issuer_uri) || !valid_absolute_uri(trusted_subject_uri) {
        return Err(ScittAttestationError::EmptyIssuerOrSubjectUri);
    }
    if !nonempty(trusted_key_id) || trusted_key_id.len() > 256 {
        return Err(ScittAttestationError::InvalidKeyId);
    }

    let encoded_value = Value::from_slice(signed_statement_bytes)
        .map_err(|_| ScittAttestationError::CborDecoding)?;
    let Value::Tag(tag, inner) = encoded_value else {
        return Err(ScittAttestationError::InvalidCoseTag);
    };
    if tag != SCITT_COSE_SIGN1_TAG {
        return Err(ScittAttestationError::InvalidCoseTag);
    }
    let sign1 = CoseSign1::from_cbor_value(*inner)
        .map_err(|_| ScittAttestationError::InvalidCoseStructure)?;
    check_cose_profile(
        &sign1,
        trusted_key_id,
        trusted_issuer_uri,
        trusted_subject_uri,
    )?;

    let public_key_array: [u8; 32] = trusted_public_key_bytes
        .try_into()
        .map_err(|_| ScittAttestationError::InvalidPublicKey)?;
    let verifying_key = VerifyingKey::from_bytes(&public_key_array)
        .map_err(|_| ScittAttestationError::InvalidPublicKey)?;
    sign1.verify_signature(&[], |signature_bytes, to_be_signed| {
        let signature = Signature::from_slice(signature_bytes)
            .map_err(|_| "invalid Ed25519 signature")?;
        verifying_key
            .verify_strict(to_be_signed, &signature)
            .map_err(|_| "invalid Ed25519 signature")
    })
    .map_err(|_| ScittAttestationError::InvalidSignature)?;

    let statement_bytes = sign1
        .payload
        .as_deref()
        .ok_or(ScittAttestationError::MissingCosePayload)?;
    let statement: InTotoStatementV1 = serde_json::from_slice(statement_bytes)
        .map_err(|_| ScittAttestationError::InvalidStatementPayload)?;

    if statement.type_uri != IN_TOTO_STATEMENT_V1 {
        return Err(ScittAttestationError::WrongInTotoStatementType);
    }
    if statement.predicate_type != SUBTERRA_SCIENCE_RETURN_PREDICATE_V1 {
        return Err(ScittAttestationError::WrongPredicateType);
    }
    if statement.subject.len() != 1 {
        return Err(ScittAttestationError::WrongSubjectCount);
    }
    let subject = &statement.subject[0];
    if subject.name != IN_TOTO_SUBJECT_NAME {
        return Err(ScittAttestationError::WrongSubjectName);
    }
    if subject.digest.len() != 1 {
        return Err(ScittAttestationError::MissingSha256Digest);
    }
    let declared_sha256 = subject
        .digest
        .get("sha256")
        .ok_or(ScittAttestationError::MissingSha256Digest)?;
    if !is_canonical_sha256(declared_sha256) {
        return Err(ScittAttestationError::MalformedSha256Digest);
    }
    if *declared_sha256 != sha256_hex(payload_bytes) {
        return Err(ScittAttestationError::PayloadDigestMismatch);
    }

    let record = ScienceReturnRecord {
        site_id: statement.predicate.site_id.clone(),
        observation_id: statement.predicate.observation_id.clone(),
        payload: Some(ScienceReturnPayload {
            sha256: declared_sha256.clone(),
            byte_length: payload_bytes.len() as u64,
            attestation: None,
        }),
        events: statement.predicate.events.clone(),
    };
    record
        .verify_payload_bytes(payload_bytes)
        .map_err(ScittAttestationError::InvalidScienceReturnRecord)?;

    Ok(statement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::science_return::ScienceReturnStage;

    const ISSUER_URI: &str = "https://science.example.org/issuers/robotics-team-01";
    const SUBJECT_URI: &str = "urn:subterra:site:synthetic-site-a:observation:obs-001";
    const KEY_ID: &str = "sensor-signing-key-01";
    const ACTOR_ID: &str = "synthetic-robot-01";

    fn event(
        sequence: u64,
        stage: ScienceReturnStage,
        event_id: &str,
    ) -> ScienceReturnEvent {
        let mut item = ScienceReturnEvent {
            sequence,
            event_id: event_id.to_string(),
            stage,
            recorded_at: format!("2026-11-16T12:{:02}:00Z", sequence),
            clock_id: "robot-clock-01".to_string(),
            clock_uncertainty_ms: 1_000,
            actor_id: ACTOR_ID.to_string(),
            evidence_ref: format!("evidence-{event_id}"),
            calibration_ref: None,
            custody_ref: None,
            delivery_receipt_ref: None,
            admission_ref: None,
            disposition_reason: None,
        };
        match stage {
            ScienceReturnStage::IntegrityVerified => {
                item.calibration_ref = Some("calibration-01".to_string());
            }
            ScienceReturnStage::Persisted => {
                item.custody_ref = Some("storage-custody-01".to_string());
            }
            ScienceReturnStage::Delivered => {
                item.custody_ref = Some("delivery-custody-01".to_string());
                item.delivery_receipt_ref = Some("delivery-receipt-01".to_string());
            }
            ScienceReturnStage::Admitted => {
                item.custody_ref = Some("science-custody-01".to_string());
                item.delivery_receipt_ref = Some("delivery-receipt-01".to_string());
                item.admission_ref = Some("science-admission-01".to_string());
            }
            _ => {}
        }
        item
    }

    fn valid_record() -> ScienceReturnRecord {
        ScienceReturnRecord {
            site_id: "synthetic-site-a".to_string(),
            observation_id: "obs-001".to_string(),
            payload: Some(ScienceReturnPayload {
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
                byte_length: 3,
                attestation: None,
            }),
            events: vec![
                event(1, ScienceReturnStage::Proposed, "proposed"),
                event(2, ScienceReturnStage::Executed, "executed"),
                event(3, ScienceReturnStage::Acquired, "acquired"),
                event(4, ScienceReturnStage::IntegrityVerified, "verified"),
                event(5, ScienceReturnStage::Persisted, "persisted"),
                event(6, ScienceReturnStage::Delivered, "delivered"),
                event(7, ScienceReturnStage::Admitted, "admitted"),
            ],
        }
    }

    fn signed_fixture() -> (Vec<u8>, [u8; 32]) {
        let signing_key = SigningKey::from_bytes(&[9_u8; 32]);
        let signed = create_scitt_signed_science_return_statement(
            &valid_record(),
            b"abc",
            ISSUER_URI,
            SUBJECT_URI,
            KEY_ID,
            &signing_key,
        )
        .unwrap();
        (signed, signing_key.verifying_key().to_bytes())
    }

    #[test]
    fn scitt_signed_statement_roundtrips_and_verifies() {
        let (signed, public_key) = signed_fixture();
        let statement = verify_scitt_signed_science_return_statement(
            &signed,
            b"abc",
            ISSUER_URI,
            SUBJECT_URI,
            KEY_ID,
            &public_key,
        )
        .unwrap();
        assert_eq!(statement.type_uri, IN_TOTO_STATEMENT_V1);
        assert_eq!(statement.predicate.site_id, "synthetic-site-a");
        assert_eq!(statement.predicate.observation_id, "obs-001");
        assert_eq!(statement.subject[0].digest["sha256"], sha256_hex(b"abc"));
    }

    #[test]
    fn signed_statement_uses_cbor_tag_18_and_registered_ed25519_algorithm() {
        let (signed, _) = signed_fixture();
        let encoded = Value::from_slice(&signed).unwrap();
        let Value::Tag(tag, inner) = encoded else {
            panic!("expected tag 18");
        };
        assert_eq!(tag, SCITT_COSE_SIGN1_TAG);
        let sign1 = CoseSign1::from_cbor_value(*inner).unwrap();
        assert_eq!(sign1.protected.header.alg, Some(iana::Algorithm::Ed25519));
        assert_eq!(sign1.protected.header.key_id, KEY_ID.as_bytes());
        assert_eq!(
            sign1.protected.header.content_type,
            Some(coset::ContentType::Text(IN_TOTO_JSON_CONTENT_TYPE.to_string()))
        );
        assert!(protected_cwt_claims(&sign1).is_ok());
    }

    #[test]
    fn wrong_public_key_is_rejected() {
        let (signed, _) = signed_fixture();
        let wrong_key = SigningKey::from_bytes(&[10_u8; 32]).verifying_key().to_bytes();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &wrong_key,
            ),
            Err(ScittAttestationError::InvalidSignature)
        );
    }

    #[test]
    fn wrong_issuer_subject_or_key_id_is_rejected() {
        let (signed, public_key) = signed_fixture();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abc",
                "https://science.example.org/other-issuer",
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::IssuerMismatch)
        );
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abc",
                ISSUER_URI,
                "urn:subterra:another-subject",
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::SubjectMismatch)
        );
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                "other-key",
                &public_key,
            ),
            Err(ScittAttestationError::ProtectedKeyIdMismatch)
        );
    }

    #[test]
    fn altered_payload_bytes_are_rejected() {
        let (signed, public_key) = signed_fixture();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abd",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::PayloadDigestMismatch)
        );
    }

    #[test]
    fn altered_signature_is_rejected() {
        let (mut signed, public_key) = signed_fixture();
        if let Value::Tag(_, inner) = Value::from_slice(&signed).unwrap() {
            let mut sign1 = CoseSign1::from_cbor_value(*inner).unwrap();
            sign1.signature[0] ^= 0x01;
            signed = Value::Tag(
                SCITT_COSE_SIGN1_TAG,
                Box::new(sign1.to_cbor_value().unwrap()),
            )
            .to_vec()
            .unwrap();
        }
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::InvalidSignature)
        );
    }

    #[test]
    fn untagged_cose_sign1_is_not_accepted_as_a_scitt_signed_statement() {
        let (signed, public_key) = signed_fixture();
        let tagged = Value::from_slice(&signed).unwrap();
        let Value::Tag(_, inner) = tagged else {
            panic!("expected tag wrapper");
        };
        let untagged = (*inner).to_vec().unwrap();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &untagged,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::InvalidCoseTag)
        );
    }

    #[test]
    fn signed_statement_rejects_unknown_public_key_id_substitution() {
        let (signed, public_key) = signed_fixture();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                "untrusted-key",
                &public_key,
            ),
            Err(ScittAttestationError::ProtectedKeyIdMismatch)
        );
    }

    #[test]
    fn creator_rejects_unverified_or_invalid_payload_records() {
        let mut record = valid_record();
        record.payload.as_mut().unwrap().sha256 = "00".repeat(32);
        assert!(matches!(
            create_scitt_signed_science_return_statement(
                &record,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &SigningKey::from_bytes(&[9_u8; 32]),
            ),
            Err(ScittAttestationError::InvalidScienceReturnRecord(_))
        ));
        assert_eq!(
            create_scitt_signed_science_return_statement(
                &valid_record(),
                b"abc",
                "not-a-uri",
                SUBJECT_URI,
                KEY_ID,
                &SigningKey::from_bytes(&[9_u8; 32]),
            ),
            Err(ScittAttestationError::EmptyIssuerOrSubjectUri)
        );
    }

    #[test]
    fn invalid_event_history_inside_a_signed_statement_is_rejected() {
        let (mut signed, public_key) = signed_fixture();
        if let Value::Tag(_, inner) = Value::from_slice(&signed).unwrap() {
            let mut sign1 = CoseSign1::from_cbor_value(*inner).unwrap();
            let mut statement: InTotoStatementV1 =
                serde_json::from_slice(sign1.payload.as_ref().unwrap()).unwrap();
            statement.predicate.events[3].calibration_ref = None;
            sign1.payload = Some(serde_json::to_vec(&statement).unwrap());
            let original_header = sign1.protected.clone();
            let re_signed = CoseSign1Builder::new()
                .protected(original_header)
                .payload(sign1.payload.clone().unwrap())
                .create_signature(&[], |to_be_signed| {
                    use ed25519_dalek::Signer;
                    SigningKey::from_bytes(&[9_u8; 32])
                        .sign(to_be_signed)
                        .to_bytes()
                        .to_vec()
                })
                .build();
            signed = Value::Tag(
                SCITT_COSE_SIGN1_TAG,
                Box::new(re_signed.to_cbor_value().unwrap()),
            )
            .to_vec()
            .unwrap();
        }
        assert!(matches!(
            verify_scitt_signed_science_return_statement(
                &signed,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::InvalidScienceReturnRecord(_))
        ));
    }

    #[test]
    fn non_ed25519_algorithm_in_protected_header_is_rejected() {
        let (signed, public_key) = signed_fixture();
        let Value::Tag(_, inner) = Value::from_slice(&signed).unwrap() else {
            panic!("expected tag wrapper");
        };
        let mut sign1 = CoseSign1::from_cbor_value(*inner).unwrap();
        sign1.protected.original_data = None;
        sign1.protected.header.alg = Some(iana::Algorithm::A128GCM);
        let tampered = Value::Tag(
            SCITT_COSE_SIGN1_TAG,
            Box::new(sign1.to_cbor_value().unwrap()),
        )
        .to_vec()
        .unwrap();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &tampered,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::WrongCoseAlgorithm)
        );
    }

    #[test]
    fn unexpected_unprotected_security_header_is_rejected() {
        let (signed, public_key) = signed_fixture();
        let Value::Tag(_, inner) = Value::from_slice(&signed).unwrap() else {
            panic!("expected tag wrapper");
        };
        let mut sign1 = CoseSign1::from_cbor_value(*inner).unwrap();
        sign1.unprotected.key_id = KEY_ID.as_bytes().to_vec();
        let tampered = Value::Tag(
            SCITT_COSE_SIGN1_TAG,
            Box::new(sign1.to_cbor_value().unwrap()),
        )
        .to_vec()
        .unwrap();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &tampered,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::UnexpectedUnprotectedSecurityHeader)
        );
    }
    #[test]
    fn wrong_protected_content_type_is_rejected() {
        let (signed, public_key) = signed_fixture();
        let Value::Tag(_, inner) = Value::from_slice(&signed).unwrap() else {
            panic!("expected tag wrapper");
        };
        let mut sign1 = CoseSign1::from_cbor_value(*inner).unwrap();
        sign1.protected.original_data = None;
        sign1.protected.header.content_type =
            Some(coset::ContentType::Text("application/json".to_string()));
        let tampered = Value::Tag(
            SCITT_COSE_SIGN1_TAG,
            Box::new(sign1.to_cbor_value().unwrap()),
        )
        .to_vec()
        .unwrap();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &tampered,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::WrongCoseContentType)
        );
    }

    #[test]
    fn protected_cwt_claims_are_not_accepted_from_the_unprotected_header() {
        let (signed, public_key) = signed_fixture();
        let Value::Tag(_, inner) = Value::from_slice(&signed).unwrap() else {
            panic!("expected tag wrapper");
        };
        let mut sign1 = CoseSign1::from_cbor_value(*inner).unwrap();
        sign1.unprotected.rest.push((
            Label::Int(CWT_CLAIMS_HEADER_LABEL),
            cwt_issuer_subject_claims(ISSUER_URI, SUBJECT_URI),
        ));
        let tampered = Value::Tag(
            SCITT_COSE_SIGN1_TAG,
            Box::new(sign1.to_cbor_value().unwrap()),
        )
        .to_vec()
        .unwrap();
        assert_eq!(
            verify_scitt_signed_science_return_statement(
                &tampered,
                b"abc",
                ISSUER_URI,
                SUBJECT_URI,
                KEY_ID,
                &public_key,
            ),
            Err(ScittAttestationError::UnexpectedUnprotectedSecurityHeader)
        );
    }

}
