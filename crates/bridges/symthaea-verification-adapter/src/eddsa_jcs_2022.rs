//! Narrow Ed25519 + JCS verification for the W3C `eddsa-jcs-2022` suite.
//!
//! This module intentionally verifies exactly one detached DataIntegrityProof.
//! It does not perform controller-document retrieval; that responsibility stays
//! with the durable resolution adapter. It consumes the already-resolved public
//! method material so the cryptographic step cannot silently select a different key.
//!
//! The implementation follows the W3C 2025 Recommendation's JCS suite recipe:
//! JCS canonicalization, SHA-256 over the proof configuration and transformed
//! document, concatenation in that order, and pure Ed25519 verification.
//!
//! The numeric check below is an explicit Symthaea interoperability hardening
//! profile. RFC 7493 states the binary64 constraint as a SHOULD NOT rather than
//! a universal MUST, so this stricter rejection is deliberately not presented as
//! a generic claim of I-JSON conformance.

use ed25519_dalek::{Signature, VerifyingKey};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use symthaea_epistemic_types::{
    ClaimVerificationMethod, CryptographicVerificationReceipt, VerificationEvidence,
    VerificationFailure, VerificationMethodResolution, VerificationRequest,
};

use crate::{ResolvedVerificationMethod, ResolvedVerificationMethodMaterial, SnapshotError};
use crate::strict_json::{parse_strict_json, validate_strict_ijson_value};

pub const EDDSA_JCS_2022: &str = "eddsa-jcs-2022";
const DATA_INTEGRITY_PROOF: &str = "DataIntegrityProof";
const MAX_SECURED_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;

/// Verify a wire-format JSON document using a strict I-JSON parser before
/// entering the JCS/Data Integrity pipeline.
///
/// The ordinary `Value` entry point is subject to the same strict validation
/// before JCS, while raw network/storage JSON additionally enters through the
/// duplicate-name-aware parser. JCS requires adaptation to I-JSON; Symthaea
/// additionally applies a stronger numeric interoperability profile.
pub fn verify_eddsa_jcs_2022_json(
    request: &VerificationRequest,
    resolution: &VerificationMethodResolution,
    resolved_method: &ResolvedVerificationMethod,
    secured_document_json: &[u8],
) -> Result<CryptographicVerificationReceipt, SnapshotError> {
    if secured_document_json.len() > MAX_SECURED_DOCUMENT_BYTES {
        return Err(SnapshotError::Verification(
            VerificationFailure::Structural(
                "secured JSON document exceeds the verifier safety limit".into(),
            ),
        ));
    }
    let secured_document = parse_strict_json(secured_document_json)?;
    verify_eddsa_jcs_2022(
        request,
        resolution,
        resolved_method,
        &secured_document,
    )
}



/// Verify one secured JSON document containing exactly one
/// `DataIntegrityProof` using the already-resolved Multikey material.
///
/// The returned receipt records the exact intermediate digests used by the
/// verifier. The `cryptographic_input_digest` is a Symthaea evidence artifact:
/// SHA-256 of the 64-byte W3C `hashData` input supplied to Ed25519.
pub fn verify_eddsa_jcs_2022(
    request: &VerificationRequest,
    resolution: &VerificationMethodResolution,
    resolved_method: &ResolvedVerificationMethod,
    secured_document: &Value,
) -> Result<CryptographicVerificationReceipt, SnapshotError> {
    request.validate_structure()?;
    resolution.validate_structure()?;
    validate_strict_ijson_value(secured_document)?;
    if !resolution.matches_request(request) {
        return Err(SnapshotError::Verification(
            VerificationFailure::ResolutionRequestMismatch,
        ));
    }

    if resolved_method.method != resolution.verification_method
        || resolved_method.method != request.verification_method
        || resolved_method.method_type() != resolution.verification_method_type
        || resolved_method.material_digest != resolution.verification_method_material_digest
        || resolved_method.controller != resolution.resolved_verification_method_controller
        || resolved_method.lifecycle != resolution.verification_method_lifecycle
    {
        return Err(SnapshotError::Verification(
            VerificationFailure::ResolutionEvidenceMismatch,
        ));
    }

    let public_key_multibase = match &resolved_method.material {
        ResolvedVerificationMethodMaterial::Multikey {
            public_key_multibase,
        } => public_key_multibase,
        ResolvedVerificationMethodMaterial::JsonWebKey { .. } => {
            return Err(SnapshotError::Verification(
                VerificationFailure::Structural(
                    "eddsa-jcs-2022 requires Multikey verification material".into(),
                ),
            ));
        }
    };

    if resolved_method.method_type() != "Multikey"
        || resolution.verification_method_type != "Multikey"
    {
        return Err(SnapshotError::Verification(
            VerificationFailure::Structural(
                "eddsa-jcs-2022 resolution must be typed as Multikey".into(),
            ),
        ));
    }

    // Recompute the material identity from the exact handoff material. This is
    // deliberately independent of the resolution receipt's stored digest.
    let mut material_object = Map::new();
    material_object.insert(
        "publicKeyMultibase".into(),
        Value::String(public_key_multibase.clone()),
    );
    let recomputed_material_digest = crate::verification_method_material_digest(
        &resolved_method.method,
        "Multikey",
        &material_object,
    )?;
    if recomputed_material_digest != resolved_method.material_digest {
        return Err(SnapshotError::Verification(
            VerificationFailure::ResolutionEvidenceMismatch,
        ));
    }

    let secured_object = secured_document.as_object().ok_or_else(|| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "eddsa-jcs-2022 secured document root must be a JSON object".into(),
        ))
    })?;

    let proof_value = secured_object.get("proof").ok_or_else(|| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "eddsa-jcs-2022 secured document must contain a proof object".into(),
        ))
    })?;
    let proof = proof_value.as_object().ok_or_else(|| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "eddsa-jcs-2022 supports exactly one proof object, not a proof set".into(),
        ))
    })?;

    let proof_type = required_string(proof, "type")?;
    if proof_type != DATA_INTEGRITY_PROOF {
        return Err(SnapshotError::Verification(
            VerificationFailure::Structural(
                "eddsa-jcs-2022 proof type must be DataIntegrityProof".into(),
            ),
        ));
    }

    let cryptosuite = required_string(proof, "cryptosuite")?;
    if cryptosuite != EDDSA_JCS_2022 {
        return Err(SnapshotError::Verification(
            VerificationFailure::Structural(
                "proof cryptosuite is not eddsa-jcs-2022".into(),
            ),
        ));
    }

    let verification_method = required_string(proof, "verificationMethod")?;
    if verification_method != request.verification_method.as_str() {
        return Err(SnapshotError::Verification(
            VerificationFailure::VerificationMethodMismatch {
                expected: request.verification_method.clone(),
                actual: ClaimVerificationMethod::new(verification_method.to_owned())?,
            },
        ));
    }

    let proof_purpose = required_string(proof, "proofPurpose")?;
    if proof_purpose != request.proof_purpose.as_str() {
        return Err(SnapshotError::Verification(
            VerificationFailure::ProofPurposeMismatch {
                expected: request.proof_purpose.clone(),
                actual: symthaea_epistemic_types::ClaimProofPurpose::new(
                    proof_purpose.to_owned(),
                )?,
            },
        ));
    }

    validate_freshness_field(
        proof,
        "created",
        request.freshness.proof_created.as_deref(),
    )?;
    validate_freshness_field(
        proof,
        "expires",
        request.freshness.proof_expires.as_deref(),
    )?;
    validate_optional_string_field(
        proof,
        "domain",
        request.freshness.proof_domain.as_deref(),
        "proof domain",
    )?;
    validate_optional_string_field(
        proof,
        "challenge",
        request.freshness.proof_challenge.as_deref(),
        "proof challenge",
    )?;

    // Data Integrity's JCS suite requires proofOptions @context to be a prefix
    // of the secured document context and then replaces unsecuredDocument.@context
    // with that proof context before canonicalization.
    let mut unsecured_document = secured_document.clone();
    let unsecured_object = unsecured_document.as_object_mut().ok_or_else(|| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "eddsa-jcs-2022 secured document root must be a JSON object".into(),
        ))
    })?;
    unsecured_object.remove("proof");

    let mut proof_options = Value::Object(proof.clone());
    let proof_options_object = proof_options.as_object_mut().ok_or_else(|| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "eddsa-jcs-2022 proof options must be a JSON object".into(),
        ))
    })?;
    let proof_value_multibase = proof_options_object
        .remove("proofValue")
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| {
            SnapshotError::Verification(VerificationFailure::Structural(
                "eddsa-jcs-2022 proofValue must be a string".into(),
            ))
        })?;
    let proof_bytes = decode_multibase_signature(&proof_value_multibase)?;

    if let Some(proof_context) = proof_options_object.get("@context") {
        let document_context = unsecured_object.get("@context").ok_or_else(|| {
            SnapshotError::Verification(VerificationFailure::Structural(
                "proof @context is present but secured document has no @context".into(),
            ))
        })?;
        if !context_starts_with(document_context, proof_context) {
            return Err(SnapshotError::Verification(
                VerificationFailure::Structural(
                    "proof @context must be an ordered prefix of secured document @context"
                        .into(),
                ),
            ));
        }
        unsecured_object.insert("@context".into(), proof_context.clone());
    }

    let transformed_document = serde_jcs::to_vec(&unsecured_document).map_err(|error| {
        SnapshotError::Verification(VerificationFailure::Structural(format!(
            "eddsa-jcs-2022 document JCS canonicalization failed: {error}"
        )))
    })?;
    let canonical_proof_config = serde_jcs::to_vec(&proof_options).map_err(|error| {
        SnapshotError::Verification(VerificationFailure::Structural(format!(
            "eddsa-jcs-2022 proof configuration JCS canonicalization failed: {error}"
        )))
    })?;

    let transformed_document_hash = Sha256::digest(&transformed_document);
    let proof_config_hash = Sha256::digest(&canonical_proof_config);

    // W3C ordering is proofConfigHash || transformedDocumentHash.
    let mut hash_data = Vec::with_capacity(64);
    hash_data.extend_from_slice(&proof_config_hash);
    hash_data.extend_from_slice(&transformed_document_hash);

    let public_key_bytes = decode_ed25519_multikey(public_key_multibase)?;
    let mut public_key_array = [0u8; 32];
    public_key_array.copy_from_slice(&public_key_bytes);
    let verifying_key = VerifyingKey::from_bytes(&public_key_array).map_err(|_| {
        SnapshotError::Verification(VerificationFailure::CryptographicVerificationFailed)
    })?;
    let signature = Signature::from_slice(&proof_bytes).map_err(|_| {
        SnapshotError::Verification(VerificationFailure::CryptographicVerificationFailed)
    })?;
    verifying_key
        .verify_strict(&hash_data, &signature)
        .map_err(|_| {
            SnapshotError::Verification(VerificationFailure::CryptographicVerificationFailed)
        })?;

    let transformed_document_digest = hex::encode(transformed_document_hash);
    let proof_configuration_digest = hex::encode(proof_config_hash);
    let cryptographic_input_digest = hex::encode(Sha256::digest(&hash_data));

    let expected_transformed_document_digest = request
        .expected_transformed_document_digest
        .as_ref()
        .ok_or(SnapshotError::Verification(
            VerificationFailure::MissingExpectedTransformedDocumentDigest,
        ))?;
    if expected_transformed_document_digest != &transformed_document_digest {
        return Err(SnapshotError::Verification(
            VerificationFailure::TransformedDocumentDigestMismatch {
                expected: Some(expected_transformed_document_digest.clone()),
                actual: Some(transformed_document_digest.clone()),
            },
        ));
    }

    // Evidence identity is defined over the full proof's JCS form, independent
    // of the formatting of the submitted JSON representation.
    let proof_for_digest = Value::Object(proof.clone());
    let canonical_proof = serde_jcs::to_vec(&proof_for_digest).map_err(|error| {
        SnapshotError::Verification(VerificationFailure::Structural(format!(
            "eddsa-jcs-2022 full proof canonicalization failed: {error}"
        )))
    })?;
    let proof_digest = hex::encode(Sha256::digest(&canonical_proof));

    CryptographicVerificationReceipt::from_adapter_verification(
        request,
        resolution,
        DATA_INTEGRITY_PROOF,
        EDDSA_JCS_2022,
        transformed_document_digest,
        proof_configuration_digest,
        cryptographic_input_digest,
        proof_digest,
        proof_value_multibase,
    )
    .map_err(SnapshotError::Verification)
}

/// Verify the proof and immediately package the cryptographic result into the
/// substrate-neutral evidence envelope.
///
/// This is the preferred integration path because the caller cannot accidentally
/// replace the typed cryptographic receipt with free-form digest strings.
pub fn verify_eddsa_jcs_2022_evidence(
    request: &VerificationRequest,
    resolution: &VerificationMethodResolution,
    resolved_method: &ResolvedVerificationMethod,
    secured_document: &Value,
) -> Result<VerificationEvidence, SnapshotError> {
    let receipt = verify_eddsa_jcs_2022(
        request,
        resolution,
        resolved_method,
        secured_document,
    )?;
    VerificationEvidence::from_adapter_attestation(request, resolution.clone(), receipt)
        .map_err(SnapshotError::Verification)
}

use crate::strict_json::{parse_strict_json, validate_strict_ijson_value};

fn required_string(
    object: &Map<String, Value>,
    field: &str,
) -> Result<&str, SnapshotError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            SnapshotError::Verification(VerificationFailure::Structural(format!(
                "{field} must be a non-empty string"
            )))
        })
}

fn validate_freshness_field(
    proof: &Map<String, Value>,
    field: &str,
    expected: Option<&str>,
) -> Result<(), SnapshotError> {
    let actual = match proof.get(field) {
        None => None,
        Some(value) => Some(value.as_str().ok_or_else(|| {
            SnapshotError::Verification(VerificationFailure::Structural(format!(
                "proof {field} must be a string"
            )))
        })?),
    };
    if actual != expected {
        return Err(SnapshotError::Verification(
            VerificationFailure::FreshnessMismatch,
        ));
    }
    Ok(())
}

fn validate_optional_string_field(
    proof: &Map<String, Value>,
    field: &str,
    expected: Option<&str>,
    label: &str,
) -> Result<(), SnapshotError> {
    let actual = match proof.get(field) {
        None => None,
        Some(value) => Some(value.as_str().ok_or_else(|| {
            SnapshotError::Verification(VerificationFailure::Structural(format!(
                "proof {field} must be a string"
            )))
        })?),
    };
    if actual != expected {
        let error = match field {
            "domain" => VerificationFailure::DomainMismatch {
                expected: expected.unwrap_or_default().to_owned(),
                actual: actual.map(str::to_owned),
            },
            "challenge" => VerificationFailure::ChallengeMismatch {
                expected: expected.unwrap_or_default().to_owned(),
                actual: actual.map(str::to_owned),
            },
            _ => VerificationFailure::Structural(format!("{label} mismatch")),
        };
        return Err(SnapshotError::Verification(error));
    }
    Ok(())
}

fn context_values(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(values) => values.clone(),
        other => vec![other.clone()],
    }
}

fn context_starts_with(document_context: &Value, proof_context: &Value) -> bool {
    let document_values = context_values(document_context);
    let proof_values = context_values(proof_context);
    document_values.len() >= proof_values.len()
        && document_values[..proof_values.len()] == proof_values[..]
}

fn decode_multibase_signature(value: &str) -> Result<Vec<u8>, SnapshotError> {
    let encoded = value.strip_prefix('z').ok_or_else(|| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "proofValue must use the base58-btc multibase prefix".into(),
        ))
    })?;
    let bytes = bs58::decode(encoded).into_vec().map_err(|_| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "proofValue must contain valid base58-btc data".into(),
        ))
    })?;
    if bytes.len() != 64 {
        return Err(SnapshotError::Verification(
            VerificationFailure::Structural(
                "Ed25519 proofValue must decode to exactly 64 bytes".into(),
            ),
        ));
    }
    Ok(bytes)
}

fn decode_ed25519_multikey(value: &str) -> Result<Vec<u8>, SnapshotError> {
    let encoded = value.strip_prefix('z').ok_or_else(|| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "Ed25519 Multikey must use the base58-btc multibase prefix".into(),
        ))
    })?;
    let bytes = bs58::decode(encoded).into_vec().map_err(|_| {
        SnapshotError::Verification(VerificationFailure::Structural(
            "Ed25519 Multikey must contain valid base58-btc data".into(),
        ))
    })?;
    if bytes.len() != 34 || bytes[0] != 0xed || bytes[1] != 0x01 {
        return Err(SnapshotError::Verification(
            VerificationFailure::Structural(
                "eddsa-jcs-2022 requires the 0xed01 Ed25519 Multikey header".into(),
            ),
        ));
    }
    Ok(bytes[2..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{
        ClaimAuthorIdentity, ClaimControllerIdentity, ClaimProofPurpose,
        ClaimVerificationMethod, ClaimVerificationRelationship,
        ControllerDocumentIntegrityPolicy, ControllerDocumentNetworkPolicy,
        VerificationFreshnessContext, VerificationRequest,
    };
    use crate::{ControllerDocumentSnapshotFile, JsonControllerDocumentSnapshotAdapter};

    const CONTROLLER: &str =
        "did:key:z6MkrJVnaZkeFzdQyMZu1cgjg7k1pZZ6pvBQ7XJPt4swbTQ2";
    const METHOD: &str =
        "did:key:z6MkrJVnaZkeFzdQyMZu1cgjg7k1pZZ6pvBQ7XJPt4swbTQ2#z6MkrJVnaZkeFzdQyMZu1cgjg7k1pZZ6pvBQ7XJPt4swbTQ2";
    const PUBLIC_KEY: &str =
        "z6MkrJVnaZkeFzdQyMZu1cgjg7k1pZZ6pvBQ7XJPt4swbTQ2";
    const PROOF_VALUE: &str =
        "z2HnFSSPPBzR36zdDgK8PbEHeXbR56YF24jwMpt3R1eHXQzJDMWS93FCzpvJpwTWd3GAVFuUfjoJdcnTMuVor51aX";

    fn vector_request() -> VerificationRequest {
        VerificationRequest {
            schema_version: 1,
            claim_representation_digest: "00".repeat(32),
            statement_digest: "11".repeat(32),
            author: ClaimAuthorIdentity::new("author:w3c-vector").unwrap(),
            proof_purpose: ClaimProofPurpose::new("assertionMethod").unwrap(),
            verification_method: ClaimVerificationMethod::new(METHOD).unwrap(),
            expected_controller: ClaimControllerIdentity::new(CONTROLLER).unwrap(),
            expected_verification_relationship:
                ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            expected_transformed_document_digest: Some(
                "59b7cb6251b8991add1ce0bc83107e3db9dbbab5bd2c28f687db1a03abc92f19".into(),
            ),
            controller_document_integrity_policy: ControllerDocumentIntegrityPolicy::Unpinned,
            controller_document_network_policy: ControllerDocumentNetworkPolicy::strict_for_url(
                METHOD,
            )
            .unwrap(),
            freshness: VerificationFreshnessContext {
                proof_created: Some("2023-02-24T23:36:38Z".into()),
                proof_expires: None,
                proof_domain: None,
                proof_challenge: None,
                verification_time: "2023-02-24T23:36:38Z".into(),
                expected_domain: None,
                expected_challenge: None,
            },
        }
    }

    fn vector_credential() -> Value {
        serde_json::json!({
            "@context": [
                "https://www.w3.org/ns/credentials/v2",
                "https://www.w3.org/ns/credentials/examples/v2"
            ],
            "id": "urn:uuid:58172aac-d8ba-11ed-83dd-0b3aef56cc33",
            "type": ["VerifiableCredential", "AlumniCredential"],
            "name": "Alumni Credential",
            "description": "A minimum viable example of an Alumni Credential.",
            "issuer": "https://vc.example/issuers/5678",
            "validFrom": "2023-01-01T00:00:00Z",
            "credentialSubject": {
                "id": "did:example:abcdefgh",
                "alumniOf": "The School of Examples"
            }
        })
    }

    fn vector_proof() -> Value {
        serde_json::json!({
            "type": "DataIntegrityProof",
            "cryptosuite": EDDSA_JCS_2022,
            "created": "2023-02-24T23:36:38Z",
            "verificationMethod": METHOD,
            "proofPurpose": "assertionMethod",
            "@context": [
                "https://www.w3.org/ns/credentials/v2",
                "https://www.w3.org/ns/credentials/examples/v2"
            ],
            "proofValue": PROOF_VALUE
        })
    }

    fn resolved_vector(
        request: &VerificationRequest,
    ) -> (VerificationMethodResolution, ResolvedVerificationMethod) {
        let controller_document = serde_json::json!({
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://w3id.org/security/multikey/v1"
            ],
            "id": CONTROLLER,
            "verificationMethod": [{
                "id": METHOD,
                "type": "Multikey",
                "controller": CONTROLLER,
                "publicKeyMultibase": PUBLIC_KEY
            }],
            "assertionMethod": [METHOD]
        });
        let snapshot = ControllerDocumentSnapshotFile::new(
            CONTROLLER,
            "2023-02-24T23:36:38Z",
            "2023-02-24T23:36:39Z",
            "application/cid",
            serde_json::to_string(&controller_document).unwrap(),
        )
        .unwrap();
        let expected = snapshot.snapshot_reference().unwrap();
        let adapter = JsonControllerDocumentSnapshotAdapter::new(
            tempfile::NamedTempFile::new().unwrap().path(),
            expected,
        )
        .unwrap();
        adapter
            .resolve_snapshot_with_material(request, snapshot)
            .unwrap()
    }

    fn secured_document() -> Value {
        let mut credential = vector_credential();
        credential.as_object_mut().unwrap().insert(
            "proof".into(),
            vector_proof(),
        );
        credential
    }

    #[test]
    fn verifies_w3c_eddsa_jcs_2022_vector() {
        let request = vector_request();
        request.validate_structure().unwrap();
        let (resolution, resolved_method) = resolved_vector(&request);
        let receipt = verify_eddsa_jcs_2022(
            &request,
            &resolution,
            &resolved_method,
            &secured_document(),
        )
        .unwrap();

        assert_eq!(
            receipt.transformed_document_digest,
            "59b7cb6251b8991add1ce0bc83107e3db9dbbab5bd2c28f687db1a03abc92f19"
        );
        assert_eq!(
            receipt.proof_configuration_digest,
            "66ab154f5c2890a140cb8388a22a160454f80575f6eae09e5a097cabe539a1db"
        );
        assert_eq!(receipt.proof_value_multibase, PROOF_VALUE);
        assert_eq!(receipt.cryptosuite, EDDSA_JCS_2022);
        assert_eq!(
            receipt.cryptographic_input_digest,
            "953c5ff7db5ca4e543ede6262df2676fd21f9a4dbba7db0ad9266bda28020325"
        );

        let evidence = verify_eddsa_jcs_2022_evidence(
            &request,
            &resolution,
            &resolved_method,
            &secured_document(),
        )
        .unwrap();
        assert_eq!(
            evidence.cryptographic_verification.claim_representation_digest,
            request.claim_representation_digest
        );
        assert!(evidence.validate_structure().is_ok());
    }

    #[test]
    fn rejects_oversized_wire_document() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let oversized = vec![b' '; MAX_SECURED_DOCUMENT_BYTES + 1];

        assert!(matches!(
            verify_eddsa_jcs_2022_json(
                &request,
                &resolution,
                &resolved_method,
                &oversized,
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("safety limit")
        ));
    }

    #[test]
    fn wire_verifier_rejects_negative_zero_before_verification() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let wire =
            br#"{"proof":{"type":"DataIntegrityProof","cryptosuite":"eddsa-jcs-2022","verificationMethod":"ignored","proofPurpose":"assertionMethod","proofValue":"ignored"},"amount":-0}"#;

        assert!(matches!(
            verify_eddsa_jcs_2022_json(
                &request,
                &resolution,
                &resolved_method,
                wire,
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("negative zero")
        ));
    }

    #[test]
    fn strict_json_detects_duplicate_keys_after_unicode_escape_decoding() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let duplicate = br#"{"proof":{"type":"DataIntegrityProof"},"a":1,"\u0061":2}"#;

        assert!(matches!(
            verify_eddsa_jcs_2022_json(
                &request,
                &resolution,
                &resolved_method,
                duplicate,
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("duplicate JSON object member name")
        ));
    }

    #[test]
    fn strict_json_accepts_valid_unicode_surrogate_pair() {
        let value = parse_strict_json(br#""\uD800\uDEAD""#).unwrap();
        assert_eq!(value.as_str(), Some("\u{101ad}"));
    }

    #[test]
    fn rejects_duplicate_wire_object_member_names() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let duplicate = br#"{"proof":{"type":"DataIntegrityProof"},"proof":{"type":"DataIntegrityProof"}}"#;

        assert!(matches!(
            verify_eddsa_jcs_2022_json(
                &request,
                &resolution,
                &resolved_method,
                duplicate,
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("duplicate JSON object member name")
        ));
    }

    #[test]
    fn rejects_unicode_noncharacters_in_wire_values_and_member_names() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);

        let value_noncharacter = br#"{"proof":{"type":"DataIntegrityProof"},"label":"\uFDD0"}"#;
        assert!(matches!(
            verify_eddsa_jcs_2022_json(
                &request,
                &resolution,
                &resolved_method,
                value_noncharacter,
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("Unicode noncharacter")
        ));

        let key_noncharacter = br#"{"proof":{"type":"DataIntegrityProof"},"\uFFFF":"value"}"#;
        assert!(matches!(
            verify_eddsa_jcs_2022_json(
                &request,
                &resolution,
                &resolved_method,
                key_noncharacter,
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("Unicode noncharacter")
        ));
    }

    #[test]
    fn rejects_trailing_wire_data() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let mut document = serde_json::to_vec(&secured_document()).unwrap();
        document.extend_from_slice(br#"null"#);

        assert!(matches!(
            verify_eddsa_jcs_2022_json(
                &request,
                &resolution,
                &resolved_method,
                &document,
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("trailing data")
        ));
    }

    #[test]
    fn strict_wire_entry_point_matches_value_entry_point() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let document = secured_document();

        let from_value = verify_eddsa_jcs_2022(
            &request,
            &resolution,
            &resolved_method,
            &document,
        )
        .unwrap();
        let wire = verify_eddsa_jcs_2022_json(
            &request,
            &resolution,
            &resolved_method,
            &serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();

        assert_eq!(from_value, wire);
    }

    #[test]
    fn wire_json_formatting_does_not_change_the_verified_receipt() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let canonical = serde_jcs::to_string(&secured_document()).unwrap();
        let formatted = serde_json::to_string_pretty(&secured_document()).unwrap();

        let canonical_receipt = verify_eddsa_jcs_2022_json(
            &request,
            &resolution,
            &resolved_method,
            canonical.as_bytes(),
        )
        .unwrap();
        let formatted_receipt = verify_eddsa_jcs_2022_json(
            &request,
            &resolution,
            &resolved_method,
            formatted.as_bytes(),
        )
        .unwrap();

        assert_eq!(canonical_receipt, formatted_receipt);
    }

    #[test]
    fn value_entry_point_rejects_non_ijson_unicode_and_unsafe_integers() {
        let mut noncharacter_document = secured_document();
        noncharacter_document["credentialSubject"]["alumniOf"] =
            Value::String("\u{fdd0}".into());
        assert!(matches!(
            validate_strict_ijson_value(&noncharacter_document),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("Unicode noncharacter")
        ));

        let unsafe_integer: Value =
            serde_json::json!({"value": 9007199254740993u64});
        assert!(matches!(
            validate_strict_ijson_value(&unsafe_integer),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("IEEE-754")
        ));

        let max_integer: Value = serde_json::json!({"value": u64::MAX});
        assert!(matches!(
            validate_strict_ijson_value(&max_integer),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("IEEE-754")
        ));

        let exact_large_integer: Value = serde_json::json!({"value": 1u64 << 53});
        assert!(validate_strict_ijson_value(&exact_large_integer).is_ok());
    }

    #[test]
    fn requires_exact_expected_transformed_document_binding() {
        let mut request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);

        request.expected_transformed_document_digest = None;
        assert!(matches!(
            verify_eddsa_jcs_2022(
                &request,
                &resolution,
                &resolved_method,
                &secured_document(),
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::MissingExpectedTransformedDocumentDigest
            ))
        ));

        let mut request = vector_request();
        request.expected_transformed_document_digest = Some("00".repeat(32));
        assert!(matches!(
            verify_eddsa_jcs_2022(
                &request,
                &resolution,
                &resolved_method,
                &secured_document(),
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::TransformedDocumentDigestMismatch { .. }
            ))
        ));
    }

    #[test]
    fn rejects_document_tampering() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let mut document = secured_document();
        document["credentialSubject"]["alumniOf"] = Value::String("tampered".into());

        assert!(matches!(
            verify_eddsa_jcs_2022(&request, &resolution, &resolved_method, &document),
            Err(SnapshotError::Verification(
                VerificationFailure::CryptographicVerificationFailed
            ))
        ));
    }

    #[test]
    fn rejects_proof_value_tampering() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let mut document = secured_document();
        document["proof"]["proofValue"] =
            Value::String(format!("z{}", bs58::encode([7u8; 64]).into_string()));

        assert!(matches!(
            verify_eddsa_jcs_2022(&request, &resolution, &resolved_method, &document),
            Err(SnapshotError::Verification(
                VerificationFailure::CryptographicVerificationFailed
            ))
        ));
    }

    #[test]
    fn rejects_wrong_proof_context_prefix() {
        let request = vector_request();
        let (resolution, resolved_method) = resolved_vector(&request);
        let mut document = secured_document();
        document["@context"] = serde_json::json!(["https://wrong.example/context"]);

        assert!(matches!(
            verify_eddsa_jcs_2022(&request, &resolution, &resolved_method, &document),
            Err(SnapshotError::Verification(VerificationFailure::Structural(
                _
            )))
        ));
    }

    #[test]
    fn fully_rebound_wrong_public_key_reaches_cryptographic_failure() {
        let request = vector_request();
        let (mut resolution, mut resolved_method) = resolved_vector(&request);
        let alternate = "z6Mkf5rGMoatrSj1f4CyvuHBeXJELe9RPdzo2PKGNCKVtZxP";

        if let ResolvedVerificationMethodMaterial::Multikey {
            public_key_multibase,
        } = &mut resolved_method.material
        {
            *public_key_multibase = alternate.into();
        }

        let mut material_object = Map::new();
        material_object.insert(
            "publicKeyMultibase".into(),
            Value::String(alternate.into()),
        );
        resolved_method.material_digest = crate::verification_method_material_digest(
            &resolved_method.method,
            "Multikey",
            &material_object,
        )
        .unwrap();
        resolution.verification_method_material_digest =
            resolved_method.material_digest.clone();

        assert!(matches!(
            verify_eddsa_jcs_2022(
                &request,
                &resolution,
                &resolved_method,
                &secured_document(),
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::CryptographicVerificationFailed
            ))
        ));
    }

    #[test]
    fn rejects_lifecycle_substitution_before_signature_check() {
        let request = vector_request();
        let (resolution, mut resolved_method) = resolved_vector(&request);
        resolved_method.lifecycle =
            symthaea_epistemic_types::VerificationMethodLifecycle::new(
                Some("2026-10-06T00:00:00Z"),
                Some("2026-10-05T00:30:00Z"),
            )
            .unwrap();

        assert!(matches!(
            verify_eddsa_jcs_2022(
                &request,
                &resolution,
                &resolved_method,
                &secured_document()
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::ResolutionEvidenceMismatch
            ))
        ));
    }

    #[test]
    fn rejects_key_material_substitution_before_signature_check() {
        let request = vector_request();
        let (resolution, mut resolved_method) = resolved_vector(&request);
        if let ResolvedVerificationMethodMaterial::Multikey {
            public_key_multibase,
        } = &mut resolved_method.material
        {
            *public_key_multibase = "z6Mkf5rGMoatrSj1f4CyvuHBeXJELe9RPdzo2PKGNCKVtZxP".into();
        }
        assert!(matches!(
            verify_eddsa_jcs_2022(
                &request,
                &resolution,
                &resolved_method,
                &secured_document()
            ),
            Err(SnapshotError::Verification(
                VerificationFailure::ResolutionEvidenceMismatch
            ))
        ));
    }
}
