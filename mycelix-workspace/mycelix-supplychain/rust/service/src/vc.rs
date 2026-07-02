// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifiable Credential operations
//!
//! This module provides signing and verification of Verifiable Credentials
//! using EdDSA (Ed25519) signatures in JWT format.

use anyhow::{Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use claim_model::SupplyEventVC;
use crypto::KeyPair;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to VC JWT verification
#[derive(Debug, Error)]
pub enum VcVerificationError {
    #[error("Invalid JWT format: expected 3 parts separated by dots")]
    InvalidJwtFormat,
    #[error("Unsupported algorithm: {0}. Only EdDSA is supported")]
    UnsupportedAlgorithm(String),
    #[error("Invalid header encoding: {0}")]
    InvalidHeaderEncoding(String),
    #[error("Invalid payload encoding: {0}")]
    InvalidPayloadEncoding(String),
    #[error("Invalid signature encoding: {0}")]
    InvalidSignatureEncoding(String),
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    #[error("Invalid issuer DID format: {0}")]
    InvalidIssuerDid(String),
    #[error("Could not extract public key from issuer DID")]
    PublicKeyExtractionFailed,
    #[error("VC validation failed: {0}")]
    VcValidationFailed(String),
    #[error("VC expired at {0}")]
    VcExpired(String),
}

/// JWT header for VC verification
#[derive(Debug, Serialize, Deserialize)]
pub struct JwtHeader {
    pub alg: String,
    pub typ: String,
}

/// Sign a VC and return a JWT
pub fn sign_vc(keypair: &KeyPair, vc: &SupplyEventVC) -> Result<String> {
    // Validate VC before signing
    vc.validate()
        .map_err(|e| anyhow::anyhow!("VC validation failed: {}", e))?;

    // Serialize VC to JSON
    let vc_json = serde_json::to_value(vc)?;

    // Create signed JWT
    let jwt = crypto::create_vc_jwt(keypair, &vc_json)?;

    Ok(jwt)
}

/// Verify a VC JWT signature and return the parsed VC
///
/// This function:
/// 1. Parses the JWT into header, payload, and signature parts
/// 2. Validates the JWT header (must use EdDSA algorithm)
/// 3. Extracts the issuer's public key from the DID
/// 4. Verifies the Ed25519 signature
/// 5. Validates the VC structure and expiration
/// 6. Returns the verified SupplyEventVC
pub fn verify_vc_jwt(jwt: &str) -> Result<SupplyEventVC> {
    // Step 1: Split JWT into parts
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return Err(VcVerificationError::InvalidJwtFormat.into());
    }

    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let signature_b64 = parts[2];

    // Step 2: Decode and validate header
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|e| VcVerificationError::InvalidHeaderEncoding(e.to_string()))?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes)
        .map_err(|e| VcVerificationError::InvalidHeaderEncoding(e.to_string()))?;

    // Validate algorithm is EdDSA
    if header.alg != "EdDSA" {
        return Err(VcVerificationError::UnsupportedAlgorithm(header.alg).into());
    }

    // Step 3: Decode payload
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|e| VcVerificationError::InvalidPayloadEncoding(e.to_string()))?;
    let vc: SupplyEventVC = serde_json::from_slice(&payload_bytes)
        .map_err(|e| VcVerificationError::InvalidPayloadEncoding(e.to_string()))?;

    // Step 4: Extract public key from issuer DID
    let public_key = extract_public_key_from_did(&vc.issuer)?;

    // Step 5: Decode signature
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(signature_b64)
        .map_err(|e| VcVerificationError::InvalidSignatureEncoding(e.to_string()))?;

    if signature_bytes.len() != 64 {
        return Err(VcVerificationError::InvalidSignatureEncoding(format!(
            "Expected 64 bytes, got {}",
            signature_bytes.len()
        ))
        .into());
    }

    let signature = Signature::from_bytes(
        signature_bytes
            .as_slice()
            .try_into()
            .map_err(|_| VcVerificationError::InvalidSignatureEncoding("Invalid length".into()))?,
    );

    // Step 6: Verify signature
    let message = format!("{}.{}", header_b64, payload_b64);
    crypto::verify_signature(&public_key, message.as_bytes(), &signature)
        .map_err(|_| VcVerificationError::SignatureVerificationFailed)?;

    // Step 7: Validate VC structure
    vc.validate()
        .map_err(|e| VcVerificationError::VcValidationFailed(e.to_string()))?;

    // Step 8: Check expiration
    if let Some(expiration) = &vc.expiration_date {
        if *expiration < chrono::Utc::now() {
            return Err(VcVerificationError::VcExpired(expiration.to_rfc3339()).into());
        }
    }

    Ok(vc)
}

/// Extract Ed25519 public key from a DID
///
/// Supports:
/// - did:key:<hex-encoded-public-key> (as generated by crypto crate)
fn extract_public_key_from_did(did: &str) -> Result<VerifyingKey> {
    // Parse DID format: did:key:<hex-encoded-public-key>
    if !did.starts_with("did:key:") {
        return Err(VcVerificationError::InvalidIssuerDid(format!(
            "Expected 'did:key:' prefix, got: {}",
            did
        ))
        .into());
    }

    let key_part = &did[8..]; // Skip "did:key:"

    // Decode hex-encoded public key
    let key_bytes = hex::decode(key_part).map_err(|e| {
        VcVerificationError::InvalidIssuerDid(format!("Invalid hex encoding: {}", e))
    })?;

    if key_bytes.len() != 32 {
        return Err(VcVerificationError::InvalidIssuerDid(format!(
            "Expected 32-byte public key, got {} bytes",
            key_bytes.len()
        ))
        .into());
    }

    let key_array: [u8; 32] = key_bytes.try_into().map_err(|_| {
        VcVerificationError::PublicKeyExtractionFailed
    })?;

    VerifyingKey::from_bytes(&key_array)
        .map_err(|_| VcVerificationError::PublicKeyExtractionFailed.into())
}

/// Verify a VC JWT with an explicit public key (useful when DID resolution is external)
pub fn verify_vc_jwt_with_key(jwt: &str, public_key: &VerifyingKey) -> Result<SupplyEventVC> {
    // Step 1: Split JWT into parts
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return Err(VcVerificationError::InvalidJwtFormat.into());
    }

    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let signature_b64 = parts[2];

    // Step 2: Decode and validate header
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|e| VcVerificationError::InvalidHeaderEncoding(e.to_string()))?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes)
        .map_err(|e| VcVerificationError::InvalidHeaderEncoding(e.to_string()))?;

    if header.alg != "EdDSA" {
        return Err(VcVerificationError::UnsupportedAlgorithm(header.alg).into());
    }

    // Step 3: Decode payload
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|e| VcVerificationError::InvalidPayloadEncoding(e.to_string()))?;
    let vc: SupplyEventVC = serde_json::from_slice(&payload_bytes)
        .map_err(|e| VcVerificationError::InvalidPayloadEncoding(e.to_string()))?;

    // Step 4: Decode and verify signature
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(signature_b64)
        .map_err(|e| VcVerificationError::InvalidSignatureEncoding(e.to_string()))?;

    if signature_bytes.len() != 64 {
        return Err(VcVerificationError::InvalidSignatureEncoding(format!(
            "Expected 64 bytes, got {}",
            signature_bytes.len()
        ))
        .into());
    }

    let signature = Signature::from_bytes(
        signature_bytes
            .as_slice()
            .try_into()
            .map_err(|_| VcVerificationError::InvalidSignatureEncoding("Invalid length".into()))?,
    );

    let message = format!("{}.{}", header_b64, payload_b64);
    crypto::verify_signature(public_key, message.as_bytes(), &signature)
        .map_err(|_| VcVerificationError::SignatureVerificationFailed)?;

    // Step 5: Validate VC
    vc.validate()
        .map_err(|e| VcVerificationError::VcValidationFailed(e.to_string()))?;

    // Step 6: Check expiration
    if let Some(expiration) = &vc.expiration_date {
        if *expiration < chrono::Utc::now() {
            return Err(VcVerificationError::VcExpired(expiration.to_rfc3339()).into());
        }
    }

    Ok(vc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use claim_model::{CredentialSubject, EventType, Facility};

    fn create_test_vc(keypair: &KeyPair) -> SupplyEventVC {
        SupplyEventVC {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            vc_type: vec!["VerifiableCredential".to_string()],
            issuer: keypair.did(),
            issuance_date: Utc::now(),
            expiration_date: None,
            credential_subject: CredentialSubject {
                event_type: EventType::Produced,
                product_id: "SKU-001".to_string(),
                batch_id: "BATCH-001".to_string(),
                prev_batch_ids: None,
                quantity: 100.0,
                unit: "kg".to_string(),
                facility: Facility {
                    id: "FAC-001".to_string(),
                    name: "Test Facility".to_string(),
                    location: None,
                },
                timestamp: Utc::now(),
                shipment: None,
                certification: None,
                metadata: None,
            },
            proof: None,
        }
    }

    #[test]
    fn test_sign_and_verify_vc() {
        let keypair = KeyPair::generate();
        let vc = create_test_vc(&keypair);

        // Sign the VC
        let jwt = sign_vc(&keypair, &vc).expect("Signing should succeed");

        // Verify the JWT
        let verified_vc = verify_vc_jwt(&jwt).expect("Verification should succeed");

        // Check fields match
        assert_eq!(verified_vc.issuer, vc.issuer);
        assert_eq!(verified_vc.credential_subject.batch_id, vc.credential_subject.batch_id);
        assert_eq!(verified_vc.credential_subject.event_type, vc.credential_subject.event_type);
    }

    #[test]
    fn test_verify_with_explicit_key() {
        let keypair = KeyPair::generate();
        let vc = create_test_vc(&keypair);

        let jwt = sign_vc(&keypair, &vc).expect("Signing should succeed");

        // Verify with explicit public key
        let verified_vc = verify_vc_jwt_with_key(&jwt, &keypair.public_key())
            .expect("Verification should succeed");

        assert_eq!(verified_vc.issuer, vc.issuer);
    }

    #[test]
    fn test_tampered_jwt_fails() {
        let keypair = KeyPair::generate();
        let vc = create_test_vc(&keypair);

        let jwt = sign_vc(&keypair, &vc).expect("Signing should succeed");

        // Tamper with the JWT payload
        let parts: Vec<&str> = jwt.split('.').collect();
        let tampered_jwt = format!("{}.{}.{}", parts[0], "dGFtcGVyZWQ", parts[2]);

        // Verification should fail
        let result = verify_vc_jwt(&tampered_jwt);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let keypair1 = KeyPair::generate();
        let keypair2 = KeyPair::generate();
        let vc = create_test_vc(&keypair1);

        let jwt = sign_vc(&keypair1, &vc).expect("Signing should succeed");

        // Verify with wrong key should fail
        let result = verify_vc_jwt_with_key(&jwt, &keypair2.public_key());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_jwt_format() {
        let result = verify_vc_jwt("not.a.valid.jwt.format");
        assert!(result.is_err());

        let result = verify_vc_jwt("only-one-part");
        assert!(result.is_err());
    }

    #[test]
    fn test_expired_vc_fails() {
        let keypair = KeyPair::generate();
        let mut vc = create_test_vc(&keypair);

        // Set expiration in the past
        vc.expiration_date = Some(Utc::now() - chrono::Duration::hours(1));

        let jwt = sign_vc(&keypair, &vc).expect("Signing should succeed");

        // Verification should fail due to expiration
        let result = verify_vc_jwt(&jwt);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expired"));
    }

    #[test]
    fn test_extract_public_key_from_did() {
        let keypair = KeyPair::generate();
        let did = keypair.did();

        let extracted_key = extract_public_key_from_did(&did).expect("Key extraction should succeed");
        assert_eq!(extracted_key.to_bytes(), keypair.public_key().to_bytes());
    }

    #[test]
    fn test_invalid_did_format() {
        // Wrong prefix
        let result = extract_public_key_from_did("did:web:example.com");
        assert!(result.is_err());

        // Invalid hex
        let result = extract_public_key_from_did("did:key:not-valid-hex");
        assert!(result.is_err());

        // Wrong length
        let result = extract_public_key_from_did("did:key:deadbeef");
        assert!(result.is_err());
    }
}
