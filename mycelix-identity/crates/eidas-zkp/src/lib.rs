// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! eIDAS 2.0 compliant ZKP selective disclosure for EUDI Wallet.

use serde::{Deserialize, Serialize};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Cryptosuite identifier for DASTARK proofs in W3C Data Integrity.
pub const CRYPTOSUITE_DASTARK: &str = "dastark-2026";

/// Cryptosuite for Dilithium5 post-quantum signatures.
pub const CRYPTOSUITE_DILITHIUM5: &str = "dilithium5-2026";

/// An eIDAS-compliant verifiable credential with selective-disclosure metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EidasCredential {
    /// W3C VC 2.0 context.
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Credential ID.
    pub id: String,
    /// Types (must include "VerifiableCredential").
    #[serde(rename = "type")]
    pub types: Vec<String>,
    /// Issuer DID.
    pub issuer: String,
    /// Issuance date (ISO 8601).
    #[serde(rename = "issuanceDate")]
    pub issuance_date: String,
    /// Expiration date.
    #[serde(rename = "expirationDate")]
    pub expiration_date: Option<String>,
    /// Subject with claims.
    #[serde(rename = "credentialSubject")]
    pub credential_subject: EidasSubject,
    /// Issuer proof.
    pub proof: EidasProof,
}

/// W3C-style verifiable presentation for holder-authenticated disclosure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EidasPresentation {
    /// W3C VC 2.0 context.
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Presentation ID.
    pub id: String,
    /// Types (must include "VerifiablePresentation").
    #[serde(rename = "type")]
    pub types: Vec<String>,
    /// Holder DID.
    pub holder: String,
    /// Redacted credentials included in the presentation.
    #[serde(rename = "verifiableCredential")]
    pub verifiable_credential: Vec<EidasCredential>,
    /// Holder authentication proof.
    pub proof: EidasProof,
}

/// Credential subject with typed claims for eIDAS.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EidasSubject {
    /// Subject DID.
    pub id: String,
    /// Claims (key-value pairs).
    pub claims: serde_json::Value,
    /// Merkle root of all claims (for selective disclosure).
    #[serde(rename = "claimsMerkleRoot")]
    pub claims_merkle_root: Option<String>,
}

/// DASTARK or Dilithium proof in W3C Data Integrity format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EidasProof {
    /// Must be "DataIntegrityProof".
    #[serde(rename = "type")]
    pub proof_type: String,
    /// "dastark-2026" for issuer proofs, "dilithium5-2026" for holder auth.
    pub cryptosuite: String,
    /// ISO 8601 creation time.
    pub created: String,
    /// DID URL of verification method.
    #[serde(rename = "verificationMethod")]
    pub verification_method: String,
    /// "assertionMethod" for issuance, "authentication" for presentation.
    #[serde(rename = "proofPurpose")]
    pub proof_purpose: String,
    /// Multibase-encoded proof value.
    #[serde(rename = "proofValue")]
    pub proof_value: String,
    /// Selective disclosure metadata for redacted credentials.
    #[serde(
        rename = "selectiveDisclosure",
        skip_serializing_if = "Option::is_none"
    )]
    pub selective_disclosure: Option<SelectiveDisclosureMetadata>,
}

/// Metadata about which claims are disclosed vs ZK-proven.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectiveDisclosureMetadata {
    /// Claims revealed in plaintext (key names).
    pub disclosed_claims: Vec<String>,
    /// Claims proven via ZKP without revealing value.
    pub proven_claims: Vec<ProvenClaim>,
    /// Merkle proofs for disclosed claims.
    pub merkle_proofs: Vec<MerkleClaimProof>,
}

/// A claim proven via ZKP without revealing its value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenClaim {
    /// Claim key (e.g., "age", "salary_bracket").
    pub claim_key: String,
    /// What property is proven (e.g., "range", "membership", "equality").
    pub proof_type: String,
    /// Human-readable description (e.g., "age >= 18").
    pub description: String,
    /// STARK proof bytes (multibase z-encoded).
    pub proof_value: String,
}

/// Merkle proof for a disclosed claim.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleClaimProof {
    /// Claim key.
    pub claim_key: String,
    /// Merkle path (hashes from leaf to root).
    pub merkle_path: Vec<String>,
    /// Leaf hash.
    pub leaf_hash: String,
}

/// Create an eIDAS-compliant credential with DASTARK proof.
pub fn create_eidas_credential(
    issuer_did: &str,
    subject_did: &str,
    claims: serde_json::Value,
    proof_value: &str,
) -> EidasCredential {
    let now = now_iso_8601();
    EidasCredential {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".to_string(),
            "https://mycelix.net/ns/eidas/v1".to_string(),
        ],
        id: format!("urn:uuid:{}", uuid_v4()),
        types: vec![
            "VerifiableCredential".to_string(),
            "EidasCredential".to_string(),
        ],
        issuer: issuer_did.to_string(),
        issuance_date: now.clone(),
        expiration_date: None,
        credential_subject: EidasSubject {
            id: subject_did.to_string(),
            claims,
            claims_merkle_root: None,
        },
        proof: EidasProof {
            proof_type: "DataIntegrityProof".to_string(),
            cryptosuite: CRYPTOSUITE_DASTARK.to_string(),
            created: now,
            verification_method: format!("{}#dastark-key-1", issuer_did),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: proof_value.to_string(),
            selective_disclosure: None,
        },
    }
}

/// Create a selective disclosure presentation from a credential.
pub fn create_selective_presentation(
    credential: &EidasCredential,
    disclosed_keys: &[&str],
    proven_claims: Vec<ProvenClaim>,
    holder_proof_value: &str,
) -> EidasPresentation {
    let disclosed_claims = filter_claims(&credential.credential_subject.claims, disclosed_keys);
    let selective_disclosure = SelectiveDisclosureMetadata {
        disclosed_claims: disclosed_keys.iter().map(|key| key.to_string()).collect(),
        proven_claims,
        merkle_proofs: vec![],
    };
    let holder = credential.credential_subject.id.clone();
    let now = now_iso_8601();

    let redacted_credential = EidasCredential {
        context: credential.context.clone(),
        id: credential.id.clone(),
        types: credential.types.clone(),
        issuer: credential.issuer.clone(),
        issuance_date: credential.issuance_date.clone(),
        expiration_date: credential.expiration_date.clone(),
        credential_subject: EidasSubject {
            id: credential.credential_subject.id.clone(),
            claims: disclosed_claims,
            claims_merkle_root: credential.credential_subject.claims_merkle_root.clone(),
        },
        proof: EidasProof {
            selective_disclosure: Some(selective_disclosure),
            ..credential.proof.clone()
        },
    };

    EidasPresentation {
        context: credential.context.clone(),
        id: format!("urn:uuid:{}", uuid_v4()),
        types: vec![
            "VerifiablePresentation".to_string(),
            "EidasPresentation".to_string(),
        ],
        holder: holder.clone(),
        verifiable_credential: vec![redacted_credential],
        proof: EidasProof {
            proof_type: "DataIntegrityProof".to_string(),
            cryptosuite: CRYPTOSUITE_DILITHIUM5.to_string(),
            created: now,
            verification_method: format!("{}#dilithium5-key-1", holder),
            proof_purpose: "authentication".to_string(),
            proof_value: holder_proof_value.to_string(),
            selective_disclosure: None,
        },
    }
}

fn filter_claims(claims: &serde_json::Value, disclosed_keys: &[&str]) -> serde_json::Value {
    let Some(object) = claims.as_object() else {
        return serde_json::Value::Null;
    };

    let mut filtered = serde_json::Map::new();
    for key in disclosed_keys {
        if let Some(value) = object.get(*key) {
            filtered.insert((*key).to_string(), value.clone());
        }
    }
    serde_json::Value::Object(filtered)
}

// Simple UUID v4 (no external dep)
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        t.as_secs() as u32,
        (t.subsec_nanos() >> 16) as u16,
        (t.subsec_nanos() & 0x0fff) as u16,
        0x8000 | (t.as_millis() as u16 & 0x3fff),
        t.as_nanos() as u64 & 0x0000_ffff_ffff_ffff
    )
}

fn now_iso_8601() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::Duration;

    #[test]
    fn test_create_eidas_credential() {
        let credential = create_eidas_credential(
            "did:mycelix:issuer001",
            "did:mycelix:holder001",
            serde_json::json!({
                "givenName": "Alice",
                "familyName": "Smith",
                "dateOfBirth": "1990-03-15",
                "nationality": "NL",
                "age": 36
            }),
            "z3DASTARK_PROOF_BASE58...",
        );

        assert_eq!(credential.proof.cryptosuite, CRYPTOSUITE_DASTARK);
        assert_eq!(credential.proof.proof_type, "DataIntegrityProof");
        assert!(credential.types.contains(&"EidasCredential".to_string()));
        assert!(credential
            .context
            .contains(&"https://www.w3.org/ns/credentials/v2".to_string()));
        assert_ne!(credential.issuance_date, "2026-04-07T00:00:00Z");
        assert!(credential.issuance_date.ends_with('Z'));
        assert!(credential.proof.created.ends_with('Z'));
    }

    #[test]
    fn test_timestamps_are_parseable_and_recent() {
        let credential = create_eidas_credential(
            "did:mycelix:issuer001",
            "did:mycelix:holder001",
            serde_json::json!({ "age": 36 }),
            "z3PROOF",
        );

        let issuance = OffsetDateTime::parse(&credential.issuance_date, &Rfc3339).unwrap();
        let created = OffsetDateTime::parse(&credential.proof.created, &Rfc3339).unwrap();
        let now = OffsetDateTime::now_utc();

        assert!(now - issuance < Duration::seconds(5));
        assert!(now - created < Duration::seconds(5));
    }

    #[test]
    fn test_selective_presentation() {
        let credential = create_eidas_credential(
            "did:mycelix:issuer001",
            "did:mycelix:holder001",
            serde_json::json!({
                "givenName": "Alice",
                "familyName": "Smith",
                "dateOfBirth": "1990-03-15",
                "nationality": "NL",
                "age": 36
            }),
            "z3ISSUER_PROOF...",
        );

        let presentation = create_selective_presentation(
            &credential,
            &["nationality"],
            vec![ProvenClaim {
                claim_key: "age".to_string(),
                proof_type: "range".to_string(),
                description: "age >= 18".to_string(),
                proof_value: "z3STARK_AGE_RANGE_PROOF...".to_string(),
            }],
            "z3HOLDER_AUTH_PROOF...",
        );

        assert_eq!(presentation.holder, "did:mycelix:holder001");
        assert_eq!(presentation.proof.cryptosuite, CRYPTOSUITE_DILITHIUM5);
        assert_eq!(presentation.proof.proof_purpose, "authentication");

        let disclosed = presentation.verifiable_credential[0]
            .credential_subject
            .claims
            .as_object()
            .unwrap();
        assert!(disclosed.contains_key("nationality"));
        assert!(!disclosed.contains_key("givenName"));
        assert!(!disclosed.contains_key("dateOfBirth"));
        assert!(!disclosed.contains_key("age"));

        let selective_disclosure = presentation.verifiable_credential[0]
            .proof
            .selective_disclosure
            .as_ref()
            .unwrap();
        assert_eq!(selective_disclosure.disclosed_claims, vec!["nationality"]);
        assert_eq!(selective_disclosure.proven_claims.len(), 1);
        assert_eq!(selective_disclosure.proven_claims[0].claim_key, "age");
        assert_eq!(
            selective_disclosure.proven_claims[0].description,
            "age >= 18"
        );
    }

    #[test]
    fn test_presentation_serializes_as_vp_shape() {
        let credential = create_eidas_credential(
            "did:mycelix:issuer001",
            "did:mycelix:holder001",
            serde_json::json!({
                "nationality": "NL",
                "age": 36
            }),
            "z3ISSUER_PROOF...",
        );

        let presentation = create_selective_presentation(
            &credential,
            &["nationality"],
            vec![ProvenClaim {
                claim_key: "age".to_string(),
                proof_type: "range".to_string(),
                description: "age >= 18".to_string(),
                proof_value: "z3AGE_PROOF".to_string(),
            }],
            "z3HOLDER_AUTH",
        );

        let json = serde_json::to_value(&presentation).unwrap();
        assert!(json.get("holder").is_some());
        assert!(json.get("verifiableCredential").is_some());
        assert!(json.get("issuer").is_none());
        assert!(json.get("issuanceDate").is_none());
        assert!(json.get("credentialSubject").is_none());

        let top_proof = json.get("proof").unwrap();
        assert_eq!(
            top_proof.get("cryptosuite").unwrap(),
            CRYPTOSUITE_DILITHIUM5
        );
        assert!(top_proof.get("selectiveDisclosure").is_none());

        let embedded = json
            .get("verifiableCredential")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .clone();
        assert_eq!(
            embedded.get("proof").unwrap().get("cryptosuite").unwrap(),
            CRYPTOSUITE_DASTARK
        );
        assert!(embedded
            .get("proof")
            .unwrap()
            .get("selectiveDisclosure")
            .is_some());
    }

    #[test]
    fn test_cryptosuite_identifiers() {
        assert_eq!(CRYPTOSUITE_DASTARK, "dastark-2026");
        assert_eq!(CRYPTOSUITE_DILITHIUM5, "dilithium5-2026");
    }
}
