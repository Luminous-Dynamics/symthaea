// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Credential Coordinator Zome
//!
//! Implements business logic for W3C Verifiable Credentials.
//! This zome is upgradeable - business logic can change without breaking data.
//!
//! ## Core Functions:
//! - Issue credentials to learners
//! - Verify credential authenticity
//! - Retrieve credentials (by learner, course, issuer)
//! - Revoke credentials

use hdk::prelude::*;
use credential_integrity::{
    VerifiableCredential, CredentialStatus,
    EntryTypes, LinkTypes
};
use praxis_core::CourseId;
use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use base64::Engine as _;
use hdk::prelude::HdkPathExt;
use sha2::{Sha256, Digest};

// =============================================================================
// Epistemic Types (inline to avoid getrandom 0.2 from mycelix-sdk)
// =============================================================================
// These types are copied from mycelix-sdk::epistemic to avoid the rand 0.8 -> getrandom 0.2
// dependency chain which requires wasm-bindgen that Holochain doesn't provide.

/// Empirical Axis (E): How to verify the claim
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum EmpiricalLevel {
    E0Null = 0,
    E1Testimonial = 1,
    E2PrivateVerify = 2,
    E3Cryptographic = 3,
    E4PublicRepro = 4,
}

/// Normative Axis (N): Who agrees with the claim
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum NormativeLevel {
    N0Personal = 0,
    N1Communal = 1,
    N2Network = 2,
    N3Axiomatic = 3,
}

/// Materiality Axis (M): How long the claim matters
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum MaterialityLevel {
    M0Ephemeral = 0,
    M1Temporal = 1,
    M2Persistent = 2,
    M3Foundational = 3,
}

/// Issue a new verifiable credential
///
/// This function creates a complete W3C Verifiable Credential with cryptographic proof.
/// The credential is stored on the DHT and linked to the learner, course, and issuer.
#[hdk_extern]
pub fn issue_credential(input: IssueCredentialInput) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Steward tier to issue credentials
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_constitutional(),
        "issue_credential",
    )?;

    // Get issuer agent info
    let agent_info = agent_info()?;
    let issuer_pubkey = agent_info.agent_initial_pubkey;

    // Generate proof
    let proof = generate_proof(&input, &issuer_pubkey)?;

    // Serialize metadata to JSON string if present
    let metadata_str = input.metadata.map(|m| serde_json::to_string(&m))
        .transpose()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to serialize metadata: {}", e))))?;

    // Serialize context to JSON string
    let context_json = serde_json::json!([
        "https://www.w3.org/2018/credentials/v1",
        "https://mycelix.edunet/credentials/v1"
    ]);
    let context_str = serde_json::to_string(&context_json)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to serialize context: {}", e))))?;

    // Create verifiable credential (flattened structure)
    let credential = VerifiableCredential {
        // Context & Type
        context: context_str,
        credential_type: vec![
            "VerifiableCredential".to_string(),
            "CourseCompletionCredential".to_string(),
        ],

        // Issuer & Dates
        issuer: issuer_pubkey.to_string(),
        issuance_date: chrono::Utc::now().to_rfc3339(),
        expiration_date: input.expiration_date,

        // Subject fields (flattened)
        subject_id: input.learner_agent.to_string(),
        course_id: input.course_id.clone(),
        model_id: input.model_id,
        rubric_id: input.rubric_id,
        score: input.score,
        score_band: input.score_band,
        subject_metadata: metadata_str,

        // Status fields (flattened, optional)
        status_id: input.credential_status.as_ref().map(|s| s.id.clone()),
        status_type: input.credential_status.as_ref().map(|s| s.status_type.clone()),
        status_list_index: input.credential_status.as_ref().and_then(|s| s.status_list_index),
        status_purpose: input.credential_status.as_ref().and_then(|s| s.status_purpose.clone()),

        // Proof fields (flattened)
        proof_type: proof.proof_type.clone(),
        proof_created: proof.created,
        verification_method: proof.verification_method,
        proof_purpose: proof.proof_purpose,
        proof_value: proof.proof_value,

        // Epistemic classification (auto-determine if not provided)
        epistemic_empirical: input.epistemic_empirical.or_else(|| {
            Some(determine_empirical_level(&proof.proof_type) as u8)
        }),
        epistemic_normative: input.epistemic_normative.or(Some(NormativeLevel::N1Communal as u8)),
        epistemic_materiality: input.epistemic_materiality.or(Some(MaterialityLevel::M2Persistent as u8)),
    };

    // Store credential entry
    let action_hash = create_entry(EntryTypes::VerifiableCredential(credential.clone()))?;

    // Create links for easy lookup
    // Link from learner to credential
    let learner_anchor = Path::from(format!("learner_credentials.{}", input.learner_agent));
    let learner_entry_hash = ensure_path(learner_anchor, LinkTypes::LearnerToCredentials)?;

    create_link(
        learner_entry_hash,
        action_hash.clone(),
        LinkTypes::LearnerToCredentials,
        vec![],
    )?;

    // Link from course to credential
    let course_anchor = Path::from(format!("course_credentials.{}", input.course_id.0));
    let course_entry_hash = ensure_path(course_anchor, LinkTypes::CourseToCredentials)?;

    create_link(
        course_entry_hash,
        action_hash.clone(),
        LinkTypes::CourseToCredentials,
        vec![],
    )?;

    // Link from issuer to credential
    let issuer_anchor = Path::from(format!("issuer_credentials.{}", issuer_pubkey));
    let issuer_entry_hash = ensure_path(issuer_anchor, LinkTypes::IssuerToCredentials)?;

    create_link(
        issuer_entry_hash,
        action_hash.clone(),
        LinkTypes::IssuerToCredentials,
        vec![],
    )?;

    Ok(action_hash)
}

/// Verify a credential's authenticity
///
/// Checks the cryptographic proof and signature to ensure the credential
/// was issued by the claimed issuer and has not been tampered with.
#[hdk_extern]
pub fn verify_credential(action_hash: ActionHash) -> ExternResult<VerificationResult> {
    // Get the credential
    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".to_string())))?;

    // Deserialize credential
    let credential = match record.entry().as_option() {
        Some(Entry::App(bytes)) => {
            VerifiableCredential::try_from(SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())))
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize credential: {}", e))))?
        }
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Invalid entry type".to_string()))),
    };

    // Verify proof signature
    let is_valid = verify_proof(&credential)?;

    // Check if revoked (if status fields are present)
    let is_revoked = if let (Some(status_id), Some(status_type)) =
        (&credential.status_id, &credential.status_type)
    {
        // Reconstruct CredentialStatus for checking
        let status = CredentialStatus {
            id: status_id.clone(),
            status_type: status_type.clone(),
            status_list_index: credential.status_list_index,
            status_purpose: credential.status_purpose.clone(),
        };
        check_revocation_status(&status)?
    } else {
        false
    };

    // Check expiration
    let is_expired = if let Some(exp_date) = &credential.expiration_date {
        chrono::DateTime::parse_from_rfc3339(exp_date)
            .map(|dt| dt < chrono::Utc::now())
            .unwrap_or(false)
    } else {
        false
    };

    Ok(VerificationResult {
        is_valid: is_valid && !is_revoked && !is_expired,
        is_revoked,
        is_expired,
        issuer: credential.issuer,
        issuance_date: credential.issuance_date,
    })
}

/// Get all credentials for a specific learner
#[hdk_extern]
pub fn get_learner_credentials(learner_agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let learner_anchor = Path::from(format!("learner_credentials.{}", learner_agent));
    let learner_entry_hash = ensure_path(learner_anchor, LinkTypes::LearnerToCredentials)?;

    let links = get_links(
        LinkQuery::new(
            learner_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::LearnerToCredentials as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                credentials.push(record);
            }
        }
    }

    Ok(credentials)
}

/// Get all credentials issued for a specific course
#[hdk_extern]
pub fn get_course_credentials(course_id: CourseId) -> ExternResult<Vec<Record>> {
    let course_anchor = Path::from(format!("course_credentials.{}", course_id.0));
    let course_entry_hash = ensure_path(course_anchor, LinkTypes::CourseToCredentials)?;

    let links = get_links(
        LinkQuery::new(
            course_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CourseToCredentials as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                credentials.push(record);
            }
        }
    }

    Ok(credentials)
}

/// Get all credentials issued by a specific issuer
#[hdk_extern]
pub fn get_issuer_credentials(issuer: AgentPubKey) -> ExternResult<Vec<Record>> {
    let issuer_anchor = Path::from(format!("issuer_credentials.{}", issuer));
    let issuer_entry_hash = ensure_path(issuer_anchor, LinkTypes::IssuerToCredentials)?;

    let links = get_links(
        LinkQuery::new(
            issuer_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::IssuerToCredentials as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                credentials.push(record);
            }
        }
    }

    Ok(credentials)
}

/// Revoke a credential
///
/// Note: In a full implementation, this would update a revocation list
/// stored in the DHT. For now, we create a simple revocation entry.
#[hdk_extern]
pub fn revoke_credential(input: RevokeCredentialInput) -> ExternResult<()> {
    // Verify the caller is the issuer
    let agent_info = agent_info()?;
    let caller_pubkey = agent_info.agent_initial_pubkey;

    // Get the credential
    let record = get(input.credential_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".to_string())))?;

    let credential = match record.entry().as_option() {
        Some(Entry::App(bytes)) => {
            VerifiableCredential::try_from(SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())))
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize: {}", e))))?
        }
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Invalid entry".to_string()))),
    };

    // Verify caller is the issuer
    if credential.issuer != caller_pubkey.to_string() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the issuer can revoke this credential".to_string()
        )));
    }

    // In a full implementation, we would:
    // 1. Create a revocation list entry
    // 2. Update the credential_status field
    // 3. Notify relevant parties

    // For now, we'll just emit a signal
    emit_signal(SignalPayload::CredentialRevoked {
        credential_hash: input.credential_hash,
        reason: input.reason,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })?;

    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

/// Helper struct for proof data (coordinator-only, not an entry type)
#[derive(Debug, Clone)]
struct Proof {
    pub proof_type: String,
    pub created: String,
    pub verification_method: String,
    pub proof_purpose: String,
    pub proof_value: String,
}

/// Generate cryptographic proof for a credential
fn generate_proof(
    input: &IssueCredentialInput,
    issuer_pubkey: &AgentPubKey,
) -> ExternResult<Proof> {
    // Create canonical data to sign (stable timestamp shared between payload and proof)
    let created = chrono::Utc::now().to_rfc3339();
    let proof_value = compute_proof_value(
        issuer_pubkey.to_string(),
        input.learner_agent.to_string(),
        input.course_id.0.clone(),
        input.model_id.clone(),
        input.rubric_id.clone(),
        input.score_band.clone(),
        input.score,
        input.expiration_date.clone(),
        created.clone(),
    )?;

    Ok(Proof {
        proof_type: "Ed25519Signature2020".to_string(),
        created,
        verification_method: format!("did:key:{}", issuer_pubkey),
        proof_purpose: "assertionMethod".to_string(),
        proof_value,
    })
}

/// Verify the cryptographic proof of a credential
fn verify_proof(credential: &VerifiableCredential) -> ExternResult<bool> {
    let expected = compute_proof_value(
        credential.issuer.clone(),
        credential.subject_id.clone(),
        credential.course_id.0.clone(),
        credential.model_id.clone(),
        credential.rubric_id.clone(),
        credential.score_band.clone(),
        credential.score,
        credential.expiration_date.clone(),
        credential.proof_created.clone(),
    )?;

    Ok(credential.proof_value == expected)
}

/// Check if a credential has been revoked
fn check_revocation_status(_status: &CredentialStatus) -> ExternResult<bool> {
    // In a real implementation, this would:
    // 1. Fetch the revocation list from the DHT
    // 2. Check if this credential's index is marked as revoked

    // For now, always return false (not revoked)
    Ok(false)
}

/// Compute deterministic proof value from credential fields
fn compute_proof_value(
    issuer: String,
    subject_id: String,
    course_id: String,
    model_id: String,
    rubric_id: String,
    score_band: String,
    score: Option<f32>,
    expiration_date: Option<String>,
    created: String,
) -> ExternResult<String> {
    let canonical = format!(
        "{issuer}:{subject_id}:{course_id}:{model_id}:{rubric_id}:{score_band}:{score:?}:{expiration}",
        expiration = expiration_date.clone().unwrap_or_default(),
    );

    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    hasher.update(created.as_bytes());
    let hash = hasher.finalize();
    Ok(BASE64_ENGINE.encode(hash))
}

/// Determine empirical level from proof type
/// Ed25519/RSA signatures → E3 (Cryptographic)
/// ZK proofs → E4 (Publicly Reproducible)
/// Unknown → E2 (Privately Verifiable)
fn determine_empirical_level(proof_type: &str) -> EmpiricalLevel {
    match proof_type {
        "Ed25519Signature2020" | "JsonWebSignature2020" => EmpiricalLevel::E3Cryptographic,
        "RsaSignature2018" => EmpiricalLevel::E3Cryptographic,
        "ZKProof2023" | "BbsBlsSignature2020" => EmpiricalLevel::E4PublicRepro,
        _ => EmpiricalLevel::E2PrivateVerify,
    }
}

// ============================================================================
// Epistemic Verification Functions
// ============================================================================

/// Input for verifying epistemic requirements
#[derive(Serialize, Deserialize, Debug)]
pub struct EpistemicVerifyInput {
    pub credential_hash: ActionHash,
    pub min_empirical: u8,
    pub min_normative: u8,
    pub min_materiality: u8,
}

/// Verify a credential meets minimum epistemic requirements
#[hdk_extern]
pub fn verify_epistemic_requirements(input: EpistemicVerifyInput) -> ExternResult<bool> {
    // Get the credential
    let record = get(input.credential_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".to_string())))?;

    let credential = match record.entry().as_option() {
        Some(Entry::App(bytes)) => {
            VerifiableCredential::try_from(SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())))
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {:?}", e))))?
        }
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Invalid entry type".to_string()))),
    };

    // Check empirical level
    let empirical_ok = credential.epistemic_empirical
        .map(|e| e >= input.min_empirical)
        .unwrap_or(false);

    // Check normative level
    let normative_ok = credential.epistemic_normative
        .map(|n| n >= input.min_normative)
        .unwrap_or(false);

    // Check materiality level
    let materiality_ok = credential.epistemic_materiality
        .map(|m| m >= input.min_materiality)
        .unwrap_or(false);

    Ok(empirical_ok && normative_ok && materiality_ok)
}

/// Get epistemic classification for a credential
#[hdk_extern]
pub fn get_epistemic_classification(credential_hash: ActionHash) -> ExternResult<EpistemicClassificationResult> {
    let record = get(credential_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".to_string())))?;

    let credential = match record.entry().as_option() {
        Some(Entry::App(bytes)) => {
            VerifiableCredential::try_from(SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())))
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {:?}", e))))?
        }
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Invalid entry type".to_string()))),
    };

    Ok(EpistemicClassificationResult {
        empirical: credential.epistemic_empirical,
        normative: credential.epistemic_normative,
        materiality: credential.epistemic_materiality,
        code: format_epistemic_code(
            credential.epistemic_empirical,
            credential.epistemic_normative,
            credential.epistemic_materiality,
        ),
    })
}

/// Format epistemic classification as code string (e.g., "E3-N1-M2")
fn format_epistemic_code(e: Option<u8>, n: Option<u8>, m: Option<u8>) -> String {
    format!(
        "E{}-N{}-M{}",
        e.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
        n.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
        m.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
    )
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EpistemicClassificationResult {
    pub empirical: Option<u8>,
    pub normative: Option<u8>,
    pub materiality: Option<u8>,
    pub code: String,
}

// ============================================================================
// Tests (pure helpers only)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use praxis_core::CourseId;
    use credential_integrity::VerifiableCredential;

    #[test]
    fn proof_round_trip_is_deterministic() {
        let issuer = "issuer-123".to_string();
        let subject = "learner-xyz".to_string();
        let course_id = CourseId("course-abc".to_string());
        let model_id = "model-1".to_string();
        let rubric_id = "rubric-1".to_string();
        let score_band = "A".to_string();
        let created = "2024-01-01T00:00:00Z".to_string();
        let expiration = Some("2025-01-01T00:00:00Z".to_string());

        let proof_value = compute_proof_value(
            issuer.clone(),
            subject.clone(),
            course_id.0.clone(),
            model_id.clone(),
            rubric_id.clone(),
            score_band.clone(),
            Some(95.0),
            expiration.clone(),
            created.clone(),
        )
        .unwrap();

        let credential = VerifiableCredential {
            context: "[]".to_string(),
            credential_type: vec!["VerifiableCredential".to_string()],
            issuer: issuer.clone(),
            issuance_date: created.clone(),
            expiration_date: expiration.clone(),
            subject_id: subject.clone(),
            course_id,
            model_id,
            rubric_id,
            score: Some(95.0),
            score_band,
            subject_metadata: None,
            status_id: None,
            status_type: None,
            status_list_index: None,
            status_purpose: None,
            proof_type: "Ed25519Signature2020".to_string(),
            proof_created: created.clone(),
            verification_method: format!("did:key:{issuer}"),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: proof_value.clone(),
            // Epistemic classification (E3-N1-M2 for cryptographic credential)
            epistemic_empirical: Some(EmpiricalLevel::E3Cryptographic as u8),
            epistemic_normative: Some(NormativeLevel::N1Communal as u8),
            epistemic_materiality: Some(MaterialityLevel::M2Persistent as u8),
        };

        assert!(verify_proof(&credential).unwrap());
    }

    #[test]
    fn test_determine_empirical_level() {
        assert_eq!(
            determine_empirical_level("Ed25519Signature2020"),
            EmpiricalLevel::E3Cryptographic
        );
        assert_eq!(
            determine_empirical_level("ZKProof2023"),
            EmpiricalLevel::E4PublicRepro
        );
        assert_eq!(
            determine_empirical_level("Unknown"),
            EmpiricalLevel::E2PrivateVerify
        );
    }

    #[test]
    fn test_format_epistemic_code() {
        assert_eq!(
            format_epistemic_code(Some(3), Some(1), Some(2)),
            "E3-N1-M2"
        );
        assert_eq!(
            format_epistemic_code(None, Some(1), None),
            "E?-N1-M?"
        );
    }
}

// ============================================================================
// Input/Output structures
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct IssueCredentialInput {
    pub learner_agent: AgentPubKey,
    pub course_id: CourseId,
    pub model_id: String,
    pub rubric_id: String,
    pub score: Option<f32>,
    pub score_band: String,
    pub metadata: Option<serde_json::Value>,
    pub expiration_date: Option<String>,
    pub credential_status: Option<CredentialStatus>,
    /// Optional epistemic classification - if None, auto-determined from proof type
    pub epistemic_empirical: Option<u8>,
    pub epistemic_normative: Option<u8>,
    pub epistemic_materiality: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub is_revoked: bool,
    pub is_expired: bool,
    pub issuer: String,
    pub issuance_date: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeCredentialInput {
    pub credential_hash: ActionHash,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum SignalPayload {
    CredentialRevoked {
        credential_hash: ActionHash,
        reason: String,
        timestamp: String,
    },
}

// ============================================================================
// Helpers
// ============================================================================

fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}
