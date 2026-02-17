//! # Credential Integrity Zome
//!
//! Defines entry types and validation rules for W3C Verifiable Credentials.
//! This zome is immutable - entry definitions cannot change without breaking data.
//!
//! ## Design Note
//! This implementation flattens the W3C VC structure to work with Holochain's HDI.
//! All nested structures (CredentialSubject, CredentialStatus, Proof) are flattened
//! into the main VerifiableCredential entry type to ensure HDI compatibility.

use hdi::prelude::*;
use edunet_core::CourseId;
// Note: Epistemic levels stored as u8 for HDI compatibility
// See mycelix_sdk::epistemic for level definitions

/// Verifiable Credential entry (Flattened structure for HDI compatibility)
///
/// Follows W3C Verifiable Credentials Data Model but flattened for Holochain HDI.
/// All fields from nested objects (subject, status, proof) are included at top level.
///
/// Note: Field names use snake_case for HDI compatibility. W3C spec fields are
/// documented but stored differently to work with Holochain's serialization.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiableCredential {
    // ========== Context & Type ==========
    /// JSON-LD context (W3C: @context) - stored as serialized JSON string
    pub context: String,

    /// Credential type (W3C: type) - e.g., ["VerifiableCredential", "CourseCompletionCredential"]
    pub credential_type: Vec<String>,

    // ========== Issuer & Dates ==========
    /// Issuer identifier (agent public key as string)
    pub issuer: String,

    /// Issuance date (ISO 8601 format)
    pub issuance_date: String,

    /// Optional expiration date (ISO 8601 format)
    pub expiration_date: Option<String>,

    // ========== Credential Subject (flattened) ==========
    /// Learner identifier (agent public key as string)
    pub subject_id: String,

    /// Course completed
    pub course_id: CourseId,

    /// Model ID (for FL provenance)
    pub model_id: String,

    /// Rubric/assessment ID
    pub rubric_id: String,

    /// Achievement score (0-100)
    pub score: Option<f32>,

    /// Score band (e.g., "A", "Pass", "Mastery")
    pub score_band: String,

    /// Additional subject metadata (stored as serialized JSON string)
    pub subject_metadata: Option<String>,

    // ========== Credential Status (flattened, optional) ==========
    /// Status list identifier
    pub status_id: Option<String>,

    /// Status method type (e.g., "StatusList2021Entry")
    pub status_type: Option<String>,

    /// Index in status list (for revocation checking)
    pub status_list_index: Option<u32>,

    /// Status purpose (e.g., "revocation", "suspension")
    pub status_purpose: Option<String>,

    // ========== Cryptographic Proof (flattened) ==========
    /// Proof type (e.g., "Ed25519Signature2020")
    pub proof_type: String,

    /// Proof creation timestamp (ISO 8601)
    pub proof_created: String,

    /// Verification method (public key reference, e.g., "did:key:...")
    pub verification_method: String,

    /// Proof purpose (e.g., "assertionMethod")
    pub proof_purpose: String,

    /// Signature value (base64 encoded)
    pub proof_value: String,

    // ========== Epistemic Classification (Mycelix SDK) ==========
    /// Empirical level - how the credential can be verified
    /// E0=Null, E1=Testimonial, E2=PrivateVerify, E3=Cryptographic, E4=PublicRepro
    pub epistemic_empirical: Option<u8>,

    /// Normative level - who agrees with the credential
    /// N0=Personal, N1=Communal, N2=Network, N3=Axiomatic
    pub epistemic_normative: Option<u8>,

    /// Materiality level - how long the credential matters
    /// M0=Ephemeral, M1=Temporal, M2=Persistent, M3=Foundational
    pub epistemic_materiality: Option<u8>,
}

// Helper structs for coordinator zome (not HDI entry types)
// These are used for input/output but not stored directly

/// Subject of the credential (learner's achievement)
/// Used by coordinator zome for construction/parsing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CredentialSubject {
    pub id: String,
    pub course_id: CourseId,
    pub model_id: String,
    pub rubric_id: String,
    pub score: Option<f32>,
    pub score_band: String,
    pub metadata: Option<String>,
}

/// Credential status for revocation checking
/// Used by coordinator zome for construction/parsing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CredentialStatus {
    pub id: String,
    #[serde(rename = "type")]
    pub status_type: String,
    pub status_list_index: Option<u32>,
    pub status_purpose: Option<String>,
}

/// Cryptographic proof
/// Used by coordinator zome for construction/parsing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Proof {
    #[serde(rename = "type")]
    pub proof_type: String,
    pub created: String,
    pub verification_method: String,
    pub proof_purpose: String,
    pub proof_value: String,
}

/// All entry types for this integrity zome
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    VerifiableCredential(VerifiableCredential),
}

/// All link types for this integrity zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from learner to their credentials
    LearnerToCredentials,
    /// Links from course to credentials issued for that course
    CourseToCredentials,
    /// Links from issuer to credentials they issued
    IssuerToCredentials,
}

/// Validation function for VerifiableCredential entries
pub fn validate_verifiable_credential(
    credential: VerifiableCredential,
) -> ExternResult<ValidateCallbackResult> {
    // ========== Credential Type Validation ==========
    if !credential
        .credential_type
        .contains(&"VerifiableCredential".to_string())
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential type must include 'VerifiableCredential'".to_string(),
        ));
    }

    // ========== Issuer Validation ==========
    if credential.issuer.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer cannot be empty".to_string(),
        ));
    }

    // ========== Date Validation ==========
    if credential.issuance_date.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuance date cannot be empty".to_string(),
        ));
    }

    // ========== Subject Validation ==========
    if credential.subject_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject ID (learner) cannot be empty".to_string(),
        ));
    }

    if credential.score_band.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Score band cannot be empty".to_string(),
        ));
    }

    // Validate score if present
    if let Some(score) = credential.score {
        if score < 0.0 || score > 100.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Score must be between 0 and 100".to_string(),
            ));
        }
    }

    // ========== Proof Validation ==========
    if credential.proof_type.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof type cannot be empty".to_string(),
        ));
    }

    if credential.proof_value.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof value cannot be empty".to_string(),
        ));
    }

    if credential.proof_created.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof creation timestamp cannot be empty".to_string(),
        ));
    }

    if credential.verification_method.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Verification method cannot be empty".to_string(),
        ));
    }

    if credential.proof_purpose.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof purpose cannot be empty".to_string(),
        ));
    }

    // ========== Epistemic Classification Validation ==========
    // If epistemic fields are provided, validate they're within valid range
    if let Some(e) = credential.epistemic_empirical {
        if e > 4 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Epistemic empirical level {} invalid (must be 0-4)", e),
            ));
        }
        // Educational credentials should be at least E1 (Testimonial)
        if e < 1 {
            return Ok(ValidateCallbackResult::Invalid(
                "Educational credentials must be at least E1 (Testimonial)".to_string(),
            ));
        }
    }

    if let Some(n) = credential.epistemic_normative {
        if n > 3 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Epistemic normative level {} invalid (must be 0-3)", n),
            ));
        }
        // Educational credentials should be at least N1 (Communal)
        if n < 1 {
            return Ok(ValidateCallbackResult::Invalid(
                "Educational credentials must be at least N1 (Communal) - institution recognized".to_string(),
            ));
        }
    }

    if let Some(m) = credential.epistemic_materiality {
        if m > 3 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Epistemic materiality level {} invalid (must be 0-3)", m),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => match store_entry.action.hashed.content.entry_type() {
            EntryType::App(app_entry_def) => {
                let entry = store_entry.entry;
                match EntryTypes::deserialize_from_type(
                    app_entry_def.zome_index,
                    app_entry_def.entry_index,
                    &entry,
                )? {
                    Some(EntryTypes::VerifiableCredential(credential)) => {
                        validate_verifiable_credential(credential)
                    }
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
