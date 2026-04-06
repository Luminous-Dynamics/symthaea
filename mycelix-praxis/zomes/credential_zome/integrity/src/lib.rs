// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use praxis_core::CourseId;
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

// =============================================================================
// Assessment Entry Types
// =============================================================================

/// Assessment definition (created by teacher)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Assessment {
    /// Knowledge node this assesses
    pub node_hash: ActionHash,
    /// Type of assessment
    pub assessment_type: AssessmentType,
    /// Title
    pub title: String,
    /// Instructions
    pub instructions: Option<String>,
    /// Maximum score
    pub max_score: u32,
    /// Passing score
    pub passing_score: u32,
    /// Rubric criteria
    pub rubric: Vec<RubricCriterion>,
    /// Bloom's level being assessed
    pub bloom_level: Option<String>,
    /// Time limit in minutes (None = untimed)
    pub time_limit_minutes: Option<u32>,
    /// Creator
    pub creator: AgentPubKey,
    /// Creation timestamp
    pub created_at: i64,
}

/// Types of assessment
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssessmentType {
    Quiz,
    UnitTest,
    Project,
    Portfolio,
    Observation,
    Performance,
    SelfAssessment,
}

impl Default for AssessmentType {
    fn default() -> Self {
        AssessmentType::Quiz
    }
}

/// Rubric criterion for assessment
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RubricCriterion {
    pub name: String,
    pub description: String,
    pub max_points: u32,
    pub levels: Vec<RubricLevel>,
}

/// Performance level within a rubric criterion
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RubricLevel {
    /// Label e.g. "Exceeds", "Meets", "Approaching", "Beginning"
    pub label: String,
    pub points: u32,
    pub description: String,
}

/// Student's result on an assessment
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StudentResult {
    pub assessment_hash: ActionHash,
    pub student: AgentPubKey,
    pub score: u32,
    pub rubric_scores: Vec<u32>,
    pub attempt_number: u32,
    pub reasoning_trace: Option<String>,
    pub consciousness_level_permille: Option<u16>,
    pub feedback: Option<String>,
    pub graded_by: Option<AgentPubKey>,
    pub completed_at: i64,
}

/// Report card for a student in a period
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct ReportCard {
    pub student: AgentPubKey,
    pub classroom_hash: Option<ActionHash>,
    pub period: String,
    pub subject_grades: Vec<SubjectGrade>,
    pub total_mastery_permille: u16,
    pub attendance_days: u32,
    pub absence_days: u32,
    pub teacher_narrative: Option<String>,
    pub generated_at: i64,
}

/// Grade for a subject
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectGrade {
    pub subject: String,
    /// Grade band e.g. "A", "B", "C", "D", "F" or "4", "3", "2", "1"
    pub grade_band: String,
    pub mastery_permille: u16,
    pub standards_met: u32,
    pub standards_total: u32,
}

// =============================================================================
// Helper structs for coordinator zome (not HDI entry types)
// These are used for input/output but not stored directly
// =============================================================================

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
    #[entry_type(required_validations = 3, visibility = "public")]
    Assessment(Assessment),
    #[entry_type(required_validations = 1, visibility = "private")]
    StudentResult(StudentResult),
    #[entry_type(required_validations = 1, visibility = "private")]
    ReportCard(ReportCard),
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
    /// Node -> Assessments for that node
    NodeToAssessments,
    /// Student -> Their assessment results
    StudentToResults,
    /// Student -> Their report cards
    StudentToReportCards,
    /// Assessment -> Student results
    AssessmentToResults,
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
        if !score.is_finite() || score < 0.0 || score > 100.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Score must be a finite number between 0 and 100".to_string(),
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

/// Validation function for Assessment entries
pub fn validate_assessment(assessment: &Assessment) -> ExternResult<ValidateCallbackResult> {
    if assessment.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Assessment title cannot be empty".to_string(),
        ));
    }
    if assessment.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Assessment title too long (max 200)".to_string(),
        ));
    }
    if assessment.passing_score > assessment.max_score {
        return Ok(ValidateCallbackResult::Invalid(
            "Passing score cannot exceed max score".to_string(),
        ));
    }
    if assessment.rubric.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many rubric criteria (max 20)".to_string(),
        ));
    }
    for criterion in &assessment.rubric {
        if criterion.name.len() > 100 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rubric criterion name too long (max 100)".to_string(),
            ));
        }
        if criterion.levels.len() > 10 {
            return Ok(ValidateCallbackResult::Invalid(
                "Too many rubric levels (max 10)".to_string(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for StudentResult entries
pub fn validate_student_result(result: &StudentResult) -> ExternResult<ValidateCallbackResult> {
    if result.attempt_number == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Attempt number must be at least 1".to_string(),
        ));
    }
    if let Some(cl) = result.consciousness_level_permille {
        if cl > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Consciousness level permille must be 0-1000".to_string(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for ReportCard entries
pub fn validate_report_card(card: &ReportCard) -> ExternResult<ValidateCallbackResult> {
    if card.period.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Report card period cannot be empty".to_string(),
        ));
    }
    if card.total_mastery_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total mastery permille must be 0-1000".to_string(),
        ));
    }
    for grade in &card.subject_grades {
        if grade.subject.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Subject name cannot be empty".to_string(),
            ));
        }
        if grade.mastery_permille > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Subject mastery permille must be 0-1000".to_string(),
            ));
        }
        if grade.standards_met > grade.standards_total {
            return Ok(ValidateCallbackResult::Invalid(
                "Standards met cannot exceed standards total".to_string(),
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
                    Some(EntryTypes::Assessment(assessment)) => {
                        validate_assessment(&assessment)
                    }
                    Some(EntryTypes::StudentResult(result)) => {
                        validate_student_result(&result)
                    }
                    Some(EntryTypes::ReportCard(card)) => {
                        validate_report_card(&card)
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
