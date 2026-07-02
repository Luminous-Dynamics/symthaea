// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PoGQ Validation Integrity Zome
//!
//! Defines entry types and validation rules for Proof of Gradient Quality.
//! This zome validates gradient contributions for federated learning.

use hdi::prelude::*;

/// Maximum allowed L2 norm for gradients
pub const MAX_GRADIENT_NORM: f32 = 10.0;
/// Minimum gradient elements for valid contribution
pub const MIN_GRADIENT_ELEMENTS: u32 = 100;
/// Required epsilon for differential privacy
pub const REQUIRED_EPSILON: f32 = 10.0;
/// Required sigma (noise) for differential privacy
pub const REQUIRED_SIGMA: f32 = 5.0;

/// A gradient contribution submitted by an FL participant
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GradientSubmission {
    /// The agent who submitted the gradient (as string for compatibility)
    pub agent: String,
    /// FL round number
    pub round: u64,
    /// Hash of the gradient values (commitment)
    pub gradient_hash: String,
    /// L2 norm of the gradient
    pub l2_norm: f32,
    /// Number of gradient elements
    pub num_elements: u32,
    /// Sigma (noise) added for differential privacy
    pub noise_sigma: f32,
    /// Claimed epsilon value for DP
    pub claimed_epsilon: f32,
    /// Statistical moments for anomaly detection
    pub mean: f32,
    pub variance: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    /// Timestamp of submission
    pub submitted_at: i64,
}

/// Result of validating a gradient submission
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ValidationResult {
    /// Hash of the submission being validated (as string for compatibility)
    pub submission_hash: String,
    /// The agent who submitted the gradient
    pub agent: String,
    /// FL round number
    pub round: u64,
    /// PoGQ metrics from validation (inlined)
    pub dp_compliant: bool,
    pub epsilon: f32,
    pub sigma: f32,
    pub l2_norm: f32,
    pub bounds_valid: bool,
    pub quality_score: f32,
    /// Whether the submission passed all checks
    pub is_valid: bool,
    /// Any error message if validation failed
    pub error_message: Option<String>,
    /// Validator who performed the validation
    pub validator: String,
    /// Timestamp of validation
    pub validated_at: i64,
}

/// An agent's accumulated trust record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AgentTrustRecord {
    /// The agent this record belongs to
    pub agent: String,
    /// Total number of submissions
    pub total_submissions: u64,
    /// Number of valid submissions
    pub valid_submissions: u64,
    /// Average quality score
    pub avg_quality_score: f32,
    /// Number of Byzantine flags
    pub byzantine_flags: u32,
    /// Last update timestamp
    pub updated_at: i64,
    /// Version for conflict resolution
    pub version: u64,
}

/// Entry types for PoGQ validation
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    GradientSubmission(GradientSubmission),
    ValidationResult(ValidationResult),
    AgentTrustRecord(AgentTrustRecord),
}

/// Link types for PoGQ validation
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from agent to their submissions
    AgentToSubmissions,
    /// Link from agent to their trust record
    AgentToTrustRecord,
    /// Link from submission to validation result
    SubmissionToValidation,
    /// Link from round to submissions
    RoundToSubmissions,
}

// === Validation Functions ===

/// Validate GradientSubmission entries
pub fn validate_gradient_submission(submission: GradientSubmission) -> ExternResult<ValidateCallbackResult> {
    // 1. Check agent is non-empty
    if submission.agent.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent ID cannot be empty".to_string(),
        ));
    }

    // 2. Check gradient hash is non-empty
    if submission.gradient_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Gradient hash cannot be empty".to_string(),
        ));
    }

    // 3. Check gradient hash length (should be a hash)
    if submission.gradient_hash.len() < 16 || submission.gradient_hash.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid gradient hash length".to_string(),
        ));
    }

    // 4. Check L2 norm is non-negative and finite
    if submission.l2_norm < 0.0 || !submission.l2_norm.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "L2 norm must be non-negative and finite".to_string(),
        ));
    }

    // 5. Check minimum gradient elements
    if submission.num_elements < MIN_GRADIENT_ELEMENTS {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Too few gradient elements: {} < {}",
            submission.num_elements, MIN_GRADIENT_ELEMENTS
        )));
    }

    // 6. Check differential privacy parameters
    if submission.claimed_epsilon <= 0.0 || !submission.claimed_epsilon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Epsilon must be positive and finite".to_string(),
        ));
    }

    if submission.noise_sigma <= 0.0 || !submission.noise_sigma.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Noise sigma must be positive and finite".to_string(),
        ));
    }

    // 7. Check variance is non-negative
    if submission.variance < 0.0 || !submission.variance.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Variance must be non-negative and finite".to_string(),
        ));
    }

    // 8. Check other statistical moments are finite
    if !submission.mean.is_finite() || !submission.skewness.is_finite() || !submission.kurtosis.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Statistical moments must be finite".to_string(),
        ));
    }

    // 9. Epsilon must not exceed required value (lower is more private)
    if submission.claimed_epsilon > REQUIRED_EPSILON {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Epsilon {} exceeds maximum allowed {}",
            submission.claimed_epsilon, REQUIRED_EPSILON
        )));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate ValidationResult entries
pub fn validate_validation_result(result: ValidationResult) -> ExternResult<ValidateCallbackResult> {
    // 1. Check submission hash is non-empty
    if result.submission_hash.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Submission hash cannot be empty".to_string(),
        ));
    }

    // 2. Check agent is non-empty
    if result.agent.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent ID cannot be empty".to_string(),
        ));
    }

    // 3. Check validator is non-empty
    if result.validator.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Validator ID cannot be empty".to_string(),
        ));
    }

    // 4. Quality score must be in [0, 1]
    if result.quality_score < 0.0 || result.quality_score > 1.0 || !result.quality_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Quality score must be between 0 and 1".to_string(),
        ));
    }

    // 5. Epsilon must be positive
    if result.epsilon <= 0.0 || !result.epsilon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Epsilon must be positive".to_string(),
        ));
    }

    // 6. Sigma must be positive
    if result.sigma <= 0.0 || !result.sigma.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Sigma must be positive".to_string(),
        ));
    }

    // 7. L2 norm must be non-negative
    if result.l2_norm < 0.0 || !result.l2_norm.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "L2 norm must be non-negative".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate AgentTrustRecord entries
pub fn validate_agent_trust_record(record: AgentTrustRecord) -> ExternResult<ValidateCallbackResult> {
    // 1. Check agent is non-empty
    if record.agent.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent ID cannot be empty".to_string(),
        ));
    }

    // 2. Valid submissions cannot exceed total
    if record.valid_submissions > record.total_submissions {
        return Ok(ValidateCallbackResult::Invalid(
            "Valid submissions cannot exceed total submissions".to_string(),
        ));
    }

    // 3. Average quality score must be in [0, 1]
    if record.avg_quality_score < 0.0 || record.avg_quality_score > 1.0 || !record.avg_quality_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Average quality score must be between 0 and 1".to_string(),
        ));
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
                    Some(EntryTypes::GradientSubmission(submission)) => {
                        validate_gradient_submission(submission)
                    }
                    Some(EntryTypes::ValidationResult(result)) => {
                        validate_validation_result(result)
                    }
                    Some(EntryTypes::AgentTrustRecord(record)) => {
                        validate_agent_trust_record(record)
                    }
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
