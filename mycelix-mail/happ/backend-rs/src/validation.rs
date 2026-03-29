// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Input validation for API requests
//!
//! Provides validation functions for DIDs, trust scores, and other inputs.

use crate::error::{AppError, AppResult};

/// Maximum allowed subject length (characters)
pub const MAX_SUBJECT_LENGTH: usize = 500;

/// Maximum allowed body length (bytes)
pub const MAX_BODY_LENGTH: usize = 10 * 1024 * 1024; // 10 MB

/// Minimum allowed trust score
pub const MIN_TRUST_SCORE: f64 = 0.0;

/// Maximum allowed trust score
pub const MAX_TRUST_SCORE: f64 = 1.0;

/// Validate a DID format
///
/// DIDs must follow the format: `did:<method>:<method-specific-id>`
/// Allowed methods: mycelix, key, web, ion, ethr, pkh
pub fn validate_did(did: &str) -> AppResult<()> {
    if did.is_empty() {
        return Err(AppError::ValidationError("DID cannot be empty".to_string()));
    }

    if !did.starts_with("did:") {
        return Err(AppError::ValidationError(
            "DID must start with 'did:'".to_string(),
        ));
    }

    let parts: Vec<&str> = did.split(':').collect();
    if parts.len() < 3 {
        return Err(AppError::ValidationError(
            "DID must have format 'did:<method>:<identifier>'".to_string(),
        ));
    }

    let method = parts[1];
    if method.is_empty() {
        return Err(AppError::ValidationError(
            "DID method cannot be empty".to_string(),
        ));
    }

    // Validate method characters (lowercase alphanumeric only)
    if !method.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        return Err(AppError::ValidationError(
            "DID method must be lowercase alphanumeric".to_string(),
        ));
    }

    let identifier = parts[2..].join(":");
    if identifier.is_empty() {
        return Err(AppError::ValidationError(
            "DID identifier cannot be empty".to_string(),
        ));
    }

    // Validate identifier length
    if identifier.len() > 256 {
        return Err(AppError::ValidationError(
            "DID identifier too long (max 256 characters)".to_string(),
        ));
    }

    Ok(())
}

/// Validate a trust score
pub fn validate_trust_score(score: f64) -> AppResult<()> {
    if score.is_nan() {
        return Err(AppError::ValidationError(
            "Trust score cannot be NaN".to_string(),
        ));
    }

    if score < MIN_TRUST_SCORE || score > MAX_TRUST_SCORE {
        return Err(AppError::ValidationError(format!(
            "Trust score must be between {} and {}",
            MIN_TRUST_SCORE, MAX_TRUST_SCORE
        )));
    }

    Ok(())
}

/// Validate email subject
pub fn validate_subject(subject: &str) -> AppResult<()> {
    if subject.is_empty() {
        return Err(AppError::ValidationError(
            "Subject cannot be empty".to_string(),
        ));
    }

    if subject.chars().count() > MAX_SUBJECT_LENGTH {
        return Err(AppError::ValidationError(format!(
            "Subject too long (max {} characters)",
            MAX_SUBJECT_LENGTH
        )));
    }

    // Check for control characters (except newline/tab)
    if subject.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
        return Err(AppError::ValidationError(
            "Subject contains invalid control characters".to_string(),
        ));
    }

    Ok(())
}

/// Validate email body
pub fn validate_body(body: &str) -> AppResult<()> {
    if body.len() > MAX_BODY_LENGTH {
        return Err(AppError::ValidationError(format!(
            "Body too large (max {} bytes)",
            MAX_BODY_LENGTH
        )));
    }

    Ok(())
}

/// Validate a base64-encoded string
pub fn validate_base64(input: &str, field_name: &str) -> AppResult<Vec<u8>> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    STANDARD
        .decode(input)
        .map_err(|_| AppError::ValidationError(format!("Invalid base64 encoding for {}", field_name)))
}

/// Validate an action hash (39 bytes base64)
pub fn validate_action_hash(input: &str) -> AppResult<[u8; 39]> {
    let bytes = validate_base64(input, "action hash")?;

    if bytes.len() != 39 {
        return Err(AppError::ValidationError(format!(
            "Invalid action hash length: expected 39 bytes, got {}",
            bytes.len()
        )));
    }

    let mut arr = [0u8; 39];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Sanitize a string by removing null bytes and trimming
pub fn sanitize_string(input: &str) -> String {
    input
        .replace('\0', "")
        .trim()
        .to_string()
}

/// Rate limit configuration
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub requests_per_minute: u32,
    /// Maximum requests per hour
    pub requests_per_hour: u32,
    /// Burst allowance
    pub burst: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_hour: 1000,
            burst: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_dids() {
        assert!(validate_did("did:mycelix:abc123").is_ok());
        assert!(validate_did("did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK").is_ok());
        assert!(validate_did("did:web:example.com").is_ok());
        assert!(validate_did("did:ethr:0x123").is_ok());
    }

    #[test]
    fn test_invalid_dids() {
        assert!(validate_did("").is_err());
        assert!(validate_did("not-a-did").is_err());
        assert!(validate_did("did:").is_err());
        assert!(validate_did("did:method:").is_err());
        assert!(validate_did("did::identifier").is_err());
        assert!(validate_did("did:UPPER:identifier").is_err());
    }

    #[test]
    fn test_trust_scores() {
        assert!(validate_trust_score(0.0).is_ok());
        assert!(validate_trust_score(0.5).is_ok());
        assert!(validate_trust_score(1.0).is_ok());
        assert!(validate_trust_score(-0.1).is_err());
        assert!(validate_trust_score(1.1).is_err());
        assert!(validate_trust_score(f64::NAN).is_err());
    }

    #[test]
    fn test_subject_validation() {
        assert!(validate_subject("Hello").is_ok());
        assert!(validate_subject("").is_err());
        assert!(validate_subject(&"a".repeat(501)).is_err());
        assert!(validate_subject("Hello\x00World").is_err());
    }

    #[test]
    fn test_sanitize() {
        assert_eq!(sanitize_string("  hello  "), "hello");
        assert_eq!(sanitize_string("hello\0world"), "helloworld");
    }
}
