// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Input validation for API requests
//!
//! Provides validation rules to ensure data quality and security

use once_cell::sync::Lazy;
use regex::Regex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Field '{field}' {reason}")]
    Invalid { field: String, reason: String },
}

// Compile regexes once at startup
static BATCH_ID_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[A-Z0-9_-]{1,100}$").unwrap()
});

static FACILITY_ID_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[A-Z0-9_-]{1,100}$").unwrap()
});

/// Validate a batch ID
///
/// Rules:
/// - Not empty
/// - Max 100 characters
/// - Only uppercase letters, numbers, hyphens, and underscores
pub fn validate_batch_id(batch_id: &str) -> Result<(), ValidationError> {
    if batch_id.is_empty() {
        return Err(ValidationError::Invalid {
            field: "batch_id".to_string(),
            reason: "cannot be empty".to_string(),
        });
    }

    if batch_id.len() > 100 {
        return Err(ValidationError::Invalid {
            field: "batch_id".to_string(),
            reason: "must be 100 characters or less".to_string(),
        });
    }

    if !BATCH_ID_REGEX.is_match(batch_id) {
        return Err(ValidationError::Invalid {
            field: "batch_id".to_string(),
            reason: "must contain only uppercase letters, numbers, hyphens, and underscores".to_string(),
        });
    }

    Ok(())
}

/// Validate a facility ID
///
/// Rules:
/// - Not empty
/// - Max 100 characters
/// - Only uppercase letters, numbers, hyphens, and underscores
pub fn validate_facility_id(facility_id: &str) -> Result<(), ValidationError> {
    if facility_id.is_empty() {
        return Err(ValidationError::Invalid {
            field: "facility_id".to_string(),
            reason: "cannot be empty".to_string(),
        });
    }

    if facility_id.len() > 100 {
        return Err(ValidationError::Invalid {
            field: "facility_id".to_string(),
            reason: "must be 100 characters or less".to_string(),
        });
    }

    if !FACILITY_ID_REGEX.is_match(facility_id) {
        return Err(ValidationError::Invalid {
            field: "facility_id".to_string(),
            reason: "must contain only uppercase letters, numbers, hyphens, and underscores".to_string(),
        });
    }

    Ok(())
}

/// Validate a product ID
///
/// Rules:
/// - Not empty
/// - Max 200 characters
/// - No dangerous characters (XSS prevention)
pub fn validate_product_id(product_id: &str) -> Result<(), ValidationError> {
    if product_id.is_empty() {
        return Err(ValidationError::Invalid {
            field: "product_id".to_string(),
            reason: "cannot be empty".to_string(),
        });
    }

    if product_id.len() > 200 {
        return Err(ValidationError::Invalid {
            field: "product_id".to_string(),
            reason: "must be 200 characters or less".to_string(),
        });
    }

    // Product IDs can contain more characters (spaces, etc.)
    // but should not contain dangerous characters
    if product_id.contains(&['<', '>', '"', '\'', '`', '\0'][..]) {
        return Err(ValidationError::Invalid {
            field: "product_id".to_string(),
            reason: "contains invalid characters".to_string(),
        });
    }

    Ok(())
}

/// Validate quantity
///
/// Rules:
/// - Must be positive
/// - Must be finite (not NaN or infinity)
/// - Max value 1 billion
pub fn validate_quantity(quantity: f64) -> Result<(), ValidationError> {
    if quantity <= 0.0 {
        return Err(ValidationError::Invalid {
            field: "quantity".to_string(),
            reason: "must be greater than zero".to_string(),
        });
    }

    if !quantity.is_finite() {
        return Err(ValidationError::Invalid {
            field: "quantity".to_string(),
            reason: "must be a finite number".to_string(),
        });
    }

    if quantity > 1_000_000_000.0 {
        return Err(ValidationError::Invalid {
            field: "quantity".to_string(),
            reason: "exceeds maximum allowed value of 1 billion".to_string(),
        });
    }

    Ok(())
}

/// Validate unit of measurement
///
/// Rules:
/// - Not empty
/// - Max 20 characters
pub fn validate_unit(unit: &str) -> Result<(), ValidationError> {
    if unit.is_empty() {
        return Err(ValidationError::Invalid {
            field: "unit".to_string(),
            reason: "cannot be empty".to_string(),
        });
    }

    if unit.len() > 20 {
        return Err(ValidationError::Invalid {
            field: "unit".to_string(),
            reason: "must be 20 characters or less".to_string(),
        });
    }

    Ok(())
}

/// Validate metadata
///
/// Rules:
/// - Max 10KB size
/// - Must be valid JSON
pub fn validate_metadata(metadata: &str) -> Result<(), ValidationError> {
    if metadata.len() > 10_000 {
        return Err(ValidationError::Invalid {
            field: "metadata".to_string(),
            reason: "exceeds maximum size of 10KB".to_string(),
        });
    }

    // Ensure metadata is valid JSON
    if serde_json::from_str::<serde_json::Value>(metadata).is_err() {
        return Err(ValidationError::Invalid {
            field: "metadata".to_string(),
            reason: "must be valid JSON".to_string(),
        });
    }

    Ok(())
}

/// Validate claim ID format
///
/// Rules:
/// - Not empty
/// - Max 100 characters
pub fn validate_claim_id(claim_id: &str) -> Result<(), ValidationError> {
    if claim_id.is_empty() {
        return Err(ValidationError::Invalid {
            field: "claim_id".to_string(),
            reason: "cannot be empty".to_string(),
        });
    }

    if claim_id.len() > 100 {
        return Err(ValidationError::Invalid {
            field: "claim_id".to_string(),
            reason: "must be 100 characters or less".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_batch_id() {
        // Valid IDs
        assert!(validate_batch_id("BATCH-001").is_ok());
        assert!(validate_batch_id("BATCH_TEST_123").is_ok());
        assert!(validate_batch_id("B").is_ok());

        // Invalid IDs
        assert!(validate_batch_id("").is_err());  // Empty
        assert!(validate_batch_id("batch-001").is_err());  // Lowercase
        assert!(validate_batch_id("BATCH 001").is_err());  // Space
        assert!(validate_batch_id("BATCH@001").is_err());  // Special char
        assert!(validate_batch_id(&"X".repeat(101)).is_err());  // Too long
    }

    #[test]
    fn test_validate_facility_id() {
        assert!(validate_facility_id("ORG-FACTORY").is_ok());
        assert!(validate_facility_id("FAC_001").is_ok());

        assert!(validate_facility_id("").is_err());
        assert!(validate_facility_id("org-factory").is_err());  // Lowercase
    }

    #[test]
    fn test_validate_product_id() {
        // Valid product IDs
        assert!(validate_product_id("Organic Coffee Beans").is_ok());
        assert!(validate_product_id("SKU-12345").is_ok());
        assert!(validate_product_id("Product (Premium)").is_ok());

        // Invalid product IDs
        assert!(validate_product_id("").is_err());  // Empty
        assert!(validate_product_id("<script>").is_err());  // XSS attempt
        assert!(validate_product_id("Product\"Name").is_err());  // Dangerous char
        assert!(validate_product_id(&"X".repeat(201)).is_err());  // Too long
    }

    #[test]
    fn test_validate_quantity() {
        // Valid quantities
        assert!(validate_quantity(1.0).is_ok());
        assert!(validate_quantity(1000.5).is_ok());
        assert!(validate_quantity(0.001).is_ok());

        // Invalid quantities
        assert!(validate_quantity(0.0).is_err());  // Zero
        assert!(validate_quantity(-1.0).is_err());  // Negative
        assert!(validate_quantity(f64::INFINITY).is_err());  // Infinity
        assert!(validate_quantity(f64::NAN).is_err());  // NaN
        assert!(validate_quantity(2_000_000_000.0).is_err());  // Too large
    }

    #[test]
    fn test_validate_unit() {
        assert!(validate_unit("kg").is_ok());
        assert!(validate_unit("units").is_ok());
        assert!(validate_unit("m³").is_ok());

        assert!(validate_unit("").is_err());  // Empty
        assert!(validate_unit(&"x".repeat(21)).is_err());  // Too long
    }

    #[test]
    fn test_validate_metadata() {
        // Valid JSON metadata
        assert!(validate_metadata("{}").is_ok());
        assert!(validate_metadata(r#"{"key": "value"}"#).is_ok());
        assert!(validate_metadata(r#"{"nested": {"data": 123}}"#).is_ok());

        // Invalid metadata
        assert!(validate_metadata("not json").is_err());  // Invalid JSON
        assert!(validate_metadata("{incomplete").is_err());  // Malformed JSON
        assert!(validate_metadata(&"x".repeat(11_000)).is_err());  // Too large
    }

    #[test]
    fn test_validate_claim_id() {
        assert!(validate_claim_id("01JCXXX123").is_ok());
        assert!(validate_claim_id("claim-123").is_ok());

        assert!(validate_claim_id("").is_err());  // Empty
        assert!(validate_claim_id(&"x".repeat(101)).is_err());  // Too long
    }
}
