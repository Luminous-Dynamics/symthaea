// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Standardized error messages for integrity zome validation.
//!
//! The 68+ integrity zomes across all Mycelix clusters produce hundreds of
//! validation error messages.  This module provides canonical message templates
//! so that:
//!
//! - Error messages are consistent across all zomes
//! - Test assertions can match on predictable strings
//! - Localization (if needed) has a single source of truth
//!
//! # Usage
//!
//! ```ignore
//! use mycelix_bridge_common::error_messages as errmsg;
//!
//! // Field validation
//! if title.trim().is_empty() {
//!     return Ok(ValidateCallbackResult::Invalid(
//!         errmsg::field_empty("Case", "title"),
//!     ));
//! }
//! if title.len() > 512 {
//!     return Ok(ValidateCallbackResult::Invalid(
//!         errmsg::field_too_long("Case", "title", 512),
//!     ));
//! }
//! ```

use core::fmt;

// ---------------------------------------------------------------------------
// Field validation
// ---------------------------------------------------------------------------

/// A required field is empty or whitespace-only.
///
/// Example: `"Property id cannot be empty"`
pub fn field_empty(entry_type: &str, field: &str) -> String {
    format!("{} {} cannot be empty", entry_type, field)
}

/// A string field exceeds its maximum length.
///
/// Example: `"Property title must be 256 characters or fewer"`
pub fn field_too_long(entry_type: &str, field: &str, max: usize) -> String {
    format!(
        "{} {} must be {} characters or fewer",
        entry_type, field, max
    )
}

/// A numeric/float field is not finite (NaN or infinity).
///
/// Example: `"Building latitude must be a finite number"`
pub fn field_not_finite(entry_type: &str, field: &str) -> String {
    format!("{} {} must be a finite number", entry_type, field)
}

/// A numeric field is outside its allowed range.
///
/// Example: `"Building latitude must be between -90 and 90"`
pub fn field_out_of_range<T: fmt::Display>(
    entry_type: &str,
    field: &str,
    min: T,
    max: T,
) -> String {
    format!(
        "{} {} must be between {} and {}",
        entry_type, field, min, max
    )
}

/// A numeric field must be greater than a minimum value.
///
/// Example: `"Unit square_meters must be greater than 0"`
pub fn field_min_value<T: fmt::Display>(entry_type: &str, field: &str, min: T) -> String {
    format!("{} {} must be greater than {}", entry_type, field, min)
}

/// A field must satisfy a specific format constraint.
///
/// Example: `"Case complainant must be a valid DID"`
pub fn field_invalid_format(entry_type: &str, field: &str, expected: &str) -> String {
    format!("{} {} must be {}", entry_type, field, expected)
}

// ---------------------------------------------------------------------------
// Collection / Vec validation
// ---------------------------------------------------------------------------

/// A Vec field exceeds its maximum element count.
///
/// Example: `"Property co_owners cannot have more than 50 items"`
pub fn vec_too_long(entry_type: &str, field: &str, max: usize) -> String {
    format!(
        "{} {} cannot have more than {} items",
        entry_type, field, max
    )
}

/// A collection must have at least N items.
///
/// Example: `"Circle members must have at least 2 items"`
pub fn vec_too_short(entry_type: &str, field: &str, min: usize) -> String {
    format!(
        "{} {} must have at least {} items",
        entry_type, field, min
    )
}

// ---------------------------------------------------------------------------
// Entry lifecycle
// ---------------------------------------------------------------------------

/// An entry was not found during validation (e.g. `must_get_valid_record` failure).
///
/// Example: `"Property not found"`
pub fn entry_not_found(entry_type: &str) -> String {
    format!("{} not found", entry_type)
}

/// An entry type cannot be created in the current context.
///
/// Example: `"TitleDeed cannot be created directly; use transfer workflow"`
pub fn entry_create_forbidden(entry_type: &str, reason: &str) -> String {
    format!("{} cannot be created directly; {}", entry_type, reason)
}

/// An entry failed generic validation.
///
/// Example: `"Case failed validation: complainant and respondent must differ"`
pub fn invalid_entry(entry_type: &str, reason: &str) -> String {
    format!("{} failed validation: {}", entry_type, reason)
}

/// An entry or anchor cannot be updated once created.
///
/// Example: `"Anchor cannot be updated once created"`
pub fn update_forbidden(entry_type: &str) -> String {
    format!("{} cannot be updated once created", entry_type)
}

/// An entry cannot be deleted.
///
/// Example: `"BridgeEvent cannot be deleted once created"`
pub fn delete_forbidden(entry_type: &str) -> String {
    format!("{} cannot be deleted once created", entry_type)
}

// ---------------------------------------------------------------------------
// Author / permission checks
// ---------------------------------------------------------------------------

/// The acting agent is not the original author of the entry.
///
/// Example: `"Only the original author can update this Property"`
pub fn author_mismatch(operation: &str, entry_type: &str) -> String {
    format!(
        "Only the original author can {} this {}",
        operation, entry_type
    )
}

/// Only the original link author may delete the link.
///
/// Example: `"Only the original author can delete this link"`
pub fn link_author_mismatch() -> String {
    "Only the original author can delete this link".into()
}

// ---------------------------------------------------------------------------
// Link validation
// ---------------------------------------------------------------------------

/// A link tag exceeds the maximum allowed size.
///
/// Example: `"AllBuildings link tag must be 256 bytes or fewer"`
pub fn link_tag_too_long(link_type: &str, max: usize) -> String {
    format!("{} link tag must be {} bytes or fewer", link_type, max)
}

/// A domain value is not recognized.
///
/// Example: `"Invalid domain 'foo'. Must be one of: [\"kinship\", \"gratitude\"]"`
pub fn invalid_domain(domain: &str, valid: &[&str]) -> String {
    format!(
        "Invalid domain '{}'. Must be one of: {:?}",
        domain, valid
    )
}

// ---------------------------------------------------------------------------
// Payload / data limits
// ---------------------------------------------------------------------------

/// A payload or blob exceeds its byte limit.
///
/// Example: `"Query payload must be 65536 bytes or fewer"`
pub fn payload_too_large(payload_name: &str, max: usize) -> String {
    format!("{} must be {} bytes or fewer", payload_name, max)
}

// ---------------------------------------------------------------------------
// Deserialization
// ---------------------------------------------------------------------------

/// Failed to deserialize an entry from the DHT.
///
/// Example: `"Failed to deserialize BridgeQuery: invalid msgpack"`
pub fn deserialize_failed(entry_type: &str, detail: &str) -> String {
    format!("Failed to deserialize {}: {}", entry_type, detail)
}

/// An immutable field was changed during an update.
///
/// Example: `"BridgeQuery field 'domain' cannot be changed after creation"`
pub fn immutable_field_changed(entry_type: &str, field: &str) -> String {
    format!(
        "{} field '{}' cannot be changed after creation",
        entry_type, field
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Field validation messages ----

    #[test]
    fn test_field_empty() {
        assert_eq!(
            field_empty("Property", "id"),
            "Property id cannot be empty"
        );
    }

    #[test]
    fn test_field_too_long() {
        assert_eq!(
            field_too_long("Property", "title", 256),
            "Property title must be 256 characters or fewer"
        );
    }

    #[test]
    fn test_field_not_finite() {
        assert_eq!(
            field_not_finite("Building", "latitude"),
            "Building latitude must be a finite number"
        );
    }

    #[test]
    fn test_field_out_of_range() {
        assert_eq!(
            field_out_of_range("Building", "latitude", -90, 90),
            "Building latitude must be between -90 and 90"
        );
    }

    #[test]
    fn test_field_min_value() {
        assert_eq!(
            field_min_value("Unit", "square_meters", 0),
            "Unit square_meters must be greater than 0"
        );
    }

    #[test]
    fn test_field_invalid_format() {
        assert_eq!(
            field_invalid_format("Case", "complainant", "a valid DID"),
            "Case complainant must be a valid DID"
        );
    }

    // ---- Collection messages ----

    #[test]
    fn test_vec_too_long() {
        assert_eq!(
            vec_too_long("Property", "co_owners", 50),
            "Property co_owners cannot have more than 50 items"
        );
    }

    #[test]
    fn test_vec_too_short() {
        assert_eq!(
            vec_too_short("Circle", "members", 2),
            "Circle members must have at least 2 items"
        );
    }

    // ---- Entry lifecycle messages ----

    #[test]
    fn test_entry_not_found() {
        assert_eq!(entry_not_found("Property"), "Property not found");
    }

    #[test]
    fn test_entry_create_forbidden() {
        assert_eq!(
            entry_create_forbidden("TitleDeed", "use transfer workflow"),
            "TitleDeed cannot be created directly; use transfer workflow"
        );
    }

    #[test]
    fn test_invalid_entry() {
        assert_eq!(
            invalid_entry("Case", "complainant and respondent must differ"),
            "Case failed validation: complainant and respondent must differ"
        );
    }

    #[test]
    fn test_update_forbidden() {
        assert_eq!(
            update_forbidden("Anchor"),
            "Anchor cannot be updated once created"
        );
    }

    #[test]
    fn test_delete_forbidden() {
        assert_eq!(
            delete_forbidden("BridgeEvent"),
            "BridgeEvent cannot be deleted once created"
        );
    }

    // ---- Author / permission messages ----

    #[test]
    fn test_author_mismatch() {
        assert_eq!(
            author_mismatch("update", "Property"),
            "Only the original author can update this Property"
        );
    }

    #[test]
    fn test_link_author_mismatch() {
        assert_eq!(
            link_author_mismatch(),
            "Only the original author can delete this link"
        );
    }

    // ---- Link messages ----

    #[test]
    fn test_link_tag_too_long() {
        assert_eq!(
            link_tag_too_long("AllBuildings", 256),
            "AllBuildings link tag must be 256 bytes or fewer"
        );
    }

    #[test]
    fn test_invalid_domain() {
        let valid = &["kinship", "gratitude"];
        let msg = invalid_domain("foo", valid);
        assert!(msg.contains("foo"));
        assert!(msg.contains("kinship"));
    }

    // ---- Payload / data messages ----

    #[test]
    fn test_payload_too_large() {
        assert_eq!(
            payload_too_large("Query payload", 65536),
            "Query payload must be 65536 bytes or fewer"
        );
    }

    // ---- Deserialization messages ----

    #[test]
    fn test_deserialize_failed() {
        assert_eq!(
            deserialize_failed("BridgeQuery", "invalid msgpack"),
            "Failed to deserialize BridgeQuery: invalid msgpack"
        );
    }

    #[test]
    fn test_immutable_field_changed() {
        assert_eq!(
            immutable_field_changed("BridgeQuery", "domain"),
            "BridgeQuery field 'domain' cannot be changed after creation"
        );
    }

    // ---- Consistency checks ----

    #[test]
    fn messages_are_nonempty() {
        // Every function should produce a non-empty message
        let messages = vec![
            field_empty("X", "y"),
            field_too_long("X", "y", 10),
            field_not_finite("X", "y"),
            field_out_of_range("X", "y", 0, 1),
            field_min_value("X", "y", 0),
            field_invalid_format("X", "y", "z"),
            vec_too_long("X", "y", 5),
            vec_too_short("X", "y", 1),
            entry_not_found("X"),
            entry_create_forbidden("X", "y"),
            invalid_entry("X", "y"),
            update_forbidden("X"),
            delete_forbidden("X"),
            author_mismatch("update", "X"),
            link_author_mismatch(),
            link_tag_too_long("X", 256),
            invalid_domain("x", &["a"]),
            payload_too_large("X", 100),
            deserialize_failed("X", "y"),
            immutable_field_changed("X", "y"),
        ];
        for msg in &messages {
            assert!(!msg.is_empty(), "Empty message found");
        }
    }
}
