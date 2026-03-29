// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for the migration module, including mock Migratable implementations.

use mycelix_bridge_common::migration::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Mock Migratable type
// ═══════════════════════════════════════════════════════════════════════════════

/// A mock DHT entry that demonstrates the Migratable trait pattern.
///
/// V1: { "name": "..." }
/// V2: { "name": "...", "email": "..." }  (added email field)
/// V3: { "full_name": "...", "email": "..." }  (renamed name → full_name)
#[derive(Debug, Clone, PartialEq)]
struct MockEntry {
    full_name: String,
    email: String,
}

impl Migratable for MockEntry {
    const CURRENT_VERSION: u8 = 3;

    fn migrate_from(json: &str, from_version: u8) -> Result<Self, MigrationError> {
        match from_version {
            1 => {
                // V1: only has "name" field
                let parsed: serde_json::Value = serde_json::from_str(json)
                    .map_err(|e| MigrationError::DeserializationFailed(e.to_string()))?;
                let name = parsed["name"]
                    .as_str()
                    .ok_or_else(|| MigrationError::MigrationStepFailed {
                        from: 1,
                        to: 3,
                        reason: "missing 'name' field".into(),
                    })?;
                Ok(MockEntry {
                    full_name: name.to_string(),
                    email: String::new(), // Default for missing field
                })
            }
            2 => {
                // V2: has "name" and "email"
                let parsed: serde_json::Value = serde_json::from_str(json)
                    .map_err(|e| MigrationError::DeserializationFailed(e.to_string()))?;
                let name = parsed["name"]
                    .as_str()
                    .ok_or_else(|| MigrationError::MigrationStepFailed {
                        from: 2,
                        to: 3,
                        reason: "missing 'name' field".into(),
                    })?;
                let email = parsed["email"].as_str().unwrap_or("");
                Ok(MockEntry {
                    full_name: name.to_string(),
                    email: email.to_string(),
                })
            }
            3 => {
                // Current version: has "full_name" and "email"
                let parsed: serde_json::Value = serde_json::from_str(json)
                    .map_err(|e| MigrationError::DeserializationFailed(e.to_string()))?;
                let full_name = parsed["full_name"]
                    .as_str()
                    .ok_or_else(|| MigrationError::DeserializationFailed(
                        "missing 'full_name' field".into(),
                    ))?;
                let email = parsed["email"].as_str().unwrap_or("");
                Ok(MockEntry {
                    full_name: full_name.to_string(),
                    email: email.to_string(),
                })
            }
            _ => Err(MigrationError::UnknownVersion {
                found: from_version,
                max_supported: Self::CURRENT_VERSION,
            }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// needs_migration / is_future_version
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_needs_migration_boundary() {
    assert!(needs_migration(0, 1));
    assert!(!needs_migration(1, 1));
    assert!(!needs_migration(2, 1));
    assert!(needs_migration(254, 255));
    assert!(!needs_migration(255, 255));
}

#[test]
fn test_is_future_version_boundary() {
    assert!(is_future_version(255, 254));
    assert!(!is_future_version(0, 0));
    assert!(is_future_version(1, 0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Migratable trait — V1 → V3
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_migrate_from_v1() {
    let json = r#"{"name": "Alice"}"#;
    let entry = MockEntry::migrate_from(json, 1).unwrap();
    assert_eq!(entry.full_name, "Alice");
    assert_eq!(entry.email, ""); // Default for missing field
}

#[test]
fn test_migrate_from_v2() {
    let json = r#"{"name": "Bob", "email": "bob@example.com"}"#;
    let entry = MockEntry::migrate_from(json, 2).unwrap();
    assert_eq!(entry.full_name, "Bob");
    assert_eq!(entry.email, "bob@example.com");
}

#[test]
fn test_migrate_from_v3_current() {
    let json = r#"{"full_name": "Charlie", "email": "charlie@example.com"}"#;
    let entry = MockEntry::migrate_from(json, 3).unwrap();
    assert_eq!(entry.full_name, "Charlie");
    assert_eq!(entry.email, "charlie@example.com");
}

#[test]
fn test_migrate_unknown_version() {
    let json = r#"{"data": "something"}"#;
    let err = MockEntry::migrate_from(json, 99).unwrap_err();
    match err {
        MigrationError::UnknownVersion { found, max_supported } => {
            assert_eq!(found, 99);
            assert_eq!(max_supported, 3);
        }
        _ => panic!("Expected UnknownVersion, got {:?}", err),
    }
}

#[test]
fn test_migrate_version_zero() {
    let json = r#"{"data": "ancient"}"#;
    let err = MockEntry::migrate_from(json, 0).unwrap_err();
    assert!(matches!(err, MigrationError::UnknownVersion { .. }));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Malformed JSON handling
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_migrate_invalid_json() {
    let err = MockEntry::migrate_from("not json at all", 1).unwrap_err();
    assert!(matches!(err, MigrationError::DeserializationFailed(_)));
}

#[test]
fn test_migrate_empty_json() {
    let err = MockEntry::migrate_from("", 1).unwrap_err();
    assert!(matches!(err, MigrationError::DeserializationFailed(_)));
}

#[test]
fn test_migrate_missing_required_field() {
    // V1 requires "name" but this JSON has "wrong_field"
    let json = r#"{"wrong_field": "value"}"#;
    let err = MockEntry::migrate_from(json, 1).unwrap_err();
    assert!(matches!(err, MigrationError::MigrationStepFailed { .. }));
}

#[test]
fn test_migrate_null_field() {
    let json = r#"{"name": null}"#;
    let err = MockEntry::migrate_from(json, 1).unwrap_err();
    assert!(matches!(err, MigrationError::MigrationStepFailed { .. }));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data preservation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_migration_preserves_critical_data() {
    // V1 → V3: name should be preserved as full_name
    let v1_json = r#"{"name": "Dr. Alice Smith PhD"}"#;
    let entry = MockEntry::migrate_from(v1_json, 1).unwrap();
    assert_eq!(entry.full_name, "Dr. Alice Smith PhD");

    // V2 → V3: both name and email preserved
    let v2_json = r#"{"name": "Dr. Alice Smith PhD", "email": "alice@university.edu"}"#;
    let entry = MockEntry::migrate_from(v2_json, 2).unwrap();
    assert_eq!(entry.full_name, "Dr. Alice Smith PhD");
    assert_eq!(entry.email, "alice@university.edu");
}

#[test]
fn test_migration_handles_unicode() {
    let json = r#"{"name": "Über Größe 日本語"}"#;
    let entry = MockEntry::migrate_from(json, 1).unwrap();
    assert_eq!(entry.full_name, "Über Größe 日本語");
}

#[test]
fn test_migration_extra_fields_ignored() {
    // Extra fields in old versions should not cause errors
    let json = r#"{"name": "Alice", "legacy_field": 42, "another": true}"#;
    let entry = MockEntry::migrate_from(json, 1).unwrap();
    assert_eq!(entry.full_name, "Alice");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MigrationError display
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_error_display_formats() {
    let e1 = MigrationError::UnknownVersion { found: 10, max_supported: 3 };
    assert!(e1.to_string().contains("10"));
    assert!(e1.to_string().contains("3"));

    let e2 = MigrationError::DeserializationFailed("bad JSON".into());
    assert!(e2.to_string().contains("bad JSON"));

    let e3 = MigrationError::MigrationStepFailed {
        from: 1,
        to: 2,
        reason: "missing field".into(),
    };
    assert!(e3.to_string().contains("v1"));
    assert!(e3.to_string().contains("v2"));
    assert!(e3.to_string().contains("missing field"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Integration: needs_migration with Migratable
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_needs_migration_with_migratable() {
    // MockEntry::CURRENT_VERSION = 3
    assert!(needs_migration(1, MockEntry::CURRENT_VERSION));
    assert!(needs_migration(2, MockEntry::CURRENT_VERSION));
    assert!(!needs_migration(3, MockEntry::CURRENT_VERSION));
    assert!(!is_future_version(2, MockEntry::CURRENT_VERSION));
    assert!(is_future_version(4, MockEntry::CURRENT_VERSION));
}
