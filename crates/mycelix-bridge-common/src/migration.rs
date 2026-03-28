// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Schema migration framework for versioned DHT entries.
//!
//! All Mycelix bridge entry types carry a `schema_version: u8` field.
//! This module provides the [`Migratable`] trait for reading entries
//! that may have been written with an older schema version.
//!
//! Currently all entries are at version 1, so no actual migrations exist yet.
//! This establishes the pattern for future schema evolution.

/// Trait for entry types that support schema migration.
///
/// Implementors define `CURRENT_VERSION` and a `migrate_from` function
/// that can upgrade entries from any prior version to the current one.
pub trait Migratable: Sized {
    /// The current schema version for this entry type.
    const CURRENT_VERSION: u8;

    /// Attempt to deserialize and migrate an entry from `from_version` to `CURRENT_VERSION`.
    ///
    /// Returns `Err` if the version is unsupported or the data is corrupt.
    fn migrate_from(json: &str, from_version: u8) -> Result<Self, MigrationError>;
}

/// Error type for migration failures.
#[derive(Debug, Clone)]
pub enum MigrationError {
    /// The schema version is not recognized.
    UnknownVersion { found: u8, max_supported: u8 },
    /// The entry data could not be deserialized.
    DeserializationFailed(String),
    /// A migration step failed.
    MigrationStepFailed { from: u8, to: u8, reason: String },
}

impl core::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownVersion { found, max_supported } => {
                write!(f, "Unknown schema version {} (max supported: {})", found, max_supported)
            }
            Self::DeserializationFailed(msg) => write!(f, "Deserialization failed: {}", msg),
            Self::MigrationStepFailed { from, to, reason } => {
                write!(f, "Migration v{} → v{} failed: {}", from, to, reason)
            }
        }
    }
}

/// Check if a schema version needs migration.
pub const fn needs_migration(entry_version: u8, current_version: u8) -> bool {
    entry_version < current_version
}

/// Check if a schema version is from the future (written by a newer version of the code).
pub const fn is_future_version(entry_version: u8, current_version: u8) -> bool {
    entry_version > current_version
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_migration_detects_old_version() {
        assert!(needs_migration(0, 1));
        assert!(needs_migration(1, 2));
        assert!(!needs_migration(1, 1));
        assert!(!needs_migration(2, 1));
    }

    #[test]
    fn is_future_version_detects_newer() {
        assert!(is_future_version(2, 1));
        assert!(!is_future_version(1, 1));
        assert!(!is_future_version(0, 1));
    }

    #[test]
    fn migration_error_display() {
        let err = MigrationError::UnknownVersion { found: 5, max_supported: 3 };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));
    }
}
