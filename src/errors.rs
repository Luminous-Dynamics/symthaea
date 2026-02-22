//! Unified error types for Symthaea subsystems.
//!
//! Provides a single top-level error enum ([`SymthaeaError`]) that downstream
//! code can use instead of ad-hoc per-module error types. [`From`] impls bridge
//! the most common standard-library and crate-internal error types so that `?`
//! works naturally.
//!
//! # Design principles
//!
//! * **Lightweight** -- one enum, no new dependencies.
//! * **Non-invasive** -- existing module-level error types are *not* modified.
//!   This module is the *foundation*; retrofitting call sites is a separate task.
//! * **Extensible** -- new variants can be added as subsystems are migrated.

use std::fmt;

use crate::databases::DatabaseError;

/// Top-level error categories for the Symthaea system.
///
/// Each variant wraps a human-readable message describing what went wrong.
/// More structured payloads can be added later as subsystems adopt this type.
#[derive(Debug)]
pub enum SymthaeaError {
    /// HDC encoding/decoding errors.
    Hdc(String),
    /// Consciousness computation errors (Phi, Psi, Sigma).
    Consciousness(String),
    /// Database storage/retrieval errors.
    Database(String),
    /// Configuration validation errors.
    Config(String),
    /// Network/communication errors.
    Network(String),
    /// General runtime errors (I/O, serialization, etc.).
    Runtime(String),
}

impl fmt::Display for SymthaeaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdc(msg) => write!(f, "HDC error: {msg}"),
            Self::Consciousness(msg) => write!(f, "Consciousness error: {msg}"),
            Self::Database(msg) => write!(f, "Database error: {msg}"),
            Self::Config(msg) => write!(f, "Config error: {msg}"),
            Self::Network(msg) => write!(f, "Network error: {msg}"),
            Self::Runtime(msg) => write!(f, "Runtime error: {msg}"),
        }
    }
}

impl std::error::Error for SymthaeaError {}

// ---------------------------------------------------------------------------
// From impls -- standard library types
// ---------------------------------------------------------------------------

impl From<std::io::Error> for SymthaeaError {
    fn from(err: std::io::Error) -> Self {
        Self::Runtime(err.to_string())
    }
}

impl From<serde_json::Error> for SymthaeaError {
    fn from(err: serde_json::Error) -> Self {
        Self::Runtime(err.to_string())
    }
}

// ---------------------------------------------------------------------------
// From impls -- crate-internal error types
// ---------------------------------------------------------------------------

impl From<DatabaseError> for SymthaeaError {
    fn from(err: DatabaseError) -> Self {
        Self::Database(err.to_string())
    }
}

// ---------------------------------------------------------------------------
// Convenience type alias
// ---------------------------------------------------------------------------

/// Convenience [`Result`] alias using [`SymthaeaError`].
pub type SymthaeaResult<T> = Result<T, SymthaeaError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let cases: Vec<(SymthaeaError, &str)> = vec![
            (SymthaeaError::Hdc("bad dim".into()), "HDC error: bad dim"),
            (
                SymthaeaError::Consciousness("phi diverged".into()),
                "Consciousness error: phi diverged",
            ),
            (
                SymthaeaError::Database("connection lost".into()),
                "Database error: connection lost",
            ),
            (
                SymthaeaError::Config("missing field".into()),
                "Config error: missing field",
            ),
            (
                SymthaeaError::Network("timeout".into()),
                "Network error: timeout",
            ),
            (
                SymthaeaError::Runtime("unexpected EOF".into()),
                "Runtime error: unexpected EOF",
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: SymthaeaError = io_err.into();
        match &err {
            SymthaeaError::Runtime(msg) => assert!(
                msg.contains("file missing"),
                "expected 'file missing' in message, got: {msg}"
            ),
            other => panic!("expected Runtime variant, got: {other:?}"),
        }
        // Also verify Display round-trips sensibly.
        assert!(err.to_string().contains("Runtime error"));
    }

    #[test]
    fn test_from_serde_error() {
        let bad_json = "{ not valid json }}}";
        let serde_err = serde_json::from_str::<serde_json::Value>(bad_json).unwrap_err();
        let err: SymthaeaError = serde_err.into();
        match &err {
            SymthaeaError::Runtime(msg) => assert!(
                !msg.is_empty(),
                "serde error message should not be empty"
            ),
            other => panic!("expected Runtime variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_from_database_error() {
        let db_err = DatabaseError::ConnectionFailed("host unreachable".into());
        let err: SymthaeaError = db_err.into();
        match &err {
            SymthaeaError::Database(msg) => assert!(
                msg.contains("host unreachable"),
                "expected 'host unreachable' in message, got: {msg}"
            ),
            other => panic!("expected Database variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_error_is_std_error() {
        // Verify that SymthaeaError satisfies std::error::Error.
        fn assert_std_error<E: std::error::Error>() {}
        assert_std_error::<SymthaeaError>();
    }

    #[test]
    fn test_result_alias() {
        let ok: SymthaeaResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: SymthaeaResult<u32> = Err(SymthaeaError::Config("bad".into()));
        assert!(err.is_err());
    }
}
