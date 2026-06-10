// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    /// Cognitive loop errors (cycle execution, panic recovery).
    CognitiveLoop(String),
    /// Language subsystem errors (LLM backends, domain plugins, Phi monitor).
    Language(String),
    /// Perception subsystem errors (encoding, projection, physiological data).
    Perception(String),
    /// Pipeline integration errors (conscious pipeline, session management).
    Pipeline(String),
    /// Physiology subsystem errors (coherence, social coordination).
    Physiology(String),
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
            Self::CognitiveLoop(msg) => write!(f, "Cognitive loop error: {msg}"),
            Self::Language(msg) => write!(f, "Language error: {msg}"),
            Self::Perception(msg) => write!(f, "Perception error: {msg}"),
            Self::Pipeline(msg) => write!(f, "Pipeline error: {msg}"),
            Self::Physiology(msg) => write!(f, "Physiology error: {msg}"),
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

impl From<String> for SymthaeaError {
    fn from(msg: String) -> Self {
        Self::Runtime(msg)
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

// -- Infrastructure errors (somatic error bridge) --

impl From<crate::infrastructure::somatic_error_bridge::InfrastructureError> for SymthaeaError {
    fn from(err: crate::infrastructure::somatic_error_bridge::InfrastructureError) -> Self {
        Self::Runtime(format!("infrastructure: {err:?}"))
    }
}

// -- Swarm / Network errors (always available) --

impl From<crate::swarm::NetworkError> for SymthaeaError {
    fn from(err: crate::swarm::NetworkError) -> Self {
        Self::Network(err.to_string())
    }
}

impl From<crate::swarm::SwarmError> for SymthaeaError {
    fn from(err: crate::swarm::SwarmError) -> Self {
        Self::Network(err.to_string())
    }
}

impl From<crate::swarm::HandshakeError> for SymthaeaError {
    fn from(err: crate::swarm::HandshakeError) -> Self {
        Self::Network(err.to_string())
    }
}

// -- Action errors (always available) --

impl From<crate::action::ExecutionError> for SymthaeaError {
    fn from(err: crate::action::ExecutionError) -> Self {
        Self::Runtime(err.to_string())
    }
}

impl From<crate::action::ActionError> for SymthaeaError {
    fn from(err: crate::action::ActionError) -> Self {
        Self::Runtime(err.to_string())
    }
}

// -- Feature-gated error bridges --

#[cfg(feature = "school_learning")]
impl From<crate::school::curriculum_loader::LoadError> for SymthaeaError {
    fn from(err: crate::school::curriculum_loader::LoadError) -> Self {
        Self::Config(err.to_string())
    }
}

#[cfg(feature = "full_language")]
impl From<crate::language::learning_persistence::LearningPersistenceError> for SymthaeaError {
    fn from(err: crate::language::learning_persistence::LearningPersistenceError) -> Self {
        Self::Language(err.to_string())
    }
}

#[cfg(feature = "web_research_module")]
impl From<crate::web_research::researcher::WebResearcherCreationError> for SymthaeaError {
    fn from(err: crate::web_research::researcher::WebResearcherCreationError) -> Self {
        Self::Runtime(err.to_string())
    }
}

// ---------------------------------------------------------------------------
// Convenience type alias
// ---------------------------------------------------------------------------

/// Convenience [`Result`] alias using [`SymthaeaError`].
pub type SymthaeaResult<T> = Result<T, SymthaeaError>;

// ---------------------------------------------------------------------------
// Contextual error enrichment
// ---------------------------------------------------------------------------

/// Severity of an error — determines recovery strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// The subsystem can recover; skip or degrade gracefully.
    #[default]
    Recoverable,
    /// The system cannot continue meaningfully.
    Fatal,
}

/// Structured context attached to a [`SymthaeaError`].
#[derive(Debug, Default, Clone)]
pub struct ErrorContext {
    /// Which subsystem produced this error (e.g., `"causal_explanation"`).
    pub subsystem: Option<String>,
    /// Cycle number when the error occurred.
    pub cycle: Option<usize>,
    /// Error severity.
    pub severity: ErrorSeverity,
}

/// A [`SymthaeaError`] enriched with structured context.
///
/// Created via [`SymthaeaError::in_subsystem`] or [`SymthaeaError::with_context`].
/// This is additive — existing code continues to use plain `SymthaeaError`.
#[derive(Debug)]
pub struct ContextualError {
    /// The underlying error.
    pub error: SymthaeaError,
    /// Structured context describing where/when the error occurred.
    pub context: ErrorContext,
}

impl SymthaeaError {
    /// Wrap this error with a subsystem label.
    pub fn in_subsystem(self, subsystem: &str) -> ContextualError {
        ContextualError {
            error: self,
            context: ErrorContext {
                subsystem: Some(subsystem.to_string()),
                ..Default::default()
            },
        }
    }

    /// Wrap this error with full structured context.
    pub fn with_context(self, ctx: ErrorContext) -> ContextualError {
        ContextualError {
            error: self,
            context: ctx,
        }
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Recoverable => write!(f, "recoverable"),
            ErrorSeverity::Fatal => write!(f, "fatal"),
        }
    }
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref sub) = self.context.subsystem {
            write!(f, "[{sub}] {}", self.error)
        } else {
            write!(f, "{}", self.error)
        }
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

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
            SymthaeaError::Runtime(msg) => {
                assert!(!msg.is_empty(), "serde error message should not be empty")
            }
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
    #[allow(clippy::unnecessary_literal_unwrap)]
    fn test_result_alias() {
        let ok: SymthaeaResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: SymthaeaResult<u32> = Err(SymthaeaError::Config("bad".into()));
        assert!(err.is_err());
    }

    #[test]
    fn test_contextual_error_in_subsystem() {
        let err = SymthaeaError::Consciousness("phi diverged".into());
        let ctx_err = err.in_subsystem("causal_explanation");

        assert_eq!(
            ctx_err.to_string(),
            "[causal_explanation] Consciousness error: phi diverged"
        );
        assert_eq!(
            ctx_err.context.subsystem.as_deref(),
            Some("causal_explanation")
        );
        assert_eq!(ctx_err.context.severity, ErrorSeverity::Recoverable);
        assert!(ctx_err.context.cycle.is_none());
    }

    #[test]
    fn test_contextual_error_with_context() {
        let err = SymthaeaError::CognitiveLoop("timeout".into());
        let ctx = ErrorContext {
            subsystem: Some("moral_algebra".into()),
            cycle: Some(42),
            severity: ErrorSeverity::Fatal,
        };
        let ctx_err = err.with_context(ctx);

        assert_eq!(
            ctx_err.to_string(),
            "[moral_algebra] Cognitive loop error: timeout"
        );
        assert_eq!(ctx_err.context.cycle, Some(42));
        assert_eq!(ctx_err.context.severity, ErrorSeverity::Fatal);
    }

    #[test]
    fn test_from_network_error() {
        use crate::swarm::NetworkError;
        let err: SymthaeaError = NetworkError::Timeout {
            operation: "ping".into(),
            timeout_ms: 5000,
        }
        .into();
        match &err {
            SymthaeaError::Network(msg) => assert!(msg.contains("ping"), "got: {msg}"),
            other => panic!("expected Network variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_from_swarm_error() {
        use crate::swarm::SwarmError;
        let err: SymthaeaError = SwarmError::NotInitialized.into();
        match &err {
            SymthaeaError::Network(msg) => {
                assert!(!msg.is_empty(), "swarm error message should not be empty")
            }
            other => panic!("expected Network variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_from_handshake_error() {
        use crate::swarm::HandshakeError;
        let err: SymthaeaError = HandshakeError::ProtocolViolation {
            message: "bad nonce".into(),
        }
        .into();
        match &err {
            SymthaeaError::Network(msg) => assert!(msg.contains("bad nonce"), "got: {msg}"),
            other => panic!("expected Network variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_from_execution_error() {
        use crate::action::ExecutionError;
        let err: SymthaeaError = ExecutionError::Unsupported("nope".into()).into();
        match &err {
            SymthaeaError::Runtime(msg) => assert!(msg.contains("nope"), "got: {msg}"),
            other => panic!("expected Runtime variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_from_action_error() {
        use crate::action::ActionError;
        let err: SymthaeaError = ActionError::ValidationFailed("missing param".into()).into();
        match &err {
            SymthaeaError::Runtime(msg) => {
                assert!(msg.contains("missing param"), "got: {msg}")
            }
            other => panic!("expected Runtime variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_from_infrastructure_error() {
        use crate::infrastructure::somatic_error_bridge::InfrastructureError;
        let err: SymthaeaError = InfrastructureError::AsyncTaskPanicked {
            task_name: "heartbeat".into(),
        }
        .into();
        match &err {
            SymthaeaError::Runtime(msg) => {
                assert!(msg.contains("heartbeat"), "got: {msg}")
            }
            other => panic!("expected Runtime variant, got: {other:?}"),
        }
    }
}
