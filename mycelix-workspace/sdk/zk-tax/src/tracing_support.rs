//! Tracing and observability support for ZK Tax proofs.
//!
//! This module provides structured logging and instrumentation using the `tracing` crate.
//! Enable with the `tracing-support` feature.
//!
//! # Usage
//!
//! ```rust,ignore
//! use mycelix_zk_tax::tracing_support::{init_tracing, TracingConfig, LogLevel};
//!
//! // Initialize with defaults (INFO level, no JSON)
//! init_tracing(TracingConfig::default());
//!
//! // Or with custom settings
//! init_tracing(TracingConfig {
//!     level: LogLevel::Debug,
//!     json_output: true,
//!     service_name: "zk-tax-api".to_string(),
//! });
//! ```
//!
//! # Spans and Events
//!
//! All major operations are instrumented with spans:
//! - `prove`: Proof generation (includes income range, jurisdiction, year)
//! - `verify`: Proof verification
//! - `find_bracket`: Bracket lookup operations
//! - `compress`/`decompress`: Proof compression operations
//!
//! Events are logged at appropriate levels:
//! - `INFO`: Operation start/complete, key metrics
//! - `DEBUG`: Intermediate steps, cache hits/misses
//! - `TRACE`: Detailed internal state
//! - `WARN`: Non-fatal issues (fallback behaviors)
//! - `ERROR`: Operation failures

use std::fmt;

#[cfg(feature = "tracing-support")]
pub use tracing::{
    debug, error, event, info, info_span, instrument, span, trace, warn,
    Level, Span,
};

/// Log level for tracing output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LogLevel {
    /// Only errors
    Error,
    /// Errors and warnings
    Warn,
    /// Default level - errors, warnings, and info
    #[default]
    Info,
    /// More verbose - includes debug messages
    Debug,
    /// Maximum verbosity
    Trace,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Error => write!(f, "error"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Trace => write!(f, "trace"),
        }
    }
}

#[cfg(feature = "tracing-support")]
impl From<LogLevel> for tracing::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => tracing::Level::ERROR,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Trace => tracing::Level::TRACE,
        }
    }
}

/// Configuration for tracing initialization.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Log level filter
    pub level: LogLevel,
    /// Whether to output JSON-formatted logs
    pub json_output: bool,
    /// Service name for structured logging
    pub service_name: String,
    /// Whether to include file/line info
    pub include_location: bool,
    /// Whether to include target (module path)
    pub include_target: bool,
    /// Whether to include thread IDs
    pub include_thread_ids: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            json_output: false,
            service_name: "zk-tax".to_string(),
            include_location: false,
            include_target: true,
            include_thread_ids: false,
        }
    }
}

impl TracingConfig {
    /// Create a development configuration with debug logging.
    pub fn development() -> Self {
        Self {
            level: LogLevel::Debug,
            json_output: false,
            service_name: "zk-tax-dev".to_string(),
            include_location: true,
            include_target: true,
            include_thread_ids: false,
        }
    }

    /// Create a production configuration with JSON output.
    pub fn production() -> Self {
        Self {
            level: LogLevel::Info,
            json_output: true,
            service_name: "zk-tax".to_string(),
            include_location: false,
            include_target: true,
            include_thread_ids: true,
        }
    }

    /// Set the log level.
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }

    /// Enable JSON output.
    pub fn with_json(mut self) -> Self {
        self.json_output = true;
        self
    }

    /// Set the service name.
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }
}

/// Initialize tracing with the given configuration.
///
/// This should be called once at application startup.
///
/// # Panics
///
/// Panics if tracing has already been initialized.
#[cfg(feature = "tracing-support")]
pub fn init_tracing(config: TracingConfig) {
    use tracing_subscriber::{
        fmt::{self, format::FmtSpan},
        prelude::*,
        EnvFilter,
    };

    // Build the filter from the level
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()));

    // Build the subscriber
    let subscriber = tracing_subscriber::registry().with(filter);

    if config.json_output {
        // JSON formatting for production
        let fmt_layer = fmt::layer()
            .json()
            .with_span_events(FmtSpan::CLOSE)
            .with_file(config.include_location)
            .with_line_number(config.include_location)
            .with_target(config.include_target)
            .with_thread_ids(config.include_thread_ids);

        subscriber.with(fmt_layer).init();
    } else {
        // Human-readable format for development
        let fmt_layer = fmt::layer()
            .with_span_events(FmtSpan::CLOSE)
            .with_file(config.include_location)
            .with_line_number(config.include_location)
            .with_target(config.include_target)
            .with_thread_ids(config.include_thread_ids);

        subscriber.with(fmt_layer).init();
    }

    tracing::info!(
        service = %config.service_name,
        level = %config.level,
        "Tracing initialized"
    );
}

/// Initialize tracing with default configuration.
#[cfg(feature = "tracing-support")]
pub fn init_default_tracing() {
    init_tracing(TracingConfig::default());
}

/// Try to initialize tracing, returning Ok if successful.
///
/// Unlike `init_tracing`, this won't panic if tracing is already initialized.
#[cfg(feature = "tracing-support")]
pub fn try_init_tracing(config: TracingConfig) -> Result<(), Box<dyn std::error::Error>> {
    use tracing_subscriber::{
        fmt::{self, format::FmtSpan},
        prelude::*,
        EnvFilter,
    };

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()));

    let subscriber = tracing_subscriber::registry().with(filter);

    if config.json_output {
        let fmt_layer = fmt::layer()
            .json()
            .with_span_events(FmtSpan::CLOSE)
            .with_file(config.include_location)
            .with_line_number(config.include_location)
            .with_target(config.include_target)
            .with_thread_ids(config.include_thread_ids);

        subscriber.with(fmt_layer).try_init()?;
    } else {
        let fmt_layer = fmt::layer()
            .with_span_events(FmtSpan::CLOSE)
            .with_file(config.include_location)
            .with_line_number(config.include_location)
            .with_target(config.include_target)
            .with_thread_ids(config.include_thread_ids);

        subscriber.with(fmt_layer).try_init()?;
    }

    Ok(())
}

// ============================================================================
// Instrumentation Macros (work with or without tracing)
// ============================================================================

/// Create a span for proof generation.
#[cfg(feature = "tracing-support")]
#[macro_export]
macro_rules! prove_span {
    ($jurisdiction:expr, $year:expr, $filing_status:expr) => {
        tracing::info_span!(
            "prove",
            jurisdiction = %$jurisdiction,
            year = $year,
            filing_status = %$filing_status,
        )
    };
}

/// Create a span for proof verification.
#[cfg(feature = "tracing-support")]
#[macro_export]
macro_rules! verify_span {
    ($jurisdiction:expr, $year:expr) => {
        tracing::info_span!(
            "verify",
            jurisdiction = %$jurisdiction,
            year = $year,
        )
    };
}

/// Log a proof generation event.
#[cfg(feature = "tracing-support")]
#[macro_export]
macro_rules! log_proof_generated {
    ($bracket:expr, $rate:expr, $duration:expr) => {
        tracing::info!(
            bracket = $bracket,
            rate_bps = $rate,
            duration_ms = $duration.as_millis() as u64,
            "Proof generated"
        );
    };
}

/// Log a proof verification event.
#[cfg(feature = "tracing-support")]
#[macro_export]
macro_rules! log_proof_verified {
    ($valid:expr, $duration:expr) => {
        tracing::info!(
            valid = $valid,
            duration_ms = $duration.as_millis() as u64,
            "Proof verified"
        );
    };
}

/// Log a bracket lookup event.
#[cfg(feature = "tracing-support")]
#[macro_export]
macro_rules! log_bracket_found {
    ($income:expr, $bracket:expr, $rate:expr) => {
        tracing::debug!(
            income = $income,
            bracket = $bracket,
            rate_bps = $rate,
            "Bracket found"
        );
    };
}

/// Log a cache hit.
#[cfg(feature = "tracing-support")]
#[macro_export]
macro_rules! log_cache_hit {
    ($key:expr) => {
        tracing::debug!(
            key = %$key,
            "Cache hit"
        );
    };
}

/// Log a cache miss.
#[cfg(feature = "tracing-support")]
#[macro_export]
macro_rules! log_cache_miss {
    ($key:expr) => {
        tracing::debug!(
            key = %$key,
            "Cache miss"
        );
    };
}

// No-op versions when tracing is disabled
#[cfg(not(feature = "tracing-support"))]
#[macro_export]
macro_rules! prove_span {
    ($jurisdiction:expr, $year:expr, $filing_status:expr) => {{
        struct NoOpGuard;
        NoOpGuard
    }};
}

#[cfg(not(feature = "tracing-support"))]
#[macro_export]
macro_rules! verify_span {
    ($jurisdiction:expr, $year:expr) => {{
        struct NoOpGuard;
        NoOpGuard
    }};
}

#[cfg(not(feature = "tracing-support"))]
#[macro_export]
macro_rules! log_proof_generated {
    ($bracket:expr, $rate:expr, $duration:expr) => {};
}

#[cfg(not(feature = "tracing-support"))]
#[macro_export]
macro_rules! log_proof_verified {
    ($valid:expr, $duration:expr) => {};
}

#[cfg(not(feature = "tracing-support"))]
#[macro_export]
macro_rules! log_bracket_found {
    ($income:expr, $bracket:expr, $rate:expr) => {};
}

#[cfg(not(feature = "tracing-support"))]
#[macro_export]
macro_rules! log_cache_hit {
    ($key:expr) => {};
}

#[cfg(not(feature = "tracing-support"))]
#[macro_export]
macro_rules! log_cache_miss {
    ($key:expr) => {};
}

// ============================================================================
// Metrics Collection
// ============================================================================

/// Metrics collected during operations.
#[derive(Debug, Clone, Default)]
pub struct OperationMetrics {
    /// Duration of the operation in milliseconds
    pub duration_ms: u64,
    /// Number of bytes in the proof
    pub proof_size_bytes: Option<usize>,
    /// Number of brackets checked
    pub brackets_checked: Option<usize>,
    /// Whether result was from cache
    pub cache_hit: bool,
    /// Additional key-value metrics
    pub extra: Vec<(String, String)>,
}

impl OperationMetrics {
    /// Create new metrics with a duration.
    pub fn new(duration: std::time::Duration) -> Self {
        Self {
            duration_ms: duration.as_millis() as u64,
            ..Default::default()
        }
    }

    /// Add proof size.
    pub fn with_proof_size(mut self, size: usize) -> Self {
        self.proof_size_bytes = Some(size);
        self
    }

    /// Mark as cache hit.
    pub fn with_cache_hit(mut self) -> Self {
        self.cache_hit = true;
        self
    }

    /// Add a custom metric.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.push((key.into(), value.into()));
        self
    }

    /// Log these metrics.
    #[cfg(feature = "tracing-support")]
    pub fn log(&self, operation: &str) {
        tracing::info!(
            operation = %operation,
            duration_ms = self.duration_ms,
            proof_size_bytes = ?self.proof_size_bytes,
            cache_hit = self.cache_hit,
            "Operation complete"
        );
    }

    /// No-op when tracing disabled.
    #[cfg(not(feature = "tracing-support"))]
    pub fn log(&self, _operation: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_display() {
        assert_eq!(LogLevel::Error.to_string(), "error");
        assert_eq!(LogLevel::Warn.to_string(), "warn");
        assert_eq!(LogLevel::Info.to_string(), "info");
        assert_eq!(LogLevel::Debug.to_string(), "debug");
        assert_eq!(LogLevel::Trace.to_string(), "trace");
    }

    #[test]
    fn test_tracing_config_defaults() {
        let config = TracingConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert!(!config.json_output);
        assert_eq!(config.service_name, "zk-tax");
    }

    #[test]
    fn test_tracing_config_builder() {
        let config = TracingConfig::default()
            .with_level(LogLevel::Debug)
            .with_json()
            .with_service_name("my-service");

        assert_eq!(config.level, LogLevel::Debug);
        assert!(config.json_output);
        assert_eq!(config.service_name, "my-service");
    }

    #[test]
    fn test_development_config() {
        let config = TracingConfig::development();
        assert_eq!(config.level, LogLevel::Debug);
        assert!(config.include_location);
    }

    #[test]
    fn test_production_config() {
        let config = TracingConfig::production();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.json_output);
        assert!(config.include_thread_ids);
    }

    #[test]
    fn test_operation_metrics() {
        let metrics = OperationMetrics::new(std::time::Duration::from_millis(100))
            .with_proof_size(1024)
            .with_cache_hit()
            .with_extra("bracket", "3");

        assert_eq!(metrics.duration_ms, 100);
        assert_eq!(metrics.proof_size_bytes, Some(1024));
        assert!(metrics.cache_hit);
        assert_eq!(metrics.extra.len(), 1);
    }
}
