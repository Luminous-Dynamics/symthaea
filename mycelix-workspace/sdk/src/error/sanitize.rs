//! Error Sanitization Module
//!
//! Provides error sanitization to prevent sensitive data leakage in production.
//! In debug builds, full error details are preserved for development.
//! In release builds, PII and sensitive financial data are redacted.
//!
//! # Security Assumptions
//!
//! - **Production builds** should be compiled with `--release` to enable sanitization
//! - **Debug builds** intentionally expose full errors for development
//! - **Regex patterns** are designed to catch common PII formats but may not be exhaustive
//!
//! # Threat Model
//!
//! - Adversary may have access to error logs or API responses
//! - Error messages should not reveal private financial information
//! - Cryptographic keys and identifiers should be redacted
//!
//! # Example
//!
//! ```rust
//! use mycelix_sdk::error::{SanitizedError, sanitize_message};
//!
//! let error = std::io::Error::new(
//!     std::io::ErrorKind::Other,
//!     "Income $75,000.00 exceeds bracket"
//! );
//!
//! let sanitized = SanitizedError::new(error);
//! // In release: "Income [REDACTED_AMOUNT] exceeds bracket"
//! // In debug: "Income $75,000.00 exceeds bracket"
//! ```

use std::fmt;

/// Wrapper for errors that sanitizes sensitive data in production builds.
///
/// In debug builds (`#[cfg(debug_assertions)]`), the full error message is preserved.
/// In release builds, sensitive patterns (income, SSN, keys) are redacted.
pub struct SanitizedError<E> {
    inner: E,
    #[cfg(debug_assertions)]
    full_message: String,
}

impl<E: std::error::Error> SanitizedError<E> {
    /// Create a new sanitized error wrapper.
    ///
    /// The original error is preserved for logging and chaining.
    /// The display message is sanitized in release builds.
    pub fn new(error: E) -> Self {
        Self {
            #[cfg(debug_assertions)]
            full_message: format!("{:?}", error),
            inner: error,
        }
    }

    /// Get a reference to the inner error.
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Consume the wrapper and return the inner error.
    pub fn into_inner(self) -> E {
        self.inner
    }
}

impl<E: std::error::Error> fmt::Display for SanitizedError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(debug_assertions)]
        {
            write!(f, "{}", self.full_message)
        }

        #[cfg(not(debug_assertions))]
        {
            write!(f, "{}", sanitize_message(&self.inner.to_string()))
        }
    }
}

impl<E: std::error::Error> fmt::Debug for SanitizedError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(debug_assertions)]
        {
            write!(f, "SanitizedError {{ {} }}", self.full_message)
        }

        #[cfg(not(debug_assertions))]
        {
            write!(
                f,
                "SanitizedError {{ {} }}",
                sanitize_message(&self.inner.to_string())
            )
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SanitizedError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.inner)
    }
}

/// Sanitize a message by removing potentially sensitive information.
///
/// This function removes:
/// - Financial amounts (e.g., "$75,000.00", "123.45 USD")
/// - Social Security Numbers (e.g., "123-45-6789")
/// - Long hexadecimal strings (potential keys/hashes)
/// - Email addresses
/// - IP addresses
///
/// # Security Note
///
/// This function uses regex patterns that may not catch all sensitive data.
/// Always review error messages in security-sensitive contexts.
pub fn sanitize_message(msg: &str) -> String {
    let mut sanitized = msg.to_string();

    // Remove income/financial values with currency symbols
    // Matches: $1,234.56, $1234, EUR 1234.56, 1234.56 USD
    sanitized = INCOME_PATTERN
        .replace_all(&sanitized, "[REDACTED_AMOUNT]")
        .to_string();

    // Remove potential SSN patterns (XXX-XX-XXXX)
    sanitized = SSN_PATTERN
        .replace_all(&sanitized, "[REDACTED_ID]")
        .to_string();

    // Remove hex keys longer than 16 chars (potential crypto keys/hashes)
    sanitized = KEY_PATTERN
        .replace_all(&sanitized, "[REDACTED_KEY]")
        .to_string();

    // Remove email addresses
    sanitized = EMAIL_PATTERN
        .replace_all(&sanitized, "[REDACTED_EMAIL]")
        .to_string();

    // Remove IPv4 addresses
    sanitized = IPV4_PATTERN
        .replace_all(&sanitized, "[REDACTED_IP]")
        .to_string();

    sanitized
}

/// Check if a message contains potentially sensitive data.
///
/// This is useful for logging decisions - sensitive messages
/// might be logged at a different level or to a different sink.
pub fn contains_sensitive_data(msg: &str) -> bool {
    INCOME_PATTERN.is_match(msg)
        || SSN_PATTERN.is_match(msg)
        || KEY_PATTERN.is_match(msg)
        || EMAIL_PATTERN.is_match(msg)
        || IPV4_PATTERN.is_match(msg)
}

// Lazy-compiled regex patterns for performance
use once_cell::sync::Lazy;
use regex::Regex;

static INCOME_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\$[\d,]+\.?\d*|\d+\.\d{2}\s*(?:USD|EUR|GBP|JPY|CNY|INR|CAD|AUD|CHF|KRW|BRL)|(?:USD|EUR|GBP)\s*\d+\.?\d*)")
        .expect("hardcoded INCOME_PATTERN regex should compile")
});

static SSN_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").expect("hardcoded SSN_PATTERN regex should compile")
});

static KEY_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[a-fA-F0-9]{16,}\b").expect("hardcoded KEY_PATTERN regex should compile")
});

static EMAIL_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        .expect("hardcoded EMAIL_PATTERN regex should compile")
});

static IPV4_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
        .expect("hardcoded IPV4_PATTERN regex should compile")
});

/// A result type that automatically sanitizes errors on display.
pub type SanitizedResult<T, E> = Result<T, SanitizedError<E>>;

/// Extension trait for converting Results to SanitizedResults.
pub trait SanitizeExt<T, E: std::error::Error> {
    /// Convert the error in this Result to a SanitizedError.
    fn sanitize_err(self) -> SanitizedResult<T, E>;
}

impl<T, E: std::error::Error> SanitizeExt<T, E> for Result<T, E> {
    fn sanitize_err(self) -> SanitizedResult<T, E> {
        self.map_err(SanitizedError::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_income_values() {
        let msg = "Income $75,000.00 is outside bracket bounds";
        let sanitized = sanitize_message(msg);
        assert!(sanitized.contains("[REDACTED_AMOUNT]"));
        assert!(!sanitized.contains("75,000"));
    }

    #[test]
    fn test_sanitize_currency_formats() {
        assert_eq!(
            sanitize_message("Amount: 1234.56 USD"),
            "Amount: [REDACTED_AMOUNT]"
        );
        assert_eq!(
            sanitize_message("EUR 999.99 was transferred"),
            "[REDACTED_AMOUNT] was transferred"
        );
    }

    #[test]
    fn test_sanitize_ssn() {
        let msg = "SSN 123-45-6789 invalid";
        let sanitized = sanitize_message(msg);
        assert!(sanitized.contains("[REDACTED_ID]"));
        assert!(!sanitized.contains("123-45-6789"));
    }

    #[test]
    fn test_sanitize_hex_keys() {
        let msg = "Key 0123456789abcdef0123 invalid";
        let sanitized = sanitize_message(msg);
        assert!(sanitized.contains("[REDACTED_KEY]"));
        assert!(!sanitized.contains("0123456789abcdef0123"));
    }

    #[test]
    fn test_sanitize_email() {
        let msg = "Contact user@example.com for help";
        let sanitized = sanitize_message(msg);
        assert!(sanitized.contains("[REDACTED_EMAIL]"));
        assert!(!sanitized.contains("user@example.com"));
    }

    #[test]
    fn test_sanitize_ip() {
        let msg = "Connection from 192.168.1.100 failed";
        let sanitized = sanitize_message(msg);
        assert!(sanitized.contains("[REDACTED_IP]"));
        assert!(!sanitized.contains("192.168.1.100"));
    }

    #[test]
    fn test_contains_sensitive_data() {
        assert!(contains_sensitive_data("Income $50,000"));
        assert!(contains_sensitive_data("SSN: 123-45-6789"));
        assert!(contains_sensitive_data("Key: abcdef0123456789"));
        assert!(!contains_sensitive_data("Normal error message"));
    }

    #[test]
    fn test_sanitized_error_wrapper() {
        let error = std::io::Error::new(std::io::ErrorKind::Other, "Failed for income $100,000.00");
        let sanitized = SanitizedError::new(error);

        // The inner error should still be accessible
        assert!(sanitized.inner().to_string().contains("100,000"));
    }

    #[test]
    fn test_preserves_non_sensitive() {
        let msg = "Bracket validation failed: lower bound exceeded";
        let sanitized = sanitize_message(msg);
        assert_eq!(msg, sanitized);
    }
}
