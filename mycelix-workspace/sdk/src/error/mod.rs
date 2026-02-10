//! Error Handling Module
//!
//! This module provides error types and sanitization utilities for the Mycelix SDK.
//!
//! # Security Considerations
//!
//! Error messages can leak sensitive information. This module provides:
//!
//! - **Sanitization**: Remove PII and financial data from error messages
//! - **Debug vs Release**: Full details in debug, redacted in release
//! - **Structured errors**: Machine-readable error codes for programmatic handling
//!
//! # Example
//!
//! ```rust
//! use mycelix_sdk::error::{sanitize_message, SanitizedError, SanitizeExt};
//!
//! fn process_income(income: u64) -> Result<(), Box<dyn std::error::Error>> {
//!     if income > 1_000_000 {
//!         return Err(format!("Income ${} exceeds maximum", income).into());
//!     }
//!     Ok(())
//! }
//!
//! // In production, use sanitized errors
//! let result = process_income(5_000_000);
//! if let Err(e) = result {
//!     // Log sanitized version
//!     println!("Error: {}", sanitize_message(&e.to_string()));
//! }
//! ```

mod sanitize;

pub use sanitize::{
    contains_sensitive_data, sanitize_message, SanitizeExt, SanitizedError, SanitizedResult,
};
