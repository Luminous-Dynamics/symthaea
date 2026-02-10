//! Cryptographic Utilities
//!
//! This module provides security-critical cryptographic operations including:
//!
//! - **Constant-time comparisons**: Prevent timing side-channel attacks
//! - **Secure memory handling**: Zeroize sensitive data on drop
//!
//! # Security Design
//!
//! All functions in this module are designed with security as the primary concern:
//!
//! 1. **Timing-attack resistance**: Comparisons take constant time
//! 2. **Memory safety**: Sensitive data is zeroized when dropped
//! 3. **Fail-safe defaults**: Operations fail securely on error
//!
//! # Usage
//!
//! ```rust
//! use mycelix_sdk::crypto::{secure_compare, secure_compare_hash};
//!
//! // Compare two byte slices in constant time
//! let computed_hash = [0u8; 32];
//! let expected_hash = [0u8; 32];
//!
//! if secure_compare(&computed_hash, &expected_hash) {
//!     // Hashes match - proceed
//! } else {
//!     // Hashes differ - reject
//! }
//!
//! // Compare fixed-size hashes
//! let hash_a: [u8; 32] = [1u8; 32];
//! let hash_b: [u8; 32] = [1u8; 32];
//! assert!(secure_compare_hash(&hash_a, &hash_b));
//! ```

mod constant_time;

// Re-export public API
pub use constant_time::{
    is_zero, is_zero_hash, secure_compare, secure_compare_64, secure_compare_hash,
};
