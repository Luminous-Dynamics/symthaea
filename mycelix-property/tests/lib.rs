#![allow(dead_code)] // Mock helpers are shared across test modules
//! mycelix-property Test Suite
//!
//! Comprehensive test coverage for the property hApp including:
//! - Unit tests for each zome (registry, transfer, disputes, commons, bridge)
//! - Integration tests for cross-zome operations
//! - Edge case and error condition coverage
//!
//! ## Test Structure
//!
//! - tests/lib.rs - This file (test suite entry point)
//! - tests/unit/ - Unit tests for each zome
//! - tests/integration/ - Cross-zome integration tests
//! - tests/byzantine/ - (future) Byzantine fault tolerance tests
//!
//! ## Running Tests
//!
//! Run all tests: `cargo test -p mycelix-property-tests`
//!
//! Run specific module: `cargo test -p mycelix-property-tests registry`
//!
//! Run with verbose output: `cargo test -p mycelix-property-tests -- --nocapture`

pub mod unit;
pub mod integration;

// Re-export for convenience
pub use unit::*;
pub use integration::*;
