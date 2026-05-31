// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! mycelix-media Test Suite
//!
//! Comprehensive test coverage for the media hApp including:
//! - Unit tests for each zome (publication, attribution, curation, factcheck, bridge)
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
//! Run all tests: `cargo test -p mycelix-media-tests`
//!
//! Run specific module: `cargo test -p mycelix-media-tests publication`
//!
//! Run with verbose output: `cargo test -p mycelix-media-tests -- --nocapture`

pub mod integration;
pub mod unit;

// Re-export for convenience
pub use integration::*;
pub use unit::*;
