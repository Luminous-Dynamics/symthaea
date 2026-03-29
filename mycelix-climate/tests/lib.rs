// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! mycelix-climate Test Suite
//!
//! Comprehensive test coverage for the climate hApp including:
//! - Unit tests for each zome (carbon, projects, bridge)
//! - Integration tests for cross-zome operations
//! - Edge case and error condition coverage
//!
//! ## Test Structure
//!
//! - tests/lib.rs - This file (test suite entry point)
//! - tests/unit/ - Unit tests for each zome
//! - tests/integration/ - Cross-zome integration tests
//!
//! ## Running Tests
//!
//! Run all tests: `cargo test -p mycelix-climate-tests`
//!
//! Run specific module: `cargo test -p mycelix-climate-tests carbon`
//!
//! Run with verbose output: `cargo test -p mycelix-climate-tests -- --nocapture`

pub mod unit;
pub mod integration;

// Re-export for convenience
pub use unit::*;
pub use integration::*;
