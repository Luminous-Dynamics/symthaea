#![allow(dead_code)]
// Mock helpers are shared across test modules

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! mycelix-property Test Suite
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

pub mod integration;
pub mod unit;

// Re-export for convenience
pub use integration::*;
pub use unit::*;
