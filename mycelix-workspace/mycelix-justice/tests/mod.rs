// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Mycelix Justice Test Suite
//!
//! Comprehensive test coverage for the mycelix-justice hApp dispute resolution system.
//!
//! ## Test Modules
//!
//! - `integrity_tests`: Entry validation tests
//! - `case_lifecycle_tests`: Case management and status transitions
//! - `evidence_tests`: Evidence submission and chain of custody
//! - `arbitration_tests`: Arbitration workflows and decisions
//! - `enforcement_tests`: Enforcement action execution
//! - `restorative_tests`: Restorative justice circles
//! - `access_control_tests`: Party permissions and authorization
//!
//! ## Test Categories
//!
//! 1. **Entry Validation** - Tests that validation rules are enforced
//! 2. **Status Transitions** - Tests case/arbitration lifecycle
//! 3. **Access Control** - Tests party permissions
//! 4. **Evidence Integrity** - Tests chain of custody
//! 5. **Cross-Zome Integration** - Tests coordination between zomes

// Test modules
mod access_control_tests;
mod arbitration_tests;
mod case_lifecycle_tests;
mod enforcement_tests;
mod evidence_tests;
mod integrity_tests;
mod restorative_tests;

// Re-export lazy_static for test modules
#[macro_use]
extern crate lazy_static;
