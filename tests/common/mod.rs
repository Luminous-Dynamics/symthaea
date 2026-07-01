// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared Test Infrastructure
//!
//! Common utilities, builders, fixtures, mocks, and assertions for integration tests.
//!
//! # Usage
//!
//! Add this to your test file:
//! ```rust,ignore
//! mod common;
//! use common::prelude::*;
//! ```
//!
//! Or import specific modules:
//! ```rust,ignore
//! mod common;
//! use common::builders::MemoryRecordBuilder;
//! use common::fixtures::TempDatabase;
//! ```

pub mod assertions;
pub mod builders;
pub mod fixtures;
pub mod mocks;

/// Prelude module for convenient imports
#[allow(unused_imports)]
pub mod prelude {
    pub use super::assertions::*;
    pub use super::builders::*;
    pub use super::fixtures::*;
    pub use super::mocks::*;
}