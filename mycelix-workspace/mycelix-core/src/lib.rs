// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Core - Holochain zomes for the Mycelix ecosystem
//!
//! This crate serves as the root of the Mycelix Core project,
//! which contains multiple Holochain zomes for decentralized applications.
//!
//! ## Zomes
//!
//! - **bridge**: Inter-hApp communication, reputation sharing, credential verification
//! - **federated_learning**: Byzantine-resistant federated learning
//! - **agents**: Agent management and discovery

/// Version of the Mycelix Core library
pub const VERSION: &str = "0.1.0";

/// Placeholder module for workspace root
pub mod placeholder {
    /// Returns the current version
    pub fn version() -> &'static str {
        super::VERSION
    }
}
