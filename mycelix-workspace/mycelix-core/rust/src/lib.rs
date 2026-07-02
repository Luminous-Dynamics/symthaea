// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-Core Rust Components
//!
//! High-performance implementations for federated learning operations:
//! - Secure aggregation
//! - Byzantine detection
//! - Differential privacy
//! - Cryptographic primitives

pub mod aggregation;
pub mod byzantine;
pub mod crypto;

/// Initialize the Mycelix Rust core
pub fn init() {
    println!("Mycelix Rust core initialized v{}", env!("CARGO_PKG_VERSION"));
}

/// Version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
    }

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
