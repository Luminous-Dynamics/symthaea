// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration test support for the mycelix-craft hApp.
//!
//! Tests use Holochain sweettest to spin up real conductors.
//! Run with: `cargo test --release -- --ignored --test-threads=2`

use std::path::PathBuf;

/// Path to the compiled Craft DNA bundle.
///
/// Requires: `hc dna pack dna/` run from mycelix-craft root first.
pub fn craft_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ → mycelix-craft/
    path.push("dna");
    path.push("mycelix_craft.dna");
    path
}
