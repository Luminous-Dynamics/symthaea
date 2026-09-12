// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;

/// Restart continuity must never "solve" reconstruction by routing persisted episodes back through
/// ordinary cognition insertion. Those APIs intentionally mint fresh occurrence UUIDs and would
/// silently destroy the identity continuity that quarantine/restore evidence binds.
#[test]
fn production_continuity_store_does_not_call_fresh_uuid_insertion_apis() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(manifest_dir.join("src/lib.rs")).unwrap();

    for forbidden in [".store_if_significant(", ".store_if_significant_with_id("] {
        assert!(
            !source.contains(forbidden),
            "restart continuity production code must not call fresh-UUID insertion API {forbidden:?}"
        );
    }
}

/// The continuity crate owns exact persistence/recovery, not the live replay engine. Keeping the
/// production dependency graph one-way prevents a future convenience refactor from making the
/// persistence layer itself an alternate cognitive store.
#[test]
fn production_dependencies_do_not_directly_depend_on_symthaea_memory() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap();
    let dependencies = manifest
        .split("[dev-dependencies]")
        .next()
        .expect("manifest has production dependency section");
    assert!(
        !dependencies.contains("symthaea-memory"),
        "continuity persistence must consume assurance envelopes/contracts, not become a second live memory engine"
    );
}
