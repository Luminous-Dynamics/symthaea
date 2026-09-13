// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::{Path, PathBuf};

const BATCH_IMPORT_CALL: &str = ".from_validated_persisted_active_state(";
const BATCH_IMPORT_CALL_ASSOCIATED: &str = "EpisodicMemory::from_validated_persisted_active_state(";
const POINT_RESTORE_CALL: &str = ".restore_validated_persisted_occurrence(";

#[test]
fn only_welfare_restart_adapter_calls_exact_uuid_batch_constructor_in_production() {
    let root = repository_root();
    let mut offenders = Vec::new();
    visit_rust_sources(&root.join("crates"), &mut |path, source| {
        if is_test_source(path) || is_import_mechanism(path) {
            return;
        }
        if source.contains(BATCH_IMPORT_CALL) || source.contains(BATCH_IMPORT_CALL_ASSOCIATED) {
            let expected = path.ends_with(
                "crates/domains/symthaea-welfare-assurance/src/canonical_memory_restart.rs",
            );
            if !expected {
                offenders.push(path.to_path_buf());
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "production exact-UUID batch import bypasses welfare restart adapter: {offenders:#?}"
    );
}

#[test]
fn exact_point_restore_has_no_production_caller_before_governed_restore_adapter_exists() {
    let root = repository_root();
    let mut offenders = Vec::new();
    visit_rust_sources(&root.join("crates"), &mut |path, source| {
        if is_test_source(path) || is_import_mechanism(path) {
            return;
        }
        if source.contains(POINT_RESTORE_CALL) {
            offenders.push(path.to_path_buf());
        }
    });
    assert!(
        offenders.is_empty(),
        "exact persisted point restore gained an unapproved production caller: {offenders:#?}"
    );
}

#[test]
fn approved_restart_adapter_never_routes_through_fresh_uuid_insertion() {
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/canonical_memory_restart.rs"),
    )
    .unwrap();
    for forbidden in [".store_if_significant(", ".store_if_significant_with_id("] {
        assert!(
            !source.contains(forbidden),
            "approved restart adapter must not call fresh-UUID insertion API {forbidden:?}"
        );
    }
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("welfare-assurance crate must live under crates/domains")
        .to_path_buf()
}

fn is_test_source(path: &Path) -> bool {
    path.components().any(|component| component.as_os_str() == "tests")
        || path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with("_test.rs") || name.ends_with("_tests.rs"))
}

fn is_import_mechanism(path: &Path) -> bool {
    path.ends_with("crates/domains/symthaea-memory/src/episodic_replay/persisted_import.rs")
}

fn visit_rust_sources(root: &Path, visitor: &mut impl FnMut(&Path, &str)) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "target") {
                continue;
            }
            visit_rust_sources(&path, visitor);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path).expect("workspace Rust source must be readable");
            visitor(&path, &source);
        }
    }
}
