// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::{Path, PathBuf};

const BARRIER_IMPL_MARKER: &str = "RestoredContinuityPromotionBarrier for";
const APPROVED_IMPL: &str =
    "crates/domains/symthaea-restored-continuity-anchor/src/lib.rs";

#[test]
fn production_restored_continuity_barrier_has_one_purpose_separated_implementation() {
    let root = repository_root();
    let mut implementations = Vec::new();
    visit_rust_sources(&root, &mut |path, source| {
        if is_test_source(path) {
            return;
        }
        if source.contains(BARRIER_IMPL_MARKER) {
            implementations.push(path.to_path_buf());
        }
    });

    assert_eq!(
        implementations.len(),
        1,
        "expected exactly one production restored-continuity promotion barrier implementation; found {implementations:#?}"
    );
    assert!(
        implementations[0].ends_with(APPROVED_IMPL),
        "restored-continuity promotion barrier bypassed purpose-separated adapter: {implementations:#?}"
    );
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

fn should_skip_directory(path: &Path) -> bool {
    path.file_name().is_some_and(|name| {
        matches!(
            name.to_str(),
            Some("target" | ".git" | ".direnv" | "node_modules" | "vendor")
        )
    })
}

fn visit_rust_sources(root: &Path, visitor: &mut impl FnMut(&Path, &str)) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if should_skip_directory(&path) {
                continue;
            }
            visit_rust_sources(&path, visitor);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path).expect("repository Rust source must be readable");
            visitor(&path, &source);
        }
    }
}
