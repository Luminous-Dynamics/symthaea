// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Black-box confirmation that the sandbox boundary holds from outside the crate,
//! exercised through the public `Sandbox` API rather than the internal
//! `resolve_in_sandbox` function directly (that has its own thorough unit-test table
//! in `src/tools.rs`).

use std::fs;
use symthaea_audit::tools::Sandbox;

fn build_repo() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("repo");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(tmp.path().join("secret.txt"), "should never be readable").unwrap();
    (tmp, root)
}

#[test]
fn read_file_allows_in_tree_paths() {
    let (_tmp, root) = build_repo();
    let sandbox = Sandbox::new(&root, vec![]).unwrap();
    let preview = sandbox.read_file("src/main.rs").unwrap();
    assert!(preview.content.contains("fn main"));
}

#[test]
fn read_file_denies_parent_traversal() {
    let (_tmp, root) = build_repo();
    let sandbox = Sandbox::new(&root, vec![]).unwrap();
    let result = sandbox.read_file("../secret.txt");
    assert!(
        result.is_err(),
        "expected the sandbox to deny reading outside its root"
    );
}

#[test]
fn read_file_denies_absolute_escape() {
    let (tmp, root) = build_repo();
    let sandbox = Sandbox::new(&root, vec![]).unwrap();
    let secret = tmp.path().join("secret.txt");
    let result = sandbox.read_file(secret.to_str().unwrap());
    assert!(result.is_err());
}

#[cfg(unix)]
#[test]
fn recursive_list_dir_denies_internal_symlink_escape() {
    use std::os::unix::fs::symlink;
    let (tmp, root) = build_repo();
    let outside_dir = tmp.path().join("outside_dir");
    fs::create_dir_all(&outside_dir).unwrap();
    fs::write(outside_dir.join("leak.txt"), "leaked").unwrap();
    symlink(&outside_dir, root.join("src/escape_link")).unwrap();

    let sandbox = Sandbox::new(&root, vec![]).unwrap();
    let listing = sandbox.list_dir("src", true).unwrap();
    assert!(
        !listing.entries.iter().any(|e| e.contains("leak.txt")),
        "recursive list_dir must not follow an internal symlink outside the sandbox root: {:?}",
        listing.entries
    );
}

#[test]
fn run_check_without_allowlist_is_rejected() {
    let (_tmp, root) = build_repo();
    let sandbox = Sandbox::new(&root, vec![]).unwrap();
    assert!(sandbox.run_check("echo hi").is_err());
}

#[test]
fn run_check_with_allowlist_permits_exact_match_only() {
    let (_tmp, root) = build_repo();
    let sandbox = Sandbox::new(&root, vec!["true".to_string()]).unwrap();
    assert!(sandbox.run_check("true").is_ok());
    assert!(sandbox.run_check("false").is_err());
}
