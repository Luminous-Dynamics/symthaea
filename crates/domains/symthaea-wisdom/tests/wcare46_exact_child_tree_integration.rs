// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("resolve repository root")
}

fn git_blob(root: &Path, relative: &str) -> String {
    let output = Command::new("git")
        .args(["hash-object", relative])
        .current_dir(root)
        .output()
        .expect("run git hash-object");
    assert!(output.status.success(), "git hash-object failed for {relative}");
    String::from_utf8(output.stdout)
        .expect("git hash is utf-8")
        .trim()
        .to_owned()
}

#[test]
fn wcare46_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        (
            "docs/release/evidence/WCARE46_EXACT_CHILD_TREE_INTEGRATION_PROTOCOL_V1.md",
            "40e3f34b41a5d9ccb8e813b60837423aa28ae253",
        ),
        (
            "docs/release/evidence/WCARE46_PREFLIGHT_RESULT_SCHEMA_V1.json",
            "1b263b8aad3bbddfc973371599ee6e0ca39358e3",
        ),
        ("scripts/wcare46-preflight.py", "fd921dcd66ea85e872fd5e1bf30b1609a67ea432"),
        ("scripts/wcare46_selftest.py", "7a6c38e974fd0be3f8762df6e5b31976272b506d"),
        ("scripts/wcare46-integrity.sh", "0af7eba72ceec4ab4e70043823ed7f730016bcc1"),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-46 byte drift: {path}");
    }
}

#[test]
fn wcare46_exact_integration_executes_and_refuses_promotion() {
    let root = repo_root();
    let output = Command::new("bash")
        .arg("scripts/wcare46-integrity.sh")
        .current_dir(&root)
        .output()
        .expect("run WCARE-46 integrity gate");
    assert!(
        output.status.success(),
        "WCARE-46 integrity gate failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("integrity stdout is utf-8");
    for marker in [
        "PASS_PROTOCOL_INTEGRITY",
        "PASS_WCARE46_EXACT_TREE_INTEGRATION",
        "\"exact_child_tree_integration_verified\":true",
        "\"standalone_lock_present\":false",
        "\"child_execution_established\":false",
        "\"runtime_authority_granted\":false",
    ] {
        assert!(stdout.contains(marker), "WCARE-46 receipt missing {marker}: {stdout}");
    }
}
