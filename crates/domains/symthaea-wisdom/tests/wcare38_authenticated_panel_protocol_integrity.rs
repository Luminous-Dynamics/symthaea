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

fn assert_blob(root: &Path, relative: &str, expected: &str) {
    assert_eq!(git_blob(root, relative), expected, "blob drift: {relative}");
}

fn run_python_selftest(root: &Path, relative: &str, pass_marker: &str) {
    let output = Command::new("python3")
        .arg(relative)
        .current_dir(root)
        .output()
        .unwrap_or_else(|error| panic!("run {relative}: {error}"));
    assert!(
        output.status.success(),
        "{relative} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("self-test stdout is utf-8");
    assert!(stdout.contains(pass_marker), "{relative} missing {pass_marker}: {stdout}");
}

#[test]
fn wcare38_exact_review_unit_is_frozen() {
    let root = repo_root();
    for (path, sha) in [
        (
            "docs/release/evidence/WCARE38_AUTHENTICATED_PANEL_PROTOCOL_V1.md",
            "ebffc2a6bf8f5114a6e852791ef881c4b0219bcf",
        ),
        (
            "docs/release/evidence/WCARE38_AUTHENTICATED_PANEL_PLAN_SCHEMA_V1.json",
            "2b2650bf1b5b7269209e36415e5a74fcd77ed080",
        ),
        (
            "docs/release/evidence/WCARE38_ATTESTATION_PACKAGE_MANIFEST_SCHEMA_V1.json",
            "a87ecc428c4974672a01f322367fba62735c0bd2",
        ),
        (
            "docs/release/evidence/WCARE38_AUTHENTICATED_PANEL_RESULT_SCHEMA_V1.json",
            "f374cd71a864eb179d68c815abf4e9b0420de9ca",
        ),
        (
            "scripts/wcare38-qualify.py",
            "885b952a8a75fba59a75544429b0338fd0ff2519",
        ),
        (
            "scripts/wcare38_qualify_authenticated_panel.py",
            "cf198d78e76eab05a64c3489bdd51f538f026a1d",
        ),
        (
            "scripts/wcare38_monotonicity_selftest.py",
            "e9e82d13cfdef768cb5873359d77f98de465495a",
        ),
        (
            "scripts/wcare38_adversarial_selftest.py",
            "b522f8e81114ee9d41383e5409f6bf6c465c89df",
        ),
        (
            "scripts/wcare38-integrity.sh",
            "a8cd54ae1cfda782fe89a308c3aae0a91d7a6dd5",
        ),
    ] {
        assert_blob(&root, path, sha);
    }
}

#[test]
fn wcare38_dependency_free_selftests_execute() {
    let root = repo_root();
    run_python_selftest(
        &root,
        "scripts/wcare38_monotonicity_selftest.py",
        "PASS_MONOTONICITY_SELFTEST",
    );
    run_python_selftest(
        &root,
        "scripts/wcare38_adversarial_selftest.py",
        "PASS_ADVERSARIAL_SOURCE_SELFTEST",
    );
}
