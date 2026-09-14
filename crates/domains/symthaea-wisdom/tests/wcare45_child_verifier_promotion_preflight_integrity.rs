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
fn wcare45_preflight_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        (
            "docs/release/evidence/WCARE45_CHILD_VERIFIER_PROMOTION_PROTOCOL_V1.md",
            "85db0ee96aa0e3673f8e8f586a7bdff93f569b4a",
        ),
        (
            "docs/release/evidence/WCARE45_PREFLIGHT_RESULT_SCHEMA_V1.json",
            "b0640098c16b59ec126031dd4f7eeae7698dc1b5",
        ),
        ("scripts/wcare45-preflight.py", "bd76fe8cd51c6d45599030d4f89a60fdffe9f575"),
        ("scripts/wcare45_selftest.py", "ef02f19b46f4728326af908212d1a66a979c01de"),
        ("scripts/wcare45-integrity.sh", "9fbb01a8d21f9fd6a236d77c843dafdf46101264"),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-45 byte drift: {path}");
    }
}

#[test]
fn wcare45_preflight_executes_and_refuses_promotion() {
    let root = repo_root();
    let output = Command::new("bash")
        .arg("scripts/wcare45-integrity.sh")
        .current_dir(&root)
        .output()
        .expect("run WCARE-45 integrity gate");
    assert!(
        output.status.success(),
        "WCARE-45 integrity gate failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("integrity stdout is utf-8");
    for marker in [
        "PASS_PROTOCOL_INTEGRITY",
        "\"child_verifier_lineage_established\":false",
        "\"authenticated_preregistered_replication_established\":false",
        "\"runtime_authority_granted\":false",
    ] {
        assert!(stdout.contains(marker), "WCARE-45 receipt missing {marker}: {stdout}");
    }
}
