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
fn wcare47_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        (
            "docs/release/evidence/WCARE47_STANDALONE_LOCK_ADMISSION_PROTOCOL_V1.md",
            "331d911cb60a0e9e899a4f9ca359381702090b66",
        ),
        (
            "docs/release/evidence/WCARE47_LOCK_ADMISSION_RESULT_SCHEMA_V1.json",
            "e16c822c4b41bf2cede04745ad4f0cefdab973d9",
        ),
        ("scripts/wcare47_lock_admit.py", "0456dd638b11d17efdc64ce3d64cbcebab060308"),
        ("scripts/wcare47_selftest.py", "bea208b796d20b01c69f9b12c1073775c7cc9293"),
        ("scripts/wcare47-integrity.sh", "9f9a29399dad9d8f0793a736f28382109cfb7579"),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-47 byte drift: {path}");
    }
}

#[test]
fn wcare47_prelock_blocker_executes_and_refuses_admission() {
    let root = repo_root();
    let output = Command::new("bash")
        .arg("scripts/wcare47-integrity.sh")
        .current_dir(&root)
        .output()
        .expect("run WCARE-47 integrity gate");
    assert!(
        output.status.success(),
        "WCARE-47 integrity gate failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("integrity stdout is utf-8");
    for marker in [
        "PASS_PROTOCOL_INTEGRITY",
        "PASS_WCARE47_PRELOCK_FAIL_CLOSED",
        "\"classification\":\"INFRASTRUCTURE_INDETERMINATE\"",
        "\"detail\":\"candidate_lock_missing\"",
        "\"lock_admitted\":false",
        "\"wcare42_executable_qualification_established\":false",
        "\"runtime_authority_granted\":false",
    ] {
        assert!(stdout.contains(marker), "WCARE-47 receipt missing {marker}: {stdout}");
    }
}
