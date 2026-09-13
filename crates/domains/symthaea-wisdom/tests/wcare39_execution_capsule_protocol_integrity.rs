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
fn wcare39_exact_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        ("docs/release/evidence/WCARE39_EXECUTION_CAPSULE_PROTOCOL_V1.md", "c196824cba4afd25ec6e59d6215a33e24a97cbb3"),
        ("docs/release/evidence/WCARE39_EXECUTION_CAPSULE_SCHEMA_V1.json", "0d3936654d947062664999bbb59063e9f24f1b24"),
        ("docs/release/evidence/WCARE39_COMMAND_PLAN_SCHEMA_V1.json", "e2b0a027daaf3518a21e36b5f6c97952879ad470"),
        ("docs/release/evidence/WCARE39_EXECUTION_STATUS_SCHEMA_V1.json", "35afbd3c4dbd7e5bd765509ffb1839b543c228af"),
        ("scripts/wcare39_execution_capsule.py", "17c43109a59047052737b39a5d12465b2d824eb1"),
        ("scripts/wcare39_selftest.py", "42f0bc87ffafe3c359af7774f50545b80a17c8fe"),
        ("scripts/wcare39-integrity.sh", "3ed04e6f3f4f1aa2067a141584ff855e65479d97"),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-39 byte drift: {path}");
    }
}

#[test]
fn wcare39_synthetic_campaign_executes() {
    let root = repo_root();
    let output = Command::new("python3")
        .arg("scripts/wcare39_selftest.py")
        .current_dir(&root)
        .output()
        .expect("run WCARE-39 synthetic campaign");
    assert!(
        output.status.success(),
        "WCARE-39 synthetic campaign failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("self-test stdout is utf-8");
    for marker in [
        "PASS_WCARE39_SELFTEST",
        "\"qualified_subject_failure_observed\":true",
        "\"stale_output_rejected\":true",
        "\"prepared_binding_tamper_rejected\":true",
        "\"missing_subject_drift_observed\":true",
        "\"wcare37_lock_blocker_preserved\":true",
        "\"sensitive_literal_rejected\":true",
    ] {
        assert!(stdout.contains(marker), "missing self-test marker {marker}: {stdout}");
    }
}
