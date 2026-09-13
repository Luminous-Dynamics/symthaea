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
        (
            "docs/release/evidence/WCARE39_EXECUTION_CAPSULE_PROTOCOL_V1.md",
            "af313132a55160a2db2423efacf5cd8a2c88f605",
        ),
        (
            "docs/release/evidence/WCARE39_EXECUTION_CAPSULE_SCHEMA_V1.json",
            "0d3936654d947062664999bbb59063e9f24f1b24",
        ),
        (
            "docs/release/evidence/WCARE39_COMMAND_PLAN_SCHEMA_V1.json",
            "e2b0a027daaf3518a21e36b5f6c97952879ad470",
        ),
        (
            "docs/release/evidence/WCARE39_EXECUTION_STATUS_SCHEMA_V1.json",
            "14be5a2cd585c0e790e99bce1ebf7c7273cf1b9b",
        ),
        (
            "scripts/wcare39_execution_capsule.py",
            "17c43109a59047052737b39a5d12465b2d824eb1",
        ),
        (
            "scripts/wcare39-qualify.sh",
            "5a86b1055b1d9415d09679ccc7d06aca94de30e2",
        ),
        (
            "scripts/wcare39_selftest.py",
            "dd32058e770cd9014beced33e8ba34805a029ca3",
        ),
        (
            "scripts/wcare39-integrity.sh",
            "ceb840c5d254949fa905c52ec72fcb1baed1698a",
        ),
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
        "\"qualifier_exit_contract_verified\":true",
    ] {
        assert!(stdout.contains(marker), "missing self-test marker {marker}: {stdout}");
    }
}
