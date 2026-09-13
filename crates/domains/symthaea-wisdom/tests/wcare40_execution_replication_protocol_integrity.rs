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

fn run_python(root: &Path, relative: &str, markers: &[&str]) {
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
    for marker in markers {
        assert!(stdout.contains(marker), "{relative} missing {marker}: {stdout}");
    }
}

#[test]
fn wcare40_exact_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        (
            "docs/release/evidence/WCARE40_EXECUTION_REPLICATION_PROTOCOL_V1.md",
            "da09bf7adef07065a22bc5a2dd6e471d08aa139e",
        ),
        (
            "docs/release/evidence/WCARE40_REPLICATION_PLAN_SCHEMA_V1.json",
            "244f8b8fa063238d22c408365f5655f365cc70b6",
        ),
        (
            "docs/release/evidence/WCARE40_BUILDER_PROVENANCE_SCHEMA_V1.json",
            "d9dcceb5da983cc411e44cb1527bca2052bd7cd2",
        ),
        (
            "docs/release/evidence/WCARE40_BUILDER_RELATION_SCHEMA_V1.json",
            "8b00caac8bf934636789b74d048cabcce63b29ee",
        ),
        (
            "docs/release/evidence/WCARE40_REPLICATION_RESULT_SCHEMA_V1.json",
            "8b3a3978518fb5782e6e134b362fb34c89bc81ee",
        ),
        (
            "scripts/wcare40_verify_replication.py",
            "a368dc46b33272d6f424212a29ac26d5170b2e5a",
        ),
        (
            "scripts/wcare40-qualify.py",
            "13675951a5a8bd21c341cf6779341aa9e497b4d1",
        ),
        (
            "scripts/wcare40_selftest.py",
            "f782b10b17ebf41ce42647c4ee84eb938b4d0fb5",
        ),
        (
            "scripts/wcare40_frontdoor_selftest.py",
            "b95fc828b71a8f980bba0b3b5d5c29c25a6870e9",
        ),
        (
            "scripts/wcare40-integrity.sh",
            "a4a8e825ff61cf274663c25dd3c8ba1109ba673a",
        ),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-40 byte drift: {path}");
    }
}

#[test]
fn wcare40_adversarial_campaigns_execute() {
    let root = repo_root();
    run_python(
        &root,
        "scripts/wcare40_selftest.py",
        &[
            "PASS_WCARE40_SELFTEST",
            "\"same_builder_retry_does_not_multiply_independence\":true",
            "\"pass_fail_contradiction_preserved\":true",
            "\"required_receipt_contradiction_preserved\":true",
            "\"missing_preregistered_slot_is_indeterminate\":true",
            "\"forged_shared_fault_domain_rejected\":true",
            "\"single_execution_cannot_be_replication\":true",
        ],
    );
    run_python(
        &root,
        "scripts/wcare40_frontdoor_selftest.py",
        &[
            "PASS_WCARE40_FRONTDOOR_SELFTEST",
            "\"frontdoor_supported_path_verified\":true",
            "\"forged_final_outcome_rejected\":true",
        ],
    );
}
