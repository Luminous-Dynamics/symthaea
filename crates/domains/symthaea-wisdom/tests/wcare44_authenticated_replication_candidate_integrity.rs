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

fn run_python(root: &Path, script: &str) -> String {
    let output = Command::new("python3")
        .arg(script)
        .current_dir(root)
        .output()
        .unwrap_or_else(|error| panic!("run {script}: {error}"));
    assert!(
        output.status.success(),
        "{script} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("python self-test stdout is utf-8")
}

#[test]
fn wcare44_candidate_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        (
            "docs/release/evidence/WCARE44_AUTHENTICATED_REPLICATION_AGGREGATION_PROTOCOL_V1.md",
            "2c1b0f51454b6d5e7b0c98c3d6055fbd63caee3b",
        ),
        (
            "docs/release/evidence/WCARE44_BUILDER_AUTH_OBSERVATION_SCHEMA_V1.json",
            "f129ae7fef8d20856277a97f6f549b9313e98e8c",
        ),
        (
            "docs/release/evidence/WCARE44_TEMPORAL_OBSERVATION_SCHEMA_V1.json",
            "2fabbb71170d0efe34e01a2166d197a44e056b3b",
        ),
        (
            "docs/release/evidence/WCARE44_CANDIDATE_RESULT_SCHEMA_V1.json",
            "2db071910816c1354bc8e14a46e8118ea8052a97",
        ),
        (
            "scripts/wcare44_candidate_kernel.py",
            "c5e0f52e34c4753028a00cf6891c8725d04e6b9a",
        ),
        (
            "scripts/wcare44_candidate_qualify.py",
            "44d3ea2572430f828e4dce2fcaf4c6c09756f52e",
        ),
        (
            "scripts/wcare44_candidate_selftest.py",
            "1449466d0f237871360904c0228f12e8f7af2231",
        ),
        (
            "scripts/wcare44_frontdoor_selftest.py",
            "184cdcdaef2aeecd5e627519ced5f50ff567a619",
        ),
        (
            "scripts/wcare44-integrity.sh",
            "51620ff3b13f2740390d9f8a33672f7d8e936640",
        ),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-44 byte drift: {path}");
    }
}

#[test]
fn wcare44_candidate_and_frontdoor_campaigns_execute() {
    let root = repo_root();
    let candidate = run_python(&root, "scripts/wcare44_candidate_selftest.py");
    for marker in [
        "\"classification\":\"PASS_WCARE44_CANDIDATE_SELFTEST\"",
        "\"full_candidate_conjunction_without_final_promotion_verified\":true",
        "\"runtime_authority_granted\":false",
    ] {
        assert!(candidate.contains(marker), "candidate campaign missing {marker}: {candidate}");
    }

    let frontdoor = run_python(&root, "scripts/wcare44_frontdoor_selftest.py");
    for marker in [
        "\"classification\":\"PASS_WCARE44_FRONTDOOR_SELFTEST\"",
        "\"wrong_builder_verifier_identity_rejected\":true",
        "\"wrong_temporal_verifier_identity_rejected\":true",
        "\"wrong_wcare40_frontdoor_identity_rejected\":true",
        "\"wrong_wcare40_core_identity_rejected\":true",
        "\"malformed_optional_field_rejected\":true",
        "\"final_promotion_remains_blocked\":true",
        "\"runtime_authority_granted\":false",
    ] {
        assert!(frontdoor.contains(marker), "frontdoor campaign missing {marker}: {frontdoor}");
    }
}

#[test]
fn wcare44_integrity_gate_preserves_non_promotion() {
    let root = repo_root();
    let output = Command::new("bash")
        .arg("scripts/wcare44-integrity.sh")
        .current_dir(&root)
        .output()
        .expect("run WCARE-44 integrity gate");
    assert!(
        output.status.success(),
        "WCARE-44 integrity gate failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("integrity stdout is utf-8");
    for marker in [
        "PASS_PROTOCOL_INTEGRITY",
        "\"child_verifier_lineage_established\":false",
        "\"builder_authentication_established\":false",
        "\"preregistration_temporal_precedence_established\":false",
        "\"authenticated_preregistered_replication_established\":false",
        "\"runtime_authority_granted\":false",
    ] {
        assert!(stdout.contains(marker), "WCARE-44 integrity receipt missing {marker}: {stdout}");
    }
}
